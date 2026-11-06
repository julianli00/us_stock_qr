from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import execution_gap_shield_v2 as rules
from us_quant import ssd_execution_gap_shield_v1 as framework
from us_quant.candidate_receipts import write_new_receipt
from us_quant.data import MarketData
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, lifecycle = framework.base, framework.ROOT, framework.lifecycle
reference, unique_refs, counts = framework.reference, framework.unique_refs, framework.counts
ID, KIND = rules.ID, framework.KIND
CANDIDATE = {"id": ID, "weighting": "same_Prod_delay_shield_law_with_retained_exact_proof_V2"}
FOLDER = base.REPORTS / "execution-gap-shield-20261106-v2"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/execution_gap_shield_20261106_registration_v2.json"
PUBLIC = ROOT / "evidence/execution_gap_shield_20261106_results_v2.json"
OWN = (
    "config/execution-gap-shield-v2.json",
    "src/us_quant/execution_gap_shield_v2.py",
    "src/us_quant/ssd_execution_gap_shield_v2.py",
    "tests/test_execution_gap_shield_v2.py",
    "EXECUTION_GAP_SHIELD_V2.md",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals()},
        argdefs=function.__defaults__,
    )


def dependencies():
    return bound(framework._dependencies)()


events = bound(framework.events)
check_frozen = bound(framework.check_frozen)


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "V2 cannot rewrite old failures or current authoritative counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "execution_gap_shield_V2_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    source_event = read_json(base.checked_reference(catalogue["source_registered_event"]))
    rules.require(
        source_event["kind"] == "registered"
        and source_event["candidate_id"] == "expert_pool_six_confidence_rated_prod_v1"
        and catalogue["new_V2_scenarios_targets_accounts_or_financial_outputs_computed"] is False
        and catalogue["terminal_V1_run_resolver_verifier_or_cash_replay_executed"] is False
        and catalogue["old_source_learner_shadow_target_or_account_replayed"] is False
        and catalogue["same_economic_law_sha256"] == rules.ECONOMIC_SHA256
        and closed["actual_authoritative_counts"]["current_journal_event_count"] == 1494,
        "Use valid original Prod sources,not the terminal V1 account or targets.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, source_event["body"]["frozen"], refs


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v2.json")
    suites = ElementTree.parse(base.checked_reference(proof["test_outputs"][0])).getroot()
    suites = suites.findall(".//testsuite")
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")),
        "Require exact-source V2 mixed-asset and retained-comparison validation.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_mixed_asset_full_frame_cash_parity",
            "test_corrupted_frame_operands_retained_before_failure",
            "test_mixed_asset_account_prefix_is_exact",
            "test_actual_V2_packet_entry_defers_historical_outputs",
            "test_economic_law_not_retuned",
        }.issubset(names),
        "The V2 proof and unchanged-law seams must actually be exercised.",
    )
    first = read_json(FOLDER / "validation-source-v1.json")
    rules.require(
        first["exit_code"] == 1 and first["passed_tests"] == 29,
        "Preserve the actual initial CSV frequency-metadata test failure.",
    )
    return [
        *proof["test_outputs"],
        reference(FOLDER / "validation-source-v2.json"),
        reference(FOLDER / "validation-source-v1.json"),
        *first["test_outputs"],
        *first["versioned_prototypes"],
    ]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial V2 freeze.",
        )
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1494
            and history[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(e["candidate_id"] == ID for e in history),
            "Use a separate unused V2 identity at the preserved terminal1494 baseline.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, parent, refs = source_inputs()
        validation = validation_refs()
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": "separate_V2_same_law_canonical_frame_retained_exact_proof",
            "construction": "same_cached_six_Prod_delay_shield_law_technical_successor_V2",
            "decision_program_kind": KIND,
            "prior_journal_sequence": 1494,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 303,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs([*parent["validation_evidence"], *validation]),
            "same_economic_law_sha256": rules.ECONOMIC_SHA256,
            "actual_comparison_operands_retained_before_assertions": True,
            "exact_all_column_and_position_equality_relaxed": False,
            "terminal_V1_repaired_retried_or_replayed": False,
            "independent_alpha_discovery_claimed": False,
            "source_expert_pool_or_admission_changed": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_V2_scenarios_targets_accounts_or_financial_outputs_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_separate_V2_program")
    return reference(INTENT)


inputs = bound(framework.inputs)
packet_body = bound(framework.packet_body)


def checked_packet(frozen):
    return bound(framework.checked_packet)(frozen)


register = bound(framework.register)
publication = bound(framework.publication)
write_run = bound(framework.write_run)


def retain_cash_comparison(directory, actual, replay, report):
    for name, frame in (
        ("actual-account.csv", actual.frame),
        ("cash-engine-account.csv", replay.frame),
        ("actual-weights.csv", actual.weights),
        ("cash-engine-weights.csv", replay.weights),
    ):
        write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "comparison.json",
        {
            "retained_at": utc_now(),
            **report,
            "retained_before_acceptance_assertion": True,
            "artifacts": [
                reference(directory / p)
                for p in (
                    "actual-account.csv",
                    "cash-engine-account.csv",
                    "actual-weights.csv",
                    "cash-engine-weights.csv",
                )
            ],
        },
    )


def evaluate_path(data, nominal, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    rules.require(
        not any(e["body"]["path_key"] == key for e in events("path_started")),
        "Do not replay a complete or partial new V2 path.",
    )
    directory = FOLDER / "accounts" / key
    base.append_event(
        base.JOURNAL,
        "path_started",
        ID,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    new_output_directory(directory)
    matrices, comparisons = directory / "reference-inputs", directory / "cash-comparison"
    new_output_directory(matrices)
    new_output_directory(comparisons)
    trace = directory / "chronological-trace.jsonl"
    counter = 0
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        def retain_inputs(state, target, sample):
            nonlocal counter
            path = matrices / f"{counter:06d}.npz"
            np.savez_compressed(
                path,
                starts=sample.starts.asi8,
                ends=sample.ends.asi8,
                waiting=sample.waiting,
                opening=sample.opening,
                fill_mark=sample.fill_mark,
            )
            retain(
                {
                    "kind": "reference_input",
                    "as_of": str(state.as_of.date()),
                    "scenario_operands": reference(path),
                    "actual_delay": sample.delay,
                    "main_asset_values": state.asset_values.tolist(),
                    "cash": state.cash,
                    "nav": state.nav,
                    "peak": state.peak,
                    "nominal": target.tolist(),
                }
            )
            counter += 1

        output = rules.run(
            data,
            nominal,
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            before_decision=lifecycle.runtime_guard,
            retain=retain,
            retain_inputs=retain_inputs,
        )
    artifacts = write_run(directory, output)

    def retain_comparison(actual, replay, report):
        retain_cash_comparison(comparisons, actual, replay, report)

    feedback = rules.verify_run(
        data,
        output,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
        retain_comparison=retain_comparison,
    )
    feedback["actual_retained_cash_comparison"] = reference(comparisons / "comparison.json")
    independent = base.independent_equity(
        data,
        output.issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    independent_path = directory / "independent-bt.csv"
    write_text_atomic(independent_path, independent.to_csv(float_format="%.17g"))
    own = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        output.result.frame,
        independent,
        data.risk_free.loc[output.result.frame.index],
        10000,
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], s] for s in ("SPY", "QQQ"))
    gates = base.gates(own, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    artifacts += [
        reference(trace),
        reference(independent_path),
        *[reference(p) for p in sorted(matrices.glob("*.npz"))],
        *[reference(p) for p in sorted(comparisons.iterdir())],
    ]
    assessments = [
        r["assessment"] for r in output.decisions if "reference_status" in r["assessment"]
    ]
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "decision_program_kind": KIND,
        "metrics": own,
        "execution": base.execution_statistics(
            output.result,
            data,
            10000,
            scenario["cost_bps_per_side"],
            1,
        ),
        "gates": gates,
        "all_point_gates_pass": all(gates.values()),
        "excess_net_cagr_vs_spy": own["cagr"] - spy["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"] - qqq["cagr"],
        "independent_bt_max_equity_difference_usd": difference,
        "actual_capital_queue_and_accounting_verification": feedback,
        "decision_reason_counts": dict(Counter(r["reason"] for r in output.decisions)),
        "finite_reference_status_counts": dict(Counter(a["reference_status"] for a in assessments)),
        "retained_reference_inputs": counter,
        "maximum_independent_scalar_money_error_USD": max(
            a["maximum_scalar_money_error_USD"] for a in assessments
        ),
        "artifacts": artifacts,
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        ID,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def prefix_audits(data, nominal):
    folder = FOLDER / "capital-prefix-audits-v2"
    new_output_directory(folder)
    source = FOLDER / "accounts/10y-base"
    saved = {
        name: pd.read_csv(
            source / name,
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    decisions = read_json(source / "decisions.json")["decisions"]
    results = []
    for position in read_json(rules.POLICY)["prefix_positions"]:
        lifecycle.runtime_guard()
        day = saved["account.csv"].index[position]
        clipped = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        output = rules.run(
            clipped,
            nominal.loc[:day],
            "2016-10-06",
            str(day.date()),
            cost_bps=5,
            delay=1,
            before_decision=lifecycle.runtime_guard,
        )
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        artifacts = write_run(directory, output)
        frames = (
            ("account.csv", output.result.frame),
            ("weights.csv", output.result.weights),
            ("desired.csv", output.desired),
            ("issued.csv", output.issued),
        )
        for name, _ in frames:
            path = directory / ("expected-" + name)
            write_text_atomic(path, saved[name].loc[:day].to_csv(float_format="%.17g"))
            artifacts.append(reference(path))
        expected_decisions = [r for r in decisions if pd.Timestamp(r["signal_session"]) <= day]
        write_new_receipt(
            directory / "expected-decisions.json",
            {
                "decisions": expected_decisions,
                "retained_before_assertion": True,
            },
        )
        artifacts.append(reference(directory / "expected-decisions.json"))
        for name, actual in frames:
            expected = saved[name].loc[:day]
            rules.require(
                expected.index.equals(actual.loc[:day].index)
                and expected.columns.equals(actual.columns)
                and np.array_equal(
                    expected.to_numpy(dtype=float),
                    actual.loc[:day].to_numpy(dtype=float),
                    equal_nan=True,
                ),
                "The full new V2 account/target prefix must remain exactly unchanged.",
            )
        rules.require(
            output.decisions == expected_decisions,
            "A truncated new V2 run changed an earlier capital or reference decision.",
        )
        results.append(
            {
                "last_session": str(day.date()),
                "formal_return_position": position,
                "all_frame_target_position_and_decision_prefixes_exact": True,
                "both_comparison_operands_retained_before_assertions": True,
                "artifacts": artifacts,
            }
        )
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefixes": results,
            "new_policy_auxiliary_prefix_runs": 5,
            "independent_economic_trials": 0,
            "source_learner_or_old_account_replayed": False,
        },
    )
    return reference(folder / "completion.json")


record_failure = bound(framework.record_failure)
evaluate = bound(framework.evaluate)
export = bound(framework.export)


def main():
    parser = argparse.ArgumentParser(description="Separate V2 exact-account engineering successor.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published V2 preregistration SHA.")
    result = (
        evaluate(args.commit)
        if args.action == "evaluate"
        else {
            "freeze": freeze,
            "register": register,
            "export": export,
        }[args.action]()
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
