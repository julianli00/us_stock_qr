from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import main_risk_confidence_prod_v1 as rules
from us_quant import ssd_confidence_rated_prod_v1 as source
from us_quant import ssd_execution_gap_shield_v2 as template
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, lifecycle = template.base, template.ROOT, template.lifecycle
reference, unique_refs, counts = template.reference, template.unique_refs, template.counts
ID, KIND = rules.ID, template.KIND
CANDIDATE = {"id": ID, "weighting": "monthly_Prod_confidence_from_actual_main_prefix_risk"}
FOLDER = base.REPORTS / "main-risk-confidence-prod-20261107-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/main_risk_confidence_prod_20261107_registration_v1.json"
PUBLIC = ROOT / "evidence/main_risk_confidence_prod_20261107_results_v1.json"
OWN = (
    "config/main-risk-confidence-prod-v1.json",
    "src/us_quant/main_risk_confidence_prod_v1.py",
    "src/us_quant/ssd_main_risk_confidence_prod_v1.py",
    "tests/test_main_risk_confidence_prod_v1.py",
    "MAIN_RISK_CONFIDENCE_PROD_V1.md",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals()},
        argdefs=function.__defaults__,
    )


def dependencies():
    return bound(template.framework._dependencies)()


events = bound(template.events)
check_frozen = bound(template.check_frozen)


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "Main-context learning cannot override authoritative counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "main_risk_confidence_prod_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, _, reader_refs, clocks, benchmarks = source.source_inputs()
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    registered = read_json(base.checked_reference(catalogue["source_registered_Prod_event"]))
    periods = read_json(base.checked_reference(catalogue["cached_six_source_month_audits"]))[
        "periods"
    ]
    rules.require(
        tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
        and tuple(e.identity.mechanism_group for e in experts) == rules.MECHANISMS
        and catalogue["source_clocks"] == clocks
        and catalogue["admitted_reader_references"] == reader_refs
        and len(periods) == 119
        and closed["actual_authoritative_counts"]["current_journal_event_count"] == 1506
        and registered["candidate_id"] == source.rules.ID
        and catalogue["new_main_context_confidences_learning_targets_or_accounts_computed"] is False
        and catalogue["old_source_policy_shadow_or_master_replayed"] is False,
        "Keep all admitted source identities, clocks, sealed months and old outcomes.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            *reader_refs,
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
            *[
                reference(FOLDER / n)
                for n in (
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
    return (
        data,
        rules.SourceBundle(tuple(experts), tuple(periods)),
        registered["body"]["frozen"],
        refs,
        benchmarks,
    )


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
        "Bind successful actual-main context, chronology and accounting checks.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_all_closes_not_only_month_ends_define_main_peak",
            "test_prior_origin_confidence_not_current_end_weights_loss",
            "test_mixed_monthly_main_cash_and_independent_BT",
            "test_path_local_learner_starts_fresh",
            "test_actual_packet_entry_defers_historical_outputs",
        }.issubset(names),
        "Exercise every real main-risk program Seam before funding.",
    )
    first = read_json(FOLDER / "validation-source-v1.json")
    rules.require(
        first["exit_code"] == 1
        and first["passed_tests"] == 36
        and first["junit_totals"]
        == {
            "tests": 37,
            "failures": 1,
            "errors": 0,
            "skipped": 0,
        },
        "Preserve the first mixed-asset fixture's incorrect pre-fill exposure assertion.",
    )
    return [
        *proof["test_outputs"],
        reference(FOLDER / "validation-source-v2.json"),
        proof["validation_runner"],
        *proof["versioned_prototypes"],
        reference(FOLDER / "validation-source-v1.json"),
        *first["test_outputs"],
        *first["versioned_prototypes"],
    ]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the complete or partial first actual-main confidence freeze.",
        )
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1506
            and history[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(e["candidate_id"] == ID for e in history),
            "Freeze one unused coupled-main identity at the unchanged closed1506 head.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, _ = source_inputs()
        validation, deps = validation_refs(), dependencies()
        frozen = {
            **parent,
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": "canonical_monthly_actual_main_confidence_program",
            "construction": "main_risk_conditional_participation_original_six_Prod",
            "decision_program_kind": KIND,
            "prior_journal_sequence": 1506,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 304,
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
            "learning_state_is_path_local_actual_main": True,
            "five_year_warm_source_learning_state_reused": False,
            "participation_is_capital_cap_or_drawdown_guarantee": False,
            "source_policies_shadows_or_old_masters_replayed": False,
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
                "new_main_context_confidences_learning_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_dynamic_main_context_program")
    return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(ref)
    rules.require(
        sys.version == intent["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == intent["package_versions"],
        "Keep the exact accepted frozen environment.",
    )
    data, bundle, _, refs, benchmarks = source_inputs()
    rules.require(refs == intent["source_references"], "No admitted input may change.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, bundle, frozen, benchmarks


packet_body = bound(template.packet_body)


def checked_packet(frozen):
    return bound(template.framework.checked_packet)(frozen)


register = bound(template.register)
publication = bound(template.publication)


def write_run(directory, output):
    return bound(template.write_run)(directory, output)


def retain_cash_comparison(directory, actual, replay, report):
    return bound(template.retain_cash_comparison)(directory, actual, replay, report)


def evaluate_path(data, bundle, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    rules.require(
        not any(e["body"]["path_key"] == key for e in events("path_started")),
        "Do not replay a complete or partial coupled-main account.",
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
    comparisons = directory / "cash-comparison"
    new_output_directory(comparisons)
    trace = directory / "chronological-trace.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        output = rules.run(
            data,
            bundle,
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    artifacts = write_run(directory, output)
    raw_path = directory / "raw-kernel-auxiliary-gross.csv"
    write_text_atomic(raw_path, output.raw_kernel_auxiliary_gross.to_csv(float_format="%.17g"))

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
    expected_updates = 119 if window["years"] == 10 else 59
    rules.require(
        feedback["complete_month_updates"] == expected_updates,
        "Do not train the initial partial month or import a precapital main context.",
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
        reference(raw_path),
        reference(independent_path),
        *[reference(p) for p in sorted(comparisons.iterdir())],
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
        "actual_capital_learning_and_accounting_verification": feedback,
        "path_local_learning_initialization": True,
        "confidence_is_forced_defensive_capital": False,
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


def prefix_audits(data, bundle):
    folder = FOLDER / "capital-learning-prefix-audits-v1"
    new_output_directory(folder)
    source_folder = FOLDER / "accounts/10y-base"
    saved = {
        name: pd.read_csv(
            source_folder / name,
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    decisions = read_json(source_folder / "decisions.json")["decisions"]
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
            bundle,
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
        for name, frame in frames:
            expected = saved[name].loc[:day]
            rules.require(
                frame.loc[:day].index.equals(expected.index)
                and frame.columns.equals(expected.columns)
                and np.array_equal(
                    frame.loc[:day].to_numpy(dtype=float),
                    expected.to_numpy(dtype=float),
                    equal_nan=True,
                )
                and np.array_equal(
                    np.signbit(frame.loc[:day].to_numpy(dtype=float)),
                    np.signbit(expected.to_numpy(dtype=float)),
                ),
                "A coupled main/learner prefix changed a prior actual capital coordinate.",
            )
        rules.require(
            output.decisions == expected_decisions,
            "A coupled main prefix changed a prior confidence,learning state or target.",
        )
        results.append(
            {
                "last_session": str(day.date()),
                "formal_return_position": position,
                "all_main_capital_learning_confidence_and_target_prefixes_exact": True,
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


record_failure = bound(template.record_failure)


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not RESULTS.exists()
            and not events("technical_failure")
            and not events("review")
            and not events("path_started"),
            "Do not replay completed,failed or partial actual-main confidence accounts.",
        )
        data, bundle, frozen, benchmarks = inputs()
        published = publication(frozen, commit)
        start = read_json(FOLDER / "actual-funding-start-v1.json")
        checkpoint = start["actual_continuity_checkpoint_readback"]
        rules.require(
            checkpoint["saved_at"] < start["started_at"]
            and start["actual_preregistration_commit"] == commit,
            "Actual prefunding checkpoint/publication must precede coupled main learning.",
        )
        base.append_event(
            base.JOURNAL,
            "publication",
            None,
            {
                "study_id": rules.STUDY,
                "verification": published,
                "candidate_accounts_previously_started": False,
                "historical_new_meta_outputs_previously_computed": False,
            },
        )
        try:
            paths = {}
            protocol = read_json(base.PROTOCOL)
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, bundle, frozen, window, scenario, benchmarks)
            audits = prefix_audits(data, bundle)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": (
                        "actual_main_context_participation_learning_and_capital_prefixes_v1"
                    ),
                    "proof_stage": "after_four_actual_published_program_paths",
                    "new_main_capital_prefix_audit": audits,
                    "all_original_formal_dates_and_same_policy_parameters_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {k: reference(FOLDER / "accounts" / k / "complete.json") for k in paths},
                "all_four_paths_same_candidate_and_frozen_parameters": True,
                "independent_forward_validation": False,
                "order_authority": False,
            }
            base.append_event(base.JOURNAL, "review", ID, review)
            write_new_receipt(
                RESULTS,
                {
                    "completed_at": utc_now(),
                    "frozen": frozen,
                    "publication": published,
                    "review": review,
                    "paths": paths,
                    "new_main_capital_prefix_audit": audits,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_actual_full_results_without_old_replay")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(set(saved["paths"]) == base.PATH_KEYS, "Retain all four actual paths.")
        for key, ref in saved["review"]["paths"].items():
            base.verify_path(ref, saved["frozen"], ID, key)
        fields = (
            "path_key",
            "metrics",
            "execution",
            "gates",
            "all_point_gates_pass",
            "independent_bt_max_equity_difference_usd",
            "actual_capital_learning_and_accounting_verification",
            "path_local_learning_initialization",
            "confidence_is_forced_defensive_capital",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "actual_main_risk_confidence_Prod_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "paths": {k: {name: v[name] for name in fields} for k, v in saved["paths"].items()},
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "new_policy_auxiliary_prefix_runs": 5,
                "all_real_prefix_audits_exact": True,
                "source_wealth_guarantee_transferred_to_main": False,
                "path_local_5y_initialization_differs_from_old_warm_source_control": True,
                "isolated_confidence_causal_effect_claimed": False,
                "old_source_policy_shadow_or_master_replayed": False,
                "raw_prices_targets_holdings_daily_books_or_learning_rows_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly actual-main risk confidence research.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published dynamic-program preregistration SHA.")
    value = (
        evaluate(args.commit)
        if args.action == "evaluate"
        else {
            "freeze": freeze,
            "register": register,
            "export": export,
        }[args.action]()
    )
    print(json.dumps(value, indent=2, allow_nan=False))
