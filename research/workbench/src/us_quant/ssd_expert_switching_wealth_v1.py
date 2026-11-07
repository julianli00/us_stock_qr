from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_switching_wealth_v1 as rules
from us_quant import ssd_main_risk_confidence_prod_v1 as template
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import review_market
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
CANDIDATE = {"id": ID, "weighting": "Jeffreys_universal_complete_expert_identity_sequences"}
FOLDER = base.REPORTS / "expert-switching-wealth-20261107-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/expert_switching_wealth_20261107_registration_v1.json"
PUBLIC = ROOT / "evidence/expert_switching_wealth_20261107_results_v1.json"
OWN = (
    "config/expert-switching-wealth-v1.json",
    "src/us_quant/expert_switching_wealth_v1.py",
    "src/us_quant/ssd_expert_switching_wealth_v1.py",
    "tests/test_expert_switching_wealth_v1.py",
    "EXPERT_SWITCHING_WEALTH_V1.md",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals()},
        argdefs=function.__defaults__,
    )


def dependencies():
    return bound(template.template.framework._dependencies)()


events = bound(template.events)
check_frozen = bound(template.check_frozen)
packet_body = bound(template.packet_body)


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "A switching study cannot overwrite authoritative financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "expert_switching_wealth_" + phase,
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
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    registered = read_json(base.checked_reference(catalogue["source_registered_Prod_event"]))
    rules.require(
        closed["actual_authoritative_counts"]["current_journal_event_count"] == 1518
        and registered["kind"] == "registered"
        and registered["candidate_id"] == "expert_pool_six_confidence_rated_prod_v1"
        and catalogue["expert_ids"] == list(rules.EXPERT_IDS)
        and catalogue["mechanism_groups"] == list(rules.MECHANISMS)
        and catalogue["source_clocks"]
        == registered["body"]["frozen"]["source_policy_and_expert_admission_clocks"]
        and catalogue["new_switching_forecasts_learning_targets_or_accounts_computed"] is False
        and catalogue["old_source_learner_weights_update_generator_or_account_replayed"] is False
        and catalogue["old_fixed_policy_integral_repaired_or_reexecuted"] is False,
        "Keep the exact admitted six,source clocks and preserved J1518 closure.",
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
            *catalogue["fixed_policy_mixture_prior_art"],
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
    return catalogue, registered["body"]["frozen"], refs


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
        "Bind successful source-sequence,clock and exact-accounting synthetic validation.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_all119_months_match_independent_hazard_integration",
            "test_next_identity_not_posterior_of_matured_identity",
            "test_full_mixed_asset_cash_and_BT",
            "test_actual_packet_entry_defers_source_learning",
            "test_old_policy_integrals_and_learners_are_not_replayed",
        }.issubset(names),
        "Exercise actual mathematical and funded decision-program seams.",
    )
    first = read_json(FOLDER / "validation-source-v1.json")
    rules.require(
        first["exit_code"] == 1
        and first["passed_tests"] == 35
        and first["junit_totals"] == {"tests": 36, "failures": 1, "errors": 0, "skipped": 0},
        "Retain the first forbidden-entrypoint fixture's missing old oracle-name failure.",
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
            "Preserve the first complete or partial switching-program freeze.",
        )
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1518
            and history[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(e["candidate_id"] == ID for e in history),
            "Freeze one unused sequence-mixture identity at the unchanged J1518 head.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, parent, refs = source_inputs()
        deps, validation = dependencies(), validation_refs()
        old_method_fields = {
            "confidence_for_loss_fixed_at_prior_month_origin",
            "mixability_gap_floor",
            "negative_Jensen_gap_clipped",
            "normalized_confidence_regret_is_main_wealth",
            "normalized_source_regret_is_main_wealth",
        }
        frozen = {
            **{k: v for k, v in parent.items() if k not in old_method_fields},
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": (
                "new_switching_source_forecasts_original_monthly_main_execution"
            ),
            "expert_pool_definition_role": (
                "unchanged_six_complete_policies_switching_sequence_prior"
            ),
            "construction": "Jeffreys_integrated_source_expert_identity_sequence_wealth",
            "decision_program_kind": KIND,
            "declared_financial_cohort_size": 1,
            "decision_information_clock": "matured_source_month_then_NEXTidentity_forecast",
            "prior_journal_sequence": 1518,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 305,
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
            "shared_source_learning_not_actual_main_capital": True,
            "source_game_score_is_main_NET_wealth": False,
            "latent_switch_rate_is_market_probability_or_turnover": False,
            "old_fixed_policy_integral_repaired_or_reexecuted": False,
            "old_learner_source_generator_shadow_or_account_replayed": False,
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
                "new_switching_forecasts_learning_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_source_learning")
    return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(ref)
    rules.require(
        sys.version == intent["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == intent["package_versions"],
        "Keep the exact frozen interpreter and accepted package set.",
    )
    catalogue, _, refs = source_inputs()
    rules.require(
        refs == intent["source_references"], "Every captured source reference must remain."
    )
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": catalogue["source_market"]}, ROOT)
    benchmark_refs = [
        r for r in catalogue["source_benchmark_artifacts"] if r["path"].endswith("/results.json")
    ]
    rules.require(len(benchmark_refs) == 1, "Use the original unique completed benchmark set.")
    benchmarks = read_json(base.checked_reference(benchmark_refs[0]))
    rules.require(len(benchmarks["benchmarks"]) == 8, "Keep both benchmarks on all original paths.")
    return data, catalogue, frozen, benchmarks


def checked_packet(frozen):
    return bound(template.template.framework.checked_packet)(frozen)


register = bound(template.register)
publication = bound(template.publication)
record_failure = bound(template.record_failure)


def write_run(directory, output):
    return bound(template.template.write_run)(directory, output)


def retain_cash_comparison(directory, actual, replay, report):
    return bound(template.template.retain_cash_comparison)(directory, actual, replay, report)


class SavedOpinionFeed:
    def __init__(self, catalogue):
        self.rows = {}
        counter = 0
        with base.checked_reference(catalogue["cached_complete_policy_opinions"]).open() as stream:
            while line := stream.readline():
                before = json.loads(line)
                after_line = stream.readline()
                rules.require(bool(after_line), "Retain complete original opinion record pairs.")
                after = json.loads(after_line)
                day = pd.Timestamp(before["decision_session"])
                rules.require(
                    before["phase"] == "before_decision"
                    and after["phase"] == "after_decision"
                    and before["decision_session"] == after["decision_session"]
                    and day not in self.rows,
                    "Use only original opinion fields,not a future or recomputed source.",
                )
                self.rows[day] = rules.OpinionSnapshot(
                    day,
                    np.array(
                        [[float.fromhex(v) for v in row] for row in before["known_targets_hex"]]
                    ),
                    tuple(pd.Timestamp(d) for d in before["known_target_dates"]),
                )
                counter += 2
        rules.require(
            counter == 242 and len(self.rows) == 121 and list(self.rows) == sorted(self.rows),
            "Keep all121causal complete-policy opinion snapshots without old learner replay.",
        )

    def observe(self, day):
        rules.require(day in self.rows, "A missing native source snapshot cannot be filled.")
        return self.rows[day]


def prepare_source_targets(data, catalogue):
    rules.require(
        not (FOLDER / "source-targets-started-v1.json").exists(),
        "Do not replay a partial or complete new switching source learner.",
    )
    write_new_receipt(
        FOLDER / "source-targets-started-v1.json",
        {
            "started_at": utc_now(),
            "actual_registration": reference(PACKET),
            "new_source_learning_previously_started": False,
        },
    )
    periods = read_json(base.checked_reference(catalogue["cached_six_source_month_audits"]))[
        "periods"
    ]
    directory = FOLDER / "source-targets"
    new_output_directory(directory)
    trace, counter = directory / "input-resolution.jsonl", [0]
    with trace.open("x") as stream:

        def retain(row):
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            stream.flush()
            counter[0] += 1

        prepared = rules.build_targets(
            data.close.index,
            SavedOpinionFeed(catalogue),
            periods,
            before=lifecycle.runtime_guard,
            retain=retain,
        )
    rules.require(
        prepared.proof["source_decisions"] == 121
        and prepared.proof["complete_source_intervals"] == 119
        and counter[0] == 242,
        "Retain every new source forecast and complete original learning interval.",
    )
    write_text_atomic(directory / "targets.csv", prepared.targets.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "audits.json", {"audits": prepared.audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "completed_at": utc_now(),
            **prepared.proof,
            "input_resolution_records": counter[0],
            "retained_records": reference(trace),
            "new_main_accounts_previously_started": False,
            "order_authority": False,
        },
    )
    return prepared


def source_prefixes(data, catalogue, prepared):
    folder = FOLDER / "source-prefix-audits-v1"
    new_output_directory(folder)
    periods = read_json(base.checked_reference(catalogue["cached_six_source_month_audits"]))[
        "periods"
    ]
    results = []
    for position in read_json(rules.POLICY)["prefix_positions"]:
        lifecycle.runtime_guard()
        day = data.close.loc["2016-10-06":].index[position]
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        prefix = rules.build_targets(
            data.close.loc[:day].index,
            SavedOpinionFeed(catalogue),
            periods,
            before=lifecycle.runtime_guard,
        )
        expected_audits = [a for a in prepared.audits if pd.Timestamp(a["decision_session"]) <= day]
        for name, frame in (
            ("actual-targets.csv", prefix.targets),
            ("expected-targets.csv", prepared.targets.loc[:day]),
        ):
            write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
        write_new_receipt(directory / "actual-audits.json", {"audits": prefix.audits})
        write_new_receipt(directory / "expected-audits.json", {"audits": expected_audits})
        exact = (
            prefix.targets.index.equals(prepared.targets.loc[:day].index)
            and np.array_equal(prefix.targets, prepared.targets.loc[:day], equal_nan=True)
            and np.array_equal(np.signbit(prefix.targets), np.signbit(prepared.targets.loc[:day]))
            and prefix.audits == expected_audits
        )
        write_new_receipt(
            directory / "comparison.json",
            {
                "last_session": str(day.date()),
                "source_forecasts_targets_and_states_exact": exact,
                "both_operands_retained_before_assertion": True,
                "artifacts": [reference(p) for p in sorted(directory.iterdir())],
            },
        )
        rules.require(exact, "A future source label or opinion changed an earlier new forecast.")
        results.append(reference(directory / "comparison.json"))
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefixes": results,
            "new_economic_trials": 0,
            "old_learner_or_source_replayed": False,
        },
    )
    return reference(folder / "completion.json")


def evaluate_path(data, prepared, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    rules.require(
        not any(e["body"]["path_key"] == key for e in events("path_started")),
        "Never replay a complete or partial switching-funded account.",
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
    comparison_dir = directory / "cash-comparison"
    new_output_directory(comparison_dir)
    trace = directory / "chronological-trace.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        output = rules.run(
            data,
            prepared,
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
    checked = rules.verify_run(
        data,
        prepared,
        output,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
        retain_comparison=lambda actual, replay, report: retain_cash_comparison(
            comparison_dir,
            actual,
            replay,
            report,
        ),
    )
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
    metrics = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    other, difference = base.independent_metrics(
        output.result.frame,
        independent,
        data.risk_free.loc[output.result.frame.index],
        10000,
    )
    base.check_metrics(other, metrics)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], s] for s in ("SPY", "QQQ"))
    gates = base.gates(
        metrics,
        spy,
        qqq,
        lookup[window["years"], "base", "SPY"],
        scenario["id"],
    )
    artifacts += [
        reference(trace),
        reference(raw_path),
        reference(independent_path),
        *[reference(p) for p in sorted(comparison_dir.iterdir())],
    ]
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "decision_program_kind": KIND,
        "metrics": metrics,
        "gates": gates,
        "all_point_gates_pass": all(gates.values()),
        "execution": base.execution_statistics(
            output.result,
            data,
            10000,
            scenario["cost_bps_per_side"],
            1,
        ),
        "independent_bt_max_equity_difference_usd": difference,
        "actual_capital_and_source_selection_verification": checked,
        "source_forecast_proof": reference(FOLDER / "source-targets/proof.json"),
        "source_game_score_is_main_NET_wealth": False,
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


def capital_prefixes(data, prepared):
    folder = FOLDER / "capital-prefix-audits-v1"
    new_output_directory(folder)
    source_folder = FOLDER / "accounts/10y-base"
    saved = {
        name: pd.read_csv(
            source_folder / name, index_col=0, parse_dates=True, float_precision="round_trip"
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
        prefix_source = rules.PreparedSourceTargets(
            prepared.targets.loc[:day],
            [a for a in prepared.audits if pd.Timestamp(a["decision_session"]) <= day],
            prepared.proof,
        )
        output = rules.run(
            clipped,
            prefix_source,
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
        write_new_receipt(directory / "expected-decisions.json", {"decisions": expected_decisions})
        artifacts.append(reference(directory / "expected-decisions.json"))
        exact = output.decisions == expected_decisions
        for name, frame in frames:
            expected = saved[name].loc[:day]
            exact = exact and (
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
                )
            )
        write_new_receipt(
            directory / "comparison.json",
            {
                "last_session": str(day.date()),
                "all_capital_and_saved_source_decisions_exact": exact,
                "both_operands_retained_before_assertion": True,
                "artifacts": artifacts,
            },
        )
        rules.require(exact, "A future source forecast changed a prior actual main-capital prefix.")
        results.append(reference(directory / "comparison.json"))
    write_new_receipt(
        folder / "completion.json",
        {
            "completed_at": utc_now(),
            "prefixes": results,
            "new_economic_trials": 0,
        },
    )
    return reference(folder / "completion.json")


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not RESULTS.exists()
            and not events("technical_failure")
            and not events("review")
            and not events("path_started")
            and not (FOLDER / "source-targets-started-v1.json").exists(),
            "Do not replay a partial or complete source learner or funded switching study.",
        )
        data, catalogue, frozen, benchmarks = inputs()
        published = publication(frozen, commit)
        start = read_json(FOLDER / "actual-funding-start-v1.json")
        checkpoint = start["actual_continuity_checkpoint_readback"]
        rules.require(
            checkpoint["saved_at"] < start["started_at"]
            and start["actual_preregistration_commit"] == commit,
            "An actual remote program and saved checkpoint must precede new source learning.",
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
            prepared = prepare_source_targets(data, catalogue)
            source_audits = source_prefixes(data, catalogue, prepared)
            protocol, paths = read_json(base.PROTOCOL), {}
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, prepared, frozen, window, scenario, benchmarks)
            capital_audits = capital_prefixes(data, prepared)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "switching_source_forecasts_and_all_actual_capital_prefixes_v1",
                    "proof_stage": "after_four_actual_remotely_published_program_paths",
                    "source_forecast_proof": reference(FOLDER / "source-targets/proof.json"),
                    "source_prefix_audits": source_audits,
                    "capital_prefix_audits": capital_audits,
                    "original_financial_contract_and_full_dates_retained": True,
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
                    "source_forecast_proof": reference(FOLDER / "source-targets/proof.json"),
                    "source_prefix_audits": source_audits,
                    "capital_prefix_audits": capital_audits,
                    "review": review,
                    "paths": paths,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("source_learning_funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_all_actual_economic_outcomes")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(
            set(saved["paths"]) == base.PATH_KEYS, "Keep all four actual switching paths."
        )
        for key, ref in saved["review"]["paths"].items():
            base.verify_path(ref, saved["frozen"], ID, key)
        source_proof = read_json(base.checked_reference(saved["source_forecast_proof"]))
        fields = (
            "path_key",
            "metrics",
            "execution",
            "gates",
            "all_point_gates_pass",
            "independent_bt_max_equity_difference_usd",
            "actual_capital_and_source_selection_verification",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "expert_switching_source_wealth_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "source_forecast_verification": {
                    k: v
                    for k, v in source_proof.items()
                    if k not in ("retained_records", "completed_at")
                },
                "paths": {
                    k: {name: value[name] for name in fields} for k, value in saved["paths"].items()
                },
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "new_source_prefix_inspections": 5,
                "new_capital_prefix_runs": 5,
                "all_source_and_capital_prefixes_exact": True,
                "latent_switch_rate_is_market_probability_or_turnover": False,
                "source_game_score_is_actual_main_wealth_or_bound": False,
                "old_fixed_policy_integral_repaired_or_reexecuted": False,
                "old_learner_source_generator_shadow_or_account_replayed": False,
                "raw_prices_daily_books_targets_source_logits_or_states_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete-expert switching sequence research.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published schema3program commit.")
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
