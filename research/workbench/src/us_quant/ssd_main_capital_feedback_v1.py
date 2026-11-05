from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import main_capital_feedback_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as source
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_variance_disagreement_research_v1 import verify_feedback
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT, ID = base.ROOT, rules.ID
FOLDER = base.REPORTS / "main-capital-feedback-20261104-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/main_capital_feedback_20261104_registration_v1.json"
PUBLIC = ROOT / "evidence/main_capital_feedback_20261104_results_v1.json"
KIND = "causal_main_capital_decision_program_v1"
OWN = (
    "config/main-capital-feedback-v1.json",
    "src/us_quant/main_capital_feedback_v1.py",
    "src/us_quant/ssd_main_capital_feedback_v1.py",
    "tests/test_main_capital_feedback_v1.py",
    "tests/test_ssd_main_capital_feedback_v1.py",
    "MAIN_CAPITAL_FEEDBACK_V1.md",
    "config/main-capital-cost-estimate-v1.json",
    "src/us_quant/main_capital_cost_estimate_v1.py",
    "tests/test_main_capital_cost_estimate_v1.py",
    "MAIN_CAPITAL_COST_ESTIMATE_V1.md",
)
reference, unique_refs = base.reference, source.unique_refs


def events(kind):
    return [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == kind and e["candidate_id"] == ID
    ]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("A dynamic study cannot override authoritative journal totals.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "main_capital_feedback_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def validate_policy(policy):
    expected = {
        "schema_version": 1,
        "study_id": rules.STUDY,
        "candidate_id": ID,
        "candidate_count": 1,
        "study_size_is_a_total_research_quota": False,
        "prior_journal_sequence": 910,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "prior_completed_configurations": 254,
        "packet_kind": KIND,
        "capital_usd_per_formal_path": 10000.0,
        "initial_drawdown_peak_usd": 10000.0,
        "initialization": "fresh_path_local_learning_at_real_capital_anchor",
        "ordinary_decisions": "true_NYSE_month_end_after_admitted_16_15_ET_clock",
        "initialization_difference_from_cached_five_year_control_disclosed": True,
        "comparison_is_isolated_cost_input_causal_effect": False,
        "historical_target_proof_occurs_after_published_program_execution": True,
        "historical_meta_outputs_computed_before_publication": False,
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "selection_score_decimal_places": 12,
        "amortization_sessions": 21,
        "assessment_cost_bps_per_side": 5.0,
        "assessment_fixed_model_ticket_usd": 1.0,
        "assessment_convention_same_in_base_and_stress": True,
        "actual_weight_band": 0.025,
        "target_budget": 0.98,
        "formal_paths": ["10y-base", "10y-stress", "5y-base", "5y-stress"],
        "prefix_audit_ten_year_base_return_positions": [125, 503, 1006, 1509, 2012],
        "prefix_replays_are_independent_economic_trials": False,
        "new_source_requests": 0,
        "new_source_policy_generator_or_shadow_account_runs": 0,
        "new_economic_factor_definitions": 0,
        "new_expert_admissions": 0,
        "global_expert_rosters_changed": False,
        "new_RF_or_source_assessment_Adapter": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "broker_orders": 0,
        "order_authority": False,
    }
    refs = {
        "source_expert_pool_policy",
        "cached_original_eight_context_results",
        "cost_Module_preparation_closure",
    }
    if set(policy) != set(expected) | refs or any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in expected.items()
    ):
        raise QuantError(
            "Keep the one frozen dynamic law, real anchors and original financial terms."
        )
    if policy["source_expert_pool_policy"] != reference(source.rules.POLICY):
        raise QuantError("Declare the actual unchanged original-eight source pool.")
    for key in refs:
        base.checked_reference(policy[key])


def packages():
    return sorted((d.metadata["Name"], d.version) for d in distributions())


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for row in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(row)


def validation_refs():
    xml = FOLDER / "study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if not suites or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Complete the dynamic Interface, chronology and publication-fence tests.")
    proof = read_json(FOLDER / "validation-source-v1.json")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["ruff_returncode"] != 0
        or proof["synthetic_only"] is not True
    ):
        raise QuantError("Validation must match exact current code and synthetic inputs.")
    return [
        reference(FOLDER / p)
        for p in (
            "validation-source-v1.json",
            "study-tests-v1.xml",
            "study-tests-v1.log",
            "study-ruff-v1.log",
        )
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial original dynamic-program freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        validate_policy(policy)
        data, experts, parent, refs, clocks, _ = source.source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 910 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError(
                "The pre-outcome dynamic study requires the actual closed910 baseline."
            )
        ready = read_json(FOLDER / "source-input-readiness-v1.json")
        if (
            refs != ready["source_references"]
            or clocks != ready["source_introduction_and_information_clocks"]
            or len(experts) != 8
            or len(data.close) != 2805
            or len(data.close.columns) != 14
        ):
            raise QuantError("Keep the original eight source identities, clocks and full market.")
        deps = unique_refs([*source.dependencies(), *[reference(ROOT / p) for p in OWN]])
        all_refs = [*deps, *refs]
        code = [
            r
            for r in all_refs
            if r["path"].startswith(("src/", "tests/"))
            or ("/" not in r["path"] and r["path"].endswith(".md"))
        ]
        frozen = {
            **parent,
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "expert_pool": policy["source_expert_pool_policy"],
            "construction": "known_main_capital_close_mark_cost_feedback_original_eight",
            "decision_program_kind": KIND,
            "prepublication_readiness": "synthetic_Interface_and_pinned_archived_inputs_only",
            "historical_target_proof_stage": "after_actual_published_program_execution",
            "main_capital_initialization": policy["initialization"],
            "prior_journal_sequence": 910,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "code": unique_refs([*parent["code"], *code]),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in all_refs if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *refs,
                    policy["cost_Module_preparation_closure"],
                    policy["cached_original_eight_context_results"],
                    reference(FOLDER / "source-input-readiness-v1.json"),
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "design-decision-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "decision_information_clock": rules.pool.CLOCK,
            "new_source_requests": 0,
            "new_source_model_or_shadow_account_runs": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "package_versions": packages(),
                "historical_new_meta_scores_targets_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_dynamic_program_without_historical_outputs")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [list(row) for row in packages()]:
        raise QuantError("The frozen isolated source environment changed.")
    data, experts, _, refs, clocks, benchmarks = source.source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Dynamic execution cannot replace any original admitted source input.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def packet_body(frozen, registered, readiness):
    return {
        "schema_version": 3,
        "program_id": "ssd_continuous_research_v1",
        "study_id": rules.STUDY,
        "target_delivery_kind": KIND,
        "complete_fixed_cohort_ids": [ID],
        "cohort_registrations": [registered["sha256"]],
        "entries": [
            {
                "candidate_id": ID,
                "specification": {
                    "financial_criteria_id": "ssd_dual_benchmark_net_cagr_v1",
                    "decision_program_kind": KIND,
                    "frozen": frozen,
                },
                "registered_event_sha256": registered["sha256"],
                "registered_at": registered["at"],
                "program_readiness": readiness,
                "historical_target_derivation_proof": None,
            }
        ],
        "historical_new_meta_scores_or_targets_computed": False,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "historical_target_proof_deferred_until_published_execution": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def checked_packet(frozen):
    registered = events("registered")
    if len(registered) != 1 or registered[0]["body"] != {
        "candidate": {"id": ID, "weighting": "original_rolling_pair_close_mark_main_capital_cost"},
        "frozen": frozen,
    }:
        raise QuantError("Require the one actual unique dynamic-law registration.")
    body = read_json(PACKET)
    expected = packet_body(frozen, registered[0], reference(READY))
    if {k: v for k, v in body.items() if k != "created_at"} != expected:
        raise QuantError("Publish the complete explicit dynamic program, not a fake target proof.")
    ready = read_json(READY)
    if (
        ready["readiness_kind"] != "synthetic_Interface_and_archived_input_identity_only"
        or ready["historical_new_meta_scores_targets_accounts_computed"] is not False
        or ready["historical_target_derivation_verified"] is not False
    ):
        raise QuantError(
            "Synthetic readiness cannot be laundered into real historical target proof."
        )
    return body


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        _, _, frozen, _ = inputs()
        if events("registered") or events("technical_failure") or PACKET.exists():
            raise QuantError("Recover the retained dynamic registration; never register it twice.")
        registered = base.append_event(
            base.JOURNAL,
            "registered",
            ID,
            {
                "candidate": {
                    "id": ID,
                    "weighting": "original_rolling_pair_close_mark_main_capital_cost",
                },
                "frozen": frozen,
            },
        )
        write_new_receipt(
            READY,
            {
                "checked_at": utc_now(),
                "readiness_kind": "synthetic_Interface_and_archived_input_identity_only",
                "implementation": reference(INTENT),
                "validation": validation_refs(),
                "source_input_readiness": reference(FOLDER / "source-input-readiness-v1.json"),
                "historical_new_meta_scores_targets_accounts_computed": False,
                "historical_target_derivation_verified": False,
                "future_price_or_main_account_simulation_used": False,
            },
        )
        write_new_receipt(
            PACKET, {**packet_body(frozen, registered, reference(READY)), "created_at": utc_now()}
        )
        checked_packet(frozen)
        progress("registered_before_outputs", next_step="publish_dynamic_program_before_funding")
    return reference(PACKET)


def publication(frozen, commit):
    checked_packet(frozen)
    verifier = FunctionType(
        base.published_inputs.__code__,
        {**base.published_inputs.__globals__, "PACKET": PACKET},
        argdefs=base.published_inputs.__defaults__,
    )
    return verifier(frozen, commit)


def write_run(directory, result):
    frames = {
        "account.csv": result.result.frame,
        "weights.csv": result.result.weights,
        "desired.csv": result.desired,
        "issued.csv": result.issued,
    }
    for name, frame in frames.items():
        write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "decisions.json",
        {"decisions": result.decisions, "pending_after_window": result.pending_after_window},
    )
    return [reference(directory / name) for name in (*frames, "decisions.json")]


def evaluate_path(data, experts, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    complete = [e for e in events("path_completed") if e["body"]["path_key"] == key]
    if complete:
        if len(complete) != 1:
            raise QuantError("A dynamic path cannot select among repeated completed accounts.")
        return base.verify_path(complete[0]["body"]["receipt"], frozen, ID, key)
    if any(e["body"]["path_key"] == key for e in events("path_started")):
        raise QuantError("Preserve a partial dynamic account; do not silently replay it.")
    fenced = [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == "publication" and e["body"].get("study_id") == rules.STUDY
    ]
    if len(fenced) != 1 or not events("registered") or events("technical_failure"):
        raise QuantError("No historical dynamic output before actual registration and publication.")
    lifecycle.runtime_guard()
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
    trace = directory / "chronological-trace.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        output = rules.run(
            data,
            rules.CloseMarkedPairPolicy(experts, tuple(data.close.columns), data.risk_free),
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    feedback = verify_feedback(
        data, output.desired, output.result, output.issued, output.decisions, window, scenario
    )
    capital_inputs = rules.verify_capital_inputs(output)
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
    own = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        output.result.frame, independent, data.risk_free.loc[output.result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], symbol] for symbol in ("SPY", "QQQ"))
    criteria = base.gates(own, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    statistics = base.execution_statistics(
        output.result, data, 10000, scenario["cost_bps_per_side"], 1
    )
    artifacts = write_run(directory, output)
    write_text_atomic(directory / "independent-bt.csv", independent.to_csv(float_format="%.17g"))
    artifacts.extend([reference(trace), reference(directory / "independent-bt.csv")])
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "decision_program_kind": KIND,
        "metrics": own,
        "execution": statistics,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"] - spy["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"] - qqq["cagr"],
        "independent_bt_max_equity_difference_usd": difference,
        "independent_monthly_feedback": feedback,
        "actual_capital_input_verification": capital_inputs,
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


def prefix_audits(data, experts):
    folder = FOLDER / "main-capital-prefix-audits-v1"
    new_output_directory(folder)
    source_folder = FOLDER / "accounts/10y-base"
    frames = {
        name: pd.read_csv(
            source_folder / name, index_col=0, parse_dates=True, float_precision="round_trip"
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    decisions = read_json(source_folder / "decisions.json")["decisions"]
    dates = frames["account.csv"].index
    results = []
    for position in read_json(rules.POLICY)["prefix_audit_ten_year_base_return_positions"]:
        lifecycle.runtime_guard()
        day = dates[position]
        prefix = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        output = rules.run(
            prefix,
            rules.CloseMarkedPairPolicy(experts, tuple(prefix.close.columns), prefix.risk_free),
            "2016-10-06",
            str(day.date()),
            cost_bps=5,
            delay=1,
            before_decision=lifecycle.runtime_guard,
        )
        for name, frame in (
            ("account.csv", output.result.frame),
            ("weights.csv", output.result.weights),
            ("desired.csv", output.desired),
            ("issued.csv", output.issued),
        ):
            actual, expected = frame.loc[:day], frames[name].loc[:day]
            if (
                not actual.index.equals(expected.index)
                or not actual.columns.equals(expected.columns)
                or not np.array_equal(
                    actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True
                )
            ):
                raise QuantError("A real main-capital prefix differs from its published full run.")
        expected_decisions = [
            row for row in decisions if pd.Timestamp(row["signal_session"]) <= day
        ]
        if output.decisions != expected_decisions:
            raise QuantError("A real dynamic prefix changed an earlier source/selection decision.")
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        results.append(
            {
                "last_session": str(day.date()),
                "formal_return_position": position,
                "capital_and_all_decisions_exact": True,
                "artifacts": write_run(directory, output),
            }
        )
    path = folder / "completion.json"
    write_new_receipt(
        path,
        {
            "completed_at": utc_now(),
            "version": "main_capital_prefix_audit_v1",
            "prefixes": results,
            "auxiliary_main_book_audit_runs": 5,
            "extra_economic_rules_or_qualifying_windows": 0,
            "all_runs_after_actual_preregistration_publication": True,
            "source_Interface_scope_not_expanded": True,
        },
    )
    return reference(path)


def record_failure(stage, error):
    interrupted = lifecycle.phase(datetime.now(timezone.utc)) != "research"
    if events("registered") and not events("review") and not events("technical_failure"):
        if not interrupted:
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                ID,
                {"stage": stage, "reason": str(error), "partial_outputs_retained": True},
            )
    path = FOLDER / f"{stage}-failure-v1.json"
    if not path.exists():
        write_new_receipt(
            path,
            {
                "failed_at": utc_now(),
                "stage": stage,
                "reason": str(error),
                "authorization_interrupted": interrupted,
                "partial_outputs_retained": True,
            },
        )
    progress(
        "interrupted" if interrupted else "technical_failure_preserved",
        failure=str(error),
        next_step="preserve_terminal_or_interrupted_identity_without_retuning",
    )


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal dynamic-program failure without retry.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("Completed dynamic results cannot be changed on recovery.")
            for key, row in saved["review"]["paths"].items():
                base.verify_path(row, frozen, ID, key)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            e
            for e in base.journal_events(base.JOURNAL)
            if e["kind"] == "publication" and e["body"].get("study_id") == rules.STUDY
        ]
        if not fences:
            if events("path_started") or events("target_proof"):
                raise QuantError("No dynamic historical output may predate its publication fence.")
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
        elif len(fences) != 1 or fences[0]["body"]["verification"]["published_commit"] != commit:
            raise QuantError("Keep the unique original actual dynamic-program publication fence.")
        try:
            paths = {}
            protocol = read_json(base.PROTOCOL)
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, experts, frozen, window, scenario, benchmarks)
            audits = prefix_audits(data, experts)
            if events("target_proof"):
                raise QuantError("The actual dynamic target proof already exists; preserve it.")
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "actual_published_dynamic_outputs_and_main_capital_prefixes_v1",
                    "proof_stage": "after_four_actual_published_program_paths",
                    "issued_and_desired_outputs": [
                        row
                        for result in paths.values()
                        for row in result["artifacts"]
                        if row["path"].endswith(("/issued.csv", "/desired.csv", "/decisions.json"))
                    ],
                    "real_main_capital_prefix_audit": audits,
                    "all_original_formal_dates_and_same_policy_parameters_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {
                    key: reference(FOLDER / "accounts" / key / "complete.json") for key in paths
                },
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
                    "real_main_capital_prefix_audit": audits,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_all_actual_results_without_source_replay")
    return reference(RESULTS)


def export():
    data = read_json(RESULTS)
    if set(data["paths"]) != base.PATH_KEYS or len(events("review")) != 1:
        raise QuantError("Export only the complete actually reviewed dynamic study.")
    for key, ref in data["review"]["paths"].items():
        base.verify_path(ref, data["frozen"], ID, key)
    controls = read_json(
        base.checked_reference(read_json(rules.POLICY)["cached_original_eight_context_results"])
    )
    body = {
        "schema_version": 1,
        "record_type": "november_actual_main_capital_feedback_four_path_results",
        "study_id": rules.STUDY,
        "candidate_id": ID,
        "actual_completed_at": data["completed_at"],
        "status": data["review"]["status"],
        "registration": reference(PACKET),
        "frozen_specification_sha256": digest_json(data["frozen"]),
        "private_results_sha256": file_digest(RESULTS),
        "actual_preregistration": data["publication"],
        "target_delivery_kind": KIND,
        "paths": {
            key: {
                name: row[name]
                for name in (
                    "path_key",
                    "metrics",
                    "execution",
                    "gates",
                    "all_point_gates_pass",
                    "independent_bt_max_equity_difference_usd",
                    "independent_monthly_feedback",
                    "actual_capital_input_verification",
                )
            }
            for key, row in data["paths"].items()
        },
        "cached_context_controls_not_replayed": {
            name: {
                "status": row["status"],
                "paths": {
                    key: {"metrics": path["metrics"], "execution": path["execution"]}
                    for key, path in row["paths"].items()
                },
            }
            for name, row in controls["candidates"].items()
        },
        "path_local_initialization_difference_from_cached_five_year_control_disclosed": True,
        "comparison_identifies_isolated_cost_input_causal_effect": False,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "new_financial_rules_evaluated": 1,
        "new_formal_funded_paths": 4,
        "separately_versioned_auxiliary_main_book_prefix_runs": 5,
        "all_real_prefix_audits_exact": True,
        "new_source_requests": 0,
        "source_policy_generators_or_accounts_replayed": False,
        "raw_source_prices_targets_daily_books_private_intents_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_new_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Explicit dynamic main-capital research program.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published dynamic-program preregistration SHA.")
    if args.action == "evaluate":
        result = evaluate(args.commit)
    else:
        result = {"freeze": initialize, "register": register, "export": export}[args.action]()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
