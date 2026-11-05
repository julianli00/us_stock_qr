from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from itertools import zip_longest
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import saved_inclusion_execution_v1 as saved
from us_quant import ssd_research_v1 as base
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

ROOT = base.ROOT
FOLDER = base.REPORTS / "signed-activity-execution-diagnosis-20261105-v1"
POLICY = ROOT / "config/signed-activity-execution-diagnosis-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/signed_activity_execution_diagnosis_20261105_v1.json"
OWN = (
    "config/signed-activity-execution-diagnosis-v1.json",
    "src/us_quant/saved_inclusion_execution_v1.py",
    "src/us_quant/ssd_signed_activity_execution_diagnosis_v1.py",
    "tests/test_saved_inclusion_execution_v1.py",
    "SIGNED_ACTIVITY_EXECUTION_DIAGNOSIS_V1.md",
    "tests/conftest.py",
)


def unique_refs(rows):
    found = {}
    for row in rows:
        if row["path"] in found and found[row["path"]] != row:
            raise QuantError("Do not select conflicting saved source identities.")
        found[row["path"]] = row
    return [found[key] for key in sorted(found)]


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_signed_activity_execution_diagnosis_20261105_v1",
        "prior_journal_sequence": 1144,
        "prior_completed_configurations": 274,
        "input_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "signed-activity-execution-diagnosis-20261105-v1/source-input-catalogue-v1.json"
            ),
            "sha256": "31e1ad93d0c64e17cd123c152bf300262509d62a0b54b051815c882857d3633b",
        },
        "all_original_meta_decisions": 121,
        "paired_main_book_cases": 8,
        "saved_main_books": 16,
        "original_paths": ["10y-base", "10y-stress", "5y-base", "5y-stress"],
        "new_extra_expert_id": "signed_style_activity_participation_v1",
        "money_reconciliation_tolerance_USD": saved.MONEY_TOLERANCE,
        "equal_inclusion_identity_tolerance": saved.TARGET_TOLERANCE,
        "model_ticket_USD": 1.0,
        "issue_groups": ["both_issued", "only_old_issued", "only_new_issued", "neither_issued"],
        "equal_identity": (
            "saved_equal9_equals_eight_ninths_saved_equal8_plus_one_ninth_saved_extra_opinion"
        ),
        "score_selection_and_targets_recomputed": False,
        "matched_capital_seed_is_new_meta_or_source_opinion": False,
        "out_of_window_execution": "explicit_censored_not_zero_fee_success",
        "issued_numeric_noop": "retained_actual_zero_cost_outcome_not_dropped",
        "old_and_new_accounts_use_same_counterfactual_capital": False,
        "cost_difference_is_causal_marginal_expert_effect": False,
        "target_L1_is_executed_turnover": False,
        "new_financial_evaluations": 0,
        "new_source_requests": 0,
        "original_models_targets_accounts_or_benchmarks_replayed": False,
        "financial_gates_source_scope_or_global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Freeze factual participation and actual fee accounting, not a new policy."
        )


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


reference = base.reference


def inputs():
    policy = read_json(POLICY)
    validate(policy)
    catalogue = read_json(base.checked_reference(policy["input_catalogue"]))
    refs = unique_refs([policy["input_catalogue"], *catalogue["source_references"]])
    for ref in refs:
        base.checked_reference(ref)
    closed = read_json(base.checked_reference(catalogue["new_financial_completion"]))
    if (
        closed["actual_authoritative_counts"]["current_journal_event_count"] != 1144
        or closed["new_financial_evaluations"] != 2
        or closed["expert_pool_changed"] is not False
        or catalogue["new_diagnostic_outcomes_computed"] is not False
    ):
        raise QuantError("Consume the exact completed signed-activity cohort, not a fresh run.")
    return catalogue, refs


def progress(phase, **extra):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in extra and extra[k] != v for k, v in counts.items()):
        raise QuantError("A read-only diagnosis cannot override authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "signed_activity_execution_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **extra,
            "new_financial_evaluations_in_this_study": 0,
            "investment_goal_met": False,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first diagnostic freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 171450
            or design["new_diagnostic_outcomes_computed"]
        ):
            raise QuantError("Declare the saved paired-book question before diagnostic outcomes.")
        _, refs = inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or sum(int(s.get("tests", 0)) for s in suites) != validation["passed_tests"]
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Bind the exact synthetic-tested saved-data Interface before outcomes."
            )
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts["current_journal_event_count"] != 1144 or counts["new_pending_candidates"] != 0:
            raise QuantError("Keep the actual J1144/274 pre-diagnosis baseline.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "criteria": read_json(POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "design": reference(FOLDER / "design-decision-v1.json"),
                "invocation": reference(FOLDER / "actual-invocation-v1.json"),
                "preservation_baseline": reference(FOLDER / "preservation-baseline-v1.json"),
                "actual_counts": counts,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_diagnostic_outcomes_computed": False,
                "new_financial_evaluations": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="one_streamed_saved_meta_and_paired_execution_pass")
    return reference(INTENT)


def _pairs(path):
    with path.open() as stream:
        while line := stream.readline():
            second = stream.readline()
            if not second:
                raise QuantError("A missing resolved source record cannot be filled.")
            yield json.loads(line), json.loads(second)


def _frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def analyze():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "analysis-started-v1.json").exists():
        raise QuantError("Preserve the first complete or partial diagnosis, without replay.")
    with lifecycle.research_job():
        intent = read_json(INTENT)
        for ref in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        versions = [
            list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
        ]
        if versions != intent["package_versions"] or read_json(POLICY) != intent["criteria"]:
            raise QuantError("The frozen diagnostic law, code or pinned environment changed.")
        catalogue, refs = inputs()
        if refs != intent["source_references"]:
            raise QuantError("Do not select a changed source inventory after diagnostic outputs.")
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "analysis_intent": reference(INTENT)},
        )
        meta, cases = [], {}
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        try:
            closed = read_json(base.checked_reference(catalogue["new_financial_completion"]))
            financial = read_json(base.checked_reference(closed["financial_freeze"]))
            pool = read_json(base.checked_reference(financial["frozen"]["policy"]))
            identities = {
                "experts": pool["expert_ids"],
                "old8": catalogue["old_rule_ids"],
                "new9": catalogue["new_rule_ids"],
            }
            with trace.open("x") as stream:

                def keep(row):
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                old = _pairs(base.checked_reference(catalogue["traces"]["old8"]))
                new = _pairs(base.checked_reference(catalogue["traces"]["new9"]))
                for position, pair in enumerate(zip_longest(old, new)):
                    lifecycle.runtime_guard()
                    if pair[0] is None or pair[1] is None:
                        raise QuantError("Do not crop unmatched source decisions.")
                    left, right = pair
                    if (
                        left[0]["decision_index"] != position
                        or right[0]["decision_index"] != position
                    ):
                        raise QuantError(
                            "Retain every source decision in its exact original order."
                        )
                    keep(
                        {
                            "record": "matched_meta_input",
                            "position": position,
                            "source_references": catalogue["traces"],
                            "old_equal_hex": [
                                float(x).hex()
                                for x in left[1]["combined_targets"][identities["old8"][0]]
                            ],
                            "new_equal_hex": [
                                float(x).hex()
                                for x in right[1]["combined_targets"][identities["new9"][0]]
                            ],
                            "extra_hex": [
                                float(x).hex() for x in right[0]["observations"][-1]["target"]
                            ],
                        }
                    )
                    result = saved.inspect_meta(left, right, identities)
                    meta.append(result)
                    keep({"record": "matched_meta_resolved", **result})
                if len(meta) != 121:
                    raise QuantError(
                        "Keep all121recorded meta decisions, including unscored origins."
                    )
                dates = pd.DatetimeIndex([r["decision_session"] for r in meta])
                for role, paths in catalogue["books"].items():
                    for key, sides in paths.items():
                        lifecycle.runtime_guard()
                        frames = {side: _frame(row["account"]) for side, row in sides.items()}
                        expected = sessions(
                            "2016-10-06" if key.startswith("10") else "2021-10-06", "2026-10-05"
                        )
                        if any(not f.index.equals(expected) for f in frames.values()):
                            raise QuantError("Every original daily source-book date must remain.")
                        decisions = {
                            side: read_json(base.checked_reference(row["decisions"]))["decisions"]
                            for side, row in sides.items()
                        }
                        keep(
                            {
                                "record": "paired_book_input",
                                "role": role,
                                "path": key,
                                "saved_books": sides,
                                "cost_order_hex": {
                                    side: [
                                        [float(v).hex() for v in values]
                                        for values in f[["cost", "orders"]].to_numpy()
                                    ]
                                    for side, f in frames.items()
                                },
                            }
                        )
                        rows, summary = saved.compare_execution(
                            frames["old8"], frames["new9"], decisions["old8"], decisions["new9"]
                        )
                        for side, row in sides.items():
                            receipt = read_json(base.checked_reference(row["receipt"]))
                            if (
                                abs(
                                    summary[
                                        side.replace("8", "").replace("9", "") + "_total_cost_USD"
                                    ]
                                    - receipt["execution"]["total_cost_usd"]
                                )
                                > saved.MONEY_TOLERANCE
                                or summary[
                                    side.replace("8", "").replace("9", "") + "_model_tickets"
                                ]
                                != receipt["execution"]["model_order_count"]
                            ):
                                raise QuantError(
                                    "Actual fee/ticket totals must match the published receipt."
                                )
                        for row in rows:
                            as_of = pd.Timestamp(row["signal_session"])
                            index = int(dates.searchsorted(as_of, side="right")) - 1
                            if index < 0:
                                raise QuantError(
                                    "A capital seed cannot invent a missing meta opinion."
                                )
                            row["known_meta_opinion_session"] = str(dates[index].date())
                            row["capital_seed_observes_carried_meta_opinion"] = (
                                row["initial_capital"] and dates[index] < as_of
                            )
                            row["recorded_extra_selected_in_known_meta_opinion"] = (
                                meta[index]["new_selected_pair_contains_extra"]
                                if role == "rolling"
                                else None
                            )
                        summary["ordinary_decisions_only_issue_groups"] = {
                            group: sum(
                                r["issue_group"] == group and not r["initial_capital"] for r in rows
                            )
                            for group in intent["criteria"]["issue_groups"]
                        }
                        write_new_receipt(FOLDER / f"{role}-{key}-cases-v1.json", {"cases": rows})
                        cases[role + ":" + key] = summary
                        keep(
                            {
                                "record": "paired_book_resolved",
                                "role": role,
                                "path": key,
                                "summary": summary,
                            }
                        )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_cases": list(cases),
                    "terminal_no_retry_or_repair": True,
                    "new_financial_evaluations": 0,
                },
            )
            progress(
                "terminal_failure_preserved", next_step="do_not_retry_frozen_diagnostic_identity"
            )
            raise
        mature = [r for r in meta if r["new_scored"]]
        output = {
            "completed_at": utc_now(),
            "study_id": read_json(POLICY)["study_id"],
            "analysis_intent": reference(INTENT),
            "actual_input_resolved_records": reference(trace),
            "meta_decisions": len(meta),
            "new_scored_decisions": len(mature),
            "new_unscored_decisions": len(meta) - len(mature),
            "mature_extra_selected_decisions": sum(
                r["new_selected_pair_contains_extra"] for r in mature
            ),
            "mature_extra_selected_with_BIL_opinion": sum(
                r["new_selected_pair_contains_extra"] and r["new_extra_opinion_is_actual_BIL_only"]
                for r in mature
            ),
            "mature_source_pair_same_as_original8": sum(
                r["old_scored"] and r["selected_pairs_exactly_equal"] for r in mature
            ),
            "mature_source_pair_changed_without_extra_selected": sum(
                r["old_scored"]
                and not r["new_selected_pair_contains_extra"]
                and not r["selected_pairs_exactly_equal"]
                for r in mature
            ),
            "maximum_equal_identity_residual": max(r["equal_inclusion_max_residual"] for r in meta),
            "maximum_fraction_equal_identity_residual": max(
                r["independent_fraction_equal_inclusion_max_residual"] for r in meta
            ),
            "paired_main_book_cases": cases,
            "saved_main_books": 16,
            "new_financial_evaluations": 0,
            "new_source_requests": 0,
            "original_sources_scores_targets_accounts_or_benchmarks_replayed": False,
            "cost_difference_is_causal_marginal_expert_effect": False,
            "financial_gates_or_global_pool_changed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_new_receipt(OUTPUT, output)
        write_new_receipt(PUBLIC, output)
        progress(
            "complete_awaiting_publication",
            complete_public_result=reference(PUBLIC),
            next_step="verify_unchanged_sources_publish_only_aggregate_evidence_then_close",
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Read-only saved inclusion/execution diagnosis.")
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze}[args.action]()))


if __name__ == "__main__":
    main()
