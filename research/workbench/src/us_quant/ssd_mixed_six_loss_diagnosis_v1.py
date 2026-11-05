from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import saved_main_book_decomposition_v1 as marks
from us_quant import ssd_calendar_expert_inclusion_v1 as dependency_template
from us_quant import ssd_research_v1 as base
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = base.ROOT
FOLDER = base.REPORTS / "mixed-six-loss-diagnosis-20261105-v1"
POLICY = ROOT / "config/mixed-six-loss-diagnosis-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/mixed_six_loss_diagnosis_20261105_v1.json"
IDS = (
    "expert_pool_mixed_mechanism_six_equal_v1",
    "expert_pool_mixed_mechanism_six_rolling_pair_v1",
)
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
ASSETS = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)
OWN = (
    "config/mixed-six-loss-diagnosis-v1.json",
    "src/us_quant/saved_main_book_decomposition_v1.py",
    "src/us_quant/ssd_mixed_six_loss_diagnosis_v1.py",
    "tests/test_saved_main_book_decomposition_v1.py",
    "MIXED_SIX_LOSS_DIAGNOSIS_V1.md",
    "tests/conftest.py",
)
reference, unique_refs = base.reference, dependency_template.unique_refs


def validate(policy):
    prefix = "reports/ssd-continuous-research-v1/mixed-mechanism-six-financial-20261105-v1/"
    expected = {
        "schema_version": 1,
        "study_id": "ssd_mixed_six_loss_diagnosis_20261105_v1",
        "financial_freeze": {
            "path": prefix + "financial-implementation-freeze-v1.json",
            "sha256": "f6d9d667554ff1205931f760c56d4365c8e81fd41c17bd485223b773cfcfb06c",
        },
        "financial_results": {
            "path": prefix + "results.json",
            "sha256": "523db60447650faeb0a0badf289e13162e73c6621ffd32ff22ecf00968a2732d",
        },
        "financial_completion": {
            "path": prefix + "completion-v1.json",
            "sha256": "f888b9d4fa1492d2f775480d20cfb82bde183282395333fbace3b57f9c53ced9",
        },
        "financial_public_result": {
            "path": "evidence/mixed_mechanism_six_financial_20261105_results_v1.json",
            "sha256": "b1a35048b0ad4d20628a8262d812f4c0a3fb39b22a0d0f26c4a9070c24d59740",
        },
        "candidate_ids": list(IDS),
        "path_keys": list(PATHS),
        "initial_capital_USD": 10000,
        "monetary_reconciliation_tolerance_USD": marks.MONEY_TOLERANCE,
        "saved_drawdown_agreement_tolerance": marks.METRIC_TOLERANCE,
        "peak_convention": "latest_preceding_running_max_including_initial_capital",
        "trough_convention": "earliest_maximum_observed_drawdown",
        "common_clock_comparison": "both_own_peak_to_trough_intervals_for_each_original_path",
        "mark_identity": (
            "previous_adjusted_units_overnight_current_adjusted_units_intraday_minus_saved_cost"
        ),
        "idle_cash_interest": 0,
        "original_full_return_lattice_preserved": True,
        "existing_accounting_identity_reused_not_new_strategy": True,
        "source_data_reader": "unchanged_canonical_review_market",
        "raw_source_prices_units_targets_or_daily_rows_exported": False,
        "new_financial_evaluations": 0,
        "new_source_requests": 0,
        "source_selectors_targets_accounts_or_BT_replayed": False,
        "financial_criteria_or_global_pool_changed": False,
        "fee_free_or_expert_causal_counterfactual_claimed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Freeze the complete saved-book diagnosis, not a new portfolio criterion.")


def dependencies():
    walker = dependency_template.dependencies
    return FunctionType(
        walker.__code__, {**walker.__globals__, **globals()}, argdefs=walker.__defaults__
    )()


def source_inputs():
    policy = read_json(POLICY)
    validate(policy)
    refs = [
        policy[k]
        for k in (
            "financial_freeze",
            "financial_results",
            "financial_completion",
            "financial_public_result",
        )
    ]
    for row in refs:
        base.checked_reference(row)
    intent = read_json(base.checked_reference(policy["financial_freeze"]))
    bundle = read_json(base.checked_reference(policy["financial_results"]))
    closed = read_json(base.checked_reference(policy["financial_completion"]))
    if (
        closed["status"] != "both_complete_policies_rejected_all_eight_original_paths_cycle_closed"
        or set(bundle["candidate_result_refs"]) != set(IDS)
        or closed["actual_authoritative_counts"]["current_journal_event_count"] != 1109
    ):
        raise QuantError("Read only the eight actual completed mixed-six main accounts.")
    market = intent["frozen"]["market"]
    refs.extend(market.values())
    cases = {}
    for identifier in IDS:
        result_ref = bundle["candidate_result_refs"][identifier]
        result = read_json(base.checked_reference(result_ref))
        refs.append(result_ref)
        if set(result["paths"]) != set(PATHS) or result["review"]["status"] != "rejected":
            raise QuantError("No complete failed policy or original financial path may be omitted.")
        cases[identifier] = {}
        for key in PATHS:
            receipt_ref = result["review"]["paths"][key]
            receipt = read_json(base.checked_reference(receipt_ref))
            if receipt != result["paths"][key] or receipt["candidate_id"] != identifier:
                raise QuantError("The actual source receipt or complete-policy identity changed.")
            artifacts = {r["path"].rsplit("/", 1)[-1]: r for r in receipt["artifacts"]}
            if set(artifacts) != {
                "account.csv",
                "weights.csv",
                "issued.csv",
                "independent-bt.csv",
                "decisions.json",
            }:
                raise QuantError(
                    "Retain all actual account, target and independent accounting evidence."
                )
            refs.extend([receipt_ref, *artifacts.values()])
            cases[identifier][key] = {
                "receipt": receipt_ref,
                "account": artifacts["account.csv"],
                "weights": artifacts["weights.csv"],
            }
    refs = unique_refs(refs)
    for row in refs:
        base.checked_reference(row)
    return market, cases, refs


def progress(phase, **extra):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in extra and extra[k] != v for k, v in counts.items()):
        raise QuantError("A diagnostic cannot override authoritative financial journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "mixed_six_loss_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **extra,
            "investment_goal_met": False,
            "new_financial_evaluations_in_this_study": 0,
        },
    )


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    xml, _ = [base.checked_reference(r) for r in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
        or not suites
        or sum(int(s.get("tests", 0)) for s in suites) != proof["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Bind the actual tested ledger Interface before reading new PnL outcomes.")
    return [*proof["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial loss-diagnosis freeze.")
    with lifecycle.research_job():
        design = read_json(DESIGN)
        if design["actual_native_event_seq"] != 168449 or design["new_PnL_outputs_computed"]:
            raise QuantError(
                "Declare the saved-eight-book question before new diagnostic outcomes."
            )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, refs = source_inputs()
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 1109
            or counts["total_legacy_plus_ssd_completed_configurations"] != 271
            or counts["new_pending_candidates"] != 0
        ):
            raise QuantError("Keep the actual J1109/271 pre-diagnosis financial baseline.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "policy": reference(POLICY),
                "criteria": read_json(POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "validation_evidence": validation_refs(),
                "design": reference(DESIGN),
                "invocation": reference(FOLDER / "actual-invocation-v1.json"),
                "preservation_baseline": reference(FOLDER / "preservation-baseline-v1.json"),
                "actual_counts": counts,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_real_PnL_diagnostic_outcomes_computed": False,
                "new_financial_evaluations": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="one_read_only_full_book_decomposition_pass")
    return reference(INTENT)


def analyze():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "analysis-started-v1.json").exists():
        raise QuantError("Preserve the complete or partial first actual diagnosis without replay.")
    with lifecycle.research_job():
        intent = read_json(INTENT)
        for row in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(row)
        versions = [
            list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
        ]
        if versions != intent["package_versions"] or read_json(POLICY) != intent["criteria"]:
            raise QuantError(
                "The exact diagnostic implementation, law or accepted packages changed."
            )
        market, cases, refs = source_inputs()
        if refs != intent["source_references"]:
            raise QuantError("The actual saved books, prices or financial receipts changed.")
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "analysis_intent": reference(INTENT)},
        )
        case_results, common = {}, []
        try:
            data = base.review_market({"market": market}, ROOT)
            if tuple(data.close.columns) != ASSETS:
                raise QuantError("Keep the original fourteen-asset adjusted quote coordinates.")
            for key in PATHS:
                lifecycle.runtime_guard()
                calendars, decompositions, summaries = {}, {}, {}
                for identifier in IDS:
                    case = cases[identifier][key]
                    write_new_receipt(
                        FOLDER / f"{identifier}-{key}-input-v1.json",
                        {"captured_at": utc_now(), "case": case, "market": market},
                    )
                    account, weights = [
                        pd.read_csv(
                            base.checked_reference(case[name]),
                            index_col=0,
                            parse_dates=True,
                            float_precision="round_trip",
                        )
                        for name in ("account", "weights")
                    ]
                    start = "2016-10-06" if key.startswith("10") else "2021-10-06"
                    if (
                        not account.index.equals(sessions(start, "2026-10-05"))
                        or tuple(weights.columns) != ASSETS
                        or not np.array_equal(
                            account["risk_free"].to_numpy(), data.risk_free.loc[account.index]
                        )
                    ):
                        raise QuantError(
                            "No return date, original RF or actual asset may be cropped."
                        )
                    daily, summary = marks.decompose(
                        account,
                        weights,
                        data.open.loc[account.index],
                        data.close.loc[account.index],
                    )
                    saved = read_json(base.checked_reference(case["receipt"]))
                    gap = abs(
                        summary["maximum_observed_drawdown"] - saved["metrics"]["max_drawdown"]
                    )
                    if gap > marks.METRIC_TOLERANCE:
                        raise QuantError("The original initial-peak maximum drawdown must agree.")
                    summary.update(
                        {
                            "candidate_id": identifier,
                            "path_key": key,
                            "saved_financial_receipt": case["receipt"],
                            "saved_maximum_drawdown_difference": gap,
                        }
                    )
                    write_text_atomic(
                        FOLDER / f"{identifier}-{key}-decomposition-v1.csv",
                        daily.to_csv(float_format="%.17g"),
                    )
                    write_new_receipt(FOLDER / f"{identifier}-{key}-result-v1.json", summary)
                    calendars[identifier], decompositions[identifier] = account.index, daily
                    summaries[identifier] = summary
                    case_results[identifier + ":" + key] = summary
                if not calendars[IDS[0]].equals(calendars[IDS[1]]):
                    raise QuantError("Compare only exact common original saved close dates.")
                for origin in IDS:
                    episode = summaries[origin]["own_maximum_drawdown_episode"]
                    common.append(
                        {
                            "path_key": key,
                            "episode_origin": origin,
                            "episode_status": summaries[origin]["episode_status"],
                            "both_stored_accounts_at_identical_clocks": {
                                identifier: marks.interval(
                                    decompositions[identifier],
                                    episode["start_position"],
                                    episode["end_position"],
                                )
                                for identifier in IDS
                            }
                            if episode is not None
                            else None,
                            "new_capital_or_matched_window_financial_gate": False,
                        }
                    )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_case_ids": list(case_results),
                    "terminal_no_retry_or_repair": True,
                    "new_financial_evaluations": 0,
                },
            )
            progress(
                "terminal_failure_preserved", next_step="do_not_retry_or_repair_frozen_identity"
            )
            raise
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "study_id": read_json(POLICY)["study_id"],
                "analysis_intent": reference(INTENT),
                "original_eight_main_book_cases": case_results,
                "common_clock_episode_views": common,
                "source_references": refs,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "existing_accounting_identity_reused_not_new_strategy": True,
                "source_selectors_targets_accounts_or_BT_replayed": False,
                "new_financial_evaluations": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "complete", next_step="publish_sanitized_aggregate_diagnosis_without_financial_replay"
        )
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": saved["study_id"],
                "completed_at": saved["completed_at"],
                "source_financial_public_result": read_json(POLICY)["financial_public_result"],
                "original_eight_main_book_cases": saved["original_eight_main_book_cases"],
                "common_clock_episode_views": saved["common_clock_episode_views"],
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "raw_source_prices_units_targets_or_daily_rows_exported": False,
                "new_financial_evaluations": 0,
                "source_selectors_targets_accounts_or_BT_replayed": False,
                "fee_free_or_expert_causal_counterfactual_claimed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Read-only eight-book mixed-six loss diagnosis.")
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
