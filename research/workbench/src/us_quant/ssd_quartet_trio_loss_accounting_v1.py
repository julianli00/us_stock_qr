from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import quartet_trio_loss_accounting_v1 as rules
from us_quant import ssd_growth_macro_defense_hedge_v1 as source
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "quartet-trio-loss-accounting-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/quartet_trio_loss_accounting_20261106_v1.json"
OWN = (
    "config/quartet-trio-loss-accounting-v1.json",
    "src/us_quant/quartet_trio_loss_accounting_v1.py",
    "src/us_quant/ssd_quartet_trio_loss_accounting_v1.py",
    "tests/test_quartet_trio_loss_accounting_v1.py",
    "tests/test_saved_main_book_decomposition_v1.py",
    "tests/conftest.py",
    "QUARTET_TRIO_LOSS_ACCOUNTING_V1.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    previous = read_json(base.checked_reference(baseline["source_closed_completion"]))
    refs = [
        *previous["exact_evidence_references"],
        baseline["source_closed_completion"],
        *inputs["market"].values(),
        *inputs["original_book_references"],
        *[
            inputs[k]
            for k in (
                "quartet_private_results",
                "quartet_public_results",
                "trio_private_results",
                "trio_public_results",
            )
        ],
        *[
            reference(FOLDER / name)
            for name in (
                "source-inputs-v1.json",
                "design-decision-v1.json",
                "actual-invocation-v1.json",
                "preservation-baseline-v1.json",
                "capture-preflight-v1.py",
            )
        ],
    ]
    return unique_refs(refs)


def check_frozen():
    frozen = read_json(INTENT)
    for ref in (
        frozen["implementation_dependencies"]
        + frozen["source_references"]
        + frozen["validation_evidence"]
    ):
        base.checked_reference(ref)
    rules.require(
        frozen["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN],
        "Preserve the exact first frozen paired-book diagnosis.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep the accepted full Python version and every captured package row.",
    )
    return frozen


def progress(phase, **fields):
    counts = source.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "A saved-book diagnosis cannot change authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quartet_trio_loss_accounting_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial paired-loss freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        rules.validate(read_json(rules.POLICY))
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            source.current_counts(base.journal_events(base.JOURNAL)) == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1320,
            "Freeze against the actual unchanged J1320/289 baseline.",
        )
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, _ = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful accounting, group, episode-clock, prefix and no-replay checks.",
        )
        refs = source_refs()
        for ref in refs:
            base.checked_reference(ref)
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "policy": reference(rules.POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "implementation_dependencies": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "new_actual_PnL_or_episode_diagnosis_computed": False,
                "new_financial_trials": 0,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "analysis-attempt-v1.json").exists(),
        "Preserve the first complete or partial actual diagnosis instead of replaying it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(FOLDER / "analysis-attempt-v1.json", {"started_at": utc_now()})
        inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
        data = base.review_market({"market": inputs["market"]}, ROOT)
        rules.require(
            tuple(data.close.columns) == tuple(rules.ASSETS),
            "Keep the original fourteen-asset adjusted price coordinates.",
        )
        cases = []
        directory = FOLDER / "cases"
        new_output_directory(directory)
        try:
            for new_id, old_id in zip(source.IDS, source.trio.IDS, strict=True):
                for key in rules.PATHS:
                    lifecycle.runtime_guard()
                    case = inputs["cases"][new_id][key]
                    case_directory = directory / (new_id + "-" + key)
                    new_output_directory(case_directory)
                    write_new_receipt(
                        case_directory / "input-v1.json",
                        {
                            "captured_at": utc_now(),
                            "case": case,
                            "market": inputs["market"],
                            "new_PnL_outputs_computed": False,
                        },
                    )
                    books = {}
                    for role, identifier in (("quartet", new_id), ("cached_trio", old_id)):
                        evidence = case[role]
                        receipt = read_json(base.checked_reference(evidence["receipt"]))
                        account, weights = frame(evidence["account"]), frame(evidence["weights"])
                        expected = sessions(
                            "2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05"
                        )
                        rules.require(
                            evidence["candidate_id"] == receipt["candidate_id"] == identifier
                            and evidence["path_key"] == receipt["path_key"] == key
                            and account.index.equals(expected)
                            and len(expected) == (2512 if key.startswith("10y") else 1254)
                            and tuple(weights.columns) == tuple(rules.ASSETS)
                            and account["risk_free"].equals(data.risk_free.loc[expected])
                            and evidence["cached_max_drawdown"]
                            == receipt["metrics"]["max_drawdown"]
                            and receipt["broker_orders"] == 0,
                            "No formal date, cached source identity, "
                            "original RF or funding role changes.",
                        )
                        books[role] = rules.StoredBook(
                            account, weights, evidence["cached_max_drawdown"]
                        )
                    dates = books["quartet"].account.index
                    opened, closed = data.open.loc[dates], data.close.loc[dates]
                    daily, report = rules.compare(books, opened, closed)
                    prefix_counts = (1, 21, 126, len(dates) // 2, len(dates))
                    prefix_checks = {}
                    for role in rules.ROLES:
                        for stop in prefix_counts:
                            early, _ = rules.marks.decompose(
                                books[role].account.iloc[:stop],
                                books[role].weights.iloc[:stop],
                                opened.iloc[:stop],
                                closed.iloc[:stop],
                            )
                            rules.require(
                                early.equals(daily[role].iloc[:stop]),
                                "Future saved rows changed a previous daily monetary identity.",
                            )
                        prefix_checks[role] = list(prefix_counts)
                        write_text_atomic(
                            case_directory / (role + "-daily-v1.csv"),
                            daily[role].to_csv(float_format="%.17g"),
                        )
                    report.update(
                        {
                            "path_key": key,
                            "quartet_id": new_id,
                            "cached_trio_id": old_id,
                            "saved_source_refs": case,
                            "daily_identity_prefix_counts": prefix_checks,
                            "whole_episode_selection_is_prefix_invariant": False,
                        }
                    )
                    write_new_receipt(case_directory / "report-v1.json", report)
                    cases.append(report)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_paired_cases": len(cases),
                    "partial_evidence_preserved": True,
                    "terminal_no_retry_or_repair": True,
                    "new_financial_trials": 0,
                },
            )
            progress(
                "terminal_failure_preserved", next_step="do_not_retry_or_patch_frozen_identity"
            )
            raise
        counts = source.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"],
            "Readonly loss bookkeeping cannot append a financial lifecycle event.",
        )
        rules.require(len(cases) == 8, "Retain every paired path and all sixteen full books.")
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "implementation_freeze": reference(INTENT),
                "paired_cases": cases,
                "existing_full_books": 16,
                "common_clock_episode_origins": 16,
                "actual_authoritative_counts": counts,
                "new_financial_trials": 0,
                "source_strategy_account_BT_or_CAGR_Sharpe_replayed": False,
                "new_stored_interval_PnL_arithmetic": True,
                "source_full_window_returns_edited_or_requalified": False,
                "fee_free_or_expert_causal_counterfactual_claimed": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_analysis_complete",
            private_result=reference(RESULTS),
            implementation_freeze=reference(INTENT),
            new_financial_trials=0,
        )
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "paired_cases": [
                    {k: v for k, v in case.items() if k != "saved_source_refs"}
                    for case in saved["paired_cases"]
                ],
                "existing_full_books": 16,
                "common_clock_episode_origins": 16,
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "raw_prices_units_targets_or_daily_rows_exported": False,
                "new_financial_trials": 0,
                "source_strategy_account_BT_or_CAGR_Sharpe_replayed": False,
                "new_stored_interval_PnL_arithmetic": True,
                "source_full_window_returns_edited_or_requalified": False,
                "fee_free_or_expert_causal_counterfactual_claimed": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Readonly paired quartet/trio loss bookkeeping.")
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
