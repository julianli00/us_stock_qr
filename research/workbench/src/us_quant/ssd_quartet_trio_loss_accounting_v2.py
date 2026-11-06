from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import quartet_trio_loss_accounting_v2 as rules
from us_quant import ssd_quartet_trio_loss_accounting_v1 as prior
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

base, ROOT, lifecycle = prior.base, prior.ROOT, prior.lifecycle
reference, unique_refs = prior.reference, prior.unique_refs
FOLDER = base.REPORTS / "quartet-trio-loss-accounting-20261106-v2"
INTENT = FOLDER / "diagnostic-implementation-freeze-v2.json"
RESULTS = FOLDER / "results-v2.json"
PUBLIC = ROOT / "evidence/quartet_trio_loss_accounting_20261106_v2.json"
OWN = (
    "config/quartet-trio-loss-accounting-v2.json",
    "src/us_quant/quartet_trio_loss_accounting_v2.py",
    "src/us_quant/ssd_quartet_trio_loss_accounting_v2.py",
    "tests/test_quartet_trio_loss_accounting_v2.py",
    "tests/test_quartet_trio_loss_accounting_v1.py",
    "tests/test_saved_main_book_decomposition_v1.py",
    "tests/conftest.py",
    "QUARTET_TRIO_LOSS_ACCOUNTING_V2.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    continuation = read_json(base.checked_reference(read_json(rules.POLICY)["continuation_inputs"]))
    terminal = read_json(base.checked_reference(continuation["V1_terminal_cycle"]))
    refs = [
        *terminal["exact_evidence_references"],
        *[
            continuation[k]
            for k in (
                "original_source_inputs",
                "V1_frozen_implementation",
                "V1_terminal_cycle",
                "V1_failure_projection",
                "V1_narrow_counterexample",
            )
        ],
        *[
            reference(FOLDER / name)
            for name in (
                "continuation-inputs-v2.json",
                "design-decision-v2.json",
                "preservation-baseline-v2.json",
                "capture-continuation-v2.py",
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
        "Preserve exact separately frozen V2 code and every immutable V1 artifact.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v2.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep the accepted full Python version and every captured package row.",
    )
    return frozen


def progress(phase, **fields):
    counts = prior.source.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "A numeric-prefix continuation cannot change financial trial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quartet_trio_loss_accounting_v2_" + phase,
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
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v2.json").exists(),
        "Preserve the first complete or partial V2 numeric-prefix freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v2.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        rules.validate(policy)
        baseline = read_json(FOLDER / "preservation-baseline-v2.json")
        continuation = read_json(base.checked_reference(policy["continuation_inputs"]))
        rules.require(
            prior.source.current_counts(base.journal_events(base.JOURNAL))
            == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1320
            and continuation["original_V1_all_column_prefix_contract_remains_failed"] is True
            and continuation["completed_V1_paired_cases_reused"] == 2
            and continuation["remaining_paired_cases_for_full_arithmetic"] == 6,
            "Keep V1 terminal and reuse two completed cases without changing the source problem.",
        )
        validation = read_json(FOLDER / "validation-source-v2.json")
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
            "Bind synthetic shape-reduction evidence and exact primitive/scalar prefix checks.",
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
                    reference(FOLDER / "validation-source-v2.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "same_economic_question_continuation": True,
                "original_V1_all_column_prefix_contract_remains_failed": True,
                "actual_V2_prefix_view_or_PnL_arithmetic_started": False,
                "new_financial_trials": 0,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "analysis-attempt-v2.json").exists(),
        "Preserve the first complete or partial actual V2 diagnosis; do not replay it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(FOLDER / "analysis-attempt-v2.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        continuation = read_json(base.checked_reference(policy["continuation_inputs"]))
        inputs = read_json(base.checked_reference(policy["source_inputs"]))
        data = base.review_market({"market": inputs["market"]}, ROOT)
        rules.require(
            tuple(data.close.columns) == tuple(rules.ASSETS),
            "Keep the original fourteen-asset adjusted price coordinates.",
        )
        cases, reused_count, computed_count = [], 0, 0
        directory = FOLDER / "cases"
        new_output_directory(directory)
        try:
            for new_id, old_id in zip(prior.source.IDS, prior.source.trio.IDS, strict=True):
                for key in rules.PATHS:
                    lifecycle.runtime_guard()
                    case = inputs["cases"][new_id][key]
                    case_directory = directory / (new_id + "-" + key)
                    new_output_directory(case_directory)
                    reuse = continuation["V1_completed_case_reuse"].get(new_id, {}).get(key)
                    write_new_receipt(
                        case_directory / "input-v2.json",
                        {
                            "captured_at": utc_now(),
                            "case": case,
                            "market": inputs["market"],
                            "immutable_V1_completed_case_reuse": reuse,
                            "actual_V2_case_outputs_computed": False,
                        },
                    )
                    expected = sessions(
                        "2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05"
                    )
                    prefix_counts = (1, 21, 126, len(expected) // 2, len(expected))
                    prefix_details, daily_refs = {}, {}
                    if reuse is not None:
                        report = read_json(base.checked_reference(reuse["report"]))
                        rules.require(
                            report["saved_source_refs"] == case
                            and report["quartet_id"] == new_id
                            and report["cached_trio_id"] == old_id
                            and report["path_key"] == key,
                            "Reuse only the exact completed V1 source case, not a partial report.",
                        )
                        for role in rules.ROLES:
                            daily = prior.frame(reuse["daily"][role])
                            rules.require(
                                daily.index.equals(expected)
                                and reuse["original_all_column_exact_prefix_counts"][role]
                                == list(prefix_counts),
                                "The stronger original exact-prefix evidence "
                                "must cover all five cuts.",
                            )
                            prefix_details[role] = {
                                "original_stronger_V1_prefix_evidence_reused": True,
                                "full_book_decomposition_reexecuted": False,
                                "canonical_projection_locality_checks": [
                                    rules.check_prefix(daily, daily.iloc[:stop])
                                    for stop in prefix_counts
                                ],
                            }
                            daily_refs[role] = reuse["daily"][role]
                        reused_count += 1
                    else:
                        books = {}
                        for role, identifier in (("quartet", new_id), ("cached_trio", old_id)):
                            evidence = case[role]
                            receipt = read_json(base.checked_reference(evidence["receipt"]))
                            account = prior.frame(evidence["account"])
                            weights = prior.frame(evidence["weights"])
                            rules.require(
                                evidence["candidate_id"] == receipt["candidate_id"] == identifier
                                and evidence["path_key"] == receipt["path_key"] == key
                                and account.index.equals(expected)
                                and tuple(weights.columns) == tuple(rules.ASSETS)
                                and account["risk_free"].equals(data.risk_free.loc[expected])
                                and evidence["cached_max_drawdown"]
                                == receipt["metrics"]["max_drawdown"]
                                and receipt["broker_orders"] == 0,
                                "No date, source role, native RF, "
                                "saved drawdown or funding changes.",
                            )
                            books[role] = rules.StoredBook(
                                account, weights, evidence["cached_max_drawdown"]
                            )
                        opened, closed = data.open.loc[expected], data.close.loc[expected]
                        daily, report = rules.compare(books, opened, closed)
                        for role in rules.ROLES:
                            checks = []
                            for stop in prefix_counts:
                                early, _ = rules.marks.decompose(
                                    books[role].account.iloc[:stop],
                                    books[role].weights.iloc[:stop],
                                    opened.iloc[:stop],
                                    closed.iloc[:stop],
                                )
                                checks.append(rules.check_prefix(daily[role], early))
                            prefix_details[role] = {
                                "original_stronger_V1_prefix_evidence_reused": False,
                                "full_book_decomposition_reexecuted": False,
                                "new_V2_full_book_diagnostic_arithmetic": True,
                                "canonical_projection_locality_checks": checks,
                            }
                            path = case_directory / (role + "-daily-v2.csv")
                            write_text_atomic(path, daily[role].to_csv(float_format="%.17g"))
                            daily_refs[role] = reference(path)
                        computed_count += 1
                    report.update(
                        {
                            "path_key": key,
                            "quartet_id": new_id,
                            "cached_trio_id": old_id,
                            "saved_source_refs": case,
                            "daily_diagnostic_refs": daily_refs,
                            "V2_prefix_contract": rules.PREFIX_CONTRACT,
                            "canonical_prefix_details": prefix_details,
                            "completed_V1_case_reused_without_decomposition": reuse is not None,
                            "source_completed_V1_report": reuse["report"]
                            if reuse is not None
                            else None,
                            "original_V1_all_column_prefix_contract_remains_failed": True,
                            "whole_episode_selection_is_prefix_invariant": False,
                        }
                    )
                    write_new_receipt(case_directory / "report-v2.json", report)
                    cases.append(report)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v2.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_paired_cases": len(cases),
                    "partial_evidence_preserved": True,
                    "terminal_no_retry_or_repair": True,
                    "new_financial_trials": 0,
                },
            )
            progress("terminal_failure_preserved", next_step="do_not_retry_or_patch_frozen_V2")
            raise
        counts = prior.source.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"]
            and len(cases) == 8
            and reused_count == 2
            and computed_count == 6,
            "Keep every pair, exact completed-case reuse and unchanged financial counts.",
        )
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "implementation_freeze": reference(INTENT),
                "paired_cases": cases,
                "existing_full_books": 16,
                "common_clock_episode_origins": 16,
                "completed_V1_paired_cases_reused": reused_count,
                "new_full_book_diagnostic_paired_cases": computed_count,
                "original_V1_terminal_cycle": continuation["V1_terminal_cycle"],
                "original_V1_failure_projection": continuation["V1_failure_projection"],
                "original_V1_all_column_prefix_contract_remains_failed": True,
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
                "schema_version": 2,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "paired_cases": [
                    {
                        k: v
                        for k, v in case.items()
                        if k
                        not in {
                            "saved_source_refs",
                            "daily_diagnostic_refs",
                            "source_completed_V1_report",
                        }
                    }
                    for case in saved["paired_cases"]
                ],
                "existing_full_books": 16,
                "common_clock_episode_origins": 16,
                "completed_V1_paired_cases_reused": 2,
                "new_full_book_diagnostic_paired_cases": 6,
                "original_V1_failure_projection": saved["original_V1_failure_projection"],
                "original_V1_all_column_prefix_contract_remains_failed": True,
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
    parser = argparse.ArgumentParser(
        description="Versioned canonical-prefix paired loss diagnosis."
    )
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
