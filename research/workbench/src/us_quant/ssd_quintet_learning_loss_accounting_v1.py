from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import quintet_learning_loss_accounting_v1 as rules
from us_quant import ssd_research_v1 as base
from us_quant import ssd_style_activity_financial_v1 as dependency_source
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import read_json, utc_now, write_json, write_text_atomic

ROOT = base.ROOT
FOLDER = base.REPORTS / "quintet-learning-loss-accounting-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/quintet_learning_loss_accounting_20261106_v1.json"
reference, unique_refs = base.reference, dependency_source.unique_refs
OWN = (
    "config/quintet-learning-loss-accounting-v1.json",
    "src/us_quant/quintet_learning_loss_accounting_v1.py",
    "src/us_quant/ssd_quintet_learning_loss_accounting_v1.py",
    "tests/test_quintet_learning_loss_accounting_v1.py",
    "QUINTET_LEARNING_LOSS_ACCOUNTING_V1.md",
    "src/us_quant/saved_main_book_decomposition_v1.py",
    "src/us_quant/quartet_trio_loss_accounting_v1.py",
    "src/us_quant/quartet_trio_loss_accounting_v2.py",
    "tests/test_saved_main_book_decomposition_v1.py",
    "tests/conftest.py",
)


def dependencies():
    function = dependency_source.dependencies
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )()


def counts():
    return current_counts(base.journal_events(base.JOURNAL))


def progress(phase, **fields):
    actual = counts()
    if any(key in fields and fields[key] != value for key, value in actual.items()):
        raise QuantError("Read-only accounting cannot change authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quintet_learning_loss_accounting_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **fields,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_catalogue"]))
    if (
        catalogue["role_ids"] != rules.ROLE_IDS
        or catalogue["path_keys"] != list(rules.PATHS)
        or catalogue["new_marked_PnL_or_episode_outputs_computed"] is not False
        or catalogue["source_learner_target_account_BT_or_prior_diagnosis_replayed"] is not False
        or catalogue["user_owned_stored_main_cost_config_read_or_executed"] is not False
        or catalogue["new_financial_trials"] != 0
        or set(catalogue["books"]) != set(rules.ROLES)
    ):
        raise QuantError("Use only the declared twelve saved books and original source scope.")
    closed = read_json(base.checked_reference(catalogue["latest_financial_completion"]))
    if closed["actual_authoritative_counts"] != counts():
        raise QuantError(
            "The financial predecessor changed;do not analyze a stale current snapshot."
        )
    for role, cases in catalogue["books"].items():
        if set(cases) != set(rules.PATHS):
            raise QuantError("Keep every original full path of all three complete laws.")
        for case in cases.values():
            saved = read_json(base.checked_reference(case["receipt"]))
            if (
                case["candidate_id"] != rules.ROLE_IDS[role]
                or saved["candidate_id"] != case["candidate_id"]
                or case["cached_max_drawdown"] != saved["metrics"]["max_drawdown"]
                or case["account"] not in saved["artifacts"]
                or case["weights"] not in saved["artifacts"]
            ):
                raise QuantError("A sealed receipt,book coordinate or cached drawdown changed.")
    refs = unique_refs(
        [
            *catalogue["source_references"],
            policy["source_catalogue"],
            reference(DESIGN),
            reference(INVOCATION),
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue["market"], catalogue["books"], refs


def validation_refs():
    validation = read_json(FOLDER / "validation-source-v1.json")
    xml, _ = [base.checked_reference(ref) for ref in validation["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or validation["exit_code"] != 0
        or validation["synthetic_only"] is not True
        or not suites
        or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError(
            "Freeze actual tested three-book and signed-zero Interfaces before PnL labels."
        )
    return [*validation["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial saved-book accounting freeze.")
    with lifecycle.research_job():
        design = read_json(DESIGN)
        actual = counts()
        if (
            design["new_marked_PnL_or_episode_outputs_computed"] is not False
            or actual["current_journal_event_count"] != 1452
            or actual["current_journal_head_sha256"] != rules.PRIOR_HEAD
        ):
            raise QuantError("Freeze the complete accounting scope before new PnL outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, refs = source_inputs()
        validation = validation_refs()
        deps = dependencies()
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        if actual != baseline["actual_counts"]:
            raise QuantError("The read-only diagnosis cannot add financial registrations or paths.")
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation,
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_authoritative_counts": actual,
                "new_marked_PnL_or_episode_outputs_computed": False,
                "new_strategy_target_account_BT_or_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_analyze_saved_twelve_books_without_replays")
    return reference(INTENT)


def verify_retained(path, cases, contexts, market):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keyed = {(r["phase"], r["path_key"], r["role"]): r for r in rows}
    expected = {
        (phase, key, role)
        for phase in ("before_book", "after_book")
        for key in rules.PATHS
        for role in rules.ROLES
    }
    if len(rows) != len(keyed) or set(keyed) != expected:
        raise QuantError("Retain all24unique before/after archived-book input-resolution records.")
    for key in rules.PATHS:
        for role in rules.ROLES:
            before, after = keyed["before_book", key, role], keyed["after_book", key, role]
            if (
                before["case"] != cases[role][key]
                or before["market"] != market
                or after["summary"] != contexts[key]["own_books"][role]
            ):
                raise QuantError("A retained book input or grouped result changed.")
            base.checked_reference(after["marked_daily"])
    return len(rows)


def analyze():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "analysis-started-v1.json").exists():
        raise QuantError("Preserve the first complete or partial accounting run;do not replay it.")
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
        if (
            sys.version != intent["python_version"]
            or versions != intent["package_versions"]
            or read_json(rules.POLICY) != intent["criteria"]
            or counts() != intent["actual_authoritative_counts"]
        ):
            raise QuantError("The frozen law,input environment or financial predecessor changed.")
        market, cases, refs = source_inputs()
        if refs != intent["source_references"]:
            raise QuantError("The exact source receipts,books or market references changed.")
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": reference(INTENT),
            },
        )
        contexts, prefixes, completed_cases = {}, [], []
        rows = 0
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        try:
            data = base.review_market({"market": market}, ROOT)
            if tuple(data.close.columns) != tuple(rules.ASSETS):
                raise QuantError("Use every original fourteen-asset quote coordinate.")
            with trace.open("x") as stream:

                def retain(record):
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                for key in rules.PATHS:
                    lifecycle.runtime_guard()
                    expected = sessions(
                        "2016-10-06" if key.startswith("10") else "2021-10-06", "2026-10-05"
                    )
                    books = {}
                    for role in rules.ROLES:
                        case = cases[role][key]
                        retain(
                            {
                                "phase": "before_book",
                                "path_key": key,
                                "role": role,
                                "case": case,
                                "market": market,
                                "source_learner_target_account_BT_replayed": False,
                            }
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
                        if (
                            not account.index.equals(expected)
                            or not weights.index.equals(expected)
                            or tuple(weights.columns) != tuple(rules.ASSETS)
                            or not np.array_equal(
                                account["risk_free"].to_numpy(),
                                data.risk_free.loc[expected].to_numpy(),
                            )
                        ):
                            raise QuantError(
                                "No original return date,RF value or actual asset may be filled."
                            )
                        books[role] = rules.StoredBook(
                            account, weights, case["cached_max_drawdown"]
                        )
                    opened, closed = data.open.loc[expected], data.close.loc[expected]
                    daily, summary = rules.compare(books, opened, closed)
                    for role in rules.ROLES:
                        book = books[role]
                        for cut in (1, 21, 126, len(expected) // 2, len(expected)):
                            lifecycle.runtime_guard()
                            prefix, _ = rules.marks.decompose(
                                book.account.iloc[:cut],
                                book.weights.iloc[:cut],
                                opened.iloc[:cut],
                                closed.iloc[:cut],
                            )
                            prefixes.append(
                                {
                                    "path_key": key,
                                    "role": role,
                                    **rules.check_prefix(daily[role], prefix),
                                }
                            )
                        output = FOLDER / f"{role}-{key}-marked-v1.csv"
                        write_text_atomic(output, daily[role].to_csv(float_format="%.17g"))
                        retain(
                            {
                                "phase": "after_book",
                                "path_key": key,
                                "role": role,
                                "marked_daily": reference(output),
                                "summary": summary["own_books"][role],
                                "new_strategy_target_account_BT_or_financial_trials": 0,
                            }
                        )
                        completed_cases.append(role + ":" + key)
                        rows += len(expected)
                    contexts[key] = summary
                    del books, daily
            own_slots = sum(len(value["own_books"]) for value in contexts.values())
            common_slots = sum(
                len(row["all_three_stored_accounts_at_identical_clocks"])
                for value in contexts.values()
                for row in value["common_clock_episode_views"]
                if row["all_three_stored_accounts_at_identical_clocks"] is not None
            )
            comparisons = sum(
                len(value["adaptive_minus_control_comparisons"]) for value in contexts.values()
            )
            if (
                len(completed_cases) != 12
                or rows != 22596
                or len(prefixes) != 60
                or own_slots != 12
                or common_slots != 36
                or comparisons != 24
                or counts() != intent["actual_authoritative_counts"]
            ):
                raise QuantError(
                    "Keep all books,full rows,own/common clocks,prefixes and financial counts."
                )
            retained = verify_retained(trace, cases, contexts, market)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_cases": completed_cases,
                    "completed_contexts": list(contexts),
                    "terminal_no_retry_or_repair": True,
                    "new_financial_trials": 0,
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
                "study_id": read_json(rules.POLICY)["study_id"],
                "analysis_intent": reference(INTENT),
                "four_original_contexts": contexts,
                "actual_prefix_checks": prefixes,
                "stored_books": len(completed_cases),
                "original_return_rows": rows,
                "input_resolved_records": reference(trace),
                "actual_input_resolved_record_count": retained,
                "own_episode_slots": own_slots,
                "common_clock_interval_slots": common_slots,
                "adaptive_minus_control_comparison_slots": comparisons,
                "source_references": refs,
                "actual_authoritative_counts": counts(),
                "new_strategy_target_account_BT_or_financial_trials": 0,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "source_learner_target_account_BT_or_prior_diagnosis_replayed": False,
                "financial_criteria_or_global_pool_changed": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress("complete", next_step="publish_only_sanitized_aggregate_diagnosis")
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        if counts() != saved["actual_authoritative_counts"]:
            raise QuantError("The diagnostic must retain its actual financial snapshot.")
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": saved["study_id"],
                "completed_at": saved["completed_at"],
                "four_original_contexts": saved["four_original_contexts"],
                "actual_prefix_checks": saved["actual_prefix_checks"],
                "stored_books": saved["stored_books"],
                "original_return_rows": saved["original_return_rows"],
                "actual_input_resolved_record_count": saved["actual_input_resolved_record_count"],
                "own_episode_slots": saved["own_episode_slots"],
                "common_clock_interval_slots": saved["common_clock_interval_slots"],
                "adaptive_minus_control_comparison_slots": saved[
                    "adaptive_minus_control_comparison_slots"
                ],
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "source_financial_public_result": read_json(
                    base.checked_reference(read_json(rules.POLICY)["source_catalogue"])
                )["latest_financial_public_result"],
                "raw_source_prices_units_targets_or_daily_rows_exported": False,
                "new_strategy_target_account_BT_or_financial_trials": 0,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "source_learner_target_account_BT_or_prior_diagnosis_replayed": False,
                "fee_free_or_expert_causal_counterfactual_claimed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))
