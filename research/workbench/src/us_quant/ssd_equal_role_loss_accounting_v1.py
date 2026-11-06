from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType

import numpy as np
import pandas as pd

from us_quant import equal_role_loss_accounting_v1 as rules
from us_quant import ssd_quintet_learning_loss_accounting_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.storage import read_json, utc_now, write_json, write_text_atomic

base, ROOT, lifecycle = prior.base, prior.ROOT, prior.lifecycle
reference, unique_refs = prior.reference, prior.unique_refs
FOLDER = base.REPORTS / "equal-role-loss-accounting-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/equal_role_loss_accounting_20261106_v1.json"
OWN = (
    "config/equal-role-loss-accounting-v1.json",
    "src/us_quant/equal_role_loss_accounting_v1.py",
    "src/us_quant/ssd_equal_role_loss_accounting_v1.py",
    "tests/test_equal_role_loss_accounting_v1.py",
    "EQUAL_ROLE_LOSS_ACCOUNTING_V1.md",
    "src/us_quant/saved_main_book_decomposition_v1.py",
    "src/us_quant/quartet_trio_loss_accounting_v1.py",
    "src/us_quant/quartet_trio_loss_accounting_v2.py",
    "tests/test_quintet_learning_loss_accounting_v1.py",
    "tests/test_saved_main_book_decomposition_v1.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    return bound(prior.dependency_source.dependencies)()


counts = prior.counts


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Saved accounting cannot change financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "equal_role_loss_accounting_" + phase,
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
    return bound(prior.source_inputs)()


def validation_refs():
    return bound(prior.validation_refs)()


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial eight-book accounting freeze.",
    )
    with lifecycle.research_job():
        actual = counts()
        rules.validate(read_json(rules.POLICY))
        rules.require(
            actual["current_journal_event_count"] == 1478
            and actual == read_json(FOLDER / "preservation-baseline-v1.json")["actual_counts"],
            "Freeze against the actual unchanged J1478 predecessor.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, refs = source_inputs()
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": dependencies(),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_refs(),
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
        progress("frozen", next_step="checkpoint_then_analyze_only_eight_saved_books")
    return reference(INTENT)


def verify_retained(path, cases, contexts, market):
    return bound(prior.verify_retained)(path, cases, contexts, market)


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
        "Never replay the first complete or partial eight-book diagnosis.",
    )
    with lifecycle.research_job():
        intent = read_json(INTENT)
        for ref in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        rules.require(
            sys.version == intent["python_version"]
            and counts() == intent["actual_authoritative_counts"]
            and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
            == intent["package_versions"],
            "The frozen source environment or predecessor changed.",
        )
        market, cases, refs = source_inputs()
        rules.require(
            refs == intent["source_references"], "Keep every exact source receipt and book."
        )
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "A real saved preanalysis checkpoint must precede new marks.",
        )
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": reference(INTENT),
            },
        )
        data = base.review_market({"market": market}, ROOT)
        rules.require(
            tuple(data.close.columns) == tuple(rules.ASSETS),
            "Keep the original fourteen price coordinates.",
        )
        contexts, prefixes, completed_cases, rows = {}, [], [], 0
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with trace.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for key in rules.PATHS:
                lifecycle.runtime_guard()
                expected = prior.sessions(
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
                    rules.require(
                        account.index.equals(expected)
                        and weights.index.equals(expected)
                        and tuple(weights.columns) == tuple(rules.ASSETS)
                        and np.array_equal(
                            account["risk_free"].to_numpy(), data.risk_free.loc[expected].to_numpy()
                        ),
                        "No formal date,RF value or actual asset coordinate may be filled.",
                    )
                    books[role] = rules.StoredBook(account, weights, case["cached_max_drawdown"])
                opened, closed = data.open.loc[expected], data.close.loc[expected]
                daily, summary = rules.compare(books, opened, closed)
                for role in rules.ROLES:
                    book = books[role]
                    for cut in (1, 21, 126, len(expected) // 2, len(expected)):
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
                    path = FOLDER / f"{role}-{key}-marked-v1.csv"
                    write_text_atomic(path, daily[role].to_csv(float_format="%.17g"))
                    retain(
                        {
                            "phase": "after_book",
                            "path_key": key,
                            "role": role,
                            "marked_daily": reference(path),
                            "summary": summary["own_books"][role],
                            "new_strategy_target_account_BT_or_financial_trials": 0,
                        }
                    )
                    completed_cases.append(role + ":" + key)
                    rows += len(expected)
                contexts[key] = summary
                del books, daily
        own = sum(len(c["own_books"]) for c in contexts.values())
        common = sum(
            len(r["both_stored_accounts_at_identical_clocks"])
            for c in contexts.values()
            for r in c["common_clock_episode_views"]
            if r["both_stored_accounts_at_identical_clocks"] is not None
        )
        comparisons = sum(len(c["equal_role_minus_flat_comparisons"]) for c in contexts.values())
        rules.require(
            len(completed_cases) == own == 8
            and rows == 15064
            and len(prefixes) == 40
            and common == 16
            and comparisons == 8
            and counts() == intent["actual_authoritative_counts"],
            "Keep all eight books,own/common clocks,full rows,prefixes and financial counts.",
        )
        retained = verify_retained(trace, cases, contexts, market)
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "analysis_intent": reference(INTENT),
                "four_original_contexts": contexts,
                "actual_prefix_checks": prefixes,
                "actual_input_resolved_records": reference(trace),
                "actual_input_resolved_record_count": retained,
                "stored_books": 8,
                "original_return_rows": rows,
                "own_episode_slots": own,
                "common_clock_interval_slots": common,
                "equal_role_minus_flat_comparison_slots": comparisons,
                "actual_authoritative_counts": counts(),
                "source_financial_public_result": read_json(
                    base.checked_reference(read_json(rules.POLICY)["source_catalogue"])
                )["latest_financial_public_result"],
                "new_strategy_target_account_BT_or_financial_trials": 0,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "fee_free_or_expert_causal_counterfactual_claimed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_analysis_complete",
            private_result=reference(OUTPUT),
            analysis_intent=reference(INTENT),
            next_step="export_sanitized_aggregate_once",
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
                "study_id": read_json(rules.POLICY)["study_id"],
                **{
                    k: v
                    for k, v in saved.items()
                    if k not in ("analysis_intent", "actual_input_resolved_records")
                },
                "raw_prices_targets_units_or_daily_rows_exported": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))
