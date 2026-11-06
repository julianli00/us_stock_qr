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

from us_quant import confidence_regret_scope_v1 as rules
from us_quant import ssd_confidence_rated_prod_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "confidence-regret-scope-20261106-v1"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/confidence_regret_scope_20261106_v1.json"
OWN = (
    "config/confidence-regret-scope-v1.json",
    "src/us_quant/confidence_regret_scope_v1.py",
    "src/us_quant/ssd_confidence_regret_scope_v1.py",
    "tests/test_confidence_regret_scope_v1.py",
    "CONFIDENCE_REGRET_SCOPE_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def counts():
    return source.current_counts(base.journal_events(base.JOURNAL))


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Comparator inspection cannot alter financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "confidence_regret_scope_" + phase,
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
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    rules.require(
        counts() == closed["actual_authoritative_counts"]
        and counts()["current_journal_event_count"] == 1490
        and catalogue["source_expert_ids"] == list(source.rules.EXPERT_IDS)
        and catalogue["new_regret_scope_or_month_loss_labels_computed"] is False
        and catalogue["source_learning_targets_accounts_or_prior_diagnoses_replayed"] is False
        and set(catalogue["four_saved_books"]) == set(rules.PATHS),
        "Use the actual unchanged complete J1490 source and every saved formal book.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                value
                for value in catalogue.values()
                if isinstance(value, dict) and set(value) == {"path", "sha256"}
            ],
            *[row["account"] for row in catalogue["four_saved_books"].values()],
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
    return catalogue, refs


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial saved-regret inspection freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, refs = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        suites = (
            ElementTree.parse(base.checked_reference(validation["test_outputs"][0]))
            .getroot()
            .findall(".//testsuite")
        )
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind actual successful exact-algebra, prior-clock, month-loss and prefix checks.",
        )
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": dependencies(),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_authoritative_counts": counts(),
                "new_regret_scope_or_month_loss_labels_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_inspect_saved_game_and_books_once")
    return reference(INTENT)


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
        "Never replay the first complete or partial saved-regret diagnosis.",
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
            "The original saved inputs, environment or predecessor changed.",
        )
        catalogue, refs = source_inputs()
        rules.require(refs == intent["source_references"], "Keep every exact source reference.")
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "An actual saved checkpoint must precede new comparator and loss-month labels.",
        )
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "analysis_intent": reference(INTENT)},
        )
        old_records = [
            json.loads(line)
            for line in base.checked_reference(catalogue["source_actual_records"])
            .read_text()
            .splitlines()
        ]
        rules.require(len(old_records) == 242, "Keep all original source-controller records.")
        games, month_inputs, month_results = [], [], []
        contexts, record_count = {}, 0
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with trace.open("x") as stream:

            def retain(row):
                nonlocal record_count
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                record_count += 1

            previous = None
            for index in range(121):
                lifecycle.runtime_guard()
                before, after = old_records[2 * index : 2 * index + 2]
                retain({"phase": "game_input", "source_before": before, "source_after": after})
                result = rules.inspect_game(before, after, previous)
                rules.require(
                    result["mature"] is (index >= 2), "Keep exactly two unscored origins."
                )
                games.append(result)
                retain({"phase": "game_resolved", "result": result})
                previous = after["audit"]
            mature = [row for row in games if row["mature"]]
            rules.require(len(mature) == 119, "Keep every complete original source month.")
            for key in rules.PATHS:
                lifecycle.runtime_guard()
                case = catalogue["four_saved_books"][key]
                book = pd.read_csv(
                    base.checked_reference(case["account"]),
                    index_col=0,
                    parse_dates=True,
                    float_precision="round_trip",
                )
                expected = rules.sessions(
                    "2016-10-06" if key.startswith("10") else "2021-10-06", "2026-10-05"
                )
                rules.require(
                    book.index.equals(expected)
                    and len(book) == case["original_sessions"]
                    and np.isfinite(book.to_numpy()).all()
                    and (book["return"] > -1).all()
                    and (book["equity"] > 0).all()
                    and (book["cash"] >= 0).all(),
                    "A saved book cannot lose, fill or replace any formal return row.",
                )
                covered, results = [], []
                for game in mature:
                    dates = pd.DatetimeIndex(game["source_return_dates"])
                    if not dates.isin(book.index).all():
                        continue
                    value = {
                        "path_key": key,
                        "game_month": game,
                        "return_dates": game["source_return_dates"],
                        "NET_returns_hex": [float(v).hex() for v in book.loc[dates, "return"]],
                    }
                    retain(
                        {
                            "phase": "main_month_input",
                            "case": value,
                            "source_account": case["account"],
                        }
                    )
                    result = rules.inspect_main_month(value)
                    retain({"phase": "main_month_resolved", "result": result})
                    results.append(result)
                    month_inputs.append(value)
                    month_results.append(result)
                    covered.extend(dates)
                rules.require(
                    len(results) == intent["criteria"]["expected_main_months"][key]
                    and len(covered) == len(set(covered)),
                    "Match every eligible full month once, without an overlapping allocation.",
                )
                boundary = expected.difference(pd.DatetimeIndex(covered))
                rules.require(
                    len(boundary) == 21 and len(covered) + len(boundary) == len(book),
                    "Retain every initial-crossing and terminal boundary day separately.",
                )
                boundary_input = {
                    "path_key": key,
                    "source_account": case["account"],
                    "boundary_dates": boundary.strftime("%Y-%m-%d").tolist(),
                    "NET_returns_hex": [float(v).hex() for v in book.loc[boundary, "return"]],
                    "actual_cost_hex": [float(v).hex() for v in book.loc[boundary, "cost"]],
                }
                retain({"phase": "boundary_input", "case": boundary_input})
                summary = {
                    **rules.summarize_main(results),
                    "original_formal_return_rows": len(book),
                    "retained_boundary_return_rows": len(boundary),
                    "all_original_return_rows_retained": True,
                    "boundary_returns_and_initial_costs_remain_in_original_verdict": True,
                }
                retain({"phase": "boundary_resolved", "result": summary, "path_key": key})
                contexts[key] = summary
            prefixes = []
            for cut in (1, 2, 13, 61, 121):
                lifecycle.runtime_guard()
                prior = None
                for index in range(cut):
                    before, after = old_records[2 * index : 2 * index + 2]
                    check = rules.inspect_game(before, after, prior)
                    rules.require(
                        digest_json(check) == digest_json(games[index]),
                        "Future records changed a prior scope label.",
                    )
                    prior = after["audit"]
                through = games[cut - 1]["decision_session"]
                checked_months = 0
                for value, saved in zip(month_inputs, month_results, strict=True):
                    if saved["end"] <= through:
                        rules.require(
                            digest_json(rules.inspect_main_month(value)) == digest_json(saved),
                            "Future main records changed a prior complete-month loss label.",
                        )
                        checked_months += 1
                prefixes.append(
                    {
                        "source_decisions": cut,
                        "through": through,
                        "main_month_cases": checked_months,
                        "exact_retained_prefix": True,
                    }
                )
        rules.require(
            len(month_results) == 356
            and record_count == 962
            and sum(v["original_formal_return_rows"] for v in contexts.values()) == 7532
            and counts() == intent["actual_authoritative_counts"],
            "The complete fixed diagnosis scope and financial history must remain.",
        )
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "study_id": intent["criteria"]["study_id"],
                "analysis_intent": reference(INTENT),
                "source_complete_financial_result": catalogue["source_public_result"],
                "source_game_summary": rules.summarize_games(games, catalogue["source_expert_ids"]),
                "four_original_main_books": contexts,
                "source_decisions": 121,
                "mature_source_months": 119,
                "complete_main_month_cases": 356,
                "original_main_return_rows": 7532,
                "retained_boundary_return_rows": 84,
                "actual_input_resolved_records": reference(trace),
                "actual_input_resolution_record_count": record_count,
                "exact_retained_prefixes": prefixes,
                "actual_authoritative_counts": counts(),
                "source_learning_targets_accounts_or_prior_diagnoses_replayed": False,
                "new_financial_trials": 0,
                "new_rankings_targets_or_accounts": 0,
                "new_source_or_RF_or_event_Adapter": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_analysis_complete",
            private_result=reference(OUTPUT),
            analysis_intent=reference(INTENT),
            next_step="export_sanitized_scope_counts_once",
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
                **{
                    k: v
                    for k, v in saved.items()
                    if k not in ("analysis_intent", "actual_input_resolved_records")
                },
                "raw_source_game_or_dated_NET_rows_exported": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))
