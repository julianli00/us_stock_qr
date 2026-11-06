from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import quintet_return_cost_bridge_v1 as rules
from us_quant import ssd_benchmark_opportunity_quintet_v1 as source
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json, write_text_atomic

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "quintet-return-cost-bridge-20261106-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/quintet_return_cost_bridge_20261106_v1.json"
OWN = (
    "config/quintet-return-cost-bridge-v1.json",
    "src/us_quant/quintet_return_cost_bridge_v1.py",
    "src/us_quant/ssd_quintet_return_cost_bridge_v1.py",
    "tests/test_quintet_return_cost_bridge_v1.py",
    "QUINTET_RETURN_COST_BRIDGE_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    previous = read_json(base.checked_reference(inputs["opportunity_completed_cycle"]))
    return unique_refs(
        [
            *previous["exact_evidence_references"],
            *inputs["exact_book_and_receipt_refs"],
            *[
                inputs[k]
                for k in (
                    "risk_free",
                    "opportunity_completed_cycle",
                    "opportunity_public_results",
                    "cached_volatility_public_results",
                )
            ],
            *[
                reference(FOLDER / name)
                for name in (
                    "source-inputs-v1.json",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                    "capture-preflight-v1.py",
                    "capture-RF-literal-failure-v1.json",
                    "capture-preflight-failed-RF-literal-v1.py",
                )
            ],
        ]
    )


def progress(phase, **fields):
    actual = source.template.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "A return-cost bridge cannot substitute projected financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quintet_return_cost_bridge_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **fields,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


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
        "Keep exact separately frozen bridge code and successful synthetic validation.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(p) for p in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep the full accepted Python string and every captured package row.",
    )
    return frozen


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial bridge freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        rules.validate(read_json(rules.POLICY))
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            source.template.current_counts(base.journal_events(base.JOURNAL))
            == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1416,
            "Freeze against actual J1416 with all 297 financial completions preserved.",
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
            "Bind synthetic dollar/log identities, exact prefixes and source-scope refusal.",
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
                "actual_bridge_outputs_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress(
            "frozen",
            implementation_freeze=reference(INTENT),
            next_step="checkpoint_before_actual_bridge_rows",
        )
    return reference(INTENT)


def validate_loaded_book(account, expected, risk_free):
    rules.require(
        account.index.equals(expected)
        and tuple(account.columns) == rules.COLUMNS
        and account["risk_free"].equals(risk_free.loc[expected]),
        "Keep every original return date, recorded-book column and exact canonical RF.",
    )


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not RESULT.exists() and not (FOLDER / "analysis-attempt-v1.json").exists(),
        "Preserve the first complete or partial actual bridge, without replaying it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(
            FOLDER / "analysis-attempt-v1.json",
            {
                "started_at": utc_now(),
                "frozen_intent": reference(INTENT),
            },
        )
        policy = read_json(rules.POLICY)
        inputs = read_json(base.checked_reference(policy["source_inputs"]))
        rf = pd.read_csv(
            base.checked_reference(inputs["risk_free"]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        rules.require(list(rf.columns) == ["risk_free"], "Use the original canonical RF column.")
        records = FOLDER / "actual-inputs-resolved-v1.jsonl"
        cases, prefix_count, row_count = [], 0, 0
        with records.open("x") as stream:
            for new_id, paths in inputs["cases"].items():
                for key in rules.PATHS:
                    lifecycle.runtime_guard()
                    expected = sessions(
                        "2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05"
                    )
                    case, books, checks = paths[key], {}, {}
                    for role in rules.ROLES:
                        evidence = case[role]
                        receipt = read_json(base.checked_reference(evidence["receipt"]))
                        rules.require(
                            receipt["candidate_id"] == evidence["candidate_id"]
                            and receipt["path_key"] == key
                            and receipt["broker_orders"] == 0
                            and receipt["frozen_specification_sha256"]
                            == evidence["registered_frozen_sha256"],
                            "Require the exact already completed source identity and money book.",
                        )
                        account = pd.read_csv(
                            base.checked_reference(evidence["account"]),
                            index_col=0,
                            parse_dates=True,
                            float_precision="round_trip",
                        )
                        validate_loaded_book(account, expected, rf["risk_free"])
                        stream.write(
                            json.dumps(
                                {
                                    "phase": "before_book_bridge",
                                    "candidate_id": evidence["candidate_id"],
                                    "path_key": key,
                                    "role": role,
                                    "account": evidence["account"],
                                    "receipt": evidence["receipt"],
                                    "return_rows": len(account),
                                    "actual_bridge_rows_computed": False,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                        daily, summary = rules.decompose(account)
                        rules.require(
                            abs(
                                summary["recorded_paid_costs_USD"]
                                - evidence["cached_paid_costs_USD"]
                            )
                            <= rules.MONEY_TOLERANCE,
                            "Recorded full-book costs must equal existing paid-cost evidence.",
                        )
                        cuts = (1, 21, 126, len(expected) // 2, len(expected))
                        checks[role] = []
                        for stop in cuts:
                            lifecycle.runtime_guard()
                            early, _ = rules.decompose(account.iloc[:stop])
                            checks[role].append(rules.check_prefix(daily, early))
                            prefix_count += 1
                        path = FOLDER / "daily" / f"{new_id}-{key}-{role}.csv"
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        write_text_atomic(path, daily.to_csv(float_format="%.17g"))
                        books[role] = (summary, reference(path))
                        stream.write(
                            json.dumps(
                                {
                                    "phase": "after_book_bridge",
                                    "candidate_id": evidence["candidate_id"],
                                    "path_key": key,
                                    "role": role,
                                    "daily_bridge": reference(path),
                                    "summary": summary,
                                    "prefix_checks": checks[role],
                                },
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                        row_count += len(account)
                    summaries = {role: value[0] for role, value in books.items()}
                    paired = rules.paired_summary(summaries)
                    cases.append(
                        {
                            "opportunity_candidate_id": new_id,
                            "path_key": key,
                            "source_case_references": case,
                            "both_original_books": summaries,
                            "opportunity_minus_cached_volatility": (
                                paired["opportunity_minus_cached_volatility"]
                            ),
                            "daily_bridges": {role: value[1] for role, value in books.items()},
                            "prefix_checks": checks,
                        }
                    )
        rules.require(
            len(cases) == 8 and prefix_count == 80 and row_count == 30128,
            "Keep all original sixteen books, eighty prefixes and every formal return row.",
        )
        counts = source.template.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"], "A bridge is zero new trials."
        )
        write_new_receipt(
            RESULT,
            {
                "completed_at": utc_now(),
                "study_id": policy["study_id"],
                "cases": cases,
                "input_resolved_records": reference(records),
                "actual_input_and_resolved_record_count": 32,
                "total_recorded_return_rows": row_count,
                "exact_prefix_checks": prefix_count,
                "actual_authoritative_counts": counts,
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "fee_free_funded_policy_counterfactual_claimed": False,
                "fee_or_expert_causal_alpha_attribution_claimed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress(
            "complete",
            diagnostic=reference(RESULT),
            next_step="publish_only_scoped_readonly_bridge_without_new_financial_metrics",
        )
    return reference(RESULT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULT)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": saved["study_id"],
                "completed_at": saved["completed_at"],
                "paired_cases": [
                    {
                        "opportunity_candidate_id": case["opportunity_candidate_id"],
                        "path_key": case["path_key"],
                        "both_original_book_summaries": case["both_original_books"],
                        "opportunity_minus_cached_volatility": case[
                            "opportunity_minus_cached_volatility"
                        ],
                    }
                    for case in saved["cases"]
                ],
                "actual_input_and_resolved_record_count": saved[
                    "actual_input_and_resolved_record_count"
                ],
                "total_recorded_return_rows": saved["total_recorded_return_rows"],
                "exact_prefix_checks": saved["exact_prefix_checks"],
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "source_CAGR_Sharpe_or_qualification_recomputed": False,
                "fee_free_funded_policy_counterfactual_claimed": False,
                "fee_or_expert_causal_alpha_attribution_claimed": False,
                "raw_prices_targets_or_daily_book_rows_exported": False,
                "new_financial_trials": 0,
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
