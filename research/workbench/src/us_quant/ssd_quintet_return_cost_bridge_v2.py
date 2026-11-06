from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import quintet_return_cost_bridge_v2 as rules
from us_quant import ssd_quintet_return_cost_bridge_v1 as prior
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.storage import read_json, utc_now, write_text_atomic

base, ROOT, lifecycle, source = prior.base, prior.ROOT, prior.lifecycle, prior.source
reference, unique_refs = prior.reference, prior.unique_refs
FOLDER = base.REPORTS / "quintet-return-cost-bridge-20261106-v2"
INTENT = FOLDER / "diagnostic-implementation-freeze-v2.json"
RESULT = FOLDER / "diagnostic-result-v2.json"
PUBLIC = ROOT / "evidence/quintet_return_cost_bridge_20261106_v2.json"
OWN = (
    "config/quintet-return-cost-bridge-v2.json",
    "src/us_quant/quintet_return_cost_bridge_v2.py",
    "src/us_quant/ssd_quintet_return_cost_bridge_v2.py",
    "tests/test_quintet_return_cost_bridge_v2.py",
    "QUINTET_RETURN_COST_BRIDGE_V2.md",
    *prior.OWN,
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    return bound(prior.dependencies)()


def progress(phase, **fields):
    return bound(prior.progress)("v2_" + phase, **fields)


def source_refs():
    continuation = read_json(
        base.checked_reference(read_json(rules.POLICY)["reader_continuation_inputs"])
    )
    return unique_refs(
        [
            *continuation["original_v1_evidence_refs"],
            read_json(rules.POLICY)["reader_continuation_inputs"],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-reader-continuation-v2.py",
                    "preservation-baseline-v2.json",
                    "design-decision-v2.json",
                )
            ],
        ]
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
        "Keep exact V2 code and every immutable failed V1 artifact.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v2.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(p) for p in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep the accepted full interpreter and every original distribution row.",
    )
    return frozen


def canonical_rf(inputs, continuation):
    rules.require(
        inputs["risk_free"] == continuation["original_frozen_market"]["risk_free"]
        and continuation["new_RF_view_or_new_source_adapter"] is False
        and continuation["unchanged_original_canonical_reader"]
        == "us_quant.research_program.review_market",
        "Use the original identical RF file and pinned authoritative reader,not a new view.",
    )
    return base.review_market({"market": continuation["original_frozen_market"]}, ROOT).risk_free


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v2.json").exists(),
        "Preserve the first complete or partial separately versioned V2 freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v2.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        rules.validate(policy)
        continuation = read_json(base.checked_reference(policy["reader_continuation_inputs"]))
        baseline = read_json(FOLDER / "preservation-baseline-v2.json")
        rules.require(
            source.template.current_counts(base.journal_events(base.JOURNAL))
            == baseline["actual_counts"]
            and continuation["actual_V2_bridge_rows_computed"] == 0
            and continuation["all16_RF_columns_exact_against_original_reader"] is True,
            "Preserve all original financial counts and freeze V2 before actual outputs.",
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
            "Bind original-reader refusal checks and unchanged first V1 arithmetic tests.",
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
                    reference(prior.FOLDER / "validation-source-v1.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "actual_V2_bridge_outputs_computed": False,
                "original_V1_reader_contract_remains_failed": True,
                "original_V1_arithmetic_and_all_limits_reused_unchanged": True,
                "new_RF_view_or_source_adapter": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress(
            "frozen",
            implementation_freeze=reference(INTENT),
            next_step="checkpoint_before_actual_V2_bridge_rows",
        )
    return reference(INTENT)


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not RESULT.exists() and not (FOLDER / "analysis-attempt-v2.json").exists(),
        "Preserve the first complete or partial actual V2 run;never repeat failed V1.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(
            FOLDER / "analysis-attempt-v2.json",
            {
                "started_at": utc_now(),
                "frozen_intent": reference(INTENT),
            },
        )
        policy = read_json(rules.POLICY)
        inputs = read_json(base.checked_reference(policy["source_inputs"]))
        continuation = read_json(base.checked_reference(policy["reader_continuation_inputs"]))
        risk_free = canonical_rf(inputs, continuation)
        records = FOLDER / "actual-inputs-resolved-v2.jsonl"
        cases, prefix_count, row_count = [], 0, 0
        with records.open("x") as stream:
            for new_id, paths in inputs["cases"].items():
                for key in rules.PATHS:
                    lifecycle.runtime_guard()
                    expected = sessions(
                        "2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05"
                    )
                    case, summaries, daily_refs, checks = paths[key], {}, {}, {}
                    for role in rules.ROLES:
                        evidence = case[role]
                        receipt = read_json(base.checked_reference(evidence["receipt"]))
                        account = pd.read_csv(
                            base.checked_reference(evidence["account"]),
                            index_col=0,
                            parse_dates=True,
                            float_precision="round_trip",
                        )
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
                        prior.validate_loaded_book(account, expected, risk_free)
                        rules.require(
                            receipt["candidate_id"] == evidence["candidate_id"]
                            and receipt["path_key"] == key
                            and receipt["broker_orders"] == 0
                            and receipt["frozen_specification_sha256"]
                            == evidence["registered_frozen_sha256"],
                            "Use the exact already completed book identity and frozen law.",
                        )
                        daily, summary = rules.decompose(account)
                        rules.require(
                            abs(
                                summary["recorded_paid_costs_USD"]
                                - evidence["cached_paid_costs_USD"]
                            )
                            <= rules.MONEY_TOLERANCE,
                            "Keep the same recorded paid costs without a refund or replay.",
                        )
                        checks[role] = []
                        for stop in (1, 21, 126, len(expected) // 2, len(expected)):
                            lifecycle.runtime_guard()
                            early, _ = rules.decompose(account.iloc[:stop])
                            checks[role].append(rules.check_prefix(daily, early))
                            prefix_count += 1
                        path = FOLDER / "daily" / f"{new_id}-{key}-{role}.csv"
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        write_text_atomic(path, daily.to_csv(float_format="%.17g"))
                        summaries[role], daily_refs[role] = summary, reference(path)
                        stream.write(
                            json.dumps(
                                {
                                    "phase": "after_book_bridge",
                                    "candidate_id": evidence["candidate_id"],
                                    "path_key": key,
                                    "role": role,
                                    "daily_bridge": daily_refs[role],
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
                            "daily_bridges": daily_refs,
                            "prefix_checks": checks,
                        }
                    )
        rules.require(
            len(cases) == 8 and prefix_count == 80 and row_count == 30128,
            "Retain all original sixteen books and eighty exact scalar/primitive prefixes.",
        )
        counts = source.template.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"], "V2 adds zero financial trials."
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
                "original_V1_reader_contract_remains_failed": True,
                "unchanged_original_canonical_RF_reader": True,
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
            next_step="publish_only_readonly_V2_bridge_with_V1_failure_preserved",
        )
    return reference(RESULT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULT)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 2,
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
                "original_V1_reader_contract_remains_failed": True,
                "unchanged_original_canonical_RF_reader": True,
                "new_RF_view_or_source_adapter": False,
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
