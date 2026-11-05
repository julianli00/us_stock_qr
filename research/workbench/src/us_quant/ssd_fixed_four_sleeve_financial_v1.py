from __future__ import annotations

import argparse
import json
from fractions import Fraction
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import fixed_four_sleeve_expert_v1 as law
from us_quant import ssd_calendar_seasonal_financial_v1 as shared
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import digest_json, file_digest, read_json, utc_now, write_json

ROOT, ID = base.ROOT, law.ID
STUDY = "ssd_fixed_four_sleeve_financial_20261105_v1"
POLICY = ROOT / "config/fixed-four-sleeve-financial-v1.json"
FOLDER = base.REPORTS / "fixed-four-sleeve-financial-20261105-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
PREPARED = FOLDER / "published-native-targets-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/fixed_four_sleeve_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/fixed_four_sleeve_financial_20261105_results_v1.json"
KIND = shared.KIND
lifecycle, reference, unique_refs = shared.lifecycle, base.reference, shared.unique_refs
run_calendar, verify_opinions = shared.run_calendar, shared.verify_opinions
OWN = (
    "config/fixed-four-sleeve-financial-v1.json",
    "src/us_quant/ssd_fixed_four_sleeve_financial_v1.py",
    "tests/test_ssd_fixed_four_sleeve_financial_v1.py",
    "FIXED_FOUR_SLEEVE_FINANCIAL_V1.md",
    "config/fixed-four-sleeve-expert-v1.json",
    "src/us_quant/fixed_four_sleeve_expert_v1.py",
    "tests/test_fixed_four_sleeve_expert_v1.py",
    "FIXED_FOUR_SLEEVE_EXPERT_V1.md",
)
REFERENCES = (
    "primary_policy",
    "primary_preparation_freeze",
    "primary_preparation_completion",
    "market_parent_freeze",
)


def _shared(name, *args, **kwargs):
    function = getattr(shared, name)
    isolated = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    return isolated(*args, **kwargs)


def events(kind):
    return _shared("events", kind)


def packages():
    return shared.packages()


def dependencies():
    return _shared("dependencies")


def check_frozen(frozen):
    return shared.check_frozen(frozen)


def inputs():
    return _shared("inputs")


def packet_body(frozen, registered, readiness):
    return _shared("packet_body", frozen, registered, readiness)


def checked_packet(frozen):
    return _shared("checked_packet", frozen)


def register():
    return _shared("register")


def publication(frozen, commit):
    return _shared("publication", frozen, commit)


def publication_fence():
    return _shared("publication_fence")


def prepare_source(data, frozen):
    return _shared("prepare_source", data, frozen)


def source_frame(prepared):
    return _shared("source_frame", prepared)


def evaluate_path(data, targets, frozen, window, scenario, benchmarks):
    return _shared("evaluate_path", data, targets, frozen, window, scenario, benchmarks)


def prefix_audits(data, targets):
    return _shared("prefix_audits", data, targets)


def record_failure(stage, error):
    return _shared("record_failure", stage, error)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Fixed-policy progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "fixed_four_sleeve_financial_" + phase,
            "study_id": STUDY,
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def validate_policy(policy):
    original = read_json(shared.POLICY)
    references = set(REFERENCES)
    fixed = {
        **{key: value for key, value in original.items() if key not in references},
        "study_id": STUDY,
        "candidate_id": ID,
        "prior_journal_sequence": 992,
        "prior_journal_head_sha256": (
            "1586cb6695a5f09652895362b77e3b4afc2db36e8cfecf19a8f49a625bec1332"
        ),
        "prior_completed_configurations": 261,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | references
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in fixed.items()
        )
        or policy["primary_policy"] != reference(law.POLICY)
    ):
        raise QuantError("Preserve the frozen four-bucket law and original funded contract.")
    for key in references:
        base.checked_reference(policy[key])
    primary = read_json(base.checked_reference(policy["primary_preparation_freeze"]))
    if (
        primary["prior_completed_configurations"] != 261
        or primary["prior_journal_sequence"] != 992
        or primary["registered"] is not False
        or primary["real_new_policy_targets_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("Primary preparation is not a financial run or inherited prior259.")
    for row in primary["implementation_and_test_inputs"]:
        base.checked_reference(row)
    law.validate(read_json(law.POLICY))


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v2.json")
    suites = ElementTree.parse(FOLDER / "financial-tests-v2.xml").getroot().findall(".//testsuite")
    if (
        not suites
        or any(int(row.get(key, 0)) for row in suites for key in ("failures", "errors", "skipped"))
        or proof["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or proof["passed_tests"] != sum(int(row.get("tests", 0)) for row in suites)
        or proof["ruff_returncode"] != 0
        or proof["synthetic_only"] is not True
    ):
        raise QuantError("Bind the exact synthetic financial program and publication checks.")
    return [
        reference(FOLDER / name)
        for name in (
            "validation-source-v2.json",
            "financial-tests-v2.xml",
            "financial-tests-v2.log",
            "financial-ruff-v2.log",
        )
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve complete or partial fixed-policy financial preparation.")
    with lifecycle.research_job():
        policy = read_json(POLICY)
        validate_policy(policy)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 992
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 261
        ):
            raise QuantError("Require the actual closedJ992 baseline before new outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        parent = read_json(base.checked_reference(policy["market_parent_freeze"]))["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        law.checked_close(data.close)
        deps = unique_refs(
            [*dependencies(), reference(shared.POLICY), reference(ROOT / "tests/conftest.py")]
        )
        frozen = {
            **{
                key: value
                for key, value in parent.items()
                if key not in {"expert_pool", "expert_pool_definition_role"}
                and not key.startswith("source_")
            },
            "study_id": STUDY,
            "policy": reference(POLICY),
            "primary_signal_policy": policy["primary_policy"],
            "execution_policy": reference(POLICY),
            "account_namespace": "fixed_four_sleeve_published_program_v1",
            "construction": law.CANDIDATE["construction"],
            "construction_inputs": [policy["primary_policy"]],
            "decision_program_kind": KIND,
            "expert_pool_definition_role": "one_complete_source_policy_not_pool_expansion",
            "main_capital_initialization": policy["initialization"],
            "prepublication_readiness": "synthetic_Interface_and_pinned_market_identity_only",
            "historical_target_proof_stage": policy["historical_target_proof_stage"],
            "prior_journal_sequence": 992,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 261,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / path) for path in OWN if path.endswith(".md")],
                ]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *[policy[key] for key in REFERENCES[1:]],
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "financial-design-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    *validation_refs(),
                    *[
                        reference(FOLDER / name)
                        for name in (
                            "preflight-failure-v1.json",
                            "validation-source-v1.json",
                            "financial-tests-v1.xml",
                            "financial-tests-v1.log",
                            "financial-ruff-v1.log",
                        )
                    ],
                ]
            ),
            "decision_information_clock": law.CLOCK,
            "new_source_requests": 0,
            "new_economic_factor_definitions": 0,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "package_versions": packages(),
                "new_real_source_targets_accounts_or_metrics_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_program_without_real_source_outputs")
    return reference(INTENT)


def verify_targets(close, targets, audits, records):
    dates = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    if (
        not targets.index.equals(close.index)
        or not targets.columns.equals(close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
        or len(records) != 2 * len(dates)
    ):
        raise QuantError("Preserve every native month end without a price warmup or fake anchor.")
    defensive = ("TLT", "GLD", "BIL")
    oracle = [
        float(Fraction(49, 1200))
        if name in law.CORE
        else float(Fraction(49, 200))
        if name in defensive
        else 0.0
        for name in law.UNIVERSE
    ]
    for index, day in enumerate(dates):
        expected_before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": law.CLOCK,
            "fixed_capital_buckets": ["six_style_core", *defensive],
            "bucket_weights_within_ETF_budget": [0.25] * 4,
            "price_or_future_return_observed_to_choose_weight": False,
        }
        expected_audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": law.CLOCK,
            "fixed_capital_buckets": ["six_style_core", *defensive],
            "bucket_weights_within_ETF_budget": [0.25] * 4,
            "selected_assets": [*law.CORE, *defensive],
            "BIL_is_actual_ETF_not_idle_cash": True,
            "idle_cash_reserve": 0.02,
            "price_return_covariance_or_forecast_used_to_choose_weights": False,
            "capital_buckets_claimed_independent_strategy_experts": False,
            "counterfactual_historical_policy_availability": True,
            "actual_policy_introduction_or_admission_backdated": False,
            "mature_NET_score_inferred_from_fixed_target_availability": False,
            "independent_forward_validation": False,
        }
        before, after = records[2 * index : 2 * index + 2]
        if (
            before != expected_before
            or audits[index] != expected_audit
            or after["record"] != "decision_resolved"
            or after["audit"] != expected_audit
            or after["complete_target"] != targets.loc[day].tolist()
            or after["asset_order"] != list(close.columns)
            or not np.array_equal(oracle, targets.loc[day].to_numpy())
        ):
            raise QuantError("Independent fractional weights or fixed-input-before-target failed.")
    return {
        "native_month_end_opinions": len(dates),
        "all_independent_fractional_four_bucket_targets_exact": True,
        "source_price_binary64_inputs_checked": 0,
        "market_prices_used_for_weight_choice": False,
        "BIL_2450bps_of_capital_confused_with_idle_cash": False,
    }


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal fixed-policy financial failure.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("A completed fixed-policy result cannot change on recovery.")
            for key, row in saved["review"]["paths"].items():
                base.verify_path(row, frozen, ID, key)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            row
            for row in base.journal_events(base.JOURNAL)
            if row["kind"] == "publication" and row["body"].get("study_id") == STUDY
        ]
        if not fences:
            if events("path_started") or PREPARED.exists():
                raise QuantError("No fixed-policy historical output before actual publication.")
            base.append_event(
                base.JOURNAL,
                "publication",
                None,
                {
                    "study_id": STUDY,
                    "verification": published,
                    "candidate_accounts_previously_started": False,
                    "real_source_targets_previously_computed": False,
                },
            )
        elif len(fences) != 1 or fences[0]["body"]["verification"]["published_commit"] != commit:
            raise QuantError("Keep the unique actual fixed-policy publication fence.")
        try:
            prepared = prepare_source(data, frozen)
            targets = source_frame(prepared)
            paths = {}
            protocol = read_json(base.PROTOCOL)
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, targets, frozen, window, scenario, benchmarks)
            audits = prefix_audits(data, targets)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": (
                        "actual_published_fixed_four_sleeve_" "native_and_main_book_outputs_v1"
                    ),
                    "proof_stage": "after_four_actual_published_program_paths",
                    "native_source_proof": prepared,
                    "real_main_capital_prefix_audit": audits,
                    "all_original_dates_and_fixed_policy_retained": True,
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
                    "native_source_proof": prepared,
                    "real_main_capital_prefix_audit": audits,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_and_separate_technical_expert_admission")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    data = read_json(RESULTS)
    if set(data["paths"]) != base.PATH_KEYS or len(events("review")) != 1:
        raise QuantError("Export only a complete actually reviewed fixed policy.")
    for key, ref in data["review"]["paths"].items():
        base.verify_path(ref, data["frozen"], ID, key)
    body = {
        "schema_version": 1,
        "record_type": "published_fixed_four_sleeve_four_path_results",
        "study_id": STUDY,
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
                    "actual_source_opinion_verification",
                )
            }
            for key, row in data["paths"].items()
        },
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "new_financial_rules_evaluated": 1,
        "new_formal_funded_paths": 4,
        "separately_versioned_auxiliary_main_book_prefix_runs": 5,
        "all_real_prefix_audits_exact": True,
        "real_native_targets_generated_after_actual_publication": True,
        "technical_expert_admission_review_complete": False,
        "expert_pool_changed": False,
        "prior_policies_or_accounts_or_benchmarks_replayed": False,
        "raw_prices_targets_daily_books_private_intents_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_new_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="One published fixed four-sleeve source, no orders."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published preregistration commit.")
    actions = {"freeze": initialize, "register": register, "export": export}
    value = evaluate(args.commit) if args.action == "evaluate" else actions[args.action]()
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
