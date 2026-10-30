from __future__ import annotations

import argparse
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_path_admissible_baskets_v1 as rules
from us_quant import ssd_expert_online_log_growth_v1 as previous
from us_quant import ssd_expert_tail_risk_budget_v1 as source
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.ssd_expert_meta_research_v1 import failure
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-path-baskets-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-path-baskets-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_path_baskets_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_path_baskets_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-path-admissible-baskets-v1.json",
    "src/us_quant/expert_path_admissible_baskets_v1.py",
    "src/us_quant/ssd_expert_path_baskets_v1.py",
    "tests/test_expert_path_admissible_baskets_v1.py",
    "tests/test_ssd_expert_path_baskets_v1.py",
)
SHARED_CODE = "src/us_quant/expert_net_wealth_mixture_v1.py"
SHARED_POLICY = "config/expert-net-wealth-mixture-v1.json"
PARENT_SPEC = "3c9ba6f21342237d0d5e26a88e3c1c4f1cab25491498e07023d5a1fe0e466edb"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame


def dependency_inputs():
    policy = read_json(rules.POLICY)
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("The fixed original dependency parent changed.")
    return (
        unique_refs([*parent["code"], reference(ROOT / SHARED_CODE)]),
        unique_refs([*parent["policies"], reference(ROOT / SHARED_POLICY)]),
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("Keep the exact preserved ancestor dependency specification.")
    check_frozen(parent)
    previous.checked_packet(parent)
    saved = read_json(base.checked_reference(policy["source_results"]))
    if (
        saved["registration_packet"] != policy["source_registration"]
        or saved["reviews"][0]["status"] != "rejected"
        or len(saved["reviews"][0]["paths"]) != 4
    ):
        raise QuantError("Prior complete financial rejections cannot be relabelled or replayed.")
    data, experts, old_policy, _, old_refs, clocks, _, benchmarks = source.source_inputs()
    if any(
        policy[k] != old_policy[k]
        for k in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
        )
    ):
        raise QuantError("Change the allocator, not its valid experts or funded execution.")
    old_intent = read_json(previous.INTENT)
    dependency_code, dependency_policies = dependency_inputs()
    refs = unique_refs(
        [
            *old_refs,
            *old_intent["source_references"],
            *old_intent["model_and_test_files"],
            *old_intent["validation_evidence"],
            reference(previous.INTENT),
            *read_json(DESIGN)["source_references"],
            policy["source_registration"],
            policy["source_results"],
            policy["saved_assessment_divergence"],
            reference(ROOT / SHARED_CODE),
            reference(ROOT / SHARED_POLICY),
            *dependency_code,
            *dependency_policies,
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    required = {
        "test_all63_equal_capital_baskets_are_assessed_as_paths",
        "test_chronological_order_changes_risk_not_terminal_growth",
        "test_admissibility_is_not_average_individual_drawdown",
        "test_no_admissible_subset_is_explicit_funding_standby",
        "test_negative_finite_active_growth_is_not_a_cash_cutoff",
        "test_future_source_and_benchmark_changes_preserve_prior_targets",
        "test_independent_scalar_all_subset_weights_risks_scores_and_targets",
        "test_tampered_basket_risk_score_clock_or_weights_is_rejected",
        "test_cached_loader_never_invokes_closed_entrypoints",
        "test_complete_local_import_closure_is_bound",
        "test_complete_dependencies_precede_the_new_publication_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_all_original_paths_without_private_rows",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed basket, clock, source and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current basket rule and driver bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve any existing complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            len(events) != 540
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 49
            or counts["new_completed_strategy_evaluations"] != 45
            or counts["new_technical_failures"] != 4
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 50358
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual post-diagnosis journal and invocation.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 50358,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_meta_allocations_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_path_baskets_frozen",
            active_compute_job=None,
            expert_path_baskets_intent=reference(INTENT),
            pending_steps=["register_new_basket_law_before_real_assessments_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_allocations_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome basket allocation freeze changed.")
    dependency_code, dependency_policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *[reference(ROOT / p) for p in OWN[1:]],
                reference(ROOT / SHARED_CODE),
                *dependency_code,
            ]
        ),
        "policies": unique_refs(
            [
                *parent["policies"],
                reference(rules.POLICY),
                reference(ROOT / SHARED_POLICY),
                *dependency_policies,
            ]
        ),
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": (
            "unchanged_six_representatives_ordered_NET_path_admissibility"
        ),
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 223,
        "prior_journal_sequence": 540,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_ordered_NET_path_admissible_equal_baskets",
        "archive_import_dependencies_are_additional_strategy_experts": False,
        "public_archive_policy_pins_are_additional_active_allocators": False,
        "prior_evidence": unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(DESIGN),
                reference(INVOCATION),
                reference(INTENT),
            ]
        ),
        "validation_evidence": unique_refs(
            [*parent["validation_evidence"], *validation_references()]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, policy, frozen, benchmarks


def close_array(actual, expected, message, tolerance=1e-12):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or float(abs(a - b).max(initial=0)) > tolerance
    ):
        raise QuantError(message)


def scoring_returns():
    policy = read_json(rules.POLICY)
    result = {}
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    lookup = {r["path"]: r for r in parent["benchmark_artifacts"]}
    if set(policy["scoring_benchmarks"]) != {"SPY", "QQQ"}:
        raise QuantError("Both original NET scoring benchmarks are required.")
    for symbol in ("SPY", "QQQ"):
        ref = policy["scoring_benchmarks"][symbol]
        if lookup.get(ref["path"]) != ref:
            raise QuantError("Do not substitute raw prices or another benchmark cost scenario.")
        book = frame(ref)
        if (
            len(book) != 2512
            or str(book.index[0].date()) != "2016-10-06"
            or str(book.index[-1].date()) != "2026-10-05"
        ):
            raise QuantError("Keep the full original NET benchmark calendar.")
        result[symbol] = book["return"]
    return pd.DataFrame(result)


def independent_assessment(returns, benchmarks, targets, previous):
    curves = {}
    benchmark_growth = {}
    for label, start, stop in (("full_126", 0, 126), ("older_84", 0, 84), ("recent_42", 84, 126)):
        current = [1.0] * 6
        path = [tuple(current)]
        for day in returns[start:stop]:
            current = [current[i] * (1 + float(day[i])) for i in range(6)]
            path.append(tuple(current))
        curves[label] = path
        if label != "full_126":
            benchmark_growth[label] = [
                math.log(math.prod(1 + float(row[i]) for row in benchmarks[start:stop]))
                * 252
                / (stop - start)
                for i in range(2)
            ]
    cards = []
    for mask in range(1, 64):
        chosen = tuple(i for i in range(6) if mask & (1 << i))
        size = len(chosen)
        risks, scores = {}, {}
        for label, start, stop in (
            ("full_126", 0, 126),
            ("older_84", 0, 84),
            ("recent_42", 84, 126),
        ):
            peak = 1.0
            risk = 0.0
            value = 1.0
            for row in curves[label]:
                value = math.fsum(row[i] for i in chosen) / size
                peak = max(peak, value)
                risk = max(risk, 1 - value / peak)
            risks[label] = risk
            if label != "full_126":
                growth = math.log(value) * 252 / (stop - start)
                for j, symbol in enumerate(("SPY", "QQQ")):
                    scores[f"{label}:{symbol}"] = growth - benchmark_growth[label][j]
        score = min(scores.values())
        desired = [math.fsum(float(row[i]) for i in chosen) / size for row in targets]
        distance = math.fsum(abs(float(previous[i]) - desired[i]) for i in range(len(previous)))
        cards.append(
            {
                "mask": mask,
                "selected_indices": list(chosen),
                "expert_count": size,
                "assessment_drawdowns": risks,
                "annualized_NET_log_growth_differences": scores,
                "minimum_active_growth": score,
                "rank_score": round(score, 10),
                "nominal_target_L1_distance": distance,
                "rank_target_L1": round(distance, 12),
                "assessment_admissible": all(value <= 0.15 - 1e-10 for value in risks.values()),
            }
        )
    admissible = [c for c in cards if c["assessment_admissible"]]
    selected = (
        min(
            admissible,
            key=lambda c: (
                -c["rank_score"],
                c["rank_target_L1"],
                -c["expert_count"],
                c["selected_indices"],
            ),
        )
        if admissible
        else None
    )
    return cards, selected


def verify_targets(data, experts, targets, audits, benchmarks=None):
    policy = read_json(rules.POLICY)
    dates = rules.decision_sessions(data.close.index, policy)
    if benchmarks is None:
        benchmarks = scoring_returns()
    if (
        len(audits) != len(dates)
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
    ):
        raise QuantError("Keep every actual nominal decision and complete target coordinate.")
    previous = np.zeros(len(data.close.columns))
    maximum_weight = maximum_target = maximum_risk = maximum_score = maximum_distance = 0.0
    warmup = mature_count = standby = cases = 0
    mature_started = False
    for day, audit in zip(dates, audits, strict=True):
        observed = rules.observations(experts, day)
        sample = benchmarks.loc[:day].tail(126)
        mature = all(o.score_mature for o in observed) and len(sample) == 126
        if mature_started and not mature:
            raise QuantError("A missing post-maturity assessment cannot be a warmup.")
        mature_started = mature_started or mature
        if (
            audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if day == dates[0] else "completed_exchange_month_end")
            or audit["source_assessment_mature"] is not mature
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["source_observation_counts"] != [len(o.net_returns) for o in observed]
            or audit["source_observation_start"]
            != (
                str(observed[0].net_returns.index[0].date())
                if len(observed[0].net_returns)
                else None
            )
            or audit["source_observation_end"]
            != (
                str(observed[0].net_returns.index[-1].date())
                if len(observed[0].net_returns)
                else None
            )
            or audit["benchmark_observations"] != len(sample)
            or audit["source_shadow_wealth_or_protection_floor_transferred"] is not False
            or audit["assessment_admissibility_is_final_qualification"] is not False
            or audit["independent_forward_validation"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
        ):
            raise QuantError("A saved basket information clock or state separation changed.")
        close_array(
            audit["previous_nominal_meta_target"], previous, "The previous nominal target changed."
        )
        target_matrix = np.column_stack([o.target.to_numpy() for o in observed])
        assessment = audit["assessment"]
        if mature:
            if any(not o.net_returns.index.equals(sample.index) for o in observed):
                raise QuantError(
                    "All expert and benchmark observations need the same matured dates."
                )
            actual_cards = assessment["all_subset_assessments"]
            cards, selected = independent_assessment(
                np.column_stack([o.net_returns.to_numpy() for o in observed]),
                sample.to_numpy(),
                target_matrix,
                previous,
            )
            if len(actual_cards) != 63:
                raise QuantError("A basket cannot hide a rejected risk or opportunity case.")
            for actual, expected in zip(actual_cards, cards, strict=True):
                if any(
                    actual[k] != expected[k]
                    for k in (
                        "mask",
                        "selected_indices",
                        "expert_count",
                        "assessment_admissible",
                        "rank_score",
                        "rank_target_L1",
                    )
                ):
                    raise QuantError(
                        "Independent admissibility, ranking or tie selection disagrees."
                    )
                for label in ("full_126", "older_84", "recent_42"):
                    error = abs(
                        float(actual["assessment_drawdowns"][label])
                        - expected["assessment_drawdowns"][label]
                    )
                    if not math.isfinite(error):
                        raise QuantError(
                            "A nonfinite source-risk statistic cannot pass a certificate."
                        )
                    maximum_risk = max(maximum_risk, error)
                for label in ("older_84:SPY", "older_84:QQQ", "recent_42:SPY", "recent_42:QQQ"):
                    error = abs(
                        float(actual["annualized_NET_log_growth_differences"][label])
                        - expected["annualized_NET_log_growth_differences"][label]
                    )
                    if not math.isfinite(error):
                        raise QuantError(
                            "A nonfinite NET growth statistic cannot pass a certificate."
                        )
                    maximum_score = max(maximum_score, error)
                if not all(
                    math.isfinite(float(actual[k]))
                    for k in ("minimum_active_growth", "nominal_target_L1_distance")
                ):
                    raise QuantError(
                        "Every recorded growth or distance statistic must remain finite."
                    )
                maximum_score = max(
                    maximum_score,
                    abs(float(actual["minimum_active_growth"]) - expected["minimum_active_growth"]),
                )
                maximum_distance = max(
                    maximum_distance,
                    abs(
                        float(actual["nominal_target_L1_distance"])
                        - expected["nominal_target_L1_distance"]
                    ),
                )
            if max(maximum_risk, maximum_score, maximum_distance) > 1e-10:
                raise QuantError(
                    "Independent source-path risk, NET score or target distance disagrees."
                )
            admissible = sum(c["assessment_admissible"] for c in cards)
            if (
                assessment["enumerated_nonempty_subsets"] != 63
                or assessment["assessment_admissible_subsets"] != admissible
                or assessment["selected_mask"] != (selected["mask"] if selected else None)
                or assessment["selected_expert_indices"]
                != (selected["selected_indices"] if selected else [])
                or assessment["funding_standby"] is not (selected is None)
                or any(
                    assessment[k] is not False
                    for k in (
                        "assessment_paths_are_actual_main_or_future_drawdown_guarantee",
                        "subset_cases_are_independent_trials_or_validation",
                        "target_L1_is_actual_turnover_or_fee_forecast",
                    )
                )
            ):
                raise QuantError(
                    "The complete basket enumeration does not certify the applied choice."
                )
            mature_count += 1
            cases += 63
            if selected is None:
                expected_weights = None
                standby += 1
                if (
                    assessment["reason"] != "no_observed_path_admissible_subset"
                    or assessment["selected_minimum_active_growth"] is not None
                ):
                    raise QuantError("Funding standby cannot fabricate a score or an expert.")
            else:
                expected_weights = np.zeros(6)
                expected_weights[selected["selected_indices"]] = 1 / selected["expert_count"]
                if assessment["reason"] != "ordered_NET_drawdown_admissible_robust_active_growth":
                    raise QuantError("A selected basket cannot change its allocation law.")
                close_array(
                    [assessment["selected_minimum_active_growth"]],
                    [selected["minimum_active_growth"]],
                    "The selected source growth statistic changed.",
                    1e-10,
                )
        else:
            warmup += 1
            expected_weights = np.full(6, 1 / 6)
            if (
                assessment["reason"] != "unscored_initial_equal_weights"
                or assessment["all_subset_assessments"] != []
                or assessment["enumerated_nonempty_subsets"] != 0
                or assessment["assessment_admissible_subsets"] is not None
                or assessment["selected_mask"] is not None
                or assessment["selected_minimum_active_growth"] is not None
                or assessment["selected_expert_indices"] != list(range(6))
                or assessment["funding_standby"] is not False
                or assessment["safety_or_success_claimed"] is not False
            ):
                raise QuantError(
                    "An unscored initial intention cannot pass source risk assessment."
                )
        if expected_weights is None:
            if audit["expert_weights"] is not None:
                raise QuantError("A funding standby is not an expert-capital mixture.")
            desired = np.zeros(len(previous))
            desired[list(data.close.columns).index("BIL")] = 0.98
        else:
            close_array(
                audit["expert_weights"],
                expected_weights,
                "Independent basket capital weights disagree.",
            )
            maximum_weight = max(
                maximum_weight,
                float(abs(np.asarray(audit["expert_weights"]) - expected_weights).max()),
            )
            desired = np.array(
                [
                    math.fsum(
                        float(expected_weights[i]) * float(observed[i].target[symbol])
                        for i in range(6)
                    )
                    for symbol in data.close.columns
                ]
            )
        close_array(
            targets.loc[day].to_numpy(), desired, "Independent underlying basket target disagrees."
        )
        maximum_target = max(
            maximum_target, float(abs(targets.loc[day].to_numpy() - desired).max())
        )
        previous = targets.loc[day].to_numpy().copy()
    return {
        "all_monthly_decisions_checked": len(audits),
        "mature_assessment_decisions": mature_count,
        "unscored_initial_decisions": warmup,
        "funding_standby_decisions": standby,
        "independently_checked_subset_cases": cases,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_scalar_target_difference": maximum_target,
        "maximum_independent_subset_drawdown_difference": maximum_risk,
        "maximum_independent_active_growth_score_difference": maximum_score,
        "maximum_independent_nominal_target_distance_difference": maximum_distance,
        "main_account_or_old_source_pipeline_replayed": False,
        "observed_risk_admissibility_is_main_future_drawdown_guarantee": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation rather than regenerate it.")
        for ref in saved["prepared"].values():
            base.checked_reference(ref)
        return saved["prepared"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve unresolved preparation instead of repeating it.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    benchmarks = scoring_returns()
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        benchmarks,
        read_json(rules.POLICY),
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits, benchmarks)
    if (
        len(audits) != 121
        or independent["mature_assessment_decisions"] != 114
        or independent["unscored_initial_decisions"] != 7
        or independent["independently_checked_subset_cases"] != 7182
    ):
        raise QuantError("Keep all original decisions, 63 cases and explicit source maturity.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            benchmarks,
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future observations changed an earlier basket decision.")
        prefixes.append(str(day.date()))
    new_output_directory(TARGETS)
    write_text_atomic(TARGETS / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = targets.ffill().loc[anchor : window["end"]]
        active = [
            symbol
            for symbol in participation.source.donor.rules.CORE
            if (known[symbol] > 1e-10).any()
        ]
        if anchor not in known.index or known.isna().any().any() or len(active) < 3:
            raise QuantError("Keep the original anchor and actual factor coverage.")
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": (
                "formal main paths,not normalized source risk or growth assessments"
            ),
            "original_source_signal_recomputed": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
        },
    )
    prepared = {
        "targets": reference(TARGETS / "targets.csv"),
        "proof": reference(TARGETS / "proof.json"),
    }
    check_frozen(frozen)
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "prepared": prepared}
    )
    return prepared


def checked_packet(frozen):
    packet = read_json(PACKET)
    entry = common.packet_entry(CANDIDATE, frozen)
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != [entry]
        or packet["new_registered_candidates"] != 1
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or not pd.Timestamp(entry["registered_at"])
        <= pd.Timestamp(entry["target_proof_at"])
        <= created
        or any(created > pd.Timestamp(e["at"]) for e in matching_events("path_started", ID))
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The complete pre-account basket allocation packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, frozen, _ = inputs()
        old = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] != ID]
        if len(old) != 540 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve every old completed or failed financial identity.")
        if matching_events("technical_failure", ID):
            raise QuantError("A terminal new version cannot be silently retried.")
        found = matching_events("registered", ID)
        body = {"candidate": CANDIDATE, "frozen": frozen}
        if found and (len(found) != 1 or found[0]["body"] != body):
            raise QuantError("Registered code, inputs and policy cannot change.")
        if not found:
            append_event(base.JOURNAL, "registered", ID, body)
        try:
            if not matching_events("target_proof", ID):
                current.progress(
                    "preparing_registered_ordered_path_baskets",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, frozen)
                append_event(
                    base.JOURNAL,
                    "target_proof",
                    ID,
                    {
                        "registered_event_sha256": matching_events("registered", ID)[0]["sha256"],
                        **prepared,
                    },
                )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [common.packet_entry(CANDIDATE, frozen)],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 1,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "formal main paths, not source risk assessments",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_path_baskets_preparation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_path_baskets_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "expert_path_baskets_registered_waiting_publication",
            active_compute_job=None,
            expert_path_baskets_registration=reference(PACKET),
            pending_steps=["publish_exact_basket_packet_before_four_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence, not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def state(frozen):
    reviews, failed, completed = (
        matching_events(k, ID) for k in ("review", "technical_failure", "path_completed")
    )
    if (
        len(reviews) > 1
        or len(failed) > 1
        or len({e["body"]["path_key"] for e in completed}) != len(completed)
    ):
        raise QuantError("Duplicate terminal events cannot select another winner.")
    paths = {
        e["body"]["path_key"]: base.verify_path(
            e["body"]["receipt"], frozen, ID, e["body"]["path_key"]
        )
        for e in completed
    }
    if reviews:
        if (
            failed
            or set(paths) != base.PATH_KEYS
            or reviews[0]["body"]["paths"] != paths
            or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"] is not True
        ):
            raise QuantError("A complete review requires every actual original path.")
        return reviews[0]["body"]
    return {
        "candidate_id": ID,
        "status": "technical_failure" if failed else "pending",
        "paths": paths,
        "failure_event_sha256": failed[0]["sha256"] if failed else None,
        "failure_stage": failed[0]["body"]["stage"] if failed else None,
    }


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != {ID: state(frozen)}:
                raise QuantError("Read preserved complete outcomes instead of replaying them.")
            return reference(RESULTS)
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_path_baskets",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            if not matching_events("review", ID) and not matching_events("technical_failure", ID):
                target = frame(matching_events("target_proof", ID)[0]["body"]["targets"])
                paths = {}
                for window in read_json(base.PROTOCOL)["windows"]:
                    for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_path_baskets",
                            active_candidate=ID,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, ID, window, scenario, benchmarks
                        )
                actual = participation.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in actual.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    ID,
                    {
                        "candidate_id": ID,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": actual,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            write_new_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": {ID: state(frozen)},
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "new_shadow_account_runs": 0,
                    "new_shadow_recommendation_artifacts": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "expert_path_baskets_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_path_baskets_results=reference(RESULTS),
                pending_steps=["publish_all_four_basket_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_path_baskets_evaluation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_path_baskets_evaluation_interrupted", active_compute_job=None)
            raise


def public_result():
    base.storage_guard()
    _, _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = state(frozen), read_json(RESULTS)
    if saved["candidates"] != {ID: actual} or actual["status"] == "pending":
        raise QuantError("Publish actual complete outcomes or retained technical failure only.")
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"]["cagr"]
        for r in benchmarks["benchmarks"]
    }
    paths = []
    for key, path in actual["paths"].items():
        year, scenario = key.split("-")
        years = int(year[:-1])
        paths.append(
            {
                **{
                    k: path[k]
                    for k in (
                        "path_key",
                        "metrics",
                        "execution",
                        "gates",
                        "all_point_gates_pass",
                        "excess_net_cagr_vs_spy",
                        "excess_net_cagr_vs_qqq",
                        "stress_vs_base_cost_qqq_diagnostic",
                        "independent_bt_max_equity_difference_usd",
                        "independent_monthly_feedback",
                    )
                },
                "years": years,
                "scenario": scenario,
                "benchmark_net_cagr": {s: lookup[years, scenario, s] for s in ("SPY", "QQQ")},
                "actual_factor_participation": actual.get("actual_factor_participation", {}).get(
                    key
                ),
            }
        )
    counts = common.counts(
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] == ID]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": [
                {
                    "candidate_id": ID,
                    "status": actual["status"],
                    "paths": paths,
                    "reviewed_at": actual.get("reviewed_at"),
                    "failure_event_sha256": actual.get("failure_event_sha256"),
                    "failure_stage": actual.get("failure_stage"),
                    "all_four_paths_same_candidate_and_frozen_parameters": actual.get(
                        "all_four_paths_same_candidate_and_frozen_parameters", False
                    ),
                    "independent_forward_validation": False,
                    "order_authority": False,
                }
            ],
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="One ordered NET drawdown-admissible expert basket policy; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Hierarchical expert research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication SHA.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Hierarchical expert research blocked: {error}\n")


if __name__ == "__main__":
    main()
