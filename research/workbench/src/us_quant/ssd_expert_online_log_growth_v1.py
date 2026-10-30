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
from us_quant import expert_online_log_growth_v1 as rules
from us_quant import ssd_expert_hierarchical_risk_v1 as previous
from us_quant import ssd_expert_tail_risk_budget_v1 as source
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
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
FOLDER = base.REPORTS / "expert-online-log-growth-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-online-log-growth-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_online_log_growth_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_online_log_growth_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-online-log-growth-v1.json",
    "src/us_quant/expert_online_log_growth_v1.py",
    "src/us_quant/ssd_expert_online_log_growth_v1.py",
    "tests/test_expert_online_log_growth_v1.py",
    "tests/test_ssd_expert_online_log_growth_v1.py",
)
SHARED_CODE = "src/us_quant/expert_net_wealth_mixture_v1.py"
SHARED_POLICY = "config/expert-net-wealth-mixture-v1.json"
PARENT_SPEC = "46f5d7d6ac07e0e0f0ad24664b07ae4920d5d95f07ebb4c4489feaa191e3e5f8"
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
            policy["saved_opportunity_summary"],
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
        "test_identity_curvature_and_equal_periods_preserve_prior",
        "test_growth_feedback_changes_weights_without_covariance_ranking",
        "test_all_simplex_faces_and_border_certificate_agree",
        "test_full_month_clock_partial_skip_and_maturity_are_explicit",
        "test_future_source_changes_preserve_prior_allocations",
        "test_independent_monthly_state_weights_and_targets",
        "test_scalar_verifier_rejects_state_clock_score_and_weight_tampering",
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
        raise QuantError("Require completed online-state, clock, source and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current online rule and driver bytes.")
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
            len(events) != 528
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 48
            or counts["new_completed_strategy_evaluations"] != 44
            or counts["new_technical_failures"] != 4
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 48799
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual post-diagnosis journal and invocation.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 48799,
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
            "expert_online_log_growth_frozen",
            active_compute_job=None,
            expert_online_log_growth_intent=reference(INTENT),
            pending_steps=["register_new_online_law_before_real_updates_or_targets"],
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
        raise QuantError("The actual pre-outcome online allocation freeze changed.")
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
        "expert_pool_definition_role": "unchanged_six_representatives_online_NET_log_growth",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 222,
        "prior_journal_sequence": 528,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_causal_second_order_NET_log_growth",
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


def independent_projection(matrix, linear):
    candidates = []
    for mask in range(1, 64):
        face = tuple(i for i in range(6) if mask & (1 << i))
        n = len(face)
        block = np.zeros((n + 1, n + 1))
        block[:n, :n] = matrix[np.ix_(face, face)]
        block[:n, n] = block[n, :n] = 1
        try:
            answer = np.linalg.solve(block, np.r_[linear[list(face)], 1.0])
        except np.linalg.LinAlgError as error:
            raise QuantError("The independent positive-curvature bordered solve failed.") from error
        raw = answer[:n]
        if not np.isfinite(raw).all() or (raw < -1e-12).any() or abs(float(raw.sum()) - 1) > 1e-12:
            continue
        weights = np.zeros(6)
        weights[list(face)] = np.maximum(raw, 0)
        weights /= math.fsum(weights)
        objective = math.fsum(
            0.5 * float(weights[i]) * float(matrix[i, j]) * float(weights[j])
            for i in range(6)
            for j in range(6)
        ) - math.fsum(float(weights[i]) * float(linear[i]) for i in range(6))
        candidates.append((objective, face, weights))
    if not candidates:
        raise QuantError("The independent simplex projection has no feasible face.")
    return min(candidates, key=lambda value: (value[0], value[1]))[2]


def check_recorded_certificate(matrix, linear, weights, update):
    positive = weights > 1e-12
    gradient = np.array(
        [
            math.fsum(float(matrix[i, j]) * float(weights[j]) for j in range(6)) - float(linear[i])
            for i in range(6)
        ]
    )
    multiplier = float(update["simplex_multiplier"])
    if not math.isfinite(multiplier):
        raise QuantError("A projection multiplier cannot be nonfinite.")
    dual = gradient + multiplier
    errors = [
        float(abs(dual[positive]).max(initial=0)),
        float(np.maximum(-dual[~positive], 0).max(initial=0)),
        float(abs(weights * dual).max()),
    ]
    close_array(update["nonnegative_coordinate_duals"], dual, "The coordinate duals disagree.")
    close_array(
        [
            update["maximum_positive_stationarity_error"],
            update["maximum_inactive_dual_violation"],
            update["maximum_complementarity_error"],
        ],
        errors,
        "The recorded KKT residual changed.",
    )
    objective = math.fsum(
        0.5 * float(weights[i]) * float(matrix[i, j]) * float(weights[j])
        for i in range(6)
        for j in range(6)
    ) - math.fsum(float(weights[i]) * float(linear[i]) for i in range(6))
    close_array(
        [update["quadratic_objective"]],
        [objective],
        "The quadratic objective disagrees.",
        tolerance=1e-10,
    )
    face = update["selected_face"]
    if (
        max(errors) > 1e-10
        or update["enumerated_nonempty_faces"] != 63
        or not 1 <= update["feasible_faces"] <= 63
        or not face
        or sorted(set(face)) != face
        or any(type(i) is not int or not 0 <= i < 6 for i in face)
        or any(weights[i] != 0 for i in range(6) if i not in face)
    ):
        raise QuantError("The independent global quadratic KKT certificate failed.")
    return max(errors)


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
    ):
        raise QuantError("The complete initial/monthly target calendar changed.")
    learner = np.full(6, 1 / 6)
    applied_previous = learner.copy()
    curvature = np.eye(6)
    previous_day = None
    periods = warmup = 0
    maximum_weight = maximum_target = maximum_curvature = maximum_score = maximum_kkt = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != rules.MECHANISMS
            or audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or any(o.target_known_at > day or o.as_of != day for o in observed)
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["source_observation_counts"] != [len(o.net_returns) for o in observed]
            or any(
                not o.risk_free.equals(observed[0].risk_free)
                or not o.net_returns.index.equals(observed[0].net_returns.index)
                for o in observed
            )
            or audit["source_shadow_wealth_or_protection_floor_transferred"] is not False
            or audit["learning_score_is_actual_main_net_wealth"] is not False
            or audit["universal_NET_wealth_identity_or_bound_claimed"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("Independent source identities, clocks or capital separation changed.")
        close_array(
            audit["previous_applied_expert_weights"],
            applied_previous,
            "Previous main intentions changed.",
        )
        update = audit["online_update"]
        full = previous_day is not None and is_month_end(previous_day)
        partial = previous_day is not None and not is_month_end(previous_day)
        if audit["initial_partial_source_month_not_assimilated"] is not partial:
            raise QuantError("The initial partial learning interval was redated.")
        if full:
            period_dates = sessions(next_session(previous_day), day)
            if any(not period_dates.isin(o.net_returns.index).all() for o in observed):
                raise QuantError("An independent full learning interval is missing.")
            relatives = np.array(
                [
                    math.exp(
                        math.fsum(math.log1p(float(v)) for v in o.net_returns.loc[period_dates])
                    )
                    for o in observed
                ]
            )
            score = math.fsum(float(learner[i]) * float(relatives[i]) for i in range(6))
            gradient = relatives / score
            after = np.array(
                [
                    [
                        math.fsum((float(curvature[i, j]), float(gradient[i]) * float(gradient[j])))
                        for j in range(6)
                    ]
                    for i in range(6)
                ]
            )
            linear = np.array(
                [
                    math.fsum(float(after[i, j]) * float(learner[j]) for j in range(6))
                    + float(gradient[i])
                    for i in range(6)
                ]
            )
            expected = independent_projection(after, linear)
            if update is None:
                raise QuantError("Every complete source month must be assimilated exactly once.")
            close_array(
                update["source_NET_period_relatives"],
                relatives,
                "The NET source relatives disagree.",
            )
            close_array(
                update["learner_weights_before"],
                learner,
                "Future learner weights leaked into past assessment.",
            )
            close_array(update["gradient"], gradient, "The log-growth gradient changed.")
            close_array(update["curvature_before"], curvature, "Previous curvature changed.")
            close_array(
                update["curvature_after"], after, "The independent curvature update disagrees."
            )
            close_array(
                update["learner_weights_after"], expected, "Independent bordered weights disagree."
            )
            close_array(
                [update["pre_update_learning_score_relative"]],
                [score],
                "The pre-assimilation learning score changed.",
            )
            periods += 1
            if (
                update["complete_learning_intervals"] != periods
                or update["source_period_previous_decision"] != str(previous_day.date())
                or update["source_period_first_session"] != str(period_dates[0].date())
                or update["source_period_end"] != str(day.date())
                or update["source_period_sessions"] != len(period_dates)
                or update["score_recorded_before_assimilation"] is not True
                or update["persistent_array_buffer_bytes"] != 336
                or update["buffer_bytes_are_total_process_RSS"] is not False
                or update["learning_score_is_actual_main_net_wealth"] is not False
            ):
                raise QuantError("The source learning clock or statistical-state scope changed.")
            actual_weights = np.asarray(update["learner_weights_after"], dtype=float)
            if (
                actual_weights.shape != (6,)
                or not np.isfinite(actual_weights).all()
                or (actual_weights < 0).any()
                or abs(float(actual_weights.sum()) - 1) > 1e-12
            ):
                raise QuantError("The recorded online allocation is not funded.")
            maximum_kkt = max(
                maximum_kkt, check_recorded_certificate(after, linear, actual_weights, update)
            )
            maximum_weight = max(maximum_weight, float(abs(actual_weights - expected).max()))
            maximum_curvature = max(
                maximum_curvature, float(abs(np.asarray(update["curvature_after"]) - after).max())
            )
            maximum_score = max(
                maximum_score, abs(float(update["pre_update_learning_score_relative"]) - score)
            )
            learner, curvature = expected, after
        elif update is not None:
            raise QuantError("A fabricated initial source month cannot update learning state.")
        mature = all(len(o.net_returns) == 126 and o.score_mature for o in observed)
        applied = learner if mature else np.full(6, 1 / 6)
        warmup += not mature
        if (
            audit["complete_learning_intervals"] != periods
            or audit["source_assessment_mature"] is not mature
            or audit["reason"]
            != ("online_NET_log_growth_weights" if mature else "unscored_initial_equal_weights")
        ):
            raise QuantError("Unmatured source history cannot become a successful score.")
        close_array(
            audit["learner_expert_weights"], learner, "The recorded learning state changed."
        )
        close_array(
            audit["applied_expert_weights"], applied, "The applied maturity-gated weights disagree."
        )
        expected_target = [
            math.fsum(float(applied[i]) * float(observed[i].target[symbol]) for i in range(6))
            for symbol in data.close.columns
        ]
        close_array(
            targets.loc[day].to_numpy(),
            expected_target,
            "The independent underlying target disagrees.",
        )
        maximum_weight = max(
            maximum_weight, float(abs(np.asarray(audit["applied_expert_weights"]) - applied).max())
        )
        maximum_target = max(
            maximum_target, float(abs(targets.loc[day].to_numpy() - expected_target).max())
        )
        previous_day, applied_previous = day, applied
    return {
        "all_monthly_decisions_checked": len(audits),
        "complete_learning_intervals": periods,
        "unscored_initial_decisions": warmup,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_scalar_target_difference": maximum_target,
        "maximum_independent_curvature_difference": maximum_curvature,
        "maximum_independent_prequential_score_difference": maximum_score,
        "maximum_independent_KKT_error": maximum_kkt,
        "every_month_independently_replayed_as_statistics_not_account": True,
        "main_account_or_old_source_pipeline_replayed": False,
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
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        read_json(rules.POLICY),
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits)
    if (
        len(audits) != 121
        or independent["complete_learning_intervals"] != 119
        or independent["unscored_initial_decisions"] != 7
    ):
        raise QuantError(
            "Keep all original decisions, complete months and explicit source maturity."
        )
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future observations changed an earlier online allocation.")
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
            raise QuantError("The original anchor and actual factor coverage remain required.")
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            **independent,
            "initial_partial_source_months_not_assimilated": sum(
                a["initial_partial_source_month_not_assimilated"] for a in audits
            ),
            "learning_state_is_main_wealth": False,
            "universal_NET_wealth_theorem_claimed": False,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": "formal main paths, not surrogate source learning scores",
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
        raise QuantError("The complete pre-account online allocation packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, frozen, _ = inputs()
        old = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] != ID]
        if len(old) != 528 or old[-1]["sha256"] != rules.PRIOR_HEAD:
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
                    "preparing_registered_online_log_growth",
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
            failure("expert_online_log_growth_preparation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_online_log_growth_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_online_log_growth_registered_waiting_publication",
            active_compute_job=None,
            expert_online_log_growth_registration=reference(PACKET),
            pending_steps=["publish_exact_online_packet_before_four_accounts"],
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
                "evaluating_expert_online_log_growth",
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
                            "evaluating_expert_online_log_growth",
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
                "expert_online_log_growth_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_online_log_growth_results=reference(RESULTS),
                pending_steps=["publish_all_four_online_log_growth_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_online_log_growth_evaluation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_online_log_growth_evaluation_interrupted", active_compute_job=None
            )
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
        description="One online NET log-growth complete-expert policy; no trading."
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
