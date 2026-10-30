from __future__ import annotations

import argparse
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from us_quant import expert_full_covariance_risk_budget_v1 as rules
from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-full-covariance-risk-budget-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-full-covariance-risk-budget-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_full_covariance_risk_budget_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_full_covariance_risk_budget_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-full-covariance-risk-budget-v1.json",
    "src/us_quant/expert_full_covariance_risk_budget_v1.py",
    "src/us_quant/ssd_expert_full_covariance_risk_budget_v1.py",
    "tests/test_expert_full_covariance_risk_budget_v1.py",
    "tests/test_ssd_expert_full_covariance_risk_budget_v1.py",
)
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs = source.check_frozen, source.published_inputs
frame = source.frame


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("Keep the exact admitted six-expert source specification.")
    check_frozen(parent)
    source.checked_packet(parent)
    controls = read_json(base.checked_reference(policy["source_controls"]))
    dual = read_json(base.checked_reference(policy["additional_dual_relative_control"]))
    if (
        controls["frozen_specification_sha256"] != PARENT_SPEC
        or set(controls["candidates"]) != set(source.IDS)
        or set(dual["candidates"]) != {"expert_pool_six_dual_relative_pair_v1"}
        or any(
            r["status"] != "rejected" or len(r["paths"]) != 4
            for r in [*controls["candidates"].values(), *dual["candidates"].values()]
        )
    ):
        raise QuantError("Preserve all three complete rejected controls without replay.")
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
            "switch_cost_bps_per_side",
            "switch_commission_per_changed_asset_usd",
            "changed_asset_numeric_tolerance",
        )
    ):
        raise QuantError("Change the allocator, not its admitted pool or funded execution.")
    old_intent = read_json(source.INTENT)
    design = read_json(DESIGN)
    refs = unique_refs(
        [
            *old_refs,
            *old_intent["source_references"],
            *old_intent["model_and_test_files"],
            *old_intent["validation_evidence"],
            reference(source.INTENT),
            *design["source_references"],
            *[
                policy[k]
                for k in (
                    "source_registration",
                    "source_controls",
                    "additional_dual_relative_control",
                    "risk_diagnosis",
                )
            ],
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
    if not {
        "test_whole_pool_budgets_are_equal_without_asset_group_constraints",
        "test_singular_positive_covariance_is_not_regularized",
        "test_missing_history_and_zero_risk_are_explicit_not_success",
        "test_future_source_changes_preserve_prior_allocations",
        "test_scalar_optimizer_verifies_weights_covariance_and_targets",
        "test_scalar_oracle_rejects_covariance_weight_risk_and_cost_tampering",
        "test_cached_loader_never_invokes_completed_policy_entrypoints",
        "test_all_dependencies_are_checked_before_publication_fence",
        "test_failed_publication_cannot_start_new_capital",
        "test_public_result_preserves_all_paths_without_private_rows",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
    }.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed risk, source, scalar, clock and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"]
        != [
            reference(ROOT / p) for p in (*OWN, "tests/test_expert_initial_monthly_execution_v1.py")
        ]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current allocator and driver bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve any complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            len(events) != 502
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 45
            or counts["new_completed_strategy_evaluations"] != 42
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 45112
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual post-diagnosis journal and invocation.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 45112,
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
            "expert_full_covariance_frozen",
            active_compute_job=None,
            expert_full_covariance_intent=reference(INTENT),
            pending_steps=["register_new_allocator_before_real_weights_or_targets"],
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
        raise QuantError("The actual pre-outcome allocator freeze changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs([*parent["code"], *[reference(ROOT / p) for p in OWN[1:]]]),
        "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
        "source_six_expert_pool_policy": parent["policy"],
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_full_covariance_risk_budget",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 220,
        "prior_journal_sequence": 502,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "all_six_complete_experts_full_NET_RF_covariance_equal_risk_budget",
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


def scalar_covariance(columns):
    centered = [[v - values[0] for v in values] for values in columns]
    means = [math.fsum(v) / len(v) for v in centered]
    return np.array(
        [
            [
                math.fsum(
                    (a - means[i]) * (b - means[j])
                    for a, b in zip(centered[i], centered[j], strict=True)
                )
                / (len(centered[i]) - 1)
                for j in range(6)
            ]
            for i in range(6)
        ]
    )


def independent_weights(covariance):
    matrix = covariance / float(np.trace(covariance))

    def objective(x):
        return 0.5 * float(x @ matrix @ x) - math.fsum(math.log(v) for v in x) / 6

    result = minimize(
        objective,
        np.ones(6),
        jac=lambda x: matrix @ x - 1 / (6 * x),
        method="SLSQP",
        bounds=[(1e-12, None)] * 6,
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    if not result.success or not np.isfinite(result.x).all() or (result.x <= 0).any():
        raise QuantError(f"The independent risk-budget optimizer failed: {result.message}")
    return result.x / math.fsum(result.x)


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Preserve the complete initial/monthly target lattice and calendar.")
    previous = np.full(6, 1 / 6)
    maximum_target = maximum_weight = maximum_covariance = maximum_budget = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["previous_expert_weights"] != previous.tolist()
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or any(o.target_known_at is None or o.target_known_at > day for o in observed)
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError(
                "The archived expert clocks, coordinates or capital separation changed."
            )
        row = audit["allocation"]
        actual = np.asarray(audit["expert_weights"], dtype=float)
        if (
            actual.shape != (6,)
            or not np.isfinite(actual).all()
            or (actual < 0).any()
            or abs(float(actual.sum()) - 1) > 1e-12
            or row["safety_or_success_claimed"] is not False
        ):
            raise QuantError("The scalar allocator needs one finite long-only expert vector.")
        chosen = np.full(6, 1 / 6)
        if not all(o.score_mature for o in observed):
            if (
                row["reason"] != "unscored_initial_equal_weights"
                or row["risk_estimate_available"] is not False
                or row["available_shadow_observations"] != [len(o.net_returns) for o in observed]
            ):
                raise QuantError("Unmatured history cannot acquire a risk-fit success.")
        else:
            index, rf = observed[0].net_returns.index, observed[0].risk_free
            if any(
                not o.net_returns.index.equals(index) or not o.risk_free.equals(rf)
                for o in observed
            ):
                raise QuantError("Independent risk fitting requires identical source/RF dates.")
            excess = [
                [float(v) - float(r) for v, r in zip(o.net_returns, rf, strict=True)]
                for o in observed
            ]
            covariance = scalar_covariance(excess)
            recorded = np.asarray(row["covariance"], dtype=float)
            if (
                recorded.shape != (6, 6)
                or not np.isfinite(recorded).all()
                or row["risk_sample_start"] != str(index[0].date())
                or row["risk_sample_end"] != str(index[-1].date())
            ):
                raise QuantError("The scalar covariance or actual mature clock changed.")
            maximum_covariance = max(maximum_covariance, float(abs(recorded - covariance).max()))
            degenerate = any(all(v == values[0] for v in values) for values in excess)
            if degenerate:
                if (
                    row["reason"] != "unscored_nonpositive_individual_variance_equal_weights"
                    or row["risk_estimate_available"] is not False
                ):
                    raise QuantError("Zero individual risk cannot become a successful allocation.")
            else:
                if (
                    row["reason"] != "full_126_NET_RF_covariance_equal_risk_budget"
                    or row["risk_estimate_available"] is not True
                    or row["observations_used_for_fit"] != 126
                    or row["variance_floor_or_regularization_applied"] is not False
                    or row["return_score_or_rank_used"] is not False
                    or row["solver"] != "positive_cyclic_coordinate_descent"
                    or not 1 <= row["iterations"] <= 10000
                    or row["all_six_expert_weights_positive"] is not True
                    or (actual <= 0).any()
                    or not np.isfinite(
                        [
                            row["scaled_budget_equation_max_residual"],
                            row["scaled_covariance_minimum_eigenvalue"],
                            row["forecast_annualized_RF_excess_volatility"],
                            row["estimated_incremental_switch_cost_fraction"],
                        ]
                    ).all()
                ):
                    raise QuantError(
                        "Verify the actual full positive risk allocation, not pair scores."
                    )
                chosen = independent_weights(covariance)
                x = np.asarray(row["unnormalized_solution"], dtype=float)
                recorded_shares = np.asarray(
                    row["signed_variance_contribution_shares"], dtype=float
                )
                if (
                    x.shape != (6,)
                    or not np.isfinite(x).all()
                    or (x <= 0).any()
                    or recorded_shares.shape != (6,)
                    or not np.isfinite(recorded_shares).all()
                ):
                    raise QuantError(
                        "The scalar risk proof needs complete finite optimizer vectors."
                    )
                matrix = covariance / float(np.trace(covariance))
                budget = actual * (covariance @ actual) / float(actual @ covariance @ actual)
                residual = float(abs(x * (matrix @ x) - 1 / 6).max())
                maximum_budget = max(maximum_budget, float(abs(budget - 1 / 6).max()), residual)
                if (
                    x.shape != (6,)
                    or not np.isfinite(x).all()
                    or (x <= 0).any()
                    or max(abs(x / math.fsum(x) - actual)) > 1e-12
                    or residual > 1e-9
                    or abs(residual - row["scaled_budget_equation_max_residual"]) > 1e-12
                    or max(abs(budget - recorded_shares)) > 1e-10
                    or abs(
                        float(np.linalg.eigvalsh(matrix).min())
                        - row["scaled_covariance_minimum_eigenvalue"]
                    )
                    > 1e-12
                    or abs(
                        math.sqrt(252 * float(actual @ covariance @ actual))
                        - row["forecast_annualized_RF_excess_volatility"]
                    )
                    > 1e-10
                ):
                    raise QuantError("Independent scalar whole-pool risk budgets disagree.")
                for label, bounds in (("older_84", slice(0, 84)), ("recent_42", slice(84, 126))):
                    sample = [v[bounds] for v in excess]
                    cov = scalar_covariance(sample)
                    variance = float(actual @ cov @ actual)
                    sub = row["subperiod_risk_stability"][label]
                    rf_means = np.asarray(sub["RF_excess_shadow_mean_returns"], dtype=float)
                    net_means = np.asarray(sub["NET_shadow_mean_returns"], dtype=float)
                    if (
                        sub["observations"] != len(sample[0])
                        or sub["start"] != str(index[bounds][0].date())
                        or sub["end"] != str(index[bounds][-1].date())
                        or sub["heldout_validation"] is not False
                        or rf_means.shape != (6,)
                        or net_means.shape != (6,)
                        or not np.isfinite(rf_means).all()
                        or not np.isfinite(net_means).all()
                        or not np.isfinite(
                            [
                                sub["combined_RF_excess_mean_return"],
                                sub["combined_annualized_RF_excess_volatility"],
                            ]
                        ).all()
                        or max(abs(rf_means - [math.fsum(v) / len(v) for v in sample])) > 1e-12
                        or max(
                            abs(
                                net_means
                                - [
                                    math.fsum(o.net_returns.iloc[bounds]) / len(sample[0])
                                    for o in observed
                                ]
                            )
                        )
                        > 1e-12
                        or abs(
                            sub["combined_RF_excess_mean_return"]
                            - math.fsum(
                                actual[i] * math.fsum(sample[i]) / len(sample[i]) for i in range(6)
                            )
                        )
                        > 1e-12
                        or abs(
                            sub["combined_annualized_RF_excess_volatility"]
                            - math.sqrt(252 * variance)
                        )
                        > 1e-10
                        or sub["zero_combined_variance"] != (variance == 0)
                    ):
                        raise QuantError(
                            "Full-fit subperiod diagnostics are not held-out validation."
                        )
                    if variance > 0:
                        subshares = np.asarray(
                            sub["signed_variance_contribution_shares"], dtype=float
                        )
                        if (
                            subshares.shape != (6,)
                            or not np.isfinite(subshares).all()
                            or max(abs(subshares - actual * (cov @ actual) / variance)) > 1e-9
                        ):
                            raise QuantError("Subperiod signed marginal risks must also reconcile.")
                    elif sub["signed_variance_contribution_shares"] is not None:
                        raise QuantError("Zero total subperiod risk has undefined risk shares.")
                change = [
                    abs(
                        math.fsum(
                            (actual[i] - previous[i]) * float(observed[i].target[s])
                            for i in range(6)
                        )
                    )
                    for s in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                if (
                    abs(cost - row["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or row["estimated_incremental_changed_assets"] != sum(v > 1e-12 for v in change)
                    or row["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError("The switch proxy must not be another main-account charge.")
        maximum_weight = max(maximum_weight, float(abs(actual - chosen).max()))
        expected = np.array(
            [
                math.fsum(actual[i] * float(observed[i].target[s]) for i in range(6))
                for s in data.close.columns
            ]
        )
        maximum_target = max(
            maximum_target, float(abs(targets.loc[day].to_numpy() - expected).max())
        )
        previous = actual
    if (
        maximum_weight > 1e-6
        or maximum_target > 1e-12
        or maximum_covariance > 1e-12
        or maximum_budget > 1e-8
    ):
        raise QuantError("The independent optimizer/scalar weights, risks or targets disagree.")
    return {
        "all_monthly_decisions_checked": len(audits),
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_covariance_difference": maximum_covariance,
        "maximum_independent_risk_budget_equation_error": maximum_budget,
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
    if len(audits) != 121:
        raise QuantError("Keep all initial/monthly decisions and every formal return day.")
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
            raise QuantError("Future expert history changed an earlier covariance allocation.")
        prefixes.append(str(day.date()))
    new_output_directory(TARGETS)
    write_text_atomic(TARGETS / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = targets.ffill().loc[anchor : window["end"]]
        active = [
            s for s in source.previous.previous.source.donor.rules.CORE if (known[s] > 1e-10).any()
        ]
        if anchor not in known.index or known.isna().any().any() or len(active) < 3:
            raise QuantError("Every formal anchor and target factor coverage remains required.")
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            "unscored_initial_decisions": sum(
                a["allocation"]["reason"] == "unscored_initial_equal_weights" for a in audits
            ),
            "explicit_degenerate_covariance_decisions": sum(
                a["allocation"]["reason"]
                == "unscored_nonpositive_individual_variance_equal_weights"
                for a in audits
            ),
            "solved_covariance_decisions": sum(
                a["allocation"]["risk_estimate_available"] for a in audits
            ),
            "initial_capital_decision_is_not_a_fabricated_month_end": True,
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": "formal main paths, not mature source covariance fits",
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
        raise QuantError("The complete pre-account covariance packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, frozen, _ = inputs()
        old = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] != ID]
        if len(old) != 502 or old[-1]["sha256"] != rules.PRIOR_HEAD:
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
                    "preparing_registered_full_covariance",
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
            source.previous.previous.source.failure(
                "expert_full_covariance_preparation", (ID,), error
            )
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_full_covariance_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_full_covariance_registered_waiting_publication",
            active_compute_job=None,
            expert_full_covariance_registration=reference(PACKET),
            pending_steps=["publish_exact_full_covariance_packet_before_four_accounts"],
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
        raise QuantError("Duplicate completion or terminal events cannot select another winner.")
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
                "evaluating_expert_full_covariance",
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
                            "evaluating_expert_full_covariance",
                            active_candidate=ID,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, ID, window, scenario, benchmarks
                        )
                participation = source.previous.previous.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
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
                        "actual_factor_participation": participation,
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
                "expert_full_covariance_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_full_covariance_results=reference(RESULTS),
                pending_steps=["publish_all_four_full_covariance_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.previous.previous.source.failure(
                "expert_full_covariance_evaluation", (ID,), error
            )
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_full_covariance_evaluation_interrupted", active_compute_job=None
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
        description="One full-covariance expert allocator; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Full-covariance research cannot request live sources.")

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
        parser.exit(2, f"Full-covariance research blocked: {error}\n")


if __name__ == "__main__":
    main()
