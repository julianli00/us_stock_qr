from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import five_hierarchical_risk_v1 as covariance_source
from us_quant import quintet_anytime_hedge_v1 as prior
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quintet-hedge-risk-units-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = prior.EXPERT_IDS, prior.MECHANISMS, prior.CLOCK
ID = "expert_pool_quintet_hedge_risk_units_v1"
CANDIDATES = ({"id": ID, "weighting": "cached_Hedge_beliefs_in_source_volatility_units"},)
PRIOR_HEAD = "8526850396a8054097558ac674e567d22dc395c77a31c99c1659b23dd563d817"
N, LOOKBACK, TOLERANCE = 5, 126, 1e-12
combine, sample_covariance = prior.combine, covariance_source.sample_covariance


def expected_policy():
    return {
        "schema_version": 1,
        "study_id": "ssd_quintet_hedge_risk_units_20261106_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 187844,
        "prior_journal_sequence": 1428,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 298,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-hedge-risk-units-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "64d03a17733271ac7c6beba9f2f27075c7bd2e243be3b2aa218ab78f71c1d073",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "risk_history_sessions": LOOKBACK,
        "sample_covariance_ddof": 1,
        "all_risk_observations_used": True,
        "covariance_integrity_tolerance": TOLERANCE,
        "independent_numeric_tolerance": TOLERANCE,
        "capital_weight_rule": "cached_belief_divided_by_current_native_NET_RF_std_then_unit_sum",
        "unmatured_or_nonpositive_individual_risk": "explicit_unscored_passthrough_cached_beliefs",
        "cached_learning_state_reexecuted": False,
        "learning_rate_or_history_changed": False,
        "covariance_regularization_or_variance_floor": False,
        "full_covariance_ERC_or_global_solver_invoked": False,
        "main_drawdown_guarantee": False,
        "normalized_source_regret_transfers_to_sized_capital": False,
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": TOLERANCE,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "decision_information_clock": CLOCK,
        "admitted_legacy_defensive_views_reused_unchanged": True,
        "all_source_native_RF_claimed_equal_without_view": False,
        "mechanism_labels_are_independent_risks": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_source_RF_views": 0,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "study_size_is_a_total_research_quota": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError(
            "Freeze cached beliefs and one original-window volatility-unit conversion."
        )


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the full admitted calendar and original initial decision.")
    return pd.DatetimeIndex(
        [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    )


def checked_observations(observed, day):
    formal = (
        sessions("2016-10-06", day) if day >= pd.Timestamp("2016-10-06") else pd.DatetimeIndex([])
    )
    expected = formal[-LOOKBACK:]
    if (
        tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != day
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > day
            for o in observed
        )
        or any(
            not o.net_returns.index.equals(expected) or not o.risk_free.index.equals(expected)
            for o in observed
        )
        or any(o.score_mature != (len(expected) == LOOKBACK) for o in observed)
        or any(
            not np.isfinite(o.net_returns).all()
            or (o.net_returns <= -1).any()
            or not np.isfinite(o.risk_free).all()
            for o in observed
        )
        or any(not o.risk_free.equals(observed[0].risk_free) for o in observed)
    ):
        raise QuantError(
            "Missing or invalid complete source state cannot become an unscored risk fit."
        )
    return observed


def observations(experts, day):
    return checked_observations([expert.observe(day, LOOKBACK) for expert in experts], day)


def capital_weights(observed, beliefs):
    probabilities = np.asarray(beliefs)
    if (
        len(observed) != N
        or probabilities.shape != (N,)
        or not np.isfinite(probabilities).all()
        or (probabilities <= 0).any()
        or abs(math.fsum(float(v) for v in probabilities) - 1) > TOLERANCE
    ):
        raise QuantError("Five exact cached positive belief fractions are required.")
    checked_observations(observed, observed[0].as_of)
    if any(not o.score_mature for o in observed):
        return probabilities.copy(), {
            "risk_sizing_available": False,
            "reason": "unscored_incomplete_126_passthrough_beliefs",
            "risk_observations": [len(o.net_returns) for o in observed],
            "covariance": None,
            "safety_or_success_claimed": False,
        }
    rf = observed[0].risk_free
    matrix = np.column_stack([o.net_returns.to_numpy() - rf.to_numpy() for o in observed])
    covariance = sample_covariance(matrix)
    if (
        covariance.shape != (N, N)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=0, atol=TOLERANCE)
    ):
        raise QuantError("Use the complete unchanged five-coordinate sample covariance.")
    diagonal = np.diag(covariance)
    trace = float(np.trace(covariance))
    if (diagonal < 0).any() or not math.isfinite(trace):
        raise QuantError("Negative or nonfinite variance cannot become valid risk immaturity.")
    if trace == 0:
        if (covariance != 0).any():
            raise QuantError("A zero-variance covariance matrix must be genuinely zero.")
    elif float(np.linalg.eigvalsh(covariance / trace).min()) < -TOLERANCE:
        raise QuantError("Invalid covariance cannot be regularized into successful risk sizing.")
    if (diagonal == 0).any():
        return probabilities.copy(), {
            "risk_sizing_available": False,
            "reason": "unscored_nonpositive_individual_variance_passthrough_beliefs",
            "risk_observations": LOOKBACK,
            "covariance": covariance.tolist(),
            "safety_or_success_claimed": False,
        }
    deviation = np.sqrt(diagonal)
    mass = probabilities / deviation
    total = math.fsum(float(value) for value in mass)
    if not np.isfinite(mass).all() or not math.isfinite(total) or total <= 0:
        raise QuantError("Risk-unit normalization must remain finite without a variance floor.")
    weights = np.array([float(value) / total for value in mass])
    contributions = weights * (covariance @ weights)
    raw_contributions = probabilities * (covariance @ probabilities)
    return weights, {
        "risk_sizing_available": True,
        "reason": "cached_beliefs_in_realized_source_volatility_units",
        "risk_observations": LOOKBACK,
        "risk_start": str(observed[0].net_returns.index[0].date()),
        "risk_end": str(observed[0].net_returns.index[-1].date()),
        "covariance": covariance.tolist(),
        "source_daily_NET_RF_std": deviation.tolist(),
        "raw_belief_marginal_variance_contributions": raw_contributions.tolist(),
        "capital_marginal_variance_contributions": contributions.tolist(),
        "all126_observations_fit_risk_not_heldout_validation": True,
        "full_covariance_ERC_or_variance_target_claimed": False,
        "main_drawdown_guarantee": False,
        "safety_or_success_claimed": False,
    }


def build_targets(
    experts, cached_decisions, calendar, assets, policy, before_decision=None, retain=None
):
    dates = decision_sessions(calendar, policy)
    if len(experts) != N or len(set(assets)) != len(assets):
        raise QuantError("Use exactly five complete cached-source policies and unique coordinates.")
    by_date = {row["decision_session"]: row for row in cached_decisions}
    if len(by_date) != len(cached_decisions):
        raise QuantError("The sealed belief stream must have unique original decisions.")
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    audits, previous = [], np.full(N, 0.2)
    for day in dates:
        if before_decision is not None:
            before_decision()
        key = str(day.date())
        if key not in by_date:
            raise QuantError("A missing original belief cannot be filled or recomputed.")
        observed = observations(experts, day)
        if any(tuple(o.target.index) != tuple(assets) for o in observed):
            raise QuantError("Preserve every original underlying recommendation coordinate.")
        beliefs = np.asarray(by_date[key]["expert_weights"])
        if retain is not None:
            retain(
                {
                    "phase": "before_decision",
                    "decision_session": key,
                    "cached_beliefs_hex": [float(v).hex() for v in beliefs],
                    "known_target_dates": [str(o.target_known_at.date()) for o in observed],
                    "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
                    "source_NET_returns_hex": [
                        [float(v).hex() for v in o.net_returns] for o in observed
                    ],
                    "source_RF_hex": [float(v).hex() for v in observed[0].risk_free],
                    "previous_capital_weights_hex": [float(v).hex() for v in previous],
                    "candidate_accounts_or_metrics_computed": False,
                }
            )
        weights, risk = capital_weights(observed, beliefs)
        row = combine(observed, weights)
        target.loc[day] = row
        matrix = np.stack([o.target.to_numpy() for o in observed])
        change = (weights - previous) @ matrix
        cost = float(5 * abs(change).sum() + (abs(change) > TOLERANCE).sum())
        audit = {
            "decision_session": key,
            "cached_beliefs": beliefs.tolist(),
            "capital_weights": weights.tolist(),
            "previous_capital_weights": previous.tolist(),
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "risk_sizing": risk,
            "incremental_switch_proxy_USD": cost,
            "proxy_is_diagnostic_not_actual_main_cost": True,
            "cached_learner_state_reexecuted": False,
            "normalized_source_regret_transfers_to_sized_capital": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "phase": "after_decision",
                    "decision_session": key,
                    "audit": audit,
                    "unified_target_hex": [float(v).hex() for v in row],
                }
            )
        previous = weights
    return target, audits
