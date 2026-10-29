from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/strategy-expert-meta-v1.json"
PRIOR_HEAD = "3373afc2600c9aaf76cfc47503312a7c6927070335ecf15a6b27014f14d8bb2d"
EXPERT_IDS = (
    "equity_first_month_end_commitment_v1",
    "benchmark_opportunity_joint_tail_v1",
    "symmetric_garch_growth_bil_v1",
)
CANDIDATES = (
    {"id": "expert_pool_equal_weight_v1", "weighting": "equal"},
    {"id": "expert_pool_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_strategy_expert_meta_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 207,
        "prior_journal_sequence": 344,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "selection_score_decimal_places": 12,
        "shadow_reference_path": "10y-base",
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "switch_cost_amortization_sessions": 21,
        "changed_asset_numeric_tolerance": 1e-12,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()):
        raise QuantError(
            "Keep the fixed expert pool, chronological scoring and single-capital rule."
        )
    if any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Expert composition is not forward qualification or trading authority.")


def combine(observations: list[ExpertObservation], weights: np.ndarray) -> pd.Series:
    if (
        len(observations) != 3
        or weights.shape != (3,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
        or any(o.target is None for o in observations)
    ):
        raise QuantError(
            "Combine complete experts with one nonnegative unit-sum capital allocation."
        )
    targets = [o.target for o in observations]
    if any(not t.index.equals(targets[0].index) for t in targets):
        raise QuantError(
            "Combine one common admitted asset universe, not independent account wealth."
        )
    row = pd.Series(weights @ np.stack([t.to_numpy() for t in targets]), index=targets[0].index)
    if not np.isfinite(row).all() or (row < 0).any() or abs(float(row.sum()) - 0.98) > 1e-12:
        raise QuantError("Unified asset targets must preserve the cash-funded invested budget.")
    return row


def choose_pair(
    observations: list[ExpertObservation], previous: np.ndarray
) -> tuple[np.ndarray, dict]:
    equal = np.full(3, 1 / 3)
    if (
        len(observations) != 3
        or any(o.target is None for o in observations)
        or previous.shape != (3,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(float(previous.sum()) - 1) > 1e-12
        or len({o.identity.identifier for o in observations}) != 3
        or any(o.as_of != observations[0].as_of for o in observations)
        or any(o.target_known_at is None or o.target_known_at > o.as_of for o in observations)
        or any(o.score_mature != (len(o.net_returns) == 126) for o in observations)
    ):
        raise QuantError("An unavailable expert is not a score or an implicit safe portfolio.")
    if any(not o.score_mature for o in observations):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "available_shadow_observations": [len(o.net_returns) for o in observations],
            "pairs": [],
            "selected_pair": None,
            "scores_available": False,
            "safety_or_success_claimed": False,
        }
    dates = observations[0].net_returns.index
    rf = observations[0].risk_free
    if (
        len(dates) != 126
        or any(
            not o.net_returns.index.equals(dates) or not o.risk_free.equals(rf)
            for o in observations
        )
        or any(o.as_of < dates[-1] for o in observations)
    ):
        raise QuantError(
            "Training and validation require126identical matured dates and the original RF."
        )
    excess = np.column_stack([o.net_returns.to_numpy() - rf.to_numpy() for o in observations])
    if not np.isfinite(excess).all():
        raise QuantError("Missing mature observations cannot become successful scores.")
    train, validation = excess[:84], excess[84:]
    covariance = np.cov(train, rowvar=False, ddof=1)
    deviation = np.sqrt(np.diag(covariance))
    current_reference = combine(observations, previous)
    records = []
    usable = []
    for pair in itertools.combinations(range(3), 2):
        names = sorted(observations[i].identity.identifier for i in pair)
        row = {"pair": names}
        if (
            observations[pair[0]].identity.mechanism_group
            == observations[pair[1]].identity.mechanism_group
        ):
            row["unscorable_reason"] = "same_mechanism_group"
            records.append(row)
            continue
        if any(not math.isfinite(deviation[i]) or deviation[i] <= 0 for i in pair):
            row["unscorable_reason"] = "zero_training_variance"
            records.append(row)
            continue
        weights = np.zeros(3)
        inverse = 1 / deviation[list(pair)]
        weights[list(pair)] = inverse / inverse.sum()
        target = combine(observations, weights)
        change = abs(target - current_reference).to_numpy()
        switched = change > 1e-12
        cost_fraction = float(change.sum() * 5 / 10000 + switched.sum() / 10000)
        scores = []
        for sample in (train, validation):
            portfolio = sample @ weights
            sigma = float(np.std(portfolio, ddof=1))
            if sigma <= 0 or not math.isfinite(sigma):
                break
            scores.append(float(math.sqrt(252) * (np.mean(portfolio) - cost_fraction / 21) / sigma))
        if len(scores) != 2 or not np.isfinite(scores).all():
            row["unscorable_reason"] = "nonpositive_or_nonfinite_pair_risk"
            records.append(row)
            continue
        contribution = weights * (covariance @ weights)
        variance = float(contribution.sum())
        if variance <= 0 or abs(contribution[pair[0]] - contribution[pair[1]]) > 1e-12 * variance:
            raise QuantError(
                "The pair's trained weights must equalize marginal variance contributions."
            )
        correlation = float(
            covariance[pair[0], pair[1]] / (deviation[pair[0]] * deviation[pair[1]])
        )
        row.update(
            expert_weights=weights.tolist(),
            train_net_risk_score=scores[0],
            validation_net_risk_score=scores[1],
            selection_score=min(scores),
            selection_score_key=round(min(scores), 12),
            training_correlation=correlation,
            training_marginal_variance_contributions=contribution.tolist(),
            estimated_incremental_switch_cost_fraction=cost_fraction,
            estimated_incremental_changed_assets=int(switched.sum()),
            cost_proxy_is_actual_main_account_cost=False,
        )
        records.append(row)
        usable.append(row)
    if not usable:
        return equal, {
            "reason": "unscored_degenerate_pair_equal_weights",
            "pairs": records,
            "selected_pair": None,
            "scores_available": False,
            "safety_or_success_claimed": False,
        }
    chosen = sorted(usable, key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
    return np.asarray(chosen["expert_weights"]), {
        "reason": "rolling_trained_pair_validated_on_mature_past",
        "train_start": str(dates[0].date()),
        "train_end": str(dates[83].date()),
        "validation_start": str(dates[84].date()),
        "validation_end": str(dates[-1].date()),
        "pairs": records,
        "selected_pair": chosen["pair"],
        "scores_available": True,
        "safety_or_success_claimed": False,
    }


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
):
    validate(policy)
    if len(experts) != 3 or not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise QuantError("Use the fixed three-expert pool and ordered original calendar.")
    targets = {c["id"]: pd.DataFrame(np.nan, index=calendar, columns=universe) for c in CANDIDATES}
    audits = []
    previous = np.full(3, 1 / 3)
    available_started = False
    for day in calendar:
        if not is_month_end(day):
            continue
        if before_decision is not None:
            before_decision()
        observations = [expert.observe(day, 126) for expert in experts]
        if tuple(o.identity.identifier for o in observations) != EXPERT_IDS:
            raise QuantError("Expert identity/order cannot change after the pool is fixed.")
        if any(o.target is None for o in observations):
            if available_started:
                raise QuantError(
                    "A post-start unavailable expert cannot silently disappear from the pool."
                )
            continue
        available_started = True
        equal = np.full(3, 1 / 3)
        rolling, selection = choose_pair(observations, previous)
        targets[CANDIDATES[0]["id"]].loc[day] = combine(observations, equal)
        targets[CANDIDATES[1]["id"]].loc[day] = combine(observations, rolling)
        audits.append(
            {
                "decision_session": str(day.date()),
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observations],
                "expert_introduced_at": [o.identity.introduced_at for o in observations],
                "expert_shadow_last_observation": [
                    str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                    for o in observations
                ],
                "equal_expert_weights": equal.tolist(),
                "previous_rolling_expert_weights": previous.tolist(),
                "rolling_expert_weights": rolling.tolist(),
                "selection": selection,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "independent_forward_validation": False,
            }
        )
        previous = rolling
    if not audits:
        raise QuantError("No complete expert recommendation exists before the formal accounts.")
    return targets, audits
