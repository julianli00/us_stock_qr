from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/strategy-expert-pool-four-v1.json"
PRIOR_HEAD = "e1b0200ca142bbf165455d63ff07a75a8802d542404c281bad46a23b94f01aaf"
EXPERT_IDS = (
    "equity_first_month_end_commitment_v1",
    "benchmark_opportunity_joint_tail_v1",
    "symmetric_garch_growth_bil_v1",
    "semiconductor_ratchet_protection",
)
MECHANISMS = (
    "committed_price_trend_defense",
    "joint_tail_opportunity",
    "conditional_variance_growth",
    "wealth_feedback_protection",
)
CANDIDATES = (
    {"id": "expert_pool_four_equal_weight_v1", "weighting": "equal"},
    {"id": "expert_pool_four_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_strategy_expert_pool_four_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 209,
        "prior_new_registered_candidates": 34,
        "prior_new_completed_strategy_evaluations": 31,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 374,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
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
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()):
        raise QuantError("Freeze the four admitted experts and the unchanged original scoring law.")
    if any(
        policy["methodology"].get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A larger expert pool is not qualification or authority to trade.")


def combine(observed: list[ExpertObservation], weights: np.ndarray) -> pd.Series:
    n = len(observed)
    if (
        n < 2
        or weights.shape != (n,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
        or any(o.target is None for o in observed)
    ):
        raise QuantError("Combine complete experts with nonnegative unit-sum capital weights.")
    targets = [o.target for o in observed]
    if any(not t.index.equals(targets[0].index) for t in targets):
        raise QuantError("Expert recommendations must share the same admitted asset coordinates.")
    row = pd.Series(weights @ np.stack([t.to_numpy() for t in targets]), index=targets[0].index)
    if not np.isfinite(row).all() or (row < 0).any() or abs(float(row.sum()) - 0.98) > 1e-12:
        raise QuantError("The unified target must preserve the funded ETF budget and idle cash.")
    return row


def choose_pair(observed: list[ExpertObservation], previous: np.ndarray) -> tuple[np.ndarray, dict]:
    n = len(observed)
    if (
        n < 2
        or any(o.target is None for o in observed)
        or previous.shape != (n,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(float(previous.sum()) - 1) > 1e-12
        or len({o.identity.identifier for o in observed}) != n
        or any(o.as_of != observed[0].as_of for o in observed)
        or any(o.target_known_at is None or o.target_known_at > o.as_of for o in observed)
        or any(o.score_mature != (len(o.net_returns) == 126) for o in observed)
        or any(not o.net_returns.index.equals(o.risk_free.index) for o in observed)
        or any(
            len(o.net_returns)
            and (
                o.net_returns.index[-1] != o.as_of
                or not o.net_returns.index.equals(
                    sessions(o.net_returns.index[0], o.net_returns.index[-1])
                )
            )
            for o in observed
        )
        or any(
            not np.isfinite(o.net_returns).all() or not np.isfinite(o.risk_free).all()
            for o in observed
        )
    ):
        raise QuantError("Unavailable or incomplete expert state is not a successful assessment.")
    equal = np.full(n, 1 / n)
    if any(not o.score_mature for o in observed):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "available_shadow_observations": [len(o.net_returns) for o in observed],
            "pairs": [],
            "selected_pair": None,
            "scores_available": False,
            "safety_or_success_claimed": False,
        }
    dates, rf = observed[0].net_returns.index, observed[0].risk_free
    if (
        len(dates) != 126
        or any(
            not o.net_returns.index.equals(dates) or not o.risk_free.equals(rf) for o in observed
        )
        or any(o.as_of < dates[-1] for o in observed)
    ):
        raise QuantError(
            "Use identical mature dates and the original RF for training and validation."
        )
    excess = np.column_stack([o.net_returns.to_numpy() - rf.to_numpy() for o in observed])
    if not np.isfinite(excess).all():
        raise QuantError("Missing mature observations cannot become successful scores.")
    train, validation = excess[:84], excess[84:]
    covariance = np.cov(train, rowvar=False, ddof=1)
    deviation = np.sqrt(np.diag(covariance))
    current_reference = combine(observed, previous)
    records, usable = [], []
    for pair in itertools.combinations(range(n), 2):
        names = sorted(observed[i].identity.identifier for i in pair)
        row = {"pair": names}
        if observed[pair[0]].identity.mechanism_group == observed[pair[1]].identity.mechanism_group:
            row["unscorable_reason"] = "same_mechanism_group"
            records.append(row)
            continue
        if any(not math.isfinite(deviation[i]) or deviation[i] <= 0 for i in pair):
            row["unscorable_reason"] = "zero_training_variance"
            records.append(row)
            continue
        weights = np.zeros(n)
        inverse = 1 / deviation[list(pair)]
        weights[list(pair)] = inverse / inverse.sum()
        target = combine(observed, weights)
        change = abs(target - current_reference).to_numpy()
        cost_fraction = float(change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000)
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
                "Trained pair weights must preserve equal marginal variance contributions."
            )
        row.update(
            expert_weights=weights.tolist(),
            train_net_risk_score=scores[0],
            validation_net_risk_score=scores[1],
            selection_score=min(scores),
            selection_score_key=round(min(scores), 12),
            training_correlation=float(
                covariance[pair[0], pair[1]] / (deviation[pair[0]] * deviation[pair[1]])
            ),
            training_marginal_variance_contributions=contribution.tolist(),
            estimated_incremental_switch_cost_fraction=cost_fraction,
            estimated_incremental_changed_assets=int((change > 1e-12).sum()),
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


def decision_sessions(calendar: pd.DatetimeIndex, policy: dict) -> pd.DatetimeIndex:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.is_monotonic_increasing
        or not calendar.is_unique
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the real initial anchor and every original source-calendar session.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    dates = decision_sessions(calendar, policy)
    if len(experts) != 4 or len(universe) != len(set(universe)):
        raise QuantError(
            "Use the fixed four complete experts and unique admitted asset coordinates."
        )
    targets = {c["id"]: pd.DataFrame(np.nan, index=calendar, columns=universe) for c in CANDIDATES}
    audits = []
    previous = np.full(4, 0.25)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
            or any(o.target is None for o in observed)
            or any(tuple(o.target.index) != universe for o in observed)
            or position > 0
            and any(not len(o.net_returns) for o in observed)
        ):
            raise QuantError("A missing post-start expert cannot be dropped or silently replaced.")
        equal = np.full(4, 0.25)
        rolling, selection = choose_pair(observed, previous)
        targets[CANDIDATES[0]["id"]].loc[day] = combine(observed, equal)
        targets[CANDIDATES[1]["id"]].loc[day] = combine(observed, rolling)
        audits.append(
            {
                "decision_session": str(day.date()),
                "decision_role": "initial_capital"
                if position == 0
                else "completed_exchange_month_end",
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
                "preserved_archive_identity_introduced_at": [
                    o.identity.introduced_at for o in observed
                ],
                "expert_shadow_last_observation": [
                    str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                    for o in observed
                ],
                "equal_expert_weights": equal.tolist(),
                "previous_rolling_expert_weights": previous.tolist(),
                "rolling_expert_weights": rolling.tolist(),
                "selection": selection,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
        )
        previous = rolling
    return targets, audits
