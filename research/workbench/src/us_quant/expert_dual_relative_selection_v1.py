from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-dual-relative-selection-v1.json"
PRIOR_HEAD = "31dd509b070c9ac219da939f123d40c153df30b24c269331691c21e75fa934d6"
EXPERT_IDS, MECHANISMS = source.EXPERT_IDS, source.MECHANISMS
CANDIDATES = (
    {"id": "expert_pool_six_dual_relative_pair_v1", "weighting": "train_erc_validate_dual_active"},
)
BENCHMARKS = ("SPY", "QQQ")
combine = source.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_dual_relative_selection_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 219,
        "prior_new_registered_candidates": 44,
        "prior_new_completed_strategy_evaluations": 41,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 490,
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
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()
    ):
        raise QuantError(
            "Freeze the single active-score change and all original risk/execution parameters."
        )
    if (
        set(policy.get("scoring_benchmarks", {})) != set(BENCHMARKS)
        or not isinstance(policy.get("methodology"), dict)
        or any(
            policy["methodology"].get(k) is not False
            for k in (
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError(
            "Use both original tradable references without qualification or trading claims."
        )


def choose_pair(
    observed: list[ExpertObservation],
    previous: np.ndarray,
    benchmark_returns: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    n = len(observed)
    if (
        n != 6
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
        raise QuantError(
            "Unavailable or incomplete expert state is not a successful active assessment."
        )
    equal = np.full(n, 1 / n)
    if any(not o.score_mature for o in observed):
        return equal, {
            "reason": "unscored_initial_equal_weights",
            "available_shadow_observations": [len(o.net_returns) for o in observed],
            "pairs": [],
            "selected_pair": None,
            "scores_available": False,
            "scoring_benchmark_last_observation": None,
            "safety_or_success_claimed": False,
        }
    dates, rf = observed[0].net_returns.index, observed[0].risk_free
    if (
        any(not o.net_returns.index.equals(dates) or not o.risk_free.equals(rf) for o in observed)
        or not isinstance(benchmark_returns, pd.DataFrame)
        or not isinstance(benchmark_returns.index, pd.DatetimeIndex)
        or not benchmark_returns.index.is_unique
        or not benchmark_returns.columns.is_unique
        or not set(BENCHMARKS).issubset(benchmark_returns.columns)
        or not dates.isin(benchmark_returns.index).all()
    ):
        raise QuantError("Use exact matured source and original SPY/QQQ dates without padding.")
    benchmarks = benchmark_returns.loc[dates, list(BENCHMARKS)]
    if not benchmarks.index.equals(dates) or not np.isfinite(benchmarks).all().all():
        raise QuantError(
            "Missing or nonfinite benchmark observations cannot become neutral scores."
        )
    raw = np.column_stack([o.net_returns.to_numpy() for o in observed])
    excess = raw - rf.to_numpy()[:, None]
    covariance = np.cov(excess[:84], rowvar=False, ddof=1)
    deviations = np.sqrt(np.diag(covariance))
    current_reference = combine(observed, previous)
    records, usable = [], []
    for pair in itertools.combinations(range(n), 2):
        names = sorted(observed[i].identity.identifier for i in pair)
        row = {"pair": names}
        if observed[pair[0]].identity.mechanism_group == observed[pair[1]].identity.mechanism_group:
            row["unscorable_reason"] = "same_mechanism_group"
            records.append(row)
            continue
        if any(not math.isfinite(deviations[i]) or deviations[i] <= 0 for i in pair):
            row["unscorable_reason"] = "zero_training_variance"
            records.append(row)
            continue
        weights = np.zeros(n)
        inverse = 1 / deviations[list(pair)]
        weights[list(pair)] = inverse / inverse.sum()
        target = combine(observed, weights)
        change = abs(target - current_reference).to_numpy()
        cost = float(change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000)
        descriptions, scores = {}, []
        for symbol in BENCHMARKS:
            # Subtract first so identical references retain exactly zero active risk.
            active = (raw - benchmarks[symbol].to_numpy()[:, None]) @ weights
            values = {}
            for label, sample in (("train", active[:84]), ("validation", active[84:])):
                if np.all(sample == sample[0]):
                    break
                tracking = float(np.std(sample, ddof=1))
                if tracking <= 0 or not math.isfinite(tracking):
                    break
                score = float(math.sqrt(252) * (np.mean(sample) - cost / 21) / tracking)
                if not math.isfinite(score):
                    break
                scores.append(score)
                values[label + "_net_active_risk_score"] = score
                values[label + "_tracking_risk"] = tracking
            descriptions[symbol] = values
        if len(scores) != 4:
            row["unscorable_reason"] = "nonpositive_or_nonfinite_active_tracking_risk"
            records.append(row)
            continue
        contribution = weights * (covariance @ weights)
        variance = float(contribution.sum())
        if variance <= 0 or abs(contribution[pair[0]] - contribution[pair[1]]) > 1e-12 * variance:
            raise QuantError("Keep the original RF-excess trained pair ERC risk weights.")
        row.update(
            expert_weights=weights.tolist(),
            benchmark_scores=descriptions,
            selection_score=min(scores),
            selection_score_key=round(min(scores), 12),
            training_correlation=float(
                covariance[pair[0], pair[1]] / (deviations[pair[0]] * deviations[pair[1]])
            ),
            training_marginal_variance_contributions=contribution.tolist(),
            training_risk_basis="original_RF_excess_returns",
            estimated_incremental_switch_cost_fraction=cost,
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
            "scoring_benchmark_last_observation": str(dates[-1].date()),
            "safety_or_success_claimed": False,
        }
    chosen = sorted(usable, key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
    return np.asarray(chosen["expert_weights"]), {
        "reason": "mature_dual_benchmark_worst_train_validation_active_score",
        "train_start": str(dates[0].date()),
        "train_end": str(dates[83].date()),
        "validation_start": str(dates[84].date()),
        "validation_end": str(dates[-1].date()),
        "pairs": records,
        "selected_pair": chosen["pair"],
        "scores_available": True,
        "scoring_benchmark_last_observation": str(dates[-1].date()),
        "safety_or_success_claimed": False,
    }


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Keep the genuine initial anchor and every original calendar session.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    benchmarks: pd.DataFrame,
    before_decision=None,
) -> tuple[pd.DataFrame, list[dict]]:
    dates = decision_sessions(calendar, policy)
    if len(experts) != 6 or len(universe) != len(set(universe)):
        raise QuantError(
            "Use the unchanged fixed six complete experts and unique asset coordinates."
        )
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits, previous = [], np.full(6, 1 / 6)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
            or any(o.target is None or tuple(o.target.index) != universe for o in observed)
            or position > 0
            and any(not len(o.net_returns) for o in observed)
        ):
            raise QuantError("Missing source experts cannot be silently dropped or substituted.")
        weights, selection = choose_pair(observed, previous, benchmarks)
        targets.loc[day] = combine(observed, weights)
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
                "previous_expert_weights": previous.tolist(),
                "expert_weights": weights.tolist(),
                "selection": selection,
                "counterfactual_historical_expert_availability": True,
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
                "independent_forward_validation": False,
            }
        )
        previous = weights
    return targets, audits
