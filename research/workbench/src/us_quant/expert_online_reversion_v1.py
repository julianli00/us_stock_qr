from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v4 as pool
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.online_reversion import update_weights
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-online-reversion-v1.json"
PRIOR_HEAD = "476d19a6aa712a2aa5c2717fd518e3c7cc9d9c1c300311104ad8654ec46682e6"
EXPERT_IDS, MECHANISMS = pool.EXPERT_IDS, pool.MECHANISMS
CANDIDATES = (
    {
        "id": "expert_pool_six_olmar_reversion_v1",
        "method": "olmar",
        "window": 5,
        "epsilon": 10.0,
    },
    {
        "id": "expert_pool_six_pamr_reversion_v1",
        "method": "pamr",
        "window": 1,
        "epsilon": 0.5,
    },
)
TOLERANCE = 1e-10


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_online_reversion_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 224,
        "prior_new_registered_candidates": 52,
        "prior_new_completed_strategy_evaluations": 46,
        "prior_new_technical_failures": 6,
        "prior_journal_sequence": 556,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "actual_continuation_event": 53048,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "initial_decision_session": "2016-10-05",
        "shadow_reference_path": "10y-base",
        "formation_price_observations": 6,
        "reference_capital_usd": 10000,
        "learning_clock": "each_complete_NET_source_day_at_its_close",
        "learning_state_resets_for_five_year_account": False,
        "learning_continues_through_last_source_session": True,
        "initial_expert_weights": "equal_six",
        "unscored_main_intention": "equal_six_without_a_success_claim",
        "constant_forecast_rule": "retain_previous_weights_without_new_cross_expert_information",
        "main_decision_clock": "actual_initial_anchor_and_genuine_month_ends",
        "actual_weight_trade_band": 0.025,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "independent_weight_and_target_tolerance": TOLERANCE,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        key not in policy or type(policy[key]) is not type(value) or policy[key] != value
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze the two prior-art reversal laws and the unchanged six experts.")
    methodology = policy.get("methodology")
    if not isinstance(methodology, dict) or any(
        methodology.get(key) is not False
        for key in (
            "old_asset_level_reversion_pipeline_replayed",
            "old_19_asset_universe_or_DBC_admitted",
            "new_standalone_expert_added",
            "pool_selected_from_complete_returns",
            "learning_state_is_main_wealth",
            "shadow_wealth_summed_into_main_capital",
            "source_feedback_floor_transferred",
            "universal_NET_wealth_identity_or_bound_claimed",
            "headroom_stress_V1_repaired_or_retried",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A reversal forecast cannot transfer wealth or repair source invalidity.")


def advance(
    weights: np.ndarray, history: np.ndarray, candidate: dict, observations: int
) -> tuple[np.ndarray, dict]:
    prior, levels = np.asarray(weights, dtype=float), np.asarray(history, dtype=float)
    if (
        candidate not in CANDIDATES
        or type(observations) is not int
        or observations < 2
        or prior.shape != (6,)
        or not np.isfinite(prior).all()
        or (prior < 0).any()
        or abs(float(prior.sum()) - 1) > 1e-12
        or levels.shape != (min(observations, 6), 6)
        or not np.isfinite(levels).all()
        or (levels <= 0).any()
    ):
        raise QuantError("Reversal learning needs finite NET levels and a funded six-expert prior.")
    forecast = None
    result = prior.copy()
    if observations < 6:
        reason = "unscored_formation_warmup"
    else:
        forecast = (
            levels[-5:].mean(axis=0) / levels[-1]
            if candidate["method"] == "olmar"
            else levels[-1] / levels[-2]
        )
        if not np.isfinite(forecast).all() or (forecast <= 0).any():
            raise QuantError("A reversal forecast must contain only known positive finite levels.")
        if float(forecast.max()) == float(forecast.min()):
            reason = "constant_forecast_no_new_relative_information"
        else:
            result = update_weights(prior, forecast, candidate["method"], candidate["epsilon"])
            reason = "observed_NET_expert_reversal_update"
    if not np.isfinite(result).all() or (result < 0).any() or abs(float(result.sum()) - 1) > 1e-12:
        raise QuantError("The statistical reversal update violated its nonnegative unit budget.")
    return result, {
        "algorithm_weights_before": prior.tolist(),
        "forecast": None if forecast is None else forecast.tolist(),
        "algorithm_weights_after": result.tolist(),
        "formation_price_observations": observations,
        "formation_mature": observations >= 6,
        "reason": reason,
        "criterion_or_profit_guaranteed": False,
    }


def decision_sessions(calendar: pd.DatetimeIndex, policy: dict) -> pd.DatetimeIndex:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
    ):
        raise QuantError("Retain the actual initial anchor and every original exchange session.")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


def observations(experts: tuple[StrategyExpert, ...], day: pd.Timestamp, source_day: int):
    observed = [expert.observe(day, 2) for expert in experts]
    expected_count = min(source_day, 2)
    if (
        len(observed) != 6
        or tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != day
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > day
            or len(o.net_returns) != expected_count
            or not o.net_returns.index.equals(o.risk_free.index)
            or not np.isfinite(o.net_returns).all()
            or not np.isfinite(o.risk_free).all()
            or (o.net_returns <= -1).any()
            or o.score_mature != (len(o.net_returns) == 2)
            or expected_count
            and (
                o.net_returns.index[-1] != day
                or not o.net_returns.index.equals(
                    sessions(o.net_returns.index[0], o.net_returns.index[-1])
                )
            )
            for o in observed
        )
        or any(
            not o.net_returns.index.equals(observed[0].net_returns.index)
            or not o.risk_free.equals(observed[0].risk_free)
            for o in observed
        )
    ):
        raise QuantError("Every new learner day needs the same genuinely observed source clock.")
    return observed


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    record_learning=None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Form statistical learner states, not accounts; record each input before its update."""
    dates = decision_sessions(calendar, policy)
    if len(experts) != 6 or len(universe) != len(set(universe)):
        raise QuantError("Keep the fixed complete experts and unique underlying asset coordinates.")
    targets = {c["id"]: pd.DataFrame(np.nan, index=calendar, columns=universe) for c in CANDIDATES}
    weights = {c["id"]: np.full(6, 1 / 6) for c in CANDIDATES}
    previous_main = {name: values.copy() for name, values in weights.items()}
    levels = np.full(6, 10000.0)
    history = levels[None, :].copy()
    audits = []
    latest = {
        c["id"]: {"reason": "unscored_formation_warmup", "forecast": None} for c in CANDIDATES
    }
    updates = 0
    source_calendar = calendar[calendar >= dates[0]]
    for source_day, day in enumerate(source_calendar):
        if day in dates and before_decision is not None:
            before_decision()
        observed = observations(experts, day, source_day)
        if any(tuple(o.target.index) != universe for o in observed):
            raise QuantError("Every complete expert must keep the original admitted asset lattice.")
        if source_day:
            relatives = np.array([1 + float(o.net_returns.iloc[-1]) for o in observed])
            before = levels.copy()
            levels = before * relatives
            if not np.isfinite(levels).all() or (levels <= 0).any():
                raise QuantError("Source NET learning levels must remain finite and positive.")
            history = np.vstack((history, levels))[-6:]
            for candidate in CANDIDATES:
                name = candidate["id"]
                if record_learning is not None:
                    record_learning(
                        {
                            "stage": "input",
                            "session": str(day.date()),
                            "candidate_id": name,
                            "source_price_observations": source_day + 1,
                            "source_NET_relatives": relatives.tolist(),
                            "source_wealth_before": before.tolist(),
                            "source_history_after": history.tolist(),
                            "algorithm_weights_before": weights[name].tolist(),
                        }
                    )
                weights[name], latest[name] = advance(
                    weights[name], history, candidate, source_day + 1
                )
                if record_learning is not None:
                    record_learning(
                        {
                            "stage": "resolved",
                            "session": str(day.date()),
                            "candidate_id": name,
                            **latest[name],
                        }
                    )
            updates += source_day + 1 >= 6
        if day not in dates:
            continue
        for candidate in CANDIDATES:
            name = candidate["id"]
            targets[name].loc[day] = pool.combine(observed, weights[name])
        audits.append(
            {
                "decision_session": str(day.date()),
                "decision_role": "initial_capital"
                if len(audits) == 0
                else "completed_exchange_month_end",
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
                "preserved_archive_identity_introduced_at": [
                    o.identity.introduced_at for o in observed
                ],
                "source_shadow_last_observation": str(day.date()) if source_day else None,
                "source_price_observations": source_day + 1,
                "source_NET_daily_observations": source_day,
                "statistical_daily_updates_per_candidate": updates,
                "source_assessment_mature": source_day + 1 >= 6,
                "previous_applied_expert_weights": {
                    name: value.tolist() for name, value in previous_main.items()
                },
                "applied_expert_weights": {name: value.tolist() for name, value in weights.items()},
                "latest_reversal_update_reason": {
                    name: row["reason"] for name, row in latest.items()
                },
                "learning_state_is_main_wealth": False,
                "source_shadow_wealth_or_protection_floor_transferred": False,
                "universal_NET_wealth_identity_or_bound_claimed": False,
                "counterfactual_historical_expert_availability": True,
                "independent_forward_validation": False,
            }
        )
        previous_main = {name: value.copy() for name, value in weights.items()}
    return targets, audits
