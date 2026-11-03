from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as pool
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_net_wealth_mixture_v1 import period_relative
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-minimax-opportunity-v1.json"
PRIOR_HEAD = "2d8640e7ae7a166b96083fa241951fe60aa73f1d74214992c445b71a35e5b5d4"
EXPERT_IDS, MECHANISMS, CLOCK = pool.EXPERT_IDS, pool.MECHANISMS, pool.CLOCK
CANDIDATES = (
    {
        "id": "expert_pool_eight_minimax_monthly_opportunity_v1",
        "weighting": "equal_pair_worst_mature_month_log_opportunity_plus_switch_proxy",
    },
)
combine = pool.combine


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_minimax_opportunity_20261103_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_journal_sequence": 722,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_registered": 68,
        "prior_completed": 60,
        "prior_technical_failures": 8,
        "prior_funded_paths": 240,
        "prior_combined_completed": 238,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "decision_information_clock": CLOCK,
        "initial_decision_session": "2016-10-05",
        "source_history_delivery_lookback": 2512,
        "first_complete_month_anchor": "2016-10-31",
        "minimum_completed_months": 1,
        "maximum_completed_months": 119,
        "learning_memory": "all_expanding_completed_full_exchange_month_intervals",
        "initial_partial_source_month_assimilated": False,
        "learning_resets_for_five_year_account": False,
        "pair_menu_size": 28,
        "pair_fractions": [0.5, 0.5],
        "selection_score_decimal_places": 12,
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": 1e-12,
        "switch_penalty": "minus_log1p_of_negative_one_time_proxy_fraction",
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Freeze one eight-source opportunity law and original funding rules.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(key) is not False
        for key in (
            "score_is_actual_main_net_wealth",
            "source_feedback_floor_or_wealth_guarantee_transferred",
            "opportunity_bound_or_independent_samples_claimed",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("An assessment opportunity loss is not a funded-return guarantee.")


def choose_pair(
    monthly: np.ndarray, opinions: np.ndarray, previous: np.ndarray
) -> tuple[np.ndarray, dict]:
    if (
        monthly.ndim != 2
        or monthly.shape[1] != 8
        or not 0 <= monthly.shape[0] <= 119
        or not np.isfinite(monthly).all()
        or (monthly <= 0).any()
        or opinions.ndim != 2
        or opinions.shape[0] != 8
        or opinions.shape[1] == 0
        or not np.isfinite(opinions).all()
        or (opinions < 0).any()
        or any(abs(math.fsum(row) - 0.98) > 1e-12 for row in opinions)
        or previous.shape != (8,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(math.fsum(previous) - 1) > 1e-12
    ):
        raise QuantError("Require complete positive source months and valid funded opinions.")
    if not len(monthly):
        return np.full(8, 0.125), {
            "score_mature": False,
            "reason": "no_complete_source_month",
            "pair_records": [],
            "selected_pair": None,
        }
    reference = monthly.max(axis=1)
    records = []
    for first, second in itertools.combinations(range(8), 2):
        weights = np.zeros(8)
        weights[[first, second]] = 0.5
        relatives = [
            math.fsum((0.5 * float(row[first]), 0.5 * float(row[second]))) for row in monthly
        ]
        if not all(math.isfinite(value) and value > 0 for value in relatives):
            raise QuantError("Every statistical pair month needs a finite positive relative.")
        losses = [
            math.log(float(best) / relative)
            for best, relative in zip(reference, relatives, strict=True)
        ]
        change = (weights - previous) @ opinions
        changed = int((abs(change) > 1e-12).sum())
        cost = float(5 * abs(change).sum() + changed)
        fraction = cost / 10000
        if (
            not all(math.isfinite(value) and value >= 0 for value in losses)
            or not math.isfinite(fraction)
            or not 0 <= fraction < 1
        ):
            raise QuantError("Invalid opportunity loss or switching cost cannot become safe.")
        penalty = -math.log1p(-fraction)
        worst = max(losses)
        score = worst + penalty
        if not math.isfinite(score):
            raise QuantError("A nonfinite monthly opportunity score cannot be ranked.")
        records.append(
            {
                "pair": sorted((EXPERT_IDS[first], EXPERT_IDS[second])),
                "expert_weights": weights.tolist(),
                "monthly_opportunity_losses": losses,
                "worst_month_indices": [i for i, loss in enumerate(losses) if loss == worst],
                "worst_opportunity_loss": worst,
                "switch_cost_proxy_usd": cost,
                "switch_changed_asset_coordinates": changed,
                "one_time_switch_log_penalty": penalty,
                "raw_score": score,
                "score_key_12dp": round(score, 12),
            }
        )
    selected = min(records, key=lambda row: (row["score_key_12dp"], row["pair"]))
    return np.asarray(selected["expert_weights"]), {
        "score_mature": True,
        "completed_intervals": len(monthly),
        "pair_records": records,
        "selected_pair": selected["pair"],
        "selected_score_key_12dp": selected["score_key_12dp"],
        "score_is_actual_main_net_wealth_or_risk_bound": False,
    }


def _input_record(position, day, observed, previous, monthly, intervals, anchor):
    rows = []
    for item in observed:
        net = (
            item.net_returns.loc[item.net_returns.index > anchor]
            if anchor is not None
            else item.net_returns.iloc[:0]
        )
        rf = (
            item.risk_free.loc[item.risk_free.index > anchor]
            if anchor is not None
            else item.risk_free.iloc[:0]
        )
        rows.append(
            {
                "id": item.identity.identifier,
                "mechanism_group": item.identity.mechanism_group,
                "introduced_at": item.identity.introduced_at,
                "as_of": str(item.as_of.date()),
                "target_known_at": str(item.target_known_at.date())
                if item.target_known_at is not None
                else None,
                "asset_order": list(item.target.index) if item.target is not None else None,
                "target_binary64_hex": [float(x).hex() for x in item.target]
                if item.target is not None
                else None,
                "full_prefix_return_count": len(item.net_returns),
                "source_delivery_score_mature": item.score_mature,
                "new_NET_dates": net.index.strftime("%Y-%m-%d").tolist(),
                "new_NET_binary64_hex": [float(x).hex() for x in net],
                "new_RF_dates": rf.index.strftime("%Y-%m-%d").tolist(),
                "new_RF_binary64_hex": [float(x).hex() for x in rf],
            }
        )
    return {
        "record": "decision_input",
        "decision_index": position,
        "as_of": str(day.date()),
        "decision_information_clock": CLOCK,
        "previous_expert_weights_binary64_hex": [float(x).hex() for x in previous],
        "previous_matured_months_binary64_hex": [[float(x).hex() for x in row] for row in monthly],
        "previous_completed_intervals": intervals.copy(),
        "new_interval_left_anchor": str(anchor.date()) if anchor is not None else None,
        "observations": rows,
        "initial_partial_month_assimilated": False,
    }


def _check_observations(observed: list[ExpertObservation], day, expected, universe) -> None:
    if (
        tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != day
            or o.target is None
            or tuple(o.target.index) != universe
            or o.target_known_at is None
            or o.target_known_at > day
            or not o.net_returns.index.equals(expected)
            or not o.risk_free.index.equals(expected)
            or not np.isfinite(o.net_returns).all()
            or (o.net_returns <= -1).any()
            or not np.isfinite(o.risk_free).all()
            or not o.risk_free.equals(observed[0].risk_free)
            for o in observed
        )
    ):
        raise QuantError("Preserve every source identity, opinion, native NET and exact common RF.")


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    retain=None,
) -> tuple[pd.DataFrame, list[dict]]:
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
        or len(experts) != 8
        or not universe
        or len(universe) != len(set(universe))
    ):
        raise QuantError("Keep the full genuine source calendar and all eight admitted experts.")
    decisions = [initial, *[day for day in calendar if day > initial and is_month_end(day)]]
    target = pd.DataFrame(np.nan, index=calendar, columns=universe)
    previous, monthly, intervals, audits = np.full(8, 0.125), [], [], []
    anchor = None
    for position, day in enumerate(decisions):
        if before_decision is not None:
            before_decision()
        observed = [expert.observe(day, 2512) for expert in experts]
        if retain is not None:
            retain(_input_record(position, day, observed, previous, monthly, intervals, anchor))
        expected = calendar[(calendar > initial) & (calendar <= day)]
        _check_observations(observed, day, expected, universe)
        if is_month_end(day):
            if anchor is not None:
                relative = period_relative(observed, anchor, day)
                monthly.append(relative)
                intervals.append([str(anchor.date()), str(day.date())])
            anchor = day
        history = np.asarray(monthly) if monthly else np.empty((0, 8))
        weights, selection = choose_pair(
            history, np.stack([o.target.to_numpy() for o in observed]), previous
        )
        combined = combine(observed, weights)
        target.loc[day] = combined
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if position == 0 else "completed_exchange_month_end",
            "decision_information_clock": CLOCK,
            "completed_intervals": [row.copy() for row in intervals],
            "source_month_relatives_binary64_hex": [
                [float(x).hex() for x in row] for row in monthly
            ],
            "previous_expert_weights": previous.tolist(),
            "expert_weights": weights.tolist(),
            "selection": selection,
            "counterfactual_historical_expert_availability": True,
            "source_intramonth_feedback_or_shadow_wealth_guarantee_transferred": False,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "target_binary64_hex": [float(x).hex() for x in combined],
                }
            )
        previous = weights
    return target, audits
