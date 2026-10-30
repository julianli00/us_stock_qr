from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import observed_weight_execution as monthly
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE as ORIGINAL_UNIVERSE
from us_quant.data import MarketData
from us_quant.expert_master_protection_v1 import inputs as nominal_inputs
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-headroom-stress-v1.json"
PRIOR_HEAD = "e56b8138eaf88267def7882a36a796425ed539c98bfbdd3829b07c548f39ec24"
UNIVERSE = (*ORIGINAL_UNIVERSE, "IJR", "PKW")
CANDIDATES = (
    {
        "id": "expert_pool_six_online_headroom_stress_v1",
        "source_candidate_id": "expert_pool_six_online_log_growth_v1",
    },
    {
        "id": "expert_pool_six_path_baskets_headroom_stress_v1",
        "source_candidate_id": "expert_pool_six_path_admissible_baskets_v1",
    },
)
NOMINAL_HASHES = (
    "fc60f51ee70f34cae9e3874677f897f3d52415444c66cc6889e7edb8d21fb4e1",
    "d6f60a12c677f85ee0bcc33bf8ff58e057715f5f90cab9d5b62d32db17c510e6",
)


@dataclass(frozen=True)
class MainCloseState:
    as_of: pd.Timestamp
    nav: float
    peak: float
    current_weights: pd.Series

    def validate(self):
        if (
            pd.Timestamp(self.as_of).tzinfo is not None
            or not math.isfinite(self.nav)
            or not math.isfinite(self.peak)
            or self.nav <= 0
            or self.peak < self.nav
            or self.peak < 10000
            or tuple(self.current_weights.index) != UNIVERSE
            or not np.isfinite(self.current_weights).all()
            or (self.current_weights < 0).any()
            or float(self.current_weights.sum()) > 1 + 1e-10
        ):
            raise QuantError(
                "Use actual known positive main NAV, retained peak and funded holdings."
            )


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_headroom_stress_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 224,
        "prior_new_registered_candidates": 50,
        "prior_new_completed_strategy_evaluations": 46,
        "prior_new_technical_failures": 4,
        "prior_journal_sequence": 552,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "main_initial_capital_usd": 10000,
        "retained_peak_floor_fraction": 0.85,
        "original_drawdown_limit": 0.15,
        "stress_lookback_return_sessions": 126,
        "stress_price_observations": 127,
        "stress_horizon": (
            "next_session_through_first_future_month_end_plus_original_delay_inclusive"
        ),
        "stress_scenarios": "all_overlapping_horizon_adjusted_close_ratios",
        "controller_decision_clock": "actual_initial_anchor_and_completed_exchange_month_ends_only",
        "main_peak_updated_each_actual_close": True,
        "current_close_cost_allowance": (
            "NAV_times_current_gross_plus_0.98_times_bps_plus14fixed_tickets"
        ),
        "allowance_fixed_asset_tickets": 14,
        "commission_per_order": 1.0,
        "maximum_fraction": 1.0,
        "minimum_fraction": 0.0,
        "infeasible_mode": "BIL_funding_standby_with_contradiction_witness_not_success",
        "short_initial_history_mode": "unscored_BIL_funding_standby",
        "actual_weight_trade_band": 0.025,
        "cash_reserve": 0.02,
        "invested_target_budget": 0.98,
        "independent_fractional_tolerance": 1e-10,
        "independent_money_state_tolerance_usd": 1e-6,
        "target_budget_tolerance": 1e-12,
        "final_issued_targets_precomputed_before_accounts": False,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
        "minimum_economic_factor_families": 3,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    expected_nominals = {
        CANDIDATES[0]["source_candidate_id"]: {
            "path": "data/ssd-expert-online-log-growth-20261030-v1/targets.csv",
            "sha256": NOMINAL_HASHES[0],
        },
        CANDIDATES[1]["source_candidate_id"]: {
            "path": "data/ssd-expert-path-baskets-20261030-v1/targets.csv",
            "sha256": NOMINAL_HASHES[1],
        },
    }
    if (
        not isinstance(policy, dict)
        or any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items())
        or policy.get("nominal_sources") != expected_nominals
        or not isinstance(policy.get("methodology"), dict)
        or any(
            policy["methodology"].get(k) is not False
            for k in (
                "source_roles_selected_as_archive_return_winners",
                "reference_stress_loss_is_next_return_forecast",
                "main_drawdown_guarantee",
                "source_shadow_wealth_or_peak_imported",
                "cost_allowance_deducted_as_actual_fee",
                "cost_allowance_is_future_open_cost_cap",
                "future_calendar_is_future_price_information",
                "old_fixed_daily_ratchet_repaired_or_retuned",
                "frequency_or_fee_difference_is_isolated_causal_experiment",
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError(
            "Freeze the two nominal roles, main stress transition and all original gates."
        )


def review_horizon(day, delay):
    day = pd.Timestamp(day)
    if type(delay) is not int or delay not in (1, 2):
        raise QuantError("Use an original one/two-session execution delay.")
    future = sessions(day + pd.Timedelta(days=1), day + pd.Timedelta(days=45))
    month_ends = [d for d in future if is_month_end(d)]
    if not month_ends:
        raise QuantError("The admitted exchange calendar needs the next scheduled review.")
    review = execution = month_ends[0]
    for _ in range(delay):
        execution = next_session(execution)
    horizon = len(sessions(next_session(day), execution))
    if not 1 <= horizon < 126:
        raise QuantError("The known review horizon must fit inside the fixed stress sample.")
    return review, execution, horizon


def affine_interval(intercepts, slopes, budget):
    if (
        intercepts.ndim != 1
        or intercepts.shape != slopes.shape
        or not len(intercepts)
        or not np.isfinite(intercepts).all()
        or not np.isfinite(slopes).all()
        or not math.isfinite(budget)
    ):
        raise QuantError("A stress interval needs actual finite affine loss constraints.")
    lower, upper = 0.0, 1.0
    lower_index = upper_index = zero_index = None
    for i, (a, b) in enumerate(zip(intercepts, slopes, strict=True)):
        if b == 0:
            if a > budget and zero_index is None:
                zero_index = i
        else:
            bound = float((budget - a) / b)
            if not math.isfinite(bound):
                raise QuantError("An affine constraint has a nonfinite fractional bound.")
            if b > 0 and bound < upper:
                upper, upper_index = bound, i
            elif b < 0 and bound > lower:
                lower, lower_index = bound, i
    feasible = zero_index is None and lower <= upper
    beta = float(upper) if feasible else 0.0
    residual = float((intercepts + slopes * beta - budget).max())
    if feasible and (not 0 <= beta <= 1 or residual > 1e-10):
        raise QuantError("The maximum stress fraction failed its finite interval certificate.")
    return beta, {
        "feasible": feasible,
        "lower_fraction": lower,
        "upper_fraction": upper,
        "binding_lower_scenario": lower_index,
        "binding_upper_scenario": upper_index,
        "zero_slope_violation_scenario": zero_index,
        "maximum_scenario_residual_at_applied_fraction": residual,
        "reason": "maximum_observed_stress_feasible_fraction"
        if feasible
        else "observed_stress_interval_infeasible_funding_standby",
    }


def _validate_close_inputs(nominal, state, history, cost, delay):
    state.validate()
    if (
        (cost, delay) not in ((5.0, 1), (20.0, 2))
        or tuple(nominal.index) != UNIVERSE
        or not np.isfinite(nominal).all()
        or (nominal < 0).any()
        or abs(float(nominal.sum()) - 0.98) > 1e-12
        or not isinstance(history, pd.DataFrame)
        or history.empty
        or tuple(history.columns) != UNIVERSE
        or not history.index.equals(sessions(history.index[0], history.index[-1]))
        or history.index[-1] != state.as_of
        or not np.isfinite(history.tail(127)).all().all()
        or (history.tail(127) <= 0).any().any()
    ):
        raise QuantError(
            "Use known-close funded intentions, exact asset prices and original costs."
        )


def _scaled_target(nominal, beta):
    desired = nominal.copy()
    risk = desired.index != "BIL"
    desired.loc[risk] = nominal.loc[risk] * beta
    desired["BIL"] = float(nominal["BIL"]) + float((nominal.loc[risk] - desired.loc[risk]).sum())
    if (
        not np.isfinite(desired).all()
        or (desired < 0).any()
        or abs(float(desired.sum()) - 0.98) > 1e-12
    ):
        raise QuantError("The stress target must preserve the long-only funded ETF budget.")
    return desired


def resolve(nominal, state: MainCloseState, history, cost, delay):
    _validate_close_inputs(nominal, state, history, cost, delay)
    review, execution, horizon = review_horizon(state.as_of, delay)
    headroom = float(state.nav - 0.85 * state.peak)
    allowance = float(state.nav * (float(state.current_weights.sum()) + 0.98) * cost / 10000 + 14.0)
    budget = float((headroom - allowance) / state.nav)
    common = {
        "known_main_NAV_USD": state.nav,
        "known_main_peak_USD": state.peak,
        "signed_main_headroom_USD": headroom,
        "current_close_cost_allowance_USD": allowance,
        "available_stress_loss_fraction": budget,
        "next_review_session": str(review.date()),
        "next_review_execution_session": str(execution.date()),
        "known_calendar_horizon_sessions": horizon,
        "reference_is_actual_future_loss_or_drawdown_guarantee": False,
        "allowance_is_deducted_fee_or_guaranteed_future_open_cap": False,
    }
    if len(history) < 127:
        return _scaled_target(nominal, 0), {
            **common,
            "stress_sample_available": False,
            "applied_fraction": 0.0,
            "feasible": None,
            "reason": "unscored_short_price_history_funding_standby",
            "scenarios": [],
            "certificate": None,
        }
    sample = history.tail(127)
    ratios = sample.iloc[horizon:].to_numpy() / sample.iloc[:-horizon].to_numpy()
    bil = UNIVERSE.index("BIL")
    intercepts = 1 - (0.02 + 0.98 * ratios[:, bil])
    slopes = ((ratios[:, bil, None] - ratios) * nominal.to_numpy()).sum(axis=1)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise QuantError("Historical adjusted stress ratios must remain finite and positive.")
    beta, certificate = affine_interval(intercepts, slopes, budget)
    scenarios = [
        {
            "first_price_session": str(sample.index[i].date()),
            "last_price_session": str(sample.index[i + horizon].date()),
            "constant_BIL_reference_loss": float(intercepts[i]),
            "nonBIL_fraction_loss_slope": float(slopes[i]),
        }
        for i in range(len(ratios))
    ]
    return _scaled_target(nominal, beta), {
        **common,
        "stress_sample_available": True,
        "applied_fraction": beta,
        "feasible": certificate["feasible"],
        "reason": certificate["reason"],
        "scenarios": scenarios,
        "certificate": certificate,
    }


def independent_resolve(nominal, state: MainCloseState, history, cost, delay):
    _validate_close_inputs(nominal, state, history, cost, delay)
    day = pd.Timestamp(state.as_of)
    future = sessions(day + pd.Timedelta(days=1), day + pd.Timedelta(days=45))
    review = next(d for d in future if is_month_end(d))
    execution = review
    for _ in range(delay):
        execution = next_session(execution)
    horizon = len(sessions(next_session(day), execution))
    allowance = state.nav * (math.fsum(float(v) for v in state.current_weights) + 0.98)
    allowance = allowance * cost / 10000 + 14.0
    budget = (state.nav - 0.85 * state.peak - allowance) / state.nav
    rows = []
    lower, upper = 0.0, 1.0
    lower_index = upper_index = zero_index = None
    if len(history) >= 127:
        sample = history.tail(127)
        prices = sample.to_numpy()
        for i in range(127 - horizon):
            ratio = [
                float(prices[i + horizon, j]) / float(prices[i, j]) for j in range(len(UNIVERSE))
            ]
            bil = ratio[UNIVERSE.index("BIL")]
            a = 1 - (0.02 + 0.98 * bil)
            b = math.fsum(float(nominal.iloc[j]) * (bil - ratio[j]) for j in range(len(UNIVERSE)))
            if b == 0:
                if a > budget and zero_index is None:
                    zero_index = i
            elif b > 0:
                bound = (budget - a) / b
                if bound < upper:
                    upper, upper_index = bound, i
            else:
                bound = (budget - a) / b
                if bound > lower:
                    lower, lower_index = bound, i
            rows.append((a, b))
        feasible = zero_index is None and lower <= upper
        beta = upper if feasible else 0.0
    else:
        feasible, beta = None, 0.0
    result, removed = nominal.copy(), 0.0
    for symbol in nominal.index:
        if symbol != "BIL":
            result[symbol] = float(nominal[symbol]) * beta
            removed += float(nominal[symbol]) - float(result[symbol])
    result["BIL"] = float(nominal["BIL"]) + removed
    return result, {
        "applied_fraction": float(beta),
        "feasible": feasible,
        "current_close_cost_allowance_USD": float(allowance),
        "available_stress_loss_fraction": float(budget),
        "known_calendar_horizon_sessions": horizon,
        "next_review_session": str(review.date()),
        "next_review_execution_session": str(execution.date()),
        "scenarios": rows,
        "lower_fraction": float(lower),
        "upper_fraction": float(upper),
        "binding_lower_scenario": lower_index,
        "binding_upper_scenario": upper_index,
        "zero_slope_violation_scenario": zero_index,
    }


def verify_resolution(nominal, state, history, cost, delay, desired, audit):
    if (
        not isinstance(desired, pd.Series)
        or not desired.index.equals(nominal.index)
        or not np.isfinite(desired).all()
        or (desired < 0).any()
        or abs(float(desired.sum()) - 0.98) > 1e-12
    ):
        raise QuantError(
            "The recorded reference target must preserve the exact funded asset lattice."
        )
    expected, scalar = independent_resolve(nominal, state, history, cost, delay)
    target_error = float(abs(desired - expected).max())
    fraction_error = abs(float(audit["applied_fraction"]) - scalar["applied_fraction"])
    money_error = abs(
        float(audit["current_close_cost_allowance_USD"])
        - scalar["current_close_cost_allowance_USD"]
    )
    budget_error = abs(
        float(audit["available_stress_loss_fraction"]) - scalar["available_stress_loss_fraction"]
    )
    state_errors = [
        abs(float(audit["known_main_NAV_USD"]) - state.nav),
        abs(float(audit["known_main_peak_USD"]) - state.peak),
        abs(float(audit["signed_main_headroom_USD"]) - (state.nav - 0.85 * state.peak)),
    ]
    if (
        not np.isfinite(desired).all()
        or not np.isfinite(
            [target_error, fraction_error, money_error, budget_error, *state_errors]
        ).all()
        or target_error > 1e-10
        or fraction_error > 1e-10
        or money_error > 1e-6
        or max(state_errors) > 1e-6
        or budget_error > 1e-10
        or audit["feasible"] is not scalar["feasible"]
        or audit["stress_sample_available"] is not (len(history) >= 127)
        or audit["known_calendar_horizon_sessions"] != scalar["known_calendar_horizon_sessions"]
        or audit["next_review_session"] != scalar["next_review_session"]
        or audit["next_review_execution_session"] != scalar["next_review_execution_session"]
        or len(audit["scenarios"]) != len(scalar["scenarios"])
        or audit["reference_is_actual_future_loss_or_drawdown_guarantee"] is not False
        or audit["allowance_is_deducted_fee_or_guaranteed_future_open_cap"] is not False
    ):
        raise QuantError(
            "The independent main stress interval, cost allowance or target disagrees."
        )
    maximum_scenario_error = 0.0
    sample = history.tail(127)
    for i, (row, (a, b)) in enumerate(zip(audit["scenarios"], scalar["scenarios"], strict=True)):
        errors = [
            abs(float(row["constant_BIL_reference_loss"]) - a),
            abs(float(row["nonBIL_fraction_loss_slope"]) - b),
        ]
        if (
            not np.isfinite(errors).all()
            or max(errors) > 1e-10
            or row["first_price_session"] != str(sample.index[i].date())
            or row["last_price_session"]
            != str(sample.index[i + scalar["known_calendar_horizon_sessions"]].date())
        ):
            raise QuantError(
                "A recorded stress scenario disagrees with independent adjusted ratios."
            )
        maximum_scenario_error = max(maximum_scenario_error, *errors)
    if len(history) >= 127:
        certificate = audit["certificate"]
        reason = (
            "maximum_observed_stress_feasible_fraction"
            if scalar["feasible"]
            else ("observed_stress_interval_infeasible_funding_standby")
        )
        if (
            certificate is None
            or certificate["feasible"] is not scalar["feasible"]
            or any(
                not math.isfinite(float(certificate[k]))
                or abs(float(certificate[k]) - scalar[k]) > 1e-10
                for k in ("lower_fraction", "upper_fraction")
            )
            or audit["reason"] != reason
            or certificate["reason"] != reason
        ):
            raise QuantError("The maximum-fraction or contradiction witness changed.")
        for field, bound, sign, inactive in (
            ("binding_lower_scenario", "lower_fraction", -1, 0.0),
            ("binding_upper_scenario", "upper_fraction", 1, 1.0),
        ):
            index = certificate[field]
            if index is None:
                if abs(certificate[bound] - inactive) > 1e-10:
                    raise QuantError(
                        "A nontrivial interval bound needs an observed scenario witness."
                    )
                continue
            if type(index) is not int or not 0 <= index < len(audit["scenarios"]):
                raise QuantError("The interval witness is outside the actual scenario lattice.")
            row = audit["scenarios"][index]
            a, b = row["constant_BIL_reference_loss"], row["nonBIL_fraction_loss_slope"]
            if (
                sign * b <= 0
                or abs((audit["available_stress_loss_fraction"] - a) / b - certificate[bound])
                > 1e-10
            ):
                raise QuantError(
                    "The recorded binding scenario does not certify its interval bound."
                )
        zero = certificate["zero_slope_violation_scenario"]
        if zero is not None:
            if type(zero) is not int or not 0 <= zero < len(audit["scenarios"]):
                raise QuantError("A zero-slope contradiction needs an actual scenario.")
            row = audit["scenarios"][zero]
            if (
                row["nonBIL_fraction_loss_slope"] != 0
                or row["constant_BIL_reference_loss"] <= audit["available_stress_loss_fraction"]
            ):
                raise QuantError("The zero-slope scenario does not prove a contradiction.")
        if (
            audit["feasible"] is False
            and zero is None
            and certificate["lower_fraction"] <= certificate["upper_fraction"]
        ):
            raise QuantError("An infeasible interval cannot hide a missing contradiction witness.")
        expected_residual = max(
            a + b * scalar["applied_fraction"] - scalar["available_stress_loss_fraction"]
            for a, b in scalar["scenarios"]
        )
        if (
            not math.isfinite(float(certificate["maximum_scenario_residual_at_applied_fraction"]))
            or abs(certificate["maximum_scenario_residual_at_applied_fraction"] - expected_residual)
            > 1e-10
        ):
            raise QuantError(
                "The reference interval residual cannot become a success-shaped default."
            )
    elif (
        audit["certificate"] is not None
        or audit["reason"] != "unscored_short_price_history_funding_standby"
    ):
        raise QuantError("Unavailable initial stress history must remain explicitly unscored.")
    return {
        "maximum_target_difference": target_error,
        "maximum_fraction_difference": fraction_error,
        "maximum_money_allowance_difference_USD": money_error,
        "maximum_scenario_coefficient_difference": maximum_scenario_error,
    }


def run(data: MarketData, baseline, candidate, policy, start, end, cost, delay):
    validate(policy)
    if candidate not in CANDIDATES or (cost, delay) not in ((5.0, 1), (20.0, 2)):
        raise QuantError("Use one frozen source role and an original cost/delay scenario.")
    dates, active = nominal_inputs(data, baseline, start, end)
    execution_policy = read_json(monthly.POLICY)
    monthly.validate(execution_policy)
    band, original_policy = monthly.pure_band(execution_policy)
    anchor = previous_session(dates[0])
    issued = pd.DataFrame(np.nan, index=baseline.index, columns=baseline.columns)
    pending, decisions, rows, weights = {}, [], [], []
    units = np.zeros(len(baseline.columns))
    cash = previous_nav = peak = 10000.0
    stress_mature_started = False

    def decide(day, nav, current, initial=False):
        nonlocal stress_mature_started
        known = active.loc[:day]
        source_day, nominal = known.index[-1], known.iloc[-1]
        state = MainCloseState(day, nav, peak, current)
        desired, stress = resolve(nominal, state, data.close.loc[:day], cost, delay)
        if stress_mature_started and not stress["stress_sample_available"]:
            raise QuantError("Post-maturity stress prices cannot silently become unscored.")
        stress_mature_started = stress_mature_started or stress["stress_sample_available"]
        covariance = monthly.prior.observed_covariance(
            data, day, original_policy["covariance_sessions"]
        )
        reason = (
            "initial_capital"
            if initial
            else "pending_request"
            if pending
            else monthly.prior.rebalance_reason(current, desired, covariance, band)
        )
        requested = reason not in ("hold", "pending_request")
        execution = None
        if requested:
            if pending:
                raise QuantError("A monthly stress transition cannot overwrite a pending request.")
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending[execution] = desired.to_numpy().copy()
            issued.loc[day] = desired
        decisions.append(
            {
                "signal_session": str(day.date()),
                "source_nominal_session": str(source_day.date()),
                "initial_capital": initial,
                "known_main_NAV_USD": nav,
                "known_main_peak_USD": peak,
                "actual_current_gross_exposure": float(current.sum()),
                "maximum_observed_target_gap": float(abs(current - desired).max()),
                "desired_reference_target": desired.tolist(),
                "reason": reason,
                "request_issued": requested,
                "execution_session": str(execution.date()) if execution is not None else None,
                "pending_executions": [str(d.date()) for d in pending],
                "stress": stress,
            }
        )

    decide(anchor, 10000.0, pd.Series(0.0, index=baseline.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, costs, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, 1.0
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("The actual main stress-controlled account borrowed cash or failed.")
        current = pd.Series(values / nav, index=baseline.columns)
        rows.append(
            (nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders)
        )
        weights.append(current.to_numpy())
        peak = max(peak, nav)
        if is_month_end(day):
            decide(day, nav, current)
        previous_nav = nav
    account = pd.DataFrame(
        rows,
        index=dates,
        columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"],
    )
    account["risk_free"] = data.risk_free.loc[dates]
    return (
        BacktestResult(account, pd.DataFrame(weights, index=dates, columns=baseline.columns)),
        issued,
        decisions,
    )


def verify_feedback(data, baseline, result, issued, decisions, start, end, cost, delay):
    dates, active = nominal_inputs(data, baseline, start, end)
    anchor = previous_session(dates[0])
    opportunities = pd.DatetimeIndex([anchor, *[d for d in dates if is_month_end(d)]])
    if (
        not result.frame.index.equals(dates)
        or not result.weights.index.equals(dates)
        or not result.weights.columns.equals(baseline.columns)
        or not issued.index.equals(baseline.index)
        or not issued.columns.equals(baseline.columns)
        or [r["signal_session"] for r in decisions] != [str(d.date()) for d in opportunities]
        or not result.frame["risk_free"].equals(data.risk_free.loc[dates])
        or not np.isfinite(result.frame).all().all()
        or not np.isfinite(result.weights).all().all()
        or (result.weights < 0).any().any()
        or (result.frame["cash"] < 0).any()
        or (result.frame["gross_exposure"] > 1 + 1e-10).any()
        or abs(result.weights.sum(axis=1) - result.frame["gross_exposure"]).max() > 1e-10
        or abs(
            result.frame["cash"] / result.frame["equity"] + result.frame["gross_exposure"] - 1
        ).max()
        > 1e-10
        or abs(
            float(result.frame.iloc[0]["equity"]) / (1 + float(result.frame.iloc[0]["return"]))
            - 10000
        )
        > 1e-8
    ):
        raise QuantError(
            "Keep fresh main state, original opportunities and every formal account day."
        )
    band, original_policy = monthly.pure_band(read_json(monthly.POLICY))
    pending, expected_issued = {}, []
    maximum_target = maximum_fraction = maximum_money = maximum_scenario = 0.0
    for position, row in enumerate(decisions):
        day = pd.Timestamp(row["signal_session"])
        initial = position == 0
        pending = {d: value for d, value in pending.items() if d > day}
        known = active.loc[:day]
        source_day, nominal = known.index[-1], known.iloc[-1]
        nav = 10000.0 if initial else float(result.frame.loc[day, "equity"])
        peak = 10000.0 if initial else max(10000.0, float(result.frame.loc[:day, "equity"].max()))
        current = pd.Series(0.0, index=baseline.columns) if initial else result.weights.loc[day]
        state = MainCloseState(day, nav, peak, current)
        desired, scalar = independent_resolve(nominal, state, data.close.loc[:day], cost, delay)
        recorded_desired = pd.Series(row["desired_reference_target"], index=baseline.columns)
        proof = verify_resolution(
            nominal, state, data.close.loc[:day], cost, delay, recorded_desired, row["stress"]
        )
        covariance = monthly.prior.observed_covariance(
            data, day, original_policy["covariance_sessions"]
        )
        reason = (
            "initial_capital"
            if initial
            else "pending_request"
            if pending
            else monthly.prior.rebalance_reason(current, desired, covariance, band)
        )
        requested = reason not in ("hold", "pending_request")
        execution = None
        if requested:
            if pending:
                raise QuantError(
                    "The independent monthly state found an overwritten pending request."
                )
            execution = day
            for _ in range(delay):
                execution = next_session(execution)
            pending[execution] = True
            expected_issued.append(day)
            error = float(abs(issued.loc[day] - desired).max())
            if issued.loc[day].isna().any() or error > 1e-10:
                raise QuantError(
                    "The actual issued target differs from the scalar stress transition."
                )
            maximum_target = max(maximum_target, error)
        elif not issued.loc[day].isna().all():
            raise QuantError("A held monthly opportunity cannot acquire an invented request.")
        fields = {
            "known_main_NAV_USD": nav,
            "known_main_peak_USD": peak,
            "actual_current_gross_exposure": float(current.sum()),
            "maximum_observed_target_gap": float(abs(current - desired).max()),
        }
        errors = [abs(float(row[k]) - value) for k, value in fields.items()]
        if (
            not np.isfinite(errors).all()
            or max(errors) > 1e-6
            or row["source_nominal_session"] != str(source_day.date())
            or row["initial_capital"] is not initial
            or row["request_issued"] is not requested
            or row["reason"] != reason
            or row["execution_session"]
            != (str(execution.date()) if execution is not None else None)
            or row["pending_executions"] != [str(d.date()) for d in pending]
            or abs(row["stress"]["applied_fraction"] - scalar["applied_fraction"]) > 1e-10
        ):
            raise QuantError("The actual peak, source clock, band or pending stress state changed.")
        maximum_fraction = max(maximum_fraction, proof["maximum_fraction_difference"])
        maximum_money = max(maximum_money, proof["maximum_money_allowance_difference_USD"], *errors)
        maximum_scenario = max(maximum_scenario, proof["maximum_scenario_coefficient_difference"])
    if not issued.dropna(how="all").index.equals(pd.DatetimeIndex(expected_issued)):
        raise QuantError(
            "Every actual monthly issued request needs its unique original signal date."
        )
    return {
        "checked_decision_states": len(decisions),
        "checked_issued_requests": len(expected_issued),
        "independent_scalar_stress_and_actual_continuous_main_peak_checked": True,
        "single_pending_request_original_band_and_session_delays_checked": True,
        "maximum_issued_target_difference": maximum_target,
        "maximum_fraction_difference": maximum_fraction,
        "maximum_money_state_difference_USD": maximum_money,
        "maximum_scenario_coefficient_difference": maximum_scenario,
        "initial_capital_and_peak_are_fresh_main_10000": True,
        "source_shadow_wealth_or_peak_imported": False,
        "main_drawdown_guarantee": False,
    }
