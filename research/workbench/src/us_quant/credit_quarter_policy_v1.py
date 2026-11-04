from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/credit-quarter-policy-v1.json"
ID = "credit_quarter_style_defense_v1"
CANDIDATE = {
    "id": ID,
    "construction": "monthly_lagged_relative_credit_style_or_gold_bill",
}
CORE = ("MTUM", "VLUE", "QUAL", "USMV")
DEFENSE = ("GLD", "BIL")
UNIVERSE = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"


def validate(policy: dict) -> None:
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_credit_quarter_complete_policy_20261104_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "candidate_id": ID,
        "candidates": [CANDIDATE],
        "mechanism_group": "relative_credit_market_condition",
        "universe": list(UNIVERSE),
        "core_symbols": list(CORE),
        "defensive_symbols": list(DEFENSE),
        "credit_lookback_intervals": 63,
        "credit_observation_lag_sessions": 1,
        "risk_on_condition": "stored_admitted_relative_total_return_strictly_greater_than_zero",
        "risk_on_weighting": "equal_capital_over_four_existing_style_ETFs",
        "risk_off_weighting": "equal_capital_over_GLD_and_actual_BIL",
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "decision_lattice": "genuine_completed_month_ends_after_admitted_input_warmup",
        "decision_information_clock": CLOCK,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "new_economic_factor_definitions": 0,
        "new_traded_assets": 0,
        "parameter_search": False,
    }
    if any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items()):
        raise QuantError("Preserve the single complete credit law, clock, budget and universe.")
    if any(value is not False for value in policy.get("methodology", {}).values()) or set(
        policy.get("methodology", {})
    ) != {
        "real_new_policy_targets_accounts_or_financial_outcomes_computed",
        "source_credit_observations_recomputed",
        "new_expert_technically_admitted",
        "existing_pool_changed",
        "independent_forward_validation",
        "investment_objective_verified",
        "order_authority",
        "automatic_live_deployment",
    }:
        raise QuantError(
            "A declared source policy is not admission, qualification or order authority."
        )


def checked_inputs(close: pd.DataFrame, credit: pd.DataFrame) -> None:
    needed = {
        "source_start_session",
        "source_end_session",
        "relative_total_return",
        "history_mature",
    }
    if (
        not isinstance(close, pd.DataFrame)
        or not isinstance(credit, pd.DataFrame)
        or not isinstance(close.index, pd.DatetimeIndex)
        or close.empty
        or close.index.tz is not None
        or not close.index.is_unique
        or not close.index.is_monotonic_increasing
        or not close.index.equals(sessions(close.index[0], close.index[-1]))
        or tuple(close.columns) != UNIVERSE
        or any(dtype.kind not in "fi" for dtype in close.dtypes)
        or not np.isfinite(close.to_numpy()).all()
        or (close <= 0).any().any()
        or not credit.index.equals(close.index)
        or not needed.issubset(credit.columns)
        or credit["history_mature"].dtype != np.dtype("bool")
        or credit["relative_total_return"].dtype != np.dtype("float64")
    ):
        raise QuantError("Use the whole fourteen-asset market and admitted input session lattice.")
    expected = np.arange(len(close)) >= 64
    values = credit["relative_total_return"].to_numpy()
    if (
        not np.array_equal(credit["history_mature"].to_numpy(), expected)
        or not np.isnan(values[~expected]).all()
        or not np.isfinite(values[expected]).all()
    ):
        raise QuantError("Keep all64 genuine cold-history sessions; never invent a safe signal.")
    for position in np.flatnonzero(expected):
        row = credit.iloc[position]
        if (
            row["source_end_session"] != close.index[position - 1]
            or row["source_start_session"] != close.index[position - 64]
        ):
            raise QuantError(
                "The fixed credit input must retain its prior-session63-interval clock."
            )


def decision(value: float) -> tuple[pd.Series, list[str]]:
    if (
        not isinstance(value, int | float | np.floating)
        or type(value) is bool
        or not np.isfinite(value)
    ):
        raise QuantError("A missing credit input is not a defensive success.")
    selected = list(CORE if value > 0 else DEFENSE)
    row = pd.Series(0.0, index=UNIVERSE)
    row.loc[selected] = 0.98 / len(selected)
    return row, selected


def build_targets(close, credit, policy, before_decision=None, retain=None):
    validate(policy)
    checked_inputs(close, credit)
    targets = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    audits = []
    for day in close.index:
        observation = credit.loc[day]
        if not is_month_end(day) or not observation["history_mature"]:
            continue
        if before_decision is not None:
            before_decision()
        value = float(observation["relative_total_return"])
        before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": CLOCK,
            "credit_source_start": str(observation["source_start_session"].date()),
            "credit_source_end": str(observation["source_end_session"].date()),
            "credit_relative_binary64_hex": value.hex(),
            "credit_history_mature": True,
            "source_credit_observations_recomputed": False,
        }
        if retain is not None:
            retain(before)
        row, selected = decision(value)
        targets.loc[day] = row
        audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": CLOCK,
            "credit_relative_strictly_positive": value > 0,
            "state": "four_style_core" if value > 0 else "gold_and_actual_bill_defense",
            "selected_assets": selected,
            "credit_source_start": before["credit_source_start"],
            "credit_source_end": before["credit_source_end"],
            "counterfactual_historical_policy_availability": True,
            "actual_policy_introduction_or_admission_backdated": False,
            "pure_credit_spread_or_future_protection_claim": False,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "audit": audit,
                    "asset_order": list(close.columns),
                    "complete_target": row.tolist(),
                }
            )
    return targets, audits
