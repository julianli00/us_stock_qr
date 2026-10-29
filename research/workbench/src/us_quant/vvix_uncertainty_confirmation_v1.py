from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.anfci_residual_confirmation_v1 import CORE, UNIVERSE, donor_budget
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import FACTOR_IDS
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/vvix-uncertainty-confirmation-v1.json"
LOOKBACK = 63
PRIOR_HEAD = "2e7bae878fbcd48b785b33524a79d41e3cb52450e6bc2014ac210447b155e806"
CANDIDATES = (
    {
        "id": "two_segment_vvix_uncertainty_confirmation_v1",
        "confirmation": "VVIX_quarter_increase_and_original_dollar_guard",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_vvix_uncertainty_confirmation_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 196,
        "prior_journal_sequence": 216,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_two_segment_variation_v1",
        "vvix_comparison_sessions": LOOKBACK,
        "confirmation": "current_VVIX_native_level_strictly_above_63_session_prior_level",
        "source_monthly_opportunities": 131,
        "missing_required_quote_behavior": "stop_version_without_filling_or_defaulting",
        "decision_information_clock": "after_16_15_ET_index_close_before_later_open_execution",
        "dollar_risk_off_investment_scale": 0.5,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        key not in policy or policy[key] != value or type(policy[key]) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "source_admission_repaired",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed VVIX information rule, clock and financial boundaries.")


def build_targets(
    data: MarketData, desired: pd.DataFrame, vvix: pd.Series, policy: dict
) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    index = data.close.index
    if (
        tuple(data.close.columns) != UNIVERSE
        or not index.equals(sessions(index[0], index[-1]))
        or not desired.index.equals(index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
        or not isinstance(vvix, pd.Series)
        or vvix.name != "VVIX"
        or not vvix.index.equals(index)
        or not pd.api.types.is_numeric_dtype(vvix)
        or not np.isfinite(vvix.to_numpy()).all()
        or (vvix <= 0).any()
    ):
        raise QuantError("Keep complete native VVIX quotes and the exact saved donor/calendar.")
    active = desired.dropna(how="all")
    if active.empty or not all(is_month_end(day) for day in active.index):
        raise QuantError("Use only the original completed month-end opportunities.")
    target = pd.DataFrame(np.nan, index=index, columns=desired.columns)
    audits = []
    for day, old in active.iterrows():
        position = index.get_loc(day)
        if not isinstance(position, int) or position < LOOKBACK:
            raise QuantError("A quarter comparison needs its actual63-session prior quote.")
        prior = index[position - LOOKBACK]
        current_level, prior_level = float(vvix.loc[day]), float(vvix.loc[prior])
        budget, guarded = donor_budget(old)
        increased = current_level > prior_level
        confirmed = bool(guarded and increased)
        row = old.copy()
        if guarded and not confirmed:
            non_bil = row.index.drop("BIL")
            row.loc[non_bil] *= 0.98 / budget
            row["BIL"] = 0.0
        if (
            not np.isfinite(row).all()
            or (row < 0).any()
            or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("VVIX confirmation changed funding or created invalid weights.")
        target.loc[day] = row
        audits.append(
            {
                "decision_session": str(day.date()),
                "VVIX_reference_session": str(prior.date()),
                "VVIX_current_native_level": current_level,
                "VVIX_prior_native_level": prior_level,
                "VVIX_level_increased": bool(increased),
                "original_dollar_guard_active": guarded,
                "uncertainty_guard_confirmed": confirmed,
                "candidate_target_created": True,
                "information_clock": "after_16_15_ET_index_close",
                "VVIX_index_return_or_variance_premium_claimed": False,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(old: pd.Series, current_native: Decimal, prior_native: Decimal) -> pd.Series:
    if (
        not current_native.is_finite()
        or not prior_native.is_finite()
        or min(current_native, prior_native) <= 0
    ):
        raise QuantError("The independent oracle requires actual positive native quotes.")
    guarded = abs(float(old["BIL"]) - 0.49) <= 1e-12
    result = old.copy()
    if guarded and current_native <= prior_native:
        for name in result.index:
            result[name] = 0.0 if name == "BIL" else float(old[name]) * 2
    return result
