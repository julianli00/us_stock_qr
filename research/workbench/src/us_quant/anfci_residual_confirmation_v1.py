from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import daily_variance_disagreement_v1 as donor
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/anfci-residual-confirmation-v1.json"
CORE, STOCKS, UNIVERSE = donor.CORE, donor.STOCKS, donor.UNIVERSE
PRIOR_HEAD = "9109b83e7425b310895446f2b9fdeb344003805cd25cd15d1efe7b049196db71"
CANDIDATES = (
    {
        "id": "two_segment_anfci_residual_confirmation_v1",
        "confirmation": "fresh_anfci_positive_and_original_dollar_guard",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_anfci_residual_confirmation_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 194,
        "prior_journal_sequence": 190,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_two_segment_variation_v1",
        "anfci_threshold": 0.0,
        "maximum_observation_age_calendar_days": 14,
        "vintage_cutoff": "previous_NYSE_session",
        "required_anfci_information": "current_own_vintage_level_only",
        "require_fresh_anfci_before_every_monthly_decision": True,
        "stale_input_behavior": "explicit_no_new_request_keep_actual_positions",
        "dollar_risk_off_investment_scale": 0.5,
        "source_monthly_opportunities": 131,
        "expected_fresh_decisions": 130,
        "expected_source_abstentions": 1,
        "source_full_225_query_admission_passed": False,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "core_symbols": list(CORE),
        "factor_ids": list(donor.FACTOR_IDS),
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
        raise QuantError("Keep the fixed residual-stress and explicit abstention contract.")


def donor_budget(row: pd.Series) -> tuple[float, bool]:
    if (
        tuple(row.index) != UNIVERSE
        or not np.isfinite(row).all()
        or (row < 0).any()
        or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
        or (row.loc[list(donor.UNUSED)].abs() > 1e-12).any()
    ):
        raise QuantError("Require the complete unchanged donor allocation.")
    budget = float(row.loc[[*STOCKS, "GLD"]].sum())
    guarded = bool(np.isclose(row["BIL"], 0.49, rtol=0, atol=1e-12))
    if not np.isclose(row["BIL"], 0.49 if guarded else 0.0, rtol=0, atol=1e-12) or not np.isclose(
        budget, 0.49 if guarded else 0.98, rtol=0, atol=1e-12
    ):
        raise QuantError("The original dollar-only budget must remain identifiable.")
    equity = float(row.loc[list(STOCKS)].sum())
    if (
        equity <= 0
        or not 0.30 - 1e-12 <= equity / budget <= 0.70 + 1e-12
        or not np.isclose(row["SOXX"], equity / 2, rtol=0, atol=1e-12)
        or not np.allclose(row.loc[list(CORE)], equity / 8, rtol=0, atol=1e-12)
    ):
        raise QuantError("Donor stock/gold or relative stock composition changed.")
    return budget, guarded


def align_readings(records: list[dict], decisions: pd.DatetimeIndex) -> pd.DataFrame:
    rows = {}
    for record in records:
        day = pd.Timestamp(record["query_session"])
        cutoff = pd.Timestamp(record["vintage_cutoff"])
        observed = pd.Timestamp(record["observation_date"])
        value = record["actual_latest_value"]
        age = (day - observed).days
        if (
            day in rows
            or pd.isna(day)
            or pd.isna(cutoff)
            or pd.isna(observed)
            or cutoff != previous_session(day)
            or observed > cutoff
            or observed.weekday() != 4
            or type(record["age_calendar_days"]) is not int
            or record["age_calendar_days"] != age
            or age < 0
            or type(record["fresh"]) is not bool
            or record["fresh"] != (age <= 14)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
        ):
            raise QuantError("Source query identity, vintage, age or recorded eligibility changed.")
        rows[day] = {
            "vintage_cutoff": cutoff,
            "observation_date": observed,
            "age_calendar_days": age,
            "fresh": record["fresh"],
            "accepted_value": float(value) if record["fresh"] else np.nan,
        }
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    if (
        decisions.empty
        or decisions.has_duplicates
        or not decisions.is_monotonic_increasing
        or not decisions.isin(frame.index).all()
    ):
        raise QuantError("A required source query cannot be omitted or synthesized.")
    return frame.loc[decisions]


def build_targets(
    data: MarketData, desired: pd.DataFrame, records: list[dict], policy: dict
) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
    ):
        raise QuantError("Keep the complete market and original sparse donor target calendar.")
    active = desired.dropna(how="all")
    if active.empty or not all(is_month_end(day) for day in active.index):
        raise QuantError("Only original completed monthly opportunities are allowed.")
    readings = align_readings(records, active.index)
    target = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    audits = []
    for day, old in active.iterrows():
        budget, guarded = donor_budget(old)
        reading = readings.loc[day]
        fresh = bool(reading["fresh"])
        confirmed = bool(guarded and reading["accepted_value"] > 0) if fresh else None
        if fresh:
            multiplier = 1.0 if confirmed else 0.98 / budget
            row = old.copy()
            non_bil = row.index.drop("BIL")
            row.loc[non_bil] *= multiplier
            row["BIL"] = 0.49 if confirmed else 0.0
            if (
                not np.isfinite(row).all()
                or (row < 0).any()
                or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
            ):
                raise QuantError(
                    "Residual confirmation changed funding or introduced invalid targets."
                )
            target.loc[day] = row
        audits.append(
            {
                "decision_session": str(day.date()),
                "vintage_cutoff": str(reading["vintage_cutoff"].date()),
                "observation_date": str(reading["observation_date"].date()),
                "age_calendar_days": int(reading["age_calendar_days"]),
                "fresh_required_reading": fresh,
                "original_dollar_guard_active": guarded,
                "accepted_ANFCI_value": float(reading["accepted_value"]) if fresh else None,
                "residual_stress_confirmed": confirmed,
                "candidate_target_created": fresh,
                "source_abstention_reason": None if fresh else "anfci_exceeds_frozen_14day_age",
                "stale_numeric_reading_used": False,
            }
        )
    return {CANDIDATES[0]["id"]: target}, audits


def independent_row(row: pd.Series, record: dict, day: pd.Timestamp) -> pd.Series:
    if (day - pd.Timestamp(record["observation_date"])).days > 14:
        return pd.Series(np.nan, index=row.index)
    half = abs(float(row["BIL"]) - 0.49) <= 1e-12
    tight = float(record["actual_latest_value"]) > 0
    result = row.copy()
    if half and not tight:
        for name in result.index:
            result[name] = 0.0 if name == "BIL" else float(row[name]) * 2
    return result
