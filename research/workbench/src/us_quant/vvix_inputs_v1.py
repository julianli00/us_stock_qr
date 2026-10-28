from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.near_term_volatility_inputs import parse_history as parse_existing

SERIES = ("VVIX", "VIX")
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv"
REFERENCES = (
    "source_index",
    "new_raw_csv",
    "existing_vix_csv",
    "existing_option_manifest",
    "history_acquisition",
    "methodology",
    "methodology_acquisition",
    "advertised_resources",
)


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_vvix_source_readiness_20261028_v1",
        "scope": "archival_daily_closes_and_original_monthly_query_coverage_only",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "required_market_context_sessions": 2805,
        "original_monthly_query_count": 131,
        "new_source_series": ["VVIX"],
        "existing_comparator_series": ["VIX"],
        "daily_input_column_order": list(SERIES),
        "new_source_url": URL,
        "native_new_source_csv_columns": ["DATE", "VVIX"],
        "native_value_column": "VVIX",
        "native_units": "annualized_expected_volatility_percentage_points",
        "maximum_source_bytes": 5_000_000,
        "missing_data_rule": (
            "retain_all_calendar_rows_and_missing_cells_report_incomplete_no_fill_or_safe_state"
        ),
        "full_daily_coverage_required_for_full_admission": True,
        "old_near_term_VVIX_admitted_false_record_unchanged": True,
        "new_registered_factor_definitions": 0,
        "new_registered_candidates": 0,
        "new_strategy_evaluations": 0,
    }
    if (
        any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in expected.items()
        )
        or any(
            policy.get(key) is not False
            for key in (
                "strategy_states_targets_or_returns_computed",
                "existing_W42_or_forward_rules_changed",
                "intraday_first_publication_or_full_history_methodology_verified",
                "investment_objective_verified",
                "order_authority",
            )
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError(
            "Keep the distinct VVIX source, original coverage and source-only boundary."
        )


def parse_history(path: Path, name: str, last_allowed_date: pd.Timestamp) -> pd.Series:
    if name == "VIX":
        return parse_existing(path, name, last_allowed_date)
    if (
        name != "VVIX"
        or not isinstance(last_allowed_date, pd.Timestamp)
        or last_allowed_date.tz is not None
        or last_allowed_date != last_allowed_date.normalize()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 5_000_000
    ):
        raise QuantError("Require the bounded VVIX file and its actual acquisition-date limit.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["DATE", "VVIX"]:
        raise QuantError("Use the actual DATE/VVIX schema, not a substituted OHLC or return field.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="raise"))
        values = pd.to_numeric(frame["VVIX"], errors="raise").to_numpy()
    except (ValueError, TypeError) as error:
        raise QuantError("VVIX dates and annualized native quotes must be explicit.") from error
    if (
        index.empty
        or not index.is_unique
        or not index.is_monotonic_increasing
        or (index.weekday >= 5).any()
        or index[-1] > last_allowed_date
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise QuantError("VVIX quotes must be positive, finite, chronological and not future data.")
    return pd.Series(values, index=index, name=name)


def align_closes(index: pd.DatetimeIndex, histories: dict[str, pd.Series]) -> pd.DataFrame:
    if not index.equals(sessions("2015-08-10", "2026-10-05")) or set(histories) != set(SERIES):
        raise QuantError("Keep every original market date and exactly the VVIX/VIX identities.")
    aligned = pd.DataFrame(index=index)
    for name in SERIES:
        series = histories[name]
        if (
            not isinstance(series, pd.Series)
            or series.name != name
            or not isinstance(series.index, pd.DatetimeIndex)
            or series.index.tz is not None
            or not series.index.is_unique
            or not series.index.is_monotonic_increasing
            or (series.index.weekday >= 5).any()
            or not pd.api.types.is_numeric_dtype(series)
            or not np.isfinite(series.to_numpy()).all()
            or (series <= 0).any()
        ):
            raise QuantError("Alignment cannot hide malformed or non-native input histories.")
        aligned[name] = series.reindex(index)
    return aligned


def coverage(aligned: pd.DataFrame) -> dict:
    if list(aligned.columns) != list(SERIES) or not aligned.index.equals(
        sessions("2015-08-10", "2026-10-05")
    ):
        raise QuantError("Coverage must retain the complete original calendar and column order.")
    monthly = list(decision_pairs(aligned.index))
    if len(monthly) != 131:
        raise QuantError("The original 131 month-end opportunities changed.")
    result = {}
    for name in SERIES:
        values = aligned[name]
        finite = values.dropna().to_numpy()
        if not np.isfinite(finite).all() or (finite <= 0).any():
            raise QuantError("Invalid observed quotes cannot be relabelled as missing coverage.")
        month_values = values.loc[monthly]
        result[name] = {
            "context_sessions": len(values),
            "observed_daily_values": int(values.notna().sum()),
            "missing_daily_dates": values.index[values.isna()].strftime("%Y-%m-%d").tolist(),
            "original_monthly_queries": len(monthly),
            "observed_monthly_values": int(month_values.notna().sum()),
            "missing_monthly_dates": month_values.index[month_values.isna()]
            .strftime("%Y-%m-%d")
            .tolist(),
        }
    daily_ready = all(not row["missing_daily_dates"] for row in result.values())
    return {
        "status": "complete_daily_coverage" if daily_ready else "incomplete_daily_coverage",
        "full_daily_coverage": daily_ready,
        "all_original_monthly_queries_covered": all(
            not row["missing_monthly_dates"] for row in result.values()
        ),
        "series": result,
        "values_filled_or_market_dates_dropped": False,
        "strategy_eligibility_or_safe_state_inferred": False,
    }
