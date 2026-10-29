from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import market_calendar, previous_session, sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.near_term_volatility_inputs import parse_history as parse_existing_vix

SERIES = ("VXN", "VIX")
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"
REFERENCES = (
    "source_index",
    "original_monthly_query_index",
    "new_raw_csv",
    "existing_vix_csv",
    "existing_option_manifest",
    "source_intent",
    "dashboard_acquisition",
    "advertised_resources",
    "definition_before_history",
    "history_acquisition",
    "methodology",
    "methodology_text",
    "methodology_acquisition",
)


def validate_policy(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_vxn_source_readiness_20261029_v1",
        "scope": "archival_daily_closes_and_original_query_coverage_only",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "required_market_context_sessions": 2805,
        "original_monthly_query_count": 131,
        "new_source_series": ["VXN"],
        "existing_comparator_series": ["VIX"],
        "daily_input_column_order": list(SERIES),
        "new_source_url": URL,
        "native_new_source_csv_columns": ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"],
        "native_value_column": "CLOSE",
        "native_units": "annualized_option_implied_volatility_percentage_points",
        "maximum_source_bytes": 5_000_000,
        "regular_index_close_ET": "16:15",
        "source_completed_session_ceiling": {"VXN": "2026-10-28", "VIX": "2026-10-09"},
        "independent_native_value_tolerance": 1e-12,
        "missing_data_rule": (
            "retain_all_original_rows_and_missing_cells_no_fill_drop_or_safe_state"
        ),
        "full_daily_coverage_required_for_full_admission": True,
        "new_registered_factor_definitions": 0,
        "new_registered_candidates": 0,
        "new_strategy_evaluations": 0,
    }
    if (
        not isinstance(policy, dict)
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in expected.items()
        )
        or any(
            policy.get(key) is not False
            for key in (
                "old_prices_quotes_methods_and_timestamps_replaced",
                "ratios_strategy_states_targets_or_returns_computed",
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
            "Preserve the VXN identity, full original source coverage and source-only boundary."
        )


def last_completed_session(acquired_at):
    try:
        stamp = pd.Timestamp(acquired_at)
    except (ValueError, TypeError) as error:
        raise QuantError("Use the actual timezone-aware source acquisition timestamp.") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise QuantError("Use the actual timezone-aware source acquisition timestamp.")
    local = stamp.tz_convert("America/New_York")
    day = local.tz_localize(None).normalize()
    available = market_calendar().date_to_session(day, direction="previous")
    if available == day and local.time() < time(16, 15):
        return previous_session(available)
    return available


def parse_history(path: Path, name: str, ceiling: pd.Timestamp) -> pd.Series:
    if (
        name not in SERIES
        or not isinstance(ceiling, pd.Timestamp)
        or ceiling.tz is not None
        or ceiling != ceiling.normalize()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 5_000_000
    ):
        raise QuantError(
            "Require the bounded native file and its actual completed-session ceiling."
        )
    if name == "VIX":
        return parse_existing_vix(path, name, ceiling)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
        raise QuantError("VXN must retain its actual native OHLC header and CLOSE field.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="raise"))
        values = pd.to_numeric(frame["CLOSE"], errors="raise").to_numpy()
    except (ValueError, TypeError) as error:
        raise QuantError("VXN source dates and native closing quotes must be explicit.") from error
    if (
        index.empty
        or not index.is_unique
        or not index.is_monotonic_increasing
        or (index.weekday >= 5).any()
        or index[-1] > ceiling
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise QuantError("VXN closes must be positive, finite, chronological and not future data.")
    return pd.Series(values, index=index, name=name)


def align_closes(index, histories):
    if not index.equals(sessions("2015-08-10", "2026-10-05")) or set(histories) != set(SERIES):
        raise QuantError("Preserve every original market date and the VXN/VIX identities.")
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
            raise QuantError(
                "Alignment cannot conceal invalid or mislabeled native index histories."
            )
        aligned[name] = series.reindex(index)
    return aligned


def coverage(aligned):
    if list(aligned.columns) != list(SERIES) or not aligned.index.equals(
        sessions("2015-08-10", "2026-10-05")
    ):
        raise QuantError("Keep the full original calendar and declared source-column order.")
    monthly = list(decision_pairs(aligned.index))
    if len(monthly) != 131:
        raise QuantError("The original monthly query lattice changed.")
    by_series = {}
    for name in SERIES:
        values = aligned[name]
        observed = values.dropna().to_numpy()
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise QuantError("Malformed observations cannot become missing cells or safe states.")
        selected = values.loc[monthly]
        by_series[name] = {
            "context_sessions": len(values),
            "observed_daily_values": int(values.notna().sum()),
            "missing_daily_dates": values.index[values.isna()].strftime("%Y-%m-%d").tolist(),
            "original_monthly_queries": len(monthly),
            "observed_monthly_values": int(selected.notna().sum()),
            "missing_monthly_dates": selected.index[selected.isna()].strftime("%Y-%m-%d").tolist(),
        }
    ready = all(not row["missing_daily_dates"] for row in by_series.values())
    return {
        "status": "complete_daily_coverage" if ready else "incomplete_daily_coverage",
        "full_daily_coverage": ready,
        "all_original_monthly_queries_covered": all(
            not row["missing_monthly_dates"] for row in by_series.values()
        ),
        "series": by_series,
        "values_filled_or_market_dates_dropped": False,
        "strategy_eligibility_or_safe_state_inferred": False,
    }
