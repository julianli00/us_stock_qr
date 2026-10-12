from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.near_term_volatility_inputs import parse_history as parse_comparator
from us_quant.storage import digest_json, file_digest

SERIES = ("GVZ", "VIX")
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv"
MISSING = ("2018-12-03", "2019-02-20", "2019-07-05", "2020-10-16")
REFERENCES = ("source_index", "research_snapshot", "previous_preservation", "previous_option_manifest",
              "primary_history_document", "actual_initial_requests", "actual_methodology_requests",
              "primary_methodology_pdf", "primary_filter_results_pdf")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"gold_option_closing_source_20261011",
        "scope":"source_only_daily_coverage_and_original_monthly_query_availability",
        "data_start":"2015-08-10", "as_of":"2026-10-05", "required_market_context_sessions":2805,
        "original_monthly_query_count":131, "new_source_series":["GVZ"], "existing_comparator_series":["VIX"],
        "daily_input_column_order":list(SERIES), "new_source_url":URL, "native_new_source_csv_columns":["DATE", "GVZ"],
        "native_value_column":"GVZ", "maximum_source_bytes":5_000_000, "known_missing_gvz_daily_dates":list(MISSING),
        "missing_data_rule":"retain_exact_missing_daily_quotes_without_fill_daily_readiness_false_unknown_gap_fatal",
        "monthly_query_rule":"original131monthends_only_each_actual_native_close_required_no_fill_or_pause",
        "new_registered_factor_definitions":0, "new_registered_candidates":0, "new_strategy_evaluations":0,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(policy.get(key) is not False for key in (
            "full_daily_readiness", "strategy_states_targets_or_returns_computed", "existing_W42_or_forward_rules_changed",
            "intraday_first_publication_or_full_history_methodology_verified", "investment_objective_verified",
            "order_authority"))
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the exact gold-option source gaps,original monthly query scope and source-only authority.")


def parse_history(path: Path, series: str, last_allowed_date: pd.Timestamp) -> pd.Series:
    if series == "VIX":
        return parse_comparator(path, series, last_allowed_date)
    if (series != "GVZ" or not isinstance(last_allowed_date, pd.Timestamp) or last_allowed_date.tz is not None
        or last_allowed_date != last_allowed_date.normalize()
        or not path.is_file() or path.stat().st_size > 5_000_000):
        raise QuantError("Use the bounded native gold-option CSV and actual acquisition-date limit.")
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["DATE", "GVZ"]:
        raise QuantError("The native GVZ value field cannot be replaced by a guessed OHLC or price-return schema.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="raise"))
        values = pd.to_numeric(frame["GVZ"], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise QuantError("Require actual native dates and numeric annualized gold-volatility quotes.") from exc
    if (index.empty or index.tz is not None or not index.is_unique or not index.is_monotonic_increasing
        or (index.weekday >= 5).any() or index[-1] > last_allowed_date
        or not np.isfinite(values).all() or (values <= 0).any()):
        raise QuantError("Gold option-index values must be finite,positive,chronological and not future data.")
    return pd.Series(values, index=index, name=series)


def align_closes(index: pd.DatetimeIndex, histories: dict[str, pd.Series]) -> pd.DataFrame:
    if not index.equals(sessions("2015-08-10", "2026-10-05")) or set(histories) != set(SERIES):
        raise QuantError("Retain the full original2805NYSEcontext and genuine GVZ/VIX identities.")
    aligned = pd.DataFrame(index=index)
    for name in SERIES:
        source = histories[name]
        if (not isinstance(source, pd.Series) or source.name != name or not isinstance(source.index, pd.DatetimeIndex)
            or source.index.tz is not None or not source.index.is_unique or not source.index.is_monotonic_increasing
            or not pd.api.types.is_numeric_dtype(source) or not np.isfinite(source.to_numpy()).all()
            or (source <= 0).any() or (source.index.weekday >= 5).any()):
            raise QuantError("Each option view needs one genuine native positive daily history without substitutes.")
        values = source.reindex(index)
        missing = values.index[values.isna()].strftime("%Y-%m-%d").tolist()
        if missing != (list(MISSING) if name == "GVZ" else []):
            raise QuantError("Only the four exact unfilled GVZ source gaps may remain;unknown or repaired gaps stop.")
        aligned[name] = values
    return aligned


def monthly_closes(aligned: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(aligned, pd.DataFrame) or list(aligned.columns) != list(SERIES)
        or not aligned.index.equals(sessions("2015-08-10", "2026-10-05"))
        or aligned["GVZ"].index[aligned["GVZ"].isna()].strftime("%Y-%m-%d").tolist() != list(MISSING)
        or aligned["VIX"].isna().any()):
        raise QuantError("Keep the blocked full daily cohort,not a filled or shortened monthly-ready surrogate.")
    values = aligned.loc[list(decision_pairs(aligned.index))]
    if len(values) != 131 or not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise QuantError("Every original monthly decision requires actual positive native quotes;no standby is allowed.")
    return values


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory / name) for name in (
        "gold_volatility_inputs.py", "near_term_volatility_inputs.py", "dollar_risk_guard.py",
        "calendar.py", "storage.py")})
