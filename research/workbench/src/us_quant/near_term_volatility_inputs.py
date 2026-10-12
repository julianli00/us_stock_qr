from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json, file_digest

SERIES = ("VIX9D", "VIX", "VIX3M")
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"
REFERENCES = ("source_index", "research_snapshot", "previous_preservation", "previous_option_manifest",
              "previous_methodology_manifest", "primary_history_document", "primary_methodology_pdf",
              "primary_methodology_text")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"near_term_spx_option_source_20261011",
        "scope":"source_only_daily_closes_and_existing_horizon_comparators",
        "data_start":"2015-08-10", "as_of":"2026-10-05", "required_market_context_sessions":2805,
        "new_source_series":["VIX9D"], "existing_comparator_series":["VIX", "VIX3M"],
        "daily_input_column_order":list(SERIES), "source_urls":{"VIX9D":URL}, "new_source_field":"CLOSE",
        "native_units":"annualized_option_implied_volatility_percentage_points", "maximum_source_bytes":5_000_000,
        "missing_data_rule":"every_original_nyse_session_required_unknown_gap_fatal_no_fill",
        "new_registered_factor_definitions":0, "new_registered_candidates":0, "new_strategy_evaluations":0,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(policy.get(key) is not False for key in (
            "VVIX_admitted", "strategy_states_targets_or_returns_computed", "existing_W42_or_forward_rules_changed",
            "intraday_first_publication_or_full_history_methodology_verified", "investment_objective_verified",
            "order_authority"))
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the one advertised near-term index,unchanged comparators and source-only authority.")


def parse_history(path: Path, series: str, last_allowed_date: pd.Timestamp) -> pd.Series:
    if (series not in SERIES or not isinstance(last_allowed_date, pd.Timestamp)
        or last_allowed_date.tz is not None or last_allowed_date != last_allowed_date.normalize()
        or not path.is_file() or path.stat().st_size > 5_000_000):
        raise QuantError("Use only bounded actual Cboe CSV histories and the actual acquisition-date limit.")
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
        raise QuantError("The official source must identify its exact native daily closing field.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="raise"))
        values = pd.to_numeric(frame["CLOSE"], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise QuantError("Cboe source dates and closes must be explicit native numeric observations.") from exc
    if (index.empty or index.tz is not None or not index.is_unique or not index.is_monotonic_increasing
        or (index.weekday >= 5).any() or index[-1] > last_allowed_date
        or not np.isfinite(values).all() or (values <= 0).any()):
        raise QuantError("Daily Cboe closes must be unique,chronological,finite,positive and not future data.")
    return pd.Series(values, index=index, name=series)


def align_closes(index: pd.DatetimeIndex, histories: dict[str, pd.Series]) -> pd.DataFrame:
    if not index.equals(sessions("2015-08-10", "2026-10-05")) or set(histories) != set(SERIES):
        raise QuantError("Retain every original2805NYSEsession and the exact three declared horizon inputs.")
    aligned = pd.DataFrame(index=index)
    for name in SERIES:
        source = histories[name]
        if (not isinstance(source, pd.Series) or source.name != name or not isinstance(source.index, pd.DatetimeIndex)
            or source.index.tz is not None or not source.index.is_unique or not source.index.is_monotonic_increasing
            or not pd.api.types.is_numeric_dtype(source) or not np.isfinite(source.to_numpy()).all()
            or (source <= 0).any() or (source.index.weekday >= 5).any()):
            raise QuantError("Each horizon must retain its genuine distinct native daily closing series.")
        values = source.reindex(index)
        if values.isna().any():
            missing = values.index[values.isna()].strftime("%Y-%m-%d").tolist()
            raise QuantError(f"Missing {name} option-index closes cannot be filled or discarded: {missing[:10]}")
        aligned[name] = values
    return aligned


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory / name) for name in (
        "near_term_volatility_inputs.py", "calendar.py", "storage.py")})
