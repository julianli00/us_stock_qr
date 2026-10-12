from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.near_term_volatility_inputs import parse_history as parse_comparator
from us_quant.storage import digest_json, file_digest

SERIES = ("SKEW", "VIX")
URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
WHITEPAPER = "https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf"
MISSING = ("2017-09-14", "2018-12-03", "2019-07-05", "2024-11-29")
MONTH_MISSING = ("2024-11-29",)
REFERENCES = ("source_index", "research_snapshot", "previous_preservation", "previous_option_manifest",
              "actual_acquisition", "primary_dashboard", "primary_whitepaper", "new_source_history")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"option_tail_asymmetry_source_20261011",
        "scope":"source_only_SKEW_native_closes_and_primary_interpretation",
        "data_start":"2015-08-10", "as_of":"2026-10-05",
        "source_context_start":"2014-01-02", "source_context_end":"2026-10-05",
        "source_context_nyse_sessions":3208, "original_market_context_nyse_sessions":2805,
        "original_monthly_queries":131, "series":list(SERIES),
        "dashboard_url":"https://www.cboe.com/us/indices/dashboard/SKEW/",
        "new_source_url":URL, "primary_whitepaper_url":WHITEPAPER, "native_new_source_csv_columns":["DATE", "SKEW"],
        "maximum_source_bytes":5_000_000, "known_missing_skew_daily_dates":list(MISSING),
        "known_missing_skew_monthly_queries":list(MONTH_MISSING),
        "missing_data_rule":"retain_exact_daily_and_monthly_NaN_quotes_no_fill_no_safe_state_unknown_gap_fatal",
        "new_registered_factor_definitions":0, "new_registered_candidates":0, "new_strategy_evaluations":0,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(policy.get(key) is not False for key in (
            "full_source_readiness", "full_monthly_query_readiness", "strategy_states_targets_or_returns_computed",
            "physical_crash_probabilities_or_option_trading_returns_computed", "existing_W42_or_forward_rules_changed",
            "investment_objective_verified", "order_authority"))
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the exact tail-asymmetry source units,gaps,contexts and source-only research scope.")


def dashboard_resources(html: str) -> dict:
    blocks = [block for block in re.findall(r"CTX\[cSymbol\]\s*=\s*'([^']*)';", html)
              if "<h2>Cboe SKEW Index (SKEW)</h2>" in block]
    if re.search(r"var symbol = 'SKEW';", html) is None or len(blocks) != 1:
        raise QuantError("Require one scoped static SKEW resource block without executing remote Javascript.")
    document = BeautifulSoup(blocks[0], "html.parser")
    result = {}
    for name, url in (("history", URL), ("whitepaper", WHITEPAPER)):
        links = [link for link in document.find_all("a", href=True) if link["href"] == url]
        if len(links) != 1:
            raise QuantError("Use only the exact primary SKEW history and whitepaper advertisements.")
        result[name] = url
    text = document.get_text(" ", strip=True)
    if "30-day horizon" not in text or "out-of-the-money options" not in text:
        raise QuantError("The primary scoped dashboard must identify the SPX option horizon and skewness role.")
    return result


def parse_history(path: Path, series: str, last_allowed_date: pd.Timestamp) -> pd.Series:
    if series == "VIX":
        return parse_comparator(path, series, last_allowed_date)
    if (series != "SKEW" or not isinstance(last_allowed_date, pd.Timestamp) or last_allowed_date.tz is not None
        or last_allowed_date != last_allowed_date.normalize()
        or not path.is_file() or path.stat().st_size > 5_000_000):
        raise QuantError("Use only bounded actual native SKEW history and the actual acquisition-date limit.")
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["DATE", "SKEW"]:
        raise QuantError("Tail-asymmetry index points cannot be replaced by a volatility/price/OHLC field.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="raise"))
        values = pd.to_numeric(frame["SKEW"], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise QuantError("Require actual native dates and numeric finite tail-asymmetry index points.") from exc
    if (index.empty or index.tz is not None or not index.is_unique or not index.is_monotonic_increasing
        or (index.weekday >= 5).any() or index[-1] > last_allowed_date
        or not np.isfinite(values).all() or (values <= 0).any()):
        raise QuantError("Actual SKEW values must be positive,chronological,distinct and not future data.")
    return pd.Series(values, index=index, name=series)


def align_context(histories: dict[str, pd.Series]) -> pd.DataFrame:
    index = sessions("2014-01-02", "2026-10-05")
    if len(index) != 3208 or set(histories) != set(SERIES):
        raise QuantError("Retain the entire original source-precontext and both genuine native series.")
    result = pd.DataFrame(index=index)
    for name in SERIES:
        source = histories[name]
        if (not isinstance(source, pd.Series) or source.name != name or not isinstance(source.index, pd.DatetimeIndex)
            or source.index.tz is not None or not source.index.is_unique or not source.index.is_monotonic_increasing
            or not pd.api.types.is_numeric_dtype(source) or not np.isfinite(source.to_numpy()).all()
            or (source <= 0).any() or (source.index.weekday >= 5).any()):
            raise QuantError("Each native tail/volatility view requires a genuine finite positive daily source.")
        aligned = source.reindex(index)
        if aligned.index[aligned.isna()].strftime("%Y-%m-%d").tolist() != (list(MISSING) if name == "SKEW" else []):
            raise QuantError("Only exact documented unfilled SKEW gaps are allowed;unknown or repaired gaps stop.")
        result[name] = aligned
    return result


def monthly_queries(context: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(context, pd.DataFrame) or list(context.columns) != list(SERIES)
        or not context.index.equals(sessions("2014-01-02", "2026-10-05"))
        or context["SKEW"].index[context["SKEW"].isna()].strftime("%Y-%m-%d").tolist() != list(MISSING)
        or context["VIX"].isna().any()):
        raise QuantError("Keep the exact daily source gaps and native identities in their full context.")
    market = sessions("2015-08-10", "2026-10-05")
    result = context.loc[list(decision_pairs(market))].copy()
    if (len(result) != 131 or result["SKEW"].index[result["SKEW"].isna()].strftime("%Y-%m-%d").tolist() != list(MONTH_MISSING)
        or np.isinf(result.to_numpy()).any() or (result.dropna() <= 0).any().any()):
        raise QuantError("Every original monthly query and the one genuine missing reading must stay explicit.")
    result["source_available"] = result["SKEW"].notna()
    result["source_reason"] = None
    result.loc[list(MONTH_MISSING), "source_reason"] = "actual_SKEW_quote_missing_no_safe_default"
    return result


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory / name) for name in (
        "skew_inputs.py", "near_term_volatility_inputs.py", "dollar_risk_guard.py", "calendar.py", "storage.py")})
