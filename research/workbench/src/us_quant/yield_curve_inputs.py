from __future__ import annotations

import io
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.financial_conditions_guard import needed_vintage_dates
from us_quant.inflation_inputs import request_fields as existing_request_fields
from us_quant.macro_factor_tilt import align_observations
from us_quant.storage import digest_json, file_digest

SERIES = {
    "T10Y3M":"10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity",
    "T10Y2Y":"10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
}
REFERENCES = ("source_index", "research_snapshot", "previous_preservation", "previous_t10y3m_source",
              "previous_macro_policy", "previous_macro_helper")


def request_fields(series: str, dates: list[str]) -> dict:
    if series not in SERIES:
        raise QuantError("Use only the two declared Treasury constant-maturity spreads.")
    return existing_request_fields(dates)


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"treasury_curve_vintage_source_20261011",
        "scope":"source_only_vintages_and_existing_input_version_comparison", "series":list(SERIES),
        "data_start":"2015-08-10", "as_of":"2026-10-05", "source_observation_start":"2014-01-01",
        "source_observation_end":"2026-10-05", "query_sessions_per_series":225, "monthly_reference_pairs":131,
        "vintage_cutoff":"previous_nyse_session", "publication_delay_sessions":2,
        "maximum_observation_age_days":7, "maximum_dates_per_request":40, "maximum_entered_dates_characters":500,
        "independent_probe_dates":["2015-08-28", "2021-02-25", "2026-09-29"],
        "new_registered_factor_definitions":0, "new_registered_candidates":0, "new_strategy_evaluations":0,
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(policy.get(key) is not False for key in (
            "strategy_states_targets_or_returns_computed", "T10Y3M_counted_as_new_factor_or_asset",
            "new_independent_factor_families_claimed", "existing_W42_or_forward_rules_changed",
            "investment_objective_verified", "order_authority"))
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the source-only Treasury curve cohort,lag,age and unchanged research authority.")
    expected = {
        **{name:f"https://fred.stlouisfed.org/series/{name}" for name in SERIES},
        **{f"{name}_form":f"https://alfred.stlouisfed.org/series/downloaddata?seid={name}" for name in SERIES},
        "NYFed_context":"https://www.newyorkfed.org/research/capital_markets/ycfaq",
    }
    if policy.get("primary_definitions") != expected:
        raise QuantError("Use the exact primary series definitions and advertised public vintage forms.")


def cohort(index: pd.DatetimeIndex) -> tuple[list[pd.Timestamp], list[str]]:
    if not index.equals(sessions("2015-08-10", "2026-10-05")):
        raise QuantError("Yield-source validation must retain the original2805NYSEsession context.")
    pairs = decision_pairs(index)
    queries, dates = sorted(set(pairs) | set(pairs.values())), needed_vintage_dates(index)
    if len(pairs) != 131 or len(queries) != 225 or len(dates) != 225:
        raise QuantError("Do not omit any original monthly or63-session reference query.")
    return queries, dates


def archive_metadata(readme: str) -> dict[str, str]:
    lines = readme.splitlines()
    headers = [position for position, line in enumerate(lines)
               if re.fullmatch(r"Title\s+Real-Time Start\s+Real-Time End", line.strip())]
    if len(headers) != 1:
        raise QuantError("Require the actual ALFRED metadata table with explicit real-time columns.")
    position = headers[0]
    value_width = lines[position].index("Real-Time Start")
    sections = ("Title", "Source", "Release", "Units", "Frequency", "Seasonal Adjustment", "Notes")
    values = {name:[] for name in sections}
    current = "Title"
    for line in lines[position+1:]:
        if line.strip().startswith("---"):
            break
        value = line[:value_width].strip()
        if value in sections:
            current = value
        elif value:
            values[current].append(value)
    result = {name:" ".join(parts) for name, parts in values.items()}
    if any(not result[name] for name in sections[:-1]):
        raise QuantError("The archive is missing a required scoped series metadata field.")
    return result


def parse_archive(path: Path, series: str, dates: list[str]) -> pd.DataFrame:
    request_fields(series, dates)
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (len(archive.namelist()) != 2 or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())):
                raise QuantError("Use the exact bounded two-file Treasury spread archive.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            metadata = archive_metadata(readme)
            expected_metadata = {
                "Title":SERIES[series], "Source":"Federal Reserve Bank of St. Louis",
                "Release":"Interest Rate Spreads", "Units":"Percent", "Frequency":"Daily",
                "Seasonal Adjustment":"Not Seasonally Adjusted",
            }
            if (re.search(rf"^Series ID:\s*{series}\s*$", readme, re.M) is None
                or "Output Format: Observations by Vintage Date, All Observations" not in readme
                or any(metadata[name] != value for name, value in expected_metadata.items())):
                raise QuantError("The scoped archive metadata do not substantiate the exact native daily Treasury spread.")
            match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if match is None or match[1].split() != dates:
                raise QuantError("Every exact requested Treasury vintage must appear in the primary README.")
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("A yield-curve vintage must be an actual readable ZIP,not an HTML success-shaped response.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    expected = ["observation_date", *(f"{series}_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected:
        raise QuantError("The actual source columns changed the declared spread or its requested vintages.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (index.empty or not index.is_unique or not index.is_monotonic_increasing or index.tz is not None
        or (index.weekday >= 5).any() or (index < pd.Timestamp("2014-01-01")).any()
        or (index > pd.Timestamp("2026-10-05")).any()):
        raise QuantError("Daily spreads require distinct chronological business-day observations.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any():
            raise QuantError("A Treasury spread vintage contains missing,nonfinite or future data.")
        result[day] = values
    return result


def select(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (not isinstance(observed, pd.Series) or not isinstance(observed.index, pd.DatetimeIndex)
        or observed.index.tz is not None or not observed.index.is_unique
        or not observed.index.is_monotonic_increasing or not pd.api.types.is_numeric_dtype(observed)
        or (observed.index.weekday >= 5).any() or np.isinf(observed.to_numpy()).any()
        or not isinstance(query, pd.Timestamp) or query.tz is not None or query != query.normalize()
        or query not in sessions(query.date().isoformat(), query.date().isoformat())
        or cutoff != str(previous_session(query).date())):
        raise QuantError("Use only the exact previous-NYSE-date spread snapshot and actual query session.")
    known = observed.dropna()
    if known.empty or (known.index > pd.Timestamp(cutoff)).any():
        raise QuantError("Missing or future snapshots cannot be filled from current histories.")
    values, audit = align_observations(observed, pd.DatetimeIndex([query]))
    row = audit.loc[query]
    return float(values.loc[query]), {
        "query_session":str(query.date()), "vintage_cutoff":cutoff,
        "observation_date":str(row["observation_date"].date()),
        "assumed_available_session":str(row["available_session"].date()),
        "age_calendar_days":int(row["age_calendar_days"]),
        "cutoff_strictly_before_query":True,
    }


def query_readiness(index: pd.DatetimeIndex, vintages: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    queries, dates = cohort(index)
    if set(vintages) != set(SERIES) or any(set(values) != set(dates) for values in vintages.values()):
        raise QuantError("Both complete225snapshot cohorts are required,without current-history substitution.")
    rows = []
    for name in SERIES:
        for query in queries:
            cutoff = str(previous_session(query).date())
            value, audit = select(vintages[name][cutoff], query, cutoff)
            rows.append({"series":name, **audit, "spread_percentage_points":value})
    return pd.DataFrame(rows)


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory / name) for name in (
        "yield_curve_inputs.py", "inflation_inputs.py", "macro_factor_tilt.py",
        "financial_conditions_guard.py", "dollar_risk_guard.py", "calendar.py", "storage.py")})
