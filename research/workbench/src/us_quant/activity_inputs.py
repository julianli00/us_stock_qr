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
from us_quant.inflation_inputs import request_fields as prior_date_fields
from us_quant.storage import digest_json, file_digest

SERIES = {
    "CFNAIMA3": "Chicago Fed National Activity Index: Three Month Moving Average",
    "CFNAIDIFF": "Chicago Fed National Activity Index: Diffusion Index",
}


def request_fields(series: str, dates: list[str]) -> dict:
    if series not in SERIES:
        raise QuantError("Use only the two declared Chicago Fed real-activity series.")
    return {**prior_date_fields(dates), "form[obs_end_date]":"2026-08-01"}


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version":1, "study_id":"chicago_real_activity_vintage_source_20261011",
        "scope":"historical_monthly_source_readiness_only", "series":list(SERIES),
        "data_start":"2015-08-10", "as_of":"2026-10-05",
        "source_observation_start":"2014-01-01", "source_observation_end":"2026-08-01",
        "monthly_reference_pairs":131, "query_sessions_per_series":225, "required_vintage_dates_per_series":225,
        "vintage_cutoff":"second_previous_nyse_session", "maximum_completed_period_age_days":62,
        "freshness_basis":"query_date_minus_completed_observation_month_end_not_month_label_or_download_date",
        "missing_stale_or_unverified_input":"block_source_readiness_preserve_failure_no_fill_or_standby_strategy",
        "maximum_dates_per_request":40, "maximum_entered_dates_characters":500,
        "independent_probe_dates":["2015-08-27", "2021-02-24", "2026-09-28"],
        "new_registered_factor_definitions":0, "new_strategy_evaluations":0, "new_registered_candidates":0,
    }
    if (not isinstance(policy, dict) or any(policy.get(name) != value for name, value in expected.items())
        or any(policy.get(name) is not False for name in (
            "strategy_states_targets_or_returns_computed", "new_independent_factor_families_claimed",
            "existing_W42_or_forward_rules_changed", "investment_objective_verified", "order_authority"))
        or any(not isinstance(policy.get(name), dict) or set(policy[name]) != {"path", "sha256"}
               for name in ("source_index", "research_snapshot", "previous_preservation"))
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the frozen monthly vintage cohort,freshness,no-portfolio scope and original programme.")
    definitions = {
        **{name:f"https://fred.stlouisfed.org/series/{name}" for name in SERIES},
        **{f"{name}_form":f"https://alfred.stlouisfed.org/series/downloaddata?seid={name}" for name in SERIES},
        "publisher_current_data":"https://www.chicagofed.org/research/data/cfnai/current-data",
        "publisher_advertised_background_pdf":"https://www.chicagofed.org/-/media/publications/cfnai/background/cfnai-background-pdf.pdf",
    }
    if policy.get("source_definitions") != definitions:
        raise QuantError("Preserve the exact primary definitions and publicly advertised vintage forms.")


def cohort(index: pd.DatetimeIndex) -> tuple[list[pd.Timestamp], list[str]]:
    if not index.equals(sessions("2015-08-10", "2026-10-05")):
        raise QuantError("Activity inputs retain the complete original2805NYSEsessions.")
    pairs = decision_pairs(index)
    queries = sorted(set(pairs) | set(pairs.values()))
    dates = sorted({str(previous_session(previous_session(day)).date()) for day in queries})
    if len(pairs) != 131 or len(queries) != 225 or len(dates) != 225:
        raise QuantError("Activity readiness cannot omit any original monthly or63-session reference query.")
    return queries, dates


def parse_archive(path: Path, series: str, dates: list[str]) -> pd.DataFrame:
    request_fields(series, dates)
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (len(archive.namelist()) != 2 or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())):
                raise QuantError("Use the exact bounded two-member monthly activity archive.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            if not all(term in readme for term in (
                f"Series ID: {series}", SERIES[series], "Observations by Vintage Date, All Observations",
                "Federal Reserve Bank of Chicago", "Chicago Fed National Activity Index",
                "Index", "Monthly", "Not Seasonally Adjusted")):
                raise QuantError("The archive does not substantiate the declared monthly activity series.")
            match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if match is None or match[1].split() != dates:
                raise QuantError("The primary README must identify every requested activity vintage exactly.")
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("Activity vintages must be an actual readable ZIP,not an HTML success-shaped response.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    expected = ["observation_date", *(f"{series}_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected:
        raise QuantError("Activity columns changed the declared series or exact requested vintage dates.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (index.empty or not index.is_unique or not index.is_monotonic_increasing or index.tz is not None
        or not (index.day == 1).all()
        or (index < pd.Timestamp("2014-01-01")).any() or (index > pd.Timestamp("2026-08-01")).any()):
        raise QuantError("Activity observations require distinct chronological first-of-month period labels.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if (not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any()
            or series == "CFNAIDIFF" and (abs(values[known]) > 1 + 1e-12).any()):
            raise QuantError("A monthly vintage has invalid/future data or impossible weighted diffusion values.")
        result[day] = values
    return result


def select(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (not isinstance(observed, pd.Series) or not isinstance(observed.index, pd.DatetimeIndex)
        or not pd.api.types.is_numeric_dtype(observed)
        or not isinstance(query, pd.Timestamp) or query.tz is not None or query != query.normalize()
        or query not in sessions(query.date().isoformat(), query.date().isoformat())
        or cutoff != str(previous_session(previous_session(query)).date())
        or not observed.index.is_unique or not observed.index.is_monotonic_increasing
        or observed.index.tz is not None or not (observed.index.day == 1).all()
        or np.isinf(observed.to_numpy()).any()):
        raise QuantError("Only exact second-prior-NYSE monthly snapshots may be selected.")
    known = observed.dropna()
    if known.empty or (known.index > pd.Timestamp(cutoff)).any():
        raise QuantError("Missing or future activity snapshots cannot use current-history fallback.")
    month = known.index[-1]
    completed = month + pd.offsets.MonthEnd(0)
    if completed > pd.Timestamp(cutoff):
        raise QuantError("A first-of-month label is not a completed or released monthly observation.")
    age = (query - completed).days
    if not 0 <= age <= 62:
        raise QuantError("Monthly completed-period freshness exceeds the frozen62day source contract.")
    value = float(known.iloc[-1])
    if not np.isfinite(value):
        raise QuantError("A selected real-activity observation must be finite,not a default risk state.")
    return value, {
        "query_session":str(query.date()), "vintage_cutoff":cutoff,
        "observation_month":str(month.date()), "completed_period_end":str(completed.date()),
        "completed_period_age_calendar_days":int(age),
        "snapshot_available_two_nyse_sessions_before_query":True,
        "month_label_claimed_first_release":False,
    }


def query_readiness(index: pd.DatetimeIndex, vintages: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    queries, dates = cohort(index)
    if (set(vintages) != set(SERIES)
        or any(set(vintages[name]) != set(dates) for name in SERIES)):
        raise QuantError("Every one of the450series/query snapshots is required;no smaller fallback cohort.")
    records = []
    for name in SERIES:
        for query in queries:
            cutoff = str(previous_session(previous_session(query)).date())
            value, audit = select(vintages[name][cutoff], query, cutoff)
            records.append({"series":name, **audit, "index_value":value})
    return pd.DataFrame(records)


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json({name:file_digest(directory / name) for name in (
        "activity_inputs.py", "inflation_inputs.py", "dollar_risk_guard.py", "calendar.py", "storage.py")})
