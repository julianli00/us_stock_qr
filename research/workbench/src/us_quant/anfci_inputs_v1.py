from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant import ssd_research_v1 as base
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = base.ROOT
POLICY = ROOT / "config/anfci-inputs-v1.json"
DATA = ROOT / "data/ssd-anfci-vintages-20261028-v1"
REPORTS = base.REPORTS / "anfci-vintage-admission-20261028-v1"
PUBLIC = ROOT / "evidence/anfci_input_readiness_20261028_v1.json"


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "source_id": "ssd_anfci_vintages_20261028_v1",
        "series": "ANFCI",
        "source": "Federal Reserve Bank of Chicago",
        "frequency": "Weekly, Ending Friday",
        "units": "Index",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "url": "https://alfred.stlouisfed.org/series/downloaddata?seid=ANFCI",
        "method": "POST",
        "form_units": "lin",
        "file_type": "2",
        "file_format": "csv",
        "observation_start": "2014-01-01",
        "observation_end": "2026-10-02",
        "vintage_cutoff": "previous_NYSE_session",
        "query_count": 225,
        "original_monthly_decisions": 131,
        "reference_query_lag_sessions": 63,
        "maximum_dates_per_request": 40,
        "maximum_entered_dates_characters": 500,
        "maximum_observation_age_calendar_days": 14,
        "maximum_archive_bytes": 10_000_000,
        "maximum_member_bytes": 5_000_000,
        "automatic_request_retries": 0,
        "transport": "anonymous_public_form_http",
        "maximum_same_origin_download_redirects": 2,
        "allowed_redirect_statuses": [301, 302, 303],
        "credential_store_access": False,
        "custom_user_agent_or_TLS_override": False,
        "missing_value_marker": ".",
        "full_admission_requires_all_queries_fresh": True,
        "current_history_substitution_allowed": False,
        "missing_observations_filled": False,
        "raw_inputs_or_aligned_values_public": False,
        "strategy_rule_selected": False,
        "new_factor_definitions": 0,
        "new_strategy_evaluations": 0,
        "order_authority": False,
    }
    if any(
        key not in policy or policy[key] != value or type(policy[key]) is not type(value)
        for key, value in expected.items()
    ):
        raise QuantError("The fixed ANFCI source, timing, freshness or no-retry boundary changed.")


def checked_plan(policy: dict) -> dict:
    validate(policy)
    plan = read_json(base.checked_reference(policy["query_plan"]))
    base.checked_reference(policy["source_review"])
    base.checked_reference(policy["form_contract"])
    dates, queries = plan["vintage_cutoffs"], plan["query_sessions"]
    if (
        len(dates) != 225
        or len(queries) != 225
        or dates != sorted(set(dates))
        or queries != sorted(set(queries))
        or len(plan["batches"]) != 6
        or [day for batch in plan["batches"] for day in batch] != dates
        or any(
            str(previous_session(day).date()) != cutoff
            for day, cutoff in zip(queries, dates, strict=True)
        )
    ):
        raise QuantError("Preserve all225 original prior-session vintage cutoffs.")
    for batch in plan["batches"]:
        validate_dates(batch)
    return plan


def validate_dates(dates: list[str]) -> None:
    if (
        not dates
        or dates != sorted(set(dates))
        or len(dates) > 40
        or len(" ".join(dates)) > 500
        or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in dates)
    ):
        raise QuantError("Vintage dates must obey the original advertised bounded form.")
    try:
        pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    except ValueError as error:
        raise QuantError("A requested vintage is not an actual calendar date.") from error


def form_fields(dates: list[str], policy: dict) -> dict[str, str]:
    validate_dates(dates)
    return {
        "form[units]": policy["form_units"],
        "form[obs_start_date]": policy["observation_start"],
        "form[obs_end_date]": policy["observation_end"],
        "form[entered_vintage_dates]": " ".join(dates),
        "form[file_type]": policy["file_type"],
        "form[file_format]": policy["file_format"],
        "form[download_data]": "",
    }


def verify_request(receipt: dict, dates: list[str], policy: dict, earliest: str) -> dict:
    exchanges = receipt["exchanges"]
    if not exchanges or len(exchanges) > policy["maximum_same_origin_download_redirects"] + 1:
        raise QuantError("Require a bounded actual source request/response chain.")
    first = exchanges[0]
    stamp = pd.Timestamp(first["started_at"])
    if (
        first["method"] != "POST"
        or first["url"] != policy["url"]
        or stamp.tzinfo is None
        or stamp < pd.Timestamp(earliest)
        or exchanges[-1]["status"] != 200
        or receipt["automatic_retries"] != 0
    ):
        raise QuantError(
            "Require one successful original form submission after its recorded attempt."
        )
    actual = parse_qs(first["encoded_body"], keep_blank_values=True, strict_parsing=True)
    expected = {key: [value] for key, value in form_fields(dates, policy).items()}
    if actual != expected:
        raise QuantError("The actual POST fields differ from the frozen source request.")
    for previous, following in zip(exchanges[:-1], exchanges[1:], strict=True):
        destination = urljoin(previous["url"], previous["headers"].get("location", ""))
        parsed = urlsplit(destination)
        if (
            previous["status"] not in policy["allowed_redirect_statuses"]
            or parsed.scheme != "https"
            or parsed.netloc != "alfred.stlouisfed.org"
            or following["url"] != destination
            or following["method"] != "GET"
            or following["encoded_body"] is not None
        ):
            raise QuantError(
                "Only a bounded same-origin server-directed GET may follow a redirect."
            )
    return {
        "actual_started_at": stamp.isoformat(),
        "url": first["url"],
        "method": first["method"],
        "HTTP_status": exchanges[-1]["status"],
        "actual_POST_submissions": 1,
        "server_directed_GETs": len(exchanges) - 1,
        "actual_form_fields": form_fields(dates, policy),
    }


def archive_contents(path: Path, dates: list[str], policy: dict) -> tuple[str, bytes]:
    validate_dates(dates)
    if path.stat().st_size > policy["maximum_archive_bytes"]:
        raise QuantError("The source archive exceeds the predeclared size limit.")
    name = f"vintages_starting_{dates[0]}.csv"
    try:
        with ZipFile(path) as archive:
            if (
                set(archive.namelist()) != {"README.txt", name}
                or len(archive.infolist()) != 2
                or any(
                    item.file_size > policy["maximum_member_bytes"] for item in archive.infolist()
                )
            ):
                raise QuantError("Require the bounded two-member ALFRED archive without extras.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            content = archive.read(name)
    except (BadZipFile, UnicodeError) as error:
        raise QuantError("The downloaded object is not a readable ALFRED ZIP.") from error
    compact = " ".join(readme.split())
    if (
        not re.search(r"Series ID:\s*ANFCI\b", readme)
        or "Adjusted National Financial Conditions Index" not in compact
        or "Federal Reserve Bank of Chicago" not in compact
        or "Weekly, Ending Friday" not in compact
        or "Not Seasonally Adjusted" not in compact
        or "Index" not in compact
        or "Output Format: Observations by Vintage Date, All Observations" not in compact
    ):
        raise QuantError("The archive metadata does not identify the actual plain ANFCI series.")
    match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
    if match is None or match[1].split() != dates:
        raise QuantError("Archive README does not confirm every requested vintage.")
    return readme, content


def parse_archive(path: Path, dates: list[str], policy: dict) -> tuple[pd.DataFrame, dict]:
    readme, content = archive_contents(path, dates, policy)
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    names = ["observation_date", *[f"ANFCI_{day.replace('-', '')}" for day in dates]]
    if list(frame.columns) != names:
        raise QuantError("Source columns must identify exactly the requested ANFCI vintages.")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("observation_date"), format="%Y-%m-%d", errors="raise")
        )
    except ValueError as error:
        raise QuantError("ANFCI observation dates are malformed.") from error
    if (
        index.empty
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or not (index.weekday == 4).all()
        or index[0] < pd.Timestamp(policy["observation_start"])
        or index[-1] > pd.Timestamp(policy["observation_end"])
    ):
        raise QuantError(
            "ANFCI must retain unique chronological Friday dates in the requested range."
        )
    result = pd.DataFrame(index=index)
    for day, name in zip(dates, names[1:], strict=True):
        missing = frame[name].eq(".").to_numpy()
        try:
            values = pd.to_numeric(frame[name].mask(missing), errors="raise").to_numpy(dtype=float)
        except ValueError as error:
            raise QuantError("ANFCI values contain an unknown nonnumeric token.") from error
        if (
            not np.array_equal(np.isnan(values), missing)
            or np.isinf(values).any()
            or missing.all()
            or (index[~missing] > pd.Timestamp(day)).any()
        ):
            raise QuantError("A vintage has invalid, wholly missing or future observations.")
        result[day] = values
    independent = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if independent[0] != names or len(independent) - 1 != len(result):
        raise QuantError("Independent raw CSV shape differs.")
    maximum, compared = 0.0, 0
    for position, row in enumerate(independent[1:]):
        if len(row) != len(names) or row[0] != str(index[position].date()):
            raise QuantError("Independent CSV dates or widths differ.")
        for column, cell in enumerate(row[1:]):
            actual = result.iloc[position, column]
            if cell == ".":
                if not pd.isna(actual):
                    raise QuantError("A source missing value was silently replaced.")
            else:
                value = float(cell)
                if not np.isfinite(value) or not np.isfinite(actual):
                    raise QuantError("Independent source numeric parsing failed.")
                maximum = max(maximum, abs(value - float(actual)))
                compared += 1
    if maximum > 1e-12:
        raise QuantError("Independent raw-source values disagree with the primary parser.")
    return result, {
        "observation_rows": len(result),
        "vintage_columns": len(dates),
        "finite_cells_independently_compared": compared,
        "maximum_independent_numeric_difference": maximum,
        "README_vintage_metadata_verified": True,
        "README_text_length": len(readme),
        "raw_future_missing_cells_preserved": True,
    }


def assess_queries(values: dict[str, pd.Series], queries: list[str], policy: dict) -> dict:
    rows = []
    for query in queries:
        cutoff = str(previous_session(query).date())
        if cutoff not in values:
            raise QuantError("An exact required own-vintage snapshot is absent.")
        actual = values[cutoff].dropna()
        if actual.empty or not np.isfinite(actual).all() or actual.index[-1] > pd.Timestamp(cutoff):
            raise QuantError("A vintage lacks valid actually available observations.")
        latest = actual.index[-1]
        age = (pd.Timestamp(query) - latest).days
        if age < 0:
            raise QuantError("A query uses an observation from its future.")
        rows.append(
            {
                "query_session": query,
                "vintage_cutoff": cutoff,
                "observation_date": str(latest.date()),
                "age_calendar_days": age,
                "fresh": age <= policy["maximum_observation_age_calendar_days"],
                "actual_latest_value": float(actual.iloc[-1]),
            }
        )
    stale = [row for row in rows if not row["fresh"]]
    return {
        "query_count": len(rows),
        "fresh_query_count": len(rows) - len(stale),
        "stale_query_count": len(stale),
        "maximum_observation_age_calendar_days": max(row["age_calendar_days"] for row in rows),
        "all_queries_fresh": not stale,
        "query_readings": rows,
        "stale_queries": [
            {key: value for key, value in row.items() if key != "actual_latest_value"}
            for row in stale
        ],
        "no_fill_age_relaxation_or_missingness_strategy": True,
    }
