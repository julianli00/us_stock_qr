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
from us_quant.macro_factor_tilt import align_observations
from us_quant.storage import digest_json, file_digest

FORM_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=T10YIE"
FIELDS = {
    "form[units]": "lin",
    "form[obs_start_date]": "2014-01-01",
    "form[obs_end_date]": "2026-10-05",
    "form[file_type]": "2",
    "form[file_format]": "csv",
    "form[download_data]": "",
}


def request_fields(dates: list[str]) -> dict:
    if (
        not dates
        or dates != sorted(set(dates))
        or len(dates) > 40
        or len(" ".join(dates)) > 500
        or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in dates)
    ):
        raise QuantError(
            "Inflation vintages need exact ordered dates within the public form limit."
        )
    try:
        for day in dates:
            pd.Timestamp(day)
    except ValueError as exc:
        raise QuantError("Inflation vintage dates must be actual calendar dates.") from exc
    return {**FIELDS, "form[entered_vintage_dates]": " ".join(dates)}


def parse_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    request_fields(dates)
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (
                len(archive.namelist()) != 2
                or set(archive.namelist()) != {"README.txt", member}
                or any(item.file_size > 5_000_000 for item in archive.infolist())
            ):
                raise QuantError("Use the exact bounded two-file T10YIE source archive.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            if not all(
                term in readme
                for term in (
                    "Series ID: T10YIE",
                    "10-Year Breakeven Inflation Rate",
                    "Observations by Vintage Date, All Observations",
                    "Federal Reserve Bank of St. Louis",
                    "Interest Rate Spreads",
                    "Percent",
                    "Daily",
                )
            ):
                raise QuantError(
                    "The historical archive does not substantiate the exact inflation series."
                )
            match = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if match is None or match[1].split() != dates:
                raise QuantError(
                    "Every requested inflation vintage must appear in the actual README."
                )
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("Inflation data must be an actual historical ZIP, not HTML.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    expected = ["observation_date", *(f"T10YIE_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected:
        raise QuantError("The inflation columns differ from the exact historical as-of dates.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (
        index.empty
        or not index.is_unique
        or not index.is_monotonic_increasing
        or (index.weekday >= 5).any()
    ):
        raise QuantError("Inflation observations need unique chronological business dates.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any():
            raise QuantError("An inflation snapshot contains invalid or future observations.")
        result[day] = values
    return result


def query_readiness(index: pd.DatetimeIndex, vintages: dict[str, pd.Series]) -> pd.DataFrame:
    if not index.equals(sessions("2015-08-10", "2026-10-05")):
        raise QuantError("The inflation input study retains the exact prior 2,805-session index.")
    required = needed_vintage_dates(index)
    pairs = decision_pairs(index)
    if len(required) != 225 or len(pairs) != 131 or set(vintages) != set(required):
        raise QuantError(
            "Inflation readiness cannot omit any original month or reference snapshot."
        )
    records = []
    for query in sorted(set(pairs) | set(pairs.values())):
        cutoff = str(previous_session(query).date())
        observed = vintages[cutoff]
        known = observed.dropna()
        if known.empty or (known.index > pd.Timestamp(cutoff)).any():
            raise QuantError(
                "Only observations actually present in the earlier snapshot are eligible."
            )
        values, audit = align_observations(observed, pd.DatetimeIndex([query]))
        row = audit.loc[query]
        records.append(
            {
                "query_session": str(query.date()),
                "vintage_cutoff": cutoff,
                "observation_date": str(row["observation_date"].date()),
                "assumed_available_session": str(row["available_session"].date()),
                "age_calendar_days": int(row["age_calendar_days"]),
                "yield_percent": float(values.loc[query]),
            }
        )
    return pd.DataFrame(records).set_index("query_session")


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "inflation_input_readiness_20261011",
        "scope": "historical_monthly_source_readiness_only",
        "series": "T10YIE",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "source_observation_start": "2014-01-01",
        "source_observation_end": "2026-10-05",
        "required_vintage_dates": 225,
        "monthly_comparison_grid": 131,
        "publication_delay_sessions": 2,
        "maximum_observation_age_days": 7,
        "vintage_cutoff": "previous_nyse_session",
        "form_url": FORM_URL,
        "maximum_dates_per_request": 40,
        "maximum_entered_dates_characters": 500,
        "new_registered_factor_definitions": 0,
        "new_strategy_evaluations": 0,
    }
    if any(policy.get(key) != value for key, value in expected.items()) or any(
        policy.get(key) is not False
        for key in (
            "new_factor_family_independence_claimed",
            "strategy_states_or_returns_computed",
            "TIP_counted_as_new_asset_or_factor",
            "forward_rules_changed",
            "existing_queued_proposals_replaced",
            "order_authority",
        )
    ):
        raise QuantError("The source-only inflation scope, lag or no-performance boundary changed.")


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json(
        {
            name: file_digest(folder / name)
            for name in (
                "inflation_inputs.py",
                "macro_factor_tilt.py",
                "financial_conditions_guard.py",
                "dollar_risk_guard.py",
                "calendar.py",
                "storage.py",
            )
        }
    )
