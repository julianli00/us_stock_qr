from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from us_quant.calendar import completed_session, market_calendar
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, utc_now, write_json

SUPPORTED_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})
ANNUAL_MIN_DAYS = 330
ANNUAL_MAX_DAYS = 380
MAX_ANNUAL_AGE_DAYS = 550


def parsed_date(value: object, label: str, *, exchange_bound: bool = False) -> date:
    if not isinstance(value, str):
        raise QuantError(f"{label} must be an ISO calendar date.")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise QuantError(f"Invalid {label}: {value}") from exc
    if result.isoformat() != value:
        raise QuantError(f"{label} must be YYYY-MM-DD.")
    if exchange_bound and not 2000 <= result.year <= 2035:
        raise QuantError(f"{label} falls outside the supported exchange calendar.")
    return result


@lru_cache(maxsize=2048)
def available_at(filed: str) -> pd.Timestamp:
    day = pd.Timestamp(parsed_date(filed, "filing date", exchange_bound=True))
    calendar = market_calendar()
    session = calendar.date_to_session(day, direction="next")
    if session == day:
        session = calendar.next_session(session)
    # Company Facts supplies a filing date, not a precise public-release timestamp.
    return calendar.session_close(session)


def decision_cutoff(as_of: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(parsed_date(as_of, "as-of date", exchange_bound=True))
    session = market_calendar().date_to_session(day, direction="previous")
    if session > completed_session():
        raise QuantError("The requested as-of close has not occurred yet.")
    return session, market_calendar().session_close(session)


@dataclass(frozen=True)
class FilingFact:
    tag: str
    period_start: str | None
    period_end: str
    filed: str
    accession: str
    form: str
    value: float
    available_at: str


def disclosed_records(
    payload: dict, tag: str, cutoff: pd.Timestamp
) -> tuple[list[FilingFact], dict]:
    try:
        source = payload["facts"]["us-gaap"][tag]["units"]["USD"]
    except (KeyError, TypeError) as exc:
        raise QuantError(f"Required US-GAAP/USD fact is absent: {tag}.") from exc
    if not isinstance(source, list):
        raise QuantError(f"{tag} USD facts must be a list.")
    records = []
    audit = {"source_records": len(source), "not_yet_available": 0, "unsupported_form": 0}
    for row in source:
        if not isinstance(row, dict):
            raise QuantError(f"{tag} has a malformed fact record.")
        if row.get("form") not in SUPPORTED_FORMS:
            audit["unsupported_form"] += 1
            continue
        filed = parsed_date(row.get("filed"), f"{tag} filing date")
        if filed > cutoff.date():
            audit["not_yet_available"] += 1
            continue
        known_at = available_at(filed.isoformat())
        if known_at > cutoff:
            audit["not_yet_available"] += 1
            continue
        end = parsed_date(row.get("end"), f"{tag} period end")
        start = parsed_date(row["start"], f"{tag} period start") if "start" in row else None
        value = row.get("val")
        accession = row.get("accn")
        if end > filed or (start is not None and start > end):
            raise QuantError(f"{tag} has inconsistent reporting and filing dates.")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise QuantError(f"{tag} has a nonnumeric or nonfinite disclosed value.")
        if not isinstance(accession, str) or not re.fullmatch(
            r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession
        ):
            raise QuantError(f"{tag} is missing a valid SEC accession.")
        records.append(
            FilingFact(
                tag=tag,
                period_start=start.isoformat() if start else None,
                period_end=end.isoformat(),
                filed=filed.isoformat(),
                accession=accession,
                form=row["form"],
                value=float(value),
                available_at=known_at.isoformat(),
            )
        )
    audit["available_records"] = len(records)
    return records, audit


def select_fact(
    records: list[FilingFact],
    tag: str,
    *,
    annual: bool = False,
    period_start: str | None = None,
    period_end: str | None = None,
) -> FilingFact:
    candidates = []
    for fact in records:
        if fact.tag != tag:
            raise QuantError("Mixed taxonomy tags in one financial selection.")
        if annual:
            if fact.period_start is None:
                continue
            days = (
                date.fromisoformat(fact.period_end) - date.fromisoformat(fact.period_start)
            ).days + 1
            if not ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS:
                continue
        elif fact.period_start is not None:
            continue
        if period_start is not None and fact.period_start != period_start:
            continue
        if period_end is not None and fact.period_end != period_end:
            continue
        candidates.append(fact)
    if not candidates:
        raise QuantError(f"No available, period-aligned {'annual ' if annual else ''}{tag} fact.")
    latest = max((fact.period_end, fact.filed) for fact in candidates)
    matches = [fact for fact in candidates if (fact.period_end, fact.filed) == latest]
    versions = {(fact.period_start, fact.period_end, fact.value) for fact in matches}
    if len(versions) != 1:
        raise QuantError(
            f"Conflicting {tag} facts on the same latest filing date; order is unknown."
        )
    return min(matches, key=lambda fact: (fact.accession, fact.form))


def annual_quality_features(payload: dict, as_of: str) -> dict:
    cik, entity = payload.get("cik"), payload.get("entityName")
    if type(cik) is not int or not 0 < cik < 10**10 or not isinstance(entity, str) or not entity:
        raise QuantError("A Company Facts input must identify one SEC CIK and entity name.")
    session, cutoff = decision_cutoff(as_of)
    tags = ("Assets", "Liabilities", "NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities")
    available, audit = {}, {}
    for tag in tags:
        available[tag], audit[tag] = disclosed_records(payload, tag, cutoff)
    income = select_fact(available["NetIncomeLoss"], "NetIncomeLoss", annual=True)
    if income.period_start is None:
        raise QuantError("An annual income statement requires a start date.")
    age = (session.date() - date.fromisoformat(income.period_end)).days
    if age > MAX_ANNUAL_AGE_DAYS:
        raise QuantError("Annual financial information is stale; no carry-forward beyond 550 days.")
    cash_flow = select_fact(
        available["NetCashProvidedByUsedInOperatingActivities"],
        "NetCashProvidedByUsedInOperatingActivities",
        annual=True,
        period_start=income.period_start,
        period_end=income.period_end,
    )
    assets = select_fact(available["Assets"], "Assets", period_end=income.period_end)
    prior_end = (date.fromisoformat(income.period_start) - timedelta(days=1)).isoformat()
    prior_assets = select_fact(available["Assets"], "Assets", period_end=prior_end)
    liabilities = select_fact(available["Liabilities"], "Liabilities", period_end=income.period_end)
    if min(assets.value, prior_assets.value) <= 0 or liabilities.value < 0:
        raise QuantError("Assets must be positive and reported liabilities nonnegative.")
    mean_assets = assets.value / 2 + prior_assets.value / 2
    if not math.isfinite(mean_assets) or mean_assets <= 0:
        raise QuantError("Cannot compute a finite, positive average asset base.")
    values = {
        "return_on_average_assets": income.value / mean_assets,
        "cash_return_on_average_assets": cash_flow.value / mean_assets,
        "cash_minus_income_over_average_assets": (cash_flow.value - income.value) / mean_assets,
        "liabilities_to_assets": liabilities.value / assets.value,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise QuantError("Nonfinite annual quality features.")
    inputs = {
        "net_income": income,
        "operating_cash_flow": cash_flow,
        "assets": assets,
        "prior_assets": prior_assets,
        "liabilities": liabilities,
    }
    if any(pd.Timestamp(fact.available_at) > cutoff for fact in inputs.values()):
        raise QuantError("A financial feature used a disclosure unavailable at its decision time.")
    return {
        "mode": "point_in_time_financial_feature_preparation",
        "cik": cik,
        "entity": entity,
        "as_of_date": as_of,
        "decision_session": session.date().isoformat(),
        "decision_cutoff_utc": cutoff.isoformat(),
        "annual_period_start": income.period_start,
        "annual_period_end": income.period_end,
        "annual_data_age_days": age,
        "features": values,
        "provenance": {key: asdict(value) for key, value in inputs.items()},
        "record_audit": audit,
        "availability_policy": "first complete NYSE session strictly after the SEC filing date",
        "objective_verified": False,
        "strategy_qualified": False,
        "order_authority": False,
        "limitations": [
            "Descriptive annual features only; no evidence of profitable stock selection.",
            "Accounting asset-return ratios are not portfolio CAGR or trading returns.",
            "Filing-date filtering is not proof that the provider never revised its extraction.",
            "Company Facts is not a complete historical security master or constituent dataset.",
            "The date-only filing proxy does not establish exact intraday publication time.",
            "Annual flows are not quarterly/YTD sums; matching fiscal periods are required.",
            "Only these exact US-GAAP/USD tags and 10-K/10-Q forms are supported.",
            "Missing delisted prices and historical membership block backtest qualification.",
        ],
    }


def write_features(source: Path, as_of: str, output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite a financial-feature audit artifact.")
    result = annual_quality_features(read_json(source), as_of)
    result.update(
        {
            "created_at": utc_now(),
            "source_file": str(source),
            "source_sha256": file_digest(source),
            "implementation_sha256": file_digest(Path(__file__)),
        }
    )
    write_json(output, result)
    return result


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "fundamentals", help="Build filing-time-correct annual features; not trading signals."
    )
    command.add_argument("--facts", type=Path, required=True)
    command.add_argument("--as-of", required=True)
    command.add_argument("--output", type=Path, required=True)
