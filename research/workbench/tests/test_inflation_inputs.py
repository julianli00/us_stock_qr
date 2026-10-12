from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.financial_conditions_guard import needed_vintage_dates
from us_quant.inflation_inputs import (
    parse_archive,
    query_readiness,
    request_fields,
    validate_policy,
)
from us_quant.storage import read_json

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, rows=None, series="T10YIE", dates=DATES):
    path = tmp_path / "inflation.zip"
    readme = (
        f"Series ID: {series}\n10-Year Breakeven Inflation Rate\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Federal Reserve Bank of St. Louis\nInterest Rate Spreads\nPercent\nDaily\n"
        "Vintage Dates Specified:\n----------\n" + "\n".join(dates) + "\n----------\n"
    )
    content = (
        "observation_date,T10YIE_20260828,T10YIE_20260901\n"
        + "\n".join(rows or ["2026-08-28,2.10,2.09", "2026-08-31,,2.11"])
        + "\n"
    )
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{dates[0]}.csv", content)
    return path


def test_percent_quote_units_and_revisions_are_not_converted_to_security_returns(tmp_path):
    frame = parse_archive(archive(tmp_path), DATES)
    assert frame.iloc[0, 0] == 2.10
    assert frame.iloc[0, 1] == 2.09
    assert np.isnan(frame.iloc[1, 0])
    assert frame.iloc[1, 1] == 2.11
    assert request_fields(DATES)["form[units]"] == "lin"


@pytest.mark.parametrize("problem", ["series", "dates", "future", "duplicate", "weekend", "inf"])
def test_incorrect_series_vintages_or_observations_are_not_silently_repaired(tmp_path, problem):
    options = {}
    if problem == "series":
        options["series"] = "DFII10"
    elif problem == "dates":
        options["dates"] = ["2026-08-28", "2026-09-02"]
    elif problem == "future":
        options["rows"] = ["2026-08-31,2.1,2.1"]
    elif problem == "duplicate":
        options["rows"] = ["2026-08-28,2.1,2.1", "2026-08-28,2.1,2.1"]
    elif problem == "weekend":
        options["rows"] = ["2026-08-29,,2.1"]
    else:
        options["rows"] = ["2026-08-28,2.1,inf"]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **options), DATES)


@pytest.mark.parametrize(
    "dates",
    [
        DATES[::-1],
        [],
        ["2026-08-28"] * 41,
        ["not-a-date"],
        ["2026-02-30"],
    ],
)
def test_public_form_date_limits_and_exact_order_are_retained(dates):
    with pytest.raises(QuantError):
        request_fields(dates)


def test_every_month_and_reference_query_requires_an_earlier_actual_source_snapshot():
    index = sessions("2015-08-10", "2026-10-05")
    source_dates = sessions("2014-01-01", "2026-10-05")
    vintages = {
        cutoff: pd.Series(2.0, index=source_dates[source_dates <= cutoff], name="T10YIE")
        for cutoff in needed_vintage_dates(index)
    }
    ready = query_readiness(index, vintages)
    assert len(ready) == 225
    assert ready["yield_percent"].eq(2.0).all()
    assert ready["age_calendar_days"].max() <= 7
    assert (
        pd.to_datetime(ready["vintage_cutoff"]).to_numpy() < pd.to_datetime(ready.index).to_numpy()
    ).all()
    bad = dict(vintages)
    bad.pop(next(iter(bad)))
    with pytest.raises(QuantError, match="omit"):
        query_readiness(index, bad)


@pytest.mark.parametrize("problem", ["stale", "future", "index"])
def test_missing_stale_or_future_inputs_block_source_readiness_instead_of_becoming_cash(problem):
    index = sessions("2015-08-10", "2026-10-05")
    source_dates = sessions("2014-01-01", "2026-10-05")
    vintages = {
        cutoff: pd.Series(2.0, index=source_dates[source_dates <= cutoff])
        for cutoff in needed_vintage_dates(index)
    }
    cutoff = next(iter(vintages))
    if problem == "stale":
        vintages[cutoff] = pd.Series([2.0], index=pd.to_datetime(["2014-01-02"]))
    elif problem == "future":
        vintages[cutoff] = pd.Series([2.0], index=pd.to_datetime(["2026-10-05"]))
    else:
        index = index[:-1]
    with pytest.raises(QuantError):
        query_readiness(index, vintages)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_observation_age_days", 14),
        ("publication_delay_sessions", 1),
        ("required_vintage_dates", 224),
        ("new_registered_factor_definitions", 1),
        ("strategy_states_or_returns_computed", True),
        ("TIP_counted_as_new_asset_or_factor", True),
        ("new_factor_family_independence_claimed", True),
        ("existing_queued_proposals_replaced", True),
        ("forward_rules_changed", True),
        ("order_authority", True),
    ],
)
def test_readiness_cannot_become_an_early_strategy_or_factor_independence_claim(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/inflation-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
