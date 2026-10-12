from __future__ import annotations

import textwrap
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.profit_inputs import SERIES, cohort, parse_archive, query_readiness, request_fields, select, validate_policy
from us_quant.storage import read_json

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, series="CPATAX", rows=None, dates=DATES, metadata=None, extra=False):
    fields = {
        "Title": SERIES[series], "Source": "U.S. Bureau of Economic Analysis",
        "Release": "Gross Domestic Product", "Units": "Billions of Dollars",
        "Frequency": "Quarterly", "Seasonal Adjustment": "Seasonally Adjusted Annual Rate",
        "Notes": "Quarter labels are not release dates.", **(metadata or {}),
    }
    lines = []
    for name, value in fields.items():
        if name != "Title":
            lines.append(name)
        lines.extend(f"{part:<88}2026-01-01  Current" for part in textwrap.wrap(value, width=88))
    readme = (f"Series ID: {series}\nOutput Format: Observations by Vintage Date, All Observations\n"
              + f"{'Title':<88}Real-Time Start  Real-Time End\n" + "\n".join(lines)
              + "\n----------\nVintage Dates Specified:\n----------\n" + "\n".join(dates) + "\n----------\n")
    content = ("observation_date," + ",".join(f"{series}_{day.replace('-', '')}" for day in DATES) + "\n"
               + "\n".join(rows if rows is not None else ["2026-01-01,2500,2501", "2026-04-01,2600,2599"]) + "\n")
    path = tmp_path/"quarter.zip"
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{dates[0]}.csv", content)
        if extra:
            output.writestr("unexpected.csv", "Do not ignore changed archive members.")
    return path


@pytest.mark.parametrize("series", list(SERIES))
def test_native_nominal_quarterly_saar_levels_revisions_and_wrapped_current_title_are_preserved(tmp_path, series):
    frame = parse_archive(archive(tmp_path, series=series), series, DATES)
    assert frame.iloc[0].tolist() == [2500, 2501]
    assert frame.iloc[-1].tolist() == [2600, 2599]
    assert request_fields(series, DATES)["form[units]"] == "lin"
    assert request_fields(series, DATES)["form[obs_end_date]"] == "2026-04-01"


@pytest.mark.parametrize("field,value", [
    ("Title", "Corporate Profits After Tax (without IVA and CCAdj)"),
    ("Source", "A different agency"), ("Release", "Corporate Company Filings"),
    ("Units", "Percent"), ("Frequency", "Monthly"),
    ("Seasonal Adjustment", "Not Seasonally Adjusted"),
])
def test_wrong_scoped_fields_or_unadjusted_profit_alias_cannot_pass(tmp_path, field, value):
    with pytest.raises(QuantError, match="metadata"):
        parse_archive(archive(tmp_path, metadata={field: value}), "CPATAX", DATES)


def test_negative_aggregate_profits_are_not_rejected_by_a_fitted_source_sign_gate(tmp_path):
    frame = parse_archive(archive(tmp_path, rows=["2020-04-01,-10,20"]), "CPATAX", DATES)
    assert frame.iloc[0, 0] == -10
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, series="GDP", rows=["2020-04-01,0,20"]), "GDP", DATES)


@pytest.mark.parametrize("problem", [
    "dates", "duplicate", "not_quarter_start", "not_period_start", "future", "nonfinite",
    "outside_range", "after_range", "extra", "missing_snapshot", "empty",
])
def test_invalid_quarterly_archives_are_not_filled_or_shortened(tmp_path, problem):
    options = {}
    if problem == "dates":
        options["dates"] = ["2026-08-28", "2026-09-02"]
    elif problem == "duplicate":
        options["rows"] = ["2026-01-01,2500,2501", "2026-01-01,2500,2501"]
    elif problem == "not_quarter_start":
        options["rows"] = ["2026-02-01,2500,2501"]
    elif problem == "not_period_start":
        options["rows"] = ["2026-01-02,2500,2501"]
    elif problem == "future":
        options["rows"] = ["2026-10-01,2500,2501"]
    elif problem == "nonfinite":
        options["rows"] = ["2026-01-01,2500,inf"]
    elif problem == "outside_range":
        options["rows"] = ["2013-10-01,2500,2501"]
    elif problem == "after_range":
        options["rows"] = ["2026-07-01,2500,2501"]
    elif problem == "extra":
        options["extra"] = True
    elif problem == "empty":
        options["rows"] = []
    else:
        options["rows"] = ["2026-01-01,,."]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **options), "CPATAX", DATES)


@pytest.mark.parametrize("dates", [DATES[::-1], [], ["2026-08-28"]*41, ["bad-date"], ["2026-02-30"]])
def test_original_bounded_date_request_helper_is_not_relaxed(dates):
    with pytest.raises(QuantError):
        request_fields("CPATAX", dates)


def test_series_alias_or_growth_rate_is_not_the_declared_nominal_level_adapter():
    with pytest.raises(QuantError):
        request_fields("CP", DATES)
    with pytest.raises(QuantError):
        request_fields("GDPC1", DATES)


def test_quarter_end_not_quarter_start_or_month_end_defines_age_and_release_buffer():
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    value, audit = select(pd.Series([2600.0], index=pd.to_datetime(["2026-04-01"])), query, cutoff)
    assert value == 2600
    assert audit["completed_period_end"] == "2026-06-30"
    assert audit["completed_period_age_calendar_days"] == 62
    assert audit["vintage_cutoff"] == "2026-08-27"
    assert audit["quarter_label_claimed_first_release"] is False


def test_exact186day_boundary_is_a_new_quarterly_contract_not_a_monthly_age_relaxation():
    observed = pd.Series([2500.0], index=pd.to_datetime(["2025-01-01"]))
    query = pd.Timestamp("2025-10-03")
    assert select(observed, query, str(previous_session(previous_session(query)).date()))[1]["completed_period_age_calendar_days"] == 186
    query = pd.Timestamp("2025-10-06")
    with pytest.raises(QuantError, match="freshness"):
        select(observed, query, str(previous_session(previous_session(query)).date()))


@pytest.mark.parametrize("problem", ["empty", "future", "incomplete_quarter", "wrong_cutoff", "string",
                                   "bool", "complex", "weekend_query"])
def test_unreleased_or_unverified_quarter_data_never_becomes_a_safe_signal(problem):
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    observed = pd.Series([2500.0], index=pd.to_datetime(["2026-04-01"]))
    if problem == "empty":
        observed[:] = np.nan
    elif problem == "future":
        observed.index = pd.to_datetime(["2026-10-01"])
    elif problem == "incomplete_quarter":
        observed.index = pd.to_datetime(["2026-07-01"])
    elif problem == "wrong_cutoff":
        cutoff = "2026-08-28"
    elif problem == "string":
        observed = observed.astype(str)
    elif problem == "bool":
        observed = observed.astype(bool)
    elif problem == "complex":
        observed = observed.astype(complex)
    else:
        query = pd.Timestamp("2026-08-29")
    with pytest.raises(QuantError):
        select(observed, query, cutoff)


def snapshots(index):
    _, dates = cohort(index)
    quarters = pd.date_range("2014-01-01", "2026-04-01", freq="QS")
    return {
        name: {cutoff: pd.Series(2500.0, index=quarters[quarters+pd.offsets.QuarterEnd(startingMonth=3) <= pd.Timestamp(cutoff)])
               for cutoff in dates}
        for name in SERIES
    }


def test_all450independent_query_snapshots_and_original2805market_sessions_remain_required():
    index = sessions("2015-08-10", "2026-10-05")
    values = snapshots(index)
    selected = query_readiness(index, values)
    assert len(selected) == 450
    assert selected["series"].value_counts().to_dict() == {"CPATAX": 225, "GDP": 225}
    values["GDP"].pop(next(iter(values["GDP"])))
    with pytest.raises(QuantError, match="450"):
        query_readiness(index, values)


def test_source_does_not_force_independently_latest_quarters_to_match_or_compute_a_margin():
    index = sessions("2015-08-10", "2026-10-05")
    values = snapshots(index)
    cutoff = "2026-05-27"
    if cutoff not in values["CPATAX"]:
        cutoff = next(day for day in values["CPATAX"] if day.startswith("2026-05"))
    values["CPATAX"][cutoff] = values["CPATAX"][cutoff].iloc[:-1]
    selected = query_readiness(index, values)
    rows = selected.loc[selected["vintage_cutoff"].eq(cutoff)]
    assert rows["observation_quarter"].nunique() == 2
    assert "profit_margin" not in selected and "growth_signal" not in selected


@pytest.mark.parametrize("field,value", [
    ("maximum_completed_period_age_days", 279), ("source_observation_end", "2026-10-05"),
    ("vintage_cutoff", "previous_nyse_session"), ("query_sessions_per_series", 224),
    ("source_definitions", {}), ("new_registered_factor_definitions", 1),
    ("new_registered_candidates", 2), ("source_ratios_states_targets_or_returns_computed", True),
    ("new_independent_factor_families_claimed", True), ("existing_W42_or_forward_rules_changed", True),
    ("investment_objective_verified", True), ("order_authority", True),
])
def test_quarterly_source_scope_clock_and_no_portfolio_boundary_cannot_be_relabelled(field, value):
    policy = read_json(Path(__file__).parents[1]/"config/profit-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_frozen_policy_validates_original_market_asof_and_full_query_grid():
    policy = read_json(Path(__file__).parents[1]/"config/profit-inputs.json")
    validate_policy(policy)
    assert policy["as_of"] == "2026-10-05"
    assert len(cohort(sessions(policy["data_start"], policy["as_of"]))[1]) == 225
