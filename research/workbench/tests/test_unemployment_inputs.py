from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.activity_inputs import cohort
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.unemployment_inputs import (
    parse_archive,
    query_readiness,
    request_fields,
    select,
    validate_policy,
)

DATES = ["2026-08-27", "2026-09-01"]


def archive(tmp_path, *, rows=None, fields=None, dates=DATES, declared=None, extra=False,
            title_change="2019-09-06"):
    metadata = {
        "Title": "Unemployment Rate", "Source": "U.S. Bureau of Labor Statistics",
        "Release": "Employment Situation", "Units": "Percent", "Frequency": "Monthly",
        "Seasonal Adjustment": "Seasonally Adjusted", "Notes": "Household survey. LNS14000000",
        **(fields or {}),
    }
    lines = [f"{'Title':<88}Real-Time Start  Real-Time End"]
    for key, value in metadata.items():
        if key == "Title" and value == "Unemployment Rate":
            if dates[0] <= "2019-09-05":
                lines.append(f"{'Civilian Unemployment Rate':<88}1960-03-15  2019-09-05")
            if dates[-1] >= "2019-09-06":
                lines.append(f"{value:<88}{title_change}  Current")
        else:
            if key != "Title":
                lines.append(key)
            lines.append(f"{value:<88}2026-01-01  Current")
    readme = (
        "Series ID: UNRATE\nOutput Format: Observations by Vintage Date, All Observations\n"
        + "\n".join(lines) + "\n----------\nVintage Dates Specified:\n----------\n"
        + "\n".join(declared if declared is not None else dates) + "\n----------\n"
    )
    content = "observation_date," + ",".join(f"UNRATE_{day.replace('-', '')}" for day in dates) + "\n"
    content += "\n".join(rows if rows is not None else ["2026-06-01,4.1,4.2", "2026-07-01,4.2,4.3"]) + "\n"
    path = tmp_path / "unemployment.zip"
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{(declared if declared is not None else dates)[0]}.csv", content)
        if extra:
            output.writestr("unexpected.csv", "Do not ignore extra source members.")
    return path


def test_native_percent_revisions_remain_separate_in_their_own_snapshots(tmp_path):
    frame = parse_archive(archive(tmp_path), DATES)
    assert frame.iloc[0].tolist() == [4.1, 4.2]
    assert request_fields(DATES)["form[units]"] == "lin"
    assert request_fields(DATES)["form[obs_end_date]"] == "2026-09-01"


def test_real_title_history_is_metadata_not_a_series_or_numeric_revision(tmp_path):
    frame = parse_archive(archive(tmp_path), DATES)
    assert frame.columns.tolist() == DATES
    with pytest.raises(QuantError, match="title interval"):
        parse_archive(archive(tmp_path, title_change="2019-09-05"), DATES)
    with pytest.raises(QuantError, match="identity"):
        parse_archive(archive(tmp_path, fields={"Notes": "An unrelated population."}), DATES)


@pytest.mark.parametrize("dates", [
    ["2015-08-27", "2017-08-29"], ["2019-09-04", "2019-09-06"],
    ["2026-08-27", "2026-09-01"],
])
def test_native_title_history_is_scoped_to_the_requested_cohort_without_aliasing(tmp_path, dates):
    frame = parse_archive(archive(tmp_path, dates=dates, rows=["2014-01-01,4.1,4.2"]), dates)
    assert frame.iloc[0].tolist() == [4.1, 4.2]


@pytest.mark.parametrize("key,value", [
    ("Title", "Real-time Sahm Rule Recession Indicator"), ("Source", "A different source"),
    ("Release", "Unemployment Insurance Weekly Claims"), ("Units", "Percentage Points"),
    ("Frequency", "Weekly"), ("Seasonal Adjustment", "Not Seasonally Adjusted"),
])
def test_other_labor_definitions_units_or_releases_cannot_pass(tmp_path, key, value):
    with pytest.raises(QuantError, match="identity|title interval"):
        parse_archive(archive(tmp_path, fields={key: value}), DATES)


@pytest.mark.parametrize("rows", [
    [], ["2026-07-01,.,."], ["2026-07-01,4.2,inf"], ["2026-07-01,-0.1,4.2"],
    ["2026-07-01,4.2,100.1"], ["2026-08-01,4.2,4.3"], ["2026-07-02,4.2,4.3"],
    ["2013-12-01,4.2,4.3"], ["2026-10-01,4.2,4.3"],
    ["2026-07-01,4.2,4.3", "2026-07-01,4.2,4.3"],
])
def test_invalid_missing_future_or_duplicate_months_are_not_repaired(tmp_path, rows):
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, rows=rows), DATES)


def test_changed_vintage_list_and_extra_archive_members_cannot_pass(tmp_path):
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, declared=["2026-08-27", "2026-09-02"]), DATES)
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, extra=True), DATES)


@pytest.mark.parametrize("dates", [[], DATES[::-1], [DATES[0]] * 41, ["2026-02-30"], ["wrong-date"]])
def test_existing_bounded_vintage_request_validation_is_reused(dates):
    with pytest.raises(QuantError):
        request_fields(dates)


def observed():
    return pd.Series(4.2, index=pd.date_range("2024-01-01", "2026-07-01", freq="MS"))


def test_source_requires15consecutive_same_vintage_months_but_does_not_compute_a_rule():
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    value, audit = select(observed(), query, cutoff)
    assert value == 4.2
    assert audit["completed_period_end"] == "2026-07-31"
    assert audit["completed_period_age_calendar_days"] == 31
    assert audit["history_start_month"] == "2025-05-01"
    assert audit["consecutive_history_months"] == 15
    assert audit["rolling_means_or_recession_state_computed"] is False
    assert audit["month_label_claimed_first_release"] is False


@pytest.mark.parametrize("problem", ["gap", "missing_value", "short", "stale", "future", "bool",
                                   "complex", "string", "wrong_cutoff", "nonfinite", "empty", "range"])
def test_unknown_history_or_invalid_series_never_defaults_to_a_safe_regime(problem):
    values = observed()
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    if problem == "gap":
        values = values.drop(pd.Timestamp("2025-07-01"))
    elif problem == "missing_value":
        values.loc["2025-07-01"] = np.nan
    elif problem == "short":
        values = values.iloc[-14:]
    elif problem == "stale":
        values = values.loc[:"2026-04-01"]
    elif problem == "future":
        values.loc[pd.Timestamp("2026-08-01")] = 4.2
    elif problem == "wrong_cutoff":
        cutoff = "2026-08-28"
    elif problem == "nonfinite":
        values.iloc[-1] = np.inf
    elif problem == "empty":
        values[:] = np.nan
    elif problem == "range":
        values.iloc[-1] = -1
    else:
        values = values.astype({"bool": bool, "complex": complex, "string": str}[problem])
    with pytest.raises(QuantError):
        select(values, query, cutoff)


def test_full225queries_and_two_prior_nyse_cutoffs_are_preserved():
    index = sessions("2015-08-10", "2026-10-05")
    _, dates = cohort(index)
    months = pd.date_range("2014-01-01", "2026-09-01", freq="MS")
    vintages = {
        cutoff: pd.Series(4.2, index=months[months + pd.offsets.MonthEnd(0) <= pd.Timestamp(cutoff)])
        for cutoff in dates
    }
    frame = query_readiness(index, vintages)
    assert len(frame) == 225
    assert frame["consecutive_history_months"].eq(15).all()
    assert "sahm_gap" not in frame and "risk_off" not in frame
    vintages.pop(dates[0])
    with pytest.raises(QuantError, match="225"):
        query_readiness(index, vintages)


@pytest.mark.parametrize("key,value", [
    ("series", "SAHMREALTIME"), ("required_consecutive_months_per_query", 14),
    ("maximum_completed_period_age_days", 93), ("vintage_cutoff", "previous_nyse_session"),
    ("query_sessions", 224), ("source_observation_end", "2026-10-05"),
    ("source_statistics_states_targets_or_returns_computed", True),
    ("new_registered_factor_definitions", 1), ("new_registered_candidates", 2),
    ("new_strategy_evaluations", 1), ("existing_W42_or_forward_rules_changed", True),
    ("new_independent_factor_families_claimed", True), ("investment_objective_verified", True),
    ("order_authority", True), ("primary_definitions", {}),
])
def test_frozen_source_scope_and_authority_cannot_be_relabelled(key, value):
    policy = read_json(Path(__file__).parents[1] / "config/unemployment-inputs.json")
    policy[key] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_frozen_source_contract_retains_the_original_formal_market_asof():
    validate_policy(read_json(Path(__file__).parents[1] / "config/unemployment-inputs.json"))
