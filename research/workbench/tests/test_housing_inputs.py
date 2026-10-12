from __future__ import annotations

import textwrap
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.housing_inputs import SERIES, cohort, parse_archive, query_readiness, request_fields, select, validate_policy
from us_quant.storage import read_json

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, series="PERMIT", rows=None, dates=DATES, metadata=None, extra=False,
            real_time_end="2026-09-01"):
    fields = {
        "Title": SERIES[series],
        "Source": "U.S. Census Bureau U.S. Department of Housing and Urban Development",
        "Release": "New Residential Construction", "Units": "Thousands of Units",
        "Frequency": "Monthly", "Seasonal Adjustment": "Seasonally Adjusted Annual Rate",
        "Notes": "Period labels are not first publication times.",
        **(metadata or {}),
    }
    lines = []
    for name, value in fields.items():
        if name != "Title":
            lines.append(name)
        lines.extend(f"{part:<88}2026-01-01  {real_time_end}" for part in textwrap.wrap(value, width=88))
    readme = (
        f"Series ID: {series}\nOutput Format: Observations by Vintage Date, All Observations\n"
        + f"{'Title':<88}Real-Time Start  Real-Time End\n"
        + "\n".join(lines)
        + "\n----------\nVintage Dates Specified:\n----------\n"
        + "\n".join(dates) + "\n----------\n"
    )
    content = (
        "observation_date," + ",".join(f"{series}_{day.replace('-', '')}" for day in DATES) + "\n"
        + "\n".join(rows if rows is not None else ["2026-06-01,1500,1501", "2026-07-01,1600,1599"]) + "\n"
    )
    path = tmp_path / "housing.zip"
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{dates[0]}.csv", content)
        if extra:
            output.writestr("other.csv", "Unexpected member must not be ignored.")
    return path


@pytest.mark.parametrize("series", list(SERIES))
def test_native_monthly_units_saar_revisions_and_long_metadata_are_preserved(tmp_path, series):
    frame = parse_archive(archive(tmp_path, series=series), series, DATES)
    assert frame.iloc[0].tolist() == [1500, 1501]
    assert frame.iloc[-1].tolist() == [1600, 1599]
    assert request_fields(series, DATES)["form[units]"] == "lin"
    assert request_fields(series, DATES)["form[file_type]"] == "2"
    assert request_fields(series, DATES)["form[obs_end_date]"] == "2026-08-01"


@pytest.mark.parametrize("series", list(SERIES))
def test_actual_current_realtime_end_is_metadata_not_part_of_the_wrapped_native_title(tmp_path, series):
    frame = parse_archive(archive(tmp_path, series=series, real_time_end="Current"), series, DATES)
    assert frame.iloc[0].tolist() == [1500, 1501]


@pytest.mark.parametrize("field,value", [
    ("Title", "New Privately-Owned Housing Units Under Construction"),
    ("Source", "A different publisher"),
    ("Release", "Housing Prices"),
    ("Units", "Percent Thousands of Units"),
    ("Frequency", "Weekly Monthly"),
    ("Seasonal Adjustment", "Not Seasonally Adjusted"),
])
def test_metadata_terms_elsewhere_do_not_substantiate_a_wrong_scoped_field(tmp_path, field, value):
    with pytest.raises(QuantError, match="metadata"):
        parse_archive(archive(tmp_path, metadata={field: value}), "PERMIT", DATES)


@pytest.mark.parametrize("problem", [
    "dates", "future", "duplicate", "not_month_start", "negative", "nonfinite",
    "outside_range", "after_range", "extra", "missing_snapshot", "empty",
])
def test_invalid_archives_are_not_repaired_filled_or_relabelled_safe(tmp_path, problem):
    options = {}
    if problem == "dates":
        options["dates"] = ["2026-08-28", "2026-09-02"]
    elif problem == "future":
        options["rows"] = ["2026-09-01,1500,1501"]
    elif problem == "duplicate":
        options["rows"] = ["2026-07-01,1500,1501", "2026-07-01,1500,1501"]
    elif problem == "not_month_start":
        options["rows"] = ["2026-07-02,1500,1501"]
    elif problem == "negative":
        options["rows"] = ["2026-07-01,-1,1501"]
    elif problem == "nonfinite":
        options["rows"] = ["2026-07-01,1500,inf"]
    elif problem == "outside_range":
        options["rows"] = ["2013-12-01,1500,1501"]
    elif problem == "after_range":
        options["rows"] = ["2026-09-01,.,1501"]
    elif problem == "extra":
        options["extra"] = True
    elif problem == "empty":
        options["rows"] = []
    else:
        options["rows"] = ["2026-07-01,,."]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **options), "PERMIT", DATES)


def test_zero_unit_count_is_not_rejected_by_an_invented_positive_threshold(tmp_path):
    frame = parse_archive(archive(tmp_path, rows=["2022-01-01,0,1501"]), "PERMIT", DATES)
    assert frame.iloc[0, 0] == 0
    assert frame.index[0].weekday() == 5


def test_html_http200_body_is_not_a_housing_vintage(tmp_path):
    path = tmp_path / "housing.zip"
    path.write_text("<html>Data is unavailable</html>")
    with pytest.raises(QuantError, match="readable ZIP"):
        parse_archive(path, "PERMIT", DATES)


@pytest.mark.parametrize("dates", [DATES[::-1], [], ["2026-08-28"]*41, ["bad-date"], ["2026-02-30"]])
def test_advertised_request_date_order_bounds_and_calendar_validation_are_reused(dates):
    with pytest.raises(QuantError):
        request_fields("PERMIT", dates)


def test_current_history_or_another_housing_series_is_not_an_authorized_adapter():
    with pytest.raises(QuantError):
        request_fields("PERMIT1", DATES)


def test_actual_earlier_snapshot_is_not_month_start_release_or_download_availability():
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    observed = pd.Series([1600.0], index=pd.to_datetime(["2026-07-01"]))
    value, audit = select(observed, query, cutoff)
    assert value == 1600
    assert audit["vintage_cutoff"] == "2026-08-27"
    assert audit["completed_period_end"] == "2026-07-31"
    assert audit["completed_period_age_calendar_days"] == 31
    assert audit["month_label_claimed_first_release"] is False


def test_62day_freshness_boundary_reuses_the_prior_completed_month_contract():
    observed = pd.Series([1500.0], index=pd.to_datetime(["2026-06-01"]))
    query = pd.Timestamp("2026-08-31")
    assert select(observed, query, str(previous_session(previous_session(query)).date()))[1]["completed_period_age_calendar_days"] == 62
    query = pd.Timestamp("2026-09-01")
    with pytest.raises(QuantError, match="freshness"):
        select(observed, query, str(previous_session(previous_session(query)).date()))


@pytest.mark.parametrize("problem", ["empty", "future", "incomplete_period", "wrong_cutoff", "string",
                                   "negative", "weekend_query"])
def test_missing_unreleased_or_wrongly_timed_source_never_becomes_a_default_risk_state(problem):
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    observed = pd.Series([1500.0], index=pd.to_datetime(["2026-07-01"]))
    if problem == "empty":
        observed[:] = np.nan
    elif problem == "future":
        observed.index = pd.to_datetime(["2026-09-01"])
    elif problem == "incomplete_period":
        observed.index = pd.to_datetime(["2026-08-01"])
    elif problem == "wrong_cutoff":
        cutoff = "2026-08-28"
    elif problem == "string":
        observed = observed.astype(str)
    elif problem == "negative":
        observed[:] = -1
    else:
        query = pd.Timestamp("2026-08-29")
    with pytest.raises(QuantError):
        select(observed, query, cutoff)


def vintages(index):
    _, dates = cohort(index)
    months = pd.date_range("2014-01-01", "2026-08-01", freq="MS")
    return {
        name: {
            cutoff: pd.Series(1500.0, index=months[months+pd.offsets.MonthEnd(0) <= pd.Timestamp(cutoff)])
            for cutoff in dates
        }
        for name in SERIES
    }


def test_all450snapshots_and_original2805sessions_and131pairs_are_required():
    index = sessions("2015-08-10", "2026-10-05")
    snapshots = vintages(index)
    frame = query_readiness(index, snapshots)
    assert len(frame) == 450
    assert frame["series"].value_counts().to_dict() == {"PERMIT": 225, "HOUST": 225}
    assert frame["thousands_of_units_saar"].eq(1500).all()
    snapshots["HOUST"].pop(next(iter(snapshots["HOUST"])))
    with pytest.raises(QuantError, match="450"):
        query_readiness(index, snapshots)


def test_source_readiness_cannot_shorten_calendar_or_substitute_a_series():
    index = sessions("2015-08-10", "2026-10-05")
    snapshots = vintages(index)
    with pytest.raises(QuantError):
        query_readiness(index[1:], snapshots)
    snapshots["PERMIT1"] = snapshots.pop("PERMIT")
    with pytest.raises(QuantError):
        query_readiness(index, snapshots)


@pytest.mark.parametrize("field,value", [
    ("maximum_completed_period_age_days", 93), ("source_observation_end", "2026-10-05"),
    ("vintage_cutoff", "previous_nyse_session"), ("query_sessions_per_series", 224),
    ("source_definitions", {}), ("new_registered_factor_definitions", 1),
    ("new_registered_candidates", 2), ("strategy_states_targets_or_returns_computed", True),
    ("new_independent_factor_families_claimed", True), ("existing_W42_or_forward_rules_changed", True),
    ("investment_objective_verified", True), ("order_authority", True),
])
def test_source_only_freshness_scope_and_existing_research_cannot_be_relabelled(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/housing-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_frozen_policy_validates_without_changing_original_observation_or_market_range():
    policy = read_json(Path(__file__).parents[1] / "config/housing-inputs.json")
    validate_policy(policy)
    assert policy["as_of"] == "2026-10-05"
    assert len(cohort(sessions(policy["data_start"], policy["as_of"]))[1]) == 225
