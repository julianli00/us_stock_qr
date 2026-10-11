from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.activity_inputs import SERIES, cohort, parse_archive, query_readiness, request_fields, select, validate_policy
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, series="CFNAIMA3", rows=None, dates=DATES, title=None, extra=False):
    path = tmp_path / "activity.zip"
    readme = (
        f"Series ID: {series}\n{title or SERIES[series]}\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Federal Reserve Bank of Chicago\nChicago Fed National Activity Index\n"
        "Index\nMonthly\nNot Seasonally Adjusted\nVintage Dates Specified:\n----------\n"
        + "\n".join(dates) + "\n----------\n"
    )
    content = (
        "observation_date," + ",".join(f"{series}_{day.replace('-', '')}" for day in DATES) + "\n"
        + "\n".join(rows or ["2026-06-01,-0.25,-0.24", "2026-07-01,-0.31,-0.30"]) + "\n"
    )
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{dates[0]}.csv", content)
        if extra:
            output.writestr("other.csv", "do not ignore unexpected source members")
    return path


@pytest.mark.parametrize("series", list(SERIES))
def test_actual_monthly_levels_and_revisions_are_not_prices_returns_or_recession_flags(tmp_path, series):
    frame = parse_archive(archive(tmp_path, series=series), series, DATES)
    assert frame.iloc[0, 0] == -0.25
    assert frame.iloc[0, 1] == -0.24
    assert request_fields(series, DATES)["form[units]"] == "lin"
    assert request_fields(series, DATES)["form[file_type]"] == "2"


def test_first_of_month_labels_may_be_weekends_not_daily_business_dates(tmp_path):
    frame = parse_archive(archive(tmp_path, rows=["2022-01-01,-0.2,-0.1"]), "CFNAIMA3", DATES)
    assert frame.index[0].weekday() == 5


@pytest.mark.parametrize("problem", ["title", "dates", "future", "duplicate", "not_month_start",
                                   "nonfinite", "outside_range", "extra", "missing_snapshot"])
def test_unverified_or_missing_monthly_archives_fail_without_conversion_or_repair(tmp_path, problem):
    options = {}
    if problem == "title":
        options["title"] = "Chicago Fed Financial Conditions"
    elif problem == "dates":
        options["dates"] = ["2026-08-28", "2026-09-02"]
    elif problem == "future":
        options["rows"] = ["2026-09-01,-0.2,-0.2"]
    elif problem == "duplicate":
        options["rows"] = ["2026-07-01,-0.2,-0.1", "2026-07-01,-0.2,-0.1"]
    elif problem == "not_month_start":
        options["rows"] = ["2026-07-02,-0.2,-0.1"]
    elif problem == "nonfinite":
        options["rows"] = ["2026-07-01,-0.2,inf"]
    elif problem == "outside_range":
        options["rows"] = ["2013-12-01,-0.2,-0.1"]
    elif problem == "extra":
        options["extra"] = True
    else:
        options["rows"] = ["2026-07-01,,."]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **options), "CFNAIMA3", DATES)


def test_weighted_diffusion_has_structural_unit_bounds_but_activity_magnitude_does_not(tmp_path):
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, series="CFNAIDIFF", rows=["2026-07-01,-1.1,0.1"]), "CFNAIDIFF", DATES)
    frame = parse_archive(archive(tmp_path, rows=["2020-04-01,-9.2,-9.3"]), "CFNAIMA3", DATES)
    assert frame.iloc[0, 0] == -9.2


@pytest.mark.parametrize("dates", [DATES[::-1], [], ["2026-08-28"]*41, ["bad-date"], ["2026-02-30"]])
def test_existing_advertised_date_limits_and_order_are_not_relaxed(dates):
    with pytest.raises(QuantError):
        request_fields("CFNAIMA3", dates)


def test_unknown_series_or_a_current_history_alias_is_not_an_activity_adapter():
    with pytest.raises(QuantError):
        request_fields("CFNAIDI", DATES)


def test_actual_monthly_observation_range_does_not_shorten_market_or_query_windows():
    policy = read_json(Path(__file__).parents[1] / "config/activity-inputs.json")
    validate_policy(policy)
    assert request_fields("CFNAIMA3", DATES)["form[obs_end_date]"] == "2026-08-01"
    assert policy["as_of"] == "2026-10-05"
    assert len(cohort(sessions(policy["data_start"], policy["as_of"]))[1]) == 225


def test_snapshot_lag_is_applied_to_actual_vintage_not_assumed_month_start_release():
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    observed = pd.Series([-0.25], index=pd.to_datetime(["2026-07-01"]))
    value, audit = select(observed, query, cutoff)
    assert value == -0.25
    assert audit["completed_period_end"] == "2026-07-31"
    assert audit["completed_period_age_calendar_days"] == 31
    assert audit["month_label_claimed_first_release"] is False
    assert audit["vintage_cutoff"] == "2026-08-27"


def test_exact_62day_boundary_is_measured_from_completed_month_end():
    observed = pd.Series([0.1], index=pd.to_datetime(["2026-06-01"]))
    query = pd.Timestamp("2026-08-31")
    assert select(observed, query, str(previous_session(previous_session(query)).date()))[1]["completed_period_age_calendar_days"] == 62
    query = pd.Timestamp("2026-09-01")
    with pytest.raises(QuantError, match="freshness"):
        select(observed, query, str(previous_session(previous_session(query)).date()))


@pytest.mark.parametrize("problem", ["empty", "future", "incomplete_period", "wrong_cutoff", "string", "weekend_query"])
def test_missing_unreleased_or_wrongly_timed_data_never_become_a_cash_default(problem):
    query = pd.Timestamp("2026-08-31")
    cutoff = str(previous_session(previous_session(query)).date())
    observed = pd.Series([0.1], index=pd.to_datetime(["2026-07-01"]))
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
    else:
        query = pd.Timestamp("2026-08-29")
    with pytest.raises(QuantError):
        select(observed, query, cutoff)


def vintages(index):
    _, dates = cohort(index)
    months = pd.date_range("2014-01-01", "2026-09-01", freq="MS")
    result = {}
    for name in SERIES:
        result[name] = {}
        for cutoff in dates:
            completed = months[months + pd.offsets.MonthEnd(0) <= pd.Timestamp(cutoff)]
            result[name][cutoff] = pd.Series(0.1, index=completed, name=name)
    return result


def test_all_450_series_query_snapshots_are_required_on_the_original_market_calendar():
    index = sessions("2015-08-10", "2026-10-05")
    snapshots = vintages(index)
    ready = query_readiness(index, snapshots)
    assert len(ready) == 450
    assert ready["series"].value_counts().to_dict() == {"CFNAIMA3":225, "CFNAIDIFF":225}
    assert ready["index_value"].eq(0.1).all()
    snapshots["CFNAIDIFF"].pop(next(iter(snapshots["CFNAIDIFF"])))
    with pytest.raises(QuantError, match="450"):
        query_readiness(index, snapshots)


def test_source_readiness_cannot_shorten_the_formal_market_context_or_replace_a_series():
    index = sessions("2015-08-10", "2026-10-05")
    snapshots = vintages(index)
    with pytest.raises(QuantError):
        query_readiness(index[1:], snapshots)
    snapshots["CFNAI"] = snapshots.pop("CFNAIMA3")
    with pytest.raises(QuantError):
        query_readiness(index, snapshots)


@pytest.mark.parametrize("field,value", [("maximum_completed_period_age_days", 93),
                                        ("source_observation_end", "2026-10-05"),
                                        ("vintage_cutoff", "previous_nyse_session"),
                                        ("query_sessions_per_series", 224),
                                        ("new_registered_factor_definitions", 1),
                                        ("new_registered_candidates", 2),
                                        ("strategy_states_targets_or_returns_computed", True),
                                        ("new_independent_factor_families_claimed", True),
                                        ("existing_W42_or_forward_rules_changed", True),
                                        ("investment_objective_verified", True),
                                        ("order_authority", True)])
def test_source_only_freshness_scope_and_existing_research_cannot_be_relabelled(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/activity-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
