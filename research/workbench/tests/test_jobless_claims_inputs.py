from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.jobless_claims_inputs import (
    INITIAL_MEMBER,
    inspect_queries,
    needed_vintage_dates,
    parse_archive,
    parse_initial_archive,
    query_readiness,
    request_fields,
    validate_policy,
)
from us_quant.storage import read_json

DATES = ["2026-09-24", "2026-10-01"]


def archive(tmp_path, rows=None, *, initial=False, series="ICSA", dates=DATES):
    path = tmp_path / "claims.zip"
    output = (
        "Observations, Initial Release Only"
        if initial
        else "Observations by Vintage Date, All Observations"
    )
    readme = (
        f"Series ID: {series}\nInitial Claims\nOutput Format: {output}\n"
        "U.S. Employment and Training Administration\n"
        "Unemployment Insurance Weekly Claims Report\nNumber\n"
        "Weekly, Ending Saturday\nSeasonally Adjusted\n"
        "Vintage Dates Specified:\n----------\n" + "\n".join(dates) + "\n----------\n"
    )
    if initial:
        header = "period_start_date,ICSA,realtime_start_date"
        default = ["2026-09-19,200000,2026-09-24", "2026-09-26,201000,2026-10-01"]
    else:
        header = "observation_date,ICSA_20260924,ICSA_20261001"
        default = ["2026-09-19,200000,199000", "2026-09-26,,201000"]
    with ZipFile(path, "w") as result:
        result.writestr("README.txt", readme)
        result.writestr(
            INITIAL_MEMBER if initial else f"vintages_starting_{dates[0]}.csv",
            header + "\n" + "\n".join(default if rows is None else rows) + "\n",
        )
    return path


def test_claims_keep_native_counts_historical_revisions_and_unreleased_week_missing(tmp_path):
    parsed = parse_archive(archive(tmp_path), DATES)
    assert parsed.iloc[0, 0] == 200000
    assert parsed.iloc[0, 1] == 199000
    assert np.isnan(parsed.iloc[1, 0])
    assert parsed.iloc[1, 1] == 201000
    fields = request_fields(DATES)
    assert fields["form[units]"] == "lin"
    assert fields["form[obs_end_date]"] == "2026-10-03"


def test_initial_releases_have_separate_calendar_dates_not_week_ending_availability(tmp_path):
    result = parse_initial_archive(archive(tmp_path, initial=True), DATES)
    assert result.index[0] == pd.Timestamp("2026-09-19")
    assert result.iloc[0]["first_release_date"] == pd.Timestamp("2026-09-24")
    assert result.iloc[0]["initial_claims"] == 200000
    assert request_fields(DATES, initial_release=True)["form[file_type]"] == "4"


@pytest.mark.parametrize(
    "problem",
    [
        "series",
        "date_header",
        "duplicate",
        "weekday",
        "future",
        "negative",
        "fraction",
        "inf",
    ],
)
def test_invalid_or_changed_snapshot_is_explicitly_rejected_not_repaired(tmp_path, problem):
    kwargs = {}
    if problem == "series":
        kwargs["series"] = "IC4WSA"
    elif problem == "date_header":
        kwargs["dates"] = ["2026-09-24", "2026-10-02"]
    elif problem == "duplicate":
        kwargs["rows"] = ["2026-09-19,200000,199000", "2026-09-19,200000,199000"]
    elif problem == "weekday":
        kwargs["rows"] = ["2026-09-18,200000,199000"]
    elif problem == "future":
        kwargs["rows"] = ["2026-09-26,200000,199000"]
    else:
        value = {"negative": "-1", "fraction": "200000.5", "inf": "inf"}[problem]
        kwargs["rows"] = [f"2026-09-19,{value},199000"]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **kwargs), DATES)


@pytest.mark.parametrize(
    "rows",
    [
        ["2026-09-19,200000,2026-09-19"],
        ["2026-09-19,200000,2026-09-25"],
        ["2026-09-19,,2026-09-24"],
        ["2026-09-18,200000,2026-09-24"],
    ],
)
def test_initial_release_metadata_cannot_be_inferred_or_forged(tmp_path, rows):
    with pytest.raises(QuantError):
        parse_initial_archive(archive(tmp_path, rows, initial=True), DATES)


@pytest.mark.parametrize(
    "dates",
    [
        [],
        DATES[::-1],
        ["2026-09-24"] * 41,
        ["2026-02-30"],
        [None],
        None,
    ],
)
def test_public_form_limits_and_calendar_dates_are_not_relaxed(dates):
    with pytest.raises(QuantError):
        request_fields(dates)


def test_http200_html_is_not_a_successful_archive(tmp_path):
    path = tmp_path / "claims.zip"
    path.write_text("<html>form validation error</html>")
    with pytest.raises(QuantError, match="HTTP200 HTML"):
        parse_archive(path, DATES)


@pytest.fixture
def inputs(request):
    release_days = getattr(request, "param", 4)
    index = sessions("2015-08-10", "2026-10-05")
    weekly = pd.date_range("2014-01-04", "2026-10-03", freq="W-SAT")
    first = pd.DataFrame(
        {
            "initial_claims": 250000.0,
            "first_release_date": weekly + pd.Timedelta(days=release_days),
        },
        index=weekly,
    )
    vintages = {
        day: pd.Series(
            250000.0,
            index=weekly[first["first_release_date"] <= pd.Timestamp(day)],
        )
        for day in needed_vintage_dates(index)
    }
    return index, vintages, first


def test_all_months_and_reference_queries_use_two_earlier_sessions_and_actual_first_release(inputs):
    index, vintages, first = inputs
    ready = query_readiness(index, vintages, first)
    assert len(ready) == 225
    assert ready["initial_claims"].eq(250000).all()
    assert ready["information_delay_sessions"].eq(2).all()
    assert ready["age_calendar_days"].max() <= 14
    assert (ready["first_release_date"] <= ready["vintage_cutoff"]).all()
    query = "2026-09-30"
    assert ready.loc[query, "vintage_cutoff"] == "2026-09-28"
    assert ready.loc[query, "observation_date"] == "2026-09-19"
    assert ready.loc[query, "first_release_date"] == "2026-09-23"
    assert ready["source_ready"].all()
    assert ready["blocker"].isna().all()


@pytest.mark.parametrize("inputs", [5], indirect=True)
def test_holiday_age_blocks_remain_explicit_in_the_complete_query_grid(inputs):
    index, vintages, first = inputs
    inspected = inspect_queries(index, vintages, first)
    blocked = inspected.loc[~inspected["source_ready"]]
    assert len(inspected) == 225
    assert list(blocked.index) == [
        "2015-11-30",
        "2017-11-27",
        "2018-04-02",
        "2018-11-26",
        "2020-11-30",
        "2025-12-29",
    ]
    assert blocked["age_calendar_days"].eq(16).all()
    assert blocked["blocker"].eq("observation_age_exceeds_frozen_fourteen_days").all()
    assert blocked["initial_claims"].eq(250000).all()
    with pytest.raises(QuantError, match="readiness blocked"):
        query_readiness(index, vintages, first)


def test_later_revisions_and_initial_release_values_do_not_replace_earlier_snapshots(inputs):
    index, vintages, first = inputs
    before = query_readiness(index, vintages, first)
    first["initial_claims"] += 1000
    for cutoff in vintages:
        if cutoff >= "2021-01-01":
            vintages[cutoff] *= 2
    after = query_readiness(index, vintages, first)
    earlier = before.index < "2021-01-01"
    assert before.loc[earlier, "initial_claims"].equals(after.loc[earlier, "initial_claims"])
    assert after["first_released_initial_claims"].eq(251000).all()
    assert after.loc[~earlier, "initial_claims"].eq(500000).all()


@pytest.mark.parametrize("problem", ["initial_object", "snapshot_object", "missing_release"])
def test_missing_or_malformed_source_objects_are_explicit_errors(inputs, problem):
    index, vintages, first = inputs
    if problem == "initial_object":
        first = None
    elif problem == "snapshot_object":
        vintages[next(iter(vintages))] = None
    else:
        first = first.drop(pd.Timestamp("2026-09-19"))
    with pytest.raises(QuantError):
        query_readiness(index, vintages, first)


@pytest.mark.parametrize("problem", ["missing", "stale", "future", "index", "first_release"])
def test_missing_or_unreleased_inputs_do_not_become_cash_or_a_shorter_window(inputs, problem):
    index, vintages, first = inputs
    cutoff = "2026-09-28"
    if problem == "missing":
        vintages.pop(cutoff)
    elif problem == "stale":
        vintages[cutoff] = pd.Series([250000], index=pd.to_datetime(["2026-09-05"]))
    elif problem == "future":
        vintages[cutoff] = pd.Series([250000], index=pd.to_datetime(["2026-10-03"]))
    elif problem == "index":
        index = index[:-1]
    else:
        first.loc[pd.Timestamp("2026-09-19"), "first_release_date"] = pd.Timestamp("2026-10-01")
    with pytest.raises(QuantError):
        query_readiness(index, vintages, first)


@pytest.mark.parametrize(
    "field,value",
    [
        ("information_delay_sessions", 1),
        ("maximum_observation_age_days", 21),
        ("required_vintage_dates", 224),
        ("first_release_date_validation_required", False),
        ("first_release_date_validation_required", 1),
        ("new_registered_factor_definitions", 1),
        ("strategy_states_or_returns_computed", True),
        ("new_factor_family_independence_claimed", True),
        ("existing_queued_proposals_replaced", True),
        ("forward_rules_changed", True),
        ("order_authority", True),
        ("order_authority", 0),
    ],
)
def test_source_readiness_cannot_become_a_factor_or_early_strategy_result(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/jobless-claims-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
