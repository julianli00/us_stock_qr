from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions, previous_session
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.yield_curve_inputs import SERIES, archive_metadata, cohort, parse_archive, query_readiness, request_fields, select, validate_policy

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, series="T10Y3M", rows=None, dates=DATES, title=None, extra=False, wrapped=False):
    path = tmp_path / "spread.zip"
    title = title or SERIES[series]
    head, continuation = title.rsplit(" ", 1) if wrapped else (title, "")
    readme = (
        f"Series ID: {series}\nOutput Format: Observations by Vintage Date, All Observations\n"
        + f"{'Title':<108}Real-Time Start  Real-Time End\n"
        + f"{head:<108}2014-01-27       Current\n"
        + (continuation + "\n" if continuation else "")
        + "Source\nFederal Reserve Bank of St. Louis\nRelease\nInterest Rate Spreads\n"
        "Units\nPercent\nFrequency\nDaily\nSeasonal Adjustment\nNot Seasonally Adjusted\n"
        "Notes\nNative constant-maturity spread\n----------\n"
        "Vintage Dates Specified:\n----------\n" + "\n".join(dates) + "\n----------\n"
    )
    content = (
        "observation_date," + ",".join(f"{series}_{day.replace('-', '')}" for day in DATES) + "\n"
        + "\n".join(rows or ["2026-08-28,-0.25,-0.24", "2026-08-31,,0.0"]) + "\n"
    )
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(f"vintages_starting_{dates[0]}.csv", content)
        if extra:
            output.writestr("unexpected.csv", "do not ignore")
    return path


@pytest.mark.parametrize("series", list(SERIES))
def test_negative_and_zero_spreads_keep_native_percentage_points_not_prices_or_returns(tmp_path, series):
    frame = parse_archive(archive(tmp_path, series=series), series, DATES)
    assert frame.iloc[0, 0] == -0.25
    assert frame.iloc[0, 1] == -0.24
    assert frame.iloc[1, 1] == 0.0
    assert pd.isna(frame.iloc[1, 0])
    assert request_fields(series, DATES)["form[units]"] == "lin"


@pytest.mark.parametrize("series", list(SERIES))
def test_actual_metadata_column_wrapping_preserves_exact_title_without_consuming_period_dates(tmp_path, series):
    path = archive(tmp_path, series=series, wrapped=True)
    with ZipFile(path) as source:
        metadata = archive_metadata(source.read("README.txt").decode())
    assert metadata["Title"] == SERIES[series]
    assert metadata["Source"] == "Federal Reserve Bank of St. Louis"
    assert "2014-01-27" not in metadata["Title"]
    assert parse_archive(path, series, DATES).iloc[0, 0] == -0.25


def test_unscoped_flat_metadata_cannot_substitute_for_the_actual_archive_table():
    with pytest.raises(QuantError, match="metadata table"):
        archive_metadata("Title\nT10Y3M\nSource\nFederal Reserve Bank of St. Louis\n")


@pytest.mark.parametrize("problem", ["title", "dates", "future", "duplicate", "weekend", "nonfinite",
                                   "outside", "extra", "missing"])
def test_invalid_or_unverified_vintage_archive_is_not_silently_repaired(tmp_path, problem):
    options = {}
    if problem == "title":
        options["title"] = "10-Year Inflation-Indexed Rate"
    elif problem == "dates":
        options["dates"] = ["2026-08-28", "2026-09-02"]
    elif problem == "future":
        options["rows"] = ["2026-08-31,0.1,0.1"]
    elif problem == "duplicate":
        options["rows"] = ["2026-08-28,0.1,0.1", "2026-08-28,0.1,0.1"]
    elif problem == "weekend":
        options["rows"] = ["2026-08-29,,0.1"]
    elif problem == "nonfinite":
        options["rows"] = ["2026-08-28,0.1,inf"]
    elif problem == "outside":
        options["rows"] = ["2013-12-31,0.1,0.1"]
    elif problem == "extra":
        options["extra"] = True
    else:
        options["rows"] = ["2026-08-28,.,."]
    with pytest.raises(QuantError):
        parse_archive(archive(tmp_path, **options), "T10Y3M", DATES)


@pytest.mark.parametrize("dates", [DATES[::-1], [], ["2026-08-28"]*41, ["bad-date"], ["2026-02-30"]])
def test_advertised_date_limits_order_and_calendar_are_unchanged(dates):
    with pytest.raises(QuantError):
        request_fields("T10Y3M", dates)


def test_other_maturities_or_recession_labels_cannot_substitute_for_declared_spreads():
    with pytest.raises(QuantError):
        request_fields("USREC", DATES)


def test_source_selection_retains_previous_vintage_and_second_strictly_later_nyse_lag():
    query = pd.Timestamp("2026-09-01")
    observed = pd.Series([0.1, -0.2, -0.3], index=pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"]))
    value, audit = select(observed, query, "2026-08-31")
    assert value == -0.2
    assert audit["observation_date"] == "2026-08-28"
    assert audit["assumed_available_session"] == "2026-09-01"
    assert audit["age_calendar_days"] == 4


@pytest.mark.parametrize("problem", ["empty", "future", "wrong_cutoff", "stale", "nonfinite", "string", "weekend_query"])
def test_missing_stale_or_future_inputs_never_become_default_economic_states(problem):
    query = pd.Timestamp("2026-09-01")
    cutoff = "2026-08-31"
    observed = pd.Series([0.1], index=pd.to_datetime(["2026-08-28"]))
    if problem == "empty":
        observed[:] = np.nan
    elif problem == "future":
        observed.index = pd.to_datetime(["2026-09-01"])
    elif problem == "wrong_cutoff":
        cutoff = "2026-08-28"
    elif problem == "stale":
        observed.index = pd.to_datetime(["2026-08-20"])
    elif problem == "nonfinite":
        observed[:] = np.inf
    elif problem == "string":
        observed = observed.astype(str)
    else:
        query = pd.Timestamp("2026-08-29")
    with pytest.raises(QuantError):
        select(observed, query, cutoff)


def test_exact_seven_day_age_is_not_a_relaxed_proxy_threshold():
    observed = pd.Series([-0.1], index=pd.to_datetime(["2026-08-25"]))
    query = pd.Timestamp("2026-09-01")
    assert select(observed, query, str(previous_session(query).date()))[1]["age_calendar_days"] == 7
    query = pd.Timestamp("2026-09-02")
    with pytest.raises(QuantError):
        select(observed, query, str(previous_session(query).date()))


def test_full_450query_source_grid_is_required_without_current_history_fallback():
    index = sessions("2015-08-10", "2026-10-05")
    _, dates = cohort(index)
    history = sessions("2014-01-01", "2026-10-05")
    vintages = {name:{cutoff:pd.Series(-0.1, index=history[history <= cutoff]) for cutoff in dates} for name in SERIES}
    result = query_readiness(index, vintages)
    assert len(result) == 450 and result["spread_percentage_points"].eq(-0.1).all()
    vintages["T10Y2Y"].pop(dates[0])
    with pytest.raises(QuantError, match="complete"):
        query_readiness(index, vintages)


@pytest.mark.parametrize("field,value", [("publication_delay_sessions", 1), ("maximum_observation_age_days", 14),
                                        ("query_sessions_per_series", 224), ("T10Y3M_counted_as_new_factor_or_asset", True),
                                        ("new_registered_factor_definitions", 1), ("new_registered_candidates", 2),
                                        ("strategy_states_targets_or_returns_computed", True),
                                        ("new_independent_factor_families_claimed", True),
                                        ("existing_W42_or_forward_rules_changed", True), ("order_authority", True)])
def test_source_lag_freshness_and_existing_factor_scope_cannot_be_relabelled(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/yield-curve-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
