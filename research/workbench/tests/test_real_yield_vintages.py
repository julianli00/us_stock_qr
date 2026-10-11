from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.real_yield_vintages import parse_vintage_archive, select_vintage, validate_policy
from us_quant.storage import read_json

DATES = ["2026-08-28", "2026-09-01"]


def archive(tmp_path, *, columns=None, rows=None, dates=DATES, series="DFII10", frequency="Daily"):
    path = tmp_path / "vintages.zip"
    names = columns or ["observation_date", *(f"DFII10_{day.replace('-', '')}" for day in dates)]
    body = rows or ["2026-08-28,-0.25,-0.24", "2026-08-31,,0.05"]
    readme = (
        f"Series ID: {series}\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Source\nBoard of Governors of the Federal Reserve System (US)\n"
        "Release\nH.15 Selected Interest Rates\nUnits\nPercent\n"
        f"Frequency\n{frequency}\nVintage Dates Specified:\n----------\n"
        + "\n".join(dates)
        + "\n----------\n"
    )
    with ZipFile(path, "w") as output:
        output.writestr("README.txt", readme)
        output.writestr(
            f"vintages_starting_{dates[0]}.csv", ",".join(names) + "\n" + "\n".join(body) + "\n"
        )
    return path


def test_archived_percent_yields_negative_values_and_native_revisions_are_preserved(tmp_path):
    frame = parse_vintage_archive(archive(tmp_path), DATES)
    assert frame.iloc[0, 0] == -0.25
    assert frame.iloc[0, 1] == -0.24
    assert np.isnan(frame.iloc[1, 0])
    assert frame.iloc[1, 1] == 0.05
    assert list(frame.columns) == DATES


@pytest.mark.parametrize(
    "problem", ["series", "weekly", "columns", "future", "duplicate", "weekend", "inf"]
)
def test_bad_daily_series_or_future_observations_cannot_enter_historical_queries(tmp_path, problem):
    options = {}
    if problem == "series":
        options["series"] = "NFCI"
    elif problem == "weekly":
        options["frequency"] = "Weekly, Ending Friday"
    elif problem == "columns":
        options["columns"] = ["observation_date", "DFII10_20260828", "unrelated"]
    elif problem == "future":
        options["rows"] = ["2026-08-31,1.0,1.1"]
    elif problem == "duplicate":
        options["rows"] = ["2026-08-28,1.0,1.1", "2026-08-28,1.0,1.1"]
    elif problem == "weekend":
        options["rows"] = ["2026-08-29,,1.1"]
    else:
        options["rows"] = ["2026-08-28,1.0,inf"]
    with pytest.raises(QuantError):
        parse_vintage_archive(archive(tmp_path, **options), DATES)


def test_request_dates_and_readme_must_match_exactly_not_just_an_archive_filename(tmp_path):
    path = archive(tmp_path)
    with pytest.raises(QuantError, match="confirm every"):
        parse_vintage_archive(path, ["2026-08-28", "2026-09-02"])
    with pytest.raises(QuantError, match="bounded"):
        parse_vintage_archive(path, DATES[::-1])
    with pytest.raises(QuantError, match="bounded"):
        parse_vintage_archive(path, ["2026-08-28"] * 41)


def test_previous_session_vintage_retains_the_unchanged_two_session_observation_lag():
    observed = pd.Series(
        [1.0, 1.1, 9.0],
        index=pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-01"]),
    )
    value, audit = select_vintage(observed, pd.Timestamp("2026-09-02"), "2026-09-01")
    assert value == 1.1
    assert audit["observation_date"] == "2026-08-31"
    assert audit["assumed_observation_available_session"] == "2026-09-02"
    assert audit["cutoff_strictly_before_query"]
    assert audit["age_calendar_days"] == 2


def test_snapshot_futures_or_wrong_cutoff_do_not_override_legacy_lag():
    observed = pd.Series([1.0], index=pd.to_datetime(["2026-09-02"]))
    with pytest.raises(QuantError, match="without future"):
        select_vintage(observed, pd.Timestamp("2026-09-02"), "2026-09-01")
    with pytest.raises(QuantError, match="previous-NYSE"):
        select_vintage(observed, pd.Timestamp("2026-09-03"), "2026-09-01")


def test_stale_and_empty_vintage_inputs_block_instead_of_relaxing_the_age_limit():
    stale = pd.Series([1.0], index=pd.to_datetime(["2026-08-10"]))
    with pytest.raises(QuantError, match="stale"):
        select_vintage(stale, pd.Timestamp("2026-09-02"), "2026-09-01")
    empty = pd.Series([np.nan], index=pd.to_datetime(["2026-08-31"]))
    with pytest.raises(QuantError, match="No finite"):
        select_vintage(empty, pd.Timestamp("2026-09-02"), "2026-09-01")


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_vintage_count", 224),
        ("monthly_comparison_count", 130),
        ("publication_delay_sessions", 1),
        ("maximum_observation_age_days", 14),
        ("change_sessions", 21),
        ("order_authority", True),
        ("forward_rules_changed", True),
        ("qualification_policy_changed", True),
        ("source_comparison_is_portfolio_performance", True),
    ],
)
def test_source_diagnosis_cannot_become_an_unregistered_strategy_or_lag_search(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/real-yield-vintages.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_return_window_slice_cannot_prove_unchanged_initial_signal():
    dates = sessions("2026-08-31", "2026-09-04")
    first = pd.Timestamp("2026-09-02")
    anchor = previous_session(first)
    before = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    before.loc["2026-08-31"] = [0.60, 0.38]
    before.loc["2026-09-03"] = [0.40, 0.58]
    after = before.copy()
    after.loc["2026-08-31"] = [0.50, 0.48]
    old, new = seed_window(before, first), seed_window(after, first)
    assert old.loc[first:].equals(new.loc[first:])
    assert not old.loc[anchor:].equals(new.loc[anchor:])
    assert anchor == pd.Timestamp("2026-09-01")
    assert np.isfinite(old.loc[anchor]).all()
    assert np.isfinite(new.loc[anchor]).all()


def test_published_vintage_target_proofs_include_seed_and_every_formal_session():
    evidence = read_json(
        Path(__file__).parents[1] / "evidence/real_yield_vintage_20261011_results_v2.json"
    )
    proofs = evidence["formal_target_proofs_including_explicit_seed"]
    assert len(proofs) == 4
    assert {(row["candidate_id"], row["years"]) for row in proofs} == {
        (name, years)
        for name in ("macro_real_yield_factor_tilt", "macro_real_yield_tilt_curve_defense")
        for years in (10, 5)
    }
    for row in proofs:
        first, end = row["formal_return_start"], row["formal_return_end"]
        count = len(sessions(first, end))
        assert row["initial_seed_signal_session"] == str(previous_session(first).date())
        assert row["formal_return_sessions"] == count
        assert row["target_shape_including_seed"] == [count + 1, 8]
        assert row["seed_row_is_explicitly_present_and_finite"] is True
        assert row["seeded_original_and_source_replaced_targets_byte_identical"] is True
        assert row["this_is_not_a_fully_PIT_daily_feature_dataset"] is True
