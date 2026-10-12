from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import (
    SOURCE_URLS,
    build_from_inputs,
    decision_pairs,
    dollar_comparisons,
    load_vintages,
    parse_release,
    release_calendar,
    validate_policy,
    vintage_at,
)
from us_quant.dual_horizon import seed_window
from us_quant.multifactor_stability import FACTORS
from us_quant.sector_growth_balance import SECTORS
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/dollar-risk-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2018-06-29", ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, *SECTORS)
    )


def release_document(release, dates, *, new=False, values=None):
    values = (
        ["100.0000", "101.0000", "102.0000", "103.0000", "104.0000"] if values is None else values
    )
    headers = "".join(f'<th id="a{i + 3}">{day}</th>' for i, day in enumerate(dates))
    numbers = "".join(f"<td>{value}</td>" for value in values)
    unit = "JAN06=100" if new else "JAN97=100"
    return (
        f"<div>Release Date: {release}</div>"
        '<table class="statistics"><thead><tr><th id="a1">COUNTRY</th>'
        f'<th id="a2">CURRENCY</th>{headers}</tr></thead>'
        f"<tr><th>1) BROAD</th><td>{unit}</td>{numbers}</tr></table>"
    )


def input_changes(market, dollar=1.0, real=1.0):
    return pd.DataFrame(
        {"dollar_change": dollar, "real_yield_change": real},
        index=pd.DatetimeIndex(decision_pairs(market.close.index)),
    )


def test_actual_tuesday_release_is_unavailable_that_day_and_uses_next_nyse_session():
    vintage = parse_release(
        release_document(
            "October 11, 2016",
            ["Oct. 3", "Oct. 4", "Oct. 5", "Oct. 6", "Oct. 7"],
        ),
        pd.Timestamp("2016-10-11"),
    )
    assert vintage["available_session"].eq(pd.Timestamp("2016-10-12")).all()
    assert vintage["released_at"].eq("2016-10-11T16:15:00-04:00").all()
    for query in ("2016-10-10", "2016-10-11"):
        with pytest.raises(QuantError, match="actually released"):
            vintage_at(vintage, pd.Timestamp(query), "DTWEXB")
    assert vintage_at(vintage, pd.Timestamp("2016-10-12"), "DTWEXB")["value"] == 104.0
    with pytest.raises(QuantError, match="stale"):
        vintage_at(vintage, pd.Timestamp("2016-10-25"), "DTWEXB")


def test_year_rollover_and_explicit_missing_holiday_values_are_not_filled():
    vintage = parse_release(
        release_document(
            "January 6, 2020",
            ["Dec. 30", "Dec. 31", "Jan. 1", "Jan. 2", "Jan. 3"],
            new=True,
            values=["100", "101", "ND", "103", "ND"],
        ),
        pd.Timestamp("2020-01-06"),
    )
    assert len(vintage) == 3
    assert vintage["observation_date"].min() == pd.Timestamp("2019-12-30")
    selected = vintage_at(vintage, pd.Timestamp("2020-01-07"), "DTWEXBGS")
    assert selected["observation_date"] == pd.Timestamp("2020-01-02")
    assert selected["value"] == 103
    assert selected["unit_regime"] == "goods_services_january_2006_month"


def test_wrong_date_unscoped_index_wrong_units_and_preintroduction_backfill_fail():
    document = release_document(
        "August 31, 2015", ["Aug. 24", "Aug. 25", "Aug. 26", "Aug. 27", "Aug. 28"]
    )
    with pytest.raises(QuantError, match="archive URL"):
        parse_release(document, pd.Timestamp("2015-08-24"))
    with pytest.raises(QuantError, match="absent"):
        parse_release(document.replace("1) BROAD", "3) OITP"), pd.Timestamp("2015-08-31"))
    with pytest.raises(QuantError, match="indexation"):
        parse_release(document.replace("JAN97=100", "MAR73=100"), pd.Timestamp("2015-08-31"))
    with pytest.raises(QuantError, match="introduction"):
        parse_release(document.replace("JAN97=100", "JAN06=100"), pd.Timestamp("2015-08-31"))
    with pytest.raises(QuantError, match="neither numeric"):
        parse_release(document.replace("104.0000", "revised"), pd.Timestamp("2015-08-31"))


def test_official_calendar_keeps_nonmonday_dates_and_rejects_mislabelled_months():
    document = (
        '[{"yearValue":"2019","Months":'
        '[{"MonthValue":"201902","Dates":["20190205","20190211"]}]}]'
    )
    assert release_calendar(document).equals(pd.to_datetime(["2019-02-05", "2019-02-11"]))
    with pytest.raises(QuantError, match="inconsistent"):
        release_calendar(document.replace("20190205", "20190105"))
    with pytest.raises(QuantError, match="unique"):
        release_calendar(document.replace("20190211", "20190205"))


def test_publisher_retirement_uses_same_method_at_both_endpoints_and_never_splices_levels():
    index = sessions("2019-08-01", "2020-01-31")
    rows = []
    for i, day in enumerate(index):
        for series, value, regime in (
            ("DTWEXB", 100 + i, "goods_january_1997"),
            ("DTWEXBGS", 1000 - i, "goods_services_january_2006_month"),
        ):
            rows.append(
                {
                    "series": series,
                    "observation_date": day - pd.Timedelta(days=3),
                    "release_date": day - pd.Timedelta(days=1),
                    "available_session": day,
                    "unit_regime": regime,
                    "value": value,
                }
            )
    vintages = pd.DataFrame(rows)
    changes, audit = dollar_comparisons(vintages, index)
    assert (changes.loc[:"2019-12-31"] > 0).all()
    assert changes.loc["2020-01-31"] < 0
    assert audit.loc["2019-12-31", "series"] == "DTWEXB"
    assert audit.loc["2020-01-31", "series"] == "DTWEXBGS"
    assert audit["current_series"].equals(audit["reference_series"])
    future = vintages.copy()
    future.loc[future["available_session"] > pd.Timestamp("2019-12-31"), "value"] *= 10
    pd.testing.assert_series_equal(
        dollar_comparisons(future, index)[0].loc[:"2019-12-31"], changes.loc[:"2019-12-31"]
    )
    wrong = vintages.copy()
    wrong.loc[
        (wrong["series"] == "DTWEXBGS") & (wrong["available_session"] < pd.Timestamp("2020-01-01")),
        "unit_regime",
    ] = "goods_services_january_2006_day"
    with pytest.raises(QuantError, match="cannot splice"):
        dollar_comparisons(wrong, index)


@pytest.mark.parametrize("real,confirmed", [(-1.0, False), (0.0, False), (1.0, True)])
def test_dollar_and_real_yield_confirmation_halves_the_identical_full_risk_budget(
    market, policy, real, confirmed
):
    base = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        policy["original_control_candidate"]
    ]
    outputs = build_from_inputs(market, input_changes(market, real=real), policy)
    for candidate in policy["candidates"]:
        active = outputs[candidate["id"]].dropna(how="all")
        original = base.loc[active.index]
        half = not candidate["real_yield_confirmation"] or confirmed
        assets = active.columns.drop("BIL")
        np.testing.assert_allclose(active[assets], original[assets] * (0.5 if half else 1))
        np.testing.assert_allclose(active["BIL"], 0.49 if half else 0)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (active.loc[:, [*FACTORS, "SOXX", "GLD"]] > 0).all().all()
        assert active.index.equals(original.dropna(how="all").index)
        assert outputs[candidate["id"]].isna().all(axis=1).equals(base.isna().all(axis=1))


@pytest.mark.parametrize("dollar", [-1.0, 0.0])
def test_nonrising_dollar_keeps_original_targets_without_a_hindsight_threshold(
    market, policy, dollar
):
    expected = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        policy["original_control_candidate"]
    ]
    for target in build_from_inputs(market, input_changes(market, dollar=dollar), policy).values():
        pd.testing.assert_frame_equal(target, expected)


def test_future_macro_or_sector_observations_cannot_retune_past_targets(market, policy):
    inputs = input_changes(market)
    before = build_from_inputs(market, inputs, policy)
    cutoff = pd.Timestamp("2017-06-30")
    inputs.loc[inputs.index > cutoff] *= -1
    closing, opening = market.close.copy(), market.open.copy()
    after = closing.index > cutoff
    closing.loc[after, "SOXX"] *= np.linspace(1, 2, after.sum())
    opening.loc[after, "SOXX"] *= np.linspace(1, 2, after.sum())
    changed = replace(market, close=closing, raw_close=closing.copy(), open=opening)
    for name, target in build_from_inputs(changed, inputs, policy).items():
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], target.loc[:cutoff])
    with pytest.raises(QuantError, match="exactly the original"):
        build_from_inputs(market, inputs.iloc[1:], policy)


def test_source_hash_change_fails_before_parsing_or_using_a_current_csv(
    tmp_path, monkeypatch, policy
):
    monkeypatch.setattr("us_quant.dollar_risk_guard.SOURCE", tmp_path)
    files = {
        name: file_digest(write_fixture_file(tmp_path / name))
        for name in set(SOURCE_URLS) | {"acquisition.json", "archive-acquisition.json"}
    }
    write_json(
        tmp_path / "verified-manifest.json",
        {
            "schema_version": 1,
            "policy_sha256": file_digest(ROOT / "config/dollar-risk-guard.json"),
            "raw_inputs_not_to_be_published": True,
            "strategy_outcomes_computed": False,
            "current_fred_values_used_for_targets": False,
            "source_urls": SOURCE_URLS,
            "files": files,
            "releases": [],
            "quarantined_releases": [],
        },
    )
    write_text_atomic(tmp_path / "DTWEXB.csv", "changed source")
    with pytest.raises(QuantError, match="hash"):
        load_vintages(pd.to_datetime(["2017-01-31"]), policy)


def write_fixture_file(path):
    write_text_atomic(path, "synthetic source-seal fixture,not actual market data")
    return path


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_registered_source_replay_and_independent_cash_funded_paths(
    market, policy, monkeypatch, tmp_path, cost, delay
):
    inputs = input_changes(market)
    inputs.iloc[::2, 0] = -1
    monkeypatch.setattr(
        "us_quant.dollar_risk_guard.load_inputs", lambda index, config: (inputs, None, None)
    )
    dependencies = (
        "src/us_quant/dollar_risk_guard.py",
        "config/dollar-risk-guard.json",
        "src/us_quant/sector_growth_balance.py",
        "config/sector-growth-balance.json",
        "src/us_quant/growth_factor_satellite.py",
        "config/growth-factor-satellite.json",
        "src/us_quant/multifactor_stability.py",
        "src/us_quant/macro_factor_tilt.py",
    )
    frozen = {}
    for name in dependencies:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        frozen[name] = file_digest(path)
    for name in (
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        path = write_fixture_file(tmp_path / name)
        frozen[name] = file_digest(path)
    candidate = policy["candidates"][1]
    spec = {"id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
    start, end = "2016-10-06", "2017-03-31"
    targets = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    pd.testing.assert_frame_equal(
        targets, seed_window(build_from_inputs(market, inputs, policy)[candidate["id"]], start)
    )
    funded = simulate(
        market, targets, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    audit = independent_equity(
        market, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(funded.frame["equity"], audit["equity"], rtol=0, atol=1e-8)
    assert funded.frame["cash"].min() >= 0
    omitted = {
        **spec,
        "frozen_files": {
            name: digest
            for name, digest in frozen.items()
            if not name.endswith("DFII10-curl-response.csv")
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(omitted, market, start, end, cost, delay, tmp_path, {})


@pytest.mark.parametrize(
    "key,value",
    [("risk_change_sessions", 21), ("risk_off_investment_scale", 0.75), ("cash_reserve", 0.0)],
)
def test_frozen_risk_window_scale_and_cash_cannot_be_retuned(policy, key, value):
    policy[key] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_exact_audited_date_conflicts_use_later_date_and_unknown_conflicts_stay_blocked():
    template = release_document(
        "May 23, 2016", ["May 16", "May 17", "May 18", "May 19", "May 20"]
    ).replace(
        '<table class="statistics">',
        '<table class="pubtables" title="Foreign Exchange Rates -- H.10 Weekly">',
    )
    vintage = parse_release(template, pd.Timestamp("2016-05-24"), conservative_dates=True)
    assert vintage["reported_release_date"].eq(pd.Timestamp("2016-05-23")).all()
    assert vintage["release_date"].eq(pd.Timestamp("2016-05-24")).all()
    assert vintage["available_session"].eq(pd.Timestamp("2016-05-25")).all()
    assert vintage["date_conflict"].all()
    with pytest.raises(QuantError, match="archive URL"):
        parse_release(template, pd.Timestamp("2016-05-25"), conservative_dates=True)
    later = release_document("June 2, 2016", ["May 23", "May 24", "May 25", "May 26", "May 27"])
    vintage = parse_release(later, pd.Timestamp("2016-05-31"), conservative_dates=True)
    assert vintage["release_date"].eq(pd.Timestamp("2016-06-02")).all()
    assert vintage["available_session"].eq(pd.Timestamp("2016-06-03")).all()
    with pytest.raises(QuantError, match="actually released"):
        vintage_at(vintage, pd.Timestamp("2016-06-02"), "DTWEXB")
