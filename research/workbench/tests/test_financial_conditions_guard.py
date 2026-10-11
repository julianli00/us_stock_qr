from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.financial_conditions_guard import (
    CANDIDATES,
    build_from_inputs,
    needed_vintage_dates,
    parse_vintage_archive,
    validate_policy,
)
from us_quant.multifactor_stability import FACTORS
from us_quant.sector_growth_balance import SECTORS
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, read_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/financial-conditions-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10",
        "2018-06-29",
        ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, *SECTORS),
    )


def create_archive(path, dates, content, *, readme_dates=None):
    readme = (
        "Series ID: NFCI\nFederal Reserve Bank of Chicago\nWeekly, Ending Friday\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Vintage Dates Specified:\n----------\n"
        + "\n".join(dates if readme_dates is None else readme_dates)
        + "\n----------\n"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("README.txt", readme)
        archive.writestr(f"vintages_starting_{dates[0]}.csv", content)


def comparisons(market, dollar=1.0, level=-0.2, change=0.1):
    months = (
        sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
            "four_factor_semiconductor_sector50"
        ]
        .dropna(how="all")
        .index
    )
    return pd.DataFrame(
        {"dollar_change": dollar, "conditions_level": level, "conditions_change": change},
        index=months,
    ).assign(level_available=True, change_available=True)


def test_actual_vintage_columns_preserve_rounding_and_unavailable_future_rows(tmp_path):
    dates = ["2015-08-28", "2016-09-29"]
    path = tmp_path / "actual-shape.zip"
    create_archive(
        path,
        dates,
        "observation_date,NFCI_20150828,NFCI_20160929\n"
        "2015-08-21,-0.74,-0.71\n2015-08-28,,-0.67\n2016-09-23,,-0.60\n",
    )
    frame = parse_vintage_archive(path, dates)
    assert frame.loc["2015-08-21", "2015-08-28"] == -0.74
    assert pd.isna(frame.loc["2015-08-28", "2015-08-28"])
    assert frame.loc["2015-08-21", "2015-08-28"] != frame.loc["2015-08-21", "2016-09-29"]
    assert frame["2015-08-28"].dropna().index[-1] == pd.Timestamp("2015-08-21")


@pytest.mark.parametrize("problem", ["future_value", "wrong_column", "wrong_readme", "not_friday"])
def test_false_current_history_or_mislabeled_vintages_are_rejected(tmp_path, problem):
    dates = ["2015-08-28"]
    path = tmp_path / "bad.zip"
    content = "observation_date,NFCI_20150828\n2015-08-21,-0.74\n"
    readme_dates = None
    if problem == "future_value":
        content += "2015-09-04,-0.65\n"
    elif problem == "wrong_column":
        content = content.replace("NFCI_20150828", "NFCI")
    elif problem == "wrong_readme":
        readme_dates = ["2026-09-29"]
    else:
        content = content.replace("2015-08-21", "2015-08-20")
    create_archive(path, dates, content, readme_dates=readme_dates)
    with pytest.raises(QuantError):
        parse_vintage_archive(path, dates)


def test_html_or_extra_archive_member_cannot_be_admitted_as_a_vintage(tmp_path):
    path = tmp_path / "response.zip"
    write_text_atomic(path, "<html>Download form validation failed</html>")
    with pytest.raises(QuantError, match="actual readable ZIP"):
        parse_vintage_archive(path, ["2015-08-28"])
    create_archive(path, ["2015-08-28"], "observation_date,NFCI_20150828\n2015-08-21,-0.74\n")
    with ZipFile(path, "a") as archive:
        archive.writestr("../unsupported.csv", "untrusted")
    with pytest.raises(QuantError, match="two-file"):
        parse_vintage_archive(path, ["2015-08-28"])


def test_vintage_queries_always_end_on_the_previous_actual_market_session(market):
    dates = needed_vintage_dates(market.close.index)
    assert dates[0] == "2015-08-28"
    assert dates[-1] == "2018-06-28"
    assert dates == sorted(set(dates))
    assert all(pd.Timestamp(day).weekday() < 5 for day in dates)


def test_tightness_and_deterioration_are_distinct_source_defined_confirmations(market, policy):
    original = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    inputs = comparisons(market, level=-0.2, change=0.1)
    outputs = build_from_inputs(market, inputs, policy)
    pd.testing.assert_frame_equal(outputs[CANDIDATES[0]["id"]], original)
    active = outputs[CANDIDATES[1]["id"]].dropna(how="all")
    assets = active.columns.drop("BIL")
    np.testing.assert_allclose(active[assets], original.loc[active.index, assets] / 2)
    np.testing.assert_allclose(active["BIL"], 0.49)
    assert (active.loc[:, [*FACTORS, "SOXX", "GLD"]] > 0).all().all()
    np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "dollar,level,change", [(-0.1, 0.1, 0.1), (0.0, 0.1, 0.1), (0.1, 0.0, 0.0)]
)
def test_equality_or_nonrising_dollar_does_not_create_unregistered_risk_off_rules(
    market, policy, dollar, level, change
):
    original = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    for target in build_from_inputs(
        market, comparisons(market, dollar=dollar, level=level, change=change), policy
    ).values():
        pd.testing.assert_frame_equal(target, original)


def test_future_inputs_or_prices_cannot_retune_a_past_target(market, policy):
    inputs = comparisons(market)
    before = build_from_inputs(market, inputs, policy)
    cutoff = pd.Timestamp("2017-06-30")
    inputs.loc[inputs.index > cutoff, "conditions_level"] += 10
    prices, opening = market.close.copy(), market.open.copy()
    later = prices.index > cutoff
    prices.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    after = build_from_inputs(
        replace(market, close=prices, open=opening, raw_close=prices.copy()), inputs, policy
    )
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], after[name].loc[:cutoff])


def test_declared_source_gaps_pause_only_the_candidate_that_requires_the_missing_reading(
    market, policy
):
    inputs = comparisons(market)
    first, second = pd.Timestamp("2017-01-31"), pd.Timestamp("2017-02-28")
    inputs.loc[first, ["conditions_level", "conditions_change"]] = np.nan
    inputs.loc[first, ["level_available", "change_available"]] = False
    inputs.loc[second, "conditions_change"] = np.nan
    inputs.loc[second, "change_available"] = False
    outputs = build_from_inputs(market, inputs, policy)
    for target in outputs.values():
        assert target.loc[first].isna().all()
    assert outputs[CANDIDATES[0]["id"]].loc[second].notna().all()
    assert outputs[CANDIDATES[1]["id"]].loc[second].isna().all()
    start, end = "2016-10-06", "2017-03-31"
    funded = simulate(
        market,
        seed_window(outputs[CANDIDATES[1]["id"]], start),
        start,
        end,
        initial_capital=10000,
        cost_bps=5,
        commission=1,
        delay=1,
    )
    assert funded.frame.loc["2017-02-01", "orders"] == 0
    assert funded.frame.loc["2017-03-01", "orders"] == 0
    assert funded.frame.index[0] == pd.Timestamp(start)
    assert funded.frame.index[-1] == pd.Timestamp(end)
    assert funded.frame.loc["2017-02-01", "return"] != 0
    invalid = inputs.copy()
    invalid.loc[first, "conditions_level"] = -0.5
    with pytest.raises(QuantError, match="source availability"):
        build_from_inputs(market, invalid, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("risk_change_sessions", 21),
        ("tight_conditions_threshold", -0.25),
        ("risk_off_investment_scale", 0.75),
        ("maximum_observation_age_days", 28),
        ("vintage_cutoff", "current_download"),
        ("new_economic_factor_definitions", 1),
    ],
)
def test_windows_thresholds_vintage_timing_and_factor_count_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_registered_targets_and_independent_accounts_agree(
    market, policy, monkeypatch, tmp_path, cost, delay
):
    inputs = comparisons(market)
    inputs.iloc[::2, 2] *= -1
    monkeypatch.setattr(
        "us_quant.financial_conditions_guard.load_inputs",
        lambda index, config: (inputs, {}),
    )
    dependencies = (
        "src/us_quant/financial_conditions_guard.py",
        "config/financial-conditions-guard.json",
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
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        frozen[name] = file_digest(destination)
    for name in (
        "data/financial-conditions-vintages-20261011/verified-manifest-v2.json",
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        write_text_atomic(tmp_path / name, "synthetic source-seal fixture,not actual data")
        frozen[name] = file_digest(tmp_path / name)
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
    independent = independent_equity(
        market, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(funded.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert funded.frame["cash"].min() >= 0
    missing = {
        **spec,
        "frozen_files": {
            name: digest for name, digest in frozen.items() if "dollar_risk_guard.py" not in name
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(missing, market, start, end, cost, delay, tmp_path, {})
