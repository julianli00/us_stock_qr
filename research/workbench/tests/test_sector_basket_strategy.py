from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import sector_basket_strategy as basket
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.factor_research import factor_scores
from us_quant.storage import read_json
from us_quant.storage import file_digest, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/sector-basket-strategy.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2017-06-30",
        ("SPY", "IEF", "GLD", "BIL", *basket.FACTORS, *basket.sources.SECTORS),
    )


def changes(market):
    days = [day for index, day in enumerate(market.close.index) if index >= 252 and is_month_end(day)]
    return pd.Series([0.01 if index % 2 else -0.01 for index in range(len(days))], index=days)


def test_skip_only_selector_matches_original_factor_scores_after_full_original_warmup(market_factory):
    old_policy = read_json(ROOT / "config/factor-research.json")
    data = market_factory("2010-01-04", "2014-06-30", ("SPY", *basket.sources.SECTORS))
    scores = factor_scores(data.close, data.risk_free, old_policy)
    shares, metadata = basket.sector_selection(data.close)
    eligible = scores.loc[(scores["momentum"] > 0) & scores["above_trend"] & (scores["volatility"] > 1e-10)]
    selected = sorted(eligible.index, key=lambda symbol: (-float(eligible.loc[symbol, "momentum"]), symbol))[:3]
    expected = pd.Series(0.0, index=shares.index)
    inverse = 1 / scores.loc[selected, "volatility"]
    expected.loc[selected] = (0.98 * len(selected) / 3 * inverse / inverse.sum()).clip(upper=0.40) / 0.98
    expected["BIL"] = 1 - expected.drop("BIL").sum()
    pd.testing.assert_series_equal(shares, expected)
    assert metadata["selected_symbols"] == selected


def test_missing_sector_slots_stay_cash_without_renormalizing():
    dates = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.DataFrame(100.0, index=dates, columns=basket.sources.SECTORS)
    prices["XLB"] = 100 * np.exp(np.linspace(0, 0.2, len(dates)) + 0.002 * np.sin(np.arange(len(dates))))
    shares, metadata = basket.sector_selection(prices)
    assert metadata["selected_symbols"] == ["XLB"]
    assert shares["XLB"] == pytest.approx(1 / 3)
    assert shares["BIL"] == pytest.approx(2 / 3)
    flat, metadata = basket.sector_selection(prices.assign(XLB=100.0))
    assert metadata["selected_symbols"] == []
    assert flat["BIL"] == 1


def test_fully_invested_unit_sector_basket_cannot_create_rounding_only_negative_bills(market):
    day = pd.Timestamp("2017-03-31")
    shares, selection = basket.sector_selection(market.close.loc[:day])
    assert len(selection["selected_symbols"]) == 3
    assert shares["BIL"] >= 0
    assert shares.sum() == pytest.approx(1, abs=1e-12)


def test_actual_half_core_monthly_targets_and_dollar_pair_retain_all_four_families(market, policy):
    dollar = changes(market)
    frames, audit = basket.build_from_inputs(market, policy, dollar)
    plain = frames[basket.CANDIDATES[0]["id"]].dropna(how="all")
    guarded = frames[basket.CANDIDATES[1]["id"]].dropna(how="all")
    assert (plain[list(basket.FACTORS)] > 0).all().all()
    assert all(is_month_end(day) for day in plain.index)
    np.testing.assert_allclose(plain.sum(axis=1), 0.98, atol=1e-12)
    np.testing.assert_allclose(guarded.sum(axis=1), 0.98, atol=1e-12)
    non_bil = plain.columns != "BIL"
    for day in dollar.index:
        if dollar[day] > 0:
            np.testing.assert_allclose(guarded.loc[day, non_bil], plain.loc[day, non_bil] * 0.5)
            assert guarded.loc[day, "BIL"] > plain.loc[day, "BIL"]
        else:
            pd.testing.assert_series_equal(guarded.loc[day], plain.loc[day])
    assert len(audit) == len(plain) * 2
    assert plain[["SPY", "IEF"]].eq(0).all().all()


def test_zero_dollar_change_does_not_trigger_half_risk(market, policy):
    frames, _ = basket.build_from_inputs(market, policy, changes(market) * 0)
    pd.testing.assert_frame_equal(frames[basket.CANDIDATES[0]["id"]], frames[basket.CANDIDATES[1]["id"]])


def test_future_prices_do_not_change_earlier_selected_baskets_or_risk_targets(market, policy):
    dollar = changes(market)
    before, _ = basket.build_from_inputs(market, policy, dollar)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multiplier = np.linspace(1, 1.5, later.sum())
    close.loc[later, "XLK"] *= multiplier
    opening.loc[later, "XLK"] *= multiplier
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, _ = basket.build_from_inputs(changed, policy, dollar)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("problem", ["missing", "nan", "duplicate"])
def test_incomplete_dollar_queries_are_not_filled_with_safe_defaults(market, policy, problem):
    dollar = changes(market)
    if problem == "missing":
        dollar = dollar.iloc[1:]
    elif problem == "nan":
        dollar.iloc[0] = np.nan
    else:
        dollar.index = [dollar.index[0]] * len(dollar)
    with pytest.raises(QuantError, match="complete historical dollar"):
        basket.build_from_inputs(market, policy, dollar)


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 1), ("momentum_sessions", 63), ("skip_sessions", 0),
        ("standalone_sector_cap", 0.98), ("satellite_share_of_equity_bucket", 0.75),
        ("dollar_risk_off_scale", 0.25), ("equity_bucket_min", 0.0),
        ("sector_symbols", ["SOXX"]), ("new_economic_factor_definitions", 9),
    ],
)
def test_scope_cannot_select_sectors_or_tune_parameters_after_results(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        basket.validate(policy)


def test_replay_accepts_the_complete_transitive_dollar_guard_dependencies(tmp_path, policy):
    names = (
        "src/us_quant/sector_basket_strategy.py", "config/sector-basket-strategy.json",
        "src/us_quant/sector_basket_sources.py", "src/us_quant/sector_growth_balance.py",
        "src/us_quant/factor_research.py", "src/us_quant/multifactor_stability.py",
        "src/us_quant/dual_horizon.py", "src/us_quant/dollar_risk_guard.py",
        "src/us_quant/macro_factor_tilt.py", "src/us_quant/growth_factor_satellite.py",
        "config/sector-basket-sources.json", "config/factor-research.json", "config/dual-horizon.json",
        "config/dollar-risk-guard.json", "config/sector-growth-balance.json",
        "config/growth-factor-satellite.json", "evidence/factor_round_20261007_registration.json",
        "evidence/factor_gold_risk_20261010_registration.json",
        "evidence/sector_basket_source_readiness_20261011.json",
        "data/sector-basket-sources-20261011/manifest.json",
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("src/") or name.startswith("config/"):
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_text('{"synthetic":true}')
    write_json(tmp_path / "config/sector-basket-strategy.json", policy)
    spec = {
        "id": basket.CANDIDATES[0]["id"], "configuration": basket.CANDIDATES[0],
        "frozen_files": {name: file_digest(tmp_path / name) for name in names},
    }
    assert registered_generator(spec, tmp_path)[0] == "sector_basket_strategy"
    del spec["frozen_files"]["config/growth-factor-satellite.json"]
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, tmp_path)
