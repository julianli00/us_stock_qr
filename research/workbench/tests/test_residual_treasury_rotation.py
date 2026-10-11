from __future__ import annotations

import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import residual_treasury_rotation as rotation
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.storage import file_digest, read_json, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/residual-treasury-rotation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30",
                          ("SPY", "QQQ", "GLD", "BIL", "IEF", "TLT", *rotation.FACTORS, "XLK", "SOXX"))


def source(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for index, day in enumerate(frame.index):
        if index >= 252 and is_month_end(day):
            frame.loc[day] = 0.0
            equity_scale = (index % 3) / 3
            gold_scale = (index % 4) / 3
            frame.loc[day, list(rotation.FACTORS)] = 0.98 * 0.6 * equity_scale / 8
            frame.loc[day, "SOXX"] = 0.98 * 0.6 * equity_scale / 2
            frame.loc[day, "GLD"] = 0.98 * 0.4 * gold_scale
            frame.loc[day, "BIL"] = 0.98 - frame.loc[day].sum()
    return frame


def test_strict_treasury_eligibility_and_lexical_ties(market):
    history = market.close.iloc[:127].copy()
    bills = np.linspace(100, 102, 127)
    history["BIL"] = bills
    history["IEF"] = bills
    history["TLT"] = np.linspace(100, 101, 127)
    for method in (row["defense_method"] for row in rotation.CANDIDATES):
        shares, audit = rotation.treasury_shares(history, method)
        assert shares.to_dict() == {"IEF": 0.0, "TLT": 0.0, "BIL": 1.0}
        assert audit["eligible_treasuries"] == []
    history["IEF"] = np.linspace(100, 103, 127)
    history["TLT"] = history["IEF"]
    shares, audit = rotation.treasury_shares(history, "highest_positive_excess")
    assert shares.to_dict() == {"IEF": 1.0, "TLT": 0.0, "BIL": 0.0}
    assert audit["eligible_treasuries"] == ["IEF", "TLT"]
    history["TLT"] = np.linspace(100, 104, 127)
    shares, _ = rotation.treasury_shares(history, "highest_positive_excess")
    assert shares["TLT"] == 1.0


def test_inverse_volatility_uses_all_eligible_treasuries_and_63_sample_returns(market):
    history = market.close.iloc[:127].copy()
    history["BIL"] = np.linspace(100, 101, 127)
    history["IEF"] = np.linspace(100, 108, 127) + 0.1 * np.sin(np.arange(127))
    history["TLT"] = np.linspace(100, 110, 127) + 0.4 * np.cos(np.arange(127))
    shares, audit = rotation.treasury_shares(history, "all_positive_inverse_volatility")
    inverse = {}
    for name in ("IEF", "TLT"):
        levels = history[name].tolist()
        daily = [right / left - 1 for left, right in zip(levels, levels[1:], strict=False)]
        inverse[name] = 1 / statistics.stdev(daily[-63:])
    for name in inverse:
        assert shares[name] == pytest.approx(inverse[name] / sum(inverse.values()), abs=1e-12)
    assert audit["eligible_treasuries"] == ["IEF", "TLT"]
    assert shares["BIL"] == 0.0
    history["IEF"] = history["BIL"]
    shares, audit = rotation.treasury_shares(history, "all_positive_inverse_volatility")
    assert shares.to_dict() == {"IEF": 0.0, "TLT": 1.0, "BIL": 0.0}
    assert audit["eligible_treasuries"] == ["TLT"]


def test_eligible_degenerate_volatility_is_not_a_safe_fallback(market):
    history = market.close.iloc[:127].copy()
    history["BIL"] = np.linspace(100, 99, 127)
    history["IEF"] = 100.0
    history["TLT"] = history["BIL"]
    with pytest.raises(QuantError, match="volatility"):
        rotation.treasury_shares(history, "all_positive_inverse_volatility")
    shares, _ = rotation.treasury_shares(history, "highest_positive_excess")
    assert shares["IEF"] == 1.0


@pytest.mark.parametrize("shares", [(1.0, 0.0, 0.0), (0.3, 0.7, 0.0), (0.0, 0.0, 1.0)])
def test_routing_replaces_only_bills_and_keeps_every_other_target(market, shares):
    original = source(market).dropna(how="all").iloc[0]
    defensive = pd.Series(shares, index=["IEF", "TLT", "BIL"])
    actual = rotation.route_residual(original, defensive)
    unchanged = original.index.difference(defensive.index)
    pd.testing.assert_series_equal(actual.loc[unchanged], original.loc[unchanged])
    assert actual.sum() == pytest.approx(0.98, abs=1e-12)
    for name in ("IEF", "TLT"):
        assert actual[name] == pytest.approx(original[name] + original["BIL"] * defensive[name])
    assert actual["BIL"] == pytest.approx(original["BIL"] * defensive["BIL"])
    assert original["IEF"] == original["TLT"] == 0.0


def test_zero_bill_budget_never_adds_treasury_exposure(market):
    original = pd.Series(0.0, index=market.close.columns)
    original["GLD"] = 0.98
    defensive = pd.Series({"IEF": 0.5, "TLT": 0.5, "BIL": 0.0})
    pd.testing.assert_series_equal(rotation.route_residual(original, defensive), original)
    original["GLD"] = 0.0
    original["BIL"] = 0.98
    actual = rotation.route_residual(original, defensive)
    assert actual["IEF"] == actual["TLT"] == 0.49
    assert actual.drop(["IEF", "TLT"]).eq(0).all()


def test_only_original_monthly_decisions_and_full_source_shape_are_preserved(market, policy):
    original = source(market)
    frames, diagnostics = rotation.build_from_source(market, original, policy)
    for frame in frames.values():
        assert frame.index.equals(original.index)
        assert frame.columns.equals(original.columns)
        assert frame.isna().equals(original.isna())
        pd.testing.assert_frame_equal(frame.drop(columns=["IEF", "TLT", "BIL"]),
                                      original.drop(columns=["IEF", "TLT", "BIL"]))
        assert all(is_month_end(day) for day in frame.dropna(how="all").index)
        assert np.allclose(frame.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert len(diagnostics) == len(original.dropna(how="all")) * 2


def test_future_prices_cannot_change_prior_eligibility_or_targets(market, policy):
    original = source(market)
    before, _ = rotation.build_from_source(market, original, policy)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    close.loc[later, "TLT"] *= np.linspace(1, 2, later.sum())
    opening.loc[later, "TLT"] *= np.linspace(1, 2, later.sum())
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, _ = rotation.build_from_source(changed, original, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("start,end", [("2016-10-06", "2016-12-30"), ("2017-01-06", "2017-06-30")])
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_independent_funded_and_bt_accounts_keep_each_windows_full_seed(market, policy, start, end, cost, delay):
    frames, _ = rotation.build_from_source(market, source(market), policy)
    for frame in frames.values():
        targets = seed_window(frame, start)
        own = simulate(market, targets, start, end, initial_capital=10000,
                       cost_bps=cost, commission=1, delay=delay)
        independent = independent_equity(market, targets, start, end, capital=10000,
                                         cost_bps=cost, commission=1, delay=delay)
        np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0
        if delay == 2:
            assert own.frame.iloc[0]["equity"] == 10000


@pytest.mark.parametrize("problem", ["partial", "negative", "budget", "empty", "nonmonthly", "columns"])
def test_invalid_original_targets_fail_without_a_default(market, policy, problem):
    original = source(market)
    day = original.dropna(how="all").index[0]
    if problem == "partial":
        original.iloc[0, 0] = 0.1
    elif problem == "negative":
        original.loc[day, "BIL"] = -0.1
    elif problem == "budget":
        original.loc[day] *= 2
    elif problem == "empty":
        original[:] = np.nan
    elif problem == "nonmonthly":
        original.iloc[253] = original.loc[day]
    else:
        original = original.drop(columns="TLT")
    with pytest.raises(QuantError):
        rotation.build_from_source(market, original, policy)


@pytest.mark.parametrize("problem", ["short", "missing", "duplicate", "nonfinite", "nonpositive", "method"])
def test_incomplete_defensive_history_is_not_invented(market, problem):
    history = market.close.iloc[:127].copy()
    method = "highest_positive_excess"
    if problem == "short":
        history = history.iloc[1:]
    elif problem == "missing":
        history = history.drop(columns="BIL")
    elif problem == "duplicate":
        history.index = pd.DatetimeIndex([history.index[0]] * len(history))
    elif problem == "nonfinite":
        history.iloc[0, 0] = np.nan
    elif problem == "nonpositive":
        history.iloc[0, 0] = 0.0
    else:
        method = "unregistered_method"
    with pytest.raises(QuantError):
        rotation.treasury_shares(history, method)


@pytest.mark.parametrize("field,value", [("momentum_sessions", 63), ("volatility_sessions", 126),
                                        ("treasury_assets", ["IEF", "TLT", "GLD"]), ("cash_reserve", 0.0),
                                        ("new_configurations", 10), ("new_economic_factor_definitions", 1)])
def test_source_rules_and_fixed_defensive_subset_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        rotation.validate(policy)


def test_finite_replay_requires_the_complete_original_trend_dependency_graph(tmp_path, policy):
    names = (
        "src/us_quant/residual_treasury_rotation.py", "config/residual-treasury-rotation.json",
        "src/us_quant/semiconductor_trend_transfer.py", "config/semiconductor-trend-transfer.json",
        "evidence/trend_transfer_20261011_registration.json",
        "src/us_quant/defensive_factor_rotation.py", "config/defensive-factor-rotation.json",
        "src/us_quant/trend_factor_guard.py", "config/trend-factor-guard.json",
        "config/allocation-research.json", "evidence/dollar_risk_guard_20261011_registration.json",
        "src/us_quant/factor_gold_risk.py", "config/factor-gold-risk.json",
        "src/us_quant/sector_growth_balance.py", "config/sector-growth-balance.json",
        "src/us_quant/growth_factor_satellite.py", "config/growth-factor-satellite.json",
        "src/us_quant/multifactor_stability.py", "src/us_quant/dollar_risk_guard.py",
        "config/dollar-risk-guard.json", "src/us_quant/macro_factor_tilt.py",
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith(("src/", "config/")):
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_text('{"synthetic":true}')
    write_json(tmp_path / "config/residual-treasury-rotation.json", policy)
    spec = {
        "id": rotation.CANDIDATES[0]["id"], "configuration": rotation.CANDIDATES[0],
        "frozen_files": {name: file_digest(tmp_path / name) for name in names},
    }
    assert registered_generator(spec, tmp_path)[0] == "residual_treasury_rotation"
    for name in ("config/factor-gold-risk.json", "src/us_quant/trend_factor_guard.py"):
        incomplete = {**spec, "frozen_files": {key: value for key, value in spec["frozen_files"].items() if key != name}}
        with pytest.raises(QuantError, match="frozen helper dependency"):
            registered_generator(incomplete, tmp_path)
