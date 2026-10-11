from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import semiconductor_volatility_target as transfer
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.growth_factor_satellite import build_targets as original_targets
from us_quant.growth_portfolio_protection import monthly_targets as original_monthly
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/semiconductor-volatility-target.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", ("SPY", "GLD", "BIL"))


def sources(data):
    result = {}
    for candidate in transfer.CANDIDATES:
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for index, day in enumerate(frame.index):
            if index >= 63 and is_month_end(day):
                frame.loc[day] = [0.49, 0.49, 0]
        result[candidate["id"]] = frame
    return result


def test_transfer_exactly_reproduces_prior_growth_projection_and_event_law(market_factory, policy):
    columns = ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *transfer.FACTORS)
    data = market_factory("2015-08-10", "2017-06-30", columns)
    old_policy = read_json(ROOT / "config/growth-factor-satellite.json")
    monthly = original_monthly(data, read_json(ROOT / "config/growth-portfolio-protection.json"))
    baselines = {candidate["id"]: monthly.copy() for candidate in transfer.CANDIDATES}
    actual, _ = transfer.build_from_baselines(data, baselines, policy)
    expected = original_targets(data, old_policy)["four_factor_growth_satellite50"]
    for values in actual.values():
        pd.testing.assert_frame_equal(values, expected)


def test_projection_uses_analytic_risky_scale_and_preserves_existing_bills():
    names = ["SPY", "GLD", "BIL"]
    covariance = pd.DataFrame(np.diag([0.04, 0.04, 0]), index=names, columns=names)
    source = pd.Series([0.49, 0.49, 0], index=names)
    actual = transfer.cap_volatility(source, covariance, 0.12, 0.98)
    scale = 0.12 / np.sqrt(source.to_numpy() @ covariance.to_numpy() @ source.to_numpy())
    np.testing.assert_allclose(actual.drop("BIL"), source.drop("BIL") * scale, rtol=0, atol=1e-13)
    assert actual["BIL"] == pytest.approx(0.98 * (1 - scale))
    guarded = source.copy()
    guarded.loc[["SPY", "GLD"]] *= 0.5
    guarded["BIL"] = 0.49
    pd.testing.assert_series_equal(transfer.cap_volatility(guarded, covariance, 0.12, 0.98), guarded)


def test_all_issued_targets_obey_forecast_ceiling_without_upscaling_monthly_risk(market, policy):
    frames, diagnostics = transfer.build_from_baselines(market, sources(market), policy)
    issued = [row for row in diagnostics if row["target_issued"]]
    assert issued
    assert all(row["desired_forecast_volatility"] <= 0.12 + 1e-10 for row in issued)
    assert all(row["desired_bil_weight"] + 1e-12 >= row["source_bil_weight"] for row in diagnostics)
    for frame in frames.values():
        active = frame.dropna(how="all")
        assert (active >= 0).all().all()
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_monthly_updates_emit_even_if_daily_risk_changes_are_inside_target_band(market, policy):
    frames, audit = transfer.build_from_baselines(market, sources(market), policy)
    for row in audit:
        if row["source_monthly_update"]:
            assert row["target_issued"] is True
        elif row["target_issued"] and row["target_change_from_previous_issued"] is not None:
            assert row["target_change_from_previous_issued"] + 1e-12 >= 0.05
    assert all(frame.isna().all(axis=1).any() for frame in frames.values())


def test_future_observations_cannot_change_previously_issued_covariance_targets(market, policy):
    baselines = sources(market)
    before, _ = transfer.build_from_baselines(market, baselines, policy)
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multiplier = np.linspace(1, 2, later.sum())
    close.loc[later, "SPY"] *= multiplier
    opening.loc[later, "SPY"] *= multiplier
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after, _ = transfer.build_from_baselines(changed, baselines, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_each_seeded_source_replays_with_original_funded_accounting_and_bt(market, policy, cost, delay):
    frames, _ = transfer.build_from_baselines(market, sources(market), policy)
    for frame in frames.values():
        targets = seed_window(frame, "2016-10-06")
        own = simulate(market, targets, "2016-10-06", "2017-06-30",
                       initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
        separate = independent_equity(market, targets, "2016-10-06", "2017-06-30",
                                     capital=10000, cost_bps=cost, commission=1, delay=delay)
        np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
        assert own.frame["cash"].min() >= 0


@pytest.mark.parametrize("problem", ["missing_source", "partial", "negative", "budget", "empty"])
def test_incomplete_monthly_sources_cannot_be_repaired_by_a_default_allocation(market, policy, problem):
    baselines = sources(market)
    identifier = transfer.CANDIDATES[0]["id"]
    active = baselines[identifier].dropna(how="all").index[0]
    if problem == "missing_source":
        baselines.pop(identifier)
    elif problem == "partial":
        baselines[identifier].iloc[0, 0] = 0.1
    elif problem == "negative":
        baselines[identifier].loc[active, "BIL"] = -0.1
    elif problem == "budget":
        baselines[identifier].loc[active] *= 2
    else:
        baselines[identifier][:] = np.nan
    with pytest.raises(QuantError):
        transfer.build_from_baselines(market, baselines, policy)


@pytest.mark.parametrize(
    "field,value", [("target_volatility", 0.10), ("target_change_band", 0.01),
                    ("volatility_sessions", 21), ("new_configurations", 10),
                    ("new_economic_factor_definitions", 1), ("cash_reserve", 0)],
)
def test_original_target_level_band_and_observation_window_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        transfer.validate(policy)
