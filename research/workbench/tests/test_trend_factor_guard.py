from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_targets
from us_quant.trend_factor_guard import BASELINE, build_targets, gated_target, indicators, validate_policy

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/trend-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def trending(market, equity_drift: float, gold_drift: float):
    close = market.close.copy()
    time = np.arange(len(close))
    for symbol in FACTORS:
        close[symbol] = 100 * np.exp(equity_drift * time + 0.005 * np.sin(time / 7))
    close["GLD"] = 100 * np.exp(gold_drift * time + 0.005 * np.sin(time / 7))
    close["BIL"] = 100 * np.exp(0.00004 * time)
    opening = close.shift(1).fillna(100) * 1.0001
    return replace(market, open=opening, close=close, raw_close=close.copy())


def test_only_removed_sleeve_budget_goes_to_actual_bills(market):
    baseline = monthly_targets(market, read_json(BASELINE)).dropna(how="all").iloc[-1]
    target = gated_target(baseline, (True, False))
    pd.testing.assert_series_equal(target.loc[list(FACTORS)], baseline.loc[list(FACTORS)])
    assert target["GLD"] == 0
    assert target["BIL"] == pytest.approx(baseline["GLD"])
    assert target.sum() == pytest.approx(0.98)
    cash = gated_target(baseline, (False, False))
    assert cash["BIL"] == pytest.approx(0.98)
    assert cash.drop("BIL").eq(0).all()


def test_joint_and_component_gates_are_distinct_and_preserve_active_factor_shares(market, policy):
    data = trending(market, 0.002, -0.001)
    targets = build_targets(data, policy)
    joint = targets[policy["candidates"][0]["id"]].dropna(how="all")
    component = targets[policy["candidates"][1]["id"]].dropna(how="all")
    assert joint.iloc[-1]["GLD"] > 0
    assert component.iloc[-1]["GLD"] == 0
    assert component.iloc[-1]["BIL"] > 0
    for frame in (joint, component):
        assert (frame >= 0).all().all()
        np.testing.assert_allclose(frame.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        eq = frame.loc[:, list(FACTORS)]
        active = eq.sum(axis=1) > 0
        np.testing.assert_allclose(eq.loc[active].div(eq.loc[active].sum(axis=1), axis=0), 0.25)


def test_two_falling_sleeves_go_to_bills_without_minimum_equity_exposure(market, policy):
    data = trending(market, -0.001, -0.001)
    for frame in build_targets(data, policy).values():
        active = frame.dropna(how="all")
        assert active["BIL"].eq(0.98).all()
        assert active.drop(columns="BIL").eq(0).all().all()


def test_indicator_uses_previous_close_targets_and_real_warmup(market):
    baseline = monthly_targets(market, read_json(BASELINE))
    original = indicators(market, baseline)
    first_target = baseline.dropna(how="all").index[0]
    assert original.loc[:first_target, "joint"].isna().all()
    changed = baseline.copy()
    changed.loc[first_target] = 0.0
    changed.loc[first_target, "BIL"] = 0.98
    modified = indicators(market, changed)
    pd.testing.assert_series_equal(original.loc[:first_target, "joint"], modified.loc[:first_target, "joint"])
    assert not original["joint"].equals(modified["joint"])


def test_no_daily_rebalance_when_gates_are_unchanged(market, policy):
    data = trending(market, 0.001, 0.001)
    for frame in build_targets(data, policy).values():
        active = frame.dropna(how="all")
        assert active.index[0] <= pd.Timestamp("2016-09-30")
        assert all(is_month_end(day) for day in active.index[1:])
        assert active["BIL"].eq(0).all()


def test_future_data_and_snapshot_truncation_preserve_prior_targets(market, policy):
    original = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    later = market.close.index > cutoff
    closing, opening, rates = market.close.copy(), market.open.copy(), market.risk_free.copy()
    closing.loc[later, ["GLD", "MTUM"]] *= np.linspace(1, 1.5, later.sum())[:, None]
    opening.loc[later, ["GLD", "MTUM"]] *= np.linspace(1, 1.5, later.sum())[:, None]
    rates.loc[later] += 0.0001
    changed = replace(market, open=opening, close=closing, raw_close=closing.copy(), risk_free=rates)
    updated = build_targets(changed, policy)
    truncated = replace(
        market,
        open=market.open.loc[:cutoff],
        close=market.close.loc[:cutoff],
        raw_close=market.raw_close.loc[:cutoff],
        volume=market.volume.loc[:cutoff],
        risk_free=market.risk_free.loc[:cutoff],
    )
    past = build_targets(truncated, policy)
    for name, frame in original.items():
        pd.testing.assert_frame_equal(frame.loc[:cutoff], updated[name].loc[:cutoff])
        pd.testing.assert_frame_equal(frame.loc[:cutoff], past[name])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_source_replay_preserves_delayed_execution_and_independent_accounting(market, policy, cost, delay):
    candidate = policy["candidates"][1]
    source = ROOT / "src/us_quant/trend_factor_guard.py"
    config = ROOT / "config/trend-factor-guard.json"
    spec = {
        "id": candidate["id"],
        "configuration": candidate,
        "frozen_files": {
            path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in (
                source,
                config,
                BASELINE,
                source.with_name("factor_gold_risk.py"),
                source.with_name("multifactor_stability.py"),
            )
        },
    }
    start, end = "2016-10-06", "2017-01-31"
    targets = registered_targets(spec, market, start, end, cost, delay, ROOT, {})
    pd.testing.assert_frame_equal(
        targets, seed_window(build_targets(market, policy)[candidate["id"]], start)
    )
    own = simulate(
        market, targets, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cash"].min() >= 0
    assert own.frame["cost"].sum() > 0
    if delay == 1:
        assert own.frame["orders"].iloc[0] > 0
    else:
        assert own.frame["orders"].iloc[0] == 0


def test_same_close_request_never_changes_that_sessions_account(market, policy):
    data = trending(market, 0.001, 0.001)
    signal = seed_window(build_targets(data, policy)[policy["candidates"][0]["id"]], "2016-10-06")
    day = pd.Timestamp("2016-10-18")
    changed = signal.copy()
    changed.loc[day] = 0.0
    changed.loc[day, "BIL"] = 0.98
    before = simulate(data, signal, "2016-10-06", "2016-11-01")
    after = simulate(data, changed, "2016-10-06", "2016-11-01")
    pd.testing.assert_frame_equal(before.frame.loc[:day], after.frame.loc[:day])
    assert not before.frame.loc[day + pd.Timedelta(days=1) :].equals(
        after.frame.loc[day + pd.Timedelta(days=1) :]
    )


def test_no_post_outcome_trend_window_changes(policy):
    policy["trend_sessions"] = 63
    with pytest.raises(QuantError):
        validate_policy(policy)
