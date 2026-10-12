from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.growth_portfolio_protection import monthly_targets, protect_target, run, validate_policy
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/growth-portfolio-protection.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS))


def test_growth_core_retains_meaningful_four_factor_budgets_without_embedded_leverage(market, policy):
    targets = monthly_targets(market, policy).dropna(how="all")
    equity = targets[[*FACTORS, "QQQ"]].sum(axis=1)
    np.testing.assert_allclose(targets["QQQ"], equity / 2)
    for symbol in FACTORS:
        np.testing.assert_allclose(targets[symbol], equity / 8)
    np.testing.assert_allclose(targets.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert targets[["SPY", "IEF", "TLT", "BIL"]].eq(0).all().all()
    assert targets.index[0] < pd.Timestamp("2016-10-05")


def test_floor_multiplier_is_derived_from_drawdown_goal_and_only_reduces_risk(market, policy):
    core = monthly_targets(market, policy).dropna(how="all").iloc[0]
    peak_target = protect_target(core, 10000, 10000, 0.15)
    pd.testing.assert_series_equal(peak_target, core)
    reduced = protect_target(core, 9500, 10000, 0.15)
    scale = (9500 - 8500) / (9500 * 0.15)
    np.testing.assert_allclose(reduced.drop("BIL"), core.drop("BIL") * scale)
    assert reduced["BIL"] == pytest.approx(0.98 * (1 - scale))
    floor = protect_target(core, 8500, 10000, 0.15)
    assert floor["BIL"] == pytest.approx(0.98)
    assert floor.drop("BIL").eq(0).all()
    assert reduced.sum() == pytest.approx(0.98)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_feedback_targets_and_independent_accounts_include_costs_and_delays(market, policy, cost, delay):
    baseline = monthly_targets(market, policy)
    for candidate in policy["candidates"]:
        result, targets, decisions = run(market, baseline, candidate, policy, "2016-10-06", "2017-06-30", cost, delay)
        independent = independent_equity(
            market, targets, "2016-10-06", "2017-06-30",
            capital=10000, cost_bps=cost, commission=1, delay=delay,
        )
        np.testing.assert_allclose(result.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
        assert result.frame["cash"].min() >= 0 and result.frame["cost"].sum() > 0
        if candidate["portfolio_insurance"]:
            assert decisions and decisions[0]["reason"] == "initial_capital"
            for before, after in zip(decisions, decisions[1:]):
                assert pd.Timestamp(before["execution_session"]) <= pd.Timestamp(after["signal_session"])
            active = targets.dropna(how="all")
            np.testing.assert_allclose(active.loc[:, list(FACTORS)].sum(axis=1), active["QQQ"])
            assert active["BIL"].max() > 0
            np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_future_closes_cannot_change_past_cost_dependent_protection_targets(market, policy):
    cutoff = pd.Timestamp("2017-06-30")
    candidate = policy["candidates"][1]
    before, old_targets, old_decisions = run(
        market, monthly_targets(market, policy), candidate, policy,
        "2016-10-06", "2018-01-31", 20, 2,
    )
    close, opening = market.close.copy(), market.open.copy()
    future = close.index > cutoff
    close.loc[future, "QQQ"] *= np.linspace(1, 1.4, future.sum())
    opening.loc[future, "QQQ"] *= np.linspace(1, 1.4, future.sum())
    changed = replace(market, close=close, raw_close=close.copy(), open=opening)
    after, new_targets, new_decisions = run(
        changed, monthly_targets(changed, policy), candidate, policy,
        "2016-10-06", "2018-01-31", 20, 2,
    )
    pd.testing.assert_frame_equal(before.frame.loc[:cutoff], after.frame.loc[:cutoff])
    pd.testing.assert_frame_equal(old_targets.loc[:cutoff], new_targets.loc[:cutoff])
    assert [row for row in old_decisions if row["signal_session"] <= str(cutoff.date())] == [
        row for row in new_decisions if row["signal_session"] <= str(cutoff.date())
    ]


def test_each_horizon_resets_actual_peak_and_initial_account_instead_of_inheriting_prior_returns(market, policy):
    own, targets, decisions = run(
        market, monthly_targets(market, policy), policy["candidates"][1], policy,
        "2017-01-03", "2017-06-30", 5, 1,
    )
    assert decisions[0]["known_nav_usd"] == decisions[0]["known_peak_usd"] == 10000
    assert decisions[0]["signal_session"] == "2016-12-30"
    assert decisions[0]["bil_target_weight"] == 0
    assert own.frame.index[0] == pd.Timestamp("2017-01-03")


def test_month_end_update_is_retained_when_a_previous_protection_request_is_pending(market, policy):
    end = "2016-11-10"
    close, opening = market.close.loc[:end].copy(), market.open.loc[:end].copy()
    after_anchor = close.index > pd.Timestamp("2016-10-05")
    initial = close.loc["2016-10-05"]
    close.loc[after_anchor] = np.outer(
        1.0002 ** np.arange(1, after_anchor.sum() + 1), initial
    )
    opening.loc[after_anchor] = np.outer(
        1.0002 ** np.arange(after_anchor.sum()), initial
    )
    affected = close.index >= pd.Timestamp("2016-10-28")
    for symbol in (*FACTORS, "QQQ", "GLD"):
        close.loc[affected, symbol] *= 0.8
        opening.loc[affected, symbol] *= 0.8
    shocked = replace(
        market, close=close, raw_close=close.copy(), open=opening,
        volume=market.volume.loc[:end], risk_free=market.risk_free.loc[:end],
    )
    own, targets, decisions = run(
        shocked, monthly_targets(shocked, policy), policy["candidates"][1], policy,
        "2016-10-06", "2016-11-10", 20, 2,
    )
    pending = next(row for row in decisions if row["signal_session"] == "2016-10-28")
    assert pending["execution_session"] == "2016-11-01"
    update = next(row for row in decisions if row["signal_session"] == "2016-11-01")
    assert update["reason"] == "monthly_target"
    assert update["execution_session"] == "2016-11-03"
    assert targets.loc["2016-10-31"].isna().all()
    nav = np.r_[10000, own.frame["equity"].to_numpy()]
    assert float(-(nav / np.maximum.accumulate(nav) - 1).min()) > 0.15


def test_registered_targets_regenerate_cost_and_delay_paths_not_a_cached_winner(market, policy):
    source = ROOT / "src/us_quant/growth_portfolio_protection.py"
    candidate = policy["candidates"][1]
    paths = (
        source, ROOT / "config/growth-portfolio-protection.json",
        source.with_name("growth_factor_satellite.py"),
        ROOT / "config/growth-factor-satellite.json",
        source.with_name("multifactor_stability.py"),
        source.with_name("cash_funded_accounting_v2.py"),
    )
    spec = {
        "id": candidate["id"], "configuration": candidate,
        "frozen_files": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in paths},
    }
    cache = {}
    for cost, delay in ((5, 1), (20, 2)):
        targets = registered_targets(spec, market, "2016-10-06", "2017-06-30", cost, delay, ROOT, cache)
        expected = run(
            market, monthly_targets(market, policy), candidate, policy,
            "2016-10-06", "2017-06-30", cost, delay,
        )[1]
        pd.testing.assert_frame_equal(targets, expected)


def test_no_post_outcome_protection_multiplier_or_growth_fraction_search(policy):
    policy["growth_share_of_equity"] = 0.75
    with pytest.raises(QuantError):
        validate_policy(policy)
