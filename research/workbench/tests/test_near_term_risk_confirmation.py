from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import near_term_risk_confirmation as guard
from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/near-term-risk-confirmation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def inputs(data, near=25.0, far=20.0, three_month=30.0, dollar=0.1):
    closes = pd.DataFrame({"VIX9D":near, "VIX":far, "VIX3M":three_month}, index=data.close.index)
    usd = pd.Series(dollar, index=original(data).dropna(how="all").index, name="dollar_change")
    return closes, usd


def test_actual_nine_day_and_thirty_day_legs_are_used_not_the_old_thirty_three_month_pair(market, policy):
    closes, usd = inputs(market)
    source = original(market)
    for frame in guard.build_from_inputs(market, closes, usd, policy).values():
        active = source.dropna(how="all").index
        assets = source.columns.drop("BIL")
        assert len(frame.dropna(how="all")) == 131
        np.testing.assert_allclose(frame.loc[active, assets], source.loc[active, assets]/2)
        np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)


@pytest.mark.parametrize("near,far,three_month,dollar", [(10.0, 20.0, 15.0, 0.1), (25.0, 20.0, 30.0, 0.0),
                                                        (25.0, 20.0, 30.0, -0.1)])
def test_nontrigger_or_nonrising_dollar_keeps_original_even_when_old_pair_is_inverted(market, policy, near, far, three_month, dollar):
    closes, usd = inputs(market, near, far, three_month, dollar)
    for frame in guard.build_from_inputs(market, closes, usd, policy).values():
        pd.testing.assert_frame_equal(frame, original(market))


def test_equality_uses_the_original_option_threshold_and_halves_risk(market, policy):
    closes, usd = inputs(market, near=20.0, far=20.0)
    frames = guard.build_from_inputs(market, closes, usd, policy)
    active = original(market).dropna(how="all").index
    for frame in frames.values():
        np.testing.assert_allclose(frame.loc[active, "BIL"], 0.49, atol=1e-12)


def test_monthly_dollar_state_is_carried_but_option_changes_emit_only_daily_events(market, policy):
    closes, usd = inputs(market, near=10.0)
    month = usd.index[10]
    following = market.close.index[market.close.index.get_loc(month)+1]
    closes.loc[following, "VIX9D"] = 30.0
    frames = guard.build_from_inputs(market, closes, usd, policy)
    monthly, daily = (frames[row["id"]] for row in guard.CANDIDATES)
    assert monthly.loc[following].isna().all()
    assert daily.loc[following, "BIL"] == pytest.approx(0.49)
    after = market.close.index[market.close.index.get_loc(following)+1]
    assert daily.loc[after, "BIL"] == 0.0
    assert len(daily.dropna(how="all")) == len(monthly.dropna(how="all"))+2
    usd.loc[month] = -0.1
    off = guard.build_from_inputs(market, closes, usd, policy)[guard.CANDIDATES[1]["id"]]
    assert off.loc[following].isna().all()


@pytest.mark.parametrize("problem", ["missing_day", "wrong_fields", "zero", "nan", "string", "missing_usd"])
def test_missing_or_misidentified_inputs_never_default_to_a_safe_state(market, policy, problem):
    closes, usd = inputs(market)
    if problem == "missing_day":
        closes = closes.iloc[1:]
    elif problem == "wrong_fields":
        closes = closes.rename(columns={"VIX9D":"VXST"})
    elif problem in ("zero", "nan"):
        closes.iloc[0, 0] = 0.0 if problem == "zero" else np.nan
    elif problem == "string":
        closes["VIX9D"] = "missing"
    else:
        usd = usd.iloc[1:]
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, closes, usd, policy)


def test_future_prices_and_option_closes_cannot_change_previous_requests(market, policy):
    closes, usd = inputs(market)
    before = guard.build_from_inputs(market, closes, usd, policy)
    boundary = pd.Timestamp("2021-12-31")
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > boundary
    closing.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1.0, 1.4, later.sum())
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    changed_closes = closes.copy()
    changed_closes.loc[later, "VIX9D"] = 5.0
    after = guard.build_from_inputs(changed, changed_closes, usd, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_dated_on_off_requests_keep_their_queue_and_funded_bt_calendar(market, policy, cost, delay):
    closes, usd = inputs(market)
    closes.iloc[::7, 0] = 10.0
    desired = guard.build_from_inputs(market, closes, usd, policy)[guard.CANDIDATES[1]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert len(desired.dropna(how="all")) > 131


@pytest.mark.parametrize("field,value", [("option_comparison", ["VIX", "VIX3M"]),
                                        ("primitive_role_mapping", {"VIX":"VIX", "VIX3M":"VIX3M"}),
                                        ("risk_off_investment_scale", 0.75), ("option_ratio_risk_threshold", 1.1),
                                        ("new_economic_factor_definitions", 2)])
def test_frozen_horizons_role_adapter_scale_and_thresholds_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_generator_requires_complete_genuine_source_and_existing_event_helper_graph():
    spec = {"id":guard.CANDIDATES[0]["id"], "configuration":guard.CANDIDATES[0],
            "frozen_files":{"src/us_quant/near_term_risk_confirmation.py":"0"*64,
                            "config/near-term-risk-confirmation.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
