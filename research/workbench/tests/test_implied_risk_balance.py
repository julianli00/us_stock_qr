from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import implied_risk_balance as guard
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
    return read_json(ROOT / "config/implied-risk-balance.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "GLD", "BIL", *guard.FACTORS, "TLT", "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def inputs(data, gold=20.0, equity=20.0, dollar=0.0):
    index = original(data).dropna(how="all").index
    return pd.DataFrame({"GVZ":gold, "VIX":equity}, index=index), pd.Series(dollar, index=index, name="dollar_change")


@pytest.mark.parametrize("gold,equity,share", [(20.0, 20.0, 0.5), (30.0, 10.0, 0.7), (10.0, 30.0, 0.3)])
def test_native_option_budget_is_clamped_and_preserves_all_relative_factor_satellite_weights(market, policy, gold, equity, share):
    options, dollar = inputs(market, gold, equity)
    for frame in guard.build_from_inputs(market, options, dollar, policy).values():
        active = frame.dropna(how="all")
        assert len(active) == 131 and frame.isna().equals(original(market).isna())
        np.testing.assert_allclose(active[list(guard.EQUITY)].sum(axis=1), 0.98*share, atol=1e-12)
        np.testing.assert_allclose(active["SOXX"], 0.98*share/2, atol=1e-12)
        np.testing.assert_allclose(active[list(guard.FACTORS)], 0.98*share/8, atol=1e-12)
        np.testing.assert_allclose(active["GLD"], 0.98*(1-share), atol=1e-12)
        assert active["BIL"].eq(0).all()
        assert active[[*guard.FACTORS, "SOXX", "GLD"]].gt(0).all().all()


def test_positive_original_dollar_guard_halves_nonbil_only_after_the_new_budget(market, policy):
    options, dollar = inputs(market, gold=15.0, equity=25.0, dollar=0.1)
    frames = guard.build_from_inputs(market, options, dollar, policy)
    plain, guarded = (frames[row["id"]] for row in guard.CANDIDATES)
    active = plain.dropna(how="all").index
    assets = plain.columns.drop("BIL")
    np.testing.assert_allclose(guarded.loc[active, assets], plain.loc[active, assets]/2, atol=1e-12)
    np.testing.assert_allclose(guarded.loc[active, "BIL"], 0.49, atol=1e-12)
    np.testing.assert_allclose(guarded.loc[active].sum(axis=1), 0.98, rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_dollar_equality_or_negative_sign_does_not_trigger(market, policy, value):
    options, dollar = inputs(market, dollar=value)
    frames = guard.build_from_inputs(market, options, dollar, policy)
    pd.testing.assert_frame_equal(frames[guard.CANDIDATES[0]["id"]], frames[guard.CANDIDATES[1]["id"]])


def test_common_unit_rescaling_leaves_risk_ratio_and_weights_unchanged(market, policy):
    options, dollar = inputs(market, gold=13.5, equity=22.5)
    before = guard.build_from_inputs(market, options, dollar, policy)
    after = guard.build_from_inputs(market, options*100, dollar, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier], after[identifier])


@pytest.mark.parametrize("problem", ["missing_month", "wrong_labels", "zero", "nan", "infinite", "string", "missing_dollar", "boolean_dollar"])
def test_monthly_missingness_or_invalid_native_quotes_never_default_to_cash_or_a_budget(market, policy, problem):
    options, dollar = inputs(market)
    if problem == "missing_month":
        options = options.iloc[1:]
    elif problem == "wrong_labels":
        options = options.rename(columns={"GVZ":"GLD"})
    elif problem in ("zero", "nan", "infinite"):
        options.iloc[0, 0] = {"zero":0.0, "nan":np.nan, "infinite":np.inf}[problem]
    elif problem == "string":
        options["GVZ"] = "missing"
    elif problem == "boolean_dollar":
        dollar = dollar.gt(-1)
    else:
        dollar = dollar.iloc[1:]
    with pytest.raises(QuantError):
        guard.build_from_inputs(market, options, dollar, policy)


def test_future_prices_or_option_proxies_cannot_change_earlier_monthly_budgets(market, policy):
    options, dollar = inputs(market)
    before = guard.build_from_inputs(market, options, dollar, policy)
    boundary = pd.Timestamp("2021-12-31")
    closing, opening = market.close.copy(), market.open.copy()
    late = closing.index > boundary
    closing.loc[late, "SOXX"] *= np.linspace(1.0, 1.4, late.sum())
    opening.loc[late, "SOXX"] *= np.linspace(1.0, 1.4, late.sum())
    changed = replace(market, close=closing, open=opening, raw_close=closing.copy())
    later_options = options.copy()
    later_options.loc[options.index > boundary, "GVZ"] = 50.0
    after = guard.build_from_inputs(changed, later_options, dollar, policy)
    for identifier in before:
        pd.testing.assert_frame_equal(before[identifier].loc[:boundary], after[identifier].loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_full_actual_returns_and_monthly_budget_requests_match_funded_and_bt(market, policy, cost, delay):
    options, dollar = inputs(market, dollar=0.1)
    options.iloc[::2, 0] = 15.0
    desired = guard.build_from_inputs(market, options, dollar, policy)[guard.CANDIDATES[1]["id"]]
    seeded = seed_window(desired, "2021-10-06")
    own = simulate(market, seeded, "2021-10-06", "2026-10-05",
                   initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    separate = independent_equity(market, seeded, "2021-10-06", "2026-10-05",
                                  capital=10000, cost_bps=cost, commission=1, delay=delay)
    assert len(own.frame) == 1254 and own.frame["cash"].ge(0).all()
    np.testing.assert_allclose(own.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert desired.loc[list(guard.inputs.MISSING)].isna().all().all()
    assert len(desired.dropna(how="all")) == 131


@pytest.mark.parametrize("field,value", [("equity_share_min", 0.2), ("equity_share_max", 0.8),
                                        ("growth_share_of_equity", 0.75), ("risk_off_investment_scale", 0.75),
                                        ("new_economic_factor_definitions", 2), ("budget_proxy_indices", ["GLD", "SPY"])])
def test_original_bounds_composition_and_proxy_source_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        guard.validate(policy)


def test_generator_requires_exact_monthly_source_gaps_and_all_original_helper_dependencies():
    spec = {"id":guard.CANDIDATES[0]["id"], "configuration":guard.CANDIDATES[0],
            "frozen_files":{"src/us_quant/implied_risk_balance.py":"0"*64,
                            "config/implied-risk-balance.json":"0"*64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
