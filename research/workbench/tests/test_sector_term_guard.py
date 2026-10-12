from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.multifactor_stability import FACTORS
from us_quant.sector_growth_balance import SECTORS
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.sector_term_guard import CANDIDATES, build_from_inputs, validate_policy
from us_quant.storage import file_digest, read_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/sector-term-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10",
        "2017-06-30",
        ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, *SECTORS),
    )


def inputs(market, dollar_level=-1.0):
    monthly = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    return (
        pd.DataFrame({"VIX": 15.0, "VIX3M": 20.0}, index=market.close.index),
        pd.Series(dollar_level, index=monthly.dropna(how="all").index),
        monthly,
    )


def test_daily_inversion_at_equal_threshold_halves_all_risk_not_only_equity(market, policy):
    terms, dollar, monthly = inputs(market)
    terms.loc["2016-10-10", "VIX"] = 20.0
    outputs = build_from_inputs(market, terms, dollar, policy)
    baseline = monthly.ffill()
    for target in outputs.values():
        assert target.loc["2016-10-10", "BIL"] == pytest.approx(0.49)
        assets = market.close.columns.drop("BIL")
        np.testing.assert_allclose(
            target.loc["2016-10-10", assets], baseline.loc["2016-10-10", assets] / 2
        )
        np.testing.assert_allclose(target.loc["2016-10-11"], baseline.loc["2016-10-11"])
        assert target.loc["2016-10-12"].isna().all()
        assert (target.dropna(how="all").loc[:, [*FACTORS, "SOXX", "GLD"]] > 0).all().all()
        np.testing.assert_allclose(target.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_monthly_dollar_state_persists_and_or_guard_does_not_churn_on_inversion_reversals(
    market, policy
):
    terms, dollar, monthly = inputs(market, 1.0)
    terms.loc["2016-10-10", "VIX"] = 25
    outputs = build_from_inputs(market, terms, dollar, policy)
    combined = outputs[CANDIDATES[1]["id"]]
    assert combined.loc["2016-10-10"].isna().all()
    assert combined.loc["2016-10-11"].isna().all()
    assert combined.dropna(how="all").index.equals(monthly.dropna(how="all").index)
    np.testing.assert_allclose(combined.dropna(how="all")["BIL"], 0.49)
    assert outputs[CANDIDATES[0]["id"]].loc["2016-10-10"].notna().all()


def test_a_new_month_updates_the_budget_even_without_a_risk_state_transition(market, policy):
    terms, dollar, monthly = inputs(market, 1.0)
    target = build_from_inputs(market, terms, dollar, policy)[CANDIDATES[1]["id"]]
    assert target.loc["2016-10-31"].notna().all()
    np.testing.assert_allclose(
        target.loc["2016-10-31", market.close.columns.drop("BIL")],
        monthly.loc["2016-10-31", market.close.columns.drop("BIL")] / 2,
    )


def test_one_day_inversion_and_reversal_remain_two_dated_pending_targets(market, policy):
    terms, dollar, monthly = inputs(market)
    terms.loc["2016-10-10", "VIX"] = 20
    changed = build_from_inputs(market, terms, dollar, policy)[CANDIDATES[0]["id"]]
    assert changed.loc["2016-10-10", "BIL"] == pytest.approx(0.49)
    assert changed.loc["2016-10-11", "BIL"] == 0
    start, end = "2016-10-06", "2016-10-14"
    actual = simulate(
        market,
        seed_window(changed, start),
        start,
        end,
        initial_capital=10000,
        cost_bps=20,
        commission=1,
        delay=2,
    )
    control = simulate(
        market,
        seed_window(monthly, start),
        start,
        end,
        initial_capital=10000,
        cost_bps=20,
        commission=1,
        delay=2,
    )
    pd.testing.assert_frame_equal(actual.frame.loc[:"2016-10-11"], control.frame.loc[:"2016-10-11"])
    assert actual.frame.loc["2016-10-12", "orders"] > 0
    assert actual.weights.loc["2016-10-12", "BIL"] > 0.45
    assert actual.frame.loc["2016-10-13", "orders"] > 0
    assert actual.weights.loc["2016-10-13", "BIL"] < 0.01


def test_future_option_dollar_or_sector_inputs_cannot_change_past_targets(market, policy):
    terms, dollar, monthly = inputs(market)
    before = build_from_inputs(market, terms, dollar, policy)
    cutoff = pd.Timestamp("2017-01-31")
    terms.loc[terms.index > cutoff, "VIX"] *= 3
    dollar.loc[dollar.index > cutoff] *= -1
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    close.loc[later, "SOXX"] *= np.linspace(1, 2, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1, 2, later.sum())
    after = build_from_inputs(
        replace(market, close=close, open=opening, raw_close=close.copy()), terms, dollar, policy
    )
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], after[name].loc[:cutoff])


@pytest.mark.parametrize("problem", ["missing_day", "zero", "nan", "monthly_date"])
def test_term_or_monthly_inputs_cannot_be_filled_or_substituted(market, policy, problem):
    terms, dollar, monthly = inputs(market)
    if problem == "missing_day":
        terms = terms.iloc[1:]
    elif problem == "zero":
        terms.iloc[-1, 0] = 0
    elif problem == "nan":
        terms.iloc[-1, 0] = np.nan
    else:
        dollar = dollar.iloc[1:]
    with pytest.raises(QuantError, match="complete official"):
        build_from_inputs(market, terms, dollar, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("option_ratio_risk_threshold", 0.95),
        ("risk_off_investment_scale", 0.75),
        ("cash_reserve", 0.0),
        ("target_event_policy", "daily_rebalance"),
        ("new_economic_factor_definitions", 1),
    ],
)
def test_ratio_scale_event_frequency_and_factor_count_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_registered_complete_target_replay_and_independent_funding(
    market, policy, monkeypatch, tmp_path, cost, delay
):
    terms, dollar, monthly = inputs(market)
    terms.loc["2016-10-10", "VIX"] = 20.0
    monkeypatch.setattr(
        "us_quant.sector_term_guard.load_inputs", lambda data, config: (terms, dollar, None)
    )
    dependencies = (
        "src/us_quant/sector_term_guard.py",
        "config/sector-term-guard.json",
        "src/us_quant/volatility_term_risk.py",
        "config/volatility-term-risk.json",
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
        "data/sector-term-source-20261011/verified-manifest.json",
        "data/volatility-term-source-20261010/manifest.json",
        "data/volatility-term-source-20261010/VIX.csv",
        "data/volatility-term-source-20261010/VIX3M.csv",
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        write_text_atomic(tmp_path / name, "synthetic source seal,not actual market data")
        frozen[name] = file_digest(tmp_path / name)
    candidate = policy["candidates"][0]
    spec = {"id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
    start, end = "2016-10-06", "2017-03-31"
    targets = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    pd.testing.assert_frame_equal(
        targets,
        seed_window(build_from_inputs(market, terms, dollar, policy)[candidate["id"]], start),
    )
    actual = simulate(
        market, targets, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(actual.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert actual.frame["cash"].min() >= 0
    missing = {
        **spec,
        "frozen_files": {
            name: value for name, value in frozen.items() if not name.endswith("VIX3M.csv")
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(missing, market, start, end, cost, delay, tmp_path, {})
