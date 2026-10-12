from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.factor_gold_risk import monthly_targets
from us_quant.momentum_implementation import (
    BASELINE,
    build_targets,
    issuer_definition,
    validate_policy,
)
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/momentum-implementation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FACTORS, "PDP"))


def product_html(ticker="PDP"):
    metadata = {
        "ticker": ticker,
        "isin": "US46137V8375",
        "shareClassInceptionDate": "2007-03-01",
        "assetClass": "Equity",
        "region": "United States",
        "investmentMethod": "Passive",
        "shareClassFullName": "Invesco Dorsey Wright Momentum ETF",
    }
    tags = "".join(f'<meta name="{key}" content="{value}">' for key, value in metadata.items())
    description = {
        "description": "An index fund tracks the Technical Leaders Index: 100 US companies, relative strength, quarterly."
    }
    return tags + f'<div data-model-json="{escape(json.dumps(description), quote=True)}"></div>'


def test_identity_requires_actual_metadata_and_fund_specific_definition():
    result = issuer_definition(product_html())
    assert result["ticker"] == "PDP" and result["index_code"] == "DWTL"
    assert not result["full_historical_methodology_verified"]
    with pytest.raises(QuantError, match="issuer"):
        issuer_definition(product_html("SPMO"))
    with pytest.raises(QuantError, match="issuer"):
        issuer_definition(product_html().replace("relative strength", "unrelated"))
    with pytest.raises(QuantError, match="valid JSON"):
        issuer_definition('<div data-model-json="broken"></div>')


def test_family_budgets_and_equity_gold_totals_are_identical(market, policy):
    prior = replace(
        market,
        open=market.open.drop(columns="PDP"),
        close=market.close.drop(columns="PDP"),
        raw_close=market.raw_close.drop(columns="PDP"),
        volume=market.volume.drop(columns="PDP"),
    )
    baseline = monthly_targets(prior, read_json(BASELINE)).dropna(how="all")
    for candidate in policy["candidates"]:
        target = build_targets(market, policy)[candidate["id"]].dropna(how="all")
        unchanged = ["SPY", "IEF", "GLD", "BIL", "VLUE", "QUAL", "USMV"]
        pd.testing.assert_frame_equal(target[unchanged], baseline[unchanged])
        np.testing.assert_allclose(target[["MTUM", "PDP"]].sum(axis=1), baseline["MTUM"])
        equity = target[[*FACTORS, "PDP"]].sum(axis=1)
        np.testing.assert_allclose(
            target["PDP"] / equity, 0.25 * candidate["pdp_share_of_momentum"]
        )
        np.testing.assert_allclose(target.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (target >= 0).all().all()
        assert (target[["VLUE", "QUAL", "USMV", "PDP"]] > 0).all().all()


def test_pdp_future_returns_cannot_change_past_or_baseline_risk_budgets(market, policy):
    before = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > cutoff
    closing.loc[later, "PDP"] *= np.linspace(1, 1.4, later.sum())
    opening.loc[later, "PDP"] *= np.linspace(1, 1.4, later.sum())
    after = build_targets(
        replace(market, close=closing, raw_close=closing.copy(), open=opening), policy
    )
    for name in before:
        pd.testing.assert_frame_equal(before[name], after[name])


def test_future_original_factor_prices_cannot_change_prior_targets(market, policy):
    before = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    closing, opening = market.close.copy(), market.open.copy()
    later = closing.index > cutoff
    closing.loc[later, "QUAL"] *= np.linspace(1, 1.4, later.sum())
    opening.loc[later, "QUAL"] *= np.linspace(1, 1.4, later.sum())
    after = build_targets(
        replace(market, close=closing, raw_close=closing.copy(), open=opening), policy
    )
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], after[name].loc[:cutoff])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_source_bound_targets_reproduce_independent_accounts(market, policy, cost, delay):
    source = ROOT / "src/us_quant/momentum_implementation.py"
    config = ROOT / "config/momentum-implementation.json"
    candidate = policy["candidates"][1]
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
                source.with_name("factor_replication.py"),
                source.with_name("multifactor_stability.py"),
            )
        },
    }
    start, end = "2016-10-06", "2017-03-31"
    target = registered_targets(spec, market, start, end, cost, delay, ROOT, {})
    pd.testing.assert_frame_equal(
        target, seed_window(build_targets(market, policy)[candidate["id"]], start)
    )
    actual = simulate(
        market, target, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, target, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(actual.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert actual.frame["cost"].sum() > 0 and actual.frame["cash"].min() >= 0


def test_fixed_family_mix_and_window_cannot_be_retuned(policy):
    policy["candidates"][1]["pdp_share_of_momentum"] = 0.75
    with pytest.raises(QuantError):
        validate_policy(policy)
