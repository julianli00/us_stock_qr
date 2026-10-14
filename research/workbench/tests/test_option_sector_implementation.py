from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import option_sector_implementation as study
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/option-sector-implementation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "TLT", "GLD", "BIL", *study.FACTORS, "QQQ", "XLK", "SOXX"))


def original(data):
    return sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]


def test_only_the_original_industry_sleeve_moves_with_no_new_weight_grid(market, policy):
    source = original(market)
    targets = study.build_from_original(market, source, policy)
    for candidate in study.CANDIDATES:
        actual = targets[candidate["id"]]
        name = candidate["replacement_symbol"]
        pd.testing.assert_frame_equal(actual[actual.columns.drop(["SOXX", name])],
                                      source[source.columns.drop(["SOXX", name])])
        pd.testing.assert_series_equal(actual[name], source["SOXX"], check_names=False)
        assert actual.dropna(how="all")["SOXX"].eq(0).all()
        assert actual.isna().equals(source.isna()) and len(actual.dropna(how="all")) == 131
        np.testing.assert_allclose(actual.dropna(how="all").sum(axis=1), .98, rtol=0, atol=1e-12)


@pytest.mark.parametrize("problem", ["missing_month", "partial", "negative", "nonfinite", "preallocated", "columns"])
def test_incomplete_or_changed_original_targets_are_fatal(market, policy, problem):
    source = original(market)
    day = source.dropna(how="all").index[0]
    if problem == "missing_month":
        source.loc[day] = np.nan
    elif problem == "partial":
        source.loc[day, "QQQ"] = np.nan
    elif problem == "negative":
        source.loc[day, "QQQ"] = -.1
    elif problem == "nonfinite":
        source.loc[day, "QQQ"] = np.inf
    elif problem == "preallocated":
        source.loc[day, "XLK"] = .1
    else:
        source = source.drop(columns="XLK")
    with pytest.raises(QuantError):
        study.build_from_original(market, source, policy)


@pytest.mark.parametrize("field,value", [
    ("approved_round_configuration_cap", 8), ("approved_round_batch_cap", 4),
    ("batch_number", 1), ("cash_reserve", 0.0), ("actual_weight_trade_band", .05),
    ("new_economic_factor_definitions", 2), ("data_start", "2017-01-01"),
])
def test_round_limits_and_original_risk_contract_cannot_change(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        study.validate(policy)


def test_an_arbitrary_sector_or_extra_share_cannot_be_smuggled_into_this_batch(policy):
    policy["candidates"][0]["replacement_symbol"] = "SPY"
    with pytest.raises(QuantError):
        study.validate(policy)


def test_generator_requires_the_complete_original_allocation_and_execution_graph():
    candidate = study.CANDIDATES[0]
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator({
            "id": candidate["id"], "configuration": candidate,
            "frozen_files": {"src/us_quant/option_sector_implementation.py": "0" * 64,
                             "config/option-sector-implementation.json": "0" * 64},
        }, ROOT)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_funded_feedback_execution_matches_independent_bt(market, policy, cost, delay):
    candidate = study.CANDIDATES[0]
    target = study.build_from_original(market, original(market), policy)[candidate["id"]]
    result, issued, decisions = study.run(
        market, target, candidate, policy, "2021-10-06", "2026-10-05", cost, delay,
    )
    separate = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                 capital=10000, cost_bps=cost, commission=1, delay=delay)
    np.testing.assert_allclose(result.frame["equity"], separate["equity"], rtol=0, atol=1e-8)
    assert len(result.frame) == 1254 and decisions[0]["initial_capital"]
    assert (result.frame["cash"] >= 0).all()
