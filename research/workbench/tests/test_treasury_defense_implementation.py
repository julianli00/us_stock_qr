from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import treasury_defense_implementation as study
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/treasury-defense-implementation.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2026-10-05",
                          ("SPY", "IEF", "TLT", "GLD", "BIL", *study.FACTORS, "QQQ", "XLK", "SOXX"))


def dollar(data, value=0.0):
    return pd.Series(value, index=study.monthly_index(data))


def test_correct_actual_treasury_tickers_and_equity_proportions(market, policy):
    outputs = study.build_from_inputs(market, dollar(market), policy)
    for candidate in study.CANDIDATES:
        target = outputs[candidate["id"]].dropna(how="all")
        defense = candidate["defensive_symbol"]
        assert len(target) == 131 and target["GLD"].eq(0).all()
        assert target[["QQQ", "XLK", "BIL"]].eq(0).all().all()
        np.testing.assert_allclose(target.sum(axis=1), .98, rtol=0, atol=1e-12)
        equity = target[["SOXX", *study.FACTORS]].sum(axis=1)
        np.testing.assert_allclose(target["SOXX"], equity * .5, rtol=0, atol=1e-12)
        np.testing.assert_allclose(target[list(study.FACTORS)].to_numpy(),
                                   np.repeat((equity * .125).to_numpy()[:, None], 4, axis=1))
        assert (target[defense] > 0).all()
        assert ((equity / .98) >= .30 - 1e-12).all() and ((equity / .98) <= .70 + 1e-12).all()


def test_strict_positive_dollar_halves_all_non_bil_funds(market, policy):
    plain = study.build_from_inputs(market, dollar(market, 0), policy)
    guarded = study.build_from_inputs(market, dollar(market, 1e-8), policy)
    declining = study.build_from_inputs(market, dollar(market, -.01), policy)
    for candidate in study.CANDIDATES:
        identifier = candidate["id"]
        pd.testing.assert_frame_equal(plain[identifier], declining[identifier])
        active = guarded[identifier].dropna(how="all")
        np.testing.assert_allclose(active["BIL"], .49, rtol=0, atol=1e-12)
        np.testing.assert_allclose(active.drop(columns="BIL"),
                                  plain[identifier].dropna(how="all").drop(columns="BIL") * .5,
                                  rtol=0, atol=1e-12)


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "wrong_calendar", "bool"])
def test_dollar_gaps_are_fatal_not_safe_defaults(market, policy, problem):
    values = dollar(market)
    if problem == "missing":
        values.iloc[0] = np.nan
    elif problem == "nonfinite":
        values.iloc[0] = np.inf
    elif problem == "wrong_calendar":
        values = values.iloc[1:]
    else:
        values = values.astype(bool)
    with pytest.raises(QuantError):
        study.build_from_inputs(market, values, policy)


@pytest.mark.parametrize("field,value", [
    ("volatility_sessions", 21), ("equity_share_min", .10), ("equity_share_max", .90),
    ("approved_round_configuration_cap", 8), ("batch_number", 4),
    ("actual_weight_trade_band", .05), ("cash_reserve", 0.0),
])
def test_no_parameter_grid_or_budget_extension(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        study.validate(policy)


def test_no_arbitrary_defensive_fund_substitution(policy):
    policy["candidates"][0]["defensive_symbol"] = "GLD"
    with pytest.raises(QuantError):
        study.validate(policy)


def test_complete_shared_source_and_risk_graph_is_required():
    candidate = study.CANDIDATES[0]
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator({
            "id": candidate["id"], "configuration": candidate,
            "frozen_files": {"src/us_quant/treasury_defense_implementation.py": "0" * 64,
                             "config/treasury-defense-implementation.json": "0" * 64},
        }, ROOT)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_band_funding_matches_independent_bt(market, policy, cost, delay):
    candidate = study.CANDIDATES[0]
    target = study.build_from_inputs(market, dollar(market), policy)[candidate["id"]]
    result, issued, _ = study.run(market, target, candidate, policy, "2021-10-06", "2026-10-05", cost, delay)
    independent = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                    capital=10000, cost_bps=cost, commission=1, delay=delay)
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert len(result.frame) == 1254 and (result.frame["cash"] >= 0).all()
