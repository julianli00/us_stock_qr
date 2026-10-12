from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import producer_gold_routing as routing
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, previous_session
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.research_program import verified_etf_market
from us_quant.storage import file_digest, read_json, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/producer-gold-routing.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", (*routing.sources.ORIGINAL, *routing.PRODUCERS))


def source(data):
    original = pd.DataFrame(np.nan, index=data.close.index, columns=routing.sources.ORIGINAL)
    released = pd.Series(np.nan, index=original.index, name="gold_released_weight")
    for position, day in enumerate(original.index):
        if position >= 252 and is_month_end(day):
            original.loc[day] = 0.0
            original.loc[day, list(routing.FACTORS)] = 0.98 * 0.6 / 8
            original.loc[day, "SOXX"] = 0.98 * 0.6 / 2
            original.loc[day, ["GLD", "BIL"]] = 0.98 * 0.4 / 2
            released.loc[day] = 0.98 * 0.4 / 2
    return original, released


def row():
    result = pd.Series(0.0, index=(*routing.sources.ORIGINAL, *routing.PRODUCERS))
    result.loc[list(routing.FACTORS)] = 0.02
    result["SOXX"], result["GLD"], result["BIL"] = 0.10, 0.20, 0.60
    return result


def prices(market, xle=120.0, xlb=110.0, bil=102.0):
    index = market.close.index[:127]
    return pd.DataFrame({name:np.linspace(100.0, end, len(index))
                         for name, end in (("XLE", xle), ("XLB", xlb), ("BIL", bil))}, index=index)


@pytest.mark.parametrize("method", [row["producer_method"] for row in routing.CANDIDATES])
def test_strict_relative_eligibility_leaves_original_bil_when_no_producer_qualifies(market, method):
    shares, audit = routing.producer_shares(prices(market, xle=102, xlb=90), method)
    assert audit["eligible_producers"] == []
    pd.testing.assert_series_equal(shares, pd.Series([0.0, 0.0, 1.0], index=["XLE", "XLB", "BIL"]))
    pd.testing.assert_series_equal(routing.route_gold_release(row(), 0.60, shares), row())


def test_winner_uses_actual_return_then_lexical_ties_not_asset_order(market):
    shares, _ = routing.producer_shares(prices(market), "highest_positive_excess")
    assert shares["XLE"] == 1.0
    shares, _ = routing.producer_shares(prices(market, xle=120, xlb=120), "highest_positive_excess")
    assert shares["XLB"] == 1.0


def test_diversified_is_prior63_sample_inverse_volatility_for_every_eligible_fund(market):
    history = prices(market)
    shares, _ = routing.producer_shares(history, "all_positive_inverse_volatility")
    volatility = history[["XLE", "XLB"]].pct_change(fill_method=None).iloc[1:].tail(63).std(ddof=1)
    inverse = 1 / volatility
    pd.testing.assert_series_equal(shares.loc[["XLE", "XLB"]], inverse / inverse.sum(), check_names=False)
    assert shares["BIL"] == 0.0


def test_zero_eligible_volatility_is_fatal_not_a_default_allocation(market):
    history = prices(market, bil=100)
    history["XLE"] = 100 * 1.001 ** np.arange(len(history))
    with pytest.raises(QuantError, match="degenerate"):
        routing.producer_shares(history, "all_positive_inverse_volatility")


@pytest.mark.parametrize("problem", ["short", "missing", "duplicate", "nonfinite", "zero", "method"])
def test_bad_selector_inputs_never_become_a_cash_success(market, problem):
    history, method = prices(market), "highest_positive_excess"
    if problem == "short":
        history = history.iloc[1:]
    elif problem == "missing":
        history = history.drop(columns="XLB")
    elif problem == "duplicate":
        history.index = [history.index[0]] * len(history)
    elif problem == "nonfinite":
        history.iloc[-1, 0] = np.nan
    elif problem == "zero":
        history.iloc[-1, 0] = 0
    else:
        method = "best_after_costs"
    with pytest.raises(QuantError):
        routing.producer_shares(history, method)


@pytest.mark.parametrize("released,expected", [(0.0, 0.0), (0.20, 0.20), (0.60, 0.40)])
def test_only_gold_release_is_routed_and_cap_residual_and_other_budgets_are_preserved(released, expected):
    original = row()
    shares = pd.Series([0.25, 0.75, 0.0], index=["XLE", "XLB", "BIL"])
    result = routing.route_gold_release(original, released, shares)
    assert result[["XLE", "XLB"]].sum() == expected
    assert result["BIL"] == original["BIL"] - expected
    unchanged = original.index.difference(["XLE", "XLB", "BIL"])
    pd.testing.assert_series_equal(result.loc[unchanged], original.loc[unchanged])
    assert np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)


def test_diversified_rounding_cannot_borrow_when_bil_contains_only_gold_release():
    original = row()
    original["BIL"], original["GLD"] = 0.20, 0.60
    shares = pd.Series([0.3333333333333333, 0.6666666666666667, 0.0], index=["XLE", "XLB", "BIL"])
    result = routing.route_gold_release(original, 0.20, shares)
    assert result["BIL"] == 0.0
    assert result.min() >= 0.0


@pytest.mark.parametrize("released", [-0.01, 0.61, np.nan])
def test_unproven_or_equity_standby_cash_cannot_be_routed(released):
    with pytest.raises(QuantError):
        routing.route_gold_release(row(), released, pd.Series([1.0, 0.0, 0.0], index=["XLE", "XLB", "BIL"]))


def test_complete_original_core_gold_and_month_dates_are_exact(market, policy):
    original, released = source(market)
    frames, _ = routing.build_from_source(market, original, released, policy)
    for frame in frames.values():
        assert frame.dropna(how="all").index.equals(original.dropna(how="all").index)
        unchanged = original.columns.difference(["BIL"])
        pd.testing.assert_frame_equal(frame.loc[:, unchanged], original.loc[:, unchanged])
        assert (frame.dropna(how="all")["BIL"] >=
                original.dropna(how="all")["BIL"] - released.dropna() - 1e-12).all()


def test_future_producer_prices_cannot_change_prior_selectors_targets_or_feedback(market, policy):
    original, released = source(market)
    before = routing.build_from_source(market, original, released, policy)[0][routing.CANDIDATES[0]["id"]]
    boundary = pd.Timestamp("2017-01-31")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multiplier = np.linspace(1.0, 1.5, later.sum())
    close.loc[later, "XLE"] *= multiplier
    opening.loc[later, "XLE"] *= multiplier
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after = routing.build_from_source(changed, original, released, policy)[0][routing.CANDIDATES[0]["id"]]
    pd.testing.assert_frame_equal(before.loc[:boundary], after.loc[:boundary])
    result_before, issued_before, _ = routing.run(market, before, routing.CANDIDATES[0], policy,
                                                   "2016-10-06", "2017-06-30", 20.0, 2)
    result_after, issued_after, _ = routing.run(changed, after, routing.CANDIDATES[0], policy,
                                                "2016-10-06", "2017-06-30", 20.0, 2)
    pd.testing.assert_frame_equal(result_before.frame.loc[:boundary], result_after.frame.loc[:boundary])
    pd.testing.assert_frame_equal(issued_before.loc[:boundary], issued_after.loc[:boundary])


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_fourteen_asset_holding_band_replays_through_funded_accounting_and_independent_bt(market, policy, cost, delay):
    original, released = source(market)
    desired = routing.build_from_source(market, original, released, policy)[0][routing.CANDIDATES[0]["id"]]
    own, issued, decisions = routing.run(market, desired, routing.CANDIDATES[0], policy,
                                          "2016-10-06", "2017-06-30", cost, delay)
    separate = simulate(market, issued, "2016-10-06", "2017-06-30",
                        initial_capital=10000, cost_bps=cost, commission=1, delay=delay)
    audit = independent_equity(market, issued, "2016-10-06", "2017-06-30",
                              capital=10000, cost_bps=cost, commission=1, delay=delay)
    pd.testing.assert_frame_equal(own.frame, separate.frame)
    pd.testing.assert_frame_equal(own.weights, separate.weights)
    np.testing.assert_allclose(own.frame["equity"], audit["equity"], rtol=0, atol=1e-8)
    assert len(own.weights.columns) == 14
    assert decisions[0]["reason"] == "initial_capital"
    anchor = previous_session(pd.Timestamp("2016-10-06"))
    pd.testing.assert_series_equal(issued.loc[anchor], desired.loc[:anchor].dropna(how="all").iloc[-1],
                                   check_names=False)


@pytest.mark.parametrize("field,value", [("source_id", "later_winning_source"), ("momentum_sessions", 63),
                                        ("standalone_producer_cap", 0.60), ("cash_origin", "all_bil"),
                                        ("producer_assets", ["XLE"]), ("new_economic_factor_definitions", 2)])
def test_frozen_source_budget_subset_or_windows_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        routing.validate(policy)


def test_source_only_preparation_is_not_an_admission_adapter():
    with pytest.raises(QuantError, match="explicitly audited"):
        verified_etf_market({"verified_etf_source":{"adapter":"producer_sector_sources_20261011"}}, ROOT)


@pytest.mark.parametrize("wrong_manifest", [False, True])
def test_finite_admission_dispatch_binds_exact_producer_manifest(tmp_path, monkeypatch, market, policy, wrong_manifest):
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"synthetic_source":True})
    policy["market_source"] = {"path":manifest_path.name, "sha256":file_digest(manifest_path)}
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    checked = []
    monkeypatch.setattr(routing, "verified_market", lambda value, root: checked.append((value, root)) or market)
    source = {"adapter":"producer_gold_routing_20261011", "policy":policy_path.name,
              "policy_sha256":file_digest(policy_path), "producer_manifest":manifest_path.name,
              "producer_manifest_sha256":"0" * 64 if wrong_manifest else file_digest(manifest_path)}
    if wrong_manifest:
        with pytest.raises(QuantError, match="exact independently verified"):
            verified_etf_market({"verified_etf_source":source}, tmp_path)
        assert checked == []
    else:
        assert verified_etf_market({"verified_etf_source":source}, tmp_path) is market
        assert checked == [(policy, tmp_path)]


def test_finite_generator_requires_complete_source_and_original_execution_dependencies():
    spec = {"id":routing.CANDIDATES[0]["id"], "configuration":routing.CANDIDATES[0],
            "frozen_files":{"src/us_quant/producer_gold_routing.py":"0" * 64,
                            "config/producer-gold-routing.json":"0" * 64}}
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)
