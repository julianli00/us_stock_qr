from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.multifactor_stability import (
    BASE,
    FACTORS,
    audit_path,
    bounded_factor_shares,
    build_targets,
    cap_volatility,
    corporate_actions,
    exposure_diagnostics,
    fetch,
    fingerprint,
    gate_result,
    load_market,
    register,
    target_weights,
    validate_policy,
    verify_registration,
)
from us_quant.storage import digest_json, read_json, write_json
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/multifactor-stability.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2014-01-02", "2016-02-29", (*BASE, *FACTORS))


@pytest.fixture
def registration(policy, tmp_path, monkeypatch):
    import us_quant.multifactor_stability as module

    old = read_json(ROOT / policy["comparison"]["old_data_registration"])
    monkeypatch.setattr(module, "snapshot_descriptor", lambda *args: old["sources"]["base"])
    return register(policy, tmp_path, tmp_path / "registration.json")


def test_distinct_economic_factors_not_multiple_momentum_windows(policy):
    validate_policy(policy)
    assert [f["factor"] for f in policy["factors"]] == [
        "momentum",
        "value",
        "quality",
        "low_volatility",
    ]
    assert all(f["intended_daily_leverage"] == 1 for f in policy["factors"])
    assert policy["methodology"]["implementation_type"].endswith("not_direct_stock_scoring")


@pytest.mark.parametrize(
    "key,value",
    [
        ("factor_symbols", ["QLD", "VLUE", "QUAL", "USMV"]),
        ("capital_usd", 100000),
        ("cash_reserve", 0),
        ("prior_disclosed_configurations", 0),
        ("as_of", "2026-10-09"),
        ("new_configurations", 5),
    ],
)
def test_no_silent_leverage_or_protocol_changes(policy, key, value):
    policy[key] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_risk_limit_and_fund_inception_cannot_be_relaxed(policy):
    original = deepcopy(policy)
    policy["goals"]["max_drawdown_at_most"] = 0.25
    with pytest.raises(QuantError):
        validate_policy(policy)
    original["factors"][2]["inception"] = "2015-01-01"
    with pytest.raises(QuantError):
        validate_policy(original)


@pytest.mark.parametrize(
    "name",
    [
        "independent_forward_validation",
        "automatic_baseline_replacement",
        "order_authority",
        "persistent_automation_started",
    ],
)
def test_no_trading_or_independent_forward_claim(policy, name):
    policy["methodology"][name] = True
    with pytest.raises(QuantError, match="authority"):
        validate_policy(policy)


def test_candidate_paths_and_missing_controls_are_rejected(policy):
    changed = deepcopy(policy)
    changed["candidates"][0]["id"] = "../escape"
    with pytest.raises(QuantError, match="identifier"):
        validate_policy(changed)
    policy["candidates"].pop()
    with pytest.raises(QuantError, match="six"):
        validate_policy(policy)


def test_bounded_risk_shares_keep_every_factor_and_prevent_domination():
    vol = pd.Series([0.005, 0.01, 0.3, 0.5], index=FACTORS)
    allocation = bounded_factor_shares(vol, 0.15, 0.35)
    assert allocation.sum() == pytest.approx(1)
    assert (allocation >= 0.15).all() and (allocation <= 0.35).all()
    np.testing.assert_allclose(allocation, [0.35, 0.35, 0.15, 0.15], atol=1e-10)
    with pytest.raises(QuantError):
        bounded_factor_shares(vol * 0, 0.15, 0.35)


def test_volatility_cap_scales_down_includes_bil_and_does_not_add_leverage():
    target = pd.Series({"MTUM": 0.49, "QUAL": 0.49, "BIL": 0.0})
    covariance = pd.DataFrame(
        np.diag([0.16, 0.16, 0.0001]), index=target.index, columns=target.index
    )
    capped = cap_volatility(target, covariance, 0.10, 0.98)
    assert capped.sum() == pytest.approx(0.98)
    assert capped["BIL"] > 0
    assert capped["MTUM"] / capped["QUAL"] == pytest.approx(1)
    assert np.sqrt(capped @ covariance @ capped) == pytest.approx(0.10)
    pd.testing.assert_series_equal(
        cap_volatility(capped, covariance, 0.2, 0.98),
        capped,
    )
    covariance.loc["BIL", "BIL"] = 1.0
    with pytest.raises(QuantError, match="defensive"):
        cap_volatility(target, covariance, 0.10, 0.98)


def test_balanced_targets_are_not_a_single_factor_rotation(market, policy):
    weights = target_weights(market.close, policy["candidates"][1], policy)
    np.testing.assert_allclose(weights.loc[list(FACTORS)], [0.147] * 4)
    assert weights["IEF"] == pytest.approx(0.196)
    assert weights["GLD"] == pytest.approx(0.196)
    assert weights["SPY"] == 0
    assert weights.sum() == pytest.approx(0.98)


def test_common_breadth_scale_preserves_factor_proportions(market, policy):
    history = market.close.copy()
    count = len(history)
    for symbol, rate in zip(FACTORS, [0.0005, 0.0002, -0.0002, -0.0005], strict=True):
        history[symbol] = 100 * np.exp(np.arange(count) * rate)
    history["BIL"] = 100 * np.exp(np.arange(count) * 0.00001)
    weights = target_weights(history, policy["candidates"][3], policy)
    np.testing.assert_allclose(weights.loc[list(FACTORS)], [0.147 / 2] * 4)
    assert weights["BIL"] == pytest.approx(0.294)
    for symbol in FACTORS:
        history[symbol] = 100 * np.exp(np.arange(count) * -0.0001)
    cash_defense = target_weights(history, policy["candidates"][3], policy)
    assert cash_defense.loc[list(FACTORS)].sum() == 0
    assert cash_defense["BIL"] == pytest.approx(0.588)


def test_targets_are_causal_complete_monthly_and_unleveraged(market, policy):
    signals = build_targets(market, policy)
    boundary = pd.Timestamp("2015-09-30")
    close = market.close.copy()
    future = close.index > boundary
    close.loc[future, "QUAL"] *= np.linspace(1, 1.4, future.sum())
    modified = replace(market, close=close, raw_close=close.copy(), open=close * 0.999)
    after = build_targets(modified, policy)
    for name, signal in signals.items():
        pd.testing.assert_frame_equal(signal.loc[:boundary], after[name].loc[:boundary])
        defined = signal.dropna(how="all")
        assert all(is_month_end(day) for day in defined.index)
        assert (defined >= 0).all().all()
        np.testing.assert_allclose(defined.sum(axis=1), 0.98)
        assert defined["SPY"].eq(0).all()
        assert not {"QLD", "SSO", "TQQQ", "UPRO"} & set(defined.columns)
        factor = defined.loc[:, list(FACTORS)]
        active = factor.sum(axis=1) > 1e-12
        shares = factor.loc[active].div(factor.loc[active].sum(axis=1), axis=0)
        assert (shares >= 0.15 - 1e-10).all().all()
        assert (shares <= 0.35 + 1e-10).all().all()


def test_missing_factor_or_insufficient_warmup_is_not_filled(market, policy):
    with pytest.raises(QuantError):
        target_weights(market.close.iloc[:252], policy["candidates"][0], policy)
    with pytest.raises(QuantError):
        target_weights(market.close.drop(columns=["QUAL"]), policy["candidates"][0], policy)


def test_factor_labels_do_not_hide_common_market_risk(market):
    close = market.close.copy()
    rng = np.random.default_rng(10)
    market_returns = close["SPY"].pct_change(fill_method=None).fillna(0).to_numpy()
    for index, symbol in enumerate(FACTORS):
        changes = 0.95 * market_returns + rng.normal(0, 0.00015, len(close)) + index * 0.00001
        close[symbol] = 100 * np.cumprod(1 + changes)
    shared = replace(market, close=close, raw_close=close.copy())
    diag = exposure_diagnostics(shared, "2015-01-02", "2016-02-29")
    assert diag["declared_factor_sleeves"] == 4
    assert diag["covariance_effective_dimension"] < 1.1
    assert all(x["market_r_squared"] > 0.99 for x in diag["market_regressions"].values())
    assert not diag["independent_alpha_sources_proven"]


def test_gates_distinguish_safer_from_higher_return(policy):
    previous = {
        "start": "2016-10-06",
        "end": "2026-10-05",
        "sessions": 2512,
        "cagr": 0.20,
        "sharpe": 1.18,
        "max_drawdown": 0.20,
        "annualized_volatility": 0.17,
    }
    spy = {**previous, "cagr": 0.15}
    safe = {
        **previous,
        "cagr": 0.11,
        "sharpe": 0.9,
        "max_drawdown": 0.14,
        "annualized_volatility": 0.10,
    }
    gates = gate_result(safe, spy, previous, policy)
    assert gates["drawdown_at_most_15pct"] and gates["volatility_below_previous"]
    assert not gates["net_cagr_above_spy"] and not gates["net_excess_sharpe_above_1"]
    safe["max_drawdown"] = 0.150001
    safe["sharpe"] = 1.0
    gates = gate_result(safe, spy, previous, policy)
    assert not gates["drawdown_at_most_15pct"] and not gates["net_excess_sharpe_above_1"]
    with pytest.raises(QuantError, match="identical"):
        gate_result(safe, {**spy, "sessions": 1254}, previous, policy)


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_original_and_independent_equity_match(market, policy, scenario_index):
    signal = buy_and_hold_signals(market.close, "QUAL", "2015-01-02")
    own, independent, audit = audit_path(
        market, signal, "2015-01-02", "2016-02-29", policy, policy["scenarios"][scenario_index]
    )
    assert audit["independent_metrics_passed"] and audit["no_account_leverage"]
    assert audit["max_equity_difference_usd"] < 1e-6
    assert np.allclose(own.frame["equity"], independent["equity"], atol=1e-6)
    if scenario_index:
        assert own.frame.iloc[0]["equity"] == 10000
        assert own.frame.iloc[0]["orders"] == 0


def test_preregistration_retains_prior_count_and_freezes_implementation(
    policy, registration, monkeypatch
):
    assert registration["total_disclosed_configurations"] == 84
    assert registration["new_price_snapshot_obtained"] is False
    verify_registration(policy, registration)
    assert registration["implementation_sha256"] == fingerprint()
    monkeypatch.setattr("us_quant.multifactor_stability.fingerprint", lambda: "changed")
    with pytest.raises(QuantError, match="code"):
        verify_registration(policy, registration)


def test_unexplained_corporate_adjustment_is_not_a_factor_return():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"close": [100.0, 100.0], "adj_close": [90.0, 100.0]}, index=dates)
    payload = {"chart": {"result": [{"events": {}}]}}
    with pytest.raises(QuantError, match="adjustment"):
        corporate_actions(payload, frame, "QUAL")
    payload["chart"]["result"][0]["events"] = {
        "dividends": {"one": {"date": int(pd.Timestamp("2020-01-03T14:30:00Z").timestamp())}}
    }
    assert corporate_actions(payload, frame, "QUAL")["unexplained_adjustment_jumps"] == 0


def test_failed_public_request_has_no_synthetic_or_success_shaped_output(
    policy, registration, tmp_path, monkeypatch
):
    class FailedSession:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise requests.HTTPError("HTTP429: research provider limited")

    monkeypatch.setattr("us_quant.multifactor_stability.requests.Session", FailedSession)
    output = tmp_path / "new-snapshot"
    with pytest.raises(QuantError, match="no proxy fallback"):
        fetch(policy, registration, output)
    assert not output.exists()


def test_load_refuses_a_relabelled_market_snapshot(policy, registration, tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    write_json(
        source / "manifest.json",
        {
            "policy_sha256": digest_json(policy),
            "registration_sha256": digest_json(registration),
            "data_start": "2013-01-01",
            "data_end": policy["as_of"],
            "files": {},
        },
    )
    with pytest.raises(QuantError, match="snapshot"):
        load_market(policy, registration, tmp_path, source)
