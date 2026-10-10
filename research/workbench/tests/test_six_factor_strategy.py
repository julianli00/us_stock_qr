from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import shutil

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS
from us_quant.six_factor_strategy import (
    BASELINE,
    FAMILIES,
    SOURCE_POLICY,
    build_targets,
    register,
    validate_policy,
)
from us_quant.storage import file_digest, read_json, write_json
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/six-factor-strategy.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FAMILIES))


def test_all_families_stay_positive_with_identical_original_equity_gold_budgets(market, policy):
    original = replace(
        market,
        open=market.open.drop(columns=["IJR", "PKW"]),
        close=market.close.drop(columns=["IJR", "PKW"]),
        raw_close=market.raw_close.drop(columns=["IJR", "PKW"]),
        volume=market.volume.drop(columns=["IJR", "PKW"]),
    )
    baseline = monthly_targets(original, read_json(BASELINE)).dropna(how="all")
    results = build_targets(market, policy)
    for candidate in policy["candidates"]:
        target = results[candidate["id"]].dropna(how="all")
        assert target.index.equals(baseline.index)
        pd.testing.assert_frame_equal(
            target[["SPY", "IEF", "GLD", "BIL"]], baseline[["SPY", "IEF", "GLD", "BIL"]]
        )
        equity = target.loc[:, list(FAMILIES)].sum(axis=1)
        np.testing.assert_allclose(
            equity, baseline.loc[:, list(FACTORS)].sum(axis=1), rtol=0, atol=1e-12
        )
        shares = target.loc[:, list(FAMILIES)].div(equity, axis=0)
        assert (shares >= 1 / 12 - 1e-12).all().all()
        assert (shares <= 1 / 4 + 1e-12).all().all()
        assert (target >= 0).all().all()
        np.testing.assert_allclose(target.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        if candidate["factor_weighting"] == "equal":
            np.testing.assert_allclose(shares, 1 / 6, rtol=0, atol=1e-12)


def test_bounded_inverse_volatility_reuses_fixed_limits_not_a_fitted_ranking(market, policy):
    close = market.close.copy()
    daily = close["PKW"].pct_change(fill_method=None).fillna(0)
    close["PKW"] = 100 * (1 + daily * 0.01).cumprod()
    changed = replace(market, close=close, raw_close=close.copy(), open=close * 0.999)
    frame = build_targets(changed, policy)[policy["candidates"][1]["id"]].dropna(how="all")
    equity = frame.loc[:, list(FAMILIES)].sum(axis=1)
    np.testing.assert_allclose(frame["PKW"] / equity, 0.25, rtol=0, atol=1e-10)


def test_future_observations_and_open_prices_cannot_change_past_targets(market, policy):
    before = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    close.loc[later, ["IJR", "PKW", "QUAL", "GLD"]] *= np.linspace(1, 1.4, later.sum())[:, None]
    opening *= 1.3
    updated = build_targets(replace(market, close=close, raw_close=close.copy(), open=opening), policy)
    for name in before:
        pd.testing.assert_frame_equal(before[name].loc[:cutoff], updated[name].loc[:cutoff])


def test_zero_volatility_family_cannot_be_silently_replaced_by_default_weights(market, policy):
    close = market.close.copy()
    close["PKW"] = 100
    with pytest.raises(QuantError, match="finite volatilities"):
        build_targets(replace(market, close=close, raw_close=close.copy(), open=close * 0.999), policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_frozen_generator_and_independent_cash_accounts(market, policy, tmp_path, cost, delay):
    source = ROOT / "src/us_quant/six_factor_strategy.py"
    files = (
        source,
        ROOT / "config/six-factor-strategy.json",
        BASELINE,
        SOURCE_POLICY,
        source.with_name("factor_gold_risk.py"),
        source.with_name("factor_family_sources.py"),
        source.with_name("multifactor_stability.py"),
        source.with_name("factor_replication.py"),
    )
    frozen = {}
    for path in files:
        relative = path.relative_to(ROOT)
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        frozen[relative.as_posix()] = file_digest(destination)
    manifest = tmp_path / "data/new-factor-family-source-20261011/manifest.json"
    write_json(manifest, {"synthetic_generator_fixture_only": True})
    frozen[manifest.relative_to(tmp_path).as_posix()] = file_digest(manifest)
    candidate = policy["candidates"][1]
    spec = {"id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
    start, end = "2016-10-06", "2017-03-31"
    target = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    pd.testing.assert_frame_equal(
        target, seed_window(build_targets(market, policy)[candidate["id"]], start)
    )
    own = simulate(
        market, target, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, target, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cost"].sum() > 0 and own.frame["cash"].min() >= 0
    if delay == 1:
        assert own.frame["orders"].iloc[0] > 0
    else:
        assert own.frame["orders"].iloc[0] == 0
    omitted = {
        **spec,
        "frozen_files": {
            key: value
            for key, value in frozen.items()
            if key != "config/factor-family-expansion.json"
        },
    }
    with pytest.raises(QuantError, match="missing a frozen helper"):
        registered_targets(omitted, market, start, end, cost, delay, tmp_path, {})


@pytest.mark.parametrize("change", ["bounds", "eligibility", "family"])
def test_frozen_families_and_limits_cannot_be_retuned_after_outcomes(policy, change):
    if change == "bounds":
        policy["maximum_share_of_equity"] = 0.5
    elif change == "family":
        policy["factor_symbols"][-1] = "PDP"
    else:
        policy["eligible_not_before"] = "2026-10-11T09:00:00+08:00"
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_registration_not_before_constraint_refuses_early_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "us_quant.six_factor_strategy.timestamp",
        lambda: pd.Timestamp("2026-10-17T08:59:59+08:00").to_pydatetime(),
    )
    receipt = tmp_path / "receipt.json"
    with pytest.raises(QuantError, match="not eligible"):
        register(tmp_path / "not_prepared", receipt)
    assert not receipt.exists()
