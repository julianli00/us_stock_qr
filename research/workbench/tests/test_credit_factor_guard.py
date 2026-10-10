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
from us_quant.credit_factor_guard import CONTROL, build_from_inputs, load_credit, validate_policy
from us_quant.dual_horizon import seed_window
from us_quant.macro_factor_tilt import align_observations
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/credit-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def test_compression_can_recover_before_elevated_credit_levels(market, policy):
    credit = pd.Series(2.0, index=market.close.index)
    credit.iloc[-63:] = np.linspace(4, 3, 63)
    results = build_from_inputs(market, credit, policy)
    strict = results[policy["candidates"][0]["id"]].dropna(how="all").iloc[-1]
    recovery = results[policy["candidates"][1]["id"]].dropna(how="all").iloc[-1]
    assert strict["BIL"] == pytest.approx(0.98 * 0.70)
    assert strict.loc[list(FACTORS)].sum() == 0
    assert recovery["BIL"] == 0
    np.testing.assert_allclose(recovery.loc[list(FACTORS)], 0.98 * 0.70 / 4)
    assert strict["GLD"] == recovery["GLD"] == pytest.approx(0.98 * 0.30)


def test_rising_or_equal_credit_levels_do_not_create_unregistered_equity_floor(market, policy):
    for credit in (
        pd.Series(np.linspace(1, 2, len(market.close)), index=market.close.index),
        pd.Series(2.0, index=market.close.index),
    ):
        for frame in build_from_inputs(market, credit, policy).values():
            active = frame.dropna(how="all")
            assert active.loc[:, list(FACTORS)].eq(0).all().all()
            np.testing.assert_allclose(active["BIL"], 0.98 * 0.70)
            np.testing.assert_allclose(active["GLD"], 0.98 * 0.30)
            np.testing.assert_allclose(active.sum(axis=1), 0.98)
            assert (active >= 0).all().all()


def test_declining_credit_retains_the_original_static_family_budgets(market, policy):
    credit = pd.Series(np.linspace(3, 2, len(market.close)), index=market.close.index)
    for frame in build_from_inputs(market, credit, policy).values():
        active = frame.dropna(how="all")
        assert active.index[0] <= pd.Timestamp("2016-09-30")
        np.testing.assert_allclose(active.loc[:, list(FACTORS)], 0.98 * 0.70 / 4)
        assert active["BIL"].eq(0).all()
        np.testing.assert_allclose(active["GLD"], 0.98 * 0.30)


def test_future_credit_or_fund_revisions_cannot_change_past_targets(market, policy):
    credit = pd.Series(2 + np.sin(np.arange(len(market.close)) / 50), index=market.close.index)
    original = build_from_inputs(market, credit, policy)
    cutoff = pd.Timestamp("2017-06-30")
    after = credit.index > cutoff
    changed_credit = credit.copy()
    changed_credit.loc[after] += 10
    closing, opening = market.close.copy(), market.open.copy()
    closing.loc[after, "QUAL"] *= np.linspace(1, 1.4, after.sum())
    opening.loc[after, "QUAL"] *= np.linspace(1, 1.4, after.sum())
    changed = replace(market, close=closing, raw_close=closing.copy(), open=opening)
    updated = build_from_inputs(changed, changed_credit, policy)
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], updated[name].loc[:cutoff])


def test_raw_credit_uses_two_strictly_later_sessions_and_never_unlimited_carry():
    dates = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"])
    observed = pd.Series([1.0, 99.0], index=pd.to_datetime(["2026-10-01", "2026-10-05"]))
    known, provenance = align_observations(observed, dates)
    assert known.loc["2026-10-05"] == known.loc["2026-10-06"] == 1
    assert known.loc["2026-10-07"] == 99
    assert provenance.loc["2026-10-07", "available_session"] == pd.Timestamp("2026-10-07")
    with pytest.raises(QuantError, match="stale"):
        align_observations(observed, pd.DatetimeIndex(["2026-10-20"]))


def test_source_hash_change_or_wrong_series_is_rejected(tmp_path):
    write_text_atomic(
        tmp_path / "BAA10Y.csv", "observation_date,BAA10Y\n2026-10-01,2\n2026-10-02,3\n"
    )
    for name in ("BAA10Y-source.html", "ICE-source-restriction.html"):
        write_text_atomic(tmp_path / name, "source")
    source = {
        "schema_version": 1,
        "series": "BAA10Y",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y&cosd=2014-01-01&coed=2026-10-05",
        "raw_inputs_not_to_be_published": True,
        "files": {
            name: file_digest(tmp_path / name)
            for name in ("BAA10Y.csv", "BAA10Y-source.html", "ICE-source-restriction.html")
        },
    }
    write_json(tmp_path / "acquisition.json", source)
    values, _ = load_credit(tmp_path, pd.DatetimeIndex(["2026-10-05"]))
    assert values.iloc[0] == 2
    write_text_atomic(tmp_path / "BAA10Y.csv", "observation_date,BAA10Y\n2026-10-01,5\n")
    with pytest.raises(QuantError, match="hash"):
        load_credit(tmp_path, pd.DatetimeIndex(["2026-10-05"]))


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_frozen_source_targets_and_independent_accounts(
    market, policy, monkeypatch, tmp_path, cost, delay
):
    source = ROOT / "src/us_quant/credit_factor_guard.py"
    config = ROOT / "config/credit-factor-guard.json"
    credit = pd.Series(2 + np.sin(np.arange(len(market.close)) / 50), index=market.close.index)
    monkeypatch.setattr(
        "us_quant.credit_factor_guard.load_credit", lambda directory, index: (credit, None)
    )
    candidate = policy["candidates"][1]
    files = (
        source,
        config,
        CONTROL,
        source.with_name("adaptive_factor_allocation.py"),
        source.with_name("macro_factor_tilt.py"),
    )
    frozen = {}
    for path in files:
        relative = path.relative_to(ROOT)
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        frozen[relative.as_posix()] = file_digest(destination)
    for name in (
        "acquisition.json",
        "BAA10Y.csv",
        "BAA10Y-source.html",
        "ICE-source-restriction.html",
    ):
        relative = f"data/credit-spread-source-20261011/{name}"
        destination = tmp_path / relative
        write_text_atomic(destination, "frozen synthetic input for target-generator testing")
        frozen[relative] = file_digest(destination)
    spec = {
        "id": candidate["id"],
        "configuration": candidate,
        "frozen_files": frozen,
    }
    start, end = "2016-10-06", "2017-03-31"
    signals = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    pd.testing.assert_frame_equal(
        signals, seed_window(build_from_inputs(market, credit, policy)[candidate["id"]], start)
    )
    actual = simulate(
        market, signals, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, signals, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(actual.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert actual.frame["cash"].min() >= 0 and actual.frame["cost"].sum() > 0
    omitted = {
        **spec,
        "frozen_files": {
            key: value
            for key, value in frozen.items()
            if key != "data/credit-spread-source-20261011/BAA10Y.csv"
        },
    }
    with pytest.raises(QuantError, match="missing a frozen helper"):
        registered_targets(omitted, market, start, end, cost, delay, tmp_path, {})


def test_no_post_outcome_credit_lookback_or_threshold_tuning(policy):
    policy["credit_change_sessions"] = 63
    with pytest.raises(QuantError):
        validate_policy(policy)
