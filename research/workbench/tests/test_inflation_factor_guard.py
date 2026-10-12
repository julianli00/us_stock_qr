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
from us_quant.inflation_factor_guard import (
    CANDIDATE,
    build_from_inputs,
    check_registration_eligibility,
    validate_policy,
)
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import ResearchProgram
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, read_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/inflation-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10",
        "2017-06-30",
        ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, "XLK", "SOXX"),
    )


def inputs(market, dollar=1.0, inflation=1.0):
    baseline = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    return pd.DataFrame(
        {
            "dollar_change": dollar,
            "inflation_compensation_change": inflation,
        },
        index=baseline.dropna(how="all").index,
    ), baseline


@pytest.mark.parametrize("inflation", [-1.0, 0.0, 1.0])
def test_only_positive_compensation_preserves_gold_equities_always_halve(market, policy, inflation):
    source, baseline = inputs(market, inflation=inflation)
    targets = build_from_inputs(market, source, policy)[CANDIDATE["id"]]
    active, original = targets.dropna(how="all"), baseline.dropna(how="all")
    np.testing.assert_allclose(active[[*FACTORS, "SOXX"]], original[[*FACTORS, "SOXX"]] / 2)
    np.testing.assert_allclose(active["GLD"], original["GLD"] * (1 if inflation > 0 else 0.5))
    removed = original[market.close.columns.drop("BIL")].sum(axis=1) - active[
        market.close.columns.drop("BIL")
    ].sum(axis=1)
    np.testing.assert_allclose(active["BIL"], original["BIL"] + removed, atol=1e-12)
    np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert targets.index.equals(baseline.index)
    assert targets.isna().all(axis=1).equals(baseline.isna().all(axis=1))
    assert (active[[*FACTORS, "SOXX", "GLD"]] > 0).all().all()


@pytest.mark.parametrize("dollar", [-1.0, 0.0])
def test_no_positive_dollar_trigger_leaves_every_original_target_unchanged(market, policy, dollar):
    source, baseline = inputs(market, dollar=dollar)
    pd.testing.assert_frame_equal(
        build_from_inputs(market, source, policy)[CANDIDATE["id"]],
        baseline,
    )


def test_future_macro_and_prices_cannot_change_past_targets(market, policy):
    source, _ = inputs(market)
    before = build_from_inputs(market, source, policy)[CANDIDATE["id"]]
    cutoff = pd.Timestamp("2017-01-31")
    source.loc[source.index > cutoff] *= -1
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    close.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1, 1.5, later.sum())
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after = build_from_inputs(changed, source, policy)[CANDIDATE["id"]]
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])


@pytest.mark.parametrize("problem", ["missing", "nan", "columns", "extra"])
def test_missing_or_partial_source_decisions_are_not_a_cash_or_standby_fallback(
    market, policy, problem
):
    source, _ = inputs(market)
    if problem == "missing":
        source = source.iloc[1:]
    elif problem == "nan":
        source.iloc[-1, 1] = np.nan
    elif problem == "columns":
        source.columns = ["dollar_change", "current_inflation_download"]
    else:
        source.loc[pd.Timestamp("2016-10-12")] = [1.0, 1.0]
    with pytest.raises(QuantError, match="complete inputs"):
        build_from_inputs(market, source, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("risk_change_sessions", 21),
        ("risk_off_investment_scale", 0.75),
        ("compensation_change_threshold", 0.1),
        ("cash_reserve", 0),
        ("eligible_not_before", "2026-10-11T09:00:00+08:00"),
        ("planned_configurations", 2),
        ("new_economic_factor_definitions", 1),
        ("existing_forward_rules_changed", True),
        ("order_authority", True),
    ],
)
def test_no_scale_window_threshold_or_time_gate_search(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def frozen_spec(tmp_path, policy):
    dependencies = (
        "src/us_quant/inflation_factor_guard.py",
        "config/inflation-factor-guard.json",
        "src/us_quant/dollar_risk_guard.py",
        "config/dollar-risk-guard.json",
        "src/us_quant/sector_growth_balance.py",
        "config/sector-growth-balance.json",
        "src/us_quant/growth_factor_satellite.py",
        "config/growth-factor-satellite.json",
        "src/us_quant/multifactor_stability.py",
        "src/us_quant/macro_factor_tilt.py",
        "src/us_quant/financial_conditions_guard.py",
        "src/us_quant/inflation_inputs.py",
        "config/inflation-inputs.json",
    )
    frozen = {}
    for name in dependencies:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        frozen[name] = file_digest(destination)
    for name in (
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/inflation-inputs-20261011/manifest.json",
        "evidence/inflation_input_readiness_20261011.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        write_text_atomic(tmp_path / name, "synthetic source seal,not actual source observations")
        frozen[name] = file_digest(tmp_path / name)
    return {
        "id": CANDIDATE["id"],
        "configuration": CANDIDATE,
        "frozen_files": frozen,
        "eligible_not_before": policy["eligible_not_before"],
    }


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_complete_registered_target_replay_and_synthetic_funding_agree(
    market, policy, tmp_path, monkeypatch, cost, delay
):
    source, _ = inputs(market)
    monkeypatch.setattr("us_quant.inflation_factor_guard.load_inputs", lambda data, p: (source, {}))
    spec = frozen_spec(tmp_path, policy)
    start, end = "2016-10-06", "2016-11-04"
    actual = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    expected = seed_window(build_from_inputs(market, source, policy)[CANDIDATE["id"]], start)
    pd.testing.assert_frame_equal(actual, expected)
    result = simulate(
        market,
        actual,
        start,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    independent = independent_equity(
        market,
        actual,
        start,
        end,
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert result.frame["cash"].min() >= 0
    missing = {
        **spec,
        "frozen_files": {
            key: value
            for key, value in spec["frozen_files"].items()
            if key != "data/inflation-inputs-20261011/manifest.json"
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(missing, market, start, end, cost, delay, tmp_path, {})


def test_wrapper_and_direct_program_registration_cannot_bypass_actual_cycle_gate(
    policy, tmp_path, monkeypatch
):
    spec = frozen_spec(tmp_path, policy)
    before_cycle = pd.Timestamp("2026-10-11T14:30:00+08:00")
    with pytest.raises(QuantError, match="not eligible"):
        check_registration_eligibility(spec, tmp_path, before_cycle)
    monkeypatch.setattr("us_quant.research_program.readiness_blockers", lambda *a, **kw: [])
    program = ResearchProgram(
        tmp_path / "ledger.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=tmp_path,
        create=True,
    )
    try:
        before = program.status()
        direct = {
            **spec,
            "data_scope": "factor_etf_portfolio",
            "factor_ids": [
                "price_momentum",
                "value_exposure",
                "quality_exposure",
                "low_volatility_exposure",
            ],
            "order_authority": False,
            "leveraged_products_allowed": False,
            "history_status": "exposed_history_not_independent_holdout",
        }
        with pytest.raises(QuantError, match="not eligible"):
            program.register_candidate(
                direct, {"data_scope": "factor_etf_portfolio"}, now=before_cycle
            )
        assert program.status() == before
    finally:
        program.close()


@pytest.mark.parametrize("frozen", [None, [], "not-a-manifest", {}])
def test_malformed_frozen_manifest_retains_explicit_rejection_without_mutating_ledger(
    tmp_path, monkeypatch, frozen
):
    monkeypatch.setattr("us_quant.research_program.readiness_blockers", lambda *a, **kw: [])
    program = ResearchProgram(
        tmp_path / "ledger.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=tmp_path,
        create=True,
    )
    try:
        before = program.status()
        spec = {
            "id": CANDIDATE["id"],
            "data_scope": "factor_etf_portfolio",
            "factor_ids": [
                "price_momentum",
                "value_exposure",
                "quality_exposure",
                "low_volatility_exposure",
            ],
            "order_authority": False,
            "leveraged_products_allowed": False,
            "history_status": "exposed_history_not_independent_holdout",
            "frozen_files": frozen,
        }
        with pytest.raises(QuantError, match="must be frozen"):
            program.register_candidate(spec, {"data_scope": "factor_etf_portfolio"})
        assert program.status() == before
    finally:
        program.close()
