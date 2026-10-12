from __future__ import annotations

import json
from dataclasses import replace
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.growth_factor_satellite import composed_monthly_target
from us_quant.multifactor_stability import FACTORS
from us_quant.sector_growth_balance import (
    ISSUER_URLS,
    SECTORS,
    build_targets,
    issuer_definition,
    validate_policy,
    verified_market,
)
from us_quant.storage import file_digest, read_json, write_json
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/sector-growth-balance.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2018-06-29", ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, *SECTORS)
    )


def technology_document(ticker="XLK"):
    attrs = {
        "fund-ticker": {"value": ticker},
        "isin": {"value": "US81369Y8030"},
        "inception-date": {"value": "Dec 16 1998"},
        "benchmark": {"value": "Technology Select Sector Index"},
    }
    value = escape(json.dumps({"attrs": attrs}), quote=True)
    return (
        f'<input id="fund-quick-info" value="{value}">'
        "<p>before expenses, correspond generally to the price and yield performance "
        "of the Technology Select Sector Index,technology sector of the S&amp;P 500 Index</p>"
    )


def semiconductor_document(ticker="SOXX", benchmark="NYSE Semiconductor Index", transition=True):
    url = ISSUER_URLS["SOXX"]
    graph = {
        "@graph": [
            {"@id": url + "#fund", "alternateName": ticker, "category": "Equity", "url": url},
            {
                "@id": url + "#key-facts",
                "about": {"@id": url + "#fund"},
                "additionalProperty": [
                    {"name": "Fund Inception", "value": "Jul 10, 2001"},
                    {"name": "Asset Class", "value": "Equity"},
                    {"name": "Benchmark Index", "value": benchmark},
                ],
            },
            {
                "@id": url + "#fund-description",
                "about": {"@id": url + "#fund"},
                "description": "The fund tracks an U.S. equity index of semiconductor companies.",
            },
        ]
    }
    text = (
        '<meta name="injectable-productTicker" content="soxx">'
        '<meta name="injectable-productAssetClass" content="eq">'
        f'<script type="application/ld+json">{json.dumps(graph)}</script>'
    )
    if transition:
        text += (
            '<button title="On 6/21/2021 SOXX began to track the NYSE Semiconductor Index. '
            'Earlier data is for the PHLX SOX Semiconductor Sector Index."></button>'
        )
    return text


def test_primary_technology_identity_requires_scoped_fields_and_nonmultiplied_mandate():
    result = issuer_definition("XLK", technology_document())
    assert result["intended_equity_index_return_multiple"] == 1
    assert not result["full_historical_methodology_verified"]
    with pytest.raises(QuantError, match="index mandate"):
        issuer_definition("XLK", technology_document("TQQQ"))
    with pytest.raises(QuantError, match="index mandate"):
        issuer_definition("XLK", technology_document().replace("before expenses", "daily 2x"))
    with pytest.raises(QuantError, match="scoped"):
        issuer_definition("XLK", "XLK technology 1x")


def test_semiconductor_scope_preserves_actual_benchmark_change_not_false_continuity():
    definition = issuer_definition("SOXX", semiconductor_document())
    assert definition["disclosed_index_transition"]["session"] == "2021-06-21"
    assert not definition["full_historical_methodology_verified"]
    for document in (
        semiconductor_document("SOXL"),
        semiconductor_document(benchmark="PHLX SOX Semiconductor Sector Index"),
        semiconductor_document(transition=False),
    ):
        with pytest.raises(QuantError, match="transition"):
            issuer_definition("SOXX", document)
    with pytest.raises(QuantError, match="audited"):
        issuer_definition("SOXL", semiconductor_document())


def test_both_sector_rules_keep_four_factors_half_equity_monthly_budget_and_cash(market, policy):
    targets = build_targets(market, policy)
    for candidate in policy["candidates"]:
        active = targets[candidate["id"]].dropna(how="all")
        equity = active.loc[:, list(FACTORS)].sum(axis=1) + active[candidate["sector_symbol"]]
        np.testing.assert_allclose(active[candidate["sector_symbol"]], equity / 2)
        for factor in FACTORS:
            np.testing.assert_allclose(active[factor], equity / 8)
        np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (equity >= 0.98 * 0.30 - 1e-12).all()
        assert (equity <= 0.98 * 0.70 + 1e-12).all()
        excluded = {"SPY", "IEF", "TLT", "BIL", "QQQ", *SECTORS} - {candidate["sector_symbol"]}
        assert active.loc[:, list(excluded)].eq(0).all().all()
        assert (active >= 0).all().all()
        assert not (
            targets[candidate["id"]].isna().any(axis=1)
            & ~targets[candidate["id"]].isna().all(axis=1)
        ).any()
        assert active.index[0] < pd.Timestamp("2016-10-06")


def test_growth_column_adapter_reuses_original_rule_instead_of_duplicate_budget_logic(
    market, policy
):
    candidate = policy["candidates"][0]
    reference = market.close.drop(columns="QQQ").rename(columns={"XLK": "QQQ"})
    last = build_targets(market, policy)[candidate["id"]].dropna(how="all").iloc[-1]
    expected = composed_monthly_target(reference.loc[: last.name], candidate)
    pd.testing.assert_series_equal(
        last.drop("QQQ"), expected.rename(index={"QQQ": "XLK"}), check_names=False
    )


def test_future_sector_prices_and_risk_free_changes_cannot_retune_past_targets(market, policy):
    original = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    opening, closing, rates = market.open.copy(), market.close.copy(), market.risk_free.copy()
    later = closing.index > cutoff
    multiplier = np.linspace(1, 1.4, later.sum())[:, None]
    closing.loc[later, list(SECTORS)] *= multiplier
    opening.loc[later, list(SECTORS)] *= multiplier
    rates.loc[later] += 0.0001
    changed = replace(
        market, open=opening, close=closing, raw_close=closing.copy(), risk_free=rates
    )
    after = build_targets(changed, policy)
    for identifier in original:
        pd.testing.assert_frame_equal(
            original[identifier].loc[:cutoff], after[identifier].loc[:cutoff]
        )


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_registered_source_replay_and_independent_funded_paths(market, policy, cost, delay):
    candidate = policy["candidates"][0]
    dependencies = (
        "src/us_quant/sector_growth_balance.py",
        "config/sector-growth-balance.json",
        "src/us_quant/growth_factor_satellite.py",
        "config/growth-factor-satellite.json",
        "src/us_quant/multifactor_stability.py",
    )
    spec = {
        "id": candidate["id"],
        "configuration": candidate,
        "frozen_files": {name: file_digest(ROOT / name) for name in dependencies},
    }
    start, end = "2016-10-06", "2017-06-30"
    targets = registered_targets(spec, market, start, end, cost, delay, ROOT, {})
    pd.testing.assert_frame_equal(
        targets, seed_window(build_targets(market, policy)[candidate["id"]], start)
    )
    funded = simulate(
        market, targets, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    audit = independent_equity(
        market, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(funded.frame["equity"], audit["equity"], rtol=0, atol=1e-8)
    assert funded.frame["cash"].min() >= 0
    missing = {
        **spec,
        "frozen_files": {
            k: v for k, v in spec["frozen_files"].items() if "growth_factor_satellite.py" not in k
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(missing, market, start, end, cost, delay, ROOT, {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("growth_share_of_equity", 0.75),
        ("volatility_sessions", 126),
        ("equity_share_max", 0.90),
        ("new_economic_factor_definitions", 2),
    ],
)
def test_sector_share_window_or_catalog_count_cannot_be_retuned(policy, key, value):
    policy[key] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_manifest_change_or_symlink_blocks_before_loading_price_history(
    tmp_path, monkeypatch, policy
):
    source = tmp_path / "data/source"
    source.mkdir(parents=True)
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    monkeypatch.setattr("us_quant.sector_growth_balance.ROOT", tmp_path)
    monkeypatch.setattr("us_quant.sector_growth_balance.POLICY", policy_path)
    monkeypatch.setattr("us_quant.sector_growth_balance.SOURCE", source)
    manifest = {
        "schema_version": 1,
        "policy_sha256": "0" * 64,
        "data_start": policy["data_start"],
        "data_end": policy["as_of"],
        "sources": {"XLK": {}, "SOXX": {}},
        "synthetic_history": False,
        "strategy_outcomes_computed": False,
    }
    write_json(source / "manifest.json", manifest)
    with pytest.raises(QuantError, match="manifest"):
        verified_market(policy)
    (source / "manifest.json").unlink()
    write_json(tmp_path / "elsewhere.json", manifest)
    (source / "manifest.json").symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(QuantError, match="symlink"):
        verified_market(policy)
