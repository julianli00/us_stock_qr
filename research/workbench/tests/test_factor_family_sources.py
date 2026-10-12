from __future__ import annotations

from html import escape
from pathlib import Path

import json

import pytest

from us_quant.config import QuantError
from us_quant.factor_family_sources import (
    definitions,
    issuer_definition,
    validate_policy,
    verified_market,
)
from us_quant.storage import file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-family-expansion.json")


def small_cap_document(ticker="ijr"):
    return (
        f'<meta name="injectable-productTicker" content="{ticker}">'
        '<meta name="injectable-productAssetClass" content="eq">'
        "<span>Benchmark Index</span><span>S&amp;P SmallCap 600 Index</span>"
        "<span>Fund Inception</span><span>May 22, 2000</span>"
        "<span>Asset Class</span><span>Equity</span>"
        "<span>Bloomberg Index Ticker</span><span>SPTRSMCP</span>"
    )


def buyback_document(
    ticker="PKW", reduction="net reduction in shares outstanding of 5% or more"
):
    metadata = {
        "ticker": ticker,
        "isin": "US46137V3087",
        "shareClassInceptionDate": "2006-12-20",
        "assetClass": "Equity",
        "region": "United States",
        "investmentMethod": "Passive",
        "bloombergTicker": "DRBTR",
    }
    tags = "".join(f'<meta name="{name}" content="{value}">' for name, value in metadata.items())
    description = {
        "description": f"An index fund tracks Nasdaq US BuyBack Achievers, {reduction} over the trailing 12 months."
    }
    return tags + f'<div data-model-json="{escape(json.dumps(description), quote=True)}"></div>'


def test_small_cap_role_uses_visible_fund_fields_not_just_ticker_text():
    result = issuer_definition("IJR", small_cap_document())
    assert result["family"] == "size"
    assert result["fund_inception"] == "2000-05-22"
    assert result["futures_may_offset_cash_for_tracking"]
    assert not result["full_historical_methodology_verified"]
    with pytest.raises(QuantError, match="small-cap"):
        issuer_definition("IJR", small_cap_document("spy"))
    with pytest.raises(QuantError, match="small-cap"):
        issuer_definition("IJR", small_cap_document().replace("SmallCap 600", "LargeCap 500"))
    with pytest.raises(QuantError, match="small-cap"):
        issuer_definition("IJR", "<script>" + small_cap_document() + "</script>")


def test_buyback_role_requires_net_share_reduction_not_gross_repurchases():
    result = issuer_definition("PKW", buyback_document())
    assert result["family"] == "share_issuance"
    assert result["minimum_net_share_reduction"] == 0.05
    assert result["lookback_months"] == 12
    with pytest.raises(QuantError, match="net-share-reduction"):
        issuer_definition("PKW", buyback_document(reduction="gross repurchases of 5%"))
    with pytest.raises(QuantError, match="net-share-reduction"):
        issuer_definition(
            "PKW", buyback_document(reduction="net reduction in shares outstanding of 15% or more")
        )
    with pytest.raises(QuantError, match="net-share-reduction"):
        issuer_definition("PKW", buyback_document("PDP"))
    with pytest.raises(QuantError, match="embedded definition"):
        issuer_definition("PKW", '<div data-model-json="broken"></div>')
    with pytest.raises(QuantError, match="explicitly audited"):
        issuer_definition("SPY", buyback_document())


def test_queued_definitions_are_two_distinct_catalog_families_not_completed_strategies(policy):
    factors = definitions(policy)
    assert set(factors) == {"size_exposure_ijr_v1", "net_buyback_exposure_pkw_v1"}
    assert {row["family"] for row in factors.values()} == {"size", "share_issuance"}
    assert policy["new_strategy_configurations"] == 0
    assert policy["eligible_not_before"] == "2026-10-17T09:00:00+08:00"


@pytest.mark.parametrize("change", ["family", "symbol", "leverage", "eligibility"])
def test_source_family_or_eligibility_cannot_be_changed_silently(policy, change):
    if change == "family":
        policy["factors"][0]["family"] = "momentum"
    elif change == "symbol":
        policy["factors"][0]["parameters"]["symbol"] = "SPY"
    elif change == "leverage":
        policy["factors"][0]["parameters"]["nominal_product_multiple"] = 2
    else:
        policy["eligible_not_before"] = "2026-10-11T09:00:00+08:00"
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_source_manifest_hash_mismatch_and_symlink_fail_before_any_fund_load(
    tmp_path, monkeypatch, policy
):
    source = tmp_path / "data/source"
    source.mkdir(parents=True)
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    monkeypatch.setattr("us_quant.factor_family_sources.ROOT", tmp_path)
    monkeypatch.setattr("us_quant.factor_family_sources.POLICY", policy_path)
    monkeypatch.setattr("us_quant.factor_family_sources.SOURCE", source)
    write_json(
        source / "manifest.json",
        {
            "policy_sha256": "0" * 64,
            "data_start": policy["data_start"],
            "data_end": policy["as_of"],
            "sources": {"IJR": {}, "PKW": {}},
            "additional_source_files": {"PKW-index.html": "0" * 64, "SP-methodology-response.bin": "0" * 64},
            "synthetic_history": False,
            "strategy_outcomes_computed": False,
        },
    )
    with pytest.raises(QuantError, match="manifest"):
        verified_market(policy)
    manifest = read_json(source / "manifest.json")
    manifest["policy_sha256"] = file_digest(policy_path)
    write_json(tmp_path / "outside.json", manifest)
    (source / "manifest.json").unlink()
    (source / "manifest.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(QuantError, match="symlink"):
        verified_market(policy)
