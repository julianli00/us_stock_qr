from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from us_quant import sector_basket_sources as source
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/sector-basket-sources.json")


def issuer(symbol, policy):
    info = {
        "attrs": {
            "fund-ticker": {"value": symbol},
            "isin": {"value": "US81369Y1001"},
            "inception-date": {"value": "Dec 16 1998"},
            "benchmark": {"value": source.BENCHMARKS[symbol]},
        },
    }
    return (
        f"<input id='fund-quick-info' value='{json.dumps(info)}'>"
        f"<p>Before expenses, correspond generally to the price and yield performance "
        f"of the {source.BENCHMARKS[symbol]}, a sector of the S&amp;P 500 Index.</p>"
        f"<a href='/library-content/products/factsheets/etfs/us/factsheet-us-en-{symbol.lower()}.pdf'>"
        "Fund factsheet</a>"
    )


@pytest.mark.parametrize("symbol", source.SECTORS)
def test_every_original_sector_requires_scoped_identity_and_its_own_advertised_document(symbol, policy):
    identity = source.issuer_definition(symbol, issuer(symbol, policy), policy)
    assert identity["symbol"] == symbol
    assert identity["fund_inception"] == "1998-12-16"
    assert identity["benchmark"] == source.BENCHMARKS[symbol]
    assert identity["intended_equity_index_return_multiple"] == 1.0
    assert identity["historical_constituents_or_gross_notional_archive_verified"] is False
    assert identity["factsheet_url"].endswith(f"factsheet-us-en-{symbol.lower()}.pdf")


@pytest.mark.parametrize(
    "problem", ["general_portal", "wrong_symbol", "wrong_index", "wrong_date", "no_pdf", "foreign_pdf"],
)
def test_incomplete_or_mismatched_issuer_content_does_not_default_to_ordinary_fund(problem, policy):
    document = issuer("XLB", policy)
    if problem == "general_portal":
        document = "<html>General fund information portal</html>"
    elif problem == "wrong_symbol":
        document = document.replace('"XLB"', '"SSO"')
    elif problem == "wrong_index":
        document = document.replace(source.BENCHMARKS["XLB"], "Daily leveraged commodities index")
    elif problem == "wrong_date":
        document = document.replace("Dec 16 1998", "Dec 16 2020")
    elif problem == "no_pdf":
        document = document[:document.index("<a")]
    else:
        document = document.replace(
            "/library-content/products", "https://unrelated.example/library-content/products",
        )
    with pytest.raises(QuantError):
        source.issuer_definition("XLB", document, policy)


def test_an_unplanned_fund_cannot_be_added_after_observing_source_data(policy):
    with pytest.raises(QuantError, match="predeclared"):
        source.issuer_definition("SOXX", issuer("XLB", policy), policy)


@pytest.mark.parametrize("problem", ["cusip", "date", "index", "objective"])
def test_pdf_identity_must_match_the_actual_fund_not_just_a_generic_sector_label(policy, problem):
    identity = source.issuer_definition("XLB", issuer("XLB", policy), policy)
    text = f"{identity['benchmark']} price and yield performance CUSIP {identity['cusip']} 12/16/1998"
    if problem == "cusip":
        text = text.replace(identity["cusip"], "wrong_security")
    elif problem == "date":
        text = text.replace("12/16/1998", "12/16/2020")
    elif problem == "index":
        text = text.replace(identity["benchmark"], "S&P 500 Index")
    else:
        text = text.replace("price and yield performance", "Daily double the return")
    with pytest.raises(QuantError):
        source.factsheet_definition(text, identity)


def test_advertised_document_cannot_be_an_html_viewer(policy, tmp_path):
    identity = source.issuer_definition("XLB", issuer("XLB", policy), policy)
    path = tmp_path / "factsheet.pdf"
    path.write_text("<html>Country redirect or PDF viewer</html>")
    with pytest.raises(QuantError, match="not PDF"):
        source.verify_factsheet(path, identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sector_symbols", ["XLK", "SOXX"]),
        ("benchmarks", {"XLB": "Another index"}),
        ("data_start", "2021-10-05"),
        ("new_registered_candidates", 2),
        ("portfolio_targets_or_returns_computed", True),
        ("existing_W42_queue_changed", True),
        ("order_authority", True),
        ("investment_objective_verified", True),
        ("issuer_urls", None),
        ("source_contract", None),
    ],
)
def test_source_scope_cannot_be_shrunk_promoted_or_relabelled(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        source.validate(policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("new_quote_requests", 9),
        ("access_or_rate_limit_bypass", True),
        ("legacy_registration_or_source_hashes_rewritten", True),
        ("code_fingerprint_drift_is_not_ignored_for_old_strategy_replay", False),
        ("complete_2805_post_inception_sessions_required", False),
    ],
)
def test_source_price_access_history_or_old_fingerprint_cannot_be_repaired_silently(policy, field, value):
    policy["source_contract"][field] = value
    with pytest.raises(QuantError):
        source.validate(policy)


@pytest.mark.parametrize("problem", ["not_ready", "missing_fund", "wrong_code", "invalid_funds"])
def test_market_admission_requires_the_complete_verified_source_manifest(tmp_path, monkeypatch, policy, problem):
    root = tmp_path / "research/workbench"
    folder = root / "data/source"
    folder.mkdir(parents=True)
    path = root / "config/policy.json"
    write_json(path, policy)
    monkeypatch.setattr(source, "ROOT", root)
    monkeypatch.setattr(source, "POLICY", path)
    monkeypatch.setattr(source, "SOURCE", folder)
    fund = {
        "issuer_path": "issuer.html", "issuer_sha256": "a" * 64,
        "factsheet_path": "factsheet.pdf", "factsheet_sha256": "b" * 64,
        "identity": {}, "factsheet_identity": {},
    }
    manifest = {
        "source_policy_sha256": file_digest(path),
        "source_validator_sha256": source.fingerprint(),
        "all_sources_ready": True,
        "funds": {symbol: deepcopy(fund) for symbol in source.SECTORS},
        "file_sha256": {},
    }
    if problem == "not_ready":
        manifest["all_sources_ready"] = False
    elif problem == "missing_fund":
        manifest["funds"].pop("XLB")
    elif problem == "wrong_code":
        manifest["source_validator_sha256"] = "different_adapter"
    else:
        manifest["funds"] = None
    write_json(folder / "manifest.json", manifest)
    monkeypatch.setattr(
        source, "cached_sector_prices", lambda *args: pytest.fail("no partial source admission"),
    )
    with pytest.raises(QuantError, match="nine-sector"):
        source.verified_market(policy)
