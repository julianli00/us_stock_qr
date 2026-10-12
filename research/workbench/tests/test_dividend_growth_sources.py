from __future__ import annotations

import html
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dividend_growth_sources import (
    factsheet_definition,
    history_from_payload,
    issuer_definition,
    validate_policy,
)
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]
TRANSITION = (
    "NASDAQ US Dividend Achievers Select Index (formerly known as the "
    "Dividend Achievers Select Index) through September 19, 2021, "
    "S&P U.S. Dividend Growers Index thereafter."
)


def vig_document(overview=None, content=None, footnote=TRANSITION):
    profile = {
        "overview": {
            "fundId": "0920",
            "cusip": "921908844",
            "iovTickerSymbol": "VIG.P",
            "inceptionDate": "04/21/2006",
            "assetClass": "Domestic Stock - General",
            "managementStyle": "Index",
            **(overview or {}),
        },
        "content": {
            "benchmarkFootnote": TRANSITION,
            "strategy": (
                "S&P U.S. Dividend Growers Index consists of common stocks; "
                "investing all, or substantially all, of its assets in the stocks "
                "in approximately the same proportion."
            ),
            "productSummary": (
                "Passively managed, full-replication approach. Fund remains fully invested. "
                "Stocks growing their dividends year over year."
            ),
            **(content or {}),
        },
    }
    notes = {"performance": {"footnote_benchmark": {"text": footnote}}}
    return (
        "<title>VIG-Vanguard Dividend Appreciation ETF</title>"
        f'<div data-vgn-funds-profile="{html.escape(json.dumps(profile), quote=True)}"></div>'
        f'<div data-vgn-funds-footnotes="{html.escape(json.dumps(notes), quote=True)}"></div>'
    )


def sdy_document():
    return (
        "<title>SDY: State Street SPDR S&amp;P Dividend ETF</title>"
        "<p>The fund corresponds to an index but its objective is to correspond generally "
        "to the total return performance of the S&amp;P High Yield Dividend Aristocrats Index.</p>"
        "<p>S&amp;P Composite 1500. At least 20 consecutive years; weighted by indicated yield.</p>"
        "<p>Inception Date Nov 08 2005</p><p>USD SDY 78464A763 US78464A7634</p>"
    )


def test_vig_retains_disclosed_index_change_and_does_not_certify_all_historical_screens():
    result = issuer_definition("VIG", vig_document())
    assert result["fund_inception"] == "2006-04-21"
    assert result["prior_benchmark_through"] == "2021-09-19"
    assert result["current_benchmark_from"] == "2021-09-20"
    assert result["ordinary_full_replication_equity_objective_verified"]
    assert not result["exact_consecutive_year_screen_across_history_verified"]
    assert not result["current_rules_applied_backwards_to_old_index"]
    assert not result["original_dated_2021_announcement_verified"]


def test_sdy_exposure_keeps_yield_weighting_and_not_a_pure_independent_factor_claim():
    result = issuer_definition("SDY", sdy_document())
    assert result["current_minimum_consecutive_dividend_growth_years"] == 20
    assert result["current_weighting"] == "indicated_dividend_yield"
    assert not result["yield_and_dividend_persistence_tilts_are_isolated_pure_alpha"]
    assert not result["full_historical_methodology_and_constituents_verified"]


@pytest.mark.parametrize(
    "problem",
    [
        "symbol",
        "cusip",
        "inception",
        "asset_class",
        "strategy",
        "benchmark",
        "conflicting_model",
    ],
)
def test_changed_vig_identity_or_dated_methodology_is_rejected(problem):
    options = {}
    symbol = "VIG"
    if problem == "symbol":
        symbol = "NOBL"
    elif problem == "cusip":
        options["overview"] = {"cusip": "incorrect"}
    elif problem == "inception":
        options["overview"] = {"inceptionDate": "10/01/2026"}
    elif problem == "asset_class":
        options["overview"] = {"assetClass": "Leveraged Derivatives"}
    elif problem == "strategy":
        options["content"] = {"strategy": "Daily leveraged exposure, not common-stock replication"}
    elif problem == "benchmark":
        options["footnote"] = TRANSITION.replace("2021", "2020")
    document = vig_document(**options)
    if problem == "conflicting_model":
        document += vig_document()
    with pytest.raises(QuantError):
        issuer_definition(symbol, document)


@pytest.mark.parametrize(
    "term,replacement",
    [
        ("20 consecutive years", "10 consecutive years"),
        ("weighted by indicated yield", "equal weighted"),
        ("78464A763", "000000000"),
        ("Nov 08 2005", "Nov 08 2025"),
        ("S&amp;P Composite 1500", "unverified current stock basket"),
    ],
)
def test_changed_sdy_mandate_cannot_be_relabelled_as_the_same_fund(term, replacement):
    with pytest.raises(QuantError):
        issuer_definition("SDY", sdy_document().replace(term, replacement))


@pytest.mark.parametrize("raw", [None, "", "<html>Country selection or rate-limit message</html>"])
def test_missing_or_access_gate_html_is_not_a_verified_fund_identity(raw):
    with pytest.raises(QuantError):
        issuer_definition("VIG", raw)


def test_actual_factsheet_definition_is_a_current_crosscheck_not_a_history_archive():
    text = (
        "SDY Inception Date 11/08/2005 CUSIP 78464A763 "
        "S&P High Yield Dividend Aristocrats; at least 20 consecutive years; "
        "weights the stocks by yield."
    )
    result = factsheet_definition(text)
    assert result["symbol"] == "SDY"
    assert not result["factsheet_is_full_historical_screen_archive"]
    with pytest.raises(QuantError):
        factsheet_definition(text.replace("78464A763", "unrelated"))


@pytest.fixture
def policy():
    return read_json(ROOT / "config/dividend-growth-sources.json")


def chart(symbol="VIG"):
    index = sessions("2015-08-10", "2026-10-05")
    close = np.linspace(100, 150, len(index)).tolist()
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": symbol, "currency": "USD", "instrumentType": "ETF"},
                    "timestamp": [
                        int(
                            (day + pd.Timedelta(hours=13, minutes=30))
                            .tz_localize("UTC")
                            .timestamp()
                        )
                        for day in index
                    ],
                    "indicators": {
                        "quote": [
                            {
                                "open": close,
                                "high": [value + 1 for value in close],
                                "low": [value - 1 for value in close],
                                "close": close,
                                "volume": [100000] * len(index),
                            }
                        ],
                        "adjclose": [{"adjclose": close}],
                    },
                    "events": {},
                }
            ],
        }
    }


@pytest.mark.parametrize("symbol", ["VIG", "SDY"])
def test_complete_synthetic_history_follows_the_exact_real_source_contract(policy, symbol):
    frame, result = history_from_payload(chart(symbol), symbol, policy)
    assert len(frame) == result["actual_sessions"] == 2805
    assert result["minimum_actual_volume"] == 100000
    assert result["corporate_actions"]["unexplained_adjustment_jumps"] == 0
    assert not result["synthetic_or_preinception_history_used"]
    assert not result["independent_predictive_factor_or_portfolio_performance_verified"]


@pytest.mark.parametrize(
    "problem", ["wrong_symbol", "zero_volume", "missing_session", "unexplained_action"]
)
def test_missing_or_changed_market_history_cannot_be_made_ready(policy, problem):
    payload = chart()
    record = payload["chart"]["result"][0]
    if problem == "wrong_symbol":
        record["meta"]["symbol"] = "SPY"
    elif problem == "zero_volume":
        record["indicators"]["quote"][0]["volume"][20] = 0
    elif problem == "missing_session":
        record["timestamp"].pop()
        for values in record["indicators"]["quote"][0].values():
            values.pop()
        record["indicators"]["adjclose"][0]["adjclose"].pop()
    else:
        prices = record["indicators"]["adjclose"][0]["adjclose"]
        prices[40:] = [value * 1.1 for value in prices[40:]]
    with pytest.raises(QuantError):
        history_from_payload(payload, "VIG", policy)


def test_rate_limited_html_json_error_is_not_a_successful_history(policy):
    payload = {"chart": {"error": {"code": "Too Many Requests"}, "result": None}}
    with pytest.raises(QuantError, match="error"):
        history_from_payload(payload, "VIG", policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "strategy_evaluation"),
        ("required_market_sessions", 2000),
        ("data_start", "2021-01-01"),
        ("earliest_separate_hypothesis_cycle", "2026-10-17T09:00:00+08:00"),
        ("new_registered_factor_definitions", 2),
        ("new_factor_independence_claimed", True),
        ("portfolio_states_targets_or_returns_computed", True),
        ("current_fund_rules_are_complete_historical_methodology_proof", True),
        ("existing_W42_queue_unchanged", False),
        ("order_authority", True),
        ("order_authority", 0),
    ],
)
def test_no_early_admission_independence_claim_or_scope_relaxation(policy, field, value):
    changed = deepcopy(policy)
    changed[field] = value
    with pytest.raises(QuantError):
        validate_policy(changed)
