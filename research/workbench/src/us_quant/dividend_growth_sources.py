from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import parse_chart
from us_quant.multifactor_stability import corporate_actions
from us_quant.storage import digest_json

FUNDS = [
    {
        "symbol": "VIG",
        "issuer_url": "https://investor.vanguard.com/investment-products/etfs/profile/vig",
        "fund_inception": "2006-04-21",
        "role": "dividend_growth_equity_exposure",
        "definition_scope": "actual_fund_returns_with_disclosed_2021_benchmark_change",
    },
    {
        "symbol": "SDY",
        "issuer_url": "https://www.ssga.com/us/en/individual/etfs/spdr-sp-dividend-etf-sdy",
        "fund_inception": "2005-11-08",
        "role": "dividend_growth_equity_exposure",
        "definition_scope": "actual_yield_weighted_dividend_persistence_fund_returns",
    },
]


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "dividend_growth_source_readiness_20261011",
        "scope": "source_preparation_only",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "required_market_sessions": 2805,
        "earliest_separate_hypothesis_cycle": "2026-10-24T09:00:00+08:00",
        "existing_W42_queue_unchanged": True,
        "funds": FUNDS,
        "new_registered_factor_definitions": 0,
        "new_registered_candidates": 0,
        "new_strategy_evaluations": 0,
        "portfolio_states_targets_or_returns_computed": False,
        "new_factor_independence_claimed": False,
        "current_fund_rules_are_complete_historical_methodology_proof": False,
        "existing_frozen_forward_rules_changed": False,
        "order_authority": False,
        "automatic_live_deployment": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or (type(value) is bool and policy.get(key) is not value)
        for key, value in expected.items()
    ):
        raise QuantError(
            "The source-only dividend scope, complete history or future cycle changed."
        )


class SourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.attributes: dict[str, str] = {}
        self.skip = 0

    def handle_starttag(self, tag, attrs) -> None:
        values = dict(attrs)
        for key in ("data-vgn-funds-profile", "data-vgn-funds-footnotes"):
            if key in values:
                if key in self.attributes or not isinstance(values[key], str):
                    raise QuantError(
                        "Use a unique complete issuer model, not conflicting identities."
                    )
                self.attributes[key] = values[key]
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag) -> None:
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data) -> None:
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def plain(document: str) -> str:
    if not isinstance(document, str) or not document.strip():
        raise QuantError("An actual complete issuer document is required.")
    parser = SourceParser()
    parser.feed(document)
    result = " ".join(" ".join(parser.parts).split())
    return re.sub(r"\bTM\b", "", result.replace("\u00ae", "").replace("\u2122", "")).strip()


def issuer_definition(symbol: str, document: str) -> dict:
    if symbol not in {fund["symbol"] for fund in FUNDS}:
        raise QuantError(
            "Only the explicitly inspected VIG and SDY issuer definitions are supported."
        )
    if not isinstance(document, str) or not document.strip():
        raise QuantError("An actual issuer document is required, not a fund-name assertion.")
    parser = SourceParser()
    parser.feed(document)
    text = plain(document)
    if symbol == "VIG":
        try:
            profile = json.loads(parser.attributes["data-vgn-funds-profile"])
            footnotes = json.loads(parser.attributes["data-vgn-funds-footnotes"])
            digest_json(profile)
            digest_json(footnotes)
            overview, content = profile["overview"], profile["content"]
            footnote = footnotes["performance"]["footnote_benchmark"]["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise QuantError(
                "The actual VIG profile and benchmark footnotes must be complete."
            ) from exc
        if not isinstance(overview, dict) or not isinstance(content, dict):
            raise QuantError(
                "The VIG issuer model does not contain its complete identity and objective."
            )
        identities = {
            "fundId": "0920",
            "cusip": "921908844",
            "iovTickerSymbol": "VIG.P",
            "inceptionDate": "04/21/2006",
            "assetClass": "Domestic Stock - General",
            "managementStyle": "Index",
        }
        transition = (
            "NASDAQ US Dividend Achievers Select Index (formerly known as the "
            "Dividend Achievers Select Index) through September 19, 2021, "
            "S&P U.S. Dividend Growers Index thereafter."
        )
        if (
            any(overview.get(key) != value for key, value in identities.items())
            or "VIG-Vanguard Dividend Appreciation ETF" not in text
            or plain(content.get("benchmarkFootnote")) != transition
            or plain(footnote) != transition
            or not all(
                term in plain(content.get("strategy"))
                for term in (
                    "S&P U.S. Dividend Growers Index",
                    "common stocks",
                    "investing all, or substantially all, of its assets in the stocks",
                    "approximately the same proportion",
                )
            )
            or not all(
                term in plain(content.get("productSummary"))
                for term in (
                    "Passively managed, full-replication approach",
                    "Fund remains fully invested",
                    "growing their dividends year over year",
                )
            )
        ):
            raise QuantError(
                "VIG identity, ordinary equity objective or disclosed benchmark change differs."
            )
        return {
            "symbol": "VIG",
            "fund_inception": "2006-04-21",
            "cusip": "921908844",
            "implementation_role": "dividend_growth_equity_exposure",
            "current_benchmark": "S&P U.S. Dividend Growers Index",
            "prior_benchmark": "NASDAQ US Dividend Achievers Select Index",
            "prior_benchmark_through": "2021-09-19",
            "current_benchmark_from": "2021-09-20",
            "ordinary_full_replication_equity_objective_verified": True,
            "exact_consecutive_year_screen_across_history_verified": False,
            "full_historical_methodology_and_constituents_verified": False,
            "original_dated_2021_announcement_verified": False,
            "current_rules_applied_backwards_to_old_index": False,
        }
    required = (
        "S&P High Yield Dividend Aristocrats",
        "S&P Composite 1500",
        "20 consecutive years",
        "weighted by indicated yield",
        "Inception Date Nov 08 2005",
        "USD SDY 78464A763 US78464A7634",
    )
    if (
        "Dividend ETF" not in text
        or not all(term in text for term in required)
        or not re.search(r"correspond generally to the total return performance", text)
    ):
        raise QuantError(
            "SDY issuer identity, ordinary index exposure or dividend/yield screen differs."
        )
    return {
        "symbol": "SDY",
        "fund_inception": "2005-11-08",
        "cusip": "78464A763",
        "isin": "US78464A7634",
        "implementation_role": "dividend_growth_equity_exposure",
        "current_benchmark": "S&P High Yield Dividend Aristocrats Index",
        "current_index_universe": "S&P Composite 1500",
        "current_minimum_consecutive_dividend_growth_years": 20,
        "current_weighting": "indicated_dividend_yield",
        "ordinary_single_index_equity_objective_verified": True,
        "yield_and_dividend_persistence_tilts_are_isolated_pure_alpha": False,
        "full_historical_methodology_and_constituents_verified": False,
    }


def factsheet_definition(document: str) -> dict:
    text = plain(document)
    if not all(
        term in text
        for term in (
            "SDY",
            "Inception Date 11/08/2005",
            "CUSIP 78464A763",
            "S&P",
            "High Yield",
            "Dividend Aristocrats",
            "at least 20 consecutive years",
            "weights the stocks by yield",
        )
    ):
        raise QuantError(
            "The advertised SDY factsheet does not substantiate the exact inspected fund."
        )
    return {
        "symbol": "SDY",
        "inception": "2005-11-08",
        "cusip": "78464A763",
        "current_minimum_consecutive_dividend_growth_years": 20,
        "current_weighting": "indicated_dividend_yield",
        "factsheet_is_full_historical_screen_archive": False,
    }


def history_from_payload(payload: dict, symbol: str, policy: dict) -> tuple[pd.DataFrame, dict]:
    validate_policy(policy)
    fund = next((fund for fund in FUNDS if fund["symbol"] == symbol), None)
    if fund is None:
        raise QuantError(
            "Only the inspected ordinary dividend funds have a source-history contract."
        )
    if pd.Timestamp(fund["fund_inception"]) >= pd.Timestamp(policy["data_start"]):
        raise QuantError("Dividend source history must be entirely post-inception.")
    frame = parse_chart(payload, symbol, policy["data_start"], policy["as_of"])
    if not frame.index.equals(sessions("2015-08-10", "2026-10-05")) or len(frame) != 2805:
        raise QuantError(
            "A source history cannot shorten or fill the original complete market grid."
        )
    if (frame["adj_close"].pct_change(fill_method=None).iloc[1:].abs() > 0.60).any():
        raise QuantError("A large dividend-fund price move needs explicit source/action review.")
    actions = corporate_actions(payload, frame, symbol)
    return frame, {
        "symbol": symbol,
        "actual_sessions": len(frame),
        "first_session": str(frame.index[0].date()),
        "last_session": str(frame.index[-1].date()),
        "minimum_actual_volume": float(frame["volume"].min()),
        "corporate_actions": actions,
        "synthetic_or_preinception_history_used": False,
        "independent_predictive_factor_or_portfolio_performance_verified": False,
    }
