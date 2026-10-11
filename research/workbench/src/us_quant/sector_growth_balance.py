from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.growth_factor_satellite import composed_monthly_target
from us_quant.multifactor_stability import FACTORS, corporate_actions
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-growth-balance.json"
SOURCE = ROOT / "data/sector-growth-source-20261011"
PRIOR = ROOT / "evidence/growth_portfolio_protection_20261011_registration.json"
SECTORS = ("XLK", "SOXX")
ISSUER_URLS = {
    "XLK": "https://www.ssga.com/us/en/individual/etfs/state-street-technology-select-sector-spdr-etf-xlk",
    "SOXX": "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf",
}


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("data_start") != "2015-08-10"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("prior_evaluated_configurations") != 112
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("growth_share_of_equity") != 0.50
        or policy.get("volatility_sessions") != 63
        or policy.get("equity_share_min") != 0.30
        or policy.get("equity_share_max") != 0.70
        or policy.get("cash_reserve") != 0.02
        or policy.get("funds")
        != [
            {
                "symbol": "XLK",
                "role": "broad_information_technology_sector",
                "issuer_url": ISSUER_URLS["XLK"],
            },
            {
                "symbol": "SOXX",
                "role": "semiconductor_industry",
                "issuer_url": ISSUER_URLS["SOXX"],
            },
        ]
        or policy.get("candidates")
        != [
            {
                "id": "four_factor_technology_sector50",
                "sector_symbol": "XLK",
                "growth_share_of_equity": 0.50,
            },
            {
                "id": "four_factor_semiconductor_sector50",
                "sector_symbol": "SOXX",
                "growth_share_of_equity": 0.50,
            },
        ]
        or policy.get("comparison", {}).get("candidate_id")
        != "four_factor_growth_gold_monthly_control"
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed sector sources, growth budget or research boundaries changed.")


class IssuerParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.quick_info = []
        self.metadata = {}
        self.documents = []
        self.parts = []
        self.tooltips = []
        self.skip = 0
        self.structured = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "input" and values.get("id") == "fund-quick-info":
            try:
                self.quick_info.append(json.loads(values["value"]))
            except (KeyError, json.JSONDecodeError) as exc:
                raise QuantError("The issuer's scoped fund identity is invalid.") from exc
        if tag == "meta" and "name" in values:
            self.metadata[values["name"]] = values.get("content")
        if tag == "button" and "title" in values:
            self.tooltips.append(values["title"])
        if tag == "script" and values.get("type") == "application/ld+json":
            self.structured = []
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "script" and self.structured is not None:
            try:
                self.documents.append(json.loads("".join(self.structured)))
            except json.JSONDecodeError as exc:
                raise QuantError("The issuer's structured fund identity is invalid.") from exc
            self.structured = None
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, value):
        if self.structured is not None:
            self.structured.append(value)
        if not self.skip and value.strip():
            self.parts.append(" ".join(value.split()))


def issuer_definition(symbol: str, document: str) -> dict:
    if symbol not in SECTORS:
        raise QuantError("Only the two audited sector satellites are supported.")
    parser = IssuerParser()
    parser.feed(document)
    if symbol == "XLK":
        if len(parser.quick_info) != 1 or not isinstance(parser.quick_info[0], dict):
            raise QuantError("The technology source needs one scoped fund identity.")
        attrs = parser.quick_info[0].get("attrs", {})
        expected = {
            "fund-ticker": "XLK",
            "isin": "US81369Y8030",
            "inception-date": "Dec 16 1998",
            "benchmark": "Technology Select Sector Index",
        }
        text = " ".join(parser.parts)
        if (
            not isinstance(attrs, dict)
            or any(attrs.get(key, {}).get("value") != value for key, value in expected.items())
            or not all(
                term in text
                for term in (
                    "before expenses, correspond generally to the price and yield performance",
                    "Technology Select Sector Index",
                    "technology sector of the S&P 500 Index",
                )
            )
        ):
            raise QuantError("The technology source does not prove the expected index mandate.")
        return {
            "symbol": symbol,
            "role": "information_technology_sector_not_independent_factor",
            "fund_inception": "1998-12-16",
            "benchmark": expected["benchmark"],
            "intended_equity_index_return_multiple": 1.0,
            "full_historical_methodology_verified": False,
        }
    graph = []
    for document in parser.documents:
        if isinstance(document, dict) and isinstance(document.get("@graph"), list):
            graph.extend(document["@graph"])
    prefix = ISSUER_URLS["SOXX"]
    products = [node for node in graph if node.get("@id") == prefix + "#fund"]
    facts = [node for node in graph if node.get("@id") == prefix + "#key-facts"]
    descriptions = [node for node in graph if node.get("@id") == prefix + "#fund-description"]
    if len(products) != 1 or len(facts) != 1 or len(descriptions) != 1:
        raise QuantError("The semiconductor source needs unique scoped fund records.")
    fields = {row.get("name"): row.get("value") for row in facts[0].get("additionalProperty", [])}
    product = products[0]
    if (
        parser.metadata.get("injectable-productTicker") != "soxx"
        or parser.metadata.get("injectable-productAssetClass") != "eq"
        or product.get("alternateName") != "SOXX"
        or product.get("category") != "Equity"
        or product.get("url") != prefix
        or facts[0].get("about", {}).get("@id") != prefix + "#fund"
        or descriptions[0].get("about", {}).get("@id") != prefix + "#fund"
        or fields.get("Fund Inception") != "Jul 10, 2001"
        or fields.get("Asset Class") != "Equity"
        or fields.get("Benchmark Index") != "NYSE Semiconductor Index"
        or "U.S. equity index" not in descriptions[0].get("description", "")
        or not any(
            "On 6/21/2021 SOXX began to track the NYSE Semiconductor Index." in text
            and "PHLX SOX Semiconductor Sector Index" in text
            for text in parser.tooltips
        )
    ):
        raise QuantError("The semiconductor identity or disclosed benchmark transition differs.")
    return {
        "symbol": symbol,
        "role": "semiconductor_industry_not_independent_factor",
        "fund_inception": "2001-07-10",
        "benchmark": fields["Benchmark Index"],
        "intended_equity_index_return_multiple": 1.0,
        "disclosed_index_transition": {
            "session": "2021-06-21",
            "previous": "PHLX SOX Semiconductor Sector Index",
            "current": "NYSE Semiconductor Index",
        },
        "full_historical_methodology_verified": False,
    }


def validate_primary_documents(symbol: str, directory: Path) -> None:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    filename = "XLK-factsheet.pdf" if symbol == "XLK" else "SOXX-summary-underlying.pdf"
    path = directory / filename
    if not path.read_bytes().startswith(b"%PDF-"):
        raise QuantError("An HTML document viewer is not primary PDF evidence.")
    try:
        text = " ".join(
            " ".join((page.extract_text() or "").split()) for page in PdfReader(path).pages
        )
    except (PdfReadError, ValueError) as exc:
        raise QuantError(f"The primary {symbol} mandate document cannot be parsed.") from exc
    required = (
        ("Technology Select Sector Index", "price and yield performance", "CUSIP 81369Y803")
        if symbol == "XLK"
        else (
            "iShares Semiconductor ETF",
            "NYSE Semiconductor Index",
            "representative sampling indexing strategy",
            "investment profile similar to that of an applicable underlying index",
            "may use derivatives to gain or reduce",
        )
    )
    if not all(term in text for term in required):
        raise QuantError(f"The primary {symbol} document does not confirm index replication.")


def verified_market(policy: dict) -> MarketData:
    from us_quant.research_program import review_market, safe_file, verified_etf_market

    validate_policy(policy)
    manifest = read_json(safe_file(ROOT, (SOURCE / "manifest.json").relative_to(ROOT).as_posix()))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or set(manifest.get("sources", {})) != set(SECTORS)
        or manifest.get("synthetic_history") is not False
        or manifest.get("strategy_outcomes_computed") is not False
    ):
        raise QuantError("The sector manifest differs from the frozen source-admission policy.")
    for name, digest in manifest.get("additional_source_files", {}).items():
        safe_file(ROOT, (SOURCE / name).relative_to(ROOT).as_posix(), digest)
    prior = read_json(PRIOR)
    actual = verified_etf_market(prior["readiness"], ROOT)
    data = review_market({"market": prior["candidates"][0]["spec"]["market"]}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        if not getattr(data, name).equals(getattr(actual, name)):
            if (
                not data.close.index.equals(actual.close.index)
                or not data.close.columns.equals(actual.close.columns)
                or not np.allclose(
                    getattr(data, name), getattr(actual, name), rtol=1e-10, atol=1e-9
                )
            ):
                raise QuantError("The original audited growth/factor market was substituted.")
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The sector study must preserve the original risk-free proxy.")
    funds = {}
    for symbol in SECTORS:
        record = manifest["sources"][symbol]
        required = {f"{symbol}-issuer.html", f"{symbol}-raw.json", f"{symbol}.csv"}
        primary = "XLK-factsheet.pdf" if symbol == "XLK" else "SOXX-summary-underlying.pdf"
        if (
            not required.union({primary}).issubset(record.get("files", {}))
            or record.get("issuer_url") != ISSUER_URLS[symbol]
        ):
            raise QuantError(
                "Sector admission requires actual issuer, primary document and quotes."
            )
        for name, digest in record["files"].items():
            safe_file(ROOT, (SOURCE / name).relative_to(ROOT).as_posix(), digest)
        definition = issuer_definition(symbol, (SOURCE / f"{symbol}-issuer.html").read_text())
        validate_primary_documents(symbol, SOURCE)
        payload = read_json(SOURCE / f"{symbol}-raw.json")
        frame = parse_chart(payload, symbol, policy["data_start"], policy["as_of"])
        retained = pd.read_csv(SOURCE / f"{symbol}.csv", index_col="date", parse_dates=True)
        if (
            record.get("issuer_definition") != definition
            or not frame.index.equals(data.close.index)
            or not frame.index.equals(retained.index)
            or not frame.columns.equals(retained.columns)
            or not np.allclose(frame, retained, rtol=1e-12, atol=1e-9)
            or corporate_actions(payload, frame, symbol) != record.get("actions")
            or record.get("sessions") != len(frame)
            or record.get("first_session") != str(frame.index[0].date())
            or record.get("last_session") != str(frame.index[-1].date())
            or record.get("minimum_volume") != float(frame["volume"].min())
        ):
            raise QuantError("The sector source identity, sessions or corporate actions changed.")
        funds[symbol] = frame

    def panel(name, field):
        result = getattr(data, name).copy()
        for symbol, frame in funds.items():
            result[symbol] = frame[field]
        return result

    result = MarketData(
        panel("open", "adj_open"),
        panel("close", "adj_close"),
        panel("raw_close", "close"),
        panel("volume", "volume"),
        data.risk_free,
    )
    result.validate()
    return result


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, *SECTORS}:
        raise QuantError("The original market and both admitted sector funds must be present.")
    result = {}
    for candidate in policy["candidates"]:
        symbol = candidate["sector_symbol"]
        history = data.close.drop(columns="QQQ").rename(columns={symbol: "QQQ"})
        targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for i, day in enumerate(data.close.index):
            if i >= 63 and is_month_end(day):
                # Keep the frozen budget helper unchanged; only its growth-price input differs.
                target = composed_monthly_target(history.iloc[: i + 1], candidate)
                target = target.rename(index={"QQQ": symbol}).reindex(
                    data.close.columns, fill_value=0.0
                )
                if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
                    raise QuantError("Sector allocation cannot short funds or borrow cash.")
                targets.loc[day] = target
        result[candidate["id"]] = targets
    return result


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    data = verified_market(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Sector research evidence must remain inside the isolated workbench.")
    new_output_directory(output)
    market = {}
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        market[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            ROOT / "config/growth-factor-satellite.json",
            Path(__file__).with_name("growth_factor_satellite.py"),
            Path(__file__).with_name("multifactor_stability.py"),
            SOURCE / "manifest.json",
            PRIOR,
        )
    }
    manifest = read_json(SOURCE / "manifest.json")
    for record in manifest["sources"].values():
        frozen.update(
            {
                (SOURCE / name).relative_to(ROOT).as_posix(): digest
                for name, digest in record["files"].items()
            }
        )
    source = {
        "adapter": "sector_growth_balance_20261011",
        "policy": POLICY.relative_to(ROOT).as_posix(),
        "policy_sha256": file_digest(POLICY),
        "factor_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
        "factor_manifest_sha256": file_digest(SOURCE / "manifest.json"),
    }
    readiness = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "data_scope": "factor_etf_portfolio",
        "verified_etf_source": source,
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": source["factor_manifest"],
                "evidence_sha256": source["factor_manifest_sha256"],
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "sector_exposures_are_not_new_factor_definitions": True,
        "full_historical_methodology_verified": False,
        "direct_stock_data_still_blocked": True,
        "prospective_archive_includes_sector_funds": False,
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "strategy_outcomes_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare two fixed sector/factor/gold comparisons."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/sector-growth-prepared-20261011"
    )
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {
                    "prepared": [row["id"] for row in result["specs"]],
                    "strategy_outcomes_computed": False,
                }
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Sector research blocked: {exc}\n")


if __name__ == "__main__":
    main()
