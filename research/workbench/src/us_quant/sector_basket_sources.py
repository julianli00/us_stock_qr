from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.factor_research import snapshot_descriptor
from us_quant.dual_horizon import load_protocol
from us_quant.multifactor_stability import corporate_actions
from us_quant.research_program import review_market, safe_file
from us_quant.sector_growth_balance import IssuerParser
from us_quant.storage import digest_json, file_digest, read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-basket-sources.json"
SOURCE = ROOT / "data/sector-basket-sources-20261011"
SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
BENCHMARKS = {
    "XLB": "Materials Select Sector Index",
    "XLE": "Energy Select Sector Index",
    "XLF": "Financial Select Sector Index",
    "XLI": "Industrial Select Sector Index",
    "XLK": "Technology Select Sector Index",
    "XLP": "Consumer Staples Select Sector Index",
    "XLU": "Utilities Select Sector Index",
    "XLV": "Health Care Select Sector Index",
    "XLY": "Consumer Discretionary Select Sector Index",
}


def validate(policy: dict) -> None:
    contract = policy.get("source_contract", {}) if isinstance(policy, dict) else {}
    urls = policy.get("issuer_urls", {}) if isinstance(policy, dict) else {}
    if (
        not isinstance(policy, dict)
        or policy.get("schema_version") != 1
        or policy.get("study_id") != "diversified_sector_basket_sources_20261011"
        or policy.get("scope") != "existing_registered_sector_prices_and_primary_issuer_identity"
        or policy.get("data_start") != "2015-08-10" or policy.get("as_of") != "2026-10-05"
        or policy.get("sector_symbols") != list(SECTORS) or policy.get("benchmarks") != BENCHMARKS
        or not isinstance(contract, dict) or not isinstance(urls, dict)
        or set(urls) != set(SECTORS)
        or any(
            not isinstance(url, str) or urlparse(url).scheme != "https"
            or urlparse(url).hostname != "www.ssga.com"
            or not url.endswith(f"-{symbol.lower()}")
            for symbol, url in urls.items()
        )
        or any(
            not isinstance(policy.get(key), dict)
            or set(policy[key]) != {"path", "sha256"}
            or any(not isinstance(value, str) for value in policy[key].values())
            for key in ("original_registration", "original_policy", "core_registration")
        )
        or any(
            contract.get(key) is not True for key in (
                "use_existing_registered_raw_and_csv_cache",
                "raw_csv_parity_and_corporate_actions_required",
                "complete_2805_post_inception_sessions_required",
                "issuer_scoped_identity_and_advertised_pdf_required",
                "code_fingerprint_drift_is_not_ignored_for_old_strategy_replay",
            )
        )
        or contract.get("new_quote_requests") != 0
        or contract.get("legacy_registration_or_source_hashes_rewritten") is not False
        or contract.get("access_or_rate_limit_bypass") is not False
        or policy.get("new_registered_candidates") != 0
        or policy.get("new_strategy_evaluations") != 0
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("portfolio_targets_or_returns_computed") is not False
        or policy.get("existing_W42_queue_changed") is not False
        or policy.get("order_authority") is not False
        or policy.get("investment_objective_verified") is not False
    ):
        raise QuantError("The complete original-sector source contract or research scope changed.")


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json({
        name: file_digest(folder / name)
        for name in (
            "sector_basket_sources.py", "sector_growth_balance.py", "multifactor_stability.py",
            "factor_research.py", "data.py", "calendar.py", "dual_horizon.py", "storage.py",
        )
    })


class SectorIssuerParser(IssuerParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])


def issuer_definition(symbol: str, document: str, policy: dict) -> dict:
    validate(policy)
    if symbol not in SECTORS:
        raise QuantError("Only original predeclared sector ETF identities may be admitted.")
    parser = SectorIssuerParser()
    parser.feed(document)
    if len(parser.quick_info) != 1 or not isinstance(parser.quick_info[0], dict):
        raise QuantError("Require one scoped issuer fund identity, not a general product portal.")
    attrs = parser.quick_info[0].get("attrs", {})
    if not isinstance(attrs, dict):
        raise QuantError("The issuer fund attributes are incomplete.")
    values = {key: item.get("value") for key, item in attrs.items() if isinstance(item, dict)}
    isin = values.get("isin")
    text = " ".join(parser.parts)
    if (
        values.get("fund-ticker") != symbol
        or values.get("benchmark") != BENCHMARKS[symbol]
        or values.get("inception-date") != "Dec 16 1998"
        or not isinstance(isin, str) or not re.fullmatch(r"US[0-9A-Z]{9}[0-9]", isin)
        or "price and yield performance" not in text
        or "S&P 500 Index" not in text or BENCHMARKS[symbol] not in text
    ):
        raise QuantError("The primary issuer does not substantiate the expected ordinary sector fund.")
    ending = f"/factsheet-us-en-{symbol.lower()}.pdf"
    urls = {
        urljoin(policy["issuer_urls"][symbol], link)
        for link in parser.links if urlparse(link).path.lower().endswith(ending)
    }
    if len(urls) != 1:
        raise QuantError("The exact fund factsheet must be uniquely advertised by its own issuer page.")
    url = urls.pop()
    if urlparse(url).scheme != "https" or urlparse(url).hostname != "www.ssga.com":
        raise QuantError("An unrelated or unverified factsheet provider cannot substitute for the issuer.")
    return {
        "symbol": symbol, "isin": isin, "cusip": isin[2:11],
        "fund_inception": "1998-12-16", "benchmark": BENCHMARKS[symbol],
        "ordinary_sector_index_fund_objective": True,
        "intended_equity_index_return_multiple": 1.0,
        "factsheet_url": url, "full_historical_sector_definitions_verified": False,
        "historical_constituents_or_gross_notional_archive_verified": False,
    }


def factsheet_definition(text: str, identity: dict) -> dict:
    normalized = " ".join(text.split())
    if (
        identity["benchmark"] not in normalized
        or "price and yield performance" not in normalized
        or f"CUSIP {identity['cusip']}" not in normalized
        or "12/16/1998" not in normalized
    ):
        raise QuantError("The actual advertised factsheet disagrees with the scoped sector identity.")
    return {
        "symbol": identity["symbol"], "cusip": identity["cusip"],
        "benchmark": identity["benchmark"], "fund_inception": identity["fund_inception"],
        "primary_identity_crosscheck_passed": True,
        "factsheet_is_full_historical_mandate_archive": False,
    }


def verify_factsheet(path: Path, identity: dict) -> dict:
    if not path.read_bytes().startswith(b"%PDF-"):
        raise QuantError("The actual advertised factsheet is not PDF evidence.")
    try:
        document = PdfReader(path)
        if document.is_encrypted or not 1 <= len(document.pages) <= 16:
            raise QuantError("Encrypted or unbounded documents cannot support source admission.")
        text = " ".join(page.extract_text() or "" for page in document.pages)
    except (PdfReadError, ValueError) as exc:
        raise QuantError("The actual primary factsheet could not be parsed.") from exc
    return {**factsheet_definition(text, identity), "actual_pdf_pages": len(document.pages)}


def cached_sector_prices(policy: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    validate(policy)
    old_policy = read_json(safe_file(
        ROOT, policy["original_policy"]["path"], policy["original_policy"]["sha256"],
    ))
    registration = read_json(safe_file(
        ROOT, policy["original_registration"]["path"], policy["original_registration"]["sha256"],
    ))
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    if old_policy["sector_universe"] != list(SECTORS) or registration["windows"] != comparison.windows():
        raise QuantError("Keep the original sector universe and unchanged formal window contract.")
    cache = ROOT / "data/factor-round-20261007"
    for label, symbols in (
        ("base", set(comparison.symbols) | {"IRX"}),
        ("supplement", set(old_policy["supplement_symbols"])),
    ):
        if snapshot_descriptor(cache / label, symbols, comparison) != registration["sources"][label]:
            raise QuantError("A registered raw source, CSV, manifest or acquisition time changed.")
    dates = sessions(policy["data_start"], policy["as_of"])
    prices, checks = {}, {}
    for symbol in SECTORS:
        csv_path, raw_path = cache / "supplement" / f"{symbol}.csv", cache / "supplement/raw" / f"{symbol}.json"
        full = pd.read_csv(csv_path, index_col="date", parse_dates=["date"], float_precision="round_trip")
        payload = read_json(raw_path)
        chart = payload["chart"]["result"][0]
        if (
            chart["meta"].get("symbol") != symbol
            or chart["meta"].get("instrumentType") != "ETF" or chart["meta"].get("currency") != "USD"
        ):
            raise QuantError("The registered price response does not identify the exact ordinary USD ETF.")
        parsed = parse_chart(payload, symbol, comparison.data_start, comparison.as_of)
        if (
            not parsed.index.equals(full.index) or not parsed.columns.equals(full.columns)
            or not np.allclose(parsed, full, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("Original raw JSON and registered cached CSV are not the same price evidence.")
        frame = full.loc[policy["data_start"]:policy["as_of"]]
        if (
            len(frame) != 2805 or not frame.index.equals(dates)
            or not np.isfinite(frame.to_numpy()).all()
            or (frame[["adj_open", "adj_close", "close", "volume"]] <= 0).any().any()
        ):
            raise QuantError("Require all 2805 actual post-inception observations, not a reduced window.")
        actions = corporate_actions(payload, full, symbol)
        prices[symbol] = frame
        checks[symbol] = {
            "csv_path": csv_path.relative_to(ROOT).as_posix(), "csv_sha256": file_digest(csv_path),
            "raw_path": raw_path.relative_to(ROOT).as_posix(), "raw_sha256": file_digest(raw_path),
            "first_session": dates[0].strftime("%Y-%m-%d"),
            "last_session": dates[-1].strftime("%Y-%m-%d"),
            "complete_sessions": len(frame),
            "maximum_raw_csv_price_error": float(abs(parsed.to_numpy() - full.to_numpy()).max()),
            **actions,
        }
    return prices, {
        "source_adapter": "versioned_price_only_registered_raw_csv_verification",
        "original_strategy_implementation_replayed": False,
        "original_registered_sources_and_acquisition_times_preserved": True,
        "original_supplement_acquired_at": registration["sources"]["supplement"]["retrieved_at"],
        "original_core_and_sector_source_descriptors_match": True,
        "fund_checks": checks, "new_price_requests": 0,
    }


def verified_market(policy: dict) -> MarketData:
    validate(policy)
    manifest = read_json(safe_file(SOURCE, "manifest.json"))
    funds = manifest.get("funds", {}) if isinstance(manifest, dict) else {}
    files = manifest.get("file_sha256", {}) if isinstance(manifest, dict) else {}
    if (
        not isinstance(manifest, dict)
        or manifest.get("source_policy_sha256") != file_digest(POLICY)
        or manifest.get("source_validator_sha256") != fingerprint()
        or manifest.get("all_sources_ready") is not True
        or not isinstance(funds, dict) or set(funds) != set(SECTORS)
        or not isinstance(files, dict)
        or any(not isinstance(path, str) or not isinstance(digest, str) for path, digest in files.items())
        or any(
            not isinstance(fund, dict)
            or not {"issuer_path", "issuer_sha256", "factsheet_path", "factsheet_sha256",
                    "identity", "factsheet_identity"}.issubset(fund)
            for fund in funds.values()
        )
    ):
        raise QuantError("The complete nine-sector issuer and actual-price source must be ready.")
    for name, digest in files.items():
        safe_file(ROOT, name, digest)
    identities = []
    for symbol in SECTORS:
        fund = manifest["funds"][symbol]
        issuer = safe_file(ROOT, fund["issuer_path"], fund["issuer_sha256"])
        identity = issuer_definition(symbol, issuer.read_text(), policy)
        factsheet = safe_file(ROOT, fund["factsheet_path"], fund["factsheet_sha256"])
        actual_pdf = verify_factsheet(factsheet, identity)
        if identity != fund["identity"] or actual_pdf != fund["factsheet_identity"]:
            raise QuantError("A retained source identity disagrees with the actual issuer/PDF.")
        identities.append(identity["cusip"])
    if len(set(identities)) != 9:
        raise QuantError("Nine symbols cannot be aliases of the same inspected fund.")
    prices, _ = cached_sector_prices(policy)
    original = read_json(safe_file(
        ROOT, policy["core_registration"]["path"], policy["core_registration"]["sha256"],
    ))
    specs = [row["spec"] for row in original["candidates"] if row["spec"]["id"] == "four_factor_gold_risk_balance"]
    if len(specs) != 1:
        raise QuantError("Require the exact original four-factor market registration.")
    core = review_market({"market": specs[0]["market"]}, ROOT)

    def panel(name: str, field: str) -> pd.DataFrame:
        result = getattr(core, name).copy()
        for symbol in SECTORS:
            if not prices[symbol].index.equals(result.index):
                raise QuantError("Sector and original factor source calendars differ.")
            result[symbol] = prices[symbol][field]
        return result

    result = MarketData(
        panel("open", "adj_open"), panel("close", "adj_close"),
        panel("raw_close", "close"), panel("volume", "volume"), core.risk_free,
    )
    result.validate()
    if len(result.close.columns) != 17 or len(result.close) != 2805:
        raise QuantError("Require the full exact seventeen-asset original-factor/sector market.")
    return result
