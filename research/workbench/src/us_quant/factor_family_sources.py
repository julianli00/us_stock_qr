from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.factor_replication import original_market
from us_quant.multifactor_stability import corporate_actions
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/factor-family-expansion.json"
SOURCE = ROOT / "data/new-factor-family-source-20261011"
ROLE_MAP = {
    "size_exposure_ijr_v1": ("IJR", "size"),
    "net_buyback_exposure_pkw_v1": ("PKW", "share_issuance"),
}


def validate_policy(policy: dict) -> None:
    factors = policy.get("factors", [])
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("data_start") != "2015-08-10"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("eligible_not_before") != "2026-10-17T09:00:00+08:00"
        or policy.get("planned_week") != "2026-W42"
        or policy.get("new_factor_definitions_in_queue") != 2
        or policy.get("new_strategy_configurations") != 0
        or policy.get("funds")
        != [
            {"symbol": "IJR", "factor_id": "size_exposure_ijr_v1", "family": "size"},
            {"symbol": "PKW", "factor_id": "net_buyback_exposure_pkw_v1", "family": "share_issuance"},
        ]
        or not isinstance(factors, list)
        or len(factors) != 2
        or {row.get("id") for row in factors} != set(ROLE_MAP)
        or policy.get("order_authority") is not False
        or policy.get("automatic_live_deployment") is not False
    ):
        raise QuantError("The audited new-family sources, eligibility or authority changed.")
    from us_quant.research_program import validate_factor

    for factor in factors:
        validate_factor(factor)
        symbol, family = ROLE_MAP[factor["id"]]
        if (
            factor["family"] != family
            or factor["parameters"].get("symbol") != symbol
            or factor["parameters"].get("implementation") != "actual_ETF_exposure"
            or factor["parameters"].get("nominal_product_multiple") != 1
        ):
            raise QuantError("A new factor must match its audited ETF economic family.")


def definitions(policy: dict) -> dict:
    validate_policy(policy)
    return {factor["id"]: factor for factor in policy["factors"]}


def issuer_definition(symbol: str, document: str) -> dict:
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.metadata = {}
            self.descriptions = []
            self.parts = []
            self.skip = 0

        def walk(self, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "description" and isinstance(item, str):
                        self.descriptions.append(item)
                    self.walk(item)
            elif isinstance(value, list):
                for item in value:
                    self.walk(item)

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "meta" and "name" in values:
                self.metadata[values["name"]] = values.get("content")
            if "data-model-json" in values:
                try:
                    self.walk(json.loads(values["data-model-json"]))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise QuantError("The issuer's embedded definition is invalid.") from exc
            if tag in ("script", "style"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = max(0, self.skip - 1)

        def handle_data(self, value):
            if not self.skip and value.strip():
                self.parts.append(" ".join(value.split()))

    parser = Parser()
    parser.feed(document)
    if symbol == "IJR":
        fields = {
            name: parser.parts[index + 1]
            for index, name in enumerate(parser.parts[:-1])
            if name
            in {"Benchmark Index", "Fund Inception", "Asset Class", "Bloomberg Index Ticker"}
        }
        if (
            parser.metadata.get("injectable-productTicker") != "ijr"
            or parser.metadata.get("injectable-productAssetClass") != "eq"
            or fields.get("Benchmark Index") != "S&P SmallCap 600 Index"
            or fields.get("Fund Inception") != "May 22, 2000"
            or fields.get("Asset Class") != "Equity"
            or fields.get("Bloomberg Index Ticker") != "SPTRSMCP"
        ):
            raise QuantError("The source does not identify the expected US small-cap fund.")
        return {
            "symbol": symbol,
            "family": "size",
            "fund_inception": "2000-05-22",
            "benchmark": fields["Benchmark Index"],
            "nominal_daily_return_multiple": 1.0,
            "futures_may_offset_cash_for_tracking": True,
            "full_historical_methodology_verified": False,
        }
    if symbol == "PKW":
        required = {
            "ticker": "PKW",
            "isin": "US46137V3087",
            "shareClassInceptionDate": "2006-12-20",
            "assetClass": "Equity",
            "region": "United States",
            "investmentMethod": "Passive",
            "bloombergTicker": "DRBTR",
        }
        terms = (
            "index fund",
            "Nasdaq US BuyBack Achievers",
            "net reduction in shares outstanding of 5% or more",
            "trailing 12 months",
        )
        if any(parser.metadata.get(key) != value for key, value in required.items()) or not any(
            all(term in text for term in terms) for text in parser.descriptions
        ):
            raise QuantError("The source does not prove the expected net-share-reduction mandate.")
        return {
            "symbol": symbol,
            "family": "share_issuance",
            "fund_inception": "2006-12-20",
            "benchmark": "Nasdaq US BuyBack Achievers",
            "index_code": "DRB",
            "minimum_net_share_reduction": 0.05,
            "lookback_months": 12,
            "nominal_daily_return_multiple": 1.0,
            "full_historical_methodology_verified": False,
        }
    raise QuantError("Only the two explicitly audited new-family funds are supported.")


def verified_market(policy: dict) -> MarketData:
    validate_policy(policy)
    from us_quant.research_program import safe_file

    manifest = read_json(
        safe_file(ROOT, (SOURCE / "manifest.json").relative_to(ROOT).as_posix())
    )
    if (
        manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or set(manifest.get("sources", {})) != {"IJR", "PKW"}
        or set(manifest.get("additional_source_files", {}))
        != {"PKW-index.html", "SP-methodology-response.bin"}
        or manifest.get("synthetic_history") is not False
        or manifest.get("strategy_outcomes_computed") is not False
    ):
        raise QuantError("The new-family manifest no longer matches the audited source policy.")
    for name, digest in manifest["additional_source_files"].items():
        safe_file(ROOT, (SOURCE / name).relative_to(ROOT).as_posix(), digest)
    if "NASDAQ US BuyBack Achievers" not in (SOURCE / "PKW-index.html").read_text():
        raise QuantError("The Nasdaq source no longer identifies the expected buyback index.")
    original = original_market()
    funds = {}
    for symbol, record in manifest["sources"].items():
        expected_files = {f"{symbol}-issuer.html", f"{symbol}-raw.json", f"{symbol}.csv"}
        if set(record["files"]) != expected_files:
            raise QuantError("The actual issuer and quote inputs must all remain frozen.")
        for name, digest in record["files"].items():
            safe_file(ROOT, (SOURCE / name).relative_to(ROOT).as_posix(), digest)
        definition = issuer_definition(symbol, (SOURCE / f"{symbol}-issuer.html").read_text())
        if definition != record["issuer_definition"]:
            raise QuantError("A factor role differs from the actual issuer definition.")
        payload = read_json(SOURCE / f"{symbol}-raw.json")
        parsed = parse_chart(payload, symbol, policy["data_start"], policy["as_of"])
        retained = pd.read_csv(SOURCE / f"{symbol}.csv", index_col="date", parse_dates=True)
        if (
            not retained.index.equals(parsed.index)
            or not retained.columns.equals(parsed.columns)
            or not np.allclose(retained, parsed, rtol=1e-12, atol=1e-9)
            or not parsed.index.equals(original.close.index)
            or corporate_actions(payload, parsed, symbol) != record["actions"]
            or record["sessions"] != len(parsed)
            or record["first_session"] != str(parsed.index[0].date())
            or record["last_session"] != str(parsed.index[-1].date())
            or record["minimum_volume"] != float(parsed["volume"].min())
        ):
            raise QuantError("A factor fund's history differs from its actual raw observations.")
        funds[symbol] = parsed

    def panel(name, field):
        frame = getattr(original, name).copy()
        for symbol in ("IJR", "PKW"):
            frame[symbol] = funds[symbol][field]
        return frame

    result = MarketData(
        panel("open", "adj_open"),
        panel("close", "adj_close"),
        panel("raw_close", "close"),
        panel("volume", "volume"),
        original.risk_free,
    )
    result.validate()
    return result


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    data = verified_market(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("New-family preparation must remain inside the isolated workbench.")
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
    source = {
        "adapter": "additional_factor_families_20261011",
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
        "new_definitions_queued_not_registered": True,
        "historical_methodology_continuity_unverified": True,
        "independent_factor_returns_unverified": True,
        "direct_stock_data_still_blocked": True,
        "existing_forward_archive_includes_new_funds": False,
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "market-reference.json", market)
    return {
        "readiness": readiness,
        "market": market,
        "eligible_not_before": policy["eligible_not_before"],
        "queued_factor_ids": list(definitions(policy)),
        "candidate_specifications_created": False,
        "strategy_outcomes_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare audited new-family data; no strategy returns."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/factor-family-prepared-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key not in ("market", "readiness")}
            )
        )
    except QuantError as exc:
        parser.exit(2, f"New-family data preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
