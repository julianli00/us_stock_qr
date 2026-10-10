from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.factor_gold_risk import monthly_targets
from us_quant.factor_replication import original_market
from us_quant.multifactor_stability import FACTORS, corporate_actions
from us_quant.research_program import safe_file
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/momentum-implementation.json"
BASELINE = ROOT / "config/factor-gold-risk.json"
SOURCE = ROOT / "data/pdp-source-check-20261011"
SOURCE_FILES = {
    "PDP.csv",
    "PDP-yahoo.json",
    "issuer-product.html",
    "nasdaq-methodology.pdf",
    "nasdaq-index-factsheet.pdf",
}


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("data_start") != "2015-08-10"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("original_factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("replacement_symbol") != "PDP"
        or policy.get("replacement_family") != "momentum"
        or policy.get("baseline_policy") != BASELINE.relative_to(ROOT).as_posix()
        or policy.get("source_manifest") != (SOURCE / "manifest.json").relative_to(ROOT).as_posix()
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_pdp_momentum_implementation", "pdp_share_of_momentum": 1.0},
            {"id": "four_factor_dual_momentum_implementation", "pdp_share_of_momentum": 0.5},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed momentum implementation or original family budgets changed.")


def issuer_definition(document: str) -> dict:
    class ProductParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.metadata = {}
            self.descriptions = []

        def descriptions_from(self, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "description" and isinstance(item, str):
                        self.descriptions.append(item)
                    self.descriptions_from(item)
            elif isinstance(value, list):
                for item in value:
                    self.descriptions_from(item)

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "meta" and "name" in values:
                self.metadata[values["name"]] = values.get("content")
            if "data-model-json" in values:
                try:
                    self.descriptions_from(json.loads(values["data-model-json"]))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise QuantError("The issuer's product definition is not valid JSON.") from exc

    parser = ProductParser()
    parser.feed(document)
    expected = {
        "ticker": "PDP",
        "isin": "US46137V8375",
        "shareClassInceptionDate": "2007-03-01",
        "assetClass": "Equity",
        "region": "United States",
        "investmentMethod": "Passive",
        "shareClassFullName": "Invesco Dorsey Wright Momentum ETF",
    }
    phrases = (
        "index fund",
        "Technical Leaders Index",
        "100 US companies",
        "relative strength",
        "quarterly",
    )
    if any(parser.metadata.get(key) != value for key, value in expected.items()) or not any(
        all(phrase in text for phrase in phrases) for text in parser.descriptions
    ):
        raise QuantError("The issuer does not identify the expected passive US momentum fund.")
    return {
        **expected,
        "index": "Dorsey Wright Technical Leaders Index",
        "index_code": "DWTL",
        "family": "momentum",
        "daily_leveraged_target": False,
        "proprietary_stock_scores_reconstructed": False,
        "full_historical_methodology_verified": False,
    }


def verified_market(policy: dict) -> MarketData:
    validate_policy(policy)
    manifest = read_json(
        safe_file(ROOT, (SOURCE / "manifest.json").relative_to(ROOT).as_posix())
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or set(manifest.get("files", {})) != SOURCE_FILES
        or manifest.get("synthetic_history") is not False
        or manifest.get("strategy_outcomes_computed") is not False
    ):
        raise QuantError("The momentum fund source does not match its frozen data policy.")
    for name, digest in manifest["files"].items():
        safe_file(ROOT, (SOURCE / name).relative_to(ROOT).as_posix(), digest)
    definition = issuer_definition((SOURCE / "issuer-product.html").read_text())
    if definition != manifest.get("issuer_definition"):
        raise QuantError("The verified momentum mandate differs from its source manifest.")
    for name in ("nasdaq-methodology.pdf", "nasdaq-index-factsheet.pdf"):
        if not (SOURCE / name).read_bytes().startswith(b"%PDF-"):
            raise QuantError("The index-provider document is not a real PDF.")
    payload = read_json(SOURCE / "PDP-yahoo.json")
    parsed = parse_chart(payload, "PDP", policy["data_start"], policy["as_of"])
    if corporate_actions(payload, parsed, "PDP") != manifest.get("corporate_actions"):
        raise QuantError("The momentum fund corporate-action evidence changed.")
    retained = pd.read_csv(SOURCE / "PDP.csv", index_col="date", parse_dates=True)
    if (
        not retained.index.equals(parsed.index)
        or not retained.columns.equals(parsed.columns)
        or not np.allclose(retained, parsed, rtol=1e-12, atol=1e-9)
    ):
        raise QuantError("The momentum price CSV differs from the actual provider observations.")
    old = original_market()
    if not old.close.index.equals(parsed.index):
        raise QuantError("The alternative fund must retain every original historical session.")

    def panel(name, field):
        frame = getattr(old, name).copy()
        frame["PDP"] = parsed[field]
        return frame

    data = MarketData(
        panel("open", "adj_open"),
        panel("close", "adj_close"),
        panel("raw_close", "close"),
        panel("volume", "volume"),
        old.risk_free,
    )
    data.validate()
    return data


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "GLD", "BIL", "PDP", *FACTORS}:
        raise QuantError("The verified momentum implementation universe was changed.")
    prior = MarketData(
        data.open.drop(columns="PDP"),
        data.close.drop(columns="PDP"),
        data.raw_close.drop(columns="PDP"),
        data.volume.drop(columns="PDP"),
        data.risk_free,
    )
    baseline = monthly_targets(prior, read_json(BASELINE))
    outputs = {}
    for candidate in policy["candidates"]:
        target = baseline.copy()
        momentum = target["MTUM"].copy()
        target["PDP"] = momentum * candidate["pdp_share_of_momentum"]
        target["MTUM"] = momentum * (1 - candidate["pdp_share_of_momentum"])
        target = target.reindex(columns=data.close.columns)
        active = target.dropna(how="all")
        if (active < 0).any().any() or not np.allclose(
            active.sum(axis=1), 0.98, rtol=0, atol=1e-12
        ):
            raise QuantError("The implementation mix violates the original cash-funded budget.")
        outputs[candidate["id"]] = target
    return outputs


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    data = verified_market(policy)
    new_output_directory(output)
    panels = {}
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"market-{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        panels[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    readiness = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "data_scope": policy["data_scope"],
        "verified_etf_source": {
            "adapter": "momentum_implementation_20261011",
            "policy": POLICY.relative_to(ROOT).as_posix(),
            "policy_sha256": file_digest(POLICY),
            "factor_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
            "factor_manifest_sha256": file_digest(SOURCE / "manifest.json"),
        },
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
                "evidence_sha256": file_digest(SOURCE / "manifest.json"),
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "new_economic_factor_definitions": 0,
        "full_historical_methodology_verified": False,
        "stock_data_still_blocked": True,
    }
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            BASELINE,
            Path(__file__).with_name("factor_gold_risk.py"),
            Path(__file__).with_name("factor_replication.py"),
            Path(__file__).with_name("multifactor_stability.py"),
        )
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": panels,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "returns_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed momentum-fund implementations.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/momentum-implementation-20261011"
    )
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {"prepared": [spec["id"] for spec in result["specs"]], "returns_computed": False}
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Momentum implementation blocked: {exc}\n")


if __name__ == "__main__":
    main()
