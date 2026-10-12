from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_gold_risk import monthly_targets
from us_quant.factor_research import factor_scores
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, verified_etf_market
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/factor-momentum-comparison.json"
BASELINE = ROOT / "config/factor-gold-risk.json"
PRIOR_REGISTRATION = ROOT / "evidence/factor_gold_risk_20261010_registration.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("score_parameters")
        != {
            "sector_universe": list(FACTORS),
            "parameters": {
                "momentum_sessions": 126,
                "skip_sessions": 21,
                "residual_fit_sessions": 252,
                "volatility_sessions": 63,
                "trend_sessions": 200,
            },
        }
        or policy.get("ranked_factor_shares") != [0.35, 0.30, 0.20, 0.15]
        or policy.get("baseline_policy") != BASELINE.relative_to(ROOT).as_posix()
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_total_momentum_tilt", "score": "momentum"},
            {"id": "four_factor_residual_momentum_tilt", "score": "residual"},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed factor-momentum comparison or matched budgets changed.")


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    baseline = monthly_targets(data, read_json(BASELINE))
    result = {
        candidate["id"]: pd.DataFrame(
            np.nan, index=data.close.index, columns=data.close.columns
        )
        for candidate in policy["candidates"]
    }
    warmup = policy["score_parameters"]["parameters"]["residual_fit_sessions"]
    for i, day in enumerate(data.close.index):
        if i < warmup or baseline.loc[day].isna().all():
            continue
        scores = factor_scores(
            data.close.iloc[: i + 1],
            data.risk_free.iloc[: i + 1],
            policy["score_parameters"],
        )
        if not np.isfinite(scores[["momentum", "residual"]].to_numpy()).all():
            raise QuantError("Factor momentum needs finite observed scores.")
        equity = float(baseline.loc[day, list(FACTORS)].sum())
        for candidate in policy["candidates"]:
            ordered = sorted(
                FACTORS, key=lambda symbol: (-float(scores.loc[symbol, candidate["score"]]), symbol)
            )
            target = baseline.loc[day].copy()
            target.loc[ordered] = equity * np.asarray(policy["ranked_factor_shares"])
            if (target < 0).any() or not np.isclose(
                target.sum(), 1 - policy["cash_reserve"], rtol=0, atol=1e-12
            ):
                raise QuantError("Factor momentum targets must remain long-only and cash funded.")
            result[candidate["id"]].loc[day] = target
    return result


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR_REGISTRATION)
    data = verified_etf_market(prior["readiness"], ROOT)
    panels = prior["candidates"][0]["spec"]["market"]
    frozen_market = review_market({"market": panels}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        actual, frozen = getattr(data, name), getattr(frozen_market, name)
        if (
            not actual.index.equals(frozen.index)
            or not actual.columns.equals(frozen.columns)
            or not np.allclose(actual, frozen, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("The reused market panels differ from the verified actual funds.")
    if not np.allclose(data.risk_free, frozen_market.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The reused risk-free inputs differ from the verified snapshot.")
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    frozen_files = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            BASELINE,
            Path(__file__).with_name("factor_gold_risk.py"),
            Path(__file__).with_name("factor_research.py"),
        )
    }
    new_output_directory(output)
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": panels,
            "frozen_files": frozen_files,
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
    parser = argparse.ArgumentParser(
        description="Prepare a fixed total-versus-residual factor-momentum comparison."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/factor-momentum-comparison-20261011"
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
        parser.exit(2, f"Factor-momentum preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
