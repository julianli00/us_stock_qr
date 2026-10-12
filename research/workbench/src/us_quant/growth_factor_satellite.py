from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.defensive_factor_rotation import verified_market as factor_market
from us_quant.dual_horizon import load_prices, load_protocol
from us_quant.multifactor_stability import FACTORS, cap_volatility
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/growth-factor-satellite.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("growth_asset") != "QQQ"
        or policy.get("growth_asset_is_new_factor") is not False
        or policy.get("volatility_sessions") != 63
        or policy.get("equity_share_min") != 0.30
        or policy.get("equity_share_max") != 0.70
        or policy.get("target_volatility") != 0.12
        or policy.get("target_change_band") != 0.05
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_daily_target_control", "growth_share_of_equity": 0.0},
            {"id": "four_factor_growth_satellite50", "growth_share_of_equity": 0.5},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
    ):
        raise QuantError("The growth-satellite study or matched risk constraints changed.")


def verified_market(policy: dict) -> MarketData:
    validate_policy(policy)
    prior = factor_market(read_json(ROOT / "config/defensive-factor-rotation.json"))
    base = load_prices(
        load_protocol(ROOT / "config/dual-horizon.json"), ROOT / "data/factor-round-20261007/base"
    )

    def panel(name):
        values = getattr(prior, name).copy()
        values["QQQ"] = getattr(base, name).loc[prior.close.index, "QQQ"]
        return values

    data = MarketData(
        panel("open"), panel("close"), panel("raw_close"), panel("volume"), prior.risk_free
    )
    data.validate()
    return data


def composed_monthly_target(history: pd.DataFrame, candidate: dict) -> pd.Series:
    prior = history.pct_change(fill_method=None).iloc[1:].tail(63)
    if len(prior) != 63 or not np.isfinite(prior.to_numpy()).all():
        raise QuantError("A full trailing risk window is required.")
    composition = pd.Series(0.0, index=history.columns)
    growth = candidate["growth_share_of_equity"]
    composition.loc[list(FACTORS)] = (1 - growth) / 4
    composition["QQQ"] = growth
    equity_vol = (prior @ composition).std(ddof=1)
    gold_vol = prior["GLD"].std(ddof=1)
    if not np.isfinite([equity_vol, gold_vol]).all() or min(equity_vol, gold_vol) <= 1e-10:
        raise QuantError("Risk balance requires positive observed sleeve volatility.")
    equity = float(np.clip(gold_vol / (equity_vol + gold_vol), 0.30, 0.70))
    result = composition * equity * 0.98
    result["GLD"] = (1 - equity) * 0.98
    return result


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS}:
        raise QuantError("The verified growth/factor universe is incomplete or substituted.")
    daily = data.close.pct_change(fill_method=None)
    result = {}
    for candidate in policy["candidates"]:
        monthly, previous = None, None
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for i, day in enumerate(data.close.index):
            if i < 63:
                continue
            update = is_month_end(day)
            if update:
                monthly = composed_monthly_target(data.close.iloc[: i + 1], candidate)
            if monthly is None:
                continue
            covariance = daily.iloc[i - 62 : i + 1].cov() * 252
            target = cap_volatility(monthly, covariance, policy["target_volatility"], 0.98)
            if previous is None or update or float(abs(target - previous).max()) + 1e-12 >= 0.05:
                if (target < -1e-12).any() or not np.isclose(target.sum(), 0.98, atol=1e-10):
                    raise QuantError("Growth/factor targets violate cash funding.")
                frame.loc[day] = target
                previous = target.copy()
        result[candidate["id"]] = frame
    return result


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
        path = output / f"{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        panels[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    readiness = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "data_scope": "factor_etf_portfolio",
        "verified_etf_source": {
            "adapter": "growth_factor_satellite_20261011",
            "policy": POLICY.relative_to(ROOT).as_posix(),
            "policy_sha256": file_digest(POLICY),
        },
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": POLICY.relative_to(ROOT).as_posix(),
                "evidence_sha256": file_digest(POLICY),
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "stock_fundamental_data_still_blocked": True,
        "QQQ_is_growth_beta_not_a_new_factor": True,
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "data_scope": "factor_etf_portfolio",
            "factor_ids": policy["factor_ids"],
            "evaluation_as_of": policy["as_of"],
            "market": panels,
            "frozen_files": {
                POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
                Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
            },
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "returns_computed": False}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fixed growth-beta/factor-core comparison."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/growth-factor-satellite-20261011"
    )
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {"prepared": [item["id"] for item in result["specs"]], "returns_computed": False}
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Growth/factor preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
