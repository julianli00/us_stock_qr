from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import load_prices, load_protocol
from us_quant.multifactor_stability import FACTORS, load_market
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/defensive-factor-rotation.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("defensive_assets") != ["GLD", "TLT", "IEF"]
        or policy.get("cash_asset") != "BIL"
        or policy.get("momentum_sessions") != 126
        or policy.get("volatility_sessions") != 63
        or policy.get("equity_share_min") != 0.30
        or policy.get("equity_share_max") != 0.70
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_defense_momentum", "defense_method": "highest_positive_excess"},
            {
                "id": "four_factor_defense_diversified",
                "defense_method": "all_positive_inverse_volatility",
            },
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
    ):
        raise QuantError("The fixed defensive-rotation study has changed.")


def verified_market(policy: dict) -> MarketData:
    validate_policy(policy)
    base = ROOT / "data/factor-round-20261007/base"
    factors = load_market(
        read_json(ROOT / "config/multifactor-stability.json"),
        read_json(ROOT / "evidence/multifactor_stability_20261010_registration_v2.json"),
        base,
        ROOT / "data/multifactor-stability-20261010",
    )
    original = load_prices(load_protocol(ROOT / "config/dual-horizon.json"), base)
    index = factors.close.index

    def combine(name):
        frame = getattr(factors, name).copy()
        frame["TLT"] = getattr(original, name).loc[index, "TLT"]
        return frame

    data = MarketData(
        combine("open"),
        combine("close"),
        combine("raw_close"),
        combine("volume"),
        factors.risk_free,
    )
    data.validate()
    return data


def target(history: pd.DataFrame, candidate: dict, policy: dict) -> pd.Series:
    validate_policy(policy)
    required = {"SPY", "IEF", "TLT", "GLD", "BIL", *FACTORS}
    if (
        len(history) < 127
        or set(history.columns) != required
        or not history.index.is_unique
        or not history.index.is_monotonic_increasing
        or not np.isfinite(history.to_numpy()).all()
        or (history <= 0).any().any()
        or candidate not in policy["candidates"]
    ):
        raise QuantError("Defensive rotation requires complete prior prices and registered rules.")
    move = history.iloc[-1] / history.iloc[-127] - 1
    recent = history.pct_change(fill_method=None).iloc[1:].tail(63)
    eligible = [symbol for symbol in policy["defensive_assets"] if move[symbol] > move["BIL"]]
    weights = pd.Series(0.0, index=history.columns)
    if not eligible:
        equity_share = 0.30
        defense = pd.Series({"BIL": 1.0})
    else:
        if candidate["defense_method"] == "highest_positive_excess":
            selected = sorted(eligible, key=lambda symbol: (-float(move[symbol]), symbol))[0]
            defense = pd.Series({selected: 1.0})
        else:
            volatility = recent.loc[:, eligible].std(ddof=1)
            if (volatility <= 1e-10).any() or not np.isfinite(volatility).all():
                raise QuantError("Eligible defense volatility is missing or degenerate.")
            defense = (1 / volatility) / (1 / volatility).sum()
        equity_vol = recent.loc[:, list(FACTORS)].mean(axis=1).std(ddof=1)
        defensive_returns = recent.loc[:, list(defense.index)] @ defense
        defense_vol = defensive_returns.std(ddof=1)
        if (
            not np.isfinite([equity_vol, defense_vol]).all()
            or min(equity_vol, defense_vol) <= 1e-10
        ):
            raise QuantError("Observed sleeve volatility cannot support a risk allocation.")
        equity_share = float(np.clip(defense_vol / (equity_vol + defense_vol), 0.30, 0.70))
    weights.loc[list(FACTORS)] = 0.98 * equity_share / 4
    weights.loc[defense.index] = 0.98 * (1 - equity_share) * defense
    if (weights < 0).any() or not np.isclose(weights.sum(), 0.98, atol=1e-12):
        raise QuantError("Defensive rotation violated its cash-funded budget.")
    return weights


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    result = {}
    for candidate in policy["candidates"]:
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for index, day in enumerate(data.close.index):
            if index >= 126 and is_month_end(day):
                frame.loc[day] = target(data.close.iloc[: index + 1], candidate, policy)
        if frame.dropna(how="all").empty:
            raise QuantError("No complete defensive-rotation targets exist.")
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
            "adapter": "defensive_factor_rotation_20261011",
            "policy": POLICY.relative_to(ROOT).as_posix(),
            "policy_sha256": file_digest(POLICY),
        },
        "capabilities": {
            name: {
                "verified": True,
                "evidence_path": POLICY.relative_to(ROOT).as_posix(),
                "evidence_sha256": file_digest(POLICY),
            }
            for name in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
        "stock_data_still_blocked": True,
        "new_market_observations": False,
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
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
            "frozen_files": {
                POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
                Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
            },
            "leveraged_products_allowed": False,
            "order_authority": False,
            "history_status": "exposed_history_not_independent_holdout",
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "strategy_outcomes_computed": False}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fixed factor-core/defensive-rotation research."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/defensive-factor-rotation-20261011"
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
        parser.exit(2, f"Defensive-factor preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
