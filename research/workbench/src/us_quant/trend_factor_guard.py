from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/trend-factor-guard.json"
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
        or policy.get("baseline_policy") != BASELINE.relative_to(ROOT).as_posix()
        or policy.get("trend_sessions") != 200
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_gold_joint_trend_guard", "guard": "joint"},
            {"id": "four_factor_gold_component_trend_guard", "guard": "component"},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The two fixed trend guards or original risk budgets changed.")


def indicators(data: MarketData, baseline: pd.DataFrame) -> pd.DataFrame:
    daily = data.close.pct_change(fill_method=None)
    equity = (1 + daily.loc[:, list(FACTORS)].mean(axis=1)).cumprod()
    prior_targets = baseline.ffill().shift(1)
    joint_returns = (prior_targets * daily).sum(axis=1, min_count=len(data.close.columns))
    joint = (1 + joint_returns.dropna()).cumprod().reindex(data.close.index)
    return pd.DataFrame({"joint": joint, "equity": equity, "gold": data.close["GLD"]})


def gated_target(baseline: pd.Series, state: tuple[bool, bool]) -> pd.Series:
    target = baseline.copy()
    for active, symbols in zip(state, (list(FACTORS), ["GLD"]), strict=True):
        if not active:
            removed = float(target.loc[symbols].sum())
            target.loc[symbols] = 0.0
            target["BIL"] += removed
    if (target < 0).any() or not np.isfinite(target).all() or not np.isclose(
        target.sum(), 0.98, rtol=0, atol=1e-12
    ):
        raise QuantError("Trend targets must remain finite, long-only and cash funded.")
    return target


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    baseline = monthly_targets(data, read_json(BASELINE))
    levels = indicators(data, baseline)
    means = levels.rolling(policy["trend_sessions"], min_periods=policy["trend_sessions"]).mean()
    known = baseline.ffill()
    result = {}
    for candidate in policy["candidates"]:
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        names = ["joint"] if candidate["guard"] == "joint" else ["equity", "gold"]
        previous = None
        for day in data.close.index:
            if means.loc[day, names].isna().any() or known.loc[day].isna().any():
                continue
            if candidate["guard"] == "joint":
                active = bool(levels.loc[day, "joint"] > means.loc[day, "joint"])
                state = (active, active)
            else:
                state = tuple(bool(levels.loc[day, name] > means.loc[day, name]) for name in names)
            if previous is None or state != previous or is_month_end(day):
                frame.loc[day] = gated_target(known.loc[day], state)
            previous = state
        result[candidate["id"]] = frame
    return result


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    original = read_json(PRIOR_REGISTRATION)
    data = verified_etf_market(original["readiness"], ROOT)
    panels = original["candidates"][0]["spec"]["market"]
    frozen = review_market({"market": panels}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        actual, retained = getattr(data, name), getattr(frozen, name)
        if (
            not actual.index.equals(retained.index)
            or not actual.columns.equals(retained.columns)
            or not np.allclose(actual, retained, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("The retained market differs from the audited actual-fund snapshot.")
    if not np.allclose(data.risk_free, frozen.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The retained risk-free series differs from the audited snapshot.")
    readiness = {**original["readiness"], "checked_at": utc_now()}
    frozen_files = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (
            POLICY,
            Path(__file__),
            BASELINE,
            Path(__file__).with_name("factor_gold_risk.py"),
            Path(__file__).with_name("multifactor_stability.py"),
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
    parser = argparse.ArgumentParser(description="Prepare two causal factor/gold trend guards.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/trend-factor-guard-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {"prepared": [spec["id"] for spec in result["specs"]], "returns_computed": False}
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Trend-guard preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
