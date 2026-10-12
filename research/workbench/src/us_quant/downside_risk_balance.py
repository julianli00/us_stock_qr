from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/downside-risk-balance.json"
PRIOR = ROOT / "evidence/growth_portfolio_protection_20261011_registration.json"


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
        or policy.get("growth_share_of_equity") != 0.5
        or policy.get("risk_sessions") != 63
        or policy.get("tail_probability") != 0.05
        or policy.get("equity_share_min") != 0.30
        or policy.get("equity_share_max") != 0.70
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "four_factor_growth_gold_semideviation", "risk_measure": "semideviation"},
            {"id": "four_factor_growth_gold_expected_loss", "risk_measure": "expected_loss"},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("The fixed downside estimators, core composition or funding protocol changed.")


def risk_estimate(excess: np.ndarray, method: str, probability: float = 0.05) -> float:
    values = np.asarray(excess, dtype=float)
    if (
        values.shape != (63,)
        or not np.isfinite(values).all()
        or probability != 0.05
        or method not in ("semideviation", "expected_loss")
    ):
        raise QuantError("Downside risk needs exactly63finite observations and a frozen estimator.")
    if method == "semideviation":
        return float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    ordered = np.sort(values)
    mass = len(ordered) * probability
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail_mean = (ordered[:whole].sum() + fraction * ordered[whole]) / mass
    return max(0.0, float(-tail_mean))


def risk_budget(equity_risk: float, gold_risk: float) -> float:
    if (
        not np.isfinite([equity_risk, gold_risk]).all()
        or min(equity_risk, gold_risk) < 0
        or equity_risk + gold_risk <= 1e-10
    ):
        raise QuantError("Two zero or invalid risk estimates cannot identify a portfolio budget.")
    return float(np.clip(gold_risk / (equity_risk + gold_risk), 0.30, 0.70))


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS}:
        raise QuantError("Downside estimates require the original audited actual growth/factor panel.")
    composition = pd.Series(0.0, index=data.close.columns)
    composition.loc[list(FACTORS)] = 0.125
    composition["QQQ"] = 0.50
    daily = data.close.pct_change(fill_method=None)
    outputs = {
        row["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for row in policy["candidates"]
    }
    for i, day in enumerate(data.close.index):
        if i < 63 or not is_month_end(day):
            continue
        returns = daily.iloc[i - 62 : i + 1]
        rates = data.risk_free.loc[returns.index]
        equity_excess = (returns @ composition - rates).to_numpy()
        gold_excess = (returns["GLD"] - rates).to_numpy()
        for candidate in policy["candidates"]:
            equity_risk = risk_estimate(equity_excess, candidate["risk_measure"])
            gold_risk = risk_estimate(gold_excess, candidate["risk_measure"])
            equity = risk_budget(equity_risk, gold_risk)
            target = composition * equity * 0.98
            target["GLD"] = (1 - equity) * 0.98
            if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
                raise QuantError("Downside balancing cannot short assets or borrow cash.")
            outputs[candidate["id"]].loc[day] = target
    return outputs


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR)
    actual = verified_etf_market(prior["readiness"], ROOT)
    market = prior["candidates"][0]["spec"]["market"]
    retained = review_market({"market": market}, ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        old, verified = getattr(retained, name), getattr(actual, name)
        if (
            not old.index.equals(verified.index)
            or not old.columns.equals(verified.columns)
            or not np.allclose(old, verified, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("The downside study cannot substitute its actual fund-price sources.")
    if not np.allclose(retained.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("Downside thresholds must use the original frozen risk-free proxy.")
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Downside preparation must stay inside the isolated workbench.")
    new_output_directory(output)
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    frozen = {
        path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in (POLICY, Path(__file__))
    }
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"], "configuration": candidate,
            "factor_ids": policy["factor_ids"],
            "data_scope": policy["data_scope"],
            "evaluation_as_of": policy["as_of"],
            "market": market,
            "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in actual.close.columns},
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "strategy_outcomes_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed downside-risk budgets.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/downside-risk-balance-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(json.dumps({
            "prepared": [spec["id"] for spec in result["specs"]],
            "strategy_outcomes_computed": False,
        }))
    except QuantError as exc:
        parser.exit(2, f"Downside-risk study blocked: {exc}\n")


if __name__ == "__main__":
    main()
