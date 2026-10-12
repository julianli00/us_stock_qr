from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import portfolio_refinement as prior
from us_quant.backtest import BacktestResult
from us_quant.breadth_gold_relative import build_targets as source_targets
from us_quant.calendar import is_month_end, next_session, previous_session
from us_quant.candidate_receipts import local_path
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import seed_window
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/observed-weight-execution.json"
CANDIDATES = (
    {"id": "semiconductor_graded_gold_half_observed_band025", "source_id": "semiconductor_breadth_half_gold_relative"},
    {"id": "semiconductor_graded_gold_off_observed_band025", "source_id": "semiconductor_breadth_off_gold_relative"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation",
              "prior_band_policy", "prior_band_implementation")


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "graded_gold_prior_observed_weight_execution_20261011",
        "data_scope": "factor_etf_portfolio", "data_start": "2015-08-10", "as_of": "2026-10-05",
        "prior_evaluated_configurations": 140, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "actual_weight_trade_band": 0.025, "capital_usd": 10000,
        "commission_per_order": 1.0, "cash_reserve": 0.02, "candidates": list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep both original sources,actual2.5ppband and unchanged costs/capital.")


def pure_band(policy: dict) -> tuple[dict, dict]:
    original = read_json(safe_file(ROOT, policy["prior_band_policy"]["path"], policy["prior_band_policy"]["sha256"]))
    prior.validate_policy(original)
    selected = [row for row in original["candidates"] if row["id"] == "portfolio_observed_weight_band025"]
    if (len(selected) != 1 or selected[0]["qld_cap"] is not None
        or selected[0]["volatility_target"] is not None or selected[0]["weight_band"] != 0.025):
        raise QuantError("Use only the unchanged original pure observed-weight band without risk overrides.")
    return selected[0], original


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    pure_band(policy)
    original = source_targets(data, read_json(ROOT / policy["source_policy"]["path"]))
    return {candidate["id"]: original[candidate["source_id"]] for candidate in CANDIDATES}


def run(
    data: MarketData, baseline: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    data.validate()
    if (candidate not in CANDIDATES or type(delay) is not int or delay < 1
        or not np.isfinite(cost) or not 0 <= cost < 100
        or not isinstance(baseline, pd.DataFrame) or not baseline.index.equals(data.close.index)
        or not baseline.columns.equals(data.close.columns)
        or (baseline.isna().any(axis=1) & ~baseline.isna().all(axis=1)).any()):
        raise QuantError("Observed execution requires the complete original source and causal delay/cost inputs.")
    active = baseline.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)):
        raise QuantError("Original desired targets must stay fully specified,monthly,long-only and funded.")
    band, original_policy = pure_band(policy)
    dates = data.close.loc[start:end].index
    if len(dates) < 2:
        raise QuantError("The execution adapter needs complete original return observations.")
    known = seed_window(baseline, str(dates[0].date()))
    anchor = previous_session(dates[0])
    issued = pd.DataFrame(np.nan, index=baseline.index, columns=baseline.columns)
    pending, decisions = {}, []
    rows, weights = [], []
    units = np.zeros(len(data.close.columns))
    cash = previous_nav = float(policy["capital_usd"])

    def decide(day: pd.Timestamp, current: pd.Series, initial: bool = False) -> None:
        desired = known.loc[day]
        covariance = prior.observed_covariance(data, day, original_policy["covariance_sessions"])
        reason = "initial_capital" if initial else prior.rebalance_reason(current, desired, covariance, band)
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if reason != "hold":
            if execution in pending:
                raise QuantError("Do not overwrite a pending original dated trade request.")
            pending[execution] = desired.to_numpy().copy()
            issued.loc[day] = desired
        decisions.append({"signal_session": str(day.date()), "execution_session": str(execution.date()),
                          "initial_capital": initial, "reason": reason,
                          "maximum_observed_target_gap": float(abs(current - desired).max())})

    decide(anchor, pd.Series(0.0, index=baseline.columns), initial=True)
    for day in dates:
        opening, closing = data.open.loc[day].to_numpy(), data.close.loc[day].to_numpy()
        costs = turnover = 0.0
        orders = 0
        if day in pending:
            values, cash, costs, turnover, orders = rebalance(
                units * opening, cash, pending.pop(day), cost, policy["commission_per_order"],
            )
            if turnover > 0 or orders > 0:
                units = values / opening
        values = units * closing
        nav = float(values.sum() + cash)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise QuantError("Observed-weight execution became insolvent or borrowed cash.")
        current = pd.Series(values / nav, index=baseline.columns)
        rows.append((nav, nav / previous_nav - 1, cash, float(current.sum()), turnover, costs, orders))
        weights.append(current.to_numpy())
        if not known.loc[day].isna().all():
            if not is_month_end(day):
                raise QuantError("Only original monthly source updates may inspect the no-trade band.")
            decide(day, current)
        previous_nav = nav
    frame = pd.DataFrame(rows, index=dates,
                         columns=["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"])
    frame["risk_free"] = data.risk_free.loc[dates]
    return BacktestResult(frame, pd.DataFrame(weights, index=dates, columns=baseline.columns)), issued, decisions


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    registration = read_json(ROOT / policy["source_registration"]["path"])
    records = {row["spec"]["id"]: row for row in registration["candidates"]}
    if set(records) != {row["source_id"] for row in CANDIDATES}:
        raise QuantError("Use both exact original graded-gold source registrations.")
    parent = records[CANDIDATES[0]["source_id"]]
    spec = parent["spec"]
    if any(row["spec"]["market"] != spec["market"] or row["spec"]["frozen_files"] != spec["frozen_files"]
           for row in records.values()):
        raise QuantError("Both source contexts must keep their identical frozen market and complete dependencies.")
    data = review_market({"market": spec["market"]}, ROOT)
    ref = parent["readiness"]
    readiness = read_json(safe_file(ROOT, ref["path"], ref["sha256"]))
    verified = verified_etf_market(readiness, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(verified, name)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)):
            raise QuantError("Preserve every original actual price and frozen risk-free observation.")
    baselines = source_baselines(data, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare one new private source-only observed-execution directory.")
    new_output_directory(output)
    readiness = deepcopy(readiness)
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    frozen = dict(spec["frozen_files"])
    for key in REFERENCES:
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    for name in ("src/us_quant/cash_funded_accounting_v2.py", "src/us_quant/dual_horizon.py",
                 "src/us_quant/factor_research.py"):
        frozen[name] = file_digest(ROOT / name)
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    frozen["src/us_quant/observed_weight_execution.py"] = file_digest(Path(__file__))
    for candidate in CANDIDATES:
        actual = {"id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
                  "data_scope": "factor_etf_portfolio", "evaluation_as_of": policy["as_of"],
                  "market": spec["market"], "asset_leverage": spec["asset_leverage"], "frozen_files": frozen,
                  "history_status": "exposed_history_not_independent_holdout",
                  "leveraged_products_allowed": False, "order_authority": False}
        write_json(output / f"{candidate['id']}-spec.json", actual)
        write_text_atomic(output / f"{candidate['id']}-source-targets.csv",
                          baselines[candidate["id"]].to_csv(float_format="%.17g"))
    return {"prepared_source_targets_per_candidate": len(next(iter(baselines.values())).dropna(how="all")),
            "actual_path_dependent_trade_targets_computed": False,
            "candidate_returns_or_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare original observed-weight trade-band transfer, without accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/observed-weight-execution-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Observed-weight preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
