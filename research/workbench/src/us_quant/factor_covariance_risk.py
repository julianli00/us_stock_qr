from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import (
    file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/factor-covariance-risk.json"
FACTORS = ("MTUM", "VLUE", "QUAL", "USMV")
ASSETS = (*FACTORS, "GLD", "IEF")
CANDIDATES = (
    {"id": "four_factor_covariance_risk_budget", "method": "risk_budget"},
    {"id": "four_factor_covariance_max_diversification", "method": "maximum_diversification"},
)


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "study_id": "factor_covariance_risk_20261011",
        "data_scope": "factor_etf_portfolio",
        "as_of": "2026-10-05",
        "prior_evaluated_configurations": 120,
        "new_configurations": 2,
        "factor_ids": [
            "price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure",
        ],
        "factor_symbols": list(FACTORS),
        "risk_assets": list(ASSETS),
        "covariance_sessions": 252,
        "covariance_diagonal_shrinkage": 0.10,
        "equity_share_min": 0.30,
        "equity_share_max": 0.70,
        "factor_share_of_equity_min": 0.15,
        "factor_share_of_equity_max": 0.35,
        "each_defensive_share_min": 0.25,
        "each_defensive_share_max": 0.75,
        "cash_reserve": 0.02,
        "capital_usd": 10000.0,
        "commission_per_order": 1.0,
        "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict)
        or any(policy.get(key) != value for key, value in constants.items())
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in ("source_registration", "source_context")
        )
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Keep the two fixed covariance-only allocations and original gates.")


def constraint_matrix() -> np.ndarray:
    total = np.ones(6)
    equity = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    defense = total - equity
    rows = [equity - 0.30 * total, 0.70 * total - equity]
    for index in range(6):
        individual = np.eye(6)[index]
        group, low, high = (equity, 0.15, 0.35) if index < 4 else (defense, 0.25, 0.75)
        rows.extend((individual - low * group, high * group - individual))
    return np.array(rows)


def allocation(covariance: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    if (
        not isinstance(covariance, np.ndarray) or covariance.shape != (6, 6)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        or (np.diag(covariance) <= 0).any()
        or method not in {candidate["method"] for candidate in CANDIDATES}
    ):
        raise QuantError("Risk allocation requires a finite symmetric six-asset covariance.")
    matrix = covariance / np.trace(covariance)
    if np.linalg.eigvalsh(matrix).min() <= 1e-12:
        raise QuantError("Risk covariance must be positive definite; no hidden regularization.")
    limits = constraint_matrix()
    constraints = [LinearConstraint(limits, 0.0, np.inf)]
    standalone = np.sqrt(np.diag(matrix))
    initial = np.ones(6)
    if method == "maximum_diversification":
        initial /= standalone.sum()
        constraints.append(LinearConstraint(standalone, 1.0, 1.0))

    def objective(values: np.ndarray) -> float:
        variance = 0.5 * float(values @ matrix @ values)
        return variance - float(np.log(values).mean()) if method == "risk_budget" else variance

    def gradient(values: np.ndarray) -> np.ndarray:
        derivative = matrix @ values
        return derivative - 1.0 / (6 * values) if method == "risk_budget" else derivative

    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(1e-12, None)] * 6, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all() or (result.x <= 0).any():
        raise QuantError(f"The fixed risk optimizer failed: {result.message}")
    weights = result.x / result.x.sum()
    if (
        (limits @ weights < -1e-9).any()
        or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
        or (
            method == "maximum_diversification"
            and not np.isclose(standalone @ result.x, 1.0, rtol=0, atol=1e-9)
        )
    ):
        raise QuantError("The solved allocation violates its fixed group or normalization constraints.")
    variance = float(weights @ covariance @ weights)
    risk_shares = weights * (covariance @ weights) / variance
    return weights, {
        "method": method,
        "unnormalized_solution": result.x.tolist(),
        "scaled_objective": objective(result.x),
        "forecast_annualized_volatility": float(np.sqrt(variance)),
        "diversification_ratio": float(weights @ np.sqrt(np.diag(covariance)) / np.sqrt(variance)),
        "signed_variance_contribution_shares": risk_shares.tolist(),
        "maximum_deviation_from_one_sixth_risk_share": float(abs(risk_shares - 1 / 6).max()),
        "group_constraints_binding_count": int((abs(limits @ weights) <= 1e-8).sum()),
        "equal_risk_contributions_guaranteed_by_constraints": False,
    }


def build_targets(data: MarketData, policy: dict) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if set(data.close.columns) != {*ASSETS, "SPY", "BIL"}:
        raise QuantError("Use the original verified eight assets; no industry or synthetic substitute.")
    daily = data.close.loc[:, list(ASSETS)].pct_change(fill_method=None)
    targets = {
        candidate["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for candidate in CANDIDATES
    }
    diagnostics = []
    for index, day in enumerate(data.close.index):
        if index < 252 or not is_month_end(day):
            continue
        prior = daily.iloc[index - 251:index + 1]
        if len(prior) != 252 or not np.isfinite(prior.to_numpy()).all():
            raise QuantError("Each covariance target requires 252 actual complete prior returns.")
        sample = prior.cov().to_numpy() * 252
        covariance = 0.9 * sample + 0.1 * np.diag(np.diag(sample))
        for candidate in CANDIDATES:
            weights, audit = allocation(covariance, candidate["method"])
            frame = targets[candidate["id"]]
            frame.loc[day] = 0.0
            frame.loc[day, list(ASSETS)] = 0.98 * weights
            diagnostics.append({
                "candidate_id": candidate["id"], "decision_session": day.strftime("%Y-%m-%d"),
                "observed_returns_first_session": prior.index[0].strftime("%Y-%m-%d"),
                "observed_returns_last_session": prior.index[-1].strftime("%Y-%m-%d"),
                "observed_returns_count": len(prior),
                "covariance": covariance.tolist(), **audit,
            })
    if any(frame.dropna(how="all").empty for frame in targets.values()):
        raise QuantError("No complete month-end risk target is available in the verified data.")
    return targets, diagnostics


def prepare(output: Path) -> dict:
    policy = read_json(CONFIG)
    validate(policy)
    source, context = (
        read_json(safe_file(ROOT, policy[key]["path"], policy[key]["sha256"]))
        for key in ("source_registration", "source_context")
    )
    if (
        context["scope"] != "primary_paper_metadata_and_risk_only_hypothesis_not_performance_replication"
        or len(context["papers"]) != 2
        or context["candidate_targets_returns_or_metrics_computed"] is not False
    ):
        raise QuantError("Preserve the independently checked source context before target preparation.")
    for paper in context["papers"]:
        safe_file(ROOT, paper["private_raw_response"], paper["private_raw_response_sha256"])
    old = next(
        (row["spec"] for row in source["candidates"] if row["spec"]["id"] == "four_factor_gold_risk_balance"),
        None,
    )
    if old is None:
        raise QuantError("The exact original factor/gold source registration is unavailable.")
    data = review_market({"market": old["market"]}, ROOT)
    verified = verified_etf_market(source["readiness"], ROOT)
    for key in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, key), getattr(verified, key)
        if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-10, atol=1e-9):
            raise QuantError("The reused market differs from its original actual-source adapter.")
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare in a new isolated private data directory.")
    new_output_directory(output)
    frames, diagnostics = build_targets(data, policy)
    write_json(output / "risk-diagnostics.json", {"diagnostics": diagnostics})
    readiness = deepcopy(source["readiness"])
    readiness.update({
        "checked_at": utc_now(),
        "source_check": policy["source_context"],
        "revalidation_not_new_market_observation": True,
    })
    write_json(output / "readiness.json", readiness)
    frozen = {
        **old["frozen_files"],
        CONFIG.relative_to(ROOT).as_posix(): file_digest(CONFIG),
        "src/us_quant/factor_covariance_risk.py": file_digest(Path(__file__)),
        **{policy[key]["path"]: policy[key]["sha256"] for key in ("source_registration", "source_context")},
        **{paper["private_raw_response"]: paper["private_raw_response_sha256"]
           for paper in context["papers"]},
    }
    specs = []
    for candidate in CANDIDATES:
        spec = {
            "id": candidate["id"], "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio", "configuration": candidate,
            "evaluation_as_of": policy["as_of"], "market": old["market"],
            "frozen_files": frozen, "asset_leverage": old["asset_leverage"],
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False, "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(
            output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"),
        )
        specs.append(spec)
    return {
        "specs": specs, "readiness": readiness,
        "defined_targets_per_candidate": len(next(iter(frames.values())).dropna(how="all")),
        "candidate_returns_or_metrics_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two covariance-only factor portfolios; no orders.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument(
        "--prepared", type=Path, default=Path("data/factor-covariance-risk-20261011"),
    )
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({
            "prepared_configurations": len(result["specs"]),
            "defined_targets_per_candidate": result["defined_targets_per_candidate"],
            "candidate_returns_or_metrics_computed": False,
            "investment_objective_verified": False,
        }, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Covariance risk preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
