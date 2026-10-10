from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from us_quant.config import QuantError
from us_quant.defensive_factor_rotation import verified_market
from us_quant.dual_horizon import load_protocol
from us_quant.storage import file_digest, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ("SPY", "MTUM", "VLUE", "QUAL", "USMV", "GLD", "IEF", "TLT", "BIL")


def static_sharpe_envelope(excess: pd.DataFrame) -> dict:
    if (
        len(excess) < 3
        or excess.shape[1] < 2
        or not np.isfinite(excess.to_numpy()).all()
        or not excess.index.is_unique
        or not excess.index.is_monotonic_increasing
    ):
        raise QuantError("The diagnostic needs complete, ordered excess returns.")
    mean = excess.mean().to_numpy() * 252
    covariance = excess.cov().to_numpy() * 252
    if not np.allclose(covariance, covariance.T):
        raise QuantError("Invalid diagnostic covariance.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= max(eigenvalues[-1], 1e-12) * 1e-12:
        raise QuantError("The diagnostic covariance is not reliably positive definite.")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise QuantError(
            "The diagnostic covariance must be positive definite; no hidden regularization."
        ) from exc
    if mean.max() <= 0:
        return {
            "status": "no_positive_sample_excess_mean",
            "numerical_upper_bound": 0.0,
            "strategy_qualified": False,
            "scope": "fixed_nonnegative_weights_only",
        }
    volatility = np.sqrt(np.diag(covariance))
    scaled_mean = mean / volatility
    correlation = covariance / np.outer(volatility, volatility)
    start = np.zeros(len(mean))
    winner = int(scaled_mean.argmax())
    start[winner] = 1 / scaled_mean[winner]
    solution = minimize(
        lambda x: 0.5 * float(x @ correlation @ x),
        start,
        jac=lambda x: correlation @ x,
        method="SLSQP",
        bounds=[(0, None)] * len(mean),
        constraints={
            "type": "eq",
            "fun": lambda x: float(scaled_mean @ x) - 1,
            "jac": lambda x: scaled_mean,
        },
        options={"ftol": 1e-13, "maxiter": 3000},
    )
    x = solution.x
    if (
        not solution.success
        or not np.isfinite(x).all()
        or (x < -1e-10).any()
        or abs(float(scaled_mean @ x) - 1) > 1e-8
    ):
        raise QuantError("The optimistic fixed-weight problem did not solve within tolerance.")
    variance = float(x @ correlation @ x)
    residual = correlation @ x - variance * scaled_mean
    multipliers = np.maximum(residual, 0)
    dual_vector = variance * scaled_mean + multipliers
    dual = variance - 0.5 * float(dual_vector @ np.linalg.solve(correlation, dual_vector))
    margin = 1e-9 * max(abs(dual), 1.0)
    lower_variance = 2 * (dual - margin)
    gap = 0.5 * variance - dual
    if lower_variance <= 0 or gap < -1e-8 or gap > 1e-7:
        raise QuantError("A sufficiently tight numerical dual certificate was not established.")
    weights = np.maximum(x, 0) / volatility
    weights /= weights.sum()
    portfolio_excess = excess.to_numpy() @ weights
    attained = float(portfolio_excess.mean() / portfolio_excess.std(ddof=1) * np.sqrt(252))
    upper = float(1 / np.sqrt(lower_variance))
    if attained > upper + 1e-8 or abs(attained - 1 / np.sqrt(variance)) > 1e-8:
        raise QuantError("Independent Sharpe and the optimization certificate disagree.")
    return {
        "status": "numerically_certified_static_diagnostic",
        "attained_in_sample_sharpe": attained,
        "numerical_upper_bound": upper,
        "primal_dual_gap": gap,
        "floating_point_safety_margin": margin,
        "coordinate_system": "equivalent_unit_volatility_coordinates_without_regularization",
        "annualized_sample_excess_means": dict(zip(excess.columns, mean.tolist(), strict=True)),
        "annualized_sample_covariance": covariance.tolist(),
        "asset_order": list(excess.columns),
        "strategy_qualified": False,
        "optimal_weights_not_published_as_trade_recommendations": True,
        "scope": "fixed_nonnegative_weights_daily_rebalanced_close_to_close_zero_cost",
    }


def run_diagnostic() -> dict:
    data = verified_market(read_json(ROOT / "config/defensive-factor-rotation.json"))
    windows = load_protocol(ROOT / "config/dual-horizon.json").windows()
    results = {}
    for window in windows:
        returns = (
            data.close.loc[:, list(ASSETS)]
            .pct_change(fill_method=None)
            .loc[window["first_return_session"] : window["last_session"]]
        )
        excess = (returns * 0.98).sub(data.risk_free.loc[returns.index], axis=0)
        results[f"{window['years']}y"] = {
            "window": window,
            **static_sharpe_envelope(excess),
        }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "implementation_sha256": file_digest(Path(__file__)),
        "mode": "hindsight_feasibility_diagnostic_not_strategy",
        "assets": list(ASSETS),
        "idle_cash_fraction": 0.02,
        "modeled_transaction_cost": 0.0,
        "allocation_knowledge": "Uses the entire window deliberately; not available at its start.",
        "windows": results,
        "same_weights_across_windows": False,
        "independent_forward_validation": False,
        "new_strategy_evaluations": 0,
        "investment_objective_verified": False,
        "order_authority": False,
        "limitations": [
            "The bound applies only to the specified fixed-weight daily-close return class.",
            "It does not bound dynamic strategies, other assets or different entry assumptions.",
            "Costs are ignored and hindsight is used; this cannot qualify a strategy.",
            "It does not bound every cost-bearing execution or adaptive rule.",
            "Each horizon is optimized separately, unlike the required shared fixed rule.",
            "Floating-point dual tolerances are disclosed; no portfolio weights are recommended.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimistic static-allocation diagnostic; not a strategy."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evidence/static_allocation_feasibility_v2.json"
    )
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise QuantError("Refusing to replace a previously recorded feasibility diagnostic.")
        result = run_diagnostic()
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "mode": result["mode"],
                    "windows": {
                        key: {"numerical_upper_bound": value["numerical_upper_bound"]}
                        for key, value in result["windows"].items()
                    },
                    "new_strategy_evaluations": 0,
                    "investment_objective_verified": False,
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Allocation feasibility diagnostic blocked: {exc}\n")


if __name__ == "__main__":
    main()
