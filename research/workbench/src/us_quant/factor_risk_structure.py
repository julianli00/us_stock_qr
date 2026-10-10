from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.factor_family_sources import verified_market
from us_quant.multifactor_stability import exposure_diagnostics
from us_quant.research_program import ResearchProgram, safe_file
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/factor-risk-structure.json"
COHORTS = {
    "original_four_equity": ["MTUM", "VLUE", "QUAL", "USMV"],
    "six_equity_families": ["MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW"],
    "six_equity_plus_defensive_assets": ["MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW", "GLD", "IEF"],
}


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("horizons_years") != [10, 5]
        or policy.get("cohorts") != COHORTS
        or policy.get("market_symbol") != "SPY"
        or policy.get("rolling_sessions") != 252
        or policy.get("rolling_endpoints") != "completed_NYSE_month_ends"
        or policy.get("annualization_sessions") != 252
        or policy.get("new_strategy_evaluations") != 0
        or any(
            policy.get(name) is not False
            for name in (
                "pending_candidate_performance_computed",
                "register_queued_factors_early",
                "order_authority",
            )
        )
    ):
        raise QuantError("The fixed risk-structure scope or no-performance boundary changed.")


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json({
        name: file_digest(folder / name)
        for name in (
            "factor_risk_structure.py", "factor_family_sources.py", "factor_replication.py",
            "multifactor_stability.py", "data.py", "calendar.py", "storage.py",
        )
    })


def matrix_structure(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
        or (np.diag(matrix) <= 0).any()
    ):
        raise QuantError("Risk structure requires a finite symmetric covariance with positive variances.")
    scale = float(abs(matrix).max())
    eigenvalues = np.linalg.eigvalsh(matrix / scale)
    if eigenvalues[0] < -max(1e-15, float(eigenvalues[-1]) * 1e-12):
        raise QuantError("Risk dependence cannot use a non-positive-semidefinite matrix.")
    clipped = np.maximum(eigenvalues, 0.0)
    trace = float(clipped.sum())
    return {
        "participation_ratio_dimension": float(trace**2 / (clipped @ clipped)),
        "leading_eigenvalue_fraction": float(clipped[-1] / trace),
        "smallest_numerical_eigenvalue": float(eigenvalues[0] * scale),
        "roundoff_negative_eigenvalue_clipped": bool((eigenvalues < 0).any()),
    }


def dependence(returns: pd.DataFrame) -> dict:
    if (
        len(returns) < 3
        or len(returns.columns) < 2
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or not returns.columns.is_unique
        or not all(isinstance(symbol, str) for symbol in returns.columns)
        or not np.isfinite(returns.to_numpy(dtype=float)).all()
        or (returns.std(ddof=1) <= 1e-10).any()
    ):
        raise QuantError("Dependence diagnostics need complete, ordered, nonconstant asset observations.")
    covariance = returns.cov()
    correlation = returns.corr()
    pairs = correlation.to_numpy()[np.triu_indices(len(correlation), 1)]
    return {
        "asset_count": len(returns.columns),
        "sessions": len(returns),
        "correlation_matrix": correlation.to_dict(),
        "daily_covariance_matrix": covariance.to_dict(),
        "pairwise_correlation_minimum": float(pairs.min()),
        "pairwise_correlation_mean": float(pairs.mean()),
        "pairwise_correlation_maximum": float(pairs.max()),
        "covariance_structure": matrix_structure(covariance.to_numpy()),
        "correlation_structure": matrix_structure(correlation.to_numpy()),
    }


def market_structure(
    returns: pd.DataFrame, market: pd.Series, risk_free: pd.Series
) -> dict:
    raw = dependence(returns)
    if (
        not returns.index.equals(market.index)
        or not returns.index.equals(risk_free.index)
        or not np.isfinite(market).all()
        or not np.isfinite(risk_free).all()
    ):
        raise QuantError("Market-risk inputs must be finite and exactly aligned.")
    x = np.column_stack([np.ones(len(market)), market - risk_free])
    y = returns.sub(risk_free, axis=0).to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        raise QuantError("The observed market regression is rank deficient.")
    residuals = pd.DataFrame(y - x @ coefficients, index=returns.index, columns=returns.columns)
    total = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    if (total <= 1e-15).any():
        raise QuantError("Constant excess returns cannot identify market-risk exposure.")
    estimates = {
        name: {
            "beta_vs_spy": float(coefficients[1, i]),
            "market_r_squared": float(1 - (residuals[name] ** 2).sum() / total[i]),
            "annualized_asset_volatility": float(returns[name].std(ddof=1) * np.sqrt(252)),
            "annualized_market_residual_volatility": float(residuals[name].std(ddof=1) * np.sqrt(252)),
        }
        for i, name in enumerate(returns.columns)
    }
    constant_residual = residuals.columns[residuals.std(ddof=1) <= 1e-10].tolist()
    residual = (
        {"estimable": False, "reason": "Near-zero market residual variance", "symbols": constant_residual}
        if constant_residual
        else {"estimable": True, **dependence(residuals)}
    )
    return {
        "raw_asset_risk": raw,
        "in_sample_market_regressions": estimates,
        "market_residual_risk": residual,
        "market_fit_is_out_of_sample_forecast": False,
        "market_residual_returns_are_investable_strategy": False,
        "independent_alpha_proven": False,
    }


def frozen_inputs(policy: dict):
    validate_policy(policy)
    inputs = {}
    for key in ("source_policy", "source_manifest", "source_audit", "research_snapshot"):
        reference = policy[key]
        inputs[key] = read_json(safe_file(ROOT, reference["path"], reference["sha256"]))
    data = verified_market(inputs["source_policy"])
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        state = program.status()
        if state != inputs["research_snapshot"]:
            raise QuantError("The descriptive risk study must bind its exact unchanged research state.")
    finally:
        program.close()
    return data, state


def register(output: Path) -> dict:
    policy = read_json(POLICY)
    data, state = frozen_inputs(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Risk-study evidence must remain inside the isolated workbench.")
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "registered_at": utc_now(),
        "policy": policy,
        "policy_sha256": file_digest(POLICY),
        "auditor_sha256": fingerprint(),
        "event_chain_sha256": state["event_chain_sha256"],
        "source_market_sessions": len(data.close),
        "risk_structure_results_computed": False,
        "candidate_performance_computed": False,
        "new_strategy_evaluations": 0,
        "order_authority": False,
    }
    write_json(output / "registration.json", result)
    return result


def run(registration_path: Path, output: Path) -> dict:
    policy = read_json(POLICY)
    data, before = frozen_inputs(policy)
    registration_path = (
        registration_path if registration_path.is_absolute() else ROOT / registration_path
    )
    if registration_path.is_symlink() or not registration_path.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Risk registration must remain inside the isolated workbench.")
    registration = read_json(safe_file(ROOT, registration_path.relative_to(ROOT).as_posix()))
    if (
        registration["policy"] != policy
        or registration["policy_sha256"] != file_digest(POLICY)
        or registration["auditor_sha256"] != fingerprint()
        or registration["event_chain_sha256"] != before["event_chain_sha256"]
        or pd.Timestamp(registration["registered_at"]).tzinfo is None
        or pd.Timestamp(registration["registered_at"]) > pd.Timestamp(utc_now())
        or registration["risk_structure_results_computed"] is not False
        or registration["candidate_performance_computed"] is not False
    ):
        raise QuantError("Risk structure must use exactly the registered source, cohorts and code.")
    daily = data.close.pct_change(fill_method=None)
    studies = {}
    for years in policy["horizons_years"]:
        end = pd.Timestamp(policy["as_of"])
        dates = sessions(end - pd.DateOffset(years=years) + pd.Timedelta(days=1), end)
        observations = daily.loc[dates]
        market = observations["SPY"]
        rates = data.risk_free.loc[dates]
        cohorts = {}
        for name, symbols in policy["cohorts"].items():
            values = observations.loc[:, symbols]
            summary = market_structure(values, market, rates)
            rolling = []
            for i, day in enumerate(dates):
                if i < 251 or not is_month_end(day):
                    continue
                window = dates[i - 251 : i + 1]
                rolling_risk = dependence(values.loc[window])
                rolling.append({
                    "end": str(day.date()),
                    "start": str(window[0].date()),
                    "sessions": len(window),
                    "correlation_dimension": rolling_risk["correlation_structure"][
                        "participation_ratio_dimension"
                    ],
                    "covariance_dimension": rolling_risk["covariance_structure"][
                        "participation_ratio_dimension"
                    ],
                    "leading_correlation_component_fraction": rolling_risk["correlation_structure"][
                        "leading_eigenvalue_fraction"
                    ],
                })
            if not rolling:
                raise QuantError("No complete registered rolling dependence windows exist.")
            summary["rolling_252_session_risk"] = {
                "overlapping_windows_not_independent": True,
                "windows": len(rolling),
                "minimum_correlation_dimension": min(row["correlation_dimension"] for row in rolling),
                "median_correlation_dimension": float(
                    np.median([row["correlation_dimension"] for row in rolling])
                ),
                "maximum_correlation_dimension": max(row["correlation_dimension"] for row in rolling),
                "maximum_leading_correlation_component_fraction": max(
                    row["leading_correlation_component_fraction"] for row in rolling
                ),
                "records": rolling,
            }
            cohorts[name] = summary
        legacy = exposure_diagnostics(data, str(dates[0].date()), str(dates[-1].date()))
        original = cohorts["original_four_equity"]
        if (
            not np.isclose(
                original["raw_asset_risk"]["correlation_structure"]["participation_ratio_dimension"],
                legacy["covariance_effective_dimension"], rtol=0, atol=1e-12,
            )
            or any(
                not np.isclose(
                    original["in_sample_market_regressions"][symbol][key],
                    value, rtol=0, atol=1e-12,
                )
                for symbol, estimates in legacy["market_regressions"].items()
                for key, value in estimates.items()
            )
        ):
            raise QuantError("Generalized risk diagnostics disagree with the preserved four-factor helper.")
        studies[str(years)] = {
            "window": {"start": str(dates[0].date()), "end": str(dates[-1].date()), "sessions": len(dates)},
            "cohorts": cohorts,
            "original_four_matches_unchanged_legacy_helper": True,
            "legacy_effective_dimension_field_is_correlation_based": True,
        }
    _, after = frozen_inputs(policy)
    if after != before:
        raise QuantError("The read-only dependence study cannot modify or race changed research state.")
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Risk-study results must remain inside the isolated workbench.")
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "diagnostic_id": policy["diagnostic_id"],
        "registration_sha256": file_digest(registration_path),
        "auditor_sha256": fingerprint(),
        "studies": studies,
        "source_market_sessions": len(data.close),
        "event_chain_sha256": before["event_chain_sha256"],
        "research_state_unchanged": True,
        "queued_definitions_registered_by_diagnostic": False,
        "candidate_rules_or_weights_selected": False,
        "pending_candidate_performance_computed": False,
        "statistics_are_in_sample_descriptive": True,
        "new_strategy_evaluations": 0,
        "independent_factor_alpha_proven": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive factor-risk structure, not strategy returns.")
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "data/factor-risk-structure-20261011/registration.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "data/factor-risk-structure-20261011"
        if args.action == "register" else "reports/factor-risk-structure-20261011"
    )
    try:
        result = register(output) if args.action == "register" else run(args.registration, output)
        print(json.dumps(
            {key: value for key, value in result.items() if key not in ("policy", "studies")},
            indent=2,
        ))
        for years, study in result.get("studies", {}).items():
            print(json.dumps({
                "years": years,
                "risk_dimensions": {
                    name: values["raw_asset_risk"]["correlation_structure"][
                        "participation_ratio_dimension"
                    ]
                    for name, values in study["cohorts"].items()
                },
            }))
    except QuantError as exc:
        parser.exit(2, f"Factor-risk diagnostic blocked: {exc}\n")


if __name__ == "__main__":
    main()
