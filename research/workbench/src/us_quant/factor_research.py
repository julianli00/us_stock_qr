from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import DualProtocol, load_prices, load_protocol, metrics, seed_window
from us_quant.metrics import block_bootstrap
from us_quant.storage import (
    digest_json,
    file_digest,
    implementation_fingerprint,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals


def validate_policy(policy: dict) -> None:
    expected_goal = {
        "same_rule_in_both_windows": True,
        "net_excess_sharpe_strictly_above": 1.0,
        "net_cagr_above_spy": True,
        "base_and_stress_required": True,
    }
    if (
        policy.get("schema_version") != 1
        or policy.get("primary_goal") != expected_goal
        or policy.get("retained_secondary_goals")
        != {"net_cagr_strictly_above": 0.20, "max_drawdown_at_most": 0.15}
        or policy.get("prior_disclosed_configurations", 0) < 52
        or policy.get("capital_usd") != 10000.0
        or policy.get("cash_reserve") != 0.02
        or policy.get("base_delay_sessions") != 1
        or policy.get("stress_delay_sessions") != 2
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("independent_future_performance_claim") is not False
    ):
        raise QuantError("Factor research may not relax its objective or enable trading.")
    numeric = [
        policy.get("cost_bps_per_side"),
        policy.get("stress_cost_bps_per_side"),
        policy.get("commission_per_order"),
    ]
    if (
        any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric)
        or not 0 < numeric[0] < numeric[1] < 100
        or numeric[2] <= 0
    ):
        raise QuantError("Factor research requires finite, positive base and stress costs.")
    parameters = policy["parameters"]
    lookbacks = (
        "momentum_sessions",
        "skip_sessions",
        "residual_fit_sessions",
        "volatility_sessions",
        "variance_sessions",
        "trend_sessions",
        "top_k",
    )
    if (
        any(type(parameters.get(key)) is not int or parameters[key] < 1 for key in lookbacks)
        or not parameters["skip_sessions"]
        < parameters["momentum_sessions"]
        < parameters["residual_fit_sessions"]
        or parameters["top_k"] > len(policy["sector_universe"])
    ):
        raise QuantError("Invalid factor lookbacks or selection counts.")
    for key in (
        "max_sector_weight",
        "qqq_reference_volatility",
        "qld_reference_volatility",
        "growth_gold_reference_volatility",
    ):
        value = parameters.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise QuantError("Invalid fixed factor weight or risk reference.")
    for key in ("sector_universe", "supplement_symbols"):
        values = policy[key]
        if (
            not values
            or len(values) != len(set(values))
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[A-Z]{1,6}", value)
                for value in values
            )
        ):
            raise QuantError("Invalid declared ETF universe.")
    if not set(policy["sector_universe"]) | {"QLD"} <= set(policy["supplement_symbols"]):
        raise QuantError("The supplemental universe omits a required factor input.")
    if "macro_universe" in policy:
        universe = policy["macro_universe"]
        if (
            not universe
            or len(universe) != len(set(universe))
            or not set(universe) <= {"QLD", "GLD", "TLT", "IEF", "DBC"}
            or not 2 <= parameters.get("macro_top_k", 0) <= len(universe)
            or not 1 <= parameters.get("rank_top_k", 0) <= len(universe)
            or parameters.get("covariance_sessions") != 252
            or parameters.get("macro_momentum_sessions") != 126
            or parameters.get("covariance_diagonal_shrinkage") != 0.10
            or parameters.get("max_macro_weight") != 0.80
            or parameters.get("trend_horizons") != [63, 126, 252]
        ):
            raise QuantError("Invalid fixed macro allocation experiment.")
    known = set()
    for candidate in policy["candidates"]:
        identifier = candidate.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-z0-9_]+", identifier)
            or identifier in known
            or type(candidate.get("embedded_leverage")) is not bool
        ):
            raise QuantError("Invalid or duplicate factor candidate identity.")
        family = candidate.get("family")
        if family == "sector":
            valid = candidate.get("factor") in {"momentum", "residual", "continuity"}
        elif family == "ensemble":
            parts = candidate.get("components", [])
            valid = bool(parts) and len(parts) == len(set(parts)) and set(parts) <= known
        elif family == "variance":
            valid = candidate.get("asset") in {"QQQ", "QLD"} and candidate.get("reference") in {
                "qqq_reference_volatility",
                "qld_reference_volatility",
            }
        elif family == "growth_gold":
            valid = type(candidate.get("managed")) is bool
        elif family == "macro":
            valid = "macro_universe" in policy and candidate.get("allocation") in {
                "equal",
                "minimum_variance",
                "maximum_diversification",
            }
        elif family == "macro_rank":
            valid = "macro_universe" in policy
        elif family == "growth_gold_trend":
            valid = "macro_universe" in policy and candidate.get("trend") in {
                "sma",
                "multi_horizon",
            }
        else:
            valid = False
        if not valid:
            raise QuantError("Unsupported or incomplete preregistered factor candidate.")
        known.add(identifier)
    count = policy.get("new_configuration_count", 8)
    if (
        type(count) is not int
        or not 1 <= count <= 8
        or len(known) != count
        or not policy.get("sources")
    ):
        raise QuantError("The factor round must retain its complete declared candidate list.")


def fingerprint() -> str:
    return digest_json(
        {
            "factor_module": file_digest(Path(__file__)),
            "frozen_core": implementation_fingerprint(),
            "dual_horizon": file_digest(Path(__file__).with_name("dual_horizon.py")),
            "bt_audit": file_digest(Path(__file__).with_name("bt_audit.py")),
            "versions": {
                name: version(name)
                for name in ("numpy", "pandas", "bt", "exchange-calendars", "scipy")
            },
        }
    )


def snapshot_descriptor(root: Path, required: set[str], comparison: DualProtocol) -> dict:
    manifest_path = root / "manifest.json"
    if root.is_symlink() or manifest_path.is_symlink():
        raise QuantError("A source snapshot or manifest may not be a symlink.")
    manifest = read_json(manifest_path)
    names = manifest.get("files", {})
    expected = {f"{symbol}.csv" for symbol in required} | {
        f"raw/{symbol}.json" for symbol in required
    }
    if (
        set(names) != expected
        or manifest.get("data_start") != comparison.data_start
        or manifest.get("data_end") != comparison.as_of
    ):
        raise QuantError("Snapshot universe or dates differ from the declared source.")
    for name, expected_hash in names.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or file_digest(path) != expected_hash
        ):
            raise QuantError(f"Snapshot artifact is missing, unsafe or revised: {name}")
    retrieved = pd.Timestamp(manifest.get("retrieved_at"))
    if pd.isna(retrieved) or retrieved.tzinfo is None:
        raise QuantError("Snapshot retrieval time must be recorded with a timezone.")
    return {
        "manifest_sha256": file_digest(manifest_path),
        "retrieved_at": manifest["retrieved_at"],
        "data_start": manifest["data_start"],
        "data_end": manifest["data_end"],
        "files": names,
    }


def register(
    policy: dict, comparison: DualProtocol, base: Path, supplement: Path, output: Path
) -> dict:
    validate_policy(policy)
    if output.exists():
        raise QuantError("Refusing to replace a factor preregistration.")
    if "previous_round" in policy:
        root = Path(__file__).resolve().parents[2]
        previous = root / policy["previous_round"]["results"]
        if (
            previous.is_symlink()
            or not previous.resolve().is_relative_to(root)
            or file_digest(previous) != policy["previous_round"]["sha256"]
        ):
            raise QuantError("The disclosed previous-round evidence changed.")
    record = {
        "schema_version": 1,
        "registered_at": utc_now(),
        "round_id": policy["round_id"],
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "implementation_sha256": fingerprint(),
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "windows": comparison.windows(),
        "sources": {
            "base": snapshot_descriptor(base, set(comparison.symbols) | {"IRX"}, comparison),
            "supplement": snapshot_descriptor(
                supplement, set(policy["supplement_symbols"]), comparison
            ),
        },
        "prior_disclosed_configurations": policy["prior_disclosed_configurations"],
        "total_disclosed_after_round": (
            policy["prior_disclosed_configurations"] + len(policy["candidates"])
        ),
        "history_previously_exposed": True,
        "new_data_or_independent_holdout": False,
        "old_forward_ledger_modified": False,
        "order_authority": False,
    }
    write_json(output, record)
    return record


def verify_registration(policy: dict, comparison: DualProtocol, record: dict) -> None:
    validate_policy(policy)
    if (
        record.get("policy_sha256") != digest_json(policy)
        or record.get("comparison_sha256") != digest_json(asdict(comparison))
        or record.get("implementation_sha256") != fingerprint()
        or record.get("candidate_ids") != [item["id"] for item in policy["candidates"]]
        or record.get("windows") != comparison.windows()
        or record.get("total_disclosed_after_round")
        != policy["prior_disclosed_configurations"] + len(policy["candidates"])
        or record.get("order_authority") is not False
        or record.get("history_previously_exposed") is not True
    ):
        raise QuantError("Factor policy, code, windows or authority changed after registration.")
    registered = pd.Timestamp(record.get("registered_at"))
    if pd.isna(registered) or registered.tzinfo is None:
        raise QuantError("Factor registration time must include its timezone.")


def stage_snapshots(
    policy: dict,
    comparison: DualProtocol,
    record: dict,
    base: Path,
    supplement: Path,
    output: Path,
) -> dict:
    verify_registration(policy, comparison, record)
    for label, source, symbols in (
        ("base", base, set(comparison.symbols) | {"IRX"}),
        ("supplement", supplement, set(policy["supplement_symbols"])),
    ):
        if snapshot_descriptor(source, symbols, comparison) != record["sources"][label]:
            raise QuantError("Source snapshot changed after factor registration.")
    new_output_directory(output)
    for label, source in (("base", base), ("supplement", supplement)):
        for relative in ("manifest.json", *record["sources"][label]["files"]):
            destination = output / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
    return {"snapshots_staged": 2, "original_retrieval_dates_preserved": True}


def load_data(policy: dict, comparison: DualProtocol, record: dict, root: Path) -> MarketData:
    verify_registration(policy, comparison, record)
    for label, symbols in (
        ("base", set(comparison.symbols) | {"IRX"}),
        ("supplement", set(policy["supplement_symbols"])),
    ):
        if snapshot_descriptor(root / label, symbols, comparison) != record["sources"][label]:
            raise QuantError("Local snapshot no longer matches the factor preregistration.")
    original = load_prices(comparison, root / "base")
    extra = {
        symbol: pd.read_csv(
            root / "supplement" / f"{symbol}.csv", index_col="date", parse_dates=["date"]
        )
        for symbol in policy["supplement_symbols"]
    }

    def merge(frame: pd.DataFrame, field: str) -> pd.DataFrame:
        return pd.concat(
            [frame, pd.DataFrame({symbol: values[field] for symbol, values in extra.items()})],
            axis=1,
        )

    data = MarketData(
        merge(original.open, "adj_open"),
        merge(original.close, "adj_close"),
        merge(original.raw_close, "close"),
        merge(original.volume, "volume"),
        original.risk_free,
    )
    data.validate()
    return data


def information_discreteness(returns: pd.DataFrame) -> pd.Series:
    if returns.empty or not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise QuantError("Information discreteness requires valid observed daily returns.")
    total = np.expm1(np.log1p(returns).sum())
    return np.sign(total) * ((returns < 0).mean() - (returns > 0).mean())


def inverse_variance_scale(variance: float, reference_volatility: float) -> float:
    if (
        not math.isfinite(variance)
        or variance < 0
        or not math.isfinite(reference_volatility)
        or reference_volatility <= 0
    ):
        raise QuantError("Invalid ex-ante variance or fixed risk reference.")
    return min(1.0, reference_volatility**2 / variance) if variance > 1e-12 else 1.0


def covariance_weights(covariance: np.ndarray, method: str, cap: float) -> np.ndarray:
    count = len(covariance)
    if (
        covariance.shape != (count, count)
        or count < 2
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T)
        or (np.diag(covariance) <= 0).any()
        or np.linalg.eigvalsh(covariance).min() < -1e-10
        or not math.isfinite(cap)
        or not 1 / count <= cap <= 1
        or method not in {"minimum_variance", "maximum_diversification"}
    ):
        raise QuantError("Invalid causal covariance or allocation constraints.")
    matrix = covariance / np.trace(covariance)
    volatility = np.sqrt(np.diag(matrix))

    def objective(weight: np.ndarray) -> float:
        variance = float(weight @ matrix @ weight)
        if method == "minimum_variance":
            return variance
        return -float(volatility @ weight) / math.sqrt(variance)

    def gradient(weight: np.ndarray) -> np.ndarray:
        if method == "minimum_variance":
            return 2 * matrix @ weight
        variance = float(weight @ matrix @ weight)
        numerator = float(volatility @ weight)
        return -volatility / np.sqrt(variance) + numerator * (matrix @ weight) / variance**1.5

    result = minimize(
        objective,
        np.full(count, 1 / count),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, cap)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(count)},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    weights = result.x
    if (
        not result.success
        or not np.isfinite(weights).all()
        or abs(weights.sum() - 1) > 1e-8
        or (weights < -1e-10).any()
        or (weights > cap + 1e-10).any()
    ):
        raise QuantError("Constrained allocation solver failed; no equal-weight fallback.")
    return weights


def macro_weights(history: pd.DataFrame, candidate: dict, policy: dict) -> pd.Series:
    p, universe = policy["parameters"], policy["macro_universe"]
    budget = 1 - policy["cash_reserve"]
    weight = pd.Series(0.0, index=history.columns)
    returns = history.loc[:, universe].pct_change(fill_method=None).iloc[1:]
    score = (
        history.loc[:, universe].iloc[-1]
        / history.loc[:, universe].iloc[-p["macro_momentum_sessions"] - 1]
        - 1
    )
    eligible = list(score.index[score > 0])
    if candidate["family"] == "macro_rank":
        recent = returns.tail(p["volatility_sessions"])
        volatility = recent.std(ddof=1)
        correlation = recent.corr()
        if not np.isfinite(correlation.to_numpy()).all():
            raise QuantError("Correlation ranks require nonconstant observed returns.")
        mean_correlation = (correlation.sum() - 1) / (len(universe) - 1)
        score = (
            score.rank(pct=True) + (-volatility).rank(pct=True) + (-mean_correlation).rank(pct=True)
        ) / 3
        top_k = p["rank_top_k"]
    else:
        top_k = p["macro_top_k"]
    selected = sorted(eligible, key=lambda symbol: (-float(score[symbol]), symbol))[:top_k]
    if not selected:
        return weight
    allocated = np.full(len(selected), 1 / len(selected))
    method = candidate.get("allocation", "equal")
    if len(selected) > 1 and method != "equal":
        covariance = returns.loc[:, selected].tail(p["covariance_sessions"]).cov().to_numpy()
        shrinkage = p["covariance_diagonal_shrinkage"]
        covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
        allocated = covariance_weights(covariance, method, p["max_macro_weight"])
    weight.loc[selected] = allocated * budget * len(selected) / top_k
    return weight


def factor_scores(history: pd.DataFrame, risk_free: pd.Series, policy: dict) -> pd.DataFrame:
    p = policy["parameters"]
    if len(history) <= p["residual_fit_sessions"]:
        raise QuantError("Factor scores require the complete registered regression warmup.")
    if (
        not np.isfinite(history.to_numpy()).all()
        or (history <= 0).any().any()
        or not history.index.is_unique
        or not history.index.is_monotonic_increasing
        or not risk_free.index.equals(history.index)
        or not np.isfinite(risk_free).all()
    ):
        raise QuantError("Factor inputs must be finite, sorted and aligned observed data.")
    sector = policy["sector_universe"]
    daily = history.pct_change(fill_method=None).iloc[1:]
    observed = daily.loc[:, sector].iloc[-p["momentum_sessions"] : -p["skip_sessions"]]
    momentum = np.expm1(np.log1p(observed).sum())
    discreteness = information_discreteness(observed)
    training = daily.tail(p["residual_fit_sessions"])
    rf = risk_free.loc[training.index]
    x = np.column_stack([np.ones(len(training)), training["SPY"] - rf])
    y = training.loc[:, sector].sub(rf, axis=0).to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        raise QuantError("The observed market regression is rank deficient.")
    residual = pd.DataFrame(y - x @ coefficients, index=training.index, columns=sector)
    residual = residual.iloc[-p["momentum_sessions"] : -p["skip_sessions"]]
    deviation = residual.std(ddof=1)
    score = pd.Series(0.0, index=sector)
    nonconstant = deviation > 1e-10
    score.loc[nonconstant] = residual.loc[:, nonconstant].sum() / deviation.loc[nonconstant]
    return pd.DataFrame(
        {
            "momentum": momentum,
            "residual": score,
            "continuity": (momentum.rank(pct=True) + (-discreteness).rank(pct=True)) / 2,
            "information_discreteness": discreteness,
            "volatility": daily.loc[:, sector].tail(p["volatility_sessions"]).std(ddof=1),
            "above_trend": (
                history.loc[:, sector].iloc[-1]
                > history.loc[:, sector].tail(p["trend_sessions"]).mean()
            ),
        }
    )


def build_signals(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    required = set(policy["sector_universe"]) | {"SPY", "QQQ", "QLD", "GLD", "BIL"}
    if not required <= set(data.close.columns):
        raise QuantError("Price panel omits a preregistered factor asset.")
    signals = {
        candidate["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for candidate in policy["candidates"]
    }
    p, budget = policy["parameters"], 1 - policy["cash_reserve"]
    for i, day in enumerate(data.close.index):
        if i < p["residual_fit_sessions"] or not is_month_end(day):
            continue
        history = data.close.iloc[: i + 1]
        scores = factor_scores(history, data.risk_free.loc[history.index], policy)
        daily = history.pct_change(fill_method=None).iloc[1:]
        for candidate in policy["candidates"]:
            weight = pd.Series(0.0, index=history.columns)
            family = candidate["family"]
            if family == "sector":
                eligible = scores.loc[
                    (scores["momentum"] > 0)
                    & scores["above_trend"]
                    & (scores["volatility"] > 1e-10)
                ]
                if candidate["factor"] == "residual":
                    eligible = eligible.loc[eligible["residual"] > 0]
                selected = sorted(
                    eligible.index,
                    key=lambda symbol: (-float(eligible.loc[symbol, candidate["factor"]]), symbol),
                )[: p["top_k"]]
                if selected:
                    inverse = 1 / scores.loc[selected, "volatility"]
                    allocated = budget * len(selected) / p["top_k"] * inverse / inverse.sum()
                    weight.loc[selected] = allocated.clip(upper=p["max_sector_weight"])
            elif family == "ensemble":
                weight = sum(signals[name].loc[day] for name in candidate["components"])
                weight /= len(candidate["components"])
            elif family == "variance":
                asset = candidate["asset"]
                variance = float(daily[asset].tail(p["variance_sessions"]).var(ddof=1) * 252)
                weight[asset] = budget * inverse_variance_scale(variance, p[candidate["reference"]])
            elif family in {"growth_gold", "growth_gold_trend"}:
                assets = ["QLD", "GLD"]
                volatility = daily.loc[:, assets].tail(p["volatility_sessions"]).std(ddof=1)
                if (volatility <= 1e-10).any() or not np.isfinite(volatility).all():
                    raise QuantError("Growth/gold risk weights need nonconstant observed returns.")
                allocated = (1 / volatility) / (1 / volatility).sum() * budget
                if family == "growth_gold" and candidate["managed"]:
                    covariance = daily.loc[:, assets].tail(p["variance_sessions"]).cov() * 252
                    variance = float(
                        allocated.to_numpy() @ covariance.to_numpy() @ allocated.to_numpy()
                    )
                    allocated *= inverse_variance_scale(
                        variance, p["growth_gold_reference_volatility"]
                    )
                if family == "growth_gold_trend":
                    if candidate["trend"] == "sma":
                        active = (
                            history.loc[:, assets].iloc[-1]
                            > history.loc[:, assets].tail(p["trend_sessions"]).mean()
                        )
                    else:
                        active = sum(
                            (
                                history.loc[:, assets].iloc[-1]
                                / history.loc[:, assets].iloc[-lookback - 1]
                                - 1
                            )
                            > (history["BIL"].iloc[-1] / history["BIL"].iloc[-lookback - 1] - 1)
                            for lookback in p["trend_horizons"]
                        ) / len(p["trend_horizons"])
                    allocated *= active
                weight.loc[assets] = allocated
            else:
                weight = macro_weights(history, candidate, policy)
            if family != "ensemble":
                weight["BIL"] = budget - weight.sum()
            if (
                not np.isfinite(weight).all()
                or (weight < -1e-12).any()
                or not np.isclose(weight.sum(), budget, atol=1e-10)
            ):
                raise QuantError("Factor target violates its cash-funded long-only budget.")
            signals[candidate["id"]].loc[day] = weight.clip(lower=0)
    if any(frame.dropna(how="all").empty for frame in signals.values()):
        raise QuantError("No complete post-warmup factor decisions are available.")
    return signals


def goal_gates(strategy: dict, benchmark: dict, policy: dict) -> dict:
    if any(strategy.get(key) != benchmark.get(key) for key in ("start", "end", "sessions")):
        raise QuantError("A factor result must use the same benchmark return window.")
    if (
        type(strategy.get("sessions")) is not int
        or strategy["sessions"] < 2
        or not all(math.isfinite(value) for value in (strategy["cagr"], benchmark["cagr"]))
    ):
        raise QuantError("Factor acceptance requires a complete finite result.")
    sharpe = strategy["sharpe"]
    return {
        "net_excess_sharpe_above_1": bool(
            sharpe is not None
            and math.isfinite(sharpe)
            and sharpe > policy["primary_goal"]["net_excess_sharpe_strictly_above"]
        ),
        "net_cagr_above_spy": bool(strategy["cagr"] > benchmark["cagr"]),
    }


def run_path(
    data: MarketData, signal: pd.DataFrame, start: str, end: str, policy: dict, stress: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cost = policy["stress_cost_bps_per_side" if stress else "cost_bps_per_side"]
    delay = policy["stress_delay_sessions" if stress else "base_delay_sessions"]
    result = simulate(
        data,
        signal,
        start,
        end,
        initial_capital=policy["capital_usd"],
        cost_bps=cost,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    independent = independent_equity(
        data,
        signal,
        start,
        end,
        capital=policy["capital_usd"],
        cost_bps=cost,
        commission=policy["commission_per_order"],
        delay=delay,
    )
    difference = float(abs(independent["equity"] - result.frame["equity"]).max())
    if not math.isfinite(difference) or difference > policy["capital_usd"] * 1e-8:
        raise QuantError("Factor cash/equity disagrees with the independent accounting engine.")
    return (
        result.frame,
        independent,
        {
            "max_equity_difference_usd": difference,
            "initial_capital_usd": policy["capital_usd"],
            "cost_bps_per_side": cost,
            "commission_per_order": policy["commission_per_order"],
            "execution_delay_sessions": delay,
        },
    )


def evaluate(
    policy: dict, comparison: DualProtocol, record: dict, data: MarketData, output: Path
) -> dict:
    verify_registration(policy, comparison, record)
    if not data.close.index.equals(sessions(comparison.data_start, comparison.as_of)):
        raise QuantError("Factor data must cover the full declared historical snapshot.")
    signals = build_signals(data, policy)
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "round_id": policy["round_id"],
        "created_at": utc_now(),
        "registration_sha256": digest_json(record),
        "implementation_sha256": fingerprint(),
        "as_of": comparison.as_of,
        "prior_disclosed_configurations": policy["prior_disclosed_configurations"],
        "new_configurations": len(policy["candidates"]),
        "total_disclosed_configurations": record["total_disclosed_after_round"],
        "primary_goal": policy["primary_goal"],
        "retained_secondary_goals": policy["retained_secondary_goals"],
        "candidates": {},
        "benchmarks": {},
        "base_joint_passes": [],
        "base_and_stress_joint_passes": [],
        "old_all_four_gates_joint_passes": [],
        "independent_accounting_paths": 0,
        "history_previously_exposed": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "current_trade_ideas": [],
        "order_authority": False,
        "limitations": [
            "Fixed rules are evaluated on previously exposed, overlapping historical windows.",
            "ETF factors are adaptations, not replications of the cited stock/factor papers.",
            "All trial counts include controls and correlated configuration variants.",
            "QLD variants use embedded daily-reset leverage and may exceed old risk limits.",
            "Bootstrap intervals are conditional and do not correct for selection across trials.",
            "Prices are adjusted historical research data, not first-known live execution quotes.",
            "No paused forward record, broker account, holdings or notification service changed.",
        ],
    }
    windows = comparison.windows()
    earlier = {
        "years": "nonoverlap_early",
        "first_return_session": windows[0]["first_return_session"],
        "last_session": str(
            sessions(
                windows[0]["first_return_session"],
                pd.Timestamp(windows[1]["first_return_session"]) - pd.Timedelta(days=1),
            )[-1].date()
        ),
    }
    benchmark_paths = {}
    for window in (*windows, earlier):
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        result["benchmarks"][key] = {}
        for stress in (False, True):
            label = "stress" if stress else "base"
            frame, independent, audit = run_path(
                data, buy_and_hold_signals(data.close, "SPY", start), start, end, policy, stress
            )
            benchmark_paths[key, label] = frame
            result["benchmarks"][key][label] = {"metrics": metrics(frame), "accounting": audit}
            result["independent_accounting_paths"] += 1
            write_text_atomic(output / f"spy/{key}-{label}.csv", frame.to_csv(float_format="%.12g"))
            write_text_atomic(
                output / f"spy/{key}-{label}-bt.csv", independent.to_csv(float_format="%.12g")
            )
    for candidate in policy["candidates"]:
        identifier = candidate["id"]
        row = {"definition": candidate, "windows": {}}
        for window in (*windows, earlier):
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            initialized = seed_window(signals[identifier], start)
            row["windows"][key] = {}
            for stress in (False, True):
                label = "stress" if stress else "base"
                frame, independent, audit = run_path(data, initialized, start, end, policy, stress)
                values = metrics(frame)
                gates = goal_gates(values, result["benchmarks"][key][label]["metrics"], policy)
                if stress:
                    gates["also_beats_base_cost_spy"] = bool(
                        values["cagr"] > result["benchmarks"][key]["base"]["metrics"]["cagr"]
                    )
                secondary = {
                    "net_cagr_above_20pct": bool(values["cagr"] > 0.20),
                    "max_drawdown_at_most_15pct": bool(values["max_drawdown"] <= 0.15),
                }
                row["windows"][key][label] = {
                    "metrics": values,
                    "primary_gates": gates,
                    "secondary_gates": secondary,
                    "primary_pass": all(gates.values()),
                    "accounting": audit,
                }
                result["independent_accounting_paths"] += 1
                if not stress:
                    row["windows"][key]["conditional_bootstrap"] = block_bootstrap(
                        frame["return"],
                        benchmark_paths[key, label]["return"],
                        frame["risk_free"],
                        samples=1000,
                        block=21,
                        seed=20261007,
                    )
                prefix = f"{identifier}/{key}-{label}"
                write_text_atomic(output / f"{prefix}.csv", frame.to_csv(float_format="%.12g"))
                write_text_atomic(
                    output / f"{prefix}-bt.csv", independent.to_csv(float_format="%.12g")
                )
        row["base_both_windows_pass"] = all(
            row["windows"][key]["base"]["primary_pass"] for key in ("10y", "5y")
        )
        row["base_and_stress_both_windows_pass"] = all(
            row["windows"][key][label]["primary_pass"]
            for key in ("10y", "5y")
            for label in ("base", "stress")
        )
        row["retained_old_all_four_goals_pass"] = bool(
            row["base_and_stress_both_windows_pass"]
            and all(
                all(row["windows"][key][label]["secondary_gates"].values())
                for key in ("10y", "5y")
                for label in ("base", "stress")
            )
        )
        result["candidates"][identifier] = row
        for flag, destination in (
            ("base_both_windows_pass", "base_joint_passes"),
            ("base_and_stress_both_windows_pass", "base_and_stress_joint_passes"),
            ("retained_old_all_four_goals_pass", "old_all_four_gates_joint_passes"),
        ):
            if row[flag]:
                result[destination].append(identifier)
        write_text_atomic(
            output / f"{identifier}/issued-signals.csv",
            signals[identifier].dropna(how="all").to_csv(float_format="%.12g"),
        )
        write_json(
            output / "progress.json",
            {
                "round_id": policy["round_id"],
                "completed_candidates": list(result["candidates"]),
                "planned_candidates": record["candidate_ids"],
                "complete": False,
            },
        )
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    result["historical_primary_target_passed"] = bool(result["base_and_stress_joint_passes"])
    write_json(output / "results.json", result)
    write_json(
        output / "progress.json",
        {
            "round_id": policy["round_id"],
            "completed_candidates": list(result["candidates"]),
            "planned_candidates": record["candidate_ids"],
            "complete": True,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregistered ETF factor research; never orders.")
    parser.add_argument("stage", choices=("register", "stage-data", "evaluate"))
    parser.add_argument("--policy", type=Path, default=Path("config/factor-research.json"))
    parser.add_argument("--comparison", type=Path, default=Path("config/dual-horizon.json"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path("evidence/factor_round_20261007_registration.json"),
    )
    parser.add_argument("--source-base", type=Path)
    parser.add_argument("--source-supplement", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/factor-round-20261007"))
    parser.add_argument("--output", type=Path, default=Path("reports/factor-round-20261007"))
    args = parser.parse_args()
    try:
        policy, comparison = read_json(args.policy), load_protocol(args.comparison)
        if args.stage in {"register", "stage-data"} and (
            args.source_base is None or args.source_supplement is None
        ):
            raise QuantError(
                "Snapshot source paths are required; no implicit external cache reads."
            )
        if args.stage == "register":
            result = register(
                policy, comparison, args.source_base, args.source_supplement, args.registration
            )
            summary = {
                "registered_at": result["registered_at"],
                "candidate_ids": result["candidate_ids"],
                "total_disclosed_after_round": result["total_disclosed_after_round"],
                "history_previously_exposed": True,
            }
        else:
            record = read_json(args.registration)
            if args.stage == "stage-data":
                summary = stage_snapshots(
                    policy, comparison, record, args.source_base, args.source_supplement, args.data
                )
            else:
                data = load_data(policy, comparison, record, args.data)
                result = evaluate(policy, comparison, record, data, args.output)
                summary = {
                    key: result[key]
                    for key in (
                        "total_disclosed_configurations",
                        "base_joint_passes",
                        "base_and_stress_joint_passes",
                        "old_all_four_gates_joint_passes",
                        "independent_accounting_paths",
                        "historical_primary_target_passed",
                        "investment_objective_verified",
                        "order_authority",
                    )
                }
        print(json.dumps(summary, indent=2, allow_nan=False))
    except QuantError as exc:
        parser.exit(2, f"Factor research blocked: {exc}\n")


if __name__ == "__main__":
    main()
