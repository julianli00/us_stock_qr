from __future__ import annotations

import argparse
import json
import math
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

from us_quant.backtest import BacktestResult, simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import (
    completed_session,
    is_month_end,
    next_session,
    previous_session,
    sessions,
)
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.dual_horizon import load_prices, load_protocol, metrics, seed_window
from us_quant.factor_research import snapshot_descriptor
from us_quant.factor_validation import check_metrics, independent_metrics
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

ROOT = Path(__file__).resolve().parents[2]
FACTORS = ("MTUM", "VLUE", "QUAL", "USMV")
BASE = ("SPY", "IEF", "GLD", "BIL")
PREVIOUS = "equal_growth_gold_min_variance_ensemble"


def validate_policy(policy: dict) -> None:
    goals = {
        "same_rule_in_10y_and_5y": True,
        "base_and_stress_required": True,
        "net_excess_sharpe_strictly_above": 1.0,
        "net_cagr_above_spy": True,
        "max_drawdown_at_most": 0.15,
        "lower_volatility_and_drawdown_than_prior_strategy": True,
        "retained_old_cagr_goal": 0.20,
        "no_leveraged_products": True,
    }
    if (
        policy.get("schema_version") != 1
        or policy.get("prior_disclosed_configurations") != 78
        or policy.get("new_configurations") != 6
        or policy.get("data_start") != "2014-01-02"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("capital_usd") != 10000
        or policy.get("cash_reserve") != 0.02
        or policy.get("commission_per_order") != 1.0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("base_symbols") != list(BASE)
        or policy.get("goals") != goals
        or policy.get("parameters")
        != {
            "volatility_sessions": 63,
            "breadth_horizons": [63, 126, 252],
            "factor_share_min": 0.15,
            "factor_share_max": 0.35,
            "volatility_target": 0.10,
        }
        or policy.get("scenarios")
        != [
            {"id": "base", "cost_bps_per_side": 5.0, "delay_sessions": 1},
            {"id": "stress", "cost_bps_per_side": 20.0, "delay_sessions": 2},
            {"id": "higher_cost", "cost_bps_per_side": 50.0, "delay_sessions": 2},
        ]
    ):
        raise QuantError("The multifactor universe, funding or frozen goals were changed.")
    expected = [
        ("equal", 1.0, 0.0, 0.0, False, False),
        ("equal", 0.6, 0.2, 0.2, False, False),
        ("equal", 0.6, 0.2, 0.2, False, True),
        ("equal", 0.6, 0.2, 0.2, True, False),
        ("inverse_volatility", 0.6, 0.2, 0.2, False, False),
        ("inverse_volatility", 0.6, 0.2, 0.2, True, True),
    ]
    keys = (
        "factor_weighting",
        "equity_budget",
        "ief_budget",
        "gold_budget",
        "breadth_defense",
        "volatility_cap",
    )
    candidates = policy.get("candidates", [])
    if [tuple(item.get(key) for key in keys) for item in candidates] != expected:
        raise QuantError("All six distinct preregistered controls must remain disclosed.")
    ids = [item.get("id") for item in candidates]
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) for name in ids):
        raise QuantError("Unsafe multifactor candidate identifier.")
    if len(set(ids)) != 6:
        raise QuantError("Duplicate multifactor candidate identifier.")
    definitions = policy.get("factors", [])
    if (
        [item.get("symbol") for item in definitions] != list(FACTORS)
        or [item.get("factor") for item in definitions]
        != ["momentum", "value", "quality", "low_volatility"]
        or any(item.get("intended_daily_leverage") != 1.0 for item in definitions)
        or any(
            pd.Timestamp(item["inception"]) >= pd.Timestamp(policy["data_start"])
            for item in definitions
        )
        or any(
            not item.get("issuer_url", "").startswith("https://www.ishares.com/us/products/")
            for item in definitions
        )
    ):
        raise QuantError(
            "Distinct factor definitions and actual post-inception funds are required."
        )
    method = policy.get("methodology", {})
    if (
        method.get("implementation_type")
        != "multi_factor_etf_sleeve_portfolio_not_direct_stock_scoring"
        or method.get("history_previously_exposed") is not True
        or any(
            method.get(key) is not False
            for key in (
                "independent_forward_validation",
                "automatic_baseline_replacement",
                "order_authority",
                "persistent_automation_started",
            )
        )
        or policy.get("comparison", {}).get("old_candidate") != PREVIOUS
    ):
        raise QuantError(
            "Multifactor research must not claim stock/PIT coverage or trading authority."
        )


def fingerprint() -> str:
    return digest_json(
        {
            "multifactor_module": file_digest(Path(__file__)),
            "frozen_core": implementation_fingerprint(),
            "dual_horizon": file_digest(Path(__file__).with_name("dual_horizon.py")),
            "factor_helpers": file_digest(Path(__file__).with_name("factor_research.py")),
            "independent_metrics": file_digest(Path(__file__).with_name("factor_validation.py")),
            "independent_accounting": file_digest(Path(__file__).with_name("bt_audit.py")),
            "versions": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "bt",
                    "ffn",
                    "exchange-calendars",
                )
            },
        }
    )


def prior_evidence(policy: dict) -> dict:
    item = policy["comparison"]
    path = ROOT / item["old_results"]
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(ROOT)
        or file_digest(path) != item["old_results_sha256"]
    ):
        raise QuantError("The preserved high-risk comparison evidence changed.")
    return read_json(path)


def register(policy: dict, base_path: Path, output: Path) -> dict:
    validate_policy(policy)
    if output.exists():
        raise QuantError("Refusing to overwrite a multifactor preregistration.")
    prior_evidence(policy)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    descriptor = snapshot_descriptor(base_path, set(comparison.symbols) | {"IRX"}, comparison)
    previous_registration = read_json(ROOT / policy["comparison"]["old_data_registration"])
    if descriptor != previous_registration["sources"]["base"]:
        raise QuantError("The benchmark/defense snapshot differs from the previously audited data.")
    record = {
        "schema_version": 1,
        "round_id": policy["round_id"],
        "registered_at": utc_now(),
        "policy_sha256": digest_json(policy),
        "implementation_sha256": fingerprint(),
        "windows": comparison.windows(),
        "base_source": descriptor,
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "factor_symbols": list(FACTORS),
        "prior_disclosed_configurations": 78,
        "new_configurations": 6,
        "total_disclosed_configurations": 84,
        "new_price_snapshot_obtained": False,
        "history_previously_exposed": True,
        "order_authority": False,
    }
    write_json(output, record)
    return record


def verify_registration(policy: dict, record: dict) -> None:
    validate_policy(policy)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    if (
        record.get("round_id") != policy["round_id"]
        or record.get("policy_sha256") != digest_json(policy)
        or record.get("implementation_sha256") != fingerprint()
        or record.get("windows") != comparison.windows()
        or record.get("candidate_ids") != [item["id"] for item in policy["candidates"]]
        or record.get("factor_symbols") != list(FACTORS)
        or record.get("total_disclosed_configurations") != 84
        or record.get("history_previously_exposed") is not True
        or record.get("order_authority") is not False
    ):
        raise QuantError("Multifactor registration no longer matches code, rules or authority.")
    timestamp = pd.Timestamp(record.get("registered_at"))
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise QuantError("Multifactor registration needs a timezone-aware timestamp.")


def corporate_actions(payload: dict, frame: pd.DataFrame, symbol: str) -> dict:
    record = payload["chart"]["result"][0]
    days = set()
    events = record.get("events", {})
    for kind in ("dividends", "splits", "capitalGains"):
        for event in events.get(kind, {}).values():
            days.add(
                pd.Timestamp(event["date"], unit="s", tz="UTC")
                .tz_convert("America/New_York")
                .normalize()
                .tz_localize(None)
            )
    changes = (frame["adj_close"] / frame["close"]).pct_change(fill_method=None).abs()
    unexplained = [str(day.date()) for day in changes[changes > 0.0001].index if day not in days]
    if unexplained:
        raise QuantError(f"{symbol} has unexplained adjustment changes: {unexplained[:8]}")
    return {
        "dividend_events": len(events.get("dividends", {})),
        "split_events": len(events.get("splits", {})),
        "unexplained_adjustment_jumps": 0,
    }


def fetch(policy: dict, registration: dict, output: Path) -> dict:
    verify_registration(policy, registration)
    if output.exists():
        raise QuantError("Refusing to overwrite the multifactor source snapshot.")
    if pd.Timestamp(policy["as_of"]) > completed_session():
        raise QuantError("The requested multifactor end date has not completed.")
    collected = {}
    with requests.Session() as client:
        client.headers["User-Agent"] = "us-quant-research/0.1 (personal research)"
        for factor in policy["factors"]:
            symbol = factor["symbol"]
            try:
                document = client.get(factor["issuer_url"], timeout=(10, 60))
                document.raise_for_status()
                if symbol not in document.text or factor["instrument"] not in document.text:
                    raise QuantError(f"Unable to confirm issuer document identity for {symbol}.")
                response = client.get(
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={
                        "period1": int(pd.Timestamp(policy["data_start"], tz="UTC").timestamp()),
                        "period2": int(
                            (
                                pd.Timestamp(policy["as_of"], tz="UTC") + pd.Timedelta(days=1)
                            ).timestamp()
                        ),
                        "interval": "1d",
                        "events": "div,splits",
                    },
                    timeout=(10, 60),
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise QuantError(
                    f"Public multifactor data request failed for {symbol}; no proxy fallback: {exc}"
                ) from exc
            frame = parse_chart(payload, symbol, policy["data_start"], policy["as_of"])
            actions = corporate_actions(payload, frame, symbol)
            collected[symbol] = (response, document, frame, actions)
    new_output_directory(output)
    files, sources = {}, {}
    for symbol, (response, document, frame, actions) in collected.items():
        content = {
            f"{symbol}.csv": frame.to_csv(float_format="%.12g"),
            f"raw/{symbol}.json": response.text,
            f"issuer/{symbol}.html": document.text,
        }
        for relative, text in content.items():
            write_text_atomic(output / relative, text)
            files[relative] = file_digest(output / relative)
        sources[symbol] = {
            "price_url": response.url,
            "issuer_url": document.url,
            "first_session": str(frame.index[0].date()),
            "last_session": str(frame.index[-1].date()),
            "rows": len(frame),
            **actions,
        }
    manifest = {
        "schema_version": 1,
        "retrieved_at": utc_now(),
        "policy_sha256": digest_json(policy),
        "registration_sha256": digest_json(registration),
        "data_start": policy["data_start"],
        "data_end": policy["as_of"],
        "base_source_manifest_sha256": registration["base_source"]["manifest_sha256"],
        "base_source_retrieved_at": registration["base_source"]["retrieved_at"],
        "sources": sources,
        "files": files,
        "synthetic_preinception_prices": False,
        "current_fundamentals_used_as_historical_signals": False,
        "issuer_documents_are_current_not_historical_methodology_proof": True,
        "snapshot_type": "actual_fund_price_history_not_constituent_level_PIT_data",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_market(policy: dict, registration: dict, base_path: Path, source_path: Path) -> MarketData:
    verify_registration(policy, registration)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    descriptor = snapshot_descriptor(base_path, set(comparison.symbols) | {"IRX"}, comparison)
    if descriptor != registration["base_source"]:
        raise QuantError("The registered benchmark/defensive source snapshot changed.")
    if source_path.is_symlink() or (source_path / "manifest.json").is_symlink():
        raise QuantError("A multifactor snapshot or manifest cannot be a symlink.")
    manifest = read_json(source_path / "manifest.json")
    required = {
        f"{prefix}{symbol}{suffix}"
        for symbol in FACTORS
        for prefix, suffix in (("", ".csv"), ("raw/", ".json"), ("issuer/", ".html"))
    }
    if (
        manifest.get("registration_sha256") != digest_json(registration)
        or manifest.get("policy_sha256") != digest_json(policy)
        or manifest.get("data_start") != policy["data_start"]
        or manifest.get("data_end") != policy["as_of"]
        or manifest.get("base_source_manifest_sha256") != descriptor["manifest_sha256"]
        or set(manifest.get("files", {})) != required
        or set(manifest.get("sources", {})) != set(FACTORS)
        or manifest.get("synthetic_preinception_prices") is not False
    ):
        raise QuantError("Multifactor snapshot dates, identities or registration are inconsistent.")
    retrieved = pd.Timestamp(manifest.get("retrieved_at"))
    if (
        pd.isna(retrieved)
        or retrieved.tzinfo is None
        or retrieved < pd.Timestamp(registration["registered_at"])
    ):
        raise QuantError("Multifactor prices were not obtained under the recorded preregistration.")
    for relative, digest in manifest["files"].items():
        path = source_path / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(source_path.resolve())
            or file_digest(path) != digest
        ):
            raise QuantError("A multifactor price or issuer source file was changed.")
    old = load_prices(comparison, base_path)
    index = sessions(policy["data_start"], policy["as_of"])
    frames = {
        symbol: pd.read_csv(source_path / f"{symbol}.csv", index_col="date", parse_dates=["date"])
        for symbol in FACTORS
    }

    def assemble(existing: pd.DataFrame, field: str) -> pd.DataFrame:
        extra = pd.DataFrame({symbol: frame[field] for symbol, frame in frames.items()})
        if not extra.index.equals(index):
            raise QuantError("A factor ETF omits part of the registered historical window.")
        return pd.concat([existing.loc[index, list(BASE)], extra], axis=1)

    data = MarketData(
        assemble(old.open, "adj_open"),
        assemble(old.close, "adj_close"),
        assemble(old.raw_close, "close"),
        assemble(old.volume, "volume"),
        old.risk_free.loc[index],
    )
    data.validate()
    return data


def bounded_factor_shares(volatility: pd.Series, lower: float, upper: float) -> pd.Series:
    count = len(volatility)
    if (
        count < 2
        or not np.isfinite(volatility).all()
        or (volatility <= 1e-10).any()
        or not 0 < lower < upper <= 1
        or lower * count >= 1
        or upper * count <= 1
    ):
        raise QuantError("Factor risk budgets need finite volatilities and feasible bounds.")
    inverse = 1 / volatility
    scale = brentq(
        lambda value: float((inverse * value).clip(lower, upper).sum()) - 1,
        0,
        float(upper / inverse.min()),
        xtol=1e-14,
    )
    return (inverse * scale).clip(lower, upper)


def cap_volatility(
    target: pd.Series, covariance: pd.DataFrame, limit: float, budget: float
) -> pd.Series:
    if (
        not target.index.equals(covariance.index)
        or not covariance.index.equals(covariance.columns)
        or not np.isfinite(covariance.to_numpy()).all()
        or not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance.to_numpy()).min() < -1e-10
        or not math.isfinite(limit)
        or limit <= 0
        or not np.isfinite(target).all()
        or (target < -1e-12).any()
        or not math.isclose(float(target.sum()), budget, abs_tol=1e-10)
    ):
        raise QuantError("Invalid portfolio risk inputs or funding.")

    def volatility(weight: pd.Series) -> float:
        variance = float(weight.to_numpy() @ covariance.to_numpy() @ weight.to_numpy())
        if not math.isfinite(variance) or variance < -1e-10:
            raise QuantError("Invalid portfolio variance.")
        return math.sqrt(max(0, variance))

    if volatility(target) <= limit:
        return target.copy()
    risky = target.copy()
    risky["BIL"] = 0

    def scaled(scale: float) -> pd.Series:
        result = risky * scale
        result["BIL"] = budget - result.sum()
        return result

    if volatility(scaled(0)) > limit:
        raise QuantError("Even the defensive fund exceeds the registered volatility ceiling.")
    scale = brentq(lambda value: volatility(scaled(value)) - limit, 0, 1, xtol=1e-14)
    return scaled(scale)


def target_weights(history: pd.DataFrame, candidate: dict, policy: dict) -> pd.Series:
    if (
        len(history) < 253
        or not history.index.is_unique
        or not history.index.is_monotonic_increasing
        or tuple(history.columns) != (*BASE, *FACTORS)
        or not np.isfinite(history.to_numpy()).all()
        or (history <= 0).any().any()
    ):
        raise QuantError("Multifactor decisions need complete, sorted, post-inception history.")
    parameters, budget = policy["parameters"], 1 - policy["cash_reserve"]
    returns = history.pct_change(fill_method=None).iloc[1:]
    recent = returns.tail(parameters["volatility_sessions"])
    shares = pd.Series(0.25, index=FACTORS)
    if candidate["factor_weighting"] == "inverse_volatility":
        shares = bounded_factor_shares(
            recent.loc[:, list(FACTORS)].std(ddof=1),
            parameters["factor_share_min"],
            parameters["factor_share_max"],
        )
    defense_scale = 1.0
    if candidate["breadth_defense"]:
        positive = []
        for horizon in parameters["breadth_horizons"]:
            move = history.iloc[-1] / history.iloc[-horizon - 1] - 1
            positive.extend((move.loc[list(FACTORS)] > move["BIL"]).tolist())
        defense_scale = sum(positive) / len(positive)
    weights = pd.Series(0.0, index=history.columns)
    weights.loc[list(FACTORS)] = shares * budget * candidate["equity_budget"] * defense_scale
    weights["IEF"] = budget * candidate["ief_budget"]
    weights["GLD"] = budget * candidate["gold_budget"]
    weights["BIL"] = budget - weights.sum()
    if candidate["volatility_cap"]:
        weights = cap_volatility(
            weights, recent.cov() * 252, parameters["volatility_target"], budget
        )
    if (
        not np.isfinite(weights).all()
        or (weights < -1e-12).any()
        or not np.isclose(weights.sum(), budget, rtol=0, atol=1e-10)
        or weights["SPY"] != 0
    ):
        raise QuantError("Multifactor targets violate the unleveraged universe or cash budget.")
    equity = weights.loc[list(FACTORS)]
    if equity.sum() > 1e-12:
        allocation = equity / equity.sum()
        if (allocation < parameters["factor_share_min"] - 1e-10).any() or (
            allocation > parameters["factor_share_max"] + 1e-10
        ).any():
            raise QuantError("A factor exposure disappeared or dominated after risk controls.")
    return weights.clip(lower=0)


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    data.validate()
    targets = {
        candidate["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for candidate in policy["candidates"]
    }
    for index, day in enumerate(data.close.index):
        if index < 252 or not is_month_end(day):
            continue
        for candidate in policy["candidates"]:
            targets[candidate["id"]].loc[day] = target_weights(
                data.close.iloc[: index + 1], candidate, policy
            )
    if any(frame.dropna(how="all").empty for frame in targets.values()):
        raise QuantError("No complete monthly multifactor allocation is available.")
    return targets


def exposure_diagnostics(data: MarketData, start: str, end: str) -> dict:
    daily = data.close.pct_change(fill_method=None).loc[start:end]
    correlations = daily.loc[:, list(FACTORS)].corr()
    if not np.isfinite(correlations.to_numpy()).all():
        raise QuantError("Factor dependence diagnostics require nonconstant complete returns.")
    market = daily["SPY"] - data.risk_free.loc[daily.index]
    x = np.column_stack([np.ones(len(market)), market])
    residuals, estimates = {}, {}
    for symbol in FACTORS:
        excess = daily[symbol] - data.risk_free.loc[daily.index]
        beta, _, rank, _ = np.linalg.lstsq(x, excess, rcond=None)
        if rank != 2:
            raise QuantError("Market-exposure diagnostics are rank deficient.")
        residual = excess - x @ beta
        total = float(((excess - excess.mean()) ** 2).sum())
        if total <= 1e-15:
            raise QuantError("Constant factor returns cannot establish independent exposure.")
        residuals[symbol] = residual
        estimates[symbol] = {
            "beta_vs_spy": float(beta[1]),
            "market_r_squared": float(1 - (residual**2).sum() / total),
        }
    residual_corr = pd.DataFrame(residuals).corr()
    if not np.isfinite(residual_corr.to_numpy()).all():
        raise QuantError("Market-adjusted dependence cannot be estimated from constant residuals.")
    return {
        "return_window": {"start": start, "end": end},
        "raw_return_correlation": correlations.to_dict(),
        "market_regressions": estimates,
        "market_residual_correlation": residual_corr.to_dict(),
        "covariance_effective_dimension": float(
            len(FACTORS) ** 2 / (correlations.to_numpy() ** 2).sum()
        ),
        "declared_factor_sleeves": 4,
        "independent_alpha_sources_proven": False,
        "historical_constituent_overlap_measured": False,
        "interpretation": "Post-result dependence diagnostic, not a future-fitted signal.",
    }


def gate_result(values: dict, spy: dict, old: dict, policy: dict) -> dict:
    for other in (spy, old):
        if any(values[key] != other[key] for key in ("start", "end", "sessions")):
            raise QuantError("Multifactor comparisons must use identical accounting windows.")
    for key in ("cagr", "max_drawdown", "annualized_volatility"):
        if not all(math.isfinite(frame[key]) for frame in (values, spy, old)):
            raise QuantError("Incomplete or nonfinite comparative performance.")
    sharpe = values["sharpe"]
    return {
        "net_excess_sharpe_above_1": bool(
            sharpe is not None and math.isfinite(sharpe) and sharpe > 1
        ),
        "net_cagr_above_spy": bool(values["cagr"] > spy["cagr"]),
        "drawdown_at_most_15pct": bool(
            values["max_drawdown"] <= policy["goals"]["max_drawdown_at_most"]
        ),
        "drawdown_below_previous": bool(values["max_drawdown"] < old["max_drawdown"]),
        "volatility_below_previous": bool(
            values["annualized_volatility"] < old["annualized_volatility"]
        ),
    }


def audit_path(
    data: MarketData, signal: pd.DataFrame, start: str, end: str, policy: dict, scenario: dict
) -> tuple[BacktestResult, pd.DataFrame, dict]:
    result = simulate(
        data,
        signal,
        start,
        end,
        initial_capital=policy["capital_usd"],
        cost_bps=scenario["cost_bps_per_side"],
        commission=policy["commission_per_order"],
        delay=scenario["delay_sessions"],
    )
    independent = independent_equity(
        data,
        signal,
        start,
        end,
        capital=policy["capital_usd"],
        cost_bps=scenario["cost_bps_per_side"],
        commission=policy["commission_per_order"],
        delay=scenario["delay_sessions"],
    )
    actual, difference = independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], policy["capital_usd"]
    )
    reported = metrics(result.frame)
    check_metrics(actual, reported)
    if (result.weights < -1e-12).any().any() or (result.weights.sum(axis=1) > 1 + 1e-10).any():
        raise QuantError("A supposedly unleveraged strategy has short or borrowed exposure.")
    return (
        result,
        independent,
        {
            "max_equity_difference_usd": difference,
            "independent_metrics_passed": True,
            "no_account_leverage": True,
        },
    )


def rolling_summary(frame: pd.DataFrame, spy: pd.DataFrame, years: int, policy: dict) -> dict:
    records = []
    for end in frame.index:
        if not is_month_end(end):
            continue
        anchor = end - pd.DateOffset(years=years)
        if anchor < previous_session(frame.index[0]):
            continue
        interval = frame.loc[(frame.index > anchor) & (frame.index <= end)]
        values, benchmark = metrics(interval), metrics(spy.loc[interval.index])
        sharpe = values["sharpe"]
        records.append(
            {
                "start": values["start"],
                "end": values["end"],
                "cagr": values["cagr"],
                "sharpe": sharpe,
                "max_drawdown": values["max_drawdown"],
                "spy_cagr": benchmark["cagr"],
                "risk_goal_pass": values["max_drawdown"] <= policy["goals"]["max_drawdown_at_most"],
                "return_goals_pass": bool(
                    sharpe is not None and sharpe > 1 and values["cagr"] > benchmark["cagr"]
                ),
            }
        )
    if not records:
        raise QuantError("No complete multifactor rolling windows exist.")
    return {
        "windows": len(records),
        "risk_goal_pass_count": sum(row["risk_goal_pass"] for row in records),
        "return_goals_pass_count": sum(row["return_goals_pass"] for row in records),
        "max_drawdown": max(row["max_drawdown"] for row in records),
        "records": records,
        "independent_samples": False,
    }


def evaluate(policy: dict, registration: dict, base_path: Path, source: Path, output: Path) -> dict:
    verify_registration(policy, registration)
    data = load_market(policy, registration, base_path, source)
    previous = prior_evidence(policy)["candidates"][PREVIOUS]["windows"]
    targets = build_targets(data, policy)
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    windows = comparison.windows()
    windows.append(
        {
            "years": "nonoverlap_early",
            "first_return_session": windows[0]["first_return_session"],
            "last_session": str(previous_session(windows[1]["first_return_session"]).date()),
        }
    )
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "round_id": policy["round_id"],
        "created_at": utc_now(),
        "registration_sha256": digest_json(registration),
        "implementation_sha256": fingerprint(),
        "price_manifest_sha256": file_digest(source / "manifest.json"),
        "prior_disclosed_configurations": 78,
        "new_configurations": 6,
        "total_disclosed_configurations": 84,
        "implementation_type": policy["methodology"]["implementation_type"],
        "factor_definitions": policy["factors"],
        "as_of": policy["as_of"],
        "candidates": {},
        "spy": {},
        "factor_dependence": {},
        "previous_strategy_metrics": {
            key: {s: x["metrics"] for s, x in value.items() if s in ("base", "stress")}
            for key, value in previous.items()
        },
        "independent_accounting_paths": 0,
        "stability_qualified_candidates": [],
        "full_goal_qualified_candidates": [],
        "return_goal_qualified_candidates": [],
        "selected_stability_research_candidate": None,
        "independent_alpha_sources_proven": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
        "automatic_baseline_replacement": False,
        "limitations": [
            "ETF factor-sleeve allocation, not direct stock-level point-in-time factor scoring.",
            "Four economic labels do not imply independent holdings, returns or alpha.",
            "Actual fund returns embed historical constituent and methodology changes.",
            "Factor prices and preserved benchmark history have different retrieval vintages.",
            "Known historical windows overlap; no new independent forward observations.",
            "Monthly volatility targeting cannot guarantee a drawdown or intraday loss ceiling.",
            "Bonds/gold may fall together with equities; fund expenses are already in prices.",
            "Fractional units, model costs and no personal taxes remain simplifying assumptions.",
        ],
    }
    spy_frames = {}

    def save(prefix, own, independent):
        write_text_atomic(output / f"{prefix}.csv", own.frame.to_csv(float_format="%.12g"))
        write_text_atomic(output / f"{prefix}-bt.csv", independent.to_csv(float_format="%.12g"))
        write_text_atomic(
            output / f"{prefix}-weights.csv", own.weights.to_csv(float_format="%.12g")
        )

    for window in windows:
        key, start, end = (
            f"{window['years']}y",
            window["first_return_session"],
            window["last_session"],
        )
        result["spy"][key] = {}
        result["factor_dependence"][key] = exposure_diagnostics(data, start, end)
        for scenario in policy["scenarios"]:
            own, independent, audit = audit_path(
                data, buy_and_hold_signals(data.close, "SPY", start), start, end, policy, scenario
            )
            label = scenario["id"]
            result["spy"][key][label] = {"metrics": metrics(own.frame), "audit": audit}
            spy_frames[key, label] = own.frame
            save(f"spy/{key}-{label}", own, independent)
            result["independent_accounting_paths"] += 1
    for candidate in policy["candidates"]:
        identifier = candidate["id"]
        row = {"definition": candidate, "windows": {}}
        for window in windows:
            key, start, end = (
                f"{window['years']}y",
                window["first_return_session"],
                window["last_session"],
            )
            row["windows"][key] = {}
            for scenario in policy["scenarios"]:
                label = scenario["id"]
                own, independent, audit = audit_path(
                    data, seed_window(targets[identifier], start), start, end, policy, scenario
                )
                values = metrics(own.frame)
                old = previous[key]["base" if label == "base" else "stress"]["metrics"]
                gates = gate_result(values, result["spy"][key][label]["metrics"], old, policy)
                if label != "base":
                    gates["also_beats_base_cost_spy"] = (
                        values["cagr"] > result["spy"][key]["base"]["metrics"]["cagr"]
                    )
                row["windows"][key][label] = {
                    "metrics": values,
                    "gates": gates,
                    "all_goals_pass": all(gates.values()),
                    "mean_factor_equity_exposure": float(
                        own.weights.loc[:, list(FACTORS)].sum(axis=1).mean()
                    ),
                    "max_factor_equity_exposure": float(
                        own.weights.loc[:, list(FACTORS)].sum(axis=1).max()
                    ),
                    "mean_bil_exposure": float(own.weights["BIL"].mean()),
                    "audit": audit,
                    "old_20pct_cagr_goal": values["cagr"] > 0.20,
                }
                if label == "base":
                    row["windows"][key]["conditional_bootstrap"] = block_bootstrap(
                        own.frame["return"],
                        spy_frames[key, label]["return"],
                        own.frame["risk_free"],
                        samples=1000,
                        block=21,
                        seed=20261010,
                    )
                save(f"{identifier}/{key}-{label}", own, independent)
                result["independent_accounting_paths"] += 1
        gate_sets = [
            row["windows"][w][s]["gates"] for w in ("10y", "5y") for s in ("base", "stress")
        ]
        stability = all(
            all(
                g[key]
                for key in (
                    "drawdown_at_most_15pct",
                    "drawdown_below_previous",
                    "volatility_below_previous",
                )
            )
            for g in gate_sets
        )
        returns_pass = all(
            g["net_excess_sharpe_above_1"]
            and g["net_cagr_above_spy"]
            and g.get("also_beats_base_cost_spy", True)
            for g in gate_sets
        )
        row["stability_goals_pass"] = stability
        row["return_goals_pass"] = returns_pass
        row["full_goals_pass"] = stability and returns_pass
        result["candidates"][identifier] = row
        for passed, key in (
            (stability, "stability_qualified_candidates"),
            (returns_pass, "return_goal_qualified_candidates"),
            (stability and returns_pass, "full_goal_qualified_candidates"),
        ):
            if passed:
                result[key].append(identifier)
        write_text_atomic(
            output / f"{identifier}/issued-targets.csv",
            targets[identifier].dropna(how="all").to_csv(float_format="%.12g"),
        )
        write_json(
            output / "progress.json", {"complete": False, "completed": list(result["candidates"])}
        )
    if result["stability_qualified_candidates"]:

        def ranking(name):
            paths = [
                result["candidates"][name]["windows"][w][s]["metrics"]
                for w in ("10y", "5y")
                for s in ("base", "stress")
            ]
            return (
                max(x["max_drawdown"] for x in paths),
                -min(x["sharpe"] if x["sharpe"] is not None else -math.inf for x in paths),
                name,
            )

        result["selected_stability_research_candidate"] = min(
            result["stability_qualified_candidates"], key=ranking
        )
    first = next_session(next(iter(targets.values())).dropna(how="all").index[0])
    start = str(first.date())
    result["rolling_windows"] = {}
    for scenario in policy["scenarios"][:2]:
        label = scenario["id"]
        spy, independent, _ = audit_path(
            data,
            buy_and_hold_signals(data.close, "SPY", start),
            start,
            policy["as_of"],
            policy,
            scenario,
        )
        save(f"continuous/spy-{label}", spy, independent)
        result["independent_accounting_paths"] += 1
        for candidate in policy["candidates"]:
            name = candidate["id"]
            own, independent, audit = audit_path(
                data, targets[name], start, policy["as_of"], policy, scenario
            )
            result["rolling_windows"].setdefault(name, {})[label] = {
                "audit": audit,
                **{
                    f"{years}y": rolling_summary(own.frame, spy.frame, years, policy)
                    for years in (10, 5)
                },
            }
            save(f"continuous/{name}-{label}", own, independent)
            result["independent_accounting_paths"] += 1
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "results.json", result)
    write_json(
        output / "progress.json", {"complete": True, "completed": list(result["candidates"])}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unleveraged multifactor ETF research; never orders."
    )
    parser.add_argument("stage", choices=("register", "fetch", "evaluate"))
    parser.add_argument("--policy", type=Path, default=Path("config/multifactor-stability.json"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path("evidence/multifactor_stability_20261010_registration.json"),
    )
    parser.add_argument("--base-data", type=Path, default=Path("data/factor-round-20261007/base"))
    parser.add_argument("--data", type=Path, default=Path("data/multifactor-stability-20261010"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/multifactor-stability-20261010")
    )
    args = parser.parse_args()
    try:
        policy = read_json(args.policy)
        if args.stage == "register":
            result = register(policy, args.base_data, args.registration)
            keys = (
                "registered_at",
                "candidate_ids",
                "factor_symbols",
                "total_disclosed_configurations",
            )
        elif args.stage == "fetch":
            result = fetch(policy, read_json(args.registration), args.data)
            keys = (
                "retrieved_at",
                "data_start",
                "data_end",
                "sources",
                "synthetic_preinception_prices",
            )
        else:
            result = evaluate(
                policy, read_json(args.registration), args.base_data, args.data, args.output
            )
            keys = (
                "total_disclosed_configurations",
                "stability_qualified_candidates",
                "return_goal_qualified_candidates",
                "full_goal_qualified_candidates",
                "selected_stability_research_candidate",
                "independent_accounting_paths",
                "independent_forward_validation",
                "automatic_baseline_replacement",
            )
        print(json.dumps({key: result[key] for key in keys}, indent=2))
    except QuantError as exc:
        parser.exit(2, f"Multifactor stability research blocked: {exc}\n")


if __name__ == "__main__":
    main()
