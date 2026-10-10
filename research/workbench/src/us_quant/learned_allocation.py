from __future__ import annotations

import argparse
import re
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_horizon import (
    DualProtocol,
    gates,
    metrics,
    rolling_diagnostics,
    run_window,
    seed_window,
)
from us_quant.dual_horizon import (
    load_protocol as load_dual,
)
from us_quant.legacy_horizons import combined_prices, verify_comparison
from us_quant.metrics import block_bootstrap
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals

NUMERIC_FEATURES = (
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "volatility_21d",
    "volatility_63d",
    "trend_200d",
    "drawdown_63d",
    "spy_return_3m",
    "spy_volatility_63d",
    "ief_return_3m",
    "tip_return_3m",
)
MODEL_PARAMETERS = {
    "ridge": {"alpha": 10.0, "solver": "svd"},
    "hist_gradient_boosting": {
        "learning_rate": 0.05,
        "max_iter": 150,
        "max_leaf_nodes": 7,
        "max_depth": 3,
        "min_samples_leaf": 30,
        "l2_regularization": 10.0,
        "max_bins": 64,
        "early_stopping": False,
        "random_state": 20261006,
    },
}


def validate_policy(policy: dict, comparison: DualProtocol) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("prior_disclosed_trials") != 42
        or policy.get("capital_usd") != comparison.capital_usd
        or policy.get("as_of") != comparison.as_of
        or policy.get("horizons_years") != list(comparison.horizons_years)
        or policy.get("training_months") != 60
        or policy.get("minimum_training_months") != 36
        or policy.get("minimum_predicted_excess_return") != 0.005
        or policy.get("volatility_window") != 63
        or policy.get("cash_reserve") != 0.02
        or policy.get("cost_bps_per_side") != 7.5
        or policy.get("stress_cost_bps_per_side") != 30
        or policy.get("commission_per_order") != 1
        or policy.get("model_parameters") != MODEL_PARAMETERS
        or policy.get("features") != [*NUMERIC_FEATURES, "symbol"]
        or policy.get("hyperparameter_search") is not False
        or policy.get("random_train_test_split") is not False
        or policy.get("auto_order_submission") is not False
        or policy.get("data_reuse_disclosed") is not True
    ):
        raise QuantError("Learned allocation must use the declared fixed causal training policy.")
    combinations = set()
    identifiers = set()
    for item in policy.get("candidates", []):
        if (
            set(item) != {"id", "model", "top_k", "max_weight", "target_volatility"}
            or not isinstance(item["id"], str)
            or not re.fullmatch(r"[a-z0-9_]+", item["id"])
            or type(item["top_k"]) is not int
            or item["model"] not in MODEL_PARAMETERS
            or item["top_k"] not in {1, 3}
            or item["id"] in identifiers
            or item["max_weight"] != (0.98 if item["top_k"] == 1 else 0.40)
            or item["target_volatility"] != (0.18 if item["top_k"] == 1 else 0.15)
        ):
            raise QuantError("Invalid or modified learned allocation candidate.")
        identifiers.add(item["id"])
        combinations.add((item["model"], item["top_k"]))
    if len(identifiers) != 4 or combinations != {
        (model, count) for model in MODEL_PARAMETERS for count in (1, 3)
    }:
        raise QuantError("Exactly four registered model/allocation combinations are required.")


def implementation_hash() -> str:
    return digest_json(
        {
            "learner": file_digest(Path(__file__)),
            "comparison": file_digest(Path(__file__).with_name("dual_horizon.py")),
            "accounting": file_digest(Path(__file__).with_name("bt_audit.py")),
            "sklearn_version": version("scikit-learn"),
        }
    )


def feature_rows(data: MarketData) -> pd.DataFrame:
    data.validate()
    if not {"SPY", "IEF", "TIP", "BIL"} <= set(data.close.columns):
        raise QuantError("Required market-regime features or benchmark cash ETF are missing.")
    close = data.close
    ends = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    monthly = close.loc[ends]
    daily_return = close.pct_change(fill_method=None)
    volatility21 = daily_return.rolling(21).std(ddof=1) * np.sqrt(252)
    volatility63 = daily_return.rolling(63).std(ddof=1) * np.sqrt(252)
    trend = close / close.rolling(200).mean() - 1
    drawdown = close / close.rolling(63).max() - 1
    month_return = {n: monthly / monthly.shift(n) - 1 for n in (1, 3, 6, 12)}
    rows = []
    for i, day in enumerate(ends):
        if i < 12:
            continue
        entry = next_session(day)
        exit_day = next_session(ends[i + 1]) if i + 1 < len(ends) else pd.NaT
        label_known = (
            not pd.isna(exit_day) and entry in data.open.index and exit_day in data.open.index
        )
        bill_return = (
            data.open.at[exit_day, "BIL"] / data.open.at[entry, "BIL"] - 1 if label_known else None
        )
        for symbol in close.columns:
            row = {
                "decision_date": day,
                "symbol": symbol,
                "entry_session": entry,
                "label_available_session": exit_day,
                **{f"return_{n}m": float(month_return[n].at[day, symbol]) for n in (1, 3, 6, 12)},
                "volatility_21d": float(volatility21.at[day, symbol]),
                "volatility_63d": float(volatility63.at[day, symbol]),
                "trend_200d": float(trend.at[day, symbol]),
                "drawdown_63d": float(drawdown.at[day, symbol]),
                "spy_return_3m": float(month_return[3].at[day, "SPY"]),
                "spy_volatility_63d": float(volatility63.at[day, "SPY"]),
                "ief_return_3m": float(month_return[3].at[day, "IEF"]),
                "tip_return_3m": float(month_return[3].at[day, "TIP"]),
                "next_month_excess": (
                    float(
                        data.open.at[exit_day, symbol] / data.open.at[entry, symbol]
                        - 1
                        - bill_return
                    )
                    if label_known
                    else np.nan
                ),
            }
            if not np.isfinite([row[key] for key in NUMERIC_FEATURES]).all():
                raise QuantError("A warmed-up learned feature is nonfinite; no silent imputation.")
            rows.append(row)
    if not rows:
        raise QuantError("No complete monthly feature window exists.")
    return pd.DataFrame(rows)


def training_rows(features: pd.DataFrame, decision: pd.Timestamp, policy: dict) -> pd.DataFrame:
    first = decision - pd.DateOffset(months=policy["training_months"])
    known = (
        (features["decision_date"] >= first)
        & (features["decision_date"] < decision)
        & (features["label_available_session"] <= decision)
        & features["next_month_excess"].notna()
    )
    train = features.loc[known].copy()
    if train.empty or not np.isfinite(train["next_month_excess"]).all():
        raise QuantError("No valid matured training labels exist at this decision time.")
    if not (
        (train["entry_session"] > train["decision_date"]).all()
        and (train["label_available_session"] > train["entry_session"]).all()
    ):
        raise QuantError("A training label is not a future open-to-open holding interval.")
    return train


def model_pipeline(name: str, policy: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if version("scikit-learn") != "1.7.2":
        raise QuantError("This study requires the registered scikit-learn 1.7.2 implementation.")
    if name == "ridge":
        return make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(),
            Ridge(**policy["model_parameters"][name]),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            DictVectorizer(sparse=False),
            HistGradientBoostingRegressor(**policy["model_parameters"][name]),
        )
    raise QuantError("Unknown learned model family.")


def walk_forward_forecasts(features: pd.DataFrame, policy: dict) -> tuple[dict, list[dict]]:
    from threadpoolctl import threadpool_limits

    forecasts = {model: {} for model in policy["model_parameters"]}
    audit = []
    if features.duplicated(["decision_date", "symbol"]).any():
        raise QuantError("Duplicate cross-sectional training samples.")
    for day in pd.DatetimeIndex(features["decision_date"].unique()).sort_values():
        available = (
            (features["decision_date"] < day)
            & (features["label_available_session"] <= day)
            & features["next_month_excess"].notna()
        )
        if features.loc[available, "decision_date"].nunique() < policy["minimum_training_months"]:
            continue
        train = training_rows(features, day, policy)
        if train["decision_date"].nunique() < policy["minimum_training_months"]:
            continue
        current = features.loc[features["decision_date"] == day]
        predictors = [*NUMERIC_FEATURES, "symbol"]
        x_train = train.loc[:, predictors].to_dict("records")
        x_next = current.loc[:, predictors].to_dict("records")
        labels = train["next_month_excess"].to_numpy()
        with threadpool_limits(limits=1):
            for name in policy["model_parameters"]:
                model = model_pipeline(name, policy)
                model.fit(x_train, labels)
                predictions = model.predict(x_next)
                if not np.isfinite(predictions).all():
                    raise QuantError("A learned forecast became nonfinite.")
                forecasts[name][day] = pd.Series(
                    predictions, index=current["symbol"].tolist(), name=day
                )
        audit.append(
            {
                "decision_date": day.date().isoformat(),
                "first_training_decision": train["decision_date"].min().date().isoformat(),
                "last_training_decision": train["decision_date"].max().date().isoformat(),
                "latest_label_available": train["label_available_session"].max().date().isoformat(),
                "unique_training_months": int(train["decision_date"].nunique()),
                "training_rows": len(train),
                "training_inputs_sha256": digest_json(x_train),
                "training_labels_sha256": digest_json(labels.tolist()),
                "current_features_sha256": digest_json(x_next),
                "hyperparameters_changed": False,
            }
        )
    if not audit:
        raise QuantError("Insufficient matured history for the first walk-forward fit.")
    frames = {
        name: pd.DataFrame.from_dict(values, orient="index").sort_index()
        for name, values in forecasts.items()
    }
    return frames, audit


def allocated_signals(
    data: MarketData, forecasts: pd.DataFrame, candidate: dict, policy: dict
) -> pd.DataFrame:
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    returns = data.close.pct_change(fill_method=None)
    for day in forecasts.index:
        if day not in data.close.index or not is_month_end(day):
            raise QuantError("Learned forecasts must be made on a completed month-end session.")
        prediction = forecasts.loc[day].reindex(data.close.columns)
        if not np.isfinite(prediction).all():
            raise QuantError("The forecast universe is incomplete.")
        eligible = [
            symbol
            for symbol in data.close.columns
            if symbol != "BIL" and prediction[symbol] > policy["minimum_predicted_excess_return"]
        ]
        eligible.sort(key=lambda symbol: (-float(prediction[symbol]), symbol))
        chosen = eligible[: candidate["top_k"]]
        weights = pd.Series(0.0, index=data.close.columns)
        if chosen:
            each = min((1 - policy["cash_reserve"]) / candidate["top_k"], candidate["max_weight"])
            weights.loc[chosen] = each
            trailing = returns.loc[:day, chosen].tail(policy["volatility_window"])
            covariance = trailing.cov().to_numpy() * 252
            exposure = weights.loc[chosen].to_numpy()
            variance = float(exposure @ covariance @ exposure)
            if not np.isfinite(variance) or variance < -1e-12:
                raise QuantError("Invalid forecast allocation covariance.")
            volatility = np.sqrt(max(variance, 0))
            if volatility > candidate["target_volatility"]:
                weights.loc[chosen] *= candidate["target_volatility"] / volatility
        weights["BIL"] = 1 - policy["cash_reserve"] - float(weights.sum())
        if (
            (weights < -1e-12).any()
            or weights.drop("BIL").max() > candidate["max_weight"] + 1e-10
            or not np.isclose(weights.sum(), 1 - policy["cash_reserve"], atol=1e-12)
        ):
            raise QuantError("Learned allocation violates long-only risk or cash limits.")
        signals.loc[day] = weights.clip(lower=0)
    return signals


def register(policy: dict, comparison: DualProtocol, source_hashes: dict, output: Path) -> dict:
    validate_policy(policy, comparison)
    if output.exists():
        raise QuantError("Refusing to overwrite learned-model preregistration.")
    record = {
        "registered_at": utc_now(),
        "policy": policy,
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "windows": comparison.windows(),
        "source_fingerprints": source_hashes,
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "prior_disclosed_trials": 42,
        "new_candidates": 4,
        "total_trials_after_round": 46,
        "hypotheses_registered_before_results": True,
        "researcher_history_exposure": (
            "Previously viewed prices; causal fitting does not make a research holdout untouched."
        ),
        "order_authority": False,
    }
    write_json(output, record)
    return record


def verify_registration(
    policy: dict, comparison: DualProtocol, source_hashes: dict, path: Path
) -> dict:
    validate_policy(policy, comparison)
    record = read_json(path)
    if (
        record.get("policy_sha256") != digest_json(policy)
        or record.get("comparison_sha256") != digest_json(asdict(comparison))
        or record.get("source_fingerprints") != source_hashes
        or record.get("candidate_ids") != [item["id"] for item in policy["candidates"]]
    ):
        raise QuantError("Learned-model parameters, evidence, or preregistration changed.")
    return record


def evaluate(
    policy: dict,
    comparison: DualProtocol,
    data: MarketData,
    registration: dict,
    output: Path,
) -> dict:
    if output.exists():
        raise QuantError("Refusing to overwrite learned-model results.")
    if data.close.index[-1].date().isoformat() != comparison.as_of:
        raise QuantError("Learned-model data does not reach the declared shared endpoint.")
    features = feature_rows(data)
    forecasts, fits = walk_forward_forecasts(features, policy)
    costs = replace(
        comparison,
        cost_bps_per_side=policy["cost_bps_per_side"],
        stress_cost_bps_per_side=policy["stress_cost_bps_per_side"],
    )
    benchmark = {}
    for window in comparison.windows():
        key = f"{window['years']}y"
        start, end = window["first_return_session"], window["last_session"]
        benchmark[key] = run_window(
            data, buy_and_hold_signals(data.close, "SPY", start), start, end, comparison
        )
    rows, checks, artifacts = {}, [], {}
    for candidate in policy["candidates"]:
        signals = allocated_signals(data, forecasts[candidate["model"]], candidate, policy)
        row = {"candidate": candidate, "windows": {}}
        for window in comparison.windows():
            key = f"{window['years']}y"
            start, end = window["first_return_session"], window["last_session"]
            seeded = seed_window(signals, start)
            row["windows"][key] = {}
            for stress in (False, True):
                label = "stress" if stress else "base"
                result = run_window(data, seeded, start, end, costs, stress=stress)
                independent = independent_equity(
                    data,
                    seeded,
                    start,
                    end,
                    capital=comparison.capital_usd,
                    cost_bps=costs.stress_cost_bps_per_side if stress else costs.cost_bps_per_side,
                    commission=costs.commission_per_order,
                    delay=1 + (costs.stress_additional_delay_sessions if stress else 0),
                )
                discrepancy = float(abs(result.frame["equity"] - independent["equity"]).max())
                if discrepancy > comparison.capital_usd * 1e-8:
                    raise QuantError("Learned allocation disagrees with independent bt accounting.")
                values = metrics(result.frame)
                conditions = gates(values, metrics(benchmark[key].frame), comparison)
                row["windows"][key][label] = {
                    "metrics": values,
                    "gates": conditions,
                    "all_numeric_gates_passed": all(conditions.values()),
                }
                if not stress:
                    row["windows"][key]["conditional_bootstrap"] = block_bootstrap(
                        result.frame["return"],
                        benchmark[key].frame["return"],
                        result.frame["risk_free"],
                        samples=1000,
                        block=21,
                        seed=20261006,
                    )
                checks.append(
                    {
                        "candidate": candidate["id"],
                        "horizon": key,
                        "scenario": label,
                        "max_equity_discrepancy_usd": discrepancy,
                    }
                )
                artifacts[f"{candidate['id']}/{key}-{label}.csv"] = result.frame.to_csv(
                    float_format="%.12g"
                )
                artifacts[f"{candidate['id']}/{key}-{label}-bt.csv"] = independent.to_csv(
                    float_format="%.12g"
                )
        first_signal = signals.dropna(how="all").index[0]
        first_trade = next_session(first_signal).date().isoformat()
        ongoing = run_window(data, signals, first_trade, comparison.as_of, costs)
        spy = run_window(
            data,
            buy_and_hold_signals(data.close, "SPY", first_trade),
            first_trade,
            comparison.as_of,
            comparison,
        )
        row["rolling_windows"] = {
            f"{years}y": rolling_diagnostics(ongoing.frame, spy.frame, years, comparison)
            for years in comparison.horizons_years
        }
        row["both_horizons_pass"] = all(
            window["base"]["all_numeric_gates_passed"] for window in row["windows"].values()
        )
        row["both_horizons_stress_pass"] = all(
            window["stress"]["all_numeric_gates_passed"] for window in row["windows"].values()
        )
        artifacts[f"{candidate['id']}/signals.csv"] = signals.dropna(how="all").to_csv(
            float_format="%.12g"
        )
        rows[candidate["id"]] = row
    result = {
        "created_at": utc_now(),
        "stage": "fixed_supervised_causal_walk_forward_dual_horizons",
        "implementation_sha256": implementation_hash(),
        "registration": registration,
        "policy": policy,
        "windows": comparison.windows(),
        "prior_trials": 42,
        "new_trials": 4,
        "global_trials": 46,
        "same_fixed_learning_policy_in_both_windows": True,
        "monthly_fits": fits,
        "model_fit_count": len(fits) * len(forecasts),
        "candidates": rows,
        "base_joint_passes": [key for key, row in rows.items() if row["both_horizons_pass"]],
        "stress_joint_passes": [
            key
            for key, row in rows.items()
            if row["both_horizons_pass"] and row["both_horizons_stress_pass"]
        ],
        "independent_bt_checks": checks,
        "order_authority": False,
        "investment_objective_verified": False,
        "limitations": [
            "Labels are time-purged, but researchers previously examined the evaluation prices.",
            "Monthly ETF rows are correlated observations, not independent samples.",
            "Hyperparameters were fixed before results; no random temporal split.",
            "QLD/SSO retain embedded product leverage despite no broker margin.",
            "A 63-session volatility forecast is not a drawdown guarantee.",
            "Adjusted fractional holdings and modeled fees are not actual broker fills.",
        ],
    }
    new_output_directory(output)
    for name, text in artifacts.items():
        write_text_atomic(output / name, text)
    for name, frame in forecasts.items():
        write_text_atomic(output / f"predictions-{name}.csv", frame.to_csv(float_format="%.12g"))
    write_text_atomic(
        output / "features-and-labels.csv", features.to_csv(index=False, float_format="%.12g")
    )
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "results.json", result)
    lines = [
        "# Fixed supervised allocation: causal model fitting",
        "",
        "**Neither a training score nor a historical backtest proves the investment objective.**",
        "",
        "| Learner | 10y CAGR | Sharpe | Drawdown | 5y CAGR | Sharpe | Drawdown | Both pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in rows.items():
        cells = []
        for key in ("10y", "5y"):
            value = row["windows"][key]["base"]["metrics"]
            sharpe = "undefined" if value["sharpe"] is None else f"{value['sharpe']:.2f}"
            cells.append(f"{value['cagr']:.2%} | {sharpe} | {value['max_drawdown']:.2%}")
        lines.append(f"| {name} | {' | '.join(cells)} | {row['both_horizons_pass']} |")
    lines.extend(["", *result["limitations"], ""])
    write_text_atomic(output / "report.md", "\n".join(lines))
    return result


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "learned-allocation", help="Fixed causal supervised research, no orders."
    )
    command.add_argument("stage", choices=["register", "evaluate"])
    command.add_argument("--policy", type=Path, default=Path("config/learned-allocation.json"))
    command.add_argument("--comparison", type=Path, default=Path("config/dual-horizon.json"))
    command.add_argument(
        "--registration", type=Path, default=Path("reports/learned-allocation/registration.json")
    )
    command.add_argument(
        "--legacy-registration",
        type=Path,
        default=Path("reports/dual-horizon/legacy-comparison-registration.json"),
    )
    command.add_argument("--data", type=Path, default=Path("data/dual-horizon/market"))
    command.add_argument(
        "--supplement", type=Path, default=Path("data/dual-horizon/legacy-supplement")
    )
    command.add_argument(
        "--output", type=Path, default=Path("reports/learned-allocation/evaluation")
    )


def dispatch_learned(args: argparse.Namespace) -> dict:
    comparison = load_dual(args.comparison)
    policy = read_json(args.policy)
    validate_policy(policy, comparison)
    record = verify_comparison(Path.cwd(), comparison, args.legacy_registration)
    hashes = {
        "market": file_digest(args.data / "manifest.json"),
        "supplement": file_digest(args.supplement / "manifest.json"),
        "prior_rule_comparison": file_digest(args.legacy_registration),
        "source_license": file_digest(Path("data/learned-allocation/scikit-learn-COPYING.txt")),
    }
    if args.stage == "register":
        result = register(policy, comparison, hashes, args.registration)
        return {
            key: result[key]
            for key in (
                "registered_at",
                "candidate_ids",
                "prior_disclosed_trials",
                "new_candidates",
                "total_trials_after_round",
                "order_authority",
            )
        }
    registration = verify_registration(policy, comparison, hashes, args.registration)
    data = combined_prices(comparison, args.data, args.supplement, record)
    result = evaluate(policy, comparison, data, registration, args.output)
    return {
        key: result[key]
        for key in (
            "stage",
            "global_trials",
            "model_fit_count",
            "base_joint_passes",
            "stress_joint_passes",
            "order_authority",
            "investment_objective_verified",
        )
    }
