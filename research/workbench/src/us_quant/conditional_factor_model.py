from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from us_quant.adaptive_factor_allocation import prepare as prepare_market
from us_quant.calendar import is_month_end, next_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.learned_allocation import model_pipeline, training_rows
from us_quant.macro_factor_tilt import load_macro
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import digest_json, file_digest, read_json, write_json
from us_quant.volatility_term_risk import load_terms

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/conditional-factor-model.json"
MODEL_ASSETS = (*FACTORS, "GLD")
PRICE_FEATURES = (
    "return21",
    "return63",
    "volatility63",
    "drawdown63",
    "spy_return63",
    "spy_volatility63",
)
EXTERNAL_FEATURES = ("term_spread", "real_yield", "real_yield_change63", "vix", "vix_term_ratio")


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("as_of") != "2026-10-05"
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("training_months") != 36
        or policy.get("minimum_training_months") != 9
        or policy.get("model_parameters") != {"ridge": {"alpha": 10.0, "solver": "svd"}}
        or policy.get("clip_current_features_to_training_range") is not True
        or policy.get("minimum_predicted_excess_return") != 0.005
        or policy.get("ranked_factor_shares") != [0.35, 0.30, 0.20, 0.15]
        or policy.get("maximum_single_risk_sleeve") != 0.70
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates")
        != [
            {"id": "conditional_price_factor_model", "augmented_information": False},
            {"id": "conditional_macro_option_factor_model", "augmented_information": True},
        ]
        or policy.get("methodology", {}).get("order_authority") is not False
    ):
        raise QuantError("The conditional forecasting protocol was altered.")


def feature_rows(data: MarketData, macro: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    data.validate()
    if (
        not macro.index.equals(data.close.index)
        or not terms.index.equals(data.close.index)
        or set(macro.columns) != {"T10Y3M", "DFII10"}
        or set(terms.columns) != {"VIX", "VIX3M"}
        or not np.isfinite(macro.to_numpy()).all()
        or not np.isfinite(terms.to_numpy()).all()
        or (terms <= 0).any().any()
    ):
        raise QuantError("External observations must match all price sessions.")
    close = data.close
    ends = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    returns = close.pct_change(fill_method=None)
    rows = []
    for i, day in enumerate(ends):
        position = close.index.get_loc(day)
        if position < 63:
            continue
        history = close.loc[:day]
        recent = returns.loc[:day].tail(63)
        entry = next_session(day)
        exit_day = next_session(ends[i + 1]) if i + 1 < len(ends) else pd.NaT
        known = not pd.isna(exit_day) and entry in data.open.index and exit_day in data.open.index
        bill = data.open.loc[exit_day, "BIL"] / data.open.loc[entry, "BIL"] - 1 if known else None
        for symbol in MODEL_ASSETS:
            row = {
                "decision_date": day,
                "symbol": symbol,
                "entry_session": entry,
                "label_available_session": exit_day,
                "return21": float(history[symbol].iloc[-1] / history[symbol].iloc[-22] - 1),
                "return63": float(history[symbol].iloc[-1] / history[symbol].iloc[-64] - 1),
                "volatility63": float(recent[symbol].std(ddof=1) * np.sqrt(252)),
                "drawdown63": float(history[symbol].iloc[-1] / history[symbol].tail(63).max() - 1),
                "spy_return63": float(history["SPY"].iloc[-1] / history["SPY"].iloc[-64] - 1),
                "spy_volatility63": float(recent["SPY"].std(ddof=1) * np.sqrt(252)),
                "term_spread": float(macro.loc[day, "T10Y3M"]),
                "real_yield": float(macro.loc[day, "DFII10"]),
                "real_yield_change63": float(
                    macro.loc[day, "DFII10"] - macro["DFII10"].iloc[position - 63]
                ),
                "vix": float(terms.loc[day, "VIX"]),
                "vix_term_ratio": float(terms.loc[day, "VIX"] / terms.loc[day, "VIX3M"]),
                "next_month_excess": (
                    float(data.open.loc[exit_day, symbol] / data.open.loc[entry, symbol] - 1 - bill)
                    if known
                    else np.nan
                ),
            }
            if not np.isfinite([row[key] for key in (*PRICE_FEATURES, *EXTERNAL_FEATURES)]).all():
                raise QuantError("A warmed-up predictor is missing; no silent feature imputation.")
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["decision_date", "symbol"]).any():
        raise QuantError("Conditional model feature rows are absent or duplicated.")
    return result


def forecasts(
    features: pd.DataFrame, policy: dict, augmented: bool
) -> tuple[pd.DataFrame, list[dict]]:
    validate_policy(policy)
    numeric = [*PRICE_FEATURES, *(EXTERNAL_FEATURES if augmented else ())]
    predictors = [*numeric, "symbol"]
    records, audit = {}, []
    for day in pd.DatetimeIndex(features["decision_date"].unique()).sort_values():
        available = (
            (features["decision_date"] < day)
            & (features["label_available_session"] <= day)
            & features["next_month_excess"].notna()
        )
        if features.loc[available, "decision_date"].nunique() < policy["minimum_training_months"]:
            continue
        training = training_rows(features, day, policy)
        if training["decision_date"].nunique() < policy["minimum_training_months"]:
            continue
        current = features.loc[features["decision_date"] == day].copy()
        low, high = training[numeric].min(), training[numeric].max()
        clipped = current[numeric].clip(lower=low, upper=high, axis=1)
        clipping_count = int((clipped != current[numeric]).sum().sum())
        current[numeric] = clipped
        x_train = training[predictors].to_dict("records")
        x_now = current[predictors].to_dict("records")
        y = training["next_month_excess"].to_numpy()
        with threadpool_limits(limits=1):
            model = model_pipeline("ridge", policy)
            model.fit(x_train, y)
            prediction = model.predict(x_now)
        if not np.isfinite(prediction).all():
            raise QuantError("A conditional prediction is nonfinite.")
        records[day] = pd.Series(prediction, index=current["symbol"].tolist())
        audit.append(
            {
                "decision_date": str(day.date()),
                "latest_label_available": str(training["label_available_session"].max().date()),
                "first_training_decision": str(training["decision_date"].min().date()),
                "last_training_decision": str(training["decision_date"].max().date()),
                "unique_training_months": int(training["decision_date"].nunique()),
                "training_rows": len(training),
                "clipped_current_feature_values": clipping_count,
                "training_inputs_sha256": digest_json(x_train),
                "training_labels_sha256": digest_json(y.tolist()),
                "current_features_sha256": digest_json(x_now),
                "hyperparameters_changed": False,
            }
        )
    if not records:
        raise QuantError("No model has enough genuinely matured labels.")
    return pd.DataFrame.from_dict(records, orient="index").sort_index(), audit


def allocation(
    data: MarketData, day: pd.Timestamp, prediction: pd.Series, policy: dict
) -> pd.Series:
    if not set(MODEL_ASSETS) <= set(prediction.index) or not np.isfinite(prediction).all():
        raise QuantError("All factor and gold forecasts must be present.")
    order = sorted(FACTORS, key=lambda symbol: (-float(prediction[symbol]), symbol))
    shares = pd.Series(policy["ranked_factor_shares"], index=order)
    equity_prediction = float(prediction.loc[order] @ shares)
    equity_ready = equity_prediction > policy["minimum_predicted_excess_return"]
    gold_ready = prediction["GLD"] > policy["minimum_predicted_excess_return"]
    equity, gold = 0.0, 0.0
    if equity_ready and gold_ready:
        recent = data.close.loc[:day].pct_change(fill_method=None).iloc[1:].tail(63)
        equity_vol = float((recent[order] @ shares).std(ddof=1))
        gold_vol = float(recent["GLD"].std(ddof=1))
        if not np.isfinite([equity_vol, gold_vol]).all() or min(equity_vol, gold_vol) <= 1e-10:
            raise QuantError(
                "Forecast-based risk allocation lacks an observed volatility estimate."
            )
        equity = float(np.clip(gold_vol / (gold_vol + equity_vol), 0.30, 0.70))
        gold = 1 - equity
    elif equity_ready:
        equity = 0.70
    elif gold_ready:
        gold = 0.70
    weights = pd.Series(0.0, index=data.close.columns)
    weights.loc[order] = 0.98 * equity * shares
    weights["GLD"] = 0.98 * gold
    weights["BIL"] = 0.98 - weights.sum()
    return weights


def build_from_inputs(
    data: MarketData, macro: pd.DataFrame, terms: pd.DataFrame, policy: dict
) -> tuple[dict[str, pd.DataFrame], dict]:
    validate_policy(policy)
    features = feature_rows(data, macro, terms)
    output, audit = {}, {}
    for candidate in policy["candidates"]:
        predicted, checks = forecasts(features, policy, candidate["augmented_information"])
        frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for day in predicted.index:
            frame.loc[day] = allocation(data, day, predicted.loc[day], policy)
        output[candidate["id"]] = frame
        audit[candidate["id"]] = checks
    return output, audit


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    macro, _ = load_macro(ROOT / "data/macro-rate-access-20261010", data.close.index)
    terms = load_terms(ROOT / "data/volatility-term-source-20261010", data.close.index)
    targets, audit = build_from_inputs(data, macro, terms, policy)
    if any(
        pd.Timestamp(rows[0]["decision_date"]) > pd.Timestamp("2016-09-30")
        for rows in audit.values()
    ):
        raise QuantError("The first formal evaluation lacks a trained pre-window model.")
    return targets


def prepare(output: Path):
    policy = read_json(POLICY)
    validate_policy(policy)
    original = prepare_market(output)
    frozen = {
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        "src/us_quant/learned_allocation.py": file_digest(
            ROOT / "src/us_quant/learned_allocation.py"
        ),
        "src/us_quant/macro_factor_tilt.py": file_digest(
            ROOT / "src/us_quant/macro_factor_tilt.py"
        ),
        "src/us_quant/volatility_term_risk.py": file_digest(
            ROOT / "src/us_quant/volatility_term_risk.py"
        ),
    }
    for manifest_path, kind in (
        (ROOT / "data/macro-rate-access-20261010/verified-manifest.json", "macro"),
        (ROOT / "data/volatility-term-source-20261010/manifest.json", "options"),
    ):
        frozen[manifest_path.relative_to(ROOT).as_posix()] = file_digest(manifest_path)
        for row in read_json(manifest_path)["sources"]:
            path = manifest_path.parent / (
                row["path"] if kind == "macro" else f"{row['symbol']}.csv"
            )
            frozen[path.relative_to(ROOT).as_posix()] = file_digest(path)
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            "id": candidate["id"],
            "configuration": candidate,
            "data_scope": "factor_etf_portfolio",
            "factor_ids": policy["factor_ids"],
            "evaluation_as_of": policy["as_of"],
            "market": original["specs"][0]["market"],
            "asset_leverage": original["specs"][0]["asset_leverage"],
            "frozen_files": frozen,
            "history_status": "exposed_history_not_independent_holdout",
            "leveraged_products_allowed": False,
            "order_authority": False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    write_json(output / "readiness.json", original["readiness"])
    return {"specs": specs, "readiness": original["readiness"], "returns_computed": False}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fixed causal conditional-factor predictors."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/conditional-factor-model-20261011"
    )
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {"prepared": [row["id"] for row in result["specs"]], "returns_computed": False}
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Conditional-factor preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
