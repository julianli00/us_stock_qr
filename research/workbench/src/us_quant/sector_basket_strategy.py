from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import sector_basket_sources as sources
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import dollar_comparisons, load_vintages
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import safe_file
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/sector-basket-strategy.json"
CANDIDATES = (
    {"id": "four_factor_diversified_sector_basket", "dollar_guard": False},
    {"id": "four_factor_diversified_sector_dollar_half", "dollar_guard": True},
)


def validate(policy: dict) -> None:
    constants = {
        "schema_version": 1, "study_id": "four_factor_diversified_sector_basket_20261011",
        "data_scope": "factor_etf_portfolio", "as_of": "2026-10-05", "data_start": "2015-08-10",
        "prior_evaluated_configurations": 122, "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "sector_symbols": list(sources.SECTORS),
        "momentum_sessions": 252, "skip_sessions": 21, "trend_sessions": 200,
        "volatility_sessions": 63, "top_k": 3, "standalone_sector_cap": 0.40,
        "satellite_share_of_equity_bucket": 0.50, "equity_bucket_min": 0.30, "equity_bucket_max": 0.70,
        "dollar_change_sessions": 63, "dollar_risk_off_scale": 0.50, "cash_reserve": 0.02,
        "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in ("source_policy", "source_readiness", "source_manifest")
        )
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Keep the fixed original-sector basket, both guards and research boundaries.")


def sector_selection(history: pd.DataFrame) -> tuple[pd.Series, dict]:
    if (
        len(history) < 253 or not set(sources.SECTORS).issubset(history)
        or not history.index.is_unique or not history.index.is_monotonic_increasing
        or not np.isfinite(history.to_numpy()).all() or (history <= 0).any().any()
    ):
        raise QuantError("Sector selection needs complete prior observed positive prices.")
    prices = history.loc[:, list(sources.SECTORS)]
    daily = prices.pct_change(fill_method=None).iloc[1:]
    momentum = np.expm1(np.log1p(daily.iloc[-252:-21]).sum())
    volatility = daily.tail(63).std(ddof=1)
    above = prices.iloc[-1] > prices.tail(200).mean()
    eligible = [
        symbol for symbol in sources.SECTORS
        if momentum[symbol] > 0 and above[symbol] and volatility[symbol] > 1e-10
    ]
    selected = sorted(eligible, key=lambda symbol: (-float(momentum[symbol]), symbol))[:3]
    bucket = pd.Series(0.0, index=(*sources.SECTORS, "BIL"))
    if selected:
        inverse = 1 / volatility.loc[selected]
        standalone = (0.98 * len(selected) / 3 * inverse / inverse.sum()).clip(upper=0.40)
        bucket.loc[selected] = standalone / 0.98
    remaining = float(1 - bucket.drop("BIL").sum())
    if -1e-12 <= remaining < 0:
        remaining = 0.0
    bucket["BIL"] = remaining
    if not np.isfinite(bucket).all() or (bucket < 0).any() or not np.isclose(bucket.sum(), 1, atol=1e-12):
        raise QuantError("Missing or capped sector slots cannot create invalid funded basket shares.")
    return bucket, {
        "selected_symbols": selected, "eligible_symbols": eligible,
        "sector_momentum": momentum.to_dict(), "sector_volatility": volatility.to_dict(),
        "sector_above_trend": above.to_dict(), "sector_bucket_shares": bucket.to_dict(),
    }


def build_from_inputs(
    data: MarketData, policy: dict, dollar: pd.Series,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if set(data.close.columns) != {"SPY", "IEF", "GLD", "BIL", *FACTORS, *sources.SECTORS}:
        raise QuantError("Keep the complete original four-style and nine-sector market.")
    decisions = [
        day for index, day in enumerate(data.close.index) if index >= 252 and is_month_end(day)
    ]
    if (
        not isinstance(dollar, pd.Series) or not dollar.index.is_unique
        or not all(day in dollar.index for day in decisions)
        or not np.isfinite(dollar.loc[decisions]).all()
    ):
        raise QuantError("Every basket decision requires its own complete historical dollar query.")
    frames = {
        candidate["id"]: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for candidate in CANDIDATES
    }
    diagnostics = []
    for day in decisions:
        history = data.close.loc[:day]
        basket, selected = sector_selection(history)
        composition = pd.Series(0.0, index=data.close.columns)
        composition.loc[list(FACTORS)] = 0.5 / 4
        composition.loc[list(sources.SECTORS)] = basket.loc[list(sources.SECTORS)] * 0.5
        remaining = basket["BIL"] * 0.5
        daily = history.pct_change(fill_method=None).iloc[1:].tail(63)
        equity_volatility = float((daily @ composition).std(ddof=1))
        gold_volatility = float(daily["GLD"].std(ddof=1))
        if (
            not np.isfinite([equity_volatility, gold_volatility]).all()
            or min(equity_volatility, gold_volatility) <= 1e-10
        ):
            raise QuantError("Actual style/sector and gold risk must be finite and positive.")
        equity_bucket = float(np.clip(gold_volatility / (equity_volatility + gold_volatility), 0.30, 0.70))
        target = composition * (0.98 * equity_bucket)
        target["GLD"] = 0.98 * (1 - equity_bucket)
        target["BIL"] = 0.98 * equity_bucket * remaining
        for candidate in CANDIDATES:
            actual = target.copy()
            risk_off = candidate["dollar_guard"] and dollar[day] > 0
            if risk_off:
                non_bil = actual.index != "BIL"
                freed = float(actual.loc[non_bil].sum()) * 0.5
                actual.loc[non_bil] *= 0.5
                actual["BIL"] += freed
            if (
                not np.isfinite(actual).all() or (actual < 0).any()
                or not np.isclose(actual.sum(), 0.98, rtol=0, atol=1e-12)
                or not actual.loc[list(FACTORS)].gt(0).all()
            ):
                raise QuantError("Sector composition or dollar guard violated original cash funding.")
            frames[candidate["id"]].loc[day] = actual
            diagnostics.append({
                "candidate_id": candidate["id"], "decision_session": day.strftime("%Y-%m-%d"),
                "equity_bucket_share": equity_bucket,
                "equity_composition_daily_volatility": equity_volatility,
                "gold_daily_volatility": gold_volatility,
                "missing_sector_cash_share_inside_bucket": remaining,
                "known_dollar_change": float(dollar[day]), "dollar_half_risk_applied": bool(risk_off),
                "actual_stock_target_weight": float(actual.loc[[*FACTORS, *sources.SECTORS]].sum()),
                **selected,
            })
    if not decisions:
        raise QuantError("No complete original sector-selection warmup is available.")
    return frames, diagnostics


def dollar_inputs(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    policy = read_json(ROOT / "config/dollar-risk-guard.json")
    vintages, _ = load_vintages(index, policy)
    return dollar_comparisons(vintages, index)


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    dollar, _ = dollar_inputs(data.close.index)
    return build_from_inputs(data, policy, dollar)[0]


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate(policy)
    for key in ("source_policy", "source_readiness", "source_manifest"):
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    source_policy = read_json(ROOT / policy["source_policy"]["path"])
    data = sources.verified_market(source_policy)
    dollar, provenance = dollar_inputs(data.close.index)
    frames, diagnostics = build_from_inputs(data, policy, dollar)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private sector-basket directory without overwriting evidence.")
    new_output_directory(output)
    write_json(output / "target-diagnostics.json", {"diagnostics": diagnostics})
    write_text_atomic(output / "dollar-provenance.csv", provenance.to_csv(float_format="%.17g"))
    panels = {}
    for name, frame in (
        ("open", data.open), ("close", data.close), ("raw_close", data.raw_close),
        ("volume", data.volume), ("risk_free", data.risk_free.to_frame()),
    ):
        path = output / f"market-{name}.csv"
        write_text_atomic(path, frame.to_csv(float_format="%.17g"))
        panels[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    core = read_json(ROOT / source_policy["core_registration"]["path"])
    readiness = deepcopy(core["readiness"])
    readiness.update({
        "checked_at": utc_now(), "source_check": policy["source_readiness"],
        "verified_etf_source": {
            "adapter": "sector_basket_sources_20261011",
            "policy": policy["source_policy"]["path"], "policy_sha256": policy["source_policy"]["sha256"],
            "factor_manifest": policy["source_manifest"]["path"], "factor_manifest_sha256": policy["source_manifest"]["sha256"],
        },
        "revalidation_not_new_market_observation": True,
    })
    write_json(output / "readiness.json", readiness)
    dependencies = (
        "sector_basket_strategy.py", "sector_basket_sources.py", "sector_growth_balance.py",
        "factor_research.py", "multifactor_stability.py", "dual_horizon.py",
        "dollar_risk_guard.py", "macro_factor_tilt.py", "growth_factor_satellite.py",
    )
    frozen = {
        f"src/us_quant/{name}": file_digest(ROOT / f"src/us_quant/{name}")
        for name in dependencies
    }
    frozen[POLICY.relative_to(ROOT).as_posix()] = file_digest(POLICY)
    for key in ("source_policy", "source_manifest", "source_readiness"):
        frozen[policy[key]["path"]] = policy[key]["sha256"]
    for key in ("original_policy", "original_registration", "core_registration"):
        frozen[source_policy[key]["path"]] = source_policy[key]["sha256"]
    for name in (
        "config/dollar-risk-guard.json", "config/dual-horizon.json",
        "config/sector-growth-balance.json", "config/growth-factor-satellite.json",
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        frozen[name] = file_digest(ROOT / name)
    specs = []
    for candidate in CANDIDATES:
        spec = {
            "id": candidate["id"], "configuration": candidate, "factor_ids": policy["factor_ids"],
            "data_scope": "factor_etf_portfolio", "market": panels,
            "evaluation_as_of": policy["as_of"], "frozen_files": frozen,
            "asset_leverage": {symbol: 1.0 for symbol in data.close.columns},
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
        "defined_month_ends_per_candidate": len(next(iter(frames.values())).dropna(how="all")),
        "candidate_returns_or_metrics_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare two fixed diversified-sector factor portfolios.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/sector-basket-strategy-20261011-v3")
    args = parser.parse_args()
    try:
        result = prepare(args.prepared)
        print(json.dumps({
            "prepared_configurations": len(result["specs"]),
            "defined_month_ends_per_candidate": result["defined_month_ends_per_candidate"],
            "candidate_returns_or_metrics_computed": False, "investment_objective_verified": False,
        }, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Sector-basket preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
