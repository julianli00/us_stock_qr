from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import option_sector_implementation as market_source
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import dollar_comparisons, load_vintages
from us_quant.growth_factor_satellite import composed_monthly_target
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import safe_file
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/treasury-defense-implementation.json"
ROUND_ID = "user_confirmed_etf_restructure_20261013"
CANDIDATES = (
    {"id": "bounded_treasury_defense_ief", "defensive_symbol": "IEF", "optimization_round_id": ROUND_ID},
    {"id": "bounded_treasury_defense_tlt", "defensive_symbol": "TLT", "optimization_round_id": ROUND_ID},
)
REFERENCES = ("market_policy", "market_implementation", "source_registration", "dollar_policy")


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1, "study_id": "bounded_realized_treasury_defense_20261014_batch3",
        "optimization_round_id": ROUND_ID, "approved_round_configuration_cap": 6,
        "approved_round_batch_cap": 3, "batch_number": 3, "data_scope": "factor_etf_portfolio",
        "data_start": "2015-08-10", "as_of": "2026-10-05", "prior_evaluated_configurations": 176,
        "new_configurations": 2, "new_economic_factor_definitions": 0,
        "factor_ids": ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols": list(FACTORS), "source_id": "semiconductor_option_proxy_dollar_observed_band025",
        "volatility_sessions": 63, "equity_share_min": 0.30, "equity_share_max": 0.70,
        "cash_reserve": 0.02, "actual_weight_trade_band": 0.025, "candidates": list(CANDIDATES),
    }
    if (
        not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False
    ):
        raise QuantError("Preserve the final two fixed Treasury packages,original risk bounds and six-slot cap.")


def references(policy: dict) -> dict:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    source_policy = read_json(ROOT / policy["market_policy"]["path"])
    if source_policy["source_registration"] != policy["source_registration"]:
        raise QuantError("The final package must retain the exact original cached market registration.")
    return source_policy


def monthly_index(data: MarketData) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([day for i, day in enumerate(data.close.index) if i >= 63 and is_month_end(day)])


def build_from_inputs(data: MarketData, dollar: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    months = monthly_index(data)
    if (
        len(data.close) != 2805 or len(months) != 131
        or set(data.close.columns) != {"SPY", "IEF", "TLT", "GLD", "BIL", *FACTORS, "QQQ", "XLK", "SOXX"}
        or not isinstance(dollar, pd.Series) or not dollar.index.equals(months)
        or not pd.api.types.is_numeric_dtype(dollar) or pd.api.types.is_bool_dtype(dollar)
        or pd.api.types.is_complex_dtype(dollar) or not np.isfinite(dollar.to_numpy()).all()
    ):
        raise QuantError("Require every original market row and monthly own-vintage dollar comparison.")
    result = {}
    for candidate in CANDIDATES:
        symbol = candidate["defensive_symbol"]
        targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
        for day in months:
            # Role aliases are local inputs to the unchanged risk helper, not price substitutions.
            history = data.close.loc[:day].drop(columns=["QQQ", "GLD"]).rename(
                columns={"SOXX": "QQQ", symbol: "GLD"},
            )
            target = composed_monthly_target(history, {"growth_share_of_equity": 0.5})
            target = target.rename(index={"QQQ": "SOXX", "GLD": symbol}).reindex(data.close.columns, fill_value=0.0)
            if dollar.loc[day] > 0:
                non_bil = target.index.drop("BIL")
                freed = float(target.loc[non_bil].sum()) * 0.5
                target.loc[non_bil] *= 0.5
                target["BIL"] += freed
            if (target < 0).any() or not np.isfinite(target).all() or not np.isclose(target.sum(), .98, rtol=0, atol=1e-12):
                raise QuantError("Treasury package violates the unchanged long-only funded budget.")
            targets.loc[day] = target
        result[candidate["id"]] = targets
    return result


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    references(policy)
    vintages, _ = load_vintages(data.close.index, read_json(ROOT / policy["dollar_policy"]["path"]))
    dollar, _ = dollar_comparisons(vintages, data.close.index)
    return build_from_inputs(data, dollar, policy)


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use only the two final predefined Treasury defensive packages.")
    source = read_json(ROOT / policy["market_policy"]["path"])
    return market_source.parent.run(data, desired, market_source.parent.CANDIDATES[1],
                                    read_json(ROOT / source["source_policy"]["path"]), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    source_policy = references(policy)
    entry = market_source.source_entry(source_policy)
    data = market_source.verified_market(source_policy)
    targets = source_baselines(data, policy)
    frozen = {
        **entry["spec"]["frozen_files"],
        **{policy[key]["path"]: policy[key]["sha256"] for key in REFERENCES},
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
    }
    for name, expected in frozen.items():
        safe_file(ROOT, name, expected)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new specific private final-batch directory without account outcomes.")
    new_output_directory(output)
    ref = entry["readiness"]
    readiness = deepcopy(read_json(safe_file(ROOT, ref["path"], ref["sha256"])))
    readiness.update(checked_at=utc_now(), revalidation_not_new_market_observation=True)
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**entry["spec"], "id": candidate["id"], "configuration": candidate, "frozen_files": frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          targets[candidate["id"]].to_csv(float_format="%.17g"))
    return {"monthly_requests_each": 131, "unchanged_source_assets": 12,
            "candidate_returns_or_metrics_computed": False, "new_source_requests": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final two Treasury packages with no strategy accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/treasury-defense-implementation-20261014")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Treasury package preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
