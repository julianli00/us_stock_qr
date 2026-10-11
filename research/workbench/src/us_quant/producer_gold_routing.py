from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import breadth_gold_relative as gold
from us_quant import observed_weight_execution as execution
from us_quant import producer_sector_sources as sources
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/producer-gold-routing.json"
PRODUCERS = sources.ADDITIONAL
CANDIDATES = (
    {"id": "semiconductor_gold_released_producer_momentum", "producer_method": "highest_positive_excess"},
    {"id": "semiconductor_gold_released_producer_diversified", "producer_method": "all_positive_inverse_volatility"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "producer_policy",
              "producer_implementation", "producer_readiness", "market_source",
              "prior_selector_policy", "prior_selector_implementation", "prior_cap_policy")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"graded_gold_released_budget_producer_equity_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":148, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_id":"semiconductor_graded_gold_half_observed_band025",
        "unbanded_source_id":"semiconductor_breadth_half_gold_relative", "producer_assets":list(PRODUCERS),
        "momentum_sessions":126, "volatility_sessions":63, "standalone_producer_cap":0.40,
        "cash_origin":"only_original_gold_trend_removed_budget_not_equity_standby_cash",
        "cash_reserve":0.02, "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Freeze both producer selectors,original source,gold-only budget,cap and execution contract.")


def parent_entry(policy: dict, root: Path = ROOT) -> dict:
    validate(policy)
    for key in REFERENCES:
        safe_file(root, policy[key]["path"], policy[key]["sha256"])
    registration = read_json(root / policy["source_registration"]["path"])
    selected = [row for row in registration["candidates"] if row["spec"]["id"] == policy["source_id"]]
    if len(selected) != 1:
        raise QuantError("Require the exact original half-breadth graded-gold observed-band source.")
    for name, digest in selected[0]["spec"]["frozen_files"].items():
        safe_file(root, name, digest)
    return selected[0]


def verified_market(policy: dict, root: Path = ROOT) -> MarketData:
    entry = parent_entry(policy, root)
    producer_policy = read_json(root / policy["producer_policy"]["path"])
    canonical, context = sources.verified_market(producer_policy)
    manifest = read_json(root / policy["market_source"]["path"])
    proof = read_json(root / policy["producer_readiness"]["path"])
    if (manifest.get("source_policy_sha256") != policy["producer_policy"]["sha256"]
        or manifest.get("source_validator_sha256") != sources.fingerprint()
        or manifest.get("prepared_assets") != 14 or manifest.get("complete_price_sessions_per_asset") != 2805
        or manifest.get("original_assets") != list(sources.ORIGINAL)
        or manifest.get("additional_assets") != list(PRODUCERS) or manifest.get("context") != context
        or context["original_market"] != entry["spec"]["market"]
        or proof.get("prepared_market") != manifest["market"]
        or proof.get("source_policy") != policy["producer_policy"]
        or proof.get("ready_for_separate_rule_preparation") is not True
        or proof.get("portfolio_targets_or_returns_computed") is not False):
        raise QuantError("Require the exact independently verified producer-source preparation and original portfolio.")
    for name, ref in manifest["market"].items():
        path = safe_file(root, ref["path"], ref["sha256"])
        actual = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        expected = getattr(canonical, name)
        expected = expected.to_frame() if isinstance(expected, pd.Series) else expected
        if (not actual.index.equals(expected.index) or not actual.columns.equals(expected.columns)
            or not np.array_equal(actual.to_numpy(), expected.to_numpy())):
            raise QuantError("Lossless source exports must preserve every canonical original and additional observation.")
    # Preserve the existing account parser after separately proving the lossless exports.
    return review_market({"market":manifest["market"]}, root)


def producer_shares(history: pd.DataFrame, method: str) -> tuple[pd.Series, dict]:
    if (not isinstance(history, pd.DataFrame) or len(history) < 127
        or not {*PRODUCERS, "BIL"}.issubset(history) or not history.index.is_unique
        or not history.index.is_monotonic_increasing or not history.columns.is_unique
        or not np.isfinite(history.to_numpy()).all() or (history <= 0).any().any()
        or method not in {row["producer_method"] for row in CANDIDATES}):
        raise QuantError("Producer selection requires complete prior actual prices and one frozen126/63method.")
    change = history.iloc[-1] / history.iloc[-127] - 1
    eligible = [name for name in PRODUCERS if change[name] > change["BIL"]]
    result = pd.Series(0.0, index=(*PRODUCERS, "BIL"))
    if not eligible:
        result["BIL"] = 1.0
    elif method == "highest_positive_excess":
        selected = sorted(eligible, key=lambda name: (-float(change[name]), name))[0]
        result[selected] = 1.0
    else:
        volatility = history.loc[:, eligible].pct_change(fill_method=None).iloc[1:].tail(63).std(ddof=1)
        if not np.isfinite(volatility).all() or (volatility <= 1e-10).any():
            raise QuantError("Eligible producer volatility is degenerate;do not substitute default shares.")
        inverse = 1 / volatility
        result.loc[eligible] = inverse / inverse.sum()
    if (not np.isfinite(result).all() or (result < 0).any()
        or not np.isclose(result.sum(), 1.0, rtol=0, atol=1e-12)):
        raise QuantError("Producer shares must remain finite,long-only and fully allocated.")
    return result, {"eligible_producers":eligible,
                    "producer126sessionreturns":{name:float(change[name]) for name in PRODUCERS},
                    "bil126sessionreturn":float(change["BIL"]),
                    "selected_producer_shares":{name:float(value) for name, value in result.items()}}


def route_gold_release(source: pd.Series, released: float, shares: pd.Series) -> pd.Series:
    if (not isinstance(source, pd.Series) or list(source.index) != [*sources.ORIGINAL, *PRODUCERS]
        or not np.isfinite(source).all() or (source < 0).any()
        or not np.isclose(source.sum(), 0.98, rtol=0, atol=1e-12)
        or not source.loc[list(PRODUCERS)].eq(0).all()
        or not np.isfinite(released) or not 0 <= released <= source["BIL"]
        or not isinstance(shares, pd.Series) or list(shares.index) != [*PRODUCERS, "BIL"]
        or not np.isfinite(shares).all() or (shares < 0).any()
        or not np.isclose(shares.sum(), 1.0, rtol=0, atol=1e-12)):
        raise QuantError("Route only the independently identified gold-released part of complete original BIL targets.")
    result = source.copy()
    amount = min(float(released), 0.40)
    result.loc[list(PRODUCERS)] = amount * shares.loc[list(PRODUCERS)]
    result["BIL"] -= amount * (1 - float(shares["BIL"]))
    unchanged = source.index.difference([*PRODUCERS, "BIL"])
    if (not result.loc[unchanged].equals(source.loc[unchanged]) or (result < 0).any()
        or not np.isclose(result.sum(), 0.98, rtol=0, atol=1e-12)
        or result.loc[list(PRODUCERS)].sum() > min(released, 0.40) + 1e-12
        or result["BIL"] < source["BIL"] - released - 1e-12):
        raise QuantError("Producer routing changed another budget,stock standby cash or the funded total.")
    return result


def build_from_source(
    data: MarketData, original: pd.DataFrame, released: pd.Series, policy: dict,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    validate(policy)
    data.validate()
    if (tuple(data.close.columns) != (*sources.ORIGINAL, *PRODUCERS)
        or not isinstance(original, pd.DataFrame) or list(original.columns) != list(sources.ORIGINAL)
        or not original.index.equals(data.close.index) or not isinstance(released, pd.Series)
        or not released.index.equals(original.index)
        or (original.isna().any(axis=1) & ~original.isna().all(axis=1)).any()):
        raise QuantError("Preserve all original decision dates and the exact twelve-plus-two source columns.")
    active = original.dropna(how="all")
    if (active.empty or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all() or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not np.isfinite(released.loc[active.index]).all()
        or (released.loc[active.index] < 0).any()
        or not released.loc[original.isna().all(axis=1)].isna().all()):
        raise QuantError("Gold-released budgets must be explicit on original fully informed monthly targets only.")
    frames = {row["id"]:pd.DataFrame(np.nan, index=original.index, columns=data.close.columns) for row in CANDIDATES}
    diagnostics = []
    for candidate in CANDIDATES:
        for day in active.index:
            source = pd.Series(0.0, index=data.close.columns)
            source.loc[original.columns] = original.loc[day]
            shares, audit = producer_shares(data.close.loc[:day], candidate["producer_method"])
            target = route_gold_release(source, float(released.loc[day]), shares)
            frames[candidate["id"]].loc[day] = target
            diagnostics.append({"candidate_id":candidate["id"], "session":str(day.date()),
                                "gold_released_weight":float(released.loc[day]),
                                "source_bil_weight":float(source["BIL"]),
                                "routed_producer_weight":float(target.loc[list(PRODUCERS)].sum()), **audit})
    return frames, diagnostics


def inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.Series]:
    entry = parent_entry(policy)
    original_data = review_market({"market":entry["spec"]["market"]}, ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        expected, actual = getattr(original_data, name), getattr(data, name)
        if isinstance(actual, pd.DataFrame):
            actual = actual.loc[:,list(sources.ORIGINAL)]
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not expected.index.equals(actual.index)
            or not np.allclose(expected, actual, rtol=relative, atol=absolute)):
            raise QuantError("Do not substitute any original portfolio observation while adding producer prices.")
    execution_policy = read_json(ROOT / policy["source_policy"]["path"])
    original = execution.source_baselines(original_data, execution_policy)[policy["source_id"]]
    gold_policy = read_json(ROOT / execution_policy["source_policy"]["path"])
    _, reference = gold.inputs(original_data, gold_policy)
    released = (reference["original_gold"] - original["GLD"]).where(original["GLD"].notna())
    return original, released


def source_baselines(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    original, released = inputs(data, policy)
    return build_from_source(data, original, released, policy)[0]


def run(
    data: MarketData, desired: pd.DataFrame, candidate: dict, policy: dict,
    start: str, end: str, cost: float, delay: int,
) -> tuple[BacktestResult, pd.DataFrame, list[dict]]:
    validate(policy)
    if candidate not in CANDIDATES:
        raise QuantError("Use an exact fixed producer-routing configuration.")
    for key in ("source_policy", "source_implementation"):
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])
    return execution.run(data, desired, execution.CANDIDATES[0],
                         read_json(ROOT / policy["source_policy"]["path"]), start, end, cost, delay)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    data = verified_market(policy)
    entry = parent_entry(policy)
    original, released = inputs(data, policy)
    frames, diagnostics = build_from_source(data, original, released, policy)
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Prepare a new private producer-rule directory without rewriting any source or account.")
    new_output_directory(output)
    write_text_atomic(output / "original-source-targets.csv", original.to_csv(float_format="%.17g"))
    write_text_atomic(output / "gold-released-budget.csv", released.to_csv(float_format="%.17g"))
    write_json(output / "producer-diagnostics.json", {"diagnostics":diagnostics})
    producer_policy = read_json(ROOT / policy["producer_policy"]["path"])
    source_manifest = read_json(ROOT / policy["market_source"]["path"])
    sector_manifest = read_json(ROOT / producer_policy["sector_manifest"]["path"])
    readiness_ref = entry["readiness"]
    readiness = deepcopy(read_json(safe_file(ROOT, readiness_ref["path"], readiness_ref["sha256"])))
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    readiness["verified_etf_source"] = {
        "adapter":"producer_gold_routing_20261011",
        "policy":POLICY.relative_to(ROOT).as_posix(), "policy_sha256":file_digest(POLICY),
        "producer_manifest":policy["market_source"]["path"],
        "producer_manifest_sha256":policy["market_source"]["sha256"],
    }
    for capability in ("actual_fund_history", "post_inception_and_actions", "unleveraged_fund_identity"):
        readiness["capabilities"][capability] = {
            "verified":True, "evidence_path":policy["producer_readiness"]["path"],
            "evidence_sha256":policy["producer_readiness"]["sha256"],
        }
    write_json(output / "readiness.json", readiness)
    frozen = dict(entry["spec"]["frozen_files"])
    additions = {
        **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES},
        **{producer_policy[key]["path"]:producer_policy[key]["sha256"] for key in sources.REFERENCES},
        **sector_manifest["file_sha256"],
        **{ref["path"]:ref["sha256"] for ref in source_manifest["market"].values()},
        POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        "src/us_quant/producer_gold_routing.py":file_digest(Path(__file__)),
    }
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not replace conflicting original frozen source or helper versions.")
        frozen[name] = digest
    for candidate in CANDIDATES:
        spec = {
            "id":candidate["id"], "configuration":candidate, "factor_ids":policy["factor_ids"],
            "data_scope":"factor_etf_portfolio", "evaluation_as_of":policy["as_of"],
            "market":source_manifest["market"],
            "asset_leverage":{**entry["spec"]["asset_leverage"], **dict.fromkeys(PRODUCERS, 1.0)},
            "frozen_files":frozen, "history_status":"exposed_history_not_independent_holdout",
            "leveraged_products_allowed":False, "order_authority":False,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-desired-targets.csv",
                          frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"desired_monthly_targets_per_candidate":len(original.dropna(how="all")),
            "actual_path_dependent_trade_targets_computed":False, "candidate_returns_or_metrics_computed":False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed gold-released producer routes without portfolio accounts.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/producer-gold-routing-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Producer routing preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
