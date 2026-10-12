from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import near_term_volatility_inputs as inputs
from us_quant.candidate_receipts import local_path
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.joint_risk_confirmation import confirmation_targets
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.sector_term_guard import load_inputs as original_inputs
from us_quant.storage import file_digest, new_output_directory, read_json, utc_now, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/near-term-risk-confirmation.json"
CANDIDATES = (
    {"id":"semiconductor_dollar_vix9d_monthly_confirmation", "confirmation_schedule":"completed_month"},
    {"id":"semiconductor_dollar_vix9d_daily_confirmation", "confirmation_schedule":"daily_option_state"},
)
REFERENCES = ("source_registration", "source_policy", "source_implementation", "event_policy",
              "event_implementation", "option_policy", "option_implementation", "option_readiness", "option_manifest")


def validate(policy: dict) -> None:
    constants = {
        "schema_version":1, "study_id":"published_nine_day_option_confirmation_20261011",
        "data_scope":"factor_etf_portfolio", "data_start":"2015-08-10", "as_of":"2026-10-05",
        "prior_evaluated_configurations":156, "new_configurations":2, "new_economic_factor_definitions":0,
        "factor_ids":["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"],
        "factor_symbols":list(FACTORS), "source_candidate":"semiconductor_term_or_dollar_half_guard",
        "original_control_candidate":"four_factor_semiconductor_sector50", "option_comparison":["VIX9D", "VIX"],
        "option_ratio_risk_threshold":1.0, "risk_off_investment_scale":0.50, "cash_reserve":0.02,
        "primitive_role_mapping":{"VIX":"VIX9D", "VIX3M":"VIX"}, "candidates":list(CANDIDATES),
    }
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in constants.items())
        or any(not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"} for key in REFERENCES)
        or not isinstance(policy.get("methodology"), dict)
        or policy["methodology"].get("order_authority") is not False
        or policy["methodology"].get("automatic_live_deployment") is not False):
        raise QuantError("Keep the fixed actual9D/30D comparison,two timings,explicit roles and original funding.")


def references(policy: dict) -> None:
    validate(policy)
    for key in REFERENCES:
        safe_file(ROOT, policy[key]["path"], policy[key]["sha256"])


def option_closes(index: pd.DatetimeIndex, policy: dict) -> pd.DataFrame:
    references(policy)
    inputs.validate_policy(read_json(ROOT / policy["option_policy"]["path"]))
    proof = read_json(ROOT / policy["option_readiness"]["path"])
    manifest = read_json(ROOT / policy["option_manifest"]["path"])
    if (proof["policy"] != policy["option_policy"] or proof["source_validator_sha256"] != inputs.fingerprint()
        or proof["private_source_manifest"] != policy["option_manifest"]
        or proof["full_source_readiness"] is not True or proof["independent_aligned_closes_checked"] != 8415
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or manifest["policy_sha256"] != policy["option_policy"]["sha256"]
        or manifest["source_validator_sha256"] != inputs.fingerprint()
        or manifest["full_source_readiness"] is not True or len(manifest["histories"]) != 3):
        raise QuantError("Use the complete actual named close source proof without relabelling its source-only scope.")
    for name, digest in manifest["source_files"].items():
        safe_file(ROOT, name, digest)
    histories = {}
    for entry in manifest["histories"]:
        name = entry["series"]
        if name not in inputs.SERIES or name in histories:
            raise QuantError("Only one genuine history per declared option index may be used.")
        at = pd.Timestamp(entry["acquired_at"])
        if at.tzinfo is None:
            raise QuantError("Every original acquisition timestamp must remain explicit.")
        last = at.tz_convert("America/New_York").normalize().tz_localize(None)
        if str(last.date()) != entry["acquisition_date_limit"]:
            raise QuantError("Do not retime source histories or change their genuine acquisition-date limit.")
        histories[name] = inputs.parse_history(safe_file(ROOT, entry["path"], entry["sha256"]), name, last)
    result = inputs.align_closes(index, histories)
    reference = manifest["daily_closes"]
    canonical = pd.read_csv(safe_file(ROOT, reference["path"], reference["sha256"]),
                            index_col=0, parse_dates=True, float_precision="round_trip")
    if not result.equals(canonical):
        raise QuantError("Reparsed genuine option source fields differ from the independently verified close matrix.")
    return result


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    closes = option_closes(data.close.index, policy)
    _, dollar, provenance = original_inputs(data, read_json(ROOT / policy["source_policy"]["path"]))
    return closes, dollar, provenance


def build_from_inputs(data: MarketData, closes: pd.DataFrame, dollar: pd.Series, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    data.validate()
    if (not isinstance(closes, pd.DataFrame) or not closes.index.equals(data.close.index)
        or list(closes.columns) != list(inputs.SERIES)
        or not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                   and not pd.api.types.is_complex_dtype(dtype) for dtype in closes.dtypes)
        or not np.isfinite(closes.to_numpy()).all()
        or (closes <= 0).any().any()):
        raise QuantError("Require genuine complete namedVIX9D/VIX/VIX3M closing inputs before adapting event roles.")
    baseline = sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))[
        policy["original_control_candidate"]
    ]
    # The immutable event primitive uses legacy labels for its near/far input roles.
    roles = closes[["VIX9D", "VIX"]].rename(columns={"VIX9D":"VIX", "VIX":"VIX3M"})
    return {candidate["id"]:confirmation_targets(baseline, roles, dollar, candidate["confirmation_schedule"])
            for candidate in CANDIDATES}


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    closes, dollar, _ = load_inputs(data, policy)
    return build_from_inputs(data, closes, dollar, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    references(policy)
    previous = read_json(ROOT / policy["source_registration"]["path"])
    records = [entry for entry in previous["candidates"] if entry["spec"]["id"] == policy["source_candidate"]]
    if len(records) != 1:
        raise QuantError("Require the exact original option/dollar-sector source registration.")
    original = records[0]["spec"]
    for name, digest in original["frozen_files"].items():
        safe_file(ROOT, name, digest)
    data = review_market({"market":original["market"]}, ROOT)
    audited = verified_etf_market(previous["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume", "risk_free"):
        left, right = getattr(data, name), getattr(audited, name)
        relative, absolute = (0, 1e-12) if name == "risk_free" else (1e-10, 1e-9)
        if (not left.index.equals(right.index)
            or isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=relative, atol=absolute)):
            raise QuantError("Preserve every original actual fund and risk-free source observation.")
    closes, dollar, provenance = load_inputs(data, policy)
    frames = build_from_inputs(data, closes, dollar, policy)
    baseline = sector_targets(data, read_json(ROOT / "config/sector-growth-balance.json"))[
        policy["original_control_candidate"]
    ]
    output = local_path(ROOT, output)
    if output == ROOT / "data" or not output.is_relative_to(ROOT / "data"):
        raise QuantError("Use a new isolated private near-term risk preparation directory.")
    new_output_directory(output)
    for name, frame in (("option-closes", closes), ("monthly-dollar", dollar),
                        ("dollar-provenance", provenance), ("original-source-targets", baseline)):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    manifest = read_json(ROOT / policy["option_manifest"]["path"])
    frozen = dict(original["frozen_files"])
    additions = {
        **{policy[key]["path"]:policy[key]["sha256"] for key in REFERENCES},
        **manifest["source_files"], POLICY.relative_to(ROOT).as_posix():file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix():file_digest(Path(__file__)),
        **{f"src/us_quant/{name}.py":file_digest(ROOT / f"src/us_quant/{name}.py")
           for name in ("calendar", "storage")},
    }
    source_policy = read_json(ROOT / policy["option_policy"]["path"])
    additions.update({source_policy[key]["path"]:source_policy[key]["sha256"] for key in inputs.REFERENCES})
    for name, digest in additions.items():
        if name in frozen and frozen[name] != digest:
            raise QuantError("Do not overwrite any conflicting original frozen source,policy or helper.")
        frozen[name] = digest
    readiness = deepcopy(previous["readiness"])
    readiness["checked_at"] = utc_now()
    readiness["revalidation_not_new_market_observation"] = True
    write_json(output / "readiness.json", readiness)
    for candidate in CANDIDATES:
        spec = {**original, "id":candidate["id"], "configuration":candidate, "frozen_files":frozen}
        write_json(output / f"{candidate['id']}-spec.json", spec)
        write_text_atomic(output / f"{candidate['id']}-targets.csv", frames[candidate["id"]].to_csv(float_format="%.17g"))
    return {"original_monthly_decisions":len(baseline.dropna(how="all")),
            "genuine_daily_option_close_observations":closes.size,
            "defined_target_rows":{name:len(frame.dropna(how="all")) for name, frame in frames.items()},
            "candidate_returns_or_metrics_computed":False, "new_source_requests":0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed nine-day option confirmations using the frozen event engine.")
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/near-term-risk-confirmation-20261011")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.prepared), indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Nine-day confirmation preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
