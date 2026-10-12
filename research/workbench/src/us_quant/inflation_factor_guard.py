from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs, dollar_comparisons, load_vintages
from us_quant.inflation_inputs import (
    fingerprint as input_fingerprint,
)
from us_quant.inflation_inputs import (
    parse_archive,
    query_readiness,
    request_fields,
)
from us_quant.inflation_inputs import (
    validate_policy as validate_input_policy,
)
from us_quant.research_program import (
    ResearchProgram,
    review_market,
    safe_file,
    timestamp,
    verified_etf_market,
)
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/inflation-factor-guard.json"
CONTROL = ROOT / "config/sector-growth-balance.json"
DOLLAR_POLICY = ROOT / "config/dollar-risk-guard.json"
INPUT_POLICY = ROOT / "config/inflation-inputs.json"
INPUT_SOURCE = ROOT / "data/inflation-inputs-20261011"
INPUT_READINESS = ROOT / "evidence/inflation_input_readiness_20261011.json"
PRIOR = ROOT / "evidence/dollar_risk_guard_20261011_registration.json"
CANDIDATE = {
    "id": "semiconductor_dollar_inflation_gold_guard",
    "gold_preserved_when_compensation_rising": True,
}


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "data_scope": "factor_etf_portfolio",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "eligible_not_before": "2026-10-17T09:00:00+08:00",
        "planned_configurations": 1,
        "new_economic_factor_definitions": 0,
        "original_control_candidate": "four_factor_semiconductor_sector50",
        "comparison_candidate": "semiconductor_dollar_half_guard",
        "risk_change_sessions": 63,
        "risk_off_investment_scale": 0.50,
        "compensation_change_threshold": 0.0,
        "cash_reserve": 0.02,
        "candidates": [CANDIDATE],
    }
    if any(policy.get(key) != value for key, value in expected.items()) or any(
        policy.get(key) is not False
        for key in (
            "order_authority",
            "automatic_live_deployment",
            "existing_forward_rules_changed",
            "new_factor_independence_claimed",
        )
    ):
        raise QuantError("The single fixed inflation guard, eligibility or risk budget changed.")


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, dict]:
    validate_policy(policy)
    input_policy = read_json(INPUT_POLICY)
    validate_input_policy(input_policy)
    readiness = read_json(INPUT_READINESS)
    manifest = read_json(
        safe_file(
            ROOT,
            readiness["source_manifest"],
            readiness["source_manifest_sha256"],
        )
    )
    if (
        readiness["policy_sha256"] != file_digest(INPUT_POLICY)
        or readiness["readiness_code_sha256"] != input_fingerprint()
        or manifest["policy_sha256"] != file_digest(INPUT_POLICY)
        or manifest["all_requested_batches_complete"] is not True
        or readiness["strategy_states_targets_or_returns_computed"] is not False
    ):
        raise QuantError(
            "Use the original sealed source-only inflation readiness, not revised inputs."
        )
    for name, digest in manifest["source_files"].items():
        safe_file(INPUT_SOURCE, name, digest)
    vintages = {}
    for batch in manifest["batches"]:
        dates = batch["vintage_dates"]
        request = read_json(safe_file(INPUT_SOURCE, batch["request_path"], batch["request_sha256"]))
        if (
            request["advertised_form_fields"] != request_fields(dates)
            or request["http_success"] is not True
            or request["sha256"] != batch["sha256"]
            or set(vintages).intersection(dates)
        ):
            raise QuantError(
                "An inflation source request changed or duplicated a historical snapshot."
            )
        frame = parse_archive(safe_file(INPUT_SOURCE, batch["path"], batch["sha256"]), dates)
        vintages.update({day: frame[day].dropna() for day in dates})
    selected = query_readiness(data.close.index, vintages)
    selected.index = pd.to_datetime(selected.index)
    pairs = decision_pairs(data.close.index)
    inflation = pd.Series(
        {
            day: selected.loc[day, "yield_percent"] - selected.loc[reference, "yield_percent"]
            for day, reference in pairs.items()
        },
        name="inflation_compensation_change",
    )
    dollar_vintages, _ = load_vintages(data.close.index, read_json(DOLLAR_POLICY))
    dollar, dollar_audit = dollar_comparisons(dollar_vintages, data.close.index)
    comparisons = pd.concat([dollar, inflation], axis=1)
    if not np.isfinite(comparisons.to_numpy()).all():
        raise QuantError(
            "Every frozen monthly risk query needs both actual inflation and dollar inputs."
        )
    return comparisons, {"inflation_selected_queries": selected, "dollar_provenance": dollar_audit}


def build_from_inputs(
    data: MarketData, comparisons: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    original = sector_targets(data, read_json(CONTROL))[policy["original_control_candidate"]]
    if (
        list(comparisons.columns) != ["dollar_change", "inflation_compensation_change"]
        or not comparisons.index.equals(original.dropna(how="all").index)
        or not np.isfinite(comparisons.to_numpy()).all()
    ):
        raise QuantError("The inflation guard needs complete inputs on every original month-end.")
    targets = original.copy()
    noncash = targets.columns.drop("BIL")
    for day, row in comparisons.iterrows():
        if row["dollar_change"] <= 0:
            continue
        reduce = noncash.drop("GLD") if row["inflation_compensation_change"] > 0 else noncash
        removed = float(targets.loc[day, reduce].sum()) * 0.5
        targets.loc[day, reduce] *= 0.5
        targets.loc[day, "BIL"] += removed
    active = targets.dropna(how="all")
    if (active < 0).any().any() or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12):
        raise QuantError(
            "Inflation targets cannot short, borrow or change the original invested budget."
        )
    return {CANDIDATE["id"]: targets}


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, comparisons, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR)
    original = prior["candidates"][0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    actual = verified_etf_market(prior["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        left, right = getattr(data, name), getattr(actual, name)
        if (
            not left.index.equals(right.index)
            or not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError(
                "Inflation preparation cannot replace the frozen sector/factor market."
            )
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("Retain the original market risk-free observations.")
    comparisons, provenance = load_inputs(data, policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Use a separate preparation directory inside the workbench.")
    new_output_directory(output)
    for name, frame in {"risk-comparisons": comparisons, **provenance}.items():
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    target = build_from_inputs(data, comparisons, policy)[CANDIDATE["id"]]
    write_text_atomic(output / "frozen-targets.csv", target.to_csv(float_format="%.17g"))
    manifest = read_json(INPUT_SOURCE / "manifest.json")
    frozen = {
        **original["frozen_files"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        INPUT_POLICY.relative_to(ROOT).as_posix(): file_digest(INPUT_POLICY),
        INPUT_READINESS.relative_to(ROOT).as_posix(): file_digest(INPUT_READINESS),
        (INPUT_SOURCE / "manifest.json").relative_to(ROOT).as_posix(): file_digest(
            INPUT_SOURCE / "manifest.json"
        ),
        "src/us_quant/inflation_inputs.py": file_digest(ROOT / "src/us_quant/inflation_inputs.py"),
        "src/us_quant/financial_conditions_guard.py": file_digest(
            ROOT / "src/us_quant/financial_conditions_guard.py"
        ),
        PRIOR.relative_to(ROOT).as_posix(): file_digest(PRIOR),
    }
    frozen.update(
        {
            (INPUT_SOURCE / name).relative_to(ROOT).as_posix(): digest
            for name, digest in manifest["source_files"].items()
        }
    )
    spec = {
        **original,
        "id": CANDIDATE["id"],
        "configuration": CANDIDATE,
        "frozen_files": frozen,
        "eligible_not_before": policy["eligible_not_before"],
    }
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    write_json(output / f"{CANDIDATE['id']}-spec.json", spec)
    write_json(output / "readiness.json", readiness)
    audit = {
        "prepared_at": utc_now(),
        "monthly_decisions": len(comparisons),
        "target_shape": list(target.shape),
        "target_sha256": file_digest(output / "frozen-targets.csv"),
        "all_four_equity_families_remain_positive": bool(
            (target.dropna(how="all")[["MTUM", "VLUE", "QUAL", "USMV"]] > 0).all().all()
        ),
        "eligible_not_before": policy["eligible_not_before"],
        "candidate_registration_performed": False,
        "strategy_outcomes_computed": False,
    }
    write_json(output / "target-budget-audit.json", audit)
    return {"spec": spec, "readiness": readiness, "target_budget_audit": audit}


def check_registration_eligibility(spec: dict, root: Path, at) -> None:
    from us_quant.strategy_replay import registered_generator

    name, policy = registered_generator(spec, root)
    validate_policy(policy)
    if (
        name != "inflation_factor_guard"
        or spec.get("eligible_not_before") != policy["eligible_not_before"]
    ):
        raise QuantError(
            "The inflation candidate needs its exact frozen eligible-cycle declaration."
        )
    if pd.Timestamp(at) < pd.Timestamp(policy["eligible_not_before"]):
        raise QuantError(
            f"Inflation registration is not eligible before {policy['eligible_not_before']}."
        )


def register(prepared: Path, receipt: Path) -> dict:
    prepared = prepared if prepared.is_absolute() else ROOT / prepared
    receipt = receipt if receipt.is_absolute() else ROOT / receipt
    if (
        prepared.is_symlink()
        or not prepared.resolve().is_relative_to(ROOT.resolve())
        or receipt.is_symlink()
        or not receipt.resolve().is_relative_to(ROOT.resolve())
        or receipt.exists()
    ):
        raise QuantError("Use unchanged preparation and a new receipt inside the workbench.")
    spec = read_json(
        safe_file(ROOT, (prepared / f"{CANDIDATE['id']}-spec.json").relative_to(ROOT).as_posix())
    )
    check_registration_eligibility(spec, ROOT, timestamp())
    readiness = read_json(
        safe_file(ROOT, (prepared / "readiness.json").relative_to(ROOT).as_posix())
    )
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3", read_json(ROOT / "config/research-program.json")
    )
    try:
        result = {
            "spec": spec,
            "readiness": readiness,
            "registration": program.register_candidate(spec, readiness),
            "registered_before_strategy_outcomes": True,
            "order_authority": False,
        }
        write_json(receipt, result)
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one frozen future inflation guard; never orders."
    )
    parser.add_argument("action", choices=("prepare", "register"))
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "data/inflation-guard-prepared-20261011"
    )
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "evidence/inflation_guard_actual_registration.json"
    )
    args = parser.parse_args()
    try:
        result = (
            prepare(args.prepared)
            if args.action == "prepare"
            else register(args.prepared, args.receipt)
        )
        print(
            json.dumps(
                {
                    "candidate_id": result["spec"]["id"],
                    "eligible_not_before": result["spec"]["eligible_not_before"],
                    "candidate_registration_performed": args.action == "register",
                    "strategy_outcomes_computed": False,
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Inflation candidate blocked: {exc}\n")


if __name__ == "__main__":
    main()
