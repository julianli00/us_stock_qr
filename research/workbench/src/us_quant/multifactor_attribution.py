from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol, metrics, seed_window
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.multifactor_stability import (
    BASE,
    FACTORS,
    ROOT,
    audit_path,
    build_targets,
    gate_result,
    load_market,
)
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

EVIDENCE = ROOT / "evidence/multifactor_stability_20261010_results.json"
REGISTRATION = ROOT / "evidence/multifactor_stability_20261010_registration_v2.json"
MARKET_MANIFEST = ROOT / "evidence/multifactor_stability_20261010_market.json"
POLICY = ROOT / "config/multifactor-stability.json"
LEDGERS = ROOT / "reports/multifactor-stability-20261010"


def spy_budget_ablation(targets: pd.DataFrame) -> pd.DataFrame:
    if tuple(targets.columns) != (*BASE, *FACTORS):
        raise QuantError("Ablation requires the same complete unleveraged target universe.")
    if (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any():
        raise QuantError("Ablation cannot fill partially specified target rows.")
    result = targets.copy()
    known = targets.dropna(how="all")
    if (
        known.empty
        or not np.isfinite(known.to_numpy()).all()
        or (known < 0).any().any()
        or (known.sum(axis=1) > 1).any()
        or not known["SPY"].eq(0).all()
    ):
        raise QuantError("Ablation only accepts cash-funded factor targets without prior SPY.")
    result.loc[known.index, "SPY"] = known.loc[:, list(FACTORS)].sum(axis=1)
    result.loc[known.index, list(FACTORS)] = 0.0
    if not np.allclose(result.loc[known.index].sum(axis=1), known.sum(axis=1), atol=1e-12):
        raise QuantError("Ablation changed the total invested budget.")
    return result


def register(output: Path) -> dict:
    if output.exists():
        raise QuantError("Refusing to replace a multifactor attribution plan.")
    evidence = read_json(EVIDENCE)
    plan = {
        "schema_version": 1,
        "registered_at": utc_now(),
        "attribution_module_sha256": file_digest(Path(__file__)),
        "evidence_sha256": file_digest(EVIDENCE),
        "registration_sha256": file_digest(REGISTRATION),
        "market_manifest_sha256": file_digest(MARKET_MANIFEST),
        "candidates": evidence["stability_qualified_candidates"],
        "windows": ["10y", "5y"],
        "scenarios": ["base", "stress", "higher_cost"],
        "method": (
            "Replace four factor ETF targets with SPY at their combined weight; preserve dates, "
            "IEF/GLD/BIL weights, target cash, capital, costs and execution delay."
        ),
        "interpretation": (
            "Equity-budget-matched ablation, not an equal-volatility portfolio or an independent "
            "SPY timing strategy. The defense schedule comes from the frozen factor rule."
        ),
        "new_hypotheses": 0,
        "candidate_parameters_changed": False,
        "independent_forward_validation": False,
        "order_authority": False,
    }
    if not plan["candidates"]:
        raise QuantError("No stability-qualified candidate exists for this attribution plan.")
    write_json(output, plan)
    return plan


def validate(plan: dict, base_path: Path, source: Path, output: Path) -> dict:
    if (
        plan.get("attribution_module_sha256") != file_digest(Path(__file__))
        or plan.get("evidence_sha256") != file_digest(EVIDENCE)
        or plan.get("registration_sha256") != file_digest(REGISTRATION)
        or plan.get("market_manifest_sha256") != file_digest(MARKET_MANIFEST)
        or plan.get("candidate_parameters_changed") is not False
        or plan.get("order_authority") is not False
        or plan.get("windows") != ["10y", "5y"]
        or plan.get("scenarios") != ["base", "stress", "higher_cost"]
        or plan.get("new_hypotheses") != 0
    ):
        raise QuantError("The fixed multifactor attribution plan or evidence changed.")
    if (source / "manifest.json").read_bytes() != MARKET_MANIFEST.read_bytes():
        raise QuantError(
            "The current market snapshot differs from the pre-evaluation public manifest."
        )
    policy, registration, evidence = read_json(POLICY), read_json(REGISTRATION), read_json(EVIDENCE)
    if set(plan["candidates"]) != set(evidence["stability_qualified_candidates"]):
        raise QuantError("Attribution must include every stability-qualified candidate.")
    data = load_market(policy, registration, base_path, source)
    for relative, digest in evidence["artifact_sha256"].items():
        path = LEDGERS / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(LEDGERS)
            or file_digest(path) != digest
        ):
            raise QuantError("A saved multifactor ledger, target or weight file has changed.")
    configured = {item["id"] for item in policy["candidates"]}
    if (
        set(evidence["candidates"]) != configured
        or evidence["total_disclosed_configurations"] != 84
    ):
        raise QuantError("Multifactor research cannot omit unsuccessful candidates.")
    paths, difference_max = 0, 0.0
    for relative in evidence["artifact_sha256"]:
        if not relative.endswith("-bt.csv"):
            continue
        ordinary = LEDGERS / relative.replace("-bt.csv", ".csv")
        frame = pd.read_csv(ordinary, index_col=0, parse_dates=True)
        independent = pd.read_csv(LEDGERS / relative, index_col=0, parse_dates=True)
        values, difference = independent_metrics(
            frame, independent, data.risk_free.loc[frame.index], policy["capital_usd"]
        )
        if not relative.startswith("continuous/"):
            name, leaf = relative.split("/")
            window, scenario = leaf.removesuffix("-bt.csv").rsplit("-", 1)
            reported = (
                evidence["spy"][window][scenario]
                if name == "spy"
                else evidence["candidates"][name]["windows"][window][scenario]
            )
            check_metrics(values, reported["metrics"])
            if name != "spy":
                old_scenario = "base" if scenario == "base" else "stress"
                old = evidence["previous_strategy_metrics"][window][old_scenario]
                recalculated = gate_result(
                    metrics(frame), evidence["spy"][window][scenario]["metrics"], old, policy
                )
                if scenario != "base":
                    recalculated["also_beats_base_cost_spy"] = bool(
                        values["cagr"] > evidence["spy"][window]["base"]["metrics"]["cagr"]
                    )
                if (
                    recalculated != reported["gates"]
                    or all(recalculated.values()) != reported["all_goals_pass"]
                ):
                    raise QuantError("A published risk or return pass flag is inconsistent.")
        paths += 1
        difference_max = max(difference_max, difference)
    if paths != evidence["independent_accounting_paths"] or paths != 77:
        raise QuantError("Multifactor evidence omits a required primary/rolling accounting path.")
    targets = build_targets(data, policy)
    for name, target in targets.items():
        retained = pd.read_csv(LEDGERS / name / "issued-targets.csv", index_col=0, parse_dates=True)
        if not retained.index.equals(target.dropna(how="all").index) or not np.allclose(
            retained, target.dropna(how="all"), rtol=0, atol=5e-12
        ):
            raise QuantError("The complete saved factor targets no longer reproduce.")
    new_output_directory(output)
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "plan_sha256": digest_json(plan),
        "evidence_sha256": file_digest(EVIDENCE),
        "original_paths_independently_audited": paths,
        "max_original_equity_difference_usd": difference_max,
        "all_six_candidate_targets_reproduced": True,
        "total_disclosed_configurations": 84,
        "new_hypotheses": 0,
        "attribution": {},
        "ablation_accounting_paths": 0,
        "stability_improved_on_registered_windows": bool(
            evidence["stability_qualified_candidates"]
        ),
        "all_risk_and_return_goals_met": bool(evidence["full_goal_qualified_candidates"]),
        "direct_stock_multifactor_backtest": False,
        "independent_forward_validation": False,
        "order_authority": False,
        "automatic_baseline_replacement": False,
        "interpretation": plan["interpretation"],
    }
    comparison = load_protocol(ROOT / "config/dual-horizon.json")
    for name in plan["candidates"]:
        replacement = spy_budget_ablation(targets[name])
        result["attribution"][name] = {}
        for window in comparison.windows():
            key, start, end = (
                f"{window['years']}y",
                window["first_return_session"],
                window["last_session"],
            )
            result["attribution"][name][key] = {}
            for scenario in policy["scenarios"]:
                label = scenario["id"]
                own, independent, audit = audit_path(
                    data, seed_window(replacement, start), start, end, policy, scenario
                )
                actual = metrics(own.frame)
                original = evidence["candidates"][name]["windows"][key][label]["metrics"]
                result["attribution"][name][key][label] = {
                    "factor_portfolio": original,
                    "same_equity_budget_spy_control": actual,
                    "factor_minus_spy_control_cagr": original["cagr"] - actual["cagr"],
                    "factor_minus_spy_control_sharpe": original["sharpe"] - actual["sharpe"],
                    "factor_minus_spy_control_drawdown": original["max_drawdown"]
                    - actual["max_drawdown"],
                    "audit": audit,
                }
                prefix = f"{name}/{key}-{label}"
                write_text_atomic(output / f"{prefix}.csv", own.frame.to_csv(float_format="%.12g"))
                write_text_atomic(
                    output / f"{prefix}-bt.csv", independent.to_csv(float_format="%.12g")
                )
                result["ablation_accounting_paths"] += 1
    result["artifact_sha256"] = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*.csv"))
    }
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent multifactor audit and budget ablation."
    )
    parser.add_argument("stage", choices=("register", "evaluate"))
    parser.add_argument(
        "--plan", type=Path, default=Path("evidence/multifactor_attribution_20261010_plan.json")
    )
    parser.add_argument("--base-data", type=Path, default=Path("data/factor-round-20261007/base"))
    parser.add_argument("--data", type=Path, default=Path("data/multifactor-stability-20261010"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/multifactor-attribution-20261010")
    )
    args = parser.parse_args()
    try:
        if args.stage == "register":
            result = register(args.plan)
            keys = ("registered_at", "candidates", "new_hypotheses", "candidate_parameters_changed")
        else:
            result = validate(read_json(args.plan), args.base_data, args.data, args.output)
            keys = (
                "original_paths_independently_audited",
                "ablation_accounting_paths",
                "all_six_candidate_targets_reproduced",
                "stability_improved_on_registered_windows",
                "all_risk_and_return_goals_met",
                "independent_forward_validation",
            )
        print(json.dumps({key: result[key] for key in keys}, indent=2))
    except QuantError as exc:
        parser.exit(2, f"Multifactor attribution blocked: {exc}\n")


if __name__ == "__main__":
    main()
