from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import equity_budget_attribution as budget
from us_quant.bt_audit import independent_equity
from us_quant.candidate_receipts import input_reference, local_path, write_new_receipt
from us_quant.cash_funded_accounting_v2 import engine_reference, simulate
from us_quant.config import QuantError
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.research_program import engine_fingerprint, review_market, safe_file
from us_quant.research_weekly import verify_publication
from us_quant.review_receipts import evidence_time
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/equity-component-attribution.json"
COMPONENTS = {
    "momentum": ("MTUM", "PDP"),
    "value": ("VLUE",),
    "quality": ("QUAL", "SPHQ"),
    "low_volatility": ("USMV",),
    "growth_technology": ("QQQ", "XLK"),
    "semiconductor": ("SOXX",),
    "all_styles": ("MTUM", "PDP", "VLUE", "QUAL", "SPHQ", "USMV"),
    "all_satellites": ("QQQ", "XLK", "SOXX"),
}
REFERENCES = (
    "source_budget_policy",
    "source_results",
    "source_verification",
    "source_read_precision",
    "source_preservation",
)


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "diagnostic_id": "equity_component_attribution_20261011",
        "scope": "complete_recurring_cohort_conditional_component_replacement",
        "included_candidate_count": 36,
        "horizons_years": [10, 5],
        "scenarios": ["base", "stress"],
        "capital_usd": 10000.0,
        "base_cost_bps": 5.0,
        "stress_cost_bps": 20.0,
        "base_delay_sessions": 1,
        "stress_delay_sessions": 2,
        "commission_per_order": 1.0,
        "components": {name: list(symbols) for name, symbols in COMPONENTS.items()},
        "logical_component_controls": 288,
        "logical_horizon_scenario_comparisons": 1152,
        "paired_accounting": "frozen_cash_funded_noop_v2_with_verified_original_legs_reused",
        "new_strategy_evaluations": 0,
        "new_registered_candidates": 0,
        "new_economic_factor_definitions": 0,
        "independent_alpha_or_additive_attribution_claimed": False,
        "qualification_policy_changed": False,
        "existing_forward_rules_changed": False,
        "order_authority": False,
        "investment_objective_verified": False,
    }
    if (
        not isinstance(policy, dict)
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in constants.items()
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError("The fixed component groups, complete scope or interpretation changed.")


def component_control(targets: pd.DataFrame, component: str) -> tuple[pd.DataFrame, bool]:
    if component not in COMPONENTS:
        raise QuantError("Use an explicitly frozen component, not a result-selected asset subset.")
    budget.spy_budget_control(targets)
    selected = [symbol for symbol in COMPONENTS[component] if symbol in targets]
    active = targets.dropna(how="all")
    exposed = bool(active[selected].gt(0).any().any())
    result = targets.copy()
    if exposed:
        result.loc[active.index, "SPY"] = active["SPY"] + active[selected].sum(axis=1)
        result.loc[active.index, selected] = 0.0
    unchanged = [symbol for symbol in targets if symbol not in {*selected, "SPY"}]
    if (
        not result.isna().equals(targets.isna())
        or not result[unchanged].equals(targets[unchanged])
        or not np.allclose(
            result.loc[active.index].sum(axis=1), active.sum(axis=1), rtol=0, atol=1e-12,
        )
        or (not exposed and not result.equals(targets))
    ):
        raise QuantError("A component replacement changed unselected targets, budgets or pauses.")
    return result, exposed


def fingerprint() -> str:
    return digest_json({
        "component_module_sha256": file_digest(Path(__file__)),
        "frozen_budget_auditor_sha256": budget.fingerprint(),
    })


def frozen_sources(policy: dict) -> tuple[dict, list[dict], dict]:
    validate_policy(policy)
    values = {
        name: read_json(safe_file(ROOT, policy[name]["path"], policy[name]["sha256"]))
        for name in REFERENCES
    }
    state, entries = budget.frozen_inputs(values["source_budget_policy"])
    public, proof = values["source_results"], values["source_verification"]
    if (
        public["status"] != "completed"
        or public["candidate_count"] != 36
        or public["all_288_paired_leg_accounts_and_144_comparisons_verified"] is not True
        or public["all_original_target_defense_and_seed_weights_independently_verified"] is not True
        or public["new_strategy_evaluations"] != 0
        or public["order_authority"] is not False
        or proof["results_sha256"] != policy["source_results"]["sha256"]
        or proof["checked_paired_comparisons"] != 144
        or proof["old_review_bodies_or_original_accounts_rewritten"] is not False
        or values["source_read_precision"]["old_equity_paths_or_review_metrics_rewritten"] is not False
        or values["source_preservation"]["current_engine_sha256"] != engine_fingerprint()
        or values["source_preservation"]["current_event_head"] != state["event_chain_sha256"]
    ):
        raise QuantError("Reuse only the complete verified original funded source legs.")
    private = read_json(safe_file(
        ROOT, public["private_result_source"], public["private_result_source_sha256"],
    ))
    pairs = public["paired_comparisons"]
    expected = {
        (entry["spec"]["id"], years, scenario)
        for entry in entries for years in (10, 5) for scenario in ("base", "stress")
    }
    actual = {(row["candidate_id"], row["years"], row["scenario"]) for row in pairs}
    if (
        len(pairs) != 144 or actual != expected
        or pairs != private["paired_comparisons"]
        or len(private["artifact_sha256"]) != 720
        or digest_json(private["artifact_sha256"]) != public["private_artifact_hash_manifest_sha256"]
        or public["paired_accounting_engine"] != engine_reference()
    ):
        raise QuantError("The original source outcomes or private artifact manifest changed.")
    return state, entries, {
        **values,
        "private": private,
        "pairs": {(row["candidate_id"], row["years"], row["scenario"]): row for row in pairs},
    }


def registration_body(policy: dict, state: dict, entries: list[dict], source: dict) -> dict:
    public = source["source_results"]
    return {
        "schema_version": 1,
        "policy": policy,
        "policy_sha256": file_digest(POLICY),
        "auditor_sha256": fingerprint(),
        "engine_sha256": engine_fingerprint(),
        "event_chain_sha256": state["event_chain_sha256"],
        "included_candidate_ids": [entry["spec"]["id"] for entry in entries],
        "original_bundle_references": [entry["bundle_reference"] for entry in entries],
        "component_ids": list(COMPONENTS),
        "logical_component_controls": 288,
        "logical_horizon_scenario_comparisons": 1152,
        "source_private_result": {
            "path": public["private_result_source"],
            "sha256": public["private_result_source_sha256"],
        },
        "accounting_engine": engine_reference(),
        "new_component_accounting_outcomes_computed": False,
        "history_and_previous_budget_control_outcomes_already_exposed": True,
        "derived_controls_are_registered_trading_candidates": False,
        "source_accounts_reused_not_rewritten": True,
        "new_strategy_evaluations": 0,
        "order_authority": False,
        "investment_objective_verified": False,
    }


def register(path: Path) -> dict:
    path = local_path(ROOT, path)
    if path.suffix != ".json" or not path.is_relative_to(ROOT / "evidence"):
        raise QuantError("Use a new metadata-only component registration under evidence.")
    policy = read_json(POLICY)
    state, entries, source = frozen_sources(policy)
    at = utc_now()
    if evidence_time(source["source_results"]["completed_at"]) > evidence_time(at):
        raise QuantError("Component registration cannot predate the exposed source outcomes.")
    with_program = budget.program()
    try:
        if with_program.status() != state:
            raise QuantError("The complete source cohort changed before component registration.")
        result = {**registration_body(policy, state, entries, source), "registered_at": at}
        write_new_receipt(path, result)
        actual, _ = input_reference(ROOT, path)
        if actual != result or with_program.status() != state:
            raise QuantError("The original component registration was not durably preserved.")
        return result
    finally:
        with_program.close()


def published_inputs(reference: dict, entries: list[dict], policy: dict, commit: str) -> dict:
    original = read_json(safe_file(
        ROOT, policy["source_budget_policy"]["path"], policy["source_budget_policy"]["sha256"],
    ))
    inherited = budget.published_inputs(reference, entries, original, commit)
    required = {
        POLICY.relative_to(ROOT).as_posix(),
        "src/us_quant/equity_component_attribution.py",
        *(policy[name]["path"] for name in REFERENCES),
    }
    own = verify_publication(ROOT, reference, required, commit)
    return {
        **own,
        "inherited_budget_publication": inherited,
        "all_old_and_new_frozen_dependencies_verified": True,
    }


def control_account(data, targets: pd.DataFrame, start: str, end: str, cost: float, delay: int):
    if (cost, delay) not in ((5.0, 1), (20.0, 2)):
        raise QuantError("Component accounts must retain original base or stress execution.")
    account = simulate(
        data, targets, start, end, initial_capital=10000, cost_bps=cost, commission=1, delay=delay,
    )
    separate = independent_equity(
        data, targets, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay,
    )
    metrics, error = independent_metrics(
        account.frame, separate, data.risk_free.loc[account.frame.index], 10000,
    )
    return account, separate, {
        "metrics": metrics,
        "max_independent_equity_error_usd": error,
        "total_cost_usd": float(account.frame["cost"].sum()),
        "total_orders": int(account.frame["orders"].sum()),
        "minimum_cash_usd": float(account.frame["cash"].min()),
    }


def original_leg(data, source: dict, candidate: str, years: int, scenario: str):
    old = source["source_results"]
    folder = Path(old["private_result_source"]).parent
    prefix = f"{candidate}/{years}y-{scenario}-source_rule_funded_v2"
    references, frames = {}, {}
    for suffix in ("account", "bt"):
        name = f"{prefix}-{suffix}.csv"
        reference = {
            "path": (folder / name).as_posix(),
            "sha256": source["private"]["artifact_sha256"][name],
        }
        frames[suffix] = pd.read_csv(
            safe_file(ROOT, reference["path"], reference["sha256"]),
            index_col=0, parse_dates=True, float_precision="round_trip",
        )
        references[suffix] = reference
    metadata = source["pairs"][(candidate, years, scenario)]["source_rule_funded_v2"]
    actual, _ = independent_metrics(
        frames["account"], frames["bt"], data.risk_free.loc[frames["account"].index], 10000,
    )
    check_metrics(actual, metadata["metrics"])
    if (
        abs(float(frames["account"]["cost"].sum()) - metadata["total_cost_usd"]) > 1e-8
        or int(frames["account"]["orders"].sum()) != metadata["total_orders"]
    ):
        raise QuantError("The reused original source account changed costs or orders.")
    return frames["account"], metadata, references


def differences(original: dict, control: dict) -> dict:
    return {
        key: original["metrics"][key] - control["metrics"][key]
        if original["metrics"][key] is not None and control["metrics"][key] is not None
        else None
        for key in ("cagr", "sharpe", "max_drawdown")
    }


def summaries(pairs: list[dict]) -> dict:
    result = {}
    for name in COMPONENTS:
        rows = [row for row in pairs if row["component_id"] == name]
        active = [row for row in rows if row["selected_component_exposure_present"]]
        candidates = {row["candidate_id"] for row in active}
        full = [
            [row for row in active if row["candidate_id"] == candidate]
            for candidate in sorted(candidates)
        ]
        result[name] = {
            "logical_comparisons": len(rows),
            "nonidentity_comparisons": len(active),
            "identity_comparisons": len(rows) - len(active),
            "source_cagr_above_control_nonidentity_pairs": sum(
                row["source_minus_component_control"]["cagr"] > 0 for row in active
            ),
            "source_sharpe_above_control_nonidentity_pairs": sum(
                row["source_minus_component_control"]["sharpe"] is not None
                and row["source_minus_component_control"]["sharpe"] > 0 for row in active
            ),
            "all_four_paths_exposed_source_count": sum(len(rows) == 4 for rows in full),
            "source_cagr_above_control_in_all_four_exposed_paths": sum(
                len(rows) == 4
                and all(row["source_minus_component_control"]["cagr"] > 0 for row in rows)
                for rows in full
            ),
        }
    return result


def run(registration_path: Path, output: Path, published_commit: str) -> dict:
    registration, reference = input_reference(ROOT, registration_path)
    policy = read_json(POLICY)
    state, entries, source = frozen_sources(policy)
    expected = registration_body(policy, state, entries, source)
    if (
        set(registration) != {*expected, "registered_at"}
        or any(registration.get(key) != value for key, value in expected.items())
        or not evidence_time(source["source_results"]["completed_at"])
        <= evidence_time(registration.get("registered_at")) <= evidence_time(utc_now())
    ):
        raise QuantError("Preserve the exact registered component scope, source evidence and code.")
    publication = published_inputs(reference, entries, policy, published_commit)
    output = local_path(ROOT, output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports"):
        raise QuantError("Use a new isolated private reports directory for component accounts.")
    with_program = budget.program()
    result = None
    created = False
    try:
        if with_program.status() != state:
            raise QuantError("The original research cohort changed after component registration.")
        new_output_directory(output)
        created = True
        write_new_receipt(output / "publication.json", {
            **publication,
            "verified_on_remote_at": utc_now(),
            "registration_predates_new_component_accounts": True,
            "investment_objective_verified": False,
            "order_authority": False,
        })
        result = {
            "schema_version": 1, "status": "running", "started_at": utc_now(),
            "registration_sha256": reference["sha256"],
            "policy_sha256": file_digest(POLICY), "auditor_sha256": fingerprint(),
            "published_registration_commit": published_commit,
            "original_source_candidate_count": 36,
            "logical_component_controls": 288, "logical_horizon_scenario_comparisons": 1152,
            "paired_comparisons": [], "computed_nonidentity_control_accounts": 0,
            "reused_identity_control_accounts": 0, "verified_original_source_accounts": 0,
            "accounting_engine": engine_reference(),
            "new_strategy_evaluations": 0, "new_registered_candidates": 0,
            "history_already_exposed": True,
            "independent_alpha_or_additive_attribution_claimed": False,
            "order_authority": False, "investment_objective_verified": False,
        }
        for entry in entries:
            spec, bundle = entry["spec"], entry["bundle"]
            data, cache = review_market(bundle, ROOT), {}
            for years in (10, 5):
                for scenario in ("base", "stress"):
                    path = next(
                        row for row in bundle["paths"]
                        if (row["years"], row["scenario"]) == (years, scenario)
                    )
                    cost, delay = (5.0, 1) if scenario == "base" else (20.0, 2)
                    if (
                        path["capital_usd"] != 10000 or path["cost_bps"] != cost
                        or path["delay_sessions"] != delay or path["commission_per_order"] != 1
                    ):
                        raise QuantError("A component path cannot change original execution.")
                    retained = pd.read_csv(
                        safe_file(ROOT, path["targets"]["path"], path["targets"]["sha256"]),
                        index_col=0, parse_dates=True, float_precision="round_trip",
                    )
                    frame, original, source_references = original_leg(
                        data, source, spec["id"], years, scenario,
                    )
                    start, end = original["metrics"]["start"], original["metrics"]["end"]
                    generated = registered_targets(
                        spec, data, start, end, cost, delay, ROOT, cache,
                    )
                    if (
                        not retained.index.equals(generated.index)
                        or not retained.columns.equals(generated.columns)
                        or not retained.isna().equals(generated.isna())
                        or not np.allclose(retained, generated, equal_nan=True, rtol=0, atol=1e-10)
                        or frame.index[0].strftime("%Y-%m-%d") != start
                        or frame.index[-1].strftime("%Y-%m-%d") != end
                    ):
                        raise QuantError("Retain every original source-regenerated target and seed.")
                    result["verified_original_source_accounts"] += 1
                    for component in COMPONENTS:
                        result["current_comparison"] = {
                            "candidate_id": spec["id"], "years": years,
                            "scenario": scenario, "component_id": component,
                        }
                        write_json(output / "progress.json", result)
                        replacement, exposed = component_control(retained, component)
                        if exposed:
                            account, independent, metadata = control_account(
                                data, replacement, start, end, cost, delay,
                            )
                            prefix = output / spec["id"] / f"{years}y-{scenario}-{component}"
                            for suffix, values in (
                                ("targets", replacement), ("account", account.frame),
                                ("bt", independent),
                            ):
                                write_text_atomic(
                                    prefix.with_name(prefix.name + f"-{suffix}.csv"),
                                    values.to_csv(float_format="%.17g"),
                                )
                            result["computed_nonidentity_control_accounts"] += 1
                        else:
                            metadata = original
                            result["reused_identity_control_accounts"] += 1
                        result["paired_comparisons"].append({
                            **result["current_comparison"],
                            "selected_component_exposure_present": exposed,
                            "identity_control_source_leg_reused": not exposed,
                            "original_source_account_references": source_references,
                            "source_rule_funded_v2": original,
                            "component_budget_spy_control": metadata,
                            "source_minus_component_control": differences(original, metadata),
                        })
        expected_pairs = {
            (entry["spec"]["id"], years, scenario, component)
            for entry in entries for years in (10, 5)
            for scenario in ("base", "stress") for component in COMPONENTS
        }
        actual_pairs = {
            (row["candidate_id"], row["years"], row["scenario"], row["component_id"])
            for row in result["paired_comparisons"]
        }
        if (
            len(result["paired_comparisons"]) != 1152 or actual_pairs != expected_pairs
            or result["verified_original_source_accounts"] != 144
            or result["computed_nonidentity_control_accounts"]
            + result["reused_identity_control_accounts"] != 1152
            or with_program.status() != state
        ):
            raise QuantError("The component diagnostic omitted comparisons or changed old research.")
        result.update({
            "status": "completed", "completed_at": utc_now(),
            "component_summaries": summaries(result["paired_comparisons"]),
            "all_original_reviews_and_research_state_unchanged": True,
            "artifact_sha256": {
                path.relative_to(output).as_posix(): file_digest(path)
                for path in sorted(output.rglob("*.csv"))
            },
        })
        write_json(output / "results.json", result)
        return result
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        if created:
            write_json(output / "failure.json", {
                "recorded_at": utc_now(), "error_type": type(exc).__name__, "reason": str(exc),
                "completed_comparisons_retained": (
                    len(result["paired_comparisons"]) if result is not None else 0
                ),
                "current_comparison": result.get("current_comparison") if result else None,
                "partial_outputs_deleted_or_filled": False,
                "investment_objective_verified": False, "order_authority": False,
            })
        raise
    finally:
        with_program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister complete conditional component replacements; never orders.",
    )
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration", type=Path,
        default=Path("evidence/equity_component_attribution_20261011_registration.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/equity-component-attribution-20261011"),
    )
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        if args.action == "register":
            result = register(args.registration)
            print(json.dumps({
                "registered_at": result["registered_at"],
                "candidate_count": len(result["included_candidate_ids"]),
                "component_count": len(result["component_ids"]),
                "logical_comparisons": result["logical_horizon_scenario_comparisons"],
                "new_component_accounting_outcomes_computed": False,
                "investment_objective_verified": False,
            }, indent=2))
        else:
            result = run(args.registration, args.output, args.published_commit)
            print(json.dumps({
                "logical_comparisons": len(result["paired_comparisons"]),
                "computed_nonidentity_accounts": result["computed_nonidentity_control_accounts"],
                "reused_identity_accounts": result["reused_identity_control_accounts"],
                "verified_original_accounts": result["verified_original_source_accounts"],
                "new_strategy_evaluations": 0, "investment_objective_verified": False,
            }, indent=2))
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Component attribution blocked; preserve original evidence: {exc}\n")


if __name__ == "__main__":
    main()
