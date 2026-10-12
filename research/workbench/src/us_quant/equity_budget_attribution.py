from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.bt_audit import independent_equity
from us_quant.candidate_receipts import input_reference, local_path, write_new_receipt
from us_quant.cash_funded_accounting_v2 import engine_reference, simulate
from us_quant.config import QuantError
from us_quant.factor_validation import independent_metrics
from us_quant.research_program import ResearchProgram, engine_fingerprint, review_market, safe_file
from us_quant.research_weekly import git as git_read
from us_quant.research_weekly import verify_publication
from us_quant.review_receipts import complete_paths, evidence_time, verify_references
from us_quant.selection_validation import original_bundle, verified_accounts
from us_quant.storage import (
    digest_bytes,
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
POLICY = ROOT / "config/equity-budget-attribution.json"
EQUITY = ("SPY", "MTUM", "VLUE", "QUAL", "USMV", "SPHQ", "PDP", "QQQ", "XLK", "SOXX")
DEFENSE = ("BIL", "GLD", "IEF", "TLT")
REFERENCES = ("source_snapshot", "source_bound_audit", "prior_art")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "diagnostic_id": "equity_budget_attribution_20261011",
        "scope": "completed_recurring_reviews_descriptive_budget_controls",
        "as_of": "2026-10-05",
        "data_start": "2015-08-10",
        "included_candidate_count": 36,
        "prior_completed_configurations": 120,
        "horizons_years": [10, 5],
        "scenarios": ["base", "stress"],
        "capital_usd": 10000.0,
        "base_cost_bps": 5.0,
        "stress_cost_bps": 20.0,
        "base_delay_sessions": 1,
        "stress_delay_sessions": 2,
        "commission_per_order": 1.0,
        "equity_symbols": list(EQUITY),
        "non_equity_symbols": list(DEFENSE),
        "paired_accounting": "both_legs_frozen_cash_funded_noop_v2",
        "derived_control_portfolios": 36,
        "paired_horizon_scenario_comparisons": 144,
        "highlighted_reference_ids": [
            "semiconductor_dollar_half_guard",
            "four_factor_semiconductor_sector50",
            "macro_real_yield_factor_tilt",
        ],
        "new_strategy_evaluations": 0,
        "new_registered_candidates": 0,
        "new_economic_factor_definitions": 0,
        "independent_alpha_or_risk_matching_claimed": False,
        "qualification_policy_changed": False,
        "existing_forward_rules_changed": False,
        "order_authority": False,
        "investment_objective_verified": False,
    }
    if (
        not isinstance(policy, dict)
        or any(
            policy.get(key) != value or (type(value) is bool and policy.get(key) is not value)
            for key, value in constants.items()
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError(
            "The complete equity-budget control scope, costs or interpretation changed."
        )


def spy_budget_control(targets: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(targets, pd.DataFrame)
        or not targets.index.is_unique
        or not targets.index.is_monotonic_increasing
        or not targets.columns.is_unique
        or "SPY" not in targets
        or "BIL" not in targets
        or not set(targets.columns).issubset({*EQUITY, *DEFENSE})
        or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
    ):
        raise QuantError(
            "Budget attribution requires classified assets and complete original target rows."
        )
    active = targets.dropna(how="all")
    if (
        active.empty
        or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any()
        or (active.sum(axis=1) > 1 + 1e-10).any()
    ):
        raise QuantError("Budget controls require finite, long-only, funded original targets.")
    equities = [symbol for symbol in targets.columns if symbol in EQUITY]
    result = targets.copy()
    result.loc[active.index, equities] = 0.0
    result.loc[active.index, "SPY"] = active[equities].sum(axis=1)
    if (
        not result.isna().equals(targets.isna())
        or not np.allclose(
            result.loc[active.index].sum(axis=1),
            active.sum(axis=1),
            rtol=0,
            atol=1e-12,
        )
        or not result.loc[:, targets.columns.intersection(DEFENSE)].equals(
            targets.loc[:, targets.columns.intersection(DEFENSE)],
        )
    ):
        raise QuantError("The SPY control changed an original budget, defensive target or pause.")
    return result


def frozen_inputs(policy: dict) -> tuple[dict, list[dict]]:
    validate_policy(policy)
    state = read_json(
        safe_file(
            ROOT,
            policy["source_snapshot"]["path"],
            policy["source_snapshot"]["sha256"],
        )
    )
    audit = read_json(
        safe_file(
            ROOT,
            policy["source_bound_audit"]["path"],
            policy["source_bound_audit"]["sha256"],
        )
    )
    safe_file(ROOT, policy["prior_art"]["path"], policy["prior_art"]["sha256"])
    complete = state["candidate_reviews"]
    ids = [row["candidate_id"] for row in complete]
    if (
        len(ids) != 36
        or len(set(ids)) != 36
        or state["total_evaluated_configurations"] != 120
        or state["pending_candidate_ids"]
        or state["research_champion"] is not None
        or audit["reviewed_candidates"] != 36
        or audit["regenerated_strategy_paths"] != 144
        or audit["event_chain_sha256"] != state["event_chain_sha256"]
        or audit["engine_sha256"] != engine_fingerprint()
        or audit["ledger_unchanged"] is not True
        or [row["candidate_id"] for row in audit["reviews"]] != ids
    ):
        raise QuantError("The exact complete source-bound cohort or live research engine changed.")
    specs = {spec["id"]: spec for spec in state["registered_candidates"]}
    entries = []
    for review in complete:
        spec = specs[review["candidate_id"]]
        bundle, reference = original_bundle(review)
        complete_paths(bundle)
        complete_paths(review)
        if (
            bundle["candidate_spec_sha256"] != digest_json(spec)
            or bundle["market"] != spec["market"]
            or bundle["as_of"] != policy["as_of"]
            or bundle.get("order_authority") is not False
            or any(
                type(value) not in (float, int) or value != 1
                for value in spec["asset_leverage"].values()
            )
            or not set(spec["asset_leverage"]).issubset({*EQUITY, *DEFENSE})
        ):
            raise QuantError(
                "A descriptive control cannot substitute a source, universe or candidate."
            )
        verify_references(ROOT, bundle)
        for path, digest in spec["frozen_files"].items():
            safe_file(ROOT, path, digest)
        entries.append(
            {"spec": spec, "review": review, "bundle": bundle, "bundle_reference": reference}
        )
    return state, entries


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json(
        {
            name: file_digest(folder / name)
            for name in (
                "equity_budget_attribution.py",
                "cash_funded_accounting_v2.py",
                "backtest.py",
                "bt_audit.py",
                "factor_validation.py",
                "selection_validation.py",
                "research_weekly.py",
                "candidate_receipts.py",
                "review_receipts.py",
                "data.py",
                "calendar.py",
                "storage.py",
            )
        }
    )


def program() -> ResearchProgram:
    return ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )


def register(path: Path) -> dict:
    path = local_path(ROOT, path)
    if path.suffix != ".json" or not path.is_relative_to(ROOT / "evidence"):
        raise QuantError("Use a new explicit metadata-only diagnostic registration under evidence.")
    policy = read_json(POLICY)
    state, entries = frozen_inputs(policy)
    with_program = program()
    try:
        if with_program.status() != state:
            raise QuantError("The live research cohort changed before control preregistration.")
        registration = {
            "schema_version": 1,
            "registered_at": utc_now(),
            "policy": policy,
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "included_candidate_ids": [entry["spec"]["id"] for entry in entries],
            "original_bundle_references": [entry["bundle_reference"] for entry in entries],
            "accounting_engine_for_both_descriptive_legs": engine_reference(),
            "new_control_accounting_outcomes_computed": False,
            "history_already_exposed": True,
            "derived_controls_are_registered_trading_candidates": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
            "investment_objective_verified": False,
        }
        write_new_receipt(path, registration)
        actual, _ = input_reference(ROOT, path)
        if actual != registration or with_program.status() != state:
            raise QuantError("The diagnostic preregistration was not durably preserved.")
        return registration
    finally:
        with_program.close()


def paired_accounts(
    data,
    targets: pd.DataFrame,
    start: str,
    end: str,
    cost: float,
    delay: int,
) -> tuple[dict, dict]:
    if (cost, delay) not in ((5.0, 1), (20.0, 2)):
        raise QuantError(
            "Paired budget accounts must retain the original base or stress execution."
        )
    control = spy_budget_control(targets)
    accounts, metadata = {}, {}
    for name, signals in (
        ("source_rule_funded_v2", targets),
        ("same_equity_budget_spy_control", control),
    ):
        account = simulate(
            data,
            signals,
            start,
            end,
            initial_capital=10000,
            cost_bps=cost,
            commission=1,
            delay=delay,
        )
        independent = independent_equity(
            data,
            signals,
            start,
            end,
            capital=10000,
            cost_bps=cost,
            commission=1,
            delay=delay,
        )
        metrics, error = independent_metrics(
            account.frame,
            independent,
            data.risk_free.loc[account.frame.index],
            10000,
        )
        accounts[name] = (signals, account, independent)
        metadata[name] = {
            "metrics": metrics,
            "max_independent_equity_error_usd": error,
            "total_cost_usd": float(account.frame["cost"].sum()),
            "total_orders": int(account.frame["orders"].sum()),
            "minimum_cash_usd": float(account.frame["cash"].min()),
        }
    original, replacement = metadata.values()
    metadata["source_minus_budget_control"] = {
        key: original["metrics"][key] - replacement["metrics"][key]
        if original["metrics"][key] is not None and replacement["metrics"][key] is not None
        else None
        for key in ("cagr", "sharpe", "max_drawdown")
    }
    return accounts, metadata


def published_inputs(reference: dict, entries: list[dict], policy: dict, commit: str) -> dict:
    required = {
        POLICY.relative_to(ROOT).as_posix(),
        *(
            f"src/us_quant/{name}"
            for name in (
                "equity_budget_attribution.py",
                "cash_funded_accounting_v2.py",
                "backtest.py",
                "bt_audit.py",
                "factor_validation.py",
                "selection_validation.py",
                "research_weekly.py",
                "candidate_receipts.py",
                "review_receipts.py",
                "data.py",
                "calendar.py",
                "storage.py",
            )
        ),
    }
    required.update(policy[key]["path"] for key in REFERENCES)
    for entry in entries:
        required.update(
            path
            for path in entry["spec"]["frozen_files"]
            if path.startswith(("src/", "config/", "evidence/"))
        )
    repo = ROOT.parents[1]
    prefix = ROOT.relative_to(repo).as_posix()
    allowlist = read_json(repo / "config/publication_allowlist.json")
    manifest_path = allowlist["bundle_manifest"]
    manifest = read_json(safe_file(repo, manifest_path, allowlist["bundle_manifest_sha256"]))
    imported = {f"{prefix}/{path}" for path in manifest["file_sha256"]}
    explicit = {path for path in required if f"{prefix}/{path}" in allowlist["files"]}
    bundled = required - explicit
    if any(f"{prefix}/{path}" not in imported for path in bundled):
        raise QuantError("A required frozen diagnostic dependency is not approved for publication.")
    publication = verify_publication(ROOT, reference, explicit, commit)
    for path in [manifest_path, *(f"{prefix}/{path}" for path in sorted(bundled))]:
        actual = safe_file(repo, path)
        if digest_bytes(git_read(ROOT, ["show", f"{commit}:{path}"])) != file_digest(actual):
            raise QuantError("The actual published import manifest or core dependency differs.")
    publication["verified_imported_core_file_count"] = len(bundled)
    publication["published_import_manifest_verified"] = True
    return publication


def run(registration_path: Path, output: Path, published_commit: str) -> dict:
    registration, reference = input_reference(ROOT, registration_path)
    policy = read_json(POLICY)
    state, entries = frozen_inputs(policy)
    if (
        registration.get("policy") != policy
        or registration.get("policy_sha256") != file_digest(POLICY)
        or registration.get("auditor_sha256") != fingerprint()
        or registration.get("engine_sha256") != engine_fingerprint()
        or registration.get("event_chain_sha256") != state["event_chain_sha256"]
        or registration.get("new_control_accounting_outcomes_computed") is not False
        or registration.get("included_candidate_ids") != [entry["spec"]["id"] for entry in entries]
        or registration.get("original_bundle_references")
        != [entry["bundle_reference"] for entry in entries]
        or registration.get("accounting_engine_for_both_descriptive_legs") != engine_reference()
        or registration.get("history_already_exposed") is not True
        or registration.get("derived_controls_are_registered_trading_candidates") is not False
        or registration.get("order_authority") is not False
        or registration.get("investment_objective_verified") is not False
        or not max(evidence_time(entry["review"]["at"]) for entry in entries)
        <= evidence_time(registration.get("registered_at"))
        <= pd.Timestamp(utc_now())
    ):
        raise QuantError(
            "The source/control comparison must keep its exact preregistration and code."
        )
    publication = published_inputs(reference, entries, policy, published_commit)
    output = local_path(ROOT, output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports"):
        raise QuantError(
            "Use a new isolated private output directory for descriptive control accounts."
        )
    with_program = program()
    created = False
    result = None
    try:
        if with_program.status() != state:
            raise QuantError("The actual research cohort changed after diagnostic registration.")
        new_output_directory(output)
        created = True
        write_json(
            output / "publication.json",
            {
                **publication,
                "verified_on_remote_at": utc_now(),
                "registration_predates_new_control_accounts": True,
                "order_authority": False,
                "investment_objective_verified": False,
            },
        )
        result = {
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "registration_sha256": reference["sha256"],
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "published_registration_commit": published_commit,
            "candidate_count": len(entries),
            "paired_comparisons": [],
            "derived_control_portfolios": 36,
            "control_leg_accounts": 0,
            "reaccounted_original_leg_accounts": 0,
            "paired_accounting_engine": engine_reference(),
            "new_strategy_evaluations": 0,
            "new_registered_candidates": 0,
            "same_budget_is_same_beta_or_risk": False,
            "descriptive_control_is_independent_strategy_validation": False,
            "order_authority": False,
            "investment_objective_verified": False,
        }
        write_json(output / "progress.json", result)
        for entry in entries:
            spec, review, bundle = entry["spec"], entry["review"], entry["bundle"]
            data = review_market(bundle, ROOT)
            cache = {}
            for years in (10, 5):
                original = verified_accounts(bundle, review, years, policy)
                for scenario in ("base", "stress"):
                    result["current_comparison"] = {
                        "candidate_id": spec["id"],
                        "years": years,
                        "scenario": scenario,
                    }
                    write_json(output / "progress.json", result)
                    path = next(
                        row
                        for row in bundle["paths"]
                        if (row["years"], row["scenario"]) == (years, scenario)
                    )
                    retained = pd.read_csv(
                        safe_file(ROOT, path["targets"]["path"], path["targets"]["sha256"]),
                        index_col=0,
                        parse_dates=True,
                        float_precision="round_trip",
                    )
                    recorded_metrics = next(
                        row["metrics"]
                        for row in review["paths"]
                        if (row["years"], row["scenario"]) == (years, scenario)
                    )
                    dates = original[scenario]["strategy"].index
                    start, end = str(dates[0].date()), str(dates[-1].date())
                    generated = registered_targets(
                        spec,
                        data,
                        start,
                        end,
                        path["cost_bps"],
                        path["delay_sessions"],
                        ROOT,
                        cache,
                    )
                    if (
                        not retained.index.equals(generated.index)
                        or not retained.columns.equals(generated.columns)
                        or not retained.isna().equals(generated.isna())
                        or not np.allclose(retained, generated, equal_nan=True, rtol=0, atol=1e-10)
                    ):
                        raise QuantError(
                            "A regenerated target differs from its original reviewed evidence."
                        )
                    accounts, metrics = paired_accounts(
                        data,
                        retained,
                        start,
                        end,
                        path["cost_bps"],
                        path["delay_sessions"],
                    )
                    rebased = accounts["source_rule_funded_v2"][1].frame
                    original_frame = original[scenario]["strategy"]
                    if not rebased.index.equals(original_frame.index):
                        raise QuantError("Reaccounting shortened the original comparison calendar.")
                    metrics["preserved_original_review"] = {
                        "accounting_engine": bundle.get("accounting_engine"),
                        "legacy_unversioned_accounting": "accounting_engine" not in bundle,
                        "metrics": recorded_metrics,
                        "maximum_equity_difference_from_paired_funded_v2_usd": float(
                            abs(rebased["equity"] - original_frame["equity"]).max()
                        ),
                        "review_body_or_ledger_rewritten": False,
                    }
                    for name, (_, account, independent) in accounts.items():
                        prefix = output / spec["id"] / f"{years}y-{scenario}-{name}"
                        for suffix, frame in (
                            ("account", account.frame),
                            ("bt", independent),
                        ):
                            write_text_atomic(
                                prefix.with_name(prefix.name + f"-{suffix}.csv"),
                                frame.to_csv(float_format="%.17g"),
                            )
                    control_path = output / spec["id"] / f"{years}y-{scenario}-control-targets.csv"
                    write_text_atomic(
                        control_path,
                        accounts["same_equity_budget_spy_control"][0].to_csv(float_format="%.17g"),
                    )
                    result["paired_comparisons"].append(
                        {
                            "candidate_id": spec["id"],
                            "years": years,
                            "scenario": scenario,
                            **metrics,
                        }
                    )
                    result["control_leg_accounts"] += 1
                    result["reaccounted_original_leg_accounts"] += 1
                    write_json(output / "progress.json", result)
        if (
            len(result["paired_comparisons"]) != 144
            or result["control_leg_accounts"] != 144
            or result["reaccounted_original_leg_accounts"] != 144
            or with_program.status() != state
        ):
            raise QuantError(
                "The descriptive budget attribution omitted paths or changed prior research."
            )
        result.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "all_original_reviews_and_research_state_unchanged": True,
                "artifact_sha256": {
                    path.relative_to(output).as_posix(): file_digest(path)
                    for path in sorted(output.rglob("*.csv"))
                },
            }
        )
        write_json(output / "results.json", result)
        return result
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        if created:
            write_json(
                output / "failure.json",
                {
                    "recorded_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "completed_comparisons_retained": (
                        len(result["paired_comparisons"]) if result is not None else 0
                    ),
                    "current_comparison": result.get("current_comparison")
                    if result is not None
                    else None,
                    "partial_outputs_deleted_or_filled": False,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
        raise
    finally:
        with_program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister descriptive equity-budget controls; never orders."
    )
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path(
            "evidence/equity_budget_attribution_20261011_registration.json",
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/equity-budget-attribution-20261011")
    )
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        if args.action == "register":
            result = register(args.registration)
            print(
                json.dumps(
                    {
                        "registered_at": result["registered_at"],
                        "candidate_count": len(result["included_candidate_ids"]),
                        "new_control_accounting_outcomes_computed": False,
                        "investment_objective_verified": False,
                    },
                    indent=2,
                )
            )
        else:
            result = run(args.registration, args.output, args.published_commit)
            print(
                json.dumps(
                    {
                        "descriptive_control_portfolios": result["derived_control_portfolios"],
                        "paired_horizon_scenario_comparisons": len(result["paired_comparisons"]),
                        "control_leg_accounts": result["control_leg_accounts"],
                        "reaccounted_original_leg_accounts": result[
                            "reaccounted_original_leg_accounts"
                        ],
                        "new_strategy_evaluations": 0,
                        "investment_objective_verified": False,
                    },
                    indent=2,
                )
            )
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(
            2,
            f"Equity-budget attribution blocked; preserve all original accounts: {exc}\n",
        )


if __name__ == "__main__":
    main()
