from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.selection_validation import (
    fingerprint as original_fingerprint,
)
from us_quant.selection_validation import (
    joint_maximum,
    original_bundle,
    verified_accounts,
)
from us_quant.selection_validation import (
    validate_policy as validate_original_policy,
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

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/selection-expansion.json"
REFERENCES = (
    "source_snapshot",
    "source_bound_audit",
    "original_policy",
    "original_registration",
    "original_results",
    "original_reproducibility",
)


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "audit_id": "recurring_selection_expansion_20261011",
        "scope": "completed_recurring_reviews_expanded_only",
        "as_of": "2026-10-05",
        "included_candidate_count": 36,
        "original_included_candidate_count": 22,
        "earlier_configurations_not_in_joint_inference": 84,
        "total_disclosed_configurations": 120,
        "horizons_years": [10, 5],
        "cost_comparisons": [
            {"strategy_scenario": "base", "spy_scenario": "base"},
            {"strategy_scenario": "stress", "spy_scenario": "stress"},
            {"strategy_scenario": "stress", "spy_scenario": "base"},
        ],
        "capital_usd": 10000.0,
        "bootstrap_samples": 4000,
        "block_sessions": 21,
        "seed": 20261011,
        "familywise_alpha": 0.05,
        "annualization_sessions": 252,
        "order_authority": False,
        "change_research_qualification_policy": False,
        "independent_forward_validation": False,
    }
    if (
        any(policy.get(key) != value for key, value in constants.items())
        or any(
            policy.get(key) is not False
            for key in (
                "order_authority",
                "change_research_qualification_policy",
                "independent_forward_validation",
            )
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError(
            "The fixed expanded selection scope, cost comparisons or settings changed."
        )


def validate_cohort(state: dict, audit: dict, original_ids: list, policy: dict) -> None:
    ids = [row["candidate_id"] for row in state["candidate_reviews"]]
    if (
        len(ids) != 36
        or len(set(ids)) != 36
        or len(original_ids) != 22
        or ids[:22] != original_ids
        or state["total_evaluated_configurations"] != 120
        or state["pending_candidate_ids"]
        or audit["reviewed_candidates"] != 36
        or audit["regenerated_strategy_paths"] != 144
        or audit["ledger_unchanged"] is not True
        or audit["event_chain_sha256"] != state["event_chain_sha256"]
        or {row["candidate_id"] for row in audit["reviews"]} != set(ids)
        or len(policy["cost_comparisons"]) * len(ids) != 108
    ):
        raise QuantError(
            "The complete expanded cohort differs from its exact audited original prefix."
        )


def source_state(policy: dict) -> tuple[dict, dict]:
    validate_policy(policy)
    values = {
        key: read_json(safe_file(ROOT, policy[key]["path"], policy[key]["sha256"]))
        for key in REFERENCES
    }
    state, audit = values["source_snapshot"], values["source_bound_audit"]
    validate_original_policy(values["original_policy"])
    original = values["original_registration"]
    if original["policy"] != values["original_policy"] or (
        original["auditor_sha256"] != original_fingerprint()
    ):
        raise QuantError(
            "The original selection policy or unchanged statistical implementation changed."
        )
    for key in (
        "as_of",
        "horizons_years",
        "cost_comparisons",
        "capital_usd",
        "bootstrap_samples",
        "block_sessions",
        "seed",
        "familywise_alpha",
        "annualization_sessions",
    ):
        if policy[key] != values["original_policy"][key]:
            raise QuantError(
                "The expanded diagnostic cannot retune the inherited inference settings."
            )
    validate_cohort(state, audit, original["included_candidate_ids"], policy)
    if audit["engine_sha256"] != engine_fingerprint():
        raise QuantError("The current source-bound audit does not match the live research engine.")
    return state, values


def fingerprint() -> str:
    return digest_json(
        {
            "expanded_adapter": file_digest(Path(__file__)),
            "unchanged_statistical_and_source_implementation": original_fingerprint(),
        }
    )


def register(output: Path) -> dict:
    policy = read_json(POLICY)
    state, values = source_state(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Selection registration must stay in the isolated workbench.")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        if program.status() != state:
            raise QuantError("The live cohort changed; register a new explicit diagnostic version.")
        new_output_directory(output)
        result = {
            "schema_version": 1,
            "registered_at": utc_now(),
            "policy": policy,
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "included_candidate_ids": [row["candidate_id"] for row in state["candidate_reviews"]],
            "original_prefix_candidate_ids": values["original_registration"][
                "included_candidate_ids"
            ],
            "included_comparisons_per_horizon": 108,
            "old_selection_artifacts_sha256": {
                key: policy[key]["sha256"] for key in REFERENCES if key.startswith("original_")
            },
            "joint_statistical_outcomes_computed": False,
            "settings_frozen_before_joint_resampling": True,
            "candidate_history_already_exposed": True,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        write_json(output / "registration.json", result)
        return result
    finally:
        program.close()


def matrices(state: dict, policy: dict) -> tuple[dict, dict]:
    sources, result = {}, {}
    for years in policy["horizons_years"]:
        differences = {}
        for review in state["candidate_reviews"]:
            name = review["candidate_id"]
            bundle, source = original_bundle(review)
            sources[name] = source
            accounts = verified_accounts(bundle, review, years, policy)
            for scenario in policy["cost_comparisons"]:
                own, spy = scenario["strategy_scenario"], scenario["spy_scenario"]
                strategy, benchmark = accounts[own]["strategy"], accounts[spy]["spy"]
                if (strategy["return"] <= -1).any() or (benchmark["return"] <= -1).any():
                    raise QuantError("Insolvent observations cannot enter log-growth inference.")
                differences[f"{name}::{own}_vs_{spy}"] = np.log1p(strategy["return"]) - np.log1p(
                    benchmark["return"]
                )
        frame = pd.DataFrame(differences)
        if frame.shape[1] != 108 or not np.isfinite(frame.to_numpy()).all():
            raise QuantError("The expanded inference matrix must contain every exact comparison.")
        result[str(years)] = frame
    return result, sources


def verify_original_prefix(actual: dict, expected: dict) -> None:
    if actual["comparisons"] != expected["comparisons"] or (
        len(actual["comparison_results"]) != len(expected["comparison_results"])
    ):
        raise QuantError("The original selection prefix shape changed.")
    for key in (
        "scope_limited_omnibus_p_value",
        "positive_part_maximum_daily_log_advantage",
        "bootstrap_maximum_critical_daily_log_advantage",
    ):
        if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-15):
            raise QuantError(
                "The original selection outcome did not reproduce on its original returns."
            )
    for current, prior in zip(
        actual["comparison_results"], expected["comparison_results"], strict=True
    ):
        if current["comparison_id"] != prior["comparison_id"]:
            raise QuantError("The original comparison ordering changed.")
        for key in (
            "observed_mean_daily_log_advantage",
            "simultaneous_lower_mean_daily_log_advantage",
        ):
            if not np.isclose(current[key], prior[key], rtol=0, atol=1e-15):
                raise QuantError("An original selection mean or confidence bound changed.")


def run(registration_path: Path, output: Path) -> dict:
    registration_path = (
        registration_path if registration_path.is_absolute() else ROOT / registration_path
    )
    registration = read_json(safe_file(ROOT, registration_path.relative_to(ROOT).as_posix()))
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Expanded selection evidence must remain inside the workbench.")
    policy = read_json(POLICY)
    state, values = source_state(policy)
    if (
        registration["policy"] != policy
        or registration["policy_sha256"] != file_digest(POLICY)
        or registration["auditor_sha256"] != fingerprint()
        or registration["program_engine_sha256"] != engine_fingerprint()
        or registration["event_chain_sha256"] != state["event_chain_sha256"]
        or registration["included_candidate_ids"]
        != [row["candidate_id"] for row in state["candidate_reviews"]]
        or pd.Timestamp(registration["registered_at"]).tzinfo is None
        or pd.Timestamp(registration["registered_at"]) > pd.Timestamp(utc_now())
        or registration["joint_statistical_outcomes_computed"] is not False
        or registration["settings_frozen_before_joint_resampling"] is not True
    ):
        raise QuantError("Use the exact actual preregistration and unchanged expanded cohort.")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        before = program.status()
        if before != state:
            raise QuantError("The live cohort changed after diagnostic registration.")
        frames, sources = matrices(state, policy)
        analyses, prefixes = {}, {}
        for offset, years in enumerate(policy["horizons_years"]):
            frame = frames[str(years)]
            parameters = {
                "samples": 4000,
                "block": 21,
                "seed": 20261011 + offset,
                "alpha": 0.025,
            }
            prefix = joint_maximum(frame.iloc[:, :66], **parameters)
            verify_original_prefix(prefix, values["original_results"]["analyses"][str(years)])
            analysis = joint_maximum(frame, **parameters)
            if (
                analysis["bootstrap_maximum_critical_daily_log_advantage"] + 1e-15
                < (prefix["bootstrap_maximum_critical_daily_log_advantage"])
            ):
                raise QuantError(
                    "The synchronous expanded cohort cannot narrow the old maximum bound."
                )
            analyses[str(years)], prefixes[str(years)] = analysis, prefix
        if program.status() != before:
            raise QuantError("A read-only inference diagnostic cannot change historical research.")
        result = {
            "schema_version": 1,
            "created_at": utc_now(),
            "audit_id": policy["audit_id"],
            "registration": {
                "path": registration_path.relative_to(ROOT).as_posix(),
                "sha256": file_digest(registration_path),
            },
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": before["event_chain_sha256"],
            "analyses": analyses,
            "original_prefix_replays": prefixes,
            "included_candidate_count": 36,
            "included_comparisons_per_horizon": 108,
            "original_prefix_candidate_count": 22,
            "earlier_configurations_excluded_from_joint_inference": 84,
            "total_disclosed_configurations": 120,
            "bonferroni_omnibus_p_value_for_scoped_family": min(
                1.0, 2 * min(row["scope_limited_omnibus_p_value"] for row in analyses.values())
            ),
            "source_bundles": sources,
            "old_prefix_outcomes_reproduced_and_artifacts_unchanged": True,
            "overlapping_horizons_assumed_independent": False,
            "scope_limitations": policy["methodology"]["limits"],
            "ledger_unchanged": True,
            "full_search_selection_adjusted_alpha_verified": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        new_output_directory(output)
        for years, frame in frames.items():
            write_text_atomic(
                output / f"{years}y-log-growth-differences.csv", frame.to_csv(float_format="%.17g")
            )
        write_json(output / "results.json", result)
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand conditional selection inference; never orders."
    )
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "data/selection-expansion-20261011/registration.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "data/selection-expansion-20261011"
        if args.action == "register"
        else "reports/selection-expansion-20261011"
    )
    try:
        result = register(output) if args.action == "register" else run(args.registration, output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in (
                        "policy",
                        "analyses",
                        "original_prefix_replays",
                        "source_bundles",
                        "included_candidate_ids",
                        "original_prefix_candidate_ids",
                    )
                },
                indent=2,
            )
        )
        if args.action == "run":
            for years, analysis in result["analyses"].items():
                print(
                    json.dumps(
                        {
                            "years": years,
                            "comparisons": analysis["comparisons"],
                            "scoped_omnibus_p_value": analysis["scope_limited_omnibus_p_value"],
                            "positive_simultaneous_lower_bounds": sum(
                                row["strictly_positive_simultaneous_lower_bound"]
                                for row in analysis["comparison_results"]
                            ),
                        }
                    )
                )
    except QuantError as exc:
        parser.exit(2, f"Expanded selection audit blocked: {exc}\n")


if __name__ == "__main__":
    main()
