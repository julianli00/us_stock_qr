from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/selection-validation.json"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("scope") != "completed_recurring_reviews_only"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("included_candidate_count") != 22
        or policy.get("earlier_configurations_not_in_joint_inference") != 84
        or policy.get("total_disclosed_configurations") != 106
        or policy.get("horizons_years") != [10, 5]
        or policy.get("cost_comparisons")
        != [
            {"strategy_scenario": "base", "spy_scenario": "base"},
            {"strategy_scenario": "stress", "spy_scenario": "stress"},
            {"strategy_scenario": "stress", "spy_scenario": "base"},
        ]
        or policy.get("capital_usd") != 10000.0
        or policy.get("bootstrap_samples") != 4000
        or policy.get("block_sessions") != 21
        or policy.get("seed") != 20261011
        or policy.get("familywise_alpha") != 0.05
        or policy.get("annualization_sessions") != 252
        or any(
            policy.get(key) is not False
            for key in (
                "order_authority",
                "change_research_qualification_policy",
                "independent_forward_validation",
            )
        )
    ):
        raise QuantError("The frozen joint-selection audit scope or statistical settings changed.")


def fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    return digest_json(
        {
            name: file_digest(directory / name)
            for name in (
                "selection_validation.py",
                "factor_validation.py",
                "metrics.py",
                "calendar.py",
                "storage.py",
            )
        }
    )


def circular_positions(
    count: int, samples: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    if (
        any(type(value) is not int for value in (count, samples, block))
        or count < block
        or block < 2
        or samples < 1
    ):
        raise QuantError("Invalid joint circular-block resampling dimensions.")
    starts = rng.integers(0, count, size=(samples, (count + block - 1) // block))
    return ((starts[:, :, None] + np.arange(block)) % count).reshape(samples, -1)[:, :count]


def joint_maximum(
    differences: pd.DataFrame, *, samples: int, block: int, seed: int, alpha: float
) -> dict:
    if (
        type(block) is not int
        or len(differences) < block
        or block < 2
        or type(samples) is not int
        or samples < 100
        or type(seed) is not int
        or not 0 < alpha < 1
        or differences.empty
        or not differences.index.is_unique
        or not differences.index.is_monotonic_increasing
        or not differences.columns.is_unique
        or not all(isinstance(name, str) for name in differences.columns)
        or not np.isfinite(differences.to_numpy(dtype=float)).all()
    ):
        raise QuantError("Joint selection inference needs finite, aligned and unique observations.")
    values = differences.to_numpy(dtype=float)
    means = values.mean(axis=0)
    centered = values - means
    count = len(values)
    rng = np.random.default_rng(seed)
    maxima = []
    for offset in range(0, samples, 64):
        positions = circular_positions(count, min(64, samples - offset), block, rng)
        counts = np.stack([np.bincount(row, minlength=count) for row in positions])
        bootstrap_means = counts @ centered / count
        maxima.extend(np.maximum(0.0, bootstrap_means.max(axis=1)))
    maxima = np.asarray(maxima)
    observed = max(0.0, float(means.max()))
    critical = float(np.quantile(maxima, 1 - alpha, method="higher"))
    return {
        "sessions": count,
        "comparisons": len(differences.columns),
        "bootstrap_samples": samples,
        "block_sessions": block,
        "seed": seed,
        "alpha_allocated_to_horizon": alpha,
        "positive_part_maximum_daily_log_advantage": observed,
        "bootstrap_maximum_critical_daily_log_advantage": critical,
        "scope_limited_omnibus_p_value": float(
            (1 + (maxima >= observed).sum()) / (samples + 1)
        ),
        "monte_carlo_p_value_resolution": 1 / (samples + 1),
        "bootstrap_maxima_sha256": digest_json(maxima.tolist()),
        "comparison_results": [
            {
                "comparison_id": name,
                "observed_mean_daily_log_advantage": float(mean),
                "simultaneous_lower_mean_daily_log_advantage": float(mean - critical),
                "strictly_positive_simultaneous_lower_bound": bool(mean > critical),
            }
            for name, mean in zip(differences.columns, means, strict=True)
        ],
    }


def frozen_state(policy: dict) -> dict:
    validate_policy(policy)
    values = {}
    for key in ("source_snapshot", "source_bound_audit"):
        source = policy[key]
        values[key] = read_json(safe_file(ROOT, source["path"], source["sha256"]))
    state, audit = values["source_snapshot"], values["source_bound_audit"]
    ids = [item["candidate_id"] for item in state["candidate_reviews"]]
    if (
        len(ids) != policy["included_candidate_count"]
        or len(set(ids)) != len(ids)
        or state["total_evaluated_configurations"] != policy["total_disclosed_configurations"]
        or state["pending_candidate_ids"]
        or audit["reviewed_candidates"] != len(ids)
        or audit["regenerated_strategy_paths"] != 4 * len(ids)
        or not audit["ledger_unchanged"]
        or audit["engine_sha256"] != engine_fingerprint()
        or audit["event_chain_sha256"] != state["event_chain_sha256"]
        or {item["candidate_id"] for item in audit["reviews"]} != set(ids)
    ):
        raise QuantError("The registered selection cohort differs from its source-bound audit.")
    return state


def register(output: Path) -> dict:
    policy = read_json(POLICY)
    state = frozen_state(policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Selection registration must remain inside the isolated workbench.")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
    )
    try:
        if program.status() != state:
            raise QuantError("The live research cohort changed; register a new audit version.")
        new_output_directory(output)
        record = {
            "schema_version": 1,
            "registered_at": utc_now(),
            "policy": policy,
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "included_candidate_ids": [
                item["candidate_id"] for item in state["candidate_reviews"]
            ],
            "settings_frozen_before_joint_resampling": True,
            "candidate_history_already_exposed": True,
            "joint_statistical_outcomes_computed": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        write_json(output / "registration.json", record)
        return record
    finally:
        program.close()


def original_bundle(review: dict) -> tuple[dict, dict]:
    for path in sorted((ROOT / "reports").glob(f"*/{review['candidate_id']}/bundle.json")):
        safe_file(ROOT, path.relative_to(ROOT).as_posix())
        bundle = read_json(path)
        if digest_json(bundle) == review["evidence_sha256"]:
            return bundle, {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
    raise QuantError("The original reviewed bundle is unavailable; do not substitute another attempt.")


def verified_accounts(bundle: dict, review: dict, years: int, policy: dict) -> dict:
    expected_pairs = {(year, scenario) for year in (10, 5) for scenario in ("base", "stress")}
    for record in (bundle, review):
        paths = record.get("paths", [])
        if len(paths) != 4 or {(item["years"], item["scenario"]) for item in paths} != expected_pairs:
            raise QuantError("Joint inference needs all four previously reviewed account paths.")
    source = bundle["market"]["risk_free"]
    rates = pd.read_csv(
        safe_file(ROOT, source["path"], source["sha256"]), index_col=0, parse_dates=True
    )
    if list(rates.columns) != ["risk_free"]:
        raise QuantError("The audit needs the frozen market's explicit risk-free series.")
    dates = sessions(
        pd.Timestamp(policy["as_of"]) - pd.DateOffset(years=years) + pd.Timedelta(days=1),
        policy["as_of"],
    )
    result = {}
    for scenario in ("base", "stress"):
        path = next(
            item
            for item in bundle["paths"]
            if (item["years"], item["scenario"]) == (years, scenario)
        )
        prior = next(
            item
            for item in review["paths"]
            if (item["years"], item["scenario"]) == (years, scenario)
        )
        expected_cost, expected_delay = (5, 1) if scenario == "base" else (20, 2)
        if (
            path["capital_usd"] != policy["capital_usd"]
            or path["cost_bps"] != expected_cost
            or path["delay_sessions"] != expected_delay
            or path["commission_per_order"] != 1
        ):
            raise QuantError("Selection inputs cannot change capital, costs or execution delay.")
        frames = {}
        for key in ("strategy", "strategy_bt", "spy", "spy_bt"):
            source = path[key]
            frame = pd.read_csv(
                safe_file(ROOT, source["path"], source["sha256"]),
                index_col=0,
                parse_dates=True,
            )
            if not frame.index.equals(dates):
                raise QuantError("A selection-audit account does not cover the exact frozen sessions.")
            frames[key] = frame
        for own, independent, metric_key in (
            ("strategy", "strategy_bt", "metrics"),
            ("spy", "spy_bt", "benchmark"),
        ):
            actual, _ = independent_metrics(
                frames[own],
                frames[independent],
                rates.loc[dates, "risk_free"],
                policy["capital_usd"],
            )
            check_metrics(actual, prior[metric_key])
        result[scenario] = frames
    return result


def run(registration_path: Path, output: Path) -> dict:
    registration_path = (
        registration_path if registration_path.is_absolute() else ROOT / registration_path
    )
    registration_path = safe_file(ROOT, registration_path.relative_to(ROOT).as_posix())
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Selection results must remain inside the isolated workbench.")
    registration = read_json(registration_path)
    policy = read_json(POLICY)
    state = frozen_state(policy)
    if (
        registration["policy"] != policy
        or registration["policy_sha256"] != file_digest(POLICY)
        or registration["auditor_sha256"] != fingerprint()
        or registration["program_engine_sha256"] != engine_fingerprint()
        or registration["event_chain_sha256"] != state["event_chain_sha256"]
        or registration["included_candidate_ids"]
        != [item["candidate_id"] for item in state["candidate_reviews"]]
        or pd.Timestamp(registration["registered_at"]).tzinfo is None
        or pd.Timestamp(registration["registered_at"]) > pd.Timestamp(utc_now())
        or registration["settings_frozen_before_joint_resampling"] is not True
        or registration["joint_statistical_outcomes_computed"] is not False
    ):
        raise QuantError("Joint inference must use the exact registered cohort, settings and code.")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
    )
    try:
        before = program.status()
        if before != state:
            raise QuantError("The live cohort changed after selection-audit registration.")
        analyses, sources = {}, {}
        for index, years in enumerate(policy["horizons_years"]):
            differences, details = {}, {}
            for review in state["candidate_reviews"]:
                name = review["candidate_id"]
                bundle, source = original_bundle(review)
                sources[name] = source
                accounts = verified_accounts(bundle, review, years, policy)
                for comparison in policy["cost_comparisons"]:
                    own, spy = comparison["strategy_scenario"], comparison["spy_scenario"]
                    label = f"{name}::{own}_vs_{spy}"
                    strategy, benchmark = accounts[own]["strategy"], accounts[spy]["spy"]
                    if (strategy["return"] <= -1).any() or (benchmark["return"] <= -1).any():
                        raise QuantError("Insolvent daily returns cannot enter log-growth inference.")
                    differences[label] = np.log1p(strategy["return"]) - np.log1p(
                        benchmark["return"]
                    )
                    details[label] = {"candidate_id": name, **comparison}
            summary = joint_maximum(
                pd.DataFrame(differences),
                samples=policy["bootstrap_samples"],
                block=policy["block_sessions"],
                seed=policy["seed"] + index,
                alpha=policy["familywise_alpha"] / len(policy["horizons_years"]),
            )
            for row in summary["comparison_results"]:
                row.update(details[row["comparison_id"]])
                row["annualized_252_log_growth_advantage"] = (
                    row["observed_mean_daily_log_advantage"] * policy["annualization_sessions"]
                )
                row["simultaneous_lower_annualized_252_log_growth_advantage"] = (
                    row["simultaneous_lower_mean_daily_log_advantage"]
                    * policy["annualization_sessions"]
                )
            analyses[str(years)] = summary
        if program.status() != before:
            raise QuantError(
                "The read-only selection audit cannot modify or race changed research state."
            )
        result = {
            "schema_version": 1,
            "created_at": utc_now(),
            "audit_id": policy["audit_id"],
            "registration": {
                "path": registration_path.relative_to(ROOT).as_posix(),
                "sha256": file_digest(registration_path),
            },
            "auditor_sha256": fingerprint(),
            "event_chain_sha256": before["event_chain_sha256"],
            "analyses": analyses,
            "included_candidate_count": len(before["candidate_reviews"]),
            "included_comparisons_per_horizon": len(before["candidate_reviews"])
            * len(policy["cost_comparisons"]),
            "overlapping_horizons_assumed_independent": False,
            "bonferroni_omnibus_p_value_for_scoped_family": min(
                1.0,
                len(analyses)
                * min(item["scope_limited_omnibus_p_value"] for item in analyses.values()),
            ),
            "earlier_configurations_excluded_from_joint_inference": policy[
                "earlier_configurations_not_in_joint_inference"
            ],
            "total_disclosed_configurations": before["total_evaluated_configurations"],
            "source_bundles": sources,
            "ledger_unchanged": True,
            "original_qualification_policy_unchanged": True,
            "scope_limitations": policy["methodology"]["selection_limits"],
            "full_search_selection_adjusted_alpha_verified": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        new_output_directory(output)
        write_json(output / "results.json", result)
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only joint selection diagnostics; no orders.")
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "data/selection-validation-20261011/registration.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "data/selection-validation-20261011"
        if args.action == "register"
        else "reports/selection-validation-20261011"
    )
    try:
        result = register(output) if args.action == "register" else run(args.registration, output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("analyses", "source_bundles", "policy", "included_candidate_ids")
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
        parser.exit(2, f"Selection audit blocked: {exc}\n")


if __name__ == "__main__":
    main()
