from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dollar_risk_guard import decision_pairs, dollar_comparisons, load_vintages
from us_quant.jobless_claims_inputs import (
    fingerprint as input_fingerprint,
)
from us_quant.jobless_claims_inputs import (
    inspect_queries,
    native_counts,
    parse_archive,
    parse_initial_archive,
    request_fields,
)
from us_quant.jobless_claims_inputs import (
    validate_policy as validate_input_policy,
)
from us_quant.research_program import review_market, safe_file, verified_etf_market
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
POLICY = ROOT / "config/claims-factor-guard.json"
CONTROL = ROOT / "config/sector-growth-balance.json"
DOLLAR_POLICY = ROOT / "config/dollar-risk-guard.json"
INPUT_POLICY = ROOT / "config/jobless-claims-inputs.json"
INPUT_SOURCE = ROOT / "data/claims-inputs-20261011"
INPUT_READINESS = ROOT / "evidence/jobless_claims_input_readiness_20261011.json"
CONTEXT = ROOT / "data/claims-confirmation-context-20261011"
PRIOR = ROOT / "evidence/dollar_risk_guard_20261011_registration.json"
CANDIDATE = {
    "id": "semiconductor_dollar_claims_confirmation",
    "confirmation": "four_week_claims_mean_increased_since_63_nyse_sessions",
}
INCOMPLETE_WINDOW = {
    "query_session": "2018-11-26",
    "vintage_cutoff": "2018-11-21",
    "latest_observation_date": "2018-11-17",
    "missing_observation_dates": ["2018-10-27"],
}
REVISION = {
    "original_unregistered_policy": (
        "reports/claims-confirmation-plan-20261011/availability-rejected-policy-v1.json"
    ),
    "original_unregistered_implementation": (
        "reports/claims-confirmation-plan-20261011/availability-rejected-implementation-v1.py"
    ),
    "original_source_failure": (
        "reports/claims-confirmation-plan-20261011/source-window-failure-v1.json"
    ),
    "portfolio_outcomes_computed_before_revision": False,
    "source_age_limit_relaxed": False,
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
        "claims_smoothing_weeks": 4,
        "claims_change_threshold": 0.0,
        "risk_off_investment_scale": 0.50,
        "cash_reserve": 0.02,
        "stale_input_behavior": "pause_rebalance_and_record_gap",
        "known_incomplete_window": INCOMPLETE_WINDOW,
        "data_only_revision": REVISION,
        "candidates": [CANDIDATE],
        "order_authority": False,
        "automatic_live_deployment": False,
        "existing_forward_rules_changed": False,
        "new_factor_independence_claimed": False,
        "source_age_limit_relaxed": False,
    }
    if any(
        policy.get(key) != value or (type(value) is bool and policy.get(key) is not value)
        for key, value in expected.items()
    ):
        raise QuantError("The single claims confirmation, source standby or cycle gate changed.")


def four_week_queries(
    index: pd.DatetimeIndex,
    vintages: dict[str, pd.Series],
    initial_releases: pd.DataFrame,
) -> pd.DataFrame:
    selected = inspect_queries(index, vintages, initial_releases)
    means, blockers = [], []
    for query, row in selected.iterrows():
        if not row["source_ready"]:
            means.append(np.nan)
            blockers.append(row["blocker"])
            continue
        observed = pd.Timestamp(row["observation_date"])
        weeks = pd.date_range(observed - pd.Timedelta(days=21), observed, freq="W-SAT")
        values = native_counts(vintages[row["vintage_cutoff"]]).reindex(weeks)
        missing = values.index[values.isna()].strftime("%Y-%m-%d").tolist()
        if missing:
            actual = {
                "query_session": query,
                "vintage_cutoff": row["vintage_cutoff"],
                "latest_observation_date": row["observation_date"],
                "missing_observation_dates": missing,
            }
            if actual == INCOMPLETE_WINDOW:
                means.append(np.nan)
                blockers.append("recorded_missing_four_week_component_2018_10_27")
                continue
        if len(values) != 4 or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
            raise QuantError("Four-week claims means require four actual consecutive source weeks.")
        means.append(float(values.mean()))
        blockers.append(None)
    selected["claims_four_week_mean"] = means
    selected["claims_mean_blocker"] = blockers
    selected.index = pd.to_datetime(selected.index)
    return selected


def load_inputs(data: MarketData, policy: dict) -> tuple[pd.DataFrame, dict]:
    validate_policy(policy)
    context_files()
    validate_input_policy(read_json(INPUT_POLICY))
    proof = read_json(INPUT_READINESS)
    manifest = read_json(
        safe_file(
            ROOT,
            proof["source_manifest"],
            proof["source_manifest_sha256"],
        )
    )
    if (
        proof["policy_sha256"] != file_digest(INPUT_POLICY)
        or proof["readiness_code_sha256"] != input_fingerprint()
        or manifest["policy_sha256"] != file_digest(INPUT_POLICY)
        or manifest["all_requested_batches_complete"] is not True
        or manifest["initial_release_batches_complete"] is not True
        or proof["strategy_states_targets_or_returns_computed"] is not False
        or proof["all_required_queries_ready"] is not False
    ):
        raise QuantError(
            "Use the exact prior claims source proof with its explicit freshness blocks."
        )
    for name, digest in manifest["source_files"].items():
        safe_file(INPUT_SOURCE, name, digest)
    vintages, initial = {}, []
    for first in (False, True):
        key = "initial_release_batches" if first else "batches"
        for batch in manifest[key]:
            dates = batch["vintage_dates"]
            request = read_json(
                safe_file(
                    INPUT_SOURCE,
                    batch["request_path"],
                    batch["request_sha256"],
                )
            )
            if (
                request["advertised_form_fields"] != request_fields(dates, initial_release=first)
                or request["http_success"] is not True
                or request["sha256"] != batch["sha256"]
            ):
                raise QuantError("The frozen claims source request or archive changed.")
            path = safe_file(INPUT_SOURCE, batch["path"], batch["sha256"])
            if first:
                initial.append(parse_initial_archive(path, dates))
            else:
                if set(vintages).intersection(dates):
                    raise QuantError("A historical claims snapshot was duplicated.")
                frame = parse_archive(path, dates)
                vintages.update({day: frame[day].dropna() for day in dates})
    selected = four_week_queries(
        data.close.index,
        vintages,
        pd.concat(initial).sort_index(),
    )
    blocked = selected.index[~selected["source_ready"]].strftime("%Y-%m-%d").tolist()
    if blocked != [row["query_session"] for row in proof["actual_frozen_freshness_blocks"]]:
        raise QuantError(
            "A source gap changed; the frozen standby policy cannot absorb unknown gaps."
        )
    missing_window = (
        selected.index[
            selected["claims_mean_blocker"].eq("recorded_missing_four_week_component_2018_10_27")
        ]
        .strftime("%Y-%m-%d")
        .tolist()
    )
    if missing_window != [INCOMPLETE_WINDOW["query_session"]]:
        raise QuantError("The specifically recorded incomplete claims window changed.")
    pairs = decision_pairs(data.close.index)
    changes = pd.Series(
        {
            day: (
                selected.loc[day, "claims_four_week_mean"]
                - selected.loc[reference, "claims_four_week_mean"]
            )
            for day, reference in pairs.items()
        },
        name="claims_change",
    )
    dollars, _ = load_vintages(data.close.index, read_json(DOLLAR_POLICY))
    dollar, dollar_audit = dollar_comparisons(dollars, data.close.index)
    comparisons = pd.concat([dollar, changes], axis=1)
    comparisons["claims_change_available"] = comparisons["claims_change"].notna()
    return comparisons, {"claims_selected_queries": selected, "dollar_provenance": dollar_audit}


def build_from_inputs(
    data: MarketData, comparisons: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    original = sector_targets(data, read_json(CONTROL))[policy["original_control_candidate"]]
    if (
        not comparisons.index.equals(original.dropna(how="all").index)
        or list(comparisons.columns)
        != [
            "dollar_change",
            "claims_change",
            "claims_change_available",
        ]
        or not np.isfinite(comparisons["dollar_change"]).all()
        or np.isinf(comparisons["claims_change"]).any()
        or comparisons["claims_change_available"].dtype != bool
        or not np.array_equal(
            np.isfinite(comparisons["claims_change"]),
            comparisons["claims_change_available"],
        )
    ):
        raise QuantError(
            "Claims confirmation requires every month and explicit source availability."
        )
    targets = original.copy()
    assets = targets.columns.drop("BIL")
    for day, row in comparisons.iterrows():
        if not row["claims_change_available"]:
            targets.loc[day] = np.nan
        elif row["dollar_change"] > 0 and row["claims_change"] > 0:
            removed = float(targets.loc[day, assets].sum()) * 0.5
            targets.loc[day, assets] *= 0.5
            targets.loc[day, "BIL"] += removed
    active = targets.dropna(how="all")
    if (active < 0).any().any() or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12):
        raise QuantError("Claims confirmation cannot short, borrow or change the original budget.")
    return {CANDIDATE["id"]: targets}


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _ = load_inputs(data, policy)
    return build_from_inputs(data, comparisons, policy)


def context_files() -> dict[str, str]:
    path = CONTEXT / "context-verification.json"
    context = read_json(path)
    if (
        context["stock_return_predictability_proven"] is not False
        or context["fixed_three_month_zero_change_rule_validated_by_primary_paper"] is not False
        or len(context["moving_average_probe_checks"]) != 3
        or set(context["primary_documents"])
        != {
            "Fed-claims-context.html",
            "IC4WSA-series.html",
            "IC4WSA-download.html",
        }
        or any(
            row["difference_from_official_same_vintage_average"] != 0
            for row in context["moving_average_probe_checks"]
        )
    ):
        raise QuantError("The context cannot relabel a smoothing convention as predictive alpha.")
    frozen = {path.relative_to(ROOT).as_posix(): file_digest(path)}
    for name, record in context["primary_documents"].items():
        frozen[(CONTEXT / name).relative_to(ROOT).as_posix()] = record["sha256"]
        frozen[(CONTEXT / record["request_path"]).relative_to(ROOT).as_posix()] = record[
            "request_sha256"
        ]
    for name, digest in (
        (context["moving_average_probe_path"], context["moving_average_probe_sha256"]),
        (context["moving_average_probe_request"], context["moving_average_probe_request_sha256"]),
    ):
        frozen[(CONTEXT / name).relative_to(ROOT).as_posix()] = digest
    for name, digest in frozen.items():
        safe_file(ROOT, name, digest)
    return frozen


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    context = context_files()
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
            raise QuantError("The claims plan cannot replace the original frozen market.")
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The claims plan retains the original frozen risk-free series.")
    comparisons, provenance = load_inputs(data, policy)
    targets = build_from_inputs(data, comparisons, policy)[CANDIDATE["id"]]
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Use an isolated preparation path inside the workbench.")
    new_output_directory(output)
    for name, frame in {"risk-comparisons": comparisons, **provenance}.items():
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    write_text_atomic(output / "frozen-targets.csv", targets.to_csv(float_format="%.17g"))
    manifest = read_json(INPUT_SOURCE / "manifest.json")
    frozen = {**original["frozen_files"], **context}
    failure = read_json(safe_file(ROOT, REVISION["original_source_failure"]))
    if (
        failure["actual_incomplete_windows"] != [INCOMPLETE_WINDOW]
        or failure["strategy_targets_generated"] is not False
        or failure["portfolio_returns_or_strategy_outcomes_computed"] is not False
    ):
        raise QuantError("The pre-outcome source-window failure must remain exactly recorded.")
    safe_file(ROOT, REVISION["original_unregistered_policy"], failure["original_policy_sha256"])
    safe_file(
        ROOT, REVISION["original_unregistered_implementation"], failure["original_code_sha256"]
    )
    for key in (
        "original_unregistered_policy",
        "original_unregistered_implementation",
        "original_source_failure",
    ):
        frozen[REVISION[key]] = file_digest(ROOT / REVISION[key])
    for path in (
        POLICY,
        Path(__file__),
        INPUT_POLICY,
        INPUT_READINESS,
        INPUT_SOURCE / "manifest.json",
        PRIOR,
        ROOT / "src/us_quant/jobless_claims_inputs.py",
        ROOT / "src/us_quant/inflation_inputs.py",
        ROOT / "src/us_quant/financial_conditions_guard.py",
    ):
        frozen[path.relative_to(ROOT).as_posix()] = file_digest(path)
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
    active = targets.dropna(how="all")
    paused = comparisons.index[~comparisons["claims_change_available"]]
    audit = {
        "prepared_at": utc_now(),
        "monthly_decisions": len(comparisons),
        "active_target_decisions": len(active),
        "paused_monthly_decisions": paused.strftime("%Y-%m-%d").tolist(),
        "target_shape": list(targets.shape),
        "target_sha256": file_digest(output / "frozen-targets.csv"),
        "all_four_equity_families_remain_positive": bool(
            (active[["MTUM", "VLUE", "QUAL", "USMV"]] > 0).all().all()
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
        name != "claims_factor_guard"
        or spec.get("eligible_not_before") != policy["eligible_not_before"]
    ):
        raise QuantError("The claims candidate needs its original eligible-cycle declaration.")
    if pd.Timestamp(at) < pd.Timestamp(policy["eligible_not_before"]):
        raise QuantError(
            f"Claims registration is not eligible before {policy['eligible_not_before']}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one fixed future claims guard; never orders."
    )
    parser.add_argument("action", choices=("prepare", "register", "recover"))
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "data/claims-guard-prepared-20261011"
    )
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "reports/claims-guard-registration.json"
    )
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare(args.prepared)
            print(
                json.dumps(
                    {
                        "candidate_id": CANDIDATE["id"],
                        "target_budget_audit": result["target_budget_audit"],
                        "strategy_outcomes_computed": False,
                    },
                    indent=2,
                )
            )
        else:
            from us_quant.candidate_receipts import register_or_recover

            prepared = args.prepared if args.prepared.is_absolute() else ROOT / args.prepared
            print(
                json.dumps(
                    register_or_recover(
                        prepared / f"{CANDIDATE['id']}-spec.json",
                        prepared / "readiness.json",
                        args.receipt,
                        allow_new_registration=args.action == "register",
                    ),
                    indent=2,
                )
            )
    except (QuantError, OSError, sqlite3.Error) as exc:
        parser.exit(
            2, f"Claims candidate blocked; preserve sources and all existing reviews: {exc}\n"
        )


if __name__ == "__main__":
    main()
