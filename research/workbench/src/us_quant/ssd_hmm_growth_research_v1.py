from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import hmm_growth_bil_v1 as rules
from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant import ssd_split_session_research_v1 as donor
from us_quant import ssd_variance_disagreement_research_v1 as source
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_text_atomic,
)

ROOT = base.ROOT
FOLDER = base.REPORTS / "hmm-growth-bil-20261029-v1"
TARGETS = ROOT / "data/ssd-hmm-growth-bil-20261029-v1"
PACKET = ROOT / "evidence/hmm_growth_bil_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/hmm_growth_bil_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/hmm-growth-bil-v1.json",
    "src/us_quant/hmm_growth_bil_v1.py",
    "src/us_quant/ssd_hmm_growth_research_v1.py",
    "tests/test_hmm_growth_bil_v1.py",
    "tests/test_ssd_hmm_growth_research_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    "evidence/split_session_variation_20261028_registration_v1.json",
    "src/us_quant/volatility_hmm_v1.py",
    "tests/test_volatility_hmm_v1.py",
    "config/ordered-path-growth-v2.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def inputs():
    events = base.journal_events(base.JOURNAL)
    if len(events) < 263 or events[262]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all200completed configurations and original failures.")
    data, _, _, parent, original, benchmarks = donor.inputs()
    donor.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["decision_calendar_registration"]))["entries"][
        0
    ]
    floor = read_json(base.checked_reference(policy["prior_floor_policy"]))
    if (
        policy["decision_calendar_registration"] != base.reference(donor.PACKET)
        or entry["target_proof"]["targets"] != policy["decision_calendar_targets"]
        or floor["stock_target_minimum"] != 0.147
        or floor["invested_target_budget"] != 0.98
    ):
        raise QuantError("The preserved calendar or inherited stock-floor reference changed.")
    calendar = source.frame(policy["decision_calendar_targets"])
    dates = rules.eligible_dates(data, calendar)
    if (
        len(data.close) != 2805
        or len(calendar.dropna()) != 131
        or len(dates) != 122
        or str(dates[0].date()) != "2016-08-31"
    ):
        raise QuantError("Keep the full original market and exact observed252return warmup.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"]
        + [base.reference(ROOT / name) for name in OWN_EXTENSIONS[1:]]
        + [
            base.reference(ROOT / name)
            for name in ("src/us_quant/volatility_hmm_v1.py", "tests/test_volatility_hmm_v1.py")
        ],
        "policies": parent["policies"]
        + [base.reference(rules.POLICY), policy["prior_floor_policy"]],
        "decision_calendar_registration": policy["decision_calendar_registration"],
        "decision_calendar_targets": policy["decision_calendar_targets"],
        "prior_floor_policy": policy["prior_floor_policy"],
        "prior_completed_configurations": 200,
        "prior_journal_sequence": 263,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "causal_current_filtered_volatility_growth_BIL_package",
        "original_policy_role": "source_market_and_calendar_only_not_old_allocation",
        "parent_target_values_or_macro_option_values_used": False,
        "model_parameters_fit_with_future_prices": False,
        "prior_evidence": parent["prior_evidence"]
        + [
            policy["decision_calendar_registration"],
            policy["decision_calendar_targets"],
            entry["target_proof"]["proof"],
            base.reference(donor.PUBLIC_RESULTS),
            base.reference(ROOT / "evidence/core_activity_momentum_20261029_results_v1.json"),
            base.reference(FOLDER / "research-intent.json"),
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(base.REPORTS / name)
            for name in ("hmm-growth-offline-tests-v1.xml", "hmm-growth-offline-tests-v1.log")
        ],
        "new_source_requests": 0,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, calendar, None, frozen, original, benchmarks


def verify_saved_targets(data, calendar, information, targets, audits):
    dates = rules.eligible_dates(data, calendar)
    if (
        len(dates) != 122
        or len(audits) != 122
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or [row["decision_session"] for row in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every eligible original date and the undefined prehistory.")
    target_error = probability_error = likelihood_error = 0.0
    for day, audit in zip(dates, audits, strict=True):
        oracle, probabilities, likelihood, pooled = rules.independent_row(data, day, audit)
        model = audit["model"]
        difference = float(np.max(abs(targets.loc[day] - oracle)))
        p_error = abs(probabilities[0] - model["low_variance_state_probability"])
        ll_error = abs(likelihood - model["log_likelihood"])
        if (
            not np.isfinite(difference + p_error + ll_error)
            or difference > 1e-10
            or p_error > 1e-10
            or ll_error > 1e-8
        ):
            raise QuantError("Independent scaled filtering or funding reconstruction differs.")
        target_error, probability_error, likelihood_error = (
            max(target_error, difference),
            max(probability_error, p_error),
            max(likelihood_error, ll_error),
        )
        position = data.close.index.get_loc(day)
        variance = np.asarray(model["state_variances_daily"])
        transition = np.asarray(model["transition"])
        objective = rules.hmm.penalized_objective(likelihood, transition, variance, pooled)
        previous = model["penalized_log_posterior"] - model["last_objective_improvement"]
        if (
            audit["first_prior_close"] != str(data.close.index[position - 252].date())
            or audit["last_observation"] != str(day.date())
            or audit["observed_returns"] != 252
            or model["converged"] is not True
            or not 1 <= model["iterations"] <= 500
            or variance[0] >= variance[1]
            or model["initial_probabilities"] != [0.5, 0.5]
            or abs(model["pooled_daily_squared_return"] - pooled) > 1e-12
            or abs(model["penalized_log_posterior"] - objective) > 1e-8
            or model["last_objective_improvement"] > 1e-7 * max(1, abs(previous))
            or model["last_objective_improvement"] < -1e-10 * max(1, abs(previous))
            or abs(model["filtered_probabilities_at_last_observation"][0] - probabilities[0])
            > 1e-10
            or abs(audit["stock_target"] - float(oracle.loc[[*rules.CORE, "SOXX"]].sum())) > 1e-10
            or abs(audit["BIL_target"] - float(oracle["BIL"])) > 1e-10
            or audit["past_targets_revised"] is not False
            or audit["future_probability_calibration_or_risk_guarantee_claimed"] is not False
        ):
            raise QuantError("The stored model, convergence evidence or causal clock changed.")
    return target_error, probability_error, likelihood_error


def prove_targets(data, calendar, information, protocol):
    retained = []

    def save_fit(audit):
        path = TARGETS / IDS[0] / "fit-audits" / (audit["decision_session"] + ".json")
        common.retain_receipt(path, {"recorded_at": utc_now(), "audit": audit}, ("recorded_at",))
        retained.append(base.reference(path))

    targets, audits = rules.build_targets(
        data, calendar, read_json(rules.POLICY), retain_fit=save_fit, before_fit=runtime_guard
    )
    target = targets[IDS[0]]
    target_error, p_error, ll_error = verify_saved_targets(
        data, calendar, information, target, audits
    )
    coverage = []
    for window in protocol["windows"]:
        anchor = previous_session(window["start"])
        known = target.ffill().loc[anchor : window["end"]]
        if (
            anchor not in known.index
            or known.isna().any().any()
            or any(not (known[name] >= 0.147 / 8 - 1e-12).all() for name in rules.CORE)
        ):
            raise QuantError(
                "Full formal anchors and four meaningful style allocations must remain."
            )
        coverage.append({"years": window["years"], "positive_factor_symbols": list(rules.CORE)})
    dates = target.dropna().index
    prefixes = []
    for position in (0, 12, 36, 60, len(dates) - 1):
        runtime_guard()
        end = dates[position]
        clipped = MarketData(
            data.open.loc[:end],
            data.close.loc[:end],
            data.raw_close.loc[:end],
            data.volume.loc[:end],
            data.risk_free.loc[:end],
        )
        early, audit = rules.target_for_day(clipped, end)
        if not np.allclose(early, target.loc[end], rtol=0, atol=1e-10) or audit != audits[position]:
            raise QuantError("Future observations altered an already dated model fit.")
        prefixes.append(str(end.date()))
    return (
        targets,
        audits,
        {
            "checked_at": utc_now(),
            "monthly_decisions": 122,
            "original_source_opportunities": 131,
            "undefined_prehistory_months": 9,
            "first_eligible_source_month": "2016-08-31",
            "maximum_independent_target_difference": target_error,
            "maximum_independent_filter_probability_difference": p_error,
            "maximum_independent_log_likelihood_difference": ll_error,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "retained_model_fit_receipts": retained,
            "current_filter_not_past_smoothed_trading_states": True,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "global_likelihood_optimum_or_probability_calibration_claimed": False,
        },
    )


def checked_packet(frozen: dict) -> dict:
    common.output_guard(PACKET)
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 1
        or packet["raw_inputs_and_target_matrices_remain_private"] is not True
        or any(
            packet.get(key) is not False
            for key in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "order_authority",
                "independent_forward_validation",
            )
        )
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(row["target_proof_at"]) > created for row in entries)
        or any(
            created > pd.Timestamp(row["at"])
            for identifier in IDS
            for row in base.matching_events("path_started", identifier)
        )
    ):
        raise QuantError("Keep the actual unchanged pre-account HMM-growth registration.")
    return packet


def failure(stage: str, error: Exception) -> None:
    identifier = IDS[0]
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if (
        not expired
        and base.matching_events("registered", identifier)
        and not base.matching_events("review", identifier)
        and not base.matching_events("technical_failure", identifier)
    ):
        base.append_event(
            base.JOURNAL,
            "technical_failure",
            identifier,
            {
                "stage": stage,
                "reason": str(error),
                "partial_outputs_retained": True,
                "candidate_accounts_started": bool(
                    base.matching_events("path_started", identifier)
                ),
            },
        )
    common.progress(
        "hmm_growth_expired_preserved" if expired else "hmm_growth_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, options, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 23
            or prior["new_completed_strategy_evaluations"] != 22
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve23registrations,22completions andtheoldfailure.")
        candidate, identifier = rules.CANDIDATES[0], IDS[0]
        spec = {"candidate": candidate, "frozen": frozen}
        existing = base.matching_events("registered", identifier)
        if existing and (len(existing) != 1 or existing[0]["body"] != spec):
            raise QuantError("A registered HMM-growth rule cannot change.")
        if not existing:
            runtime_guard()
            base.append_event(base.JOURNAL, "registered", identifier, spec)
        if base.matching_events("technical_failure", identifier):
            raise QuantError("Preserve a failed preparation without same-version replay.")
        if not base.matching_events("target_proof", identifier):
            common.progress(
                "preparing_registered_hmm_growth",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                runtime_guard()
                targets, audits, proof = prove_targets(
                    data, desired, options, read_json(base.PROTOCOL)
                )
                check_frozen(frozen)
                directory = TARGETS / identifier
                path = directory / "targets.csv"
                common.output_guard(path)
                content = targets[identifier].to_csv(float_format="%.17g")
                if path.exists():
                    if file_digest(path) != digest_bytes(content.encode()):
                        raise QuantError("Preserve conflicting partial HMM-growth targets.")
                else:
                    write_text_atomic(path, content)
                common.retain_receipt(directory / "model-audits.json", {"decisions": audits})
                proof["model_audits"] = base.reference(directory / "model-audits.json")
                common.retain_receipt(directory / "proof.json", proof, ("checked_at",))
                runtime_guard()
                base.append_event(
                    base.JOURNAL,
                    "target_proof",
                    identifier,
                    {
                        "registered_event_sha256": base.matching_events("registered", identifier)[
                            0
                        ]["sha256"],
                        "targets": base.reference(path),
                        "proof": base.reference(directory / "proof.json"),
                    },
                )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("hmm_growth_preparation", error)
                raise
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [common.packet_entry(candidate, frozen)],
                "new_registered_candidates": 1,
                "candidate_accounts_or_metrics_computed": False,
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        if not base.matching_events("path_started", identifier):
            common.progress(
                "hmm_growth_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                hmm_growth_registration=base.reference(PACKET),
                pending_steps=["publish_hmm_growth_registration", "four_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen: dict, original: dict, commit: str) -> dict:
    inherited = base.published_inputs(original, commit)
    prior = verify_publication(
        ROOT, frozen["daily_proxy_registration"], set(source.EXTENSIONS), commit
    )
    own = verify_publication(ROOT, base.reference(PACKET), set(EXTENSIONS), commit)
    events = [
        row
        for row in base.matching_events("publication")
        if row["body"].get("study_id") == frozen["study_id"]
    ]
    if events:
        if len(events) != 1 or events[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with this study's actual original publication commit.")
        return events[0]["body"]["publication"]
    receipt = {
        **own,
        "original_dependency_publication": inherited,
        "daily_proxy_dependency_publication": prior,
    }
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen: dict) -> dict:
    result = {}
    for identifier in IDS:
        reviews = base.matching_events("review", identifier)
        failed = base.matching_events("technical_failure", identifier)
        paths = {
            row["body"]["path_key"]: base.verify_path(
                row["body"]["receipt"], frozen, identifier, row["body"]["path_key"]
            )
            for row in base.matching_events("path_completed", identifier)
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("Completed HMM-growth review and paths disagree.")
            result[identifier] = reviews[0]["body"]
        else:
            result[identifier] = {
                "candidate_id": identifier,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit: str) -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, desired, options, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        common.output_guard(RESULTS)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Completed evidence differs; never replay accounts.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        common.progress(
            "evaluating_hmm_growth",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            hmm_growth_publication=published,
        )
        identifier = IDS[0]
        try:
            if not base.matching_events("review", identifier) and not base.matching_events(
                "technical_failure", identifier
            ):
                target_event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(target_event["targets"])
                proof = read_json(base.checked_reference(target_event["proof"]))
                audits = read_json(base.checked_reference(proof["model_audits"]))["decisions"]
                verify_saved_targets(data, desired, options, targets, audits)
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_hmm_growth",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = source.evaluate_path(
                            data, targets, frozen, identifier, window, scenario, benchmarks
                        )
                check_frozen(frozen)
                base.legacy_snapshot(read_json(base.POLICY))
                base.append_event(
                    base.JOURNAL,
                    "review",
                    identifier,
                    {
                        "candidate_id": identifier,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed"
                        if all(row["all_point_gates_pass"] for row in paths.values())
                        else "rejected",
                        "paths": paths,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            common.retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            common.progress(
                "hmm_growth_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                hmm_growth_results=base.reference(RESULTS),
                pending_steps=["publish_all_hmm_growth_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("hmm_growth_evaluation", error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "hmm_growth_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result() -> dict:
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal HMM-growth outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("An unfinished study is not a completed result.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for identifier, row in actual.items():
        paths = []
        for key, item in row["paths"].items():
            year, scenario = key.split("-")
            year = int(year[:-1])
            paths.append(
                {
                    **{
                        name: item[name]
                        for name in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": year,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        name: lookup[year, scenario, name] for name in ("SPY", "QQQ")
                    },
                }
            )
        reviews.append(
            {
                "candidate_id": identifier,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [row for row in base.journal_events(base.JOURNAL) if row["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": base.reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": base.reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return base.reference(PUBLIC_RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="One HMM-growth information study; no trading.")
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The HMM-growth study cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published preregistration commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            base.storage_guard()
            result = common.counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"HMM-growth research blocked: {error}\n")


if __name__ == "__main__":
    main()
