from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from us_quant import conditional_variance_growth_v1 as rules
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
FOLDER = base.REPORTS / "conditional-variance-growth-20261029-v1"
TARGETS = ROOT / "data/ssd-conditional-variance-growth-20261029-v1"
PACKET = ROOT / "evidence/conditional_variance_growth_20261029_registration_v1.json"
PREPARED = FOLDER / "prepared-target-bundle.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/conditional_variance_growth_20261029_results_v1.json"
OWN_EXTENSIONS = (
    "config/conditional-variance-growth-v1.json",
    "src/us_quant/conditional_variance_v1.py",
    "src/us_quant/conditional_variance_growth_v1.py",
    "src/us_quant/ssd_conditional_variance_research_v1.py",
    "tests/test_conditional_variance_v1.py",
    "tests/test_conditional_variance_growth_v1.py",
    "tests/test_ssd_conditional_variance_research_v1.py",
)
EXTENSIONS = (
    *OWN_EXTENSIONS,
    *donor.EXTENSIONS,
    "evidence/split_session_variation_20261028_registration_v1.json",
    "evidence/hmm_growth_bil_20261029_results_v1.json",
)
IDS = tuple(row["id"] for row in rules.CANDIDATES)
check_frozen = source.check_frozen


def inputs():
    events = base.journal_events(base.JOURNAL)
    if len(events) < 321 or events[320]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve all205completed configurations and the original failures.")
    data, _, _, parent, original, benchmarks = donor.inputs()
    donor.checked_packet(parent)
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    entry = read_json(base.checked_reference(policy["decision_calendar_registration"]))["entries"][
        0
    ]
    comparison = read_json(base.checked_reference(policy["completed_hmm_comparison"]))
    if (
        policy["decision_calendar_registration"] != base.reference(donor.PACKET)
        or entry["target_proof"]["targets"] != policy["decision_calendar_targets"]
        or comparison["new_completed_configurations"] != 1
        or comparison["reviews"][0]["candidate_id"] != "causal_two_state_growth_bil_v1"
    ):
        raise QuantError("The original calendar and completed HMM comparison must remain frozen.")
    calendar = source.frame(policy["decision_calendar_targets"])
    dates = rules.eligible_dates(data, calendar)
    if (
        len(data.close) != 2805
        or len(calendar.dropna()) != 131
        or len(dates) != 122
        or str(dates[0].date()) != "2016-08-31"
    ):
        raise QuantError("Keep the full original market and252actualreturn warmup.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": parent["code"] + [base.reference(ROOT / name) for name in OWN_EXTENSIONS[1:]],
        "policies": parent["policies"] + [base.reference(rules.POLICY)],
        "decision_calendar_registration": policy["decision_calendar_registration"],
        "decision_calendar_targets": policy["decision_calendar_targets"],
        "completed_hmm_comparison": policy["completed_hmm_comparison"],
        "prior_completed_configurations": 205,
        "prior_journal_sequence": 321,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "paired_conditional_variance_of_fixed_stock_risk_proxy",
        "original_policy_role": "frozen_market_and_calendar_only_not_old_allocation",
        "parent_target_values_or_macro_option_values_used": False,
        "model_parameters_fit_with_future_prices": False,
        "prior_evidence": parent["prior_evidence"]
        + [
            policy["decision_calendar_registration"],
            policy["decision_calendar_targets"],
            entry["target_proof"]["proof"],
            policy["completed_hmm_comparison"],
            base.reference(ROOT / "evidence/vxn_relative_uncertainty_20261029_results_v1.json"),
            *[
                base.reference(FOLDER / name)
                for name in (
                    "research-intent.json",
                    "methodology-search-failure-native-event.json",
                    "garch-gjr-documentation-native-event.json",
                    "analytical-forecasting-native-event.json",
                )
            ],
        ],
        "validation_evidence": parent["validation_evidence"]
        + [
            base.reference(FOLDER / name)
            for name in ("offline-tests-checked-v1.xml", "offline-tests-checked-v1.log")
        ],
        "new_source_requests": 0,
        "new_source_requests_scope": "market_macro_and_option_data",
        "methodology_documentation_fetches": 2,
        "formal_windows_shortened": False,
    }
    check_frozen(frozen)
    return data, calendar, None, frozen, original, benchmarks


def verify_saved_targets(data, calendar, targets, audits, candidate):
    dates = rules.eligible_dates(data, calendar)
    if (
        len(dates) != 122
        or len(audits) != 122
        or not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or [row["information"]["decision_session"] for row in audits]
        != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every eligible original date and undefined prehistory.")
    target_error = variance_error = likelihood_error = 0.0
    for day, audit in zip(dates, audits, strict=True):
        stored_row, _ = rules.target_for_day(data, day, candidate, audit)
        oracle, independent = rules.independent_row(data, day, audit)
        fitted = audit["model"]
        parameters, probabilities = rules.model.decode(
            fitted["coordinates"], candidate["asymmetric"]
        )
        pooled = fitted["pooled_daily_squared_return"]
        native_parameters = [float(parameters[0] * pooled), *parameters[1:].tolist()]
        observed_variances = np.asarray(fitted["conditional_variances_daily"], dtype=float)
        forecasts = np.asarray(fitted["forecast_variances_daily"], dtype=float)
        if (
            observed_variances.shape != (252,)
            or forecasts.shape != (21,)
            or not np.isfinite(observed_variances).all()
            or not np.isfinite(forecasts).all()
            or (observed_variances <= 0).any()
            or (forecasts <= 0).any()
            or fitted["parameters_scaled"] != parameters.tolist()
            or fitted["parameters_daily"] != native_parameters
            or fitted["asymmetric"] is not candidate["asymmetric"]
            or fitted["optimizer_success"] is not True
            or fitted["optimizer"] != "L-BFGS-B"
            or fitted["optimizer_initializations"] != 1
            or fitted["observations"] != 252
            or fitted["forecast_sessions"] != 21
            or fitted["initial_normalized_variance"] != 1.0
            or fitted["initial_coordinates"]
            != rules.model.initial_coordinates(candidate["asymmetric"]).tolist()
            or not 0 <= fitted["optimizer_iterations"] <= 1000
            or not 0 < fitted["persistence"] < 1
            or abs(fitted["persistence"] - (parameters[1] + parameters[2] + parameters[3] / 2))
            > 1e-14
            or abs(fitted["positive_persistence_slack"] - probabilities[0]) > 1e-14
            or abs(pooled - independent["pooled_squared_return"]) > 1e-9 * pooled
            or fitted["mean_negative_log_likelihood_scaled"]
            > fitted["initial_mean_negative_log_likelihood_scaled"] + 1e-10
            or abs(fitted["mean_negative_log_likelihood_scaled"] - fitted["optimizer_objective"])
            > 1e-10
            or abs(
                fitted["mean_negative_log_likelihood_daily"]
                - fitted["mean_negative_log_likelihood_scaled"]
                - 0.5 * np.log(pooled)
            )
            > 1e-10
            or fitted["globally_optimal_or_statistically_identified"] is not False
            or fitted["innovation_symmetry_empirically_verified"] is not False
        ):
            raise QuantError(
                "The saved fit, numerical constraints or convergence evidence changed."
            )
        expected_variances = np.asarray(independent["variances"])
        expected_forecasts = np.asarray(independent["forecast_variances"])
        t_error = max(
            float(np.max(abs(targets.loc[day] - oracle))),
            float(np.max(abs(targets.loc[day] - stored_row))),
        )
        v_error = max(
            float(np.max(abs(observed_variances - expected_variances) / expected_variances)),
            float(np.max(abs(forecasts - expected_forecasts) / expected_forecasts)),
            abs(fitted["mean_forecast_variance_daily"] / independent["mean_forecast_variance"] - 1),
        )
        ll_error = abs(
            fitted["mean_negative_log_likelihood_daily"]
            - independent["mean_negative_log_likelihood"]
        )
        if (
            not np.isfinite(t_error + v_error + ll_error)
            or t_error > 1e-10
            or v_error > 1e-9
            or ll_error > 1e-8
        ):
            raise QuantError("Independent native-unit convolution, forecasting or funding differs.")
        target_error = max(target_error, t_error)
        variance_error = max(variance_error, v_error)
        likelihood_error = max(likelihood_error, ll_error)
    return {
        "maximum_independent_target_difference": target_error,
        "maximum_independent_variance_relative_difference": variance_error,
        "maximum_independent_mean_likelihood_difference": likelihood_error,
    }


def registration_hashes(frozen):
    result = {}
    for candidate in rules.CANDIDATES:
        registered = base.matching_events("registered", candidate["id"])
        if len(registered) != 1 or registered[0]["body"] != {
            "candidate": candidate,
            "frozen": frozen,
        }:
            raise QuantError("Both actual immutable registrations must precede each fit.")
        result[candidate["id"]] = registered[0]["sha256"]
    return result


def fit_path(identifier, day, directory):
    return TARGETS / identifier / directory / (str(day.date()) + ".json")


def verify_fit_receipts(identifier, audits, references, frozen):
    if len(references) != len(audits):
        raise QuantError("Each dated fit needs its retained receipt.")
    registrations = registration_hashes(frozen)
    latest_registration = max(
        pd.Timestamp(base.matching_events("registered", name)[0]["at"]) for name in IDS
    )
    for audit, reference in zip(audits, references, strict=True):
        day = pd.Timestamp(audit["information"]["decision_session"])
        if reference != base.reference(fit_path(identifier, day, "fit-audits")):
            raise QuantError("A fit receipt belongs to a different candidate or date.")
        retained = read_json(base.checked_reference(reference))
        attempt = read_json(base.checked_reference(retained["attempt"]))
        started, recorded = (
            pd.Timestamp(attempt["started_at"]),
            pd.Timestamp(retained["recorded_at"]),
        )
        if (
            retained["audit"] != audit
            or attempt["candidate_id"] != identifier
            or attempt["decision_session"] != str(day.date())
            or attempt["registered_event_sha256s"] != registrations
            or attempt["frozen_specification_sha256"] != digest_json(frozen)
            or started.tzinfo is None
            or recorded.tzinfo is None
            or pd.isna(started)
            or pd.isna(recorded)
            or not latest_registration <= started <= recorded <= pd.Timestamp(utc_now())
        ):
            raise QuantError("Retained fitting clocks or registration/source bindings disagree.")


def prove_targets(data, calendar, information, protocol, frozen):
    retained = {identifier: [] for identifier in IDS}
    registrations = registration_hashes(frozen)

    def load_fit(identifier, day):
        path = fit_path(identifier, day, "fit-audits")
        common.output_guard(path)
        return read_json(path)["audit"] if path.exists() else None

    def before_fit(identifier, day, reuse):
        runtime_guard()
        path = fit_path(identifier, day, "fit-attempts")
        common.output_guard(path)
        if reuse:
            if not path.exists():
                raise QuantError(
                    "A completed fit without its actual start receipt cannot be reused."
                )
            return
        if path.exists():
            raise QuantError(
                "Preserve an unresolved fit attempt; do not silently repeat optimization."
            )
        common.retain_receipt(
            path,
            {
                "started_at": utc_now(),
                "candidate_id": identifier,
                "decision_session": str(day.date()),
                "registered_event_sha256s": registrations,
                "frozen_specification_sha256": digest_json(frozen),
            },
        )
        common.progress(
            "preparing_registered_conditional_variance",
            active_candidate=identifier,
            active_model_session=str(day.date()),
            retained_model_fits=sum(len(rows) for rows in retained.values()),
        )

    def save_fit(audit):
        identifier = audit["candidate_id"]
        day = pd.Timestamp(audit["information"]["decision_session"])
        path = fit_path(identifier, day, "fit-audits")
        common.retain_receipt(
            path,
            {
                "recorded_at": utc_now(),
                "attempt": base.reference(fit_path(identifier, day, "fit-attempts")),
                "audit": audit,
            },
            ("recorded_at",),
        )
        retained[identifier].append(base.reference(path))

    targets, audits = rules.build_targets(
        data,
        calendar,
        read_json(rules.POLICY),
        retain_fit=save_fit,
        before_fit=before_fit,
        load_fit=load_fit,
    )
    proofs = {}
    for candidate in rules.CANDIDATES:
        runtime_guard()
        identifier = candidate["id"]
        target = targets[identifier]
        metrics = verify_saved_targets(data, calendar, target, audits[identifier], candidate)
        verify_fit_receipts(identifier, audits[identifier], retained[identifier], frozen)
        coverage = []
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            known = target.ffill().loc[anchor : window["end"]]
            if (
                anchor not in known.index
                or known.isna().any().any()
                or any(not (known[name] > 1e-10).all() for name in rules.CORE)
            ):
                raise QuantError(
                    "Keep every original anchor and all four registered style families."
                )
            coverage.append(
                {
                    "years": window["years"],
                    "positive_factor_symbols": list(rules.CORE),
                    "minimum_combined_stock_target": float(
                        known.loc[:, list(rules.STOCKS)].sum(axis=1).min()
                    ),
                }
            )
        prefixes = []
        dates = target.dropna().index
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
            full_inputs, full_information = rules.observed_returns(data, end)
            clipped_inputs, clipped_information = rules.observed_returns(clipped, end)
            restored, audit = rules.target_for_day(
                clipped, end, candidate, audits[identifier][position]
            )
            if (
                not np.array_equal(full_inputs, clipped_inputs)
                or full_information != clipped_information
                or not np.allclose(restored, target.loc[end], rtol=0, atol=1e-10)
                or audit != audits[identifier][position]
            ):
                raise QuantError(
                    "Future observations changed a dated fit's inputs or saved allocation."
                )
            prefixes.append(str(end.date()))
        proofs[identifier] = {
            "checked_at": utc_now(),
            "monthly_decisions": 122,
            "original_source_opportunities": 131,
            "undefined_prehistory_months": 9,
            "first_eligible_source_month": "2016-08-31",
            **metrics,
            "prefix_input_and_saved_model_truncations": prefixes,
            "prefix_checks_rerun_optimization": False,
            "formal_factor_coverage": coverage,
            "retained_model_fit_receipts": retained[identifier],
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "global_optimum_or_forecast_calibration_claimed": False,
        }
    return targets, audits, proofs


def checked_packet(frozen):
    common.output_guard(PACKET)
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
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
        raise QuantError("Keep the actual unchanged pre-account conditional-variance registration.")
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    for identifier in identifiers:
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
        "conditional_variance_expired_preserved" if expired else "conditional_variance_blocked",
        active_compute_job=None,
        active_candidate=None,
        failure=str(error),
    )


def prepare_bundle(data, calendar, information, frozen):
    common.output_guard(PREPARED)
    if PREPARED.exists():
        bundle = read_json(PREPARED)
        if bundle["frozen_specification_sha256"] != digest_json(frozen) or set(
            bundle["entries"]
        ) != set(IDS):
            raise QuantError("Preserve a conflicting prepared bundle without replay.")
        for entry in bundle["entries"].values():
            for key in ("targets", "proof", "model_audits"):
                base.checked_reference(entry[key])
        return bundle["entries"]
    targets, audits, proofs = prove_targets(
        data,
        calendar,
        information,
        read_json(base.PROTOCOL),
        frozen,
    )
    check_frozen(frozen)
    entries = {}
    for identifier in IDS:
        directory = TARGETS / identifier
        path = directory / "targets.csv"
        common.output_guard(path)
        content = targets[identifier].to_csv(float_format="%.17g")
        if path.exists():
            if file_digest(path) != digest_bytes(content.encode()):
                raise QuantError("Preserve conflicting partial conditional-variance targets.")
        else:
            write_text_atomic(path, content)
        common.retain_receipt(directory / "model-audits.json", {"decisions": audits[identifier]})
        proof = proofs[identifier]
        proof["model_audits"] = base.reference(directory / "model-audits.json")
        common.retain_receipt(directory / "proof.json", proof, ("checked_at",))
        entries[identifier] = {
            "targets": base.reference(path),
            "proof": base.reference(directory / "proof.json"),
            "model_audits": base.reference(directory / "model-audits.json"),
        }
    common.retain_receipt(
        PREPARED,
        {"frozen_specification_sha256": digest_json(frozen), "entries": entries},
    )
    return entries


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, calendar, information, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([row for row in events if row["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 28
            or prior["new_completed_strategy_evaluations"] != 27
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Preserve28registrations,27completions and the old technical failure.")
        for candidate in rules.CANDIDATES:
            identifier = candidate["id"]
            specification = {"candidate": candidate, "frozen": frozen}
            existing = base.matching_events("registered", identifier)
            if existing and (len(existing) != 1 or existing[0]["body"] != specification):
                raise QuantError("A registered conditional-variance rule cannot change.")
            if not existing:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", identifier, specification)
        if any(base.matching_events("technical_failure", identifier) for identifier in IDS):
            raise QuantError("Preserve a failed preparation without same-version replay.")
        pending = [name for name in IDS if not base.matching_events("target_proof", name)]
        if pending:
            common.progress(
                "preparing_registered_conditional_variance",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                prepared = prepare_bundle(data, calendar, information, frozen)
                check_frozen(frozen)
                for identifier in pending:
                    runtime_guard()
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        identifier,
                        {
                            "registered_event_sha256": base.matching_events(
                                "registered", identifier
                            )[0]["sha256"],
                            "targets": prepared[identifier]["targets"],
                            "proof": prepared[identifier]["proof"],
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("conditional_variance_preparation", pending, error)
                raise
            except KeyboardInterrupt:
                common.progress(
                    "conditional_variance_interrupted_preserved",
                    active_compute_job=None,
                    active_candidate=None,
                )
                raise
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [
                    common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES
                ],
                "new_registered_candidates": 2,
                "candidate_accounts_or_metrics_computed": False,
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        if not any(base.matching_events("path_started", name) for name in IDS):
            common.progress(
                "conditional_variance_registered_waiting_publication",
                active_compute_job=None,
                active_candidate=None,
                active_model_session=None,
                conditional_variance_registration=base.reference(PACKET),
                pending_steps=["publish_conditional_variance_registration", "eight_funded_paths"],
            )
        return base.reference(PACKET)


def publication(frozen, original, commit):
    inherited = base.published_inputs(original, commit)
    prior = verify_publication(
        ROOT,
        frozen["daily_proxy_registration"],
        set(source.EXTENSIONS),
        commit,
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
        base.JOURNAL,
        "publication",
        None,
        {"study_id": frozen["study_id"], "publication": receipt},
    )
    return receipt


def states(frozen):
    result = {}
    for identifier in IDS:
        reviews = base.matching_events("review", identifier)
        failed = base.matching_events("technical_failure", identifier)
        paths = {
            row["body"]["path_key"]: base.verify_path(
                row["body"]["receipt"],
                frozen,
                identifier,
                row["body"]["path_key"],
            )
            for row in base.matching_events("path_completed", identifier)
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("Completed conditional-variance review and paths disagree.")
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


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, calendar, _, frozen, original, benchmarks = inputs()
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
            "evaluating_conditional_variance",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            conditional_variance_publication=published,
        )
        identifier = None
        try:
            for candidate in rules.CANDIDATES:
                identifier = candidate["id"]
                if base.matching_events("review", identifier) or base.matching_events(
                    "technical_failure", identifier
                ):
                    continue
                target_event = base.matching_events("target_proof", identifier)[0]["body"]
                targets = source.frame(target_event["targets"])
                proof = read_json(base.checked_reference(target_event["proof"]))
                audits = read_json(base.checked_reference(proof["model_audits"]))["decisions"]
                verify_saved_targets(data, calendar, targets, audits, candidate)
                verify_fit_receipts(
                    identifier, audits, proof["retained_model_fit_receipts"], frozen
                )
                paths = {}
                protocol = read_json(base.PROTOCOL)
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        common.progress(
                            "evaluating_conditional_variance",
                            active_candidate=identifier,
                            active_path=key,
                        )
                        paths[key] = source.evaluate_path(
                            data,
                            targets,
                            frozen,
                            identifier,
                            window,
                            scenario,
                            benchmarks,
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
                "conditional_variance_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                conditional_variance_results=base.reference(RESULTS),
                pending_steps=["publish_all_conditional_variance_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("conditional_variance_evaluation", (identifier,) if identifier else (), error)
            raise
        except KeyboardInterrupt:
            common.progress(
                "conditional_variance_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result():
    base.storage_guard()
    _, _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    if not RESULTS.exists():
        raise QuantError("Publish only an actual terminal conditional-variance outcome.")
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("An unfinished paired study is not a completed result.")
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
                },
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
                    "all_four_paths_same_candidate_and_frozen_parameters",
                    False,
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            },
        )
    counts = common.counts(
        [row for row in base.journal_events(base.JOURNAL) if row["candidate_id"] in IDS],
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


def main():
    parser = argparse.ArgumentParser(
        description="Paired conditional-variance research; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("The conditional-variance study cannot request live sources.")

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
        parser.exit(2, f"Conditional-variance research blocked: {error}\n")


if __name__ == "__main__":
    main()
