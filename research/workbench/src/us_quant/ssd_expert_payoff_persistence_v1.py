from __future__ import annotations

import argparse
import ast
import json
import math
import os
from fractions import Fraction
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_payoff_persistence_v1 as rules
from us_quant import ssd_expert_contextual_contest_v1 as risk_template
from us_quant import ssd_expert_hierarchical_risk_v1 as import_template
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-payoff-persistence-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-payoff-persistence-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_payoff_persistence_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_payoff_persistence_20261030_results_v1.json"
IDS = tuple(candidate["id"] for candidate in rules.CANDIDATES)
OWN = (
    "config/expert-payoff-persistence-v1.json",
    "src/us_quant/expert_payoff_persistence_v1.py",
    "src/us_quant/ssd_expert_payoff_persistence_v1.py",
    "tests/test_expert_payoff_persistence_v1.py",
    "tests/test_ssd_expert_payoff_persistence_v1.py",
)
EXTRA_TEST_INPUTS = (
    "tests/test_expert_contextual_contest_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
)
SHARED_CODE = "src/us_quant/strategy_expert_pool_v4.py"
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame


def dependency_inputs():
    return FunctionType(import_template.dependency_inputs.__code__, {**globals(), "ast": ast})()


def source_inputs():
    policy, design = read_json(rules.POLICY), read_json(DESIGN)
    rules.validate(policy)
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or design["policy"] != reference(rules.POLICY)
        or design["actual_invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["source_specification_sha256"] != PARENT_SPEC
        or design["real_new_payoff_mass_risk_allocations_or_targets_computed"] is not False
        or design["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError(
            "Preserve the actual pre-outcome persistence design and six-expert source."
        )
    check_frozen(parent)
    source.checked_packet(parent)
    data, experts, old_policy, _, old_refs, clocks, admission_refs, benchmarks = (
        source.source_inputs()
    )
    if any(
        policy[key] != old_policy[key]
        for key in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
        )
    ):
        raise QuantError("Keep valid experts,maturity and original main execution unchanged.")
    refs = unique_refs(
        [
            *old_refs,
            *admission_refs,
            *design["source_references"],
            reference(source.INTENT),
            *[
                policy[key]
                for key in (
                    "source_registration",
                    "source_controls",
                    "previous_contextual_results",
                    "previous_credit_diagnosis",
                    "previous_growth_results",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_fixed_persistence_rules_are_not_a_repaired_integral_or_reversal",
        "test_exact_continuation_and_uniform_refresh",
        "test_complete_month_relative_rounds_once_without_truncating_binary64",
        "test_initial_month_maturity_and_persistent_learning_clock",
        "test_future_source_changes_preserve_prior_targets_and_audits",
        "test_preupdate_inputs_survive_a_failing_advance",
        "test_independent_integer_mass_matches_exact_rational_path",
        "test_independent_proof_rejects_mass_score_hex_and_target_tampering",
        "test_cached_sources_and_verifier_never_call_closed_entrypoints",
        "test_complete_import_closure_does_not_mutate_template_globals",
        "test_failed_publication_cannot_start_main_capital",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed exact-state,causality,publication and accounting checks."
        )
    inputs = [reference(ROOT / path) for path in (*OWN, *EXTRA_TEST_INPUTS)]
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != inputs
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(suite.get("tests", 0)) for suite in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind current code,fixtures and successful test outputs.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve a complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        invocation = read_json(INVOCATION)
        if (
            len(events) != 583
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 56
            or counts["new_completed_strategy_evaluations"] != 48
            or counts["new_technical_failures"] != 8
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
            or read_json(DESIGN)["actual_continuation_event"] != 55981
        ):
            raise QuantError("Freeze against the actual unchanged journal and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 55981,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / path) for path in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_payoff_mass_risk_allocations_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_payoff_persistence_frozen",
            active_compute_job=None,
            expert_payoff_persistence_intent=reference(INTENT),
            pending_steps=["register_both_fixed_rules_before_actual_payoff_learning"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / path) for path in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != PARENT_SPEC
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_payoff_mass_risk_allocations_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome payoff implementation freeze changed.")
    code, policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *code,
                *[reference(ROOT / path) for path in (*OWN[1:], *EXTRA_TEST_INPUTS)],
            ]
        ),
        "policies": unique_refs([*parent["policies"], *policies, reference(rules.POLICY)]),
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_positive_payoff_mass",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 226,
        "prior_journal_sequence": 583,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "continuation_vs_annual_refresh_same_joint_loss_tilt",
        "archive_import_dependencies_are_additional_strategy_experts": False,
        "public_archive_policy_pins_are_additional_active_allocators": False,
        "prior_evidence": unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(DESIGN),
                reference(INVOCATION),
                reference(INTENT),
            ]
        ),
        "validation_evidence": unique_refs(
            [
                *parent["validation_evidence"],
                *validation_references(),
            ]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def integer_advance(mass, relatives, refresh):
    if (
        len(mass) != 6
        or any(type(value) is not int or value <= 0 for value in mass)
        or len(relatives) != 6
        or type(refresh) is not bool
        or any(not math.isfinite(float(value)) or float(value) <= 0 for value in relatives)
    ):
        raise QuantError("The independent learner requires positive homogeneous integer mass.")
    ratios = [float(value).as_integer_ratio() for value in relatives]
    denominator = math.lcm(*(value[1] for value in ratios))
    raw = [mass[i] * ratios[i][0] * (denominator // ratios[i][1]) for i in range(6)]
    total = sum(raw)
    if total <= 0:
        raise QuantError("The independent source interval has no positive learning payoff.")
    score = Fraction(total, sum(mass) * denominator)
    values = [66 * value + total for value in raw] if refresh else raw
    divisor = math.gcd(*values)
    result = tuple(value // divisor for value in values)
    return result, score


def independent_monthly_relatives(observed, period):
    result = []
    for item in observed:
        numerator = denominator = 1
        for value in item.net_returns.loc[period]:
            n, d = float(value).as_integer_ratio()
            if n + d <= 0:
                raise QuantError("An independent source daily relative is not positive.")
            numerator *= n + d
            denominator *= d
            divisor = math.gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor
        result.append(float(Fraction(numerator, denominator)))
    return result


def verify_targets(data, experts, targets, audits, records):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if set(targets) != set(IDS) or len(audits) != len(dates):
        raise QuantError("The independent proof requires every original decision and both rules.")
    stream = iter(records)
    mass = {name: (1,) * 6 for name in IDS}
    previous_day, complete, unscored = None, 0, 0
    maximum_weight = maximum_target = 0.0
    for position, day in enumerate(dates):
        observed = rules.observations(experts, day)
        inputs_record = next(stream, None)
        expected_before = {
            name: [rules.rational_record(Fraction(value, sum(values))) for value in values]
            for name, values in mass.items()
        }
        period, relatives = None, None
        partial = previous_day is not None and not is_month_end(previous_day)
        if previous_day is not None and not partial:
            period = sessions(next_session(previous_day), day)
            relatives = independent_monthly_relatives(observed, period)
        if (
            not isinstance(inputs_record, dict)
            or inputs_record.get("stage") != "input"
            or inputs_record.get("session") != str(day.date())
            or inputs_record.get("previous_decision")
            != (None if previous_day is None else str(previous_day.date()))
            or inputs_record.get("mass_before") != expected_before
            or inputs_record.get("completed_intervals_before") != {name: complete for name in IDS}
            or inputs_record.get("monthly_relative_inputs") != relatives
            or inputs_record.get("monthly_relative_input_hex")
            != (None if relatives is None else [value.hex() for value in relatives])
            or inputs_record.get("source_return_sessions")
            != observed[0].net_returns.index.strftime("%Y-%m-%d").tolist()
            or inputs_record.get("source_NET_returns") != [o.net_returns.tolist() for o in observed]
            or inputs_record.get("source_RF") != observed[0].risk_free.tolist()
        ):
            raise QuantError(f"Independent saved payoff input mismatch at{day.date()}.")
        if relatives is not None:
            complete += 1
        mature = all(item.score_mature for item in observed)
        unscored += not mature
        allocations = {}
        for index, name in enumerate(IDS):
            score = None
            if relatives is not None:
                mass[name], score = integer_advance(mass[name], relatives, index == 1)
            fractions = tuple(Fraction(value, sum(mass[name])) for value in mass[name])
            weights, risk = (
                risk_template.independent_risk(observed, fractions)
                if mature
                else (np.full(6, 1 / 6), None)
            )
            actual = next(stream, None)
            if not isinstance(actual, dict) or (
                actual.get("stage"),
                actual.get("session"),
                actual.get("candidate_id"),
            ) != ("allocation_resolved", str(day.date()), name):
                raise QuantError("Missing or misordered persisted payoff allocation.")
            allocation = actual["allocation"]
            if (
                allocation["learning_mass_after_update"]
                != [rules.rational_record(value) for value in fractions]
                or allocation["completed_intervals"] != complete
                or allocation["annual_refresh_applied"]
                is not (index == 1 and relatives is not None)
                or allocation["prequential_source_score_relative"]
                != (None if score is None else rules.rational_record(score))
            ):
                raise QuantError(f"Independent exact payoff mass or score mismatch at{day.date()}.")
            expected_reason = (
                "mature_payoff_mass_and_joint_loss_assessment"
                if mature
                else "unscored_initial_equal_weights"
            )
            if allocation["reason"] != expected_reason:
                raise QuantError("Preserve explicit original main risk-maturity behavior.")
            if mature:
                candidate_risk = allocation["co_loss_assessment"]
                if (
                    candidate_risk["expert_weights_after_tilt"]
                    != allocation["applied_expert_weights"]
                ):
                    raise QuantError(
                        "The saved shared-risk weights differ from the actual allocation."
                    )
                for key, value in risk.items():
                    if candidate_risk[key] != value:
                        raise QuantError(f"Independent shared joint-loss mismatch:{key}.")
            elif allocation["co_loss_assessment"] is not None:
                raise QuantError("An unscored initial intention cannot invent a risk assessment.")
            actual_weights = np.asarray(allocation["applied_expert_weights"], dtype=float)
            if actual_weights.shape != (6,) or not np.isfinite(actual_weights).all():
                raise QuantError("Require finite complete resolved expert fractions.")
            difference = float(abs(actual_weights - weights).max())
            maximum_weight = max(maximum_weight, difference)
            if difference > rules.TOLERANCE:
                raise QuantError(f"Independent payoff risk composition disagrees at{day.date()}.")
            expected_target = np.array(
                [
                    math.fsum(
                        float(weights[i]) * float(observed[i].target[symbol]) for i in range(6)
                    )
                    for symbol in data.close.columns
                ]
            )
            actual_target = targets[name].loc[day].to_numpy(dtype=float)
            recorded_target = np.asarray(actual["unified_target"], dtype=float)
            if (
                actual_target.shape != expected_target.shape
                or recorded_target.shape != expected_target.shape
                or not np.isfinite(actual_target).all()
                or not np.isfinite(recorded_target).all()
                or not np.array_equal(recorded_target, actual_target)
            ):
                raise QuantError("Retain the actual complete resolved underlying target.")
            maximum_target = max(maximum_target, float(abs(actual_target - expected_target).max()))
            if maximum_target > rules.TOLERANCE:
                raise QuantError(f"Independent payoff underlying target disagrees at{day.date()}.")
            allocations[name] = allocation
        whole = next(stream, None)
        audit = audits[position]
        if (
            not isinstance(whole, dict)
            or whole != {"stage": "resolved", "session": str(day.date()), "audit": audit}
            or audit["allocations"] != allocations
            or audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["previous_decision"]
            != (None if previous_day is None else str(previous_day.date()))
            or audit["source_period_first_session"]
            != (None if period is None else str(period[0].date()))
            or audit["source_period_last_session"]
            != (None if period is None else str(period[-1].date()))
            or audit["source_period_sessions"] != (0 if period is None else len(period))
            or audit["complete_learning_intervals"] != complete
            or audit["source_assessment_mature"] is not mature
            or audit["source_observation_counts"] != [len(o.net_returns) for o in observed]
            or audit["initial_partial_source_month_not_assimilated"] is not partial
            or audit["monthly_relative_inputs"] != relatives
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or any(
                audit.get(key) is not False
                for key in (
                    "learning_state_is_main_wealth",
                    "learning_mass_is_calibrated_probability",
                    "source_shadow_wealth_or_protection_floor_transferred",
                    "independent_forward_validation",
                )
            )
        ):
            raise QuantError(
                "The whole payoff decision audit or genuine introduction clock changed."
            )
        previous_day = day
    if next(stream, None) is not None:
        raise QuantError("Do not drop or append real learning records.")
    return {
        "decisions_checked": len(dates),
        "complete_source_months_checked": complete,
        "unscored_initial_decisions": unscored,
        "exact_integer_learning_mass_and_source_scores_match": True,
        "binary64_monthly_inputs_and_hex_match_independent_integer_products": True,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_underlying_target_difference": maximum_target,
        "source_joint_loss_statistics_match_exactly": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation rather than regenerate it.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve the first unresolved preparation;no silent retry.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    new_output_directory(TARGETS)
    learning = TARGETS / "learning-input-output-records.jsonl"
    with learning.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        targets, audits = rules.build_targets(
            experts,
            data.close,
            read_json(rules.POLICY),
            before_decision=runtime_guard,
            record_learning=retain,
        )
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    for name in IDS:
        new_output_directory(TARGETS / name)
        write_text_atomic(
            TARGETS / name / "targets.csv", targets[name].to_csv(float_format="%.17g")
        )
    independent = verify_targets(
        data, experts, targets, audits, risk_template.learning_records(learning)
    )
    if (
        len(audits) != 121
        or independent["complete_source_months_checked"] != 119
        or (independent["unscored_initial_decisions"] != 7)
    ):
        raise QuantError(
            "Retain every original decision,complete month and initial maturity label."
        )
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day],
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future source observations changed a prior payoff allocation.")
        prefixes.append(str(day.date()))
    entries = {}
    for name in IDS:
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            known = targets[name].ffill().loc[previous_session(window["start"]) : window["end"]]
            active = [
                symbol
                for symbol in participation.source.donor.rules.CORE
                if (known[symbol] > 1e-10).any()
            ]
            if (
                previous_session(window["start"]) not in known.index
                or known.isna().any().any()
                or (len(active) < 3)
            ):
                raise QuantError(
                    "Keep every formal anchor and measured registered-factor coverage."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": active})
        directory = TARGETS / name
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "decision_count": len(audits),
                **independent,
                "learning_records": reference(learning),
                "learning_inputs_saved_before_updates": True,
                "information_audits": reference(TARGETS / "information-audits.json"),
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": (
                    "formal main paths,not learning mass or source scores"
                ),
                "original_source_signal_recomputed": False,
                "new_shadow_account_runs": 0,
                "independent_forward_validation": False,
            },
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    check_frozen(frozen)
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": entries}
    )
    return entries


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["study_id"] != frozen["study_id"]
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(entry["target_proof_at"]) > created for entry in entries)
        or any(
            created > pd.Timestamp(event["at"])
            for name in IDS
            for event in matching_events("path_started", name)
        )
        or any(
            packet.get(key) is not False
            for key in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The complete actual pre-account persistence registration changed.")
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in identifiers:
            if (
                matching_events("registered", name)
                and not matching_events("review", name)
                and not (matching_events("technical_failure", name))
            ):
                append_event(
                    base.JOURNAL,
                    "technical_failure",
                    name,
                    {
                        "stage": stage,
                        "reason": str(error),
                        "partial_outputs_retained": True,
                        "candidate_accounts_started": bool(matching_events("path_started", name)),
                    },
                )
    current.progress(
        "expert_payoff_persistence_expired" if expired else "expert_payoff_persistence_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_payoff_persistence_failure=str(error),
        pending_steps=["preserve_terminal_or_interrupted_persistence_version_without_retry"],
    )


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        old = [
            event
            for event in base.journal_events(base.JOURNAL)
            if event["candidate_id"] not in IDS
            and not (
                event["kind"] == "publication"
                and event["body"].get("study_id") == frozen["study_id"]
            )
        ]
        if len(old) != 583 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve every old completed and failed financial identity.")
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("Registered code,policy and inputs cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        pending = [name for name in IDS if not matching_events("target_proof", name)]
        try:
            if pending:
                current.progress(
                    "preparing_registered_expert_payoff_persistence",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, frozen)
                for name in pending:
                    runtime_guard()
                    append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": matching_events("registered", name)[0][
                                "sha256"
                            ],
                            **prepared[name],
                        },
                    )
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
                    "new_economic_hypotheses": 2,
                    "both_rules_share_risk_and_execution": True,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "main paths,not learning mass or source scores",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_payoff_persistence_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_payoff_persistence_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_payoff_persistence_registered_waiting_publication",
            active_compute_job=None,
            expert_payoff_persistence_registration=reference(PACKET),
            pending_steps=["publish_exact_persistence_packet_before_all_eight_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        ref["path"]
        for ref in frozen["code"] + frozen["policies"]
        if f"research/workbench/{ref['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence,not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed, completed = (
            matching_events(kind, name)
            for kind in ("review", "technical_failure", "path_completed")
        )
        if len({event["body"]["path_key"] for event in completed}) != len(completed) or (
            len(reviews) > 1 or len(failed) > 1
        ):
            raise QuantError(
                "Duplicate completion or terminal events cannot choose another winner."
            )
        paths = {
            event["body"]["path_key"]: base.verify_path(
                event["body"]["receipt"], frozen, name, event["body"]["path_key"]
            )
            for event in completed
        }
        if reviews:
            if (
                failed
                or set(paths) != base.PATH_KEYS
                or reviews[0]["body"]["paths"] != paths
                or (
                    reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"]
                    is not True
                )
            ):
                raise QuantError("A complete review requires every actual original path.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or (
                saved["candidates"] != states(frozen)
            ):
                raise QuantError("Read preserved outcomes instead of replaying completed accounts.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_payoff_persistence",
                active_compute_job={
                    "pid": os.getpid(),
                    "study_id": frozen["study_id"],
                },
            )
            for name in IDS:
                active = (name,)
                if matching_events("review", name) or matching_events("technical_failure", name):
                    continue
                target = frame(matching_events("target_proof", name)[0]["body"]["targets"])
                paths = {}
                for window in read_json(base.PROTOCOL)["windows"]:
                    for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_payoff_persistence",
                            active_candidate=name,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                families = participation.actual_factor_participation(paths, frozen)
                passed = all(path["all_point_gates_pass"] for path in paths.values()) and all(
                    row["minimum_three_registered_families_pass"] for row in families.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": families,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            write_new_receipt(
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
                    "new_shadow_account_runs": 0,
                    "new_shadow_recommendation_artifacts": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "expert_payoff_persistence_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_payoff_persistence_results=reference(RESULTS),
                pending_steps=["publish_all_eight_actual_payoff_persistence_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_payoff_persistence_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_payoff_persistence_evaluation_interrupted", active_compute_job=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = states(frozen), read_json(RESULTS)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Publish actual complete outcomes or retained technical failures only.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for name, row in actual.items():
        paths = []
        for key, path in row["paths"].items():
            year, scenario = key.split("-")
            years = int(year[:-1])
            paths.append(
                {
                    **{
                        field: path[field]
                        for field in (
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
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        symbol: lookup[years, scenario, symbol] for symbol in ("SPY", "QQQ")
                    },
                    "actual_factor_participation": row.get("actual_factor_participation", {}).get(
                        key
                    ),
                }
            )
        reviews.append(
            {
                "candidate_id": name,
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
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] in IDS]
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
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed positive-payoff persistence study;no trading."
    )
    parser.add_argument("action", choices=("initialize", "register", "evaluate", "public-result"))
    parser.add_argument("--commit")
    args = parser.parse_args(argv)
    if args.action == "evaluate" and not args.commit:
        parser.error("--commit is required before any new account")
    functions = {"initialize": initialize, "register": register, "public-result": public_result}
    print(
        json.dumps(
            evaluate(args.commit) if args.action == "evaluate" else functions[args.action](),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
