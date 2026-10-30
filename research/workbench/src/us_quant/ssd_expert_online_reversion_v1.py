from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_online_reversion_v1 as rules
from us_quant import ssd_expert_hierarchical_risk_v1 as import_template
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant.calendar import previous_session, sessions
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
FOLDER = base.REPORTS / "expert-online-reversion-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-online-reversion-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_online_reversion_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_online_reversion_20261030_results_v1.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/expert-online-reversion-v1.json",
    "src/us_quant/expert_online_reversion_v1.py",
    "src/us_quant/ssd_expert_online_reversion_v1.py",
    "tests/test_expert_online_reversion_v1.py",
    "tests/test_ssd_expert_online_reversion_v1.py",
)
SHARED_CODE = "src/us_quant/online_reversion.py"
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame


def dependency_inputs():
    # The frozen traversal captures OWN; give a private copy this study's globals.
    traversal = FunctionType(import_template.dependency_inputs.__code__, {**globals(), "ast": ast})
    return traversal()


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    design = read_json(DESIGN)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or design["policy"] != reference(rules.POLICY)
        or design["actual_invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["source_specification_sha256"] != PARENT_SPEC
        or design["real_new_learning_or_targets_computed"] is not False
        or design["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("Keep the actual pre-outcome design and exact six-expert source.")
    check_frozen(parent)
    source.checked_packet(parent)
    data, experts, old_policy, _, old_refs, clocks, admission_refs, benchmarks = (
        source.source_inputs()
    )
    if any(
        policy[k] != old_policy[k]
        for k in (
            "expert_ids",
            "mechanism_groups",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
        )
    ):
        raise QuantError("Reverse the allocation premise, not expert admission or execution.")
    controls = read_json(base.checked_reference(policy["source_controls"]))
    if (
        controls["registration_packet"] != policy["source_registration"]
        or {r["candidate_id"] for r in controls["reviews"]} != set(source.IDS)
        or any(r["status"] != "rejected" or len(r["paths"]) != 4 for r in controls["reviews"])
    ):
        raise QuantError("Keep every exposed original control and its financial rejection.")
    refs = unique_refs(
        [
            *old_refs,
            *admission_refs,
            *design["source_references"],
            reference(source.INTENT),
            *[
                policy[k]
                for k in (
                    "source_registration",
                    "source_controls",
                    "previous_growth_results",
                    "prior_terminal_failure",
                    "prior_art_policy",
                    "pure_update_implementation",
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
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    required = {
        "test_prior_art_parameters_and_countercyclical_role_are_fixed",
        "test_prior_art_updates_and_flat_forecast_guard",
        "test_initial_formation_and_monthly_main_clock",
        "test_future_source_changes_preserve_prior_targets_and_records",
        "test_source_gaps_and_information_clock_fail_closed",
        "test_independent_daily_replay_and_composition",
        "test_independent_verifier_rejects_learning_and_clock_tampering",
        "test_learning_inputs_are_saved_before_a_failing_update",
        "test_cached_sources_never_call_closed_entrypoints",
        "test_complete_import_closure_is_bound_without_mutating_the_template",
        "test_publication_checks_all_dependencies_before_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_original_eight_paths_without_private_rows",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed learning, clock, independent and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"]
        != [
            reference(ROOT / p) for p in (*OWN, "tests/test_expert_initial_monthly_execution_v1.py")
        ]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current implementation and test bytes.")
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
            len(events) != 556
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 52
            or counts["new_completed_strategy_evaluations"] != 46
            or counts["new_technical_failures"] != 6
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
            or read_json(DESIGN)["actual_continuation_event"] != 53048
        ):
            raise QuantError("Freeze against the actual post-failure journal and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 53048,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_meta_allocations_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_online_reversion_frozen",
            active_compute_job=None,
            expert_online_reversion_intent=reference(INTENT),
            pending_steps=["register_both_rules_before_real_daily_learning_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != PARENT_SPEC
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_allocations_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome implementation freeze changed.")
    code, policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs([*parent["code"], *code, *[reference(ROOT / p) for p in OWN[1:]]]),
        "policies": unique_refs([*parent["policies"], *policies, reference(rules.POLICY)]),
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_countercyclical_NET_learning",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 224,
        "prior_journal_sequence": 556,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_two_fixed_OLMAR_PAMR_allocation_laws",
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
            [*parent["validation_evidence"], *validation_references()]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def close_array(actual, expected, message, tolerance=rules.TOLERANCE):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or float(abs(a - b).max(initial=0)) > tolerance
    ):
        raise QuantError(message)


def independent_projection(proposal):
    centered = [float(v) - max(proposal) for v in proposal]
    active = list(range(6))
    while active:
        threshold = (math.fsum(centered[i] for i in active) - 1) / len(active)
        next_active = [i for i in active if centered[i] > threshold]
        if next_active == active:
            answer = [max(value - threshold, 0.0) for value in centered]
            total = math.fsum(answer)
            if not math.isfinite(total) or total <= 0:
                raise QuantError("The independent Euclidean projection has no finite unit budget.")
            return np.array([value / total for value in answer])
        active = next_active
    raise QuantError("The independent Euclidean active set is empty.")


def independent_update(prior, history, candidate, count):
    if count < 6:
        return np.array(prior), None, "unscored_formation_warmup"
    forecast = [
        math.fsum(row[i] for row in history[-5:]) / 5 / history[-1][i]
        if candidate["method"] == "olmar"
        else history[-1][i] / history[-2][i]
        for i in range(6)
    ]
    if max(forecast) == min(forecast):
        return np.array(prior), forecast, "constant_forecast_no_new_relative_information"
    mean = math.fsum(forecast) / 6
    score = math.fsum(float(prior[i]) * forecast[i] for i in range(6))
    loss = max(
        0.0,
        candidate["epsilon"] - score
        if candidate["method"] == "olmar"
        else score - candidate["epsilon"],
    )
    variance = math.fsum((value - mean) ** 2 for value in forecast)
    if variance == 0 or loss == 0:
        result = np.array(prior)
    else:
        sign = 1 if candidate["method"] == "olmar" else -1
        multiplier = loss / variance
        proposal = [float(prior[i]) + sign * multiplier * (forecast[i] - mean) for i in range(6)]
        if not all(math.isfinite(value) for value in proposal):
            raise QuantError("Independent reversal arithmetic is nonfinite.")
        result = independent_projection(proposal)
    return result, forecast, "observed_NET_expert_reversal_update"


def learning_records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def verify_targets(data, experts, targets, audits, records):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or len(audits) != len(dates)
        or any(
            not target.index.equals(data.close.index)
            or not target.columns.equals(data.close.columns)
            or not target.dropna(how="all").index.equals(dates)
            for target in targets.values()
        )
    ):
        raise QuantError("Keep the entire original calendar and initial/month-end target lattice.")
    recorded = iter(records)
    weights = {name: np.full(6, 1 / 6) for name in IDS}
    previous_main = {name: value.copy() for name, value in weights.items()}
    levels = [10000.0] * 6
    history = [levels.copy()]
    reasons = {name: "unscored_formation_warmup" for name in IDS}
    updates = decision = rows = 0
    maximum_weight = maximum_target = maximum_forecast = 0.0
    source_calendar = data.close.index[data.close.index >= dates[0]]

    def take(stage, day, name):
        nonlocal rows
        row = next(recorded, None)
        if (
            row is None
            or row["stage"] != stage
            or row["session"] != str(day.date())
            or row["candidate_id"] != name
        ):
            raise QuantError("Every recorded learner input/output must retain its exact clock.")
        rows += 1
        return row

    for source_day, day in enumerate(source_calendar):
        observed = [expert.observe(day, 2) for expert in experts]
        expected_dates = (
            sessions(source_calendar[max(1, source_day - 1)], day)
            if source_day
            else pd.DatetimeIndex([])
        )
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or tuple(o.identity.mechanism_group for o in observed) != rules.MECHANISMS
            or any(
                o.as_of != day
                or o.target is None
                or o.target_known_at is None
                or o.target_known_at > day
                or len(o.net_returns) != min(source_day, 2)
                or not o.net_returns.index.equals(expected_dates)
                or not o.net_returns.index.equals(o.risk_free.index)
                or not np.isfinite(o.net_returns).all()
                or not np.isfinite(o.risk_free).all()
                or (o.net_returns <= -1).any()
                or source_day
                and o.net_returns.index[-1] != day
                for o in observed
            )
            or any(
                not o.net_returns.index.equals(observed[0].net_returns.index)
                or not o.risk_free.equals(observed[0].risk_free)
                for o in observed
            )
        ):
            raise QuantError("Independent source availability and NET observation clocks disagree.")
        if source_day:
            relatives = [1 + float(o.net_returns.iloc[-1]) for o in observed]
            before = levels.copy()
            levels = [before[i] * relatives[i] for i in range(6)]
            history = [*history, levels.copy()][-6:]
            for candidate in rules.CANDIDATES:
                name = candidate["id"]
                entry = take("input", day, name)
                if entry["source_price_observations"] != source_day + 1:
                    raise QuantError("A learning observation was dropped or backdated.")
                for field, expected in (
                    ("source_NET_relatives", relatives),
                    ("source_wealth_before", before),
                    ("source_history_after", history),
                    ("algorithm_weights_before", weights[name]),
                ):
                    close_array(entry[field], expected, "The retained learning operands changed.")
                expected, forecast, reason = independent_update(
                    weights[name], history, candidate, source_day + 1
                )
                resolved = take("resolved", day, name)
                close_array(
                    resolved["algorithm_weights_before"],
                    weights[name],
                    "The independent previous allocation changed.",
                )
                close_array(
                    resolved["algorithm_weights_after"],
                    expected,
                    "Independent reversal weights disagree.",
                )
                maximum_weight = max(
                    maximum_weight,
                    float(abs(np.asarray(resolved["algorithm_weights_after"]) - expected).max()),
                )
                if forecast is None:
                    if resolved["forecast"] is not None:
                        raise QuantError("Unmatured formation cannot invent a forecast.")
                else:
                    close_array(
                        resolved["forecast"], forecast, "The independent forecast disagrees."
                    )
                    maximum_forecast = max(
                        maximum_forecast,
                        float(abs(np.asarray(resolved["forecast"]) - forecast).max()),
                    )
                if (
                    resolved["formation_price_observations"] != source_day + 1
                    or resolved["formation_mature"] is not (source_day + 1 >= 6)
                    or resolved["reason"] != reason
                    or resolved["criterion_or_profit_guaranteed"] is not False
                ):
                    raise QuantError(
                        "A formation state or statistical premise became a success claim."
                    )
                weights[name], reasons[name] = expected, reason
            updates += source_day + 1 >= 6
        if day not in dates:
            continue
        audit = audits[decision]
        if (
            audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if decision == 0 else "completed_exchange_month_end")
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["source_shadow_last_observation"] != (str(day.date()) if source_day else None)
            or audit["source_price_observations"] != source_day + 1
            or audit["source_NET_daily_observations"] != source_day
            or audit["statistical_daily_updates_per_candidate"] != updates
            or audit["source_assessment_mature"] is not (source_day + 1 >= 6)
            or audit["latest_reversal_update_reason"] != reasons
            or audit["learning_state_is_main_wealth"] is not False
            or audit["source_shadow_wealth_or_protection_floor_transferred"] is not False
            or audit["universal_NET_wealth_identity_or_bound_claimed"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The independent monthly clock or capital separation changed.")
        for name in IDS:
            close_array(
                audit["previous_applied_expert_weights"][name],
                previous_main[name],
                "Previous main intentions cannot be replaced by daily learner weights.",
            )
            close_array(
                audit["applied_expert_weights"][name],
                weights[name],
                "Applied allocations disagree.",
            )
            expected_target = [
                math.fsum(float(weights[name][i]) * float(observed[i].target[s]) for i in range(6))
                for s in data.close.columns
            ]
            actual = targets[name].loc[day].to_numpy()
            close_array(actual, expected_target, "The independent underlying target disagrees.")
            maximum_target = max(maximum_target, float(abs(actual - expected_target).max()))
        previous_main = {name: value.copy() for name, value in weights.items()}
        decision += 1
    if next(recorded, None) is not None:
        raise QuantError("Extra learner records cannot be hidden behind the last main decision.")
    return {
        "independent_complete_daily_source_observations": len(source_calendar) - 1,
        "independent_statistical_updates_per_candidate": updates,
        "retained_learning_input_output_rows_checked": rows,
        "all_monthly_decisions_checked": decision,
        "unscored_initial_main_decisions": sum(not a["source_assessment_mature"] for a in audits),
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_scalar_target_difference": maximum_target,
        "maximum_independent_forecast_difference": maximum_forecast,
        "main_account_or_old_source_pipeline_replayed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation rather than regenerate it.")
        for row in saved["entries"].values():
            for ref in row.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve an unresolved preparation; no silent retry.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    new_output_directory(TARGETS)
    learning = TARGETS / "learning-input-output-records.jsonl"
    with learning.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
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
    independent = verify_targets(data, experts, targets, audits, learning_records(learning))
    if (
        len(audits) != 121
        or independent["independent_complete_daily_source_observations"] != 2512
        or independent["independent_statistical_updates_per_candidate"] != 2508
        or independent["unscored_initial_main_decisions"] != 1
    ):
        raise QuantError("Keep every original source day, formation state and main decision.")
    start = data.close.index.get_loc(pd.Timestamp("2016-10-05"))
    prefix_dates = [
        data.close.index[start],
        data.close.index[start + 4],
        data.close.index[start + 5],
        pd.Timestamp(audits[1]["decision_session"]),
        pd.Timestamp(audits[60]["decision_session"]),
        pd.Timestamp(audits[-1]["decision_session"]),
        data.close.index[-1],
    ]
    for day in prefix_dates:
        runtime_guard()
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if any(not early[n].equals(targets[n].loc[:day]) for n in IDS) or earlier != [
            a for a in audits if pd.Timestamp(a["decision_session"]) <= day
        ]:
            raise QuantError("Future source observations changed a prior reversal allocation.")
    entries = {}
    for name in IDS:
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = targets[name].ffill().loc[anchor : window["end"]]
            active = [s for s in participation.source.donor.rules.CORE if (known[s] > 1e-10).any()]
            if anchor not in known.index or known.isna().any().any() or len(active) < 3:
                raise QuantError("Retain every initial anchor and measured source-factor coverage.")
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
                "prefix_truncations": [str(d.date()) for d in prefix_dates],
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": "formal main paths,not reversal-forecast statistics",
                "original_source_signal_recomputed": False,
                "new_shadow_account_runs": 0,
                "new_shadow_recommendation_artifacts": 0,
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
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(entry["target_proof_at"]) > created for entry in entries)
        or any(
            created > pd.Timestamp(e["at"])
            for name in IDS
            for e in matching_events("path_started", name)
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
        raise QuantError("The complete actual pre-account reversal registration packet changed.")
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in identifiers:
            if (
                matching_events("registered", name)
                and not matching_events("review", name)
                and not matching_events("technical_failure", name)
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
        "expert_online_reversion_expired" if expired else "expert_online_reversion_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_reversion_study_failure=str(error),
        pending_steps=["preserve_terminal_or_interrupted_reversion_version_without_retry"],
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
        if len(old) != 556 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve every old completed and failed financial identity.")
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("Registered code, policy and source inputs cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        pending = [name for name in IDS if not matching_events("target_proof", name)]
        try:
            if pending:
                current.progress(
                    "preparing_registered_expert_online_reversion",
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
                    "entries": [common.packet_entry(c, frozen) for c in rules.CANDIDATES],
                    "new_registered_candidates": 2,
                    "new_economic_hypotheses": 2,
                    "same_countercyclical_allocation_family_not_independent_factor_count": True,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "formal main paths,not reversal-forecast statistics",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_online_reversion_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_online_reversion_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_online_reversion_registered_waiting_publication",
            active_compute_job=None,
            expert_online_reversion_registration=reference(PACKET),
            pending_steps=["publish_exact_reversion_packet_before_all_eight_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        event
        for event in matching_events("publication")
        if event["body"].get("study_id") == frozen["study_id"]
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
        if (
            len({event["body"]["path_key"] for event in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError(
                "Duplicate completion or terminal events cannot select another winner."
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
                or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"]
                is not True
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
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Read preserved outcomes instead of replaying completed accounts.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_online_reversion",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
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
                            "evaluating_expert_online_reversion",
                            active_candidate=name,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                actual = participation.actual_factor_participation(paths, frozen)
                passed = all(path["all_point_gates_pass"] for path in paths.values()) and all(
                    item["minimum_three_registered_families_pass"] for item in actual.values()
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
                        "actual_factor_participation": actual,
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
                "expert_online_reversion_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_online_reversion_results=reference(RESULTS),
                pending_steps=["publish_all_eight_reversion_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_online_reversion_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_online_reversion_evaluation_interrupted", active_compute_job=None
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
        [event for event in base.journal_events(base.JOURNAL) if event["candidate_id"] in IDS]
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


def main():
    parser = argparse.ArgumentParser(description="Two fixed expert-level reversal laws;no trading.")
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Expert-reversal research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication SHA.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Expert-reversal research blocked: {error}\n")


if __name__ == "__main__":
    main()
