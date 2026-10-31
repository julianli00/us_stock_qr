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
from us_quant import expert_pair_ensemble_v1 as rules
from us_quant import ssd_expert_pair_dominance_v1 as scalar_template
from us_quant import ssd_seven_expert_meta_v1 as source
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.ssd_expert_meta_research_v1 import failure
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-pair-ensemble-20261031-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-pair-ensemble-20261031-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_pair_ensemble_20261031_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_pair_ensemble_20261031_results_v1.json"
IDS = tuple(candidate["id"] for candidate in rules.CANDIDATES)
OWN = (
    "config/expert-pair-ensemble-v1.json",
    "src/us_quant/expert_pair_ensemble_v1.py",
    "src/us_quant/ssd_expert_pair_ensemble_v1.py",
    "tests/test_expert_pair_ensemble_v1.py",
    "tests/test_ssd_expert_pair_ensemble_v1.py",
)
EXTRA_TEST_INPUTS = (
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_ssd_six_expert_meta_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
)
DESIGN_SHA = "35d2ac2f2dfa061cb7c7686cedae1ae4b8ba65c47767574a0b043c7e4c58df4d"
SEVEN_SPEC = "fcb5e06fb0b3acb83c20cf69d0acf2d5e9aea7a1c05e267aea496c59f5dc60f2"
PARENT_SPEC = "715b1ea80d3c345541467bbbb07d8eddbfa0acb8b56a4db6748809a0382b2215"
template = source.source
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame
verify_publication = source.verify_publication


def _call(original, *args):
    return FunctionType(
        original.__code__, {**original.__globals__, **globals()}, argdefs=original.__defaults__
    )(*args)


def dependencies():
    return FunctionType(
        source.admitted.import_template.dependencies.__code__,
        {**globals(), "ast": ast, "prior": source},
    )()


def source_inputs():
    policy, design = read_json(rules.POLICY), read_json(DESIGN)
    rules.validate(policy)
    seven = read_json(base.checked_reference(policy["source_seven_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    parent = read_json(base.checked_reference(policy["source_dominance_registration"]))["entries"][
        0
    ]["specification"]["frozen"]
    if (
        reference(DESIGN)["sha256"] != DESIGN_SHA
        or digest_json(seven) != SEVEN_SPEC
        or digest_json(parent) != PARENT_SPEC
        or policy["source_seven_specification_sha256"] != SEVEN_SPEC
        or policy["source_dominance_specification_sha256"] != PARENT_SPEC
        or seven["policy"] != policy["source_scoring_policy"]
        or design["actual_continuation_event"] != 60756
        or design["actual_invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["real_new_pair_ensemble_scores_weights_targets_or_accounts_computed"] is not False
    ):
        raise QuantError("Preserve the actual pre-outcome ensemble premise and source identities.")
    check_frozen(seven)
    check_frozen(parent)
    source.checked_packet(seven)
    data, experts, old_policy, _, refs, clocks, admission_refs, benchmarks = source.source_inputs()
    if (
        tuple(expert.identity.identifier for expert in experts) != rules.EXPERT_IDS
        or any(
            policy[key] != value or old_policy[key] != value
            for key, value in design["unchanged_law"].items()
        )
        or clocks != design["source_policy_and_admission_clocks"]
    ):
        raise QuantError(
            "Full-menu aggregation cannot retune source clocks or original assessments."
        )
    old = read_json(base.checked_reference(policy["source_pair_controls"]))
    if (
        old["frozen_specification_sha256"] != SEVEN_SPEC
        or set(old["candidates"]) != set(source.IDS)
        or any(
            row["status"] not in {"rejected", "historical_point_gates_passed"}
            for row in old["candidates"].values()
        )
    ):
        raise QuantError("Preserve every actual old control without reclassification or replay.")
    refs = unique_refs(
        [
            *refs,
            *admission_refs,
            *design["source_references"],
            reference(source.PACKET),
            reference(source.INTENT),
            reference(source.PUBLIC_RESULTS),
            policy["source_seven_registration"],
            policy["source_dominance_registration"],
            policy["source_scoring_policy"],
            policy["source_pair_controls"],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, admission_refs, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_uniform_and_ordinal_aggregate_all_scorable_pairs",
        "test_tied_rank_points_are_averaged_without_lexical_weight_bias",
        "test_negative_valid_scores_keep_positive_full_support",
        "test_original_initial_and_degenerate_states_are_unscored",
        "test_actual_inputs_survive_a_failing_new_ensemble",
        "test_prefix_causality_and_separate_ensemble_states",
        "test_independent_scalar_and_integer_mass_oracle",
        "test_oracle_rejects_pair_risk_or_aggregation_mutations",
        "test_retained_ensemble_inputs_and_resolved_records_are_exact",
        "test_cached_sources_and_scalar_helper_never_replay_old_entrypoints",
        "test_shared_publication_wrappers_do_not_recurse",
        "test_failed_publication_cannot_start_main_account",
        "test_canonical_export_preserves_all_paths",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed full-support,integer,scalar,source and execution tests."
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
        raise QuantError("Validation must bind the current code and actual successful outputs.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve a complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        invocation = read_json(INVOCATION)
        if (
            len(events) != 652
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 62
            or counts["new_completed_strategy_evaluations"] != 54
            or counts["new_technical_failures"] != 8
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual unchanged journal and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 60756,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / path) for path in OWN],
                "complete_dependency_inputs": dependencies(),
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_pair_ensemble_scores_weights_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "pair_allocation_ensemble_rule_frozen",
            active_compute_job=None,
            pair_ensemble_intent=reference(INTENT),
            pending_steps=["register_both_before_actual_new_ensemble_assessments"],
        )
    return reference(INTENT)


def inputs():
    data, experts, _, parent, refs, clocks, admission_refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / path) for path in OWN]
        or intent["complete_dependency_inputs"] != dependencies()
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_pair_ensemble_scores_weights_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome ensemble implementation freeze changed.")
    deps = intent["complete_dependency_inputs"]
    frozen = {
        **parent,
        "study_id": read_json(rules.POLICY)["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *[
                    ref
                    for ref in admission_refs + deps
                    if ref["path"].startswith(("src/", "tests/"))
                ],
            ]
        ),
        "policies": unique_refs(
            [
                *parent["policies"],
                *[ref for ref in admission_refs + deps if ref["path"].startswith("config/")],
                reference(rules.POLICY),
            ]
        ),
        "source_seven_expert_pool_policy": reference(source.rules.POLICY),
        "source_previous_dominance_specification": PARENT_SPEC,
        "expert_pool": reference(rules.POLICY),
        "expert_pool_definition_role": "fixed_seven_full_scorable_pair_ERC_aggregation",
        "source_policy_and_expert_admission_clocks": clocks,
        "decision_information_clock": rules.CLOCK,
        "prior_completed_configurations": 232,
        "prior_journal_sequence": 652,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "full_support_uniform_or_tie_ordinal_pair_ERC_ensemble",
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


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or set(audits) != set(IDS)
        or any(
            not target.index.equals(data.close.index)
            or not target.columns.equals(data.close.columns)
            or not target.dropna().index.equals(dates)
            or not target.loc[~target.index.isin(dates)].isna().all().all()
            for target in targets.values()
        )
    ):
        raise QuantError("Keep every original decision and full unified target lattice.")
    maximum_target = maximum_weight = maximum_score = maximum_mass = 0.0
    per_rule = {}
    for candidate in rules.CANDIDATES:
        name, scheme = candidate["id"], candidate["aggregation"]
        rows = audits[name]
        if [row["decision_session"] for row in rows] != dates.strftime("%Y-%m-%d").tolist():
            raise QuantError("Each rule needs every chronological original decision.")
        previous = np.full(7, 1 / 7)
        unscored = mature = degenerate = pair_count = tied_decisions = 0
        support_minimum = 7
        for position, (day, audit) in enumerate(zip(dates, rows, strict=True)):
            observed = [expert.observe(day, 126) for expert in experts]
            evidence = audit["pair_ensemble"]
            if (
                audit["decision_index"] != position
                or audit["decision_information_clock"] != rules.CLOCK
                or audit["decision_role"]
                != ("initial_capital" if position == 0 else "completed_exchange_month_end")
                or tuple(item.identity.identifier for item in observed) != rules.EXPERT_IDS
                or any(
                    item.target is None
                    or item.target_known_at is None
                    or item.target_known_at > day
                    for item in observed
                )
                or audit["previous_expert_weights"] != previous.tolist()
                or audit["expert_target_known_at"]
                != [str(item.target_known_at.date()) for item in observed]
                or audit["preserved_archive_identity_introduced_at"]
                != [item.identity.introduced_at for item in observed]
                or audit["expert_shadow_last_observation"]
                != [
                    str(item.net_returns.index[-1].date()) if len(item.net_returns) else None
                    for item in observed
                ]
                or evidence["aggregation"] != scheme
                or evidence["overlapping_pairs_are_independent_experts_or_capital"] is not False
                or evidence["confidence_interval_or_significance_claimed"] is not False
                or any(
                    audit[key] is not expected
                    for key, expected in (
                        ("counterfactual_historical_expert_availability", True),
                        ("shadow_wealth_added_to_main_capital", False),
                        ("source_feedback_floor_transferred_to_main_account", False),
                        ("independent_forward_validation", False),
                    )
                )
            ):
                raise QuantError("The declared state,source clock or capital separation changed.")
            assessment = evidence["pair_assessments"]
            mass_records = []
            if not all(item.score_mature for item in observed):
                if (
                    position >= 7
                    or assessment["scores_available"] is not False
                    or assessment["pairs"] != []
                    or assessment["selected_pair"] is not None
                    or assessment["reason"] != "unscored_initial_equal_weights"
                    or assessment["safety_or_success_claimed"] is not False
                ):
                    raise QuantError("Seven unmatured states cannot become successful opinions.")
                weights, reason, total, m = np.full(7, 1 / 7), assessment["reason"], None, 0
                unscored += 1
            else:
                weight_gap, score_gap = scalar_template._verify_pair_assessments(
                    observed, previous, data.close.columns, assessment
                )
                maximum_weight = max(maximum_weight, weight_gap)
                maximum_score = max(maximum_score, score_gap)
                scored = sorted(
                    (pair for pair in assessment["pairs"] if "unscorable_reason" not in pair),
                    key=lambda pair: (-pair["selection_score_key"], pair["pair"]),
                )
                m = len(scored)
                if not scored:
                    if (
                        assessment["scores_available"] is not False
                        or assessment["reason"] != "unscored_degenerate_pair_equal_weights"
                        or assessment["selected_pair"] is not None
                    ):
                        raise QuantError("Unranked degenerate opinions cannot be hidden.")
                    weights, reason, total = np.full(7, 1 / 7), assessment["reason"], None
                    degenerate += 1
                else:
                    known = observed[0].net_returns.index
                    if (
                        assessment["scores_available"] is not True
                        or assessment["selected_pair"] != scored[0]["pair"]
                        or any(
                            assessment.get(key) != str(known[index].date())
                            for key, index in (
                                ("train_start", 0),
                                ("train_end", 83),
                                ("validation_start", 84),
                                ("validation_end", -1),
                            )
                        )
                        or known[-1] > day
                    ):
                        raise QuantError("Use the original past-only pair score and risk clock.")
                    keys = [pair["selection_score_key"] for pair in scored]
                    integers = [
                        1
                        if scheme == rules.SCHEMES[0]
                        else 2 * m - 2 * sum(other > key for other in keys) - keys.count(key) + 1
                        for key in keys
                    ]
                    denominator = sum(integers)
                    if min(integers) <= 0 or denominator != (
                        m if scheme == rules.SCHEMES[0] else m * (m + 1)
                    ):
                        raise QuantError(
                            "Independent integer rank mass needs full positive support."
                        )
                    masses = [value / denominator for value in integers]
                    total = m if scheme == rules.SCHEMES[0] else denominator / 2
                    for pair, integer, mass in zip(scored, integers, masses, strict=True):
                        mass_records.append(
                            {
                                "pair": pair["pair"],
                                "selection_score_key": pair["selection_score_key"],
                                "aggregation_points": (
                                    integer if scheme == rules.SCHEMES[0] else integer / 2
                                ),
                                "pair_mass": mass,
                            }
                        )
                    weights = np.asarray(
                        [
                            math.fsum(
                                mass * float(pair["expert_weights"][i])
                                for pair, mass in zip(scored, masses, strict=True)
                            )
                            for i in range(7)
                        ]
                    )
                    reason = "full_support_pair_ERC_aggregation"
                    mature += 1
                    pair_count += m
                    tied_decisions += len(set(keys)) != m
                    support_minimum = min(support_minimum, int((weights > 0).sum()))
            actual_masses = evidence["pair_masses"]
            if (
                evidence["reason"] != reason
                or evidence["scorable_pair_count"] != m
                or evidence["aggregation_point_total"] != total
                or len(actual_masses) != len(mass_records)
            ):
                raise QuantError("Recorded full-support aggregation disagrees with integer proof.")
            for actual, expected in zip(actual_masses, mass_records, strict=True):
                if any(
                    actual[key] != expected[key]
                    for key in ("pair", "selection_score_key", "aggregation_points")
                ) or (
                    type(actual["pair_mass"]) not in (int, float)
                    or not math.isfinite(actual["pair_mass"])
                    or actual["pair_mass"] <= 0
                ):
                    raise QuantError("Every original pair and tied rank point must be retained.")
                maximum_mass = max(maximum_mass, abs(actual["pair_mass"] - expected["pair_mass"]))
            for actual in (audit["expert_weights"], evidence["chosen_expert_weights"]):
                if (
                    np.asarray(actual).shape != (7,)
                    or not np.isfinite(actual).all()
                    or min(actual) < 0
                    or abs(math.fsum(actual) - 1) > 1e-12
                ):
                    raise QuantError("Use one nonnegative unit-sum expert allocation.")
                maximum_weight = max(maximum_weight, float(max(abs(weights - actual))))
            expected_target = [
                math.fsum(
                    float(item.target[symbol]) * float(weights[i])
                    for i, item in enumerate(observed)
                )
                for symbol in data.close.columns
            ]
            maximum_target = max(
                maximum_target, float(max(abs(targets[name].loc[day] - expected_target)))
            )
            previous = np.asarray(audit["expert_weights"])
        if unscored != min(7, len(rows)):
            raise QuantError("Measure every original unscored initial decision.")
        per_rule[name] = {
            "decisions_checked": len(rows),
            "unscored_initial_decisions": unscored,
            "mature_scored_decisions": mature,
            "mature_degenerate_equal_fallbacks": degenerate,
            "scorable_pair_allocations_aggregated": pair_count,
            "decisions_with_tied_score_keys": tied_decisions,
            "minimum_positive_expert_support_in_scored_decisions": (
                support_minimum if mature else None
            ),
        }
    if (
        maximum_target > 1e-10
        or maximum_weight > 1e-10
        or maximum_score > 1e-8
        or maximum_mass > 1e-12
    ):
        raise QuantError("Original scalar risk,score,target or exact rank-mass limits failed.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
        "maximum_independent_pair_mass_difference": maximum_mass,
        "independent_integer_rank_mass_and_full_support_verified": True,
        "per_rule": per_rule,
    }


def verify_retained_records(path, experts, targets, audits):
    count = values = 0
    with path.open() as stream:
        for name in IDS:
            for position, audit in enumerate(audits[name]):
                before, after = json.loads(stream.readline()), json.loads(stream.readline())
                day = pd.Timestamp(audit["decision_session"])
                if (
                    before["record"] != "decision_input"
                    or after["record"] != "decision_resolved"
                    or before["candidate_id"] != name
                    or after["candidate_id"] != name
                    or before["decision_index"] != position
                    or after["decision_index"] != position
                    or before["as_of"] != audit["decision_session"]
                    or before["information_clock"] != rules.CLOCK
                    or before["previous_weights"] != audit["previous_expert_weights"]
                    or before["previous_weights_binary64_hex"]
                    != [float(x).hex() for x in audit["previous_expert_weights"]]
                    or before["binary64_values_truncated_to_fixed_decimals"] is not False
                    or after["audit"] != audit
                    or after["asset_order"] != list(targets[name].columns)
                    or after["underlying_target"] != targets[name].loc[day].tolist()
                ):
                    raise QuantError("Actual ensemble inputs or resolved records changed.")
                expected = [
                    rules.observation_record(expert.observe(day, 126)) for expert in experts
                ]
                if before["observations"] != expected:
                    raise QuantError(
                        "Retained binary64 values differ from causal source observations."
                    )
                values += sum(len(row["NET_returns"]) + len(row["RF"]) for row in expected)
                count += 2
        if stream.readline():
            raise QuantError("Retain exactly one input and resolved record per rule decision.")
    return {
        "actual_input_and_resolved_record_count": count,
        "roundtrip_NET_RF_binary64_values_checked": values,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation,not regenerate targets.")
        for row in saved["entries"].values():
            for ref in row.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve unresolved preparation instead of repeating it.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits)
    independent = verify_targets(data, experts, targets, audits)
    if any(len(rows) != 121 for rows in audits.values()):
        raise QuantError("Keep all121original decisions for both independent ensemble states.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[IDS[0]][position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if any(
            not early[name].equals(targets[name].loc[:day])
            or earlier[name] != audits[name][: position + 1]
            for name in IDS
        ):
            raise QuantError("Future source inputs changed an earlier ensemble decision.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"candidates": audits})
    entries = {}
    for name in IDS:
        directory = TARGETS / name
        new_output_directory(directory)
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = targets[name].ffill().loc[anchor : window["end"]]
            actual = [
                symbol
                for symbol in source.source.previous.previous.source.donor.rules.CORE
                if (known[symbol] > 1e-10).any()
            ]
            if anchor not in known.index or known.isna().any().any() or len(actual) < 3:
                raise QuantError("Every formal anchor and genuine factor coverage is required.")
            coverage.append({"years": window["years"], "actual_factor_symbols": actual})
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "decision_count": len(audits[name]),
                "information_audits": reference(FOLDER / "information-audits-v1.json"),
                "actual_input_resolved_records": reference(log),
                "inputs_saved_before_updates_and_proof": True,
                **retained,
                **independent,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "decision_information_clock": rules.CLOCK,
                "overlapping_pairs_are_independent_experts_or_capital": False,
                "new_shadow_account_runs": 0,
                "new_shadow_recommendation_artifacts": 0,
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": "formal main paths,not mature source-shadow scores",
                "original_source_signal_recomputed": False,
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
    return _call(template.checked_packet, frozen)


def publication(frozen, commit):
    return _call(template.publication, frozen, commit)


def states(frozen):
    return _call(template.states, frozen)


def public_result():
    return _call(template.public_result)


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        old = [
            event for event in base.journal_events(base.JOURNAL) if event["candidate_id"] not in IDS
        ]
        if len(old) != 652 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve all old completed and failed financial identities.")
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("A registered policy or input cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        pending = [name for name in IDS if not matching_events("target_proof", name)]
        try:
            if pending:
                current.progress(
                    "preparing_registered_pair_ensemble_rules",
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
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "formal main paths,not source-shadow scores",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("pair_ensemble_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress("pair_ensemble_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "pair_ensemble_registered_waiting_publication",
            active_compute_job=None,
            pair_ensemble_registration=reference(PACKET),
            pending_steps=["publish_exact_packet_before_eight_accounts"],
        )
    return reference(PACKET)


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
                raise QuantError("Preserve complete outcomes without replay.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_pair_ensemble_rules",
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
                            "evaluating_pair_ensemble_rules", active_candidate=name, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                participation = template.previous.previous.actual_factor_participation(
                    paths, frozen
                )
                passed = all(path["all_point_gates_pass"] for path in paths.values()) and all(
                    value["minimum_three_registered_families_pass"]
                    for value in participation.values()
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
                        "actual_factor_participation": participation,
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
                "pair_ensemble_financial_outcomes_complete",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pair_ensemble_results=reference(RESULTS),
                pending_steps=["publish_all_eight_actual_outcomes"],
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("pair_ensemble_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "pair_ensemble_evaluation_interrupted",
                active_compute_job=None,
                active_candidate=None,
            )
            raise
    return reference(RESULTS)


def main():
    parser = argparse.ArgumentParser(description="Fixed full-support pair ensembles;no trading.")
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Pair ensemble research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Pair ensemble research blocked:{error}\n")


if __name__ == "__main__":
    main()
