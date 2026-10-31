from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
import sys
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_pair_dominance_v1 as rules
from us_quant import ssd_seven_expert_meta_v1 as source
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.ssd_expert_meta_research_v1 import failure, scalar_score
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-pair-dominance-20261031-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-pair-dominance-20261031-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_pair_dominance_20261031_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_pair_dominance_20261031_results_v1.json"
IDS = tuple(candidate["id"] for candidate in rules.CANDIDATES)
OWN = (
    "config/expert-pair-dominance-v1.json",
    "src/us_quant/expert_pair_dominance_v1.py",
    "src/us_quant/ssd_expert_pair_dominance_v1.py",
    "tests/test_expert_pair_dominance_v1.py",
    "tests/test_ssd_expert_pair_dominance_v1.py",
)
EXTRA_TEST_INPUTS = (
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_ssd_seven_expert_meta_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
)
DESIGN_SHA = "015dc6b09ebf43080dafe428ca59c8f3f8fd1d64c536e1b9c90a9356b9b9d9f8"
PARENT_SPEC = "fcb5e06fb0b3acb83c20cf69d0acf2d5e9aea7a1c05e267aea496c59f5dc60f2"
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
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if (
        reference(DESIGN)["sha256"] != DESIGN_SHA
        or digest_json(parent) != PARENT_SPEC
        or parent["policy"] != policy["source_scoring_policy"]
        or policy["source_specification_sha256"] != PARENT_SPEC
        or design["actual_continuation_event"] != 59782
        or design["actual_invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["real_new_dominance_assessments_states_targets_or_accounts_computed"] is not False
        or design["expert_ids"] != list(rules.EXPERT_IDS)
    ):
        raise QuantError(
            "Preserve the actual pre-outcome dominance premise and original seven pool."
        )
    check_frozen(parent)
    source.checked_packet(parent)
    old_results = read_json(base.checked_reference(policy["source_pair_controls"]))
    if (
        old_results["frozen_specification_sha256"] != PARENT_SPEC
        or set(old_results["candidates"]) != set(source.IDS)
        or any(
            row["status"] not in {"rejected", "historical_point_gates_passed"}
            for row in old_results["candidates"].values()
        )
    ):
        raise QuantError(
            "Reuse every original completed control without reclassification or replay."
        )
    data, experts, old_policy, _, refs, clocks, admission_refs, benchmarks = source.source_inputs()
    if (
        tuple(expert.identity.identifier for expert in experts) != rules.EXPERT_IDS
        or any(
            policy[key] != value or old_policy[key] != value
            for key, value in design["unchanged_law"].items()
        )
        or policy["decision_information_clock"] != rules.CLOCK
    ):
        raise QuantError("Pair commitment cannot retune the underlying assessment or execution.")
    menu = read_json(base.checked_reference(policy["source_menu_diagnosis"]))
    if (
        menu["new_strategy_evaluations"] != 0
        or menu["new_financial_configuration_selected"] is not False
        or menu["summary"]["recorded_seven_adjacent_pair_switches"] != 93
        or menu["summary"]["recorded_six_adjacent_pair_switches"] != 83
        or menu["summary"]["restricted_ranking_replays_original_six_strategy_or_previous_state"]
        is not False
    ):
        raise QuantError(
            "The saved menu motivation is not a replayed control or a new trading rule."
        )
    refs = unique_refs(
        [
            *refs,
            *admission_refs,
            *design["source_references"],
            reference(source.PACKET),
            reference(source.INTENT),
            reference(source.PUBLIC_RESULTS),
            policy["source_registration"],
            policy["source_scoring_policy"],
            policy["source_pair_controls"],
            policy["source_menu_diagnosis"],
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
        "test_pareto_and_empirical_envelope_gates_are_distinct",
        "test_strict_ties_do_not_switch_or_claim_confidence",
        "test_incumbent_updates_ERC_without_freezing_recommendations",
        "test_initial_reset_and_unscorable_states_are_explicit",
        "test_actual_inputs_survive_a_failing_new_assessment",
        "test_prefix_causality_and_separate_rule_states",
        "test_scalar_and_boolean_oracle_checks_all_pair_evidence",
        "test_oracle_rejects_nonselected_pair_and_gate_mutations",
        "test_cached_source_loader_never_replays_original_pipeline",
        "test_shared_publication_wrappers_do_not_recurse_or_mutate_source",
        "test_failed_publication_cannot_start_main_account",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed state,dominance,scalar,source and publication tests.")
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
        raise QuantError(
            "Validation must bind the actual current implementation and successful outputs."
        )
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
            len(events) != 629
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 60
            or counts["new_completed_strategy_evaluations"] != 52
            or counts["new_technical_failures"] != 8
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual unchanged history and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 59782,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / path) for path in OWN],
                "complete_dependency_inputs": dependencies(),
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_dominance_assessments_states_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "pair_assessment_dominance_rule_frozen",
            active_compute_job=None,
            pair_dominance_intent=reference(INTENT),
            pending_steps=["register_both_rules_before_actual_new_assessments"],
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
        or intent["real_new_dominance_assessments_states_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome dominance implementation freeze changed.")
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
        "source_seven_expert_pool_policy": parent["policy"],
        "expert_pool": reference(rules.POLICY),
        "expert_pool_definition_role": "fixed_seven_new_pair_ID_dominance_only",
        "source_policy_and_expert_admission_clocks": clocks,
        "decision_information_clock": rules.CLOCK,
        "prior_completed_configurations": 230,
        "prior_journal_sequence": 629,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "independent_incumbent_pair_states_strict_two_window_dominance",
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


def _verify_pair_assessments(observed, previous, universe, assessment):
    n = len(observed)
    values = [(item.net_returns - item.risk_free).tolist() for item in observed]
    means = [math.fsum(value[:84]) / 84 for value in values]
    deviations = [
        math.sqrt(math.fsum((x - mean) ** 2 for x in value[:84]) / 83)
        for value, mean in zip(values, means, strict=True)
    ]
    expected_pairs = {
        tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
        for i, j in itertools.combinations(range(n), 2)
    }
    if (
        len(assessment["pairs"]) != 21
        or {tuple(pair["pair"]) for pair in assessment["pairs"]} != expected_pairs
    ):
        raise QuantError("Retain all21pair assessments,including explicit unscorable states.")
    maximum_weight = maximum_score = 0.0
    for pair in assessment["pairs"]:
        i, j = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
        same = observed[i].identity.mechanism_group == observed[j].identity.mechanism_group
        reason = (
            "same_mechanism_group"
            if same
            else ("zero_training_variance" if min(deviations[i], deviations[j]) <= 0 else None)
        )
        weights = [0.0] * n
        if reason is None:
            weights[i] = deviations[j] / (deviations[i] + deviations[j])
            weights[j] = deviations[i] / (deviations[i] + deviations[j])
            portfolio = [math.fsum(weights[k] * values[k][t] for k in range(n)) for t in range(126)]
            if np.std(portfolio[:84], ddof=1) <= 0 or np.std(portfolio[84:], ddof=1) <= 0:
                reason = "nonpositive_or_nonfinite_pair_risk"
        if reason is not None:
            if pair.get("unscorable_reason") != reason:
                raise QuantError("An unscorable pair differs from its independent reason.")
            continue
        actual = np.asarray(pair["expert_weights"], dtype=float)
        contributions = np.asarray(pair["training_marginal_variance_contributions"])
        if (
            "unscorable_reason" in pair
            or actual.shape != (n,)
            or contributions.shape != (n,)
            or not np.isfinite(actual).all()
            or not np.isfinite(contributions).all()
            or (actual < 0).any()
            or abs(float(actual.sum()) - 1) > 1e-12
            or not np.isfinite(
                [
                    pair[key]
                    for key in (
                        "train_net_risk_score",
                        "validation_net_risk_score",
                        "selection_score",
                        "selection_score_key",
                        "training_correlation",
                        "estimated_incremental_switch_cost_fraction",
                    )
                ]
            ).all()
        ):
            raise QuantError("Pair evidence needs complete finite independent risk records.")
        maximum_weight = max(maximum_weight, max(abs(actual - weights)))
        covariance = (
            math.fsum((values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)) / 83
        )
        expected_contribution = np.zeros(n)
        expected_contribution[i] = weights[i] * (
            weights[i] * deviations[i] ** 2 + weights[j] * covariance
        )
        expected_contribution[j] = weights[j] * (
            weights[j] * deviations[j] ** 2 + weights[i] * covariance
        )
        if (
            abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j])) > 1e-10
            or max(abs(contributions - expected_contribution)) > 1e-12
        ):
            raise QuantError(
                "Stored pair correlation or marginal variance differs from known observations."
            )
        change = [
            abs(
                math.fsum(
                    (pair["expert_weights"][k] - previous[k]) * float(observed[k].target[symbol])
                    for k in range(n)
                )
            )
            for symbol in universe
        ]
        cost = math.fsum(change) * 5 / 10000 + sum(value > 1e-12 for value in change) / 10000
        scores = [scalar_score(portfolio[:84], cost), scalar_score(portfolio[84:], cost)]
        maximum_score = max(
            maximum_score,
            abs(scores[0] - pair["train_net_risk_score"]),
            abs(scores[1] - pair["validation_net_risk_score"]),
        )
        if (
            abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
            or pair["estimated_incremental_changed_assets"]
            != sum(value > 1e-12 for value in change)
            or pair["selection_score"]
            != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
            or pair["selection_score_key"] != round(pair["selection_score"], 12)
            or pair["cost_proxy_is_actual_main_account_cost"] is not False
        ):
            raise QuantError("The original switch proxy or pair score key changed.")
    return maximum_weight, maximum_score


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
    maximum_target = maximum_weight = maximum_score = 0.0
    per_rule = {}
    for candidate in rules.CANDIDATES:
        name, gate = candidate["id"], candidate["switch_gate"]
        rows = audits[name]
        if [row["decision_session"] for row in rows] != dates.strftime("%Y-%m-%d").tolist():
            raise QuantError("Each rule needs every original chronological decision.")
        previous, incumbent, switches, holds, unscored = np.full(7, 1 / 7), None, 0, 0, 0
        scored_count = degenerate_count = 0
        for position, (day, audit) in enumerate(zip(dates, rows, strict=True)):
            observed = [expert.observe(day, 126) for expert in experts]
            evidence = audit["switch_evidence"]
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
                or evidence["gate"] != gate
                or evidence["incumbent_pair_before"] != (list(incumbent) if incumbent else None)
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
                raise QuantError("The declared pair state,clock or capital separation changed.")
            assessment = evidence["pair_assessments"]
            if not all(item.score_mature for item in observed):
                if (
                    position >= 7
                    or assessment["reason"] != "unscored_initial_equal_weights"
                    or assessment["scores_available"] is not False
                    or assessment["pairs"] != []
                    or assessment["selected_pair"] is not None
                    or assessment["safety_or_success_claimed"] is not False
                ):
                    raise QuantError(
                        "Seven initial unmatured assessments cannot become successful scores."
                    )
                chosen, weights, reason = None, np.full(7, 1 / 7), "unscored_initial_equal_weights"
                comparisons, eligible = [], []
                unscored += 1
            else:
                weight_gap, score_gap = _verify_pair_assessments(
                    observed, previous, data.close.columns, assessment
                )
                maximum_weight, maximum_score = (
                    max(maximum_weight, weight_gap),
                    max(maximum_score, score_gap),
                )
                known = observed[0].net_returns.index
                if assessment["scores_available"] and (
                    assessment.get("train_start") != str(known[0].date())
                    or assessment.get("train_end") != str(known[83].date())
                    or assessment.get("validation_start") != str(known[84].date())
                    or assessment.get("validation_end") != str(known[-1].date())
                    or known[-1] > day
                ):
                    raise QuantError(
                        "Pair scores must use the unchanged known train/validation periods."
                    )
                scored = {
                    tuple(pair["pair"]): pair
                    for pair in assessment["pairs"]
                    if "unscorable_reason" not in pair
                }
                comparisons, eligible = [], []
                if not scored:
                    if (
                        assessment["scores_available"] is not False
                        or assessment["reason"] != "unscored_degenerate_pair_equal_weights"
                        or assessment["selected_pair"] is not None
                    ):
                        raise QuantError("An unranked degenerate assessment cannot be hidden.")
                    chosen, weights, reason = (
                        None,
                        np.full(7, 1 / 7),
                        "unscored_degenerate_pair_equal_weights",
                    )
                    degenerate_count += 1
                else:
                    scored_count += 1
                    best = sorted(
                        scored, key=lambda names: (-scored[names]["selection_score_key"], names)
                    )[0]
                    if (
                        assessment["scores_available"] is not True
                        or tuple(assessment["selected_pair"]) != best
                    ):
                        raise QuantError(
                            "Retain the original instantaneous lower-key ranking as evidence."
                        )
                    if incumbent is None or incumbent not in scored:
                        chosen = best
                        reason = (
                            "no_incumbent_select_original_best"
                            if incumbent is None
                            else "incumbent_unscorable_select_original_best"
                        )
                    else:
                        incumbent_record = scored[incumbent]
                        incumbent_train = round(incumbent_record["train_net_risk_score"], 12)
                        incumbent_validation = round(
                            incumbent_record["validation_net_risk_score"], 12
                        )
                        incumbent_upper = round(
                            max(
                                incumbent_record["train_net_risk_score"],
                                incumbent_record["validation_net_risk_score"],
                            ),
                            12,
                        )
                        for pair in assessment["pairs"]:
                            names = tuple(pair["pair"])
                            if "unscorable_reason" in pair:
                                comparisons.append(
                                    {
                                        "pair": list(names),
                                        "unscorable_reason": pair["unscorable_reason"],
                                        "qualifies": None,
                                    }
                                )
                                continue
                            train_key = round(pair["train_net_risk_score"], 12)
                            validation_key = round(pair["validation_net_risk_score"], 12)
                            qualifies = (
                                train_key > incumbent_train
                                and validation_key > incumbent_validation
                                if gate == rules.GATES[0]
                                else pair["selection_score_key"] > incumbent_upper
                            )
                            comparisons.append(
                                {
                                    "pair": list(names),
                                    "challenger_train_key": train_key,
                                    "challenger_validation_key": validation_key,
                                    "challenger_lower_key": pair["selection_score_key"],
                                    "incumbent_train_key": incumbent_train,
                                    "incumbent_validation_key": incumbent_validation,
                                    "incumbent_upper_key": incumbent_upper,
                                    "qualifies": qualifies,
                                }
                            )
                            if qualifies:
                                eligible.append(names)
                        eligible.sort(
                            key=lambda names: (-scored[names]["selection_score_key"], names)
                        )
                        chosen = eligible[0] if eligible else incumbent
                        reason = (
                            "dominant_challenger_selected"
                            if eligible
                            else "no_dominance_retain_incumbent"
                        )
                    weights = np.asarray(scored[chosen]["expert_weights"])
            if (
                evidence["reason"] != reason
                or evidence["dominance_comparisons"] != comparisons
                or evidence["qualified_challengers"] != [list(pair) for pair in eligible]
                or evidence["incumbent_pair_after"] != (list(chosen) if chosen else None)
                or not np.allclose(weights, audit["expert_weights"], rtol=0, atol=1e-12)
                or not np.allclose(weights, evidence["chosen_expert_weights"], rtol=0, atol=1e-12)
            ):
                raise QuantError(
                    "Strict gates,pair states or resolved weights disagree with the oracle."
                )
            expected_target = [
                math.fsum(
                    float(item.target[symbol]) * float(weights[index])
                    for index, item in enumerate(observed)
                )
                for symbol in data.close.columns
            ]
            maximum_target = max(
                maximum_target, float(max(abs(targets[name].loc[day] - expected_target)))
            )
            switches += incumbent is not None and chosen is not None and chosen != incumbent
            holds += reason == "no_dominance_retain_incumbent"
            previous, incumbent = weights, chosen
        if unscored != min(7, len(rows)):
            raise QuantError("Measure and preserve every original unscored initial decision.")
        per_rule[name] = {
            "decisions_checked": len(rows),
            "unscored_initial_decisions": unscored,
            "incumbent_pair_changes": switches,
            "no_dominance_incumbent_retained": holds,
            "mature_scored_decisions": scored_count,
            "mature_degenerate_equal_fallbacks": degenerate_count,
        }
    if maximum_target > 1e-10 or maximum_weight > 1e-10 or maximum_score > 1e-8:
        raise QuantError("Original scalar target,ERC weights or net-score limits failed.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
        "independent_strict_dominance_gate_and_pair_state_verified": True,
        "per_rule": per_rule,
    }


def verify_retained_records(path, experts, targets, audits):
    count = values = 0
    with path.open() as stream:
        for candidate in rules.CANDIDATES:
            name = candidate["id"]
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
                    != [float(value).hex() for value in audit["previous_expert_weights"]]
                    or before["incumbent_pair"] != audit["switch_evidence"]["incumbent_pair_before"]
                    or before["binary64_values_truncated_to_fixed_decimals"] is not False
                    or after["audit"] != audit
                    or after["asset_order"] != list(targets[name].columns)
                    or after["underlying_target"] != targets[name].loc[day].tolist()
                ):
                    raise QuantError(
                        "Actual incumbent/assessment input or resolved records changed."
                    )
                expected = [
                    rules.observation_record(expert.observe(day, 126)) for expert in experts
                ]
                if before["observations"] != expected:
                    raise QuantError(
                        "Retained binary64 inputs differ from actual causal source observations."
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
        raise QuantError("Keep all121original decisions for both independent pair-state rules.")
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
            raise QuantError("Future source inputs changed an earlier incumbent-pair decision.")
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
                raise QuantError("Every formal anchor and target factor coverage is required.")
            coverage.append({"years": window["years"], "actual_factor_symbols": actual})
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "decision_count": len(audits[name]),
                "unscored_initial_equal_weight_decisions": independent["per_rule"][name][
                    "unscored_initial_decisions"
                ],
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
                "empirical_gate_is_confidence_interval_or_significance_test": False,
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
        events = base.journal_events(base.JOURNAL)
        old = [event for event in events if event["candidate_id"] not in IDS]
        if len(old) != 629 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve all original completed and failed financial identities.")
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
                    "preparing_registered_pair_dominance_rules",
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
            failure("pair_dominance_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress("pair_dominance_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "pair_dominance_registered_waiting_publication",
            active_compute_job=None,
            pair_dominance_registration=reference(PACKET),
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
                raise QuantError("Preserve actual completed outcomes without replay.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_pair_dominance_rules",
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
                            "evaluating_pair_dominance_rules",
                            active_candidate=name,
                            active_path=key,
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
                "pair_dominance_financial_outcomes_complete",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                pair_dominance_results=reference(RESULTS),
                pending_steps=["publish_all_eight_actual_outcomes"],
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("pair_dominance_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "pair_dominance_evaluation_interrupted",
                active_compute_job=None,
                active_candidate=None,
            )
            raise
    return reference(RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed empirical-dominance pair research;no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Pair-dominance research cannot request live sources.")

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
        parser.exit(2, f"Pair-dominance research blocked:{error}\n")


if __name__ == "__main__":
    main()
