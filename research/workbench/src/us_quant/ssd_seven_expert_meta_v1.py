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
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant import ssd_vvix_expert_admission_v1 as admitted
from us_quant import strategy_expert_pool_v5 as rules
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
FOLDER = base.REPORTS / "seven-expert-meta-20261031-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-seven-expert-meta-20261031-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/seven_expert_meta_20261031_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/seven_expert_meta_20261031_results_v1.json"
IDS = tuple(row["id"] for row in rules.CANDIDATES)
OWN = (
    "config/strategy-expert-pool-seven-v1.json",
    "src/us_quant/strategy_expert_pool_v5.py",
    "src/us_quant/ssd_seven_expert_meta_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_ssd_seven_expert_meta_v1.py",
)
EXTRA_TEST_INPUTS = (
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_ssd_six_expert_meta_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
)
DESIGN_SHA = "d6f4ae5dd115d09d24a270a2de4393f92e2436ec8479f297d68a594831988326"
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame
verify_publication = source.verify_publication


def _call(template, *args):
    return FunctionType(
        template.__code__, {**template.__globals__, **globals()}, argdefs=template.__defaults__
    )(*args)


def dependencies():
    return FunctionType(
        admitted.import_template.dependencies.__code__,
        {**globals(), "ast": ast, "prior": source},
    )()


def source_inputs():
    design, policy = read_json(DESIGN), read_json(rules.POLICY)
    rules.validate(policy)
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if (
        reference(DESIGN)["sha256"] != DESIGN_SHA
        or digest_json(parent) != PARENT_SPEC
        or policy["source_specification_sha256"] != PARENT_SPEC
        or parent["policy"] != policy["source_scoring_policy"]
        or design["actual_continuation_event"] != 58339
        or design["invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["actual_new_seven_pool_scores_learning_targets_or_accounts_computed"] is not False
        or [row["id"] for row in design["selected_members"]] != list(rules.EXPERT_IDS)
        or design["required_decision_information_clock"] != rules.CLOCK
    ):
        raise QuantError("Preserve the actual pre-outcome membership design and original source.")
    check_frozen(parent)
    source.checked_packet(parent)
    old_results = read_json(base.checked_reference(policy["previous_six_results"]))
    if (
        old_results["frozen_specification_sha256"] != PARENT_SPEC
        or set(old_results["candidates"]) != set(source.IDS)
        or any(
            row["status"] not in {"rejected", "historical_point_gates_passed"}
            for row in old_results["candidates"].values()
        )
    ):
        raise QuantError("The completed six-expert controls cannot be replaced or reclassified.")
    data, experts, old_policy, _, old_refs, clocks, old_admission_refs, benchmarks = (
        source.source_inputs()
    )
    if tuple(e.identity.identifier for e in experts) != rules.EXPERT_IDS[:6] or any(
        policy[key] != value or old_policy[key] != value
        for key, value in design["unchanged_law"].items()
    ):
        raise QuantError("One pool inclusion cannot retune assessment or execution.")
    option, admission = admitted.load_admitted(
        tuple(data.close.columns),
        data.risk_free,
        decision_information_clock=policy["decision_information_clock"],
    )
    public = read_json(base.checked_reference(policy["option_public_admission"]))
    if (
        policy["option_expert_admission"] != reference(admitted.OUTPUT)
        or public["private_admission"] != policy["option_expert_admission"]
        or public["admitted_at"] != admission["admitted_at"]
        or option.identity.identifier != rules.EXPERT_IDS[6]
        or option.identity.mechanism_group != rules.MECHANISMS[6]
        or admission["new_recommendation_or_assessment_view_artifacts"] != 0
        or admission["expert"]["technical"]["four_completed_paths_checked"] != 4
        or admission["expert"]["technical"]["standalone_financial_verdict_preserved"] != "rejected"
    ):
        raise QuantError("Reuse actual clock-bound VVIX admission without replacing its source.")
    experts = (*experts, option)
    for expert in experts:
        initial = expert.observe(pd.Timestamp(policy["initial_decision_session"]), 126)
        final = expert.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(final.net_returns) != 2512
            or not final.risk_free.equals(data.risk_free.loc[final.risk_free.index])
        ):
            raise QuantError(
                "Keep genuine availability,empty initial history and original common RF."
            )
    option_intent = read_json(base.checked_reference(admission["intent"]))
    new_refs = unique_refs(
        [
            *option_intent["source_references"],
            *option_intent["validation_evidence"],
            *option_intent["implementation_and_test_inputs"],
            admission["intent"],
            policy["option_expert_admission"],
            policy["option_public_admission"],
        ]
    )
    refs = unique_refs(
        [
            *old_refs,
            *old_admission_refs,
            *new_refs,
            *design["source_references"],
            reference(source.PACKET),
            reference(source.INTENT),
            reference(source.PUBLIC_RESULTS),
            policy["source_registration"],
            policy["source_scoring_policy"],
            policy["previous_six_results"],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    clocks = [
        *clocks,
        {
            "id": option.identity.identifier,
            "source_policy_registered_at": admission["source_original_registered_at"],
            "actual_expert_admitted_at": admission["admitted_at"],
            "decision_information_clock": rules.CLOCK,
        },
    ]
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return (
        data,
        experts,
        policy,
        parent,
        refs,
        clocks,
        unique_refs([*old_admission_refs, *new_refs]),
        benchmarks,
    )


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_seven_inclusion_reuses_exact_unchanged_pair_law",
        "test_all_twenty_one_pairs_and_one_main_target",
        "test_initial_maturity_and_prefix_causality",
        "test_actual_inputs_survive_a_failing_pair_decision",
        "test_scalar_oracle_checks_all_twenty_one_pairs",
        "test_scalar_oracle_rejects_nonselected_pair_risk_or_cost_changes",
        "test_cached_loader_keeps_explicit_clock_without_old_pipeline",
        "test_full_import_capture_does_not_mutate_source_globals",
        "test_failed_publication_cannot_start_any_main_account",
        "test_canonical_export_preserves_all_paths_without_daily_rows",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed causal,21-pair scalar,cache,clock and publication tests."
        )
    inputs = [reference(ROOT / path) for path in (*OWN, *EXTRA_TEST_INPUTS)]
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != inputs
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError(
            "Validation must bind current implementation,fixtures and successful outputs."
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
            len(events) != 606
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 58
            or counts["new_completed_strategy_evaluations"] != 50
            or counts["new_technical_failures"] != 8
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError(
                "Freeze against the actual unchanged source history and scheduled origin."
            )
        deps = dependencies()
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 58339,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / path) for path in OWN],
                "complete_dependency_inputs": deps,
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_meta_scores_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "seven_expert_rule_frozen",
            active_compute_job=None,
            seven_expert_intent=reference(INTENT),
            pending_steps=["register_both_seven_expert_rules_before_actual_scores_targets"],
        )
    return reference(INTENT)


def inputs():
    data, experts, _, parent, refs, clocks, admission_refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["complete_dependency_inputs"] != dependencies()
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_scores_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome membership or implementation freeze changed.")
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
        "source_six_expert_pool_policy": parent["policy"],
        "expert_pool": reference(rules.POLICY),
        "expert_pool_definition_role": "controlled_seven_representatives_and_cached_admissions",
        "source_policy_and_expert_admission_clocks": clocks,
        "decision_information_clock": rules.CLOCK,
        "prior_completed_configurations": 228,
        "prior_journal_sequence": 606,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "seven_complete_experts_including_option_uncertainty_same_pair_law",
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
    n = len(rules.EXPERT_IDS)
    equal = np.full(n, 1 / n)
    if (
        set(targets) != set(IDS)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or any(
            not target.index.equals(data.close.index)
            or not target.columns.equals(data.close.columns)
            or not target.dropna().index.equals(dates)
            or not target.loc[~target.index.isin(dates)].isna().all().all()
            for target in targets.values()
        )
    ):
        raise QuantError("Keep the full initial/monthly target lattice and formal calendar.")
    previous_weights = equal.copy()
    maximum_target = maximum_weight = maximum_score = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["decision_information_clock"] != rules.CLOCK
            or audit["previous_rolling_expert_weights"] != previous_weights.tolist()
            or audit["equal_expert_weights"] != equal.tolist()
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or any(
                o.target is None or o.target_known_at is None or o.target_known_at > day
                for o in observed
            )
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The source clock,expert vector or capital separation changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if (
                selection["reason"] != "unscored_initial_equal_weights"
                or selection["scores_available"] is not False
                or selection["pairs"] != []
                or selection["selected_pair"] is not None
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("Unmatured observations cannot receive successful scores.")
            chosen = equal.copy()
        else:
            values = [(o.net_returns - o.risk_free).tolist() for o in observed]
            means = [math.fsum(v[:84]) / 84 for v in values]
            deviations = [
                math.sqrt(math.fsum((x - mean) ** 2 for x in value[:84]) / 83)
                for value, mean in zip(values, means, strict=True)
            ]
            expected_pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i, j in itertools.combinations(range(n), 2)
            }
            if (
                len(selection["pairs"]) != 21
                or {tuple(p["pair"]) for p in selection["pairs"]} != expected_pairs
            ):
                raise QuantError(
                    "Retain all twenty-one pairs,including explicit unscorable states."
                )
            for pair in selection["pairs"]:
                i, j = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
                same = observed[i].identity.mechanism_group == observed[j].identity.mechanism_group
                reason = (
                    "same_mechanism_group"
                    if same
                    else (
                        "zero_training_variance" if min(deviations[i], deviations[j]) <= 0 else None
                    )
                )
                weights = [0.0] * n
                if reason is None:
                    weights[i] = deviations[j] / (deviations[i] + deviations[j])
                    weights[j] = deviations[i] / (deviations[i] + deviations[j])
                    portfolio = [
                        math.fsum(weights[k] * values[k][t] for k in range(n)) for t in range(126)
                    ]
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
                            pair[k]
                            for k in (
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
                    raise QuantError("Every pair needs finite,correctly dimensioned risk records.")
                maximum_weight = max(maximum_weight, max(abs(actual - weights)))
                covariance = (
                    math.fsum(
                        (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                expected_contribution = np.zeros(n)
                expected_contribution[i] = weights[i] * (
                    weights[i] * deviations[i] ** 2 + weights[j] * covariance
                )
                expected_contribution[j] = weights[j] * (
                    weights[j] * deviations[j] ** 2 + weights[i] * covariance
                )
                if (
                    abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j]))
                    > 1e-10
                    or max(abs(contributions - expected_contribution)) > 1e-12
                ):
                    raise QuantError("Stored correlation or marginal risk differs from known data.")
                change = [
                    abs(
                        math.fsum(
                            (pair["expert_weights"][k] - previous_weights[k])
                            * float(observed[k].target[symbol])
                            for k in range(n)
                        )
                    )
                    for symbol in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                scores = [scalar_score(portfolio[:84], cost), scalar_score(portfolio[84:], cost)]
                maximum_score = max(
                    maximum_score,
                    abs(scores[0] - pair["train_net_risk_score"]),
                    abs(scores[1] - pair["validation_net_risk_score"]),
                )
                if (
                    abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or pair["estimated_incremental_changed_assets"]
                    != sum(v > 1e-12 for v in change)
                    or pair["selection_score"]
                    != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
                    or pair["selection_score_key"] != round(pair["selection_score"], 12)
                    or pair["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError("The original switch proxy or score key changed.")
            scored = [p for p in selection["pairs"] if "selection_score_key" in p]
            if scored:
                best = sorted(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                known = observed[0].net_returns.index
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["scores_available"] is not True
                    or selection["train_start"] != str(known[0].date())
                    or selection["train_end"] != str(known[83].date())
                    or selection["validation_start"] != str(known[84].date())
                    or selection["validation_end"] != str(known[-1].date())
                    or known[-1] > day
                ):
                    raise QuantError("The chronological score or chosen pair changed.")
                chosen = np.asarray(best["expert_weights"])
            else:
                if (
                    selection["reason"] != "unscored_degenerate_pair_equal_weights"
                    or selection["scores_available"] is not False
                    or selection["selected_pair"] is not None
                ):
                    raise QuantError("A degenerate unranked state cannot be hidden.")
                chosen = equal.copy()
        if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("Issued expert weights differ from their actual selection.")
        for name, weights in ((IDS[0], equal), (IDS[1], chosen)):
            expected = [
                math.fsum(float(o.target[symbol]) * weights[i] for i, o in enumerate(observed))
                for symbol in data.close.columns
            ]
            maximum_target = max(maximum_target, float(max(abs(targets[name].loc[day] - expected))))
        previous_weights = chosen
    if maximum_target > 1e-10 or maximum_weight > 1e-10 or maximum_score > 1e-8:
        raise QuantError("Independent scalar composition,risk weights or net scores disagree.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
    }


def verify_retained_records(path, experts, targets, audits):
    observed_values = 0
    with path.open() as stream:
        for position, audit in enumerate(audits):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            day = pd.Timestamp(audit["decision_session"])
            previous = audit["previous_rolling_expert_weights"]
            if (
                before["record"] != "decision_input"
                or after["record"] != "decision_resolved"
                or before["decision_index"] != position
                or after["decision_index"] != position
                or before["as_of"] != audit["decision_session"]
                or before["information_clock"] != rules.CLOCK
                or before["previous_weights"] != previous
                or before["previous_weights_binary64_hex"]
                != [float(value).hex() for value in previous]
                or before["binary64_values_truncated_to_fixed_decimals"] is not False
                or len(before["observations"]) != len(experts)
                or after["audit"] != audit
                or after["asset_order"] != list(targets[IDS[0]].columns)
                or after["combined_targets"]
                != {name: matrix.loc[day].tolist() for name, matrix in targets.items()}
            ):
                raise QuantError("Actual pre-update input or resolved decision records changed.")
            for record, expert in zip(before["observations"], experts, strict=True):
                item = expert.observe(day, 126)
                expected = {
                    "id": item.identity.identifier,
                    "mechanism_group": item.identity.mechanism_group,
                    "introduced_at": item.identity.introduced_at,
                    "as_of": str(item.as_of.date()),
                    "target_known_at": str(item.target_known_at.date())
                    if item.target_known_at is not None
                    else None,
                    "target": item.target.tolist() if item.target is not None else None,
                    "asset_order": list(item.target.index) if item.target is not None else None,
                    "return_dates": item.net_returns.index.strftime("%Y-%m-%d").tolist(),
                    "NET_returns": item.net_returns.tolist(),
                    "RF": item.risk_free.tolist(),
                    "score_mature": item.score_mature,
                }
                if record != expected:
                    raise QuantError(
                        "Retained binary64 inputs differ from actual causal source observations."
                    )
                observed_values += len(item.net_returns) + len(item.risk_free)
        if stream.readline():
            raise QuantError(
                "Retain exactly one input and one resolved record per actual decision."
            )
    return {
        "actual_input_and_resolved_record_count": 2 * len(audits),
        "roundtrip_NET_RF_binary64_values_checked": observed_values,
        "actual_input_resolved_records_match_source_and_targets": True,
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
    if len(audits) != 121:
        raise QuantError("Keep all121original initial/monthly decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future inputs changed a prior expert composition.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
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
                for symbol in source.previous.previous.source.donor.rules.CORE
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
                "decision_count": len(audits),
                "initial_capital_decision_is_not_a_fabricated_month_end": True,
                "unscored_initial_equal_weight_decisions": sum(
                    a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits
                ),
                "scored_decisions": sum(a["selection"]["scores_available"] for a in audits),
                "information_audits": reference(FOLDER / "information-audits-v1.json"),
                "actual_input_resolved_records": reference(log),
                "inputs_saved_before_updates_and_proof": True,
                **independent,
                **retained,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "decision_information_clock": rules.CLOCK,
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
    return _call(source.checked_packet, frozen)


def publication(frozen, commit):
    return _call(source.publication, frozen, commit)


def states(frozen):
    return _call(source.states, frozen)


def public_result():
    return _call(source.public_result)


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        old = [event for event in events if event["candidate_id"] not in IDS]
        if len(old) != 606 or old[-1]["sha256"] != rules.PRIOR_HEAD:
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
                    "preparing_registered_seven_expert_meta",
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
            failure("seven_expert_meta_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress("seven_expert_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "seven_expert_registered_waiting_publication",
            active_compute_job=None,
            seven_expert_registration=reference(PACKET),
            pending_steps=["publish_exact_seven_expert_packet_before_eight_accounts"],
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
                "evaluating_seven_expert_meta",
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
                            "evaluating_seven_expert_meta", active_candidate=name, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                participation = source.previous.previous.actual_factor_participation(paths, frozen)
                passed = all(path["all_point_gates_pass"] for path in paths.values()) and all(
                    row["minimum_three_registered_families_pass"] for row in participation.values()
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
                "seven_expert_meta_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                seven_expert_results=reference(RESULTS),
                pending_steps=["publish_all_eight_seven_expert_outcomes"],
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("seven_expert_meta_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "seven_expert_meta_interrupted", active_compute_job=None, active_candidate=None
            )
            raise
    return reference(RESULTS)


def main():
    parser = argparse.ArgumentParser(description="Controlled seven-expert research;no trading.")
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Seven-expert research cannot request live sources.")

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
        parser.exit(2, f"Seven-expert research blocked:{error}\n")


if __name__ == "__main__":
    main()
