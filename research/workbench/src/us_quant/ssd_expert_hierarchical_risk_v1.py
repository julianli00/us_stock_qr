from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_hierarchical_risk_v1 as rules
from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_expert_tail_risk_budget_v1 as source
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_weekly import verify_publication
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
FOLDER = base.REPORTS / "expert-hierarchical-risk-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-hierarchical-risk-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_hierarchical_risk_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_hierarchical_risk_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-hierarchical-risk-v1.json",
    "src/us_quant/expert_hierarchical_risk_v1.py",
    "src/us_quant/ssd_expert_hierarchical_risk_v1.py",
    "tests/test_expert_hierarchical_risk_v1.py",
    "tests/test_ssd_expert_hierarchical_risk_v1.py",
)
SHARED_CODE = "src/us_quant/expert_full_covariance_risk_budget_v1.py"
SHARED_POLICY = "config/expert-full-covariance-risk-budget-v1.json"
PARENT_SPEC = "13ddfc88078f0b9040725571815d899b4951f87136737253241ebe4e01313b56"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame


def dependency_inputs():
    manifest = read_json(
        base.checked_reference({"path": "IMPORT_MANIFEST.json", "sha256": base.BUNDLE_SHA256})
    )
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    prefix = ROOT.relative_to(base.REPO).as_posix() + "/"
    pending = [*OWN[1:], SHARED_CODE, "src/us_quant/__init__.py"]
    required = set()
    while pending:
        path = pending.pop()
        if path in required:
            continue
        required.add(path)
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (
                    [f"us_quant.{a.name}" for a in node.names]
                    if node.module == "us_quant"
                    else [node.module]
                )
            for name in names:
                if name.startswith("us_quant."):
                    local = "src/" + name.replace(".", "/") + ".py"
                    if (ROOT / local).is_file() and local not in required:
                        pending.append(local)
    configurations = {
        path[len(prefix) :]
        for path in contract["files"]
        if path.startswith(prefix + "config/") and path.endswith(".json")
    } | {
        path
        for path in manifest["file_sha256"]
        if path.startswith("config/") and path.endswith(".json")
    }
    configurations.discard(OWN[0])
    code = [reference(ROOT / path) for path in sorted(required - set(OWN))]
    policies = [reference(ROOT / path) for path in sorted(configurations)]
    for ref in [*code, *policies]:
        pinned = manifest["file_sha256"].get(ref["path"])
        if pinned is not None and ref["sha256"] != pinned:
            raise QuantError("An original imported dependency cannot change during new research.")
    return code, policies


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("Keep the exact preserved tail-study dependency specification.")
    check_frozen(parent)
    source.checked_packet(parent)
    saved = read_json(base.checked_reference(policy["source_results"]))
    if (
        saved["registration_packet"] != policy["source_registration"]
        or saved["reviews"][0]["status"] != "rejected"
        or len(saved["reviews"][0]["paths"]) != 4
    ):
        raise QuantError("Prior complete financial rejections cannot be relabelled or replayed.")
    data, experts, old_policy, _, old_refs, clocks, _, benchmarks = source.source_inputs()
    if any(
        policy[k] != old_policy[k]
        for k in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
        )
    ):
        raise QuantError("Change the allocator, not its valid experts or funded execution.")
    old_intent = read_json(source.INTENT)
    dependency_code, dependency_policies = dependency_inputs()
    refs = unique_refs(
        [
            *old_refs,
            *old_intent["source_references"],
            *old_intent["model_and_test_files"],
            *old_intent["validation_evidence"],
            reference(source.INTENT),
            *read_json(DESIGN)["source_references"],
            policy["source_registration"],
            policy["source_results"],
            policy["saved_cost_summary"],
            reference(ROOT / SHARED_CODE),
            reference(ROOT / SHARED_POLICY),
            *dependency_code,
            *dependency_policies,
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
        "test_uncorrelated_hierarchy_is_inverse_variance_not_global_erc",
        "test_singular_covariance_and_deterministic_ties_need_no_solver",
        "test_zero_cluster_risk_is_explicit_not_a_safety_claim",
        "test_missing_history_and_zero_individual_risk_are_unscored",
        "test_future_source_changes_preserve_prior_allocations",
        "test_scalar_hierarchy_verifies_every_merge_and_capital_split",
        "test_scalar_verifier_rejects_clock_tree_weight_and_risk_tampering",
        "test_cached_loader_never_invokes_closed_entrypoints",
        "test_complete_dependencies_precede_the_new_publication_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_all_original_paths_without_private_rows",
        "test_original_monthly_execution_is_preserved",
        "test_complete_local_import_closure_is_bound",
    }
    if not required.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed hierarchy, clock, source and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current hierarchy and driver bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve any existing complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            len(events) != 516
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 47
            or counts["new_completed_strategy_evaluations"] != 43
            or counts["new_technical_failures"] != 4
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 47275
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual post-diagnosis journal and invocation.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 47275,
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
            "expert_hierarchical_risk_frozen",
            active_compute_job=None,
            expert_hierarchical_risk_intent=reference(INTENT),
            pending_steps=["register_new_hierarchy_before_real_weights_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_allocations_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome hierarchy freeze changed.")
    dependency_code, dependency_policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *[reference(ROOT / p) for p in OWN[1:]],
                reference(ROOT / SHARED_CODE),
                *dependency_code,
            ]
        ),
        "policies": unique_refs(
            [
                *parent["policies"],
                reference(rules.POLICY),
                reference(ROOT / SHARED_POLICY),
                *dependency_policies,
            ]
        ),
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_hierarchical_risk",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 221,
        "prior_journal_sequence": 516,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_canonical_single_linkage_recursive_risk",
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
    return data, experts, policy, frozen, benchmarks


def scalar_covariance(columns):
    centered = [[v - values[0] for v in values] for values in columns]
    means = [math.fsum(v) / len(v) for v in centered]
    return np.array(
        [
            [
                math.fsum(
                    (a - means[i]) * (b - means[j])
                    for a, b in zip(centered[i], centered[j], strict=True)
                )
                / (len(centered[i]) - 1)
                for j in range(6)
            ]
            for i in range(6)
        ]
    )


def close_array(actual, expected, message, tolerance=1e-12):
    value, target = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        value.shape != target.shape
        or not np.isfinite(value).all()
        or not np.isfinite(target).all()
        or float(abs(value - target).max(initial=0)) > tolerance
    ):
        raise QuantError(message)


def independent_hierarchy(columns, covariance, allocation):
    distance = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            correlation = float(covariance[i, j]) / (
                math.sqrt(float(covariance[i, i])) * math.sqrt(float(covariance[j, j]))
            )
            if not math.isfinite(correlation) or abs(correlation) > 1 + 1e-12:
                raise QuantError("The independently checked correlations are invalid.")
            distance[i, j] = distance[j, i] = math.sqrt((1 - min(1.0, max(-1.0, correlation))) / 2)
    close_array(allocation["risk_distance"], distance, "The recorded risk distance changed.")
    active = {(i,): [i] for i in range(6)}
    merges = allocation["canonical_merges"]
    if len(merges) != 5:
        raise QuantError("A six-expert single-linkage hierarchy requires exactly five merges.")
    for recorded in merges:
        candidates = []
        groups = sorted(active)
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                edge = min(float(distance[x, y]) for x in a for y in b)
                candidates.append((round(edge, 12), a, b))
        key, left, right = sorted(candidates)[0]
        combined = tuple(sorted(left + right))
        leaves = active.pop(left) + active.pop(right)
        expected = {
            "left_members": list(left),
            "right_members": list(right),
            "distance_key": key,
            "merged_members": list(combined),
            "leaf_order": leaves,
        }
        if recorded != expected:
            raise QuantError("The recorded minimum-distance merge or canonical tie changed.")
        active[combined] = leaves
    order = next(iter(active.values()))
    if allocation["seriated_expert_coordinates"] != order:
        raise QuantError("The canonical expert seriation changed.")
    expected_weights = np.zeros(6)
    expected_splits = []

    def variance(members):
        inverse = [1 / float(covariance[i, i]) for i in members]
        total = math.fsum(inverse)
        inverse = [v / total for v in inverse]
        values = [
            math.fsum(inverse[k] * (columns[i][row] - columns[i][0]) for k, i in enumerate(members))
            for row in range(len(columns[0]))
        ]
        mean = math.fsum(values) / len(values)
        return math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)

    queue = [(order, 1.0)]
    while queue:
        members, capital = queue.pop()
        if len(members) == 1:
            expected_weights[members[0]] = capital
            continue
        left, right = members[: len(members) // 2], members[len(members) // 2 :]
        a, b = variance(left), variance(right)
        fraction = 0.5 if a == b == 0 else b / (a + b)
        expected_splits.append(
            {
                "leaf_order": members,
                "left": left,
                "right": right,
                "parent_capital_fraction": capital,
                "left_IVP_variance": a,
                "right_IVP_variance": b,
                "left_capital_fraction_of_parent": fraction,
                "both_child_risks_zero": a == b == 0,
                "exactly_one_child_risk_zero": (a == 0) != (b == 0),
            }
        )
        queue.extend(((right, capital * (1 - fraction)), (left, capital * fraction)))
    actual_splits = allocation["recursive_capital_splits"]
    if len(actual_splits) != 5:
        raise QuantError("Every recursive capital split must be retained.")
    numerical = (
        "parent_capital_fraction",
        "left_IVP_variance",
        "right_IVP_variance",
        "left_capital_fraction_of_parent",
    )
    for actual, expected in zip(actual_splits, expected_splits, strict=True):
        if {k: v for k, v in actual.items() if k not in numerical} != {
            k: v for k, v in expected.items() if k not in numerical
        }:
            raise QuantError("A zero-risk case or recursive cluster membership changed.")
        close_array(
            [actual[k] for k in numerical],
            [expected[k] for k in numerical],
            "The independent child risk or capital split disagrees.",
        )
    return expected_weights


def verify_targets(data, experts, targets, audits):
    policy = read_json(rules.POLICY)
    dates = rules.decision_sessions(data.close.index, policy)
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
    ):
        raise QuantError("The complete initial/monthly target calendar changed.")
    maximum_weight = maximum_target = maximum_covariance = 0.0
    previous = np.full(6, 1 / 6)
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        expected_clocks = [str(o.target_known_at.date()) for o in observed]
        if (
            audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["expert_target_known_at"] != expected_clocks
            or any(o.target_known_at > day or o.as_of != day for o in observed)
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["source_shadow_wealth_or_protection_floor_transferred"] is not False
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The independent expert clocks or source-state separation changed.")
        close_array(audit["previous_expert_weights"], previous, "The prior capital vector changed.")
        actual = np.asarray(audit["expert_weights"], dtype=float)
        if (
            actual.shape != (6,)
            or not np.isfinite(actual).all()
            or (actual < 0).any()
            or abs(float(actual.sum()) - 1) > 1e-12
        ):
            raise QuantError("The actual expert allocation is not a funded unit-sum vector.")
        row = audit["allocation"]
        if row["safety_or_success_claimed"] is not False:
            raise QuantError("Risk observations cannot acquire a safety guarantee.")
        chosen = np.full(6, 1 / 6)
        if not all(o.score_mature for o in observed):
            if (
                row["reason"] != "unscored_initial_equal_weights"
                or row["risk_estimate_available"] is not False
                or row["available_shadow_observations"] != [len(o.net_returns) for o in observed]
            ):
                raise QuantError("Unavailable history cannot become a successful risk assessment.")
        else:
            index, rf = observed[0].net_returns.index, observed[0].risk_free
            if any(
                not o.net_returns.index.equals(index)
                or not o.risk_free.equals(rf)
                or len(o.net_returns) != 126
                or o.net_returns.index[-1] > day
                for o in observed
            ):
                raise QuantError("The scalar risk fit needs all common matured source dates.")
            columns = [
                [float(v) - float(r) for v, r in zip(o.net_returns, rf, strict=True)]
                for o in observed
            ]
            scalar = scalar_covariance(columns)
            recorded = np.asarray(row["covariance"], dtype=float)
            close_array(recorded, scalar, "The independent covariance disagrees.")
            maximum_covariance = max(maximum_covariance, float(abs(recorded - scalar).max()))
            if row["risk_sample_start"] != str(index[0].date()) or row["risk_sample_end"] != str(
                index[-1].date()
            ):
                raise QuantError("The mature risk sample clock changed.")
            if any(all(v == values[0] for v in values) for values in columns):
                if (
                    row["reason"] != "unscored_nonpositive_individual_variance_equal_weights"
                    or row["risk_estimate_available"] is not False
                ):
                    raise QuantError("Undefined correlation cannot become risk-fit success.")
            else:
                if (
                    row["reason"] != "full_126_NET_RF_hierarchical_risk_allocation"
                    or row["risk_estimate_available"] is not True
                    or row["observations_used_for_fit"] != 126
                    or row["risk_fit_is_heldout_validation"] is not False
                    or row["return_score_or_rank_used"] is not False
                    or row["variance_floor_or_regularization_applied"] is not False
                    or row["global_optimizer_or_previous_failed_solver_used"] is not False
                    or row["risk_clusters_are_independent_factor_families"] is not False
                    or row["nominal_target_change_is_actual_turnover"] is not False
                    or row["cost_proxy_used_for_allocation"] is not False
                    or (np.diag(recorded) <= 0).any()
                ):
                    raise QuantError("Verify the actual hierarchy, not another risk-budget solver.")
                chosen = independent_hierarchy(columns, recorded, row)
                mixed = [math.fsum(chosen[i] * columns[i][k] for i in range(6)) for k in range(126)]
                mean = math.fsum(mixed) / 126
                volatility = math.sqrt(252 * math.fsum((v - mean) ** 2 for v in mixed) / 125)
                close_array(
                    [row["forecast_annualized_RF_excess_volatility"]],
                    [volatility],
                    "The independent combined risk disagrees.",
                )
                change = math.fsum(
                    abs(
                        math.fsum(
                            (actual[i] - previous[i]) * float(observed[i].target[s])
                            for i in range(6)
                        )
                    )
                    for s in data.close.columns
                )
                close_array(
                    [row["nominal_target_change_L1"]],
                    [change],
                    "Nominal target movement changed or became an actual-cost claim.",
                )
        close_array(actual, chosen, "The independent hierarchical expert weights disagree.")
        maximum_weight = max(maximum_weight, float(abs(actual - chosen).max()))
        expected = [
            math.fsum(chosen[i] * float(observed[i].target[s]) for i in range(6))
            for s in data.close.columns
        ]
        close_array(
            targets.loc[day].to_numpy(), expected, "The independent asset target disagrees."
        )
        maximum_target = max(
            maximum_target, float(abs(targets.loc[day].to_numpy() - expected).max())
        )
        previous = actual
    return {
        "all_monthly_decisions_checked": len(audits),
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_scalar_target_difference": maximum_target,
        "maximum_independent_covariance_difference": maximum_covariance,
        "every_minimum_distance_merge_and_recursive_capital_split_checked": True,
        "tree_uses_independently_verified_recorded_covariance": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation rather than regenerate it.")
        for ref in saved["prepared"].values():
            base.checked_reference(ref)
        return saved["prepared"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve unresolved preparation instead of repeating it.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        read_json(rules.POLICY),
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep every real initial/monthly decision and formal return day.")
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
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future source observations changed an earlier hierarchy.")
        prefixes.append(str(day.date()))
    new_output_directory(TARGETS)
    write_text_atomic(TARGETS / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = targets.ffill().loc[anchor : window["end"]]
        symbols = participation.source.donor.rules.CORE
        active = [s for s in symbols if (known[s] > 1e-10).any()]
        if anchor not in known.index or known.isna().any().any() or len(active) < 3:
            raise QuantError("The original anchor and actual factor coverage remain required.")
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            "unscored_initial_decisions": sum(
                a["allocation"]["reason"] == "unscored_initial_equal_weights" for a in audits
            ),
            "explicit_degenerate_risk_decisions": sum(
                a["allocation"]["reason"]
                == "unscored_nonpositive_individual_variance_equal_weights"
                for a in audits
            ),
            "hierarchical_risk_decisions": sum(
                a["allocation"]["risk_estimate_available"] for a in audits
            ),
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": "formal main paths, not mature source risk estimates",
            "original_source_signal_recomputed": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
        },
    )
    prepared = {
        "targets": reference(TARGETS / "targets.csv"),
        "proof": reference(TARGETS / "proof.json"),
    }
    check_frozen(frozen)
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "prepared": prepared}
    )
    return prepared


def checked_packet(frozen):
    packet = read_json(PACKET)
    entry = common.packet_entry(CANDIDATE, frozen)
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != [entry]
        or packet["new_registered_candidates"] != 1
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or not pd.Timestamp(entry["registered_at"])
        <= pd.Timestamp(entry["target_proof_at"])
        <= created
        or any(created > pd.Timestamp(e["at"]) for e in matching_events("path_started", ID))
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The complete pre-account hierarchy packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, frozen, _ = inputs()
        old = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] != ID]
        if len(old) != 516 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve every old completed or failed financial identity.")
        if matching_events("technical_failure", ID):
            raise QuantError("A terminal new version cannot be silently retried.")
        found = matching_events("registered", ID)
        body = {"candidate": CANDIDATE, "frozen": frozen}
        if found and (len(found) != 1 or found[0]["body"] != body):
            raise QuantError("Registered code, inputs and policy cannot change.")
        if not found:
            append_event(base.JOURNAL, "registered", ID, body)
        try:
            if not matching_events("target_proof", ID):
                current.progress(
                    "preparing_registered_hierarchical_risk",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, frozen)
                append_event(
                    base.JOURNAL,
                    "target_proof",
                    ID,
                    {
                        "registered_event_sha256": matching_events("registered", ID)[0]["sha256"],
                        **prepared,
                    },
                )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [common.packet_entry(CANDIDATE, frozen)],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 1,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "formal main paths, not source risk assessments",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_hierarchical_risk_preparation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_hierarchical_risk_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_hierarchical_risk_registered_waiting_publication",
            active_compute_job=None,
            expert_hierarchical_risk_registration=reference(PACKET),
            pending_steps=["publish_exact_hierarchy_packet_before_four_accounts"],
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
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence, not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def state(frozen):
    reviews, failed, completed = (
        matching_events(k, ID) for k in ("review", "technical_failure", "path_completed")
    )
    if (
        len(reviews) > 1
        or len(failed) > 1
        or len({e["body"]["path_key"] for e in completed}) != len(completed)
    ):
        raise QuantError("Duplicate terminal events cannot select another winner.")
    paths = {
        e["body"]["path_key"]: base.verify_path(
            e["body"]["receipt"], frozen, ID, e["body"]["path_key"]
        )
        for e in completed
    }
    if reviews:
        if (
            failed
            or set(paths) != base.PATH_KEYS
            or reviews[0]["body"]["paths"] != paths
            or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"] is not True
        ):
            raise QuantError("A complete review requires every actual original path.")
        return reviews[0]["body"]
    return {
        "candidate_id": ID,
        "status": "technical_failure" if failed else "pending",
        "paths": paths,
        "failure_event_sha256": failed[0]["sha256"] if failed else None,
        "failure_stage": failed[0]["body"]["stage"] if failed else None,
    }


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != {ID: state(frozen)}:
                raise QuantError("Read preserved complete outcomes instead of replaying them.")
            return reference(RESULTS)
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_hierarchical_risk",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            if not matching_events("review", ID) and not matching_events("technical_failure", ID):
                target = frame(matching_events("target_proof", ID)[0]["body"]["targets"])
                paths = {}
                for window in read_json(base.PROTOCOL)["windows"]:
                    for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_hierarchical_risk",
                            active_candidate=ID,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, ID, window, scenario, benchmarks
                        )
                actual = participation.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in actual.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    ID,
                    {
                        "candidate_id": ID,
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
                    "candidates": {ID: state(frozen)},
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
                "expert_hierarchical_risk_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_hierarchical_risk_results=reference(RESULTS),
                pending_steps=["publish_all_four_hierarchical_risk_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_hierarchical_risk_evaluation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_hierarchical_risk_evaluation_interrupted", active_compute_job=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = state(frozen), read_json(RESULTS)
    if saved["candidates"] != {ID: actual} or actual["status"] == "pending":
        raise QuantError("Publish actual complete outcomes or retained technical failure only.")
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"]["cagr"]
        for r in benchmarks["benchmarks"]
    }
    paths = []
    for key, path in actual["paths"].items():
        year, scenario = key.split("-")
        years = int(year[:-1])
        paths.append(
            {
                **{
                    k: path[k]
                    for k in (
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
                "benchmark_net_cagr": {s: lookup[years, scenario, s] for s in ("SPY", "QQQ")},
                "actual_factor_participation": actual.get("actual_factor_participation", {}).get(
                    key
                ),
            }
        )
    counts = common.counts(
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] == ID]
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
            "reviews": [
                {
                    "candidate_id": ID,
                    "status": actual["status"],
                    "paths": paths,
                    "reviewed_at": actual.get("reviewed_at"),
                    "failure_event_sha256": actual.get("failure_event_sha256"),
                    "failure_stage": actual.get("failure_stage"),
                    "all_four_paths_same_candidate_and_frozen_parameters": actual.get(
                        "all_four_paths_same_candidate_and_frozen_parameters", False
                    ),
                    "independent_forward_validation": False,
                    "order_authority": False,
                }
            ],
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
    parser = argparse.ArgumentParser(
        description="One hierarchical complete-expert policy; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Hierarchical expert research cannot request live sources.")

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
        parser.exit(2, f"Hierarchical expert research blocked: {error}\n")


if __name__ == "__main__":
    main()
