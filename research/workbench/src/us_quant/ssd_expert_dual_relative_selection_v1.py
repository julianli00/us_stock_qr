from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_dual_relative_selection_v1 as rules
from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
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
FOLDER = base.REPORTS / "expert-dual-relative-selection-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-dual-relative-selection-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_dual_relative_selection_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_dual_relative_selection_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-dual-relative-selection-v1.json",
    "src/us_quant/expert_dual_relative_selection_v1.py",
    "src/us_quant/ssd_expert_dual_relative_selection_v1.py",
    "tests/test_expert_dual_relative_selection_v1.py",
    "tests/test_ssd_expert_dual_relative_selection_v1.py",
)
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs = source.check_frozen, source.published_inputs
frame = source.frame


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if digest_json(parent) != PARENT_SPEC or parent["policy"] != policy["source_pool_policy"]:
        raise QuantError("Use the exact fixed six-expert source, not a newly selected pool.")
    check_frozen(parent)
    source.checked_packet(parent)
    controls = read_json(base.checked_reference(policy["source_controls"]))
    if (
        controls["frozen_specification_sha256"] != PARENT_SPEC
        or set(controls["candidates"]) != set(source.IDS)
        or any(
            r["status"] != "rejected" or len(r["paths"]) != 4
            for r in controls["candidates"].values()
        )
    ):
        raise QuantError(
            "Reuse complete rejected source controls, not replayed or filtered winners."
        )
    data, experts, old_policy, _, old_refs, clocks, _, benchmarks = source.source_inputs()
    if any(
        policy[k] != old_policy[k]
        for k in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "training_sessions",
            "validation_sessions",
            "selected_experts",
            "selection_score_decimal_places",
            "switch_cost_bps_per_side",
            "switch_commission_per_changed_asset_usd",
            "switch_cost_amortization_sessions",
            "changed_asset_numeric_tolerance",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
        )
    ):
        raise QuantError(
            "Change only selection objective, not source risk or execution parameters."
        )
    source_published_packet = read_json(
        base.checked_reference(controls["publication"]["registration_packet"])
    )
    if source_published_packet["study_id"] != old_policy["study_id"]:
        raise QuantError("The actual source publication packet changed.")
    original = read_json(base.checked_reference(source.reference(source.INTENT)))
    old_dependencies = [
        *original["source_references"],
        *original["model_and_test_files"],
        *original["validation_evidence"],
    ]
    for ref in old_dependencies:
        base.checked_reference(ref)
    benchmark_refs = {r["path"]: r for r in parent["benchmark_artifacts"]}
    returns = {}
    for name in rules.BENCHMARKS:
        ref = policy["scoring_benchmarks"][name]
        if benchmark_refs.get(ref["path"]) != ref:
            raise QuantError(
                "Use original full-capital NET benchmark books, not market-price proxies."
            )
        book = frame(ref)
        if (
            len(book) != 2512
            or str(book.index[0].date()) != "2016-10-06"
            or str(book.index[-1].date()) != "2026-10-05"
            or not book["risk_free"].equals(data.risk_free.loc[book.index])
        ):
            raise QuantError("Keep original benchmark calendar, reference wealth and RF.")
        returns[name] = book["return"]
    benchmark_returns = pd.DataFrame(returns)
    refs = unique_refs(
        [
            *old_refs,
            *old_dependencies,
            reference(source.INTENT),
            *[
                policy[k]
                for k in (
                    "source_registration",
                    "source_pool_policy",
                    "source_controls",
                    "score_alignment_diagnosis",
                )
            ],
            *policy["scoring_benchmarks"].values(),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, benchmark_returns, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    required = {
        "test_dual_objective_keeps_original_risk_weights_and_all_four_scores",
        "test_maturity_and_actual_initial_anchor_preserve_all_days",
        "test_future_benchmark_and_source_changes_do_not_change_past",
        "test_missing_or_invalid_benchmark_is_not_filled",
        "test_scalar_oracle_checks_all_fifteen_pairs_and_scores",
        "test_scalar_oracle_rejects_nonselected_score_or_risk_tampering",
        "test_cached_inputs_do_not_invoke_completed_pipeline_entrypoints",
        "test_full_dependency_verification_precedes_new_publication_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_all_four_paths_without_private_rows",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
    }
    if not required.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed active-score, clock, scalar, source and publication checks."
        )
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
        raise QuantError("Validation must bind the exact current implementation bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve complete or partial implementation freezes.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, policy, parent, refs, clocks, _, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        if (
            len(events) != 490
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 44
            or counts["new_completed_strategy_evaluations"] != 41
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
        ):
            raise QuantError("Freeze against the actual post-diagnosis financial history.")
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            design["actual_continuation_event"] != 43905
            or design["invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Preserve the true design and scheduled invocation origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 43905,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "question": policy["methodology"]["question"],
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
            "dual_relative_rule_frozen",
            active_compute_job=None,
            dual_relative_intent=reference(INTENT),
            pending_steps=["register_new_rule_before_real_active_scores_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmark_returns, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_scores_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome implementation freeze changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs([*parent["code"], *[reference(ROOT / p) for p in OWN[1:]]]),
        "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
        "source_six_expert_pool_policy": parent["policy"],
        "expert_pool": parent["expert_pool"],
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_new_dual_relative_objective",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 219,
        "prior_journal_sequence": 490,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_worst_dual_benchmark_train_validation_score",
        "scoring_benchmarks": policy["scoring_benchmarks"],
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
    return data, experts, policy, benchmark_returns, frozen, benchmarks


def scalar_stats(values, cost):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if all(v == values[0] for v in values) or variance <= 0 or not math.isfinite(variance):
        return None
    risk = math.sqrt(variance)
    return math.sqrt(252) * (mean - cost / 21) / risk, risk


def verify_targets(data, experts, benchmarks, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Preserve the full initial/monthly target lattice and formal calendar.")
    previous_weights = np.full(6, 1 / 6)
    maximum_target = maximum_weight = maximum_score = maximum_tracking = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["previous_expert_weights"] != previous_weights.tolist()
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
            raise QuantError(
                "The original source clock, expert vector or capital separation changed."
            )
        selection = audit["selection"]
        if selection["safety_or_success_claimed"] is not False:
            raise QuantError("An active assessment cannot claim safety or final qualification.")
        if not all(o.score_mature for o in observed):
            if (
                selection["reason"] != "unscored_initial_equal_weights"
                or selection["scores_available"] is not False
                or selection["pairs"] != []
                or selection["selected_pair"] is not None
                or selection["scoring_benchmark_last_observation"] is not None
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("Unmatured history cannot acquire successful active scores.")
            chosen = np.full(6, 1 / 6)
        else:
            net = [o.net_returns.tolist() for o in observed]
            rf = observed[0].risk_free.tolist()
            excess = [[v - r for v, r in zip(values, rf, strict=True)] for values in net]
            means = [math.fsum(v[:84]) / 84 for v in excess]
            deviations = [
                math.sqrt(math.fsum((x - m) ** 2 for x in v[:84]) / 83)
                for v, m in zip(excess, means, strict=True)
            ]
            expected_pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i, j in itertools.combinations(range(6), 2)
            }
            if (
                len(selection["pairs"]) != 15
                or {tuple(p["pair"]) for p in selection["pairs"]} != expected_pairs
            ):
                raise QuantError("Verify all15pairs, not only the selected winner.")
            known = observed[0].net_returns.index
            b = benchmarks.loc[known]
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
                w = np.zeros(6)
                if reason is None:
                    w[i], w[j] = (
                        deviations[j] / (deviations[i] + deviations[j]),
                        deviations[i] / (deviations[i] + deviations[j]),
                    )
                    change = [
                        abs(
                            math.fsum(
                                (w[k] - previous_weights[k]) * float(observed[k].target[s])
                                for k in range(6)
                            )
                        )
                        for s in data.close.columns
                    ]
                    cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                    expected_scores = {}
                    for symbol in rules.BENCHMARKS:
                        active = [
                            math.fsum(
                                w[k] * (net[k][t] - float(b[symbol].iloc[t])) for k in range(6)
                            )
                            for t in range(126)
                        ]
                        values = [scalar_stats(active[:84], cost), scalar_stats(active[84:], cost)]
                        if any(v is None for v in values):
                            reason = "nonpositive_or_nonfinite_active_tracking_risk"
                            break
                        expected_scores[symbol] = values
                if reason is not None:
                    if pair.get("unscorable_reason") != reason:
                        raise QuantError("Unscorable active risk must retain its actual reason.")
                    continue
                actual = np.asarray(pair["expert_weights"], dtype=float)
                contributions = np.asarray(pair["training_marginal_variance_contributions"])
                if (
                    "unscorable_reason" in pair
                    or actual.shape != (6,)
                    or contributions.shape != (6,)
                    or not np.isfinite(actual).all()
                    or not np.isfinite(contributions).all()
                    or (actual < 0).any()
                    or abs(float(actual.sum()) - 1) > 1e-12
                    or set(pair["benchmark_scores"]) != set(rules.BENCHMARKS)
                    or pair["training_risk_basis"] != "original_RF_excess_returns"
                    or not np.isfinite(
                        [
                            pair["selection_score"],
                            pair["selection_score_key"],
                            pair["estimated_incremental_switch_cost_fraction"],
                        ]
                    ).all()
                ):
                    raise QuantError(
                        "Keep finite original-dimension risk records and both active references."
                    )
                maximum_weight = max(maximum_weight, max(abs(actual - w)))
                covariance = (
                    math.fsum(
                        (excess[i][t] - means[i]) * (excess[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                expected_contribution = np.zeros(6)
                expected_contribution[i] = w[i] * (w[i] * deviations[i] ** 2 + w[j] * covariance)
                expected_contribution[j] = w[j] * (w[j] * deviations[j] ** 2 + w[i] * covariance)
                if (
                    not math.isfinite(pair["training_correlation"])
                    or abs(
                        pair["training_correlation"] - covariance / (deviations[i] * deviations[j])
                    )
                    > 1e-10
                    or max(abs(contributions - expected_contribution)) > 1e-12
                ):
                    raise QuantError("Original training dependence and marginal risk changed.")
                all_scores = []
                for symbol, values in expected_scores.items():
                    recorded = pair["benchmark_scores"][symbol]
                    for label, (score, tracking) in zip(
                        ("train", "validation"), values, strict=True
                    ):
                        if not np.isfinite(
                            [
                                recorded[label + "_net_active_risk_score"],
                                recorded[label + "_tracking_risk"],
                            ]
                        ).all():
                            raise QuantError(
                                "Every active score and tracking risk must remain finite."
                            )
                        maximum_score = max(
                            maximum_score, abs(score - recorded[label + "_net_active_risk_score"])
                        )
                        maximum_tracking = max(
                            maximum_tracking, abs(tracking - recorded[label + "_tracking_risk"])
                        )
                        all_scores.append(recorded[label + "_net_active_risk_score"])
                if (
                    abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or pair["estimated_incremental_changed_assets"]
                    != sum(v > 1e-12 for v in change)
                    or pair["selection_score"] != min(all_scores)
                    or pair["selection_score_key"] != round(pair["selection_score"], 12)
                    or pair["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError(
                        "Keep the original switch proxy and new four-score minimum exactly."
                    )
            usable = [p for p in selection["pairs"] if "selection_score_key" in p]
            if usable:
                best = sorted(usable, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["scores_available"] is not True
                    or selection["train_start"] != str(known[0].date())
                    or selection["train_end"] != str(known[83].date())
                    or selection["validation_start"] != str(known[84].date())
                    or selection["validation_end"] != str(known[-1].date())
                    or selection["scoring_benchmark_last_observation"] != str(known[-1].date())
                    or known[-1] > day
                ):
                    raise QuantError(
                        "Keep chronological active assessment and the actual best pair."
                    )
                chosen = np.asarray(best["expert_weights"])
            else:
                if (
                    selection["scores_available"] is not False
                    or selection["reason"] != "unscored_degenerate_pair_equal_weights"
                    or selection["selected_pair"] is not None
                    or selection["scoring_benchmark_last_observation"] != str(known[-1].date())
                ):
                    raise QuantError(
                        "An all-degenerate active state cannot claim a successful score."
                    )
                chosen = np.full(6, 1 / 6)
        if not np.allclose(chosen, audit["expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("Issued expert weights differ from their active selection.")
        expected = [
            math.fsum(float(o.target[s]) * chosen[i] for i, o in enumerate(observed))
            for s in data.close.columns
        ]
        maximum_target = max(maximum_target, float(max(abs(targets.loc[day] - expected))))
        previous_weights = chosen
    if (
        maximum_target > 1e-10
        or maximum_weight > 1e-10
        or maximum_score > 1e-8
        or maximum_tracking > 1e-12
    ):
        raise QuantError(
            "Independent scalar composition, ERC or active-score verification disagrees."
        )
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_active_score_difference": maximum_score,
        "maximum_independent_tracking_risk_difference": maximum_tracking,
    }


def prepare(data, experts, benchmark_returns, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError(
                "Preserve conflicting prior preparation instead of regenerating targets."
            )
        for ref in saved["prepared"].values():
            base.checked_reference(ref)
        return saved["prepared"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve unresolved preparation rather than repeat it.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        read_json(rules.POLICY),
        benchmark_returns,
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, benchmark_returns, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep all121initial/monthly decisions and every formal return day.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            benchmark_returns.loc[:day],
            before_decision=runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future benchmark/source data changed an earlier active decision.")
        prefixes.append(str(day.date()))
    new_output_directory(TARGETS)
    write_text_atomic(TARGETS / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = targets.ffill().loc[anchor : window["end"]]
        active = [
            s for s in source.previous.previous.source.donor.rules.CORE if (known[s] > 1e-10).any()
        ]
        if anchor not in known.index or known.isna().any().any() or len(active) < 3:
            raise QuantError("Every formal anchor and target factor coverage remains required.")
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            "unscored_initial_equal_weight_decisions": sum(
                a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits
            ),
            "scored_decisions": sum(a["selection"]["scores_available"] for a in audits),
            "initial_capital_decision_is_not_a_fabricated_month_end": True,
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": "formal main paths, not mature source-shadow assessments",
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
        raise QuantError("The complete pre-account single-rule packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, benchmark_returns, frozen, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        old = [e for e in events if e["candidate_id"] != ID]
        if len(old) != 490 or old[-1]["sha256"] != rules.PRIOR_HEAD:
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
                    "preparing_registered_dual_relative_rule",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, benchmark_returns, frozen)
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
                    "candidate_metrics_scope": "formal main paths, not source-shadow assessments",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.previous.previous.source.failure("dual_relative_preparation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("dual_relative_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "dual_relative_registered_waiting_publication",
            active_compute_job=None,
            dual_relative_registration=reference(PACKET),
            pending_steps=["publish_exact_dual_relative_packet_before_four_accounts"],
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
        raise QuantError("Duplicate completion or terminal events cannot select another winner.")
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
        data, _, _, _, frozen, benchmarks = inputs()
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
                "evaluating_dual_relative_rule",
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
                            "evaluating_dual_relative_rule", active_candidate=ID, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, ID, window, scenario, benchmarks
                        )
                participation = source.previous.previous.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
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
                "dual_relative_rule_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                dual_relative_results=reference(RESULTS),
                pending_steps=["publish_all_four_dual_relative_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.previous.previous.source.failure("dual_relative_evaluation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("dual_relative_evaluation_interrupted", active_compute_job=None)
            raise


def public_result():
    base.storage_guard()
    _, _, _, _, frozen, benchmarks = inputs()
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
        description="One controlled dual-benchmark-relative rule; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Dual-relative research cannot request live sources.")

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
        parser.exit(2, f"Dual-relative research blocked: {error}\n")


if __name__ == "__main__":
    main()
