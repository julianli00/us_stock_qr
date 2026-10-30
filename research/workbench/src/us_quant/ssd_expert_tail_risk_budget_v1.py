from __future__ import annotations

import argparse
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_tail_risk_budget_v1 as rules
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
FOLDER = base.REPORTS / "expert-tail-risk-budget-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-tail-risk-budget-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_tail_risk_budget_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_tail_risk_budget_20261030_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-tail-risk-budget-v1.json",
    "src/us_quant/expert_tail_risk_budget_v1.py",
    "src/us_quant/ssd_expert_tail_risk_budget_v1.py",
    "tests/test_expert_tail_risk_budget_v1.py",
    "tests/test_ssd_expert_tail_risk_budget_v1.py",
)
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs = source.check_frozen, source.published_inputs
frame = source.frame


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    parent = read_json(base.checked_reference(policy["source_registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    if digest_json(parent) != PARENT_SPEC:
        raise QuantError("Reuse the exact technically admitted six-expert source.")
    check_frozen(parent)
    source.checked_packet(parent)
    controls = read_json(base.checked_reference(policy["source_controls"]))
    dual = read_json(base.checked_reference(policy["additional_dual_relative_control"]))
    failed = read_json(base.checked_reference(policy["prior_terminal_covariance_failure"]))
    if (
        controls["frozen_specification_sha256"] != PARENT_SPEC
        or set(controls["candidates"]) != set(source.IDS)
        or set(dual["candidates"]) != {"expert_pool_six_dual_relative_pair_v1"}
        or any(
            r["status"] != "rejected" or len(r["paths"]) != 4
            for r in [*controls["candidates"].values(), *dual["candidates"].values()]
        )
        or failed["entries"][0]["status"] != "technical_failure"
        or failed["entries"][0]["candidate_id"] in rules.EXPERT_IDS
        or failed["entries"][0]["candidate_accounts_started"] is not False
    ):
        raise QuantError("Completed controls remain exposed and terminal versions are not experts.")
    data, experts, old_policy, _, old_refs, clocks, _, benchmarks = source.source_inputs()
    if any(
        policy[k] != old_policy[k]
        for k in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "training_sessions",
            "validation_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
            "switch_cost_bps_per_side",
            "switch_commission_per_changed_asset_usd",
            "switch_cost_amortization_sessions",
            "changed_asset_numeric_tolerance",
        )
    ):
        raise QuantError("Change the allocator, not the admitted pool or funded execution.")
    old_intent = read_json(source.INTENT)
    refs = [
        *old_refs,
        *old_intent["source_references"],
        *old_intent["model_and_test_files"],
        *old_intent["validation_evidence"],
        reference(source.INTENT),
        *read_json(DESIGN)["source_references"],
        *[
            policy[k]
            for k in (
                "source_registration",
                "source_controls",
                "additional_dual_relative_control",
                "risk_diagnosis",
                "prior_terminal_covariance_failure",
            )
        ],
    ]
    lookup = {r["path"]: r for r in parent["benchmark_artifacts"]}
    returns = {}
    if set(policy["scoring_benchmarks"]) != set(rules.BENCHMARKS):
        raise QuantError("Both original tradable benchmark books are required.")
    for symbol in rules.BENCHMARKS:
        ref = policy["scoring_benchmarks"][symbol]
        if lookup.get(ref["path"]) != ref:
            raise QuantError("Do not substitute raw price returns for original NET benchmarks.")
        book = frame(ref)
        if (
            len(book) != 2512
            or str(book.index[0].date()) != "2016-10-06"
            or str(book.index[-1].date()) != "2026-10-05"
            or not book["risk_free"].equals(data.risk_free.loc[book.index])
        ):
            raise QuantError("Keep the full original scoring benchmark calendar and RF.")
        returns[symbol] = book["return"]
        refs.append(ref)
    refs = unique_refs(refs)
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, pd.DataFrame(returns), benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    if not {
        "test_joint_tail_budget_has_no_old_asset_group_constraints",
        "test_negative_finite_opportunity_is_not_a_cash_cutoff",
        "test_infeasibility_requires_positive_phase_one_dual_certificate",
        "test_uncertified_solver_infeasibility_is_not_funding_standby",
        "test_future_source_and_benchmark_changes_preserve_prior_targets",
        "test_scalar_certificate_verifies_joint_tail_plan_and_targets",
        "test_scalar_oracle_rejects_risk_cost_dual_and_standby_tampering",
        "test_cached_sources_exclude_terminal_covariance_version",
        "test_complete_dependency_publication_precedes_new_fence",
        "test_publication_failure_cannot_start_any_main_account",
        "test_canonical_export_preserves_original_four_paths_without_private_rows",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
    }.issubset(names) or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError(
            "Require completed tail, infeasibility, source, scalar and funding checks."
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
        raise QuantError("Validation must bind the exact current tail allocator and driver.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve complete or partial implementation freezes.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            len(events) != 504
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 46
            or counts["new_completed_strategy_evaluations"] != 42
            or counts["new_technical_failures"] != 4
            or counts["new_pending_candidates"] != 0
            or design["actual_continuation_event"] != 45924
            or design["actual_invocation"] != reference(INVOCATION)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Freeze against the actual post-failure journal and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 45924,
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
            "expert_tail_budget_frozen",
            active_compute_job=None,
            expert_tail_budget_intent=reference(INTENT),
            pending_steps=["register_new_tail_rule_before_real_allocations_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks, summaries = source_inputs()
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
        raise QuantError("The actual pre-outcome tail-policy freeze changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *[reference(ROOT / p) for p in OWN[1:]],
                reference(ROOT / "src/us_quant/joint_tail_construction_v1.py"),
            ]
        ),
        "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
        "source_six_expert_pool_policy": parent["policy"],
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_joint_tail_budget",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 220,
        "prior_journal_sequence": 504,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "same_six_complete_experts_robust_NET_opportunity_joint_tail_budget",
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
        "funding_standby_is_strategy_expert": False,
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, policy, benchmarks, frozen, summaries


def scalar_tail(values):
    ordered = sorted((float(v) for v in values), reverse=True)
    mass = 0.05 * len(ordered)
    whole = int(math.floor(mass))
    return (math.fsum(ordered[:whole]) + (mass - whole) * ordered[whole]) / mass


def scalar_problem(observed, benchmark_returns, previous):
    index, rf = observed[0].net_returns.index, observed[0].risk_free
    returns = [
        [float(o.net_returns.iloc[t]) - float(rf.iloc[t]) for o in observed] for t in range(126)
    ]
    benchmark = [
        [float(benchmark_returns.loc[index[t], s]) - float(rf.iloc[t]) for s in rules.BENCHMARKS]
        for t in range(126)
    ]
    assets = list(observed[0].target.index)
    targets = [[float(o.target[s]) for o in observed] for s in assets]
    width = 135 + len(assets)
    c = [0.0] * width
    c[6] = -1.0
    for i in range(len(assets)):
        c[135 + i] = 5 / 10000 / 21
    rows, rhs, periods, cap_rows = [], [], {}, []
    for label, start, stop, theta, first in (
        ("older_84", 0, 84, 7, 8),
        ("recent_42", 84, 126, 92, 93),
    ):
        values = returns[start:stop]
        losses = {
            s: scalar_tail([-benchmark[t][j] for t in range(start, stop)])
            for j, s in enumerate(rules.BENCHMARKS)
        }
        budget = min(losses.values())
        means = [math.fsum(r[i] for r in values) / len(values) for i in range(6)]
        for t, scenario in enumerate(values):
            row = [0.0] * width
            row[:6], row[theta], row[first + t] = [-v for v in scenario], -1, -1
            rows.append(row)
            rhs.append(0.0)
        row = [0.0] * width
        row[theta] = 1.0
        for t in range(len(values)):
            row[first + t] = 1 / (0.05 * len(values))
        cap_rows.append(len(rows))
        rows.append(row)
        rhs.append(budget)
        row = [0.0] * width
        row[:6], row[6] = [-v for v in means], 1
        rows.append(row)
        rhs.append(0.0)
        periods[label] = {
            "observations": len(values),
            "NET_RF_excess_means": means,
            "benchmark_RF_tail_losses": losses,
            "risk_budget": budget,
            "scenarios": values,
        }
    for i, target in enumerate(targets):
        for sign in (1, -1):
            row = [0.0] * width
            row[:6], row[135 + i] = [sign * v for v in target], -1
            rows.append(row)
            rhs.append(sign * float(previous.iloc[i]))
    eq = np.zeros((1, width))
    eq[0, :6] = 1
    bounds = (
        [(0, 1)] * 6
        + [(None, None)] * 2
        + [(0, None)] * 84
        + [(None, None)]
        + [(0, None)] * (42 + len(assets))
    )
    return (
        np.asarray(c),
        np.asarray(rows),
        np.asarray(rhs),
        eq,
        np.ones(1),
        bounds,
        periods,
        cap_rows,
    )


def check_certificate(record, expected):
    if (
        set(record) != set(expected)
        or not np.isfinite(list(record.values())).all()
        or any(abs(record[k] - expected[k]) > 1e-10 for k in expected)
    ):
        raise QuantError("A retained primal-dual certificate was altered or lost precision.")


def verify_targets(data, experts, benchmarks, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every original initial/monthly recommendation and return date.")
    previous, maximum_target, maximum_input = None, 0.0, 0.0
    mixes = standby = warmup = 0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or any(
                o.target is None or o.target_known_at is None or o.target_known_at > day
                for o in observed
            )
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["previous_nominal_meta_target"]
            != (previous.tolist() if previous is not None else None)
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The independent expert clocks or main/shadow capital scope changed.")
        allocation = audit["allocation"]
        weights = np.asarray(audit["expert_weights"], dtype=float)
        if weights.shape != (6,) or not np.isfinite(weights).all() or (weights < 0).any():
            raise QuantError(
                "The expert coordinate vector cannot be partial, negative or nonfinite."
            )
        equal = np.array(
            [math.fsum(float(o.target[s]) / 6 for o in observed) for s in data.close.columns]
        )
        reference_target = (
            pd.Series(equal, index=data.close.columns) if previous is None else previous
        )
        if allocation["allocation_risk_feasibility_or_safety_claimed"] is not False:
            raise QuantError("Tail estimation and funding standby cannot claim safety.")
        if not all(o.score_mature for o in observed):
            warmup += 1
            expected = equal
            if (
                allocation["mode"] != "unscored_equal"
                or allocation["reason"] != "unmatured_initial_history"
                or allocation["risk_problem_mature"] is not False
                or allocation["available_observations"] != len(observed[0].net_returns)
                or max(abs(weights - 1 / 6)) > 1e-12
            ):
                raise QuantError("Unmatured observations cannot inherit a certified tail problem.")
        else:
            index, rf = observed[0].net_returns.index, observed[0].risk_free
            if (
                any(
                    not o.net_returns.index.equals(index) or not o.risk_free.equals(rf)
                    for o in observed
                )
                or not index.isin(benchmarks.index).all()
                or allocation["risk_sample_start"] != str(index[0].date())
                or allocation["risk_sample_end"] != str(index[-1].date())
                or allocation["risk_problem_mature"] is not True
                or allocation["joint_selection_is_independent_validation"] is not False
                or allocation["funding_standby_is_strategy_expert"] is not False
            ):
                raise QuantError(
                    "Both original subperiods need exact matured source and benchmark dates."
                )
            c, a, b, eq, rhs, bounds, periods, cap_rows = scalar_problem(
                observed, benchmarks, reference_target
            )
            for label, period in periods.items():
                recorded = allocation["periods"][label]
                if (
                    recorded["observations"] != period["observations"]
                    or not np.isfinite(
                        [
                            recorded["risk_budget"],
                            *recorded["NET_RF_excess_means"],
                            *recorded["benchmark_RF_tail_losses"].values(),
                        ]
                    ).all()
                    or set(recorded["benchmark_RF_tail_losses"]) != set(rules.BENCHMARKS)
                ):
                    raise QuantError("All recorded risk budgets and NET means must remain finite.")
                errors = [
                    abs(recorded["risk_budget"] - period["risk_budget"]),
                    *abs(
                        np.asarray(recorded["NET_RF_excess_means"]) - period["NET_RF_excess_means"]
                    ),
                    *[
                        abs(
                            recorded["benchmark_RF_tail_losses"][s]
                            - period["benchmark_RF_tail_losses"][s]
                        )
                        for s in rules.BENCHMARKS
                    ],
                ]
                maximum_input = max(maximum_input, max(errors))
            if allocation["mode"] == "funding_standby":
                standby += 1
                if (
                    allocation["reason"] != "certified_observed_joint_tail_budget_infeasible"
                    or allocation["primary_solver_status"] != 2
                    or (weights != 0).any()
                ):
                    raise QuantError("Funding standby is not another expert or a successful score.")
                pc = np.r_[np.zeros(len(c)), 1.0]
                pa = np.column_stack([a, np.zeros(len(a))])
                pa[cap_rows, -1] = -1
                peq = np.column_stack([eq, np.zeros(len(eq))])
                result = rules.lp.recorded_solution(allocation["phase_one_solution"])
                proof = rules.lp.certificate(result, pc, pa, b, peq, rhs, [*bounds, (0, None)])
                check_certificate(allocation["phase_one_certificate"], proof)
                if min(proof["objective"], proof["dual_objective"]) <= rules.lp.FEAS_TOL:
                    raise QuantError(
                        "No positive PhaseI lower bound proves observed infeasibility."
                    )
                expected = np.zeros(len(data.close.columns))
                expected[list(data.close.columns).index("BIL")] = 0.98
            elif allocation["mode"] == "expert_mix":
                mixes += 1
                if (
                    allocation["reason"] != "robust_NET_opportunity_under_joint_tail_budget"
                    or allocation["primary_solver_status"] != 0
                    or abs(float(weights.sum()) - 1) > 1e-12
                ):
                    raise QuantError(
                        "One funded simplex is required for complete-expert allocation."
                    )
                result = rules.lp.recorded_solution(allocation["primary_solution"])
                proof = rules.lp.certificate(result, c, a, b, eq, rhs, bounds)
                check_certificate(allocation["primary_certificate"], proof)
                raw = result.x[:6]
                if (raw < -rules.lp.ROUND_TOL).any():
                    raise QuantError("The raw numerical optimum exceeds frozen weight roundoff.")
                cleaned = raw.copy()
                cleaned[abs(cleaned) <= rules.lp.ROUND_TOL] = 0
                cleaned /= cleaned.sum()
                if max(abs(cleaned - weights)) > 1e-12:
                    raise QuantError(
                        "Retained expert weights differ from declared numerical cleanup."
                    )
                expected = np.array(
                    [
                        math.fsum(float(observed[i].target[s]) * weights[i] for i in range(6))
                        for s in data.close.columns
                    ]
                )
                means = []
                for label, period in periods.items():
                    scenarios = period["scenarios"]
                    loss = [-math.fsum(weights[i] * row[i] for i in range(6)) for row in scenarios]
                    tail = scalar_tail(loss)
                    order = sorted(range(len(loss)), key=lambda t: -loss[t])
                    mass = 0.05 * len(loss)
                    whole = int(math.floor(mass))
                    marginal = np.array(
                        [
                            (
                                -math.fsum(scenarios[t][i] for t in order[:whole])
                                - (mass - whole) * scenarios[order[whole]][i]
                            )
                            / mass
                            for i in range(6)
                        ]
                    )
                    mean = math.fsum(
                        math.fsum(weights[i] * row[i] for i in range(6)) for row in scenarios
                    ) / len(scenarios)
                    recorded = allocation["periods"][label]
                    if (
                        tail > period["risk_budget"] + rules.lp.FEAS_TOL
                        or not np.isfinite(
                            [
                                recorded["observed_tail_loss"],
                                recorded["achieved_RF_excess_mean"],
                                *recorded["tail_marginal_losses"],
                                *recorded["signed_tail_risk_contributions"],
                            ]
                        ).all()
                        or abs(recorded["observed_tail_loss"] - tail) > 1e-12
                        or abs(recorded["achieved_RF_excess_mean"] - mean) > 1e-12
                        or max(abs(marginal - recorded["tail_marginal_losses"])) > 1e-12
                        or max(abs(weights * marginal - recorded["signed_tail_risk_contributions"]))
                        > 1e-12
                        or abs(float((weights * marginal).sum()) - tail) > 1e-12
                    ):
                        raise QuantError(
                            "Independent fractional joint-tail or marginal-risk check failed."
                        )
                    means.append(mean)
                proportional = math.fsum(abs(expected - reference_target.to_numpy())) * 5 / 10000
                achieved = min(means) - proportional / 21
                if (
                    not np.isfinite(
                        [
                            allocation["achieved_robust_RF_opportunity_after_proportional_proxy"],
                            allocation["maximum_weight_roundoff_change"],
                        ]
                    ).all()
                    or achieved < -proof["objective"] - rules.lp.FEAS_TOL
                    or abs(
                        achieved
                        - allocation["achieved_robust_RF_opportunity_after_proportional_proxy"]
                    )
                    > 1e-12
                    or abs(
                        float(abs(raw - weights).max())
                        - allocation["maximum_weight_roundoff_change"]
                    )
                    > 1e-12
                ):
                    raise QuantError(
                        "Exact opportunity and proportional switch costs reject the record."
                    )
            else:
                raise QuantError(
                    "A mature decision needs explicit certified allocation or standby."
                )
            change = abs(expected - reference_target.to_numpy())
            proportional = math.fsum(change) * 5 / 10000
            changed = sum(v > 1e-12 for v in change)
            if (
                not np.isfinite(
                    [
                        allocation["estimated_incremental_proportional_cost_fraction"],
                        allocation["estimated_incremental_fixed_ticket_cost_fraction"],
                    ]
                ).all()
                or abs(
                    proportional - allocation["estimated_incremental_proportional_cost_fraction"]
                )
                > 1e-12
                or abs(
                    changed / 10000 - allocation["estimated_incremental_fixed_ticket_cost_fraction"]
                )
                > 1e-12
                or changed != allocation["estimated_incremental_changed_assets"]
                or allocation["cost_proxy_is_actual_main_account_cost"] is not False
            ):
                raise QuantError("Proxies cannot be silent missing values or actual main fees.")
        maximum_target = max(
            maximum_target, float(abs(targets.loc[day].to_numpy() - expected).max())
        )
        if (expected < 0).any() or abs(float(expected.sum()) - 0.98) > 1e-12:
            raise QuantError("The independent target is not one cash-funded underlying portfolio.")
        previous = pd.Series(targets.loc[day].to_numpy().copy(), index=data.close.columns)
    if maximum_target > 1e-12 or maximum_input > 1e-12:
        raise QuantError("Independent scalar scenario inputs or unified targets differ.")
    return {
        "all_monthly_decisions_checked": len(audits),
        "explicit_unmatured_equal_decisions": warmup,
        "certified_expert_mix_decisions": mixes,
        "certified_infeasible_funding_standby_decisions": standby,
        "maximum_independent_scalar_target_difference": maximum_target,
        "maximum_independent_tail_budget_or_mean_difference": maximum_input,
    }


def prepare(data, experts, benchmarks, frozen):
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
        benchmarks,
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, benchmarks, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep all initial/monthly decisions and every formal return day.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            benchmarks.loc[:day],
            before_decision=runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future observations changed an earlier joint-tail allocation.")
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
            raise QuantError(
                "Formal anchors and the original minimum factor scope remain required."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": active})
    write_new_receipt(
        TARGETS / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            **independent,
            "initial_capital_decision_is_not_a_fabricated_month_end": True,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "information_audits": reference(TARGETS / "information-audits.json"),
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "funding_standby_is_strategy_expert": False,
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": "formal main paths, not mature source-tail assessment",
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
        raise QuantError("The complete pre-account joint-tail packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, _, benchmarks, frozen, _ = inputs()
        old = [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] != ID]
        if len(old) != 504 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError(
                "Preserve all old completed or failed identities, including covarianceV1."
            )
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
                    "preparing_registered_expert_tail_budget",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, benchmarks, frozen)
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
                    "candidate_metrics_scope": "formal main paths, not source-tail assessments",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.previous.previous.source.failure("expert_tail_budget_preparation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_tail_budget_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "expert_tail_budget_registered_waiting_publication",
            active_compute_job=None,
            expert_tail_budget_registration=reference(PACKET),
            pending_steps=["publish_exact_expert_tail_packet_before_four_accounts"],
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
                "evaluating_expert_tail_budget",
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
                            "evaluating_expert_tail_budget", active_candidate=ID, active_path=key
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
                "expert_tail_budget_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_tail_budget_results=reference(RESULTS),
                pending_steps=["publish_all_four_expert_tail_budget_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.previous.previous.source.failure("expert_tail_budget_evaluation", (ID,), error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_tail_budget_evaluation_interrupted", active_compute_job=None)
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
        description="One joint-tail complete-expert policy; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Expert-tail research cannot request live sources.")

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
        parser.exit(2, f"Expert-tail research blocked: {error}\n")


if __name__ == "__main__":
    main()
