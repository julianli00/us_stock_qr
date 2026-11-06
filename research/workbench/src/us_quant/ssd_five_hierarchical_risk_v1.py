from __future__ import annotations

import argparse
import json
import math
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import five_hierarchical_risk_v1 as rules
from us_quant import ssd_conditional_volatility_quintet_v1 as prior
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_forecast_opportunity_gate_v1 import target_proof
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, template, seven, financial = (
    prior.base,
    prior.ROOT,
    prior.template,
    prior.seven,
    prior.financial,
)
lifecycle, reference, unique_refs = prior.lifecycle, prior.reference, prior.unique_refs
current_counts, source = prior.current_counts, seven
FOLDER = base.REPORTS / "five-hierarchical-risk-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/five_hierarchical_risk_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/five_hierarchical_risk_20261106_registration_v1.json"}
OWN = (
    "config/five-hierarchical-risk-v1.json",
    "src/us_quant/five_hierarchical_risk_v1.py",
    "src/us_quant/ssd_five_hierarchical_risk_v1.py",
    "tests/test_five_hierarchical_risk_v1.py",
    "FIVE_HIERARCHICAL_RISK_V1.md",
    "tests/test_conditional_volatility_quintet_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals()},
        argdefs=function.__defaults__,
    )


def dependencies():
    return bound(_dependencies)()


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("The new hierarchy cannot substitute projected financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "five_hierarchical_risk_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_catalogue"]))
    data, experts, _, refs, clocks, benchmarks = prior.source_inputs()
    parent = prior.events("registered", prior.IDS[0])[0]["body"]["frozen"]
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["source_clocks"] != clocks
        or catalogue["new_covariance_clusters_weights_targets_or_accounts_computed"] is not False
        or catalogue["old_six_targets_or_accounts_recomputed"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep the already fixed five policies, exact views and original clocks.")
    previous = read_json(base.checked_reference(catalogue["previous_completed_cycle"]))
    refs = unique_refs(
        [
            *refs,
            *previous["exact_evidence_references"],
            *catalogue["source_references"],
            policy["source_catalogue"],
            *[
                catalogue[k]
                for k in (
                    "source_quintet_policy",
                    "source_quintet_freeze",
                    "cached_quintet_results",
                    "cached_median_budget_results",
                    "previous_completed_cycle",
                    "old_six_hierarchical_policy",
                    "old_six_hierarchical_implementation",
                    "old_six_covariance_implementation",
                    "old_six_hierarchical_results",
                )
            ],
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial fixed-five hierarchy freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1381 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1381/294 without replaying recovery.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("The singleton hierarchy identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 184317
            or design["candidate_ids"] != list(IDS)
            or design["declared_financial_cohort_size"] != 1
            or design["new_covariance_clusters_weights_targets_or_accounts_computed"] is not False
        ):
            raise QuantError("Freeze the risk construction before new actual fits or targets.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Bind the dimension regression, full hierarchy and causal proof tests."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "fixed_five_full_NET_RF_single_linkage_recursive_risk",
            "expert_pool_definition_role": "unchanged_five_complete_sources_single_hierarchy",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1381,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 294,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "FIVE_HIERARCHICAL_RISK_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *refs,
                    reference(DESIGN),
                    reference(INVOCATION),
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "risk_fit_is_heldout_validation": False,
            "risk_clusters_are_independent_factor_families": False,
            "median_equity_budget_promoted_or_retuned": False,
            "old_six_or_cached_controls_replayed": False,
            "admitted_legacy_defensive_views_reused_unchanged": True,
            "all_source_native_RF_claimed_equal_without_view": False,
            "new_source_RF_views": 0,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_covariance_clusters_weights_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_any_new_actual_hierarchical_targets")
    return reference(INTENT)


def close_array(actual, expected, message):
    first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or float(abs(first - second).max(initial=0)) > 1e-12
    ):
        raise QuantError(message)


def scalar_covariance(columns):
    shifted = [[v - values[0] for v in values] for values in columns]
    means = [math.fsum(v) / len(v) for v in shifted]
    n = len(columns)
    return np.asarray(
        [
            [
                math.fsum(
                    (a - means[i]) * (b - means[j])
                    for a, b in zip(shifted[i], shifted[j], strict=True)
                )
                / (len(shifted[i]) - 1)
                for j in range(n)
            ]
            for i in range(n)
        ]
    )


def independent_hierarchy(columns, covariance, allocation):
    n, distance = len(columns), np.zeros((len(columns), len(columns)))
    for i in range(n):
        for j in range(i + 1, n):
            correlation = float(covariance[i, j]) / (
                math.sqrt(float(covariance[i, i])) * math.sqrt(float(covariance[j, j]))
            )
            if not math.isfinite(correlation) or abs(correlation) > 1 + 1e-12:
                raise QuantError("Invalid independently checked risk correlation.")
            distance[i, j] = distance[j, i] = math.sqrt((1 - min(1.0, max(-1.0, correlation))) / 2)
    close_array(allocation["risk_distance"], distance, "Risk distances changed.")
    active = {(i,): [i] for i in range(n)}
    if len(allocation["canonical_merges"]) != n - 1:
        raise QuantError("Retain every unique minimum-distance merge.")
    for recorded in allocation["canonical_merges"]:
        candidates, groups = [], sorted(active)
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                edge = min(float(distance[a, b]) for a in left for b in right)
                candidates.append((round(edge, 12), left, right))
        key, left, right = sorted(candidates)[0]
        combined, leaves = tuple(sorted(left + right)), active.pop(left) + active.pop(right)
        if recorded != {
            "left_members": list(left),
            "right_members": list(right),
            "distance_key": key,
            "merged_members": list(combined),
            "leaf_order": leaves,
        }:
            raise QuantError("Independent minimum-distance merge or canonical tie disagrees.")
        active[combined] = leaves
    order = next(iter(active.values()))
    if allocation["seriated_expert_coordinates"] != order:
        raise QuantError("The complete canonical seriation changed.")
    weights, expected_splits, queue = np.zeros(n), [], [(order, 1.0)]

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

    while queue:
        members, capital = queue.pop()
        if len(members) == 1:
            weights[members[0]] = capital
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
    actual = allocation["recursive_capital_splits"]
    if len(actual) != n - 1:
        raise QuantError("Every recursive split must be recorded.")
    numerical = (
        "parent_capital_fraction",
        "left_IVP_variance",
        "right_IVP_variance",
        "left_capital_fraction_of_parent",
    )
    for recorded, expected in zip(actual, expected_splits, strict=True):
        if {k: v for k, v in recorded.items() if k not in numerical} != (
            {k: v for k, v in expected.items() if k not in numerical}
        ):
            raise QuantError("A zero-risk case or recursive cluster membership differs.")
        close_array(
            [recorded[k] for k in numerical],
            [expected[k] for k in numerical],
            "Independent child risk or capital fraction disagrees.",
        )
    return weights


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or not targets[rules.ID].index.equals(data.close.index)
        or not targets[rules.ID].columns.equals(data.close.columns)
        or not targets[rules.ID].dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
    ):
        raise QuantError("Keep the complete original formal calendar and target coordinates.")
    previous = np.full(5, 0.2)
    max_covariance = max_weight = max_target = 0.0
    for day, audit in zip(dates, audits, strict=True):
        observed = [e.observe(day, 126) for e in experts]
        rules.checked_opinions(observed, day)
        if (
            audit["decision_session"] != str(day.date())
            or audit["decision_information_clock"] != rules.CLOCK
            or audit["source_shadow_wealth_transferred"] is not False
            or audit["independent_forward_validation"] is not False
            or audit["source_score_mature"] != [o.score_mature for o in observed]
        ):
            raise QuantError("Source clocks, maturity or capital separation changed.")
        close_array(
            audit["previous_expert_weights"], previous, "Previous statistical weights differ."
        )
        chosen, row = np.full(5, 0.2), audit["allocation"]
        if row["safety_or_success_claimed"] is not False:
            raise QuantError("An observed risk estimate cannot acquire a safety claim.")
        if not all(o.score_mature for o in observed):
            if (
                row["reason"] != "unscored_initial_equal_weights"
                or row["risk_estimate_available"] is not False
                or row["available_shadow_observations"] != [len(o.net_returns) for o in observed]
            ):
                raise QuantError("Preserve explicitly unscored initial equal allocations.")
        else:
            dates0, rf = observed[0].net_returns.index, observed[0].risk_free
            if any(
                not o.net_returns.index.equals(dates0)
                or not o.risk_free.equals(rf)
                or len(o.net_returns) != 126
                or o.net_returns.index[-1] > day
                for o in observed
            ):
                raise QuantError("Use all exact common mature NET/RF dates.")
            columns = [
                [float(v) - float(r) for v, r in zip(o.net_returns, rf, strict=True)]
                for o in observed
            ]
            scalar, covariance = scalar_covariance(columns), np.asarray(row["covariance"])
            close_array(covariance, scalar, "Independent scalar covariance differs.")
            max_covariance = max(max_covariance, float(abs(covariance - scalar).max()))
            if row["risk_sample_start"] != str(dates0[0].date()) or (
                row["risk_sample_end"] != str(dates0[-1].date())
            ):
                raise QuantError("The matured fit clock changed.")
            if any(all(v == values[0] for v in values) for values in columns):
                if (
                    row["reason"] != "unscored_nonpositive_individual_variance_equal_weights"
                    or row["risk_estimate_available"] is not False
                ):
                    raise QuantError("Undefined individual risk cannot become fit success.")
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
                    or row["cost_proxy_used_for_allocation"] is not False
                    or row["nominal_target_change_is_actual_turnover"] is not False
                ):
                    raise QuantError("Keep only the declared hierarchical risk construction.")
                chosen = independent_hierarchy(columns, covariance, row)
                mixed = [math.fsum(chosen[i] * columns[i][k] for i in range(5)) for k in range(126)]
                mean = math.fsum(mixed) / 126
                vol = math.sqrt(252 * math.fsum((v - mean) ** 2 for v in mixed) / 125)
                close_array(
                    [row["forecast_annualized_RF_excess_volatility"]],
                    [vol],
                    "Independent forecast risk differs.",
                )
                movement = math.fsum(
                    abs(
                        math.fsum(
                            (chosen[i] - previous[i]) * float(observed[i].target[s])
                            for i in range(5)
                        )
                    )
                    for s in data.close.columns
                )
                close_array(
                    [row["nominal_target_change_L1"]],
                    [movement],
                    "Nominal movement changed or became actual turnover.",
                )
        actual = np.asarray(audit["expert_weights"])
        if (
            actual.shape != (5,)
            or not np.isfinite(actual).all()
            or (actual < 0).any()
            or abs(float(actual.sum()) - 1) > 1e-12
        ):
            raise QuantError("The recorded expert allocation must remain a funded unit-sum vector.")
        close_array(actual, chosen, "Independent expert allocation differs.")
        expected = [
            math.fsum(chosen[i] * float(observed[i].target[s]) for i in range(5))
            for s in data.close.columns
        ]
        target = targets[rules.ID].loc[day].to_numpy()
        close_array(target, expected, "Independent unified asset target differs.")
        max_weight = max(max_weight, float(abs(actual - chosen).max()))
        max_target = max(max_target, float(abs(target - expected).max()))
        previous = actual
    return {
        "all_monthly_decisions_checked": len(audits),
        "maximum_independent_covariance_difference": max_covariance,
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "every_minimum_distance_merge_and_recursive_split_checked": True,
        "tree_uses_independently_verified_recorded_covariance": True,
    }


def verify_retained_records(path, experts, targets, audits):
    with path.open() as stream:
        for i, audit in enumerate(audits):
            first, second = stream.readline(), stream.readline()
            if not first or not second:
                raise QuantError("A partial hierarchy input stream remains a failure.")
            before, after = json.loads(first), json.loads(second)
            observed = [e.observe(pd.Timestamp(audit["decision_session"]), 126) for e in experts]
            if (
                before["record"] != "decision_input"
                or after["record"] != "decision_resolved"
                or before["decision_index"] != i
                or after["decision_index"] != i
                or before["as_of"] != audit["decision_session"]
                or before["observations"] != rules.snapshot(observed)
                or before["previous_expert_weights_binary64_hex"]
                != [float(v).hex() for v in audit["previous_expert_weights"]]
                or after["audit"] != audit
                or after["asset_order"] != list(targets[rules.ID].columns)
                or after["combined_targets"]
                != {rules.ID: targets[rules.ID].loc[audit["decision_session"]].tolist()}
            ):
                raise QuantError("Preserve all exact source NET/RF/opinion values and resolutions.")
        if stream.readline():
            raise QuantError("Unexpected extra hierarchy input records.")
    return {
        "actual_input_and_resolved_record_count": 2 * len(audits),
        "exact_unrounded_source_NET_RF_opinions_and_state_retained": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior hierarchical targets.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve partial risk preparation rather than repeat it.")
    if len(events("registered", rules.ID)) != 1:
        raise QuantError("Register before any new actual risk fit or target.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits)
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep every original initial and month-end decision.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if (
            not early[rules.ID].equals(targets[rules.ID].loc[:day])
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future observations changed an earlier risk allocation.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    active, coverage = targets[rules.ID].dropna(how="all"), []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError(
                "The hierarchy still needs known initial opinion and original factors."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_text_atomic(directory / "targets.csv", targets[rules.ID].to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        target_proof(
            frozen,
            {
                **retained,
                **independent,
                "decision_count": len(audits),
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "risk_allocated_decisions": sum(
                    a["allocation"]["risk_estimate_available"] for a in audits
                ),
                "new_source_or_shadow_account_runs": 0,
                "old_six_strategy_or_accounts_replayed": False,
                "admitted_legacy_defensive_views_reused_unchanged": True,
                "all_source_native_RF_claimed_equal_without_view": False,
            },
        ),
    )
    entries = {
        rules.ID: {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    }
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "entries": entries})
    return entries


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if events("registered", rules.ID) or events("technical_failure", rules.ID):
            raise QuantError("Do not replay or rescue a registered hierarchy identity.")
        base.append_event(
            base.JOURNAL,
            "registered",
            rules.ID,
            {
                "candidate": CANDIDATES[0],
                "frozen": frozen,
            },
        )
        try:
            prepared = prepare(data, experts, frozen)
            registered = events("registered", rules.ID)[0]["sha256"]
            base.append_event(
                base.JOURNAL,
                "target_proof",
                rules.ID,
                {
                    "registered_event_sha256": registered,
                    **prepared[rules.ID],
                },
            )
            entry = seven.common.packet_entry(CANDIDATES[0], frozen)
            seven.common.retain_receipt(
                PACKETS[rules.ID],
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [entry],
                    "complete_fixed_cohort_ids": list(IDS),
                    "cohort_registrations": [registered],
                    "candidate_accounts_or_metrics_computed": False,
                    "order_authority": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                },
                ("created_at",),
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                rules.ID,
                {
                    "stage": "five_hierarchical_risk_target_preparation",
                    "reason": str(error),
                    "partial_inputs_preserved": True,
                },
            )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_inputs_preserved": True,
                    "old_six_or_failed_algorithm_rescue": False,
                },
            )
            progress(
                "technical_failure_preserved", next_step="do_not_retry_or_retune_this_identity"
            )
            raise
        progress("registered_before_accounts", next_step="deduplicate_then_publish_actual_packet")
    return {rules.ID: reference(PACKETS[rules.ID])}


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        raise QuantError("Do not replay the completed fixed-five hierarchy accounts.")
    ref = scoped(financial.evaluate, CANDIDATES[0])(commit)
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": {rules.ID: ref},
            "all_declared_candidates_and_four_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "old_six_and_cached_controls_replayed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("all_four_paths_complete", next_step="publish_every_path_without_source_replay")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        controls = read_json(base.checked_reference(catalogue["cached_quintet_results"]))[
            "candidates"
        ]
        median = read_json(base.checked_reference(catalogue["cached_median_budget_results"]))
        controls = {**controls, **median["candidates"]}
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish all four original hierarchy paths.")
        contrasts = {
            name: {
                key: {
                    "metrics": {
                        m: row["metrics"][m] - old["paths"][key]["metrics"][m]
                        for m in ("cagr", "sharpe", "max_drawdown")
                    },
                    "execution": {
                        m: row["execution"][m] - old["paths"][key]["execution"][m]
                        for m in (
                            "annualized_two_sided_turnover",
                            "model_order_count",
                            "total_cost_usd",
                        )
                    },
                }
                for key, row in result["paths"].items()
            }
            for name, old in controls.items()
        }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": {
                    rules.ID: {"status": result["review"]["status"], "paths": result["paths"]}
                },
                "matched_cached_controls": {
                    name: {**old, "old_control_replayed": False, "snapshot_is_current": False}
                    for name, old in controls.items()
                },
                "new_minus_matched_cached_controls": contrasts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "source_pool_size": 5,
                "old_six_strategy_or_accounts_replayed": False,
                "risk_fit_is_heldout_validation": False,
                "risk_clusters_are_independent_factor_families": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "old_controls_or_source_models_replayed": False,
                "raw_prices_targets_or_daily_rows_exported": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual complete singleton preregistration SHA.")
    print(
        json.dumps(
            {
                "freeze": initialize,
                "register": register,
                "evaluate": lambda: evaluate(args.commit),
                "export": export,
            }[args.action]()
        )
    )
