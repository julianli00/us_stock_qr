from __future__ import annotations

import argparse
import json
import math
import os
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import protective_six_adaptive_hedge_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as archive
from us_quant import ssd_quintet_adaptive_hedge_v1 as prior
from us_quant.calendar import next_session, previous_session, sessions
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
FOLDER = base.REPORTS / "protective-six-adaptive-hedge-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/protective_six_adaptive_hedge_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/protective_six_adaptive_hedge_20261106_registration_v1.json"}
OWN = (
    "config/protective-six-adaptive-hedge-v1.json",
    "src/us_quant/protective_six_adaptive_hedge_v1.py",
    "src/us_quant/ssd_protective_six_adaptive_hedge_v1.py",
    "tests/test_protective_six_adaptive_hedge_v1.py",
    "PROTECTIVE_SIX_ADAPTIVE_HEDGE_V1.md",
    "tests/test_growth_macro_defense_hedge_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_quintet_anytime_hedge_v1.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    return bound(_dependencies)()


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    actual = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in actual.items()):
        raise QuantError("The controlled source addition must retain actual financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "protective_six_adaptive_hedge_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_catalogue"]))
    data, original, _, refs, clocks, benchmarks = archive.source_inputs()
    if tuple(e.identity.identifier for e in original) != archive.rules.EXPERT_IDS:
        raise QuantError(
            "Select six real admitted complete archives,not padded five-source records."
        )
    experts = tuple(original[i] for i in rules.SLOTS)
    selected_clocks = [clocks[i] for i in rules.SLOTS]
    parent = prior.events("registered", prior.rules.ID)[0]["body"]["frozen"]
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["selected_source_slots"] != list(rules.SLOTS)
        or catalogue["source_clocks"] != selected_clocks
        or catalogue["new_six_labels_learning_targets_or_accounts_computed"] is not False
        or catalogue["old_learner_or_source_or_control_reexecuted"] is not False
        or catalogue["source_wealth_guarantee_transferred_to_main"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError(
            "Keep the controlled membership,original source clocks and assessment views."
        )
    closed = read_json(base.checked_reference(catalogue["latest_readonly_completion"]))
    refs = unique_refs(
        [
            *refs,
            *closed["exact_evidence_references"],
            *catalogue["source_references"],
            policy["source_catalogue"],
            policy["source_five_math_code"],
            *[
                catalogue[key]
                for key in (
                    "source_eight_policy",
                    "cached_five_AdaHedge_policy",
                    "cached_five_AdaHedge_freeze",
                    "cached_five_AdaHedge_results",
                    "cached_five_AdaHedge_targets",
                    "cached_five_AdaHedge_month_audits",
                    "prior_academic_reference",
                    "latest_readonly_diagnosis",
                    "latest_readonly_completion",
                    "old_different_six_policy",
                    "old_different_six_covariance_policy",
                )
            ],
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, selected_clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial protective-six freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1452 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze one source addition against the actual J1452/300 baseline.")
        if any(row["candidate_id"] in IDS for row in history):
            raise QuantError("The one declared six-source financial identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["candidate_id"] != rules.ID
            or design["new_six_labels_learning_targets_or_accounts_computed"] is not False
            or design["old_learner_or_source_or_control_reexecuted"] is not False
        ):
            raise QuantError("Freeze the complete source inclusion before new six-source labels.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Bind six real histories,dimension-safe math,prefixes and source clocks."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "controlled_existing_wealth_feedback_source_same_AdaHedge",
            "expert_pool_definition_role": "study_specific_six_no_global_admission_change",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1452,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 300,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "PROTECTIVE_SIX_ADAPTIVE_HEDGE_V1.md"),
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
            "source_wealth_guarantee_transferred_to_main": False,
            "old_learner_or_source_or_control_reexecuted": False,
            "global_pool_changed": False,
            "new_expert_admissions": 0,
            "negative_Jensen_gap_clipped": False,
            "mixability_gap_floor": None,
            "rate_or_window_sweep": False,
            "normalized_source_regret_is_main_wealth": False,
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
                "new_six_labels_learning_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_six_source_labels")
    return reference(INTENT)


def verify_targets(data, experts, target, audits, updates):
    n = rules.N
    calendar, assets = data.close.index, tuple(data.close.columns)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    if (
        not target.index.equals(calendar)
        or tuple(target.columns) != assets
        or not target.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or len(updates) != max(0, len(dates) - 2)
    ):
        raise QuantError("Keep every original full-calendar decision and genuine full-month label.")
    oracle = rules.independent_trace([row["input_loss_labels"] for row in updates])
    cumulative, previous = [Fraction(0)] * n, np.full(n, 1 / n)
    first = pd.Timestamp("2016-10-31")
    max_weight = max_target = max_gap = max_rate = max_played = max_mix = max_cov = max_cost = 0.0
    max_marginal, played_prefix, mixed_prefix = 0.0, [], []
    for i, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = rules.observations(experts, day)
        state = audit["adaptive_state"]
        if i >= 2:
            current, independent = updates[i - 2], oracle[i - 2]
            logs, _ = rules.period_logs(observed, dates[i - 1], day)
            exact = [Fraction.from_float(float(v)) for v in logs]
            high, low = max(exact), min(exact)
            labels = [0.0 if high == low else float((high - v) / (high - low)) for v in exact]
            if (
                audit["period_update"] != current
                or current["end"] != str(day.date())
                or current["start_after"] != str(dates[i - 1].date())
                or [float(v).hex() for v in current["canonical_log_relatives"]]
                != [float(v).hex() for v in logs]
                or [float(v).hex() for v in current["input_loss_labels"]]
                != [float(v).hex() for v in labels]
                or current["prequential_capital_weights_before_assimilation"] != previous.tolist()
            ):
                raise QuantError("Six-source NET logs,loss labels or prequential timing changed.")
            cumulative = [
                old + Fraction.from_float(float(value))
                for old, value in zip(cumulative, labels, strict=True)
            ]
            expected = np.asarray(independent["weights_after"])
            max_gap = max(
                max_gap,
                abs(
                    float(Decimal(state["cumulative_mixability_gap_decimal"])) - independent["gap"]
                ),
            )
            played_prefix.append(independent["played_loss"])
            mixed_prefix.append(independent["mix_loss"])
            max_played = max(
                max_played,
                abs(float(Decimal(current["played_loss_decimal"])) - independent["played_loss"]),
                abs(
                    float(Decimal(state["cumulative_played_loss_decimal"]))
                    - math.fsum(played_prefix)
                ),
            )
            max_mix = max(
                max_mix,
                abs(float(Decimal(current["mix_loss_decimal"])) - independent["mix_loss"]),
                abs(float(Decimal(state["cumulative_mix_loss_decimal"])) - math.fsum(mixed_prefix)),
            )
            if independent["learning_rate_after"] is None:
                if state["learning_rate_is_infinite"] is not True:
                    raise QuantError(
                        "Zero mixability gap requires the analytic infinite-rate limit."
                    )
            elif state["learning_rate_is_infinite"]:
                raise QuantError("A positive six-source gap cannot become infinite-rate FTL.")
            else:
                max_rate = max(
                    max_rate,
                    abs(
                        float(Decimal(state["learning_rate_decimal"]))
                        - independent["learning_rate_after"]
                    ),
                )
        else:
            expected = np.full(n, 1 / n)
            if audit["period_update"] is not None or not state["learning_rate_is_infinite"]:
                raise QuantError("The original initial partial month cannot train the learner.")
        actual = np.asarray(audit["expert_weights"])
        if actual.shape != (n,) or not np.isfinite(actual).all() or (actual < 0).any():
            raise QuantError("Require six real finite nonnegative capital coordinates.")
        max_weight = max(max_weight, float(max(abs(actual - expected))))
        if (
            audit["previous_expert_weights"] != previous.tolist()
            or state["complete_intervals"] != max(0, i - 1)
            or state["arithmetic_precision"] != rules.PRECISION
            or audit["complete_learning_intervals"] != max(0, i - 1)
            or audit["score_mature"] != (i >= 2)
            or state["cumulative_loss_fractions"]
            != [
                {"numerator": str(v.numerator), "denominator": str(v.denominator)}
                for v in cumulative
            ]
        ):
            raise QuantError(
                "Exact six-source state,maturity,precision or prior fractions changed."
            )
        expected_target = np.array(
            [
                math.fsum(
                    float(o.target[symbol]) * float(w)
                    for o, w in zip(observed, actual, strict=True)
                )
                for symbol in assets
            ]
        )
        max_target = max(max_target, float(max(abs(target.loc[day].to_numpy() - expected_target))))
        changes = [
            abs(
                math.fsum(
                    (float(actual[j]) - float(previous[j])) * float(observed[j].target[symbol])
                    for j in range(n)
                )
            )
            for symbol in assets
        ]
        cost = math.fsum(changes) * 5 + sum(value > 1e-12 for value in changes)
        risk = audit["risk_and_cost"]
        max_cost = max(max_cost, abs(cost - risk["incremental_weight_switch_proxy_at_10000_usd"]))
        if i >= 2:
            known = sessions(next_session(first), day)
            columns = [
                (o.net_returns.loc[known] - observed[0].risk_free.loc[known]).to_numpy()
                for o in observed
            ]
            means = [math.fsum(float(v) for v in column) / len(known) for column in columns]
            covariance = np.array(
                [
                    [
                        math.fsum(
                            (float(a) - means[j]) * (float(b) - means[k])
                            for a, b in zip(columns[j], columns[k], strict=True)
                        )
                        / (len(known) - 1)
                        for k in range(n)
                    ]
                    for j in range(n)
                ]
            )
            recorded_cov = np.asarray(risk["net_excess_covariance"])
            recorded_contributions = np.asarray(risk["marginal_variance_contributions"])
            if (
                recorded_cov.shape != (n, n)
                or not np.isfinite(recorded_cov).all()
                or recorded_contributions.shape != (n,)
                or not np.isfinite(recorded_contributions).all()
            ):
                raise QuantError(
                    "Retain every genuine six-source covariance and risk contribution."
                )
            max_cov = max(max_cov, float(np.max(abs(covariance - recorded_cov))))
            contributions = np.array(
                [
                    float(actual[j])
                    * math.fsum(float(covariance[j, k]) * float(actual[k]) for k in range(n))
                    for j in range(n)
                ]
            )
            max_marginal = max(
                max_marginal,
                float(max(abs(contributions - recorded_contributions))),
                abs(math.fsum(float(v) for v in contributions) - risk["mixture_variance"]),
            )
        previous = actual
    if (
        any(
            value > 1e-12
            for value in (
                max_weight,
                max_target,
                max_gap,
                max_rate,
                max_played,
                max_mix,
                max_cov,
                max_marginal,
            )
        )
        or max_cost > 1e-10
    ):
        raise QuantError("Independent six-source state,rate,risk,cost or target check failed.")
    return {
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_mixability_gap_difference": max_gap,
        "maximum_independent_learning_rate_difference": max_rate,
        "maximum_independent_played_loss_difference": max_played,
        "maximum_independent_mix_loss_difference": max_mix,
        "maximum_independent_covariance_difference": max_cov,
        "maximum_independent_marginal_variance_difference": max_marginal,
        "maximum_independent_switch_proxy_difference_USD": max_cost,
        "complete_mature_learning_intervals": len(updates),
        "real_source_coordinates": n,
    }


def verify_retained_records(path, experts, target, audits):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 2 * len(audits):
        raise QuantError("Retain every before/after real six-source decision.")
    previous_state = rules.AdaHedge().snapshot()
    previous, anchor, count = [1 / rules.N] * rules.N, None, 0
    for i, audit in enumerate(audits):
        before, after = records[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = rules.observations(experts, day)
        logs, labels, dates = None, None, pd.DatetimeIndex([])
        if anchor is not None:
            logs, dates = rules.period_logs(observed, anchor, day)
            labels = rules.bounded_losses(logs)
        expected = {
            "phase": "before_decision",
            "decision_session": str(day.date()),
            "state": previous_state,
            "previous_weights_hex": [float(v).hex() for v in previous],
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
            "matured_period_dates": dates.strftime("%Y-%m-%d").tolist(),
            "matured_NET_returns_hex": [
                [float(v).hex() for v in o.net_returns.loc[dates]] for o in observed
            ],
            "canonical_RF_hex": [float(v).hex() for v in observed[0].risk_free.loc[dates]],
            "canonical_log_relatives_hex": None if logs is None else [float(v).hex() for v in logs],
            "normalized_loss_labels_hex": None
            if labels is None
            else [float(v).hex() for v in labels],
            "candidate_accounts_or_metrics_computed": False,
        }
        if before != expected or after != {
            "phase": "after_decision",
            "decision_session": str(day.date()),
            "audit": audit,
            "unified_target_hex": [float(v).hex() for v in target.loc[day]],
        }:
            raise QuantError("A retained native NET/RF,label,exact six-state or target changed.")
        count += sum(len(o.net_returns.loc[dates]) for o in observed)
        previous_state, previous = audit["adaptive_state"], audit["expert_weights"]
        if i > 0:
            anchor = day
    return {
        "actual_input_and_resolved_record_count": len(records),
        "source_NET_binary64_values_checked": count,
        "new_six_source_loss_binary64_values_checked": sum(
            rules.N for audit in audits if audit["period_update"] is not None
        ),
        "actual_input_resolved_records_match_source_and_targets": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists() or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial actual six-source construction.")
    if any(len(events("registered", identifier)) != 1 for identifier in IDS):
        raise QuantError(
            "Actual registration must precede all new six labels,learning and targets."
        )
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    trace = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        target, audits, updates = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    independent = verify_targets(data, experts, target, audits, updates)
    retained = verify_retained_records(trace, experts, target, audits)
    if len(audits) != 121 or len(updates) != 119:
        raise QuantError("Keep every original decision and complete source month.")
    prefixes = []
    for i in (0, 2, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[i]["decision_session"])
        early, earlier, early_updates = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if (
            not early.equals(target.loc[:day])
            or earlier != audits[: i + 1]
            or early_updates != updates[: audits[i]["complete_learning_intervals"]]
        ):
            raise QuantError("Future sixth-source state changed a prior allocation.")
        prefixes.append(str(day.date()))
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    control = pd.read_csv(
        base.checked_reference(catalogue["cached_five_AdaHedge_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    if not control.index.equals(target.index) or not control.columns.equals(target.columns):
        raise QuantError("Compare only exact original coordinates of cached five-source targets.")
    if target.equals(control):
        raise QuantError("Duplicate full target construction;do not fund a new economic trial.")
    decision_index = pd.DatetimeIndex([a["decision_session"] for a in audits])
    comparison = {
        "matched_cached_five_AdaHedge_id": prior.rules.ID,
        "materially_different_target_origins": int(
            (
                (target.loc[decision_index] - control.loc[decision_index]).abs().max(axis=1) > 1e-12
            ).sum()
        ),
        "old_control_replayed": False,
    }
    active, coverage = target.dropna(how="all"), []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError("Keep known initial opinions and genuine factors in both windows.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(
        FOLDER / "information-audits-v1.json", {"decisions": audits, "periods": updates}
    )
    write_new_receipt(FOLDER / "cached-control-target-comparison-v1.json", comparison)
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", target.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        target_proof(
            frozen,
            {
                **independent,
                **retained,
                "decision_count": len(audits),
                "initial_unscored_decision_count": sum(not a["score_mature"] for a in audits),
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(trace),
                "matched_cached_target_comparison": comparison,
                "new_source_or_shadow_account_runs": 0,
                "old_learner_or_source_or_control_reexecuted": False,
                "source_wealth_guarantee_transferred_to_main": False,
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
    return bound(template.register)()


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    return bound(prior.prior.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        control = read_json(base.checked_reference(catalogue["cached_five_AdaHedge_results"]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every original six-source main path and gate outcome.")
        contrasts = {
            key: {
                "metrics": {
                    metric: path["metrics"][metric]
                    - control["all_four_original_paths"][key]["metrics"][metric]
                    for metric in ("cagr", "sharpe", "max_drawdown")
                },
                "execution": {
                    metric: path["execution"][metric]
                    - control["all_four_original_paths"][key]["execution"][metric]
                    for metric in (
                        "annualized_two_sided_turnover",
                        "model_order_count",
                        "total_cost_usd",
                    )
                },
            }
            for key, path in result["paths"].items()
        }
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidate_id": rules.ID,
                "status": result["review"]["status"],
                "all_four_original_paths": result["paths"],
                "new_minus_cached_five_AdaHedge": contrasts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "actual_complete_source_count": rules.N,
                "global_pool_changed": False,
                "new_expert_admissions": 0,
                "source_wealth_guarantee_transferred_to_main": False,
                "old_learner_or_source_or_control_reexecuted": False,
                "negative_Jensen_gap_clipped": False,
                "mixability_gap_floor": None,
                "rate_or_window_sweep": False,
                "normalized_source_regret_is_main_wealth": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "raw_prices_targets_or_daily_book_rows_exported": False,
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
        parser.error("Use an actual remotely published singleton preregistration SHA.")
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
