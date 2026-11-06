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

from us_quant import quintet_adaptive_hedge_v1 as rules
from us_quant import ssd_quintet_anytime_hedge_v1 as prior
from us_quant.calendar import next_session, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_forecast_opportunity_gate_v1 import target_proof
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
FOLDER = base.REPORTS / "quintet-adaptive-hedge-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/quintet_adaptive_hedge_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/quintet_adaptive_hedge_20261106_registration_v1.json"}
OWN = (
    "config/quintet-adaptive-hedge-v1.json",
    "src/us_quant/quintet_adaptive_hedge_v1.py",
    "src/us_quant/ssd_quintet_adaptive_hedge_v1.py",
    "tests/test_quintet_adaptive_hedge_v1.py",
    "QUINTET_ADAPTIVE_HEDGE_V1.md",
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
    return bound(prior.dependencies)()


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    actual = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in actual.items()):
        raise QuantError("AdaHedge progress must retain actual financial journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quintet_adaptive_hedge_" + phase,
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
    data, experts, _, refs, clocks, benchmarks = prior.source_inputs()
    parent = prior.events("registered", prior.rules.ID)[0]["body"]["frozen"]
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["source_clocks"] != clocks
        or catalogue["academic_reference"] != policy["academic_reference"]
        or catalogue["new_adaptive_learning_weights_targets_or_accounts_computed"] is not False
        or catalogue["old_learner_or_source_or_control_reexecuted"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep the five exact admitted sources and sealed monthly label delivery.")
    previous = read_json(base.checked_reference(catalogue["previous_completed_cycle"]))
    refs = unique_refs(
        [
            *refs,
            *previous["exact_evidence_references"],
            *catalogue["source_references"],
            policy["source_catalogue"],
            policy["academic_reference"],
            *catalogue["cached_control_results"].values(),
            *catalogue["cached_control_targets"].values(),
            *catalogue["prior_net_mixture_policies"],
            *[
                catalogue[k]
                for k in (
                    "cached_monthly_labels",
                    "cached_raw_Hedge_policy",
                    "cached_raw_Hedge_freeze",
                    "previous_completed_cycle",
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
        raise QuantError("Preserve the first complete or partial AdaHedge freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1440 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1440/299 without changing old outcomes.")
        if any(row["candidate_id"] in IDS for row in history):
            raise QuantError("The fixed gap-adaptive singleton identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["candidate_id"] != rules.ID
            or design["new_adaptive_learning_weights_targets_or_accounts_computed"] is not False
            or design["old_learner_or_source_or_control_reexecuted"] is not False
        ):
            raise QuantError("Freeze the complete gap-adaptive law before new learning or targets.")
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
                "Bind infinity,neutral-gap,independent-state and causal-source checks."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "fixed_five_cached_month_loss_AdaHedge_mixability_gap",
            "expert_pool_definition_role": "unchanged_five_complete_sources_gap_adaptive_law",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1440,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 299,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "QUINTET_ADAPTIVE_HEDGE_V1.md"),
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
            "cached_monthly_loss_labels_changed": False,
            "old_learner_or_source_or_control_reexecuted": False,
            "negative_Jensen_gap_clipped": False,
            "mixability_gap_floor": None,
            "rate_or_window_sweep": False,
            "normalized_source_regret_is_main_wealth": False,
            "main_drawdown_guarantee": False,
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
                "new_adaptive_learning_weights_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_actual_gap_learning_targets")
    return reference(INTENT)


def verify_targets(data, experts, cached_periods, target, audits, updates):
    calendar, assets = data.close.index, tuple(data.close.columns)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    if (
        not target.index.equals(calendar)
        or tuple(target.columns) != assets
        or not target.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or len(updates) != max(0, len(dates) - 2)
    ):
        raise QuantError(
            "Every original formal target row and mature monthly decision is required."
        )
    cached = {row["end"]: row for row in cached_periods}
    used = [cached[str(day.date())] for day in dates[2:]]
    independent = rules.independent_trace([row["losses"] for row in used])
    cumulative = [Fraction(0)] * 5
    previous, first = np.full(5, 0.2), pd.Timestamp("2016-10-31")
    max_weight = max_target = max_gap = max_rate = max_played = max_mix = max_cov = max_cost = 0.0
    max_contribution = 0.0
    played_prefix, mixed_prefix = [], []
    for i, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = rules.observations(experts, day)
        state = audit["adaptive_state"]
        if i >= 2:
            period, oracle = used[i - 2], independent[i - 2]
            current = updates[i - 2]
            if (
                audit["period_update"] != current
                or current["end"] != str(day.date())
                or current["start_after"] != str(dates[i - 1].date())
                or current["canonical_log_relatives"] != period["canonical_log_relatives"]
                or current["input_loss_labels"] != period["losses"]
                or current["prequential_capital_weights_before_assimilation"] != previous.tolist()
            ):
                raise QuantError("Prequential weights or original sealed month labels changed.")
            cumulative = [
                old + Fraction.from_float(float(value))
                for old, value in zip(cumulative, period["losses"], strict=True)
            ]
            expected = np.asarray(oracle["weights_after"])
            max_gap = max(
                max_gap,
                abs(float(Decimal(state["cumulative_mixability_gap_decimal"])) - oracle["gap"]),
            )
            max_played = max(
                max_played,
                abs(float(Decimal(current["played_loss_decimal"])) - oracle["played_loss"]),
            )
            max_mix = max(
                max_mix, abs(float(Decimal(current["mix_loss_decimal"])) - oracle["mix_loss"])
            )
            played_prefix.append(oracle["played_loss"])
            mixed_prefix.append(oracle["mix_loss"])
            max_played = max(
                max_played,
                abs(
                    float(Decimal(state["cumulative_played_loss_decimal"]))
                    - math.fsum(played_prefix)
                ),
            )
            max_mix = max(
                max_mix,
                abs(float(Decimal(state["cumulative_mix_loss_decimal"])) - math.fsum(mixed_prefix)),
            )
            if oracle["learning_rate_after"] is None:
                if state["learning_rate_is_infinite"] is not True:
                    raise QuantError(
                        "Zero mixability gap requires the analytic infinite-rate limit."
                    )
            elif state["learning_rate_is_infinite"]:
                raise QuantError("A positive gap cannot be replaced by infinite-rate FTL.")
            else:
                max_rate = max(
                    max_rate,
                    abs(
                        float(Decimal(state["learning_rate_decimal"]))
                        - oracle["learning_rate_after"]
                    ),
                )
        else:
            expected = np.full(5, 0.2)
            if audit["period_update"] is not None or not state["learning_rate_is_infinite"]:
                raise QuantError("The initial partial month cannot train the learner.")
        actual = np.asarray(audit["expert_weights"])
        if actual.shape != (5,) or not np.isfinite(actual).all() or (actual < 0).any():
            raise QuantError("Every decision requires five finite nonnegative capital coordinates.")
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
            raise QuantError("Exact cumulative source labels,maturity or prior allocation changed.")
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
                    for j in range(5)
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
                        for k in range(5)
                    ]
                    for j in range(5)
                ]
            )
            recorded_covariance = np.asarray(risk["net_excess_covariance"])
            recorded_contributions = np.asarray(risk["marginal_variance_contributions"])
            if (
                recorded_covariance.shape != (5, 5)
                or not np.isfinite(recorded_covariance).all()
                or recorded_contributions.shape != (5,)
                or not np.isfinite(recorded_contributions).all()
            ):
                raise QuantError(
                    "Retain all finite source covariance and marginal risk coordinates."
                )
            max_cov = max(max_cov, float(np.max(abs(covariance - recorded_covariance))))
            contributions = np.array(
                [
                    float(actual[j])
                    * math.fsum(float(covariance[j, k]) * float(actual[k]) for k in range(5))
                    for j in range(5)
                ]
            )
            max_contribution = max(
                max_contribution,
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
                max_contribution,
            )
        )
        or max_cost > 1e-10
    ):
        raise QuantError(
            "Independent adaptive rate,gap,loss,covariance,cost or target check failed."
        )
    return {
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_mixability_gap_difference": max_gap,
        "maximum_independent_learning_rate_difference": max_rate,
        "maximum_independent_played_loss_difference": max_played,
        "maximum_independent_mix_loss_difference": max_mix,
        "maximum_independent_covariance_difference": max_cov,
        "maximum_independent_marginal_variance_difference": max_contribution,
        "maximum_independent_switch_proxy_difference_USD": max_cost,
        "complete_mature_learning_intervals": len(updates),
        "all_cached_monthly_loss_labels_reused_unchanged": True,
    }


def verify_retained_records(path, experts, cached_periods, target, audits):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cached = {row["end"]: row for row in cached_periods}
    if len(rows) != 2 * len(audits):
        raise QuantError("Retain every before/after adaptive decision and both unscored origins.")
    previous_state, previous, count = rules.AdaHedge().snapshot(), [0.2] * 5, 0
    anchor = None
    for i, audit in enumerate(audits):
        before, after = rows[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = rules.observations(experts, day)
        dates = sessions(next_session(anchor), day) if anchor is not None else pd.DatetimeIndex([])
        period = cached[str(day.date())] if i >= 2 else None
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
            "cached_loss_labels_hex": None
            if period is None
            else [float(v).hex() for v in period["losses"]],
            "cached_canonical_log_relatives_hex": None
            if period is None
            else [float(v).hex() for v in period["canonical_log_relatives"]],
            "candidate_accounts_or_metrics_computed": False,
        }
        if before != expected or after != {
            "phase": "after_decision",
            "decision_session": str(day.date()),
            "audit": audit,
            "unified_target_hex": [float(v).hex() for v in target.loc[day]],
        }:
            raise QuantError("A retained source,label,exact adaptive state or target changed.")
        count += sum(len(o.net_returns.loc[dates]) for o in observed)
        previous_state, previous = audit["adaptive_state"], audit["expert_weights"]
        if i > 0:
            anchor = day
    return {
        "actual_input_and_resolved_record_count": len(rows),
        "source_NET_binary64_values_checked": count,
        "cached_loss_binary64_values_checked": sum(
            5 for a in audits if a["period_update"] is not None
        ),
        "actual_input_resolved_records_match_source_and_targets": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists() or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial actual gap-adaptive construction.")
    if any(len(events("registered", name)) != 1 for name in IDS):
        raise QuantError(
            "Register the actual singleton before any new adaptive learning or target."
        )
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    cached = read_json(base.checked_reference(catalogue["cached_monthly_labels"]))["periods"]
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        target, audits, updates = rules.build_targets(
            experts,
            cached,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    independent = verify_targets(data, experts, cached, target, audits, updates)
    retained = verify_retained_records(log, experts, cached, target, audits)
    if len(audits) != 121 or len(updates) != 119:
        raise QuantError("Keep all121original decisions and119genuinely complete month labels.")
    prefixes = []
    for i in (0, 2, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[i]["decision_session"])
        early, earlier, early_updates = rules.build_targets(
            experts,
            cached,
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
            raise QuantError(
                "Future cached labels or source observations changed a prior decision."
            )
        prefixes.append(str(day.date()))
    comparisons = {}
    decision_index = pd.DatetimeIndex([a["decision_session"] for a in audits])
    for identifier, ref in catalogue["cached_control_targets"].items():
        control = pd.read_csv(
            base.checked_reference(ref),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        if not control.index.equals(target.index) or not control.columns.equals(target.columns):
            raise QuantError("A cached control requires unchanged full original coordinates.")
        if target.equals(control):
            raise QuantError("Duplicate full targets;do not fund a new economic trial.")
        comparisons[identifier] = {
            "materially_different_target_origins": int(
                (
                    (target.loc[decision_index] - control.loc[decision_index]).abs().max(axis=1)
                    > 1e-12
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
            raise QuantError("Both formal windows need a known seed and real factor participation.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(
        FOLDER / "information-audits-v1.json", {"decisions": audits, "periods": updates}
    )
    write_new_receipt(FOLDER / "cached-control-target-comparison-v1.json", comparisons)
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
                "actual_input_resolved_records": reference(log),
                "matched_cached_target_comparisons": comparisons,
                "new_source_or_shadow_account_runs": 0,
                "old_learner_or_source_or_control_reexecuted": False,
                "normalized_source_regret_is_main_wealth": False,
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
    return bound(prior.register)()


def scoped(function, candidate):
    return bound(prior.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(prior.cohort_publication)(*args)


def evaluate(commit):
    return bound(prior.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every original funded path,not selected passing metrics.")
        contrasts = {}
        for identifier, ref in catalogue["cached_control_results"].items():
            control = read_json(base.checked_reference(ref))["all_four_original_paths"]
            contrasts[identifier] = {
                key: {
                    "metrics": {
                        metric: path["metrics"][metric] - control[key]["metrics"][metric]
                        for metric in ("cagr", "sharpe", "max_drawdown")
                    },
                    "execution": {
                        metric: path["execution"][metric] - control[key]["execution"][metric]
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
                "new_minus_matched_cached_controls": contrasts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "cached_monthly_loss_labels_changed": False,
                "old_learner_or_source_or_control_reexecuted": False,
                "negative_Jensen_gap_clipped": False,
                "mixability_gap_floor": None,
                "rate_or_window_sweep": False,
                "normalized_source_regret_is_main_wealth": False,
                "main_drawdown_guarantee": False,
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
