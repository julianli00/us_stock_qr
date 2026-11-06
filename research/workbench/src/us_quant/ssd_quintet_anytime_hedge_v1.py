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

from us_quant import quintet_anytime_hedge_v1 as rules
from us_quant import ssd_benchmark_opportunity_quintet_v1 as prior
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
FOLDER = base.REPORTS / "quintet-anytime-hedge-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/quintet_anytime_hedge_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/quintet_anytime_hedge_20261106_registration_v1.json"}
OWN = (
    "config/quintet-anytime-hedge-v1.json",
    "src/us_quant/quintet_anytime_hedge_v1.py",
    "src/us_quant/ssd_quintet_anytime_hedge_v1.py",
    "tests/test_quintet_anytime_hedge_v1.py",
    "QUINTET_ANYTIME_HEDGE_V1.md",
    "tests/test_growth_macro_defense_hedge_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
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
    if any(k in fields and fields[k] != v for k, v in actual.items()):
        raise QuantError("Hedge progress must retain actual financial journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quintet_anytime_hedge_" + phase,
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
    parent = prior.events("registered", prior.IDS[0])[0]["body"]["frozen"]
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["source_clocks"] != clocks
        or catalogue["new_learner_losses_weights_targets_or_accounts_computed"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError(
            "Keep the exact current five complete sources and original assessment views."
        )
    previous = read_json(base.checked_reference(catalogue["previous_completed_cycle"]))
    refs = unique_refs(
        [
            *refs,
            *previous["exact_evidence_references"],
            *catalogue["source_references"],
            *catalogue["fixed_pool_prepared_target_refs"].values(),
            policy["source_catalogue"],
            policy["academic_reference"],
            *[
                catalogue[k]
                for k in (
                    "fixed_pool_policy",
                    "fixed_pool_implementation_freeze",
                    "fixed_pool_completed_results",
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
        raise QuantError("Preserve the first complete or partial anytime-Hedge freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1416 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1416/297 without changing history.")
        if any(row["candidate_id"] in IDS for row in history):
            raise QuantError("The predeclared singleton identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["candidate_id"] != rules.ID
            or design["new_learner_losses_weights_targets_or_accounts_computed"] is not False
        ):
            raise QuantError(
                "Freeze the complete monthly learning law before new weights or targets."
            )
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
                "Bind bounded-loss, causal-month, independent-weight and fence checks."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "fixed_five_mature_month_bounded_loss_decreasing_hedge",
            "expert_pool_definition_role": "unchanged_five_complete_sources_single_online_law",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1416,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 297,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "QUINTET_ANYTIME_HEDGE_V1.md"),
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
            "normalized_source_regret_is_main_wealth": False,
            "risk_or_wealth_guarantee_transferred": False,
            "switch_proxy_is_allocation_penalty_or_total_actual_cost": False,
            "cached_equal_or_rolling_controls_replayed": False,
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
                "new_learner_losses_weights_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_actual_learner_targets")
    return reference(INTENT)


def verify_targets(data, experts, targets, audits, periods):
    calendar, assets = data.close.index, tuple(data.close.columns)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or not targets[rules.ID].index.equals(calendar)
        or tuple(targets[rules.ID].columns) != assets
        or not targets[rules.ID].dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every original full-calendar target and monthly decision.")
    history, previous, anchor = [], np.full(5, 0.2), None
    max_weight = max_target = max_log = max_cost = max_covariance = 0.0
    period_index = 0
    for day, audit in zip(dates, audits, strict=True):
        observed = rules.observations(experts, day)
        if anchor is not None:
            logs, days = rules.period_logs(observed, anchor, day)
            independent_logs = [
                float(np.log1p(o.net_returns.loc[days].to_numpy()).sum()) for o in observed
            ]
            max_log = max(max_log, max(abs(logs - independent_logs)))
            history.append(logs.tolist())
            period = periods[period_index]
            if (
                period != audit["period_update"]
                or period["canonical_log_relatives"] != logs.tolist()
                or period["prequential_weights_before_assimilation"] != previous.tolist()
                or period["start_after"] != str(anchor.date())
                or period["end"] != str(day.date())
            ):
                raise QuantError(
                    "Validate old fractions before assimilating the newly mature month."
                )
            period_index += 1
        elif audit["period_update"] is not None:
            raise QuantError("The original initial partial month cannot become a learning label.")
        expected = rules.independent_weights(history)
        actual = np.asarray(audit["expert_weights"])
        if actual.shape != (5,) or not np.isfinite(actual).all() or (actual <= 0).any():
            raise QuantError("Every decision requires five complete positive fractions.")
        max_weight = max(max_weight, float(max(abs(actual - expected))))
        if (
            audit["complete_learning_intervals"] != len(history)
            or audit["learning_round"] != len(history) + 1
            or audit["previous_expert_weights"] != previous.tolist()
            or audit["score_mature"] != (len(history) > 0)
        ):
            raise QuantError("Month maturity, original initial weights or learner clock changed.")
        expected_target = np.array(
            [
                math.fsum(
                    float(o.target[symbol]) * float(w)
                    for o, w in zip(observed, actual, strict=True)
                )
                for symbol in assets
            ]
        )
        max_target = max(
            max_target, float(max(abs(targets[rules.ID].loc[day].to_numpy() - expected_target)))
        )
        change = [
            abs(
                math.fsum(
                    (float(actual[i]) - float(previous[i])) * float(observed[i].target[symbol])
                    for i in range(5)
                )
            )
            for symbol in assets
        ]
        cost = math.fsum(change) * 5 + sum(value > 1e-12 for value in change)
        risk = audit["risk_and_cost"]
        max_cost = max(max_cost, abs(cost - risk["incremental_weight_switch_proxy_at_10000_usd"]))
        if risk["risk_status"] not in ("no_complete_learning_interval",):
            first = pd.Timestamp(read_json(rules.POLICY)["first_complete_month_anchor"])
            known = sessions(next_session(first), day)
            values = [
                (o.net_returns.loc[known] - observed[0].risk_free.loc[known]).to_numpy()
                for o in observed
            ]
            means = [math.fsum(float(v) for v in column) / len(known) for column in values]
            covariance = np.array(
                [
                    [
                        math.fsum(
                            (float(a) - means[i]) * (float(b) - means[j])
                            for a, b in zip(values[i], values[j], strict=True)
                        )
                        / (len(known) - 1)
                        for j in range(5)
                    ]
                    for i in range(5)
                ]
            )
            max_covariance = max(
                max_covariance,
                float(np.max(abs(covariance - np.asarray(risk["net_excess_covariance"])))),
            )
        if day > pd.Timestamp(read_json(rules.POLICY)["initial_decision_session"]):
            anchor = day
        previous = actual
    if (
        period_index != len(periods)
        or max_weight > 1e-12
        or max_target > 1e-12
        or max_log > 1e-12
        or max_cost > 1e-10
        or max_covariance > 1e-12
    ):
        raise QuantError(
            "Independent learning probability, source-log, risk, cost or target check failed."
        )
    return {
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_period_log_difference": max_log,
        "maximum_independent_switch_proxy_difference_USD": max_cost,
        "maximum_independent_covariance_difference": max_covariance,
        "complete_mature_learning_intervals": len(periods),
    }


def verify_retained_records(path, experts, targets, audits):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 2 * len(audits):
        raise QuantError("Retain every before/after monthly decision,including unscored origins.")
    anchor, count = None, 0
    previous_losses, previous_weights = [0.0] * 5, [0.2] * 5
    for i, audit in enumerate(audits):
        before, after = rows[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = rules.observations(experts, day)
        period_dates = (
            sessions(next_session(anchor), day) if anchor is not None else pd.DatetimeIndex([])
        )
        expected = {
            "phase": "before_decision",
            "decision_session": str(day.date()),
            "mature_periods_before": max(0, i - 2),
            "cumulative_losses_hex": [float(v).hex() for v in previous_losses],
            "previous_weights_hex": [float(v).hex() for v in previous_weights],
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
            "matured_period_dates": period_dates.strftime("%Y-%m-%d").tolist(),
            "matured_NET_returns_hex": [
                [float(v).hex() for v in o.net_returns.loc[period_dates]] for o in observed
            ],
            "canonical_period_logs_hex": None
            if anchor is None
            else [float(v).hex() for v in rules.period_logs(observed, anchor, day)[0]],
            "candidate_accounts_or_metrics_computed": False,
        }
        if before != expected or after != {
            "phase": "after_decision",
            "decision_session": str(day.date()),
            "audit": audit,
            "unified_target_hex": [float(v).hex() for v in targets[rules.ID].loc[day]],
        }:
            raise QuantError(
                "Retained raw binary64 input/state or resolved monthly target changed."
            )
        count += sum(len(o.net_returns.loc[period_dates]) for o in observed)
        previous_losses, previous_weights = (
            audit["cumulative_expert_losses"],
            audit["expert_weights"],
        )
        if i > 0:
            anchor = day
    return {
        "actual_input_and_resolved_record_count": len(rows),
        "roundtrip_source_NET_binary64_values_checked": count,
        "actual_input_resolved_records_match_source_and_targets": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists() or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial actual Hedge target construction.")
    if any(len(events("registered", name)) != 1 for name in IDS):
        raise QuantError("The actual singleton registration must precede all new learner weights.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        target, audits, periods = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    targets = {rules.ID: target}
    independent = verify_targets(data, experts, targets, audits, periods)
    retained = verify_retained_records(log, experts, targets, audits)
    if len(audits) != 121 or len(periods) != 119:
        raise QuantError("Preserve all121decisions and exactly119complete source months.")
    prefixes = []
    for i in (0, 2, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[i]["decision_session"])
        early, earlier, early_periods = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if (
            not early.equals(target.loc[:day])
            or earlier != audits[: i + 1]
            or early_periods != periods[: audits[i]["complete_learning_intervals"]]
        ):
            raise QuantError("Future source observations changed a prior learned allocation.")
        prefixes.append(str(day.date()))
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    comparisons = {}
    for name, ref in catalogue["fixed_pool_prepared_target_refs"].items():
        cached = pd.read_csv(
            base.checked_reference(ref),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        if not cached.index.equals(target.index) or not cached.columns.equals(target.columns):
            raise QuantError("A matched cached control requires unchanged full coordinates.")
        if target.equals(cached):
            raise QuantError("Duplicate full target construction;do not fund a new economic trial.")
        decisions = pd.DatetimeIndex([a["decision_session"] for a in audits])
        comparisons[name] = {
            "materially_different_target_origins": int(
                ((target.loc[decisions] - cached.loc[decisions]).abs().max(axis=1) > 1e-12).sum()
            ),
            "old_control_replayed": False,
        }
    active = target.dropna(how="all")
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError(
                "Keep known initial opinions and genuine factor participation in both windows."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(
        FOLDER / "information-audits-v1.json", {"decisions": audits, "periods": periods}
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
                "normalized_loss_or_regret_is_actual_main_performance": False,
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
    lifecycle.runtime_guard()
    if RESULTS.exists():
        raise QuantError("The complete singleton's accounts must not be replayed.")
    refs = {
        candidate["id"]: scoped(financial.evaluate, candidate)(commit) for candidate in CANDIDATES
    }
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": refs,
            "all_declared_candidates_and_four_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "source_models_or_cached_controls_replayed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress(
        "all_four_paths_complete", next_step="publish_complete_outcome_without_replaying_sources"
    )
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        controls = read_json(base.checked_reference(catalogue["fixed_pool_completed_results"]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish all original paths of the frozen singleton law.")
        contrasts = {}
        for control in prior.IDS:
            contrasts[control] = {
                key: {
                    "metrics": {
                        m: path["metrics"][m]
                        - controls["candidates"][control]["paths"][key]["metrics"][m]
                        for m in ("cagr", "sharpe", "max_drawdown")
                    },
                    "execution": {
                        m: path["execution"][m]
                        - controls["candidates"][control]["paths"][key]["execution"][m]
                        for m in (
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
                "fixed_complete_expert_count": 5,
                "normalized_source_regret_is_main_wealth": False,
                "source_models_or_cached_controls_replayed": False,
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
