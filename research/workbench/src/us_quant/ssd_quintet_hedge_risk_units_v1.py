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

from us_quant import quintet_hedge_risk_units_v1 as rules
from us_quant import ssd_quintet_anytime_hedge_v1 as prior
from us_quant.calendar import previous_session
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
FOLDER = base.REPORTS / "quintet-hedge-risk-units-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/quintet_hedge_risk_units_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/quintet_hedge_risk_units_20261106_registration_v1.json"}
OWN = (
    "config/quintet-hedge-risk-units-v1.json",
    "src/us_quant/quintet_hedge_risk_units_v1.py",
    "src/us_quant/ssd_quintet_hedge_risk_units_v1.py",
    "tests/test_quintet_hedge_risk_units_v1.py",
    "QUINTET_HEDGE_RISK_UNITS_V1.md",
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
        raise QuantError("Risk-unit progress must retain actual financial journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "quintet_hedge_risk_units_" + phase,
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
        or catalogue["new_sizing_weights_targets_or_accounts_computed"] is not False
        or catalogue["cached_learning_state_or_source_generator_reexecuted"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep the exact cached beliefs, source identities and admitted RF views.")
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
                    "cached_belief_audits",
                    "cached_raw_Hedge_policy",
                    "cached_raw_Hedge_freeze",
                    "cached_raw_Hedge_results",
                    "cached_raw_Hedge_targets",
                    "previous_completed_cycle",
                    "existing_generic_covariance_code",
                    "old_equal_budget_covariance_policy",
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
        raise QuantError("Preserve the first complete or partial risk-unit sizing freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1428 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1428/298 without changing old outcomes.")
        if any(row["candidate_id"] in IDS for row in history):
            raise QuantError("The predeclared risk-sizing singleton identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["candidate_id"] != rules.ID
            or design["new_sizing_weights_targets_or_accounts_computed"] is not False
            or design["cached_learner_state_119_months_reexecuted"] is not False
        ):
            raise QuantError("Freeze the sizing Seam before actual capital fractions or targets.")
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
            raise QuantError("Bind cached-belief, risk-unit, exact source and prefix checks.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "cached_five_Hedge_beliefs_in_realized_source_volatility_units",
            "expert_pool_definition_role": "unchanged_five_sources_cached_belief_risk_sizing",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1428,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 298,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "QUINTET_HEDGE_RISK_UNITS_V1.md"),
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
            "cached_learning_state_reexecuted": False,
            "learning_rate_or_history_changed": False,
            "covariance_regularization_or_variance_floor": False,
            "full_covariance_ERC_or_global_solver_invoked": False,
            "main_drawdown_guarantee": False,
            "normalized_source_regret_transfers_to_sized_capital": False,
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
                "new_sizing_weights_targets_or_accounts_computed": False,
                "cached_learning_state_reexecuted": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_actual_sizing_targets")
    return reference(INTENT)


def verify_targets(data, experts, cached_decisions, target, audits):
    calendar, assets = data.close.index, tuple(data.close.columns)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    cached = {row["decision_session"]: row for row in cached_decisions}
    if (
        not target.index.equals(calendar)
        or tuple(target.columns) != assets
        or not target.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Every original source decision and formal target row is required.")
    previous = np.full(5, 0.2)
    max_cov = max_weight = max_target = max_cost = max_contribution = max_std = 0.0
    available = 0
    for day, audit in zip(dates, audits, strict=True):
        observed = rules.observations(experts, day)
        belief = np.asarray(cached[str(day.date())]["expert_weights"])
        if (
            audit["cached_beliefs"] != belief.tolist()
            or audit["previous_capital_weights"] != previous.tolist()
        ):
            raise QuantError("The original belief fractions or previous new capital state changed.")
        expected = belief.copy()
        risk = audit["risk_sizing"]
        if len(observed[0].net_returns) == 126:
            raw = [(o.net_returns - observed[0].risk_free).to_numpy() for o in observed]
            shifted = [[float(v) - float(column[0]) for v in column] for column in raw]
            means = [math.fsum(column) / 126 for column in shifted]
            covariance = np.array(
                [
                    [
                        math.fsum(
                            (a - means[i]) * (b - means[j])
                            for a, b in zip(shifted[i], shifted[j], strict=True)
                        )
                        / 125
                        for j in range(5)
                    ]
                    for i in range(5)
                ]
            )
            recorded_covariance = np.asarray(risk["covariance"])
            if recorded_covariance.shape != (5, 5) or not np.isfinite(recorded_covariance).all():
                raise QuantError(
                    "Retain every finite covariance coordinate,not only a risk status."
                )
            max_cov = max(max_cov, float(np.max(abs(covariance - recorded_covariance))))
            if (np.diag(covariance) > 0).all():
                deviation = np.array([math.sqrt(float(covariance[i, i])) for i in range(5)])
                recorded_std = np.asarray(risk["source_daily_NET_RF_std"])
                if recorded_std.shape != (5,) or not np.isfinite(recorded_std).all():
                    raise QuantError("Retain five finite independently checked source risk units.")
                max_std = max(max_std, float(max(abs(deviation - recorded_std))))
                mass = [float(belief[i]) / float(deviation[i]) for i in range(5)]
                total = math.fsum(mass)
                expected = np.array([value / total for value in mass])
                if risk["risk_sizing_available"] is not True:
                    raise QuantError("A valid mature risk-unit allocation was falsely unscored.")
                available += 1
                actual = np.asarray(audit["capital_weights"])
                for fractions, field in (
                    (actual, "capital_marginal_variance_contributions"),
                    (belief, "raw_belief_marginal_variance_contributions"),
                ):
                    contribution = np.array(
                        [
                            float(fractions[i])
                            * math.fsum(
                                float(covariance[i, j]) * float(fractions[j]) for j in range(5)
                            )
                            for i in range(5)
                        ]
                    )
                    recorded = np.asarray(risk[field])
                    if recorded.shape != (5,) or not np.isfinite(recorded).all():
                        raise QuantError(
                            "Retain every finite raw and sized marginal risk coordinate."
                        )
                    max_contribution = max(
                        max_contribution, float(max(abs(contribution - recorded)))
                    )
            elif risk["risk_sizing_available"] is not False:
                raise QuantError(
                    "Nonpositive individual risk cannot be scored as successful sizing."
                )
        elif risk["risk_sizing_available"] is not False or risk["covariance"] is not None:
            raise QuantError("Incomplete history cannot be a successful126row risk fit.")
        actual = np.asarray(audit["capital_weights"])
        if (
            actual.shape != (5,)
            or not np.isfinite(actual).all()
            or (actual < 0).any()
            or abs(math.fsum(float(v) for v in actual) - 1) > 1e-12
        ):
            raise QuantError(
                "Every capital allocation requires five finite nonnegative coordinates."
            )
        max_weight = max(max_weight, float(max(abs(actual - expected))))
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
                    (float(actual[i]) - float(previous[i])) * float(observed[i].target[symbol])
                    for i in range(5)
                )
            )
            for symbol in assets
        ]
        cost = math.fsum(changes) * 5 + sum(value > 1e-12 for value in changes)
        max_cost = max(max_cost, abs(cost - audit["incremental_switch_proxy_USD"]))
        previous = actual
    if (
        max_cov > 1e-12
        or max_weight > 1e-12
        or max_target > 1e-12
        or max_contribution > 1e-12
        or max_std > 1e-12
        or max_cost > 1e-10
    ):
        raise QuantError(
            "Independent source risk, sizing, contribution, cost or target check failed."
        )
    return {
        "maximum_independent_covariance_difference": max_cov,
        "maximum_independent_source_std_difference": max_std,
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_marginal_variance_difference": max_contribution,
        "maximum_independent_switch_proxy_difference_USD": max_cost,
        "actual_scored_risk_sizing_decisions": available,
        "actual_unscored_passthrough_decisions": len(audits) - available,
        "all_cached_belief_values_reused_unchanged": True,
        "cached_learning_state_reexecuted": False,
    }


def verify_retained_records(path, experts, target, audits):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 2 * len(audits):
        raise QuantError("Retain every before/after source-risk sizing decision.")
    previous, count = [0.2] * 5, 0
    for i, audit in enumerate(audits):
        before, after = rows[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = rules.observations(experts, day)
        expected = {
            "phase": "before_decision",
            "decision_session": str(day.date()),
            "cached_beliefs_hex": [float(v).hex() for v in audit["cached_beliefs"]],
            "known_target_dates": [str(o.target_known_at.date()) for o in observed],
            "known_targets_hex": [[float(v).hex() for v in o.target] for o in observed],
            "source_NET_returns_hex": [[float(v).hex() for v in o.net_returns] for o in observed],
            "source_RF_hex": [float(v).hex() for v in observed[0].risk_free],
            "previous_capital_weights_hex": [float(v).hex() for v in previous],
            "candidate_accounts_or_metrics_computed": False,
        }
        if before != expected or after != {
            "phase": "after_decision",
            "decision_session": str(day.date()),
            "audit": audit,
            "unified_target_hex": [float(v).hex() for v in target.loc[day]],
        }:
            raise QuantError(
                "A retained source, belief, capital state or underlying target changed."
            )
        count += sum(len(o.net_returns) for o in observed)
        previous = audit["capital_weights"]
    return {
        "actual_input_and_resolved_record_count": len(rows),
        "source_NET_binary64_values_checked": count,
        "actual_input_resolved_records_match_source_and_targets": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists() or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial risk-unit target construction.")
    if any(len(events("registered", name)) != 1 for name in IDS):
        raise QuantError("Register the actual singleton before any new sizing or target.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    cached = read_json(base.checked_reference(catalogue["cached_belief_audits"]))["decisions"]
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        target, audits = rules.build_targets(
            experts,
            cached,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    independent = verify_targets(data, experts, cached, target, audits)
    retained = verify_retained_records(log, experts, target, audits)
    if len(audits) != 121:
        raise QuantError("Keep every original initial and genuine month-end decision.")
    prefixes = []
    for i in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[i]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            cached,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early.equals(target.loc[:day]) or earlier != audits[: i + 1]:
            raise QuantError(
                "Future source observations or cached belief suffix changed the prefix."
            )
        prefixes.append(str(day.date()))
    control = pd.read_csv(
        base.checked_reference(catalogue["cached_raw_Hedge_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    if not control.index.equals(target.index) or not control.columns.equals(target.columns):
        raise QuantError("The cached raw-Hedge control requires identical full coordinates.")
    if target.equals(control):
        raise QuantError("Duplicate full target construction;do not fund a new trial.")
    decision_index = pd.DatetimeIndex([a["decision_session"] for a in audits])
    comparison = {
        "matched_raw_Hedge_id": prior.rules.ID,
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
            raise QuantError(
                "Both formal horizons need a known seed and genuine factor participation."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
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
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "matched_cached_target_comparison": comparison,
                "new_source_or_shadow_account_runs": 0,
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
        control = read_json(base.checked_reference(catalogue["cached_raw_Hedge_results"]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every original funded path of the singleton sizing law.")
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
                "new_minus_matched_cached_raw_Hedge": contrasts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "cached_learning_state_reexecuted": False,
                "learning_rate_or_history_changed": False,
                "full_covariance_ERC_or_global_solver_invoked": False,
                "main_drawdown_guarantee": False,
                "normalized_source_regret_transfers_to_sized_capital": False,
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
