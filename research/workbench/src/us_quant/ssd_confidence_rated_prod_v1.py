from __future__ import annotations

import argparse
import json
import math
import os
import sys
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import confidence_rated_prod_v1 as rules
from us_quant import ssd_protective_six_adaptive_hedge_v1 as prior
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_forecast_opportunity_gate_v1 import target_proof
from us_quant.ssd_quintet_anytime_hedge_v1 import evaluate as singleton_evaluate
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
FOLDER = base.REPORTS / "confidence-rated-prod-20261106-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/confidence_rated_prod_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/confidence_rated_prod_20261106_registration_v1.json"}
OWN = (
    "config/confidence-rated-prod-v1.json",
    "src/us_quant/confidence_rated_prod_v1.py",
    "src/us_quant/ssd_confidence_rated_prod_v1.py",
    "tests/test_confidence_rated_prod_v1.py",
    "CONFIDENCE_RATED_PROD_V1.md",
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
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Confidence learning cannot substitute stale or fabricated financial totals.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "confidence_rated_prod_" + phase,
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
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, _, refs, clocks, benchmarks = prior.source_inputs()
    rules.require(
        catalogue["expert_ids"] == list(rules.EXPERT_IDS)
        and catalogue["mechanism_groups"] == list(rules.MECHANISMS)
        and catalogue["source_clocks"] == clocks
        and catalogue["new_confidences_learning_targets_or_accounts_computed"] is False
        and catalogue["new_source_or_RF_or_event_Adapter"] is False
        and catalogue["old_learner_or_source_or_control_reexecuted"] is False,
        "Keep all admitted source identities, clocks, originals and cached six-source labels.",
    )
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    rules.require(
        closed["actual_authoritative_counts"]["current_journal_event_count"] == 1478,
        "Use the actual J1478 predecessor, not a stale or reset history.",
    )
    refs = unique_refs(
        [
            *refs,
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                value
                for value in catalogue.values()
                if isinstance(value, dict) and set(value) == {"path", "sha256"}
            ],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                    "academic-reference-v1.json",
                    "academic-acquisition-v1.json",
                    "academic-extracted-pages-v1.json",
                    "academic-http-headers-v1.txt",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    parent = prior.events("registered", prior.rules.ID)[0]["body"]["frozen"]
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial confidence-Prod freeze.",
    )
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1478
            and history[-1]["sha256"] == rules.PRIOR_HEAD
            and not any(e["candidate_id"] in IDS for e in history),
            "Freeze one unused confidence-learning identity at the actual J1478 head.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        suites = (
            ElementTree.parse(base.checked_reference(validation["test_outputs"][0]))
            .getroot()
            .findall(".//testsuite")
        )
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and validation["real_packet_entry_contract_exercised"] is True
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and suites
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful confidence chronology, modified-loss oracle and packet checks.",
        )
        names = {
            case.get("name", "").split("[", 1)[0]
            for suite in suites
            for case in suite.findall("testcase")
        }
        rules.require(
            {
                "test_canonical_packet_entry_accepts_new_proof_interface",
                "test_prior_confidence_not_end_month_confidence_weights_past_loss",
                "test_all119_updates_match_independent_modified_loss_reduction",
            }.issubset(names),
            "The named real packet, prequential clock and independent reduction checks must run.",
        )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "confidence_rated_second_order_excess_loss_six_complete_policies",
            "expert_pool_definition_role": (
                "unchanged_six_conditional_learning_not_source_admission"
            ),
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1478,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 302,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "confidence_is_probability_or_technical_source_eligibility": False,
            "confidence_for_loss_fixed_at_prior_month_origin": True,
            "cached_monthly_loss_labels_changed": False,
            "source_wealth_guarantee_transferred_to_main": False,
            "normalized_confidence_regret_is_main_wealth": False,
            "rate_or_window_or_budget_floor_sweep": False,
            "old_learner_or_source_or_control_reexecuted": False,
            "global_pool_changed": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "new_source_RF_views": 0,
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
                "python_version": sys.version,
                "new_confidences_learning_targets_or_accounts_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_confidence_learning")
    return reference(INTENT)


def confidence_archive(experts):
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    frame = pd.read_csv(
        base.checked_reference(catalogue["feedback_source_own_shadow"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    observed = experts[-1].observe(frame.index[-1], 2512)
    rules.require(
        len(frame) == 2512
        and frame["return"].equals(observed.net_returns)
        and str(frame.index[0].date()) == "2016-10-06"
        and str(frame.index[-1].date()) == "2026-10-05",
        "The confidence source must be its original complete own-shadow NET book.",
    )
    return rules.OwnShadowConfidenceArchive(frame["equity"])


def verify_targets(data, experts, archive, target, audits, updates):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    rules.require(
        len(audits) == 121
        and len(updates) == 119
        and target.index.equals(data.close.index)
        and target.columns.equals(data.close.columns)
        and target.dropna(how="all").index.equals(dates)
        and [row["decision_session"] for row in audits] == dates.strftime("%Y-%m-%d").tolist(),
        "Keep121decisions,119complete months and every original formal target coordinate.",
    )
    oracle = rules.independent_trace(
        [(u["input_loss_labels"], u["confidence_held_from_prior_origin"]) for u in updates]
    )
    max_weight = max_target = max_state = max_played = max_reduction = 0.0
    for i, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        wealth = archive.observe(day)
        confidence = rules.confidences(wealth)
        rules.require(
            wealth == audit["confidence_wealth_input"]
            and confidence.tolist() == audit["current_confidences"],
            "Every confidence must use its actual then-known own-shadow prefix.",
        )
        if i < 2:
            masses, eta = np.full(6, 1 / 6), np.full(6, 0.5)
        else:
            original, independent = updates[i - 2], oracle[i - 2]
            state = audit["learner_state"]
            rules.require(
                original["confidence_held_from_prior_origin"]
                == audits[i - 1]["current_confidences"],
                "A new confidence cannot retroactively weight the completed source month.",
            )
            for key, other in (
                ("unnormalized_mass_decimal", "mass_after_decimal"),
                ("squared_confidence_excess_decimal", "squares_after_decimal"),
                ("confidence_regret_decimal", "regrets_after_decimal"),
            ):
                max_state = max(
                    max_state,
                    *[
                        abs(float(Decimal(a) - Decimal(b)))
                        for a, b in zip(state[key], independent[other], strict=True)
                    ],
                )
            masses = np.array([float(v) for v in independent["mass_after_decimal"]])
            eta = np.array(independent["rates_after"])
            max_played = max(
                max_played,
                abs(float(Decimal(original["played_loss_decimal"])) - independent["played_loss"]),
            )
            max_reduction = max(max_reduction, independent["confidence_reduction_error"])
        mass = masses * eta * confidence
        expected = mass / math.fsum(mass)
        actual = np.asarray(audit["expert_weights"])
        max_weight = max(max_weight, float(abs(actual - expected).max()))
        observed = rules.observations(experts, day)
        scalar = [
            float(
                sum(
                    Fraction.from_float(float(w)) * Fraction.from_float(float(o.target.iloc[j]))
                    for w, o in zip(actual, observed, strict=True)
                )
            )
            for j in range(len(data.close.columns))
        ]
        max_target = max(max_target, float(np.max(np.abs(target.loc[day] - scalar))))
    rules.require(
        max(max_weight, max_target, max_state, max_played, max_reduction) <= 1e-12,
        "Independent modified-loss, confidence, state or rational target verification failed.",
    )
    return {
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_decimal_state_difference": max_state,
        "maximum_independent_played_loss_difference": max_played,
        "maximum_generic_confidence_reduction_error": max_reduction,
        "complete_mature_learning_intervals": len(updates),
        "source_confidence_origin_checked_per_learning_interval": True,
        "source_game_reference_is_not_main_wealth": True,
    }


def prepare(data, experts, frozen):
    rules.require(
        len(events("registered", rules.ID)) == 1
        and not PREPARED.exists()
        and not (FOLDER / "preparation-attempt-v1.json").exists(),
        "Register before the first complete or partial new confidence-state construction.",
    )
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    periods = read_json(base.checked_reference(catalogue["cached_six_source_month_audits"]))[
        "periods"
    ]
    archive = confidence_archive(experts)
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
            periods,
            archive,
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    checked = verify_targets(data, experts, archive, target, audits, updates)
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    rules.require(len(records) == 242, "Retain every before/after confidence decision.")
    previous_state, previous_confidence = rules.ConfidenceProd().snapshot(), None
    for i, audit in enumerate(audits):
        before, after = records[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = rules.observations(experts, day)
        period_dates = pd.DatetimeIndex(before["matured_period_dates"])
        update = audit["period_update"]
        expected_dates = (
            pd.DatetimeIndex([])
            if update is None
            else rules.sessions(
                rules.next_session(pd.Timestamp(audits[i - 1]["decision_session"])), day
            )
        )
        rules.require(
            before["phase"] == "before_decision"
            and before["state"] == previous_state
            and before["decision_session"] == audit["decision_session"]
            and period_dates.equals(expected_dates)
            and before["confidence_wealth_input"] == archive.observe(day)
            and before["previous_confidences_hex"] == previous_confidence
            and before["known_target_dates"] == [str(o.target_known_at.date()) for o in observed]
            and before["known_targets_hex"]
            == [[float(v).hex() for v in o.target] for o in observed]
            and before["matured_NET_returns_hex"]
            == [[float(v).hex() for v in o.net_returns.loc[period_dates]] for o in observed]
            and before["canonical_RF_hex"]
            == [float(v).hex() for v in observed[0].risk_free.loc[period_dates]]
            and before["cached_loss_labels_hex"]
            == (None if update is None else [float(v).hex() for v in update["input_loss_labels"]])
            and after
            == {
                "phase": "after_decision",
                "decision_session": audit["decision_session"],
                "audit": audit,
                "unified_target_hex": [float(v).hex() for v in target.loc[day]],
            },
            "A retained primitive, confidence origin, learner state or resolved target changed.",
        )
        previous_state = audit["learner_state"]
        previous_confidence = [float(v).hex() for v in audit["current_confidences"]]
    prefixes = []
    for index in (0, 2, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[index]["decision_session"])
        early, earlier, early_updates = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            periods,
            archive,
            before_decision=lifecycle.runtime_guard,
        )
        rules.require(
            early.equals(target.loc[:day])
            and earlier == audits[: index + 1]
            and early_updates == updates[: audits[index]["complete_learning_intervals"]],
            "Future source wealth or labels changed a prior confidence-weighted target.",
        )
        prefixes.append(str(day.date()))
    control = pd.read_csv(
        base.checked_reference(catalogue["cached_flat_six_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        control.index.equals(target.index)
        and control.columns.equals(target.columns)
        and not control.equals(target),
        "Duplicate economic targets cannot receive new funded paths.",
    )
    decision_index = pd.DatetimeIndex([a["decision_session"] for a in audits])
    changed = int(
        ((target.loc[decision_index] - control.loc[decision_index]).abs().max(axis=1) > 1e-12).sum()
    )
    rules.require(changed > 0, "Require a material nonduplicate source target.")
    active, coverage = target.dropna(how="all"), []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        rows = pd.concat(
            [active.loc[:anchor].tail(1), active.loc[anchor + pd.Timedelta(days=1) : window["end"]]]
        )
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        rules.require(len(factors) >= 3, "Preserve genuine factor participation in both windows.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(
        FOLDER / "information-audits-v1.json", {"decisions": audits, "periods": updates}
    )
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", target.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        target_proof(
            frozen,
            {
                **checked,
                "decision_count": 121,
                "complete_mature_learning_intervals": 119,
                "initial_unscored_decision_count": 2,
                "actual_input_and_resolved_record_count": 242,
                "actual_input_resolved_records": reference(trace),
                "source_NET_binary64_values_checked": sum(
                    len(row["matured_period_dates"]) * 6 for row in records[::2]
                ),
                "sealed_source_loss_values_reused": 714,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "materially_different_cached_flat_six_origins": changed,
                "old_learner_or_source_or_control_reexecuted": False,
                "source_wealth_guarantee_transferred_to_main": False,
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
    return bound(singleton_evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
        rules.require(
            set(result["paths"]) == base.PATH_KEYS, "Publish every original financial path."
        )
        contexts = {}
        for name, key in (
            ("flat_six_AdaHedge", "cached_flat_six_results"),
            ("equal_role_V2", "cached_equal_role_results"),
        ):
            old = read_json(base.checked_reference(catalogue[key]))
            contexts[name] = {
                path_key: {
                    "metrics": {
                        metric: row["metrics"][metric]
                        - old["all_four_original_paths"][path_key]["metrics"][metric]
                        for metric in ("cagr", "sharpe", "max_drawdown")
                    },
                    "execution": {
                        metric: row["execution"][metric]
                        - old["all_four_original_paths"][path_key]["execution"][metric]
                        for metric in (
                            "annualized_two_sided_turnover",
                            "model_order_count",
                            "total_cost_usd",
                        )
                    },
                }
                for path_key, row in result["paths"].items()
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
                "new_minus_cached_whole_policy_context": contexts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "actual_complete_source_count": 6,
                "confidence_is_probability_or_source_admission": False,
                "confidence_for_loss_fixed_at_prior_month_origin": True,
                "raw_masses_renormalized_after_update": False,
                "source_wealth_guarantee_transferred_to_main": False,
                "normalized_confidence_regret_is_main_wealth": False,
                "old_learner_or_source_or_control_reexecuted": False,
                "rate_or_window_or_budget_floor_sweep": False,
                "global_pool_changed": False,
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
        parser.error("Use the actual remotely published singleton preregistration SHA.")
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
