from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import commitment_macro_defense_hedge_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
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

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
lifecycle, reference, unique_refs = template.lifecycle, template.reference, template.unique_refs
source = seven
FOLDER = base.REPORTS / "commitment-macro-defense-hedge-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/commitment_macro_defense_hedge_20261106_results_v1.json"
CANDIDATES = rules.CANDIDATES
IDS = tuple(c["id"] for c in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"commitment_macro_defense_hedge_20261106_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/commitment-macro-defense-hedge-v1.json",
    "src/us_quant/commitment_macro_defense_hedge_v1.py",
    "src/us_quant/ssd_commitment_macro_defense_hedge_v1.py",
    "tests/test_commitment_macro_defense_hedge_v1.py",
    "COMMITMENT_MACRO_DEFENSE_HEDGE_V1.md",
    "tests/test_growth_macro_defense_hedge_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/conftest.py",
)


def bound(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals(), **overrides},
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
        raise QuantError("Keep actual authoritative counts rather than projected cohort totals.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "commitment_macro_defense_hedge_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def select_sources(experts):
    if tuple(e.identity.identifier for e in experts) != template.rules.EXPERT_IDS:
        raise QuantError("Select from the exact complete admitted original eight.")
    return tuple(experts[i] for i in rules.SLOTS)


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_catalogue"]))
    data, original, parent, refs, clocks, benchmarks = template.source_inputs()
    selected = select_sources(original)
    selected_clocks = [clocks[i] for i in rules.SLOTS]
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["selected_source_slots"] != list(rules.SLOTS)
        or catalogue["source_clocks"] != selected_clocks
        or catalogue["new_meta_scores_targets_or_accounts_computed"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep exact direct commitment and three original source scopes.")
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
                    "source_eight_policy",
                    "source_eight_implementation_freeze",
                    "cached_nested_growth_quartet_results",
                    "cached_nested_growth_quartet_freeze",
                    "previous_completed_cycle",
                )
            ],
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, selected, parent, refs, selected_clocks, benchmarks


def inputs():
    result = bound(template.inputs)()
    frozen = result[2]
    for candidate in CANDIDATES:
        existing = events("registered", candidate["id"])
        if existing and (
            len(existing) != 1 or existing[0]["body"] != {"candidate": candidate, "frozen": frozen}
        ):
            raise QuantError("A foreign or changed cohort identity cannot be rescued.")
    return result


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial direct-commitment cohort freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1320 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1320/289 before new outcomes.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("Both predeclared financial identities must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 181713
            or design["candidate_ids"] != list(IDS)
            or design["selected_source_slots"] != list(rules.SLOTS)
            or design["new_meta_scores_targets_or_accounts_computed"] is not False
        ):
            raise QuantError("Freeze mechanism-based membership before either new combination.")
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
            raise QuantError("Bind the exact first successful causal six-pair/cohort checks.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "direct_equity_commitment_plus_original_confirmation_defense_hedge",
            "expert_pool_definition_role": "study_specific_four_not_global_roster_replacement",
            "declared_financial_cohort_size": 2,
            "prior_journal_sequence": 1320,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 289,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "COMMITMENT_MACRO_DEFENSE_HEDGE_V1.md"),
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
            "nested_growth_controller_included": False,
            "current_opinion_risk_forecast_promoted": False,
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
                "new_meta_scores_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_either_new_combination")
    return reference(INTENT)


def verify_targets(*args):
    return bound(template.verify_targets)(*args)


def verify_retained_records(*args):
    return bound(template.verify_retained_records)(*args)


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior direct-commitment targets.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve partial actual combination evidence; no replay.")
    if any(len(events("registered", name)) != 1 for name in IDS):
        raise QuantError("Both actual registrations precede either new target combination.")
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
        raise QuantError("Keep every original initial/month-end decision.")
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
        if any(not early[n].equals(targets[n].loc[:day]) for n in IDS) or (
            earlier != audits[: position + 1]
        ):
            raise QuantError("Future data changed an earlier complete-policy allocation.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        directory = FOLDER / "targets" / name
        new_output_directory(directory)
        active = targets[name].dropna(how="all")
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = active.loc[:anchor].tail(1)
            rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
            factors = [
                s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
            ]
            if known.empty or len(factors) < 3:
                raise QuantError("Each horizon needs known initial opinion and genuine factors.")
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        write_new_receipt(
            directory / "proof.json",
            target_proof(
                frozen,
                {
                    **independent,
                    **retained,
                    "decision_count": len(audits),
                    "all_pairs_per_mature_decision": 6,
                    "prefix_truncations": prefixes,
                    "formal_factor_coverage": coverage,
                    "actual_input_resolved_records": reference(log),
                    "new_source_or_shadow_account_runs": 0,
                    "direct_commitment_source_reused": True,
                    "nested_growth_controller_included": False,
                    "admitted_legacy_defensive_views_reused_unchanged": True,
                    "all_source_native_RF_claimed_equal_without_view": False,
                },
            ),
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
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
    return bound(template.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        controls = read_json(
            base.checked_reference(catalogue["cached_nested_growth_quartet_results"])
        )
        control_ids = (
            "expert_pool_growth_macro_defense_hedge_equal_v1",
            "expert_pool_growth_macro_defense_hedge_rolling_pair_v1",
        )
        candidates, comparisons = {}, {}
        for name, old in zip(IDS, control_ids, strict=True):
            result = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(result["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every declared law and all original financial paths.")
            candidates[name] = {"status": result["review"]["status"], "paths": result["paths"]}
            comparisons[name] = {
                "matched_cached_nested_growth_id": old,
                "all_original_paths": {
                    key: {
                        "metrics": {
                            m: value["metrics"][m]
                            - controls["candidates"][old]["paths"][key]["metrics"][m]
                            for m in ("cagr", "sharpe", "max_drawdown")
                        },
                        "execution": {
                            m: value["execution"][m]
                            - controls["candidates"][old]["paths"][key]["execution"][m]
                            for m in (
                                "annualized_two_sided_turnover",
                                "model_order_count",
                                "total_cost_usd",
                            )
                        },
                    }
                    for key, value in result["paths"].items()
                },
                "old_control_replayed": False,
                "whole_policy_difference_is_isolated_source_alpha": False,
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "new_minus_matched_cached_nested_quartet": comparisons,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "declared_study_pool_size": 4,
                "nested_growth_controller_included": False,
                "current_opinion_risk_forecast_promoted": False,
                "admitted_legacy_defensive_views_reused_unchanged": True,
                "all_source_native_RF_claimed_equal_without_view": False,
                "old_controls_or_source_models_replayed": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
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
        parser.error("Use an actual remotely published complete-cohort preregistration SHA.")
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
