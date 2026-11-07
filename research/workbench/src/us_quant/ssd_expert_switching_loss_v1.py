from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_switching_loss_v1 as rules
from us_quant import ssd_expert_switching_wealth_v1 as previous
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_json,
)

base, ROOT, lifecycle = previous.base, previous.ROOT, previous.lifecycle
reference, unique_refs, counts = previous.reference, previous.unique_refs, previous.counts
ID, KIND = rules.ID, previous.KIND
CANDIDATE = {"id": ID, "weighting": "gap_adaptive_current_rate_switching_source_loss_partition"}
FOLDER = base.REPORTS / "expert-switching-loss-20261107-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/expert_switching_loss_20261107_registration_v1.json"
PUBLIC = ROOT / "evidence/expert_switching_loss_20261107_results_v1.json"
OWN = (
    "config/expert-switching-loss-v1.json",
    "src/us_quant/expert_switching_loss_v1.py",
    "src/us_quant/ssd_expert_switching_loss_v1.py",
    "tests/test_expert_switching_loss_v1.py",
    "EXPERT_SWITCHING_LOSS_V1.md",
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


events = bound(previous.events)
check_frozen = bound(previous.check_frozen)
packet_body = bound(previous.packet_body)


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "Adaptive source evidence cannot change authoritative financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "expert_switching_loss_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    registered = read_json(base.checked_reference(catalogue["source_registered_Prod_event"]))
    rules.require(
        closed["actual_authoritative_counts"]["current_journal_event_count"] == 1530
        and registered["kind"] == "registered"
        and registered["candidate_id"] == "expert_pool_six_confidence_rated_prod_v1"
        and catalogue["expert_ids"] == list(rules.EXPERT_IDS)
        and catalogue["mechanism_groups"] == list(rules.MECHANISMS)
        and catalogue["source_clocks"]
        == registered["body"]["frozen"]["source_policy_and_expert_admission_clocks"]
        and catalogue["new_gap_adaptive_sequence_forecasts_learning_targets_or_accounts_computed"]
        is False
        and catalogue["old_source_learner_weights_update_generator_or_account_replayed"] is False
        and catalogue["old_mixability_gap_repaired_clipped_or_floored"] is False,
        "Keep the original admitted sources and complete J1530 predecessor without gap repair.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
            *catalogue["fixed_comparator_adaptive_loss_prior_art"],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, registered["body"]["frozen"], refs


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    suites = ElementTree.parse(base.checked_reference(proof["test_outputs"][0])).getroot()
    suites = suites.findall(".//testsuite")
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")),
        "Bind complete actual gap/rate/source and original-capital synthetic verification.",
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        {
            "test_all119_matured_labels_match_independent_gap_rate_and_forecast",
            "test_current_rate_refilter_is_not_stale_posterior_power",
            "test_equal_losses_have_analytic_zero_gap_without_floor",
            "test_full_mixed_asset_cash_and_BT",
            "test_actual_packet_defers_real_loss_learning",
        }.issubset(names),
        "Exercise new sequence-loss algebra and real funded program seams.",
    )
    return [
        *proof["test_outputs"],
        reference(FOLDER / "validation-source-v1.json"),
        proof["validation_runner"],
        *proof["versioned_prototypes"],
    ]


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial adaptive source-program freeze.",
        )
        rows = base.journal_events(base.JOURNAL)
        rules.require(
            len(rows) == 1530
            and rows[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(row["candidate_id"] == ID for row in rows),
            "Freeze one unused adaptive-loss identity at the unchanged J1530 head.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, parent, refs = source_inputs()
        deps, validation = dependencies(), validation_refs()
        old_method_fields = {
            "confidence_for_loss_fixed_at_prior_month_origin",
            "mixability_gap_floor",
            "negative_Jensen_gap_clipped",
            "normalized_confidence_regret_is_main_wealth",
            "normalized_source_regret_is_main_wealth",
        }
        frozen = {
            **{k: v for k, v in parent.items() if k not in old_method_fields},
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": (
                "new_gap_adaptive_source_forecasts_original_monthly_execution"
            ),
            "expert_pool_definition_role": "same_six_complete_policies_gap_scaled_sequence_prior",
            "construction": "current_rate_Jeffreys_switching_source_loss_partition",
            "decision_program_kind": KIND,
            "declared_financial_cohort_size": 1,
            "decision_information_clock": "matured_source_gap_then_NEXTidentity_forecast",
            "prior_journal_sequence": 1530,
            "prior_journal_head_sha256": rows[-1]["sha256"],
            "prior_completed_configurations": 306,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs([*parent["validation_evidence"], *validation]),
            "shared_source_learning_not_actual_main_capital": True,
            "source_loss_gap_rate_is_main_NET_wealth_or_bound": False,
            "negative_source_gap_clipped": False,
            "source_gap_floor": None,
            "static_AdaHedge_regret_bound_claimed_for_sequence_class": False,
            "old_source_gap_repaired_or_source_program_reexecuted": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_gap_adaptive_sequence_forecasts_learning_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_real_loss_learning")
    return reference(INTENT)


inputs = bound(previous.inputs)


def checked_packet(frozen):
    return bound(previous.template.template.framework.checked_packet)(frozen)


register = bound(previous.register)
publication = bound(previous.publication)
record_failure = bound(previous.record_failure)


def write_run(directory, output):
    return bound(previous.template.template.framework.write_run)(directory, output)


def retain_cash_comparison(directory, actual, replay, report):
    return bound(previous.template.template.retain_cash_comparison)(
        directory, actual, replay, report
    )


class SavedOpinionFeed:
    __init__ = bound(previous.SavedOpinionFeed.__init__)
    observe = bound(previous.SavedOpinionFeed.observe)


prepare_source_targets = bound(previous.prepare_source_targets)
source_prefixes = bound(previous.source_prefixes)
evaluate_path = bound(previous.evaluate_path)
capital_prefixes = bound(previous.capital_prefixes)
evaluate = bound(previous.evaluate)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(
            set(saved["paths"]) == base.PATH_KEYS, "Keep all four adaptive-loss funded paths."
        )
        for key, ref in saved["review"]["paths"].items():
            base.verify_path(ref, saved["frozen"], ID, key)
        source_proof = read_json(base.checked_reference(saved["source_forecast_proof"]))
        fields = (
            "path_key",
            "metrics",
            "execution",
            "gates",
            "all_point_gates_pass",
            "independent_bt_max_equity_difference_usd",
            "actual_capital_and_source_selection_verification",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "gap_adaptive_switching_loss_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "source_forecast_verification": {
                    k: v
                    for k, v in source_proof.items()
                    if k not in ("retained_records", "completed_at")
                },
                "paths": {
                    k: {name: value[name] for name in fields} for k, value in saved["paths"].items()
                },
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "new_source_prefix_inspections": 5,
                "new_capital_prefix_runs": 5,
                "all_source_and_capital_prefixes_exact": True,
                "source_loss_gap_rate_is_main_NET_wealth_or_market_probability": False,
                "negative_gap_clipped_or_floored": False,
                "static_AdaHedge_regret_bound_claimed_for_sequence_class": False,
                "old_learner_source_generator_shadow_or_account_replayed": False,
                "raw_prices_daily_books_targets_source_states_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gap-adaptive complete-expert switching loss research."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published schema3program commit.")
    result = (
        evaluate(args.commit)
        if args.action == "evaluate"
        else {
            "freeze": freeze,
            "register": register,
            "export": export,
        }[args.action]()
    )
    print(json.dumps(result, indent=2, allow_nan=False))
