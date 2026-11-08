from __future__ import annotations

import argparse
import json
import sys
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_dyadic_meta_v1 as rules
from us_quant import ssd_retained_main_risk_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.storage import digest_json, utc_now, write_json

ROOT, ID = rules.ROOT, rules.ID
base, lifecycle = prior.saved.base, prior.lifecycle
framework, template = prior.saved.framework, prior.saved.execution
KIND = "causal_main_capital_decision_program_v1"
CANDIDATE = {"id": ID, "weighting": "shifted_dyadic_source_intervals_weighted_potential"}
FOLDER = ROOT / "reports/ssd-continuous-research-v1/expert-dyadic-meta-20261108-v1"
CATALOGUE = FOLDER / "source-inputs-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/expert_dyadic_meta_20261108_registration_v1.json"
PUBLIC = ROOT / "evidence/expert_dyadic_meta_20261108_results_v1.json"
OWN = (
    "config/expert-dyadic-meta-v1.json",
    "src/us_quant/expert_dyadic_meta_v1.py",
    "src/us_quant/ssd_expert_dyadic_meta_v1.py",
    "tests/test_expert_dyadic_meta_v1.py",
    "EXPERT_DYADIC_META_V1.md",
    "tests/conftest.py",
)
content_path, read_json, file_digest = prior.content_path, prior.read_json, prior.file_digest
reference, checked_reference = prior.reference, prior.checked_reference
safe_file, review_market = prior.saved.safe_file, prior.saved.review_market
new_output_directory, write_text_atomic = (
    prior.saved.new_output_directory,
    prior.saved.write_text_atomic,
)
counts, unique_refs = prior.saved.counts, prior.saved.unique_refs


def bound(function):
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def events(kind):
    return bound(framework.events)(kind)


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "Do not replace actual journal counters with auxiliary learner counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "dyadic_meta_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def source_inputs():
    policy, catalogue = read_json(rules.POLICY), read_json(CATALOGUE)
    rules.validate(policy)
    rules.require(
        reference(CATALOGUE) == policy["source_inputs"]
        and catalogue["expert_ids"] == list(rules.EXPERT_IDS)
        and catalogue["mechanism_groups"] == list(rules.MECHANISMS)
        and catalogue["actual_authoritative_counts"]["current_journal_event_count"] == 1558
        and catalogue["new_dyadic_states_forecasts_targets_or_accounts_computed"] is False
        and catalogue["failed_scenario_V1_repaired_or_replayed"] is False,
        "Keep original complete sources and the actual independent J1558 capture.",
    )
    refs = unique_refs(
        [
            reference(CATALOGUE),
            catalogue["last_completed_cycle"],
            catalogue["last_complete_financial_cycle"],
            catalogue["last_complete_read_only_cycle"],
            catalogue["cached_complete_policy_opinions"],
            catalogue["cached_six_source_month_audits"],
            catalogue["dyadic_calendar_math_source"],
            catalogue["original_math_potential_source"],
            *catalogue["source_market"].values(),
            *catalogue["source_benchmark_artifacts"],
        ]
    )
    for ref in refs:
        checked_reference(ref)
    return catalogue, catalogue["parent_frozen_specification"], refs


def validation_refs():
    versions = sorted(FOLDER.glob("validation-source-v*.json"))
    rules.require(bool(versions), "Successful exact-source full delivery acceptance is required.")
    proof = read_json(versions[-1])
    suites = (
        ElementTree.parse(checked_reference(proof["test_outputs"][0]))
        .getroot()
        .findall(".//testsuite")
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / n) for n in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        and {
            "test_all119_new_epoch_forecasts_match_recomputed_oracle",
            "test_actual_source_and_capital_JSONL_delivery",
            "test_actual_packet_defers_source_learning",
            "test_future_does_not_change_new_source_prefix",
            "test_prohibited_content_guard_before_open",
        }.issubset(names),
        "Bind numerical, source-clock, real JSONL, accounting and guarded packet acceptance.",
    )
    refs = []
    for version in versions:
        record = read_json(version)
        refs += [
            reference(version),
            *record["test_outputs"],
            record["validation_runner"],
            *record["versioned_prototypes"],
        ]
    return unique_refs(refs)


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        checked_reference(ref)


def freeze():
    prior._guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial freeze.",
        )
        catalogue, parent, refs = source_inputs()
        history = base.journal_events(base.JOURNAL)
        rules.require(
            counts() == catalogue["actual_authoritative_counts"]
            and len(history) == 1558
            and not any(r["candidate_id"] == ID for r in history),
            "Freeze one unused identity without changing the closed predecessor.",
        )
        dependencies, validation = bound(prior.saved.dependencies)(), validation_refs()
        inherited = (
            "protocol",
            "catalog",
            "legacy",
            "volatility_sources",
            "source_policy_and_expert_admission_clocks",
            "possible_economic_factor_families",
            "eligible_actual_registered_factors",
            "minimum_economic_factor_families",
        )
        frozen = {
            **{k: parent[k] for k in inherited},
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / "src/us_quant/main_capital_feedback_v1.py"),
            "execution_adapter_role": "unchanged_original_initial_monthly_actual_cash_capital",
            "construction": "six_complete_sources_dyadic_interval_meta_potential",
            "decision_program_kind": KIND,
            "declared_financial_cohort_size": 1,
            "decision_information_clock": "matured_full_source_month_then_next_epoch_forecast",
            "market": catalogue["source_market"],
            "benchmark_artifacts": catalogue["source_benchmark_artifacts"],
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in dependencies if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": refs,
            "validation_evidence": validation,
            "prior_journal_sequence": 1558,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 308,
            "protected_archival_reference_is_new_live_validation": False,
            "auxiliary_interval_learners_are_independent_economic_trials": False,
            "source_game_guarantee_is_actual_main_wealth_or_drawdown_bound": False,
            "old_source_learner_or_account_replayed": False,
            "failed_scenario_V1_repaired_or_replayed": False,
            "independent_forward_validation": False,
            "order_authority": False,
        }
        check_frozen(frozen)
        prior._permitted_refs(catalogue)
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": dependencies,
                "python_version": sys.version,
                "package_versions": prior._packages(),
                "complete_method_before_outcomes": True,
                "new_dyadic_forecasts_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_checkpoint_then_one_schema3_registration")
        return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        checked_reference(ref)
    catalogue, _, refs = source_inputs()
    rules.require(
        refs == intent["source_references"]
        and sys.version == intent["python_version"]
        and prior._packages() == intent["package_versions"],
        "Keep exact frozen method, inputs and accepted environment.",
    )
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": catalogue["source_market"]}, ROOT)
    refs = [
        r for r in catalogue["source_benchmark_artifacts"] if r["path"].endswith("/results.json")
    ]
    rules.require(len(refs) == 1, "Use the unique original completed benchmark set.")
    benchmarks = read_json(checked_reference(refs[0]))
    rules.require(len(benchmarks["benchmarks"]) == 8, "Keep both benchmarks on all four paths.")
    return data, catalogue, frozen, benchmarks


def checkpoint(path):
    receipt = read_json(path)
    cp = receipt["actual_continuity_checkpoint_readback"]
    created, saved, checked = (
        pd.Timestamp(read_json(INTENT)["created_at"]),
        pd.Timestamp(cp["saved_at"]),
        pd.Timestamp(receipt["checked_at"]),
    )
    rules.require(
        receipt["entire_state_readback_equal"] is True
        and receipt["frozen_implementation"] == reference(INTENT)
        and cp["session_id"] == lifecycle.SESSION
        and type(cp["revision"]) is int
        and cp["revision"] > 938
        and all(t.tzinfo is not None and pd.notna(t) for t in (created, saved, checked))
        and created < saved <= checked <= pd.Timestamp(utc_now()),
        "Actual whole-state checkpoint must follow this method freeze.",
    )
    return receipt


def packet_body(frozen, registered, readiness):
    return bound(framework.packet_body)(frozen, registered, readiness)


def checked_packet(frozen):
    return bound(framework.checked_packet)(frozen)


def register():
    prior._guard()
    checkpoint(FOLDER / "actual-preregistration-checkpoint-v1.json")
    rules.require(
        counts() == read_json(CATALOGUE)["actual_authoritative_counts"],
        "The predecessor changed before this one registration.",
    )
    return bound(framework.register)()


def publication(frozen, commit):
    checked_packet(frozen)
    return bound(base.published_inputs)(frozen, commit)


class SavedOpinionFeed:
    def __init__(self, catalogue):
        self.rows = {}
        with checked_reference(catalogue["cached_complete_policy_opinions"]).open() as stream:
            while line := stream.readline():
                before, after_line = json.loads(line), stream.readline()
                rules.require(bool(after_line), "Keep complete original opinion pairs.")
                after = json.loads(after_line)
                day = pd.Timestamp(before["decision_session"])
                rules.require(
                    before["phase"] == "before_decision"
                    and after["phase"] == "after_decision"
                    and before["decision_session"] == after["decision_session"]
                    and day not in self.rows,
                    "Original native opinion chronology changed.",
                )
                point = rules.OpinionSnapshot(
                    day,
                    np.array(
                        [[float.fromhex(v) for v in row] for row in before["known_targets_hex"]]
                    ),
                    tuple(pd.Timestamp(d) for d in before["known_target_dates"]),
                )
                point.check(day)
                self.rows[day] = point
        rules.require(
            self.rows and list(self.rows) == sorted(self.rows),
            "Missing or reordered original source opinions cannot be filled.",
        )

    def observe(self, day):
        rules.require(day in self.rows, "No future or neutral source snapshot.")
        return self.rows[day]


def prepare_source_targets(data, catalogue):
    return bound(template.prepare_source_targets)(data, catalogue)


def source_prefixes(data, catalogue, prepared):
    return bound(template.source_prefixes)(data, catalogue, prepared)


def write_run(directory, output):
    return bound(framework.write_run)(directory, output)


def retain_cash_comparison(directory, actual, replay, report):
    return prior.saved.retain_cash_comparison(directory, actual, replay, report)


def evaluate_path(data, prepared, frozen, window, scenario, benchmarks):
    return bound(template.evaluate_path)(data, prepared, frozen, window, scenario, benchmarks)


def capital_prefixes(data, prepared):
    return bound(template.capital_prefixes)(data, prepared)


def record_failure(stage, error):
    return bound(framework.record_failure)(stage, error)


def evaluate(commit):
    prior._guard()
    start = checkpoint(FOLDER / "actual-funding-start-v1.json")
    remote = read_json(checked_reference(start["actual_remote_publication"]))
    rules.require(
        start["actual_preregistration_commit"] == remote["actual_preregistration_commit"] == commit
        and remote["historical_outputs_previously_started"] is False
        and pd.Timestamp(read_json(PACKET)["created_at"])
        < pd.Timestamp(remote["verified_at"])
        < pd.Timestamp(start["actual_continuity_checkpoint_readback"]["saved_at"]),
        "A second actual checkpoint must follow real remote publication before source learning.",
    )
    return bound(template.evaluate)(commit)


def export():
    prior._guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(set(saved["paths"]) == base.PATH_KEYS, "Report all original complete paths.")
        for key, ref in saved["review"]["paths"].items():
            bound(base.verify_path)(ref, saved["frozen"], ID, key)
        proof = read_json(checked_reference(saved["source_forecast_proof"]))
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
                "record_type": "actual_dyadic_interval_meta_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "source_forecast_verification": {
                    k: v for k, v in proof.items() if k not in ("retained_records", "completed_at")
                },
                "paths": {k: {f: v[f] for f in fields} for k, v in saved["paths"].items()},
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "auxiliary_interval_learners_are_independent_economic_trials": False,
                "source_game_guarantee_is_actual_main_wealth_or_drawdown_bound": False,
                "failed_scenario_V1_repaired_or_replayed": False,
                "raw_prices_targets_source_states_or_daily_books_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="One independent dyadic source meta study.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate":
        rules.require(bool(args.commit), "Use the actual verified remote program commit.")
        result = evaluate(args.commit)
    else:
        result = globals()[args.action]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
