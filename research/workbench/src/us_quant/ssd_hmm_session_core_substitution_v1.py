from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import archived_hmm_session_eligibility_v1 as admitted
from us_quant import hmm_session_core_substitution_v1 as rules
from us_quant import ssd_paired_activity_information_v1 as paired
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

base, ROOT, template = paired.base, paired.ROOT, paired.template
seven, financial, lifecycle = paired.seven, paired.financial, paired.lifecycle
source = seven
reference, unique_refs = paired.reference, paired.unique_refs
FOLDER = base.REPORTS / "hmm-session-core-substitution-20261105-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/hmm_session_core_substitution_20261105_results_v1.json"
CANDIDATES = rules.CANDIDATES
IDS = tuple(c["id"] for c in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"hmm_session_core_substitution_20261105_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/hmm-session-core-substitution-v1.json",
    "src/us_quant/hmm_session_core_substitution_v1.py",
    "src/us_quant/ssd_hmm_session_core_substitution_v1.py",
    "tests/test_hmm_session_core_substitution_v1.py",
    "HMM_SESSION_CORE_SUBSTITUTION_V1.md",
    "src/us_quant/archived_hmm_session_eligibility_v1.py",
    "tests/test_paired_activity_information_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/conftest.py",
)


def bound(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals(), **overrides},
        argdefs=function.__defaults__,
    )


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def dependencies():
    return bound(_dependencies)()


def validation_refs():
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
        raise QuantError("Bind all three-pool causality, scalar and six-packet fence checks.")
    return [reference(xml), reference(log), reference(FOLDER / "validation-source-v1.json")]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("The study cannot override actual authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "hmm_session_core_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def substitute(experts, replacement, variant):
    if (
        variant not in rules.VARIANTS
        or tuple(e.identity.identifier for e in experts) != template.rules.EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in experts) != template.rules.MECHANISMS
        or replacement.identity.identifier != rules.EXPERT_IDS[variant][2]
        or replacement.identity.mechanism_group != rules.MECHANISMS[2]
    ):
        raise QuantError("Retain seven exact complete sources and only the declared slot2 sibling.")
    return (*experts[:2], replacement, *experts[3:])


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    complete = read_json(base.checked_reference(policy["source_eligibility_completion"]))
    if (
        complete["all_three_technical_eligibility_passed"] is not True
        or complete["representative_or_pool_member_selected"] is not False
        or complete["new_financial_evaluations"] != 0
        or complete["source_financial_verdicts_preserved"] is not True
        or complete["native_RF_rounding_normalization_or_view_applied"] is not False
        or complete["public_complete_review"] != policy["source_eligibility"]
    ):
        raise QuantError(
            "Require the actual separate three-source eligibility, not auto promotion."
        )
    data, original, _, refs, clocks, benchmarks = template.source_inputs()
    parent = read_json(template.INTENT)["frozen"]
    pools, by_clock = {}, {}
    refs = unique_refs(
        [
            *refs,
            *complete["exact_evidence_references"],
            policy["source_eligibility"],
            policy["source_eligibility_completion"],
            policy["matched_cached_HMM_results"],
            reference(template.INTENT),
        ]
    )
    for variant, identifier in zip(rules.VARIANTS, rules.SOURCE_IDS, strict=True):
        expert, row = admitted.load_admitted(
            policy["source_eligibility"], identifier, tuple(data.close.columns), data.risk_free
        )
        initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
        full = expert.observe(data.close.index[-1], 2512)
        if (
            row["source_financial_verdict_preserved"] != "rejected"
            or initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(full.net_returns) != 2512
            or not full.risk_free.equals(data.risk_free.loc[full.risk_free.index])
        ):
            raise QuantError(
                "Preserve complete known opinions, zero initial NET and exact native RF."
            )
        pools[variant] = substitute(original, expert, variant)
        by_clock[variant] = [
            *clocks[:2],
            {
                "id": identifier,
                "source_policy_registered_at": row["introduced_at"],
                "actual_expert_admitted_at": row["admitted_at"],
                "source_scope": "unchanged_archived_HMM_session_complete_policy",
                "source_universe_width": 12,
                "consumer_universe_width": 14,
                "native_RF_exact_common": True,
                "new_RF_or_recommendation_view": False,
                "only_one_related_sibling_in_this_pool": True,
                "decision_information_clock": rules.CLOCK,
            },
            *clocks[3:],
        ]
    for ref in refs:
        base.checked_reference(ref)
    return data, pools, parent, refs, by_clock, benchmarks


def inputs():
    result = bound(template.inputs)()
    frozen = result[2]
    for candidate in CANDIDATES:
        rows = events("registered", candidate["id"])
        if rows and (
            len(rows) != 1 or rows[0]["body"] != {"candidate": candidate, "frozen": frozen}
        ):
            raise QuantError(
                "Reject every foreign cohort identity before any new event is appended."
            )
    return result


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial controlled session freeze.")
    with lifecycle.research_job():
        if any(e["candidate_id"] in IDS for e in base.journal_events(base.JOURNAL)):
            raise QuantError("Every new six-candidate identity must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 172564
            or design["actual_new_meta_scores_targets_or_accounts_computed"] is not False
            or design["candidates"] != list(CANDIDATES)
        ):
            raise QuantError(
                "Declare every sibling/law before outcomes, not a full-history winner."
            )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1144 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual closed J1144/274 baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "three_HMM_session_source_substitutions_same_equal_rolling_laws",
            "expert_pool_definition_role": "three_separate_eight_pools_only_slot2_changes",
            "declared_financial_cohort_size": 6,
            "prior_journal_sequence": 1144,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 274,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "HMM_SESSION_CORE_SUBSTITUTION_V1.md"),
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
                    *validation_refs(),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "all_source_native_RF_exact_common_no_view": True,
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
                "real_new_scores_targets_or_accounts_computed": False,
                "original_active_rosters_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_all_six_before_any_actual_meta_outputs")
    return reference(INTENT)


def variant_targets(targets, variant):
    return bound(paired.variant_targets)(targets, variant)


def variant_audits(audits, variant):
    return bound(paired.variant_audits)(audits, variant)


def verify_targets(*args):
    return bound(paired.verify_targets)(*args)


def verify_retained_records(*args):
    return bound(paired.verify_retained_records)(*args)


def prepare(*args):
    return bound(template.prepare)(*args)


def register():
    return bound(template.register)()


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        for ref in read_json(RESULTS)["candidate_result_refs"].values():
            base.checked_reference(ref)
        return reference(RESULTS)
    refs = {}
    for candidate in CANDIDATES:
        lifecycle.runtime_guard()
        refs[candidate["id"]] = scoped(financial.evaluate, candidate)(commit)
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": refs,
            "all_six_declared_candidates_and24_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "original_active_rosters_changed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("all24_paths_complete", next_step="publish_every_outcome_no_winner_promotion")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        controls = read_json(
            base.checked_reference(read_json(rules.POLICY)["matched_cached_HMM_results"])
        )["candidates"]
        candidates, comparisons = {}, {}
        for name in IDS:
            result = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(result["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every complete original path of every cohort identity.")
            candidates[name] = {"status": result["review"]["status"], "paths": result["paths"]}
        for offset, law in enumerate(("equal", "rolling")):
            close = rules.ids("close")[offset]
            comparisons[law] = {}
            for variant in rules.VARIANTS:
                identifier = rules.ids(variant)[offset]
                control = (
                    "expert_pool_hmm_substitution_equal_v1",
                    "expert_pool_hmm_substitution_rolling_pair_v1",
                )[offset]
                comparisons[law][variant] = {
                    "candidate_id": identifier,
                    "contemporaneous_close_control": close,
                    "matched_cached_HMM_control": control,
                    "all_original_paths": {
                        key: {
                            "candidate_minus_close_metrics": {
                                metric: candidates[identifier]["paths"][key]["metrics"][metric]
                                - candidates[close]["paths"][key]["metrics"][metric]
                                for metric in ("cagr", "sharpe", "max_drawdown")
                            },
                            "candidate_minus_cached_HMM_metrics": {
                                metric: candidates[identifier]["paths"][key]["metrics"][metric]
                                - controls[control]["paths"][key]["metrics"][metric]
                                for metric in ("cagr", "sharpe", "max_drawdown")
                            },
                            "candidate_minus_cached_HMM_execution": {
                                metric: candidates[identifier]["paths"][key]["execution"][metric]
                                - controls[control]["paths"][key]["execution"][metric]
                                for metric in (
                                    "annualized_two_sided_turnover",
                                    "model_order_count",
                                    "total_cost_usd",
                                )
                            },
                        }
                        for key in sorted(base.PATH_KEYS)
                    },
                    "old_control_replayed": False,
                    "whole_policy_contrast_is_isolated_causal_session_alpha": False,
                }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "preregistered_source_comparisons": comparisons,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "three_separate_pools_of_eight": True,
                "only_one_related_sibling_each_pool": True,
                "original_rosters_changed": False,
                "winner_promotion": False,
                "source_models_targets_or_accounts_replayed": False,
                "new_source_requests": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Three-source complete HMM session substitution.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete six-candidate cohort.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
