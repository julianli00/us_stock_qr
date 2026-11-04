from __future__ import annotations

import argparse
import ast
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import paired_activity_information_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import file_digest, read_json, utc_now, write_json
from us_quant.strategy_experts_v1 import load_expert

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
source = seven
reference, unique_refs = template.reference, template.unique_refs
FOLDER = base.REPORTS / "paired-activity-information-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/paired_activity_information_20261104_results_v1.json"
DESIGN_SHA = "4651ee3b7ecb9ff3fea8213a6e8e70cfc16815d203c31e067d7d3777956e45ec"
CANDIDATES = rules.CANDIDATES
IDS = tuple(c["id"] for c in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"paired_activity_information_20261104_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/paired-activity-information-v1.json",
    "src/us_quant/paired_activity_information_v1.py",
    "src/us_quant/ssd_paired_activity_information_v1.py",
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


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("The study cannot override actual authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "paired_activity_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def check_frozen(frozen):
    return template.check_frozen(frozen)


def dependencies():
    walker = seven.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__,
        {**globals(), "ast": ast, "prior": template},
        argdefs=walker.__defaults__,
    )()


def substitute(experts, replacement, variant):
    if (
        variant not in rules.VARIANTS
        or tuple(e.identity.identifier for e in experts) != template.rules.EXPERT_IDS
        or replacement.identity.identifier != rules.EXPERT_IDS[variant][5]
        or replacement.identity.mechanism_group != rules.MECHANISMS[5]
    ):
        raise QuantError("Keep the seven other complete policies and only the declared style role.")
    return (*experts[:5], replacement, *experts[6:])


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    compatibility = read_json(base.checked_reference(policy["source_pair_compatibility"]))
    if (
        compatibility["status"]
        != "both_original_complete_policies_technically_usable_for_frozen_ablation"
        or compatibility["new_financial_registrations"] != 0
        or compatibility["old_target_or_RF_views_or_new_Adapter"] is not False
    ):
        raise QuantError("Complete the cached price-control prerequisite before financial use.")
    data, original, _, refs, clocks, benchmarks = template.source_inputs()
    parent = read_json(template.INTENT)["frozen"]
    contract = read_json(base.checked_reference(compatibility["source_contract"]))
    refs = unique_refs(
        [
            *refs,
            *contract["references"],
            *contract["reader_and_validation"],
            policy["source_pair_compatibility"],
            compatibility["source_contract"],
            compatibility["activity_original_review_not_reexecuted"],
            reference(template.INTENT),
            reference(template.PUBLIC),
        ]
    )
    pools, by_clock = {}, {}
    for key in rules.VARIANTS:
        record = compatibility[f"{'control' if key == 'price' else 'activity'}_record"]
        expert, checked = load_expert(record, tuple(data.close.columns))
        initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
        last = expert.observe(data.close.index[-1], 2512)
        if (
            checked["source_technical_and_funding_checks_passed"] is not True
            or checked["standalone_financial_verdict_preserved"] != "rejected"
            or initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(last.net_returns) != 2512
            or not last.risk_free.equals(data.risk_free.loc[last.risk_free.index])
        ):
            raise QuantError("Keep source compatibility, empty warmup and exact common native RF.")
        pools[key] = substitute(original, expert, key)
        by_clock[key] = [
            *clocks[:5],
            {
                "id": record["id"],
                "source_policy_registered_at": record["introduced_at"],
                "source_technical_prerequisite_recorded_at": compatibility["created_at"],
                "review_is_not_selected_representative_promotion": True,
                "study_role_group_mapping_only": contract["study_role_group_mapping"],
                "decision_information_clock": rules.CLOCK,
            },
            *clocks[6:],
        ]
    for ref in refs:
        base.checked_reference(ref)
    return data, pools, parent, refs, by_clock, benchmarks


def validation_refs():
    receipt = FOLDER / "validation-source-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
    ):
        raise QuantError("Bind the exact implementation bytes to complete canonical checks.")
    xml, log = [base.checked_reference(ref) for ref in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        not suites
        or xml.parent != FOLDER
        or log.parent != FOLDER
        or proof["passed_tests"] < 18
        or sum(int(s.get("tests", 0)) for s in suites) != proof["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Finish both-pool causality, scalar oracle and four-packet fence checks.")
    return [*proof["test_outputs"], reference(receipt)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first paired-information freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        if file_digest(DESIGN) != DESIGN_SHA:
            raise QuantError("The pre-outcome paired-information design changed.")
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 784 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual completed 784-event baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "paired_same_parent_activity_information_ablation_two_by_two",
            "expert_pool_definition_role": "two_study_specific_eight_pools_one_style_role",
            "declared_financial_cohort_size": 4,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "PAIRED_ACTIVITY_INFORMATION_V1.md"),
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
        progress("frozen", next_step="register_all_four_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    return bound(template.inputs)()


def variant_targets(targets, variant):
    return {name: targets[name] for name in rules.ids(variant)}


def variant_audits(audits, variant):
    return [row["variants"][variant] for row in audits]


def verify_targets(data, pools, targets, audits):
    proofs = {
        key: bound(template.verify_targets, rules=rules.oracle_context(key), IDS=rules.ids(key))(
            data, pools[key], variant_targets(targets, key), variant_audits(audits, key)
        )
        for key in rules.VARIANTS
    }
    keys = (
        "maximum_independent_target_difference",
        "maximum_independent_expert_weight_difference",
        "maximum_independent_net_score_difference",
    )
    return {
        **{name: max(p[name] for p in proofs.values()) for name in keys},
        "paired_pool_independent_oracle_checks": proofs,
        "each_variant_all28_pairs_verified": True,
        "pool_variant_candidate_ids": {key: list(rules.ids(key)) for key in rules.VARIANTS},
    }


def verify_retained_records(path, pools, targets, audits):
    paths = {
        key: path.with_name(f"actual-{key}-inputs-resolved-v1.jsonl") for key in rules.VARIANTS
    }
    streams = {}
    try:
        for key, destination in paths.items():
            streams[key] = destination.open("x")
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row.pop("source_variant")
                if key not in streams:
                    raise QuantError("An undeclared pool appeared in actual input evidence.")
                streams[key].write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
    proofs = {}
    for key in rules.VARIANTS:
        verifier = bound(
            seven.verify_retained_records, rules=rules.oracle_context(key), IDS=rules.ids(key)
        )
        proofs[key] = verifier(
            paths[key], pools[key], variant_targets(targets, key), variant_audits(audits, key)
        )
    return {
        "actual_input_and_resolved_record_count": sum(
            p["actual_input_and_resolved_record_count"] for p in proofs.values()
        ),
        "roundtrip_NET_RF_binary64_values_checked": sum(
            p["roundtrip_NET_RF_binary64_values_checked"] for p in proofs.values()
        ),
        "actual_input_resolved_records_match_source_and_targets": True,
        "paired_pool_retained_record_checks": proofs,
        "paired_pool_actual_records": {key: reference(path) for key, path in paths.items()},
    }


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
            "all_four_declared_candidates_and16_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "original_active_rosters_changed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("all16_paths_complete", next_step="publish_all_paired_results_no_winner_promotion")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        candidates = {}
        for name in IDS:
            data = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(data["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every complete original path of every cohort identity.")
            candidates[name] = {"status": data["review"]["status"], "paths": data["paths"]}
        comparisons = {}
        for offset, law in enumerate(("equal", "rolling")):
            price, activity = IDS[offset], IDS[2 + offset]
            comparisons[law] = {
                "price_control": price,
                "activity": activity,
                "all_original_paths": {
                    key: {
                        "activity_minus_price": {
                            metric: candidates[activity]["paths"][key]["metrics"][metric]
                            - candidates[price]["paths"][key]["metrics"][metric]
                            for metric in ("cagr", "sharpe", "max_drawdown")
                        },
                        "activity_minus_price_execution": {
                            metric: candidates[activity]["paths"][key]["execution"][metric]
                            - candidates[price]["paths"][key]["execution"][metric]
                            for metric in (
                                "annualized_two_sided_turnover",
                                "model_order_count",
                                "total_cost_usd",
                            )
                        },
                    }
                    for key in sorted(base.PATH_KEYS)
                },
            }
        body = {
            "schema_version": 1,
            "study_id": read_json(rules.POLICY)["study_id"],
            "completed_at": saved["completed_at"],
            "candidates": candidates,
            "preregistered_paired_comparisons": comparisons,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "study_pool_size_each_variant": 8,
            "one_related_role_only_per_pool": True,
            "original_rosters_changed": False,
            "winner_promotion": False,
            "source_models_or_accounts_replayed": False,
            "new_source_requests": 0,
            "raw_prices_targets_or_daily_rows_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        seven.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Paired information ablation in complete experts.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete four-candidate cohort SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
