from __future__ import annotations

import argparse
import ast
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant import ssd_hmm_expert_substitution_v1 as mother
from us_quant import vxn_hmm_composition_substitution_v1 as rules
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import file_digest, read_json, utc_now, write_json
from us_quant.strategy_experts_v1 import load_expert

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
source = seven
reference, unique_refs = template.reference, template.unique_refs
FOLDER = base.REPORTS / "vxn-hmm-composition-substitution-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/vxn_hmm_composition_substitution_20261104_results_v1.json"
REVIEW = base.REPORTS / "archived-vxn-expert-eligibility-20261104-v1"
DESIGN_SHA = "bacb5c432516378e5c66b5409eab78c3504206a949b90b3cc2aa4847e0911a4e"
CANDIDATES = rules.CANDIDATES
IDS = tuple(row["id"] for row in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"vxn_hmm_composition_substitution_20261104_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/vxn-hmm-composition-substitution-v1.json",
    "src/us_quant/vxn_hmm_composition_substitution_v1.py",
    "src/us_quant/ssd_vxn_hmm_composition_substitution_v1.py",
    "tests/test_vxn_hmm_composition_substitution_v1.py",
    "tests/test_hmm_expert_substitution_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Study progress cannot override actual journal-derived financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "vxn_hmm_substitution_" + phase,
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


def substitute(experts, replacement):
    if (
        tuple(e.identity.identifier for e in experts) != mother.rules.EXPERT_IDS
        or replacement.identity.identifier != rules.EXPERT_IDS[2]
        or replacement.identity.mechanism_group != rules.MECHANISMS[2]
    ):
        raise QuantError("Substitute only the declared HMM-family composition source.")
    return (*experts[:2], replacement, *experts[3:])


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    if file_digest(REVIEW / "completion-v1.json") != rules.REVIEW_SHA:
        raise QuantError("The completed VXN source qualification changed.")
    review = read_json(REVIEW / "eligibility-result-v1.json")
    if (
        review["status"] != "TECHNICALLY_USABLE_RELATED_HMM_OPTION_VARIANT_NOT_SELECTED"
        or review["selected_representative_or_pool_member"] is not False
        or review["original_active_rosters_changed"] is not False
        or review["blockers"] != []
    ):
        raise QuantError("Technical usability is not automatic pool promotion or a source waiver.")
    data, experts, parent, refs, clocks, benchmarks = mother.source_inputs()
    record = read_json(REVIEW / "archived-expert-descriptor-v1.json")
    replacement, technical = load_expert(record, tuple(data.close.columns))
    if (
        technical["source_technical_and_funding_checks_passed"] is not True
        or technical["standalone_financial_verdict_preserved"] != "rejected"
        or technical["adapter"] != "monthly"
    ):
        raise QuantError(
            "Reuse complete cached VXN policy evidence, not a financial-winner filter."
        )
    experts = substitute(experts, replacement)
    for expert in experts:
        initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
        final = expert.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(final.net_returns) != 2512
            or not final.risk_free.equals(data.risk_free.loc[final.risk_free.index])
        ):
            raise QuantError("Keep original source availability, empty warmup and exact common RF.")
    qualification = read_json(REVIEW / "eligibility-intent-v1.json")
    refs = unique_refs(
        [
            *refs,
            *qualification["source_references"],
            *qualification["implementation_dependencies"],
            reference(REVIEW / "completion-v1.json"),
            reference(REVIEW / "eligibility-intent-v1.json"),
            reference(REVIEW / "eligibility-result-v1.json"),
            reference(REVIEW / "archived-expert-descriptor-v1.json"),
            reference(REVIEW / "source-integrity-after-v1.json"),
            reference(mother.INTENT),
            reference(mother.FOLDER / "completion-v1.json"),
            policy["matched_HMM_results"],
            reference(mother.rules.POLICY),
            technical["target_reference"],
            technical["shadow_reference"],
            reference(
                base.REPORTS
                / "macro-expert-inclusion-20261104-v1"
                / "all-saved-financial-private-inventory-v1.json"
            ),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    clocks = [
        *clocks[:2],
        {
            "id": replacement.identity.identifier,
            "source_policy_registered_at": record["introduced_at"],
            "technical_eligibility_reviewed_at": review["reviewed_at"],
            "review_is_not_selected_representative_admission": True,
            "source_scope": review["source_scope"],
            "decision_information_clock": rules.CLOCK,
        },
        *clocks[3:],
    ]
    return data, experts, parent, refs, clocks, benchmarks


def validation_refs():
    path = FOLDER / "validation-source-v1.json"
    proof = read_json(path)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
    ):
        raise QuantError("Canonical checks must bind exact new substitution source bytes.")
    xml, log = [base.checked_reference(row) for row in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        not suites
        or xml.parent != FOLDER
        or log.parent != FOLDER
        or proof["passed_tests"] < 15
        or sum(int(s.get("tests", 0)) for s in suites) != proof["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Complete original-law, source-clock and real full-cohort packet checks.")
    return [*proof["test_outputs"], reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first composition-study freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        if file_digest(DESIGN) != DESIGN_SHA:
            raise QuantError("The pre-outcome information-channel experiment changed.")
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 852 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze at the real completed852-event pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "controlled_VXN_composition_for_HMM_same_family_slot",
            "expert_pool_definition_role": "study_specific_substitution_original_rosters_immutable",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "VXN_HMM_COMPOSITION_SUBSTITUTION_V1.md"),
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
            "decision_information_clock": rules.CLOCK,
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
                "both_cohort_identities_registered": False,
                "original_active_rosters_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    return bound(template.inputs)()


def verify_targets(*args):
    return bound(template.verify_targets)(*args)


def verify_retained_records(*args):
    return bound(template.verify_retained_records)(*args)


def prepare(*args):
    return bound(template.prepare)(*args)


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
        results = read_json(RESULTS)
        controls = read_json(base.checked_reference(read_json(rules.POLICY)["matched_HMM_results"]))
        candidates = {}
        for candidate, control in zip(CANDIDATES, mother.IDS, strict=True):
            name = candidate["id"]
            data = read_json(base.checked_reference(results["candidate_result_refs"][name]))
            if set(data["paths"]) != base.PATH_KEYS:
                raise QuantError("Preserve every candidate and full funded path.")
            candidates[name] = {
                "status": data["review"]["status"],
                "paths": data["paths"],
                "registered_event_sha256": events("registered", name)[0]["sha256"],
                "matched_cached_HMM_control": {
                    "candidate_id": control,
                    "paths": controls["candidates"][control]["paths"],
                    "old_control_replayed": False,
                },
            }
        body = {
            "schema_version": 1,
            "study_id": read_json(rules.POLICY)["study_id"],
            "completed_at": results["completed_at"],
            "candidates": candidates,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "study_specific_pool_size": 8,
            "one_related_family_slot_substituted_not_ninth": True,
            "original_active_rosters_unchanged": True,
            "source_models_or_accounts_replayed": False,
            "new_source_requests": 0,
            "raw_source_prices_targets_or_daily_rows_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        seven.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled VXN composition in one HMM family slot."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete-cohort preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )
