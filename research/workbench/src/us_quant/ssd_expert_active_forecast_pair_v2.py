from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_active_forecast_pair_v2 as rules
from us_quant import ssd_expert_active_forecast_pair_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, new_output_directory, read_json, utc_now, write_json

base, ROOT, template = prior.base, prior.ROOT, prior.template
seven, financial, source = prior.seven, prior.financial, prior.seven
lifecycle, reference, unique_refs = prior.lifecycle, prior.reference, prior.unique_refs
FOLDER = base.REPORTS / "expert-active-forecast-pair-20261106-v2"
INTENT = FOLDER / "financial-implementation-freeze-v2.json"
DESIGN = FOLDER / "design-decision-v2.json"
INVOCATION = FOLDER / "actual-invocation-v2.json"
PREPARED = FOLDER / "attested-target-bundle-v2.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_active_forecast_pair_20261106_results_v2.json"
CANDIDATES, IDS = rules.CANDIDATES, rules.IDS
PACKETS = {
    name: ROOT / "evidence" / f"expert_active_forecast_pair_20261106_{i}_registration_v2.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/expert-active-forecast-pair-v2.json",
    "src/us_quant/expert_active_forecast_pair_v2.py",
    "src/us_quant/ssd_expert_active_forecast_pair_v2.py",
    "tests/test_expert_active_forecast_pair_v2.py",
    "EXPERT_ACTIVE_FORECAST_PAIR_V2.md",
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
        raise QuantError("The schema Adapter cannot replace authoritative trial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "active_forecast_pair_v2_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    for key in (
        "source_economic_policy",
        "source_implementation_freeze",
        "source_prepared_targets",
        "source_terminal_closure",
        "source_public_failure",
    ):
        base.checked_reference(policy[key])
    closure = read_json(base.checked_reference(policy["source_terminal_closure"]))
    prepared = read_json(base.checked_reference(policy["source_prepared_targets"]))
    if (
        closure["terminal_V1_will_not_be_patched_or_retried"] is not True
        or closure["new_funded_paths"] != 0
        or closure["saved_source_targets"] != prepared["entries"]
    ):
        raise QuantError("Only the complete terminal V1 target artifacts may supply this Adapter.")
    registrations = [prior.events("registered", n) for n in rules.SOURCE_IDS]
    if any(len(rows) != 1 for rows in registrations):
        raise QuantError("Keep both actual V1 registrations preceding source derivation.")
    parent = registrations[0][0]["body"]["frozen"]
    if (
        any(rows[0]["body"]["frozen"] != parent for rows in registrations)
        or digest_json(parent) != closure["actual_registered_frozen_sha256"]
        or any(len(prior.events("technical_failure", n)) != 1 for n in rules.SOURCE_IDS)
        or any(prior.events("path_started", n) for n in rules.SOURCE_IDS)
    ):
        raise QuantError("Do not erase a failure or reuse a previously funded V1 identity.")
    data, experts, _, refs, clocks, benchmarks = prior.source_inputs()
    extra = [
        *[
            policy[k]
            for k in (
                "source_economic_policy",
                "source_implementation_freeze",
                "source_prepared_targets",
                "source_terminal_closure",
                "source_public_failure",
                "cached_original_eight_results",
            )
        ],
        *[ref for entry in prepared["entries"].values() for ref in entry.values()],
    ]
    refs = unique_refs([*refs, *extra])
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    result = bound(template.inputs)()
    frozen = result[2]
    for candidate in CANDIDATES:
        old = events("registered", candidate["id"])
        if old and (len(old) != 1 or old[0]["body"] != {"candidate": candidate, "frozen": frozen}):
            raise QuantError("Both distinct V2 identities must be checked before either append.")
    return result


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v2.json").exists():
        raise QuantError("Preserve the complete or partial first schema-only V2 freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1251 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze after both terminal V1 failures, not an invented reset.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("Use globally unused V2 financial-program identities.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 175938
            or design["candidate_ids"] != list(IDS)
            or design["new_target_derivations"] != 0
            or design["V1_targets_derived_before_V2_registration"] is not True
        ):
            raise QuantError("Declare the technical continuation and true earlier derivation.")
        write_new_receipt(FOLDER / "freeze-attempt-v2.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v2.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["real_packet_entry_contract_exercised"] is not True
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Require the real complete packet contract, not its successful mock.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "construction": "same_saved_V1_targets_explicit_no_source_signal_recompute_proof",
            "prior_journal_sequence": 1251,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 283,
            "declared_financial_cohort_size": 2,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_ACTIVE_FORECAST_PAIR_V2.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in deps if r["path"].startswith("config/")],
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
                    reference(FOLDER / "validation-source-v2.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "source_V1_registered_frozen_sha256": digest_json(parent),
            "source_V1_terminal_closure": read_json(rules.POLICY)["source_terminal_closure"],
            "source_V1_to_V2_identity_mapping": dict(zip(rules.SOURCE_IDS, IDS, strict=True)),
            "source_V1_targets_derived_before_V2_registration": True,
            "new_V2_target_derivations": 0,
            "source_original_signal_recomputed": False,
            "economic_law_or_parameters_changed": False,
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
                "new_V2_target_derivations": 0,
                "candidate_accounts_or_metrics_computed": False,
            },
        )
        progress("frozen", next_step="register_both_before_new_metadata_attestations")
    return reference(INTENT)


def prepare(data, experts, frozen):
    if PREPARED.exists():
        old = read_json(PREPARED)
        if old["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior V2 attestations.")
        for entry in old["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return old["entries"]
    if (FOLDER / "attestation-attempt-v2.json").exists():
        raise QuantError("Preserve a partial proof attestation rather than retrying it.")
    if any(len(events("registered", n)) != 1 for n in IDS):
        raise QuantError("Both new registrations precede either new proof attestation.")
    write_new_receipt(FOLDER / "attestation-attempt-v2.json", {"started_at": utc_now()})
    policy = read_json(rules.POLICY)
    source = read_json(base.checked_reference(policy["source_prepared_targets"]))
    entries = {}
    for old_id, name in zip(rules.SOURCE_IDS, IDS, strict=True):
        lifecycle.runtime_guard()
        old = source["entries"][old_id]
        base.checked_reference(old["targets"])
        original_proof = read_json(base.checked_reference(old["proof"]))
        registered = prior.events("registered", old_id)[0]
        proof = rules.attest(
            {
                "proof": original_proof,
                "target_reference": old["targets"],
                "proof_reference": old["proof"],
                "registered_at": registered["at"],
                "registered_event_sha256": registered["sha256"],
                "registered_frozen_sha256": digest_json(registered["body"]["frozen"]),
            },
            digest_json(frozen),
            utc_now(),
        )
        directory = FOLDER / "attested-target-proofs" / name
        new_output_directory(directory)
        write_new_receipt(directory / "proof.json", proof)
        entries[name] = {"targets": old["targets"], "proof": reference(directory / "proof.json")}
    write_new_receipt(
        PREPARED,
        {
            "frozen_sha256": digest_json(frozen),
            "entries": entries,
            "source_V1_targets_reused_byte_exact": True,
            "new_target_derivations": 0,
        },
    )
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
        policy = read_json(rules.POLICY)
        controls = read_json(base.checked_reference(policy["cached_original_eight_results"]))[
            "candidates"
        ]
        candidates, comparisons = {}, {}
        for name in IDS:
            result = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(result["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every funded path of both unchanged economic rules.")
            candidates[name] = {"status": result["review"]["status"], "paths": result["paths"]}
            comparisons[name] = {
                control: {
                    key: {
                        "metrics": {
                            m: v["metrics"][m] - old["paths"][key]["metrics"][m]
                            for m in ("cagr", "sharpe", "max_drawdown")
                        },
                        "execution": {
                            m: v["execution"][m] - old["paths"][key]["execution"][m]
                            for m in (
                                "annualized_two_sided_turnover",
                                "model_order_count",
                                "total_cost_usd",
                            )
                        },
                    }
                    for key, v in result["paths"].items()
                }
                for control, old in controls.items()
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 2,
                "study_id": policy["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "new_minus_both_cached_original_eight_controls": comparisons,
                "source_V1_failure": policy["source_public_failure"],
                "source_V1_terminal_failures_preserved": True,
                "source_V1_target_derivation_precedes_V2_registration": True,
                "new_V2_target_derivations": 0,
                "new_Ridge_fits": 0,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "economic_law_or_parameters_changed": False,
                "old_controls_replayed": False,
                "source_models_selectors_forecasters_or_accounts_replayed": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
                "global_pool_changed": False,
                "whole_policy_difference_is_isolated_source_alpha": False,
                "independent_economic_research_questions_added_by_adapter": 0,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Metadata-only proof Adapter; immutable V1 targets."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete two-packet cohort.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
