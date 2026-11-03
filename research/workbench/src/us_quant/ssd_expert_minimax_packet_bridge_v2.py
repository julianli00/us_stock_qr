from __future__ import annotations

import argparse
import ast
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import ssd_expert_minimax_opportunity_v1 as original
from us_quant import ssd_expert_target_consensus_v1 as financial
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import digest_json, file_digest, read_json, utc_now, write_json

base, ROOT, source, lifecycle = original.base, original.ROOT, original.source, original.lifecycle
reference, unique_refs = original.reference, original.unique_refs
FOLDER = base.REPORTS / "expert-minimax-opportunity-packet-bridge-20261103-v2"
CONTRACT = FOLDER / "bridge-contract-v2.json"
POLICY = ROOT / "config/expert-minimax-packet-bridge-v2.json"
INTENT = FOLDER / "financial-implementation-freeze-v2.json"
PROOF = FOLDER / "target-proof-view-v2.json"
PACKET = ROOT / "evidence/expert_minimax_packet_bridge_20261103_registration_v2.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_minimax_packet_bridge_20261103_results_v2.json"
STUDY = "ssd_expert_minimax_opportunity_packet_bridge_20261103_v2"
ID = "expert_pool_eight_minimax_monthly_opportunity_packet_bridge_v2"
CANDIDATE = {
    "id": ID,
    "weighting": original.CANDIDATE["weighting"],
    "economic_rule_id": original.ID,
    "metadata_compatibility_only": True,
}
CONTRACT_SHA = "50dcbaf5f5946c7a6b77828efdbdb558f3c494b37da40b1eccb62e8dd54ab401"
TERMINAL_SHA = "8ed479ce470707da32bedd75397fa939d21044ba9cf07553a63e7cef1107fe5d"
PRIOR_HEAD = "a083e93138321c7d7f591020acdb5c8014f5ea94ef4af2649adaf2e0f1a2409e"
OWN = (
    "config/expert-minimax-packet-bridge-v2.json",
    "src/us_quant/ssd_expert_minimax_packet_bridge_v2.py",
    "tests/test_ssd_expert_minimax_packet_bridge_v2.py",
    *original.OWN,
)


def validate():
    policy = read_json(POLICY)
    expected = {
        "schema_version": 2,
        "study_id": STUDY,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "candidate_id": ID,
        "source_candidate_id": original.ID,
        "source_economic_policy": original.rules.POLICY.relative_to(ROOT).as_posix(),
        "source_primary_freeze_sha256": original.PRIMARY_SHA,
        "source_terminal_completion_sha256": TERMINAL_SHA,
        "bridge_contract_sha256": CONTRACT_SHA,
        "prior_journal_sequence": 725,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "cached_targets_reused_without_recomputation": True,
        "source_registration_precedes_original_target_derivation": True,
        "v2_registration_predates_cached_targets": False,
        "economic_law_or_source_or_financial_gate_changed": False,
        "new_independent_economic_hypotheses_relative_to_v1": 0,
        "old_terminal_identity_repaired_or_retried": False,
        "new_source_requests": 0,
        "new_source_shadow_or_control_runs": 0,
        "new_score_or_target_runs": 0,
        "new_funded_paths_after_actual_remote_preregistration": 4,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Only the frozen explicit packet metadata bridge is permitted.")
    if file_digest(CONTRACT) != CONTRACT_SHA:
        raise QuantError("The bounded metadata-only bridge contract changed.")


def events(kind):
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["candidate_id"] == ID and row["kind"] == kind
    ]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Do not reset failed versions or authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "minimax_packet_bridge_v2_" + phase.removeprefix("consensus_"),
            "study_id": STUDY,
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def source_inputs():
    validate()
    terminal = original.FOLDER / "terminal-completion-v1.json"
    if file_digest(terminal) != TERMINAL_SHA:
        raise QuantError("Preserve the actual original terminal technical failure.")
    registrations = original.events("registered")
    proofs = original.events("target_proof")
    failures = original.events("technical_failure")
    if (
        len(registrations) != 1
        or len(proofs) != 1
        or len(failures) != 1
        or original.events("path_started")
        or original.events("review")
        or failures[0]["body"]["reason"] != "'original_source_signal_recomputed'"
        or failures[0]["body"]["candidate_accounts_started"] is not False
        or proofs[0]["body"]["registered_event_sha256"] != registrations[0]["sha256"]
    ):
        raise QuantError("Bridge only this recorded packet-schema failure with no prior funds.")
    old_frozen = registrations[0]["body"]["frozen"]
    original.check_frozen(old_frozen)
    prepared = read_json(original.PREPARED)
    if (
        prepared["frozen_sha256"] != digest_json(old_frozen)
        or prepared["prepared"]["targets"] != proofs[0]["body"]["targets"]
        or prepared["prepared"]["proof"] != proofs[0]["body"]["proof"]
    ):
        raise QuantError("Reuse exactly the outputs proven under the original registration.")
    for row in prepared["prepared"].values():
        base.checked_reference(row)
    data, experts, _, refs, clocks, benchmarks = original.source_inputs()
    refs = unique_refs(
        [
            *refs,
            reference(CONTRACT),
            reference(POLICY),
            reference(terminal),
            reference(original.INTENT),
            reference(original.PREPARED),
            *prepared["prepared"].values(),
            reference(original.FOLDER / "information-audits-v1.json"),
            reference(original.FOLDER / "actual-decision-inputs-resolved-v1.jsonl"),
        ]
    )
    for row in refs:
        base.checked_reference(row)
    return data, experts, old_frozen, refs, clocks, benchmarks


def dependencies():
    walker = source.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__,
        {**globals(), "ast": ast, "prior": original.pool_source},
        argdefs=walker.__defaults__,
    )()


def validation_refs():
    path = FOLDER / "validation-source-v2.json"
    proof = read_json(path)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
    ):
        raise QuantError("Canonical bridge checks must bind these exact source bytes.")
    xml, log = [base.checked_reference(row) for row in proof["test_outputs"]]
    if xml.parent != FOLDER or log.parent != FOLDER:
        raise QuantError("Retain actual local bridge validation evidence.")
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        not suites
        or sum(int(s.get("tests", 0)) for s in suites) != proof["passed_tests"]
        or proof["passed_tests"] < 8
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Complete the real packet contract and cached-chronology checks.")
    return [*proof["test_outputs"], reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v2.json").exists():
        raise QuantError("Preserve the first complete or partial bridge freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v2.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 725 or history[-1]["sha256"] != PRIOR_HEAD:
            raise QuantError("Freeze after the preserved V1 failure, without resetting its count.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": STUDY,
            "policy": reference(POLICY),
            "economic_policy": parent["policy"],
            "construction": "unchanged_minimax_law_cached_targets_packet_metadata_bridge_v2",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_MINIMAX_PACKET_BRIDGE_V2.md"),
                ]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "source_original_candidate": original.ID,
            "new_economic_hypotheses_relative_to_v1": 0,
            "cached_targets_recomputed": False,
        }
        original.check_frozen(frozen)
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
                "new_scores_targets_or_funded_accounts_computed": False,
                "v1_failure_preserved": True,
            },
        )
        progress("frozen", next_step="register_new_bridge_identity_before_fresh_proof")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen isolated source environment changed.")
    data, experts, _, refs, clocks, benchmarks = source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("The cached source outputs or admission scope changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    original.check_frozen(frozen)
    return data, experts, frozen, benchmarks


def proof_view(old, old_registration, old_proof, frozen):
    if (
        old["candidate_accounts_or_metrics_computed"] is not False
        or old["source_generator_or_old_account_replayed"] is not False
        or old_proof["body"]["registered_event_sha256"] != old_registration["sha256"]
    ):
        raise QuantError("A metadata view cannot turn source generation or prior funds into false.")
    return {
        **old,
        "checked_at": utc_now(),
        "frozen_specification_sha256": digest_json(frozen),
        "original_source_signal_recomputed": False,
        "new_score_or_target_calculation_in_v2": False,
        "source_original_checked_at": old["checked_at"],
        "source_original_frozen_specification_sha256": old["frozen_specification_sha256"],
        "source_original_registered_at": old_registration["at"],
        "source_original_registration_event_sha256": old_registration["sha256"],
        "source_original_target_proof_at": old_proof["at"],
        "source_original_target_proof_event_sha256": old_proof["sha256"],
        "v2_registration_predates_cached_target_derivation": False,
        "source_terminal_technical_failure_preserved": True,
        "independent_forward_validation": False,
    }


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        _, _, frozen, _ = inputs()
        if events("technical_failure"):
            raise QuantError("A failed V2 bridge is terminal; no additional schema retries.")
        body = {"candidate": CANDIDATE, "frozen": frozen}
        if not events("registered"):
            base.append_event(base.JOURNAL, "registered", ID, body)
        elif events("registered")[0]["body"] != body:
            raise QuantError("The bridge identity or unchanged economic law changed.")
        try:
            prepared = read_json(original.PREPARED)["prepared"]
            original_registration = original.events("registered")[0]
            original_proof = original.events("target_proof")[0]
            if not events("target_proof"):
                old = read_json(base.checked_reference(prepared["proof"]))
                write_new_receipt(
                    PROOF, proof_view(old, original_registration, original_proof, frozen)
                )
                base.append_event(
                    base.JOURNAL,
                    "target_proof",
                    ID,
                    {
                        "registered_event_sha256": events("registered")[0]["sha256"],
                        "targets": prepared["targets"],
                        "proof": reference(PROOF),
                        "cached_original_targets_not_recomputed": True,
                    },
                )
            entry = source.common.packet_entry(CANDIDATE, frozen)
            source.common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": STUDY,
                    "created_at": utc_now(),
                    "entries": [entry],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 0,
                    "same_economic_hypothesis_as_terminal_v1": True,
                    "source_original_registered_event_sha256": original_registration["sha256"],
                    "source_original_target_proof_event_sha256": original_proof["sha256"],
                    "cached_original_targets_not_recomputed": True,
                    "candidate_accounts_or_metrics_computed": False,
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "order_authority": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                },
                ("created_at",),
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("packet_bridge", error)
            raise
        progress("registered_before_funds", next_step="publish_exact_v2_packet_before_accounts")
    return reference(PACKET)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def publication(frozen, commit):
    return bound(financial.publication)(frozen, commit)


def record_failure(stage, error):
    return bound(financial.record_failure)(stage, error)


def evaluate_path(*args):
    return bound(financial.evaluate_path)(*args)


def evaluate(commit):
    return bound(financial.evaluate)(commit)


def export():
    saved = read_json(RESULTS)
    if set(saved["paths"]) != base.PATH_KEYS:
        raise QuantError("Publish every original path of this unchanged economic rule.")
    controls = read_json(base.checked_reference(reference(original.pool_source.PUBLIC)))
    body = {
        "schema_version": 2,
        "study_id": STUDY,
        "candidate_id": ID,
        "economic_rule_id": original.ID,
        "completed_at": saved["completed_at"],
        "status": saved["review"]["status"],
        "paths": saved["paths"],
        "cached_eight_controls": controls["candidates"],
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "v1_terminal_failure_preserved": reference(original.FOLDER / "terminal-completion-v1.json"),
        "same_economic_hypothesis_not_an_independent_trial": True,
        "v2_score_or_target_recomputation": False,
        "new_source_shadow_or_control_runs": 0,
        "new_source_requests": 0,
        "raw_source_daily_rows_or_targets_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    source.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Metadata-only packet bridge; no source replay.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published V2 preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
