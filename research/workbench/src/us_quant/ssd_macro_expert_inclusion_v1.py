from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import macro_compatible_expert_v1 as admitted
from us_quant import macro_expert_inclusion_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
source = seven
reference, unique_refs = template.reference, template.unique_refs
FOLDER = base.REPORTS / "macro-expert-inclusion-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/macro_expert_inclusion_20261104_results_v1.json"
ADMISSION_FOLDER = base.REPORTS / "macro-compatible-expert-admission-20261104-v1"
DESIGN_SHA = "324bd92c10b25af236d83c7ffdde7619e29a9caace7d6e5f85447d60b2e72acf"
CANDIDATES = rules.CANDIDATES
IDS = tuple(row["id"] for row in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"macro_expert_inclusion_20261104_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/macro-expert-inclusion-v1.json",
    "src/us_quant/macro_expert_inclusion_v1.py",
    "src/us_quant/ssd_macro_expert_inclusion_v1.py",
    "tests/test_macro_expert_inclusion_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/test_macro_policy_compatibility_v1.py",
    "tests/test_macro_compatible_expert_v1.py",
    "tests/conftest.py",
)
METHODS = (
    "MACRO_POLICY_COMPATIBILITY_V1.md",
    "MACRO_COMPATIBLE_EXPERT_ADMISSION_V1.md",
    "MACRO_EXPERT_INCLUSION_V1.md",
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
        raise QuantError("Study progress cannot override actual authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "macro_inclusion_" + phase,
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


def include(experts, macro):
    if (
        tuple(e.identity.identifier for e in experts) != template.rules.EXPERT_IDS
        or macro.identity.identifier != rules.EXPERT_IDS[-1]
        or macro.identity.mechanism_group != rules.MECHANISMS[-1]
    ):
        raise QuantError("Retain all original eight and append only the declared macro policy.")
    return (*experts, macro)


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    if file_digest(ADMISSION_FOLDER / "completion-v1.json") != rules.COMPLETION_SHA:
        raise QuantError("The completed scoped macro admission changed.")
    record = read_json(base.checked_reference(policy["new_expert_admission"]))
    if (
        record["status"] != admitted.ADMISSION_STATUS
        or record["scope"] != admitted.SCOPE
        or record["selected_representative_or_pool_member"] is not False
        or record["original_source_financial_rejection_and_native_no_view_block_preserved"]
        is not True
    ):
        raise QuantError("A deliberate study definition is not a source waiver or auto promotion.")
    data, experts, _, refs, clocks, benchmarks = template.source_inputs()
    macro = admitted.load_admitted(policy["new_expert_admission"], tuple(data.close.columns))
    experts = include(experts, macro)
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
            raise QuantError("Keep known source opinions, empty warmup and exact assessment RF.")
    parent = read_json(template.INTENT)["frozen"]
    admission_intent = read_json(ADMISSION_FOLDER / "implementation-intent-v1.json")
    contract = read_json(base.checked_reference(record["contract"]))
    refs = unique_refs(
        [
            *refs,
            *admission_intent["source_references"],
            *admission_intent["implementation_and_validation"],
            reference(ADMISSION_FOLDER / "completion-v1.json"),
            reference(ADMISSION_FOLDER / "implementation-intent-v1.json"),
            reference(ADMISSION_FOLDER / "admission-summary-v1.json"),
            reference(ADMISSION_FOLDER / "admitted-consumer-readback-v1.json"),
            reference(ADMISSION_FOLDER / "source-integrity-after-v1.json"),
            policy["new_expert_admission"],
            record["contract"],
            contract["existing_assessment"],
            contract["existing_assessment_provenance"],
            reference(template.INTENT),
            reference(template.PUBLIC),
            reference(
                base.REPORTS
                / "paired-activity-information-20261104-v1"
                / "all-saved-financial-private-inventory-v1.json"
            ),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    clocks = [
        *clocks,
        {
            "id": macro.identity.identifier,
            "source_policy_registered_at": record["expert_identity"]["source_policy_registered_at"],
            "consumer_defined_at": macro.identity.introduced_at,
            "actual_technical_admission_at": record["admitted_at"],
            "source_scope": record["scope"],
            "native_RF_still_unequal_common": True,
            "existing_common_RF_assessment_reference": contract["existing_assessment"],
            "existing_assessment_provenance": contract["existing_assessment_provenance"],
            "source_observation_is_not_native_request": True,
            "study_specific_inclusion_not_global_roster_promotion": True,
            "decision_information_clock": rules.CLOCK,
        },
    ]
    return data, experts, parent, refs, clocks, benchmarks


def validation_refs():
    path = FOLDER / "validation-source-v1.json"
    proof = read_json(path)
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
    ):
        raise QuantError("Bind the exact nine-source code and test inputs to canonical results.")
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
        raise QuantError("Finish all36-pair, macro-role, causality and full-cohort packet checks.")
    return [*proof["test_outputs"], reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first macro financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        if file_digest(DESIGN) != DESIGN_SHA:
            raise QuantError("The pre-outcome deliberate inclusion design changed.")
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 829 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze at the real completed 829-event financial baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "deliberate_scoped_macro_information_channel_inclusion",
            "expert_pool_definition_role": "study_specific9_original7and8_rosters_immutable",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / name) for name in METHODS],
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
            "macro_information_scope": admitted.SCOPE,
            "macro_native_RF_claimed_exact_common": False,
            "macro_assessment_view_already_existing": True,
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
        progress("frozen", next_step="register_both_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    return bound(template.inputs)()


def verify_targets(*args):
    return bound(template.verify_targets)(*args)


def verify_retained_records(path, experts, targets, audits):
    proof = bound(seven.verify_retained_records)(path, experts, targets, audits)
    with path.open() as stream:
        for audit in audits:
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            expected = rules.macro_context(experts, pd.Timestamp(audit["decision_session"]))
            if (
                before["macro_policy_event_context"] != expected
                or before["observations"][-1]["target_known_at"]
                != expected["original_policy_opinion_known_at"]
                or after["audit"] != audit
            ):
                raise QuantError("The actual macro source opinion/seed role or clock changed.")
    return {**proof, "actual_macro_role_context_checks": len(audits)}


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior nine-source targets.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve unfinished actual derivation instead of replaying it.")
    if any(len(events("registered", c["id"])) != 1 for c in CANDIDATES):
        raise QuantError("Both real cohort identities must precede all new scores or targets.")
    write_new_receipt(
        FOLDER / "preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
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
        raise QuantError("Keep every original121capital/month-end decision.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future data changed a prior nine-source or macro role decision.")
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
                symbol
                for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
                if (rows[symbol] > 1e-10).any()
            ]
            if known.empty or len(factors) < 3:
                raise QuantError("Both funded horizons require known anchors and genuine factors.")
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                **independent,
                **retained,
                "decision_count": len(audits),
                "all_pairs_per_mature_decision": 36,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "macro_scope_and_native_seed_opinion_roles_preserved": True,
                "macro_native_RF_still_unequal_common": True,
                "macro_uses_existing_exact_common_RF_assessment_view": True,
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
                "new_source_or_shadow_account_runs": 0,
                "independent_forward_validation": False,
            },
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "entries": entries})
    return entries


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if any(events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal cohort identity cannot be rescued or repeated.")
        for candidate in CANDIDATES:
            name = candidate["id"]
            if not events("registered", name):
                base.append_event(
                    base.JOURNAL, "registered", name, {"candidate": candidate, "frozen": frozen}
                )
            elif events("registered", name)[0]["body"] != {
                "candidate": candidate,
                "frozen": frozen,
            }:
                raise QuantError("Cohort membership or law changed after actual registration.")
        try:
            prepared = prepare(data, experts, frozen)
            for candidate in CANDIDATES:
                name = candidate["id"]
                if not events("target_proof", name):
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": events("registered", name)[0]["sha256"],
                            **prepared[name],
                        },
                    )
                entry = seven.common.packet_entry(candidate, frozen)
                seven.common.retain_receipt(
                    PACKETS[name],
                    {
                        "schema_version": 2,
                        "program_id": "ssd_continuous_research_v1",
                        "study_id": frozen["study_id"],
                        "created_at": utc_now(),
                        "entries": [entry],
                        "complete_fixed_cohort_ids": list(IDS),
                        "cohort_registrations": [events("registered", n)[0]["sha256"] for n in IDS],
                        "candidate_accounts_or_metrics_computed": False,
                        "order_authority": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                    },
                    ("created_at",),
                )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            for name in IDS:
                if not events("technical_failure", name):
                    base.append_event(
                        base.JOURNAL,
                        "technical_failure",
                        name,
                        {
                            "stage": "nine_source_macro_preparation",
                            "reason": str(error),
                            "partial_inputs_preserved": True,
                        },
                    )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "old_source_or_failed_algorithm_rescue": False,
                },
            )
            progress("technical_failure_preserved", next_step="do_not_retry_or_retune_this_cohort")
            raise
        progress(
            "registered_before_accounts", next_step="publish_both_exact_packets_before_funding"
        )
    return {name: reference(path) for name, path in PACKETS.items()}


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
        controls = read_json(template.PUBLIC)["candidates"]
        candidates = {}
        for candidate, control in zip(
            CANDIDATES,
            ("expert_pool_eight_equal_weight_v1", "expert_pool_eight_rolling_pair_v1"),
            strict=True,
        ):
            name = candidate["id"]
            data = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(data["paths"]) != base.PATH_KEYS:
                raise QuantError("Every complete original funded path must be published.")
            candidates[name] = {
                "status": data["review"]["status"],
                "paths": data["paths"],
                "matched_cached_original_eight_control": {
                    "candidate_id": control,
                    "paths": controls[control]["paths"],
                    "old_control_replayed": False,
                },
            }
        body = {
            "schema_version": 1,
            "study_id": read_json(rules.POLICY)["study_id"],
            "completed_at": saved["completed_at"],
            "candidates": candidates,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "study_specific_pool_size": 9,
            "deliberate_scoped_inclusion_not_automatic_global_ninth": True,
            "original_active_rosters_changed": False,
            "macro_admission_scope": admitted.SCOPE,
            "macro_native_RF_claimed_exact_common": False,
            "existing_macro_common_RF_assessment_only": True,
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
    parser = argparse.ArgumentParser(
        description="Deliberate scoped macro expert financial inclusion."
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


if __name__ == "__main__":
    main()
