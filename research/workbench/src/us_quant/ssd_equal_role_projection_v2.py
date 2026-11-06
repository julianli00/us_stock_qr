from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import equal_role_projection_proof_v2 as proof
from us_quant import equal_role_recovery_policy_v2 as rules
from us_quant import ssd_equal_role_projection_v1 as failed
from us_quant.candidate_receipts import write_new_receipt
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, template, seven, financial = (
    failed.base,
    failed.ROOT,
    failed.template,
    failed.seven,
    failed.financial,
)
lifecycle, reference, unique_refs = failed.lifecycle, failed.reference, failed.unique_refs
current_counts = failed.current_counts
FOLDER = base.REPORTS / "equal-role-projection-20261106-v2"
INTENT = FOLDER / "financial-implementation-freeze-v2.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v2.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/equal_role_projection_20261106_results_v2.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/equal_role_projection_20261106_registration_v2.json"}
OWN = (
    "config/equal-role-projection-v2.json",
    "src/us_quant/equal_role_recovery_policy_v2.py",
    "src/us_quant/ssd_equal_role_projection_v2.py",
    "src/us_quant/equal_role_projection_proof_v2.py",
    "tests/test_equal_role_projection_v2.py",
    "tests/test_equal_role_projection_proof_v2.py",
    "EQUAL_ROLE_PROJECTION_V2.md",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def dependencies():
    return bound(failed._dependencies)()


def progress(phase, **fields):
    actual = current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Versioned proof recovery cannot substitute stale financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "equal_role_projection_V2_" + phase,
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
    inputs = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, parent, refs, clocks, benchmarks = failed.source_inputs()
    rules.require(
        len(failed.events("technical_failure", failed.rules.ID)) == 1
        and not failed.events("path_started", failed.rules.ID)
        and inputs["failed_V1_reopened"] is False
        and inputs["economic_rule_or_tolerance_changed"] is False
        and inputs["new_real_V2_verification_or_accounts_computed"] is False,
        "Keep failed V1 terminal and require a new independent V2 verification.",
    )
    failed_intent = read_json(base.checked_reference(inputs["failed_V1_intent"]))
    recovered = read_json(base.checked_reference(inputs["proof_recovery_synthetic_validation"]))
    refs = unique_refs(
        [
            *refs,
            policy["source_inputs"],
            *failed_intent["frozen"]["code"],
            *failed_intent["frozen"]["policies"],
            *recovered["test_outputs"],
            *recovered["versioned_prototypes"],
            *[v for v in inputs.values() if isinstance(v, dict) and "path" in v and "sha256" in v],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v2.py",
                    "actual-invocation-v2.json",
                    "design-decision-v2.json",
                    "preservation-baseline-v2.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v2.json").exists(),
        "Preserve the first complete or partial V2 freeze.",
    )
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1466 and not any(e["candidate_id"] in IDS for e in history),
            "Freeze a new V2 identity at the actual J1466 baseline.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v2.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v2.json")
        suites = (
            ElementTree.parse(base.checked_reference(validation["test_outputs"][0]))
            .getroot()
            .findall(".//testsuite")
        )
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and suites
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful V2 identity and repaired proof-Interface wiring checks.",
        )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "same_role_rule_independent_cached_projection_proof_recovery_V2",
            "prior_journal_sequence": 1466,
            "prior_journal_head_sha256": read_json(rules.POLICY)["prior_journal_head_sha256"],
            "prior_completed_configurations": 301,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[-1]),
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
                    reference(FOLDER / "validation-source-v2.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "failed_V1_reopened": False,
            "cached_targets_verified_not_regenerated": True,
            "economic_rule_or_tolerance_changed": False,
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
                "real_V2_cache_verification_or_accounts_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_V2_before_real_verification")
    return reference(INTENT)


def retained_pairs(path):
    with path.open() as stream:
        while True:
            before = stream.readline()
            if not before:
                break
            after = stream.readline()
            rules.require(bool(after), "A partial retained pair cannot receive a V2 proof.")
            yield json.loads(before), json.loads(after)


def prepare(data, experts, frozen):
    rules.require(
        len(events("registered", rules.ID)) == 1
        and not PREPARED.exists()
        and not (FOLDER / "verification-attempt-v2.json").exists(),
        "Register V2 before its one real retained-target verification.",
    )
    write_new_receipt(FOLDER / "verification-attempt-v2.json", {"started_at": utc_now()})
    source_catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    target_ref, tape_ref = (
        source_catalogue["cached_partial_targets"],
        source_catalogue["cached_partial_records"],
    )
    targets = pd.read_csv(
        base.checked_reference(target_ref),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        targets.index.equals(data.close.index) and targets.columns.equals(data.close.columns),
        "A retained file must keep the original full market coordinate lattice.",
    )
    verified = proof.verify_cached(
        targets, retained_pairs(base.checked_reference(tape_ref)), frozen
    )
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = failed.source.prior.previous_session(window["start"])
        rows = targets.dropna().loc[: window["end"]]
        rows = pd.concat([rows.loc[:anchor].tail(1), rows.loc[anchor + pd.Timedelta(days=1) :]])
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        rules.require(len(factors) >= 3, "Keep actual factor participation in both formal windows.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    original_inputs = read_json(base.checked_reference(source_catalogue["original_source_inputs"]))
    flat = pd.read_csv(
        base.checked_reference(original_inputs["cached_flat_six_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        flat.index.equals(targets.index)
        and flat.columns.equals(targets.columns)
        and not flat.equals(targets),
        "A duplicate economic target cannot be funded.",
    )
    verified.update(
        {
            "formal_factor_coverage": coverage,
            "materially_different_cached_flat_six_origins": int(
                (
                    (targets.dropna() - flat.loc[targets.dropna().index]).abs().max(axis=1) > 1e-12
                ).sum()
            ),
            "retained_target_artifact_origin": target_ref,
            "actual_input_resolved_records": tape_ref,
        }
    )
    proof_path = FOLDER / "verified-target-proof-v2.json"
    write_new_receipt(proof_path, verified)
    entries = {rules.ID: {"targets": target_ref, "proof": reference(proof_path)}}
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "entries": entries})
    return entries


def register():
    return bound(template.register)()


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    return bound(failed.source.prior.prior.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        recovered = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
        original = read_json(base.checked_reference(recovered["original_source_inputs"]))
        control = read_json(base.checked_reference(original["cached_flat_six_results"]))
        rules.require(set(result["paths"]) == base.PATH_KEYS, "Retain all four original paths.")
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 2,
                "study_id": read_json(rules.POLICY)["study_id"],
                "candidate_id": rules.ID,
                "completed_at": saved["completed_at"],
                "status": result["review"]["status"],
                "all_four_original_paths": result["paths"],
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "cached_flat_six_control": original["cached_flat_six_results"],
                "new_minus_cached_flat_six": {
                    key: {
                        metric: path["metrics"][metric]
                        - control["all_four_original_paths"][key]["metrics"][metric]
                        for metric in ("cagr", "sharpe", "max_drawdown")
                    }
                    for key, path in result["paths"].items()
                },
                "failed_V1_reopened": False,
                "cached_targets_verified_not_regenerated": True,
                "economic_rule_or_tolerance_changed": False,
                "source_learning_state_reexecuted": False,
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
    if args.action == "evaluate" and not args.commit:
        parser.error("Use an actual remotely verified V2 preregistration SHA.")
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
