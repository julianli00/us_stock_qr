from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from importlib.metadata import distributions
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_adanormalhedge_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import research_program, research_weekly
from us_quant import ssd_execution_gap_shield_v1 as framework
from us_quant import ssd_expert_switching_wealth_v1 as execution
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import (
    digest_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.storage import (
    new_output_directory as new_output_directory,
)

ROOT, ID, KIND = rules.ROOT, rules.ID, execution.KIND
PROTECTED = ROOT / "config/stored-main-cost-audit-v1.json"
FOLDER = ROOT / "reports/ssd-continuous-research-v1/expert-adanormalhedge-20261107-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY, RESULTS = FOLDER / "program-readiness-v1.json", FOLDER / "results.json"
PACKET = ROOT / "evidence/expert_adanormalhedge_20261107_registration_v1.json"
PUBLIC = ROOT / "evidence/expert_adanormalhedge_20261107_results_v1.json"
CANDIDATE = {"id": ID, "weighting": "original_AdaNormalHedge_source_relative_regret_potential"}
OWN = (
    "config/expert-adanormalhedge-v1.json",
    "src/us_quant/expert_adanormalhedge_v1.py",
    "src/us_quant/ssd_expert_adanormalhedge_v1.py",
    "tests/test_expert_adanormalhedge_v1.py",
    "EXPERT_ADANORMALHEDGE_V1.md",
    "tests/conftest.py",
)


def content_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if path == PROTECTED:
        raise PermissionError("Protected configuration content access is prohibited before open.")
    physical = path.resolve()
    if physical == PROTECTED:
        raise PermissionError("Protected configuration alias content access is prohibited.")
    if not physical.is_relative_to(lifecycle.SSD):
        raise PermissionError("Keep content reads on the physical SSD research root.")
    return physical


def file_digest(path: Path) -> str:
    path = content_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    path = content_path(path)
    with path.open() as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise QuantError(f"Expected a JSON object: {path}")
    return result


def reference(path: Path) -> dict:
    path = content_path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}


def checked_reference(ref: dict) -> Path:
    path = content_path(ROOT / ref["path"])
    rules.require(file_digest(path) == ref["sha256"], "A frozen source reference changed.")
    return path


def safe_file(root: Path, relative: str, expected: str | None = None) -> Path:
    if isinstance(relative, str):
        content_path(root / relative)
    return bound(research_program.safe_file)(root, relative, expected)


def review_market(bundle: dict, root: Path):
    return bound(research_program.review_market)(bundle, root)


def verify_publication(root: Path, ref: dict, required: set[str], commit: str) -> dict:
    return bound(research_weekly.verify_publication)(root, ref, required, commit)


def protected_reference(ref: dict) -> bool:
    return any(
        Path(os.path.abspath(root / ref["path"])) == PROTECTED for root in (ROOT, ROOT.parents[1])
    )


def live_references(refs):
    return [ref for ref in refs if not protected_reference(ref)]


def unique_refs(refs):
    return execution.unique_refs(refs)


class GuardedArchiveBase:
    def __getattr__(self, name):
        if name in ("file_digest", "read_json", "safe_file"):
            return globals()[name]
        if name in ("published_inputs", "verify_path"):
            return bound(getattr(execution.base, name))
        return getattr(execution.base, name)

    def checked_reference(self, ref):
        return checked_reference(ref)

    def reference(self, path):
        return reference(path)

    def check_frozen_files(self, frozen):
        for ref in (
            frozen["code"]
            + frozen["policies"]
            + frozen["volatility_sources"]
            + frozen["benchmark_artifacts"]
            + list(frozen["market"].values())
        ):
            checked_reference(ref)


base = GuardedArchiveBase()
counts = execution.counts


def bound(function):
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def events(kind):
    return bound(framework.events)(kind)


def packet_body(frozen, registered, readiness):
    return bound(framework.packet_body)(frozen, registered, readiness)


def checked_packet(frozen):
    return bound(framework.checked_packet)(frozen)


def register():
    return bound(framework.register)()


def publication(frozen, commit):
    checked_packet(frozen)
    return base.published_inputs(frozen, commit)


def record_failure(stage, error):
    return bound(framework.record_failure)(stage, error)


def write_run(directory, output):
    return bound(framework.write_run)(directory, output)


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        checked_reference(ref)


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "Source potential work cannot overwrite authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "expert_adanormalhedge_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "order_authority": False,
        },
    )


def dependencies():
    pending = [ROOT / name for name in OWN if name.endswith(".py")]
    result = {}
    while pending:
        path = content_path(pending.pop())
        name = path.relative_to(ROOT).as_posix()
        if name in result:
            continue
        result[name] = reference(path)
        with path.open() as stream:
            tree = ast.parse(stream.read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == "us_quant":
                    names.extend("us_quant." + alias.name for alias in node.names)
            for module in names:
                if module.startswith("us_quant"):
                    relative = ROOT / "src" / module.replace(".", "/")
                    pending.extend(
                        p
                        for p in (relative.with_suffix(".py"), relative / "__init__.py")
                        if p.is_file()
                    )
    return unique_refs([*result.values(), *[reference(ROOT / name) for name in OWN]])


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(checked_reference(policy["source_inputs"]))
    closed = read_json(checked_reference(catalogue["last_completed_cycle"]))
    registered = read_json(checked_reference(catalogue["source_registered_Prod_event"]))
    rules.require(
        closed["actual_authoritative_counts"]["current_journal_event_count"] == 1542
        and registered["kind"] == "registered"
        and registered["candidate_id"] == "expert_pool_six_confidence_rated_prod_v1"
        and catalogue["expert_ids"] == list(rules.EXPERT_IDS)
        and catalogue["mechanism_groups"] == list(rules.MECHANISMS)
        and catalogue["source_clocks"]
        == registered["body"]["frozen"]["source_policy_and_expert_admission_clocks"]
        and catalogue["new_potential_regrets_scores_weights_targets_or_accounts_computed"] is False
        and catalogue["original_source_labels_or_normalization_changed"] is False
        and catalogue["old_source_learner_generator_or_account_replayed"] is False,
        "Keep the captured original source labels, complete policies and J1542 predecessor.",
    )
    refs = [
        *closed["exact_evidence_references"],
        policy["source_inputs"],
        *[v for v in catalogue.values() if isinstance(v, dict) and set(v) == {"path", "sha256"}],
        *catalogue["source_market"].values(),
        *catalogue["source_benchmark_artifacts"],
        *catalogue["fixed_teacher_unit_prior_art"],
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
    live = unique_refs(live_references(refs))
    for ref in live:
        checked_reference(ref)
    return catalogue, registered["body"]["frozen"], live


def validation_refs():
    versions = sorted(FOLDER.glob("validation-source-v*.json"))
    rules.require(bool(versions), "Actual successful synthetic evidence is required before freeze.")
    proof = read_json(versions[-1])
    xml = checked_reference(proof["test_outputs"][0])
    suites = ElementTree.parse(content_path(xml)).getroot().findall(".//testsuite")
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / name) for name in OWN]
        and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        and {
            "test_all119_matured_labels_match_independent_regret_scores",
            "test_all_zero_scores_use_uniform_without_state_reset",
            "test_full_mixed_asset_cash_and_BT",
            "test_actual_packet_defers_real_source_learning",
            "test_banned_content_guard_before_open_in_both_entrypoints",
        }.issubset(names),
        "Bind complete successful potential, clock, protection and accounting synthetic checks.",
    )
    refs = []
    for version in versions:
        record = read_json(version)
        refs.extend(
            [reference(version), *record["test_outputs"], record["validation_runner"]]
            + record["versioned_prototypes"]
        )
    return unique_refs(refs)


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial AdaNormalHedge freeze.",
        )
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1542
            and history[-1]["sha256"] == read_json(rules.POLICY)["prior_journal_head_sha256"]
            and not any(row["candidate_id"] == ID for row in history),
            "Freeze an unused identity without changing J1542 or prior failures.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, parent, refs = source_inputs()
        deps, validation = dependencies(), validation_refs()
        old_method = {
            "confidence_for_loss_fixed_at_prior_month_origin",
            "mixability_gap_floor",
            "negative_Jensen_gap_clipped",
            "normalized_confidence_regret_is_main_wealth",
            "normalized_source_regret_is_main_wealth",
        }
        frozen = {
            **{k: v for k, v in parent.items() if k not in old_method},
            "study_id": rules.STUDY,
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "execution_policy": reference(rules.POLICY),
            "execution_adapter": reference(ROOT / OWN[1]),
            "execution_adapter_role": "new_source_relative_regret_original_monthly_capital",
            "expert_pool_definition_role": "same_six_complete_policies_original_AdaNormalHedge",
            "construction": "original_AdaNormalHedge_fixed_expert_source_loss_potential",
            "decision_program_kind": KIND,
            "declared_financial_cohort_size": 1,
            "decision_information_clock": "matured_source_loss_then_NEXTpotential_forecast",
            "prior_journal_sequence": 1542,
            "prior_journal_head_sha256": history[-1]["sha256"],
            "prior_completed_configurations": 307,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[ref for ref in deps if ref["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[4]),
                ]
            ),
            "policies": unique_refs([*parent["policies"], reference(rules.POLICY)]),
            "prior_evidence": unique_refs(live_references([*parent["prior_evidence"], *refs])),
            "validation_evidence": unique_refs(
                live_references([*parent["validation_evidence"], *validation])
            ),
            "protected_archival_only_references": [
                ref for ref in parent["prior_evidence"] if protected_reference(ref)
            ],
            "protected_archival_reference_is_new_live_validation": False,
            "shared_source_learning_not_actual_main_capital": True,
            "source_regret_potential_is_actual_main_wealth_or_bound": False,
            "weight_floor_or_regret_clipping": False,
            "old_source_learner_or_account_replayed": False,
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
                "new_potential_regrets_scores_weights_targets_or_accounts_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_checkpoint_then_registration_before_source_learning")
    return reference(INTENT)


def inputs():
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        checked_reference(ref)
    rules.require(
        sys.version == intent["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == intent["package_versions"],
        "Keep the exact frozen interpreter and accepted package set.",
    )
    catalogue, _, refs = source_inputs()
    rules.require(refs == intent["source_references"], "The captured source references changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": catalogue["source_market"]}, ROOT)
    benchmark_refs = [
        ref
        for ref in catalogue["source_benchmark_artifacts"]
        if ref["path"].endswith("/results.json")
    ]
    rules.require(len(benchmark_refs) == 1, "Use the unique original completed benchmark set.")
    benchmarks = read_json(checked_reference(benchmark_refs[0]))
    rules.require(len(benchmarks["benchmarks"]) == 8, "Keep both benchmarks on all four paths.")
    return data, catalogue, frozen, benchmarks


def retain_cash_comparison(directory, actual, replay, report):
    for name, frame in (
        ("actual-frame.csv", actual.frame),
        ("expected-frame.csv", replay.frame),
        ("actual-positions.csv", actual.weights),
        ("expected-positions.csv", replay.weights),
    ):
        write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "comparison.json",
        {
            **report,
            "retained_before_acceptance_assertion": True,
            "operands": [
                reference(directory / name)
                for name in (
                    "actual-frame.csv",
                    "expected-frame.csv",
                    "actual-positions.csv",
                    "expected-positions.csv",
                )
            ],
        },
    )


class SavedOpinionFeed:
    __init__ = bound(execution.SavedOpinionFeed.__init__)
    observe = bound(execution.SavedOpinionFeed.observe)


prepare_source_targets = bound(execution.prepare_source_targets)
source_prefixes = bound(execution.source_prefixes)
evaluate_path = bound(execution.evaluate_path)
capital_prefixes = bound(execution.capital_prefixes)
evaluate = bound(execution.evaluate)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        rules.require(set(saved["paths"]) == base.PATH_KEYS, "Keep all four original funded paths.")
        for key, ref in saved["review"]["paths"].items():
            base.verify_path(ref, saved["frozen"], ID, key)
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
                "record_type": "original_adanormalhedge_four_path_results",
                "study_id": rules.STUDY,
                "candidate_id": ID,
                "actual_completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "registration": reference(PACKET),
                "private_results_sha256": file_digest(RESULTS),
                "frozen_specification_sha256": digest_json(saved["frozen"]),
                "actual_preregistration": saved["publication"],
                "source_forecast_verification": {
                    key: value
                    for key, value in proof.items()
                    if key not in ("retained_records", "completed_at")
                },
                "paths": {
                    key: {name: row[name] for name in fields} for key, row in saved["paths"].items()
                },
                "actual_authoritative_counts": counts(),
                "new_financial_rules_evaluated": 1,
                "new_formal_funded_paths": 4,
                "source_regret_potential_is_actual_main_wealth_or_bound": False,
                "raw_prices_daily_books_targets_source_states_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original AdaNormalHedge complete-expert study.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely preregistered schema3 commit.")
    result = (
        evaluate(args.commit)
        if args.action == "evaluate"
        else {"freeze": freeze, "register": register, "export": export}[args.action]()
    )
    print(json.dumps(result, indent=2, allow_nan=False))
