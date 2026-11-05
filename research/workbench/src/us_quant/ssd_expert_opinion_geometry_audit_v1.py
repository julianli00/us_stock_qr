from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from xml.etree import ElementTree

import numpy as np

from us_quant import expert_opinion_geometry_audit_v1 as law
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import read_json, utc_now

ROOT = law.ROOT
FOLDER = base.REPORTS / "expert-opinion-geometry-audit-20261105-v1"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_opinion_geometry_20261105_results_v1.json"
OWN = (
    "config/expert-opinion-geometry-audit-v1.json",
    "src/us_quant/expert_opinion_geometry_audit_v1.py",
    "src/us_quant/ssd_expert_opinion_geometry_audit_v1.py",
    "tests/test_expert_opinion_geometry_audit_v1.py",
    "EXPERT_OPINION_GEOMETRY_AUDIT_V1.md",
)


def dependencies():
    pending = [ROOT / path for path in OWN if path.endswith(".py")]
    result = {}
    while pending:
        path = pending.pop()
        key = path.relative_to(ROOT).as_posix()
        if key in result:
            continue
        result[key] = base.reference(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == "us_quant":
                    names.extend("us_quant." + alias.name for alias in node.names)
            for name in names:
                if name.startswith("us_quant"):
                    child = ROOT / "src" / (name.replace(".", "/") + ".py")
                    package = ROOT / "src" / name.replace(".", "/") / "__init__.py"
                    pending.extend(p for p in (child, package) if p.is_file())
    result.update({path: base.reference(ROOT / path) for path in OWN})
    return [result[key] for key in sorted(result)]


def source_inputs(policy):
    law.validate(policy)
    closed = read_json(base.checked_reference(policy["source_completion"]))
    expected = closed["actual_authoritative_counts"]
    if (
        expected["current_journal_event_count"] != 1062
        or expected["total_legacy_plus_ssd_completed_configurations"] != 267
        or closed["study_specific_expert_pool_size"] != 9
        or closed["status"] != "complete_both_rules_rejected_cycle_closed"
    ):
        raise QuantError("Use the exact closed nine-expert study, not a substituted source.")
    directory = base.checked_reference(policy["source_completion"]).parent
    references = {ref["path"]: ref for ref in closed["exact_evidence_references"]}
    paths = [
        directory / "actual-decision-inputs-resolved-v1.jsonl",
        directory / "information-audits-v1.json",
        ROOT / "config/lagged-spy-price-impact-expert-inclusion-v1.json",
    ]
    refs = [references[path.relative_to(ROOT).as_posix()] for path in paths]
    for ref in refs:
        base.checked_reference(ref)
    source_policy = read_json(paths[2])
    return paths[0], read_json(paths[1])["decisions"], source_policy, refs, closed


def initialize():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
            raise QuantError("Preserve the complete or partial first diagnostic freeze.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        policy = read_json(law.POLICY)
        _, _, _, refs, closed = source_inputs(policy)
        validation = read_json(FOLDER / "validation-source-v2.json")
        own = [base.reference(ROOT / path) for path in OWN]
        suites = (
            ElementTree.parse(FOLDER / "offline-audit-tests-v2.xml")
            .getroot()
            .findall(".//testsuite")
        )
        if (
            validation["implementation_and_test_inputs"] != own
            or validation["exit_code"] != 0
            or validation["ruff_returncode"] != 0
            or validation["synthetic_only"] is not True
            or not suites
            or validation["passed_tests"] != sum(int(row.get("tests", 0)) for row in suites)
            or any(
                int(row.get(key, 0)) for row in suites for key in ("failures", "errors", "skipped")
            )
        ):
            raise QuantError("Freeze the exact tested scalar-oracle and record Adapter.")
        for ref in closed["exact_evidence_references"]:
            base.checked_reference(ref)
        source_refs = [policy["source_completion"], *refs]
        for name in (
            "actual-invocation-v1.json",
            "baseline-v1.json",
            "design-decision-v1.json",
            "validation-source-v2.json",
            "offline-audit-tests-v2.xml",
            "offline-audit-tests-v2.log",
            "offline-audit-ruff-v2.log",
            "validation-source-v1.json",
            "offline-audit-tests-v1.xml",
            "offline-audit-tests-v1.log",
            "offline-audit-ruff-v1.log",
            "preflight-failure-explanation-v1.json",
        ):
            source_refs.append(base.reference(FOLDER / name))
        frozen = {
            "created_at": utc_now(),
            "study_id": law.ID,
            "criteria": policy,
            "implementation_and_test_inputs": own,
            "complete_dependency_inputs": dependencies(),
            "source_references": source_refs,
            "package_versions": sorted((d.metadata["Name"], d.version) for d in distributions()),
            "actual_geometry_calculations_executed": False,
            "new_financial_evaluations": 0,
            "order_authority": False,
        }
        write_new_receipt(INTENT, frozen)
    return base.reference(INTENT)


def _summary(cases):
    def histogram(key):
        values = [row["geometry"][key] for row in cases]
        return {str(value): values.count(value) for value in sorted(set(values))}

    selected = [
        row["geometry"]["saved_selected_pair_geometry"]
        for row in cases
        if row["geometry"]["saved_selected_pair_geometry"] is not None
    ]
    overlaps = [row["normalized_target_overlap"] for row in selected]
    if not cases or not selected:
        raise QuantError("Retain actual decision cases and saved mature pair observations.")
    return {
        "all_source_decisions_retained": len(cases),
        "all_current_opinion_pair_cases": sum(
            len(row["geometry"]["all_current_opinion_pairs"]) for row in cases
        ),
        "saved_scored_selected_pairs": len(selected),
        "unscored_selected_pair_not_applicable": len(cases) - len(selected),
        "distinct_current_opinion_histogram": histogram("distinct_current_opinions"),
        "first_eight_distinct_opinion_histogram": histogram("first_sources_distinct_opinions"),
        "exact_binary64_affine_rank_histogram": histogram("exact_binary64_affine_rank"),
        "first_eight_exact_rank_histogram": histogram("first_sources_exact_binary64_affine_rank"),
        "numerical_affine_rank_histogram": histogram("numerical_affine_rank"),
        "first_eight_numerical_rank_histogram": histogram("first_sources_numerical_affine_rank"),
        "extra_source_adds_numerical_direction_cases": sum(
            row["geometry"]["extra_source_adds_numerical_direction"] for row in cases
        ),
        "extra_source_adds_exact_binary64_direction_cases": sum(
            row["geometry"]["extra_source_adds_exact_binary64_direction"] for row in cases
        ),
        "extra_source_exact_duplicate_cases": sum(
            bool(row["geometry"]["extra_source_exact_matches"]) for row in cases
        ),
        "saved_selected_pairs_with_identical_current_opinions": sum(
            row["exact_same_opinion"] for row in selected
        ),
        "selected_pair_normalized_target_overlap": {
            "minimum": min(overlaps),
            "median": float(np.median(overlaps)),
            "maximum": max(overlaps),
        },
        "maximum_independent_scalar_gap": max(
            row["geometry"]["maximum_independent_scalar_gap"] for row in cases
        ),
        "source_scores_or_financial_accounts_replayed": False,
        "new_financial_evaluations": 0,
        "rank_or_overlap_is_independent_economic_risk": False,
        "new_membership_or_financial_acceptance_selected": False,
        "investment_objective_verified": False,
    }


def run():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        attempt = FOLDER / "actual-calculation-started-v1.json"
        if attempt.exists() or RESULTS.exists():
            raise QuantError("Preserve completed or partial geometry output without retry.")
        frozen = read_json(INTENT)
        for ref in (
            frozen["implementation_and_test_inputs"]
            + frozen["complete_dependency_inputs"]
            + frozen["source_references"]
        ):
            base.checked_reference(ref)
        policy = read_json(law.POLICY)
        trace, audits, source, _, _ = source_inputs(policy)
        calendar = sessions("2016-10-05", "2026-10-05")
        dates = [str(day.date()) for i, day in enumerate(calendar) if i == 0 or is_month_end(day)]
        if len(dates) != len(audits) or len(dates) != policy["source_decisions"]:
            raise QuantError("Retain every source decision, including initial unscored state.")
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "PID": os.getpid(),
                "diagnostic_freeze": base.reference(INTENT),
            },
        )
        cases = []
        output = FOLDER / "diagnostic-inputs-resolved-v1.jsonl"
        try:
            with trace.open() as old, output.open("x") as retained:
                for index, day in enumerate(dates):
                    lifecycle.runtime_guard()
                    before, after = old.readline(), old.readline()
                    if not before or not after:
                        raise QuantError("Missing input/resolved records cannot be omitted.")
                    current = law.retained_case(
                        json.loads(before),
                        json.loads(after),
                        audits[index],
                        index,
                        day,
                        tuple(source["expert_ids"]),
                        tuple(source["mechanism_groups"]),
                        tuple(policy["universe"]),
                    )
                    retained.write(json.dumps({"record": "diagnostic_input", **current}) + "\n")
                    retained.flush()
                    os.fsync(retained.fileno())
                    matrix = [
                        [float.fromhex(v) for v in row] for row in current["opinion_binary64_hex"]
                    ]
                    pair = current["saved_selected_pair_indices"]
                    geometry = law.inspect_opinions(
                        matrix, tuple(pair) if pair is not None else None
                    )
                    case = {"as_of": day, "source_decision_index": index, "geometry": geometry}
                    retained.write(json.dumps({"record": "diagnostic_resolved", **case}) + "\n")
                    retained.flush()
                    os.fsync(retained.fileno())
                    cases.append(case)
                if old.readline():
                    raise QuantError("Extra source records cannot be silently ignored.")
        except (QuantError, ValueError, TypeError, KeyError, IndexError, OSError) as error:
            write_new_receipt(
                FOLDER / "actual-calculation-failure-v1.json",
                {"failed_at": utc_now(), "reason": str(error), "partial_evidence_preserved": True},
            )
            raise
        result = {
            "completed_at": utc_now(),
            "study_id": law.ID,
            "diagnostic_freeze": base.reference(INTENT),
            "source_completion": policy["source_completion"],
            "retained_input_resolved_trace": base.reference(output),
            "summary": _summary(cases),
            "cases": cases,
            "new_financial_evaluations": 0,
            "order_authority": False,
        }
        write_new_receipt(RESULTS, result)
    return base.reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        result = read_json(RESULTS)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": law.ID,
                "completed_at": result["completed_at"],
                "diagnostic_freeze": result["diagnostic_freeze"],
                "source_completion": result["source_completion"],
                "summary": result["summary"],
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "raw_opinions_or_per_date_source_rows_exported": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return base.reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Read-only retained expert opinion geometry.")
    parser.add_argument("action", choices=("freeze", "run", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": initialize, "run": run, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
