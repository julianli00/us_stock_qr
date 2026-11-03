from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_target_consensus_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_seven_expert_meta_v1 as source
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT = source.base, source.ROOT
FOLDER = base.REPORTS / "expert-target-consensus-20261103-v1"
PRIMARY = FOLDER / "aggregation-implementation-freeze-v1.json"
INVOCATION = FOLDER / "actual-financial-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_target_consensus_20261103_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_target_consensus_20261103_results_v1.json"
PROGRESS = ROOT / "reports/ssd-november-research-20261103-v1/progress.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-target-consensus-v1.json",
    "src/us_quant/expert_target_consensus_v1.py",
    "src/us_quant/ssd_expert_target_consensus_v1.py",
    "src/us_quant/november_research_lifecycle_v1.py",
    "tests/test_expert_target_consensus_v1.py",
    "tests/test_ssd_expert_target_consensus_v1.py",
    "tests/test_november_research_lifecycle_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
    "tests/conftest.py",
)
reference, unique_refs = source.reference, source.unique_refs


def dependencies() -> list[dict]:
    walker = source.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__, {**globals(), "prior": source, "ast": ast}, argdefs=walker.__defaults__
    )()


def events(kind: str) -> list[dict]:
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["candidate_id"] == ID and row["kind"] == kind
    ]


def progress(phase: str, **fields) -> None:
    base.storage_guard()
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Current journal-derived progress counts cannot be overridden.")
    write_json(
        PROGRESS,
        {
            "updated_at_utc": utc_now(),
            "phase": phase,
            "authorization_id": "november-research-20261103-v1",
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def check_primary() -> dict:
    if file_digest(PRIMARY) != "d229a5cb09ac5432e2a508cb89fcac91f69e98e89b69e132a47efd45a2065a44":
        raise QuantError("Preserve the actual first-stage economic-law freeze.")
    primary = read_json(PRIMARY)
    for row in primary["aggregation_implementation_and_test_inputs"]:
        base.checked_reference(row)
    rules.validate(read_json(rules.POLICY))
    return primary


def cached_inputs():
    check_primary()
    data, experts, _, _, refs, clocks, admissions, benchmarks = source.source_inputs()
    policy = read_json(rules.POLICY)
    packet = read_json(base.checked_reference(policy["source_pool_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != policy["source_pool_specification_sha256"]
        or tuple(expert.identity.identifier for expert in experts) != rules.EXPERT_IDS
    ):
        raise QuantError("Keep the exact seven technically admitted cached policies.")
    source.check_frozen(parent)
    return data, experts, parent, unique_refs([*refs, *admissions]), clocks, benchmarks


def validation_references() -> list[dict]:
    xml = FOLDER / "offline-orchestration-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if not suites or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Complete the orchestration checks before freezing real inputs.")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    if not {
        "test_new_orchestrator_keeps_old_guards_and_pure_monthly_accounting_unchanged",
        "test_import_capture_includes_new_and_old_actual_helpers_without_mutating_sources",
        "test_independent_real_record_checker_handles_all_current_synthetic_decisions",
        "test_retention_or_oracle_mismatch_stops_preparation",
        "test_unresolved_preparation_attempt_is_never_replayed",
        "test_failed_publication_never_starts_a_main_account",
        "test_completed_account_recovery_reads_receipt_without_running_any_account",
        "test_november_progress_never_overwrites_october_global_progress",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
        "test_real_initial_target_is_not_backdated_to_a_month_end",
    }.issubset(names):
        raise QuantError("The exact source, oracle, execution and publication checks are required.")
    proof = read_json(FOLDER / "orchestration-validation-source-v1.json")
    expected = [reference(ROOT / path) for path in OWN]
    outputs = [reference(xml), reference(FOLDER / "offline-orchestration-tests-v1.log")]
    if (
        proof["implementation_and_test_inputs"] != expected
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("The successful tests must bind these exact implementation bytes.")
    return [*outputs, reference(FOLDER / "orchestration-validation-source-v1.json")]


def initialize() -> dict:
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "financial-freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(
            FOLDER / "financial-freeze-attempt-v1.json",
            {"started_at": utc_now(), "actual_invocation": reference(INVOCATION)},
        )
        data, experts, parent, refs, clocks, _ = cached_inputs()
        history = base.journal_events(base.JOURNAL)
        invocation = read_json(INVOCATION)
        if (
            len(history) != 675
            or history[-1]["sha256"] != read_json(rules.POLICY)["prior_journal_head_sha256"]
            or invocation["actual_sequence"] != 64948
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
            or digest_bytes(invocation["data"]["content"].encode())
            != "bed74a2cf3f1db748bfe0affe7d1ccca6cb2ab84bd3f4294822edf15348b3346"
        ):
            raise QuantError("Freeze against actual unchanged history and scheduled delivery.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": read_json(rules.POLICY)["source_pool_policy"],
            "construction": "exact_current_complete_expert_target_consensus",
            "authorization_id": "november-research-20261103-v1",
            "authorization_record_sha256": lifecycle.AUTHORIZATION_SHA256,
            "actual_invocation": reference(INVOCATION),
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
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
                    reference(PRIMARY),
                    reference(INVOCATION),
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    *validation_references(),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "decision_information_clock": rules.CLOCK,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "new_source_requests": 0,
        }
        source.check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "source_references": refs,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "source_sessions": len(data.close),
                "source_experts": len(experts),
                "actual_new_targets_or_accounts_computed": False,
                "candidate_registered": False,
                "order_authority": False,
            },
        )
        progress("consensus_financial_implementation_frozen", next_step="register_before_targets")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["complete_dependency_inputs"] + intent["source_references"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(pair) for pair in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen isolated package environment changed.")
    source.check_frozen(intent["frozen"])
    data, experts, _, refs, clocks, benchmarks = cached_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Cached source provenance or expert availability clocks changed.")
    return data, experts, intent["frozen"], benchmarks


def verify_preparation(data, experts, targets, audits, records: list[dict]) -> dict:
    dates = source.rules.decision_sessions(data.close.index, read_json(source.rules.POLICY))
    if (
        list(targets) != [ID]
        or list(audits) != [ID]
        or len(records) != 2 * len(dates)
        or not targets[ID].index.equals(data.close.index)
        or not targets[ID].columns.equals(data.close.columns)
        or not targets[ID].dropna(how="all").index.equals(dates)
        or len(audits[ID]) != len(dates)
    ):
        raise QuantError("Preserve the complete initial/month-end lattice and every input record.")
    maximum = 0.0
    for index, day in enumerate(dates):
        observed = [expert.observe(day, 126) for expert in experts]
        before, after = records[2 * index : 2 * index + 2]
        audit = audits[ID][index]
        if (
            before["record"] != "decision_input"
            or after["record"] != "decision_resolved"
            or before["decision_index"] != index
            or after["decision_index"] != index
            or before["as_of"] != str(day.date())
            or before["information_clock"] != rules.CLOCK
            or after["audit"] != audit
            or after["asset_order"] != list(data.close.columns)
            or after["combined_target"] != targets[ID].loc[day].tolist()
        ):
            raise QuantError("Actual retained inputs and resolved decisions differ.")
        for saved, item in zip(before["observations"], observed, strict=True):
            if (
                saved["id"] != item.identity.identifier
                or saved["mechanism_group"] != item.identity.mechanism_group
                or saved["introduced_at"] != item.identity.introduced_at
                or saved["target_known_at"] != str(item.target_known_at.date())
                or saved["asset_order"] != list(item.target.index)
                or saved["target"] != item.target.tolist()
                or saved["target_binary64_hex"] != [float(v).hex() for v in item.target]
                or saved["source_NET_observations"] != len(item.net_returns)
                or saved["source_score_mature"] != item.score_mature
            ):
                raise QuantError("Retain exact source target values before the decision.")
        expected, chosen = rules.independent_target(observed)
        ratios = [[float(v).as_integer_ratio() for v in item.target] for item in observed]
        denominator = max(bottom for row in ratios for _, bottom in row)
        rows = [[top * (denominator // bottom) for top, bottom in row] for row in ratios]
        for row in rows:
            row.append(denominator - sum(row))
        exact_totals = [
            Fraction(
                sum(abs(a - b) for other in rows for a, b in zip(row, other, strict=True)),
                denominator,
            )
            for row in rows
        ]
        if audit["consensus"]["objective_exact_fractions"] != [
            {"numerator": str(v.numerator), "denominator": str(v.denominator)} for v in exact_totals
        ]:
            raise QuantError("Every recorded objective must match independent integer geometry.")
        expected_weights = np.array(
            [1 / len(chosen) if name in chosen else 0 for name in rules.EXPERT_IDS]
        )
        if (
            chosen != audit["consensus"]["selected_expert_ids"]
            or not np.array_equal(expected_weights, np.array(audit["expert_weights"]))
            or audit["decision_information_clock"] != rules.CLOCK
        ):
            raise QuantError("The independent integer-coordinate medoid disagrees.")
        maximum = max(maximum, float(abs(expected - targets[ID].loc[day]).max()))
    if maximum > 1e-12:
        raise QuantError("The independent exact-target average differs from issued composition.")
    return {
        "decision_count": len(dates),
        "retained_input_resolved_records": len(records),
        "maximum_independent_integer_target_difference": maximum,
    }


def prepare(data, experts, frozen) -> dict:
    if PREPARED.exists():
        result = read_json(PREPARED)
        if result["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior target preparation.")
        for row in result["entries"][ID].values():
            base.checked_reference(row)
        return result["entries"][ID]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve the first unfinished preparation instead of replaying it.")
    registered = events("registered")
    if len(registered) != 1 or registered[0]["body"] != {"candidate": CANDIDATE, "frozen": frozen}:
        raise QuantError("Actual registration must precede every new real consensus target.")
    lifecycle.runtime_guard()
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
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
    records = [json.loads(line) for line in log.read_text().splitlines()]
    verification = verify_preparation(data, experts, targets, audits, records)
    if verification["decision_count"] != 121:
        raise QuantError("Keep all 121 original formal decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[ID][position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early[ID].equals(targets[ID].loc[:day]) or earlier[ID] != audits[ID][: position + 1]:
            raise QuantError("Future observations changed an earlier consensus decision.")
        prefixes.append(str(day.date()))
    active = targets[ID].dropna(how="all")
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        rows = pd.concat(
            [active.loc[:anchor].tail(1), active.loc[anchor + pd.Timedelta(days=1) : window["end"]]]
        )
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        if rows.empty or len(factors) < 3 or rows.index[0] > anchor:
            raise QuantError("Each original funded horizon needs genuine known factor coverage.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    directory = FOLDER / "targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets[ID].to_csv(float_format="%.17g"))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **verification,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records": reference(log),
            "information_audits": reference(FOLDER / "information-audits-v1.json"),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "independent_forward_validation": False,
        },
    )
    entries = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": {ID: entries}}
    )
    return entries


def record_failure(stage: str, error: Exception) -> None:
    interrupted = lifecycle.phase(datetime.now(timezone.utc)) != "research"
    if (
        events("registered")
        and not events("review")
        and not events("technical_failure")
        and not interrupted
    ):
        base.append_event(
            base.JOURNAL,
            "technical_failure",
            ID,
            {
                "stage": stage,
                "reason": str(error),
                "partial_outputs_retained": True,
                "candidate_accounts_started": bool(events("path_started")),
            },
        )
    path = FOLDER / f"{stage}-failure-v1.json"
    if not path.exists():
        write_new_receipt(
            path,
            {
                "failed_at": utc_now(),
                "stage": stage,
                "reason": str(error),
                "authorization_interrupted": interrupted,
                "partial_outputs_retained": True,
            },
        )
    progress(
        "consensus_interrupted" if interrupted else "consensus_technical_failure",
        failure=str(error),
        next_step="preserve_without_automatic_identity_retry",
    )


def register() -> dict:
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if events("technical_failure"):
            raise QuantError("A terminal financial version cannot be retried.")
        if not events("registered"):
            base.append_event(
                base.JOURNAL, "registered", ID, {"candidate": CANDIDATE, "frozen": frozen}
            )
        elif events("registered")[0]["body"] != {"candidate": CANDIDATE, "frozen": frozen}:
            raise QuantError("The actual registered policy cannot change.")
        try:
            if not events("target_proof"):
                prepared = prepare(data, experts, frozen)
                base.append_event(
                    base.JOURNAL,
                    "target_proof",
                    ID,
                    {
                        "registered_event_sha256": events("registered")[0]["sha256"],
                        **prepared,
                    },
                )
            entry = source.common.packet_entry(CANDIDATE, frozen)
            source.common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [entry],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 1,
                    "candidate_accounts_or_metrics_computed": False,
                    "order_authority": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                },
                ("created_at",),
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("preparation", error)
            raise
        progress(
            "consensus_registered_waiting_remote_publication", next_step="publish_exact_packet"
        )
    return reference(PACKET)


def publication(frozen, commit: str) -> dict:
    source.check_frozen(frozen)
    packet = read_json(PACKET)
    if packet["entries"] != [source.common.packet_entry(CANDIDATE, frozen)]:
        raise QuantError("Only the complete actually registered consensus packet may be published.")
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    if contract["bundle_manifest_sha256"] != base.BUNDLE_SHA256:
        raise QuantError("The original imported publication boundary changed.")
    manifest = read_json(
        base.checked_reference({"path": "IMPORT_MANIFEST.json", "sha256": base.BUNDLE_SHA256})
    )
    refs = {row["path"]: row["sha256"] for row in frozen["code"] + frozen["policies"]}
    explicit = {path for path in refs if "research/workbench/" + path in contract["files"]}
    inherited = set(refs) - explicit
    if any(manifest["file_sha256"].get(path) != refs[path] for path in inherited):
        raise QuantError(
            "A scientific dependency lacks explicit approval or its original import pin."
        )
    result = source.verify_publication(ROOT, reference(PACKET), explicit, commit)
    for path in inherited:
        blob = base.git(ROOT, ["show", f"{commit}:research/workbench/{path}"])
        if digest_bytes(blob) != refs[path]:
            raise QuantError("An inherited dependency differs in the actual published commit.")
    return {
        **result,
        "unchanged_pinned_import_files_verified": len(inherited),
        "original_bundle_manifest_sha256": base.BUNDLE_SHA256,
    }


def evaluate_path(data, targets, frozen, window, scenario, benchmarks) -> dict:
    key = f"{window['years']}y-{scenario['id']}"
    complete = [e for e in events("path_completed") if e["body"]["path_key"] == key]
    if complete:
        return base.verify_path(complete[0]["body"]["receipt"], frozen, ID, key)
    starts = [e for e in events("path_started") if e["body"]["path_key"] == key]
    if starts:
        prior = starts[-1]
        saved = ROOT / prior["body"]["directory"] / "complete.json"
        if saved.exists():
            result = base.verify_path(reference(saved), frozen, ID, key)
            base.append_event(
                base.JOURNAL,
                "path_completed",
                ID,
                {
                    "path_key": key,
                    "receipt": reference(saved),
                    "recovered_actual_complete_receipt": True,
                },
            )
            return result
        raise QuantError("Preserve an unfinished account instead of automatically rerunning it.")
    lifecycle.runtime_guard()
    registered = events("registered")
    fenced = [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == "publication" and row["body"].get("study_id") == frozen["study_id"]
    ]
    if len(registered) != 1 or registered[0]["body"]["frozen"] != frozen or len(fenced) != 1:
        raise QuantError("A new funded path requires its actual frozen registration and fence.")
    directory = FOLDER / "accounts" / key
    base.append_event(
        base.JOURNAL,
        "path_started",
        ID,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    new_output_directory(directory)
    policy = read_json(base.checked_reference(frozen["execution_policy"]))
    result, issued, decisions = execution.run(
        data,
        targets,
        execution.monthly.CANDIDATES[0],
        policy,
        window["start"],
        window["end"],
        scenario["cost_bps_per_side"],
        scenario["delay_sessions"],
    )
    feedback = execution.original.verify_feedback(
        data, targets, result, issued, decisions, window, scenario
    )
    independent = base.independent_equity(
        data,
        issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(result.frame["return"], result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = (lookup[window["years"], scenario["id"], symbol] for symbol in ("SPY", "QQQ"))
    criteria = base.gates(own, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    statistics = base.execution_statistics(result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": result.frame,
        "weights.csv": result.weights,
        "issued.csv": issued,
        "independent-bt.csv": independent,
    }
    for name, value in frames.items():
        write_text_atomic(directory / name, value.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": decisions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": statistics,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"] - spy["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"] - qqq["cagr"],
        "independent_bt_max_equity_difference_usd": difference,
        "independent_monthly_feedback": feedback,
        "artifacts": [reference(directory / name) for name in (*frames, "decisions.json")],
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        ID,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def evaluate(commit: str) -> dict:
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, _, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal version without evaluation.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            reviews = events("review")
            if (
                saved["frozen"] != frozen
                or len(reviews) != 1
                or reviews[0]["body"] != saved["review"]
            ):
                raise QuantError("Preserve conflicting complete results without replay.")
            for row in saved["review"]["paths"].values():
                base.checked_reference(row)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            e
            for e in base.journal_events(base.JOURNAL)
            if e["kind"] == "publication" and e["body"].get("study_id") == frozen["study_id"]
        ]
        if not fences:
            if events("path_started"):
                raise QuantError("An account cannot predate its actual publication fence.")
            base.append_event(
                base.JOURNAL,
                "publication",
                None,
                {
                    "study_id": frozen["study_id"],
                    "verification": published,
                    "candidate_accounts_previously_started": False,
                },
            )
        else:
            if len(fences) != 1 or fences[0]["body"]["verification"]["published_commit"] != commit:
                raise QuantError("Preserve the unique actual original publication fence.")
            published = fences[0]["body"]["verification"]
        targets = pd.read_csv(
            base.checked_reference(events("target_proof")[0]["body"]["targets"]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        try:
            protocol = read_json(base.PROTOCOL)
            paths = {}
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, targets, frozen, window, scenario, benchmarks)
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {
                    key: reference(FOLDER / "accounts" / key / "complete.json") for key in paths
                },
                "all_four_paths_same_candidate_and_frozen_parameters": True,
                "independent_forward_validation": False,
                "order_authority": False,
            }
            if not events("review"):
                base.append_event(base.JOURNAL, "review", ID, review)
            else:
                recorded = events("review")[0]["body"]
                if recorded["paths"] != review["paths"] or recorded["status"] != review["status"]:
                    raise QuantError("The original review cannot change on receipt recovery.")
                review = recorded
            write_new_receipt(
                RESULTS,
                {
                    "completed_at": utc_now(),
                    "frozen": frozen,
                    "publication": published,
                    "review": review,
                    "paths": paths,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding", error)
            raise
        progress("consensus_four_paths_complete", next_step="publish_complete_results")
    return reference(RESULTS)


def public_result() -> dict:
    data = read_json(RESULTS)
    frozen, paths = data["frozen"], data["paths"]
    if set(paths) != base.PATH_KEYS or data["review"]["status"] not in {
        "rejected",
        "historical_point_gates_passed",
    }:
        raise QuantError("Publish only the complete original four-path conclusion.")
    if PUBLIC.exists():
        if read_json(PUBLIC)["private_results_sha256"] != file_digest(RESULTS):
            raise QuantError("The completed private result changed after its public projection.")
        return reference(PUBLIC)
    if len(events("review")) != 1 or events("review")[0]["body"] != data["review"]:
        raise QuantError("Require the actual complete review, not an unregistered summary.")
    for key, row in data["review"]["paths"].items():
        if base.verify_path(row, frozen, ID, key) != paths[key]:
            raise QuantError("Public output must match every actual completed path receipt.")
    controls = read_json(base.checked_reference(read_json(rules.POLICY)["saved_controls"]))
    matched = {}
    for label, name in (
        ("primary", "expert_pool_seven_rolling_pair_v1"),
        ("secondary_context", "expert_pool_seven_equal_weight_v1"),
    ):
        old = controls["candidates"][name]["paths"]
        if set(old) != base.PATH_KEYS:
            raise QuantError("The saved complete controls cannot be shortened or relabelled.")
        matched[label] = {
            "candidate_id": name,
            "actual_old_paths_replayed": False,
            "paths": {
                key: {"metrics": row["metrics"], "execution": row["execution"]}
                for key, row in old.items()
            },
        }
    primary = matched["primary"]["paths"]
    diagnostic = all(
        row["metrics"]["max_drawdown"] < primary[key]["metrics"]["max_drawdown"]
        and row["execution"]["annualized_two_sided_turnover"]
        <= primary[key]["execution"]["annualized_two_sided_turnover"]
        for key, row in paths.items()
    )
    packet = {
        "schema_version": 1,
        "record_type": "november_consensus_complete_four_path_results",
        "study_id": frozen["study_id"],
        "candidate_id": ID,
        "actual_completed_at": data["completed_at"],
        "status": data["review"]["status"],
        "registration": reference(PACKET),
        "frozen_specification_sha256": digest_json(frozen),
        "private_results_sha256": file_digest(RESULTS),
        "actual_preregistration": data["publication"],
        "protocol": {"path": "config/dual-benchmark-v1.json", "sha256": base.PROTOCOL_SHA256},
        "paths": {
            key: {
                name: row[name]
                for name in (
                    "path_key",
                    "metrics",
                    "execution",
                    "gates",
                    "all_point_gates_pass",
                    "independent_bt_max_equity_difference_usd",
                    "independent_monthly_feedback",
                )
            }
            for key, row in paths.items()
        },
        "saved_matched_controls": matched,
        "directional_robustness_premise_supported": diagnostic,
        "directional_diagnostic_is_a_qualification_gate": False,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "new_financial_rules_evaluated": 1,
        "new_funded_paths": 4,
        "new_source_requests": 0,
        "source_policy_generators_or_accounts_replayed": False,
        "raw_source_targets_daily_books_private_intents_or_native_controls_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    source.common.retain_receipt(PUBLIC, packet)
    return reference(PUBLIC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen November consensus research; no orders.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Evaluation needs its actual remote preregistration commit.")
    actions = {"freeze": initialize, "register": register, "export": public_result}
    result = evaluate(args.commit) if args.action == "evaluate" else actions[args.action]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
