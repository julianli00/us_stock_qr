from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.bt_audit import independent_equity
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import (
    POLICY as PROTOCOL,
)
from us_quant.dual_benchmark_v1 import (
    POLICY_SHA256 as PROTOCOL_SHA256,
)
from us_quant.dual_benchmark_v1 import (
    REPORTS,
    execution_statistics,
    gates,
    research_job,
    runtime_guard,
)
from us_quant.equity_first_regimes_v1 import (
    CANDIDATES,
    POLICY,
    build_targets,
    independent_targets,
    simulate_targets,
    validate,
)
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.metrics import performance
from us_quant.research_program import engine_fingerprint, review_market, safe_file
from us_quant.research_weekly import git, verify_publication
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
from us_quant.volatility_term_risk import SOURCES, load_terms

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent.parent
JOURNAL = ROOT / "runtime/ssd-continuous-research-v1/events"
TARGETS = ROOT / "data/ssd-equity-first-20261027-v1"
ACCOUNTS = REPORTS / "equity-first-20261027-v1"
PACKET = ROOT / "evidence/equity_first_20261027_registration_v1.json"
BENCHMARKS = REPORTS / "benchmarks-20261027-v1"
LEGACY_SHA256 = "ed2c075d0819e1ff8bb2d6920910128e0dd2ed4be46f920fef575129d64c9317"
LEGACY_ENGINE_SHA256 = "a4be8219834c412e0c1e311180dc9ce12c5b0d60b32670a9edd4906278a21c05"
BUNDLE_SHA256 = "3b2ef6789348ac88b8fdf82805415f518da85bb72c1d652735e7aeaaa63666b6"
CODE_MODULES = (
    "ssd_research_v1",
    "equity_first_regimes_v1",
    "dual_benchmark_v1",
    "cash_funded_accounting_v2",
    "backtest",
    "bt_audit",
    "metrics",
    "data",
    "storage",
    "calendar",
    "dual_horizon",
    "factor_validation",
    "candidate_receipts",
    "semiconductor_trend_transfer",
    "multifactor_stability",
    "portfolio_refinement",
    "volatility_term_risk",
    "research_program",
    "research_weekly",
)
PATH_KEYS = {"10y-base", "10y-stress", "5y-base", "5y-stress"}


def storage_guard() -> None:
    mount = ROOT.parents[3]
    if (
        not os.path.ismount(mount)
        or ROOT.resolve() != ROOT
        or ROOT.stat().st_dev != mount.stat().st_dev
    ):
        raise QuantError(
            "The actual SSD disappeared; never create replacement internal-disk output."
        )


def journal_events(directory: Path) -> list[dict]:
    if directory.is_symlink() or directory.resolve() != directory:
        raise QuantError("The new journal must remain in its actual private directory.")
    paths = list(directory.glob("*.json"))
    if any(not re.fullmatch(r"[0-9]+\.json", path.name) for path in paths):
        raise QuantError("An unexpected journal file cannot be silently ignored.")
    result = []
    previous = "0" * 64
    for sequence, path in enumerate(sorted(paths, key=lambda item: int(item.stem)), start=1):
        if path.is_symlink() or int(path.stem) != sequence:
            raise QuantError("The new research journal has a gap or unsafe path.")
        event = read_json(path)
        body = {key: value for key, value in event.items() if key != "sha256"}
        if (
            event.get("schema_version") != 1
            or event.get("sequence") != sequence
            or event.get("previous_sha256") != previous
            or event.get("sha256") != digest_json(body)
        ):
            raise QuantError("The new research journal was revised; do not reset it.")
        clock = pd.Timestamp(event["at"])
        if (
            clock.tzinfo is None
            or clock > pd.Timestamp(utc_now())
            or (result and clock < pd.Timestamp(result[-1]["at"]))
        ):
            raise QuantError("Journal events cannot invent or backdate their actual clock.")
        result.append(event)
        previous = event["sha256"]
    return result


def append_event(directory: Path, kind: str, candidate_id: str | None, body: dict) -> dict:
    if kind not in {
        "baseline",
        "registered",
        "target_proof",
        "technical_failure",
        "publication",
        "path_started",
        "path_interrupted",
        "path_completed",
        "review",
    } or (
        candidate_id is not None
        and (
            not isinstance(candidate_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]+", candidate_id)
        )
    ):
        raise QuantError("Unknown journal event or candidate identity.")
    if (kind in {"baseline", "publication"}) != (candidate_id is None):
        raise QuantError("Actual candidate events need their own registered identity.")
    events = journal_events(directory)
    if kind == "baseline" and events:
        raise QuantError("The original new-lane baseline cannot be reset.")
    if kind != "baseline" and (not events or events[0]["kind"] != "baseline"):
        raise QuantError("Record the preserved legacy baseline before new research.")
    if kind in {"registered", "target_proof", "technical_failure", "review"} and any(
        event["kind"] == kind and event["candidate_id"] == candidate_id for event in events
    ):
        raise QuantError(
            "An actual candidate event already exists; recover rather than duplicate it."
        )
    if kind not in {"baseline", "registered", "publication"} and not any(
        event["kind"] == "registered" and event["candidate_id"] == candidate_id for event in events
    ):
        raise QuantError("No account, proof or failure may invent a missing registration.")
    if kind in {"path_started", "path_completed", "path_interrupted", "review"} and any(
        event["kind"] == "technical_failure" and event["candidate_id"] == candidate_id
        for event in events
    ):
        raise QuantError("A terminal technical failure cannot be silently resumed as success.")
    if kind == "technical_failure" and any(
        event["kind"] == "review" and event["candidate_id"] == candidate_id for event in events
    ):
        raise QuantError("A completed review cannot be retrospectively reclassified.")
    if kind.startswith("path_") and body.get("path_key") not in PATH_KEYS:
        raise QuantError("Require an actual original formal path identity.")
    if kind == "review":
        complete = {
            event["body"]["path_key"]
            for event in events
            if event["kind"] == "path_completed" and event["candidate_id"] == candidate_id
        }
        if complete != PATH_KEYS or set(body.get("paths", {})) != PATH_KEYS:
            raise QuantError("No completed review without all four actual path receipts.")
    event = {
        "schema_version": 1,
        "sequence": len(events) + 1,
        "at": utc_now(),
        "kind": kind,
        "candidate_id": candidate_id,
        "body": body,
        "previous_sha256": events[-1]["sha256"] if events else "0" * 64,
    }
    event["sha256"] = digest_json(event)
    if directory == JOURNAL:
        storage_guard()
    write_new_receipt(directory / f"{event['sequence']:012d}.json", event)
    return event


def reference(path: Path) -> dict:
    relative = path.relative_to(ROOT).as_posix()
    return {"path": relative, "sha256": file_digest(safe_file(ROOT, relative))}


def checked_reference(value: dict) -> Path:
    return safe_file(ROOT, value["path"], value["sha256"])


def check_frozen_files(frozen: dict) -> None:
    for item in (
        frozen["code"]
        + frozen["policies"]
        + frozen["volatility_sources"]
        + frozen["benchmark_artifacts"]
        + list(frozen["market"].values())
    ):
        checked_reference(item)


def legacy_snapshot(policy: dict) -> dict:
    ledger = ROOT / "runtime/research-program.sqlite3"
    if file_digest(ledger) != LEGACY_SHA256 or engine_fingerprint() != LEGACY_ENGINE_SHA256:
        raise QuantError("The accepted immutable legacy ledger or engine changed.")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        factors = {
            identifier: {
                "id": identifier,
                "semantic_sha256": digest,
                "registered_at": at,
                "family": json.loads(body)["family"],
            }
            for identifier, digest, at, body in db.execute(
                "SELECT id,semantic_sha,registered_at,body FROM factors ORDER BY id"
            )
        }
    selected = [factors[identifier] for identifier in policy["factor_ids"]]
    if len({row["family"] for row in selected}) < policy["minimum_economic_factor_families"]:
        raise QuantError("The actual previously registered economic families are insufficient.")
    return {
        "ledger_sha256": LEGACY_SHA256,
        "engine_sha256": LEGACY_ENGINE_SHA256,
        "fixed_snapshot_completed_configurations": 178,
        "fixed_snapshot_recurring_reviews": 94,
        "fixed_snapshot_registered_versions": 96,
        "fixed_snapshot_terminal_technical_versions": 2,
        "new_factor_definitions": 0,
        "used_actual_registered_factors": selected,
        "old_records_reclassified_or_written": False,
    }


def study_inputs() -> tuple[MarketData, pd.Series, dict, dict]:
    policy = read_json(POLICY)
    validate(policy)
    if file_digest(PROTOCOL) != PROTOCOL_SHA256:
        raise QuantError("The pre-benchmark acceptance contract changed.")
    legacy = legacy_snapshot(policy)
    admission = read_json(BENCHMARKS / "admission.json")
    benchmark_results = read_json(BENCHMARKS / "results.json")
    if (
        benchmark_results["protocol_sha256"] != PROTOCOL_SHA256
        or benchmark_results["admission_sha256"] != file_digest(BENCHMARKS / "admission.json")
        or benchmark_results["runner_sha256"]
        != file_digest(ROOT / "src/us_quant/dual_benchmark_v1.py")
        or benchmark_results["benchmark_accounts"] != 8
        or benchmark_results["new_strategy_evaluations"] != 0
    ):
        raise QuantError("Use the actually completed, unchanged new benchmark accounts.")
    data = review_market(admission, ROOT)
    terms = load_terms(SOURCES, data.close.index)
    benchmark_files = {"admission.json", "results.json"}
    expected = {
        (years, scenario, symbol)
        for years in (10, 5)
        for scenario in ("base", "stress")
        for symbol in ("SPY", "QQQ")
    }
    actual = set()
    for row in benchmark_results["benchmarks"]:
        key = (row["years"], row["scenario"], row["symbol"])
        if key in actual:
            raise QuantError("Duplicate benchmark labels cannot replace a missing path.")
        actual.add(key)
        name = f"{row['symbol']}-{row['years']}y-{row['scenario']}"
        benchmark_files.update({f"{name}.json", f"{name}.csv", f"{name}-bt.csv"})
    if actual != expected:
        raise QuantError("Every source-bound SPY/QQQ cost/window path is required.")
    code = [reference(ROOT / f"src/us_quant/{name}.py") for name in CODE_MODULES]
    policies = [
        reference(path)
        for path in (
            POLICY,
            PROTOCOL,
            ROOT / "config/portfolio-refinement.json",
            ROOT / "config/research-program.json",
            ROOT / "config/volatility-term-risk.json",
        )
    ]
    frozen = {
        "study_id": policy["study_id"],
        "policy": reference(POLICY),
        "protocol": reference(PROTOCOL),
        "code": code,
        "policies": policies,
        "market": admission["market"],
        "volatility_sources": [
            reference(SOURCES / name) for name in ("manifest.json", "VIX.csv", "VIX3M.csv")
        ],
        "benchmark_artifacts": [reference(BENCHMARKS / name) for name in sorted(benchmark_files)],
        "legacy": legacy,
        "python_version": list(sys.version_info[:3]),
        "package_versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "bt", "ffn", "exchange-calendars")
        },
        "order_authority": False,
    }
    return data, terms["VIX"], frozen, benchmark_results


def matching_events(kind: str, identifier: str | None = None) -> list[dict]:
    return [
        event
        for event in journal_events(JOURNAL)
        if event["kind"] == kind and event["candidate_id"] == identifier
    ]


def progress(phase: str, **fields) -> None:
    path = REPORTS / "progress.json"
    current = read_json(path)
    current.update(updated_at_utc=utc_now(), phase=phase, **fields)
    storage_guard()
    write_json(path, current)


def prove_targets(data: MarketData, vix: pd.Series, policy: dict) -> tuple[dict, dict]:
    targets = build_targets(data, vix, policy)
    oracle = independent_targets(data, vix, policy)
    differences = {}
    for identifier in targets:
        if not np.allclose(
            targets[identifier], oracle[identifier], rtol=0, atol=1e-12, equal_nan=True
        ):
            raise QuantError("The sequential target oracle differs from the registered rule.")
        differences[identifier] = float(
            abs(targets[identifier].iloc[252:] - oracle[identifier].iloc[252:]).to_numpy().max()
        )
    checked_prefixes = []
    for position in (252, 503, 1006, 1509, 2012, len(data.close) - 2):
        if position >= len(data.close):
            continue
        day = data.close.index[position]
        prefix = MarketData(
            open=data.open.loc[:day],
            close=data.close.loc[:day],
            raw_close=data.raw_close.loc[:day],
            volume=data.volume.loc[:day],
            risk_free=data.risk_free.loc[:day],
        )
        earlier = build_targets(prefix, vix.loc[:day], policy)
        for identifier in targets:
            if not earlier[identifier].equals(targets[identifier].loc[:day]):
                raise QuantError("Future observations revised an earlier desired target.")
        checked_prefixes.append(str(day.date()))
    return targets, {
        "checked_at": utc_now(),
        "sequential_oracle_all_rows": True,
        "maximum_weight_difference": differences,
        "prefix_truncations": checked_prefixes,
        "candidate_accounts_or_metrics_computed": False,
        "untraded_price_indicator_is_a_strategy_account": False,
    }


def register() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, vix, frozen, _ = study_inputs()
        policy = read_json(POLICY)
        events = journal_events(JOURNAL)
        if not events:
            append_event(
                JOURNAL,
                "baseline",
                None,
                {
                    "program_id": "ssd_continuous_research_v1",
                    "legacy": frozen["legacy"],
                    "protocol": frozen["protocol"],
                    "minimum_economic_factor_families": 3,
                    "configuration_count_limit_before_deadline": None,
                },
            )
        elif events[0]["body"]["legacy"] != frozen["legacy"]:
            raise QuantError("The new journal must keep its actual preserved legacy baseline.")
        for candidate in CANDIDATES:
            spec = {"candidate": candidate, "frozen": frozen}
            existing = matching_events("registered", candidate["id"])
            if existing:
                if len(existing) != 1 or existing[0]["body"] != spec:
                    raise QuantError("A registered identity has different code, inputs or rules.")
            else:
                runtime_guard()
                append_event(JOURNAL, "registered", candidate["id"], spec)
        if any(matching_events("technical_failure", row["id"]) for row in CANDIDATES):
            raise QuantError(
                "Preserve a failed registered code version; do not overwrite or erase it."
            )
        pending = [row for row in CANDIDATES if not matching_events("target_proof", row["id"])]
        if pending:
            runtime_guard()
            progress(
                "preparing_registered_equity_first_targets",
                active_compute_job={"pid": os.getpid(), "study_id": policy["study_id"]},
                new_registered_candidates=len(CANDIDATES),
            )
            try:
                targets, proof = prove_targets(data, vix, policy)
                check_frozen_files(frozen)
                for candidate in pending:
                    identifier = candidate["id"]
                    directory = TARGETS / identifier
                    storage_guard()
                    if not directory.exists():
                        new_output_directory(directory)
                    target_path, proof_path = directory / "targets.csv", directory / "proof.json"
                    if target_path.exists():
                        retained = pd.read_csv(
                            target_path, index_col=0, parse_dates=True, float_precision="round_trip"
                        )
                        if (
                            not retained.index.equals(targets[identifier].index)
                            or not retained.columns.equals(targets[identifier].columns)
                            or not np.allclose(
                                retained, targets[identifier], rtol=0, atol=1e-12, equal_nan=True
                            )
                        ):
                            raise QuantError(
                                "Preserve a conflicting partial target matrix; do not overwrite it."
                            )
                    else:
                        write_text_atomic(
                            target_path, targets[identifier].to_csv(float_format="%.17g")
                        )
                    if proof_path.exists():
                        retained_proof = read_json(proof_path)
                        if {
                            key: value
                            for key, value in retained_proof.items()
                            if key != "checked_at"
                        } != {key: value for key, value in proof.items() if key != "checked_at"}:
                            raise QuantError(
                                "Retained source proof differs from the registered implementation."
                            )
                    else:
                        write_new_receipt(proof_path, proof)
                    append_event(
                        JOURNAL,
                        "target_proof",
                        identifier,
                        {
                            "registered_event_sha256": matching_events("registered", identifier)[0][
                                "sha256"
                            ],
                            "targets": reference(directory / "targets.csv"),
                            "proof": reference(directory / "proof.json"),
                            "defined_target_sessions": len(targets[identifier].dropna(how="all")),
                            "candidate_accounts_or_metrics_computed": False,
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
                for candidate in pending:
                    if not matching_events("target_proof", candidate["id"]):
                        append_event(
                            JOURNAL,
                            "technical_failure",
                            candidate["id"],
                            {
                                "stage": "registered_target_preparation",
                                "reason": str(exc),
                                "candidate_accounts_started": False,
                            },
                        )
                progress(
                    "equity_first_preparation_failed", active_compute_job=None, failure=str(exc)
                )
                raise
        entries = []
        check_frozen_files(frozen)
        for candidate in CANDIDATES:
            identifier = candidate["id"]
            registration = matching_events("registered", identifier)[0]
            proof = matching_events("target_proof", identifier)[0]
            checked_reference(proof["body"]["targets"])
            checked_reference(proof["body"]["proof"])
            entries.append(
                {
                    "candidate_id": identifier,
                    "registered_at": registration["at"],
                    "registered_event_sha256": registration["sha256"],
                    "target_proof_at": proof["at"],
                    "target_proof_event_sha256": proof["sha256"],
                    "specification": registration["body"],
                    "target_proof": proof["body"],
                }
            )
        body = {
            "schema_version": 1,
            "program_id": "ssd_continuous_research_v1",
            "study_id": policy["study_id"],
            "entries": entries,
            "new_registered_candidates": len(entries),
            "new_completed_candidate_accounts": 0,
            "new_economic_factor_definitions": 0,
            "raw_inputs_and_target_matrices_remain_private": True,
            "publication_and_source_bound_evaluation_still_required": True,
            "historical_overlap_and_adaptive_selection": True,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        if PACKET.exists():
            saved = read_json(PACKET)
            if {key: value for key, value in saved.items() if key != "created_at"} != body:
                raise QuantError("The existing registration packet changed; preserve it.")
        else:
            storage_guard()
            write_new_receipt(PACKET, {**body, "created_at": utc_now()})
        if matching_events("review", CANDIDATES[0]["id"]) or matching_events(
            "review", CANDIDATES[1]["id"]
        ):
            return reference(PACKET)
        progress(
            "equity_first_registered_waiting_publication",
            active_compute_job=None,
            new_registered_candidates=len(entries),
            new_completed_strategy_evaluations=0,
            registration_packet=reference(PACKET),
            pending_steps=[
                "actual_sanitized_feature_branch_publication",
                "source_bound_four_path_evaluation",
            ],
        )
        return reference(PACKET)


def published_inputs(frozen: dict, commit: str) -> dict:
    contract = read_json(REPO / "config/publication_allowlist.json")
    if (
        contract["bundle_manifest"] != "research/workbench/IMPORT_MANIFEST.json"
        or contract["bundle_manifest_sha256"] != BUNDLE_SHA256
    ):
        raise QuantError("Keep the original pinned import manifest, not revised acceptance hashes.")
    manifest_path = safe_file(REPO, contract["bundle_manifest"], BUNDLE_SHA256)
    manifest = read_json(manifest_path)
    pinned = {
        (manifest_path.parent / name).relative_to(ROOT).as_posix(): digest
        for name, digest in manifest["file_sha256"].items()
    }
    prefix = ROOT.relative_to(REPO).as_posix()
    required = {row["path"] for row in frozen["code"] + frozen["policies"]}
    explicit = {path for path in required if f"{prefix}/{path}" in contract["files"]}
    references = {row["path"]: row["sha256"] for row in frozen["code"] + frozen["policies"]}
    inherited = required - explicit
    if any(pinned.get(path) != references[path] for path in inherited):
        raise QuantError(
            "Every dependency must be an explicit extension or unchanged pinned import."
        )
    receipt = verify_publication(ROOT, reference(PACKET), explicit, commit)
    published_contract = json.loads(
        git(ROOT, ["show", f"{commit}:config/publication_allowlist.json"])
    )
    if (
        published_contract["bundle_manifest"] != contract["bundle_manifest"]
        or published_contract["bundle_manifest_sha256"] != BUNDLE_SHA256
        or digest_bytes(git(ROOT, ["show", f"{commit}:{contract['bundle_manifest']}"]))
        != BUNDLE_SHA256
    ):
        raise QuantError("The actual published immutable import boundary differs.")
    for path in inherited:
        safe_file(ROOT, path, references[path])
        if digest_bytes(git(ROOT, ["show", f"{commit}:{prefix}/{path}"])) != references[path]:
            raise QuantError(
                "An original imported dependency differs in the actual published commit."
            )
    return {
        **receipt,
        "publication_boundary_version": "explicit_extensions_plus_pinned_original_import_v1",
        "unchanged_pinned_import_files_verified": len(inherited),
        "original_bundle_manifest_sha256": BUNDLE_SHA256,
    }


def checked_packet(frozen: dict) -> dict:
    packet = read_json(PACKET)
    if (
        packet.get("schema_version") != 1
        or packet.get("program_id") != "ssd_continuous_research_v1"
        or packet.get("study_id") != frozen["study_id"]
        or [row["candidate_id"] for row in packet["entries"]] != [row["id"] for row in CANDIDATES]
    ):
        raise QuantError("The complete actual registered cohort cannot be reduced or relabelled.")
    created = pd.Timestamp(packet["created_at"])
    if created.tzinfo is None or pd.isna(created) or created > pd.Timestamp(utc_now()):
        raise QuantError("The public packet requires its real creation timestamp.")
    for candidate, entry in zip(CANDIDATES, packet["entries"], strict=True):
        registered = matching_events("registered", candidate["id"])
        prepared = matching_events("target_proof", candidate["id"])
        if len(registered) != 1 or len(prepared) != 1:
            raise QuantError("The actual registration and target-proof events are missing.")
        registration, proof = registered[0], prepared[0]
        expected = {
            "candidate_id": candidate["id"],
            "registered_at": registration["at"],
            "registered_event_sha256": registration["sha256"],
            "target_proof_at": proof["at"],
            "target_proof_event_sha256": proof["sha256"],
            "specification": {"candidate": candidate, "frozen": frozen},
            "target_proof": proof["body"],
        }
        if (
            registration["body"] != expected["specification"]
            or entry != expected
            or proof["body"]["registered_event_sha256"] != registration["sha256"]
            or not pd.Timestamp(registration["at"]) <= pd.Timestamp(proof["at"]) <= created
        ):
            raise QuantError("Published registration identities, clocks or source proofs disagree.")
        checked_reference(proof["body"]["targets"])
        proof_content = read_json(checked_reference(proof["body"]["proof"]))
        checked_at = pd.Timestamp(proof_content["checked_at"])
        if (
            checked_at.tzinfo is None
            or not pd.Timestamp(registration["at"]) <= checked_at <= pd.Timestamp(proof["at"])
            or proof_content["candidate_accounts_or_metrics_computed"] is not False
        ):
            raise QuantError(
                "The target proof cannot predate registration or follow account outcomes."
            )
    return packet


def verify_path(value: dict, frozen: dict, identifier: str, key: str) -> dict:
    receipt = read_json(checked_reference(value))
    if (
        receipt["candidate_id"] != identifier
        or receipt["path_key"] != key
        or receipt["frozen_specification_sha256"] != digest_json(frozen)
        or key not in PATH_KEYS
    ):
        raise QuantError("A completed path belongs to different frozen research.")
    for artifact in receipt["artifacts"]:
        checked_reference(artifact)
    return receipt


def evaluate_path(
    data: MarketData,
    targets: pd.DataFrame,
    frozen: dict,
    identifier: str,
    window: dict,
    scenario: dict,
    policy: dict,
    benchmarks: dict,
) -> dict:
    key = f"{window['years']}y-{scenario['id']}"
    events = journal_events(JOURNAL)
    completed = [
        event
        for event in events
        if event["kind"] == "path_completed"
        and event["candidate_id"] == identifier
        and event["body"]["path_key"] == key
    ]
    if completed:
        if len(completed) != 1:
            raise QuantError("Multiple completed versions cannot be cherry-picked for one path.")
        return verify_path(completed[0]["body"]["receipt"], frozen, identifier, key)
    starts = [
        event
        for event in events
        if event["kind"] == "path_started"
        and event["candidate_id"] == identifier
        and event["body"]["path_key"] == key
    ]
    if starts:
        prior = starts[-1]
        receipt_path = ROOT / prior["body"]["directory"] / "complete.json"
        if receipt_path.exists():
            recovered = verify_path(reference(receipt_path), frozen, identifier, key)
            append_event(
                JOURNAL,
                "path_completed",
                identifier,
                {
                    "path_key": key,
                    "receipt": reference(receipt_path),
                    "recovered_actual_complete_receipt": True,
                },
            )
            return recovered
        if not any(
            event["kind"] == "path_interrupted"
            and event["body"].get("started_event_sha256") == prior["sha256"]
            for event in events
        ):
            append_event(
                JOURNAL,
                "path_interrupted",
                identifier,
                {
                    "path_key": key,
                    "started_event_sha256": prior["sha256"],
                    "partial_output_preserved": True,
                    "reason": "Exclusive job lock reacquired; no complete path receipt exists.",
                },
            )
    runtime_guard()
    directory = ACCOUNTS / identifier / f"{key}-attempt-{len(starts) + 1:03d}"
    append_event(
        JOURNAL,
        "path_started",
        identifier,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    storage_guard()
    new_output_directory(directory)
    run = simulate_targets(
        data,
        targets,
        policy,
        window["start"],
        window["end"],
        cost_bps=scenario["cost_bps_per_side"],
        delay=scenario["delay_sessions"],
    )
    independent = independent_equity(
        data,
        run.issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = performance(run.result.frame["return"], run.result.frame["risk_free"])
    checked, difference = independent_metrics(
        run.result.frame, independent, data.risk_free.loc[run.result.frame.index], 10000
    )
    check_metrics(checked, own)
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]
        for row in benchmarks["benchmarks"]
    }
    criteria = gates(
        own,
        lookup[(window["years"], scenario["id"], "SPY")],
        lookup[(window["years"], scenario["id"], "QQQ")],
        lookup[(window["years"], "base", "SPY")],
        scenario["id"],
    )
    disclosure = execution_statistics(run.result, data, 10000, scenario["cost_bps_per_side"], 1)
    frames = {
        "account.csv": run.result.frame,
        "weights.csv": run.result.weights,
        "issued.csv": run.issued,
        "independent-bt.csv": independent,
    }
    storage_guard()
    for name, frame in frames.items():
        write_text_atomic(directory / name, frame.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "decisions.json", {"decisions": run.decisions})
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": identifier,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": disclosure,
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"]
        - lookup[(window["years"], scenario["id"], "SPY")]["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"]
        - lookup[(window["years"], scenario["id"], "QQQ")]["cagr"],
        "stress_vs_base_cost_qqq_diagnostic": own["cagr"]
        - lookup[(window["years"], "base", "QQQ")]["cagr"]
        if scenario["id"] == "stress"
        else None,
        "independent_bt_max_equity_difference_usd": difference,
        "artifacts": [reference(directory / name) for name in (*frames, "decisions.json")],
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    append_event(
        JOURNAL,
        "path_completed",
        identifier,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def evaluate(commit: str) -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, vix, frozen, benchmarks = study_inputs()
        checked_packet(frozen)
        finished_path = ACCOUNTS / "results.json"
        if finished_path.exists():
            saved = read_json(finished_path)
            for candidate in CANDIDATES:
                identifier = candidate["id"]
                actual_reviews = matching_events("review", identifier)
                failed = matching_events("technical_failure", identifier)
                expected_review = (
                    actual_reviews[0]["body"]
                    if len(actual_reviews) == 1
                    else {
                        "candidate_id": identifier,
                        "status": "preserved_technical_failure",
                        "failure_event_sha256": failed[0]["sha256"],
                        "failure": failed[0]["body"],
                    }
                    if len(failed) == 1
                    else None
                )
                if expected_review is None or (
                    saved["reviews"].get(identifier) != expected_review
                    or saved["legacy"] != frozen["legacy"]
                ):
                    raise QuantError(
                        "The existing cohort result differs from the actual completed reviews."
                    )
                for completed in matching_events("path_completed", identifier):
                    verify_path(
                        completed["body"]["receipt"],
                        frozen,
                        identifier,
                        completed["body"]["path_key"],
                    )
            return saved
        publication = published_inputs(frozen, commit)
        append_event(JOURNAL, "publication", None, publication)
        policy, protocol = read_json(POLICY), read_json(PROTOCOL)
        regenerated = build_targets(data, vix, policy)
        reviews = {}
        progress(
            "evaluating_registered_equity_first_pair",
            active_compute_job={"pid": os.getpid(), "study_id": policy["study_id"]},
            publication=publication,
        )
        active_identifier = None
        try:
            for candidate in CANDIDATES:
                identifier = candidate["id"]
                active_identifier = identifier
                existing = matching_events("review", identifier)
                if existing:
                    reviews[identifier] = existing[0]["body"]
                    continue
                failed = matching_events("technical_failure", identifier)
                if failed:
                    reviews[identifier] = {
                        "candidate_id": identifier,
                        "status": "preserved_technical_failure",
                        "failure_event_sha256": failed[0]["sha256"],
                        "failure": failed[0]["body"],
                    }
                    continue
                proof = matching_events("target_proof", identifier)
                if len(proof) != 1:
                    raise QuantError("A complete source-bound target proof is required.")
                targets = pd.read_csv(
                    checked_reference(proof[0]["body"]["targets"]),
                    index_col=0,
                    parse_dates=True,
                    float_precision="round_trip",
                )
                if (
                    not targets.index.equals(data.close.index)
                    or not targets.columns.equals(data.close.columns)
                    or not np.allclose(
                        targets, regenerated[identifier], rtol=0, atol=1e-12, equal_nan=True
                    )
                ):
                    raise QuantError(
                        "Submitted intents do not reproduce the published source-bound rule."
                    )
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen_files(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        paths[key] = evaluate_path(
                            data, targets, frozen, identifier, window, scenario, policy, benchmarks
                        )
                        progress(
                            "evaluating_registered_equity_first_pair",
                            active_candidate=identifier,
                            last_completed_path=key,
                        )
                if set(paths) != PATH_KEYS:
                    raise QuantError("All four actual paths from one configuration are required.")
                check_frozen_files(frozen)
                legacy_snapshot(policy)
                review = {
                    "candidate_id": identifier,
                    "reviewed_at": utc_now(),
                    "status": "historical_point_gates_passed"
                    if all(row["all_point_gates_pass"] for row in paths.values())
                    else "rejected",
                    "paths": paths,
                    "all_four_paths_same_candidate_and_frozen_parameters": True,
                    "full_search_selection_adjusted_alpha_verified": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                    "automatic_live_deployment": False,
                    "order_authority": False,
                }
                append_event(JOURNAL, "review", identifier, review)
                reviews[identifier] = review
            legacy_snapshot(policy)
            check_frozen_files(frozen)
            result = {
                "completed_at": utc_now(),
                "study_id": policy["study_id"],
                "registered_candidates": len(CANDIDATES),
                "reviews": reviews,
                "legacy": frozen["legacy"],
                "publication": publication,
                "original_records_reclassified": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            path = ACCOUNTS / "results.json"
            if not path.exists():
                storage_guard()
                write_new_receipt(path, result)
            else:
                saved = read_json(path)
                if saved["reviews"] != reviews or saved["legacy"] != result["legacy"]:
                    raise QuantError(
                        "A prior completed cohort cannot be replaced with new outcomes."
                    )
                result = saved
            progress(
                "equity_first_pair_complete",
                active_compute_job=None,
                new_registered_candidates=len(CANDIDATES),
                new_completed_strategy_evaluations=sum(
                    row.get("status") in {"rejected", "historical_point_gates_passed"}
                    for row in reviews.values()
                ),
                first_pair_results=reference(path),
                pending_steps=[
                    "publish_sanitized_complete_pair_results",
                    "next_independent_economic_research",
                ],
            )
            return result
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            if (
                active_identifier is not None
                and not matching_events("review", active_identifier)
                and not matching_events("technical_failure", active_identifier)
            ):
                append_event(
                    JOURNAL,
                    "technical_failure",
                    active_identifier,
                    {
                        "stage": "registered_account_evaluation_or_validation",
                        "reason": str(exc),
                        "partial_account_outputs_retained": True,
                        "candidate_accounts_started": bool(
                            matching_events("path_started", active_identifier)
                        ),
                    },
                )
            progress("equity_first_evaluation_blocked", active_compute_job=None, failure=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate versioned SSD research journal; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual published feature-branch commit.")
            result = evaluate(args.commit)
        else:
            events = journal_events(JOURNAL)
            result = {
                "events": len(events),
                "registered": sum(event["kind"] == "registered" for event in events),
                "reviews": sum(event["kind"] == "review" for event in events),
                "technical_failures": sum(event["kind"] == "technical_failure" for event in events),
                "interrupted_path_attempts": sum(
                    event["kind"] == "path_interrupted" for event in events
                ),
                "head_sha256": events[-1]["sha256"] if events else None,
            }
        print(json.dumps(result, indent=2, allow_nan=False))
    except (QuantError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, f"SSD research blocked: {exc}\n")


if __name__ == "__main__":
    main()
