from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pandas as pd

from us_quant.config import QuantError
from us_quant.prospective_data import ProspectiveArchive, utc_now
from us_quant.prospective_factor_inputs import ExpandedArchive
from us_quant.prospective_research_accounts import ResearchAccounts
from us_quant.prospective_target_observations import TargetJournal
from us_quant.research_program import ResearchProgram, engine_fingerprint
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
STEP_NAMES = ("parent_data", "expanded_data", "target_observations", "research_model")
ORIGINS = ("agent_continuation", "session_automation", "operator")


def actual_steps(root: Path) -> list[tuple[str, Callable[[], dict]]]:
    parent = ProspectiveArchive(
        root / "data/prospective-market-v1", read_json(root / "config/prospective-data.json")
    )
    expanded = ExpandedArchive(
        root / "data/prospective-factor-inputs-v2",
        read_json(root / "config/prospective-factor-inputs.json"),
        root=root,
    )
    targets = TargetJournal(
        root / "data/prospective-target-observations-v1",
        read_json(root / "config/prospective-target-observations.json"),
        root=root,
    )
    model = ResearchAccounts(
        root / "data/prospective-research-accounts-v1",
        read_json(root / "config/prospective-research-accounts.json"),
        root=root,
    )
    return [
        ("parent_data", parent.collect),
        ("expanded_data", expanded.collect),
        ("target_observations", targets.collect),
        ("research_model", model.advance),
    ]


def research_state(root: Path) -> dict:
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        read_json(root / "config/research-program.json"),
        root=root,
    )
    try:
        state = program.status()
        return {
            "state_sha256": digest_json(state),
            "engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "complete_strategy_configurations": state["total_evaluated_configurations"],
            "complete_program_reviews": state["new_evaluated_strategy_count"],
            "known_registered_definitions": state["known_factor_definition_count"],
            "research_champion": state["research_champion"],
            "pending_candidate_ids": state["pending_candidate_ids"],
        }
    finally:
        program.close()


def moment(clock) -> pd.Timestamp:
    value = pd.Timestamp(clock())
    if pd.isna(value) or value.tzinfo is None:
        raise QuantError("Daily operational receipts need timezone-aware timestamps.")
    return value


def run(
    directory: Path,
    origin: str,
    *,
    root: Path = ROOT,
    clock=utc_now,
    step_factory=actual_steps,
    state_reader=research_state,
) -> dict:
    if origin not in ORIGINS:
        raise QuantError("Declare a supported invocation origin without claiming scheduler proof.")
    directory = directory if directory.is_absolute() else root / directory
    if (
        directory.is_symlink()
        or not directory.resolve().is_relative_to(root.resolve())
        or not directory.resolve().is_relative_to((root / "reports").resolve())
        or directory.resolve() == (root / "reports").resolve()
    ):
        raise QuantError("Daily receipts need a specific, in-workbench operational directory.")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = moment(clock)
        attempt = directory / "runs" / (
            started.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        )
        attempt.mkdir(parents=True, mode=0o700)
        record = {
            "schema_version": 1,
            "started_at": started.isoformat(),
            "declared_invocation_origin": origin,
            "native_schedule_delivery_independently_verified": False,
            "runner_sha256": file_digest(Path(__file__)),
            "run_receipt_path": (attempt / "result.json").relative_to(root).as_posix(),
            "status": "running",
            "steps": [],
            "existing_archives_initialized_or_reset": False,
            "new_factor_or_candidate_admission_performed": False,
            "new_historical_strategy_evaluations": 0,
            "actual_orders_or_broker_account": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_json(attempt / "result.json", record)
        current = None
        previous = started
        try:
            before = state_reader(root)
            record["research_before"] = before
            write_json(attempt / "result.json", record)
            steps = step_factory(root)
            if tuple(name for name, _ in steps) != STEP_NAMES:
                raise QuantError("Daily operations must preserve all four stages and their order.")
            for name, action in steps:
                current = {"name": name, "started_at": moment(clock).isoformat(), "status": "running"}
                if pd.Timestamp(current["started_at"]) < previous:
                    raise QuantError("Daily operational time cannot move backwards.")
                record["steps"].append(current)
                write_json(attempt / "result.json", record)
                result = action()
                current["result"] = result
                write_json(attempt / "result.json", record)
                if (
                    not isinstance(result, dict)
                    or result.get("investment_objective_verified") is not False
                    or result.get("order_authority") is not False
                    or result.get("action") not in (
                        "collected", "already_collected", "missed_preopen_deadline",
                        "advanced", "no_new_observed_session",
                    )
                ):
                    raise QuantError("An operational stage returned unsupported scope or status.")
                finished = moment(clock)
                if finished < pd.Timestamp(current["started_at"]):
                    raise QuantError("Daily operational completion predates its stage.")
                current["finished_at"] = finished.isoformat()
                if result["action"] == "missed_preopen_deadline":
                    raise QuantError(f"{name}: missed the actual input acquisition deadline.")
                current["status"] = "completed"
                previous, current = finished, None
                write_json(attempt / "result.json", record)
            after = state_reader(root)
            if after != before:
                raise QuantError("The daily input/observation runner cannot alter research trials or policy.")
            finished = moment(clock)
            if finished < previous:
                raise QuantError("Daily operational finish predates its final step.")
            record.update({
                "status": "completed",
                "finished_at": finished.isoformat(),
                "research_after": after,
                "historical_research_state_unchanged": True,
                "pipeline_execution_verified_by_actual_stage_returns": True,
            })
            write_json(attempt / "result.json", record)
            return record
        except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
            if current is not None:
                current["status"] = "failed"
            record.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "failure_recorded_at_utc": utc_now().isoformat(),
                "previous_completed_stage_outputs_retained": True,
                "pipeline_execution_verified_by_actual_stage_returns": False,
            })
            write_json(attempt / "result.json", record)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Receipt-backed daily research inputs, never broker orders.")
    parser.add_argument("--origin", choices=ORIGINS, default="operator")
    parser.add_argument("--directory", type=Path, default=ROOT / "reports/daily-research-operations")
    args = parser.parse_args()
    try:
        result = run(args.directory, args.origin)
        print(json.dumps({
            key: value for key, value in result.items()
            if key not in ("research_before", "research_after", "steps")
        }, indent=2))
        for step in result["steps"]:
            print(json.dumps({
                "step": step["name"], "status": step["status"],
                "action": step["result"]["action"],
            }))
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Daily research operations blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
