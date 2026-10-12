from __future__ import annotations

import argparse
import fcntl
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd

from us_quant.candidate_receipts import (
    input_reference,
    local_path,
    recorded_registration,
    register_or_recover,
    write_new_receipt,
)
from us_quant.claims_factor_guard import validate_policy as validate_claims
from us_quant.config import QuantError
from us_quant.inflation_factor_guard import validate_policy as validate_inflation
from us_quant.research_daily import ORIGINS, moment
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.review_receipts import evidence_time, record_or_recover
from us_quant.six_factor_strategy import validate_policy as validate_six
from us_quant.storage import digest_bytes, digest_json, file_digest, read_json, utc_now, write_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).resolve().parents[2]
WEEK = "2026-W42"
ELIGIBLE = "2026-10-17T09:00:00+08:00"
DIRECTORY = f"reports/weekly-research-{WEEK}"
PACKET = f"evidence/weekly_research_{WEEK}_registration.json"
AGENDA = (
    ("six_factor_equal_family", "six-factor", "six_factor_strategy"),
    ("six_factor_bounded_inverse_volatility", "six-factor", "six_factor_strategy"),
    ("semiconductor_dollar_inflation_gold_guard", "inflation-guard", "inflation_factor_guard"),
    ("semiconductor_dollar_claims_confirmation", "claims-guard", "claims_factor_guard"),
)
VALIDATORS = {
    "six_factor_strategy": validate_six,
    "inflation_factor_guard": validate_inflation,
    "claims_factor_guard": validate_claims,
}


def eligible(now: pd.Timestamp) -> None:
    if now < pd.Timestamp(ELIGIBLE):
        raise QuantError(f"The frozen weekly queue is not eligible before {ELIGIBLE}.")


def inputs(root: Path) -> list[dict]:
    entries = []
    for candidate, group, generator in AGENDA:
        prepared = Path(f"data/{group}-prepared-20261017")
        spec, spec_reference = input_reference(root, prepared / f"{candidate}-spec.json")
        readiness, readiness_reference = input_reference(root, prepared / "readiness.json")
        frozen = spec.get("frozen_files")
        if (
            spec.get("id") != candidate
            or spec.get("data_scope") != "factor_etf_portfolio"
            or spec.get("order_authority") is not False
            or not isinstance(frozen, dict)
            or not frozen
        ):
            raise QuantError("The weekly queue requires its exact original frozen candidates.")
        for path, digest in frozen.items():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise QuantError("Every queued frozen source needs an explicit SHA256.")
            safe_file(root, path, digest)
        name, policy = registered_generator(spec, root)
        if name != generator:
            raise QuantError("A queued candidate cannot switch its source-bound generator.")
        VALIDATORS[name](policy)
        entries.append(
            {
                "spec": spec,
                "spec_reference": spec_reference,
                "readiness": readiness,
                "readiness_reference": readiness_reference,
                "output": f"reports/{group}-20261017",
            }
        )
    return entries


def actual_cycle(program: ResearchProgram, entries: list[dict], now: pd.Timestamp) -> dict:
    proposals = read_json(safe_file(program.root, "config/factor-family-expansion.json"))
    row = program.db.execute("SELECT result FROM cycles WHERE week=?", (WEEK,)).fetchone()
    if row is None:
        raise QuantError("Complete the actual IJR/PKW factor cycle before queue admission.")
    cycle = json.loads(row["result"])
    if (
        proposals.get("planned_week") != WEEK
        or proposals.get("eligible_not_before") != ELIGIBLE
        or cycle.get("proposal_sha256") != digest_json(proposals)
        or cycle.get("readiness_sha256") != digest_json(entries[0]["readiness"])
        or cycle.get("data_ready") is not True
        or cycle.get("data_scope") != "factor_etf_portfolio"
        or cycle.get("order_authority") is not False
        or not pd.Timestamp(ELIGIBLE) <= evidence_time(cycle.get("at")) <= now
    ):
        raise QuantError("The queue cannot substitute another cycle, readiness or admission time.")
    for factor in proposals["factors"]:
        recorded = program.db.execute(
            "SELECT body FROM factors WHERE id=?", (factor["id"],)
        ).fetchone()
        if recorded is None or json.loads(recorded["body"]) != factor:
            raise QuantError(
                "Both original queued factor definitions must actually be in the catalog."
            )
    return cycle


def packet_body(program: ResearchProgram, entries: list[dict], cycle: dict) -> dict:
    candidates = []
    for entry in entries:
        original = recorded_registration(program, entry["spec"], entry["readiness"])
        if original is None:
            raise QuantError(
                "All four actual registrations must exist before publication or evaluation."
            )
        candidates.append({**entry, **original})
    return {
        "schema_version": 1,
        "week": WEEK,
        "eligible_not_before": ELIGIBLE,
        "engine_sha256": engine_fingerprint(),
        "runner_sha256": file_digest(Path(__file__)),
        "actual_factor_cycle": cycle,
        "candidates": candidates,
        "candidate_portfolio_returns_computed_when_packet_created": False,
        "automatic_git_or_order_submission": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def check_packet(packet: dict, expected: dict, now: pd.Timestamp) -> None:
    if set(packet) != {*expected, "recorded_at"} or any(
        packet.get(key) != value for key, value in expected.items()
    ):
        raise QuantError("The registration packet changed; preserve the exact original evidence.")
    at = evidence_time(packet.get("recorded_at"))
    if at > now or any(
        at < evidence_time(row["registration"]["registered_at"]) for row in expected["candidates"]
    ):
        raise QuantError("Publication metadata cannot precede the actual registrations.")


def git(root: Path, arguments: list[str]) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", *arguments],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise QuantError(
            f"Git publication verification failed at {arguments[0]} (exit {result.returncode})."
        )
    return result.stdout


def required_public_files(root: Path, entries: list[dict]) -> set[str]:
    paths = {
        "config/research-program.json",
        "config/factor-family-expansion.json",
        "src/us_quant/research_weekly.py",
        "src/us_quant/research_program.py",
        "src/us_quant/strategy_replay.py",
        "src/us_quant/candidate_receipts.py",
        "src/us_quant/review_receipts.py",
        "src/us_quant/cash_funded_accounting_v2.py",
        "src/us_quant/bt_audit.py",
    }
    for entry in entries:
        paths.update(
            name
            for name in entry["spec"]["frozen_files"]
            if name.startswith(("src/", "config/", "evidence/"))
        )
    for path in paths:
        safe_file(root, path)
    return paths


def verify_publication(
    root: Path,
    reference: dict,
    required: set[str],
    commit: str,
) -> dict:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise QuantError("Evaluation needs an exact, actually published Git commit SHA.")
    repo = Path(git(root, ["rev-parse", "--show-toplevel"]).decode().strip()).resolve()
    if not root.resolve().is_relative_to(repo):
        raise QuantError("Publication must belong to this isolated repository workbench.")
    branch = git(root, ["symbolic-ref", "--quiet", "--short", "HEAD"]).decode().strip()
    upstream = (
        git(
            root,
            [
                "for-each-ref",
                "--format=%(upstream:remotename)%09%(upstream:remoteref)",
                f"refs/heads/{branch}",
            ],
        )
        .decode()
        .strip()
        .split("\t")
    )
    if len(upstream) != 2 or not upstream[0] or not upstream[1].startswith("refs/heads/"):
        raise QuantError(
            "Use this research branch's configured remote, not an assumed main branch."
        )
    remote, ref = upstream
    rows = git(root, ["ls-remote", "--exit-code", remote, ref]).decode().strip().splitlines()
    if len(rows) != 1:
        raise QuantError("The configured research branch was not uniquely observed on its remote.")
    fields = rows[0].split()
    if len(fields) != 2 or fields[1] != ref or not re.fullmatch(r"[0-9a-f]{40}", fields[0]):
        raise QuantError("The actual remote publication response is invalid.")
    head = fields[0]
    git(root, ["merge-base", "--is-ancestor", commit, head])
    prefix = root.relative_to(repo).as_posix()
    packet_path = f"{prefix}/{reference['path']}"
    allowlist = read_json(safe_file(repo, "config/publication_allowlist.json"))
    try:
        published_allowlist = json.loads(
            git(root, ["show", f"{commit}:config/publication_allowlist.json"])
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise QuantError("The actual published allowlist must be a complete JSON object.") from exc
    for contract in (allowlist, published_allowlist):
        if (
            not isinstance(contract, dict)
            or not isinstance(contract.get("files"), list)
            or any(not isinstance(path, str) for path in contract["files"])
            or len(contract["files"]) != len(set(contract["files"]))
        ):
            raise QuantError("Both published and current publication boundaries must be explicit.")
    required_paths = {packet_path, *(f"{prefix}/{path}" for path in required)}
    if not all(
        required_paths.issubset(set(contract["files"]))
        for contract in (
            allowlist,
            published_allowlist,
        )
    ):
        raise QuantError(
            "Every packet, frozen code and policy must be explicitly approved for publication."
        )
    for path in required_paths:
        relative = Path(path).relative_to(root.relative_to(repo)).as_posix()
        local = safe_file(root, relative)
        published = git(root, ["show", f"{commit}:{path}"])
        if digest_bytes(published) != file_digest(local):
            raise QuantError(
                "Actual published registration, code or policy differs from frozen local bytes."
            )
    safe_file(root, reference["path"], reference["sha256"])
    return {
        "published_commit": commit,
        "configured_remote": remote,
        "configured_remote_ref": ref,
        "observed_remote_head": head,
        "registration_packet": reference,
        "verified_public_file_count": len(required_paths),
        "verification_is_independent_first_publication_time": False,
    }


def publication_receipt(
    root: Path,
    proof: dict,
    packet: dict,
    now: pd.Timestamp,
) -> dict:
    path = root / DIRECTORY / "publication.json"
    expected = {key: value for key, value in proof.items() if key != "observed_remote_head"}
    expected.update(
        {
            "schema_version": 1,
            "runner_sha256": file_digest(Path(__file__)),
            "investment_objective_verified": False,
            "order_authority": False,
        }
    )
    if path.exists():
        saved, _ = input_reference(root, path)
        if set(saved) != {*expected, "observed_remote_head", "verified_on_remote_at"} or any(
            saved.get(key) != value for key, value in expected.items()
        ):
            raise QuantError(
                "A different publication receipt cannot replace the original evidence fence."
            )
        observed = saved["observed_remote_head"]
        if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{40}", observed):
            raise QuantError("The original publication receipt has an invalid remote head.")
        git(root, ["merge-base", "--is-ancestor", proof["published_commit"], observed])
        git(root, ["merge-base", "--is-ancestor", observed, proof["observed_remote_head"]])
    else:
        saved = {
            **expected,
            "observed_remote_head": proof["observed_remote_head"],
            "verified_on_remote_at": now.isoformat(),
        }
        write_new_receipt(path, saved)
        written, _ = input_reference(root, path)
        if written != saved:
            raise QuantError("No complete publication fence was persisted; do not evaluate.")
    at = evidence_time(saved.get("verified_on_remote_at"))
    if not evidence_time(packet["recorded_at"]) <= at <= now:
        raise QuantError(
            "The publication verification time cannot precede registration or be future-dated."
        )
    return saved


def check_outputs(program: ResearchProgram, entries: list[dict], publication: dict) -> None:
    at = evidence_time(publication["verified_on_remote_at"])
    failures = program.technical_failures()
    for entry in entries:
        candidate = entry["spec"]["id"]
        directory = local_path(program.root, Path(entry["output"]) / candidate)
        review = program.db.execute(
            "SELECT at FROM reviews WHERE candidate_id=?", (candidate,)
        ).fetchone()
        if candidate in failures:
            if evidence_time(failures[candidate]["at"]) < at:
                raise QuantError("A pre-publication technical outcome cannot be relabelled.")
            continue
        if directory.exists():
            bundle, _ = input_reference(program.root, directory / "bundle.json")
            if evidence_time(bundle.get("completed_at")) < at:
                raise QuantError(
                    "Pre-publication evaluation cannot become a preregistered queue result."
                )
        elif review is not None:
            raise QuantError(
                "A recorded review lacks its original complete queue bundle; do not regenerate it."
            )
        if review is not None and evidence_time(review["at"]) < at:
            raise QuantError(
                "A review predating the actual publication fence cannot be relabelled."
            )


def run(
    action: str,
    origin: str,
    *,
    published_commit: str | None = None,
    root: Path = ROOT,
    clock=utc_now,
) -> dict:
    if action not in ("register", "evaluate") or origin not in ORIGINS:
        raise QuantError(
            "Choose a supported weekly phase and declare, not certify, its invocation origin."
        )
    root = root.resolve()
    directory = local_path(root, Path(DIRECTORY))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = moment(clock)
        attempt = directory / "runs" / (started.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        attempt.mkdir(parents=True, mode=0o700)
        path = attempt / "result.json"
        record = {
            "schema_version": 1,
            "week": WEEK,
            "phase": action,
            "started_at": started.isoformat(),
            "declared_invocation_origin": origin,
            "native_schedule_delivery_independently_verified": False,
            "runner_sha256": file_digest(Path(__file__)),
            "status": "running",
            "steps": [],
            "run_receipt_path": path.relative_to(root).as_posix(),
            "automatic_git_or_order_submission": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_json(path, record)
        program = None
        current = None
        previous = started

        def step(name, operation):
            nonlocal current, previous
            at = moment(clock)
            if at < previous:
                raise QuantError("Weekly operational timestamps cannot move backwards.")
            current = {"name": name, "started_at": at.isoformat(), "status": "running"}
            record["steps"].append(current)
            write_json(path, record)
            result = operation()
            current["result"] = result
            write_json(path, record)
            if not isinstance(result, dict):
                raise QuantError("Each weekly stage must return its explicit original result.")
            if name.startswith(("register:", "review:")) and (
                result.get("candidate_id") != name.split(":", 1)[1]
                or result.get("order_authority") is not False
                or result.get("investment_objective_verified") is not False
                or result.get("action")
                not in (
                    "registered",
                    "recovered_registration",
                    "already_recorded",
                    "review_recorded",
                    "recovered_review",
                    "technical_rejection_retained",
                )
            ):
                raise QuantError("A weekly stage returned unsupported candidate scope or status.")
            if name.startswith("register:") and (
                result.get("action")
                not in (
                    "registered",
                    "recovered_registration",
                    "already_recorded",
                )
                or result.get("new_strategy_evaluations") != 0
                or result.get("portfolio_returns_computed") is not False
            ):
                raise QuantError("Registration stages cannot submit or compute portfolio outcomes.")
            finished = moment(clock)
            if finished < at:
                raise QuantError("Weekly stage completion predates its start.")
            current.update({"status": "completed", "finished_at": finished.isoformat()})
            previous, current = finished, None
            write_json(path, record)
            return result

        try:
            eligible(started)
            if (action == "evaluate") != (published_commit is not None):
                raise QuantError("Only evaluation requires an explicit actual published commit.")
            entries = inputs(root)
            program = ResearchProgram(
                safe_file(root, "runtime/research-program.sqlite3"),
                read_json(safe_file(root, "config/research-program.json")),
                root=root,
            )
            before = program.status()
            record["research_before"] = before
            cycle = actual_cycle(program, entries, started)
            if action == "register":
                for entry in entries:
                    candidate = entry["spec"]["id"]
                    step(
                        f"register:{candidate}",
                        lambda item=entry, identifier=candidate: register_or_recover(
                            Path(item["spec_reference"]["path"]),
                            Path(item["readiness_reference"]["path"]),
                            Path(DIRECTORY) / "registrations" / f"{identifier}.json",
                            root=root,
                        ),
                    )
                expected = packet_body(program, entries, cycle)
                if (
                    program.status()["total_evaluated_configurations"]
                    != before["total_evaluated_configurations"]
                ):
                    raise QuantError("The registration phase changed completed portfolio outcomes.")
                if (root / PACKET).exists():
                    packet, reference = input_reference(root, Path(PACKET))
                    check_packet(packet, expected, moment(clock))
                else:
                    failures = program.technical_failures()
                    if any(
                        entry["spec"]["id"] in failures
                        or program.db.execute(
                            "SELECT 1 FROM reviews WHERE candidate_id=?",
                            (entry["spec"]["id"],),
                        ).fetchone()
                        or (root / entry["output"] / entry["spec"]["id"]).exists()
                        for entry in entries
                    ):
                        raise QuantError(
                            "Do not create a first registration packet after evaluation began."
                        )
                    packet = {**expected, "recorded_at": moment(clock).isoformat()}
                    write_new_receipt(root / PACKET, packet)
                    packet, reference = input_reference(root, Path(PACKET))
                    check_packet(packet, expected, moment(clock))
                record.update(
                    {
                        "status": "registered_waiting_for_publication",
                        "registration_packet": reference,
                    }
                )
            else:
                packet, reference = input_reference(root, Path(PACKET))
                check_packet(packet, packet_body(program, entries, cycle), moment(clock))
                proof = step(
                    "verify_actual_publication",
                    lambda: verify_publication(
                        root,
                        reference,
                        required_public_files(root, entries),
                        published_commit,
                    ),
                )
                publication = publication_receipt(root, proof, packet, moment(clock))
                check_outputs(program, entries, publication)
                for entry in entries:
                    candidate = entry["spec"]["id"]
                    output = Path(entry["output"])

                    def evaluate(identifier=candidate, folder=output):
                        failure = program.technical_failures().get(identifier)
                        if failure is not None:
                            return {
                                "action": "technical_rejection_retained",
                                "candidate_id": identifier,
                                "original_failure": failure,
                                "investment_objective_verified": False,
                                "order_authority": False,
                            }
                        bundle = root / folder / identifier / "bundle.json"
                        if not (root / folder / identifier).exists():
                            program.evaluate_registered(identifier, folder)
                        return record_or_recover(
                            identifier,
                            bundle,
                            folder / identifier / "review-receipt.json",
                            root=root,
                        )

                    step(f"review:{candidate}", evaluate)
                record["status"] = "all_original_outcomes_recorded"
                record["registration_packet"] = reference
                record["publication_fence"] = publication
            after = program.status()
            if after["factor_proposals"] != before["factor_proposals"] or (
                action == "evaluate"
                and after["registered_candidates"] != before["registered_candidates"]
            ):
                raise QuantError(
                    "Weekly phases cannot alter the catalog or add evaluation candidates."
                )
            record.update(
                {
                    "research_after": after,
                    "new_historical_strategy_evaluations": (
                        after["total_evaluated_configurations"]
                        - before["total_evaluated_configurations"]
                    ),
                    "new_candidate_registration_count": (
                        len(after["registered_candidates"]) - len(before["registered_candidates"])
                    ),
                    "finished_at": moment(clock).isoformat(),
                    "earlier_completed_outputs_preserved": True,
                }
            )
            if evidence_time(record["finished_at"]) < previous:
                raise QuantError("Weekly completion predates its last actual stage.")
            write_json(path, record)
            return record
        except (
            QuantError,
            OSError,
            ValueError,
            sqlite3.Error,
            subprocess.SubprocessError,
        ) as exc:
            if current is not None:
                current["status"] = "failed"
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "failure_recorded_at_utc": utc_now(),
                    "earlier_completed_outputs_preserved": True,
                }
            )
            write_json(path, record)
            raise
        finally:
            if program is not None:
                program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded publication-gated W42 research queue; never orders."
    )
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--origin", choices=ORIGINS, default="operator")
    parser.add_argument("--published-commit")
    args = parser.parse_args()
    try:
        result = run(args.action, args.origin, published_commit=args.published_commit)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("steps", "research_before", "research_after")
                },
                indent=2,
            )
        )
    except (QuantError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(
            2, f"Weekly queue blocked; preserve all registrations and original outputs: {exc}\n"
        )


if __name__ == "__main__":
    main()
