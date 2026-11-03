from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from us_quant import dual_benchmark_v1 as original
from us_quant.config import QuantError
from us_quant.storage import digest_bytes, file_digest, read_json

ROOT = Path(__file__).resolve().parents[2]
SSD = Path("/Volumes/Extreme SSD/us_stock_qr-research")
WORKSPACE = SSD / "workspace"
BRANCH = "julianli00-factor-research-handoff"
SESSION = "e99572a7-073a-4344-91b8-7252df99256c"
AUTHORIZATION = SSD / "migration/research-authorization-20261103-v1.json"
AUTHORIZATION_SHA256 = "6324d64f2c9ca99349e2b5e9f2502f0bc908091d02cb8cdede5d736c0bad6b4f"
START = datetime(2026, 11, 3, 7, 40, 50, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 30, 15, 59, tzinfo=timezone.utc)
CLOSEOUT = DEADLINE - timedelta(minutes=10)
LOCK = ROOT / "runtime/ssd-continuous-research-v1.lock"
OLD_PROMPT_SHA256 = "46d88b147d139df4fd238fbdbbbc40fbfc543a5250b05d0bfb7c0a73bb937b92"
OLD_AUTHORITY = (
    "Use /Volumes/Extreme SSD/us_stock_qr-research/migration/"
    "research-authorization-20261027.json and ibkr-data-authorization-20261027.json, "
    "plus the creator's explicit continuous-stage activation (native event 804). "
    "Migration was accepted."
)
NEW_AUTHORITY = (
    f"Use {AUTHORIZATION} (SHA256 {AUTHORIZATION_SHA256}), the new explicit November "
    "user authorization and confirmed 2026-11-30 deadline, received through creator "
    "native event 64145. This is a new nonretroactive window; preserve October "
    "authorization, expired lifecycle, shutdown receipts and all research history. "
    "Migration was accepted."
)
NOVEMBER_INSTRUCTIONS = """

NOVEMBER LIFECYCLE V1
Only this existing native session owns the one-minute continuation, with at most
one in-flight computation. This binding's configuration is not proof of a later
scheduled execution. Native event 64145 is creator-agent/idle delivery, NOT
session_automation delivery; only actually observed native scheduled calls may
be reported as such.

Before every new research action use
us_quant.november_research_lifecycle_v1.runtime_guard and the same shared
ssd-continuous-research-v1 lock. Do not patch October DEADLINE/runtime guards,
fake clocks, replay completed entrypoints, or reset the 675-event recovery
baseline. Use an explicitly versioned new November study for an independently
justified question; freeze its law, inputs and implementation before outcomes.
The initial minimum expert-pool study in the preserved instructions is already
complete, not a reason to repeat it.

The historical conditional IBKR market-data permission above does NOT reopen
the terminal unknown2157 probe, allow another startup/history request, rotate
client/provider/endpoint/UA, or reset terminal VIG/SDY429. Term eligibility
9cc114/V1 remains terminal; independent V2 remains blocked by exact RF and full
native event-Interface checks. No replacement RF view, new Adapter, filling,
cropping, source waiver or automatic admission follows a failed review.
New observations need actual acquisition timestamps and separate identities;
never backfill missed prospective days or revise the two October single-day
forward accounts.

At 2026-11-30T15:49:00Z (23:49 Beijing), enter closeout-only mode in THIS SAME
binding: start no new research, preserve partial/in-flight evidence, save the
deadline summary and checkpoint, clear this session's native continuation and
get-confirm absence. Ten minutes is an operational safety margin, not a changed
financial threshold or a second deadline timer. At or after the hard deadline
2026-11-30T15:59:00Z start no research even if closeout was interrupted; report any
late cleanup truthfully. Never create a second timer or promise 24-hour execution
while the host, SSD or native permissions are unavailable.
"""


def phase(now: datetime) -> str:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise QuantError("The November lifecycle requires an actual timezone-aware clock.")
    if now < START:
        return "not_started"
    if now >= DEADLINE:
        return "expired"
    return "closeout_only" if now >= CLOSEOUT else "research"


def status(now: datetime) -> dict:
    current = phase(now)
    return {
        "authorization_id": "november-research-20261103-v1",
        "actual_time_utc": now.astimezone(timezone.utc).isoformat(),
        "start_not_before_utc": START.isoformat(),
        "closeout_only_from_utc": CLOSEOUT.isoformat(),
        "deadline_utc": DEADLINE.isoformat(),
        "phase": current,
        "authorization_window_open": current in {"research", "closeout_only"},
        "new_research_actions_allowed": current == "research",
        "financial_protocol": "ssd_dual_benchmark_net_cagr_v1",
        "financial_protocol_sha256": original.POLICY_SHA256,
        "october_authorization_extended_retroactively": False,
        "configuration_is_scheduled_execution_proof": False,
        "order_authority": False,
    }


def runtime_guard() -> None:
    if phase(datetime.now(timezone.utc)) != "research":
        raise QuantError("November research is not open; preserve evidence and close out.")
    if (
        os.environ.get("SSD_RESEARCH_ROOT") != str(SSD)
        or SSD.resolve() != SSD
        or not os.path.ismount(SSD.parent)
        or ROOT != WORKSPACE / "research/workbench"
        or ROOT.resolve() != ROOT
        or ROOT.stat().st_dev != SSD.parent.stat().st_dev
        or Path.cwd().resolve() != WORKSPACE
        or Path(sys.prefix).resolve() != ROOT / ".venv"
        or (WORKSPACE / ".git").resolve() != WORKSPACE / ".git"
        or not (WORKSPACE / ".git").is_dir()
        or (WORKSPACE / ".git/objects/info/alternates").exists()
    ):
        raise QuantError("Keep the actual mounted SSD, isolated interpreter and independent Git.")
    git = ["git", "--no-pager", "-C", str(WORKSPACE)]
    branch = subprocess.check_output(git + ["branch", "--show-current"], text=True).strip()
    directory = subprocess.check_output(
        git + ["rev-parse", "--absolute-git-dir"], text=True
    ).strip()
    if branch != BRANCH or Path(directory) != WORKSPACE / ".git":
        raise QuantError("Keep the accepted physical SSD Git directory and handoff branch.")
    expected_environment = {
        "PIP_CACHE_DIR": str(SSD / "cache/pip"),
        "XDG_CACHE_HOME": str(SSD / "cache"),
        "MPLCONFIGDIR": str(SSD / "cache/matplotlib"),
        "TMPDIR": str(SSD / "tmp") + "/",
        "JOBLIB_TEMP_FOLDER": str(SSD / "tmp/joblib"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        raise QuantError("Source the accepted SSD caches, temporary paths and one-thread runtime.")
    if (
        AUTHORIZATION.resolve() != AUTHORIZATION
        or file_digest(AUTHORIZATION) != AUTHORIZATION_SHA256
        or file_digest(original.POLICY) != original.POLICY_SHA256
    ):
        raise QuantError("The new authorization or original frozen financial protocol changed.")
    authority = read_json(AUTHORIZATION)
    if (
        authority["research_owner"]["session_id"] != SESSION
        or authority["research_owner"]["workspace"] != str(WORKSPACE)
        or authority["research_owner"]["branch"] != BRANCH
        or authority["authorization_window"]["deadline_utc"] != "2026-11-30T15:59:00Z"
        or authority["continuation"]["every_minutes"] != 1
        or authority["continuation"]["maximum_active_research_computations"] != 1
    ):
        raise QuantError("Keep this session's exact confirmed November ownership and limits.")
    if phase(datetime.now(timezone.utc)) != "research":
        raise QuantError("The actual clock crossed into November closeout during verification.")


@contextmanager
def research_job() -> Iterator[None]:
    runtime_guard()
    with original.research_job(LOCK):
        runtime_guard()
        yield


def continuation_prompt(previous: str) -> str:
    if digest_bytes(previous.encode()) != OLD_PROMPT_SHA256:
        raise QuantError("Derive only from the actual byte-verified complete prior prompt.")
    replacements = (
        (OLD_AUTHORITY, NEW_AUTHORITY, 1),
        ("2026-10-31T23:59:00+08:00", "2026-11-30T23:59:00+08:00", 1),
        ("2026-10-31T15:59:00Z", "2026-11-30T15:59:00Z", 2),
    )
    updated = previous
    for old, new, count in replacements:
        if updated.count(old) != count:
            raise QuantError("The protected prompt has an unexpected operational authority/date.")
        updated = updated.replace(old, new)
    return updated + NOVEMBER_INSTRUCTIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit nonretroactive November research lifecycle."
    )
    parser.add_argument("action", choices=("check", "status"))
    args = parser.parse_args()
    if args.action == "check":
        runtime_guard()
    print(json.dumps(status(datetime.now(timezone.utc)), indent=2))


if __name__ == "__main__":
    main()
