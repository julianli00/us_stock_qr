from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.research_delivery import outbox_path, read_outbox  # noqa: E402
from src.us_quant.research_reporting import ReportError, STREAM, report_from_root  # noqa: E402


LABEL = "com.usstockqr.daily-research-report"
OLD_LABEL = "com.usstockqr.slack-signal-monitor"


def configuration(root: Path = ROOT) -> dict:
    template = ROOT / "config/com.usstockqr.daily-research-report.plist.template"
    config = plistlib.loads(template.read_bytes())
    private = root / "artifacts/private/research_reporting"
    replacements = {
        "__ROOT__": str(root),
        "__PYTHON__": str(root / ".venv/bin/python"),
        "__SCRIPT__": str(root / "scripts/send_slack_signal_once.py"),
        "__STDOUT__": str(private / "launchd.stdout.log"),
        "__STDERR__": str(private / "launchd.stderr.log"),
    }
    config["ProgramArguments"] = [replacements.get(item, item) for item in config["ProgramArguments"]]
    for name in ("WorkingDirectory", "StandardOutPath", "StandardErrorPath"):
        config[name] = replacements[config[name]]
    return config


def old_monitor_contained(domain: str) -> bool:
    old = subprocess.run(["launchctl", "print", f"{domain}/{OLD_LABEL}"], capture_output=True, text=True)
    absent = old.returncode != 0 and bool(re.search(r"could not find(?: specified)? service", old.stdout + old.stderr, re.I))
    disabled = subprocess.run(["launchctl", "print-disabled", domain], capture_output=True, text=True)
    marked = bool(re.search(r'"' + re.escape(OLD_LABEL) + r'"\s*=>\s*(?:disabled|true)\b', disabled.stdout))
    return absent and disabled.returncode == 0 and marked


def reviewed_session_receipt(root: Path, now: datetime) -> str:
    report = report_from_root(root, now)
    state = read_outbox(outbox_path(root))
    key = f"canonical-slack:{STREAM}:{report.session}"
    if state["deliveries"].get(key, {}).get("state") != "SENT":
        raise ReportError("reviewed_session_receipt_required_before_bootstrap")
    return report.session


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or install only the reviewed research-report LaunchAgent.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = configuration()
    if not args.install:
        print(json.dumps({"installed": False, "reason": "dry_run", "configuration": config}, sort_keys=True))
        return 0
    domain = f"gui/{os.getuid()}"
    try:
        if not old_monitor_contained(domain):
            raise ReportError("old_monitor_not_persistently_contained")
        session = reviewed_session_receipt(ROOT, datetime.now(timezone.utc))
        destination = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
        data = plistlib.dumps(config, sort_keys=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ReportError("existing_launchagent_differs_do_not_overwrite")
        private = ROOT / "artifacts/private/research_reporting"
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(data)
            destination.chmod(0o600)
        loaded = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"], capture_output=True, text=True)
        if loaded.returncode != 0:
            bootstrap = subprocess.run(["launchctl", "bootstrap", domain, str(destination)], capture_output=True, text=True)
            if bootstrap.returncode:
                raise ReportError("research_launchagent_bootstrap_failed")
        verified = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"], capture_output=True, text=True)
        if verified.returncode:
            raise ReportError("research_launchagent_not_loaded")
    except ReportError as exc:
        print(json.dumps({"installed": False, "reason": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({
        "installed": True, "label": LABEL, "reviewed_receipt_session": session,
        "interval_seconds": 900, "eligibility_timezone": "Asia/Shanghai", "eligible_after": "06:00",
        "keepalive": False, "old_monitor_contained": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
