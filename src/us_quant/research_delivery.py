from __future__ import annotations

import fcntl
import http.client
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.us_quant.research_reporting import ReportError, STREAM, read_json, report_from_root


@dataclass(frozen=True)
class Receipt:
    state: str
    code: str
    http_status: int | None = None


def configured_webhook(env_file: Path, variable: str = "SLACK_WEBHOOK_URL") -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", variable):
        raise ReportError("webhook_variable_invalid")
    value = os.environ.get(variable, "").strip()
    if not value and env_file.is_file():
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ReportError("webhook_configuration_unreadable") from None
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, candidate = line.split("=", 1)
            if key.strip() == variable:
                value = candidate.strip().strip("\"'")
    if not value:
        raise ReportError("webhook_not_configured")
    return validate_webhook(value)


def validate_webhook(value: str) -> str:
    if not isinstance(value, str):
        raise ReportError("webhook_configuration_invalid")
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise ReportError("webhook_configuration_invalid") from None
    if (
        parsed.scheme != "https" or parsed.hostname != "hooks.slack.com"
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or not re.fullmatch(r"/services/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", parsed.path)
    ):
        raise ReportError("webhook_configuration_invalid")
    return value


def post_to_slack(webhook: str, text: str, *, timeout: int = 12) -> Receipt:
    request = Request(
        validate_webhook(webhook),
        data=json.dumps({"text": text}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read(128).decode("utf-8", errors="replace").strip()
    except HTTPError as exc:
        state = "FAILED" if 400 <= exc.code < 500 else "UNKNOWN"
        return Receipt(state, f"http_{exc.code}", exc.code)
    except (URLError, OSError, http.client.HTTPException, TimeoutError):
        return Receipt("UNKNOWN", "transport_outcome_unknown")
    if status == 200 and body == "ok":
        return Receipt("SENT", "slack_receipt_ok", status)
    return Receipt("UNKNOWN", "unexpected_slack_response", status)


def outbox_path(root: Path) -> Path:
    return root / "artifacts/private/research_reporting/outbox.json"


def read_outbox(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReportError("outbox_symlink_rejected")
    if not path.exists():
        return {"schema_version": 1, "deliveries": {}, "published_ideas": {}}
    state = read_json(path)
    if (
        state.get("schema_version") != 1 or not isinstance(state.get("deliveries"), dict)
        or not isinstance(state.get("published_ideas"), dict)
    ):
        raise ReportError("outbox_schema_invalid")
    for row in state["deliveries"].values():
        if not isinstance(row, dict) or row.get("state") not in {"PENDING", "SENT", "FAILED", "UNKNOWN"}:
            raise ReportError("outbox_record_invalid")
    return state


def atomic_write(path: Path, state: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ReportError("outbox_symlink_rejected")
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".outbox-", delete=False, encoding="utf-8") as stream:
            name = stream.name
            os.chmod(name, 0o600)
            json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if name is not None and Path(name).exists():
            Path(name).unlink()


def deliver_report(
    root: Path,
    *,
    channel_alias: str = "canonical-slack",
    now: datetime | None = None,
    language: str = "zh",
    dry_run: bool = True,
    transport: Callable[[str], Receipt] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", channel_alias):
        raise ReportError("channel_alias_invalid")
    clock = now or datetime.now(timezone.utc)
    path = outbox_path(root)
    state = read_outbox(path)
    report = report_from_root(root, clock, seen=state["published_ideas"], language=language)
    if dry_run:
        return {"sent": False, "reason": "dry_run", "text": report.text, **report.metadata()}
    if not report.due:
        return {"sent": False, "reason": "before_0600_shanghai", **report.metadata()}
    if transport is None:
        raise ReportError("delivery_transport_missing")
    if path.parent.is_symlink() or not path.parent.resolve().is_relative_to(root.resolve()):
        raise ReportError("outbox_path_rejected")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_outbox(path)
        report = report_from_root(root, clock, seen=state["published_ideas"], language=language)
        key = f"{channel_alias}:{STREAM}:{report.session}"
        previous = state["deliveries"].get(key, {})
        if previous.get("state") == "SENT":
            return {"sent": False, "reason": "session_already_reported", **report.metadata()}
        if previous.get("state") in {"PENDING", "UNKNOWN"}:
            if previous["state"] == "PENDING":
                previous["state"] = "UNKNOWN"
                previous["code"] = "interrupted_delivery_requires_reconciliation"
                atomic_write(path, state)
            return {"sent": False, "reason": "delivery_outcome_unknown", "requires_review": True, **report.metadata()}
        attempt = {
            "state": "PENDING",
            "digest": report.digest,
            "session": report.session,
            "attempted_at_utc": clock.astimezone(timezone.utc).isoformat(),
            "attempt_count": int(previous.get("attempt_count", 0)) + 1,
            "idea_keys": list(report.idea_keys),
        }
        state["deliveries"][key] = attempt
        atomic_write(path, state)
        receipt = transport(report.text)
        if receipt.state not in {"SENT", "FAILED", "UNKNOWN"}:
            raise ReportError("transport_receipt_invalid")
        if receipt.state == "SENT" and receipt.http_status != 200:
            raise ReportError("transport_success_without_receipt")
        attempt.update({"state": receipt.state, "code": receipt.code, "http_status": receipt.http_status})
        if receipt.state == "SENT":
            attempt["receipt_at_utc"] = datetime.now(timezone.utc).isoformat()
            for idea in report.idea_keys:
                state["published_ideas"][idea] = report.session
        atomic_write(path, state)
        return {
            "sent": receipt.state == "SENT",
            "reason": "posted" if receipt.state == "SENT" else (
                "delivery_failed" if receipt.state == "FAILED" else "delivery_outcome_unknown"
            ),
            "receipt_verified": receipt.state == "SENT",
            "channel_history_verified": False,
            "http_status": receipt.http_status,
            "requires_review": receipt.state == "UNKNOWN",
            **report.metadata(),
        }
