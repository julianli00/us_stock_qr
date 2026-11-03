from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from us_quant import dual_benchmark_v1 as original
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant.config import QuantError
from us_quant.storage import digest_bytes, file_digest, read_json


@pytest.mark.parametrize(
    "instant,expected",
    [
        (lifecycle.START - timedelta(microseconds=1), "not_started"),
        (lifecycle.START, "research"),
        (lifecycle.CLOSEOUT - timedelta(microseconds=1), "research"),
        (lifecycle.CLOSEOUT, "closeout_only"),
        (lifecycle.DEADLINE - timedelta(microseconds=1), "closeout_only"),
        (lifecycle.DEADLINE, "expired"),
        (lifecycle.DEADLINE + timedelta(days=1), "expired"),
    ],
)
def test_new_window_and_safe_closeout_use_exact_instants(instant, expected):
    assert lifecycle.phase(instant) == expected
    report = lifecycle.status(instant)
    assert report["new_research_actions_allowed"] == (expected == "research")
    assert report["authorization_window_open"] == (expected in {"research", "closeout_only"})
    assert report["order_authority"] is False
    assert report["configuration_is_scheduled_execution_proof"] is False


def test_beijing_clock_is_compared_as_an_instant_not_a_naive_wall_time():
    beijing = timezone(timedelta(hours=8))
    assert lifecycle.phase(datetime(2026, 11, 30, 23, 49, tzinfo=beijing)) == "closeout_only"
    assert lifecycle.phase(datetime(2026, 11, 30, 23, 59, tzinfo=beijing)) == "expired"
    for bad in (datetime(2026, 11, 4), None, "2026-11-04"):
        with pytest.raises(QuantError, match="timezone-aware"):
            lifecycle.phase(bad)


def test_october_guard_and_entire_financial_protocol_are_unchanged():
    assert original.DEADLINE == datetime(2026, 10, 31, 15, 59, tzinfo=timezone.utc)
    with pytest.raises(QuantError, match="expired"):
        original.before_deadline(lifecycle.START)
    assert file_digest(original.POLICY) == original.POLICY_SHA256
    assert lifecycle.status(lifecycle.START)["financial_protocol_sha256"] == original.POLICY_SHA256
    report = lifecycle.status(lifecycle.START)
    assert report["october_authorization_extended_retroactively"] is False


def test_shared_job_checks_actual_guard_before_and_after_acquiring_the_same_lock(
    tmp_path, monkeypatch
):
    path = tmp_path / "research.lock"
    guard = Mock()
    monkeypatch.setattr(lifecycle, "LOCK", path)
    monkeypatch.setattr(lifecycle, "runtime_guard", guard)
    with lifecycle.research_job():
        assert guard.call_count == 2
        assert read_json(path)["state"] == "running"
        with pytest.raises(QuantError, match="Another research"):
            with original.research_job(path):
                pytest.fail("A second October/November job bypassed the common inode lock")
    assert read_json(path)["state"] == "released"


def test_deadline_crossing_while_acquiring_lock_preserves_released_evidence(tmp_path, monkeypatch):
    path = tmp_path / "research.lock"
    monkeypatch.setattr(lifecycle, "LOCK", path)
    monkeypatch.setattr(
        lifecycle,
        "runtime_guard",
        Mock(side_effect=[None, QuantError("crossed into closeout")]),
    )
    with pytest.raises(QuantError, match="closeout"):
        with lifecycle.research_job():
            pytest.fail("A job started after its actual guard failed")
    assert read_json(path)["state"] == "released"


def test_job_failure_is_not_swallowed_and_the_shared_lock_releases(tmp_path, monkeypatch):
    path = tmp_path / "research.lock"
    monkeypatch.setattr(lifecycle, "LOCK", path)
    monkeypatch.setattr(lifecycle, "runtime_guard", Mock())
    with pytest.raises(QuantError, match="preserve failed work"):
        with lifecycle.research_job():
            raise QuantError("preserve failed work")
    assert read_json(path)["state"] == "released"


@pytest.fixture
def previous_prompt(monkeypatch):
    previous = (
        lifecycle.OLD_AUTHORITY
        + "\nNo quota before 2026-10-31T23:59:00+08:00 (2026-10-31T15:59:00Z)."
        + "\nPreserved financial and safety text, including RF and both benchmark gates."
        + "\nAt or after 2026-10-31T15:59:00Z stop new research."
    )
    monkeypatch.setattr(lifecycle, "OLD_PROMPT_SHA256", digest_bytes(previous.encode()))
    return previous


def test_new_prompt_changes_only_explicit_authority_dates_and_adds_stricter_guards(previous_prompt):
    updated = lifecycle.continuation_prompt(previous_prompt)
    body = updated.removesuffix(lifecycle.NOVEMBER_INSTRUCTIONS)
    restored = (
        body.replace(lifecycle.NEW_AUTHORITY, lifecycle.OLD_AUTHORITY)
        .replace("2026-11-30T23:59:00+08:00", "2026-10-31T23:59:00+08:00")
        .replace("2026-11-30T15:59:00Z", "2026-10-31T15:59:00Z")
    )
    assert restored == previous_prompt
    assert "2026-11-30T15:49:00Z" in updated
    assert "terminal unknown2157" in updated and "terminal VIG/SDY429" in updated
    assert "64145 is creator-agent/idle" in updated
    assert "never backfill" in updated
    assert "2026-10-31T15:59:00Z" not in updated


def test_prompt_rejects_unverified_or_already_transformed_input(previous_prompt):
    with pytest.raises(QuantError, match="byte-verified"):
        lifecycle.continuation_prompt(previous_prompt + "\nchanged gate")
    with pytest.raises(QuantError, match="byte-verified"):
        lifecycle.continuation_prompt(lifecycle.continuation_prompt(previous_prompt))


def test_prompt_rejects_missing_or_duplicate_operational_dates(previous_prompt, monkeypatch):
    for bad in (
        previous_prompt.replace("2026-10-31T23:59:00+08:00", ""),
        previous_prompt + "\n2026-10-31T15:59:00Z",
        previous_prompt.replace(lifecycle.OLD_AUTHORITY, "different authority"),
    ):
        monkeypatch.setattr(lifecycle, "OLD_PROMPT_SHA256", digest_bytes(bad.encode()))
        with pytest.raises(QuantError, match="unexpected operational"):
            lifecycle.continuation_prompt(bad)


def test_runtime_guard_rejects_expired_time_before_reading_any_inputs(monkeypatch):
    class ClosedClock:
        @classmethod
        def now(cls, tz):
            return lifecycle.DEADLINE

    monkeypatch.setattr(lifecycle, "datetime", ClosedClock)
    monkeypatch.setattr(lifecycle, "phase", Mock(return_value="expired"))
    digest = Mock(side_effect=AssertionError("Expired guard read source files"))
    monkeypatch.setattr(lifecycle, "file_digest", digest)
    with pytest.raises(QuantError, match="not open"):
        lifecycle.runtime_guard()
    digest.assert_not_called()


def test_runtime_guard_rejects_internal_disk_fallback(monkeypatch):
    monkeypatch.setattr(lifecycle, "phase", Mock(return_value="research"))
    monkeypatch.delenv("SSD_RESEARCH_ROOT", raising=False)
    with pytest.raises(QuantError, match="mounted SSD"):
        lifecycle.runtime_guard()


@pytest.fixture
def accepted_runtime(tmp_path, monkeypatch):
    ssd = tmp_path / "ssd"
    workspace = ssd / "workspace"
    root = workspace / "research/workbench"
    root.mkdir(parents=True)
    (workspace / ".git").mkdir()
    monkeypatch.chdir(workspace)
    for key, value in (("SSD", ssd), ("WORKSPACE", workspace), ("ROOT", root)):
        monkeypatch.setattr(lifecycle, key, value)
    monkeypatch.setattr(lifecycle.sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(lifecycle.os.path, "ismount", lambda path: path == ssd.parent)
    monkeypatch.setattr(lifecycle, "phase", Mock(return_value="research"))
    monkeypatch.setattr(
        lifecycle.subprocess,
        "check_output",
        lambda args, **kwargs: (
            lifecycle.BRANCH if args[-1] == "--show-current" else str(workspace / ".git")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "file_digest",
        lambda path: (
            original.POLICY_SHA256 if path == original.POLICY else lifecycle.AUTHORIZATION_SHA256
        ),
    )
    authority = {
        "research_owner": {
            "session_id": lifecycle.SESSION,
            "workspace": str(workspace),
            "branch": lifecycle.BRANCH,
        },
        "authorization_window": {"deadline_utc": "2026-11-30T15:59:00Z"},
        "continuation": {"every_minutes": 1, "maximum_active_research_computations": 1},
    }
    monkeypatch.setattr(lifecycle, "read_json", lambda path: authority)
    for key, value in {
        "SSD_RESEARCH_ROOT": str(ssd),
        "PIP_CACHE_DIR": str(ssd / "cache/pip"),
        "XDG_CACHE_HOME": str(ssd / "cache"),
        "MPLCONFIGDIR": str(ssd / "cache/matplotlib"),
        "TMPDIR": str(ssd / "tmp") + "/",
        "JOBLIB_TEMP_FOLDER": str(ssd / "tmp/joblib"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    return authority


def test_runtime_guard_accepts_only_the_expected_complete_runtime(accepted_runtime):
    lifecycle.runtime_guard()
    assert lifecycle.phase.call_count == 2


@pytest.mark.parametrize("key,value", [("TMPDIR", "/tmp/"), ("OMP_NUM_THREADS", "2")])
def test_runtime_guard_rejects_cache_or_parallelism_drift(
    accepted_runtime, monkeypatch, key, value
):
    monkeypatch.setenv(key, value)
    with pytest.raises(QuantError, match="one-thread"):
        lifecycle.runtime_guard()


def test_runtime_guard_rejects_branch_drift(accepted_runtime, monkeypatch):
    monkeypatch.setattr(lifecycle.subprocess, "check_output", lambda *args, **kwargs: "main")
    with pytest.raises(QuantError, match="handoff branch"):
        lifecycle.runtime_guard()


def test_runtime_guard_rejects_changed_authority_or_financial_protocol(
    accepted_runtime, monkeypatch
):
    monkeypatch.setattr(lifecycle, "file_digest", lambda path: "0" * 64)
    with pytest.raises(QuantError, match="frozen financial protocol"):
        lifecycle.runtime_guard()


def test_runtime_guard_rejects_other_session_or_changed_cadence(accepted_runtime):
    for field in ("session_id", "branch"):
        old = accepted_runtime["research_owner"][field]
        accepted_runtime["research_owner"][field] = "changed"
        with pytest.raises(QuantError, match="confirmed November ownership"):
            lifecycle.runtime_guard()
        accepted_runtime["research_owner"][field] = old
    accepted_runtime["continuation"]["every_minutes"] = 2
    with pytest.raises(QuantError, match="confirmed November ownership"):
        lifecycle.runtime_guard()


def test_runtime_guard_rechecks_real_time_after_file_and_git_checks(accepted_runtime, monkeypatch):
    monkeypatch.setattr(lifecycle, "phase", Mock(side_effect=["research", "closeout_only"]))
    with pytest.raises(QuantError, match="clock crossed"):
        lifecycle.runtime_guard()
