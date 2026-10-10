from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from us_quant.config import QuantError
from us_quant.prospective_data import ProspectiveArchive, validate_policy
from us_quant.storage import read_json, write_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/prospective-data.json")


def moment(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def fixture_fetch(policy, day, output):
    write_json(
        output / "synthetic-source.json", {"session": str(day.date()), "synthetic_test_only": True}
    )
    return {
        "session": str(day.date()),
        "all_required_sources_verified": True,
        "strategy_returns_calculated": False,
    }


def test_first_snapshot_is_baseline_only_and_same_session_is_not_recounted(tmp_path, policy):
    now = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(now)
    first = archive.collect(fixture_fetch, lambda: now)
    assert first["baseline_session"] == "2026-10-09"
    assert first["complete_snapshots"] == 1 and first["observations_after_baseline"] == 0
    second = archive.collect(lambda *args: pytest.fail("duplicate should not fetch"), lambda: now)
    assert second["action"] == "already_collected"
    assert second["receipt_chain_head"] == first["receipt_chain_head"]
    assert (
        not second["strategy_returns_calculated"] and not second["eligible_for_strategy_promotion"]
    )


def test_missing_actual_observation_is_recorded_not_backfilled(tmp_path, policy):
    first = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(first)
    archive.collect(fixture_fetch, lambda: first)
    later = moment("2026-10-14T01:00:00+00:00")
    result = archive.collect(fixture_fetch, lambda: later)
    assert result["latest_session"] == "2026-10-13"
    assert result["missed_sessions"] == ["2026-10-12"]
    assert result["observations_after_baseline"] == 1
    assert not result["contiguous_since_baseline"]
    assert not (archive.directory / "receipts/2026-10-12.json").exists()


def test_next_open_deadline_blocks_late_recording(tmp_path, policy):
    first = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(first)
    late = moment("2026-10-12T15:00:00+00:00")
    result = archive.collect(lambda *args: pytest.fail("too late"), lambda: late)
    assert result["action"] == "missed_preopen_deadline"
    assert result["complete_snapshots"] == 0


def test_failed_fetch_does_not_create_a_completed_snapshot(tmp_path, policy):
    now = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(now)

    def fail(*args):
        raise QuantError("synthetic provider failure")

    with pytest.raises(QuantError, match="provider"):
        archive.collect(fail, lambda: now)
    assert archive.status()["complete_snapshots"] == 0
    failures = list((archive.directory / "attempts").glob("*/failure.json"))
    assert len(failures) == 1 and read_json(failures[0])["observation_recorded"] is False
    assert archive.collect(fixture_fetch, lambda: now)["complete_snapshots"] == 1


def test_mutated_latest_receipt_or_source_is_detected(tmp_path, policy):
    now = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(now)
    archive.collect(fixture_fetch, lambda: now)
    source = next((archive.directory / "attempts").glob("*/synthetic-source.json"))
    original = source.read_bytes()
    source.write_text("{}")
    with pytest.raises(QuantError, match="revised"):
        archive.status()
    source.write_bytes(original)
    receipt = archive.directory / "receipts/2026-10-09.json"
    record = read_json(receipt)
    record["observed_at"] = "2026-10-10T09:00:00+00:00"
    write_json(receipt, record)
    with pytest.raises(QuantError, match="head"):
        archive.status()


def test_collection_policy_cannot_enable_trading_or_backfill(policy):
    validate_policy(policy)
    for key in ("backfill_missed_observations", "compute_strategy_returns", "order_authority"):
        changed = {**policy, key: True}
        with pytest.raises(QuantError):
            validate_policy(changed)


def test_registration_cannot_reset_existing_archive(tmp_path, policy):
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(moment("2026-10-10T08:00:00+00:00"))
    with pytest.raises(QuantError, match="reset"):
        archive.initialize()


def test_capture_that_finishes_after_next_open_is_not_a_valid_observation(tmp_path, policy):
    initial = moment("2026-10-10T08:00:00+00:00")
    before = moment("2026-10-12T13:29:00+00:00")
    after = moment("2026-10-12T13:31:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(initial)
    times = iter((before, after, after))
    with pytest.raises(QuantError, match="deadline"):
        archive.collect(fixture_fetch, lambda: next(times))
    assert archive.status()["complete_snapshots"] == 0


def test_concurrent_capture_only_fetches_once(tmp_path, policy):
    now = moment("2026-10-10T08:00:00+00:00")
    archive = ProspectiveArchive(tmp_path / "archive", policy)
    archive.initialize(now)
    calls = []

    def fetch(*args):
        calls.append("called")
        return fixture_fetch(*args)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: archive.collect(fetch, lambda: now), range(2)))
    assert len(calls) == 1
    assert {result["action"] for result in results} == {"collected", "already_collected"}
    assert archive.status()["complete_snapshots"] == 1
