from __future__ import annotations

import json

import pytest

from us_quant.config import QuantError
from us_quant.ssd_research_v1 import append_event, journal_events


def test_new_journal_is_hash_chained_and_never_resets_a_baseline(tmp_path):
    directory = tmp_path / "events"
    baseline = append_event(directory, "baseline", None, {"legacy_counts": [178, 94, 96]})
    registered = append_event(directory, "registered", "test_candidate", {"frozen": "first"})
    proof = append_event(directory, "target_proof", "test_candidate", {"actual": True})
    events = journal_events(directory)
    assert len(events) == 3
    assert registered["previous_sha256"] == baseline["sha256"]
    assert proof["previous_sha256"] == registered["sha256"]
    with pytest.raises(QuantError, match="cannot be reset"):
        append_event(directory, "baseline", None, {"legacy_counts": [0, 0, 0]})
    with pytest.raises(QuantError, match="recover rather than duplicate"):
        append_event(directory, "registered", "test_candidate", {"frozen": "second"})
    assert len(journal_events(directory)) == 3


def test_missing_registration_does_not_become_a_successful_account(tmp_path):
    directory = tmp_path / "events"
    with pytest.raises(QuantError, match="baseline"):
        append_event(directory, "registered", "candidate", {})
    append_event(directory, "baseline", None, {})
    with pytest.raises(QuantError, match="missing registration"):
        append_event(directory, "path_started", "candidate", {})
    with pytest.raises(QuantError, match="registered identity"):
        append_event(directory, "registered", None, {})


def test_tampering_is_rejected_without_rewriting_old_events(tmp_path):
    directory = tmp_path / "events"
    append_event(directory, "baseline", None, {})
    append_event(directory, "registered", "candidate", {"capital": 10000})
    path = directory / "000000000002.json"
    record = json.loads(path.read_text())
    record["body"]["capital"] = 20000
    path.write_text(json.dumps(record))
    with pytest.raises(QuantError, match="revised"):
        journal_events(directory)
    assert json.loads(path.read_text())["body"]["capital"] == 20000


def test_technical_failures_remain_distinct_from_completed_reviews(tmp_path):
    directory = tmp_path / "events"
    append_event(directory, "baseline", None, {})
    append_event(directory, "registered", "candidate", {})
    append_event(directory, "technical_failure", "candidate", {"reason": "target proof failed"})
    events = journal_events(directory)
    assert sum(row["kind"] == "registered" for row in events) == 1
    assert sum(row["kind"] == "technical_failure" for row in events) == 1
    assert not any(row["kind"] == "review" for row in events)
    with pytest.raises(QuantError, match="terminal technical failure"):
        append_event(directory, "path_started", "candidate", {"path_key": "10y-base"})


def test_review_cannot_invent_missing_actual_paths(tmp_path):
    directory = tmp_path / "events"
    append_event(directory, "baseline", None, {})
    append_event(directory, "registered", "candidate", {})
    with pytest.raises(QuantError, match="four actual path receipts"):
        append_event(directory, "review", "candidate", {"paths": {"10y-base": {}}})
