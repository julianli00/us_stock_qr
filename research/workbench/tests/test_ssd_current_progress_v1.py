from __future__ import annotations

import pytest

from us_quant import ssd_current_progress_v1 as current
from us_quant.config import QuantError


def events():
    rows = [
        {
            "kind": "baseline",
            "candidate_id": None,
            "sha256": "a" * 64,
            "body": {
                "legacy": {
                    "fixed_snapshot_completed_configurations": 178,
                    "fixed_snapshot_recurring_reviews": 94,
                    "fixed_snapshot_registered_versions": 96,
                }
            },
        },
        {"kind": "registered", "candidate_id": "first", "body": {}, "sha256": "b" * 64},
    ]
    rows.extend(
        {
            "kind": "path_completed",
            "candidate_id": "first",
            "body": {"path_key": key},
            "sha256": "c" * 64,
        }
        for key in sorted(current.base.PATH_KEYS)
    )
    rows.append(
        {
            "kind": "review",
            "candidate_id": "first",
            "body": {"status": "rejected"},
            "sha256": "d" * 64,
        }
    )
    return rows


def test_current_totals_derive_from_baseline_and_completed_reviews():
    result = current.current_counts(events())
    assert result["total_legacy_plus_ssd_completed_configurations"] == 179
    assert result["new_account_paths"] == 4
    assert result["new_completed_strategy_evaluations"] == 1
    assert result["new_historical_point_gate_passes"] == 0


def test_pending_and_failed_models_do_not_become_completions():
    rows = events() + [
        {"kind": "registered", "candidate_id": "pending", "body": {}, "sha256": "e" * 64},
        {"kind": "registered", "candidate_id": "failed", "body": {}, "sha256": "f" * 64},
        {"kind": "technical_failure", "candidate_id": "failed", "body": {}, "sha256": "1" * 64},
    ]
    result = current.current_counts(rows)
    assert result["total_legacy_plus_ssd_completed_configurations"] == 179
    assert result["new_pending_candidates"] == result["new_technical_failures"] == 1


def test_progress_replaces_stale_derived_fields_without_modifying_old_updater(monkeypatch):
    seen = []
    monkeypatch.setattr(current.base, "journal_events", lambda path: events())
    monkeypatch.setattr(
        current.base, "progress", lambda phase, **fields: seen.append((phase, fields))
    )
    current.progress("synthetic", active_compute_job=None)
    assert seen[0][1]["total_legacy_plus_ssd_completed_configurations"] == 179
    with pytest.raises(QuantError, match="override"):
        current.progress("synthetic", total_legacy_plus_ssd_completed_configurations=184)
    assert len(seen) == 1


def test_missing_baseline_is_not_zero_completed_history():
    with pytest.raises(QuantError, match="baseline"):
        current.current_counts([])
