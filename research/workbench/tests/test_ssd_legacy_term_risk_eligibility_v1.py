import copy

import pytest

from us_quant import ssd_legacy_term_risk_eligibility_v1 as study
from us_quant import volatility_term_risk as original
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram
from us_quant.storage import read_json


def test_readers_never_replay_original_source_or_initialize_ledger(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A source pipeline or ResearchProgram constructor was invoked.")

    for name in (
        "prepare",
        "register",
        "evaluate",
        "build_targets",
        "load_terms",
        "monthly_targets",
    ):
        monkeypatch.setattr(original, name, forbidden)
    monkeypatch.setattr(ResearchProgram, "__init__", forbidden)
    connect = study.sqlite3.connect
    observed = []

    def only_readonly(path, *args, **kwargs):
        assert "?mode=ro&immutable=1" in path and kwargs.get("uri") is True
        observed.append(path)
        return connect(path, *args, **kwargs)

    monkeypatch.setattr(study.sqlite3, "connect", only_readonly)
    snapshot, bundle, frames, rf, identity = study.read_inputs(read_json(study.DESIGN))
    assert snapshot["candidate_id"] == study.rules.CANDIDATE
    assert len(bundle["paths"]) == len(frames) == 4
    assert identity["source_identity_and_four_path_completeness_verified"] is True
    assert len(rf) >= 2512 and len(observed) == 1


def test_dependency_capture_preserves_original_globals():
    policy = original.POLICY
    root = original.ROOT
    refs = study.dependencies()
    assert original.POLICY is policy and original.ROOT is root
    assert set(study.OWN).issubset({ref["path"] for ref in refs})


def test_second_actual_review_is_terminal(monkeypatch, tmp_path):
    (tmp_path / "review-attempt-v1.json").write_text("{}")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "RESULT", tmp_path / "missing.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "checked_intent", lambda: pytest.fail("A second review was begun."))
    with pytest.raises(QuantError, match="second pass"):
        study.review()


def test_public_review_reports_blockers_not_admission_or_raw_daily_data(monkeypatch):
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "a" * 64})
    report = study.public_report(
        {
            "completed_at": "2026-10-31T04:40:00Z",
            "intent": {"path": "intent.json", "sha256": "b" * 64},
            "summary": {
                "status": "not_admitted_current_Interface",
                "blockers": [{"reason": "native_event_calendar"}],
            },
            "source_identity": {"original_candidate_event_sequence": 17},
            "source_inputs_checked": 1316,
            "analysis_dependencies": 300,
            "passed_offline_checks": 12,
            "all_financial_counts_unchanged": {"new_registered_candidates": 64},
        }
    )
    assert report["summary"]["blockers"]
    assert report["new_expert_admitted"] is False
    assert report["raw_native_targets_or_daily_accounts_published"] is False
    assert report["new_recommendation_or_RF_view_created"] is False
    assert report["new_strategy_evaluations"] == 0


def test_modified_identity_snapshot_does_not_change_original_ledger(monkeypatch):
    actual = study.read_json

    def changed(path):
        value = actual(path)
        if path.name == "source-identity-snapshot-v1.json":
            value = copy.deepcopy(value)
            value["semantic_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(study, "read_json", changed)
    with pytest.raises(QuantError, match="identity snapshot"):
        study.legacy_snapshot(read_json(study.DESIGN))
