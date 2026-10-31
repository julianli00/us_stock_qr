import pytest

from us_quant import ssd_legacy_term_risk_eligibility_v2 as study
from us_quant import volatility_term_risk as source
from us_quant.config import QuantError
from us_quant.research_program import ResearchProgram
from us_quant.storage import read_json


def test_readers_keep_full_source_without_original_V1_check_or_review(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A stopped V1 or original source pipeline/constructor was replayed.")

    monkeypatch.setattr(study.original, "review", forbidden)
    monkeypatch.setattr(study.original.rules, "check", forbidden)
    monkeypatch.setattr(ResearchProgram, "__init__", forbidden)
    for name in ("prepare", "register", "evaluate", "build_targets", "load_terms"):
        monkeypatch.setattr(source, name, forbidden)
    snapshot, bundle, frames, rf, identity = study.read_inputs(read_json(study.DESIGN))
    assert snapshot["candidate_id"] == study.rules.CANDIDATE
    assert len(bundle["paths"]) == len(frames) == 4
    assert len(rf) >= 2512
    assert identity["original_candidate_event_sequence"] == 17
    assert frames[10, "base"]["targets"].index[0].strftime("%Y-%m-%d") == "2015-08-10"


def test_dependency_capture_preserves_original_globals():
    root, own = study.original.ROOT, study.original.OWN
    refs = study.dependencies()
    assert study.original.ROOT is root and study.original.OWN is own
    assert set(study.OWN).issubset({row["path"] for row in refs})


def test_second_actual_V2_review_is_terminal(monkeypatch, tmp_path):
    (tmp_path / "review-attempt-v1.json").write_text("{}")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "RESULT", tmp_path / "missing.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "checked_intent", lambda: pytest.fail("A second V2 pass started."))
    with pytest.raises(QuantError, match="second pass"):
        study.review()


def test_public_conclusion_does_not_claim_admission_or_create_views(monkeypatch):
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "a" * 64})
    report = study.public_report(
        {
            "completed_at": "2026-10-31T06:40:00Z",
            "intent": {"path": "intent.json", "sha256": "b" * 64},
            "summary": {"status": "not_admitted_current_Interface", "blockers": []},
            "source_identity": {"original_candidate_event_sequence": 17},
            "source_inputs_checked": 1333,
            "analysis_dependencies": 300,
            "passed_offline_checks": 14,
            "all_financial_counts_unchanged": {"new_registered_candidates": 64},
        }
    )
    assert report["old_V1_failure_remains_terminal"] is True
    assert report["full_source_prehistory_and_all25early_requests_preserved"] is True
    assert report["new_expert_admitted"] is False
    assert report["new_recommendation_RF_view_Adapter_or_source_waiver_created"] is False
    assert report["new_strategy_evaluations"] == 0
