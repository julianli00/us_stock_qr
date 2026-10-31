import hashlib
import json
from contextlib import nullcontext

import pytest
from test_expert_pair_menu_extension_v1 import saved_decisions

from us_quant import ssd_expert_pair_menu_extension_v1 as study
from us_quant import ssd_seven_expert_meta_v1 as seven
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_original_source_score_target_and_account_entrypoints_are_never_called(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A stored diagnostic replayed a strategy/source/score/account pipeline.")

    for module, names in (
        (seven, ("source_inputs", "initialize", "prepare", "register", "evaluate")),
        (seven.source, ("source_inputs", "initialize", "prepare", "register", "evaluate")),
        (study.rules.source, ("choose_pair", "combine", "build_targets")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    assert study.rules.compare(*saved_decisions())["new_financial_configuration_selected"] is False


def test_complete_dependency_capture_keeps_original_globals():
    old_own = study.import_template.OWN
    refs = study.dependencies()
    assert study.import_template.OWN is old_own
    assert set(study.OWN).issubset({ref["path"] for ref in refs})
    assert "src/us_quant/strategy_expert_pool_v5.py" in {ref["path"] for ref in refs}


def test_public_report_has_no_raw_pairs_weights_or_individual_scores(monkeypatch):
    monkeypatch.setattr(
        study, "reference", lambda path: {"path": "private.json", "sha256": "a" * 64}
    )
    result = {
        "completed_at": "2026-10-31T00:00:00+00:00",
        "intent": {"path": "intent.json", "sha256": "b" * 64},
        "summary": study.rules.compare(*saved_decisions()),
        "source_inputs_checked": 1,
        "analysis_import_and_configuration_dependencies": 1,
        "passed_offline_checks": 1,
        "all_financial_counts_unchanged": {},
        "raw_pairs": "PRIVATE",
        "raw_weights": "PRIVATE",
        "individual_scores": "PRIVATE",
    }
    public = study.public_report(result)
    assert "PRIVATE" not in json.dumps(public)
    assert public["new_strategy_evaluations"] == 0
    assert public["new_financial_configuration_selected"] is False
    assert public["source_scores_risks_targets_or_accounts_recomputed"] is False


def test_second_actual_comparison_attempt_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    (tmp_path / "comparison-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="first actual comparison"):
        study.diagnose()


@pytest.fixture
def synthetic_driver(tmp_path, monkeypatch):
    before, after = saved_decisions()
    for name, path in (
        ("FOLDER", tmp_path),
        ("INTENT", tmp_path / "intent.json"),
        ("RESULT", tmp_path / "result.json"),
        ("PUBLIC", tmp_path / "public.json"),
        ("DESIGN", tmp_path / "design.json"),
    ):
        monkeypatch.setattr(study, name, path)
    study.INTENT.write_text("{}")
    study.DESIGN.write_text("{}")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(
        study, "checked_intent", lambda: {"source_references": [], "passed_offline_checks": 1}
    )
    monkeypatch.setattr(study, "dependencies", lambda: [])
    monkeypatch.setattr(study, "read_inputs", lambda design: (before, after))
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(study, "file_digest", lambda p: study.base.LEGACY_SHA256)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study.current, "current_counts", lambda events: {"synthetic_counts_unchanged": True}
    )
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: None)
    return before, after


def test_driver_uses_one_stored_pass_and_keeps_all_counts(synthetic_driver):
    study.diagnose()
    result = read_json(study.RESULT)
    assert result["summary"]["matched_decisions"] == 121
    assert result["summary"]["added_VVIX_pair_selected"] == 1
    assert result["all_financial_counts_unchanged"] == {"synthetic_counts_unchanged": True}
    assert (
        len((study.FOLDER / "actual-matched-inputs-resolved-v1.jsonl").read_text().splitlines())
        == 242
    )


def test_driver_retains_a_failed_comparison_without_retry(synthetic_driver):
    _, after = synthetic_driver
    after[7]["selection"]["pairs"][0]["training_correlation"] += 1e-5
    with pytest.raises(QuantError):
        study.diagnose()
    assert (study.FOLDER / "failure-v1.json").is_file()
    assert not study.RESULT.exists()
    with pytest.raises(QuantError, match="first actual comparison"):
        study.diagnose()
