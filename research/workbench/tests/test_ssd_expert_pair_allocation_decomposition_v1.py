import copy
from pathlib import Path

import pytest

from us_quant import ssd_expert_pair_allocation_decomposition_v1 as study
from us_quant import ssd_expert_pair_ensemble_v1 as original
from us_quant import ssd_expert_pair_menu_extension_v1 as previous_diagnosis
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_source_pair_and_log_inputs_are_read_without_original_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A source scorer,target,account or old diagnostic entrypoint was replayed.")

    for module, names in (
        (original, ("initialize", "inputs", "prepare", "register", "evaluate", "verify_targets")),
        (original.rules, ("select", "build_targets", "combine")),
        (original.rules.source, ("choose_pair", "build_targets")),
        (previous_diagnosis, ("freeze", "diagnose", "checked_intent")),
        (previous_diagnosis.rules, ("compare",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    packet = read_json(original.PACKET)
    proof = read_json(study.base.checked_reference(packet["entries"][0]["target_proof"]["proof"]))
    design = {
        "source_information_audits": proof["information_audits"],
        "source_actual_input_resolved_log": proof["actual_input_resolved_records"],
    }
    u, o = study.read_inputs(design)
    assert len(u) == len(o) == 121
    assert u[0]["decision_session"] == "2016-10-05"
    assert u[-1]["decision_session"] == o[-1]["decision_session"] == "2026-09-30"


def test_inconsistent_frozen_source_log_is_an_error(monkeypatch):
    packet = read_json(original.PACKET)
    proof = read_json(study.base.checked_reference(packet["entries"][0]["target_proof"]["proof"]))
    design = {
        "source_information_audits": proof["information_audits"],
        "source_actual_input_resolved_log": proof["actual_input_resolved_records"],
    }
    actual = study.read_json

    def changed(path):
        value = actual(path)
        if Path(path).name == "information-audits-v1.json":
            value = copy.deepcopy(value)
            value["candidates"][study.rules.UNIFORM][0]["expert_weights"][0] += 0.1
        return value

    monkeypatch.setattr(study, "read_json", changed)
    with pytest.raises(QuantError, match="log disagrees"):
        study.read_inputs(design)


def test_complete_dependency_capture_keeps_original_globals():
    original_own = original.OWN
    original_policy = original.rules.POLICY
    refs = study.dependencies()
    assert original.OWN is original_own and original.rules.POLICY is original_policy
    assert set(study.OWN).issubset({row["path"] for row in refs})


def test_public_report_contains_only_aggregate_diagnostic_not_individual_source_rows(monkeypatch):
    monkeypatch.setattr(study, "reference", lambda path: {"path": path.name, "sha256": "a" * 64})
    report = study.public_report(
        {
            "completed_at": "2026-10-31T03:50:00Z",
            "intent": {"path": "intent.json", "sha256": "b" * 64},
            "summary": {"matched_decisions": 121, "maximum_identity_residual": 0.0},
            "source_inputs_checked": 1274,
            "analysis_import_and_configuration_dependencies": 300,
            "passed_offline_checks": 20,
            "all_financial_counts_unchanged": {"new_registered_candidates": 64},
        }
    )
    assert report["summary"]["matched_decisions"] == 121
    assert "uniform" not in report and "individual_pair_rows" not in report
    assert report["individual_pairs_weights_scores_or_raw_source_records_published"] is False
    assert report["main_account_PNL_fees_CAGR_or_drawdown_counterfactual_computed"] is False
    assert report["new_strategy_evaluations"] == report["new_funded_paths"] == 0


def test_second_actual_diagnostic_attempt_is_terminal(monkeypatch, tmp_path):
    (tmp_path / "comparison-attempt-v1.json").write_text("{}")
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "RESULT", tmp_path / "missing-result.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "checked_intent", lambda: pytest.fail("A second pass was started."))
    with pytest.raises(QuantError, match="second comparison"):
        study.diagnose()
