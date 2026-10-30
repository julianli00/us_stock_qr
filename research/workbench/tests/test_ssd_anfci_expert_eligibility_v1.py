import json

import pytest
from test_expert_anfci_scope_v1 import synthetic

from us_quant import ssd_anfci_expert_eligibility_v1 as review
from us_quant import ssd_anfci_residual_research_v1 as original
from us_quant.config import QuantError


def test_closed_financial_and_source_entrypoints_are_never_called(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A cached eligibility review replayed a closed source or account entrypoint.")

    for module, names in (
        (review.prior, ("initialize", "prepare", "register", "evaluate", "inputs")),
        (original, ("inputs", "information_inputs", "prove_targets", "register", "evaluate_path")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    policy, source, decisions = synthetic()
    documents = {
        "policy.json": policy,
        "readiness.json": source["readiness"],
        "manifest.json": source["manifest"],
        "validation.json": source["validation"],
        "proof.json": decisions["proof"],
        "audits.json": {"decisions": decisions["audits"]},
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value))
    decisions["targets"].to_csv(tmp_path / "targets.csv", float_format="%.17g")
    monkeypatch.setattr(review.base, "checked_reference", lambda ref: tmp_path / ref["path"])
    design = {
        key: {"path": name}
        for key, name in (
            ("source_policy", "policy.json"),
            ("source_readiness", "readiness.json"),
            ("source_manifest", "manifest.json"),
            ("source_validation", "validation.json"),
            ("source_targets", "targets.csv"),
            ("source_target_proof", "proof.json"),
            ("source_information_audits", "audits.json"),
        )
    }
    assert review.scope.assess_saved(*review.read_cached(design))["new_expert_admitted"] is False


def test_public_report_has_no_source_values_targets_or_financial_metrics():
    assessment = review.scope.assess_saved(*synthetic())
    assessment["raw_source_values"] = "PRIVATE"
    assessment["native_source_abstentions"][0]["target"] = "PRIVATE"
    result = {
        "completed_at": "2026-10-30T00:00:00+00:00",
        "analysis_intent": {"sha256": "a" * 64},
        "assessment": assessment,
        "original_standalone_review_preserved": {"status": "rejected"},
        "all_current_financial_counts_unchanged": {},
        "targets": "PRIVATE",
        "financial_metrics": "PRIVATE",
    }
    public = review.public_report(result)
    assert "PRIVATE" not in json.dumps(public)
    assert public["original_full_scope_nonadmission_overridden"] is False
    assert public["new_expert_admitted"] is False and public["new_financial_trials"] == 0


def test_second_actual_review_attempt_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "FOLDER", tmp_path)
    monkeypatch.setattr(review, "runtime_guard", lambda: None)
    (tmp_path / "review-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="first actual review"):
        review.review()
