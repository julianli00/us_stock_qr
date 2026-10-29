from __future__ import annotations

import copy
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from us_quant import ssd_four_expert_meta_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_strategy_expert_pool_v2.py"))
    experts, calendar, assets = helper["archived_experts"]()
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY)
    )
    return SimpleNamespace(close=targets[study.IDS[0]]), experts, targets, audits


def test_scalar_oracle_checks_the_actual_interface(synthetic):
    result = study.verify_targets(*synthetic)
    assert result["maximum_independent_target_difference"] < 1e-12
    assert result["maximum_independent_expert_weight_difference"] < 1e-12
    assert result["maximum_independent_net_score_difference"] < 1e-10


def test_scalar_oracle_rejects_changed_combined_target(synthetic):
    data, experts, targets, audits = synthetic
    changed = {k: v.copy() for k, v in targets.items()}
    day = changed[study.IDS[0]].dropna().index[-1]
    changed[study.IDS[0]].loc[day, changed[study.IDS[0]].columns[0]] += 0.01
    with pytest.raises(QuantError, match="scalar composition"):
        study.verify_targets(data, experts, changed, audits)


@pytest.mark.parametrize(
    "field",
    [
        "train_net_risk_score",
        "validation_net_risk_score",
        "estimated_incremental_switch_cost_fraction",
        "training_correlation",
    ],
)
def test_scalar_oracle_rejects_changed_score_cost_or_dependence(synthetic, field):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    scored = next(a for a in changed if a["selection"]["scores_available"])
    scored["selection"]["pairs"][0][field] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)


def test_nonselected_pair_cannot_hide_nan_weights(synthetic):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    scored = next(a for a in changed if a["selection"]["scores_available"])
    pair = next(
        p for p in scored["selection"]["pairs"] if p["pair"] != scored["selection"]["selected_pair"]
    )
    pair["expert_weights"][0] = np.nan
    with pytest.raises(QuantError, match="invalid risk"):
        study.verify_targets(data, experts, targets, changed)


def test_warmup_does_not_acquire_spliced_successful_scores(synthetic):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    changed[0]["selection"]["scores_available"] = True
    with pytest.raises(QuantError, match="Unmatured"):
        study.verify_targets(data, experts, targets, changed)


def test_actual_four_expert_inputs_have_available_but_unscored_initial_state():
    data, experts, _, _, _, _, _, _, clocks = study.source_bundle()
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert len({e.identity.mechanism_group for e in experts}) == 4
    assert experts[-1].observe("2016-09-30", 126).target is None
    assert all(e.observe("2016-10-05", 126).target is not None for e in experts)
    assert all(not e.observe("2016-10-05", 126).score_mature for e in experts)
    assert all(
        e.observe("2016-09-30", 126).target.equals(e.observe("2016-10-05", 126).target)
        for e in experts[:3]
    )
    assert (
        len(study.rules.decision_sessions(data.close.index, read_json(study.rules.POLICY))) == 121
    )
    assert clocks[-1]["source_policy_registered_at"] != clocks[-1]["actual_expert_admitted_at"]
    assert clocks[0]["source_policy_registered_at"] != clocks[0]["actual_expert_admitted_at"]


def publication_mocks(monkeypatch, found=None):
    calls = []
    commit = "a" * 40
    explicit = "src/us_quant/strategy_expert_pool_v2.py"
    imported = "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_fixed_four",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("full_dependencies")
        return {"published_commit": sha, "unchanged_pinned_import_files_verified": 1}

    def packet(root, reference, required, sha):
        assert root == study.ROOT and sha == commit
        assert reference["path"] == study.PACKET.relative_to(study.ROOT).as_posix()
        assert required == {explicit}
        calls.append("new_explicit_packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": p.relative_to(study.ROOT).as_posix(),
            "sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda p: {
            "files": [f"research/workbench/{explicit}"],
        },
    )
    monkeypatch.setattr(study, "matching_events", lambda kind: found or [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_fence"))
    return frozen, commit, calls


def test_publication_checks_full_dependencies_before_the_explicit_packet(monkeypatch):
    frozen, commit, calls = publication_mocks(monkeypatch)
    receipt = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "new_explicit_packet", "actual_fence"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_publication_resume_never_replaces_the_original_commit(monkeypatch):
    found = [
        {"body": {"study_id": "test_fixed_four", "publication": {"published_commit": "d" * 40}}}
    ]
    frozen, commit, calls = publication_mocks(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "new_explicit_packet"]
