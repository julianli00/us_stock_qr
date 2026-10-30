from __future__ import annotations

import copy
import json
import runpy
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from us_quant import ssd_seven_expert_meta_v1 as study
from us_quant import ssd_vvix_uncertainty_research_v1 as original
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_strategy_expert_pool_v5.py"))
    experts, calendar, assets = helper["archived_experts"]()
    records = []
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY), retain=records.append
    )
    return SimpleNamespace(close=targets[study.IDS[0]]), experts, targets, audits, records


def test_scalar_oracle_checks_all_twenty_one_pairs(synthetic):
    result = study.verify_targets(*synthetic[:4])
    assert result["maximum_independent_target_difference"] < 1e-12
    assert result["maximum_independent_expert_weight_difference"] < 1e-12
    assert result["maximum_independent_net_score_difference"] < 1e-10


@pytest.mark.parametrize(
    "field",
    [
        "train_net_risk_score",
        "validation_net_risk_score",
        "estimated_incremental_switch_cost_fraction",
        "training_correlation",
        "expert_weights",
        "training_marginal_variance_contributions",
    ],
)
def test_scalar_oracle_rejects_nonselected_pair_risk_or_cost_changes(synthetic, field):
    data, experts, targets, audits, _ = synthetic
    changed = copy.deepcopy(audits)
    scored = next(a for a in changed if a["selection"]["scores_available"])
    pair = next(
        p for p in scored["selection"]["pairs"] if p["pair"] != scored["selection"]["selected_pair"]
    )
    if isinstance(pair[field], list):
        pair[field][0] = np.nan
    else:
        pair[field] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)


def test_scalar_oracle_rejects_changed_combined_target(synthetic):
    data, experts, targets, audits, _ = synthetic
    changed = {key: value.copy() for key, value in targets.items()}
    day = changed[study.IDS[0]].dropna().index[-1]
    changed[study.IDS[0]].loc[day, changed[study.IDS[0]].columns[0]] += 0.01
    with pytest.raises(QuantError, match="scalar composition"):
        study.verify_targets(data, experts, changed, audits)


def test_unmatured_state_cannot_acquire_successful_scores(synthetic):
    data, experts, targets, audits, _ = synthetic
    changed = copy.deepcopy(audits)
    changed[0]["selection"]["scores_available"] = True
    with pytest.raises(QuantError, match="Unmatured"):
        study.verify_targets(data, experts, targets, changed)


def test_retained_binary64_inputs_and_resolved_records_match(synthetic, tmp_path):
    _, experts, targets, audits, records = synthetic
    path = tmp_path / "synthetic-records.jsonl"
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    checked = study.verify_retained_records(path, experts, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 2 * len(audits)
    changed = copy.deepcopy(records)
    changed[0]["previous_weights"][0] += 1e-10
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in changed))
    with pytest.raises(QuantError, match="pre-update"):
        study.verify_retained_records(path, experts, targets, audits)


def test_cached_loader_keeps_explicit_clock_without_old_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed admission/source/control pipeline was replayed.")

    for module, names in (
        (study.admitted, ("admit",)),
        (original, ("inputs", "register", "evaluate", "native_quotes", "prove_targets")),
        (study.source, ("initialize", "prepare", "register", "evaluate")),
        (study.source.rules, ("build_targets",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    real = study.admitted.load_admitted
    clocks = []

    def cached(universe, rf, *, decision_information_clock):
        clocks.append(decision_information_clock)
        return real(universe, rf, decision_information_clock=decision_information_clock)

    monkeypatch.setattr(study.admitted, "load_admitted", cached)
    data, experts, _, _, refs, introductions, _, _ = study.source_inputs()
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert clocks == [study.rules.CLOCK]
    assert len(experts) == 7 and len(refs) >= 1199
    assert introductions[-1]["source_policy_registered_at"] == "2026-10-28T23:58:17.764688+00:00"
    assert introductions[-1]["actual_expert_admitted_at"] == "2026-10-30T22:58:44.381948+00:00"
    assert all(e.observe("2016-10-05", 126).target is not None for e in experts)
    assert all(len(e.observe(data.close.index[-1], 2512).net_returns) == 2512 for e in experts)


def test_full_import_capture_does_not_mutate_source_globals():
    source_own, source_policy = study.source.OWN, study.source.rules.POLICY
    admitted_own = study.admitted.OWN
    refs = study.dependencies()
    assert study.source.OWN is source_own and study.source.rules.POLICY is source_policy
    assert study.admitted.OWN is admitted_own
    assert set(study.OWN).issubset({row["path"] for row in refs})
    assert "src/us_quant/strategy_expert_pool_v2.py" in {row["path"] for row in refs}


def publication_mocks(monkeypatch, found=None):
    calls = []
    commit = "a" * 40
    explicit = "src/us_quant/strategy_expert_pool_v5.py"
    imported = "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_seven",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {ref["path"] for ref in value["code"]} == {explicit, imported}
        calls.append("full_dependencies")
        return {"published_commit": sha, "unchanged_pinned_import_files_verified": 1}

    def packet(root, reference, required, sha):
        assert root == study.ROOT and sha == commit
        assert reference["path"] == study.PACKET.relative_to(study.ROOT).as_posix()
        assert required == {explicit}
        calls.append("explicit_packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {"path": p.relative_to(study.ROOT).as_posix(), "sha256": "c" * 64},
    )
    monkeypatch.setattr(study, "read_json", lambda p: {"files": ["research/workbench/" + explicit]})
    monkeypatch.setattr(study, "matching_events", lambda kind: found or [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_fence"))
    return frozen, commit, calls


def test_publication_checks_full_dependencies_before_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_mocks(monkeypatch)
    result = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet", "actual_fence"]
    assert result["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_publication_resume_cannot_substitute_a_later_commit(monkeypatch):
    frozen, commit, calls = publication_mocks(
        monkeypatch,
        [{"body": {"study_id": "test_seven", "publication": {"published_commit": "d" * 40}}}],
    )
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet"]


def test_failed_publication_cannot_start_any_main_account(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication is not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("A funded account started before publication.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "no-complete-result.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(study, "failure", lambda *args: calls.append("retained_failure"))
    with pytest.raises(QuantError, match="not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "retained_failure"]


def test_canonical_export_preserves_all_paths_without_daily_rows(monkeypatch):
    namespace = runpy.run_path(str(study.ROOT / "tests/test_ssd_six_expert_meta_v1.py"))
    original_test = namespace["test_canonical_export_preserves_all_paths_without_daily_rows"]
    rebound = FunctionType(original_test.__code__, {**namespace, "study": study})
    rebound(monkeypatch)


def test_original_monthly_execution_is_preserved():
    assert study.execution.evaluate_path is study.source.execution.evaluate_path
    assert study.execution is study.source.execution
