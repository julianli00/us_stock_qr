from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from us_quant import legacy_profit_expert_v1 as admitted
from us_quant import ssd_five_expert_meta_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_strategy_expert_pool_v3.py"))
    experts, calendar, assets = helper["archived_experts"]()
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY)
    )
    return SimpleNamespace(close=targets[study.IDS[0]]), experts, targets, audits


def test_scalar_oracle_checks_all_ten_pairs_and_vector_order(synthetic):
    result = study.verify_targets(*synthetic)
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
    data, experts, targets, audits = synthetic
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
    data, experts, targets, audits = synthetic
    changed = {k: v.copy() for k, v in targets.items()}
    day = changed[study.IDS[0]].dropna().index[-1]
    changed[study.IDS[0]].loc[day, changed[study.IDS[0]].columns[0]] += 0.01
    with pytest.raises(QuantError, match="scalar composition"):
        study.verify_targets(data, experts, changed, audits)


def test_unmatured_state_cannot_acquire_successful_scores(synthetic):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    changed[0]["selection"]["scores_available"] = True
    with pytest.raises(QuantError, match="Unmatured"):
        study.verify_targets(data, experts, targets, changed)


def test_cached_admission_loader_never_invokes_historical_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed admission or old target/account pipeline was invoked.")

    monkeypatch.setattr(admitted, "admit", forbidden)
    monkeypatch.setattr(admitted, "load_profit_expert", forbidden)
    monkeypatch.setattr(study.previous, "source_bundle", forbidden)
    monkeypatch.setattr(study.previous.source, "inputs", forbidden)
    monkeypatch.setattr(study.previous.source, "load_expert", forbidden)
    data, experts, _, _, refs, clocks, _, _ = study.source_inputs()
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert isinstance(experts[-1], ArchivedMonthlyExpert)
    assert all(e.observe("2016-10-05", 126).target is not None for e in experts)
    assert all(not e.observe("2016-10-05", 126).score_mature for e in experts)
    assert all(len(e.observe(data.close.index[-1], 2512).net_returns) == 2512 for e in experts)
    assert clocks[-1]["source_policy_registered_at"] != clocks[-1]["actual_expert_admitted_at"]
    assert clocks[-1]["actual_expert_admitted_at"] == "2026-10-30T00:24:25.009540+00:00"
    assert any(r["path"].endswith("admission-v1.json") for r in refs)


def publication_mocks(monkeypatch, found=None):
    calls = []
    commit = "a" * 40
    explicit = "src/us_quant/strategy_expert_pool_v3.py"
    imported = "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_five",
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
        calls.append("explicit_packet")
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


def test_publication_checks_full_dependencies_before_explicit_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_mocks(monkeypatch)
    result = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet", "actual_fence"]
    assert result["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_publication_resume_cannot_substitute_a_later_commit(monkeypatch):
    found = [
        {
            "body": {
                "study_id": "test_five",
                "publication": {"published_commit": "d" * 40},
            }
        }
    ]
    frozen, commit, calls = publication_mocks(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet"]


def test_failed_publication_cannot_start_any_main_account(monkeypatch, tmp_path):
    calls = []

    def blocked(frozen, commit):
        calls.append("publication")
        raise QuantError("Publication is not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("A funded account started before its publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(
        study.previous.source, "failure", lambda *args: calls.append("retained_failure")
    )
    with pytest.raises(QuantError, match="Publication is not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "retained_failure"]


def test_canonical_export_preserves_all_paths_without_daily_rows(monkeypatch):
    metrics = {"cagr": 0.13, "sharpe": 1.1, "max_drawdown": 0.14}
    metrics.update({f"other_scalar_{i}": i for i in range(13)})
    reviews = {}
    for name in study.IDS:
        paths = {}
        for key in ("10y-base", "10y-stress", "5y-base", "5y-stress"):
            paths[key] = {
                "path_key": key,
                "metrics": metrics,
                "execution": {"total_cost_usd": 123.0, "model_orders": 456},
                "gates": {"all_required": False},
                "all_point_gates_pass": False,
                "excess_net_cagr_vs_spy": 0.01,
                "excess_net_cagr_vs_qqq": -0.02,
                "stress_vs_base_cost_qqq_diagnostic": None,
                "independent_bt_max_equity_difference_usd": 1e-10,
                "independent_monthly_feedback": {"verified": True},
                "artifacts": [{"path": "private/daily-account.csv", "sha256": "b" * 64}],
            }
        reviews[name] = {
            "candidate_id": name,
            "status": "rejected",
            "paths": paths,
            "all_four_paths_same_candidate_and_frozen_parameters": True,
            "actual_factor_participation": {key: {"minimum_three": True} for key in paths},
        }
    saved = {
        "candidates": reviews,
        "finished_at": "2026-10-30T01:00:00Z",
        "actual_journal_event_count": 455,
        "actual_journal_head_sha256": "a" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    benchmarks = {
        "benchmarks": [
            {"years": years, "scenario": scenario, "symbol": symbol, "metrics": {"cagr": 0.12}}
            for years in (10, 5)
            for scenario in ("base", "stress")
            for symbol in ("SPY", "QQQ")
        ]
    }
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(
        study,
        "inputs",
        lambda: (None, None, {"study_id": "test_five", "protocol": {}}, benchmarks),
    )
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "states", lambda f: reviews)
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(study.common, "counts", lambda e: {"new_completed_strategy_evaluations": 2})
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": p.relative_to(study.ROOT).as_posix(),
            "sha256": "d" * 64,
        },
    )
    retained = []
    monkeypatch.setattr(study.common, "retain_receipt", lambda p, b, v: retained.append(b))
    study.public_result()
    output = retained[0]
    assert len(output["reviews"]) == 2
    assert all(len(r["paths"]) == 4 for r in output["reviews"])
    assert all(p["metrics"] == metrics for r in output["reviews"] for p in r["paths"])
    assert "private/daily-account.csv" not in str(output)
    assert output["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert output["new_completed_configurations"] == 2
