from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_expert_dual_relative_selection_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_dual_relative_selection_v1.py"))
    experts, calendar, assets, benchmarks = helper["synthetic"]()
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY), benchmarks
    )
    return SimpleNamespace(close=targets), experts, benchmarks, targets, audits


def test_scalar_oracle_checks_all_fifteen_pairs_and_scores(synthetic):
    result = study.verify_targets(*synthetic)
    assert result["maximum_independent_target_difference"] < 1e-12
    assert result["maximum_independent_expert_weight_difference"] < 1e-12
    assert result["maximum_independent_active_score_difference"] < 1e-10
    assert result["maximum_independent_tracking_risk_difference"] < 1e-12


def test_scalar_oracle_handles_exact_benchmark_replication_as_unscored():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_dual_relative_selection_v1.py"))
    experts, calendar, assets, _ = helper["synthetic"]()
    common = experts[0]

    class SameReturns:
        def __init__(self, identity):
            self.identity = identity

        def observe(self, day, lookback):
            return replace(common.observe(day, lookback), identity=self.identity)

    synced = tuple(SameReturns(e.identity) for e in experts)
    history = calendar[calendar >= "2016-10-06"]
    values = common.observe(calendar[-1], len(history)).net_returns
    benchmarks = pd.DataFrame({"SPY": values, "QQQ": values})
    targets, audits = study.rules.build_targets(
        synced, calendar, assets, read_json(study.rules.POLICY), benchmarks
    )
    result = study.verify_targets(
        SimpleNamespace(close=targets), synced, benchmarks, targets, audits
    )
    assert result["maximum_independent_target_difference"] < 1e-12
    assert all(not a["selection"]["scores_available"] for a in audits)
    assert audits[-1]["selection"]["reason"] == "unscored_degenerate_pair_equal_weights"


@pytest.mark.parametrize(
    "field",
    ["active_score", "tracking", "cost_NaN", "weights", "marginal", "correlation", "score_key"],
)
def test_scalar_oracle_rejects_nonselected_score_or_risk_tampering(synthetic, field):
    data, experts, benchmarks, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    decision = next(a for a in changed if a["selection"]["scores_available"])
    pair = next(
        p
        for p in decision["selection"]["pairs"]
        if p["pair"] != decision["selection"]["selected_pair"]
    )
    if field == "active_score":
        pair["benchmark_scores"]["QQQ"]["validation_net_active_risk_score"] += 0.1
    elif field == "tracking":
        pair["benchmark_scores"]["SPY"]["train_tracking_risk"] += 0.1
    elif field == "cost_NaN":
        pair["estimated_incremental_switch_cost_fraction"] = np.nan
    elif field == "weights":
        pair["expert_weights"][0] = np.nan
    elif field == "marginal":
        pair["training_marginal_variance_contributions"][0] = np.nan
    elif field == "correlation":
        pair["training_correlation"] = np.nan
    else:
        pair["selection_score_key"] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, benchmarks, targets, changed)


def test_scalar_oracle_rejects_changed_target_and_future_clock(synthetic):
    data, experts, benchmarks, targets, audits = synthetic
    changed_target = targets.copy()
    changed_target.loc[changed_target.dropna().index[-1], targets.columns[0]] += 0.01
    with pytest.raises(QuantError, match="scalar"):
        study.verify_targets(data, experts, benchmarks, changed_target, audits)
    changed = copy.deepcopy(audits)
    next(a for a in changed if a["selection"]["scores_available"])["selection"][
        "scoring_benchmark_last_observation"
    ] = "2026-10-05"
    with pytest.raises(QuantError, match="chronological"):
        study.verify_targets(data, experts, benchmarks, targets, changed)


def test_cached_inputs_do_not_invoke_completed_pipeline_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed strategy, admission or account pipeline was invoked.")

    for name in ("initialize", "register", "prepare", "evaluate", "public_result"):
        monkeypatch.setattr(study.source, name, forbidden)
    monkeypatch.setattr(study.source.rules, "build_targets", forbidden)
    monkeypatch.setattr(study.source.execution, "run", forbidden)
    data, experts, _, parent, refs, clocks, benchmarks, _ = study.source_inputs()
    assert len(experts) == 6 and len(clocks) == 6
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert parent["study_id"] == "ssd_strategy_expert_pool_six_20261030_v1"
    assert len(benchmarks) == 2512 and set(benchmarks) == {"SPY", "QQQ"}
    assert benchmarks.index.equals(data.close.loc["2016-10-06":].index)
    assert any(r["path"].endswith("QQQ-10y-base.csv") for r in refs)


def publication_mocks(monkeypatch, found=None):
    calls = []
    commit = "a" * 40
    explicit, imported = (
        "src/us_quant/expert_dual_relative_selection_v1.py",
        "src/us_quant/backtest.py",
    )
    frozen = {
        "study_id": "test_active",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("all_dependencies")
        return {"published_commit": sha, "unchanged_pinned_import_files_verified": 1}

    def packet(root, ref, required, sha):
        assert root == study.ROOT and sha == commit and required == {explicit}
        assert ref["path"] == study.PACKET.relative_to(study.ROOT).as_posix()
        calls.append("new_packet")
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
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_new_fence"))
    return frozen, commit, calls


def test_full_dependency_verification_precedes_new_publication_fence(monkeypatch):
    frozen, commit, calls = publication_mocks(monkeypatch)
    receipt = study.publication(frozen, commit)
    assert calls == ["all_dependencies", "new_packet", "actual_new_fence"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_publication_resume_cannot_use_later_result_commit(monkeypatch):
    frozen, commit, calls = publication_mocks(
        monkeypatch,
        [
            {
                "body": {"study_id": "test_active", "publication": {"published_commit": "d" * 40}},
            }
        ],
    )
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["all_dependencies", "new_packet"]


def test_failed_publication_cannot_start_main_capital(monkeypatch, tmp_path):
    calls = []

    def blocked(frozen, commit):
        calls.append("publication")
        raise QuantError("Publication not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("Main capital started without its publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(
        study.source.previous.previous.source,
        "failure",
        lambda *a: calls.append("retained_failure"),
    )
    with pytest.raises(QuantError, match="Publication not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "retained_failure"]


def test_canonical_export_preserves_all_four_paths_without_private_rows(monkeypatch):
    metrics = {"cagr": 0.13, "sharpe": 1.1, "max_drawdown": 0.14}
    paths = {}
    for key in ("10y-base", "10y-stress", "5y-base", "5y-stress"):
        paths[key] = {
            "path_key": key,
            "metrics": metrics,
            "execution": {"total_cost_usd": 123.0, "model_order_count": 456},
            "gates": {"all_required": False},
            "all_point_gates_pass": False,
            "excess_net_cagr_vs_spy": 0.01,
            "excess_net_cagr_vs_qqq": -0.02,
            "stress_vs_base_cost_qqq_diagnostic": None,
            "independent_bt_max_equity_difference_usd": 1e-10,
            "independent_monthly_feedback": {"verified": True},
            "artifacts": [{"path": "private/daily-account.csv", "sha256": "b" * 64}],
        }
    row = {
        "candidate_id": study.ID,
        "status": "rejected",
        "paths": paths,
        "all_four_paths_same_candidate_and_frozen_parameters": True,
        "actual_factor_participation": {key: {"minimum_three": True} for key in paths},
    }
    saved = {
        "candidates": {study.ID: row},
        "finished_at": "2026-10-30T06:00:00Z",
        "actual_journal_event_count": 502,
        "actual_journal_head_sha256": "a" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    benchmarks = {
        "benchmarks": [
            {"years": y, "scenario": c, "symbol": s, "metrics": {"cagr": 0.12}}
            for y in (10, 5)
            for c in ("base", "stress")
            for s in ("SPY", "QQQ")
        ]
    }
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(
        study,
        "inputs",
        lambda: (None, None, None, None, {"study_id": "test_active", "protocol": {}}, benchmarks),
    )
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "state", lambda f: row)
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(study.common, "counts", lambda e: {"new_completed_strategy_evaluations": 1})
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {"path": p.relative_to(study.ROOT).as_posix(), "sha256": "d" * 64},
    )
    retained = []
    monkeypatch.setattr(study.common, "retain_receipt", lambda p, b, v: retained.append(b))
    study.public_result()
    output = retained[0]
    assert len(output["reviews"]) == 1 and len(output["reviews"][0]["paths"]) == 4
    assert all(p["metrics"] == metrics for p in output["reviews"][0]["paths"])
    assert "private/daily-account.csv" not in str(output)
    assert output["original_records_reclassified"] is False
    assert output["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert output["new_completed_configurations"] == 1
