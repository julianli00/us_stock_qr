from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from us_quant import ssd_expert_full_covariance_risk_budget_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_full_covariance_risk_budget_v1.py"))
    experts, calendar, universe = helper["synthetic"]()
    targets, audits = study.rules.build_targets(
        experts, calendar, universe, read_json(study.rules.POLICY)
    )
    return SimpleNamespace(close=targets), experts, targets, audits


def test_scalar_optimizer_verifies_weights_covariance_and_targets(synthetic):
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_expert_weight_difference"] < 1e-6
    assert proof["maximum_independent_target_difference"] < 1e-12
    assert proof["maximum_independent_covariance_difference"] < 1e-12
    assert proof["maximum_independent_risk_budget_equation_error"] < 1e-8


@pytest.mark.parametrize(
    "kind",
    (
        "covariance",
        "weight",
        "risk",
        "cost_NaN",
        "subperiod",
        "x",
        "risk_end",
        "heldout",
        "mean_NaN",
    ),
)
def test_scalar_oracle_rejects_covariance_weight_risk_and_cost_tampering(synthetic, kind):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    row = next(a for a in changed if a["allocation"]["risk_estimate_available"])
    allocation = row["allocation"]
    if kind == "covariance":
        allocation["covariance"][0][1] += 0.01
    elif kind == "weight":
        row["expert_weights"][0] += 0.001
        row["expert_weights"][1] -= 0.001
    elif kind == "risk":
        allocation["signed_variance_contribution_shares"][0] += 0.1
    elif kind == "cost_NaN":
        allocation["estimated_incremental_switch_cost_fraction"] = np.nan
    elif kind == "subperiod":
        allocation["subperiod_risk_stability"]["recent_42"][
            "combined_annualized_RF_excess_volatility"
        ] += 0.1
    elif kind == "x":
        allocation["unnormalized_solution"][0] *= 2
    elif kind == "risk_end":
        allocation["risk_sample_end"] = "2026-10-05"
    elif kind == "heldout":
        allocation["subperiod_risk_stability"]["recent_42"]["heldout_validation"] = True
    else:
        allocation["subperiod_risk_stability"]["recent_42"]["combined_RF_excess_mean_return"] = (
            np.nan
        )
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)


def test_scalar_oracle_rejects_changed_underlying_target(synthetic):
    data, experts, targets, audits = synthetic
    altered = targets.copy()
    altered.loc[altered.dropna().index[-1], altered.columns[0]] += 0.01
    with pytest.raises(QuantError, match="scalar"):
        study.verify_targets(data, experts, altered, audits)


def test_cached_loader_never_invokes_completed_policy_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed policy/admission/account entrypoint was called.")

    for name in ("initialize", "prepare", "register", "evaluate", "public_result"):
        monkeypatch.setattr(study.source, name, forbidden)
    monkeypatch.setattr(study.source.rules, "build_targets", forbidden)
    monkeypatch.setattr(study.source.execution, "run", forbidden)
    data, experts, _, parent, refs, clocks, _ = study.source_inputs()
    assert len(experts) == len(clocks) == 6 and len(refs) > 654
    assert parent["study_id"] == "ssd_strategy_expert_pool_six_20261030_v1"
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert not data.close.empty


def test_all_dependencies_are_checked_before_publication_fence(monkeypatch):
    calls = []
    commit = "a" * 40
    explicit = "src/us_quant/expert_full_covariance_risk_budget_v1.py"
    imported = "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_covariance",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("all_dependencies")
        return {"unchanged_pinned_import_files_verified": 1}

    def packet(root, ref, paths, sha):
        assert root == study.ROOT and paths == {explicit} and sha == commit
        calls.append("packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(study, "reference", lambda p: {"path": "packet.json", "sha256": "c" * 64})
    monkeypatch.setattr(study, "read_json", lambda p: {"files": ["research/workbench/" + explicit]})
    monkeypatch.setattr(study, "matching_events", lambda kind: [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_new_fence"))
    proof = study.publication(frozen, commit)
    assert calls == ["all_dependencies", "packet", "actual_new_fence"]
    assert proof["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_failed_publication_cannot_start_new_capital(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("Main capital started without the actual publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda p: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(
        study.source.previous.previous.source, "failure", lambda *a: calls.append("failure")
    )
    with pytest.raises(QuantError, match="Publication not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "failure"]


def test_public_result_preserves_all_paths_without_private_rows(monkeypatch):
    paths = {}
    for key in ("10y-base", "10y-stress", "5y-base", "5y-stress"):
        paths[key] = {
            "path_key": key,
            "metrics": {"cagr": 0.13, "sharpe": 1.1, "max_drawdown": 0.16},
            "execution": {"total_cost_usd": 123.0, "model_order_count": 456},
            "gates": {"drawdown_at_most_15pct": False},
            "all_point_gates_pass": False,
            "excess_net_cagr_vs_spy": 0.01,
            "excess_net_cagr_vs_qqq": -0.02,
            "stress_vs_base_cost_qqq_diagnostic": None,
            "independent_bt_max_equity_difference_usd": 1e-10,
            "independent_monthly_feedback": {"verified": True},
            "artifacts": [{"path": "private/daily.csv", "sha256": "b" * 64}],
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
        "finished_at": "2026-10-30T07:00:00Z",
        "actual_journal_event_count": 514,
        "actual_journal_head_sha256": "a" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    benchmarks = {
        "benchmarks": [
            {"years": year, "scenario": scenario, "symbol": symbol, "metrics": {"cagr": 0.12}}
            for year in (10, 5)
            for scenario in ("base", "stress")
            for symbol in ("SPY", "QQQ")
        ]
    }
    frozen = {"study_id": "test_covariance", "protocol": {"path": "protocol.json"}}
    exported = []
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, benchmarks))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "state", lambda f: row)
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study.common, "counts", lambda events: {"new_completed_strategy_evaluations": 1}
    )
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "d" * 64})
    monkeypatch.setattr(
        study.common, "retain_receipt", lambda path, body, keys: exported.append(body)
    )
    study.public_result()
    body = exported[0]
    assert len(body["reviews"][0]["paths"]) == 4
    assert all("artifacts" not in p for p in body["reviews"][0]["paths"])
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["original_records_reclassified"] is False
    assert body["new_completed_configurations"] == 1
