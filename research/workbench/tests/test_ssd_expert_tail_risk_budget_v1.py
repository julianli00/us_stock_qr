from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_expert_tail_risk_budget_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_tail_risk_budget_v1.py"))
    experts, calendar, universe, benchmarks = helper["synthetic"]()
    benchmarks = benchmarks * 10
    targets, audits = study.rules.build_targets(
        experts, calendar, universe, read_json(study.rules.POLICY), benchmarks
    )
    return SimpleNamespace(close=targets), experts, benchmarks, targets, audits


def test_scalar_certificate_verifies_joint_tail_plan_and_targets(synthetic):
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_scalar_target_difference"] < 1e-12
    assert proof["maximum_independent_tail_budget_or_mean_difference"] < 1e-12
    assert proof["certified_expert_mix_decisions"] > 0
    assert (
        sum(
            proof[k]
            for k in (
                "explicit_unmatured_equal_decisions",
                "certified_expert_mix_decisions",
                "certified_infeasible_funding_standby_decisions",
            )
        )
        == proof["all_monthly_decisions_checked"]
    )


@pytest.mark.parametrize(
    "kind",
    (
        "risk_budget",
        "risk_NaN",
        "mean_NaN",
        "cost_NaN",
        "dual",
        "weights",
        "marginal",
        "standby_mode",
        "sample_end",
        "proxy_actual_fee",
    ),
)
def test_scalar_oracle_rejects_risk_cost_dual_and_standby_tampering(synthetic, kind):
    data, experts, benchmarks, targets, audits = synthetic
    altered = copy.deepcopy(audits)
    row = next(r for r in altered if r["allocation"]["mode"] == "expert_mix")
    allocation = row["allocation"]
    if kind == "risk_budget":
        allocation["periods"]["older_84"]["risk_budget"] += 0.1
    elif kind == "risk_NaN":
        allocation["periods"]["older_84"]["observed_tail_loss"] = np.nan
    elif kind == "mean_NaN":
        allocation["periods"]["recent_42"]["NET_RF_excess_means"][0] = np.nan
    elif kind == "cost_NaN":
        allocation["estimated_incremental_proportional_cost_fraction"] = np.nan
    elif kind == "dual":
        allocation["primary_solution"]["inequality_dual"][0] += 1
    elif kind == "weights":
        weights = row["expert_weights"]
        largest, smallest = int(np.argmax(weights)), int(np.argmin(weights))
        weights[largest] -= 0.001
        weights[smallest] += 0.001
    elif kind == "marginal":
        allocation["periods"]["recent_42"]["tail_marginal_losses"][0] += 0.1
    elif kind == "standby_mode":
        allocation["mode"] = "funding_standby"
    elif kind == "sample_end":
        allocation["risk_sample_end"] = "2026-10-05"
    else:
        allocation["cost_proxy_is_actual_main_account_cost"] = True
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, benchmarks, targets, altered)


def test_scalar_phase_one_verifies_real_standby_without_inventing_expert(synthetic):
    _, experts, _, targets, _ = synthetic

    class ObservedLoss:
        def __init__(self, source):
            self.source, self.identity = source, source.identity

        def observe(self, day, lookback):
            value = self.source.observe(day, lookback)
            return replace(value, net_returns=value.risk_free - 0.02)

    losses = tuple(ObservedLoss(e) for e in experts)
    rf = experts[0].observe(targets.index[-1], 311).risk_free
    benchmarks = pd.DataFrame({"SPY": rf, "QQQ": rf})
    new_targets, audits = study.rules.build_targets(
        losses, targets.index, tuple(targets.columns), read_json(study.rules.POLICY), benchmarks
    )
    proof = study.verify_targets(
        SimpleNamespace(close=new_targets), losses, benchmarks, new_targets, audits
    )
    assert proof["certified_infeasible_funding_standby_decisions"] > 0
    altered = copy.deepcopy(audits)
    row = next(r for r in altered if r["allocation"]["mode"] == "funding_standby")
    row["allocation"]["phase_one_solution"]["inequality_dual"][0] += 1
    with pytest.raises(QuantError):
        study.verify_targets(
            SimpleNamespace(close=new_targets), losses, benchmarks, new_targets, altered
        )


def test_changed_underlying_target_or_nominal_cost_reference_is_rejected(synthetic):
    data, experts, benchmarks, targets, audits = synthetic
    changed = targets.copy()
    changed.loc[changed.dropna().index[-1], changed.columns[0]] += 0.01
    with pytest.raises(QuantError, match="target"):
        study.verify_targets(data, experts, benchmarks, changed, audits)
    altered = copy.deepcopy(audits)
    altered[-1]["previous_nominal_meta_target"][0] += 0.01
    with pytest.raises(QuantError, match="clocks"):
        study.verify_targets(data, experts, benchmarks, targets, altered)


def test_cached_sources_exclude_terminal_covariance_version(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed policy, admission or account pipeline was invoked.")

    for name in ("initialize", "prepare", "register", "evaluate", "public_result"):
        monkeypatch.setattr(study.source, name, forbidden)
    monkeypatch.setattr(study.source.rules, "build_targets", forbidden)
    monkeypatch.setattr(study.source.execution, "run", forbidden)
    data, experts, policy, parent, refs, clocks, benchmark, _ = study.source_inputs()
    assert len(experts) == len(clocks) == 6 and len(refs) > 654
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert "expert_pool_six_full_covariance_erc_v1" not in policy["expert_ids"]
    assert parent["study_id"] == "ssd_strategy_expert_pool_six_20261030_v1"
    assert len(benchmark) == 2512 and set(benchmark) == {"SPY", "QQQ"}
    assert benchmark.index.equals(data.close.loc["2016-10-06":].index)


def test_complete_dependency_publication_precedes_new_fence(monkeypatch):
    calls = []
    commit = "a" * 40
    explicit, imported = "src/us_quant/expert_tail_risk_budget_v1.py", "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_expert_tail",
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


def test_publication_failure_cannot_start_any_main_account(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("Capital started without the actual preregistration publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda p: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, None, {}, None))
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


def test_canonical_export_preserves_original_four_paths_without_private_rows(monkeypatch):
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
        "finished_at": "2026-10-30T09:00:00Z",
        "actual_journal_event_count": 516,
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
    frozen = {"study_id": "test_expert_tail", "protocol": {"path": "protocol.json"}}
    exported = []
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, None, frozen, benchmarks))
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
