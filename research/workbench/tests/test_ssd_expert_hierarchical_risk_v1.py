from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import observed_weight_execution as monthly
from us_quant import ssd_expert_hierarchical_risk_v1 as study
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_hierarchical_risk_v1.py"))
    experts, calendar, universe = helper["synthetic"]()
    targets, audits = study.rules.build_targets(
        experts, calendar, universe, read_json(study.rules.POLICY)
    )
    return SimpleNamespace(close=targets), experts, targets, audits


def test_scalar_hierarchy_verifies_every_merge_and_capital_split(synthetic):
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_expert_weight_difference"] < 1e-12
    assert proof["maximum_independent_scalar_target_difference"] < 1e-12
    assert proof["maximum_independent_covariance_difference"] < 1e-12
    assert proof["every_minimum_distance_merge_and_recursive_capital_split_checked"] is True


@pytest.mark.parametrize(
    "kind",
    (
        "covariance",
        "NaN",
        "merge",
        "distance",
        "seriation",
        "split_risk",
        "split_capital",
        "zero_case",
        "weights",
        "clock",
        "cost_claim",
        "risk_fit_claim",
        "prior",
    ),
)
def test_scalar_verifier_rejects_clock_tree_weight_and_risk_tampering(synthetic, kind):
    data, experts, targets, audits = synthetic
    changed = copy.deepcopy(audits)
    row = next(r for r in changed if r["allocation"]["risk_estimate_available"])
    allocation = row["allocation"]
    if kind == "covariance":
        allocation["covariance"][0][0] += 0.1
    elif kind == "NaN":
        allocation["forecast_annualized_RF_excess_volatility"] = np.nan
    elif kind == "merge":
        allocation["canonical_merges"][0]["distance_key"] += 0.1
    elif kind == "distance":
        allocation["risk_distance"][0][1] += 0.1
    elif kind == "seriation":
        allocation["seriated_expert_coordinates"].reverse()
    elif kind == "split_risk":
        allocation["recursive_capital_splits"][0]["left_IVP_variance"] += 0.1
    elif kind == "split_capital":
        allocation["recursive_capital_splits"][0]["left_capital_fraction_of_parent"] += 0.1
    elif kind == "zero_case":
        allocation["recursive_capital_splits"][0]["both_child_risks_zero"] = True
    elif kind == "weights":
        a, b = int(np.argmax(row["expert_weights"])), int(np.argmin(row["expert_weights"]))
        row["expert_weights"][a] -= 0.001
        row["expert_weights"][b] += 0.001
    elif kind == "clock":
        allocation["risk_sample_end"] = "2026-10-05"
    elif kind == "cost_claim":
        allocation["nominal_target_change_is_actual_turnover"] = True
    elif kind == "risk_fit_claim":
        allocation["risk_fit_is_heldout_validation"] = True
    else:
        row["previous_expert_weights"][0] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)


def test_scalar_verifier_rejects_underlying_target_change(synthetic):
    data, experts, targets, audits = synthetic
    changed = targets.copy()
    changed.loc[changed.dropna().index[-1], changed.columns[0]] += 0.01
    with pytest.raises(QuantError, match="target"):
        study.verify_targets(data, experts, changed, audits)


def test_scalar_zero_individual_risk_is_explicit(synthetic):
    _, experts, targets, _ = synthetic

    class ZeroRisk:
        def __init__(self, source):
            self.source, self.identity = source, source.identity

        def observe(self, day, lookback):
            result = self.source.observe(day, lookback)
            return replace(result, net_returns=result.risk_free.copy())

    fixed = (ZeroRisk(experts[0]), *experts[1:])
    changed, audits = study.rules.build_targets(
        fixed, targets.index, tuple(targets.columns), read_json(study.rules.POLICY)
    )
    proof = study.verify_targets(SimpleNamespace(close=changed), fixed, changed, audits)
    assert proof["maximum_independent_expert_weight_difference"] == 0
    assert all(
        not row["allocation"]["risk_estimate_available"]
        and not row["allocation"]["safety_or_success_claimed"]
        for row in audits
    )


def test_cached_loader_never_invokes_closed_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A closed policy, solver, source or main account pipeline was invoked.")

    for owner in (study.source, study.source.source):
        for name in ("initialize", "prepare", "register", "evaluate", "public_result"):
            monkeypatch.setattr(owner, name, forbidden)
        monkeypatch.setattr(owner.rules, "build_targets", forbidden)
        monkeypatch.setattr(owner.execution, "run", forbidden)
    data, experts, policy, parent, refs, clocks, _ = study.source_inputs()
    assert len(experts) == len(clocks) == 6 and len(refs) > 678
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert parent["study_id"] == "ssd_expert_tail_risk_budget_20261030_v1"
    assert policy["methodology"]["prior_failed_covariance_solver_invoked"] is False
    assert len(data.close.loc["2016-10-06":]) == 2512


def test_complete_local_import_closure_is_bound():
    code, policies = study.dependency_inputs()
    paths = {r["path"] for r in code}
    assert study.SHARED_CODE in paths
    assert "src/us_quant/__init__.py" in paths
    assert "src/us_quant/research.py" in paths
    assert not paths.intersection(study.OWN)
    assert {r["path"] for r in policies}.issuperset(
        {"config/dual-benchmark-v1.json", study.SHARED_POLICY}
    )
    assert study.OWN[0] not in {r["path"] for r in policies}


def test_complete_dependencies_precede_the_new_publication_fence(monkeypatch):
    calls = []
    commit = "a" * 40
    explicit, imported = "src/us_quant/expert_hierarchical_risk_v1.py", "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_hierarchy",
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


def test_failed_publication_cannot_start_main_capital(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("New main capital started without its actual publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, None, {}))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(study, "failure", lambda *args: calls.append("failure"))
    with pytest.raises(QuantError, match="Publication not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "failure"]


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_monthly_execution_is_preserved(synthetic, cost, delay):
    _, _, targets, _ = synthetic
    dates, symbols = targets.index, targets.columns
    phase = np.arange(len(dates))
    close = pd.DataFrame(
        {
            s: 100 * np.exp(np.cumsum(0.0001 + 0.001 * np.sin(phase / (i + 3))))
            for i, s in enumerate(symbols)
        },
        index=dates,
    )
    opening = close.shift(1)
    opening.iloc[0] = close.iloc[0]
    data = MarketData(
        opening, close, close.copy(), close * 0 + 1e6, pd.Series(0.00005, index=dates)
    )
    known = targets.copy()
    known.loc[~known.index.map(is_month_end)] = np.nan
    policy = read_json(monthly.POLICY)
    args = (data, known, monthly.CANDIDATES[0], policy, "2017-01-03", "2017-12-29", cost, delay)
    before, before_issued, before_decisions = monthly.run(*args)
    after, after_issued, after_decisions = study.execution.run(*args)
    pd.testing.assert_frame_equal(before.frame, after.frame, check_exact=True)
    pd.testing.assert_frame_equal(before.weights, after.weights, check_exact=True)
    pd.testing.assert_frame_equal(before_issued, after_issued, check_exact=True)
    assert before_decisions == after_decisions


def test_canonical_export_preserves_all_original_paths_without_private_rows(monkeypatch):
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
        "finished_at": "2026-10-30T10:00:00Z",
        "actual_journal_event_count": 528,
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
    frozen = {"study_id": "test_hierarchy", "protocol": {"path": "protocol.json"}}
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
