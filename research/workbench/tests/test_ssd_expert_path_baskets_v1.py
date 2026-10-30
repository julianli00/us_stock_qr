from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import observed_weight_execution as monthly
from us_quant import ssd_expert_path_baskets_v1 as study
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_path_admissible_baskets_v1.py"))
    experts, calendar, universe = helper["synthetic"]()
    benchmark = helper["benchmarks"](calendar)
    targets, audits = study.rules.build_targets(
        experts, calendar, universe, benchmark, read_json(study.rules.POLICY)
    )
    return SimpleNamespace(close=targets), experts, targets, audits, benchmark


def test_independent_scalar_all_subset_weights_risks_scores_and_targets(synthetic):
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_expert_weight_difference"] < 1e-12
    assert proof["maximum_independent_scalar_target_difference"] < 1e-12
    assert proof["maximum_independent_subset_drawdown_difference"] < 1e-10
    assert proof["maximum_independent_active_growth_score_difference"] < 1e-10
    assert proof["independently_checked_subset_cases"] == 63 * 9
    assert proof["unscored_initial_decisions"] == 7
    assert proof["main_account_or_old_source_pipeline_replayed"] is False


@pytest.mark.parametrize(
    "kind",
    (
        "risk",
        "score",
        "distance",
        "NaN",
        "weights",
        "clock",
        "known",
        "maturity",
        "standby",
        "prior",
    ),
)
def test_tampered_basket_risk_score_clock_or_weights_is_rejected(synthetic, kind):
    data, experts, targets, audits, benchmark = synthetic
    changed = copy.deepcopy(audits)
    row = next(a for a in changed if a["source_assessment_mature"])
    card = row["assessment"]["all_subset_assessments"][0]
    if kind == "risk":
        card["assessment_drawdowns"]["full_126"] += 0.1
    elif kind == "score":
        card["minimum_active_growth"] += 0.1
    elif kind == "distance":
        card["nominal_target_L1_distance"] += 0.1
    elif kind == "NaN":
        card["assessment_drawdowns"]["full_126"] = np.nan
    elif kind == "weights":
        row["expert_weights"] = (
            [1 / 6] * 6 if row["expert_weights"] != [1 / 6] * 6 else [1, 0, 0, 0, 0, 0]
        )
    elif kind == "clock":
        row["source_observation_start"] = "2026-10-05"
    elif kind == "known":
        row["expert_target_known_at"][0] = "2026-10-06"
    elif kind == "maturity":
        row["source_assessment_mature"] = False
    elif kind == "standby":
        row["assessment"]["funding_standby"] = not row["assessment"]["funding_standby"]
    else:
        row["previous_nominal_meta_target"][0] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed, benchmark)


def test_changed_underlying_target_or_unscored_assessment_is_rejected(synthetic):
    data, experts, targets, audits, benchmark = synthetic
    altered = targets.copy()
    altered.loc[altered.dropna().index[-1], altered.columns[0]] += 0.01
    with pytest.raises(QuantError, match="target"):
        study.verify_targets(data, experts, altered, audits, benchmark)
    changed = copy.deepcopy(audits)
    changed[0]["assessment"]["selected_mask"] = 1
    with pytest.raises(QuantError, match="unscored"):
        study.verify_targets(data, experts, targets, changed, benchmark)


def test_scalar_enumeration_checks_explicit_standby():
    raw = np.full((126, 6), -0.02)
    bench = np.zeros((126, 2))
    targets = np.zeros((14, 6))
    targets[0] = 0.98
    weights, actual = study.rules.assess_baskets(raw, bench, targets, targets[:, 0])
    expected, selected = study.independent_assessment(raw, bench, targets, targets[:, 0])
    assert weights is None and selected is None
    assert not any(c["assessment_admissible"] for c in expected)
    assert actual["assessment_admissible_subsets"] == 0


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
    assert parent["study_id"] == "ssd_expert_online_log_growth_20261030_v1"
    assert policy["methodology"]["old_ordered_price_sleeve_LP_or_solver_invoked"] is False
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
        "study_id": "test_path_baskets",
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
    _, _, targets, _, _benchmark = synthetic
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
    frozen = {"study_id": "test_path_baskets", "protocol": {"path": "protocol.json"}}
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
