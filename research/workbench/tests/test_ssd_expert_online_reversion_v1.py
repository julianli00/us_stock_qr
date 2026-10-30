from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import observed_weight_execution as monthly
from us_quant import online_reversion as old_asset_rule
from us_quant import ssd_expert_online_reversion_v1 as study
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_online_reversion_v1.py"))
    experts, calendar, universe = helper["synthetic"]()
    records = []
    targets, audits = study.rules.build_targets(
        experts, calendar, universe, read_json(study.rules.POLICY), record_learning=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=universe))
    return data, experts, targets, audits, records


def test_independent_daily_replay_and_composition(synthetic, monkeypatch):
    def forbidden(*args):
        pytest.fail("The independent verifier reused the candidate update implementation.")

    monkeypatch.setattr(study.rules, "advance", forbidden)
    monkeypatch.setattr(study.rules, "update_weights", forbidden)
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_expert_weight_difference"] <= 1e-10
    assert proof["maximum_independent_scalar_target_difference"] <= 1e-10
    assert proof["unscored_initial_main_decisions"] == 1
    assert proof["independent_complete_daily_source_observations"] == len(synthetic[0].close) - 1
    assert proof["independent_statistical_updates_per_candidate"] == len(synthetic[0].close) - 5
    assert proof["main_account_or_old_source_pipeline_replayed"] is False


@pytest.mark.parametrize(
    "kind",
    (
        "prior",
        "relative",
        "history",
        "forecast",
        "weights",
        "NaN",
        "day",
        "maturity",
        "guarantee",
        "extra",
    ),
)
def test_independent_verifier_rejects_learning_and_clock_tampering(synthetic, kind):
    data, experts, targets, audits, records = synthetic
    changed = copy.deepcopy(records)
    entry = next(
        row for row in changed if row["stage"] == "input" and row["source_price_observations"] > 8
    )
    resolved = next(
        row
        for row in changed
        if row["stage"] == "resolved" and row["formation_price_observations"] > 8
    )
    if kind == "prior":
        entry["algorithm_weights_before"][0] += 0.01
    elif kind == "relative":
        entry["source_NET_relatives"][0] += 0.01
    elif kind == "history":
        entry["source_history_after"][0][0] += 1
    elif kind == "forecast":
        resolved["forecast"][0] += 0.01
    elif kind == "weights":
        resolved["algorithm_weights_after"][0] += 0.01
    elif kind == "NaN":
        resolved["algorithm_weights_after"][0] = np.nan
    elif kind == "day":
        entry["session"] = "2026-10-05"
    elif kind == "maturity":
        resolved["formation_mature"] = False
    elif kind == "guarantee":
        resolved["criterion_or_profit_guaranteed"] = True
    else:
        changed.append(copy.deepcopy(changed[-1]))
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, audits, changed)


@pytest.mark.parametrize("kind", ("target", "previous_main", "maturity", "clock", "capital"))
def test_changed_main_target_or_learning_capital_claim_is_rejected(synthetic, kind):
    data, experts, targets, audits, records = synthetic
    changed = copy.deepcopy(audits)
    changed_targets = {name: target.copy() for name, target in targets.items()}
    if kind == "target":
        changed_targets[study.IDS[0]].iloc[-1, 0] += 0.01
    elif kind == "previous_main":
        changed[1]["previous_applied_expert_weights"][study.IDS[0]][0] += 0.01
    elif kind == "maturity":
        changed[0]["source_assessment_mature"] = True
    elif kind == "clock":
        changed[1]["source_NET_daily_observations"] += 1
    else:
        changed[1]["learning_state_is_main_wealth"] = True
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, changed_targets, changed, records)


def test_scalar_projection_agrees_with_public_simplex_implementation():
    rng = np.random.default_rng(53048)
    for scale in (0.01, 1, 1000, 1e8):
        for _ in range(10):
            proposal = rng.normal(size=6) * scale
            expected = old_asset_rule.simplex_projection(proposal)
            actual = study.independent_projection(proposal.tolist())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_cached_sources_never_call_closed_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed policy,source learning or capital pipeline was invoked.")

    for owner in (study.source, study.source.previous, study.source.previous.previous):
        for name in ("initialize", "prepare", "register", "evaluate", "public_result"):
            monkeypatch.setattr(owner, name, forbidden)
        monkeypatch.setattr(owner.rules, "build_targets", forbidden)
    monkeypatch.setattr(old_asset_rule, "build_online_signals", forbidden)
    monkeypatch.setattr(study.rules, "build_targets", forbidden)
    monkeypatch.setattr(study.execution, "run", forbidden)
    data, experts, policy, parent, refs, clocks, _ = study.source_inputs()
    assert len(experts) == len(clocks) == 6 and len(refs) >= 669
    assert tuple(expert.identity.identifier for expert in experts) == study.rules.EXPERT_IDS
    assert parent["study_id"] == "ssd_strategy_expert_pool_six_20261030_v1"
    assert policy["methodology"]["headroom_stress_V1_repaired_or_retried"] is False
    assert len(data.close.loc["2016-10-06":]) == 2512


def test_complete_import_closure_is_bound_without_mutating_the_template():
    original = study.import_template.OWN
    code, policies = study.dependency_inputs()
    paths = {ref["path"] for ref in code}
    assert study.SHARED_CODE in paths
    assert "src/us_quant/__init__.py" in paths and "src/us_quant/research.py" in paths
    assert not paths.intersection(study.OWN)
    assert study.OWN[0] not in {ref["path"] for ref in policies}
    assert study.import_template.OWN is original
    assert study.import_template.dependency_inputs.__globals__["OWN"] is original


def test_publication_checks_all_dependencies_before_fence(monkeypatch):
    calls = []
    commit = "a" * 40
    explicit, original = study.SHARED_CODE, "src/us_quant/backtest.py"
    frozen = {
        "study_id": "test_reversion",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, original)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {ref["path"] for ref in value["code"]} == {explicit, original}
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
        pytest.fail("Main capital started without the actual preregistration publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, {}))
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(study, "failure", lambda *args: calls.append("failure"))
    with pytest.raises(QuantError, match="Publication not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "failure"]


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_monthly_execution_is_preserved(synthetic, cost, delay):
    raw = synthetic[2][study.IDS[0]]
    dates, symbols = raw.index, raw.columns
    phase = np.arange(len(dates))
    close = pd.DataFrame(
        {
            symbol: 100 * np.exp(np.cumsum(0.0001 + 0.001 * np.sin(phase / (i + 3))))
            for i, symbol in enumerate(symbols)
        },
        index=dates,
    )
    opening = close.shift(1)
    opening.iloc[0] = close.iloc[0]
    data = MarketData(
        opening, close, close.copy(), close * 0 + 1e6, pd.Series(0.00005, index=dates)
    )
    known = raw.copy()
    known.loc[~known.index.map(is_month_end)] = np.nan
    policy = read_json(monthly.POLICY)
    args = (data, known, monthly.CANDIDATES[0], policy, "2017-03-01", "2017-12-29", cost, delay)
    before, before_issued, before_decisions = monthly.run(*args)
    after, after_issued, after_decisions = study.execution.run(*args)
    pd.testing.assert_frame_equal(before.frame, after.frame, check_exact=True)
    pd.testing.assert_frame_equal(before.weights, after.weights, check_exact=True)
    pd.testing.assert_frame_equal(before_issued, after_issued, check_exact=True)
    assert before_decisions == after_decisions


def test_canonical_export_preserves_original_eight_paths_without_private_rows(monkeypatch):
    keys = ("10y-base", "10y-stress", "5y-base", "5y-stress")
    paths = {
        key: {
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
        for key in keys
    }
    rows = {
        name: {
            "candidate_id": name,
            "status": "rejected",
            "paths": paths,
            "all_four_paths_same_candidate_and_frozen_parameters": True,
            "actual_factor_participation": {key: {"minimum_three": True} for key in keys},
        }
        for name in study.IDS
    }
    saved = {
        "candidates": rows,
        "finished_at": "2026-10-30T10:00:00Z",
        "actual_journal_event_count": 580,
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
    frozen = {"study_id": "synthetic_reversion", "protocol": {"path": "protocol.json"}}
    exported = []
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, benchmarks))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: rows)
    monkeypatch.setattr(study, "read_json", lambda path: saved)
    monkeypatch.setattr(study.base, "journal_events", lambda path: [])
    monkeypatch.setattr(
        study.common, "counts", lambda events: {"new_completed_strategy_evaluations": 2}
    )
    monkeypatch.setattr(study, "reference", lambda path: {"path": path.name, "sha256": "d" * 64})
    monkeypatch.setattr(
        study.common, "retain_receipt", lambda path, body, keys: exported.append(body)
    )
    study.public_result()
    body = exported[0]
    assert len(body["reviews"]) == 2 and sum(len(r["paths"]) for r in body["reviews"]) == 8
    assert all("artifacts" not in p for row in body["reviews"] for p in row["paths"])
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["original_records_reclassified"] is False
    assert body["investment_objective_verified"] is False
