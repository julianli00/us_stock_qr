from __future__ import annotations

import copy
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import observed_weight_execution as monthly
from us_quant import ssd_expert_contextual_contest_v1 as study
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    helper = runpy.run_path(str(study.ROOT / "tests/test_expert_contextual_contest_v1.py"))
    experts, close = helper["synthetic"]()
    records = []
    targets, audits = study.rules.build_targets(
        experts, close, read_json(study.rules.POLICY), record_learning=records.append
    )
    return SimpleNamespace(close=close), experts, targets, audits, records


def test_independent_integer_counts_and_decimal_risk_composition(synthetic, monkeypatch):
    def forbidden(*args):
        pytest.fail("The independent verifier reused candidate context,credit or loss logic.")

    for name in ("market_context", "winner_credit", "co_loss_weights"):
        monkeypatch.setattr(study.rules, name, forbidden)
    proof = study.verify_targets(*synthetic)
    assert proof["maximum_independent_expert_weight_difference"] < 1e-10
    assert proof["maximum_independent_scalar_target_difference"] < 1e-10
    assert proof["complete_source_months_checked"] == 14
    assert proof["unscored_initial_decisions"] == 7
    assert proof["retained_input_output_records_checked"] == 4 * len(synthetic[3])
    assert proof["context_and_credit_integer_keys_counts_and_loss_assessment_exact"] is True
    assert proof["main_account_or_old_source_pipeline_replayed"] is False


@pytest.mark.parametrize(
    "kind",
    (
        "context",
        "credit",
        "count",
        "month",
        "posterior",
        "loss_date",
        "loss_value",
        "ratio",
        "weight",
        "NaN",
        "maturity",
        "prior_only",
        "capital",
    ),
)
def test_independent_verifier_rejects_context_credit_loss_clock_and_weight_tampering(
    synthetic, kind
):
    data, experts, targets, audits, records = synthetic
    changed = copy.deepcopy(audits)
    row = next(item for item in changed if item["source_assessment_mature"])
    allocation = row["allocations"][study.IDS[0]]
    risk = allocation["co_loss_assessment"]
    if kind == "context":
        row["current_context"] = (row["current_context"] + 1) % 4
    elif kind == "credit":
        row["monthly_credit_update"]["credit_units"][0] += 1
    elif kind == "count":
        row["credit_state_after"][0][0] += 1
    elif kind == "month":
        row["complete_learning_months"] += 1
    elif kind == "posterior":
        allocation["posterior_rational_fractions"][0]["numerator"] = "99999"
    elif kind == "loss_date":
        risk["common_loss_source_sessions"].append("2026-10-05")
    elif kind == "loss_value":
        risk["component_nonoffsetting_loss_means"][0]["numerator"] = "99999"
    elif kind == "ratio":
        risk["component_relative_losses"][0]["numerator"] = "99999"
    elif kind == "weight":
        allocation["applied_expert_weights"][0] += 0.001
    elif kind == "NaN":
        risk["expert_weights_after_tilt"][0] = np.nan
    elif kind == "maturity":
        row["source_assessment_mature"] = False
    elif kind == "prior_only":
        allocation["categorical_forecast_is_prior_only"] = True
    else:
        row["learning_state_is_main_wealth"] = True
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed, records)


@pytest.mark.parametrize("kind", ("operand", "output", "order", "extra", "target"))
def test_retained_input_and_resolved_record_tampering_is_rejected(synthetic, kind):
    data, experts, targets, audits, records = synthetic
    altered = copy.deepcopy(records)
    changed_targets = {name: target.copy() for name, target in targets.items()}
    if kind == "operand":
        altered[0]["context_input"]["current_prices"]["SPY"] += 1.0
    elif kind == "output":
        altered[1]["unified_target"][0] += 0.01
    elif kind == "order":
        altered[1], altered[2] = altered[2], altered[1]
    elif kind == "extra":
        altered.append(copy.deepcopy(altered[-1]))
    else:
        day = changed_targets[study.IDS[0]].dropna(how="all").index[-1]
        symbol = changed_targets[study.IDS[0]].columns[0]
        changed_targets[study.IDS[0]].loc[day, symbol] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, changed_targets, audits, altered)


def test_cached_sources_never_call_closed_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed policy,source learning or capital pipeline was invoked.")

    for owner in (study.source, study.source.previous, study.source.previous.previous):
        for name in ("initialize", "prepare", "register", "evaluate", "public_result"):
            monkeypatch.setattr(owner, name, forbidden)
        monkeypatch.setattr(owner.rules, "build_targets", forbidden)
    for name in ("build_targets", "market_context", "winner_credit", "co_loss_weights"):
        monkeypatch.setattr(study.rules, name, forbidden)
    monkeypatch.setattr(study.execution, "run", forbidden)
    data, experts, policy, parent, refs, clocks, _ = study.source_inputs()
    assert len(experts) == len(clocks) == 6 and len(refs) >= 668
    assert parent["study_id"] == "ssd_strategy_expert_pool_six_20261030_v1"
    assert tuple(item.identity.identifier for item in experts) == study.rules.EXPERT_IDS
    assert len(data.close.loc["2016-10-06":]) == 2512
    assert policy["methodology"]["failed_OLMAR_PAMR_V1_repaired_retried_or_recertified"] is False


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
        "study_id": "test_context",
        "code": [{"path": path, "sha256": "b" * 64} for path in (explicit, original)],
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
        pytest.fail("Main capital started without its actual preregistration publication fence.")

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
    data, _, targets, _, _ = synthetic
    close = data.close
    opening = close.shift(1)
    opening.iloc[0] = close.iloc[0]
    market = MarketData(
        opening,
        close,
        close.copy(),
        close * 0 + 1e6,
        pd.Series(0.00005, index=close.index),
    )
    known = targets[study.IDS[0]].copy()
    known.loc[~known.index.map(is_month_end)] = np.nan
    policy = read_json(monthly.POLICY)
    args = (market, known, monthly.CANDIDATES[0], policy, "2017-03-01", "2017-12-29", cost, delay)
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
        "actual_journal_event_count": 584,
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
    frozen = {"study_id": "synthetic_context", "protocol": {"path": "protocol.json"}}
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
    assert len(body["reviews"]) == 2 and sum(len(row["paths"]) for row in body["reviews"]) == 8
    assert all("artifacts" not in path for row in body["reviews"] for path in row["paths"])
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["original_records_reclassified"] is False
    assert body["investment_objective_verified"] is False
