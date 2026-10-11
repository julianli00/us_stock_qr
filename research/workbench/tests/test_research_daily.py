from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from us_quant.config import QuantError
from us_quant.research_daily import STEP_NAMES, run
from us_quant.storage import read_json


def clock():
    return datetime(2026, 10, 11, 1, 10, tzinfo=timezone.utc)


def state(root):
    return {"state_sha256": "unchanged", "complete_strategy_configurations": 112}


def success(name, calls):
    def action():
        calls.append(name)
        return {
            "action": "no_new_observed_session" if name == "research_model" else "already_collected",
            "investment_objective_verified": False, "order_authority": False,
        }
    return action


def test_four_real_step_results_are_receipted_but_declared_native_origin_is_not_proof(tmp_path):
    calls = []
    result = run(
        tmp_path / "reports/operations", "session_automation", root=tmp_path,
        clock=clock, state_reader=state,
        step_factory=lambda root: [(name, success(name, calls)) for name in STEP_NAMES],
    )
    assert tuple(calls) == STEP_NAMES
    assert result["status"] == "completed"
    assert result["pipeline_execution_verified_by_actual_stage_returns"]
    assert not result["native_schedule_delivery_independently_verified"]
    assert result["historical_research_state_unchanged"]
    assert not result["investment_objective_verified"]
    assert read_json(tmp_path / result["run_receipt_path"]) == result


def test_failure_preserves_completed_stage_receipt_and_stops_later_stages(tmp_path):
    calls = []

    def failure():
        calls.append("expanded_data")
        raise QuantError("synthetic input failure")

    actions = [(name, failure if name == "expanded_data" else success(name, calls)) for name in STEP_NAMES]
    with pytest.raises(QuantError, match="synthetic input"):
        run(
            tmp_path / "reports/operations", "agent_continuation", root=tmp_path,
            clock=clock, state_reader=state, step_factory=lambda root: actions,
        )
    assert calls == ["parent_data", "expanded_data"]
    record = read_json(next((tmp_path / "reports/operations/runs").glob("*/result.json")))
    assert record["status"] == "failed"
    assert record["steps"][0]["status"] == "completed"
    assert record["steps"][1]["status"] == "failed"
    assert record["previous_completed_stage_outputs_retained"]
    assert not record["pipeline_execution_verified_by_actual_stage_returns"]


def test_missed_deadline_is_not_counted_as_success_shaped_full_pipeline(tmp_path):
    def late():
        return {"action": "missed_preopen_deadline", "investment_objective_verified": False, "order_authority": False}
    with pytest.raises(QuantError, match="missed"):
        run(
            tmp_path / "reports/operations", "operator", root=tmp_path, clock=clock,
            state_reader=state,
            step_factory=lambda root: [(name, late) for name in STEP_NAMES],
        )
    record = read_json(next((tmp_path / "reports/operations/runs").glob("*/result.json")))
    assert record["status"] == "failed" and len(record["steps"]) == 1


def test_changed_research_state_is_detected_without_pretending_strategy_progress(tmp_path):
    states = iter((state(tmp_path), {"state_sha256": "unexpected-change"}))
    with pytest.raises(QuantError, match="alter research"):
        run(
            tmp_path / "reports/operations", "operator", root=tmp_path, clock=clock,
            state_reader=lambda root: next(states),
            step_factory=lambda root: [(name, success(name, [])) for name in STEP_NAMES],
        )
    record = read_json(next((tmp_path / "reports/operations/runs").glob("*/result.json")))
    assert record["status"] == "failed"
    assert record["new_historical_strategy_evaluations"] == 0


def test_backwards_stage_time_and_naive_clock_are_rejected(tmp_path):
    now = clock()
    earlier = datetime(2026, 10, 11, 1, 9, tzinfo=timezone.utc)
    times = iter((now, earlier))
    with pytest.raises(QuantError, match="backwards"):
        run(
            tmp_path / "reports/operations", "operator", root=tmp_path, clock=lambda: next(times),
            state_reader=state,
            step_factory=lambda root: [(name, success(name, [])) for name in STEP_NAMES],
        )
    with pytest.raises(QuantError, match="timezone-aware"):
        run(tmp_path / "reports/other", "operator", root=tmp_path, clock=lambda: datetime(2026, 10, 11))


def test_concurrent_calls_use_distinct_complete_receipts_without_resetting_archives(tmp_path):
    def invoke():
        return run(
            tmp_path / "reports/operations", "operator", root=tmp_path, clock=clock,
            state_reader=state,
            step_factory=lambda root: [(name, success(name, [])) for name in STEP_NAMES],
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: invoke(), range(2)))
    assert all(result["status"] == "completed" for result in results)
    assert results[0]["run_receipt_path"] != results[1]["run_receipt_path"]
    assert len(list((tmp_path / "reports/operations/runs").glob("*/result.json"))) == 2


def test_scope_and_step_order_cannot_be_falsified(tmp_path):
    with pytest.raises(QuantError, match="origin"):
        run(tmp_path / "reports/operations", "verified_native_run", root=tmp_path)
    with pytest.raises(QuantError, match="specific"):
        run(tmp_path, "operator", root=tmp_path)
    with pytest.raises(QuantError, match="specific"):
        run(tmp_path / "data/prospective-market-v1", "operator", root=tmp_path)
    with pytest.raises(QuantError, match="order"):
        run(
            tmp_path / "reports/operations", "operator", root=tmp_path, clock=clock,
            state_reader=state, step_factory=lambda root: [],
        )


def test_scope_violation_and_clock_error_retain_actual_returned_stage_result(tmp_path):
    returned = {
        "action": "already_collected", "investment_objective_verified": False,
        "order_authority": False,
    }
    times = iter((clock(), clock(), datetime(2026, 10, 11, 1, 12)))
    with pytest.raises(QuantError, match="timezone-aware"):
        run(
            tmp_path / "reports/clock-failure", "operator", root=tmp_path,
            clock=lambda: next(times), state_reader=state,
            step_factory=lambda root: [(name, lambda: returned) for name in STEP_NAMES],
        )
    record = read_json(next((tmp_path / "reports/clock-failure/runs").glob("*/result.json")))
    assert record["status"] == "failed"
    assert record["steps"][0]["result"] == returned
    with pytest.raises(QuantError, match="unsupported scope"):
        run(
            tmp_path / "reports/scope-failure", "operator", root=tmp_path,
            clock=clock, state_reader=state,
            step_factory=lambda root: [
                (name, lambda: {**returned, "investment_objective_verified": True})
                for name in STEP_NAMES
            ],
        )
