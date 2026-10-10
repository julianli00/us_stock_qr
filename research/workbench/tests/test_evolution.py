from __future__ import annotations

import fcntl
import sqlite3
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import us_quant.evolution as evolution
from us_quant.calendar import market_calendar
from us_quant.config import QuantError
from us_quant.evolution import EvolutionEngine, candidate_config, seed_genome, validate_policy
from us_quant.storage import read_json

REGISTERED = pd.Timestamp("2022-04-01 14:00Z")
SOURCES = {"history": "synthetic-test-fixture", "prior_trials": "14"}


def through(data, end):
    return replace(
        data,
        open=data.open.loc[:end],
        close=data.close.loc[:end],
        raw_close=data.raw_close.loc[:end],
        volume=data.volume.loc[:end],
        risk_free=data.risk_free.loc[:end],
    )


def after_close(day):
    return market_calendar().session_close(pd.Timestamp(day)) + pd.Timedelta(minutes=45)


@pytest.fixture
def base(config):
    return replace(
        config,
        data_start="2018-01-02",
        simulation_start="2019-01-02",
        development_end="2021-12-31",
        holdout_start="2022-01-03",
        as_of="2022-03-31",
        selection=replace(
            config.selection,
            first_test_year=2020,
            training_years=1,
            final_training_start="2021-01-01",
        ),
    )


@pytest.fixture
def policy():
    result = read_json(Path(__file__).parents[1] / "config/evolution.json")
    result["max_candidates"] = 1
    return result


@pytest.fixture
def market(base, market_factory):
    return market_factory("2018-01-02", "2022-08-31", base.symbols)


@pytest.fixture
def engine(tmp_path, base, policy):
    value = EvolutionEngine(tmp_path / "evolution.sqlite3", tmp_path / "reports", create=True)
    value.initialize(base, policy, SOURCES, REGISTERED)
    yield value
    value.close()


def first_cycle(engine, base, policy, market):
    return engine.cycle(
        base,
        policy,
        SOURCES,
        through(market, base.as_of),
        through(market, base.as_of),
        "first-fixture-snapshot",
        REGISTERED,
    )


def step(engine, base, policy, market, day):
    return engine.cycle(
        base,
        policy,
        SOURCES,
        through(market, base.as_of),
        through(market, day),
        f"fixture-{pd.Timestamp(day).date()}",
        after_close(day),
    )


def synthetic_evaluation(score=2.0, passed=False):
    def evaluate(config, data, identifier):
        return {
            "history_role": "synthetic_test_diagnostics",
            "objective_shortfall_score": score,
            "all_historical_numeric_gates_passed": passed,
            "paper_order_authority": False,
        }, {"test": pd.DataFrame({"equity": [10000.0]}, index=[pd.Timestamp("2022-03-31")])}

    return evaluate


def test_first_real_diagnostic_cycle_is_registered_and_repeat_safe(engine, base, policy, market):
    result = first_cycle(engine, base, policy, market)
    assert result["new_trials"] == 1 and result["cumulative_trials"] == 15
    candidate = result["candidates"][0]
    assert candidate["state"] == "awaiting_baseline"
    assert candidate["baseline_session"] == "2022-04-01"
    assert candidate["forward"] is None
    assert not candidate["historical_all_gates_passed"]
    assert not result["order_authority"] and not result["investment_objective_verified"]
    before = engine.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    again = first_cycle(engine, base, policy, market)
    assert again["repeated_cycle_no_changes"] is True
    assert engine.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == before
    saved = read_json(engine.reports / "candidates" / candidate["id"] / "history.json")
    assert saved["history_role"] == "previously_exposed_history_diagnostics_not_new_holdout"
    assert len(saved["periods"]) == 4


def test_registration_is_durable_before_evaluation(engine, base, policy, market, monkeypatch):
    def checked(config, data, identifier):
        with sqlite3.connect(engine.path) as other:
            row = other.execute(
                "SELECT status FROM candidates WHERE id=?", (identifier,)
            ).fetchone()
            assert row[0] == "registered"
            assert (
                other.execute(
                    "SELECT COUNT(*) FROM events "
                    "WHERE kind='candidate_registered_before_evaluation'"
                ).fetchone()[0]
                == 1
            )
        return synthetic_evaluation()(config, data, identifier)

    monkeypatch.setattr(evolution, "historical_diagnostics", checked)
    first_cycle(engine, base, policy, market)


def test_clock_rollback_does_not_register_a_retrospective_trial(engine, base, policy, market):
    with pytest.raises(QuantError, match="clock moved backwards"):
        engine.cycle(
            base,
            policy,
            SOURCES,
            through(market, base.as_of),
            through(market, base.as_of),
            "first-fixture-snapshot",
            REGISTERED - pd.Timedelta(minutes=1),
        )
    assert engine.status()["new_trials"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"capital_usd": 20000},
        {"auto_order_submission": True},
        {"max_candidates": 100},
        {"max_active_candidates": 4},
        {"proposal_interval_sessions": 1},
        {"minimum_forward_sessions": 5},
        {"max_cycle_sessions": 1000},
        {"prior_trials": 0},
        {"minimum_historical_score_improvement": float("nan")},
        {"stop_after": "NaT"},
        {"mutations": None},
    ],
)
def test_self_evolution_cannot_expand_authority_or_erase_trials(base, policy, change):
    with pytest.raises(QuantError):
        validate_policy({**policy, **change}, base)


def test_mutations_cannot_modify_capital_or_the_universe(base, policy):
    changed = deepcopy(policy)
    changed["mutations"][0]["field"] = "capital_usd"
    with pytest.raises(QuantError):
        validate_policy(changed, base)
    genome = {**seed_genome(base, policy), "target_volatility": 0.30}
    with pytest.raises(QuantError):
        candidate_config(base, policy, "invalid", genome)


def test_baseline_is_not_counted_as_forward_performance(engine, base, policy, market, monkeypatch):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    first_cycle(engine, base, policy, market)
    baseline = step(engine, base, policy, market, "2022-04-01")
    assert baseline["candidates"][0]["forward"]["forward_sessions"] == 0
    later = step(engine, base, policy, market, "2022-04-04")
    summary = later["candidates"][0]["forward"]
    assert summary["forward_sessions"] == 1 and summary["annualized_metrics_withheld"]
    assert "metrics" not in summary
    assert summary["total_return"] == 0


def test_missing_a_forward_close_stops_instead_of_backfilling(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    first_cycle(engine, base, policy, market)
    with pytest.raises(QuantError, match="missed forward"):
        step(engine, base, policy, market, "2022-04-04")
    assert engine.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    with pytest.raises(QuantError, match="failed cycle"):
        step(engine, base, policy, market, "2022-04-04")


def test_past_source_changes_are_not_silently_overwritten(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    first_cycle(engine, base, policy, market)
    step(engine, base, policy, market, "2022-04-01")
    raw = market.raw_close.copy()
    raw.loc["2022-04-01", "SPY"] *= 1.01
    with pytest.raises(QuantError, match="source close was revised"):
        step(engine, base, policy, replace(market, raw_close=raw), "2022-04-04")
    assert engine.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


def test_mutation_cadence_lineage_and_failed_candidates_remain_visible(
    tmp_path, base, policy, market, monkeypatch
):
    policy = {**policy, "max_candidates": 3}
    engine = EvolutionEngine(tmp_path / "mutations.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    scores = iter([2.0, 3.0, 1.5])
    monkeypatch.setattr(
        evolution,
        "historical_diagnostics",
        lambda *args: synthetic_evaluation(next(scores))(*args),
    )
    try:
        first = first_cycle(engine, base, policy, market)
        parent = first["candidates"][0]["id"]
        for day in market.close.loc["2022-04-01":"2022-04-18"].index:
            result = step(engine, base, policy, market, day)
        assert result["new_trials"] == 3 and result["cumulative_trials"] == 17
        second, third = result["candidates"][1:]
        assert second["state"] == "rejected_historical" and second["parent_id"] == parent
        assert third["parent_id"] == parent and third["mutation"] == "faster_momentum"
        assert third["state"] in {"awaiting_baseline", "observing"}
        assert second["genome"]["trend_lookback"] == 150
        assert (engine.reports / "candidates" / second["id"] / "history.json").exists()
    finally:
        engine.close()


def test_decisions_are_recorded_before_execution_and_cannot_be_postdated(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    first_cycle(engine, base, policy, market)
    for day in market.close.loc["2022-04-01":"2022-04-29"].index:
        step(engine, base, policy, market, day)
    decision = engine.connection.execute("SELECT * FROM decisions").fetchone()
    assert decision["signal_date"] == "2022-04-29"
    assert decision["execution_date"] == "2022-05-02"
    assert pd.Timestamp(decision["created_at"]) < market_calendar().session_open("2022-05-02")
    with engine.connection:
        engine.connection.execute(
            "UPDATE decisions SET created_at=?",
            ((market_calendar().session_open("2022-05-02") + pd.Timedelta(seconds=1)).isoformat(),),
        )
    with pytest.raises(QuantError, match="timing or lineage"):
        step(engine, base, policy, market, "2022-05-02")


def test_forward_promotion_is_only_a_research_stage_never_order_authority(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation(0.0, True))
    monkeypatch.setattr(evolution, "acceptance", lambda *args: {"synthetic_test_gate": True})
    monkeypatch.setattr(
        evolution,
        "block_bootstrap",
        lambda *args, **kwargs: {
            "sharpe_95pct_interval": [0.1, 1.8],
            "excess_cagr_vs_spy_95pct_interval": [0.01, 0.2],
        },
    )
    first_cycle(engine, base, policy, market)
    for day in market.close.loc["2022-04-01":].index[:64]:
        result = step(engine, base, policy, market, day)
    candidate = result["candidates"][0]
    assert candidate["state"] == "paper_review_ready"
    assert candidate["forward"]["forward_sessions"] == 63
    assert result["paper_review_candidates"] == [candidate["id"]]
    assert not result["investment_objective_verified"] and not result["order_authority"]
    assert result["broker_orders_sent"] == 0


def test_parent_selection_learns_from_mature_forward_failure(base):
    metrics = {"cagr": -0.10, "sharpe": -0.5, "max_drawdown": 0.1}
    observed = {
        "forward_sessions": 63,
        "metrics": metrics,
        "benchmark": {"cagr": 0.1},
    }
    assert evolution.parent_priority(0.1, None, base, 63) == 0.1
    assert evolution.parent_priority(0.1, {**observed, "forward_sessions": 62}, base, 63) == 0.1
    assert evolution.parent_priority(0.1, observed, base, 63) > 2.0


def test_positive_point_metrics_without_positive_uncertainty_bounds_do_not_promote(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation(0.0, True))
    monkeypatch.setattr(evolution, "acceptance", lambda *args: {"synthetic_test_gate": True})
    monkeypatch.setattr(
        evolution,
        "block_bootstrap",
        lambda *args, **kwargs: {
            "sharpe_95pct_interval": [-0.3, 1.8],
            "excess_cagr_vs_spy_95pct_interval": [-0.1, 0.2],
        },
    )
    first_cycle(engine, base, policy, market)
    for day in market.close.loc["2022-04-01":].index[:64]:
        result = step(engine, base, policy, market, day)
    assert result["candidates"][0]["state"] == "retired_forward"
    assert not result["candidates"][0]["forward"]["positive_edge_interval"]
    assert not result["paper_review_candidates"]


def test_forward_drawdown_breach_retires_instead_of_changing_risk_limits(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())

    def fixed_weights(history, candidate, config):
        weights = pd.Series(0.0, index=history.columns)
        weights.loc[["SPY", "QQQ", "IWM"]] = [0.4, 0.4, 0.18]
        return weights

    monkeypatch.setattr(evolution, "target_weights", fixed_weights)
    first_cycle(engine, base, policy, market)
    for day in market.close.loc["2022-04-01":"2022-04-29"].index:
        step(engine, base, policy, market, day)
    close, raw = market.close.copy(), market.raw_close.copy()
    close.loc["2022-05-02"] *= 0.6
    raw.loc["2022-05-02"] *= 0.6
    result = step(engine, base, policy, replace(market, close=close, raw_close=raw), "2022-05-02")
    candidate = result["candidates"][0]
    assert candidate["state"] == "retired_forward_risk"
    assert candidate["forward"]["max_drawdown"] > 0.15
    assert candidate["forward"]["observation_ended_without_assumed_liquidation"]
    assert result["broker_orders_sent"] == 0


def test_policy_or_candidate_tampering_is_rejected(engine, base, policy, market, monkeypatch):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    first_cycle(engine, base, policy, market)
    with pytest.raises(QuantError, match="risk policy"):
        engine.verify(base, {**policy, "max_candidates": 2}, SOURCES)
    with engine.connection:
        engine.connection.execute("UPDATE candidates SET genome_sha='changed'")
    with pytest.raises(QuantError, match="parameters changed"):
        engine.verify(base, policy, SOURCES)


def test_evidence_files_and_trial_count_cannot_be_silently_rewritten(
    engine, base, policy, market, monkeypatch
):
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    result = first_cycle(engine, base, policy, market)
    path = engine.reports / "candidates" / result["candidates"][0]["id"] / "test.csv"
    path.write_text("changed\n")
    with pytest.raises(QuantError, match="equity record"):
        engine.verify(base, policy, SOURCES)
    with engine.connection:
        engine.connection.execute("DELETE FROM candidates")
    with pytest.raises(QuantError, match="history was removed"):
        engine.verify(base, policy, SOURCES)


def test_cycle_budget_pauses_without_creating_more_trials(
    tmp_path, base, policy, market, monkeypatch
):
    policy = {**policy, "max_cycle_sessions": 1}
    engine = EvolutionEngine(tmp_path / "budget.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    try:
        first_cycle(engine, base, policy, market)
        result = step(engine, base, policy, market, "2022-04-01")
        assert result["cycle_action"] == "paused_cycle_budget" and result["new_trials"] == 1
        assert engine.connection.execute("SELECT COUNT(*) FROM cycles").fetchone()[0] == 1
    finally:
        engine.close()


def test_deadline_and_concurrent_cycle_fail_closed(tmp_path, base, policy, market, monkeypatch):
    policy = {**policy, "stop_after": "2022-04-01"}
    engine = EvolutionEngine(tmp_path / "deadline.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation())
    try:
        with engine.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(QuantError, match="concurrent mutations"):
                first_cycle(engine, base, policy, market)
        first_cycle(engine, base, policy, market)
        result = step(engine, base, policy, market, "2022-04-04")
        assert result["cycle_action"] == "paused_deadline"
        assert result["cumulative_trials"] == 15
    finally:
        engine.close()


def test_failed_evaluation_still_consumes_and_preserves_the_registered_trial(
    engine, base, policy, market, monkeypatch
):
    def fail(*args):
        raise QuantError("Synthetic data-quality failure")

    monkeypatch.setattr(evolution, "historical_diagnostics", fail)
    with pytest.raises(QuantError, match="data-quality"):
        first_cycle(engine, base, policy, market)
    assert engine.status()["new_trials"] == 1
    assert engine.status()["candidates"][0]["state"] == "registered"
    assert engine.connection.execute("SELECT status FROM cycles").fetchone()[0] == "failed"


def batch_policy(policy):
    return {**policy, "max_candidates": 8}


def test_batch_screen_registers_every_declared_trial_before_any_evaluation(
    tmp_path, base, policy, market, monkeypatch
):
    policy = batch_policy(policy)
    engine = EvolutionEngine(tmp_path / "batch.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation(1.0))
    first_cycle(engine, base, policy, market)
    scores = iter([0.8, 1.3, 0.7, 1.4, 0.9, 1.5, 0.6])
    checks = []

    def evaluate(config, data, identifier):
        with sqlite3.connect(engine.path) as other:
            checks.append(
                (
                    other.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                    other.execute(
                        "SELECT COUNT(*) FROM candidates WHERE status='registered'"
                    ).fetchone()[0],
                    other.execute(
                        "SELECT COUNT(*) FROM events "
                        "WHERE kind='candidate_registered_before_evaluation'"
                    ).fetchone()[0],
                )
            )
        return synthetic_evaluation(next(scores))(config, data, identifier)

    monkeypatch.setattr(evolution, "historical_diagnostics", evaluate)
    monkeypatch.setattr(evolution, "fingerprint", lambda: "reviewed-batch-engine")
    try:
        with pytest.raises(QuantError, match="accept-code-update"):
            engine.batch_screen(base, policy, SOURCES, through(market, base.as_of), REGISTERED)
        result = engine.batch_screen(
            base,
            policy,
            SOURCES,
            through(market, base.as_of),
            REGISTERED,
            accept_code_update=True,
        )
        assert checks == [(8, 7, 8)] * 7
        assert result["new_trials"] == 8 and result["cumulative_trials"] == 22
        assert result["batch_registered_trials"] == 7
        assert len(result["batch_admitted_candidates"]) == 2
        assert len(result["batch_rejected_candidates"]) == 5
        assert (
            sum(
                item["state"] in {"awaiting_baseline", "observing"} for item in result["candidates"]
            )
            == 3
        )
        assert not result["order_authority"] and result["broker_orders_sent"] == 0
        migrations = engine.connection.execute(
            "SELECT previous_engine,current_engine,risk_limits_changed FROM migrations"
        ).fetchall()
        assert len(migrations) == 1
        assert migrations[0][1] == "reviewed-batch-engine" and migrations[0][2] == 0
        registration = read_json(engine.reports / "batch-screens" / "2022-03-31-registration.json")
        assert len(registration["candidates"]) == 7
        assert not registration["risk_limits_changed"] and not registration["order_authority"]
        assert all(
            (engine.reports / "candidates" / item["id"] / "history.json").exists()
            for item in registration["candidates"]
        )
        engine.verify(base, policy, SOURCES)
    finally:
        engine.close()


def test_batch_screen_is_repeat_safe_and_preserves_control_observations(
    tmp_path, base, policy, market, monkeypatch
):
    policy = batch_policy(policy)
    engine = EvolutionEngine(tmp_path / "batch.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation(1.0))
    first_cycle(engine, base, policy, market)
    step(engine, base, policy, market, "2022-04-01")
    before_observations = engine.connection.execute("SELECT * FROM observations").fetchall()
    monkeypatch.setattr(evolution, "fingerprint", lambda: "batch-engine")
    first = engine.batch_screen(
        base,
        policy,
        SOURCES,
        through(market, base.as_of),
        REGISTERED,
        accept_code_update=True,
    )
    before = {
        table: engine.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("candidates", "events", "observations", "decisions", "migrations")
    }
    second = engine.batch_screen(base, policy, SOURCES, through(market, base.as_of), REGISTERED)
    after = {
        table: engine.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in before
    }
    assert first["cumulative_trials"] == second["cumulative_trials"] == 22
    assert second["repeated_batch_no_changes"]
    assert before == after
    assert engine.connection.execute("SELECT * FROM observations").fetchall() == before_observations
    engine.close()


def test_batch_failure_retains_all_registered_trials_and_blocks_daily_bypass(
    tmp_path, base, policy, market, monkeypatch
):
    policy = batch_policy(policy)
    engine = EvolutionEngine(tmp_path / "batch.sqlite3", tmp_path / "reports", create=True)
    engine.initialize(base, policy, SOURCES, REGISTERED)
    monkeypatch.setattr(evolution, "historical_diagnostics", synthetic_evaluation(1.0))
    first_cycle(engine, base, policy, market)
    monkeypatch.setattr(evolution, "fingerprint", lambda: "batch-engine")

    def fail(*args):
        raise QuantError("Synthetic batch failure")

    monkeypatch.setattr(evolution, "historical_diagnostics", fail)
    with pytest.raises(QuantError, match="batch failure"):
        engine.batch_screen(
            base,
            policy,
            SOURCES,
            through(market, base.as_of),
            REGISTERED,
            accept_code_update=True,
        )
    assert engine.status()["new_trials"] == 8
    assert (
        engine.connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind='batch_screen_failed_requires_review'"
        ).fetchone()[0]
        == 1
    )
    with pytest.raises(QuantError, match="operator review"):
        engine.batch_screen(base, policy, SOURCES, through(market, base.as_of), REGISTERED)
    with pytest.raises(QuantError, match="no daily-cycle bypass"):
        step(engine, base, policy, market, "2022-04-01")
    engine.close()
