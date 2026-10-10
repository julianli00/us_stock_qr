from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import simulate
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import (
    ResearchProgram,
    readiness_blockers,
    timestamp,
    validate_factor,
    validate_policy,
)
from us_quant.storage import digest_json, file_digest, read_json, write_json
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/research-program.json")


@pytest.fixture
def proposals():
    return read_json(ROOT / "config/factor-discovery-20261010.json")


@pytest.fixture
def program(tmp_path, policy):
    instance = ResearchProgram(tmp_path / "program.sqlite3", policy, root=tmp_path, create=True)
    yield instance
    instance.close()


@pytest.fixture
def registered_generator(monkeypatch):
    def generate(spec, data, start, end, cost, delay, root, cache):
        return buy_and_hold_signals(data.close, "TEST", start) * 0.98

    monkeypatch.setattr("us_quant.research_program.registered_targets", generate, raising=False)


def readiness(policy, root, now=None, *, ready=False):
    now = timestamp(now)
    proof = root / "proof.json"
    if not proof.exists():
        write_json(proof, {"synthetic_test_fixture": True})
    return {
        "schema_version": 1,
        "checked_at": now.isoformat(),
        "capabilities": {
            key: {
                "verified": ready,
                "reason": "synthetic missing-data case",
                "evidence_path": "proof.json",
                "evidence_sha256": file_digest(proof),
            }
            for key in policy["required_stock_data"]
        },
    }


def synthetic_market(tmp_path, *, winning=True):
    rng = np.random.default_rng(72)
    index = sessions("2015-01-02", "2026-10-05")
    close = pd.DataFrame(
        {
            "TEST": 100
            * np.cumprod(1 + rng.normal(0.0008 if winning else -0.0001, 0.003, len(index))),
            "SPY": 100 * np.cumprod(1 + rng.normal(0.00025, 0.006, len(index))),
        },
        index=index,
    )
    data = MarketData(
        close.shift(1).fillna(100),
        close,
        close.copy(),
        pd.DataFrame(1000000.0, index=index, columns=close.columns),
        pd.Series(0.0001, index=index, name="risk_free"),
    )
    market = {}
    for name, frame in (
        ("open", data.open),
        ("close", data.close),
        ("raw_close", data.raw_close),
        ("volume", data.volume),
        ("risk_free", data.risk_free.to_frame()),
    ):
        path = tmp_path / f"market-{name}.csv"
        frame.to_csv(path)
        market[name] = {"path": path.name, "sha256": file_digest(path)}
    return data, market


def candidate_spec(tmp_path, identifier="test_multifactor", *, winning=True):
    code = tmp_path / "frozen-rule.json"
    if not code.exists():
        write_json(code, {"rule": "synthetic test only"})
    _, market = synthetic_market(tmp_path, winning=winning)
    return {
        "id": identifier,
        "factor_ids": ["price_momentum", "quality_exposure", "value_exposure"],
        "frozen_files": {"frozen-rule.json": file_digest(code)},
        "asset_leverage": {"TEST": 1.0, "SPY": 1.0},
        "evaluation_as_of": "2026-10-05",
        "market": market,
        "history_status": "exposed_history_not_independent_holdout",
        "leveraged_products_allowed": False,
        "order_authority": False,
    }


def test_init_preserves_four_prior_families_without_claiming_a_new_strategy(program):
    status = program.status()
    assert status["known_factor_definition_count"] == 4
    assert status["new_factor_proposal_count"] == 0
    assert status["total_evaluated_configurations"] == 84
    assert status["research_champion"] is None
    assert not status["investment_objective_verified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_new_factors_per_cycle", 20),
        ("prior_evaluated_configurations", 0),
        ("minimum_economic_factor_families", 1),
        ("timezone", "UTC"),
    ],
)
def test_program_bounds_cannot_be_silently_relaxed(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_live_deployment_and_goal_relaxation_are_not_allowed(policy):
    altered = deepcopy(policy)
    altered["promotion"]["automatic_live_deployment"] = True
    with pytest.raises(QuantError):
        validate_policy(altered)
    policy["goals"]["net_excess_sharpe_strictly_above"] = 0.9
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_weekly_cycle_logs_data_blockers_and_deduplicates(program, proposals, policy, tmp_path):
    now = timestamp()
    inputs = readiness(policy, tmp_path, now)
    first = program.cycle(proposals, inputs, now=now)
    assert first["state"] == "blocked_data" and len(first["data_blockers"]) == 5
    assert len(first["new_factors"]) == 2
    before = program.status()
    second = program.cycle(proposals, inputs, now=now + timedelta(minutes=1))
    assert second["cycle_action"] == "already_recorded"
    assert program.status()["event_chain_sha256"] == before["event_chain_sha256"]
    assert program.status()["new_factor_proposal_count"] == 2
    assert program.status()["economic_factor_family_count"] == 5
    assert program.status()["total_evaluated_configurations"] == 84
    later = now + timedelta(days=7)
    third = program.cycle(proposals, readiness(policy, tmp_path, later), now=later)
    assert third["new_factors"] == [] and len(third["duplicate_definitions"]) == 2


def test_concurrent_cycle_runs_do_not_duplicate_discoveries(program, proposals, policy, tmp_path):
    inputs = readiness(policy, tmp_path)

    def run():
        other = ResearchProgram(tmp_path / "program.sqlite3", policy, root=tmp_path)
        try:
            return other.cycle(proposals, inputs)
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sum(result.get("cycle_action") == "already_recorded" for result in results) == 1
    assert program.status()["new_factor_proposal_count"] == 2


def test_renamed_factor_does_not_count_as_new_discovery(program, proposals, policy, tmp_path):
    now = timestamp()
    program.cycle(proposals, readiness(policy, tmp_path, now), now=now)
    later = now + timedelta(days=7)
    renamed = deepcopy(proposals)
    renamed["factors"][0]["id"] = "cosmetic_rename"
    result = program.cycle(renamed, readiness(policy, tmp_path, later), now=later)
    assert result["new_factors"] == []
    assert validate_factor(renamed["factors"][0]) == validate_factor(proposals["factors"][0])


def test_modified_factor_id_rolls_back_whole_cycle(program, proposals, policy, tmp_path):
    now = timestamp()
    program.cycle(proposals, readiness(policy, tmp_path, now), now=now)
    before = program.status()["event_chain_sha256"]
    proposals["factors"][0]["definition"] = "A different economic construction"
    later = now + timedelta(days=7)
    with pytest.raises(QuantError, match="reused"):
        program.cycle(proposals, readiness(policy, tmp_path, later), now=later)
    assert program.status()["event_chain_sha256"] == before


def test_factor_removal_and_event_tampering_are_detected(program, proposals, policy, tmp_path):
    program.cycle(proposals, readiness(policy, tmp_path))
    with program.db:
        program.db.execute("DELETE FROM factors WHERE id='conservative_asset_growth_v1'")
    with pytest.raises(QuantError, match="removed"):
        program.status()


def test_event_chain_cannot_hide_rewritten_failures(program):
    with program.db:
        program.db.execute("UPDATE events SET body='{}' WHERE seq=1")
    with pytest.raises(QuantError, match="chain"):
        program.status()


def test_init_cannot_reset_an_existing_ledger(program, policy, tmp_path):
    with pytest.raises(QuantError, match="reset"):
        ResearchProgram(tmp_path / "program.sqlite3", policy, root=tmp_path, create=True)


def test_stale_or_future_readiness_is_not_reused(policy, tmp_path):
    now = timestamp()
    for days in (-9, 1):
        with pytest.raises(QuantError, match="freshly"):
            readiness_blockers(
                readiness(policy, tmp_path, now + timedelta(days=days)),
                policy,
                tmp_path,
                now=now,
            )


def test_readiness_cannot_use_external_or_changed_evidence(policy, tmp_path):
    ready = readiness(policy, tmp_path, ready=True)
    ready["capabilities"]["filing_time_fundamentals"]["evidence_path"] = "../outside.json"
    with pytest.raises(QuantError, match="evidence"):
        readiness_blockers(ready, policy, tmp_path)


def test_no_candidate_backtest_with_missing_stock_history(program, policy, tmp_path):
    with pytest.raises(QuantError, match="missing point-in-time"):
        program.register_candidate(candidate_spec(tmp_path), readiness(policy, tmp_path))
    assert program.status()["registered_candidates"] == []


def test_candidate_requires_three_economic_families(program, policy, tmp_path):
    candidate = candidate_spec(tmp_path)
    candidate["factor_ids"] = ["price_momentum", "quality_exposure"]
    with pytest.raises(QuantError, match="families|diversified"):
        program.register_candidate(candidate, readiness(policy, tmp_path, ready=True))


def test_candidate_registration_is_frozen_and_duplicate_names_do_not_reset_trials(
    program, policy, tmp_path
):
    candidate = candidate_spec(tmp_path)
    registered = program.register_candidate(candidate, readiness(policy, tmp_path, ready=True))
    assert registered["spec_sha256"] == digest_json(candidate)
    candidate["id"] = "renamed_duplicate"
    with pytest.raises(QuantError, match="already registered"):
        program.register_candidate(candidate, readiness(policy, tmp_path, ready=True))
    assert len(program.status()["registered_candidates"]) == 1


def make_bundle(tmp_path, registration, *, winning=True):
    records = []
    data, market = synthetic_market(tmp_path, winning=winning)
    close = data.close
    for years in (10, 5):
        window = sessions(
            pd.Timestamp("2026-10-05") - pd.DateOffset(years=years) + pd.Timedelta(days=1),
            "2026-10-05",
        )
        start, end = str(window[0].date()), str(window[-1].date())
        targets = buy_and_hold_signals(close, "TEST", start) * 0.98
        for scenario in ("base", "stress"):
            rec = {
                "years": years,
                "scenario": scenario,
                "capital_usd": 10000.0,
                "cost_bps": 5 if scenario == "base" else 20,
                "delay_sessions": 1 if scenario == "base" else 2,
                "commission_per_order": 1,
            }
            for name, signal in (
                ("strategy", targets),
                ("spy", buy_and_hold_signals(close, "SPY", start)),
            ):
                actual = simulate(
                    data,
                    signal,
                    start,
                    end,
                    initial_capital=10000,
                    cost_bps=rec["cost_bps"],
                    commission=1,
                    delay=rec["delay_sessions"],
                )
                for suffix in ("", "_bt"):
                    path = tmp_path / f"{years}-{scenario}-{name}{suffix}.csv"
                    subset = actual.frame[["equity", "return"]] if suffix else actual.frame
                    subset.to_csv(path)
                    rec[name + suffix] = {"path": path.name, "sha256": file_digest(path)}
                if name == "strategy":
                    path = tmp_path / f"{years}-{scenario}-weights.csv"
                    actual.weights.to_csv(path)
                    rec["weights"] = {"path": path.name, "sha256": file_digest(path)}
            path = tmp_path / f"{years}-{scenario}-targets.csv"
            targets.to_csv(path)
            rec["targets"] = {"path": path.name, "sha256": file_digest(path)}
            records.append(rec)
    return {
        "candidate_spec_sha256": registration["spec_sha256"],
        "completed_at": timestamp().isoformat(),
        "as_of": "2026-10-05",
        "leveraged_products_allowed": False,
        "order_authority": False,
        "history_status": "exposed_history_not_independent_holdout",
        "paths": records,
        "market": market,
    }


@pytest.mark.parametrize("winning", [True, False])
def test_independent_review_records_success_or_failure_without_live_promotion(
    program, policy, tmp_path, winning, registered_generator
):
    registered = program.register_candidate(
        candidate_spec(tmp_path, winning=winning), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, registered, winning=winning)
    result = program.review("test_multifactor", bundle)
    assert result["historical_gates_passed"] is winning
    assert result["status"] == (
        "historical_qualified_awaiting_forward" if winning else "rejected_historical"
    )
    assert not result["live_strategy_update"] and not result["independent_forward_validation"]
    assert program.status()["total_evaluated_configurations"] == 85
    assert (program.status()["research_champion"] is not None) is winning
    with pytest.raises(QuantError, match="earlier review"):
        program.review("test_multifactor", bundle)


@pytest.mark.parametrize(
    "problem", ["changed_rule", "changed_ledger", "missing_window", "zero_cost", "changed_cutoff"]
)
def test_invalid_evaluation_cannot_update_research_champion(program, policy, tmp_path, problem):
    registered = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, registered)
    if problem == "changed_rule":
        write_json(tmp_path / "frozen-rule.json", {"rule": "retuned"})
    elif problem == "changed_ledger":
        (tmp_path / bundle["paths"][0]["strategy"]["path"]).write_text("changed")
    elif problem == "missing_window":
        bundle["paths"].pop()
    elif problem == "zero_cost":
        bundle["paths"][0]["cost_bps"] = 0
    else:
        bundle["as_of"] = "2026-10-02"
    with pytest.raises(QuantError):
        program.review("test_multifactor", bundle)
    assert program.status()["research_champion"] is None


def test_self_consistent_curve_cannot_replace_real_market_and_cost_replay(
    program, policy, tmp_path, registered_generator
):
    registered = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, registered)
    for key in ("strategy", "strategy_bt"):
        path = tmp_path / bundle["paths"][0][key]["path"]
        frame = pd.read_csv(path, index_col=0)
        frame["equity"] *= 1.1
        frame.to_csv(path)
        bundle["paths"][0][key]["sha256"] = file_digest(path)
    with pytest.raises(QuantError, match="replay"):
        program.review("test_multifactor", bundle)
    assert program.status()["research_champion"] is None


def test_review_refuses_submitted_risk_free_different_from_frozen_market(
    program, policy, tmp_path, registered_generator
):
    registered = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, registered)
    for item in bundle["paths"]:
        for name in ("strategy", "spy"):
            path = tmp_path / item[name]["path"]
            frame = pd.read_csv(path, index_col=0)
            frame["risk_free"] = -0.01
            frame.to_csv(path)
            item[name]["sha256"] = file_digest(path)
    with pytest.raises(QuantError, match="inconsistent|risk.free"):
        program.review("test_multifactor", bundle)
    assert program.status()["research_champion"] is None


def test_consistent_ledgers_cannot_use_targets_unrelated_to_the_registered_rule(
    program, policy, tmp_path, registered_generator
):
    from us_quant.research_program import review_market

    registered = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, registered)
    market = review_market(bundle, tmp_path)
    for item in bundle["paths"]:
        path = tmp_path / item["targets"]["path"]
        targets = pd.read_csv(path, index_col=0, parse_dates=True)
        targets["TEST"] *= 0.5
        targets.to_csv(path)
        item["targets"]["sha256"] = file_digest(path)
        start = str(
            (
                pd.Timestamp(bundle["as_of"])
                - pd.DateOffset(years=item["years"])
                + pd.Timedelta(days=1)
            ).date()
        )
        altered = simulate(
            market,
            targets,
            start,
            bundle["as_of"],
            initial_capital=10000,
            cost_bps=item["cost_bps"],
            commission=1,
            delay=item["delay_sessions"],
        )
        for name, frame in (
            ("strategy", altered.frame),
            ("strategy_bt", altered.frame[["equity", "return"]]),
            ("weights", altered.weights),
        ):
            path = tmp_path / item[name]["path"]
            frame.to_csv(path)
            item[name]["sha256"] = file_digest(path)
    bundle["completed_at"] = timestamp().isoformat()
    with pytest.raises(QuantError, match="registered strategy"):
        program.review("test_multifactor", bundle)
    assert program.status()["research_champion"] is None


def test_unknown_strategy_code_cannot_qualify_from_a_submitted_equity_curve(
    program, policy, tmp_path
):
    registered = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    with pytest.raises(QuantError, match="registered strategy"):
        program.review("test_multifactor", make_bundle(tmp_path, registered))
    assert program.status()["research_champion"] is None


def test_read_only_reaudit_preserves_the_entire_program_ledger(
    program, policy, tmp_path, registered_generator
):
    record = program.register_candidate(
        candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True)
    )
    bundle = make_bundle(tmp_path, record)
    program.review("test_multifactor", bundle)
    before = program.status()
    result = program.review("test_multifactor", bundle, audit_existing=True)
    assert result["read_only_reaudit"] and result["original_record_unchanged"]
    assert all(row["registered_strategy_targets_regenerated"] for row in result["paths"])
    assert all(row["frozen_market_risk_free_used"] for row in result["paths"])
    assert program.status() == before
    changed = deepcopy(bundle)
    changed["completed_at"] = timestamp().isoformat()
    with pytest.raises(QuantError, match="unchanged original"):
        program.review("test_multifactor", changed, audit_existing=True)
    assert program.status() == before


def test_dependency_changes_invalidate_program_verification(program, monkeypatch):
    monkeypatch.setattr("us_quant.research_program.replay_dependencies_hash", lambda: "changed")
    with pytest.raises(QuantError, match="engine"):
        program.status()


def test_generic_evaluator_uses_registered_generator_and_reviews_without_duplicate_trials(
    program, policy, tmp_path, registered_generator
):
    program.register_candidate(candidate_spec(tmp_path), readiness(policy, tmp_path, ready=True))
    result = program.evaluate_registered("test_multifactor", Path("reports/generic-study"))
    assert result["historical_gates_passed"]
    assert all(row["registered_strategy_targets_regenerated"] for row in result["paths"])
    assert (tmp_path / "reports/generic-study/test_multifactor/bundle.json").is_file()
    before = program.status()
    again = program.evaluate_registered("test_multifactor", Path("reports/generic-study"))
    assert again["evaluation_action"] == "already_reviewed"
    assert program.status() == before
    audit = program.audit_reviews()
    assert audit["reviewed_candidates"] == 1 and audit["regenerated_strategy_paths"] == 4
    assert program.status() == before


def test_engine_migration_preserves_history_and_requires_exact_anchors(policy, tmp_path, proposals):
    path = tmp_path / "migrate.sqlite3"
    old = ResearchProgram(path, policy, root=tmp_path, create=True)
    old.cycle(proposals, readiness(policy, tmp_path))
    before = old.status()
    with old.db:
        old.db.execute("UPDATE metadata SET engine_sha=?", ("0" * 64,))
    old.close()
    with pytest.raises(QuantError, match="migration"):
        ResearchProgram(path, policy, root=tmp_path)
    with pytest.raises(QuantError, match="preconditions"):
        ResearchProgram(
            path,
            policy,
            root=tmp_path,
            expected_previous_engine="0" * 64,
            expected_previous_event="1" * 64,
        )
    new = ResearchProgram(
        path,
        policy,
        root=tmp_path,
        expected_previous_engine="0" * 64,
        expected_previous_event=before["event_chain_sha256"],
    )
    try:
        after = new.status()
        assert after["factor_proposals"] == before["factor_proposals"]
        assert after["latest_cycle"] == before["latest_cycle"]
        assert after["event_count"] == before["event_count"] + 1
        assert after["total_evaluated_configurations"] == 84
        assert after["research_champion"] is None
    finally:
        new.close()


def test_corrupt_missing_engine_hash_never_bypasses_verification(program):
    with program.db:
        program.db.execute("UPDATE metadata SET engine_sha=NULL")
    with pytest.raises(QuantError, match="engine"):
        program.status()


def test_etf_scope_does_not_mark_stock_history_ready(policy, tmp_path, monkeypatch):
    raw = readiness(policy, tmp_path)
    assert len(readiness_blockers(raw, policy, tmp_path)) == 5
    etf = {
        "schema_version": 1,
        "checked_at": timestamp().isoformat(),
        "data_scope": "factor_etf_portfolio",
        "capabilities": {
            key: {
                "verified": True,
                "evidence_path": "proof.json",
                "evidence_sha256": file_digest(tmp_path / "proof.json"),
            }
            for key in (
                "factor_mandates",
                "actual_fund_history",
                "post_inception_and_actions",
                "unleveraged_fund_identity",
            )
        },
    }
    with pytest.raises(QuantError, match="adapter"):
        readiness_blockers(etf, policy, tmp_path)
    invoked = []
    monkeypatch.setattr(
        "us_quant.research_program.verified_etf_market",
        lambda *args: invoked.append("verified_actual_ETF_snapshot"),
    )
    assert readiness_blockers(etf, policy, tmp_path) == []
    assert invoked == ["verified_actual_ETF_snapshot"]
    assert len(readiness_blockers(raw, policy, tmp_path)) == 5
