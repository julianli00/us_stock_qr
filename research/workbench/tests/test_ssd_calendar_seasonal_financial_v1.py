from copy import deepcopy
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_calendar_seasonal_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json

PACKET = "original-test-module-scope"


def bound(function, **values):
    return FunctionType(
        function.__code__, {**function.__globals__, **values}, argdefs=function.__defaults__
    )


@pytest.fixture
def data(market_factory):
    return market_factory(start="2015-08-03", end="2019-12-31", symbols=study.law.UNIVERSE)


def native(data):
    records = []
    targets, audits = study.law.build_targets(
        data.close, read_json(study.law.POLICY), retain=records.append
    )
    return targets, audits, records


def fixture_packet():
    frozen = {"study_id": study.STUDY, "decision_program_kind": study.KIND}
    registered = {
        "sha256": "a" * 64,
        "at": "2026-11-05T03:00:00Z",
        "body": {"candidate": study.law.CANDIDATE, "frozen": frozen},
    }
    reference = {"path": "synthetic/readiness", "sha256": "b" * 64}
    readiness = {
        "readiness_kind": "synthetic_Interface_and_pinned_market_identity_only",
        "historical_new_policy_targets_computed": False,
        "historical_target_derivation_verified": False,
    }
    packet = {
        **study.packet_body(frozen, registered, reference),
        "created_at": registered["at"],
    }
    return frozen, registered, reference, readiness, packet


def checked(frozen, registered, reference, readiness, packet):
    return bound(
        study.checked_packet,
        events=lambda kind: [registered] if kind == "registered" else [],
        reference=lambda path: reference,
        read_json=lambda path: packet if path == study.PACKET else readiness,
    )(frozen)


def test_program_packet_defers_real_target_proof_and_not_source_outputs():
    frozen, registered, reference, readiness, packet = fixture_packet()
    assert checked(frozen, registered, reference, readiness, packet) == packet
    assert packet["schema_version"] == 3
    assert packet["historical_new_policy_targets_computed"] is False
    assert packet["entries"][0]["historical_target_derivation_proof"] is None
    assert packet["candidate_accounts_or_metrics_computed"] is False


@pytest.mark.parametrize(
    "key",
    [
        "candidate_accounts_or_metrics_computed",
        "historical_new_policy_targets_computed",
        "historical_target_proof_deferred_until_published_execution",
        "order_authority",
    ],
)
def test_packet_flags_or_cohort_cannot_be_relabelled(key):
    frozen, registered, reference, readiness, packet = fixture_packet()
    packet[key] = not packet[key]
    with pytest.raises(QuantError):
        checked(frozen, registered, reference, readiness, packet)


def test_synthetic_readiness_is_never_actual_historical_target_derivation():
    frozen, registered, reference, readiness, packet = fixture_packet()
    readiness["historical_target_derivation_verified"] = True
    with pytest.raises(QuantError, match="Synthetic calendar"):
        checked(frozen, registered, reference, readiness, packet)
    with pytest.raises(QuantError, match="actual unique"):
        bound(study.checked_packet, events=lambda kind: [])(frozen)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prior_completed_configurations", 228),
        ("real_source_targets_before_actual_publication", True),
        ("actual_weight_band", 0.01),
        ("capital_usd_per_formal_path", 20000.0),
        ("ordinary_decisions", "daily"),
        ("registered_expert_or_pool_admission", True),
        ("standalone_financial_failure_is_technical_exclusion", True),
    ],
)
def test_prepared_law_and_original_financial_contract_not_softened(key, value):
    invalid = deepcopy(read_json(study.POLICY))
    invalid[key] = value
    with pytest.raises(QuantError):
        study.validate_policy(invalid)


def test_actual_financial_policy_is_exact_and_has_no_hidden_control_knob():
    policy = read_json(study.POLICY)
    study.validate_policy(policy)
    policy["season_shift"] = 1
    with pytest.raises(QuantError):
        study.validate_policy(policy)


def test_no_actual_publication_means_no_calendar_source_targets():
    fake_base = SimpleNamespace(JOURNAL="synthetic", journal_events=lambda path: [])
    with pytest.raises(QuantError, match="precede real calendar outputs"):
        bound(
            study.prepare_source,
            publication_fence=bound(
                study.publication_fence, base=fake_base, events=lambda kind: []
            ),
        )(None, {})


def test_partial_source_preparation_is_not_generated_again(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "absent.json")
    monkeypatch.setattr(study, "publication_fence", lambda: {"sha256": "a" * 64})
    (tmp_path / "source-targets-started-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="incomplete calendar targets"):
        study.prepare_source(None, {})


def test_failed_remote_fence_blocks_source_and_account(monkeypatch, tmp_path):
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    from contextlib import nullcontext

    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(study, "RESULTS", tmp_path / "absent.json")

    def fail(*args):
        raise QuantError("Synthetic remote publication failure")

    monkeypatch.setattr(study, "publication", fail)
    monkeypatch.setattr(
        study, "prepare_source", lambda *args: pytest.fail("Real source targets started")
    )
    monkeypatch.setattr(
        study, "evaluate_path", lambda *args: pytest.fail("A funded account started")
    )
    with pytest.raises(QuantError, match="remote publication"):
        study.evaluate("c" * 40)


def test_complete_path_recovery_verifies_artifacts_without_execution():
    reference = {"path": "synthetic/complete", "sha256": "d" * 64}
    fake_base = SimpleNamespace(verify_path=lambda *args: {"actual_complete_verified": True})
    function = bound(
        study.evaluate_path,
        base=fake_base,
        events=lambda kind: [{"body": {"path_key": "10y-base", "receipt": reference}}]
        if kind == "path_completed"
        else [],
    )
    assert function(None, None, {}, {"years": 10}, {"id": "base"}, None) == {
        "actual_complete_verified": True
    }
    with pytest.raises(QuantError, match="Preserve a partial"):
        bound(
            study.evaluate_path,
            events=lambda kind: [{"body": {"path_key": "10y-base"}}]
            if kind == "path_started"
            else [],
        )(None, None, {}, {"years": 10}, {"id": "base"}, None)


def test_native_source_oracle_checks_every_retained_input_then_target(data):
    targets, audits, records = native(data)
    checked = study.verify_targets(data.close, targets, audits, records)
    assert checked["native_month_end_opinions"] == len(audits)
    assert checked["all_independent_calendar_targets_exact"]
    records[0]["intended_calendar_month"] = 7
    with pytest.raises(QuantError):
        study.verify_targets(data.close, targets, audits, records)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_new_source_adapter_matches_original_funded_execution_and_scalar_bt(
    data, cost, delay, monkeypatch
):
    from us_quant import observed_weight_execution as original

    targets, _, _ = native(data)
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    start, end = "2016-10-06", "2019-12-30"
    result = study.run_calendar(
        data,
        targets,
        start,
        end,
        cost_bps=cost,
        delay=delay,
    )
    baseline, issued, _ = original.run(
        data,
        targets,
        original.CANDIDATES[0],
        read_json(original.POLICY),
        start,
        end,
        cost,
        delay,
    )
    pd.testing.assert_frame_equal(result.result.frame, baseline.frame, check_exact=True)
    pd.testing.assert_frame_equal(result.issued, issued, check_exact=True)
    assert result.decisions[0]["signal_session"] == "2016-10-05"
    assert result.decisions[0]["assessment"]["source_opinion_known_at"] == "2016-09-30"
    assert result.decisions[0]["main_equity_at_decision"] == 10000
    assert study.verify_opinions(result, targets)[
        "all_actual_source_opinions_and_known_at_dates_exact"
    ]
    assert study.capital.verify_capital_inputs(result)[
        "main_dollar_holdings_cash_and_equity_match_actual_close"
    ]
    independent = study.base.independent_equity(
        data, result.issued, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    assert np.max(abs(result.result.frame["equity"] - independent["equity"])) < 1e-6
    study.verify_feedback(
        data,
        result.desired,
        result.result,
        result.issued,
        result.decisions,
        {"start": start, "end": end},
        {"delay_sessions": delay},
    )


def test_future_source_or_prices_cannot_change_a_prior_main_prefix(data, monkeypatch):
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    targets, _, _ = native(data)
    start, end = "2016-10-06", "2019-12-30"
    full = study.run_calendar(data, targets, start, end, cost_bps=5, delay=1)
    day = pd.Timestamp("2018-06-29")
    prefix_data = study.MarketData(
        data.open.loc[:day].copy(),
        data.close.loc[:day].copy(),
        data.raw_close.loc[:day].copy(),
        data.volume.loc[:day].copy(),
        data.risk_free.loc[:day].copy(),
    )
    prefix = study.run_calendar(
        prefix_data,
        targets.loc[:day],
        start,
        str(day.date()),
        cost_bps=5,
        delay=1,
    )
    pd.testing.assert_frame_equal(
        prefix.result.frame, full.result.frame.loc[:day], check_exact=True
    )
    assert prefix.decisions == [
        row for row in full.decisions if pd.Timestamp(row["signal_session"]) <= day
    ]


def test_publication_verifier_does_not_mutate_prior_packet_namespace(monkeypatch):
    def verifier(frozen, commit):
        return PACKET

    original_packet = verifier.__globals__.get("PACKET")
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study.base, "published_inputs", verifier)
    assert study.publication({}, "e" * 40) == study.PACKET
    assert verifier.__globals__.get("PACKET") is original_packet


def test_dependency_closure_includes_the_actual_source_and_funding_seams():
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/calendar_seasonal_expert_v1.py" in paths
    assert "config/calendar-seasonal-expert-v1.json" in paths
    assert "CALENDAR_SEASONAL_EXPERT_V1.md" in paths
    assert "src/us_quant/main_capital_feedback_v1.py" in paths
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
