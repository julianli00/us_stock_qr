from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_lagged_spy_price_impact_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture
def data(market_factory):
    return market_factory(start="2015-08-03", end="2019-12-31", symbols=study.law.UNIVERSE)


def native(data):
    records = []
    targets, audits = study.law.build_targets(
        data, read_json(study.law.POLICY), retain=records.append
    )
    return targets, audits, records


def test_actual_financial_policy_preserves_full_OHLCV_law_without_hidden_knobs():
    policy = read_json(study.POLICY)
    study.validate_policy(policy)
    policy["optional_input_lag"] = 0
    with pytest.raises(QuantError):
        study.validate_policy(policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("prior_completed_configurations", 228),
        ("prior_completed_configurations", 262),
        ("real_source_targets_before_actual_publication", True),
        ("actual_weight_band", 0.01),
        ("capital_usd_per_formal_path", 20000.0),
        ("ordinary_decisions", "daily"),
        ("registered_expert_or_pool_admission", True),
        ("source_input_interface", "close_only"),
        ("standalone_financial_failure_is_technical_exclusion", True),
    ],
)
def test_scope_and_original_financial_contract_cannot_drift(key, value):
    policy = deepcopy(read_json(study.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        study.validate_policy(policy)


def packet(monkeypatch):
    frozen = {"study_id": study.STUDY, "decision_program_kind": study.KIND}
    registered = {
        "sha256": "a" * 64,
        "at": "2026-11-05T09:50:44Z",
        "body": {"candidate": study.law.CANDIDATE, "frozen": frozen},
    }
    reference = {"path": "synthetic/readiness", "sha256": "b" * 64}
    readiness = {
        "readiness_kind": "synthetic_Interface_and_pinned_market_identity_only",
        "historical_new_policy_targets_computed": False,
        "historical_target_derivation_verified": False,
    }
    body = {**study.packet_body(frozen, registered, reference), "created_at": registered["at"]}
    monkeypatch.setattr(study, "events", lambda kind: [registered] if kind == "registered" else [])
    monkeypatch.setattr(study, "reference", lambda path: reference)
    monkeypatch.setattr(
        study, "read_json", lambda path: body if path == study.PACKET else readiness
    )
    return frozen, readiness, body


def test_schema3_defers_indicator_proof_and_preserves_old_namespace(monkeypatch):
    original = study.shared.PACKET
    frozen, _, body = packet(monkeypatch)
    assert study.checked_packet(frozen) == body
    assert body["schema_version"] == 3 and body["complete_fixed_cohort_ids"] == [study.ID]
    assert body["historical_new_policy_targets_computed"] is False
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert study.shared.PACKET is original


@pytest.mark.parametrize(
    "key",
    [
        "candidate_accounts_or_metrics_computed",
        "historical_new_policy_targets_computed",
        "historical_target_proof_deferred_until_published_execution",
        "order_authority",
    ],
)
def test_readiness_cannot_be_relabelled_actual_indicator_proof(monkeypatch, key):
    frozen, _, body = packet(monkeypatch)
    body[key] = not body[key]
    with pytest.raises(QuantError):
        study.checked_packet(frozen)


def test_actual_publication_required_before_any_indicator_generation(monkeypatch):
    fake = SimpleNamespace(JOURNAL="synthetic", journal_events=lambda path: [])
    monkeypatch.setattr(study, "base", fake)
    monkeypatch.setattr(study, "events", lambda kind: [])
    with pytest.raises(QuantError, match="precede real"):
        study.prepare_source(None, {})


def test_partial_source_is_never_silently_regenerated(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "absent.json")
    monkeypatch.setattr(study, "publication_fence", lambda: {"sha256": "a" * 64})
    (tmp_path / "source-targets-started-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="incomplete"):
        study.prepare_source(None, {})


def test_actual_source_Seam_passes_MarketData_not_close_frame(data, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "absent.json")
    monkeypatch.setattr(study, "publication_fence", lambda: {"sha256": "a" * 64})

    def capture(value, *args, **kwargs):
        assert value is data and value is not data.close
        raise QuantError("Synthetic stop after full MarketData was passed")

    monkeypatch.setattr(study.law, "build_targets", capture)
    with pytest.raises(QuantError, match="full MarketData"):
        study.prepare_source(data, {})


def test_failed_remote_fence_never_starts_source_or_capital(monkeypatch, tmp_path):
    from contextlib import nullcontext

    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(study, "RESULTS", tmp_path / "absent.json")

    def failed(*args):
        raise QuantError("Synthetic remote publication failure")

    monkeypatch.setattr(study, "publication", failed)
    monkeypatch.setattr(
        study, "prepare_source", lambda *args: pytest.fail("Real indicator started")
    )
    monkeypatch.setattr(study, "evaluate_path", lambda *args: pytest.fail("Capital started"))
    with pytest.raises(QuantError, match="remote publication"):
        study.evaluate("c" * 40)


@pytest.mark.parametrize("tamper", ["reported_price", "own_session", "state", "target"])
def test_independent_binary64_oracle_binds_all_lagged_inputs_and_state(data, tamper):
    targets, audits, records = native(data)
    proof = study.verify_targets(data, targets, audits, records)
    assert proof["all_independent_lagged_price_impact_targets_exact"]
    assert proof["source_price_volume_binary64_inputs_checked"] == 820 * len(audits)
    if tamper == "reported_price":
        records[0]["reported_close_binary64_hex"][0] = float(999).hex()
    elif tamper == "own_session":
        records[0]["impact_return_session_dates"][-1] = records[0]["decision_session"]
    elif tamper == "state":
        audits[0]["relative_impact_ratio"] += 0.01
    else:
        targets.loc[targets.dropna(how="all").index[0], "SPY"] = 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, targets, audits, records)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_source_adapter_preserves_original_money_and_scalar_BT(data, cost, delay, monkeypatch):
    from us_quant import observed_weight_execution as original

    targets, _, _ = native(data)
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    start, end = "2016-10-06", "2019-12-30"
    output = study.run_calendar(data, targets, start, end, cost_bps=cost, delay=delay)
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
    pd.testing.assert_frame_equal(output.result.frame, baseline.frame, check_exact=True)
    pd.testing.assert_frame_equal(output.issued, issued, check_exact=True)
    assert output.decisions[0]["signal_session"] == "2016-10-05"
    assert output.decisions[0]["assessment"]["source_opinion_known_at"] == "2016-09-30"
    assert output.decisions[0]["main_equity_at_decision"] == 10000
    assert study.verify_opinions(output, targets)[
        "all_actual_source_opinions_and_known_at_dates_exact"
    ]
    independent = study.base.independent_equity(
        data, output.issued, start, end, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    assert np.max(abs(output.result.frame["equity"] - independent["equity"])) < 1e-6


def test_real_money_prefix_exact_and_no_old_namespace_mutation(data, monkeypatch):
    from us_quant.data import MarketData

    old = study.shared.FOLDER, study.shared.ID, study.shared.law
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    targets, _, _ = native(data)
    full = study.run_calendar(data, targets, "2016-10-06", "2019-12-30", cost_bps=5, delay=1)
    day = pd.Timestamp("2018-06-29")
    prefix_data = MarketData(
        data.open.loc[:day],
        data.close.loc[:day],
        data.raw_close.loc[:day],
        data.volume.loc[:day],
        data.risk_free.loc[:day],
    )
    earlier = study.run_calendar(
        prefix_data, targets.loc[:day], "2016-10-06", str(day.date()), cost_bps=5, delay=1
    )
    pd.testing.assert_frame_equal(
        earlier.result.frame, full.result.frame.loc[:day], check_exact=True
    )
    assert earlier.decisions == [
        row for row in full.decisions if pd.Timestamp(row["signal_session"]) <= day
    ]
    assert (study.shared.FOLDER, study.shared.ID, study.shared.law) == old


def test_dependency_closure_contains_actual_volume_and_funding_Interfaces():
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/lagged_spy_price_impact_expert_v1.py" in paths
    assert "src/us_quant/data.py" in paths
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
