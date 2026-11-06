import copy
import math
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
from test_forecast_risk_normalizer_diagnosis_v1 import pair
from test_forecast_risk_normalizer_diagnosis_v2 import source_records

from us_quant import forecast_opportunity_gate_v1 as rules
from us_quant import forecast_risk_normalizer_diagnosis_v2 as certificates
from us_quant import ssd_commitment_research_v1 as contract
from us_quant import ssd_forecast_opportunity_gate_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now


def joined(source=None):
    source = source_records() if source is None else source
    _, classified = certificates.diagnose(source)
    result = []
    for i, (audit, decision) in enumerate(zip(source, classified, strict=True)):
        targets = {
            n: [0.98 if s == ("MTUM" if j == 0 else "VLUE") else 0.0 for s in rules.ASSETS]
            for j, n in enumerate(rules.SOURCE_IDS)
        }
        result.append(
            {
                "allocation_before": {
                    "record": "saved_forecast_allocation_input",
                    "source_before": {"as_of": audit["decision_session"], "decision_index": i},
                    "forecast_resolved": {
                        "audit": {
                            "forecast_available": audit["candidates"][rules.SOURCE_IDS[0]][
                                "scores_available"
                            ]
                        }
                    },
                },
                "allocation_resolved": {
                    "record": "saved_forecast_allocation_resolved",
                    "audit": copy.deepcopy(audit),
                    "targets": targets,
                },
                "certificate_before": {
                    "record": "normalizer_input",
                    "index": i,
                    "source_audit": copy.deepcopy(audit),
                },
                "certificate_resolved": {
                    "record": "normalizer_resolved",
                    "index": i,
                    "result": copy.deepcopy(decision),
                },
            }
        )
    return result


def test_positive_targets_copy_exactly_negative_and_unscored_are_explicit_BIL():
    policy = read_json(rules.POLICY)
    data = joined()
    for i, row in enumerate(data):
        targets, audit = rules.gate(row, policy)
        rules.verify(row, targets, audit, policy)
        for name, old in zip(rules.IDS, rules.SOURCE_IDS, strict=True):
            decision = audit["candidates"][name]
            if i == 7:
                assert decision["participation"] is True
                np.testing.assert_array_equal(
                    targets[name], row["allocation_resolved"]["targets"][old]
                )
            else:
                assert not decision["participation"] and decision["mode"] == "funding_standby"
                np.testing.assert_array_equal(targets[name], rules.standby())
            assert not decision["funding_standby_is_strategy_expert"]
            assert not decision["source_advisor_state_changed"]
            assert not decision["source_proxy_is_new_allocation_actual_cost_forecast"]
    assert audit["candidates"][rules.IDS[0]]["source_numerator_sign"] == "negative"


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_zero_boundary_and_one_ULP_use_raw_sign_not_rounded_score(direction):
    source = source_records()[:8]
    cost = 0.0001
    boundary = cost / 21
    mean = (
        boundary
        if direction == 0
        else math.nextafter(boundary, math.inf if direction > 0 else -math.inf)
    )
    for candidate in source[-1]["candidates"].values():
        candidate["pairs"] = [pair(p["pair"], mean=mean, cost=cost) for p in candidate["pairs"]]
        for record in candidate["pairs"]:
            record["selection_score_key"] = certificates.source_key(record["selection_score"], 12)
    row = joined(source)[-1]
    targets, audit = rules.gate(row, read_json(rules.POLICY))
    rules.verify(row, targets, audit, read_json(rules.POLICY))
    assert all(c["participation"] is (direction > 0) for c in audit["candidates"].values())


@pytest.mark.parametrize("damage", ["future", "missing", "nan", "source", "key", "mode"])
def test_missing_or_corrupted_mature_information_is_not_cash_success(damage):
    row = joined()[7]
    if damage == "future":
        row["certificate_resolved"]["result"]["decision_session"] = "2026-01-01"
    elif damage == "missing":
        row["certificate_resolved"]["result"]["models"][rules.MODELS[0]]["status"] = "missing"
    elif damage == "nan":
        for p in row["certificate_resolved"]["result"]["models"][rules.MODELS[0]]["pairs"]:
            if p["was_recorded_selected"]:
                p["used_binary64_post_cost_numerator_hex"] = "nan"
    elif damage == "source":
        row["certificate_before"]["source_audit"]["candidates"][rules.SOURCE_IDS[0]]["model"] = (
            "foreign"
        )
    elif damage == "key":
        for p in row["certificate_resolved"]["result"]["models"][rules.MODELS[0]]["pairs"]:
            if p["was_recorded_selected"]:
                p["source_NumPy_and_scalar_scaled_key_bit_exact"] = False
    else:
        row["allocation_resolved"]["targets"].pop(rules.SOURCE_IDS[0])
    with pytest.raises(QuantError):
        rules.gate(row, read_json(rules.POLICY))


def test_independent_raw_source_verifier_rejects_bad_sign_target_or_participation():
    row = joined()[7]
    policy = read_json(rules.POLICY)
    for damage in ("target", "sign", "participation", "cost"):
        targets, audit = rules.gate(row, policy)
        changed = copy.deepcopy(row)
        if damage == "target":
            targets[rules.IDS[0]] = rules.standby()
        elif damage == "sign":
            audit["candidates"][rules.IDS[0]]["source_numerator_sign"] = "negative"
        elif damage == "participation":
            audit["candidates"][rules.IDS[0]]["participation"] = False
        else:
            for p in changed["allocation_resolved"]["audit"]["candidates"][rules.SOURCE_IDS[0]][
                "pairs"
            ]:
                if (
                    p["pair"]
                    == changed["allocation_resolved"]["audit"]["candidates"][rules.SOURCE_IDS[0]][
                        "selected_pair"
                    ]
                ):
                    p["estimated_incremental_switch_cost_fraction"] = -0.001
            changed["certificate_before"]["source_audit"] = copy.deepcopy(
                changed["allocation_resolved"]["audit"]
            )
        with pytest.raises(QuantError):
            rules.verify(changed, targets, audit, policy)


def test_future_certificates_do_not_change_prior_targets_or_source_state():
    source = source_records()
    before = joined(source)
    changed = copy.deepcopy(source)
    for c in changed[-1]["candidates"].values():
        c["pairs"] = [pair(p["pair"], mean=0.003) for p in c["pairs"]]
        for p in c["pairs"]:
            p["selection_score_key"] = certificates.source_key(p["selection_score"], 12)
    later = joined(changed)
    assert before[:-1] == later[:-1]
    policy = read_json(rules.POLICY)
    original = copy.deepcopy(before)
    for a, b in zip(before[:-1], later[:-1], strict=True):
        targets, audit = rules.gate(a, policy)
        second, other = rules.gate(b, policy)
        assert audit == other
        for name in rules.IDS:
            np.testing.assert_array_equal(targets[name], second[name])
    assert before == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("strict_positive_post_cost_active_numerator_threshold", 0.001),
        ("gate_uses_rounded_score_or_fitted_threshold", True),
        ("missing_mature_certificate_is_standby", True),
        ("funding_standby_is_strategy_expert", True),
        ("source_advisor_state_recomputed_with_gate", True),
        ("source_proxy_is_new_allocation_actual_cost_forecast", True),
        ("actual_weight_trade_band", 0.01),
        ("new_source_RF_views", 1),
    ],
)
def test_no_fitted_hurdle_source_state_fee_or_qualification_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_generated_target_proof_passes_real_existing_packet_entry_contract():
    candidate, frozen = rules.CANDIDATES[0], {"study_id": "synthetic_gate"}
    evidence = study.target_proof(frozen, {"decision_count": 9})
    registered = {
        "at": "2026-11-06T00:00:00+00:00",
        "sha256": "registered",
        "body": {"candidate": candidate, "frozen": frozen},
    }
    proved = {
        "at": utc_now(),
        "sha256": "proof",
        "body": {
            "registered_event_sha256": "registered",
            "targets": {"path": "target", "sha256": "t"},
            "proof": {"path": "proof", "sha256": "p"},
        },
    }
    reader = SimpleNamespace(
        matching_events=lambda kind, identifier: [registered] if kind == "registered" else [proved],
        checked_reference=lambda ref: ref["path"],
    )
    real = FunctionType(
        contract.packet_entry.__code__,
        {**contract.packet_entry.__globals__, "base": reader, "read_json": lambda path: evidence},
    )
    assert real(candidate, frozen)["candidate_id"] == candidate["id"]
    assert evidence["original_source_signal_recomputed"] is False
    with pytest.raises(QuantError):
        study.target_proof(frozen, {"original_source_signal_recomputed": True})


def test_both_registrations_precede_either_new_gated_target(monkeypatch):
    history, packets = [], []
    frozen = {"study_id": "synthetic_gate"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": str(len(history))}
        )

    def prepare(data, experts, actual):
        assert actual == frozen and all(len(events("registered", n)) == 1 for n in rules.IDS)
        return {
            n: {
                "targets": {"path": "target", "sha256": "t"},
                "proof": {"path": "proof", "sha256": "p"},
            }
            for n in rules.IDS
        }

    common = SimpleNamespace(
        packet_entry=lambda candidate, actual: {"candidate_id": candidate["id"]},
        retain_receipt=lambda path, packet, *args: packets.append(packet),
    )

    def bind(function, **overrides):
        return FunctionType(
            function.__code__,
            {
                **function.__globals__,
                **study.__dict__,
                "lifecycle": SimpleNamespace(runtime_guard=lambda: None, research_job=nullcontext),
                "inputs": lambda: (None, None, frozen, None),
                "events": events,
                "base": SimpleNamespace(JOURNAL=None, append_event=append),
                "prepare": prepare,
                "seven": SimpleNamespace(common=common),
                "reference": lambda path: {"path": str(path), "sha256": "x"},
                "progress": lambda *args, **kwargs: None,
                **overrides,
            },
            argdefs=function.__defaults__,
        )

    monkeypatch.setattr(study, "bound", bind)
    study.register()
    assert [r["kind"] for r in history][:2] == ["registered"] * 2
    assert len(packets) == 2


def test_foreign_last_identity_stops_before_first_append(monkeypatch):
    monkeypatch.setattr(study, "bound", lambda function: lambda: (None, None, {"frozen": 1}, None))
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [{"body": {"foreign": True}}] if name == rules.IDS[1] else [],
    )
    with pytest.raises(QuantError):
        study.inputs()


def test_missing_second_complete_packet_blocks_any_funding():
    first, second = rules.IDS
    packet = {"complete_fixed_cohort_ids": list(rules.IDS), "cohort_registrations": ["r0", "r1"]}

    def events(kind, name):
        return [{"sha256": "r0" if name == first else "r1"}] if kind == "registered" else []

    def read(path):
        return (
            packet
            if path == study.PACKETS[first]
            else {**packet, "complete_fixed_cohort_ids": [second]}
        )

    checked = FunctionType(
        study.template.cohort_publication.__code__,
        {
            **study.template.cohort_publication.__globals__,
            **study.__dict__,
            "base": SimpleNamespace(JOURNAL=None, journal_events=lambda path: []),
            "events": events,
            "read_json": read,
            "scoped": lambda *args: lambda *unused: {"ok": True},
        },
    )
    with pytest.raises(QuantError):
        checked({"study_id": "synthetic_gate"}, "remotecommit")


def test_all_actual_gate_fixture_and_contract_dependencies_are_bound():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_commitment_research_v1.py" in paths
