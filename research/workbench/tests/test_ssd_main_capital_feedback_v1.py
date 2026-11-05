from copy import deepcopy
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

from us_quant import main_capital_feedback_v1 as rules
from us_quant import ssd_main_capital_feedback_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json

PACKET = "original-test-module-scope"


def bound(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **overrides},
        argdefs=function.__defaults__,
    )


def fixture_packet():
    frozen = {"study_id": rules.STUDY, "decision_program_kind": study.KIND}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-04T00:00:00Z",
        "body": {
            "candidate": {
                "id": study.ID,
                "weighting": "original_rolling_pair_close_mark_main_capital_cost",
            },
            "frozen": frozen,
        },
    }
    ready_ref = {"path": "synthetic/program-readiness.json", "sha256": "b" * 64}
    readiness = {
        "readiness_kind": "synthetic_Interface_and_archived_input_identity_only",
        "historical_new_meta_scores_targets_accounts_computed": False,
        "historical_target_derivation_verified": False,
    }
    packet = {
        **study.packet_body(frozen, registration, ready_ref),
        "created_at": registration["at"],
    }
    return frozen, registration, ready_ref, readiness, packet


def checked(frozen, registration, ready_ref, readiness, packet):
    return bound(
        study.checked_packet,
        events=lambda kind: [registration] if kind == "registered" else [],
        reference=lambda path: ready_ref,
        read_json=lambda path: packet if path == study.PACKET else readiness,
    )(frozen)


def test_dynamic_packet_explicitly_defers_historical_target_proof():
    frozen, registered, ref, readiness, packet = fixture_packet()
    assert checked(frozen, registered, ref, readiness, packet) == packet
    assert packet["schema_version"] == 3
    assert packet["target_delivery_kind"] == study.KIND
    assert packet["candidate_accounts_or_metrics_computed"] is False
    assert packet["historical_new_meta_scores_or_targets_computed"] is False
    assert packet["entries"][0]["historical_target_derivation_proof"] is None


@pytest.mark.parametrize(
    "flag",
    [
        "candidate_accounts_or_metrics_computed",
        "historical_new_meta_scores_or_targets_computed",
        "original_source_signal_recomputed",
        "investment_objective_verified",
        "order_authority",
    ],
)
def test_packet_cannot_claim_prep_is_real_outputs_or_trading(flag):
    frozen, registered, ref, readiness, packet = fixture_packet()
    packet[flag] = True
    with pytest.raises(QuantError, match="complete explicit dynamic"):
        checked(frozen, registered, ref, readiness, packet)


def test_synthetic_readiness_cannot_be_a_historical_target_proof():
    frozen, registered, ref, readiness, packet = fixture_packet()
    readiness["historical_target_derivation_verified"] = True
    with pytest.raises(QuantError, match="Synthetic readiness"):
        checked(frozen, registered, ref, readiness, packet)


def test_undeclared_cohort_or_missing_actual_registration_is_not_publishable():
    frozen, registered, ref, readiness, packet = fixture_packet()
    packet["complete_fixed_cohort_ids"].append("another_unregistered_rule")
    with pytest.raises(QuantError):
        checked(frozen, registered, ref, readiness, packet)
    with pytest.raises(QuantError, match="one actual unique"):
        bound(study.checked_packet, events=lambda kind: [])(frozen)


def test_historical_path_cannot_start_without_actual_publication_fence():
    fake_base = SimpleNamespace(JOURNAL="synthetic", journal_events=lambda path: [])
    function = bound(study.evaluate_path, base=fake_base, events=lambda kind: [])
    with pytest.raises(QuantError, match="before actual registration and publication"):
        function(None, None, {}, {"years": 10}, {"id": "base"}, None)


def test_partial_path_is_not_silently_replayed():
    fake_base = SimpleNamespace(JOURNAL="synthetic")
    function = bound(
        study.evaluate_path,
        base=fake_base,
        events=lambda kind: [{"body": {"path_key": "10y-base"}}] if kind == "path_started" else [],
    )
    with pytest.raises(QuantError, match="Preserve a partial"):
        function(None, None, {}, {"years": 10}, {"id": "base"}, None)


def test_completed_receipt_is_verified_without_recomputing_capital():
    receipt = {"path": "synthetic/complete.json", "sha256": "c" * 64}
    called = []

    def verify(reference, frozen, identifier, key):
        called.append((reference, frozen, identifier, key))
        return {"already_complete": True}

    fake_base = SimpleNamespace(verify_path=verify)
    function = bound(
        study.evaluate_path,
        base=fake_base,
        events=lambda kind: [{"body": {"path_key": "10y-base", "receipt": receipt}}]
        if kind == "path_completed"
        else [],
    )
    assert function(None, None, {}, {"years": 10}, {"id": "base"}, None) == {
        "already_complete": True
    }
    assert called == [(receipt, {}, study.ID, "10y-base")]


def test_publication_verifier_has_an_isolated_new_packet_not_old_namespace_mutation():
    def original_verifier(frozen, commit):
        return {"packet": PACKET, "commit": commit}

    fake_base = SimpleNamespace(published_inputs=original_verifier)
    new_packet = Path("synthetic/dynamic-packet.json")
    result = bound(
        study.publication, base=fake_base, PACKET=new_packet, checked_packet=lambda frozen: {}
    )({}, "d" * 40)
    assert result == {"packet": new_packet, "commit": "d" * 40}
    assert original_verifier.__globals__["PACKET"] == "original-test-module-scope"


@pytest.mark.parametrize(
    ("key", "changed"),
    [
        ("initial_drawdown_peak_usd", 0.0),
        ("lookback_sessions", 125),
        ("assessment_cost_bps_per_side", 0.0),
        ("actual_weight_band", 0.05),
        ("new_RF_or_source_assessment_Adapter", True),
        ("initialization_difference_from_cached_five_year_control_disclosed", False),
        ("prefix_replays_are_independent_economic_trials", True),
    ],
)
def test_frozen_protocol_cannot_be_softened_or_initialization_hidden(key, changed):
    original = read_json(rules.POLICY)
    invalid = deepcopy(original)
    invalid[key] = changed
    with pytest.raises(QuantError):
        study.validate_policy(invalid)


def test_actual_declared_policy_and_no_hidden_extra_knob():
    original = read_json(rules.POLICY)
    study.validate_policy(original)
    original["undisclosed_cost_penalty_multiplier"] = 0.5
    with pytest.raises(QuantError):
        study.validate_policy(original)
