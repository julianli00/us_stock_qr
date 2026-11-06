import copy
import json
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import macro_defense_hedge_trio_v1 as rules
from us_quant import ssd_macro_defense_hedge_trio_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def sources(changed_after=None, missing_initial=False):
    eight, calendar, assets = archived_experts(
        changed_after=changed_after, missing_initial=missing_initial
    )
    return study.select_sources(eight), calendar, assets


def test_exact_mechanism_slots_and_original_pair_combine_implementations():
    experts, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.SLOTS == (4, 5, 7)
    assert rules.choose_pair is study.template.rules.choose_pair
    assert rules.combine is study.template.rules.combine
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(3, 1 / 3))
    assert len(audit["pairs"]) == 3
    assert (weights > 0).sum() == 2
    assert rules.combine(observed, np.full(3, 1 / 3)).sum() == pytest.approx(0.98)
    assert all(len(p["expert_weights"]) == 3 for p in audit["pairs"])


def test_full_dates_unscored_states_and_exact_prefix_causality():
    experts, calendar, assets = sources()
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(experts, calendar, assets, policy)
    early, first = rules.build_targets(experts, calendar[calendar <= "2017-04-28"], assets, policy)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["equal_expert_weights"] == [1 / 3] * 3
    assert sum(not a["selection"]["scores_available"] for a in audits) == 7
    for name in full:
        pd.testing.assert_frame_equal(early[name], full[name].loc[:"2017-04-28"])
    assert first == [a for a in audits if a["decision_session"] <= "2017-04-28"]


def test_future_source_values_and_opinions_do_not_change_past():
    before, calendar, assets = sources()
    after, _, _ = sources(changed_after=pd.Timestamp("2017-04-28"))
    first, a = rules.build_targets(before, calendar, assets, read_json(rules.POLICY))
    later, b = rules.build_targets(after, calendar, assets, read_json(rules.POLICY))
    for name in first:
        pd.testing.assert_frame_equal(
            first[name].loc[:"2017-04-28"], later[name].loc[:"2017-04-28"]
        )
    assert [r for r in a if r["decision_session"] <= "2017-04-28"] == [
        r for r in b if r["decision_session"] <= "2017-04-28"
    ]


def test_actual_retained_inputs_and_three_pair_scalar_oracle(tmp_path):
    experts, calendar, assets = sources()
    records = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    checked = study.verify_targets(data, experts, targets, audits)
    assert checked["maximum_independent_target_difference"] < 1e-10
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    retained = study.verify_retained_records(path, experts, targets, audits)
    assert retained["actual_input_and_resolved_record_count"] == 2 * len(audits)
    assert all(len(r["observations"]) == 3 for r in records if r["record"] == "decision_input")
    damaged = copy.deepcopy(audits)
    scored = next(a for a in damaged if a["selection"]["scores_available"])
    not_selected = next(
        p for p in scored["selection"]["pairs"] if p["pair"] != scored["selection"]["selected_pair"]
    )
    not_selected["validation_net_risk_score"] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, damaged)


def test_unavailable_selected_source_is_not_silently_removed_or_filled():
    experts, calendar, assets = sources(missing_initial=True)
    with pytest.raises(QuantError):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_source_slots", [3, 5, 7]),
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("mature_pair_count", 28),
        ("decision_information_clock", "at_16_00_ET"),
        ("all_source_native_RF_claimed_equal_without_view", True),
        ("macro_proxy_is_company_EPS_or_new_independent_factor", True),
        ("mechanism_labels_are_independent_risks", True),
        ("new_source_RF_views", 1),
    ],
)
def test_no_source_window_cost_scope_or_qualification_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_both_real_registrations_precede_either_new_combination(monkeypatch):
    history, packets = [], []
    frozen = {"study_id": "synthetic_trio"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": f"event{len(history)}"}
        )

    def prepare(data, experts, actual):
        assert actual == frozen and all(len(events("registered", n)) == 1 for n in study.IDS)
        return {
            n: {
                "targets": {"path": "target", "sha256": "a"},
                "proof": {"path": "proof", "sha256": "b"},
            }
            for n in study.IDS
        }

    def retain(path, packet, *args):
        assert packet["complete_fixed_cohort_ids"] == list(study.IDS)
        assert len(packet["cohort_registrations"]) == 2
        packets.append(packet)

    common = SimpleNamespace(
        packet_entry=lambda candidate, actual: {"candidate_id": candidate["id"]},
        retain_receipt=retain,
    )

    def bind(function, **overrides):
        context = {
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
        }
        return FunctionType(function.__code__, context, argdefs=function.__defaults__)

    monkeypatch.setattr(study, "bound", bind)
    study.register()
    assert [r["kind"] for r in history][:2] == ["registered"] * 2
    assert len(packets) == 2


def test_foreign_last_identity_blocks_the_first_append(monkeypatch):
    monkeypatch.setattr(study, "bound", lambda function: lambda: (None, None, {"frozen": 1}, None))
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [{"body": {"foreign": True}}] if name == study.IDS[1] else [],
    )
    with pytest.raises(QuantError):
        study.inputs()


def test_missing_second_complete_packet_blocks_all_funding():
    first, second = study.IDS
    packet = {
        "complete_fixed_cohort_ids": list(study.IDS),
        "cohort_registrations": ["r0", "r1"],
    }

    def events(kind, name):
        return [{"sha256": "r0" if name == first else "r1"}] if kind == "registered" else []

    def read(path):
        return (
            packet
            if path == study.PACKETS[first]
            else {**packet, "complete_fixed_cohort_ids": [second]}
        )

    context = {
        **study.template.cohort_publication.__globals__,
        **study.__dict__,
        "base": SimpleNamespace(JOURNAL=None, journal_events=lambda path: []),
        "events": events,
        "read_json": read,
        "scoped": lambda *args: lambda *unused: {"ok": True},
    }
    checked = FunctionType(study.template.cohort_publication.__code__, context)
    with pytest.raises(QuantError):
        checked({"study_id": "synthetic"}, "remotecommit")


def test_real_admission_views_and_all_local_fixture_dependencies_stay_bound():
    policy = read_json(rules.POLICY)
    catalogue = read_json(study.base.checked_reference(policy["source_catalogue"]))
    assert catalogue[
        "admitted_legacy_defensive_recommendation_and_assessment_views_reused_unchanged"
    ]
    assert not catalogue["native_RF_of_all_sources_claimed_equal_without_view"]
    assert catalogue["expert_ids"] == list(rules.EXPERT_IDS)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/strategy_expert_pool_v2.py" in paths
