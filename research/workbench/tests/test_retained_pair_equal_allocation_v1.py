import copy
import json
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import retained_pair_equal_allocation_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def case(scored=True):
    ids, assets = rules.EXPERT_IDS, list(rules.ASSETS)
    opinions = np.zeros((8, 14))
    opinions[:, 0] = 0.98
    opinions[2] = 0
    opinions[2, 4] = 0.98
    previous = [0.125] * 8
    weights = [0, 0, 0.7, 0.3, 0, 0, 0, 0] if scored else previous
    dates = (
        pd.bdate_range(end="2020-01-31", periods=126).strftime("%Y-%m-%d").tolist()
        if scored
        else []
    )
    before = {
        "record": "decision_input",
        "decision_index": 0,
        "as_of": "2020-01-31",
        "information_clock": rules.CLOCK,
        "previous_weights": previous,
        "previous_weights_binary64_hex": [float(x).hex() for x in previous],
        "binary64_values_truncated_to_fixed_decimals": False,
        "observations": [
            {
                "id": ids[i],
                "mechanism_group": rules.MECHANISMS[i],
                "as_of": "2020-01-31",
                "target_known_at": "2020-01-31",
                "target": opinions[i].tolist(),
                "asset_order": assets,
                "return_dates": dates,
                "NET_returns": [0.01] * len(dates),
                "RF": [0.0] * len(dates),
                "score_mature": scored,
            }
            for i in range(8)
        ],
    }
    after = {
        "record": "decision_resolved",
        "decision_index": 0,
        "asset_order": assets,
        "audit": {
            "decision_session": "2020-01-31",
            "decision_information_clock": rules.CLOCK,
            "previous_rolling_expert_weights": previous,
            "equal_expert_weights": previous,
            "rolling_expert_weights": weights,
            "selection": {
                "selected_pair": [ids[2], ids[3]] if scored else None,
                "scores_available": scored,
                "safety_or_success_claimed": False,
            },
            "shadow_wealth_added_to_main_capital": False,
            "source_feedback_floor_transferred_to_main_account": False,
        },
        "combined_targets": {
            rules.original.CANDIDATES[0]["id"]: (np.full(8, 0.125) @ opinions).tolist(),
            rules.original.CANDIDATES[1]["id"]: (np.asarray(weights) @ opinions).tolist(),
        },
    }
    return before, after, read_json(rules.POLICY)


def test_equal_halves_change_only_execution_allocation_not_source_advisor():
    before, after, policy = case()
    unchanged = copy.deepcopy((before, after))
    target, audit = rules.allocate_retained_pair(before, after, policy)
    assert (before, after) == unchanged
    np.testing.assert_array_equal(target[[0, 4]], [0.49, 0.49])
    assert audit["new_execution_expert_weights"] == [0, 0, 0.5, 0.5, 0, 0, 0, 0]
    assert audit["source_training_ERC_weights"] == [0, 0, 0.7, 0.3, 0, 0, 0, 0]
    assert audit["source_selected_pair"] == after["audit"]["selection"]["selected_pair"]
    assert not audit["source_proxy_is_new_allocation_actual_cost"]
    assert not audit["source_advisor_state_changed"]


def test_unscored_dates_copy_actual_equal8_without_claiming_safety():
    before, after, policy = case(False)
    target, audit = rules.allocate_retained_pair(before, after, policy)
    assert target.tolist() == after["combined_targets"][policy["source_equal_candidate_id"]]
    assert audit["new_execution_expert_weights"] == [0.125] * 8
    assert not audit["source_scores_available"]
    assert "not_a_successful_score" in audit["allocation_reason"]


@pytest.mark.parametrize(
    "kind",
    [
        "future_opinion",
        "future_return",
        "missing_history",
        "nonfinite_NET",
        "RF_view",
        "id",
        "mechanism",
        "asset_order",
        "negative_target",
        "bool_target",
        "pair",
        "immature",
        "state",
        "hex",
        "clock",
        "capital",
        "source_weight",
        "truncated",
    ],
)
def test_invalid_source_state_cannot_turn_into_an_allocation(kind):
    before, after, policy = copy.deepcopy(case())
    observation = before["observations"][0]
    if kind == "future_opinion":
        observation["target_known_at"] = "2020-02-01"
    elif kind == "future_return":
        for row in before["observations"]:
            row["return_dates"][-1] = "2020-02-03"
    elif kind == "missing_history":
        observation["NET_returns"].pop()
    elif kind == "nonfinite_NET":
        observation["NET_returns"][0] = float("nan")
    elif kind == "RF_view":
        observation["RF"][0] = 1e-10
    elif kind == "id":
        observation["id"] = "unadmitted"
    elif kind == "mechanism":
        observation["mechanism_group"] = "independent_risk_claim"
    elif kind == "asset_order":
        after["asset_order"] = list(reversed(rules.ASSETS))
    elif kind == "negative_target":
        observation["target"][0] = -0.98
    elif kind == "bool_target":
        observation["target"] = [True] * 14
    elif kind == "pair":
        after["audit"]["selection"]["selected_pair"] = ["foreign", rules.EXPERT_IDS[3]]
    elif kind == "immature":
        observation["score_mature"] = False
    elif kind == "state":
        before["previous_weights"] = [1.0] + [0.0] * 7
    elif kind == "hex":
        before["previous_weights_binary64_hex"][0] = (1.0).hex()
    elif kind == "clock":
        after["audit"]["decision_information_clock"] = "earlier"
    elif kind == "capital":
        after["audit"]["source_feedback_floor_transferred_to_main_account"] = True
    elif kind == "source_weight":
        after["audit"]["rolling_expert_weights"][2] = 0.8
    else:
        before["binary64_values_truncated_to_fixed_decimals"] = True
    with pytest.raises(QuantError):
        rules.allocate_retained_pair(before, after, policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("selected_pair_fraction", 0.4),
        ("actual_weight_trade_band", 0.05),
        ("source_training_sessions", 63),
        ("source_advisor_state_recomputed_with_new_allocation", True),
        ("advisor_cost_proxy_is_new_allocation_actual_cost", True),
    ],
)
def test_no_knob_feedback_or_cost_scope_change(key, value):
    policy = read_json(rules.POLICY)
    with pytest.raises(QuantError):
        rules.validate({**policy, key: value})


def test_source_pair_identity_not_full_sample_performance_selects_support():
    before, after, policy = case()
    after["audit"]["rolling_expert_weights"] = [0.8, 0, 0.2, 0, 0, 0, 0, 0]
    after["audit"]["selection"]["selected_pair"] = [rules.EXPERT_IDS[0], rules.EXPERT_IDS[2]]
    _, audit = rules.allocate_retained_pair(before, after, policy)
    assert audit["new_execution_expert_weights"] == [0.5, 0, 0.5, 0, 0, 0, 0, 0]


def test_later_source_records_cannot_change_an_earlier_target():
    earlier = case()
    target, audit = rules.allocate_retained_pair(*earlier)
    later = copy.deepcopy(case())
    later[1]["audit"]["rolling_expert_weights"] = [0, 0, 0.4, 0.6, 0, 0, 0, 0]
    actual, actual_audit = rules.allocate_retained_pair(*earlier)
    np.testing.assert_array_equal(actual, target)
    assert actual_audit == audit


def test_actual_registration_precedes_any_new_target_derivation(monkeypatch):
    from us_quant import ssd_retained_pair_equal_allocation_v1 as study

    history, phases = [], []
    frozen = {"study_id": "synthetic_retained_pair_equal", "new": "frozen"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": f"event{len(history)}"}
        )

    def prepare(data, experts, actual):
        assert actual == frozen
        assert len(events("registered", rules.ID)) == 1
        phases.append("target")
        return {
            rules.ID: {
                "targets": {"path": "target", "sha256": "a"},
                "proof": {"path": "proof", "sha256": "b"},
            }
        }

    common = SimpleNamespace(
        packet_entry=lambda candidate, actual: {"candidate_id": candidate["id"]},
        retain_receipt=lambda *args: phases.append("packet"),
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
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    study.register()
    assert [r["kind"] for r in history] == ["registered", "target_proof"]
    assert phases == ["target", "packet"]


def test_incomplete_actual_single_cohort_packet_blocks_publication():
    from us_quant import ssd_retained_pair_equal_allocation_v1 as study

    context = {
        **study.template.cohort_publication.__globals__,
        **study.__dict__,
        "base": SimpleNamespace(JOURNAL=None, journal_events=lambda path: []),
        "events": lambda kind, name: [{"sha256": "registration"}] if kind == "registered" else [],
        "read_json": lambda path: {"complete_fixed_cohort_ids": [], "cohort_registrations": []},
        "scoped": lambda *args: pytest.fail("No publication verifier or funding may run."),
    }
    checked = FunctionType(study.template.cohort_publication.__code__, context)
    with pytest.raises(QuantError):
        checked({"study_id": "new"}, "remotecommit")


def test_conflicting_registered_identity_is_rejected_before_append(monkeypatch):
    from us_quant import ssd_retained_pair_equal_allocation_v1 as study

    monkeypatch.setattr(study, "bound", lambda function: lambda: (None, None, {"frozen": 1}, None))
    monkeypatch.setattr(study, "events", lambda kind, name: [{"body": {"foreign": True}}])
    with pytest.raises(QuantError):
        study.inputs()


def test_new_dependency_closure_and_original_accounting_helpers_remain_bound():
    from us_quant import ssd_retained_pair_equal_allocation_v1 as study

    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/expert_initial_monthly_execution_v1.py" in paths
    assert study.financial is study.template.financial


def test_real_saved_advisor_schema_is_bound_without_deriving_new_targets():
    from us_quant import ssd_retained_pair_equal_allocation_v1 as study

    saved = study.catalogue()
    with study.base.checked_reference(saved["source_tape"]).open() as stream:
        before, after = json.loads(stream.readline()), json.loads(stream.readline())
    assert before["record"] == "decision_input"
    assert after["record"] == "decision_resolved"
    assert tuple(row["id"] for row in before["observations"]) == rules.EXPERT_IDS
    assert set(after["combined_targets"]) == set(saved["source_candidate_ids"])
    assert "previous_weights_binary64_hex" in before
