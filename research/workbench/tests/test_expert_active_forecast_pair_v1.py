import copy
import math
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import expert_active_forecast_pair_v1 as rules
from us_quant import ssd_expert_active_forecast_pair_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def records():
    experts, calendar, synthetic_coordinates = archived_experts()
    coordinate_map = dict(zip(synthetic_coordinates, ("MTUM", "VLUE", "QUAL", "USMV"), strict=True))
    padded = []
    for expert in experts:
        targets = expert._targets.rename(columns=coordinate_map).reindex(
            columns=rules.ASSETS, fill_value=0.0
        )
        targets.loc[expert._targets.isna().all(axis=1)] = np.nan
        padded.append(type(expert)(expert.identity, targets, expert._shadow, rules.ASSETS))
    tape = []
    rules.original.build_targets(
        tuple(padded),
        calendar,
        rules.ASSETS,
        read_json(rules.original.POLICY),
        retain=tape.append,
    )
    result = []
    for i, (before, after) in enumerate(zip(tape[::2], tape[1::2], strict=True)):
        mature = all(o["score_mature"] for o in before["observations"])
        predictors = {
            rules.MODELS[0]: [
                0.00005 * (k + 1) + 0.00002 * math.cos(i * (k + 1)) - b * 0.00005
                for k in range(8)
                for b in range(2)
            ],
            rules.MODELS[1]: [
                0.00004 * (8 - k) + 0.00003 * math.sin(i * (k + 1)) - b * 0.00005
                for k in range(8)
                for b in range(2)
            ],
        }
        result.append(
            {
                "source_before": before,
                "source_resolved": after,
                "forecast_before": {
                    "record": "forecast_input",
                    "index": i,
                    "as_of": before["as_of"],
                    "information_clock": rules.CLOCK,
                    "binary64_values_truncated_to_fixed_decimals": False,
                    "observations": [
                        {
                            **copy.deepcopy(
                                {k: v for k, v in o.items() if k not in ("target", "asset_order")}
                            ),
                            "target_available": True,
                        }
                        for o in before["observations"]
                    ],
                },
                "forecast_resolved": {
                    "record": "forecast_resolved",
                    "index": i,
                    "audit": {
                        "as_of": before["as_of"],
                        "forecast_available": mature,
                        "predictions": predictors if mature else None,
                        "forecast_is_main_account_return_or_qualification": False,
                    },
                },
            }
        )
    return result


def new_state():
    return {n: np.full(8, 0.125) for n in rules.IDS}


def allocations(records):
    state, targets, audits = new_state(), [], []
    policy = read_json(rules.POLICY)
    for row in records:
        target, audit = rules.allocate(row, state, policy)
        proof = rules.verify(row, state, target, audit, policy)
        assert proof["maximum_independent_score_difference"] <= 1e-10
        targets.append(target)
        audits.append(audit)
        state = {n: np.asarray(audit["candidates"][n]["expert_weights"]) for n in rules.IDS}
    return targets, audits


def test_frozen_two_model_cohort_is_not_a_forecaster_or_expert_promotion():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert (
        policy["new_Ridge_fits"] == 0 and not policy["forecaster_is_an_additional_strategy_expert"]
    )
    assert policy["candidate_count_in_this_study"] == 2
    assert rules.MODELS == ("unconditional_unit_ridge", "contextual_unit_ridge")


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("all_pairs_per_mature_decision", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("new_Ridge_fits", 1),
        ("new_source_RF_views", 1),
        ("model_selected_from_complete_forecast_errors", True),
        ("independent_score_tolerance", 1e-8),
        ("ranking_risk_windows_are_independent_predictor_validation", True),
    ],
)
def test_no_window_cost_source_parameter_or_qualification_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_all_dates_null_warmup_and_all_twenty_eight_risk_pairs(records):
    targets, audits = allocations(records)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert sum(not a["candidates"][rules.IDS[0]]["scores_available"] for a in audits) == 7
    for row, target, audit in zip(records, targets, audits, strict=True):
        for name in rules.IDS:
            candidate = audit["candidates"][name]
            assert abs(sum(target[name]) - 0.98) < 1e-12
            if candidate["scores_available"]:
                assert len(candidate["pairs"]) == 28
                assert sum(w > 0 for w in candidate["expert_weights"]) == 2
            else:
                np.testing.assert_array_equal(target[name], rules.resolved_equal(row))
                assert not candidate["safety_or_success_claimed"]


def test_independent_checks_every_nonselected_score_and_marginal_risk(records):
    row = next(r for r in records if r["forecast_resolved"]["audit"]["forecast_available"])
    previous = new_state()
    targets, audit = rules.allocate(row, previous, read_json(rules.POLICY))
    proof = rules.verify(row, previous, targets, audit, read_json(rules.POLICY))
    assert proof["maximum_independent_target_difference"] < 1e-12
    for kind in (
        "forecast_risk_scores",
        "source_training_marginal_variance_contributions",
        "selection_score_key",
    ):
        damaged = copy.deepcopy(audit)
        candidate = damaged["candidates"][rules.IDS[1]]
        other = next(p for p in candidate["pairs"] if p["pair"] != candidate["selected_pair"])
        if kind == "forecast_risk_scores":
            other[kind]["SPY"]["training_risk_reference"] += 0.1
        elif kind == "selection_score_key":
            other[kind] += 0.1
        else:
            other[kind][0] += 0.1
        with pytest.raises(QuantError):
            rules.verify(row, previous, targets, damaged, read_json(rules.POLICY))


def test_own_previous_allocation_is_not_original_advisor_or_other_policy_state(records):
    row = next(r for r in records if r["forecast_resolved"]["audit"]["forecast_available"])
    a = new_state()
    b = new_state()
    b[rules.IDS[0]] = np.array([1.0] + [0.0] * 7)
    _, first = rules.allocate(row, a, read_json(rules.POLICY))
    targets, later = rules.allocate(row, b, read_json(rules.POLICY))
    assert first["candidates"][rules.IDS[1]] == later["candidates"][rules.IDS[1]]
    assert first["candidates"][rules.IDS[0]]["pairs"] != later["candidates"][rules.IDS[0]]["pairs"]
    rules.verify(row, b, targets, later, read_json(rules.POLICY))


def test_future_saved_forecasts_and_opinions_preserve_exact_prefix(records):
    cutoff = next(i for i, r in enumerate(records) if r["source_before"]["as_of"] == "2017-04-28")
    changed = copy.deepcopy(records)
    for row in changed[cutoff + 1 :]:
        for model in rules.MODELS:
            row["forecast_resolved"]["audit"]["predictions"][model] = [0.0001] * 16
        for o in row["source_before"]["observations"]:
            o["target"] = [0.98 if s == "BIL" else 0.0 for s in rules.ASSETS]
    original_targets, original_audits = allocations(records)
    changed_targets, changed_audits = allocations(changed)
    prefix_targets, prefix_audits = allocations(records[: cutoff + 1])
    assert original_audits[: cutoff + 1] == changed_audits[: cutoff + 1] == prefix_audits
    for a, b, c in zip(original_targets, changed_targets, prefix_targets, strict=False):
        for name in rules.IDS:
            np.testing.assert_array_equal(a[name], b[name])
            np.testing.assert_array_equal(a[name], c[name])


def test_negative_forecasts_rank_without_a_new_cash_cutoff(records):
    row = copy.deepcopy(
        next(r for r in records if r["forecast_resolved"]["audit"]["forecast_available"])
    )
    for model in rules.MODELS:
        row["forecast_resolved"]["audit"]["predictions"][model] = [-0.001] * 16
    targets, audit = rules.allocate(row, new_state(), read_json(rules.POLICY))
    for name in rules.IDS:
        assert audit["candidates"][name]["scores_available"]
        assert audit["candidates"][name]["selected_pair"] is not None
    rules.verify(row, new_state(), targets, audit, read_json(rules.POLICY))


def test_zero_score_keys_keep_lexical_ties_without_adding_a_safe_expert(records):
    row = copy.deepcopy(
        next(r for r in records if r["forecast_resolved"]["audit"]["forecast_available"])
    )
    opinion = [0.98 if s == "MTUM" else 0.0 for s in rules.ASSETS]
    for o in row["source_before"]["observations"]:
        o["target"] = opinion.copy()
    for model in rules.MODELS:
        row["forecast_resolved"]["audit"]["predictions"][model] = [0.0] * 16
    targets, audit = rules.allocate(row, new_state(), read_json(rules.POLICY))
    for name in rules.IDS:
        candidate = audit["candidates"][name]
        assert candidate["selected_pair"] == min(p["pair"] for p in candidate["pairs"])
        assert all(p["selection_score_key"] == 0 for p in candidate["pairs"])
        assert all(abs(p["selection_score"]) < 1e-12 for p in candidate["pairs"])
    rules.verify(row, new_state(), targets, audit, read_json(rules.POLICY))


@pytest.mark.parametrize("damage", ["future", "missing", "NET", "RF", "identity", "menu", "coerce"])
def test_bad_saved_join_is_never_filled_or_relabelled_unscored(records, damage):
    row = copy.deepcopy(
        next(r for r in records if r["forecast_resolved"]["audit"]["forecast_available"])
    )
    if damage == "future":
        row["forecast_resolved"]["audit"]["as_of"] = "2026-10-01"
    elif damage == "missing":
        row["forecast_resolved"]["audit"]["predictions"] = None
    elif damage in ("NET", "RF"):
        field = "NET_returns" if damage == "NET" else "RF"
        row["forecast_before"]["observations"][-1][field][-1] += 0.001
    elif damage == "identity":
        row["forecast_before"]["observations"][-1]["id"] = "foreign"
    elif damage == "menu":
        row["source_resolved"]["audit"]["selection"]["pairs"].pop()
    else:
        row["source_resolved"]["audit"]["selection"]["pairs"][0]["expert_weights"] = ["0.125"] * 8
    with pytest.raises(QuantError):
        rules.allocate(row, new_state(), read_json(rules.POLICY))


def test_both_registrations_precede_either_allocation(monkeypatch):
    history, packets = [], []
    frozen = {"study_id": "synthetic_saved_forecast"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": str(len(history))}
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
    assert all(p["complete_fixed_cohort_ids"] == list(study.IDS) for p in packets)


def test_foreign_last_identity_blocks_first_append(monkeypatch):
    monkeypatch.setattr(study, "bound", lambda function: lambda: (None, None, {"frozen": 1}, None))
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [{"body": {"foreign": True}}] if name == study.IDS[1] else [],
    )
    with pytest.raises(QuantError):
        study.inputs()


def test_missing_second_complete_packet_blocks_any_funding():
    first, second = study.IDS
    packet = {"complete_fixed_cohort_ids": list(study.IDS), "cohort_registrations": ["r0", "r1"]}

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
        checked({"study_id": "synthetic"}, "remotecommit")


def test_source_and_fixture_dependency_closure_is_bound_without_refits():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_expert_target_consensus_v1.py" in paths
    assert not hasattr(rules, "ridge") and not hasattr(rules, "analyze")
