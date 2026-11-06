import copy
import json
import math
from contextlib import nullcontext
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_growth_macro_defense_hedge_v1 import MappedSyntheticExpert
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import quintet_anytime_hedge_v1 as rules
from us_quant import ssd_quintet_anytime_hedge_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def sources(changed_after=None):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, ASSETS[: len(toy_assets)], strict=True))
    chosen = tuple(MappedSyntheticExpert(eight[i], mapping) for i in rules.pool.SLOTS)
    return chosen, calendar, ASSETS


def test_fixed_pool_and_initial_weights():
    experts, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS == study.prior.rules.EXPERT_IDS
    np.testing.assert_array_equal(rules.AnytimeHedge().weights(), np.full(5, 0.2))
    assert rules.CANDIDATES[0]["id"] == study.IDS[0] == rules.ID


def test_bounded_losses_equal_month_and_binary_tiny_spread():
    np.testing.assert_array_equal(rules.bounded_losses([0.3] * 5), np.zeros(5))
    row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(rules.bounded_losses(row), [1, 0.75, 0.5, 0.25, 0])
    near = np.array([0.1] * 5)
    near[-1] = np.nextafter(near[-1], np.inf)
    np.testing.assert_array_equal(rules.bounded_losses(near), [1, 1, 1, 1, 0])


def test_rate_uses_next_round_not_final_horizon_and_neutral_month_advances_clock():
    state = rules.AnytimeHedge()
    first, audit = state.update([-0.01, 0.03, 0.02, -0.02, 0.01])
    assert audit["learning_round_after_assimilation"] == 2
    assert audit["learning_rate_after_assimilation"] == 2 * math.sqrt(math.log(5) / 2)
    old_loss = state.loss.copy()
    after, second = state.update([0.1] * 5)
    np.testing.assert_array_equal(state.loss, old_loss)
    assert state.periods == 2 and second["learning_round_after_assimilation"] == 3
    assert max(after) < max(first)


def test_all_119_updates_match_independent_decimal_weights_and_bounded_regret():
    history, state = [], rules.AnytimeHedge()
    values = np.random.default_rng(42).normal(0, 0.05, size=(119, 5))
    for row in values:
        before = state.weights()
        history.append(row.tolist())
        weights, audit = state.update(row)
        assert audit["prequential_weights_before_assimilation"] == before.tolist()
        assert audit["bounded_loss_regret"] <= audit["bounded_loss_reference_bound"] + 1e-12
        if len(history) in (1, 2, 60, 119):
            np.testing.assert_allclose(
                weights, rules.independent_weights(history), rtol=0, atol=1e-12
            )
    with pytest.raises(QuantError):
        state.update([0] * 5)


@pytest.mark.parametrize("bad", [[1, 2], [0, 0, np.nan, 0, 0], [0, np.inf, 0, 0, 0]])
def test_invalid_losses_are_not_neutral(bad):
    with pytest.raises(QuantError):
        rules.bounded_losses(bad)


def test_invalid_state_does_not_become_equal_weights():
    state = rules.AnytimeHedge()
    state.loss[0] = -1
    with pytest.raises(QuantError):
        state.weights()


def test_complete_month_maturity_excludes_initial_partial_month():
    experts, calendar, assets = sources()
    target, audits, periods = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY)
    )
    assert target.index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[1]["decision_session"] == "2016-10-31"
    assert audits[0]["score_mature"] is audits[1]["score_mature"] is False
    assert audits[2]["decision_session"] == "2016-11-30"
    assert audits[2]["complete_learning_intervals"] == 1
    assert periods[0]["start_after"] == "2016-10-31"
    assert audits[0]["risk_and_cost"]["net_excess_covariance"] is None
    assert audits[2]["risk_and_cost"]["net_excess_covariance"] is not None


def test_future_source_values_and_calendar_suffix_do_not_change_prefix():
    cutoff = pd.Timestamp("2017-04-28")
    experts, calendar, assets = sources()
    future, _, _ = sources(changed_after=cutoff)
    first, a, p = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    changed, b, q = rules.build_targets(future, calendar, assets, read_json(rules.POLICY))
    early, c, r = rules.build_targets(
        experts, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(first.loc[:cutoff], changed.loc[:cutoff])
    pd.testing.assert_frame_equal(first.loc[:cutoff], early)
    assert c == [row for row in a if row["decision_session"] <= str(cutoff.date())]
    assert c == [row for row in b if row["decision_session"] <= str(cutoff.date())]
    assert r == [row for row in p if row["end"] <= str(cutoff.date())]
    assert r == [row for row in q if row["end"] <= str(cutoff.date())]


def test_independent_target_risk_cost_and_retained_binary_inputs(tmp_path):
    experts, calendar, assets = sources()
    records = []
    target, audits, periods = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    verified = study.verify_targets(data, experts, {rules.ID: target}, audits, periods)
    assert verified["maximum_independent_expert_weight_difference"] <= 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert study.verify_retained_records(path, experts, {rules.ID: target}, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    damaged[-1]["expert_weights"][0] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, {rules.ID: target}, damaged, periods)
    records[-2]["matured_NET_returns_hex"][0][0] = 0.9.hex()
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, {rules.ID: target}, audits)


def test_missing_source_opinion_is_error_not_silent_pool_reduction():
    experts, calendar, assets = sources()
    last = experts[-1]
    bad = SimpleNamespace(
        identity=last.identity,
        observe=lambda day, n: replace(last.observe(day, n), target=None, target_known_at=None),
    )
    with pytest.raises(QuantError):
        rules.build_targets((*experts[:-1], bad), calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate_coefficient", 3),
        ("learning_resets_for_five_year_account", True),
        ("initial_partial_source_month_assimilated", True),
        ("neutral_month_advances_learning_round", False),
        ("incremental_switch_proxy_affects_allocation", True),
        ("new_shadow_account_runs", 1),
        ("new_source_RF_views", 1),
        ("normalized_regret_is_actual_main_wealth", True),
        ("mechanism_labels_are_independent_risks", True),
        ("actual_weight_trade_band", 0.05),
        ("independent_numeric_tolerance", 1e-8),
    ],
)
def test_frozen_law_has_no_rate_pool_clock_or_financial_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_actual_singleton_registration_precedes_all_new_learning(monkeypatch):
    history, packets = [], []
    frozen = {"study_id": "synthetic_hedge"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {
                "kind": kind,
                "candidate_id": name,
                "body": body,
                "sha256": f"event{len(history)}",
            }
        )

    def prepare(data, experts, actual):
        assert actual == frozen and len(events("registered", rules.ID)) == 1
        return {rules.ID: {"targets": {"path": "target"}, "proof": {"path": "proof"}}}

    def retain(path, packet, *args):
        assert packet["complete_fixed_cohort_ids"] == [rules.ID]
        assert len(packet["cohort_registrations"]) == 1
        packets.append(packet)

    common = SimpleNamespace(
        packet_entry=lambda c, f: {"candidate_id": c["id"]}, retain_receipt=retain
    )

    def bind(function):
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
                "reference": lambda p: {"path": str(p), "sha256": "x"},
                "progress": lambda *a, **k: None,
            },
            argdefs=function.__defaults__,
        )

    monkeypatch.setattr(study, "bound", bind)
    study.register()
    assert history[0]["kind"] == "registered" and len(packets) == 1
