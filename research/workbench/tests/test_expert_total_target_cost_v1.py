import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_commitment_macro_defense_hedge_v1 import sources

from us_quant import expert_total_target_cost_v1 as rules
from us_quant import ssd_expert_total_target_cost_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def target(symbol):
    row = pd.Series(0.0, index=ASSETS)
    row[symbol] = 0.98
    return row


def test_unchanged_fractions_can_have_cross_origin_opinion_cost():
    old, today = target("SPY"), target("QQQ")
    assert rules.nominal_cost(today, today) == (0.0, 0)
    cost, changed = rules.nominal_cost(today, old)
    assert changed == 2 and cost == pytest.approx(1.96 * 5 / 10000 + 2 / 10000)


def test_same_desired_target_can_cost_zero_despite_a_different_reference():
    today = (target("SPY") + target("QQQ")) / 2
    assert rules.nominal_cost(today, today) == (0.0, 0)
    assert rules.nominal_cost(today, target("SPY"))[0] > 0


@pytest.mark.parametrize("bad", ["order", "budget", "negative", "nan"])
def test_invalid_nominal_reference_is_not_actual_or_neutral_holdings(bad):
    old, today = target("SPY"), target("QQQ")
    if bad == "order":
        old = old.iloc[::-1]
    elif bad == "budget":
        old.iloc[0] += 0.01
    elif bad == "negative":
        old.iloc[0], old.iloc[1] = -0.01, 0.99
    else:
        old.iloc[0] = np.nan
    with pytest.raises(QuantError):
        rules.nominal_cost(today, old)


def test_pair_risk_kernel_unchanged_and_both_cost_scopes_retained():
    experts, _, _ = sources()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    previous = np.full(4, 0.25)
    prior_target = rules.combine(
        [e.observe(pd.Timestamp("2017-03-31"), 126) for e in experts], previous
    )
    _, old = rules.prior.choose_pair(observed, previous)
    _, new = rules.choose_pair(observed, previous, prior_target)
    assert len(new["pairs"]) == 6
    for before, after in zip(old["pairs"], new["pairs"], strict=True):
        assert before["expert_weights"] == after["expert_weights"]
        assert (
            before["training_marginal_variance_contributions"]
            == (after["training_marginal_variance_contributions"])
        )
        assert (
            after["same_origin_incremental_cost_fraction_reference"]
            == (before["estimated_incremental_switch_cost_fraction"])
        )
        assert after["cost_proxy_is_actual_main_account_cost"] is False
    with pytest.raises(QuantError):
        rules.choose_pair(observed, previous, None)


def test_singleton_full_dates_nominal_state_and_unscored_origins():
    experts, calendar, assets = sources()
    targets, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert set(targets) == {rules.ID} and targets[rules.ID].index.equals(calendar)
    assert audits[0]["previous_nominal_meta_target"] is None
    assert sum(not a["selection"]["scores_available"] for a in audits) == 7
    for i, audit in enumerate(audits[1:], 1):
        assert audit["previous_nominal_meta_target"] == (
            targets[rules.ID].loc[audits[i - 1]["decision_session"]].tolist()
        )
        assert not audit["previous_nominal_target_is_actual_holdings"]


def test_future_mutation_and_prefix_do_not_change_nominal_memory():
    cutoff = pd.Timestamp("2017-04-28")
    experts, calendar, assets = sources()
    changed, _, _ = sources(changed_after=cutoff)
    first, a = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    second, b = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, c = rules.build_targets(
        experts, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(first[rules.ID].loc[:cutoff], second[rules.ID].loc[:cutoff])
    pd.testing.assert_frame_equal(first[rules.ID].loc[:cutoff], early[rules.ID])
    assert c == [x for x in a if x["decision_session"] <= str(cutoff.date())]
    assert c == [x for x in b if x["decision_session"] <= str(cutoff.date())]


def test_every_pair_score_cost_and_record_checked_independently(tmp_path):
    experts, calendar, assets = sources()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    check = study.verify_targets(data, experts, targets, audits)
    assert check["maximum_independent_total_nominal_cost_difference"] <= 1e-10
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert study.verify_retained_records(path, experts, targets, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    mature = next(a for a in damaged if a["selection"]["scores_available"])
    unused = next(
        p for p in mature["selection"]["pairs"] if p["pair"] != mature["selection"]["selected_pair"]
    )
    unused["estimated_total_nominal_target_cost_fraction"] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, damaged)
    rows[2]["previous_nominal_target"][0] += 0.01
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, targets, audits)


@pytest.mark.parametrize(
    "key,value",
    [
        ("reference_capital_usd", 20000),
        ("switch_cost_bps_per_side", 20.0),
        ("switch_commission_per_changed_asset_usd", 0.0),
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("actual_weight_trade_band", 0.05),
        ("cost_proxy_is_actual_holdings_NAV_or_main_fees", True),
        ("cached_equal_and_rolling_controls_replayed", True),
        ("candidate_count_in_this_study", 2),
    ],
)
def test_no_coefficient_window_funding_or_cached_control_retune(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_singleton_registration_precedes_any_new_target(monkeypatch, tmp_path):
    history, packets = [], []
    frozen = {"study_id": "synthetic_singleton", "declared_financial_cohort_size": 1}

    def events(kind, name):
        return [e for e in history if e["kind"] == kind and e["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {
                "kind": kind,
                "candidate_id": name,
                "body": body,
                "sha256": "event" + str(len(history)),
            }
        )

    def prepare(data, experts, actual):
        assert actual == frozen and len(events("registered", rules.ID)) == 1
        return {
            rules.ID: {
                "targets": {"path": "toy", "sha256": "a"},
                "proof": {"path": "proof", "sha256": "b"},
            }
        }

    def retain(path, packet, *args):
        assert packet["complete_fixed_cohort_ids"] == [rules.ID]
        assert len(packet["cohort_registrations"]) == 1
        packets.append(packet)

    function = study.bound(
        study.template.register,
        lifecycle=SimpleNamespace(runtime_guard=lambda: None, research_job=nullcontext),
        inputs=lambda: (None, None, frozen, None),
        events=events,
        prepare=prepare,
        base=SimpleNamespace(JOURNAL=tmp_path, append_event=append),
        seven=SimpleNamespace(
            common=SimpleNamespace(
                packet_entry=lambda candidate, actual: {"specification": {"frozen": actual}},
                retain_receipt=retain,
            )
        ),
        progress=lambda *args, **kwargs: None,
        reference=lambda path: {"path": str(path)},
    )
    function()
    assert [e["kind"] for e in history] == ["registered", "target_proof"]
    assert len(packets) == 1
