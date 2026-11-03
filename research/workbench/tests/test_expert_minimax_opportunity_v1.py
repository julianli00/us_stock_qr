import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_minimax_opportunity_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation

ASSETS = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")


class FakeExpert:
    def __init__(self, number, calendar):
        self.identity = ExpertIdentity(
            rules.EXPERT_IDS[number],
            rules.MECHANISMS[number],
            "2026-11-03T00:00:00Z",
        )
        self.returns = pd.Series(0.0001 * (number + 1), index=calendar[1:])
        self.rf = pd.Series(0.00001, index=calendar[1:])
        self.target = pd.Series(0.0, index=ASSETS)
        self.target.iloc[number] = 0.98

    def observe(self, as_of, lookback):
        net, rf = self.returns.loc[:as_of].tail(lookback), self.rf.loc[:as_of].tail(lookback)
        return ExpertObservation(
            self.identity,
            as_of,
            self.target.copy(),
            as_of,
            net.copy(),
            rf.copy(),
            len(net) == lookback,
        )


def examples(end="2017-03-31"):
    calendar = sessions("2016-10-05", end)
    return calendar, tuple(FakeExpert(i, calendar) for i in range(8))


def policy():
    return read_json(rules.POLICY)


def opinions(same=False):
    return np.tile([0.98, 0.0], (8, 1)) if same else np.eye(8) * 0.98


def test_uniform_months_use_explicit_lexical_ties_and_retain_all_pairs():
    weights, audit = rules.choose_pair(np.ones((3, 8)), opinions(True), np.full(8, 0.125))
    expected = sorted(rules.EXPERT_IDS)[:2]
    assert audit["selected_pair"] == expected
    assert len(audit["pair_records"]) == 28
    assert set(itertools.combinations(sorted(rules.EXPERT_IDS), 2)) == {
        tuple(row["pair"]) for row in audit["pair_records"]
    }
    assert sorted(weights) == [0] * 6 + [0.5, 0.5]
    assert all(row["score_key_12dp"] == 0 for row in audit["pair_records"])


def test_worst_month_opportunity_differs_from_choosing_one_source_winner():
    relatives = np.ones((2, 8))
    relatives[:, :2] = [[1.6, 0.6], [0.6, 1.6]]
    weights, audit = rules.choose_pair(relatives, opinions(True), np.full(8, 0.125))
    assert audit["selected_pair"] == sorted(rules.EXPERT_IDS[:2])
    assert np.array_equal(weights, [0.5, 0.5, 0, 0, 0, 0, 0, 0])
    assert math.isclose(
        min(row["worst_opportunity_loss"] for row in audit["pair_records"]),
        math.log(1.6 / 1.1),
        rel_tol=0,
        abs_tol=1e-15,
    )


def test_all_pair_scores_match_separate_scalar_month_and_asset_loops():
    relatives = np.array(
        [
            [0.8, 1.1, 0.9, 1.2, 1.0, 1.3, 0.95, 1.02],
            [1.4, 0.9, 1.1, 0.7, 1.05, 0.95, 1.01, 1.2],
        ]
    )
    previous = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    _, audit = rules.choose_pair(relatives, opinions(), previous)
    for row in audit["pair_records"]:
        indices = [rules.EXPERT_IDS.index(name) for name in row["pair"]]
        losses = [
            math.log(max(month) / ((month[indices[0]] + month[indices[1]]) / 2))
            for month in relatives.tolist()
        ]
        weights = [0.5 if i in indices else 0 for i in range(8)]
        change = [
            math.fsum((weights[i] - previous[i]) * opinions()[i, j] for i in range(8))
            for j in range(8)
        ]
        fee = 5 * math.fsum(abs(x) for x in change) + sum(abs(x) > 1e-12 for x in change)
        expected = max(losses) - math.log1p(-fee / 10000)
        assert row["monthly_opportunity_losses"] == pytest.approx(losses, abs=1e-15)
        assert row["switch_cost_proxy_usd"] == pytest.approx(fee, abs=1e-12)
        assert row["raw_score"] == pytest.approx(expected, abs=1e-12)
        assert row["score_key_12dp"] == round(expected, 12)


def test_one_time_switch_penalty_prefers_unchanged_pair_on_equal_assessments():
    previous = np.zeros(8)
    previous[[2, 4]] = 0.5
    weights, audit = rules.choose_pair(np.ones((1, 8)), opinions(), previous)
    assert np.array_equal(weights, previous)
    selected = next(row for row in audit["pair_records"] if row["pair"] == audit["selected_pair"])
    assert selected["switch_cost_proxy_usd"] == 0
    assert selected["one_time_switch_log_penalty"] == 0
    assert any(row["switch_cost_proxy_usd"] > 0 for row in audit["pair_records"])


def test_common_positive_month_scaling_does_not_change_opportunity_order():
    relatives = np.array(
        [
            [0.8, 1.1, 0.9, 1.2, 1.0, 1.3, 0.95, 1.02],
            [1.4, 0.9, 1.1, 0.7, 1.05, 0.95, 1.01, 1.2],
        ]
    )
    original, audit = rules.choose_pair(relatives, opinions(), np.full(8, 0.125))
    scaled, check = rules.choose_pair(
        relatives * np.array([[0.7], [1.3]]), opinions(), np.full(8, 0.125)
    )
    assert np.array_equal(original, scaled)
    assert [r["score_key_12dp"] for r in audit["pair_records"]] == [
        r["score_key_12dp"] for r in check["pair_records"]
    ]


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((1, 7)),
        np.zeros((1, 8)),
        np.full((1, 8), np.nan),
        np.ones((120, 8)),
        np.full((1, 8), np.nextafter(0.0, 1.0)),
    ],
)
def test_invalid_or_unrepresentable_months_do_not_become_safe_scores(bad):
    with pytest.raises(QuantError):
        rules.choose_pair(bad, opinions(), np.full(8, 0.125))


def test_empty_history_is_unscored_but_invalid_opinions_are_not_immaturity():
    weights, audit = rules.choose_pair(np.empty((0, 8)), opinions(), np.full(8, 0.125))
    assert np.array_equal(weights, np.full(8, 0.125))
    assert audit["score_mature"] is False
    assert audit["pair_records"] == []
    with pytest.raises(QuantError):
        rules.choose_pair(np.empty((0, 8)), np.zeros((8, 8)), weights)


def test_true_month_clock_excludes_initial_and_terminal_partial_intervals():
    calendar, experts = examples("2017-01-05")
    target, audits = rules.build_targets(experts, calendar, ASSETS, policy())
    assert [a["decision_session"] for a in audits] == [
        "2016-10-05",
        "2016-10-31",
        "2016-11-30",
        "2016-12-30",
    ]
    assert [a["selection"]["score_mature"] for a in audits] == [False, False, True, True]
    assert [len(a["completed_intervals"]) for a in audits] == [0, 0, 1, 2]
    assert audits[2]["completed_intervals"] == [["2016-10-31", "2016-11-30"]]
    assert len(audits[2]["selection"]["pair_records"]) == 28
    assert target.loc["2017-01-03":].isna().all().all()
    assert np.allclose(target.dropna().sum(axis=1), 0.98, rtol=0, atol=1e-12)


def test_prefix_truncation_and_future_poison_preserve_earlier_decisions():
    calendar, experts = examples()
    full, audits = rules.build_targets(experts, calendar, ASSETS, policy())
    for day in ("2016-10-05", "2016-10-31", "2016-11-30", "2016-12-30", "2017-02-28"):
        end = pd.Timestamp(day)
        prefix, earlier = rules.build_targets(experts, calendar[calendar <= end], ASSETS, policy())
        assert prefix.equals(full.loc[:end])
        assert earlier == [a for a in audits if a["decision_session"] <= day]
    for expert in experts:
        expert.returns.loc["2017-01-03":] = -0.5
        expert.rf.loc["2017-01-03":] = 0.1
    changed, later = rules.build_targets(experts, calendar, ASSETS, policy())
    assert changed.loc[:"2016-12-30"].equals(full.loc[:"2016-12-30"])
    assert later[:4] == audits[:4]


def test_exact_source_inputs_are_retained_before_missing_month_failure():
    calendar, experts = examples("2016-11-30")
    experts[0].returns = experts[0].returns.drop(pd.Timestamp("2016-11-15"))
    retained = []
    with pytest.raises(QuantError, match="native NET"):
        rules.build_targets(experts, calendar, ASSETS, policy(), retain=retained.append)
    assert retained[-1]["record"] == "decision_input"
    assert retained[-1]["as_of"] == "2016-11-30"
    assert "2016-11-15" not in retained[-1]["observations"][0]["new_NET_dates"]
    assert retained[-1]["observations"][0]["new_NET_binary64_hex"]
    assert retained[-2]["record"] == "decision_resolved"


def test_rf_exactness_is_not_replaced_by_a_numeric_tolerance():
    calendar, experts = examples("2016-11-30")
    experts[0].rf.iloc[0] += 1e-16
    with pytest.raises(QuantError, match="exact common RF"):
        rules.build_targets(experts, calendar, ASSETS, policy())


def test_policy_rejects_parameter_change_and_success_claims():
    for key, value in (("minimum_completed_months", 2), ("minimum_completed_months", True)):
        changed = copy.deepcopy(policy())
        changed[key] = value
        with pytest.raises(QuantError):
            rules.validate(changed)
    changed = copy.deepcopy(policy())
    changed["methodology"]["score_is_actual_main_net_wealth"] = True
    with pytest.raises(QuantError):
        rules.validate(changed)
