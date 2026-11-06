import copy
import json
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_quintet_anytime_hedge_v1 as prior_tests
from test_growth_macro_defense_hedge_v1 import MappedSyntheticExpert
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import quintet_adaptive_hedge_v1 as rules
from us_quant import ssd_quintet_adaptive_hedge_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def sources(changed_after=None):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, ASSETS[: len(toy_assets)], strict=True))
    experts = tuple(MappedSyntheticExpert(eight[i], mapping) for i in rules.prior.pool.SLOTS)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    cached = []
    for i, day in enumerate(dates[2:], start=2):
        observed = rules.observations(experts, day)
        logs, _ = rules.prior.period_logs(observed, dates[i - 1], day)
        cached.append(
            {
                "start_after": str(dates[i - 1].date()),
                "end": str(day.date()),
                "canonical_log_relatives": logs.tolist(),
                "losses": rules.prior.bounded_losses(logs).tolist(),
            }
        )
    return experts, cached, calendar, ASSETS


def test_fixed_source_pool_and_analytic_initial_rate():
    experts, _, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS == study.prior.rules.EXPERT_IDS
    assert rules.CANDIDATES[0]["id"] == rules.ID == study.IDS[0]
    state = rules.AdaHedge()
    assert state.rate() is None
    np.testing.assert_array_equal(state.weights(), np.full(5, 0.2))


def test_infinite_mix_loss_is_cumulative_minimum_difference_not_support_minimum():
    before = (Fraction(0), Fraction(1, 10), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5))
    after = tuple(a + b for a, b in zip(before, (1, 0, 1, 1, 1), strict=True))
    weights, old_mix = rules.mix(before, None)
    _, new_mix = rules.mix(after, None)
    assert weights == [1, 0, 0, 0, 0]
    assert new_mix - old_mix == Decimal("0.1")


def test_first_gap_and_rate_follow_original_equations_without_coefficient():
    state = rules.AdaHedge()
    weights, audit = state.update([0, 0.25, 0.5, 0.75, 1])
    assert audit["learning_rate_before_is_infinite"] is True
    assert Decimal(audit["played_loss_decimal"]) == Decimal("0.5")
    assert Decimal(audit["mix_loss_decimal"]) == 0
    assert state.gap == Decimal("0.5")
    assert float(state.rate()) == pytest.approx(2 * np.log(5), abs=1e-14)
    expected = np.exp(-2 * np.log(5) * np.array([0, 0.25, 0.5, 0.75, 1]))
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=0, atol=1e-12)


def test_neutral_round_advances_maturity_but_does_not_cool_or_soften_weights():
    state = rules.AdaHedge()
    state.update([0, 0.25, 0.5, 0.75, 1])
    before, rate, gap, snapshot = state.weights(), state.rate(), state.gap, state.snapshot()
    weights, audit = state.update([0] * 5)
    np.testing.assert_array_equal(weights, before)
    assert state.rate() == rate and state.gap == gap and state.periods == 2
    assert Decimal(audit["mixability_gap_increment_decimal"]) == 0
    assert state.snapshot()["cumulative_loss_fractions"] == snapshot["cumulative_loss_fractions"]


def test_all_119_updates_match_independent_direct_mixture_and_source_regret():
    state, history, actual = rules.AdaHedge(), [], []
    values = np.random.default_rng(42).uniform(0, 1, (119, 5))
    values[:, 0], values[:, 1] = 0, 1
    for row in values:
        weights, audit = state.update(row)
        history.append(row.tolist())
        actual.append((weights, audit))
        assert Decimal(audit["normalized_source_regret_decimal"]) <= (
            Decimal(audit["normalized_source_regret_reference_decimal"]) + Decimal("1e-12")
        )
    oracle = rules.independent_trace(history)
    for (weights, audit), row in zip(actual, oracle, strict=True):
        np.testing.assert_allclose(weights, row["weights_after"], rtol=0, atol=1e-12)
        assert float(Decimal(audit["mixability_gap_increment_decimal"])) >= 0
        assert float(Decimal(audit["state_after"]["cumulative_mixability_gap_decimal"])) == (
            pytest.approx(row["gap"], abs=1e-12)
        )
    with pytest.raises(QuantError):
        state.update([0, 0, 0, 0, 1])


def test_119_stable_leader_rounds_have_no_gap_clamp_or_artificial_weight_floor():
    state = rules.AdaHedge()
    for _ in range(119):
        state.update([0, 0, 0, 0, 1])
    precise, _ = rules.mix(state.losses, state.rate())
    assert precise[-1] > 0
    assert state.weights()[-1] == 0
    assert state.gap > 0 and state.periods == 119


@pytest.mark.parametrize(
    "bad",
    [
        [0, 1],
        [0.2] * 5,
        [0, 0.2, 0.4, 0.6, 0.8],
        [0, 0, -0.1, 0, 1],
        [0, 0, np.nan, 0, 1],
        [0, 0, np.inf, 0, 1],
    ],
)
def test_invalid_labels_are_not_neutral_or_rescaled(bad):
    with pytest.raises(QuantError):
        rules.AdaHedge().update(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("periods", -1),
        ("gap", Decimal("-1")),
        ("played", Decimal("NaN")),
        ("losses", (Fraction(-1),) * 5),
    ],
)
def test_invalid_adaptive_state_is_error_not_uniform_or_infinite(field, value):
    state = rules.AdaHedge()
    setattr(state, field, value)
    with pytest.raises(QuantError):
        state.weights()


def test_complete_month_clock_and_future_sources_labels_preserve_every_prefix():
    experts, cached, calendar, assets = sources()
    cutoff = pd.Timestamp("2017-04-28")
    future, future_cached, _, _ = sources(changed_after=cutoff)
    first, a, p = rules.build_targets(experts, cached, calendar, assets, read_json(rules.POLICY))
    changed, b, q = rules.build_targets(
        future, future_cached, calendar, assets, read_json(rules.POLICY)
    )
    mutated = copy.deepcopy(cached)
    for row in mutated:
        if row["end"] > str(cutoff.date()):
            row["losses"] = [np.nan] * 5
    early, c, r = rules.build_targets(
        experts, mutated, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(first.loc[:cutoff], changed.loc[:cutoff])
    pd.testing.assert_frame_equal(first.loc[:cutoff], early)
    assert c == [row for row in a if row["decision_session"] <= str(cutoff.date())]
    assert c == [row for row in b if row["decision_session"] <= str(cutoff.date())]
    assert r == [row for row in p if row["end"] <= str(cutoff.date())]
    assert r == [row for row in q if row["end"] <= str(cutoff.date())]
    assert [row["score_mature"] for row in a[:3]] == [False, False, True]
    assert p[0]["start_after"] == "2016-10-31" and p[0]["end"] == "2016-11-30"


def test_independent_weights_gap_cost_and_retained_binary_source_state(tmp_path):
    experts, cached, calendar, assets = sources()
    records = []
    target, audits, updates = rules.build_targets(
        experts, cached, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    verified = study.verify_targets(data, experts, cached, target, audits, updates)
    assert verified["maximum_independent_expert_weight_difference"] <= 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert study.verify_retained_records(path, experts, cached, target, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    damaged[-1]["expert_weights"][0] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, cached, target, damaged, updates)
    damaged = copy.deepcopy(audits)
    damaged[-1]["adaptive_state"]["cumulative_mixability_gap_decimal"] = "999"
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, cached, target, damaged, updates)
    damaged = copy.deepcopy(audits)
    damaged[-1]["risk_and_cost"]["marginal_variance_contributions"][0] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, cached, target, damaged, updates)
    records[-2]["cached_loss_labels_hex"][0] = (0.9).hex()
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, cached, target, audits)


@pytest.mark.parametrize("kind", ["missing", "label", "source_month", "clock", "RF", "opinion"])
def test_invalid_cached_label_or_complete_source_state_cannot_be_filled(kind):
    experts, cached, calendar, assets = sources()
    broken = copy.deepcopy(cached)
    if kind == "missing":
        broken = broken[1:]
    elif kind == "label":
        broken[0]["losses"][0] += 0.01
    elif kind == "source_month":
        broken[0]["canonical_log_relatives"] = [0, 0.25, 0.5, 0.75, 1]
        broken[0]["losses"] = rules.prior.bounded_losses(
            broken[0]["canonical_log_relatives"]
        ).tolist()
    elif kind == "clock":
        broken[0]["start_after"] = "2016-10-05"
    else:
        last = experts[-1]

        def observe(day, n):
            original = last.observe(day, n)
            if kind == "RF":
                return replace(original, risk_free=original.risk_free + 0.001)
            return replace(original, target=None, target_known_at=None)

        experts = (*experts[:-1], SimpleNamespace(identity=last.identity, observe=observe))
    with pytest.raises(QuantError):
        rules.build_targets(experts, broken, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", "2_sqrt_log5_over_t"),
        ("negative_Jensen_gap_clipped", True),
        ("mixability_gap_floor", 0.001),
        ("rate_or_window_sweep", True),
        ("learning_resets_for_five_year_account", True),
        ("neutral_month_advances_maturity_not_learning_rate", False),
        ("new_source_RF_views", 1),
        ("normalized_regret_is_actual_main_wealth", True),
        ("actual_weight_trade_band", 0.05),
        ("independent_numeric_tolerance", 1e-8),
    ],
)
def test_frozen_law_has_no_rate_gap_floor_clock_source_or_financial_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_actual_singleton_registration_precedes_all_new_adaptive_learning(monkeypatch):
    function = prior_tests.test_actual_singleton_registration_precedes_all_new_learning
    FunctionType(
        function.__code__,
        {**function.__globals__, "study": study, "rules": rules},
        argdefs=function.__defaults__,
    )(monkeypatch)
