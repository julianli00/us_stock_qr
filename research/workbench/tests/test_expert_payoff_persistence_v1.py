from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_expert_contextual_contest_v1 import synthetic

from us_quant import expert_payoff_persistence_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_fixed_persistence_rules_are_not_a_repaired_integral_or_reversal():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["candidates"] == list(rules.CANDIDATES)
    assert (
        policy["methodology"]["uniform_simplex_wealth_integral_repaired_or_approximated"] is False
    )
    for key, value in (
        ("refresh_denominator", 6),
        ("risk_tilt_coefficient", 2),
        ("independent_weight_and_target_tolerance", 1e-10),
    ):
        with pytest.raises(QuantError):
            rules.validate({**policy, key: value})


def test_exact_continuation_and_uniform_refresh():
    initial = rules.PayoffMassState.initial()
    values = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    continuing, score = initial.advance(values, False)
    refreshed, other_score = initial.advance(values, True)
    assert score == other_score == Fraction(7, 6)
    assert continuing.mass == (Fraction(2, 7), *([Fraction(1, 7)] * 5))
    assert refreshed.mass == tuple(Fraction(11, 12) * x + Fraction(1, 72) for x in continuing.mass)
    assert sum(refreshed.mass) == 1 and initial.completed_intervals == 0
    flat, score = initial.advance((1.0,) * 6, True)
    assert flat.mass == initial.mass and score == 1


@pytest.mark.parametrize("bad", [(0.0,) * 6, (-1.0,) * 6, (np.nan,) * 6, (1.0,) * 5])
def test_invalid_monthly_input_cannot_be_a_neutral_success(bad):
    with pytest.raises(QuantError):
        rules.PayoffMassState.initial().advance(bad, True)
    with pytest.raises(QuantError):
        replace(rules.PayoffMassState.initial(), completed_intervals=119).advance((1.0,) * 6, False)


def test_complete_month_relative_rounds_once_without_truncating_binary64():
    dates = pd.DatetimeIndex(["2017-05-01", "2017-05-02"])
    returns = [np.nextafter(0.01, 1.0), -0.003]
    observed = [SimpleNamespace(net_returns=pd.Series(returns, index=dates)) for _ in range(6)]
    expected = float((1 + Fraction.from_float(returns[0])) * (1 + Fraction.from_float(returns[1])))
    actual = rules.monthly_relatives(observed, dates)
    assert all(value.hex() == expected.hex() for value in actual)
    observed[0].net_returns.iloc[0] = -1
    with pytest.raises(QuantError):
        rules.monthly_relatives(observed, dates)


def test_initial_month_maturity_and_persistent_learning_clock():
    experts, close = synthetic()
    records = []
    targets, audits = rules.build_targets(
        experts, close, read_json(rules.POLICY), record_learning=records.append
    )
    assert len(records) == 4 * len(audits)
    assert audits[0]["complete_learning_intervals"] == audits[1]["complete_learning_intervals"] == 0
    assert audits[1]["initial_partial_source_month_not_assimilated"] is True
    assert audits[2]["complete_learning_intervals"] == 1
    assert (
        next(a["decision_session"] for a in audits if a["source_assessment_mature"]) == "2017-04-28"
    )
    assert sum(not a["source_assessment_mature"] for a in audits) == 7
    assert records[8]["stage"] == "input"
    assert records[8]["completed_intervals_before"] == {c["id"]: 0 for c in rules.CANDIDATES}
    for target in targets.values():
        np.testing.assert_allclose(target.dropna().sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert target.loc["2016-10-05"].notna().all()
    assert audits[-1]["complete_learning_intervals"] == len(audits) - 2


def test_future_source_changes_preserve_prior_targets_and_audits():
    cutoff = pd.Timestamp("2017-06-30")
    experts, close = synthetic()
    changed, later_close = synthetic(future_change=cutoff)
    baseline, first = rules.build_targets(experts, close, read_json(rules.POLICY))
    alternate, second = rules.build_targets(changed, later_close, read_json(rules.POLICY))
    early, prefix = rules.build_targets(experts, close.loc[:cutoff], read_json(rules.POLICY))
    for name in baseline:
        pd.testing.assert_frame_equal(baseline[name].loc[:cutoff], alternate[name].loc[:cutoff])
        pd.testing.assert_frame_equal(baseline[name].loc[:cutoff], early[name])
    assert prefix == first[: len(prefix)] == second[: len(prefix)]


def test_preupdate_inputs_survive_a_failing_advance(monkeypatch):
    experts, close = synthetic()
    records = []

    def fail(*args, **kwargs):
        raise QuantError("synthetic failing exact update")

    monkeypatch.setattr(rules.PayoffMassState, "advance", fail)
    with pytest.raises(QuantError, match="synthetic failing"):
        rules.build_targets(experts, close, read_json(rules.POLICY), record_learning=records.append)
    assert records[-1]["stage"] == "input"
    assert records[-1]["monthly_relative_inputs"] is not None
    assert records[-1]["monthly_relative_input_hex"] is not None
    assert all(value == 0 for value in records[-1]["completed_intervals_before"].values())


def test_learning_mass_never_mutates_prior_state():
    state = rules.PayoffMassState.initial()
    old = deepcopy(state)
    for i in range(119):
        state, _ = state.advance(tuple(1 + 0.02 * np.sin(i + j) for j in range(6)), True)
    state.validate()
    assert old == rules.PayoffMassState.initial() and state.completed_intervals == 119
