from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_contextual_contest_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity


def synthetic(end="2017-12-29", future_change=None):
    calendar = sessions("2015-10-01", end)
    universe = (*UNIVERSE, "IJR", "PKW")
    phase = np.arange(len(calendar))
    close = pd.DataFrame(
        {
            symbol: 100 * np.exp(0.00015 * phase + 0.04 * np.sin(phase / (i + 19)))
            for i, symbol in enumerate(universe)
        },
        index=calendar,
    )
    close["BIL"] = 100 * np.exp(0.00005 * phase)
    close["SPY"] = 100 * np.exp(0.00018 * phase + 0.08 * np.sin(phase / 37))
    close["IEF"] = 100 * np.exp(0.00004 * phase + 0.05 * np.cos(phase / 31))
    observed_dates = calendar[calendar > pd.Timestamp("2016-10-05")]
    source_phase = np.arange(len(observed_dates))
    experts = []
    for i, (identifier, mechanism) in enumerate(
        zip(rules.EXPERT_IDS, rules.MECHANISMS, strict=True)
    ):
        target = pd.DataFrame(0.0, index=calendar, columns=universe)
        share = 0.45 + 0.15 * np.sin(phase / (i + 13))
        target[universe[i]] = 0.98 * share
        target["BIL"] += 0.98 * (1 - share)
        returns = pd.Series(
            0.0003 + 0.006 * np.sin(source_phase / (i + 3.5)) + i * 0.00006,
            index=observed_dates,
        )
        if future_change is not None:
            returns.loc[returns.index > future_change] += (i - 2) * 0.005
            target.loc[target.index > future_change, :] = 0.0
            target.loc[target.index > future_change, universe[i]] = 0.98
        equity = 10000 * (1 + returns).cumprod()
        shadow = pd.DataFrame(
            {
                "equity": equity,
                "return": returns,
                "risk_free": 0.00005,
                "cash": equity * 0.02,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0,
            }
        )
        experts.append(
            ArchivedDenseExpert(
                ExpertIdentity(identifier, mechanism, "2026-10-29T14:13:03+00:00"),
                target,
                shadow,
                universe,
            )
        )
    if future_change is not None:
        close.loc[close.index > future_change, "SPY"] *= 1.2
        close.loc[close.index > future_change, "IEF"] *= 0.8
    return tuple(experts), close


def test_fixed_contextual_rules_are_not_a_repaired_wealth_integral():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["candidates"] == list(rules.CANDIDATES)
    assert policy["equal_winner_credit_units_per_month"] == 60
    assert policy["posterior_pseudocount_per_expert"] == 1
    assert (
        policy["methodology"]["old_uniform_simplex_wealth_integral_repaired_or_approximated"]
        is False
    )
    for key, value in (("context_lookback_sessions", 63), ("risk_tilt_coefficient", 2)):
        with pytest.raises(QuantError):
            rules.validate({**policy, key: value})


def test_exact_context_and_zero_relative_return_ties():
    record = {
        "earlier_prices": {"SPY": 1.0, "IEF": 1.0, "BIL": 1.0},
        "current_prices": {"SPY": 1.0, "IEF": 1.0, "BIL": 1.0},
    }
    assert rules.market_context(record) == 0
    record["current_prices"]["SPY"] = np.nextafter(1.0, 2.0)
    assert rules.market_context(record) == 2
    record["current_prices"]["IEF"] = np.nextafter(1.0, 2.0)
    assert rules.market_context(record) == 3
    record["current_prices"]["SPY"] = 1.0
    assert rules.market_context(record) == 1


def test_exact_month_keys_and_integer_tied_credit():
    dates = pd.DatetimeIndex(["2017-01-03"])
    observed = [SimpleNamespace(net_returns=pd.Series([0.0], index=dates)) for _ in range(6)]
    result = rules.winner_credit(observed, dates)
    assert result["winner_relative_keys"] == [10**12] * 6
    assert result["credit_units"] == [10] * 6
    observed[0].net_returns.iloc[0] = 2**-13
    observed[1].net_returns.iloc[0] = 2**-13
    result = rules.winner_credit(observed, dates)
    assert result["winner_relative_keys"][:2] == [1000122070312] * 2
    assert result["credit_units"] == [30, 30, 0, 0, 0, 0]
    observed[0].net_returns.iloc[0] = 3 * 2**-13
    assert rules.winner_credit(observed, dates)["winner_relative_keys"][0] == 1000366210938


def test_positive_posterior_prior_and_context_counts():
    state = rules.ContestState.initial()
    initial, months = state.posterior(None)
    assert initial == (Fraction(1, 6),) * 6 and months == 0
    later = state.advance(2, {"credit_units": [60, 0, 0, 0, 0, 0]})
    assert state.complete_months == 0 and later.complete_months == 1
    posterior, months = later.posterior(2)
    assert posterior == (Fraction(2, 7), *([Fraction(1, 7)] * 5)) and months == 1
    assert later.posterior(None) == (posterior, months)
    assert later.posterior(0) == ((Fraction(1, 6),) * 6, 0)
    with pytest.raises(QuantError):
        state.advance(2, {"credit_units": [59, 0, 0, 0, 0, 0]})
    with pytest.raises(QuantError):
        rules.ContestState(((1,) * 6,) * 4, 0).validate()


def loss_observations(values):
    dates = sessions("2017-01-03", "2017-12-29")[:126]
    return [
        SimpleNamespace(
            net_returns=pd.Series(value, index=dates),
            risk_free=pd.Series(0.0, index=dates),
        )
        for value in values
    ]


def test_joint_loss_tilt_and_no_loss_information_are_explicit():
    posterior = (Fraction(1, 6),) * 6
    tilted, proof = rules.co_loss_weights(loss_observations([-0.04, *([0.001] * 5)]), posterior)
    assert tilted[0] < tilted[1] and np.all(tilted > 0)
    assert len(proof["common_loss_source_sessions"]) == 126
    assert proof["source_risk_proxy_is_main_loss_or_guarantee"] is False
    assert proof["positive_hedge_gains_credited_in_proxy"] is False
    unchanged, proof = rules.co_loss_weights(loss_observations([0.001] * 6), posterior)
    np.testing.assert_allclose(unchanged, np.full(6, 1 / 6), rtol=0, atol=1e-15)
    assert proof["common_loss_source_sessions"] == []
    assert proof["risk_tilt_reason"].startswith("no_common_excess_loss_dates")


def test_initial_month_clock_maturity_and_persistent_counts():
    experts, close = synthetic()
    records = []
    targets, audits = rules.build_targets(
        experts, close, read_json(rules.POLICY), record_learning=records.append
    )
    dates = rules.decision_sessions(close.index, read_json(rules.POLICY))
    assert audits[0]["decision_role"] == "initial_capital"
    assert audits[0]["complete_learning_months"] == audits[1]["complete_learning_months"] == 0
    assert audits[1]["initial_partial_source_month_not_assimilated"] is True
    assert audits[2]["complete_learning_months"] == 1
    assert audits[-1]["complete_learning_months"] == len(audits) - 2
    assert sum(not row["source_assessment_mature"] for row in audits) == 7
    assert len(records) == 4 * len(audits)
    for target in targets.values():
        assert target.dropna(how="all").index.equals(dates)
        np.testing.assert_allclose(target.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert all(row["learning_state_is_main_wealth"] is False for row in audits)


def test_future_prices_targets_and_returns_preserve_prior_outputs():
    experts, close = synthetic()
    cutoff = pd.Timestamp("2017-03-31")
    future, changed_close = synthetic(future_change=cutoff)
    actual, audits = rules.build_targets(experts, close, read_json(rules.POLICY))
    changed, later = rules.build_targets(future, changed_close, read_json(rules.POLICY))
    prefix, earlier = rules.build_targets(experts, close.loc[:cutoff], read_json(rules.POLICY))
    for name in actual:
        pd.testing.assert_frame_equal(actual[name].loc[:cutoff], changed[name].loc[:cutoff])
        pd.testing.assert_frame_equal(actual[name].loc[:cutoff], prefix[name])
    assert earlier == [row for row in audits if row["decision_session"] <= str(cutoff.date())]
    assert earlier == [row for row in later if row["decision_session"] <= str(cutoff.date())]


def test_unconditional_control_ignores_how_price_contexts_partition_months():
    experts, close = synthetic()
    altered = close.copy()
    phase = np.arange(len(close))
    altered["SPY"] = 100 * np.exp(-0.0001 * phase)
    altered["IEF"] = 100 * np.exp(0.0002 * phase)
    original, audits = rules.build_targets(experts, close, read_json(rules.POLICY))
    changed, later = rules.build_targets(experts, altered, read_json(rules.POLICY))
    pd.testing.assert_frame_equal(
        original[rules.CANDIDATES[0]["id"]], changed[rules.CANDIDATES[0]["id"]]
    )
    assert [row["current_context"] for row in audits] != [row["current_context"] for row in later]


@pytest.mark.parametrize("kind", ("calendar", "source_gap", "future_target", "RF"))
def test_context_and_source_missingness_fail_closed(kind):
    experts, close = synthetic()
    if kind == "calendar":
        close = close.drop(close.index[300])
    else:
        original = experts[0]

        class InvalidClock:
            identity = original.identity

            def observe(self, day, lookback):
                result = original.observe(day, lookback)
                if day < pd.Timestamp("2017-05-01"):
                    return result
                if kind == "source_gap":
                    return replace(result, net_returns=result.net_returns.iloc[:-1])
                if kind == "future_target":
                    return replace(result, target_known_at=close.index[-1] + pd.Timedelta(days=1))
                return replace(result, risk_free=result.risk_free + 0.001)

        experts = (InvalidClock(), *experts[1:])
    with pytest.raises(QuantError):
        rules.build_targets(experts, close, read_json(rules.POLICY))


@pytest.mark.parametrize("kind", ("credit", "risk"))
def test_inputs_are_saved_before_a_failing_credit_or_risk_update(monkeypatch, kind):
    experts, close = synthetic()
    retained = []

    def failed(*args):
        assert retained[-1]["stage"] == "input"
        assert len(retained[-1]["credit_state_before"]) == 4
        assert len(retained[-1]["source_NET_returns"]) == 6
        raise QuantError("Synthetic failure after preserved decision operands.")

    monkeypatch.setattr(rules, "winner_credit" if kind == "credit" else "co_loss_weights", failed)
    with pytest.raises(QuantError, match="after preserved"):
        rules.build_targets(
            experts, close, read_json(rules.POLICY), record_learning=retained.append
        )
