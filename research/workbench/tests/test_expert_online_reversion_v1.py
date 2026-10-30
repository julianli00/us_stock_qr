from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_online_reversion_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE
from us_quant.online_reversion import update_weights
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity


def synthetic(end="2017-12-29", future_change=None):
    calendar = sessions("2016-10-05", end)
    universe = (*UNIVERSE, "IJR", "PKW")
    experts = []
    phase = np.arange(len(calendar))
    for i, (identifier, mechanism) in enumerate(
        zip(rules.EXPERT_IDS, rules.MECHANISMS, strict=True)
    ):
        target = pd.DataFrame(0.0, index=calendar, columns=universe)
        share = 0.45 + 0.2 * np.sin(phase / (i + 3))
        target[universe[i]] = 0.98 * share
        target["BIL"] += 0.98 * (1 - share)
        returns = pd.Series(
            0.0003 + 0.006 * np.sin(phase[1:] / (i + 2.5)) + i * 0.00007,
            index=calendar[1:],
        )
        if future_change is not None:
            returns.loc[returns.index > future_change] += (i - 2) * 0.01
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
    return tuple(experts), calendar, universe


def test_prior_art_parameters_and_countercyclical_role_are_fixed():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["candidates"] == list(rules.CANDIDATES)
    assert policy["formation_price_observations"] == 6
    assert policy["learning_state_resets_for_five_year_account"] is False
    assert policy["methodology"]["old_19_asset_universe_or_DBC_admitted"] is False
    for key, value in (("formation_price_observations", 5), ("actual_weight_trade_band", 0.01)):
        changed = {**policy, key: value}
        with pytest.raises(QuantError):
            rules.validate(changed)


@pytest.mark.parametrize("candidate", rules.CANDIDATES)
def test_prior_art_updates_and_flat_forecast_guard(candidate):
    rng = np.random.default_rng(53048)
    for _ in range(25):
        history = 10000 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (6, 6)), axis=0))
        prior = rng.dirichlet(np.ones(6))
        actual, audit = rules.advance(prior, history, candidate, 12)
        forecast = (
            history[-5:].mean(axis=0) / history[-1]
            if candidate["method"] == "olmar"
            else history[-1] / history[-2]
        )
        expected = update_weights(prior, forecast, candidate["method"], candidate["epsilon"])
        np.testing.assert_array_equal(actual, expected)
        assert audit["formation_mature"] is True
        assert audit["criterion_or_profit_guaranteed"] is False
    prior = np.array([0.6, 0.2, 0.1, 0.05, 0.025, 0.025])
    actual, audit = rules.advance(prior, np.full((6, 6), 12345.6), candidate, 6)
    np.testing.assert_array_equal(actual, prior)
    assert audit["reason"] == "constant_forecast_no_new_relative_information"


@pytest.mark.parametrize("candidate", rules.CANDIDATES)
def test_recent_relative_winners_are_not_rewarded_by_the_reversal_forecast(candidate):
    history = np.full((6, 6), 10000.0)
    history[-1, 0] = 11000.0
    history[-1, 1] = 9000.0
    result, audit = rules.advance(np.full(6, 1 / 6), history, candidate, 6)
    assert result[1] > result[0]
    assert audit["reason"] == "observed_NET_expert_reversal_update"


@pytest.mark.parametrize("bad", ("negative", "NaN", "budget", "level", "window", "candidate"))
def test_invalid_learning_inputs_have_no_funded_or_equal_weight_fallback(bad):
    prior, history, candidate = np.full(6, 1 / 6), np.full((6, 6), 10000.0), rules.CANDIDATES[0]
    if bad == "negative":
        prior[0], prior[1] = -0.1, prior[1] + prior[0] + 0.1
    elif bad == "NaN":
        history[0, 0] = np.nan
    elif bad == "budget":
        prior *= 0.9
    elif bad == "level":
        history[0, 0] = 0
    elif bad == "window":
        history = history[:-1]
    else:
        candidate = {**candidate, "epsilon": 2.0}
    with pytest.raises(QuantError):
        rules.advance(prior, history, candidate, 6)


def test_initial_formation_and_monthly_main_clock():
    experts, calendar, universe = synthetic()
    records = []
    targets, audits = rules.build_targets(
        experts, calendar, universe, read_json(rules.POLICY), record_learning=records.append
    )
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    assert dates[0] == pd.Timestamp("2016-10-05")
    assert sum(not row["source_assessment_mature"] for row in audits) == 1
    assert audits[0]["source_NET_daily_observations"] == 0
    assert audits[0]["source_price_observations"] == 1
    assert audits[-1]["statistical_daily_updates_per_candidate"] == len(calendar) - 5
    assert len(records) == 4 * (len(calendar) - 1)
    mature = [row for row in records if row["stage"] == "resolved" and row["formation_mature"]]
    assert mature[0]["formation_price_observations"] == 6
    for target in targets.values():
        assert target.dropna(how="all").index.equals(dates)
        np.testing.assert_allclose(target.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert all(row["learning_state_is_main_wealth"] is False for row in audits)


def test_future_source_changes_preserve_prior_targets_and_records():
    experts, calendar, universe = synthetic()
    cutoff = pd.Timestamp("2017-03-31")
    later, _, _ = synthetic(future_change=cutoff)
    original_records, changed_records = [], []
    original, audits = rules.build_targets(
        experts,
        calendar,
        universe,
        read_json(rules.POLICY),
        record_learning=original_records.append,
    )
    changed, later_audits = rules.build_targets(
        later, calendar, universe, read_json(rules.POLICY), record_learning=changed_records.append
    )
    early, early_audits = rules.build_targets(
        experts, calendar[calendar <= cutoff], universe, read_json(rules.POLICY)
    )
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], changed[name].loc[:cutoff])
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], early[name])
    assert early_audits == [a for a in audits if a["decision_session"] <= str(cutoff.date())]
    assert early_audits == [a for a in later_audits if a["decision_session"] <= str(cutoff.date())]
    assert [r for r in original_records if r["session"] <= str(cutoff.date())] == [
        r for r in changed_records if r["session"] <= str(cutoff.date())
    ]


@pytest.mark.parametrize("bad", ("calendar_gap", "source_gap", "future_target", "RF"))
def test_source_gaps_and_information_clock_fail_closed(bad):
    experts, calendar, universe = synthetic()
    if bad == "calendar_gap":
        calendar = calendar.delete(8)
    else:
        original = experts[0]

        class InvalidClock:
            identity = original.identity

            def observe(self, day, lookback):
                result = original.observe(day, lookback)
                if day < calendar[8]:
                    return result
                if bad == "source_gap":
                    return replace(result, net_returns=result.net_returns.iloc[:-1])
                if bad == "future_target":
                    return replace(result, target_known_at=calendar[-1] + pd.Timedelta(days=1))
                return replace(result, risk_free=result.risk_free + 0.001)

        experts = (InvalidClock(), *experts[1:])
    with pytest.raises(QuantError):
        rules.build_targets(experts, calendar, universe, read_json(rules.POLICY))


def test_learning_inputs_are_saved_before_a_failing_update(monkeypatch):
    experts, calendar, universe = synthetic()
    retained = []

    def failed(*args):
        assert retained[-1]["stage"] == "input"
        assert len(retained[-1]["source_history_after"]) == 6
        assert len(retained[-1]["algorithm_weights_before"]) == 6
        raise QuantError("Declared synthetic update failure after retained operands.")

    monkeypatch.setattr(rules, "update_weights", failed)
    with pytest.raises(QuantError, match="after retained operands"):
        rules.build_targets(
            experts, calendar, universe, read_json(rules.POLICY), record_learning=retained.append
        )
    assert retained[-1]["candidate_id"] == rules.CANDIDATES[0]["id"]
