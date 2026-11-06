from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import execution_gap_shield_v1 as model
from us_quant import ssd_execution_gap_shield_v1 as driver
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.main_capital_feedback_v1 import AllocationDecision
from us_quant.storage import read_json


def state(*, nav=10000.0, peak=10000.0, risky=0.98):
    values = np.zeros(model.N)
    values[0] = nav * risky
    return model.reference.MainMark(
        pd.Timestamp("2020-03-31"),
        values,
        nav * (1 - risky),
        nav,
        peak,
    )


def paths(delay=1, *, wait=1.0, opening=1.0, risky_mark=1.0, bil_mark=1.0):
    dates = sessions("2020-03-25", "2020-03-31")
    fill = np.ones((1, model.N))
    fill[0, 0], fill[0, model.BIL] = risky_mark, bil_mark
    return model.DelayPaths(
        dates[-1],
        delay,
        dates[-delay - 1 : -delay],
        dates[-1:],
        np.full((1, delay - 1, model.N), wait),
        np.full((1, model.N), opening),
        fill,
    )


class FixedNominal:
    def decide(self, capital):
        target = pd.Series(0.0, index=model.ASSETS)
        target.iloc[0] = 0.98
        return AllocationDecision(target, {"source_opinion_known_at": "2020-01-31"})


class FixedScenarios:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def observe(self, as_of, delay):
        assert self.value.as_of == as_of and self.value.delay == delay
        self.calls += 1
        return self.value


def choose(mark=None, sample=None):
    sample = paths() if sample is None else sample
    return model.ShieldPolicy(
        FixedNominal(),
        FixedScenarios(sample),
        5 if sample.delay == 1 else 20,
        sample.delay,
    ).decide(state() if mark is None else mark, None)


def test_frozen_policy_is_exact():
    policy = read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate({**policy, "reference_peak_fraction": 0.86})


@pytest.mark.parametrize("delay,cost", [(1, 5), (2, 20)])
def test_initial_capital_is_issued_but_not_free_of_cost(delay, cost):
    sample = paths(delay)
    policy = model.ShieldPolicy(FixedNominal(), FixedScenarios(sample), cost, delay)
    decision = policy.decide(state(risky=0), None, initial=True)
    assert decision.issue and decision.reason == "initial_capital"
    assert decision.target.sum() == pytest.approx(0.98)
    assert decision.assessment["hypothetical_full_fill_reference_certified"]


def test_pending_request_never_replaced():
    archive = FixedScenarios(paths(2))
    policy = model.ShieldPolicy(FixedNominal(), archive, 20, 2)
    decision = policy.decide(state(), pd.Timestamp("2020-04-01"))
    assert decision.reason == "pending_request_preserved"
    assert decision.target is None and not decision.issue and archive.calls == 0


def test_late_pending_fill_is_an_error():
    policy = model.ShieldPolicy(FixedNominal(), FixedScenarios(paths()), 5, 1)
    with pytest.raises(QuantError):
        policy.decide(state(), pd.Timestamp("2020-03-30"))


def test_waiting_holdings_breach_is_not_cured_by_new_target():
    decision = choose(sample=paths(2, wait=0.8))
    assert decision.assessment["reference_status"] == "unavoidable_prefill_reference_breach"
    assert decision.assessment["uncertified_defensive_fallback"]
    assert decision.target[model.BIL] == 0.98
    assert not decision.assessment["hypothetical_full_fill_reference_certified"]


def test_current_outside_is_an_uncertified_attempt():
    decision = choose(mark=state(nav=8000))
    assert decision.assessment["reference_status"] == "outside_at_current_mark"
    assert decision.assessment["uncertified_defensive_fallback"]


def test_cost_bound_exhaustion_is_explicit():
    decision = choose(mark=state(risky=1), sample=paths(opening=1e-7))
    assert decision.assessment["reference_status"] == "uncertified_conservative_cost_bound"
    assert decision.assessment["uncertified_defensive_fallback"]


def test_infeasible_even_BIL_is_not_success_shaped():
    sample = paths(risky_mark=0.5, bil_mark=0.5)
    decision = choose(sample=sample)
    assert decision.assessment["reference_status"] == "no_finite_sufficient_interval"
    assert decision.assessment["interval"]["largest_certified_nominal_fraction"] is None
    assert decision.assessment["uncertified_defensive_fallback"]


def test_band_holds_when_actual_hold_reference_is_safe():
    decision = choose()
    assert not decision.issue and decision.reason == "hold"
    assert decision.assessment["unchanged_hold_finite_reference_safe"]
    assert not decision.assessment["band_hold_is_certified_full_fill"]


def test_certified_unsafe_hold_can_override_ordinary_band():
    decision = choose(mark=state(risky=0.999), sample=paths(risky_mark=0.849))
    assert decision.assessment["maximum_observed_target_gap"] < 0.025
    assert decision.assessment["hypothetical_full_fill_reference_certified"]
    assert not decision.assessment["unchanged_hold_finite_reference_safe"]
    assert decision.issue and decision.reason == "certified_reference_overrides_band"


@pytest.mark.parametrize("upper", [Fraction(1, 10), Fraction(1, 3), Fraction(7, 11)])
def test_inside_binary64_not_rounded_above_upper(upper):
    interval = {
        "feasible": True,
        "lower": model.reference.encode(Fraction(0)),
        "upper": model.reference.encode(upper),
    }
    value = model.greatest_binary64(interval)
    assert Fraction.from_float(value) <= upper
    assert Fraction.from_float(np.nextafter(value, np.inf)) > upper


def test_no_representable_fraction_returns_none():
    a, b = 0.5, np.nextafter(0.5, np.inf)
    middle = (Fraction.from_float(a) + Fraction.from_float(float(b))) / 2
    interval = {
        "feasible": True,
        "lower": model.reference.encode(middle),
        "upper": model.reference.encode(middle),
    }
    assert model.greatest_binary64(interval) is None


def test_retention_precedes_reference_resolution():
    seen = []
    model.ShieldPolicy(FixedNominal(), FixedScenarios(paths()), 5, 1).decide(
        state(),
        None,
        retain_inputs=lambda mark, target, sample: seen.append(
            (mark.as_of, sample.as_of, target.copy())
        ),
    )
    assert len(seen) == 1 and seen[0][0] == state().as_of


def test_future_scenario_is_rejected():
    with pytest.raises(QuantError):
        replace(paths(), as_of=pd.Timestamp("2020-03-30")).validate()


@pytest.mark.parametrize("delay,cost", [(1, 5), (2, 20)])
def test_conditional_full_fill_matches_original_cash_funding(delay, cost):
    sample = paths(delay, wait=0.99, opening=0.98, risky_mark=0.75)
    mark = state()
    decision = choose(mark=mark, sample=sample)
    assert decision.assessment["hypothetical_full_fill_reference_certified"]
    assert 0 < decision.assessment["applied_nominal_fraction"] < 1
    values, cash, fee, _, _ = model.rebalance(
        mark.asset_values * sample.opening[0],
        mark.cash,
        decision.target,
        cost,
        1,
    )
    delivered = float(values @ sample.fill_mark[0] + cash)
    prior_peak = max(
        [mark.peak, *[float(row @ mark.asset_values + mark.cash) for row in sample.waiting[0]]]
    )
    assert fee > 0 and delivered >= 0.85 * prior_peak - 1e-6


def test_two_session_coefficients_agree_with_immutable_reference():
    sample = paths(2, wait=1.01, opening=0.98, risky_mark=0.75)
    mark = state()
    target = FixedNominal().decide(None).target.to_numpy()
    original = model.reference.StressSample(
        sample.as_of,
        sample.starts,
        sessions("2020-03-30", "2020-03-30"),
        sample.ends,
        sample.waiting[:, 0],
        sample.opening,
        sample.fill_mark,
    )
    old = model.reference.coefficients(target, mark, original)
    current = model.coefficients(target, mark, sample, 20)
    assert np.allclose(old[0], current[0], rtol=0, atol=1e-6)
    assert np.allclose(old[1], current[1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("cost,delay", [(0, 1), (5, 2), (20, 1), (20, True)])
def test_original_cost_delay_pairs_cannot_be_spliced(cost, delay):
    with pytest.raises(QuantError):
        model.ShieldPolicy(FixedNominal(), FixedScenarios(paths()), cost, delay)


@pytest.fixture
def data(market_factory):
    return market_factory(start="2019-12-02", end="2020-04-30", symbols=model.ASSETS)


def nominal(data):
    value = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for day in value.index:
        if day == value.index[0] or is_month_end(day):
            value.loc[day] = 0.0
            value.loc[day, "SPY" if day.month % 2 else "GLD"] = 0.98
    return value


def clipped(data, end):
    return MarketData(
        data.open.loc[:end].copy(),
        data.close.loc[:end].copy(),
        data.raw_close.loc[:end].copy(),
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end].copy(),
    )


@pytest.mark.parametrize("delay", [1, 2])
def test_known_archive_prefix_and_future_perturbation(data, delay):
    day = data.close.index[50]
    full = model.KnownDelayArchive(data.open, data.close).observe(day, delay)
    copy = clipped(data, data.close.index[-1])
    copy.open.loc[copy.open.index > day] *= 1.3
    copy.close.loc[copy.close.index > day] *= 0.8
    altered = model.KnownDelayArchive(copy.open, copy.close).observe(day, delay)
    for name in ("opening", "fill_mark", "waiting"):
        assert np.array_equal(getattr(full, name), getattr(altered, name))
    assert (full.ends <= day).all()


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_actual_account_and_independent_BT(data, cost, delay):
    output = model.run(
        data,
        nominal(data),
        "2020-02-04",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
    )
    checked = model.verify_run(data, output, "2020-02-04", "2020-03-31", cost, delay)
    assert checked["all_formal_frames_positions_cash_fees_match_cash_engine"]
    assert output.result.frame.index.equals(sessions("2020-02-04", "2020-03-31"))
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])
    independent = independent_equity(
        data,
        output.issued,
        "2020-02-04",
        "2020-03-31",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert float(abs(independent["equity"] - output.result.frame["equity"]).max()) < 1e-5
    if delay == 2:
        assert output.result.frame.iloc[0]["equity"] == 10000
        assert checked["pending_refreshes_without_replacement"] > 0


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_actual_account_prefix_is_unchanged(data, cost, delay):
    day = data.close.index[65]
    full = model.run(
        data,
        nominal(data),
        "2020-02-04",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
    )
    prefix = clipped(data, day)
    partial = model.run(
        prefix,
        nominal(data).loc[:day],
        "2020-02-04",
        str(day.date()),
        cost_bps=cost,
        delay=delay,
    )
    for left, right in (
        (full.result.frame, partial.result.frame),
        (full.result.weights, partial.result.weights),
        (full.desired, partial.desired),
        (full.issued, partial.issued),
    ):
        pd.testing.assert_frame_equal(left.loc[:day], right.loc[:day], check_exact=True)
    assert [
        r for r in full.decisions if pd.Timestamp(r["signal_session"]) <= day
    ] == partial.decisions


def test_fresh_five_year_account_does_not_inherit_a_source_peak(data):
    output = model.run(
        data,
        nominal(data),
        "2020-02-04",
        "2020-03-31",
        cost_bps=5,
        delay=1,
    )
    row = output.decisions[0]
    assert row["main_peak_at_decision"] == row["main_equity_at_decision"] == 10000
    assert row["main_cash_at_decision"] == 10000
    assert all(v == 0 for v in row["main_asset_values_at_decision"])


def test_real_dynamic_packet_entry_defers_historical_outputs(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-06T22:00:00Z",
        "body": {"candidate": driver.CANDIDATE, "frozen": frozen},
    }
    ready = {"path": "synthetic-readiness.json", "sha256": "b" * 64}
    body = driver.packet_body(frozen, registration, ready)
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert body["candidate_accounts_or_metrics_computed"] is False
    assert body["historical_target_proof_deferred_until_published_execution"] is True
    monkeypatch.setattr(
        driver, "events", lambda kind: [registration] if kind == "registered" else []
    )
    monkeypatch.setattr(driver, "reference", lambda path: ready)
    monkeypatch.setattr(
        driver,
        "read_json",
        lambda path: (
            {**body, "created_at": "2026-11-06T22:00:01Z"}
            if path == driver.PACKET
            else {
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            }
        ),
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3
