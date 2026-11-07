import itertools
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_switching_loss_v1 as model
from us_quant import ssd_expert_switching_loss_v1 as driver
from us_quant.backtest import BacktestResult
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


class MemoryOpinions:
    def __init__(self, future_after=None):
        self.future_after = future_after

    def observe(self, day):
        shift = day.month + (3 if self.future_after is not None and day > self.future_after else 0)
        raw = np.arange(1, 15, dtype=float)
        targets = np.array([np.roll(raw, shift + 2 * slot) / 105 * 0.98 for slot in range(6)])
        return model.OpinionSnapshot(day, targets, (day,) * 6)


def periods(calendar):
    dates = [d for d in calendar if d >= pd.Timestamp("2016-10-31") and is_month_end(d)]
    return [
        {
            "start_after": str(a.date()),
            "end": str(b.date()),
            "input_loss_labels": np.roll(np.linspace(0, 1, 6), b.month).tolist(),
        }
        for a, b in zip(dates[:-1], dates[1:], strict=True)
    ]


def enumerated_forecast(history, rates):
    n = len(history)
    with localcontext() as context:
        context.prec = 120
        scores = [
            [Fraction((-eta * (v - min(row))).exp()) for v in row]
            for eta, row in zip(rates, history, strict=True)
        ]

    def rising(k):
        return Fraction(math.prod(2 * i + 1 for i in range(k)), 2**k)

    mass = [Fraction(0)] * 6
    for sequence in itertools.product(range(6), repeat=n + 1):
        switches = sum(a != b for a, b in zip(sequence[:-1], sequence[1:], strict=True))
        prior = rising(switches) * rising(n - switches) / (math.factorial(n) * 6 * 5**switches)
        mass[sequence[-1]] += prior * math.prod(scores[i][sequence[i]] for i in range(n))
    return np.array([float(v / sum(mass)) for v in mass])


def test_frozen_loss_prior_rate_and_no_floor():
    policy = read_json(model.POLICY)
    model.validate(policy)
    assert policy["gap_floor"] is None and policy["negative_gap_clipped"] is False
    for key, value in (
        ("switch_prior", "Beta_one_one"),
        ("rate", "fixed_temperature"),
        ("gap_floor", 1e-12),
        ("five_year_source_learning_state_reset", True),
    ):
        with pytest.raises(QuantError):
            model.validate({**policy, key: value})


def test_equal_losses_have_analytic_zero_gap_without_floor():
    primary, independent = model.AdaptiveSwitchingLoss(), model.IndependentAdaptiveLoss()
    for _ in range(4):
        result = primary.advance(np.full(6, 0.3))
        independent.advance(np.full(6, 0.3))
        assert Decimal(result["source_mixability_gap_increment_decimal"]) == 0
        assert result["source_played_loss_decimal"] == result["source_mix_loss_decimal"]
    assert primary.rate is independent.rate is None
    assert primary.gap == 0 and len(primary.history) == 4
    np.testing.assert_allclose(primary.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-15)
    assert primary.snapshot()["negative_gap_clipped"] is False
    assert primary.snapshot()["gap_floor_applied"] is False
    with pytest.raises(QuantError, match="infinite rate"):
        model.path_partition([model.checked_labels(np.arange(6) / 5)], None)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_fixed_current_rate_path_partition_matches_sequences(length):
    history = [model.checked_labels(np.roll(np.linspace(0, 1, 6), i)) for i in range(length)]
    rate = Decimal("1.7")
    fitted = model.path_partition(history, rate)
    np.testing.assert_allclose(
        [float(v) for v in fitted.forecast],
        enumerated_forecast(history, [rate] * length),
        rtol=0,
        atol=1e-12,
    )


def test_current_rate_refilter_is_not_stale_posterior_power():
    state = model.AdaptiveSwitchingLoss()
    first = np.array([0, 1, 1, 1, 1, 1], dtype=float)
    second = np.array([1, 0, 0.2, 0.4, 0.6, 0.8])
    state.advance(first)
    old_rate = state.rate
    state.advance(second)
    assert state.rate < old_rate
    correct = enumerated_forecast(state.history, [state.rate] * 2)
    stale = enumerated_forecast(state.history, [old_rate, state.rate])
    np.testing.assert_allclose(state.predict(), correct, rtol=0, atol=1e-12)
    assert float(abs(correct - stale).max()) > 1e-6


def test_all119_matured_labels_match_independent_gap_rate_and_forecast():
    primary, independent = model.AdaptiveSwitchingLoss(), model.IndependentAdaptiveLoss()
    rng = np.random.default_rng(57)
    for _ in range(119):
        labels = rng.uniform(0, 1, 6)
        labels = (labels - labels.min()) / (labels.max() - labels.min())
        previous = primary.predict().tolist()
        update = primary.advance(labels)
        independent.advance(labels)
        assert update["prequential_forecast_before_assimilation"] == previous
        assert Decimal(update["source_mixability_gap_increment_decimal"]) >= 0
        assert abs(float(primary.gap) - independent.gap) <= 1e-12
        assert abs(float(primary.rate) - independent.rate) <= 1e-12
        assert float(abs(primary.predict() - independent.forecast).max()) <= 1e-12
        assert abs(float(primary.partition.centered_log_partition) - independent.partition) <= 1e-12
        assert abs(float(primary.partition.mean_hazard) - independent.hazard) <= 1e-12
    assert primary.snapshot()["persistent_primary_mass_cells"] == 714
    assert float(primary.predict().min()) >= 1 / (10 * 119) - 1e-12


def test_source_history_common_center_is_only_an_algebraic_invariant():
    history = [model.checked_labels(np.linspace(0, 0.5, 6))]
    with localcontext() as context:
        context.prec = model.PRECISION
        shifted = [tuple(v + Decimal("0.25") for v in history[0])]
    a, b = model.path_partition(history, Decimal(2)), model.path_partition(shifted, Decimal(2))
    assert a.forecast == b.forecast
    assert history[0][0] == 0 and shifted[0][0] == Decimal("0.25")


def test_negative_gap_state_cannot_be_clipped_into_success():
    state = model.AdaptiveSwitchingLoss()
    state.gap = Decimal("-1e-30")
    with pytest.raises(QuantError, match="failed gap"):
        state.predict()


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(5),
        np.full(6, np.nan),
        np.full(6, np.inf),
        np.ones(6, dtype=bool),
        np.array([-0.1, 0, 0, 0, 0, 0]),
    ],
)
def test_invalid_source_labels_never_become_neutral(labels):
    with pytest.raises(QuantError):
        model.AdaptiveSwitchingLoss().advance(labels)


def test_independent_nodes_cover_predictive_not_just_likelihood_degree(monkeypatch):
    assert 2 * model.QUADRATURE - 1 >= model.MAX_PERIODS
    monkeypatch.setattr(model, "QUADRATURE", 59)
    with pytest.raises(QuantError, match="forecast numerator"):
        model.independent_partition([], None)


def test_first_partial_month_is_not_assimilated():
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    result = model.build_targets(
        calendar, MemoryOpinions(), periods(calendar), retain=records.append
    )
    assert [r["complete_source_intervals"] for r in result.audits] == [0, 0, 1]
    assert result.audits[0]["period_update"] is result.audits[1]["period_update"] is None
    assert (
        result.audits[2]["period_update"]["prequential_forecast_before_assimilation"]
        == result.audits[1]["expert_weights"]
    )
    assert len(records) == 6
    assert records[-2]["matured_source_interval"]["start_after"] == "2016-10-31"


def test_missing_source_month_or_future_opinion_fails_explicitly():
    calendar = sessions("2016-10-05", "2017-01-31")
    with pytest.raises(QuantError, match="month is missing"):
        model.build_targets(calendar, MemoryOpinions(), periods(calendar)[:-1])

    class FutureOpinions(MemoryOpinions):
        def observe(self, day):
            return replace(super().observe(day), native_known_at=(next_session(day),) * 6)

    with pytest.raises(QuantError, match="then-known"):
        model.build_targets(calendar, FutureOpinions(), periods(calendar))


def test_actual_input_precedes_deliberate_new_learner_failure(monkeypatch):
    def failed(*args, **kwargs):
        raise QuantError("deliberate adaptive state failure")

    monkeypatch.setattr(model.AdaptiveSwitchingLoss, "advance", failed)
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    with pytest.raises(QuantError, match="deliberate"):
        model.build_targets(calendar, MemoryOpinions(), periods(calendar), retain=records.append)
    assert len(records) == 5 and records[-1]["phase"] == "before_decision"
    assert records[-1]["matured_source_interval"]["end"] == "2016-11-30"


@pytest.fixture
def data(market_factory):
    return market_factory(start="2016-10-05", end="2018-04-30", symbols=model.ASSETS)


@pytest.fixture
def prepared(data):
    return model.build_targets(data.close.index, MemoryOpinions(), periods(data.close.index))


def clipped(data, end):
    return MarketData(
        data.open.loc[:end].copy(),
        data.close.loc[:end].copy(),
        data.raw_close.loc[:end].copy(),
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end].copy(),
    )


def run(data, prepared, cost=5, delay=1):
    return model.run(data, prepared, "2018-02-06", "2018-03-29", cost_bps=cost, delay=delay)


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_full_mixed_asset_cash_and_BT(data, prepared, cost, delay):
    output = run(data, prepared, cost, delay)
    seen = []
    checked = model.verify_run(
        data,
        prepared,
        output,
        "2018-02-06",
        "2018-03-29",
        cost,
        delay,
        retain_comparison=lambda a, b, report: seen.append((a, b, report)),
    )
    assert checked["all_cash_financial_columns_and_positions_exact"]
    assert len(seen) == 1 and seen[0][2]["position_bit_differences"] == 0
    assert all(
        v["binary64_bit_differences"] == 0
        for v in seen[0][2]["all_frame_column_diagnostics"].values()
    )
    assert output.result.weights.iloc[: delay - 1].eq(0).all().all()
    assert (output.result.weights.iloc[delay - 1 :].gt(0).sum(axis=1) >= 3).all()
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])
    independent = independent_equity(
        data,
        output.issued,
        "2018-02-06",
        "2018-03-29",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert float(abs(independent["equity"] - output.result.frame["equity"]).max()) < 1e-6
    assert output.decisions[0]["assessment"]["source_master_known_at"] == "2018-01-31"
    assert output.decisions[0]["assessment"]["complete_source_intervals"] > 0


@pytest.mark.parametrize(
    "field", ["equity", "cash", "gross_exposure", "orders", "risk_free", "position"]
)
def test_cash_operands_retained_before_exact_failure(data, prepared, field):
    output = run(data, prepared)
    frame, weights = output.result.frame.copy(), output.result.weights.copy()
    if field == "position":
        weights.iloc[1, 0] = np.nextafter(float(weights.iloc[1, 0]), np.inf)
    elif field == "orders":
        frame.loc[frame.index[1], field] += 1
    else:
        frame.loc[frame.index[1], field] = np.nextafter(float(frame.iloc[1][field]), np.inf)
    changed = replace(output, result=BacktestResult(frame, weights))
    seen = []
    with pytest.raises(QuantError, match="cash-engine replay"):
        model.verify_run(
            data,
            prepared,
            changed,
            "2018-02-06",
            "2018-03-29",
            5,
            1,
            retain_comparison=lambda a, b, report: seen.append((a, b, report)),
        )
    assert len(seen) == 1 and seen[0][0].frame.equals(frame)


def test_concrete_record_Adapters_do_not_recurse(data, prepared, tmp_path, monkeypatch):
    output = run(data, prepared)
    monkeypatch.setattr(driver, "reference", lambda path: {"path": path.name, "sha256": "a" * 64})
    assert len(driver.write_run(tmp_path, output)) == 5
    compare = tmp_path / "comparison"
    compare.mkdir()
    driver.retain_cash_comparison(
        compare,
        output.result,
        output.result,
        model.comparison_report(output.result, output.result),
    )
    assert read_json(compare / "comparison.json")["retained_before_acceptance_assertion"]


@pytest.mark.parametrize("cut", ["2016-10-31", "2017-04-12"])
def test_future_labels_and_opinions_do_not_change_past_source_weights(data, prepared, cut):
    day = pd.Timestamp(cut)
    short = model.build_targets(
        data.close.loc[:day].index,
        MemoryOpinions(),
        periods(data.close.index),
    )
    pd.testing.assert_frame_equal(prepared.targets.loc[:day], short.targets, check_exact=True)
    assert short.audits == [
        r for r in prepared.audits if pd.Timestamp(r["decision_session"]) <= day
    ]
    altered = [
        {**row, "input_loss_labels": [0.5] * 6} if pd.Timestamp(row["end"]) > day else row
        for row in periods(data.close.index)
    ]
    future = model.build_targets(data.close.index, MemoryOpinions(day), altered)
    pd.testing.assert_frame_equal(
        prepared.targets.loc[:day], future.targets.loc[:day], check_exact=True
    )
    assert short.audits == [r for r in future.audits if pd.Timestamp(r["decision_session"]) <= day]


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
@pytest.mark.parametrize("cut", ["2018-02-28", "2018-03-13"])
def test_actual_capital_prefix_and_source_seed_remain_exact(data, prepared, cost, delay, cut):
    day = pd.Timestamp(cut)
    full = run(data, prepared, cost, delay)
    source = model.PreparedSourceTargets(
        prepared.targets.loc[:day],
        [r for r in prepared.audits if pd.Timestamp(r["decision_session"]) <= day],
        prepared.proof,
    )
    prefix = model.run(clipped(data, day), source, "2018-02-06", cut, cost_bps=cost, delay=delay)
    for a, b in (
        (full.result.frame, prefix.result.frame),
        (full.result.weights, prefix.result.weights),
        (full.desired, prefix.desired),
        (full.issued, prefix.issued),
    ):
        pd.testing.assert_frame_equal(a.loc[:day], b.loc[:day], check_exact=True)
        assert np.array_equal(np.signbit(a.loc[:day]), np.signbit(b.loc[:day]))
    assert prefix.decisions == [
        r for r in full.decisions if pd.Timestamp(r["signal_session"]) <= day
    ]


def test_actual_packet_defers_real_loss_learning(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-07T03:36:00Z",
        "body": {"candidate": driver.CANDIDATE, "frozen": frozen},
    }
    ready = {"path": "synthetic-readiness.json", "sha256": "b" * 64}
    body = driver.packet_body(frozen, registration, ready)
    assert body["schema_version"] == 3 and body["entries"][0]["candidate_id"] == model.ID
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert body["historical_new_meta_scores_or_targets_computed"] is False
    monkeypatch.setattr(
        driver, "events", lambda kind: [registration] if kind == "registered" else []
    )
    monkeypatch.setattr(driver, "reference", lambda path: ready)
    monkeypatch.setattr(
        driver,
        "read_json",
        lambda path: (
            {**body, "created_at": "2026-11-07T03:36:01Z"}
            if path == driver.PACKET
            else {
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            }
        ),
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3


def test_old_source_learners_and_gap_models_never_execute(data, monkeypatch):
    from us_quant import (
        expert_switching_wealth_v1,
        main_risk_confidence_prod_v1,
        protective_six_adaptive_hedge_v1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No old source learner or gap/account model may replay.")

    for name in ("predict", "advance"):
        monkeypatch.setattr(expert_switching_wealth_v1.SwitchCountWealth, name, forbidden)
    for name in ("weights", "update"):
        monkeypatch.setattr(protective_six_adaptive_hedge_v1.AdaHedge, name, forbidden)
    monkeypatch.setattr(main_risk_confidence_prod_v1, "run", forbidden)
    result = model.build_targets(data.close.index, MemoryOpinions(), periods(data.close.index))
    assert result.proof["complete_source_intervals"] > 0
