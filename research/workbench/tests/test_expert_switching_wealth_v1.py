import itertools
import json
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_switching_wealth_v1 as model
from us_quant import ssd_expert_switching_wealth_v1 as driver
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
        raw = np.arange(1, 15, dtype=float)
        shift = day.month + (5 if self.future_after is not None and day > self.future_after else 0)
        rows = np.array([np.roll(raw, shift + 2 * slot) / 105 * 0.98 for slot in range(6)])
        return model.OpinionSnapshot(day, rows, (day,) * 6)


def periods(calendar):
    dates = [d for d in calendar if d >= pd.Timestamp("2016-10-31") and is_month_end(d)]
    return [
        {
            "start_after": str(a.date()),
            "end": str(b.date()),
            "canonical_log_relatives": (np.sin(np.arange(6) + b.month) * 0.06).tolist(),
        }
        for a, b in zip(dates[:-1], dates[1:], strict=True)
    ]


def sequence_oracle(logs):
    n = len(logs)
    with localcontext() as context:
        context.prec = 120
        gains = [[Fraction(Decimal.from_float(float(x)).exp()) for x in row] for row in logs]

    def rising_half(k):
        return Fraction(math.prod(2 * i + 1 for i in range(k)), 2**k)

    totals = [Fraction(0)] * 6
    for sequence in itertools.product(range(6), repeat=n + 1):
        switches = sum(a != b for a, b in zip(sequence[:-1], sequence[1:], strict=True))
        probability = rising_half(switches) * rising_half(n - switches)
        probability /= math.factorial(n) * 6 * 5**switches
        score = math.prod(gains[t][sequence[t]] for t in range(n))
        totals[sequence[-1]] += probability * score
    normalizer = sum(totals)
    return np.array([float(v / normalizer) for v in totals])


def test_frozen_prior_and_financial_contract_not_a_sweep():
    policy = read_json(model.POLICY)
    model.validate(policy)
    for key, value in (
        ("switch_prior", "uniform_Beta_one_one"),
        ("actual_weight_band", 0.02),
        ("five_year_source_learning_state_reset", True),
        ("source_game_score_is_main_net_wealth", True),
    ):
        with pytest.raises(QuantError):
            model.validate({**policy, key: value})


def test_initial_prior_is_complete_uniform_and_not_a_market_probability():
    state = model.SwitchCountWealth()
    np.testing.assert_allclose(state.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-15)
    assert state.periods == 0 and state.expected_hazard() == 0.5
    assert state.snapshot()["source_game_score_is_actual_main_net_wealth"] is False
    assert state.snapshot()["persistent_primary_mass_cells"] == 6


def test_next_identity_not_posterior_of_matured_identity():
    state = model.SwitchCountWealth()
    forecast = state.predict()
    update = state.advance(np.array([math.log(2), 0, 0, 0, 0, 0]))
    posterior = np.array([float(v) for v in state.mass[0]])
    expected = posterior * 0.5 + (1 - posterior) * 0.5 / 5
    np.testing.assert_allclose(state.predict(), expected, rtol=0, atol=1e-15)
    assert not np.allclose(state.predict(), posterior, rtol=0, atol=1e-12)
    assert update["forecast_before_assimilation"] == forecast.tolist()
    assert state.expected_hazard() == 0.5


@pytest.mark.parametrize("length", [1, 2, 3])
def test_switch_count_filter_matches_explicit_integrated_sequences(length):
    logs = np.array(
        [
            [math.log(2), 0, -math.log(2), 0.1, -0.1, 0.05],
            [-0.1, math.log(2), 0.05, 0, -0.05, 0.1],
            [0, -0.15, 0.12, math.log(2), 0.07, -0.03],
        ]
    )[:length]
    state = model.SwitchCountWealth()
    for row in logs:
        state.advance(row)
    np.testing.assert_allclose(state.predict(), sequence_oracle(logs), rtol=0, atol=1e-12)


def test_all119_months_match_independent_hazard_integration():
    primary, independent = model.SwitchCountWealth(), model.HazardQuadrature()
    logs = np.random.default_rng(124).normal(0.01, 0.11, (119, 6))
    for row in logs:
        primary.advance(row)
        independent.advance(row)
        assert float(abs(primary.predict() - independent.predict()).max()) <= 1e-12
        assert abs(float(primary.log_score) - independent.game_log_score()) <= 1e-12
        assert abs(primary.expected_hazard() - independent.expected_hazard()) <= 1e-12
    assert primary.snapshot()["persistent_primary_mass_cells"] == 714
    assert all(v > 0 for row in primary.mass for v in row)


def test_independent_nodes_cover_forecast_degree_and_exact_prior_moments(monkeypatch):
    oracle = model.HazardQuadrature()
    assert 2 * model.QUADRATURE - 1 >= model.MAX_PERIODS
    for degree in range(120):
        exact = Fraction(math.comb(2 * degree, degree), 4**degree)
        assert abs(float(np.mean(oracle.hazard**degree)) - float(exact)) < 1e-14
    monkeypatch.setattr(model, "QUADRATURE", 59)
    with pytest.raises(QuantError, match="forecast numerator"):
        model.HazardQuadrature()


def test_equal_source_scores_keep_identity_and_hazard_prior():
    primary, independent = model.SwitchCountWealth(), model.HazardQuadrature()
    for _ in range(7):
        logs = np.full(6, 0.04)
        primary.advance(logs)
        independent.advance(logs)
    np.testing.assert_allclose(primary.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-15)
    assert abs(primary.expected_hazard() - 0.5) < 1e-15
    assert abs(float(primary.log_score) - 0.28) < 1e-14
    assert abs(independent.game_log_score() - 0.28) < 1e-14


def test_common_log_scale_does_not_change_allocations():
    a, b = model.SwitchCountWealth(), model.SwitchCountWealth()
    for row in np.random.default_rng(7).normal(0, 0.1, (4, 6)):
        a.advance(row)
        b.advance(row + 200)
    np.testing.assert_allclose(a.predict(), b.predict(), rtol=0, atol=1e-12)
    assert abs(float(b.log_score - a.log_score) - 800) < 1e-12


@pytest.mark.parametrize(
    "logs", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf), np.ones(6, dtype=bool)]
)
def test_invalid_source_month_cannot_be_filled(logs):
    with pytest.raises(QuantError):
        model.SwitchCountWealth().advance(logs)


def test_missing_month_and_future_native_opinion_fail_explicitly():
    calendar = sessions("2016-10-05", "2017-04-28")
    known = periods(calendar)
    with pytest.raises(QuantError, match="month is missing"):
        model.build_targets(calendar, MemoryOpinions(), known[:-1])

    class FutureOpinions(MemoryOpinions):
        def observe(self, day):
            return replace(super().observe(day), native_known_at=(next_session(day),) * 6)

    with pytest.raises(QuantError, match="then-known"):
        model.build_targets(calendar, FutureOpinions(), known)


def test_initial_partial_month_untrained_and_next_period_clock():
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    result = model.build_targets(
        calendar, MemoryOpinions(), periods(calendar), retain=records.append
    )
    assert [row["complete_source_intervals"] for row in result.audits] == [0, 0, 1]
    assert result.audits[0]["period_update"] is result.audits[1]["period_update"] is None
    assert (
        result.audits[2]["period_update"]["forecast_before_assimilation"]
        == result.audits[1]["expert_weights"]
    )
    assert len(records) == 6
    assert records[-2]["matured_source_interval"]["start_after"] == "2016-10-31"
    assert result.proof["all_source_inputs_matured_before_next_identity_forecast"]


def test_inputs_retained_before_deliberate_learning_failure(monkeypatch):
    def failed(*args, **kwargs):
        raise QuantError("deliberate new-state failure")

    monkeypatch.setattr(model.SwitchCountWealth, "advance", failed)
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    with pytest.raises(QuantError, match="deliberate"):
        model.build_targets(calendar, MemoryOpinions(), periods(calendar), retain=records.append)
    assert len(records) == 5 and records[-1]["phase"] == "before_decision"
    assert records[-1]["matured_source_interval"]["end"] == "2016-11-30"


@pytest.fixture
def data(market_factory):
    return market_factory(start="2016-10-05", end="2020-04-30", symbols=model.ASSETS)


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
    return model.run(
        data,
        prepared,
        "2020-02-04",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
    )


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_full_mixed_asset_cash_and_BT(data, prepared, cost, delay):
    result = run(data, prepared, cost, delay)
    seen = []
    checked = model.verify_run(
        data,
        prepared,
        result,
        "2020-02-04",
        "2020-03-31",
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
    assert result.result.weights.iloc[: delay - 1].eq(0).all().all()
    assert (result.result.weights.iloc[delay - 1 :].gt(0).sum(axis=1) >= 3).all()
    assert (result.result.frame["cash"] >= 0).all()
    assert result.result.frame["risk_free"].equals(data.risk_free.loc[result.result.frame.index])
    independent = independent_equity(
        data,
        result.issued,
        "2020-02-04",
        "2020-03-31",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert float(abs(independent["equity"] - result.result.frame["equity"]).max()) < 1e-6
    assert result.decisions[0]["assessment"]["source_master_known_at"] == "2020-01-31"
    assert result.decisions[0]["assessment"]["complete_source_intervals"] > 0


@pytest.mark.parametrize(
    "field", ["equity", "cash", "gross_exposure", "orders", "risk_free", "position"]
)
def test_cash_comparison_operands_retained_before_failure(data, prepared, field):
    output = run(data, prepared)
    frame, positions = output.result.frame.copy(), output.result.weights.copy()
    if field == "position":
        positions.iloc[1, 0] = np.nextafter(float(positions.iloc[1, 0]), np.inf)
    elif field == "orders":
        frame.loc[frame.index[1], field] += 1
    else:
        frame.loc[frame.index[1], field] = np.nextafter(float(frame.iloc[1][field]), np.inf)
    corrupt = replace(output, result=BacktestResult(frame, positions))
    seen = []
    with pytest.raises(QuantError, match="cash-engine replay"):
        model.verify_run(
            data,
            prepared,
            corrupt,
            "2020-02-04",
            "2020-03-31",
            5,
            1,
            retain_comparison=lambda a, b, report: seen.append((a, b, report)),
        )
    assert len(seen) == 1 and seen[0][0].frame.equals(frame)


def test_actual_file_record_and_cash_Adapters_do_not_rebind_wrapper_recursively(
    data,
    prepared,
    tmp_path,
    monkeypatch,
):
    output = run(data, prepared)
    monkeypatch.setattr(driver, "reference", lambda path: {"path": path.name, "sha256": "a" * 64})
    artifacts = driver.write_run(tmp_path, output)
    assert len(artifacts) == 5
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    report = model.comparison_report(output.result, output.result)
    driver.retain_cash_comparison(comparison, output.result, output.result, report)
    assert read_json(comparison / "comparison.json")["retained_before_acceptance_assertion"]
    assert len(list(comparison.iterdir())) == 5


@pytest.mark.parametrize("cut", ["2016-10-31", "2017-04-12", "2018-10-05"])
def test_source_learning_prefix_and_future_labels_are_causal(data, prepared, cut):
    day = pd.Timestamp(cut)
    prefix = model.build_targets(
        data.close.loc[:day].index,
        MemoryOpinions(),
        periods(data.close.index),
    )
    pd.testing.assert_frame_equal(prepared.targets.loc[:day], prefix.targets, check_exact=True)
    assert prefix.audits == [
        a for a in prepared.audits if pd.Timestamp(a["decision_session"]) <= day
    ]
    changed = [
        {**row, "canonical_log_relatives": [0.6] * 6} if pd.Timestamp(row["end"]) > day else row
        for row in periods(data.close.index)
    ]
    full = model.build_targets(
        data.close.index,
        MemoryOpinions(future_after=day),
        changed,
    )
    pd.testing.assert_frame_equal(
        prepared.targets.loc[:day], full.targets.loc[:day], check_exact=True
    )
    assert prefix.audits == [a for a in full.audits if pd.Timestamp(a["decision_session"]) <= day]


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
@pytest.mark.parametrize("cut", ["2020-02-28", "2020-03-13"])
def test_capital_prefix_has_original_initialization_and_holding_feedback(
    data, prepared, cost, delay, cut
):
    day = pd.Timestamp(cut)
    full = run(data, prepared, cost, delay)
    prefix_source = model.PreparedSourceTargets(
        prepared.targets.loc[:day],
        [a for a in prepared.audits if pd.Timestamp(a["decision_session"]) <= day],
        prepared.proof,
    )
    partial = model.run(
        clipped(data, day),
        prefix_source,
        "2020-02-04",
        cut,
        cost_bps=cost,
        delay=delay,
    )
    for a, b in (
        (full.result.frame, partial.result.frame),
        (full.result.weights, partial.result.weights),
        (full.desired, partial.desired),
        (full.issued, partial.issued),
    ):
        pd.testing.assert_frame_equal(a.loc[:day], b.loc[:day], check_exact=True)
        assert np.array_equal(np.signbit(a.loc[:day]), np.signbit(b.loc[:day]))
    assert partial.decisions == [
        r for r in full.decisions if pd.Timestamp(r["signal_session"]) <= day
    ]


def test_cached_opinion_Adapter_ignores_old_learning_state(tmp_path, monkeypatch):
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in sessions("2016-10-06", "2026-10-05") if is_month_end(d)],
    ]
    records = []
    for day in days:
        opinion = MemoryOpinions().observe(day)
        records.extend(
            [
                {
                    "phase": "before_decision",
                    "decision_session": str(day.date()),
                    "known_target_dates": [str(d.date()) for d in opinion.native_known_at],
                    "known_targets_hex": [[float(x).hex() for x in row] for row in opinion.targets],
                    "state": "do_not_use_this_old_learner_state",
                },
                {"phase": "after_decision", "decision_session": str(day.date())},
            ]
        )
    path = tmp_path / "opinions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    monkeypatch.setattr(driver.base, "checked_reference", lambda ref: path)
    feed = driver.SavedOpinionFeed({"cached_complete_policy_opinions": {}})
    row = feed.observe(days[0])
    row.check(days[0])
    assert len(feed.rows) == 121 and row.targets.shape == (6, 14)


def test_actual_packet_entry_defers_source_learning(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-07T02:26:00Z",
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
            {**body, "created_at": "2026-11-07T02:26:01Z"}
            if path == driver.PACKET
            else {
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            }
        ),
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3


def test_old_policy_integrals_and_learners_are_not_replayed(data, monkeypatch):
    from us_quant import (
        confidence_rated_prod_v1,
        expert_five_net_wealth_mixture_v1,
        expert_net_wealth_mixture_v1,
        main_risk_confidence_prod_v1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A fixed-policy integral or closed source learner cannot replay.")

    for name in ("weights", "update"):
        monkeypatch.setattr(confidence_rated_prod_v1.ConfidenceProd, name, forbidden)
        monkeypatch.setattr(expert_net_wealth_mixture_v1.TetrahedralMoments, name, forbidden)
        monkeypatch.setattr(expert_five_net_wealth_mixture_v1.PackedFiveMoments, name, forbidden)
    monkeypatch.setattr(confidence_rated_prod_v1, "build_targets", forbidden)
    monkeypatch.setattr(expert_five_net_wealth_mixture_v1, "independent_prefix_weights", forbidden)
    monkeypatch.setattr(main_risk_confidence_prod_v1, "run", forbidden)
    result = model.build_targets(data.close.index, MemoryOpinions(), periods(data.close.index))
    assert result.proof["complete_source_intervals"] > 0
