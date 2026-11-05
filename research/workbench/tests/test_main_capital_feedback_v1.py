from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import main_capital_feedback_v1 as model
from us_quant import observed_weight_execution as monthly
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation

SYMBOLS = (
    "SPY",
    "IEF",
    "TLT",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)


@pytest.fixture
def data(market_factory):
    return market_factory(start="2019-12-02", end="2021-12-31", symbols=SYMBOLS)


class SyntheticExpert:
    def __init__(self, data, position):
        self._position = position
        self._dates = data.close.index
        self._rf = data.risk_free.copy()
        elapsed = np.arange(len(self._dates), dtype=float)
        self._net = pd.Series(
            0.0002
            + 0.00002 * position
            + 0.005 * np.sin(elapsed / (3 + position) + position)
            + 0.002 * np.cos(elapsed / 11),
            index=self._dates,
        )
        self.identity = ExpertIdentity(
            model.pool.EXPERT_IDS[position],
            model.pool.MECHANISMS[position],
            "2026-11-04T00:00:00Z",
        )
        self._universe = data.close.columns

    def observe(self, as_of, lookback):
        if as_of not in self._dates:
            raise QuantError("Synthetic source day is not covered.")
        known_at = as_of
        if self._position % 2:
            prior = [day for day in self._dates if day <= as_of and is_month_end(day)]
            known_at = prior[-1] if prior else self._dates[0]
        target = pd.Series(0.0, index=self._universe)
        index = (self._position + known_at.month) % len(target)
        target.iloc[index] = 0.98
        net = self._net.loc[:as_of].tail(lookback)
        return ExpertObservation(
            self.identity,
            as_of,
            target,
            known_at,
            net.copy(),
            self._rf.loc[net.index].copy(),
            len(net) == lookback,
        )


def experts(data):
    return tuple(SyntheticExpert(data, i) for i in range(8))


def policy(data, sources=None):
    return model.CloseMarkedPairPolicy(
        experts(data) if sources is None else sources,
        tuple(data.close.columns),
        data.risk_free,
    )


def clipped(data, end):
    return MarketData(
        data.open.loc[:end].copy(),
        data.close.loc[:end].copy(),
        data.raw_close.loc[:end].copy(),
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end].copy(),
    )


def targets(data):
    frame = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for day in frame.index:
        if is_month_end(day):
            frame.loc[day] = 0.0
            frame.loc[day, "SOXX" if day.month % 2 else "GLD"] = 0.98
    return frame


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_archived_adapter_matches_original_monthly_money_and_timing_exactly(data, cost, delay):
    baseline = targets(data)
    original, issued, decisions = monthly.run(
        data,
        baseline,
        monthly.CANDIDATES[0],
        read_json(monthly.POLICY),
        "2020-06-02",
        "2021-12-30",
        cost,
        delay,
    )
    after = model.run(
        data,
        model.ArchivedOpinionPolicy(baseline),
        "2020-06-02",
        "2021-12-30",
        cost_bps=cost,
        delay=delay,
    )
    pd.testing.assert_frame_equal(original.frame, after.result.frame, check_exact=True)
    pd.testing.assert_frame_equal(original.weights, after.result.weights, check_exact=True)
    pd.testing.assert_frame_equal(issued, after.issued, check_exact=True)
    for before, current in zip(decisions, after.decisions, strict=True):
        assert all(current[key] == value for key, value in before.items())


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_actual_pair_policy_uses_main_state_and_matches_independent_bt(data, cost, delay):
    output = model.run(data, policy(data), "2020-06-02", "2021-12-30", cost_bps=cost, delay=delay)
    assert model.verify_capital_inputs(output)[
        "main_dollar_holdings_cash_and_equity_match_actual_close"
    ]
    assert output.result.frame.index.equals(sessions("2020-06-02", "2021-12-30"))
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])
    assert output.decisions[0]["signal_session"] == "2020-06-01"
    assert output.decisions[0]["main_equity_at_decision"] == 10000
    assert output.decisions[0]["main_cash_at_decision"] == 10000
    assert output.decisions[0]["assessment"]["expert_shadow_last_session"] == ["2020-06-01"] * 8
    assert all(
        row["initial_capital"] or is_month_end(pd.Timestamp(row["signal_session"]))
        for row in output.decisions
    )
    assert (output.result.frame["cash"] >= 0).all()
    assert (output.result.frame["gross_exposure"] <= 1).all()
    independent = independent_equity(
        data,
        output.issued,
        "2020-06-02",
        "2021-12-30",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert float(abs(independent["equity"] - output.result.frame["equity"]).max()) < 1e-5
    for row in output.decisions:
        if row["assessment"]["selection"]["scores_available"]:
            for candidate in row["assessment"]["selection"]["pairs"]:
                if "cost_estimate" in candidate:
                    assert (
                        candidate["cost_estimate"]["capital_usd"]
                        == (row["main_equity_at_decision"])
                    )
                    assert candidate["cost_estimation_is_actual_future_execution"] is False
    if delay == 2:
        assert output.result.frame.iloc[0]["equity"] == 10000
        assert output.result.frame.iloc[0]["cost"] == 0


@pytest.mark.parametrize("cut", [180, 240, 300, 360, 420])
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_full_capital_prefix_and_future_price_source_perturbation(data, cut, cost, delay):
    end = data.close.index[cut]
    original = model.run(data, policy(data), "2020-06-02", "2021-12-30", cost_bps=cost, delay=delay)
    prefix_data = clipped(data, end)
    prefix = model.run(
        prefix_data,
        policy(prefix_data),
        "2020-06-02",
        str(end.date()),
        cost_bps=cost,
        delay=delay,
    )
    changed = clipped(data, data.close.index[-1])
    changed.open.loc[changed.open.index > end] *= 1.4
    changed.close.loc[changed.close.index > end] *= 0.8
    changed.raw_close.loc[changed.raw_close.index > end] *= 0.8
    sources = experts(changed)
    for source in sources:
        source._net.loc[source._net.index > end] = -0.75
    altered = model.run(
        changed,
        policy(changed, sources),
        "2020-06-02",
        "2021-12-30",
        cost_bps=cost,
        delay=delay,
    )
    for other in (prefix, altered):
        pd.testing.assert_frame_equal(
            original.result.frame.loc[:end], other.result.frame.loc[:end], check_exact=True
        )
        pd.testing.assert_frame_equal(
            original.result.weights.loc[:end], other.result.weights.loc[:end], check_exact=True
        )
        pd.testing.assert_frame_equal(
            original.issued.loc[:end], other.issued.loc[:end], check_exact=True
        )
        before = [r for r in original.decisions if pd.Timestamp(r["signal_session"]) <= end]
        after = [r for r in other.decisions if pd.Timestamp(r["signal_session"]) <= end]
        assert before == after


def test_future_opening_gap_cannot_change_initial_decision(data):
    first = model.run(data, policy(data), "2020-06-02", "2020-09-30", cost_bps=5, delay=1)
    changed = clipped(data, data.close.index[-1])
    changed.open.loc["2020-06-02"] *= 1.3
    after = model.run(changed, policy(changed), "2020-06-02", "2020-09-30", cost_bps=5, delay=1)
    assert first.decisions[0] == after.decisions[0]
    assert first.result.frame.iloc[0]["equity"] != after.result.frame.iloc[0]["equity"]


def test_learning_state_cannot_be_reused_as_another_fresh_main_account(data):
    chosen = policy(data)
    model.run(data, chosen, "2020-06-02", "2020-09-30", cost_bps=5, delay=1)
    with pytest.raises(QuantError, match="fresh main account"):
        model.run(data, chosen, "2020-06-02", "2020-09-30", cost_bps=5, delay=1)


@pytest.mark.parametrize("failure", ["missing_target", "future_target", "RF", "identity"])
def test_source_failure_is_not_repaired_by_main_capital_feedback(data, failure):
    sources = list(experts(data))
    actual = sources[0].observe

    class Broken:
        def observe(self, day, lookback):
            observed = actual(day, lookback)
            if failure == "missing_target":
                return replace(observed, target=None)
            if failure == "future_target":
                return replace(observed, target_known_at=day + pd.Timedelta(days=1))
            if failure == "RF":
                return replace(observed, risk_free=observed.risk_free + 0.0001)
            return replace(
                observed, identity=replace(observed.identity, identifier="not_the_frozen_expert")
            )

    sources[0] = Broken()
    with pytest.raises(QuantError):
        model.run(
            data,
            policy(data, tuple(sources)),
            "2020-06-02",
            "2020-09-30",
            cost_bps=5,
            delay=1,
        )


def test_partial_chronological_trace_survives_a_decision_exception(data):
    retained = []
    chosen = policy(data)
    actual = chosen.decide
    calls = 0

    class FailsAfterInitialization:
        def decide(self, capital):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise QuantError("Synthetic interruption preserves previous close rows.")
            return actual(capital)

    with pytest.raises(QuantError, match="Synthetic interruption"):
        model.run(
            data,
            FailsAfterInitialization(),
            "2020-06-02",
            "2020-09-30",
            cost_bps=5,
            delay=1,
            retain=retained.append,
        )
    closes = [row for row in retained if row["kind"] == "main_close"]
    assert len(closes) > 1
    assert closes[-1]["session"] == "2020-06-30"
    assert retained[0]["kind"] == "decision"


@pytest.mark.parametrize("cost,delay", [(5, 2), (20, 1), (0, 1), (5, 0), (5, True)])
def test_execution_scenarios_cannot_be_spliced_or_softened(data, cost, delay):
    with pytest.raises(QuantError):
        model.run(data, policy(data), "2020-06-02", "2020-09-30", cost_bps=cost, delay=delay)


def test_missing_first_or_last_formal_session_is_not_silently_cropped(data):
    with pytest.raises(QuantError, match="complete formal dates"):
        model.run(data, policy(data), "2020-05-31", "2020-09-30", cost_bps=5, delay=1)


def test_pending_outside_window_is_explicit_not_filled_early(data):
    output = model.run(data, policy(data), "2020-06-02", "2020-06-30", cost_bps=20, delay=2)
    if output.decisions[-1]["reason"] != "hold":
        assert output.decisions[-1]["execution_session"] == "2020-07-02"
        assert list(output.pending_after_window) == ["2020-07-02"]
    assert output.result.frame.index[-1] == pd.Timestamp("2020-06-30")


def test_corrupt_logged_main_capital_is_rejected(data):
    output = model.run(data, policy(data), "2020-06-02", "2020-09-30", cost_bps=5, delay=1)
    output.decisions[-1]["main_cash_at_decision"] += 1
    with pytest.raises(QuantError, match="actual main book"):
        model.verify_capital_inputs(output)


def test_missing_archived_month_end_or_future_seed_is_not_carried(data):
    baseline = targets(data)
    missing = baseline.copy()
    missing.loc["2020-07-31"] = np.nan
    with pytest.raises(QuantError, match="missing post-start"):
        model.ArchivedOpinionPolicy(missing)
    late = baseline.copy()
    late.loc[late.index < pd.Timestamp("2020-07-31")] = np.nan
    with pytest.raises(QuantError, match="future opinion"):
        model.run(
            data,
            model.ArchivedOpinionPolicy(late),
            "2020-06-02",
            "2020-09-30",
            cost_bps=5,
            delay=1,
        )
