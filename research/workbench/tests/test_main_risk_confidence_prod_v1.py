from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from us_quant import main_risk_confidence_prod_v1 as model
from us_quant import ssd_main_risk_confidence_prod_v1 as driver
from us_quant.backtest import BacktestResult
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation

ASSETS = (
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)


@dataclass(frozen=True)
class MemoryExpert:
    slot: int
    assets: tuple[str, ...]
    end: pd.Timestamp
    shadow_multiplier: float = 1.0
    future_after: pd.Timestamp | None = None

    @property
    def identity(self):
        return ExpertIdentity(
            model.EXPERT_IDS[self.slot],
            model.MECHANISMS[self.slot],
            "2026-11-07T00:31:44.853Z",
        )

    def observe(self, as_of, lookback):
        if as_of > self.end or type(lookback) is not int or lookback < 2:
            raise QuantError("The memory opinion Adapter cannot invent source coverage.")
        index = sessions("2016-10-06", as_of)[-lookback:]
        raw = np.arange(1, len(self.assets) + 1, dtype=float)
        raw = np.roll(raw, as_of.month + 2 * self.slot)
        if self.future_after is not None and as_of > self.future_after:
            raw = np.roll(raw, 3)
        target = pd.Series(0.98 * raw / raw.sum(), index=self.assets)
        if self.slot == 5:
            target[:] = 0
            target.loc["BIL"] = 0.98
        return ExpertObservation(
            self.identity,
            as_of,
            target,
            as_of,
            pd.Series((self.slot + 1) * 0.0001 * self.shadow_multiplier, index=index),
            pd.Series(0.00004, index=index, name="risk_free"),
            len(index) == lookback,
        )


@pytest.fixture
def data(market_factory):
    return market_factory(start="2019-12-02", end="2020-05-29", symbols=ASSETS)


def bundle(data):
    ends = [d for d in sessions("2016-10-06", data.close.index[-1]) if is_month_end(d)]
    return model.SourceBundle(
        tuple(MemoryExpert(i, ASSETS, data.close.index[-1]) for i in range(6)),
        tuple(
            {
                "start_after": str(start.date()),
                "end": str(end.date()),
                "input_loss_labels": np.roll(np.linspace(0.1, 0.9, 6), end.month).tolist(),
            }
            for start, end in zip(ends[:-1], ends[1:], strict=True)
        ),
    )


def run(data, cost=5, delay=1, **kwargs):
    return model.run(
        data,
        bundle(data),
        "2020-02-04",
        "2020-04-30",
        cost_bps=cost,
        delay=delay,
        **kwargs,
    )


def clipped(data, end):
    return MarketData(
        data.open.loc[:end].copy(),
        data.close.loc[:end].copy(),
        data.raw_close.loc[:end].copy(),
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end].copy(),
    )


def test_frozen_one_main_participation_law():
    policy = read_json(model.POLICY)
    model.validate(policy)
    for key, changed in (
        ("actual_weight_band", 0.024),
        ("path_local_learner_reset", False),
        ("initial_partial_month_assimilated", True),
        ("participation_is_capital_cap_or_drawdown_guarantee", True),
    ):
        with pytest.raises(QuantError):
            model.validate({**policy, key: changed})


def test_all_closes_not_only_month_ends_define_main_peak():
    context = model.MainRiskContext(pd.Timestamp("2020-02-03"))
    for day in sessions("2020-02-04", "2020-02-28"):
        context.observe_close(day, 12000.0 if day == pd.Timestamp("2020-02-12") else 11000.0)
    snapshot = context.snapshot(pd.Timestamp("2020-02-28"), 11000)
    rated = model.confidence(snapshot)
    assert snapshot["actual_main_closing_peak_hex"] == float(12000).hex()
    assert rated[:5].tolist() == [1] * 5
    assert rated[5] == 1 - (11000 - 0.85 * 12000) / (0.15 * 11000)
    assert rated[5] > 0


@pytest.mark.parametrize("bad", ["2020-02-03", "2020-02-05", "2020-02-04T00:00:00Z"])
def test_main_mark_Interface_rejects_repeated_missing_or_zoned_closes(bad):
    context = model.MainRiskContext(pd.Timestamp("2020-02-03"))
    with pytest.raises(QuantError):
        context.observe_close(pd.Timestamp(bad), 10000)


def test_main_context_does_not_read_future_or_shadow_wealth():
    context = model.MainRiskContext(pd.Timestamp("2020-02-03"))
    for nav in (np.nan, np.inf, 0, -1):
        with pytest.raises(QuantError):
            context.observe_close(pd.Timestamp("2020-02-04"), nav)
    with pytest.raises(QuantError):
        context.snapshot(pd.Timestamp("2020-02-04"), 10000)
    with pytest.raises(QuantError):
        context.snapshot(pd.Timestamp("2020-02-03"), 20000)


def test_prior_origin_confidence_not_current_end_weights_loss(data):
    policy = model.MainRiskProdPolicy(bundle(data), ASSETS, pd.Timestamp("2020-02-03"))

    def decide(day, nav, initial=False):
        return policy.decide(
            model.monthly.CapitalSnapshot(
                pd.Timestamp(day),
                pd.Series(0.0, index=ASSETS),
                float(nav),
                initial,
            )
        ).assessment

    initial = decide("2020-02-03", 10000, True)
    assert initial["complete_learning_intervals"] == 0
    assert initial["current_confidences"][-1] == 0
    for day in sessions("2020-02-04", "2020-02-28"):
        policy.context.observe_close(day, 9500.0)
    first = decide("2020-02-28", 9500)
    assert first["period_update"] is None
    for day in sessions("2020-03-02", "2020-03-31"):
        policy.context.observe_close(day, 12000.0 if day.day == 12 else 9000.0)
    second = decide("2020-03-31", 9000)
    update = second["period_update"]
    assert update["start_after"] == "2020-02-28"
    assert update["confidence_held_from_prior_origin"] == first["current_confidences"]
    assert update["confidence_held_from_prior_origin"] != second["current_confidences"]
    assert second["current_confidences"][-1] == 1
    assert second["complete_learning_intervals"] == 1


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_mixed_monthly_main_cash_and_independent_BT(data, cost, delay):
    output = run(data, cost, delay)
    seen = []
    checked = model.verify_run(
        data,
        output,
        "2020-02-04",
        "2020-04-30",
        cost,
        delay,
        retain_comparison=lambda a, b, report: seen.append((a, b, report)),
    )
    assert len(seen) == 1
    assert checked["complete_month_updates"] == 2
    assert checked["all_main_closing_peak_inputs_verified"]
    assert all(
        v["binary64_bit_differences"] == 0
        for v in seen[0][2]["all_frame_column_diagnostics"].values()
    )
    waiting = output.result.weights.iloc[: delay - 1]
    assert waiting.eq(0).all().all()
    assert (output.result.weights.iloc[delay - 1 :].gt(0).sum(axis=1) >= 3).all()
    assert output.result.frame.iloc[: delay - 1]["equity"].eq(10000).all()
    assert output.result.frame.iloc[: delay - 1]["cash"].eq(10000).all()
    assert output.result.frame.iloc[: delay - 1]["cost"].eq(0).all()
    assert output.result.frame.iloc[: delay - 1]["orders"].eq(0).all()
    assert (output.result.frame["cash"] >= 0).all()
    assert output.result.frame.index.equals(sessions("2020-02-04", "2020-04-30"))
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])
    assert [r["signal_session"] for r in output.decisions] == [
        "2020-02-03",
        "2020-02-28",
        "2020-03-31",
        "2020-04-30",
    ]
    independent = independent_equity(
        data,
        output.issued,
        "2020-02-04",
        "2020-04-30",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert float(abs(independent["equity"] - output.result.frame["equity"]).max()) < 1e-6
    if delay == 2:
        assert output.result.frame.iloc[0]["equity"] == 10000


def test_canonical_gross_and_raw_auxiliary_both_retained(data):
    output = run(data)
    assert np.array_equal(
        output.result.frame["gross_exposure"],
        [float(row.copy().sum()) for row in output.result.weights.to_numpy()],
    )
    assert output.raw_kernel_auxiliary_gross.index.equals(output.result.frame.index)


def test_path_local_learner_starts_fresh(data):
    warm = run(data)
    fresh = model.run(
        data,
        bundle(data),
        "2020-03-03",
        "2020-04-30",
        cost_bps=5,
        delay=1,
    )
    first = fresh.decisions[0]["assessment"]
    assert warm.decisions[-1]["assessment"]["complete_learning_intervals"] == 2
    assert first["learner_state"] == model.learning.ConfidenceProd().snapshot()
    assert first["main_risk_context"]["initial_main_capital_anchor"] == "2020-03-02"
    assert first["main_risk_context"]["actual_main_return_rows"] == 0
    assert first["main_risk_context"]["actual_main_closing_peak_hex"] == float(10000).hex()
    assert first["complete_learning_intervals"] == 0
    assert fresh.decisions[1]["assessment"]["period_update"] is None
    assert fresh.decisions[-1]["assessment"]["complete_learning_intervals"] == 1
    assert all(
        introduced.startswith("2026-11-07") for introduced in first["source_actual_introductions"]
    )


def test_source_shadow_changes_are_not_main_risk_changes(data):
    original = bundle(data)
    changed = replace(
        original, experts=tuple(replace(e, shadow_multiplier=100) for e in original.experts)
    )
    before = run(data)
    after = model.run(
        data,
        changed,
        "2020-02-04",
        "2020-04-30",
        cost_bps=5,
        delay=1,
    )
    pd.testing.assert_frame_equal(before.result.frame, after.result.frame, check_exact=True)
    assert before.decisions == after.decisions


def test_missing_or_duplicate_sealed_month_fails_explicitly(data):
    original = bundle(data)
    missing = replace(
        original, periods=tuple(p for p in original.periods if p["end"] != "2020-03-31")
    )
    with pytest.raises(QuantError, match="matured sealed month"):
        model.run(data, missing, "2020-02-04", "2020-04-30", cost_bps=5, delay=1)
    with pytest.raises(QuantError, match="cannot occur twice"):
        model.MainRiskProdPolicy(
            replace(original, periods=(*original.periods, original.periods[-1])),
            ASSETS,
            pd.Timestamp("2020-02-03"),
        )


@pytest.mark.parametrize("case", ["missing_target", "future_target", "RF_change", "history_hole"])
def test_unavailable_source_cannot_be_filled(data, monkeypatch, case):
    observe = MemoryExpert.observe

    def invalid(self, day, lookback):
        row = observe(self, day, lookback)
        if self.slot != 2:
            return row
        if case == "missing_target":
            return replace(row, target=None)
        if case == "future_target":
            return replace(row, target_known_at=day + pd.Timedelta(days=1))
        if case == "RF_change":
            return replace(row, risk_free=row.risk_free + 1e-6)
        return replace(row, net_returns=row.net_returns.iloc[1:])

    monkeypatch.setattr(MemoryExpert, "observe", invalid)
    with pytest.raises(QuantError, match="complete-policy observations"):
        run(data)


def test_raw_mass_is_not_renormalized_and_independent_reduction_matches():
    state = model.learning.ConfidenceProd()
    history = []
    for i in range(119):
        labels = np.roll(np.array([0, 0, 0, 1, 1, 1], dtype=float), i % 3)
        participation = np.array([1.0] * 5 + [(i % 7) / 6])
        history.append((labels, participation))
        state.update(labels, participation)
    other = model.learning.independent_trace(history)[-1]
    assert abs(float(sum(state.mass)) - 1) > 1e-8
    assert (
        np.max(
            abs(
                np.array([float(x) for x in state.mass])
                - np.array([float(x) for x in other["mass_after_decimal"]])
            )
        )
        < 1e-12
    )


@pytest.mark.parametrize(
    "field",
    [
        "equity",
        "return",
        "cash",
        "gross_exposure",
        "turnover",
        "cost",
        "orders",
        "risk_free",
        "position",
    ],
)
def test_comparison_operands_retained_before_deliberate_failure(data, field):
    output = run(data)
    frame, held = output.result.frame.copy(), output.result.weights.copy()
    if field == "position":
        held.iloc[1, 0] = np.nextafter(float(held.iloc[1, 0]), np.inf)
    elif field == "orders":
        frame.loc[frame.index[1], field] += 1
    else:
        frame.loc[frame.index[1], field] = np.nextafter(float(frame.iloc[1][field]), np.inf)
    corrupted = replace(output, result=BacktestResult(frame, held))
    seen = []
    with pytest.raises(QuantError, match="cash-engine replay"):
        model.verify_run(
            data,
            corrupted,
            "2020-02-04",
            "2020-04-30",
            5,
            1,
            retain_comparison=lambda a, b, report: seen.append((a, b, report)),
        )
    assert len(seen) == 1
    assert seen[0][0].frame.equals(frame) and seen[0][0].weights.equals(held)
    assert not (seen[0][2]["full_frame_equals"] and seen[0][2]["full_positions_equal"])


def test_actual_file_comparison_Adapter_retains_four_operands(data, tmp_path, monkeypatch):
    output = run(data)
    monkeypatch.setattr(driver, "reference", lambda p: {"path": p.name, "sha256": "a" * 64})
    report = model.comparison_report(output.result, output.result)
    driver.retain_cash_comparison(tmp_path, output.result, output.result, report)
    receipt = read_json(tmp_path / "comparison.json")
    assert receipt["retained_before_acceptance_assertion"]
    assert len(receipt["artifacts"]) == 4
    for name in ("actual-account.csv", "cash-engine-account.csv"):
        actual = pd.read_csv(
            tmp_path / name,
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        assert actual.index.equals(output.result.frame.index)
        actual.index = pd.DatetimeIndex(actual.index, freq=output.result.frame.index.freq)
        pd.testing.assert_frame_equal(actual, output.result.frame, check_exact=True)


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
@pytest.mark.parametrize("cut", ["2020-02-28", "2020-03-13"])
def test_full_capital_and_learning_prefix_is_exact(data, cost, delay, cut):
    day = pd.Timestamp(cut)
    output = run(data, cost, delay)
    partial = model.run(
        clipped(data, day),
        bundle(data),
        "2020-02-04",
        cut,
        cost_bps=cost,
        delay=delay,
    )
    for whole, prefix in (
        (output.result.frame, partial.result.frame),
        (output.result.weights, partial.result.weights),
        (output.desired, partial.desired),
        (output.issued, partial.issued),
    ):
        pd.testing.assert_frame_equal(whole.loc[:day], prefix.loc[:day], check_exact=True)
        assert np.array_equal(np.signbit(whole.loc[:day]), np.signbit(prefix.loc[:day]))
    assert partial.decisions == [
        r for r in output.decisions if pd.Timestamp(r["signal_session"]) <= day
    ]


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_future_market_labels_and_opinions_do_not_change_past_main(data, cost, delay):
    day = pd.Timestamp("2020-03-13")
    before = run(data, cost, delay)
    changed = clipped(data, data.close.index[-1])
    changed.open.loc[changed.open.index > day] *= 1.07
    changed.close.loc[changed.close.index > day] *= 1.03
    changed.raw_close.loc[changed.raw_close.index > day] *= 1.03
    original = bundle(data)
    future = replace(
        original,
        experts=tuple(replace(e, future_after=day) for e in original.experts),
        periods=tuple(
            {**p, "input_loss_labels": [1.0] * 6} if pd.Timestamp(p["end"]) > day else p
            for p in original.periods
        ),
    )
    after = model.run(
        changed,
        future,
        "2020-02-04",
        "2020-04-30",
        cost_bps=cost,
        delay=delay,
    )
    for a, b in (
        (before.result.frame, after.result.frame),
        (before.result.weights, after.result.weights),
        (before.issued, after.issued),
    ):
        pd.testing.assert_frame_equal(a.loc[:day], b.loc[:day], check_exact=True)
    assert [r for r in before.decisions if pd.Timestamp(r["signal_session"]) <= day] == [
        r for r in after.decisions if pd.Timestamp(r["signal_session"]) <= day
    ]


def test_closes_are_observed_before_month_end_decisions(data):
    trace = []
    output = run(data, retain=trace.append)
    closes = [r for r in trace if r["kind"] == "main_close"]
    assert len(closes) == len(output.result.frame)
    for i, row in enumerate(trace):
        if row["kind"] == "main_risk_learning_input" and row["decision_session"] != "2020-02-03":
            assert trace[i - 1]["kind"] == "main_close"
            assert trace[i - 1]["session"] == row["decision_session"]
    for row in closes:
        known = output.result.frame.loc[: row["session"], "equity"]
        assert row["actual_main_prefix_peak"] == max(10000.0, float(known.max()))


def test_old_source_master_shadow_and_shields_are_not_executed(data, monkeypatch):
    from us_quant import execution_gap_shield_v1, execution_gap_shield_v2

    def forbidden(*args, **kwargs):
        raise AssertionError("A closed registered allocator or shadow cannot be replayed.")

    monkeypatch.setattr(model.learning, "build_targets", forbidden)
    monkeypatch.setattr(model.learning, "confidences", forbidden)
    monkeypatch.setattr(driver.source, "source_inputs", forbidden)
    monkeypatch.setattr(execution_gap_shield_v1, "run", forbidden)
    monkeypatch.setattr(execution_gap_shield_v2, "run", forbidden)
    assert len(run(data).decisions) == 4


def test_actual_packet_entry_defers_historical_outputs(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-07T00:31:44.853Z",
        "body": {"candidate": driver.CANDIDATE, "frozen": frozen},
    }
    ready = {"path": "synthetic-readiness.json", "sha256": "b" * 64}
    body = driver.packet_body(frozen, registration, ready)
    assert body["entries"][0]["candidate_id"] == model.ID
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert body["candidate_accounts_or_metrics_computed"] is False
    monkeypatch.setattr(driver, "events", lambda k: [registration] if k == "registered" else [])
    monkeypatch.setattr(driver, "reference", lambda p: ready)
    monkeypatch.setattr(
        driver,
        "read_json",
        lambda p: (
            {**body, "created_at": "2026-11-07T00:31:45Z"}
            if p == driver.PACKET
            else {
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            }
        ),
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3
