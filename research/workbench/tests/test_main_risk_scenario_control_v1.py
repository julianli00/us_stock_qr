import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import main_risk_scenario_control_v1 as model
from us_quant import scenario_main_execution_v1 as execution
from us_quant import ssd_main_risk_scenario_control_v1 as driver
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.data import MarketData


def fixture_market():
    dates = sessions("2015-08-10", "2016-11-30")
    trend = np.arange(len(dates))[:, None] * np.linspace(0.0002, 0.001, 14)
    close = pd.DataFrame(100 * np.exp(trend), index=dates, columns=model.ASSETS)
    opening = close * 0.9999
    return MarketData(
        opening,
        close,
        close.copy(),
        close * 0 + 1000,
        pd.Series(0.0001, index=dates),
    )


class MemoryOpinions:
    def __init__(self):
        self.targets = np.zeros((6, 14))
        for index in range(6):
            self.targets[index, index] = 0.98

    def observe(self, day):
        origin = max(
            [pd.Timestamp("2016-10-05")]
            + [
                d
                for d in (sessions("2016-10-06", day) if day > pd.Timestamp("2016-10-05") else ())
                if is_month_end(d)
            ]
        )
        return model.OpinionSet(origin, self.targets, (origin,) * 6)


def state(day, *, cost=5.0, delay=1, pending=False):
    values = np.zeros(14)
    values[0] = 3000
    desired = np.zeros(14)
    desired[1] = 0.98
    return model.CapitalState(
        day,
        values,
        7000.0,
        10000.0,
        cost,
        delay,
        False,
        desired if pending else None,
        delay if pending else None,
    )


def test_fixed_policy_does_not_claim_a_guarantee():
    policy = driver.read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate(
            {**policy, "historical_scenario_feasibility_is_future_drawdown_guarantee": True}
        )


def test_price_paths_mature_and_future_prices_do_not_change_past():
    data = fixture_market()
    day = pd.Timestamp("2016-10-05")
    first = model.PriceScenarioFeed(data.open, data.close).observe(day)
    changed = data.close.copy()
    changed.loc[changed.index > day] *= 2
    second = model.PriceScenarioFeed(data.open, changed).observe(day)
    np.testing.assert_array_equal(first.opening, second.opening)
    np.testing.assert_array_equal(first.closing, second.closing)
    assert all(end <= day for end in first.ends)
    short = model.PriceScenarioFeed(data.open.iloc[:20], data.close.iloc[:20]).observe(
        data.close.index[19]
    )
    assert isinstance(short, model.UnavailableHistory)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_conservative_envelope_contains_exact_cash_forecasts(cost, delay):
    data = fixture_market()
    day = pd.Timestamp("2016-10-05")
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    current = state(day, cost=cost, delay=delay)
    desired = np.zeros(14)
    desired[2] = 0.98
    lower, upper = model.wealth_envelope(batch, current, desired)
    for index in (0, 41, 84, 125):
        units = current.asset_values.copy()
        cash = current.cash
        actual = []
        for step in range(21):
            if step == delay - 1:
                values, cash, _, turnover, orders = rebalance(
                    units * batch.opening[index, step], cash, desired, cost, 1.0
                )
                if turnover > 0 or orders > 0:
                    units = values / batch.opening[index, step]
            actual.append(float((units * batch.closing[index, step]).sum() + cash))
        actual = np.asarray(actual)
        assert (lower[index] <= actual + 1e-9).all()
        assert (actual <= upper[index] + 1e-9).all()
    assessment = model.assessment(batch, current, desired)
    assert assessment["independent_relative_wealth_error"] <= 1e-12


def test_pending_target_is_preserved_not_overwritten():
    data = fixture_market()
    day = pd.Timestamp("2016-10-05")
    current = state(day, cost=20.0, delay=2, pending=True)
    decision = model.choose(
        current,
        MemoryOpinions().observe(day),
        model.PriceScenarioFeed(data.open, data.close).observe(day),
    )
    assert decision.target is None
    assert decision.assessment["status"] == "pending_request_preserved"
    assert decision.assessment["selected_action"] == "hold_with_committed_pending_target"


def test_zero_current_drawdown_and_all_cash_hold_do_not_invent_interest():
    data = fixture_market()
    day = pd.Timestamp("2016-10-05")
    current = model.CapitalState(day, np.zeros(14), 10000.0, 10000.0, 5.0, 1, True)
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    lower, upper = model.wealth_envelope(batch, current, None)
    np.testing.assert_array_equal(lower, np.full((126, 21), 10000.0))
    np.testing.assert_array_equal(lower, upper)


def test_already_breached_main_peak_is_not_a_feasibility_success():
    data = fixture_market()
    day = pd.Timestamp("2016-10-05")
    current = model.CapitalState(day, np.zeros(14), 8400.0, 10000.0, 5.0, 1, False)
    decision = model.choose(
        current,
        MemoryOpinions().observe(day),
        model.PriceScenarioFeed(data.open, data.close).observe(day),
    )
    assert not decision.assessment["forecast_feasible"]
    assert decision.assessment["status"] == "no_viable_action_defensive_standby_not_a_pass"


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_actual_daily_capital_matches_cash_and_independent_BT(cost, delay):
    data = fixture_market()
    controller = model.Controller(
        model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), cost, delay
    )
    output = execution.run(data, controller, "2016-10-06", "2016-11-30", cost_bps=cost, delay=delay)
    replay = execution.cash_replay(data, output, "2016-10-06", "2016-11-30", cost, delay)
    pd.testing.assert_frame_equal(output.result.frame, replay.frame, check_exact=True)
    pd.testing.assert_frame_equal(output.result.weights, replay.weights, check_exact=True)
    other = independent_equity(
        data,
        output.issued,
        "2016-10-06",
        "2016-11-30",
        capital=10000,
        cost_bps=cost,
        commission=1.0,
        delay=delay,
    )
    assert abs(output.result.frame["equity"] - other["equity"]).max() <= 1e-6
    assert len(output.decisions) == len(output.result.frame) + 1


def test_current_inventory_is_not_an_expert_shadow_book():
    day = pd.Timestamp("2016-10-05")
    current = state(day)
    current.check()
    assert current.nav == 10000
    bad = model.CapitalState(day, np.zeros(14), 10000.0, 9999.0, 5.0, 1, True)
    with pytest.raises(QuantError):
        bad.check()


def oscillating_market():
    data = fixture_market()
    index = np.arange(len(data.close))[:, None]
    asset = np.arange(14)[None, :]
    close = pd.DataFrame(
        100 * np.exp(0.15 * np.sin(index / 9 + asset / 3) + index * 0.0004),
        index=data.close.index,
        columns=model.ASSETS,
    )
    opening = close.shift(1).fillna(close.iloc[0]) * np.exp(0.03 * np.cos(index / 5 + asset / 2))
    return MarketData(opening, close, close.copy(), close * 0 + 1000, data.risk_free)


def points():
    opinion = MemoryOpinions()
    origins = [pd.Timestamp("2016-10-05"), pd.Timestamp("2016-10-31"), pd.Timestamp("2016-11-30")]
    result = []
    for index, day in enumerate(origins):
        rows = np.roll(opinion.targets, index, axis=1)
        p = np.full(6, 1 / 6)
        result.append(
            driver.prior.rules.SourcePoint(
                day,
                tuple(p),
                tuple(tuple(row) for row in rows),
                tuple(p @ rows),
                (day,) * 6,
                max(0, index - 1),
            )
        )
    return tuple(result)


def tape(path, records):
    rows = []
    for point in records:
        day = str(point.origin.date())
        known = [str(d.date()) for d in point.known_at]
        rows.extend(
            [
                {
                    "phase": "before_decision",
                    "decision_session": day,
                    "source_native_known_at": known,
                    "source_targets_hex": [[float(v).hex() for v in row] for row in point.targets],
                },
                {
                    "phase": "after_decision",
                    "decision_session": day,
                    "unified_target_hex": [float(v).hex() for v in point.mixed_target],
                    "audit": {
                        "decision_session": day,
                        "source_native_known_at": known,
                        "expert_weights": point.probabilities,
                        "complete_source_intervals": point.complete_intervals,
                    },
                },
            ]
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class SavedMemoryOpinions:
    def __init__(self, records):
        self.feed = driver.prior.rules.MemoryFeed(records)

    def observe(self, day):
        point = self.feed.latest(day)
        return model.OpinionSet(point.origin, np.asarray(point.targets), point.known_at)


def exact_scenario(batch, current, desired, delay, row):
    units, cash, values = current.asset_values.copy(), current.cash, []
    for step in range(21):
        if step == delay - 1:
            dollars, cash, _, turnover, orders = rebalance(
                units * batch.opening[row, step],
                cash,
                desired,
                current.cost_bps,
                1,
            )
            if turnover > 0 or orders > 0:
                units = dollars / batch.opening[row, step]
        values.append(float(units @ batch.closing[row, step] + cash))
    return np.asarray(values)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_negative_gap_envelope_contains_original_cash_engine(cost, delay):
    data = oscillating_market()
    day = pd.Timestamp("2016-10-05")
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    current = state(day, cost=cost, delay=delay)
    desired = np.zeros(14)
    desired[[0, 3, 11]] = [0.4, 0.28, 0.3]
    low, high = model.wealth_envelope(batch, current, desired)
    for row in range(126):
        actual = exact_scenario(batch, current, desired, delay, row)
        assert (low[row] <= actual + 1e-9).all()
        assert (actual <= high[row] + 1e-9).all()
        actual_peak = np.maximum.accumulate(np.r_[current.peak, actual])[1:]
        upper_peak = np.maximum.accumulate(np.r_[current.peak, high[row]])[1:]
        assert (1 - actual / actual_peak <= 1 - low[row] / upper_peak + 1e-12).all()
    assert model.assessment(batch, current, desired)["independent_relative_wealth_error"] <= 1e-12


def test_pending_remaining_delay_is_used_by_forecast():
    data, day = oscillating_market(), pd.Timestamp("2016-10-05")
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    current = replace(state(day, cost=20, delay=2, pending=True), pending_delay=1)
    proof = model.choose(current, MemoryOpinions().observe(day), batch).assessment["actions"][0]
    expected = model.assessment(batch, current, current.pending_target, execution_delay=1)
    assert proof["assessment"] == expected
    assert proof["assessment"] != model.assessment(batch, current, current.pending_target)


def test_representation_allowance_contains_rounding_only_noop():
    data, day = fixture_market(), pd.Timestamp("2016-10-05")
    sample = model.PriceScenarioFeed(data.open, data.close).observe(day)
    opening = np.ones_like(sample.opening) * (1 + 5e-11)
    closing = np.ones_like(sample.closing)
    closing[:, :, model.BIL] = np.linspace(1, 1.15, 21)
    batch = replace(sample, opening=opening, closing=closing)
    desired = np.zeros(14)
    desired[model.BIL] = 0.98
    current = model.CapitalState(day, desired * 10000, 200, 10000, 5, 1, False)
    low, high = model.wealth_envelope(batch, current, desired)
    actual = exact_scenario(batch, current, desired, 1, 0)
    assert (low[0] <= actual + 1e-9).all()
    assert (actual <= high[0] + 1e-9).all()
    _, _, fees, turnover, tickets = rebalance(
        current.asset_values * opening[0, 0], current.cash, desired, 5, 1
    )
    assert fees == turnover == tickets == 0


def test_nonpositive_small_capital_bound_is_explicit_not_feasible():
    data, day = fixture_market(), pd.Timestamp("2016-10-05")
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    current = model.CapitalState(day, np.zeros(14), 1, 10000, 5, 1, False)
    desired = MemoryOpinions().targets[0]
    result = model.assessment(batch, current, desired)
    assert result["status"] == "nonpositive_conservative_wealth_not_viable"
    assert result["score"] is None and not result["forecast_feasible"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cash", True),
        ("peak", "10000"),
        ("day", "2016-10-05"),
        ("day", pd.NaT),
        ("day", pd.Timestamp("2016-10-05", tz="UTC")),
        ("cost_bps", True),
        ("pending_delay", 0),
    ],
)
def test_invalid_capital_is_rejected_without_coercion(field, value):
    current = state(pd.Timestamp("2016-10-05"), pending=field == "pending_delay")
    with pytest.raises(QuantError):
        replace(current, **{field: value}).check()


def test_invalid_controller_call_does_not_mutate_peak_or_clock():
    data, day = fixture_market(), pd.Timestamp("2016-10-05")
    control = model.Controller(
        model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), 5, 1
    )
    with pytest.raises(QuantError):
        control.decide(day, np.zeros(14), True, True)
    assert control.last is None and control.peak == 10000
    control.decide(day, np.zeros(14), 10000, True)
    with pytest.raises(QuantError):
        control.decide(pd.Timestamp("2016-10-07"), np.zeros(14), 10000, False)
    assert control.last == day


def test_empty_market_and_unmatured_scenario_are_rejected():
    data, day = fixture_market(), pd.Timestamp("2016-10-05")
    with pytest.raises(QuantError):
        model.PriceScenarioFeed(data.open.iloc[:0], data.close.iloc[:0])
    batch = model.PriceScenarioFeed(data.open, data.close).observe(day)
    ends = (*batch.ends[:-1], pd.Timestamp("2016-10-06"))
    with pytest.raises(QuantError):
        replace(batch, ends=ends).check(day)


def test_archive_and_memory_opinions_cross_same_interface(tmp_path):
    records = points()
    archive = driver.ArchiveOpinions(tape(tmp_path / "sources.jsonl", records))
    memory = SavedMemoryOpinions(records)
    for day in sessions("2016-10-05", "2016-11-30"):
        a, b = archive.observe(day), memory.observe(day)
        assert a.origin == b.origin and a.known_at == b.known_at
        np.testing.assert_array_equal(a.targets, b.targets)
    seed = archive.observe(pd.Timestamp("2016-11-29"))
    assert seed.origin == pd.Timestamp("2016-10-31")


def test_archive_missing_month_and_future_native_known_at_are_rejected(tmp_path):
    records = points()
    with pytest.raises(QuantError):
        driver.ArchiveOpinions(tape(tmp_path / "missing.jsonl", (records[0], records[2])))
    bad = replace(records[1], known_at=(pd.Timestamp("2016-11-01"),) * 6)
    with pytest.raises(QuantError):
        driver.ArchiveOpinions(tape(tmp_path / "future.jsonl", (records[0], bad)))


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_future_source_and_prices_do_not_change_daily_prefix(tmp_path, cost, delay):
    data, cutoff = oscillating_market(), pd.Timestamp("2016-11-15")
    records = points()
    archive = driver.ArchiveOpinions(tape(tmp_path / "full.jsonl", records))
    full = execution.run(
        data,
        model.Controller(model.PriceScenarioFeed(data.open, data.close), archive, cost, delay),
        "2016-10-06",
        "2016-11-30",
        cost_bps=cost,
        delay=delay,
    )
    clipped = driver._clip(data, cutoff)
    short = driver.ArchiveOpinions(
        tape(tmp_path / "prefix.jsonl", tuple(p for p in records if p.origin <= cutoff))
    )
    prefix = execution.run(
        clipped,
        model.Controller(model.PriceScenarioFeed(clipped.open, clipped.close), short, cost, delay),
        "2016-10-06",
        str(cutoff.date()),
        cost_bps=cost,
        delay=delay,
    )
    assert driver.same_frame(prefix.result.frame, full.result.frame.loc[:cutoff])
    assert driver.same_frame(prefix.result.weights, full.result.weights.loc[:cutoff])
    assert driver.same_frame(prefix.issued, full.issued.loc[:cutoff])
    assert prefix.decisions == [
        row for row in full.decisions if pd.Timestamp(row["signal_session"]) <= cutoff
    ]
    changed = MarketData(
        *(f.copy() for f in (data.open, data.close, data.raw_close, data.volume, data.risk_free))
    )
    for f in (changed.open, changed.close, changed.raw_close, changed.volume):
        f.loc[f.index > cutoff] *= 1.2
    changed.risk_free.loc[changed.risk_free.index > cutoff] *= 2
    rows = np.roll(np.asarray(records[-1].targets), 3, axis=1)
    last = replace(
        records[-1],
        targets=tuple(map(tuple, rows)),
        mixed_target=tuple(np.asarray(records[-1].probabilities) @ rows),
    )
    changed_archive = driver.ArchiveOpinions(
        tape(tmp_path / "changed.jsonl", (*records[:-1], last))
    )
    other = execution.run(
        changed,
        model.Controller(
            model.PriceScenarioFeed(changed.open, changed.close), changed_archive, cost, delay
        ),
        "2016-10-06",
        "2016-11-30",
        cost_bps=cost,
        delay=delay,
    )
    assert driver.same_frame(other.result.frame.loc[:cutoff], full.result.frame.loc[:cutoff])
    assert [
        r for r in other.decisions if pd.Timestamp(r["signal_session"]) <= cutoff
    ] == prefix.decisions


def test_streamed_decisions_are_retained_and_verified():
    data, records = fixture_market(), []
    control = model.Controller(
        model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), 5, 1
    )
    output = execution.run(
        data,
        control,
        "2016-10-06",
        "2016-10-14",
        cost_bps=5,
        delay=1,
        retain=records.append,
        keep_decisions=False,
    )
    assert not output.decisions and output.decision_count == len(output.result.frame) + 1
    comparisons = []
    report = driver.verify_run(
        data,
        output,
        "2016-10-06",
        "2016-10-14",
        5,
        1,
        (r for r in records if r["kind"] == "main_decision_resolved"),
        lambda a, b, r: comparisons.append(r),
    )
    assert report["all_financial_frame_coordinates_exact"] and comparisons
    with pytest.raises(QuantError):
        execution.run(
            data,
            model.Controller(
                model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), 5, 1
            ),
            "2016-10-06",
            "2016-10-14",
            cost_bps=5,
            delay=1,
            keep_decisions=False,
        )


def test_cash_comparison_retained_before_rejection():
    data = fixture_market()
    output = execution.run(
        data,
        model.Controller(model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), 5, 1),
        "2016-10-06",
        "2016-10-14",
        cost_bps=5,
        delay=1,
    )
    output.result.frame.iloc[0, 0] += 1
    saved = []
    with pytest.raises(QuantError):
        driver.verify_run(
            data,
            output,
            "2016-10-06",
            "2016-10-14",
            5,
            1,
            iter(output.decisions),
            lambda a, b, r: saved.append(r),
        )
    assert saved and not saved[0]["all_financial_frame_coordinates_exact"]


def test_prohibited_content_rejected_before_open(tmp_path, monkeypatch):
    protected = tmp_path / "mock-protected.json"
    alias = tmp_path / "alias.json"
    alias.symlink_to(protected)
    monkeypatch.setattr(driver.prior.saved, "PROTECTED", protected)
    monkeypatch.setattr(driver, "PROTECTED", protected)
    opened = []
    original = Path.open

    def watch(path, *args, **kwargs):
        if path in (protected, alias):
            opened.append(path)
            raise AssertionError("A protected content inlet reached open.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watch)
    calls = (
        lambda p: driver.content_path(p),
        lambda p: driver.file_digest(p),
        lambda p: driver.read_json(p),
        lambda p: driver.reference(p),
        lambda p: driver.checked_reference({"path": str(p), "sha256": "0" * 64}),
        lambda p: driver.safe_file(tmp_path, p.name),
        lambda p: driver.base.checked_reference({"path": str(p), "sha256": "0" * 64}),
        lambda p: driver.base.reference(p),
        lambda p: driver.base.file_digest(p),
        lambda p: driver.ArchiveOpinions(p),
    )
    for path in (protected, alias):
        for call in calls:
            with pytest.raises(PermissionError):
                call(path)
    assert not opened


def test_both_preservation_loops_exclude_fake_protected_reference(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "research/workbench"
    protected = root / "config/stored-main-cost-audit-v1.json"
    monkeypatch.setattr(driver.prior, "ROOT", root)
    monkeypatch.setattr(driver.prior.saved, "ROOT", root)
    monkeypatch.setattr(driver.prior.saved, "PROTECTED", protected)
    names = [
        "GLOSSARY.md",
        "research/workbench/config/real-estate-source-intake-v1.json",
        "research/workbench/config/real-estate-structure-review-v1.json",
        "research/workbench/config/yen-currency-source-intake-v1.json",
        "research/workbench/config/yen-currency-structure-review-v1.json",
        *[f"allowed/{n}.txt" for n in range(10)],
    ]
    refs = []
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n")
        refs.append({"path": name, "sha256": "0" * 64, "mode": oct(path.stat().st_mode & 0o777)})
    refs.append(
        {"path": str(protected.relative_to(repo)), "sha256": "archival-only", "mode": "0o600"}
    )
    visited = []

    def digest(path):
        assert path != protected
        visited.append(path)
        return "0" * 64

    monkeypatch.setattr(driver.prior, "file_digest", digest)
    report = driver.prior._permitted_refs({"protected_reference_metadata": refs})
    assert report == {"archival_references": 16, "permitted_live": 15, "permitted_user_live": 5}
    assert len(set(visited)) == 15 and len(visited) == 20 and protected not in visited


def test_actual_packet_defers_historical_capital(tmp_path, monkeypatch):
    journal = []
    monkeypatch.setattr(driver, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(driver, "READY", tmp_path / "ready.json")
    monkeypatch.setattr(driver, "INTENT", tmp_path / "intent.json")
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    frozen = {"synthetic_complete_method": True}
    driver.INTENT.write_text(json.dumps({"synthetic": True}))
    monkeypatch.setattr(driver, "inputs", lambda: ({"actual_authoritative_counts": {}}, frozen))
    monkeypatch.setattr(driver, "counts", lambda: {})
    monkeypatch.setattr(driver, "checkpoint", lambda path: {"synthetic": True})
    monkeypatch.setattr(driver, "validation_refs", lambda: [])
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)
    monkeypatch.setattr(driver.prior, "_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(driver, "events", lambda kind: [r for r in journal if r["kind"] == kind])

    def append(_, kind, identity, body):
        row = {
            "kind": kind,
            "candidate_id": identity,
            "body": body,
            "sha256": "e" * 64,
            "at": driver.utc_now(),
        }
        journal.append(row)
        return row

    monkeypatch.setattr(driver, "base", SimpleNamespace(JOURNAL=tmp_path, append_event=append))

    def blocked(*a, **k):
        raise AssertionError("A closed source learner or account was replayed.")

    for name in ("prepare_source_targets", "evaluate_path", "evaluate"):
        monkeypatch.setattr(driver.prior.saved, name, blocked)
    driver.register()
    body = driver.checked_packet(frozen)
    assert body["schema_version"] == 3 and body["complete_fixed_cohort_ids"] == [model.ID]
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert not body["historical_new_meta_scores_or_targets_computed"]
    assert not body["candidate_accounts_or_metrics_computed"]
    assert [r["kind"] for r in journal] == ["registered"]
    with pytest.raises(QuantError):
        driver.register()
    assert len(journal) == 1


def test_driver_terminal_failure_preserves_outputs(tmp_path, monkeypatch):
    journal = [{"kind": "registered", "body": {}}]
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "events", lambda kind: [r for r in journal if r["kind"] == kind])
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)

    def append(_, kind, identity, body):
        journal.append({"kind": kind, "body": body})

    monkeypatch.setattr(driver, "base", SimpleNamespace(JOURNAL=tmp_path, append_event=append))
    partial = tmp_path / "partial-book.txt"
    partial.write_text("preserve synthetic partial output\n")
    driver.record_failure("funding_or_causality", QuantError("synthetic exact comparison failure"))
    driver.record_failure(
        "funding_or_causality", QuantError("duplicate must not add another failure")
    )
    proof = driver.read_json(tmp_path / "funding_or_causality-failure-v1.json")
    assert proof["partial_outputs_retained"] and partial.read_text().startswith("preserve")
    assert len([r for r in journal if r["kind"] == "technical_failure"]) == 1


def test_invalid_future_panel_still_rejected_by_existing_quality_gate():
    data = oscillating_market()
    data.close.loc[data.close.index > pd.Timestamp("2016-11-15")] *= 3
    with pytest.raises(QuantError, match="exceeds 60%"):
        data.validate()


def test_signed_zero_is_not_an_exact_position_comparison():
    first = pd.DataFrame([[0.0]], columns=["SPY"])
    second = pd.DataFrame([[-0.0]], columns=["SPY"])
    assert not driver.same_frame(first, second)


def test_later_main_window_has_fresh_capital_and_old_monthly_source():
    dates = sessions("2015-08-10", "2021-10-15")
    close = pd.DataFrame(
        100 * np.exp(np.arange(len(dates))[:, None] * np.linspace(0.0001, 0.0003, 14)),
        index=dates,
        columns=model.ASSETS,
    )
    data = MarketData(
        close * 0.9999, close, close.copy(), close * 0 + 1000, pd.Series(0.0001, index=dates)
    )
    output = execution.run(
        data,
        model.Controller(model.PriceScenarioFeed(data.open, data.close), MemoryOpinions(), 5, 1),
        "2021-10-06",
        "2021-10-15",
        cost_bps=5,
        delay=1,
    )
    seed = output.decisions[0]
    assert seed["signal_session"] == "2021-10-05"
    assert seed["main_NAV"] == seed["main_peak"] == 10000
    assert seed["assessment"]["source_origin"] == "2021-09-30"
    assert seed["assessment"]["original_source_clock_refreshed_daily"] is False


def test_actual_evaluation_fences_precede_four_paths_and_prefixes(tmp_path, monkeypatch):
    calls, journal = [], [{"kind": "registered", "body": {}}]
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(driver, "PACKET", tmp_path / "packet.json")
    driver.PACKET.write_text(json.dumps({"created_at": "2026-11-07T20:00:00+00:00"}))
    remote = tmp_path / "remote.json"
    remote.write_text(
        json.dumps(
            {
                "verified_at": "2026-11-07T20:15:00+00:00",
                "actual_preregistration_commit": "synthetic",
                "historical_outputs_previously_started": False,
            }
        )
    )
    benchmark = tmp_path / "benchmarks/results.json"
    benchmark.parent.mkdir()
    benchmark.write_text(json.dumps({"benchmarks": [None] * 8}))
    protocol = tmp_path / "protocol.json"
    windows = [{"years": 10}, {"years": 5}]
    scenarios = [{"id": "base"}, {"id": "stress"}]
    protocol.write_text(json.dumps({"windows": windows, "accounts": {"scenarios": scenarios}}))
    catalogue = {"source_market": {}, "source_benchmark_artifacts": [driver.reference(benchmark)]}
    monkeypatch.setattr(driver, "inputs", lambda: (catalogue, {}))
    monkeypatch.setattr(driver.prior, "_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)
    monkeypatch.setattr(driver, "events", lambda kind: [r for r in journal if r["kind"] == kind])

    def publication(frozen, commit):
        calls.append("publication")
        return {"synthetic_commit": commit}

    def checkpoint(path):
        calls.append("checkpoint")
        return {
            "actual_preregistration_commit": "synthetic",
            "checked_at": "2026-11-07T21:00:00+00:00",
            "actual_continuity_checkpoint_readback": {"saved_at": "2026-11-07T20:30:00+00:00"},
            "actual_remote_publication": driver.reference(remote),
        }

    def market(*a):
        assert calls == ["publication", "checkpoint"]
        calls.append("market")
        return None

    def path(data, inputs, frozen, window, scenario, benchmarks):
        key = f"{window['years']}y-{scenario['id']}"
        calls.append(key)
        location = tmp_path / "accounts" / key
        location.mkdir(parents=True)
        (location / "complete.json").write_text(json.dumps({"synthetic": True}))
        return {"all_point_gates_pass": False}

    def prefixes(*a):
        assert calls[-4:] == ["10y-base", "10y-stress", "5y-base", "5y-stress"]
        calls.append("prefixes")
        return {"synthetic_prefix_proof": True}

    def append(_, kind, identity, body):
        journal.append({"kind": kind, "body": body})

    monkeypatch.setattr(driver, "publication", publication)
    monkeypatch.setattr(driver, "checkpoint", checkpoint)
    monkeypatch.setattr(driver.prior.saved, "review_market", market)
    monkeypatch.setattr(driver, "evaluate_path", path)
    monkeypatch.setattr(driver, "prefixes", prefixes)
    monkeypatch.setattr(
        driver, "base", SimpleNamespace(JOURNAL=tmp_path, PROTOCOL=protocol, append_event=append)
    )
    driver.evaluate("synthetic")
    assert calls == [
        "publication",
        "checkpoint",
        "market",
        "10y-base",
        "10y-stress",
        "5y-base",
        "5y-stress",
        "prefixes",
    ]
    result = driver.read_json(driver.RESULTS)
    assert result["review"]["status"] == "rejected"
    assert [r["kind"] for r in journal] == ["registered", "publication", "target_proof", "review"]
    with pytest.raises(QuantError):
        driver.evaluate("synthetic")


def test_second_checkpoint_cannot_reuse_preregistration_save(tmp_path, monkeypatch):
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"created_at": "2026-11-07T20:00:00+00:00"}))
    remote = tmp_path / "remote.json"
    remote.write_text(
        json.dumps(
            {
                "verified_at": "2026-11-07T20:15:00+00:00",
                "actual_preregistration_commit": "synthetic",
                "historical_outputs_previously_started": False,
            }
        )
    )
    monkeypatch.setattr(driver, "PACKET", packet)
    start = {
        "actual_preregistration_commit": "synthetic",
        "actual_remote_publication": driver.reference(remote),
        "checked_at": "2026-11-07T21:00:00+00:00",
        "actual_continuity_checkpoint_readback": {"saved_at": "2026-11-07T19:30:00+00:00"},
    }
    with pytest.raises(QuantError):
        driver.funding_fence(start, "synthetic")
    start["actual_continuity_checkpoint_readback"]["saved_at"] = "2026-11-07T20:30:00+00:00"
    driver.funding_fence(start, "synthetic")
    with pytest.raises(QuantError):
        driver.funding_fence(start, "wrong-commit")
