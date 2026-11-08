import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_dyadic_meta_v1 as model
from us_quant import ssd_expert_dyadic_meta_v1 as driver
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData


class MemoryFeed:
    def observe(self, day):
        targets = np.zeros((6, len(model.ASSETS)))
        for i in range(6):
            targets[i, (i + day.month) % len(model.ASSETS)] = 0.98
        return model.OpinionSnapshot(day, targets, (day,) * 6)


def fixture():
    dates = sessions("2015-08-10", "2017-02-28")
    close = pd.DataFrame(
        100 * np.exp(np.arange(len(dates))[:, None] * np.linspace(0.0001, 0.0006, 14)),
        index=dates,
        columns=model.ASSETS,
    )
    data = MarketData(
        close * 0.9999, close, close.copy(), close * 0 + 1000, pd.Series(0.0001, index=dates)
    )
    days = [d for d in dates if d >= pd.Timestamp("2016-10-31") and is_month_end(d)]
    periods = [
        {
            "start_after": str(a.date()),
            "end": str(b.date()),
            "input_loss_labels": (np.arange(6) / 10 + b.month / 100).tolist(),
        }
        for a, b in zip(days[:-1], days[1:], strict=True)
    ]
    return data, periods


def test_fixed_policy_does_not_claim_strong_adaptivity_or_financial_guarantee():
    policy = driver.read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate({**policy, "faithful_Daniely_SAOL_algorithm_or_regret_guarantee": True})


def test_calendar_epochs_expire_without_global_state_reset():
    learner = model.DyadicLearner()
    assert model.active(0) == tuple((1, length) for length in model.LENGTHS)
    for period in range(17):
        learner.advance(np.full(6, 0.3))
        assert learner.periods == period + 1
        assert set(learner.epochs) == set(model.active(learner.periods))
        assert learner.retired == sum(learner.periods // length for length in model.LENGTHS)
    assert len(learner.epochs) == 8
    np.testing.assert_allclose(learner.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-12)
    json.dumps(learner.snapshot(), allow_nan=False)


def test_all119_new_epoch_forecasts_match_recomputed_oracle():
    primary, oracle = model.DyadicLearner(), model.RecomputedOracle()
    for period in range(119):
        np.testing.assert_allclose(primary.predict(), oracle.predict(), rtol=0, atol=1e-12)
        values = 0.5 + 0.4 * np.sin(period / 5 + np.arange(6))
        update = primary.advance(values)
        oracle.advance(values)
        json.dumps(update, allow_nan=False)
        np.testing.assert_allclose(primary.predict(), oracle.predict(), rtol=0, atol=1e-12)
    assert primary.retired == sum(119 // length for length in model.LENGTHS)
    with pytest.raises(QuantError):
        primary.advance(np.zeros(6))


@pytest.mark.parametrize(
    "values",
    [
        [True, 0, 0, 0, 0, 0],
        [np.bool_(True), 0, 0, 0, 0, 0],
        ["0.1"] * 6,
        [2, 0, 0, 0, 0, 0],
        [float("nan")] * 6,
        [0] * 5,
    ],
)
def test_invalid_source_labels_do_not_reset_or_uniformize(values):
    learner = model.DyadicLearner()
    with pytest.raises(QuantError):
        learner.advance(values)
    assert learner.periods == 0
    oracle = model.RecomputedOracle()
    with pytest.raises(QuantError):
        oracle.advance(values)
    assert not oracle.labels and not oracle.played


def test_boolean_source_labels_rejected_before_real_delivery_or_state_update():
    data, periods = fixture()
    periods[0]["input_loss_labels"][0] = True
    with pytest.raises(QuantError):
        model.build_targets(data.close.index, MemoryFeed(), periods)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_actual_source_and_capital_JSONL_delivery(tmp_path, cost, delay):
    data, periods = fixture()
    source_lines = []

    def retain_source(row):
        source_lines.append(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))

    prepared = model.build_targets(data.close.index, MemoryFeed(), periods, retain=retain_source)
    assert len(source_lines) == 2 * prepared.proof["source_decisions"]
    capital_lines = []
    output = model.run(
        data,
        prepared,
        "2016-10-06",
        "2017-02-28",
        cost_bps=cost,
        delay=delay,
        retain=lambda row: capital_lines.append(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
    )
    assert capital_lines and any(json.loads(v).get("kind") == "decision" for v in capital_lines)
    folder = tmp_path / "cash-comparison"
    folder.mkdir()
    checked = model.verify_run(
        data,
        prepared,
        output,
        "2016-10-06",
        "2017-02-28",
        cost,
        delay,
        retain_comparison=lambda a, b, report: driver.retain_cash_comparison(folder, a, b, report),
    )
    assert checked["all_cash_financial_columns_and_positions_exact"]
    independent = independent_equity(
        data,
        output.issued,
        "2016-10-06",
        "2017-02-28",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert abs(independent["equity"] - output.result.frame["equity"]).max() <= 1e-6
    assert (folder / "comparison.json").is_file()


def test_future_does_not_change_new_source_prefix():
    data, periods = fixture()
    cutoff = pd.Timestamp("2017-01-31")
    full = model.build_targets(data.close.index, MemoryFeed(), periods)
    changed = [dict(row) for row in periods]
    changed[-1]["input_loss_labels"] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    other = model.build_targets(data.close.index[data.close.index <= cutoff], MemoryFeed(), changed)
    pd.testing.assert_frame_equal(full.targets.loc[:cutoff], other.targets, check_exact=True)
    assert other.audits == [a for a in full.audits if pd.Timestamp(a["decision_session"]) <= cutoff]


def test_no_partial_initial_month_or_missing_matured_period():
    data, periods = fixture()
    prepared = model.build_targets(data.close.index, MemoryFeed(), periods)
    assert prepared.audits[0]["complete_source_intervals"] == 0
    assert prepared.audits[1]["complete_source_intervals"] == 0
    assert prepared.audits[2]["complete_source_intervals"] == 1
    with pytest.raises(QuantError):
        model.build_targets(data.close.index, MemoryFeed(), periods[1:])


def test_prohibited_content_guard_before_open(tmp_path, monkeypatch):
    protected = tmp_path / "mock-protected.json"
    alias = tmp_path / "alias.json"
    alias.symlink_to(protected)
    monkeypatch.setattr(driver.prior.saved, "PROTECTED", protected)
    opened, original = [], Path.open

    def watch(path, *args, **kwargs):
        if path in (protected, alias):
            opened.append(path)
            raise AssertionError("A prohibited content inlet reached open.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watch)
    calls = [
        driver.content_path,
        driver.file_digest,
        driver.read_json,
        driver.reference,
        driver.base.file_digest,
        driver.base.reference,
        lambda p: driver.checked_reference({"path": str(p), "sha256": "0" * 64}),
        lambda p: driver.safe_file(tmp_path, p.name),
    ]
    for path in (protected, alias):
        for call in calls:
            with pytest.raises(PermissionError):
                call(path)
    assert not opened


def test_actual_packet_defers_source_learning(tmp_path, monkeypatch):
    journal = []
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(driver, "READY", tmp_path / "ready.json")
    monkeypatch.setattr(driver, "INTENT", tmp_path / "intent.json")
    monkeypatch.setattr(driver, "CATALOGUE", tmp_path / "catalogue.json")
    driver.INTENT.write_text(json.dumps({"synthetic": True}))
    driver.CATALOGUE.write_text(json.dumps({"actual_authoritative_counts": {}}))
    frozen = {"synthetic_complete_method": True}
    monkeypatch.setattr(driver, "inputs", lambda: (None, {}, frozen, None))
    monkeypatch.setattr(driver, "checkpoint", lambda path: {"synthetic": True})
    monkeypatch.setattr(driver, "counts", lambda: {})
    monkeypatch.setattr(driver, "validation_refs", lambda: [])
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)
    monkeypatch.setattr(driver.lifecycle, "runtime_guard", lambda: None)
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

    def forbidden(*a, **k):
        raise AssertionError("Source states or capital were computed before publication.")

    monkeypatch.setattr(driver, "prepare_source_targets", forbidden)
    driver.register()
    packet = driver.checked_packet(frozen)
    assert packet["schema_version"] == 3
    assert packet["complete_fixed_cohort_ids"] == [model.ID]
    assert packet["entries"][0]["historical_target_derivation_proof"] is None
    assert not packet["historical_new_meta_scores_or_targets_computed"]
    assert not packet["candidate_accounts_or_metrics_computed"]


def write_opinions(path, dates):
    rows = []
    days = [
        pd.Timestamp("2016-10-05"),
        *[d for d in dates if d >= pd.Timestamp("2016-10-06") and is_month_end(d)],
    ]
    for day in days:
        point = MemoryFeed().observe(day)
        rows.extend(
            [
                {
                    "phase": "before_decision",
                    "decision_session": str(day.date()),
                    "known_targets_hex": [[float(v).hex() for v in row] for row in point.targets],
                    "known_target_dates": [str(d.date()) for d in point.native_known_at],
                },
                {"phase": "after_decision", "decision_session": str(day.date())},
            ]
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return driver.reference(path)


def full_fixture():
    dates = sessions("2015-08-10", "2026-10-05")
    close = pd.DataFrame(
        100 * np.exp(np.arange(len(dates))[:, None] * np.linspace(0.0001, 0.0006, 14)),
        index=dates,
        columns=model.ASSETS,
    )
    data = MarketData(
        close * 0.9999, close, close.copy(), close * 0 + 1000, pd.Series(0.0001, index=dates)
    )
    months = [d for d in dates if d >= pd.Timestamp("2016-10-31") and is_month_end(d)]
    periods = [
        {
            "start_after": str(a.date()),
            "end": str(b.date()),
            "input_loss_labels": (0.5 + 0.4 * np.sin(i / 5 + np.arange(6))).tolist(),
        }
        for i, (a, b) in enumerate(zip(months[:-1], months[1:], strict=True))
    ]
    assert len(periods) == 119
    return data, periods


def test_stored_opinion_adapter_matches_memory_and_rejects_missing_pair(tmp_path):
    data, _ = fixture()
    ref = write_opinions(tmp_path / "opinions.jsonl", data.close.index)
    feed = driver.SavedOpinionFeed({"cached_complete_policy_opinions": ref})
    for day in feed.rows:
        a, b = feed.observe(day), MemoryFeed().observe(day)
        np.testing.assert_array_equal(a.targets, b.targets)
        assert a.native_known_at == b.native_known_at
    broken = tmp_path / "broken.jsonl"
    broken.write_text((tmp_path / "opinions.jsonl").read_text().splitlines()[0] + "\n")
    with pytest.raises(QuantError):
        driver.SavedOpinionFeed({"cached_complete_policy_opinions": driver.reference(broken)})


def test_real_driver_all_source_capital_prefix_and_export_delivery(tmp_path, monkeypatch):
    data, periods = full_fixture()
    ref = write_opinions(tmp_path / "opinions.jsonl", data.close.index)
    source = tmp_path / "periods.json"
    source.write_text(json.dumps({"periods": periods}))
    catalogue = {
        "cached_complete_policy_opinions": ref,
        "cached_six_source_month_audits": driver.reference(source),
    }
    windows = [
        {"years": 10, "start": "2016-10-06", "end": "2026-10-05"},
        {"years": 5, "start": "2021-10-06", "end": "2026-10-05"},
    ]
    scenarios = [
        {"id": "base", "cost_bps_per_side": 5.0, "delay_sessions": 1},
        {"id": "stress", "cost_bps_per_side": 20.0, "delay_sessions": 2},
    ]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"windows": windows, "accounts": {"scenarios": scenarios}}))
    benchmark_rows = []
    for window in windows:
        dates = data.close.loc[window["start"] : window["end"]].index
        metrics = {
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "sessions": len(dates),
            "cagr": 0.01,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
        for scenario in scenarios:
            for symbol in ("SPY", "QQQ"):
                benchmark_rows.append(
                    {
                        "years": window["years"],
                        "scenario": scenario["id"],
                        "symbol": symbol,
                        "metrics": metrics,
                    }
                )
    benchmarks = {"benchmarks": benchmark_rows}
    frozen = {"synthetic_complete_method": True}
    journal = [{"kind": "registered", "candidate_id": model.ID, "body": {}}]
    original = driver.base

    class MemoryBase:
        JOURNAL = tmp_path
        PROTOCOL = protocol

        def __getattr__(self, name):
            return getattr(original, name)

        def journal_events(self, _):
            return list(journal)

        def append_event(self, _, kind, identity, body):
            row = {
                "kind": kind,
                "candidate_id": identity,
                "body": body,
                "sha256": f"{len(journal):064x}",
                "at": driver.utc_now(),
            }
            journal.append(row)
            return row

    monkeypatch.setattr(driver, "base", MemoryBase())
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(driver, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(driver, "PUBLIC", tmp_path / "public-results.json")
    driver.PACKET.write_text(json.dumps({"created_at": "2026-11-08T00:00:00+00:00"}))
    remote = tmp_path / "remote.json"
    remote.write_text(
        json.dumps(
            {
                "actual_preregistration_commit": "synthetic",
                "verified_at": "2026-11-08T00:10:00+00:00",
                "historical_outputs_previously_started": False,
            }
        )
    )
    funding = {
        "actual_preregistration_commit": "synthetic",
        "actual_remote_publication": driver.reference(remote),
        "started_at": "2026-11-08T00:30:00+00:00",
        "checked_at": "2026-11-08T00:30:00+00:00",
        "actual_continuity_checkpoint_readback": {"saved_at": "2026-11-08T00:20:00+00:00"},
    }
    (tmp_path / "actual-funding-start-v1.json").write_text(json.dumps(funding))
    monkeypatch.setattr(driver, "inputs", lambda: (data, catalogue, frozen, benchmarks))
    monkeypatch.setattr(driver, "checkpoint", lambda _: funding)
    monkeypatch.setattr(driver, "publication", lambda *_: {"synthetic_commit": "synthetic"})
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)
    monkeypatch.setattr(driver, "counts", lambda: {"synthetic_journal_count": len(journal)})
    monkeypatch.setattr(driver.prior, "_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "research_job", nullcontext)
    driver.evaluate("synthetic")
    saved = driver.read_json(driver.RESULTS)
    assert set(saved["paths"]) == {"10y-base", "10y-stress", "5y-base", "5y-stress"}
    assert saved["review"]["all_four_paths_same_candidate_and_frozen_parameters"]
    assert len([r for r in journal if r["kind"] == "path_completed"]) == 4
    assert len([r for r in journal if r["kind"] == "review"]) == 1
    assert driver.read_json(tmp_path / "source-targets/proof.json")["source_decisions"] == 121
    assert (
        driver.read_json(tmp_path / "source-prefix-audits-v1/completion.json")[
            "new_economic_trials"
        ]
        == 0
    )
    assert (
        driver.read_json(tmp_path / "capital-prefix-audits-v1/completion.json")[
            "new_economic_trials"
        ]
        == 0
    )
    driver.export()
    public = driver.read_json(driver.PUBLIC)
    assert not public["investment_objective_verified"]
    assert not public["source_game_guarantee_is_actual_main_wealth_or_drawdown_bound"]
    assert set(public["paths"]) == set(saved["paths"])
    with pytest.raises(QuantError):
        driver.evaluate("synthetic")
