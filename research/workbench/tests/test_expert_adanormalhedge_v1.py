import builtins
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_adanormalhedge_v1 as model
from us_quant import ssd_expert_adanormalhedge_v1 as driver
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData


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


def test_policy_keeps_original_algorithm_and_source_units():
    policy = driver.read_json(model.POLICY)
    model.validate(policy)
    for key, value in (
        ("algorithm", "NormalHedge_DT"),
        ("cumulative_magnitude", "elapsed_rounds"),
        ("source_loss_labels", "recentered_labels"),
        ("all_zero_scores", "reset_state"),
        ("five_year_source_learning_state_reset", True),
    ):
        with pytest.raises(QuantError):
            model.validate({**policy, key: value})


@pytest.mark.parametrize(
    "regret,magnitude", [("0", "0"), ("2.5", "4"), ("-1", "7"), ("119", "119")]
)
def test_primary_and_independent_score_match_original_potential(regret, magnitude):
    r, c = Decimal(regret), Decimal(magnitude)
    with localcontext() as context:
        context.prec = 400
        denominator = 3 * (c + 1)
        high = (max(r + 1, Decimal(0)) ** 2 / denominator).exp()
        low = (max(r - 1, Decimal(0)) ** 2 / denominator).exp()
        expected = (high - low) / 2
    actual, oracle = model.potential_score(r, c), model.independent_score(r, c)
    if expected == 0:
        assert actual == oracle == 0
    else:
        assert abs(actual / expected - 1) < Decimal("1e-25")
        assert abs(oracle / expected - 1) < Decimal("1e-25")


def test_tiny_positive_shift_never_becomes_zero_score():
    with localcontext() as context:
        context.prec = 400
        r = Decimal(-1) + Decimal("1e-100")
        actual = model.potential_score(r, Decimal(2))
        oracle = model.independent_score(r, Decimal(2))
        assert actual > 0 and oracle > 0
        assert abs(actual / oracle - 1) < Decimal("1e-100")


def test_initial_and_equal_loss_state_remains_exact():
    learner = model.AdaNormalHedge()
    for _ in range(5):
        update = learner.advance(np.full(6, 0.3))
        assert all(Decimal(v) == 0 for v in update["instantaneous_source_regret_decimal"])
        assert all(
            Decimal(v) == 0 for v in learner.snapshot()["cumulative_absolute_source_regret_decimal"]
        )
    np.testing.assert_allclose(learner.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-15)


def test_all_zero_scores_use_uniform_without_state_reset():
    state = model.RegretState(6, (Decimal(-2),) * 6, (Decimal(3),) * 6, Decimal(1))
    learner = model.AdaNormalHedge(state)
    before = learner.state
    np.testing.assert_allclose(learner.predict(), np.full(6, 1 / 6), rtol=0, atol=1e-15)
    snapshot = learner.snapshot()
    assert snapshot["all_zero_scores_use_frozen_uniform_prior"]
    assert not snapshot["all_zero_scores_reset_source_state"]
    assert learner.state == before
    learner.advance(np.array([0, 1, 0.2, 0.4, 0.6, 0.8]))
    assert learner.state.intervals == 7 and learner.state.regrets != (Decimal(0),) * 6


def test_all119_matured_labels_match_independent_regret_scores():
    primary, oracle = model.AdaNormalHedge(), model.IndependentAdaNormalHedge()
    rng = np.random.default_rng(20261107)
    for _ in range(119):
        labels = rng.uniform(0, 1, 6)
        previous = primary.predict().tolist()
        result = primary.advance(labels)
        oracle.advance(labels)
        assert result["prequential_forecast_before_assimilation"] == previous
        np.testing.assert_allclose(primary.predict(), oracle.forecast, rtol=0, atol=1e-12)
        assert primary.state.intervals == oracle.state.intervals
        for a, b in zip(primary.state.regrets, oracle.state.regrets, strict=True):
            assert abs(float(a - b)) <= 1e-12
        for a, b in zip(primary.state.magnitudes, oracle.state.magnitudes, strict=True):
            assert abs(float(a - b)) <= 1e-12
        assert abs(float(primary.state.played_loss - oracle.state.played_loss)) <= 1e-12
    with pytest.raises(QuantError, match="119"):
        primary.advance(np.zeros(6))


def test_regret_update_uses_prior_prediction_and_absolute_excess():
    learner = model.AdaNormalHedge()
    labels = np.array([0, 1, 0.2, 0.3, 0.4, 0.9])
    prior = learner.predict()
    update = learner.advance(labels)
    np.testing.assert_allclose(
        [float(Decimal(v)) for v in update["instantaneous_source_regret_decimal"]],
        prior @ labels - labels,
        rtol=0,
        atol=1e-12,
    )
    assert learner.state.magnitudes == tuple(v.copy_abs() for v in learner.state.regrets)
    assert learner.predict()[0] > prior[0]


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(5),
        np.full(6, np.nan),
        np.full(6, np.inf),
        np.ones(6, dtype=bool),
        np.full(6, -0.1),
        np.full(6, 1.1),
    ],
)
def test_invalid_loss_labels_fail_without_uniform_fallback(values):
    learner = model.AdaNormalHedge()
    with pytest.raises(QuantError):
        learner.advance(values)
    assert learner.state.intervals == 0


def test_invalid_regret_state_is_not_clipped():
    state = model.RegretState(2, (Decimal(3),) * 6, (Decimal(1),) * 6, Decimal(1))
    with pytest.raises(QuantError, match="Invalid source"):
        model.AdaNormalHedge(state)


def test_invalid_expert_loss_is_rejected_without_decimal_cancellation():
    with localcontext() as context:
        context.prec = 400
        regret = Decimal(1) + Decimal("1e-100")
    state = model.RegretState(2, (regret,) * 6, (Decimal(2),) * 6, Decimal(1))
    with pytest.raises(QuantError, match="Invalid source"):
        model.AdaNormalHedge(state)


@pytest.mark.parametrize("values", [np.zeros((2, 3)), ["invalid"] * 6])
def test_independent_invalid_labels_fail_before_conversion(values):
    with pytest.raises(QuantError):
        model.IndependentAdaNormalHedge().advance(values)


def test_independent_series_limit_fails_explicitly(monkeypatch):
    monkeypatch.setattr(model, "MAX_TERMS", 1)
    with pytest.raises(QuantError, match="numerical limit"):
        model.independent_score(Decimal(5), Decimal(10))


def test_first_partial_month_is_not_assimilated():
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    result = model.build_targets(
        calendar, MemoryOpinions(), periods(calendar), retain=records.append
    )
    assert [r["complete_source_intervals"] for r in result.audits] == [0, 0, 1]
    assert result.audits[0]["period_update"] is result.audits[1]["period_update"] is None
    assert (
        len(records) == 6 and records[-2]["matured_source_interval"]["start_after"] == "2016-10-31"
    )


def test_input_is_retained_before_deliberate_learner_failure(monkeypatch):
    def failed(*args, **kwargs):
        raise QuantError("deliberate learner failure")

    monkeypatch.setattr(model.AdaNormalHedge, "advance", failed)
    calendar = sessions("2016-10-05", "2016-11-30")
    records = []
    with pytest.raises(QuantError, match="deliberate"):
        model.build_targets(calendar, MemoryOpinions(), periods(calendar), retain=records.append)
    assert len(records) == 5 and records[-1]["phase"] == "before_decision"


def test_missing_month_or_future_native_opinion_fails():
    calendar = sessions("2016-10-05", "2017-01-31")
    with pytest.raises(QuantError, match="month is missing"):
        model.build_targets(calendar, MemoryOpinions(), periods(calendar)[:-1])

    class FutureOpinions(MemoryOpinions):
        def observe(self, day):
            return replace(super().observe(day), native_known_at=(next_session(day),) * 6)

    with pytest.raises(QuantError, match="then-known"):
        model.build_targets(calendar, FutureOpinions(), periods(calendar))


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


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_full_mixed_asset_cash_and_BT(data, prepared, cost, delay):
    output = model.run(data, prepared, "2018-02-06", "2018-03-29", cost_bps=cost, delay=delay)
    seen = []
    proof = model.verify_run(
        data,
        prepared,
        output,
        "2018-02-06",
        "2018-03-29",
        cost,
        delay,
        retain_comparison=lambda a, b, report: seen.append((a, b, report)),
    )
    assert proof["all_cash_financial_columns_and_positions_exact"]
    assert seen[0][2]["position_bit_differences"] == 0
    assert all(
        v["binary64_bit_differences"] == 0
        for v in seen[0][2]["all_frame_column_diagnostics"].values()
    )
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
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
@pytest.mark.parametrize("cut", ["2018-02-28", "2018-03-13"])
def test_actual_capital_prefix_and_source_seed_remain_exact(data, prepared, cost, delay, cut):
    day = pd.Timestamp(cut)
    full = model.run(data, prepared, "2018-02-06", "2018-03-29", cost_bps=cost, delay=delay)
    source = model.PreparedSourceTargets(
        prepared.targets.loc[:day],
        [r for r in prepared.audits if pd.Timestamp(r["decision_session"]) <= day],
        prepared.proof,
    )
    prefix = model.run(clipped(data, day), source, "2018-02-06", cut, cost_bps=cost, delay=delay)
    for actual, expected in (
        (prefix.result.frame, full.result.frame),
        (prefix.result.weights, full.result.weights),
        (prefix.desired, full.desired),
        (prefix.issued, full.issued),
    ):
        pd.testing.assert_frame_equal(actual, expected.loc[:day], check_exact=True)
        assert np.array_equal(np.signbit(actual), np.signbit(expected.loc[:day]))


@pytest.mark.parametrize("cut", ["2016-10-31", "2017-04-12"])
def test_future_labels_and_opinions_do_not_change_past_forecasts(data, prepared, cut):
    day = pd.Timestamp(cut)
    short = model.build_targets(
        data.close.loc[:day].index, MemoryOpinions(), periods(data.close.index)
    )
    altered = [
        {**row, "input_loss_labels": [0.5] * 6} if pd.Timestamp(row["end"]) > day else row
        for row in periods(data.close.index)
    ]
    future = model.build_targets(data.close.index, MemoryOpinions(day), altered)
    pd.testing.assert_frame_equal(prepared.targets.loc[:day], short.targets, check_exact=True)
    pd.testing.assert_frame_equal(
        prepared.targets.loc[:day], future.targets.loc[:day], check_exact=True
    )
    assert short.audits == [r for r in future.audits if pd.Timestamp(r["decision_session"]) <= day]


def test_old_source_learners_never_execute(data, monkeypatch):
    from us_quant import expert_switching_loss_v1, expert_switching_wealth_v1

    def forbidden(*args, **kwargs):
        raise AssertionError("No old source learner may replay.")

    monkeypatch.setattr(expert_switching_loss_v1.AdaptiveSwitchingLoss, "advance", forbidden)
    monkeypatch.setattr(expert_switching_wealth_v1.SwitchCountWealth, "advance", forbidden)
    result = model.build_targets(data.close.index, MemoryOpinions(), periods(data.close.index))
    assert result.proof["complete_source_intervals"] > 0


def test_actual_packet_defers_real_source_learning(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-07T11:00:00Z",
        "body": {"candidate": driver.CANDIDATE, "frozen": frozen},
    }
    ready = {"path": "synthetic-readiness.json", "sha256": "b" * 64}
    body = driver.packet_body(frozen, registration, ready)
    assert body["schema_version"] == 3 and body["entries"][0]["candidate_id"] == model.ID
    assert body["historical_new_meta_scores_or_targets_computed"] is False
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    monkeypatch.setattr(
        driver, "events", lambda kind: [registration] if kind == "registered" else []
    )
    monkeypatch.setattr(driver, "reference", lambda path: ready)
    monkeypatch.setattr(
        driver,
        "read_json",
        lambda path: {**body, "created_at": "2026-11-07T11:00:01Z"}
        if path == driver.PACKET
        else {
            "historical_target_derivation_verified": False,
            "historical_new_meta_scores_targets_accounts_computed": False,
        },
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3


def test_registration_uses_current_module_inputs_not_legacy_globals(tmp_path, monkeypatch):
    frozen, rows, written, calls = {"study_id": model.STUDY}, [], {}, []
    monkeypatch.setattr(
        driver,
        "lifecycle",
        SimpleNamespace(runtime_guard=lambda: None, research_job=nullcontext),
    )

    def inputs():
        calls.append("current_inputs")
        return None, None, frozen, None

    def append(directory, kind, identifier, body):
        assert kind == "registered" and identifier == model.ID
        result = {"body": body, "sha256": "a" * 64, "at": "2026-11-07T11:00:00Z"}
        rows.append(result)
        return result

    monkeypatch.setattr(driver, "inputs", inputs)
    monkeypatch.setattr(driver, "base", SimpleNamespace(JOURNAL=tmp_path, append_event=append))
    monkeypatch.setattr(driver, "events", lambda kind: rows if kind == "registered" else [])
    monkeypatch.setattr(driver, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(driver, "READY", tmp_path / "ready.json")
    monkeypatch.setattr(driver, "validation_refs", lambda: [])
    monkeypatch.setattr(driver, "reference", lambda p: {"path": p.name, "sha256": "b" * 64})
    monkeypatch.setattr(driver, "write_new_receipt", lambda p, body: written.update({p: body}))
    monkeypatch.setattr(driver, "checked_packet", lambda f: written[driver.PACKET])
    monkeypatch.setattr(driver, "progress", lambda *args, **kwargs: None)
    result = driver.register()
    assert calls == ["current_inputs"] and result["path"] == "packet.json"
    assert written[driver.PACKET]["historical_new_meta_scores_or_targets_computed"] is False
    assert written[driver.READY]["historical_target_derivation_verified"] is False


@pytest.mark.parametrize(
    "entry", ["file_digest", "read_json", "reference", "checked_reference", "safe_file"]
)
def test_banned_content_guard_before_open_in_both_entrypoints(tmp_path, monkeypatch, entry):
    fake = tmp_path / "synthetic-prohibited.json"
    monkeypatch.setattr(driver, "PROTECTED", fake)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("Protected fake must be rejected before any open.")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    argument = {"path": fake.name, "sha256": "a" * 64} if entry == "checked_reference" else fake
    with pytest.raises(PermissionError, match="before open"):
        if entry == "safe_file":
            driver.safe_file(tmp_path, fake.name, "a" * 64)
        else:
            getattr(driver, entry)(argument)
    assert not opened
    refs = [{"path": fake.name, "sha256": "a" * 64}, {"path": "permitted.json", "sha256": "b" * 64}]
    assert driver.live_references(refs) == [refs[1]]
    assert driver.live_references(refs.copy()) == [refs[1]]


@pytest.mark.parametrize("entry", ["file_digest", "read_json", "safe_file", "review_market"])
def test_borrowed_readers_use_preopen_guard_not_legacy_globals(tmp_path, monkeypatch, entry):
    fake = tmp_path / "synthetic-prohibited.json"
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "PROTECTED", fake)
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("The archived Adapter cannot open protected synthetic content.")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    with pytest.raises(PermissionError, match="before open"):
        if entry == "safe_file":
            driver.base.safe_file(tmp_path, fake.name)
        elif entry == "review_market":
            refs = {
                key: {"path": fake.name, "sha256": "a" * 64}
                for key in ("open", "close", "raw_close", "volume", "risk_free")
            }
            driver.review_market({"market": refs}, tmp_path)
        else:
            getattr(driver.base, entry)(fake)
    assert not opened
    assert driver.base.published_inputs.__globals__["safe_file"] is driver.safe_file
    assert (
        driver.base.published_inputs.__globals__["verify_publication"] is driver.verify_publication
    )
    assert driver.base.verify_path.__globals__["checked_reference"] is driver.checked_reference


def test_prohibited_alias_is_rejected_before_open(tmp_path, monkeypatch):
    fake = tmp_path / "synthetic-prohibited.json"
    alias = tmp_path / "alias.json"
    alias.symlink_to(fake)
    monkeypatch.setattr(driver, "PROTECTED", fake)
    with pytest.raises(PermissionError, match="alias"):
        driver.file_digest(alias)


def test_concrete_cash_record_adapter_does_not_recurse(data, prepared, tmp_path, monkeypatch):
    output = model.run(data, prepared, "2018-02-06", "2018-03-29", cost_bps=5, delay=1)
    monkeypatch.setattr(driver, "reference", lambda p: {"path": p.name, "sha256": "a" * 64})
    assert len(driver.write_run(tmp_path, output)) == 5
    folder = tmp_path / "comparison"
    folder.mkdir()
    driver.retain_cash_comparison(
        folder, output.result, output.result, model.comparison_report(output.result, output.result)
    )
    assert driver.read_json(folder / "comparison.json")["retained_before_acceptance_assertion"]
