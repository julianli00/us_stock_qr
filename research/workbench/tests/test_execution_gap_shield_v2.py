import ast
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import execution_gap_shield_v2 as model
from us_quant import ssd_execution_gap_shield_v2 as driver
from us_quant.backtest import BacktestResult
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import digest_json, read_json


def test_economic_law_not_retuned():
    policy = read_json(model.POLICY)
    model.validate(policy)
    original = read_json(model.ROOT / "config/execution-gap-shield-v1.json")
    assert (
        digest_json({k: v for k, v in original.items() if k not in model.IDENTITY_FIELDS})
        == model.ECONOMIC_SHA256
    )
    with pytest.raises(QuantError):
        model.validate({**policy, "ordinary_actual_holding_band": 0.024})


@pytest.mark.parametrize(
    "name", ["coefficients", "verify_scalar", "greatest_binary64", "ShieldPolicy"]
)
def test_pure_decision_law_AST_unchanged_without_executing_V1(name):
    def body(path):
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if getattr(n, "name", None) == name)
        return ast.dump(node, include_attributes=False)

    source = model.ROOT / "src/us_quant"
    assert body(source / "execution_gap_shield_v1.py") == body(
        source / "execution_gap_shield_v2.py"
    )


@pytest.fixture
def data(market_factory):
    return market_factory(start="2019-12-02", end="2020-04-30", symbols=model.ASSETS)


def nominal(data):
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    raw = np.arange(1, model.N + 1, dtype=float)
    for day in targets.index:
        if day == targets.index[0] or is_month_end(day):
            values = np.roll(raw, day.month) / raw.sum() * 0.98
            targets.loc[day] = values
    return targets


def clipped(data, end):
    return MarketData(
        data.open.loc[:end].copy(),
        data.close.loc[:end].copy(),
        data.raw_close.loc[:end].copy(),
        data.volume.loc[:end].copy(),
        data.risk_free.loc[:end].copy(),
    )


def run(data, cost=5, delay=1, **kwargs):
    return model.run(
        data,
        nominal(data),
        "2020-02-04",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
        **kwargs,
    )


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_mixed_asset_full_frame_cash_parity(data, cost, delay):
    output = run(data, cost, delay)
    seen = []
    checked = model.verify_run(
        data,
        output,
        "2020-02-04",
        "2020-03-31",
        cost,
        delay,
        retain_comparison=lambda actual, replay, report: seen.append((actual, replay, report)),
    )
    assert len(seen) == 1
    assert checked["all_column_exact_equality_acceptance_unchanged"]
    assert seen[0][2]["full_frame_equals"] and seen[0][2]["full_positions_equal"]
    assert all(
        v["binary64_bit_differences"] == 0
        for v in seen[0][2]["all_frame_column_diagnostics"].values()
    )
    assert (output.result.weights.gt(0).sum(axis=1) >= 3).any()
    assert output.result.frame.index.equals(sessions("2020-02-04", "2020-03-31"))
    assert output.result.frame["risk_free"].equals(data.risk_free.loc[output.result.frame.index])
    assert (output.result.frame["cash"] >= 0).all()
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


def test_canonical_gross_uses_actual_held_vector(data):
    output = run(data)
    expected = [float(row.copy().sum()) for row in output.result.weights.to_numpy()]
    assert np.array_equal(output.result.frame["gross_exposure"].to_numpy(), expected)


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
def test_corrupted_frame_operands_retained_before_failure(data, field):
    output = run(data)
    frame, weights = output.result.frame.copy(), output.result.weights.copy()
    if field == "position":
        weights.iloc[1, 0] = np.nextafter(float(weights.iloc[1, 0]), np.inf)
    elif field == "orders":
        frame.loc[frame.index[1], field] += 1
    else:
        frame.loc[frame.index[1], field] = np.nextafter(float(frame.iloc[1][field]), np.inf)
    corrupted = replace(output, result=BacktestResult(frame, weights))
    seen = []
    with pytest.raises(QuantError, match="cash-engine replay"):
        model.verify_run(
            data,
            corrupted,
            "2020-02-04",
            "2020-03-31",
            5,
            1,
            retain_comparison=lambda a, b, report: seen.append((a, b, report)),
        )
    assert len(seen) == 1
    assert seen[0][0].frame.equals(frame) and seen[0][0].weights.equals(weights)
    assert not (seen[0][2]["full_frame_equals"] and seen[0][2]["full_positions_equal"])


def test_file_Adapter_retains_complete_compared_frames(data, tmp_path, monkeypatch):
    output = run(data)
    monkeypatch.setattr(
        driver,
        "reference",
        lambda path: {
            "path": path.name,
            "sha256": "a" * 64,
        },
    )
    report = model.comparison_report(output.result, output.result)
    driver.retain_cash_comparison(tmp_path, output.result, output.result, report)
    recorded = read_json(tmp_path / "comparison.json")
    assert recorded["retained_before_acceptance_assertion"]
    assert len(recorded["artifacts"]) == 4
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
@pytest.mark.parametrize("cut", [60, 70])
def test_mixed_asset_account_prefix_is_exact(data, cost, delay, cut):
    day = data.close.index[cut]
    output = run(data, cost, delay)
    prefix = clipped(data, day)
    partial = model.run(
        prefix,
        nominal(data).loc[:day],
        "2020-02-04",
        str(day.date()),
        cost_bps=cost,
        delay=delay,
    )
    for full, short in (
        (output.result.frame, partial.result.frame),
        (output.result.weights, partial.result.weights),
        (output.desired, partial.desired),
        (output.issued, partial.issued),
    ):
        pd.testing.assert_frame_equal(full.loc[:day], short.loc[:day], check_exact=True)
    assert [
        r for r in output.decisions if pd.Timestamp(r["signal_session"]) <= day
    ] == partial.decisions


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_future_prices_and_opinions_do_not_change_earlier_capital(data, cost, delay):
    day = data.close.index[65]
    full = run(data, cost, delay)
    changed = clipped(data, data.close.index[-1])
    changed.open.loc[changed.open.index > day] *= 1.07
    changed.close.loc[changed.close.index > day] *= 1.03
    changed.raw_close.loc[changed.raw_close.index > day] *= 1.03
    targets = nominal(data)
    for date in targets.dropna(how="all").index:
        if date > day:
            targets.loc[date] = np.roll(targets.loc[date].to_numpy(), 2)
    after = model.run(
        changed,
        targets,
        "2020-02-04",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
    )
    for before, current in (
        (full.result.frame, after.result.frame),
        (full.result.weights, after.result.weights),
        (full.issued, after.issued),
    ):
        pd.testing.assert_frame_equal(before.loc[:day], current.loc[:day], check_exact=True)
    assert [r for r in full.decisions if pd.Timestamp(r["signal_session"]) <= day] == [
        r for r in after.decisions if pd.Timestamp(r["signal_session"]) <= day
    ]


def test_terminal_V1_functions_never_called(data, monkeypatch):
    from us_quant import execution_gap_shield_v1 as terminal

    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal V1 must not execute.")

    for name in ("run", "coefficients", "verify_scalar", "verify_run"):
        monkeypatch.setattr(terminal, name, forbidden)
    monkeypatch.setattr(terminal.ShieldPolicy, "decide", forbidden)
    output = run(data)
    assert len(output.decisions) == len(output.result.frame) + 1


def test_main_close_primitives_retained_in_chronological_order(data):
    records = []
    output = run(data, retain=records.append)
    closes = [r for r in records if r["kind"] == "main_close"]
    assert len(closes) == len(output.result.frame)
    assert closes[0]["session"] == "2020-02-04"
    assert records[0]["kind"] == "decision"
    assert records[1]["kind"] == "main_close" and records[2]["kind"] == "decision"
    assert all(len(r["units"]) == len(r["held_weights"]) == model.N for r in closes)


@pytest.mark.parametrize("upper", [Fraction(1, 10), Fraction(1, 3), Fraction(7, 11)])
def test_binary64_interval_selection_still_inside(upper):
    value = model.greatest_binary64(
        {
            "feasible": True,
            "lower": model.reference.encode(Fraction(0)),
            "upper": model.reference.encode(upper),
        }
    )
    assert Fraction.from_float(value) <= upper
    assert Fraction.from_float(float(np.nextafter(value, np.inf))) > upper


def test_actual_V2_packet_entry_defers_historical_outputs(monkeypatch):
    frozen = {"study_id": model.STUDY}
    registration = {
        "sha256": "a" * 64,
        "at": "2026-11-06T23:00:00Z",
        "body": {"candidate": driver.CANDIDATE, "frozen": frozen},
    }
    ready = {"path": "synthetic-readiness.json", "sha256": "b" * 64}
    body = driver.packet_body(frozen, registration, ready)
    assert body["entries"][0]["candidate_id"] == model.ID
    assert body["entries"][0]["historical_target_derivation_proof"] is None
    assert body["candidate_accounts_or_metrics_computed"] is False
    monkeypatch.setattr(
        driver, "events", lambda kind: [registration] if kind == "registered" else []
    )
    monkeypatch.setattr(driver, "reference", lambda path: ready)
    monkeypatch.setattr(
        driver,
        "read_json",
        lambda path: (
            {**body, "created_at": "2026-11-06T23:00:01Z"}
            if path == driver.PACKET
            else {
                "historical_target_derivation_verified": False,
                "historical_new_meta_scores_targets_accounts_computed": False,
            }
        ),
    )
    assert driver.checked_packet(frozen)["schema_version"] == 3
