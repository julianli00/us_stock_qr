import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import shield_ticket_anatomy_v1 as model
from us_quant import ssd_shield_ticket_anatomy_v1 as driver
from us_quant.calendar import next_session
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.storage import read_json


def target(symbol="SPY"):
    value = np.zeros(model.N)
    value[model.ASSETS.index(symbol)] = 0.98
    return value


def intent(
    current=None,
    previous=None,
    nominal=None,
    previous_nominal=None,
    fraction=1.0,
    previous_fraction=None,
    delay=1,
):
    current = target() if current is None else current
    nominal = current.copy() if nominal is None else nominal
    signal = pd.Timestamp("2020-01-02")
    execution = signal
    for _ in range(delay):
        execution = next_session(execution)
    return model.SavedIntent(
        signal,
        execution,
        signal,
        current,
        nominal,
        fraction,
        previous,
        previous_nominal,
        previous_fraction,
    )


def example(cost=5, delay=1, before=None, cash=10000.0, recommendation=None):
    before = np.zeros(model.N) if before is None else before
    recommendation = intent(delay=delay) if recommendation is None else recommendation
    opening, closing = np.ones(model.N), np.ones(model.N)
    previous_nav = float(before.sum() + cash)
    values, after_cash, fee, turnover, tickets = rebalance(
        before,
        cash,
        recommendation.target,
        cost,
        1,
    )
    after = values.copy()
    nav = float(after.sum() + after_cash)
    return model.SavedStep(
        recommendation.execution,
        opening,
        closing,
        before,
        after,
        cash,
        after_cash,
        previous_nav,
        nav,
        nav / previous_nav - 1,
        fee,
        turnover,
        tickets,
        cost,
        delay,
        recommendation,
    )


def test_frozen_ticket_policy_exact():
    policy = read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate({**policy, "notional_bucket_boundaries_USD": [1.0, 20.0, 100.0]})


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.nextafter(1e-6, np.inf), "below_1"),
        (0.99999, "below_1"),
        (1.0, "1_to_10"),
        (9.9999, "1_to_10"),
        (10.0, "10_to_100"),
        (99.9999, "10_to_100"),
        (100.0, "100_plus"),
    ],
)
def test_original_strict_threshold_and_fixed_bucket_endpoints(value, expected):
    assert model.bucket(value) == expected


@pytest.mark.parametrize("value", [0.0, 1e-6, -1.0, np.nan])
def test_non_ticket_notional_is_not_bucketed(value):
    with pytest.raises(QuantError):
        model.bucket(value)


@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_independent_original_cash_synthetic_ticket_reconciliation(cost, delay):
    row = model.inspect(example(cost, delay))
    assert row["actual_model_tickets"] == len(row["tickets"]) == 1
    assert row["provenance"] == "initial_funded_request"
    assert row["maximum_money_identity_error_USD"] <= 1e-6
    assert row["fee_base_identity_error_USD"] <= 1e-6


def test_partial_prefee_noop_crossing_is_retained_not_causal_wealth():
    desired = np.zeros(model.N)
    desired[0] = desired[model.ASSETS.index("GLD")] = 0.49
    held = np.zeros(model.N)
    held[0] = 4900
    rec = intent(
        desired,
        previous=target(),
        previous_nominal=target(),
        previous_fraction=1.0,
    )
    row = model.inspect(example(before=held, cash=5100, recommendation=rec))
    spy = next(t for t in row["tickets"] if t["asset"] == "SPY")
    assert spy["prefee_target_gap_USD"] == 0
    assert spy["prefee_noop_to_executed_ticket"]
    assert not row["fee_base_crossing_is_causal_wealth_attribution"]


def test_same_target_fill_is_not_a_new_nominal_change():
    rec = intent(previous=target(), previous_nominal=target(), previous_fraction=1.0)
    assert (
        model.inspect(example(recommendation=rec))["provenance"] == "same_issued_target_rebalance"
    )


def test_same_nominal_new_fraction_is_distinct():
    old_target = target() * 0.5
    old_target[model.ASSETS.index("BIL")] = 0.49
    rec = intent(
        previous=old_target,
        previous_nominal=target(),
        previous_fraction=0.5,
    )
    assert rec.label(1) == "same_nominal_changed_fraction_target"


def test_changed_nominal_and_target_are_not_isolated_causes():
    rec = intent(
        target("GLD"),
        previous=target(),
        previous_nominal=target(),
        previous_fraction=1.0,
    )
    assert rec.label(1) == "changed_nominal_and_target"


def test_future_source_or_wrong_delay_is_rejected():
    rec = intent()
    with pytest.raises(QuantError):
        replace(rec, source_known_at=rec.execution).label(1)
    with pytest.raises(QuantError):
        rec.label(2)


def test_hidden_ticket_and_wrong_fee_are_explicit_errors():
    step = example()
    for bad in (
        replace(step, model_tickets=step.model_tickets + 1),
        replace(step, paid_cost=step.paid_cost + 0.1),
        replace(step, intent=None),
    ):
        with pytest.raises(QuantError):
            model.inspect(bad)


def test_no_fill_prefix_keeps_zero_days_without_fake_fee_identity():
    units = target() * 10000
    step = model.SavedStep(
        pd.Timestamp("2020-01-03"),
        np.ones(model.N),
        np.ones(model.N),
        units,
        units.copy(),
        200,
        200,
        10000,
        10000,
        0,
        0,
        0,
        0,
        5,
        1,
        None,
    )
    row = model.inspect(step)
    summary = model.aggregate([row])
    assert summary["formal_rows"] == 1 and summary["model_tickets"] == 0
    assert summary["maximum_fee_base_identity_error_USD"] is None
    assert summary["fee_base_identity_applicable_days"] == 0


def test_scheduled_zero_ticket_fill_is_not_dropped():
    units = target() * 10000
    rec = intent(previous=target(), previous_nominal=target(), previous_fraction=1.0)
    row = model.inspect(example(before=units, cash=200, recommendation=rec))
    assert row["scheduled_fill"] and row["actual_model_tickets"] == 0
    assert model.aggregate([row])["scheduled_zero_ticket_fills"] == 1


def test_synthetic_memory_prefix_is_signed_zero_aware_and_row_local():
    rows = [model.inspect(example()), model.inspect(example(cost=20, delay=2))]
    assert driver.exact_encoding(rows[:1]) == driver.exact_encoding([model.inspect(example())])
    assert driver.exact_encoding(0.0) != driver.exact_encoding(-0.0)
    summary = model.aggregate(rows)
    assert sum(v["tickets"] for v in summary["notional_buckets"].values()) == 2


def test_actual_diagnostic_never_calls_solver_account_or_financial_metrics(monkeypatch):
    step = example()
    from us_quant import cash_funded_accounting_v2, metrics
    from us_quant import execution_gap_shield_v2 as terminal

    def forbidden(*args, **kwargs):
        raise AssertionError("A saved-record diagnostic must not execute a policy or account.")

    monkeypatch.setattr(cash_funded_accounting_v2, "rebalance", forbidden)
    monkeypatch.setattr(cash_funded_accounting_v2, "simulate", forbidden)
    monkeypatch.setattr(metrics, "performance", forbidden)
    monkeypatch.setattr(terminal, "run", forbidden)
    monkeypatch.setattr(terminal, "verify_run", forbidden)
    assert model.inspect(step)["actual_model_tickets"] == 1


def test_saved_file_Adapter_and_memory_Adapter_share_step_Interface(
    market_factory, tmp_path, monkeypatch
):
    data = market_factory(start="2020-01-02", end="2020-01-08", symbols=model.ASSETS)
    dates = data.close.index[1:4]
    signal = data.close.index[0]
    desired = target()
    issued = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    issued.loc[signal] = desired
    nominal = issued.copy()
    units, cash, previous_nav = np.zeros(model.N), 10000.0, 10000.0
    rows, weights, closes = [], [], []
    initial = {
        "signal_session": str(signal.date()),
        "execution_session": str(dates[0].date()),
        "initial_capital": True,
        "desired_target": desired.tolist(),
        "assessment": {
            "applied_nominal_fraction": 1.0,
            "source_opinion": {"source_opinion_known_at": str(signal.date())},
        },
    }
    decisions = [initial]
    trace = [
        {"kind": "reference_input", "as_of": str(signal.date()), "nominal": desired.tolist()},
        {"kind": "decision", **initial},
    ]
    for i, day in enumerate(dates):
        cost = turnover = 0.0
        tickets = 0
        if i == 0:
            values, cash, cost, turnover, tickets = rebalance(
                units * data.open.loc[day].to_numpy(),
                cash,
                desired,
                5,
                1,
            )
            units = values / data.open.loc[day].to_numpy()
        values = units * data.close.loc[day].to_numpy()
        nav = float(values.sum() + cash)
        held = values / nav
        row = [
            nav,
            nav / previous_nav - 1,
            cash,
            held.sum(),
            turnover,
            cost,
            tickets,
            float(data.risk_free.loc[day]),
        ]
        rows.append(row)
        weights.append(held)
        close = dict(
            zip(
                (
                    "equity",
                    "return",
                    "cash",
                    "gross_exposure",
                    "turnover",
                    "cost",
                    "model_tickets",
                    "risk_free",
                ),
                row,
                strict=True,
            )
        )
        close.update(
            {
                "kind": "main_close",
                "session": str(day.date()),
                "units": units.tolist(),
                "asset_values": values.tolist(),
                "held_weights": held.tolist(),
            }
        )
        trace.append(close)
        closes.append(close)
        decision = {
            "signal_session": str(day.date()),
            "execution_session": None,
            "initial_capital": False,
            "desired_target": None,
            "assessment": {},
        }
        decisions.append(decision)
        trace.append({"kind": "decision", **decision})
        previous_nav = nav
    frame = pd.DataFrame(
        rows,
        index=dates,
        columns=[
            "equity",
            "return",
            "cash",
            "gross_exposure",
            "turnover",
            "cost",
            "orders",
            "risk_free",
        ],
    )
    case = {"expected_formal_rows": len(frame)}
    for name, value in (
        ("account.csv", frame),
        ("weights.csv", pd.DataFrame(weights, index=dates, columns=model.ASSETS)),
        ("issued.csv", issued),
    ):
        path = tmp_path / name
        path.write_text(value.to_csv(float_format="%.17g"))
        case[name] = {"path": str(path), "sha256": "a" * 64}
    for name, value in (("decisions.json", {"decisions": decisions, "pending_after_window": {}}),):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        case[name] = {"path": str(path), "sha256": "a" * 64}
    path = tmp_path / "chronological-trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    case[path.name] = {"path": str(path), "sha256": "a" * 64}
    monkeypatch.setattr(driver.base, "checked_reference", lambda ref: type(path)(ref["path"]))
    adapter = driver.SavedBookAdapter(data, nominal, case, 5, 1)
    observed = [model.inspect(step) for step in adapter.steps()]
    assert len(observed) == 3 and observed[0]["actual_model_tickets"] == 1
    assert sum(r["actual_model_tickets"] for r in observed) == frame["orders"].sum()
