import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import rebalance_ticket_spillover_v1 as model
from us_quant import ssd_rebalance_ticket_spillover_v1 as driver
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError


def initial(cost=5.0, delay=1):
    dates = sessions("2020-01-02", "2020-01-10")
    anchor = previous_session(dates[0])
    prices = pd.DataFrame(100.0, index=pd.DatetimeIndex([anchor, *dates]), columns=model.ASSETS)
    target = np.zeros(14)
    target[:2] = 0.49
    signal = {
        "kind": "decision",
        "signal_session": str(anchor.date()),
        "execution_session": str((dates[0] if delay == 1 else dates[1]).date()),
        "initial_capital": True,
        "reason": "initial_capital",
        "main_asset_values_at_decision": [0.0] * 14,
        "main_cash_at_decision": 10000.0,
        "main_equity_at_decision": 10000.0,
        "desired_target": target.tolist(),
        "maximum_observed_target_gap": 0.49,
    }
    fee = (2 + cost / 10000 * 9800) / (1 + cost / 10000 * 0.98)
    post = target * (10000 - fee)
    cash, nav = 0.02 * (10000 - fee), 10000 - fee
    events = [signal]
    previous = 10000.0
    for index, date in enumerate(dates):
        fill = index == delay - 1
        if fill:
            events.append(
                {
                    "kind": "modeled_execution_input",
                    "session": str(date.date()),
                    "opening_asset_values": [0.0] * 14,
                    "cash": 10000.0,
                    "target": target.tolist(),
                }
            )
        current_nav = nav if index >= delay - 1 else 10000.0
        events.append(
            {
                "kind": "main_close",
                "session": str(date.date()),
                "asset_values": post.tolist() if index >= delay - 1 else [0.0] * 14,
                "cash": cash if index >= delay - 1 else 10000.0,
                "equity": current_nav,
                "return": current_nav / previous - 1,
                "cost": fee if fill else 0.0,
                "turnover": float(post.sum() / 10000) if fill else 0.0,
                "model_tickets": 2 if fill else 0,
            }
        )
        previous = current_nav
    return prices, events


def test_fixed_noncausal_criteria():
    policy = driver.read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate({**policy, "crossover_is_user_minimum_trade_or_turnover_cap": True})


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_archive_and_memory_same_JSONL_interface(tmp_path, cost, delay):
    prices, events = initial(cost, delay)
    path = tmp_path / "fake-trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    first_rows, second_rows = [], []
    a = model.scan(
        prices,
        prices,
        model.MemoryFeed(events),
        "2020-01-02",
        "2020-01-10",
        cost,
        delay,
        retain=lambda r: first_rows.append(json.dumps(r, allow_nan=False)),
    )
    b = model.scan(
        prices,
        prices,
        driver.ArchiveFeed(path),
        "2020-01-02",
        "2020-01-10",
        cost,
        delay,
        retain=lambda r: second_rows.append(json.dumps(r, allow_nan=False)),
    )
    assert a == b and first_rows == second_rows
    assert a["model_tickets"] == 2 and a["ticket_groups"]["initial"]["tickets"] == 2
    assert a["ticket_groups"]["below_band_spillover"]["tickets"] == 0


def test_signal_band_spillover_partition():
    before = np.zeros(14)
    before[:3] = [0.45, 0.48, 0.05]
    target = np.zeros(14)
    target[:3] = [0.48, 0.459, 0.041]
    values, desired = before * 10000, target * 10000
    signs = np.sign(desired - values)
    fee = (3 + 0.0005 * abs(desired - values).sum()) / (1 + 0.0005 * (signs * target).sum())
    post = target * (10000 - fee)
    prices = np.full(14, 100.0)
    intent = {"target": target, "gaps": abs(target - before), "initial": False}
    opening_input = {"opening_asset_values": values, "target": target, "cash": 200.0}
    close = {
        "asset_values": post,
        "cash": 0.02 * (10000 - fee),
        "cost": float(fee),
        "turnover": float(abs(post - values).sum() / 10000),
        "model_tickets": 3,
    }
    result = model.inspect_execution(intent, opening_input, close, prices, prices, 5.0)
    groups = [row["group"] for row in result["tickets"]]
    assert groups.count("direct_band_coordinate") == 1
    assert groups.count("below_band_spillover") == 2


@pytest.mark.parametrize("cost,crossover", [(5.0, 2000.0), (20.0, 500.0)])
def test_commission_crossover_is_arithmetic_not_trade_cap(cost, crossover):
    delay = 1 if cost == 5.0 else 2
    prices, events = initial(cost, delay)
    output = model.scan(
        prices, prices, model.MemoryFeed(events), "2020-01-02", "2020-01-10", cost, delay
    )
    assert output["commission_notional_crossover_usd"] == crossover
    assert not output["crossover_is_user_trade_cap"]
    assert output["fixed_fee_dominant_ticket_count"] == 0


def test_adverse_open_close_marking_uses_actual_nonconstant_prices():
    before = np.zeros(14)
    before[0] = 4000
    target = np.zeros(14)
    target[:2] = [0.5, 0.48]
    fee = (2 + 0.0005 * 5800) / (1 + 0.0005 * 0.98)
    after = target * (10000 - fee)
    opening, closing = np.full(14, 100.0), np.full(14, 100.0)
    opening[:2], closing[:2] = [80.0, 105.0], [60.0, 100.0]
    close = {
        "asset_values": after * (closing / opening),
        "cash": 0.02 * (10000 - fee),
        "cost": float(fee),
        "turnover": float(abs(after - before).sum() / 10000),
        "model_tickets": 2,
    }
    intent = {"target": target, "gaps": abs(target - before / 10000), "initial": False}
    input_row = {"opening_asset_values": before, "cash": 6000.0, "target": target}
    result = model.inspect_execution(intent, input_row, close, opening, closing, 5.0)
    assert abs(result["actual_notional_usd"] - abs(after - before).sum()) <= model.MONEY
    assert max(result["coherence_errors_usd"].values()) <= model.MONEY


def test_future_does_not_change_prefix():
    prices, events = initial()
    cutoff = pd.Timestamp("2020-01-07")
    records = []
    model.scan(
        prices,
        prices,
        model.MemoryFeed(events),
        "2020-01-02",
        "2020-01-10",
        5,
        1,
        retain=records.append,
    )
    changed = prices.copy()
    changed.loc[changed.index > cutoff] *= 2
    changed_events = [dict(row) for row in events]
    changed_events[-1]["equity"] *= 2
    prefix = []
    model.scan(
        changed,
        changed,
        model.MemoryFeed(changed_events),
        "2020-01-02",
        str(cutoff.date()),
        5,
        1,
        retain=prefix.append,
    )
    assert prefix == [r for r in records if pd.Timestamp(r["session"]) <= cutoff]


def test_fee_ticket_or_missing_close_failure_is_not_success():
    prices, events = initial()
    changed = [dict(row) for row in events]
    changed[2]["model_tickets"] = 1
    with pytest.raises(QuantError):
        model.scan(prices, prices, model.MemoryFeed(changed), "2020-01-02", "2020-01-10", 5, 1)
    with pytest.raises(QuantError):
        model.scan(prices, prices, model.MemoryFeed(events[:-1]), "2020-01-02", "2020-01-10", 5, 1)


def test_no_old_solver_or_diagnosis_called(monkeypatch):
    from us_quant import backtest, bt_audit, cash_funded_accounting_v2, shield_ticket_anatomy_v1

    def blocked(*a, **k):
        raise AssertionError("A past account/solver/BT/diagnosis was executed.")

    for module, name in (
        (backtest, "rebalance"),
        (bt_audit, "independent_equity"),
        (cash_funded_accounting_v2, "simulate"),
        (shield_ticket_anatomy_v1, "inspect"),
    ):
        monkeypatch.setattr(module, name, blocked)
    prices, events = initial()
    result = model.scan(prices, prices, model.MemoryFeed(events), "2020-01-02", "2020-01-10", 5, 1)
    assert result["formal_closes"] == 7


def test_banned_reader_before_open(tmp_path, monkeypatch):
    protected = tmp_path / "mock-protected.json"
    alias = tmp_path / "alias.json"
    alias.symlink_to(protected)
    monkeypatch.setattr(driver.prior.saved, "PROTECTED", protected)
    opened, original = [], Path.open

    def watch(path, *args, **kwargs):
        if path in (protected, alias):
            opened.append(path)
            raise AssertionError("A prohibited reader reached open.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watch)
    for path in (protected, alias):
        for fn in (
            driver.content_path,
            driver.read_json,
            driver.file_digest,
            driver.reference,
            driver.ArchiveFeed,
        ):
            with pytest.raises(PermissionError):
                fn(path)
    assert not opened


@pytest.mark.parametrize("value", [True, np.bool_(True), 1, pd.NaT])
def test_dates_and_amounts_reject_invalid_coercion(value):
    with pytest.raises(QuantError):
        model.day(value)
    if isinstance(value, bool | np.bool_):
        with pytest.raises(QuantError):
            model.number(value)
        with pytest.raises(QuantError):
            model.vector([value, *([0.0] * 13)])


def full_toy(start, end, cost, delay):
    dates = sessions(start, end)
    anchor = previous_session(dates[0])
    target = np.zeros(14)
    target[:2] = 0.49
    execution = dates[delay - 1]
    fee = (2 + cost / 10000 * 9800) / (1 + cost / 10000 * 0.98)
    post, cash, nav = target * (10000 - fee), 0.02 * (10000 - fee), 10000 - fee
    events = [
        {
            "kind": "decision",
            "signal_session": str(anchor.date()),
            "execution_session": str(execution.date()),
            "initial_capital": True,
            "reason": "initial_capital",
            "main_asset_values_at_decision": [0.0] * 14,
            "main_cash_at_decision": 10000.0,
            "main_equity_at_decision": 10000.0,
            "desired_target": target.tolist(),
            "maximum_observed_target_gap": 0.49,
        }
    ]
    previous = 10000.0
    for date in dates:
        fill = date == execution
        if fill:
            events.append(
                {
                    "kind": "modeled_execution_input",
                    "session": str(date.date()),
                    "opening_asset_values": [0.0] * 14,
                    "cash": 10000.0,
                    "target": target.tolist(),
                }
            )
        marked = nav if date >= execution else 10000.0
        events.append(
            {
                "kind": "main_close",
                "session": str(date.date()),
                "asset_values": post.tolist() if date >= execution else [0.0] * 14,
                "cash": cash if date >= execution else 10000.0,
                "equity": marked,
                "return": marked / previous - 1,
                "cost": float(fee) if fill else 0.0,
                "turnover": float(post.sum() / 10000) if fill else 0.0,
                "model_tickets": 2 if fill else 0,
            }
        )
        previous = marked
    return events, float(fee), float(post.sum())


def test_real_driver_eight_book_prefix_and_export_interface(tmp_path, monkeypatch):
    dates = sessions("2015-08-10", "2026-10-05")
    prices = pd.DataFrame(100.0, index=dates, columns=model.ASSETS)
    open_path, close_path = tmp_path / "open.csv", tmp_path / "close.csv"
    prices.to_csv(open_path)
    prices.to_csv(close_path)
    books = {}
    for family in ("dyadic", "adanormalhedge"):
        for years, start in ((10, "2016-10-06"), (5, "2021-10-06")):
            for scenario, cost, delay in (("base", 5.0, 1), ("stress", 20.0, 2)):
                key = f"{family}:{years}y-{scenario}"
                events, fee, notional = full_toy(start, "2026-10-05", cost, delay)
                trace = tmp_path / (key.replace(":", "-") + ".jsonl")
                trace.write_text("".join(json.dumps(row) + "\n" for row in events))
                books[key] = {
                    "path_key": f"{years}y-{scenario}",
                    "inputs": {"chronological-trace.jsonl": driver.reference(trace)},
                    "saved_metrics": {
                        "start": start,
                        "end": "2026-10-05",
                        "sessions": len(sessions(start, "2026-10-05")),
                    },
                    "saved_execution": {
                        "model_order_count": 2,
                        "total_cost_usd": fee,
                        "executed_notional_usd": notional,
                    },
                }
    catalogue = {
        "books": books,
        "source_prices": {
            "open": driver.reference(open_path),
            "close": driver.reference(close_path),
        },
    }
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "INTENT", tmp_path / "intent.json")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path / "diagnostic.json")
    monkeypatch.setattr(driver, "PUBLIC", tmp_path / "public.json")
    driver.INTENT.write_text(json.dumps({"synthetic": True}))
    cp = {
        "entire_state_readback_equal": True,
        "frozen_implementation": driver.reference(driver.INTENT),
        "checked_at": "2026-11-08T03:00:00+00:00",
        "actual_continuity_checkpoint_readback": {"saved_at": "2026-11-08T02:00:00+00:00"},
    }
    (tmp_path / "actual-postfreeze-checkpoint-v1.json").write_text(json.dumps(cp))
    monkeypatch.setattr(
        driver, "checked_intent", lambda: {"created_at": "2026-11-08T01:00:00+00:00"}
    )
    monkeypatch.setattr(driver, "checked_inputs", lambda: (catalogue, []))
    monkeypatch.setattr(driver, "progress", lambda *a, **k: None)
    monkeypatch.setattr(driver.prior, "_guard", lambda: None)
    monkeypatch.setattr(driver.prior.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(driver.prior.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(driver.prior.saved, "counts", lambda: {"synthetic_journal": 1570})
    driver.audit()
    result = driver.read_json(driver.OUTPUT)
    assert len(result["results"]) == 8
    assert sum(v["summary"]["formal_closes"] for v in result["results"].values()) == 15064
    assert sum(p["status"] == "exact" for p in result["prefixes"]) == 28
    assert sum(p["status"] == "before_funding_not_applicable" for p in result["prefixes"]) == 12
    driver.export()
    public = driver.read_json(driver.PUBLIC)
    assert public["new_financial_trials"] == 0 and not public["investment_objective_verified"]
    assert not public["counterfactual_selective_rebalance_policy_evaluated"]
    with pytest.raises(QuantError):
        driver.audit()
