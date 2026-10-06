from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import us_quant.paper_lab as lab
from us_quant.config import QuantError
from us_quant.free_quotes import ReferencePrice
from us_quant.paper import Outcome, PlannedOrder, submit_paper_ioc
from us_quant.paper_lab import (
    ExperimentLedger,
    LabAccount,
    experimental_order,
    regular_session,
    run_smoke,
)
from us_quant.storage import digest_json, write_json

NOW = pd.Timestamp("2026-09-29 13:36:00Z")


def reference(price=700.0):
    return ReferencePrice(
        "SPY",
        price,
        "2026-09-29T13:34:00+00:00",
        "2026-09-29T13:35:00+00:00",
        NOW.isoformat(),
        "yahoo_public_completed_1m_bar",
        "a" * 64,
    )


@pytest.fixture
def account(paper_config):
    return LabAccount(paper_config.account, 1000000.0, 1000000.0, 1000000.0, {})


@pytest.fixture
def ledger(tmp_path, paper_config, account, monkeypatch):
    monkeypatch.setattr(lab, "now_utc", lambda: NOW)
    ledger = ExperimentLedger(tmp_path / "lab.sqlite3", create=True)
    ledger.authorize(paper_config, account)
    yield ledger
    ledger.close()


class FakeLabBroker:
    def __init__(self, account, outcome="Filled", *, stop_file=None):
        self.current = account
        self.outcome = outcome
        self.orders = []
        self.stop_file = stop_file

    def account(self):
        return self.current

    def reference(self, symbol):
        assert symbol == "SPY"
        return reference()

    def send(self, order):
        self.orders.append(order)
        if self.outcome == "timeout":
            raise TimeoutError("unknown after sending")
        if self.outcome == "Cancelled" and order.action == "BUY":
            return Outcome("Cancelled", 0, 0.0, 1, 101, None)
        if self.outcome == "partial":
            return Outcome("Cancelled", 0.5, order.limit_price, 1, 101, 1.0)
        self.current = replace(
            self.current,
            positions={"SPY": 1} if order.action == "BUY" else {},
            execution_refs=(*self.current.execution_refs, order.order_ref),
        )
        if self.stop_file and order.action == "BUY":
            Path(self.stop_file).touch()
        price = 700.0 if order.action == "BUY" else 700.05
        return Outcome(
            "Filled",
            1,
            price,
            len(self.orders),
            100 + len(self.orders),
            None if self.outcome == "no_fee" else 1.0,
        )


def test_lab_budget_is_not_the_million_dollar_paper_account(ledger, paper_config):
    result = ledger.status(paper_config)
    assert result["experimental_capital_usd"] == 10000
    assert result["expected_broker_equity_at_registration"] == 1000000
    assert not result["broker_balance_reset_performed"]
    assert not result["objective_verified"] and not result["live_account_authority"]
    assert paper_config.allow_submit is False


def test_dry_run_never_places_an_order_or_reserves_a_trial(ledger, paper_config, account):
    broker = FakeLabBroker(account)
    result = run_smoke(paper_config, ledger, broker, execute=False)
    assert result["orders_sent"] == 0 and not broker.orders
    assert not ledger.status(paper_config)["experiments"]


def test_realistic_paper_smoke_roundtrip_journals_two_fills_and_returns_flat(
    ledger, paper_config, account
):
    broker = FakeLabBroker(account)
    result = run_smoke(paper_config, ledger, broker, execute=True)
    assert result["status"] == "completed" and result["orders_sent"] == 2
    assert result["realized_net_pnl"] == pytest.approx(-1.95)
    assert result["sleeve_equity_after"] == pytest.approx(9998.05)
    assert broker.current.positions == {}
    assert [order.action for order in broker.orders] == ["BUY", "SELL"]
    assert all(order.quantity == 1 and order.limit_price <= 1000 for order in broker.orders)
    assert not result["qualification_evidence"] and not result["objective_verified"]
    with pytest.raises(QuantError, match="replay"):
        run_smoke(paper_config, ledger, broker, execute=True)


def test_prior_completed_day_cannot_be_repeated_even_if_broker_history_is_empty(
    ledger, paper_config, account
):
    broker = FakeLabBroker(account)
    run_smoke(paper_config, ledger, broker, execute=True)
    broker.current = replace(broker.current, execution_refs=())
    with pytest.raises(QuantError, match="duplicate"):
        run_smoke(paper_config, ledger, broker, execute=True)
    assert len(broker.orders) == 2


def test_unfilled_entry_does_not_invent_an_exit_or_fill(ledger, paper_config, account):
    broker = FakeLabBroker(account, "Cancelled")
    result = run_smoke(paper_config, ledger, broker, execute=True)
    assert result["status"] == "not_filled" and result["orders_sent"] == 1
    assert len(broker.orders) == 1 and not broker.current.positions


@pytest.mark.parametrize("outcome", ["timeout", "partial", "stop_after_buy"])
def test_failed_or_interrupted_paper_test_is_quarantined(ledger, paper_config, account, outcome):
    broker = FakeLabBroker(
        account,
        "Filled" if outcome == "stop_after_buy" else outcome,
        stop_file=paper_config.kill_switch_file if outcome == "stop_after_buy" else None,
    )
    with pytest.raises((QuantError, TimeoutError)):
        run_smoke(paper_config, ledger, broker, execute=True)
    assert len(broker.orders) == 1
    assert ledger.status(paper_config)["experiments"][0]["status"] == "needs_reconciliation"


def test_unknown_commission_is_not_reported_as_zero(ledger, paper_config, account):
    broker = FakeLabBroker(account, "no_fee")
    result = run_smoke(paper_config, ledger, broker, execute=True)
    assert result["status"] == "completed_costs_pending"
    assert result["commission_usd"] is None and result["realized_net_pnl"] is None
    with pytest.raises(QuantError, match="fee-pending"):
        ledger.reserve(paper_config, account, pd.Timestamp("2026-09-30"))


@pytest.mark.parametrize(
    "change",
    [
        {"reported_usd_cash": -1},
        {"available_funds": float("nan")},
        {"account": "U1234567"},
        {"open_order_refs": ("someone_elses_order",)},
        {"positions": {"SPY": -1}},
        {"positions": {"SPY": 0.5}},
    ],
)
def test_account_guards_remain_strict_for_experiments(paper_config, account, change):
    with pytest.raises(QuantError):
        replace(account, **change).validate(paper_config)


def test_foreign_positions_are_not_liquidated_for_a_smoke_test(ledger, paper_config, account):
    broker = FakeLabBroker(replace(account, positions={"QQQ": 5}))
    with pytest.raises(QuantError, match="flat account"):
        run_smoke(paper_config, ledger, broker, execute=True)
    assert not broker.orders


def test_broker_reset_is_detected_without_resetting_experiment_history(
    ledger, paper_config, account
):
    broker = FakeLabBroker(
        replace(account, net_liquidation=10000, reported_usd_cash=10000, available_funds=10000)
    )
    with pytest.raises(QuantError, match="reset needs explicit"):
        run_smoke(paper_config, ledger, broker, execute=True)
    assert not broker.orders


def test_free_price_staleness_limits_and_market_hours():
    assert regular_session(NOW) == pd.Timestamp("2026-09-29")
    for value in (
        "2026-09-29 13:29Z",
        "2026-09-29 13:31Z",
        "2026-09-29 19:50Z",
        "2026-09-27 15:00Z",
    ):
        with pytest.raises(QuantError):
            regular_session(pd.Timestamp(value))
    with pytest.raises(QuantError, match="exposure"):
        experimental_order(reference(1500), "BUY", "test", NOW)
    with pytest.raises(QuantError, match="stale"):
        experimental_order(reference(), "BUY", "test", NOW + pd.Timedelta(minutes=5))
    with pytest.raises(QuantError, match="restricted"):
        experimental_order(replace(reference(), symbol="QQQ"), "BUY", "test", NOW)


def test_lab_authorization_detects_code_or_config_changes(ledger, paper_config, monkeypatch):
    with pytest.raises(QuantError, match="changed"):
        ledger.verified_authorization(replace(paper_config, max_order_notional=5000))
    monkeypatch.setattr(lab, "lab_fingerprint", lambda: "new")
    with pytest.raises(QuantError, match="changed"):
        ledger.verified_authorization(paper_config)


def test_expired_authorization_remains_readable_but_cannot_trade(ledger, paper_config, monkeypatch):
    monkeypatch.setattr(lab, "now_utc", lambda: NOW + pd.Timedelta(days=31))
    assert ledger.status(paper_config)["experimental_capital_usd"] == 10000
    with pytest.raises(QuantError, match="expired"):
        ledger.verified_authorization(paper_config)


def test_wait_is_bounded_and_never_targets_another_day(monkeypatch):
    monkeypatch.setattr(lab, "now_utc", lambda: pd.Timestamp("2026-09-29 12:00Z"))
    with pytest.raises(QuantError, match="bounded wait"):
        lab.wait_for_regular_window(3600)
    with pytest.raises(QuantError, match="at most one hour"):
        lab.wait_for_regular_window(3601)
    monkeypatch.setattr(lab, "now_utc", lambda: pd.Timestamp("2026-09-27 13:00Z"))
    with pytest.raises(QuantError, match="holiday"):
        lab.wait_for_regular_window(3600)


def test_unknown_experiment_cannot_reserve_a_ticket(ledger):
    order = experimental_order(reference(), "BUY", "unknown", NOW)
    with pytest.raises(QuantError, match="currently reserved"):
        ledger.ticket("unknown", order, reference())


def test_transport_rejects_live_or_wrong_port_before_order_submission(paper_config):
    from ib_async import Stock

    contract = Stock("SPY", "SMART", "USD", conId=756733)
    order = PlannedOrder("SPY", "BUY", 1, 700.0, "lab_test")
    config = replace(paper_config, allow_submit=True)
    ib = SimpleNamespace(
        isConnected=lambda: True,
        managedAccounts=lambda: [config.account],
        client=SimpleNamespace(host=config.host, port=4001),
    )
    with pytest.raises(QuantError, match="transport identity"):
        submit_paper_ioc(ib, config, contract, order)
    ib.client.port = config.port
    ib.managedAccounts = lambda: ["U1234567"]
    with pytest.raises(QuantError, match="transport identity"):
        submit_paper_ioc(ib, config, contract, order)


def test_shared_transport_preserves_ioc_paper_routing(paper_config):
    from ib_async import Stock

    config = replace(paper_config, allow_submit=True)
    contract = Stock("SPY", "SMART", "USD", conId=756733)
    planned = PlannedOrder("SPY", "BUY", 1, 700.5, "lab_test")
    orders = []
    fee = SimpleNamespace(execId="fill-1", currency="USD", commission=1.0)
    status = SimpleNamespace(status="Filled", filled=1, avgFillPrice=700.0, orderId=10, permId=20)

    def place_order(actual_contract, actual_order):
        assert actual_contract == contract
        orders.append(actual_order)
        return SimpleNamespace(
            isDone=lambda: True, orderStatus=status, fills=[SimpleNamespace(commissionReport=fee)]
        )

    ib = SimpleNamespace(
        isConnected=lambda: True,
        managedAccounts=lambda: [config.account],
        client=SimpleNamespace(host=config.host, port=config.port),
        placeOrder=place_order,
    )
    outcome = submit_paper_ioc(ib, config, contract, planned)
    assert outcome.status == "Filled" and outcome.commission == 1
    assert orders[0].account == config.account and orders[0].tif == "IOC"
    assert orders[0].outsideRth is False and orders[0].lmtPrice == 700.5


def test_readonly_validation_error_returns_without_waiting_or_cancelling(paper_config):
    from ib_async import Stock

    config = replace(paper_config, allow_submit=True)
    status = SimpleNamespace(
        status="ValidationError", filled=0, avgFillPrice=0, orderId=5, permId=0
    )
    trade = SimpleNamespace(
        isDone=lambda: False,
        orderStatus=status,
        fills=[],
        log=[SimpleNamespace(errorCode=321)],
    )

    def unexpected(*args, **kwargs):
        pytest.fail("A known pre-acceptance validation rejection needs no wait or cancel request.")

    ib = SimpleNamespace(
        isConnected=lambda: True,
        managedAccounts=lambda: [config.account],
        client=SimpleNamespace(host=config.host, port=config.port),
        placeOrder=lambda *args: trade,
        waitOnUpdate=unexpected,
        cancelOrder=unexpected,
    )
    result = submit_paper_ioc(
        ib,
        config,
        Stock("SPY", "SMART", "USD", conId=756733),
        PlannedOrder("SPY", "BUY", 1, 700.0, "lab_test"),
    )
    assert result.status == "ValidationError" and result.filled == 0
    assert result.permanent_id == 0 and result.error_codes == (321,)


@pytest.mark.parametrize("filled,permanent_id", [(1, 0), (0, 900)])
def test_ambiguous_validation_errors_still_require_reconciliation(
    paper_config, filled, permanent_id
):
    from ib_async import Stock

    config = replace(paper_config, allow_submit=True)
    state = {"connected": True}
    status = SimpleNamespace(
        status="ValidationError", filled=filled, avgFillPrice=0, orderId=5, permId=permanent_id
    )
    trade = SimpleNamespace(
        isDone=lambda: False, orderStatus=status, fills=[], log=[SimpleNamespace(errorCode=321)]
    )

    def place(*args):
        state["connected"] = False
        return trade

    ib = SimpleNamespace(
        isConnected=lambda: state["connected"],
        managedAccounts=lambda: [config.account],
        client=SimpleNamespace(host=config.host, port=config.port),
        placeOrder=place,
    )
    with pytest.raises(QuantError, match="uncertain"):
        submit_paper_ioc(
            ib,
            config,
            Stock("SPY", "SMART", "USD", conId=756733),
            PlannedOrder("SPY", "BUY", 1, 700.0, "lab_test"),
        )


def rejected_attempt(ledger, config, account, *, record_outcome=True):
    experiment = ledger.reserve(config, account, pd.Timestamp("2026-09-29"))
    order = experimental_order(reference(), "BUY", experiment, NOW)
    ledger.ticket(experiment, order, reference())
    if record_outcome:
        ledger.outcome(order.order_ref, Outcome("ValidationError", 0, 0, 5, 0, None, (321,)))
    ledger.finish(
        experiment,
        "needs_reconciliation",
        {
            "status": "needs_reconciliation",
            "fills": [],
            "orders_sent": 1,
            "experiment_id": experiment,
            "objective_verified": False,
        },
    )
    return experiment


def permission(granted=True):
    return {
        "mode": "broker_what_if_only",
        "order_entry_available": granted,
        "actual_orders_sent": 0,
        "error_codes": [] if granted else [321],
        "what_if_status": "PreSubmitted" if granted else "ValidationError",
    }


def test_reconciliation_retains_history_and_does_not_allow_same_day_retry(
    ledger, paper_config, account
):
    identifier = rejected_attempt(ledger, paper_config, account)
    before = ledger.connection.execute("SELECT result_json FROM experiments").fetchone()[0]
    tickets_before = ledger.connection.execute("SELECT * FROM tickets").fetchall()
    result = ledger.reconcile_rejection(paper_config, account, permission(), identifier)
    assert result["orders_sent"] == 0 and result["original_history_retained"]
    assert not result["risk_limits_changed"] and not result["same_day_retry_allowed"]
    audit = json.loads(
        ledger.connection.execute("SELECT body FROM reconciliation_audit").fetchone()[0]
    )
    assert audit["previous_result"] == json.loads(before)
    assert ledger.connection.execute("SELECT * FROM tickets").fetchall() == tickets_before
    assert ledger.status(paper_config)["experiments"][0]["status"] == "not_filled"
    with pytest.raises(QuantError, match="duplicate"):
        ledger.reserve(paper_config, account, pd.Timestamp("2026-09-29"))


@pytest.mark.parametrize("issue", ["readonly", "position", "execution", "open_order"])
def test_reconciliation_cannot_clear_a_blocker_or_broker_exposure(
    ledger, paper_config, account, issue
):
    identifier = rejected_attempt(ledger, paper_config, account)
    current = account
    proof = permission(issue != "readonly")
    if issue == "position":
        current = replace(account, positions={"SPY": 1})
    elif issue == "execution":
        current = replace(account, execution_refs=(f"{identifier}_buy",))
    elif issue == "open_order":
        current = replace(account, open_order_refs=(f"{identifier}_buy",))
    with pytest.raises(QuantError):
        ledger.reconcile_rejection(paper_config, current, proof, identifier)
    assert ledger.status(paper_config)["experiments"][0]["status"] == "needs_reconciliation"


def test_readonly_status_survives_code_revision_but_execution_does_not(
    ledger, paper_config, account, monkeypatch
):
    identifier = rejected_attempt(ledger, paper_config, account)
    monkeypatch.setattr(lab, "lab_fingerprint", lambda: "reviewed-new-engine")
    assert not ledger.status(paper_config)["execution_code_matches_authorization"]
    with pytest.raises(QuantError, match="changed"):
        ledger.verified_authorization(paper_config)
    with pytest.raises(QuantError, match="accept-code-update"):
        ledger.reconcile_rejection(paper_config, account, permission(), identifier)
    result = ledger.reconcile_rejection(
        paper_config, account, permission(), identifier, accept_code_update=True
    )
    assert not result["risk_limits_changed"]
    assert ledger.status(paper_config)["execution_code_matches_authorization"]
    audit = json.loads(
        ledger.connection.execute("SELECT body FROM reconciliation_audit").fetchone()[0]
    )
    assert audit["previous_authorization"]["engine_sha256"] != "reviewed-new-engine"
    assert audit["revised_authorization"]["engine_sha256"] == "reviewed-new-engine"
    assert (
        audit["revised_authorization"]["expires_at"]
        == audit["previous_authorization"]["expires_at"]
    )


def test_unknown_outcome_cannot_be_inferred_from_a_flat_account(ledger, paper_config, account):
    identifier = rejected_attempt(ledger, paper_config, account, record_outcome=False)
    with pytest.raises(QuantError, match="original rejection evidence"):
        ledger.reconcile_rejection(paper_config, account, permission(), identifier)


def test_legacy_rejection_requires_fingerprinted_original_evidence(
    ledger, paper_config, account, tmp_path
):
    identifier = rejected_attempt(ledger, paper_config, account, record_outcome=False)
    private, public = tmp_path / "private.json", tmp_path / "public.json"
    original = {
        "exit_code": 2,
        "error": (
            f"orderRef='{identifier}_buy', account='{paper_config.account}', "
            "ValidationError, errorCode=321, permId=0, fills=[], API in read-only mode"
        ),
    }
    write_json(private, original)
    write_json(public, {"private_original_sha256": "wrong"})
    with pytest.raises(QuantError, match="fingerprint"):
        ledger.reconcile_rejection(
            paper_config,
            account,
            permission(),
            identifier,
            legacy_receipt=public,
            legacy_private=private,
        )
    write_json(public, {"private_original_sha256": digest_json(original)})
    result = ledger.reconcile_rejection(
        paper_config,
        account,
        permission(),
        identifier,
        legacy_receipt=public,
        legacy_private=private,
    )
    assert result["resolution"] == "broker_verified_rejected_no_fill"


def test_older_attempts_need_more_than_todays_execution_cache(
    ledger, paper_config, account, monkeypatch
):
    identifier = rejected_attempt(ledger, paper_config, account)
    monkeypatch.setattr(lab, "now_utc", lambda: NOW + pd.Timedelta(days=1))
    with pytest.raises(QuantError, match="broker statements"):
        ledger.reconcile_rejection(paper_config, account, permission(), identifier)


@pytest.mark.parametrize("granted", [True, False])
def test_permission_probe_is_what_if_only_and_restores_timeout(
    paper_config, account, monkeypatch, granted
):
    import asyncio

    from ib_async import Stock

    class Event:
        def __init__(self):
            self.handlers = []

        def __iadd__(self, callback):
            self.handlers.append(callback)
            return self

        def __isub__(self, callback):
            self.handlers.remove(callback)
            return self

    event = Event()
    requests = []

    def what_if(contract, order):
        requests.append((contract, order))
        assert order.account == paper_config.account and order.totalQuantity == 1
        if granted:
            return SimpleNamespace(status="PreSubmitted", warningText="")
        for callback in event.handlers:
            callback(1, 321, "API is read-only", contract)
        raise asyncio.TimeoutError()

    ib = SimpleNamespace(
        RequestTimeout=15,
        errorEvent=event,
        whatIfOrder=what_if,
        qualifyContracts=lambda *args: [Stock("SPY", "SMART", "USD", conId=756733)],
    )
    broker = lab.IBExperimentBroker(paper_config, authorized=False)
    broker.reader = SimpleNamespace(ib=ib)
    broker.account = lambda: account
    broker.reference = lambda symbol: reference()
    monkeypatch.setattr(lab, "now_utc", lambda: NOW)
    result = broker.order_permission()
    assert result["order_entry_available"] is granted
    assert result["actual_orders_sent"] == 0 and len(requests) == 1
    assert ib.RequestTimeout == 15 and not event.handlers
    if not granted:
        assert result["error_codes"] == [321] and result["error"] == "TimeoutError"
