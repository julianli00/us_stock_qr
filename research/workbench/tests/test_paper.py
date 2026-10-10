from __future__ import annotations

import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import us_quant.paper as paper
from us_quant.config import QuantError
from us_quant.paper import (
    IBPaperBroker,
    Outcome,
    PaperLedger,
    Quote,
    Snapshot,
    build_plan,
    execute_plan,
)
from us_quant.storage import read_json, write_json

NOW = pd.Timestamp("2026-10-01 13:31:00Z")


@pytest.fixture
def signal():
    return {
        "signal_date": "2026-09-30",
        "execution_session": "2026-10-01",
        "mode": "paper_only",
        "research_qualified": True,
        "executable": True,
        "block_reasons": [],
        "weights": {"SPY": 0.35, "IEF": 0.25},
    }


@pytest.fixture
def snapshot(paper_config):
    return Snapshot(paper_config.account, "USD", 10000.0, 10000.0, {})


def quotes(*symbols):
    return {symbol: Quote(symbol, 100.0, 100.05, NOW) for symbol in symbols}


def test_plan_uses_integer_bounded_orders_and_cash(paper_config, snapshot, signal):
    orders = build_plan(paper_config, snapshot, signal, quotes("SPY", "IEF"), NOW)
    assert len(orders) == 2
    assert all(type(order.quantity) is int and order.quantity > 0 for order in orders)
    assert all(order.quantity * order.limit_price <= 4000 for order in orders)
    assert all(order.action == "BUY" and order.limit_price >= 100.05 for order in orders)
    assert sum(order.quantity * order.limit_price + 1 for order in orders) < snapshot.settled_cash


def test_sells_are_planned_before_buys(paper_config, snapshot, signal):
    snapshot = replace(snapshot, positions={"QQQ": 10}, settled_cash=9000)
    orders = build_plan(paper_config, snapshot, signal, quotes("SPY", "IEF", "QQQ"), NOW)
    assert orders[0].symbol == "QQQ" and orders[0].action == "SELL"
    assert orders[0].quantity == 10
    assert all(order.action == "BUY" for order in orders[1:])


@pytest.mark.parametrize(
    "change",
    [
        {"account": "U1234567"},
        {"port": 7496},
        {"port": 4001},
        {"host": "8.8.8.8"},
        {"paper_acknowledged": False},
        {"max_gross_exposure": 1.1},
        {"max_position_weight": 0.9},
        {"allowed_symbols": ("TQQQ",)},
        {"max_daily_loss": float("nan")},
        {"client_id": 0},
    ],
)
def test_paper_config_rejects_unsafe_routes_and_limits(paper_config, change):
    with pytest.raises(QuantError):
        replace(paper_config, **change).validate()


@pytest.mark.parametrize("account", ["DU1234567", "DUQ1234567", "DUZ1234567"])
def test_paper_identifier_supports_lettered_account_series(paper_config, account):
    assert paper.is_paper_account_id(account)
    replace(paper_config, account=account).validate()


@pytest.mark.parametrize(
    "account",
    [
        "U1234567",
        "UQ1234567",
        "DUQ",
        "DUQ123",
        "DU_REPLACE_WITH_YOUR_PAPER_ACCOUNT",
        "DUQ1234567\n",
        "duq1234567",
        None,
        1234567,
    ],
)
def test_paper_identifier_still_rejects_live_or_malformed_values(account):
    assert not paper.is_paper_account_id(account)


@pytest.mark.parametrize(
    "change",
    [
        {"market_data_type": 3},
        {"market_data_type": 2},
        {"bid": float("nan")},
        {"bid": 101.0},
        {"ask": 103.0},
        {"timestamp": NOW - pd.Timedelta(minutes=1)},
        {"timestamp": NOW + pd.Timedelta(seconds=10)},
        {"timestamp": NOW.tz_localize(None)},
    ],
)
def test_only_fresh_realtime_uncrossed_quotes_allowed(paper_config, change):
    with pytest.raises(QuantError):
        replace(quotes("SPY")["SPY"], **change).validate(paper_config, NOW)


@pytest.mark.parametrize(
    "change",
    [
        {"account": "U1234567"},
        {"currency": "EUR"},
        {"settled_cash": -1},
        {"positions": {"AAPL": 10}},
        {"positions": {"SPY": -1}},
        {"positions": {"SPY": 1.5}},
        {"open_order_refs": ("manual_order",)},
        {"net_liquidation": float("nan")},
    ],
)
def test_bad_account_state_fails_closed(paper_config, snapshot, change):
    with pytest.raises(QuantError):
        replace(snapshot, **change).validate(paper_config)


def test_missing_settled_cash_is_readable_but_never_spendable(paper_config, snapshot, signal):
    unknown = replace(snapshot, settled_cash=None)
    unknown.validate(paper_config, require_settled_cash=False)
    with pytest.raises(QuantError, match="funding cannot be assumed"):
        build_plan(paper_config, unknown, signal, quotes("SPY", "IEF"), NOW)
    ledger = PaperLedger(paper_config)
    try:
        with pytest.raises(QuantError, match="funding cannot be assumed"):
            ledger.observe(unknown, NOW)
        assert ledger.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "change",
    [
        {"research_qualified": False},
        {"executable": False},
        {"mode": "live"},
        {"weights": {"SPY": 0.8}},
        {"weights": {"SPY": -0.1}},
        {"weights": {"SPY": float("nan")}},
        {"weights": {"SPY": "0.1"}},
        {"weights": {"TQQQ": 0.1}},
        {"weights": {}},
        {"weights": {"SPY": 0.4, "QQQ": 0.4, "IEF": 0.3}},
        {"signal_date": "2026-09-29"},
        {"execution_session": "2026-10-02"},
    ],
)
def test_bad_signal_never_produces_a_plan(paper_config, snapshot, signal, change):
    with pytest.raises(QuantError):
        build_plan(paper_config, snapshot, {**signal, **change}, quotes("SPY", "IEF"), NOW)


def test_settled_cash_cannot_be_replaced_by_expected_sell_proceeds(paper_config, snapshot, signal):
    snapshot = replace(snapshot, settled_cash=50, positions={"QQQ": 30})
    with pytest.raises(QuantError, match="settled"):
        build_plan(paper_config, snapshot, signal, quotes("SPY", "IEF", "QQQ"), NOW)


def test_kill_switch_market_window_and_broker_duplicate(paper_config, snapshot, signal):
    with pytest.raises(QuantError, match="window"):
        build_plan(
            paper_config, snapshot, signal, quotes("SPY", "IEF"), NOW + pd.Timedelta(hours=1)
        )
    duplicate = replace(snapshot, executed_order_refs=("uq_20261001_prior_hash_SPY",))
    with pytest.raises(QuantError, match="never replay"):
        build_plan(paper_config, duplicate, signal, quotes("SPY", "IEF"), NOW)
    Path(paper_config.kill_switch_file).touch()
    with pytest.raises(QuantError, match="Kill switch"):
        build_plan(paper_config, snapshot, signal, quotes("SPY", "IEF"), NOW)


def test_persistent_daily_halt_does_not_reset_on_restart(paper_config, snapshot):
    ledger = PaperLedger(paper_config)
    try:
        ledger.observe(snapshot, NOW)
        with pytest.raises(QuantError, match="loss limit"):
            ledger.observe(replace(snapshot, net_liquidation=9700), NOW + pd.Timedelta(minutes=1))
    finally:
        ledger.close()
    restarted = PaperLedger(paper_config)
    try:
        with pytest.raises(QuantError, match="reconciliation"):
            restarted.observe(snapshot, NOW + pd.Timedelta(days=1))
    finally:
        restarted.close()


def test_overnight_loss_is_not_erased_at_new_day(paper_config, snapshot):
    ledger = PaperLedger(paper_config)
    try:
        ledger.observe(snapshot, NOW)
        with pytest.raises(QuantError, match="loss limit"):
            ledger.observe(replace(snapshot, net_liquidation=9700), NOW + pd.Timedelta(days=1))
    finally:
        ledger.close()


def test_high_water_drawdown_and_initial_capital_are_enforced(paper_config, snapshot):
    ledger = PaperLedger(paper_config)
    try:
        with pytest.raises(QuantError, match="initial equity"):
            ledger.observe(replace(snapshot, net_liquidation=1000000), NOW)
        ledger.observe(snapshot, NOW)
        with pytest.raises(QuantError, match="drawdown"):
            ledger.observe(replace(snapshot, net_liquidation=8900), NOW + pd.Timedelta(minutes=1))
    finally:
        ledger.close()


def test_signal_reservation_survives_restarts(paper_config, snapshot, signal):
    ledger = PaperLedger(paper_config)
    try:
        ledger.observe(snapshot, NOW)
        run_id = ledger.reserve(signal, [])
        ledger.run_state(run_id, "completed")
    finally:
        ledger.close()
    restarted = PaperLedger(paper_config)
    try:
        with pytest.raises(QuantError, match="already reserved"):
            restarted.reserve(signal, [])
    finally:
        restarted.close()


def test_allow_submit_toggle_does_not_erase_risk_state(paper_config, snapshot):
    ledger = PaperLedger(paper_config)
    try:
        ledger.observe(snapshot, NOW)
    finally:
        ledger.close()
    enabled = PaperLedger(replace(paper_config, allow_submit=True))
    try:
        enabled.observe(snapshot, NOW)
    finally:
        enabled.close()


def test_clock_rollback_is_rejected(paper_config, snapshot):
    ledger = PaperLedger(paper_config)
    try:
        ledger.observe(snapshot, NOW)
        with pytest.raises(QuantError, match="backwards"):
            ledger.observe(snapshot, NOW - pd.Timedelta(minutes=5))
    finally:
        ledger.close()


class FakeBroker:
    def __init__(self, snapshot, status="Filled", kill_switch=None):
        self.current = snapshot
        self.status = status
        self.orders = []
        self.kill_switch = kill_switch

    def snapshot(self):
        return self.current

    def quotes(self, symbols):
        return quotes(*symbols)

    def submit(self, order):
        self.orders.append(order)
        if self.status == "timeout":
            raise TimeoutError("connection lost after submission")
        quantity = order.quantity if self.status == "Filled" else order.quantity // 2
        positions = self.current.positions.copy()
        sign = 1 if order.action == "BUY" else -1
        positions[order.symbol] = positions.get(order.symbol, 0) + sign * quantity
        positions = {key: value for key, value in positions.items() if value}
        self.current = replace(
            self.current,
            positions=positions,
            settled_cash=self.current.settled_cash - sign * quantity * order.limit_price - 1,
            executed_order_refs=(*self.current.executed_order_refs, order.order_ref),
        )
        if self.kill_switch:
            Path(self.kill_switch).touch()
        return Outcome(self.status, quantity, order.limit_price, len(self.orders), 100, 1.0)


def test_dry_run_never_calls_submission(paper_config, snapshot, signal, monkeypatch):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    broker = FakeBroker(snapshot)
    ledger = PaperLedger(paper_config)
    try:
        result = execute_plan(paper_config, broker, ledger, signal)
        assert result["mode"] == "dry_run" and result["orders_sent"] == 0
        assert not broker.orders
    finally:
        ledger.close()


def test_fresh_prices_cannot_break_position_caps(paper_config, snapshot, signal, monkeypatch):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    config = replace(paper_config, allow_submit=True)

    class MovingPriceBroker(FakeBroker):
        def __init__(self, account):
            super().__init__(account)
            self.quote_requests = 0

        def quotes(self, symbols):
            self.quote_requests += 1
            if self.quote_requests == 1:
                return quotes(*symbols)
            return {symbol: Quote(symbol, 300.0, 300.05, NOW) for symbol in symbols}

    broker = MovingPriceBroker(snapshot)
    ledger = PaperLedger(config)
    try:
        with pytest.raises(QuantError, match="invalidate"):
            execute_plan(config, broker, ledger, signal, submit=True, confirmation=config.account)
        assert not broker.orders
    finally:
        ledger.close()


def test_invalid_filled_price_is_not_reported_as_success(
    paper_config, snapshot, signal, monkeypatch
):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    config = replace(paper_config, allow_submit=True)

    class BadFillBroker(FakeBroker):
        def submit(self, order):
            result = super().submit(order)
            return replace(result, average_price=order.limit_price + 10)

    broker = BadFillBroker(snapshot)
    ledger = PaperLedger(config)
    try:
        with pytest.raises(QuantError, match="uncertain"):
            execute_plan(config, broker, ledger, signal, submit=True, confirmation=config.account)
        assert len(broker.orders) == 1
        status = ledger.connection.execute("SELECT status FROM runs").fetchone()[0]
        assert status == "needs_reconciliation"
    finally:
        ledger.close()


def test_successful_paper_run_reconciles_positions(paper_config, snapshot, signal, monkeypatch):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    config = replace(paper_config, allow_submit=True)
    broker = FakeBroker(snapshot)
    ledger = PaperLedger(config)
    try:
        result = execute_plan(
            config, broker, ledger, signal, submit=True, confirmation=config.account
        )
        assert result["orders_sent"] == 2
        assert set(broker.current.positions) == {"SPY", "IEF"}
        assert ledger.connection.execute("SELECT status FROM runs").fetchone()[0] == "completed"
        with pytest.raises(QuantError, match="never replay"):
            execute_plan(config, broker, ledger, signal, submit=True, confirmation=config.account)
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "allow,confirmation", [(False, "DU1234567"), (True, "U1234567"), (True, None)]
)
def test_both_opt_in_and_account_confirmation_required(
    paper_config, snapshot, signal, monkeypatch, allow, confirmation
):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    config = replace(paper_config, allow_submit=allow)
    broker = FakeBroker(snapshot)
    ledger = PaperLedger(config)
    try:
        with pytest.raises(QuantError, match="requires"):
            execute_plan(config, broker, ledger, signal, submit=True, confirmation=confirmation)
        assert not broker.orders
    finally:
        ledger.close()


@pytest.mark.parametrize("outcome", ["Cancelled", "timeout", "kill_after_first"])
def test_partial_uncertain_or_interrupted_run_never_continues(
    paper_config, snapshot, signal, monkeypatch, outcome
):
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    config = replace(paper_config, allow_submit=True)
    broker = FakeBroker(
        snapshot,
        status=outcome if outcome != "kill_after_first" else "Filled",
        kill_switch=config.kill_switch_file if outcome == "kill_after_first" else None,
    )
    ledger = PaperLedger(config)
    try:
        with pytest.raises((QuantError, TimeoutError)):
            execute_plan(config, broker, ledger, signal, submit=True, confirmation=config.account)
        assert len(broker.orders) == 1
        assert (
            ledger.connection.execute("SELECT status FROM runs").fetchone()[0]
            == "needs_reconciliation"
        )
        with pytest.raises(QuantError, match="unfinished"):
            ledger.reserve({**signal, "signal_date": "2026-10-30"}, [])
    finally:
        ledger.close()


def install_handshake_fake(monkeypatch, accounts, server_time=NOW, fail_updates=False):
    import ib_async

    class FakeConnection:
        instances = []

        def __init__(self):
            self.connected = False
            self.disconnected = False
            self.calls = []
            self.client = SimpleNamespace(connect=self.low_level_connect)
            self.wrapper = SimpleNamespace(clientId=None)
            self.instances.append(self)

        def connect(self, *args, **kwargs):
            pytest.fail("High-level SDK connect must not fetch positions before identity checks.")

        def low_level_connect(self, *args, **kwargs):
            self.calls.append("handshake")
            self.connected = True

        def managedAccounts(self):
            self.calls.append("identity")
            return accounts

        def reqCurrentTime(self):
            self.calls.append("clock")
            return server_time.to_pydatetime()

        def reqAccountUpdates(self, account):
            self.calls.append("account_updates")
            assert accounts == [account]
            if fail_updates:
                raise TimeoutError("subscription timed out")

        def isConnected(self):
            return self.connected

        def disconnect(self):
            self.calls.append("disconnect")
            self.connected = False
            self.disconnected = True

    monkeypatch.setattr(ib_async, "IB", FakeConnection)
    monkeypatch.setattr(paper, "now_utc", lambda: NOW)
    return FakeConnection


@pytest.mark.parametrize("problem", ["live", "mixed", "clock", "different_paper", "no_accounts"])
def test_ib_connection_mismatch_disconnects_immediately(paper_config, monkeypatch, problem):
    accounts = {
        "live": ["U1234567"],
        "mixed": [paper_config.account, "U1234567"],
        "clock": [paper_config.account],
        "different_paper": ["DU7654321"],
        "no_accounts": [],
    }[problem]
    server_time = NOW - pd.Timedelta(minutes=5) if problem == "clock" else NOW
    install_handshake_fake(monkeypatch, accounts, server_time)
    broker = IBPaperBroker(paper_config)
    with pytest.raises(QuantError):
        broker.__enter__()
    assert broker.ib.disconnected and not broker.ib.connected
    assert "account_updates" not in broker.ib.calls


def test_account_subscription_only_happens_after_identity_and_clock(paper_config, monkeypatch):
    install_handshake_fake(monkeypatch, [paper_config.account])
    broker = IBPaperBroker(paper_config)
    with broker:
        assert broker.ib.calls == ["handshake", "identity", "clock", "account_updates"]
        assert broker.ib.wrapper.clientId == paper_config.client_id
    assert broker.ib.disconnected


def test_discovery_does_not_request_positions_or_balances(paper_config, monkeypatch):
    fake = install_handshake_fake(monkeypatch, [paper_config.account])
    result = paper.discover_paper_identity(4002)
    connection = fake.instances[-1]
    assert connection.calls == ["handshake", "identity", "clock", "disconnect"]
    assert result["identity_verified"] is True
    assert result["account_data_requested"] is False and result["orders_sent"] == 0
    assert result["account"] != paper_config.account


def test_lettered_paper_series_passes_the_full_identity_flow(paper_config, monkeypatch):
    config = replace(paper_config, account="DUQ1234567")
    fake = install_handshake_fake(monkeypatch, [config.account])
    result = paper.discover_paper_identity(4002)
    assert result["identity_verified"] and not result["account_data_requested"]
    assert fake.instances[-1].calls == ["handshake", "identity", "clock", "disconnect"]
    with IBPaperBroker(config) as broker:
        assert broker.ib.calls == ["handshake", "identity", "clock", "account_updates"]


def test_discovery_rejects_live_identity_without_requesting_account_data(monkeypatch):
    fake = install_handshake_fake(monkeypatch, ["U1234567"])
    with pytest.raises(QuantError, match="DU paper"):
        paper.discover_paper_identity(4002)
    assert fake.instances[-1].calls == ["handshake", "identity", "disconnect"]


def test_discovery_rejects_live_port_before_creating_a_client(monkeypatch):
    fake = install_handshake_fake(monkeypatch, ["DU1234567"])
    with pytest.raises(QuantError, match="loopback"):
        paper.discover_paper_identity(4001)
    assert not fake.instances


def test_verified_subscription_failure_disconnects_and_surfaces_error(paper_config, monkeypatch):
    install_handshake_fake(monkeypatch, [paper_config.account], fail_updates=True)
    broker = IBPaperBroker(paper_config)
    with pytest.raises(QuantError, match="subscription timed out"):
        broker.__enter__()
    assert broker.ib.disconnected and not broker.ib.connected


def test_readonly_snapshot_does_not_replace_settled_cash_with_buying_power(
    paper_config, monkeypatch
):
    install_handshake_fake(monkeypatch, [paper_config.account])
    with IBPaperBroker(paper_config, readonly=True) as broker:
        broker.ib.accountSummary = lambda account: [
            SimpleNamespace(tag="NetLiquidation", currency="USD", value="10000"),
            SimpleNamespace(tag="TotalCashValue", currency="USD", value="9000"),
            SimpleNamespace(tag="BuyingPower", currency="USD", value="40000"),
        ]
        broker.ib.accountValues = lambda account: []
        broker.ib.reqPositions = lambda: []
        broker.ib.reqAllOpenOrders = lambda: []
        broker.ib.reqExecutions = lambda query: []
        snapshot = broker.snapshot()
        assert snapshot.net_liquidation == 10000
        assert snapshot.settled_cash is None
        broker.readonly = False
        with pytest.raises(QuantError, match="funding cannot be assumed"):
            broker.snapshot()


def test_paper_binding_is_private_metadata_only_and_submission_disabled(
    paper_config, tmp_path, monkeypatch
):
    from dataclasses import asdict

    account = "DUQ1234567"
    fake = install_handshake_fake(monkeypatch, [account])
    template = tmp_path / "template.json"
    write_json(template, {**asdict(paper_config), "allow_submit": True})
    target = tmp_path / "private-paper.json"
    result = paper.save_readonly_paper_config(target, template, 4002)
    saved = read_json(target)
    assert saved["account"] == account and saved["port"] == 4002
    assert saved["paper_acknowledged"] is True and saved["allow_submit"] is False
    assert saved["initial_equity_usd"] == paper_config.initial_equity_usd
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert result["account"] != account and result["account_data_requested"] is False
    assert fake.instances[-1].calls == ["handshake", "identity", "clock", "disconnect"]
    before = target.read_bytes()
    with pytest.raises(QuantError, match="overwrite"):
        paper.save_readonly_paper_config(target, template, 4002)
    assert target.read_bytes() == before


def test_paper_binding_rejects_a_live_identity_before_writing(tmp_path, monkeypatch):
    fake = install_handshake_fake(monkeypatch, ["UQ1234567"])
    target = tmp_path / "paper.json"
    with pytest.raises(QuantError, match="DU paper"):
        paper.save_readonly_paper_config(target, tmp_path / "unused-template.json", 4002)
    assert not target.exists()
    assert fake.instances[-1].disconnected


@pytest.mark.parametrize("code", [10197, 10089])
def test_market_data_request_error_is_explicit_and_never_uses_delayed_fallback(
    paper_config, monkeypatch, code
):
    from ib_async import Stock
    from ib_async.wrapper import RequestError

    install_handshake_fake(monkeypatch, [paper_config.account])
    with IBPaperBroker(paper_config, readonly=True) as broker:
        assert broker.ib.RaiseRequestErrors is True
        broker.contracts["SPY"] = Stock("SPY", "SMART", "USD", conId=756733)
        requested_types = []
        broker.ib.reqMarketDataType = requested_types.append

        def denied_quotes(*contracts):
            raise RequestError(3, code, "Market data is unavailable")

        broker.ib.reqTickers = denied_quotes
        with pytest.raises(QuantError, match=rf"market-data request failed \({code}\)"):
            broker.quotes({"SPY"})
        assert requested_types == [1]
