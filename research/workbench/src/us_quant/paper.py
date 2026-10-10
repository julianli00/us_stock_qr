from __future__ import annotations

import asyncio
import json
import math
import os
import re
import socket
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from us_quant.calendar import require_execution_window
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json, utc_now

if TYPE_CHECKING:
    from ib_async import IB, Contract

APPROVED_ETFS = frozenset({"SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "SHY"})


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def is_paper_account_id(account: object) -> bool:
    return isinstance(account, str) and re.fullmatch(r"DU[A-Z]?[0-9]{4,}", account) is not None


def validate_paper_route(host: str, port: int, client_id: int) -> None:
    if type(port) is not int or type(client_id) is not int:
        raise QuantError("Paper port and client ID must be integers.")
    if host not in {"127.0.0.1", "localhost"} or port not in {7497, 4002}:
        raise QuantError("Only loopback TWS paper 7497 / Gateway paper 4002 are allowed.")
    if not 1 <= client_id <= 1023:
        raise QuantError("Use a dedicated nonzero client ID in [1, 1023].")


@dataclass(frozen=True)
class PaperConfig:
    host: str
    port: int
    client_id: int
    account: str
    paper_acknowledged: bool
    allow_submit: bool
    initial_equity_usd: float
    max_gross_exposure: float
    max_position_weight: float
    max_order_notional: float
    max_daily_loss: float
    max_drawdown: float
    max_spread_bps: float
    max_quote_age_seconds: int
    limit_buffer_bps: float
    minimum_trade_notional: float
    commission_reserve_per_order: float
    execution_window_minutes: int
    allowed_symbols: tuple[str, ...]
    state_file: str
    kill_switch_file: str

    def validate(self) -> None:
        validate_paper_route(self.host, self.port, self.client_id)
        if not is_paper_account_id(self.account):
            raise QuantError("Set an explicit DU paper account; live U/UQ accounts are forbidden.")
        if self.paper_acknowledged is not True or type(self.allow_submit) is not bool:
            raise QuantError(
                "Explicit paper-account acknowledgement and Boolean allow_submit required."
            )
        if (
            not self.allowed_symbols
            or len(set(self.allowed_symbols)) != len(self.allowed_symbols)
            or not set(self.allowed_symbols) <= APPROVED_ETFS
        ):
            raise QuantError(
                "Paper execution is restricted to the researched unleveraged ETF allowlist."
            )
        limits = (
            self.initial_equity_usd,
            self.max_order_notional,
            self.max_daily_loss,
            self.max_drawdown,
            self.max_spread_bps,
            self.max_quote_age_seconds,
            self.minimum_trade_notional,
            self.commission_reserve_per_order,
        )
        if not all(math.isfinite(x) and x > 0 for x in limits):
            raise QuantError("Paper risk and sizing limits must be finite and positive.")
        if not 0 < self.max_gross_exposure <= 0.98 or not 0 < self.max_position_weight <= 0.50:
            raise QuantError("Paper limits require 2% cash and at most 50% in one ETF.")
        if not 0 < self.max_daily_loss < self.max_drawdown <= 0.15:
            raise QuantError("Require daily loss < drawdown limit <= 15%.")
        if (
            not 0 <= self.limit_buffer_bps <= 20
            or self.max_spread_bps > 50
            or self.max_quote_age_seconds > 30
            or not 1 <= self.execution_window_minutes <= 60
        ):
            raise QuantError(
                "Quote, spread, price, or execution-window settings are too permissive."
            )
        if not self.state_file or not self.kill_switch_file:
            raise QuantError("Persistent state and a kill-switch path are required.")


def load_paper_config(path: Path) -> PaperConfig:
    try:
        raw = read_json(path)
        raw["allowed_symbols"] = tuple(raw["allowed_symbols"])
        config = PaperConfig(**raw)
        config.validate()
        return config
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantError(f"Invalid paper configuration: {exc}") from exc


def probe_paper_ports() -> dict:
    ports = {}
    for port in (7497, 4002):
        with socket.socket() as connection:
            connection.settimeout(0.5)
            ports[str(port)] = connection.connect_ex(("127.0.0.1", port)) == 0
    return {
        "paper_ports_listening": ports,
        "account_verified": False,
        "orders_sent": 0,
        "note": "A listening port alone is not proof of a paper account.",
    }


def verified_paper_handshake(
    ib: IB, host: str, port: int, client_id: int, expected_account: str | None = None
) -> str:
    validate_paper_route(host, port, client_id)
    if expected_account is not None and not is_paper_account_id(expected_account):
        raise QuantError("The expected account must be an explicit DU paper account.")
    ib.wrapper.clientId = client_id
    # IB.connect() in ib_async 2.1.0 fetches positions even with fetchFields=0.
    ib.client.connect(host, port, clientId=client_id, timeout=10)
    accounts = ib.managedAccounts()
    if (
        len(accounts) != 1
        or not is_paper_account_id(accounts[0])
        or (expected_account is not None and accounts[0] != expected_account)
    ):
        raise QuantError("Require one exact DU paper account and no live/mixed account session.")
    server_time = pd.Timestamp(ib.reqCurrentTime())
    if server_time.tzinfo is None or abs((now_utc() - server_time).total_seconds()) > 5:
        raise QuantError("IBKR/local clock skew exceeds five seconds or the timezone is absent.")
    if not ib.isConnected():
        raise QuantError("Paper API disconnected during identity verification.")
    return accounts[0]


def discover_paper_identity(port: int, client_id: int = 39) -> dict:
    validate_paper_route("127.0.0.1", port, client_id)
    from ib_async import IB

    ib = IB()
    ib.RequestTimeout = 15
    try:
        account = verified_paper_handshake(ib, "127.0.0.1", port, client_id)
        return {
            "port": port,
            "account": "DU***" + account[-4:],
            "identity_verified": True,
            "account_data_requested": False,
            "market_data_requested": False,
            "orders_sent": 0,
        }
    except (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError) as exc:
        raise QuantError(f"Paper identity handshake failed: {exc}") from exc
    finally:
        ib.disconnect()


def save_readonly_paper_config(path: Path, template: Path, port: int) -> dict:
    validate_paper_route("127.0.0.1", port, 39)
    if path.exists() or path.is_symlink():
        raise QuantError("Refusing to overwrite an existing local paper configuration.")
    from ib_async import IB

    ib = IB()
    ib.RequestTimeout = 15
    try:
        account = verified_paper_handshake(ib, "127.0.0.1", port, 39)
        raw = read_json(template)
        raw.update(
            {
                "host": "127.0.0.1",
                "port": port,
                "account": account,
                "paper_acknowledged": True,
                "allow_submit": False,
                "allowed_symbols": tuple(raw["allowed_symbols"]),
            }
        )
        config = PaperConfig(**raw)
        config.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise QuantError(
                "Paper configuration appeared concurrently; refusing overwrite."
            ) from exc
        with os.fdopen(descriptor, "w") as stream:
            json.dump(asdict(config), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "saved_config": str(path),
            "account": "DU***" + account[-4:],
            "paper_identity_verified": True,
            "submission_enabled": False,
            "account_data_requested": False,
            "orders_sent": 0,
        }
    except (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError) as exc:
        raise QuantError(f"Read-only paper binding failed: {exc}") from exc
    finally:
        ib.disconnect()


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    timestamp: pd.Timestamp
    market_data_type: int = 1

    def validate(self, config: PaperConfig, now: pd.Timestamp) -> None:
        if (
            self.symbol not in config.allowed_symbols
            or not all(math.isfinite(value) and value > 0 for value in (self.bid, self.ask))
            or self.bid > self.ask
        ):
            raise QuantError(f"Invalid or crossed quote for {self.symbol}.")
        if self.market_data_type != 1:
            raise QuantError("Delayed/frozen market data may not be used to submit an order.")
        if self.timestamp.tzinfo is None or now.tzinfo is None:
            raise QuantError("Quote and execution timestamps must be timezone-aware.")
        age = (now - self.timestamp).total_seconds()
        if not -2 <= age <= config.max_quote_age_seconds:
            raise QuantError(f"Stale or future-dated quote for {self.symbol}.")
        spread = (self.ask - self.bid) / ((self.ask + self.bid) / 2.0) * 10000.0
        if spread > config.max_spread_bps:
            raise QuantError(f"{self.symbol} spread exceeds the paper risk limit.")


@dataclass(frozen=True)
class Snapshot:
    account: str
    currency: str
    net_liquidation: float
    settled_cash: float | None
    positions: dict[str, int]
    open_order_refs: tuple[str, ...] = ()
    executed_order_refs: tuple[str, ...] = ()

    def required_settled_cash(self) -> float:
        if self.settled_cash is None:
            raise QuantError("IBKR did not report settled USD cash; funding cannot be assumed.")
        if not math.isfinite(self.settled_cash) or self.settled_cash < 0:
            raise QuantError("Invalid or negative settled cash.")
        return self.settled_cash

    def validate(self, config: PaperConfig, *, require_settled_cash: bool = True) -> None:
        if self.account != config.account or self.currency != "USD":
            raise QuantError("Account identity or USD base-currency check failed.")
        if not math.isfinite(self.net_liquidation) or self.net_liquidation <= 0:
            raise QuantError("Invalid account value.")
        if require_settled_cash or self.settled_cash is not None:
            self.required_settled_cash()
        if self.open_order_refs:
            raise QuantError(
                "Existing open orders require reconciliation before any new submission."
            )
        for symbol, quantity in self.positions.items():
            if symbol not in config.allowed_symbols or type(quantity) is not int or quantity < 0:
                raise QuantError(
                    "Unknown, fractional, or short positions require manual reconciliation."
                )


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    action: str
    quantity: int
    limit_price: float
    order_ref: str


@dataclass(frozen=True)
class Outcome:
    status: str
    filled: float
    average_price: float
    broker_order_id: int
    permanent_id: int
    commission: float | None = None
    error_codes: tuple[int, ...] = ()


def submit_paper_ioc(
    ib: IB, config: PaperConfig, contract: Contract, order: PlannedOrder
) -> Outcome:
    from ib_async import LimitOrder

    config.validate()
    if (
        not config.allow_submit
        or not ib.isConnected()
        or ib.managedAccounts() != [config.account]
        or ib.client.host != config.host
        or ib.client.port != config.port
    ):
        raise QuantError("The explicitly authorized paper transport identity does not match.")
    if (
        order.symbol not in config.allowed_symbols
        or contract.symbol != order.symbol
        or contract.secType != "STK"
        or contract.currency != "USD"
        or contract.conId <= 0
        or order.action not in {"BUY", "SELL"}
        or type(order.quantity) is not int
        or order.quantity <= 0
        or not math.isfinite(order.limit_price)
        or order.limit_price <= 0
        or order.quantity * order.limit_price > config.max_order_notional
        or not order.order_ref
    ):
        raise QuantError("Invalid or over-limit paper order instruction.")
    instruction = LimitOrder(
        order.action,
        order.quantity,
        order.limit_price,
        account=config.account,
        orderRef=order.order_ref,
        tif="IOC",
        outsideRth=False,
        transmit=True,
    )
    trade = ib.placeOrder(contract, instruction)

    def validation_rejected() -> bool:
        status = trade.orderStatus
        return (
            status.status == "ValidationError"
            and status.filled == 0
            and status.permId == 0
            and not trade.fills
            and any(entry.errorCode == 321 for entry in getattr(trade, "log", ()))
        )

    deadline = time.monotonic() + 20
    while (
        ib.isConnected()
        and not trade.isDone()
        and not validation_rejected()
        and time.monotonic() < deadline
    ):
        ib.waitOnUpdate(timeout=1)
    if not trade.isDone() and not validation_rejected():
        if ib.isConnected():
            ib.cancelOrder(trade.order)
            deadline = time.monotonic() + 5
            while not trade.isDone() and time.monotonic() < deadline:
                ib.waitOnUpdate(timeout=1)
        if not trade.isDone():
            raise QuantError("Order outcome is uncertain; broker reconciliation is mandatory.")
    status = trade.orderStatus
    deadline = time.monotonic() + 2
    while (
        status.status == "Filled"
        and time.monotonic() < deadline
        and any(not fill.commissionReport.execId for fill in trade.fills)
    ):
        ib.waitOnUpdate(timeout=0.25)
    commission_values = [fill.commissionReport.commission for fill in trade.fills]
    commission = (
        float(sum(commission_values))
        if commission_values
        and all(
            fill.commissionReport.execId and fill.commissionReport.currency == "USD"
            for fill in trade.fills
        )
        and all(math.isfinite(value) and 0 <= value < 1e6 for value in commission_values)
        else None
    )
    return Outcome(
        status.status,
        status.filled,
        status.avgFillPrice,
        status.orderId,
        status.permId,
        commission,
        tuple(entry.errorCode for entry in getattr(trade, "log", ()) if entry.errorCode),
    )


def validated_weights(config: PaperConfig, signal: dict) -> dict[str, float]:
    weights = signal.get("weights")
    if (
        not isinstance(weights, dict)
        or not weights
        or not set(weights) <= set(config.allowed_symbols)
    ):
        raise QuantError("Signal contains an empty or unauthorized universe.")
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= config.max_position_weight
            for value in weights.values()
        )
        or sum(weights.values()) > config.max_gross_exposure + 1e-10
    ):
        raise QuantError("Signal exceeds a long-only, concentration, or gross-exposure limit.")
    return {key: float(value) for key, value in weights.items()}


def build_plan(
    config: PaperConfig,
    snapshot: Snapshot,
    signal: dict,
    quotes: dict[str, Quote],
    now: pd.Timestamp,
) -> list[PlannedOrder]:
    config.validate()
    snapshot.validate(config)
    settled_cash = snapshot.required_settled_cash()
    if Path(config.kill_switch_file).exists():
        raise QuantError("Kill switch exists; paper trading is halted.")
    if (
        signal.get("mode") != "paper_only"
        or signal.get("research_qualified") is not True
        or signal.get("executable") is not True
        or signal.get("block_reasons")
    ):
        raise QuantError("Signal is not qualified for execution; shadow research only.")
    day = require_execution_window(signal["signal_date"], now, config.execution_window_minutes)
    if signal.get("execution_session") != day.date().isoformat():
        raise QuantError("Signal execution date is inconsistent.")
    weights = validated_weights(config, signal)
    active = set(snapshot.positions) | {key for key, value in weights.items() if value > 0}
    for symbol in active:
        if symbol not in quotes or quotes[symbol].symbol != symbol:
            raise QuantError(f"A live quote is required for {symbol}.")
        quotes[symbol].validate(config, now)
    budget = snapshot.net_liquidation - 2 * len(active) * config.commission_reserve_per_order
    if budget <= 0:
        raise QuantError("Portfolio is too small for the configured commission reserve.")
    identity = digest_json(
        {"date": signal["signal_date"], "signal": signal, "account": config.account}
    )
    proposed = []
    projected = snapshot.positions.copy()
    cash_required = 0.0
    for symbol in sorted(active):
        quote = quotes[symbol]
        buy_limit = math.ceil(quote.ask * (1 + config.limit_buffer_bps / 10000) * 100) / 100
        target = math.floor(budget * weights.get(symbol, 0.0) / buy_limit)
        existing = snapshot.positions.get(symbol, 0)
        delta = target - existing
        if delta == 0:
            continue
        action = "BUY" if delta > 0 else "SELL"
        price = (
            buy_limit
            if delta > 0
            else math.floor(quote.bid * (1 - config.limit_buffer_bps / 10000) * 100) / 100
        )
        quantity = abs(delta)
        notional = quantity * price
        if notional < config.minimum_trade_notional:
            continue
        if notional > config.max_order_notional:
            raise QuantError(f"{symbol} order exceeds the absolute notional limit.")
        if action == "SELL" and quantity > existing:
            raise QuantError("An order would create a short position.")
        if action == "BUY":
            cash_required += notional + config.commission_reserve_per_order
        projected[symbol] = target
        reference = f"uq_{day.strftime('%Y%m%d')}_{identity[:12]}_{symbol}"
        proposed.append(PlannedOrder(symbol, action, quantity, price, reference))
    # Do not fund new buys with unconfirmed or unsettled sale proceeds.
    sell_commissions = sum(
        config.commission_reserve_per_order for order in proposed if order.action == "SELL"
    )
    if cash_required + sell_commissions > settled_cash:
        raise QuantError(
            "Insufficient settled USD cash; no implicit margin or settlement assumption."
        )
    gross = 0.0
    for symbol, quantity in projected.items():
        value = quantity * (quotes[symbol].bid + quotes[symbol].ask) / 2.0
        if value > snapshot.net_liquidation * config.max_position_weight + 1e-6:
            raise QuantError("Projected position remains above the concentration limit.")
        gross += value
    if gross > snapshot.net_liquidation * config.max_gross_exposure + 1e-6:
        raise QuantError("Projected portfolio exceeds the gross exposure limit.")
    if any(ref.startswith(f"uq_{day.strftime('%Y%m%d')}_") for ref in snapshot.executed_order_refs):
        raise QuantError(
            "Broker already has an execution for this rebalance date; never replay it."
        )
    return sorted(proposed, key=lambda order: (order.action != "SELL", order.symbol))


class PaperLedger:
    def __init__(self, config: PaperConfig):
        self.config = config
        path = Path(config.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS account_state (
              account TEXT PRIMARY KEY, config_hash TEXT NOT NULL,
              high_water REAL NOT NULL, session_date TEXT NOT NULL,
              day_start_nlv REAL NOT NULL, last_nlv REAL NOT NULL, halt_reason TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, account TEXT NOT NULL, signal_date TEXT NOT NULL,
              status TEXT NOT NULL, plan_json TEXT NOT NULL, updated_at TEXT NOT NULL,
              UNIQUE(account, signal_date)
            );
            CREATE TABLE IF NOT EXISTS orders (
              order_ref TEXT PRIMARY KEY, run_id TEXT NOT NULL, status TEXT NOT NULL,
              outcome_json TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
              observed_at TEXT PRIMARY KEY, account TEXT NOT NULL,
              net_liquidation REAL NOT NULL, settled_cash REAL NOT NULL,
              positions_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def observe(self, snapshot: Snapshot, now: pd.Timestamp) -> None:
        snapshot.validate(self.config)
        settled_cash = snapshot.required_settled_cash()
        day = now.tz_convert("America/New_York").date().isoformat()
        risk_settings = asdict(self.config)
        risk_settings.pop("allow_submit")
        fingerprint = digest_json(risk_settings)
        connection = self.connection
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT config_hash, high_water, session_date, day_start_nlv, "
                "last_nlv, halt_reason, updated_at "
                "FROM account_state WHERE account=?",
                (snapshot.account,),
            ).fetchone()
            if row is None:
                if abs(snapshot.net_liquidation / self.config.initial_equity_usd - 1.0) > 0.05:
                    raise QuantError(
                        "First observation must match configured initial equity within 5%; "
                        "use a dedicated paper account reset to that capital."
                    )
                high = baseline = snapshot.net_liquidation
                halt = None
            else:
                old_hash, high, old_day, baseline, last, halt, last_observed = row
                if now < pd.Timestamp(last_observed):
                    raise QuantError("Account observation time moved backwards.")
                if old_hash != fingerprint:
                    raise QuantError(
                        "Paper configuration changed; reconcile persistent risk state first."
                    )
                if old_day != day:
                    baseline = last
                high = max(high, snapshot.net_liquidation)
            if snapshot.net_liquidation / baseline - 1.0 <= -self.config.max_daily_loss:
                halt = "Daily/overnight loss limit breached; operator reconciliation required."
            if snapshot.net_liquidation / high - 1.0 <= -self.config.max_drawdown:
                halt = "High-water drawdown limit breached; operator reconciliation required."
            connection.execute(
                "INSERT OR REPLACE INTO account_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.account,
                    fingerprint,
                    high,
                    day,
                    baseline,
                    snapshot.net_liquidation,
                    halt,
                    now.isoformat(),
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?)",
                (
                    now.isoformat(),
                    snapshot.account,
                    snapshot.net_liquidation,
                    settled_cash,
                    json.dumps(snapshot.positions, sort_keys=True),
                ),
            )
        if halt:
            raise QuantError(halt)

    def reserve(self, signal: dict, orders: list[PlannedOrder]) -> str:
        run_id = digest_json({"account": self.config.account, "signal": signal})
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            risk = self.connection.execute(
                "SELECT halt_reason FROM account_state WHERE account=?",
                (self.config.account,),
            ).fetchone()
            if risk is None or risk[0]:
                raise QuantError("Observe a healthy account before reserving a paper run.")
            incomplete = self.connection.execute(
                "SELECT run_id FROM runs WHERE account=? AND status!='completed' LIMIT 1",
                (self.config.account,),
            ).fetchone()
            if incomplete:
                raise QuantError("An unfinished paper run requires broker/ledger reconciliation.")
            try:
                self.connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, 'reserved', ?, ?)",
                    (
                        run_id,
                        self.config.account,
                        signal["signal_date"],
                        json.dumps([asdict(order) for order in orders]),
                        utc_now(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise QuantError(
                    "This account/month-end signal was already reserved; refusing replay."
                ) from exc
            self.connection.executemany(
                "INSERT INTO orders VALUES (?, ?, 'reserved', NULL, ?)",
                [(order.order_ref, run_id, utc_now()) for order in orders],
            )
        return run_id

    def order_state(self, reference: str, status: str, outcome: Outcome | None = None) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE orders SET status=?, outcome_json=?, updated_at=? WHERE order_ref=?",
                (status, json.dumps(asdict(outcome)) if outcome else None, utc_now(), reference),
            )
            if cursor.rowcount != 1:
                raise QuantError("Unknown journal order reference.")

    def run_state(self, run_id: str, status: str) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?", (status, utc_now(), run_id)
            )
            if cursor.rowcount != 1:
                raise QuantError("Unknown journal run ID.")


class Broker(Protocol):
    def snapshot(self) -> Snapshot: ...
    def quotes(self, symbols: set[str]) -> dict[str, Quote]: ...
    def submit(self, order: PlannedOrder) -> Outcome: ...


def execute_plan(
    config: PaperConfig,
    broker: Broker,
    ledger: PaperLedger,
    signal: dict,
    *,
    submit: bool = False,
    confirmation: str | None = None,
) -> dict:
    now = now_utc()
    snapshot = broker.snapshot()
    ledger.observe(snapshot, now)
    weights = validated_weights(config, signal)
    symbols = set(snapshot.positions) | {key for key, value in weights.items() if value > 0}
    quotes = broker.quotes(symbols)
    orders = build_plan(config, snapshot, signal, quotes, now_utc())
    if not submit:
        return {"mode": "dry_run", "orders_sent": 0, "orders": [asdict(order) for order in orders]}
    if config.allow_submit is not True or confirmation != config.account:
        raise QuantError(
            "Submission requires allow_submit=true and an exact DU account confirmation."
        )
    run_id = ledger.reserve(signal, orders)
    completed = False
    expected = snapshot.positions.copy()
    outcomes = []
    try:
        for order in orders:
            if Path(config.kill_switch_file).exists():
                raise QuantError("Kill switch activated; remaining paper orders were not sent.")
            current = broker.snapshot()
            ledger.observe(current, now_utc())
            if current.positions != {key: value for key, value in expected.items() if value != 0}:
                raise QuantError(
                    "Positions changed since planning; no remaining orders will be sent."
                )
            fresh_quotes = broker.quotes(set(current.positions) | {order.symbol})
            fresh_now = now_utc()
            require_execution_window(
                signal["signal_date"], fresh_now, config.execution_window_minutes
            )
            for fresh_quote in fresh_quotes.values():
                fresh_quote.validate(config, fresh_now)
            projected = current.positions.copy()
            delta = order.quantity if order.action == "BUY" else -order.quantity
            projected[order.symbol] = projected.get(order.symbol, 0) + delta
            marked_values = [
                quantity * fresh_quotes[symbol].ask for symbol, quantity in projected.items()
            ]
            if (
                any(quantity < 0 for quantity in projected.values())
                or any(
                    value > current.net_liquidation * config.max_position_weight + 1e-6
                    for value in marked_values
                )
                or sum(marked_values) > current.net_liquidation * config.max_gross_exposure + 1e-6
            ):
                raise QuantError("Fresh prices or account values invalidate the planned exposure.")
            if (
                order.action == "BUY"
                and order.quantity * order.limit_price + config.commission_reserve_per_order
                > current.required_settled_cash()
            ):
                raise QuantError("Settled cash changed; refusing to borrow for a paper purchase.")
            ledger.order_state(order.order_ref, "submitting")
            outcome = broker.submit(order)
            ledger.order_state(order.order_ref, outcome.status, outcome)
            outcomes.append({"order": asdict(order), "outcome": asdict(outcome)})
            if (
                outcome.status != "Filled"
                or outcome.filled != order.quantity
                or not math.isfinite(outcome.average_price)
                or outcome.average_price <= 0
                or (order.action == "BUY" and outcome.average_price > order.limit_price + 1e-6)
                or (order.action == "SELL" and outcome.average_price < order.limit_price - 1e-6)
            ):
                raise QuantError(
                    "Partial, rejected, cancelled, or uncertain fill; stop and reconcile."
                )
            expected[order.symbol] = expected.get(order.symbol, 0) + delta
        final = broker.snapshot()
        ledger.observe(final, now_utc())
        if final.positions != {key: value for key, value in expected.items() if value != 0}:
            raise QuantError("Final positions do not match confirmed fills.")
        ledger.run_state(run_id, "completed")
        completed = True
        return {"mode": "paper", "run_id": run_id, "orders_sent": len(outcomes), "fills": outcomes}
    finally:
        if not completed:
            ledger.run_state(run_id, "needs_reconciliation")


class IBPaperBroker:
    def __init__(self, config: PaperConfig, *, readonly: bool = True):
        config.validate()
        from ib_async import IB

        self.config = config
        self.readonly = readonly
        self.ib = IB()
        self.ib.RequestTimeout = 15
        self.ib.RaiseRequestErrors = True
        self.contracts = {}

    def __enter__(self) -> IBPaperBroker:
        from ib_async.wrapper import RequestError

        verified = False
        try:
            verified_paper_handshake(
                self.ib,
                self.config.host,
                self.config.port,
                self.config.client_id,
                expected_account=self.config.account,
            )
            self.ib.reqAccountUpdates(self.config.account)
            verified = True
            return self
        except (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError, RequestError) as exc:
            raise QuantError(f"IBKR paper connection failed: {exc}") from exc
        finally:
            if not verified:
                self.ib.disconnect()

    def __exit__(self, *args) -> None:
        self.ib.disconnect()

    def snapshot(self) -> Snapshot:
        from ib_async import ExecutionFilter

        if not self.ib.isConnected() or self.ib.managedAccounts() != [self.config.account]:
            raise QuantError(
                "Paper account disconnected or changed; reconnection is not automatic."
            )
        summary = self.ib.accountSummary(self.config.account)

        def reported_value(tag: str) -> float | None:
            entries = [item for item in summary if item.tag == tag and item.currency == "USD"]
            if len(entries) > 1:
                raise QuantError(f"Ambiguous USD {tag} account values.")
            if not entries:
                return None
            try:
                return float(entries[0].value)
            except ValueError as exc:
                raise QuantError(f"Invalid numeric USD {tag} account value.") from exc

        net_liquidation = reported_value("NetLiquidation")
        if net_liquidation is None:
            raise QuantError("IBKR did not report USD NetLiquidation.")

        for item in self.ib.accountValues(self.config.account):
            if item.tag == "CashBalance" and item.currency not in {"USD", "BASE"}:
                if abs(float(item.value)) > 0.01:
                    raise QuantError(
                        "Foreign-currency cash is unsupported in this dedicated USD account."
                    )
        positions = {}
        for item in self.ib.reqPositions():
            if item.account != self.config.account or item.position == 0:
                continue
            contract = item.contract
            if (
                contract.secType != "STK"
                or contract.currency != "USD"
                or contract.symbol not in self.config.allowed_symbols
                or item.position < 0
                or not float(item.position).is_integer()
                or contract.symbol in positions
            ):
                raise QuantError("Unsupported, duplicate, fractional, or short broker position.")
            positions[contract.symbol] = int(item.position)
        open_refs = tuple(
            trade.order.orderRef or "<unattributed>"
            for trade in self.ib.reqAllOpenOrders()
            if not trade.isDone()
        )
        executions = self.ib.reqExecutions(ExecutionFilter(acctCode=self.config.account))
        snapshot = Snapshot(
            account=self.config.account,
            currency="USD",
            net_liquidation=net_liquidation,
            settled_cash=reported_value("SettledCash"),
            positions=positions,
            open_order_refs=open_refs,
            executed_order_refs=tuple(fill.execution.orderRef for fill in executions),
        )
        snapshot.validate(self.config, require_settled_cash=not self.readonly)
        return snapshot

    def quotes(self, symbols: set[str]) -> dict[str, Quote]:
        from ib_async import Stock
        from ib_async.wrapper import RequestError

        if not symbols <= set(self.config.allowed_symbols):
            raise QuantError("Attempt to quote a non-allowlisted instrument.")
        for symbol in sorted(symbols):
            if symbol not in self.contracts:
                try:
                    matches = self.ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
                except RequestError as exc:
                    raise QuantError(
                        f"IBKR contract request failed ({exc.code}): {exc.message}"
                    ) from exc
                if len(matches) != 1:
                    raise QuantError(f"Ambiguous or unavailable IBKR contract for {symbol}.")
                contract = matches[0]
                if (
                    contract.symbol != symbol
                    or contract.secType != "STK"
                    or contract.currency != "USD"
                    or contract.conId <= 0
                ):
                    raise QuantError("Qualified contract failed symbol/currency/type checks.")
                self.contracts[symbol] = contract
        if not symbols:
            return {}
        self.ib.reqMarketDataType(1)
        try:
            tickers = self.ib.reqTickers(*(self.contracts[key] for key in sorted(symbols)))
        except RequestError as exc:
            raise QuantError(
                f"IBKR market-data request failed ({exc.code}): {exc.message}"
            ) from exc
        result = {}
        for ticker in tickers:
            if ticker.time is None:
                raise QuantError("IBKR did not provide a timestamped quote.")
            symbol = ticker.contract.symbol
            result[symbol] = Quote(
                symbol, ticker.bid, ticker.ask, pd.Timestamp(ticker.time), ticker.marketDataType
            )
        if set(result) != symbols:
            raise QuantError("IBKR did not return all requested quotes.")
        return result

    def submit(self, order: PlannedOrder) -> Outcome:
        if self.readonly or not self.config.allow_submit or order.symbol not in self.contracts:
            raise QuantError("Submission is disabled or the contract was not qualified.")
        return submit_paper_ioc(self.ib, self.config, self.contracts[order.symbol], order)
