from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import pandas as pd

from us_quant.calendar import market_calendar
from us_quant.config import QuantError
from us_quant.free_quotes import ReferencePrice, fetch_public_reference
from us_quant.paper import (
    IBPaperBroker,
    Outcome,
    PaperConfig,
    PlannedOrder,
    load_paper_config,
    now_utc,
    submit_paper_ioc,
)
from us_quant.storage import digest_json, file_digest, read_json, utc_now

CAPITAL = 10000.0
MAX_TICKET = 1000.0
MAX_DAILY_LOSS = 100.0
MAX_CUMULATIVE_LOSS = 1000.0
LIMIT_COLLAR_BPS = 10.0


def lab_fingerprint() -> str:
    root = Path(__file__).parent
    return digest_json(
        {name: file_digest(root / name) for name in ("paper_lab.py", "free_quotes.py", "paper.py")}
    )


def regular_session(now: pd.Timestamp) -> pd.Timestamp:
    if now.tzinfo is None:
        raise QuantError("An explicit execution timezone is required.")
    calendar = market_calendar()
    day = pd.Timestamp(now.tz_convert("America/New_York").date())
    if not calendar.is_session(day):
        raise QuantError("The paper lab only trades on regular exchange sessions.")
    if not (
        calendar.session_open(day) + pd.Timedelta(minutes=5)
        <= now
        <= calendar.session_close(day) - pd.Timedelta(minutes=15)
    ):
        raise QuantError("Paper lab execution is limited to 09:35 through 15 minutes before close.")
    return day


def wait_for_regular_window(maximum_seconds: int) -> None:
    if type(maximum_seconds) is not int or not 0 <= maximum_seconds <= 3600:
        raise QuantError("A paper test may wait at most one hour for today's regular session.")
    now = now_utc()
    calendar = market_calendar()
    day = pd.Timestamp(now.tz_convert("America/New_York").date())
    if not calendar.is_session(day):
        raise QuantError("No paper experiment on a market holiday.")
    start = calendar.session_open(day) + pd.Timedelta(minutes=5)
    delay = max(0.0, (start - now).total_seconds())
    if delay > maximum_seconds:
        raise QuantError("Execution window has not opened within the authorized bounded wait.")
    if delay:
        time.sleep(delay)
    regular_session(now_utc())


@dataclass(frozen=True)
class LabAccount:
    account: str
    net_liquidation: float
    reported_usd_cash: float
    available_funds: float
    positions: dict[str, int]
    open_order_refs: tuple[str, ...] = ()
    execution_refs: tuple[str, ...] = ()

    def validate(self, config: PaperConfig) -> None:
        config.validate()
        if self.account != config.account:
            raise QuantError("Experimental account identity changed.")
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.net_liquidation, self.reported_usd_cash, self.available_funds)
        ):
            raise QuantError("Experimental USD cash, available funds, and equity must be positive.")
        if self.open_order_refs:
            raise QuantError("Open orders require reconciliation before an experiment.")
        if any(
            symbol not in config.allowed_symbols or type(quantity) is not int or quantity < 0
            for symbol, quantity in self.positions.items()
        ):
            raise QuantError("Unsupported or short positions in the experimental account.")

    @property
    def funding_limit(self) -> float:
        return min(self.reported_usd_cash, self.available_funds)

    def public_dict(self) -> dict:
        value = asdict(self)
        value["account"] = "DU***" + self.account[-4:]
        return value


class LabBroker(Protocol):
    def account(self) -> LabAccount: ...
    def reference(self, symbol: str) -> ReferencePrice: ...
    def send(self, order: PlannedOrder) -> Outcome: ...


class IBExperimentBroker:
    def __init__(self, config: PaperConfig, *, authorized: bool):
        self.config = config
        self.authorized = authorized
        self.reader = IBPaperBroker(config, readonly=True)

    def __enter__(self) -> IBExperimentBroker:
        self.reader.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.reader.__exit__(*args)

    def account(self) -> LabAccount:
        snapshot = self.reader.snapshot()
        values = self.reader.ib.accountValues(self.config.account)

        def exact(tag: str) -> float:
            entries = [
                item
                for item in values
                if item.tag == tag
                and item.currency == "USD"
                and item.account == self.config.account
            ]
            if len(entries) != 1:
                raise QuantError(f"Paper lab requires exactly one reported USD {tag}.")
            try:
                return float(entries[0].value)
            except ValueError as exc:
                raise QuantError(f"Invalid reported paper-lab {tag}.") from exc

        result = LabAccount(
            snapshot.account,
            snapshot.net_liquidation,
            exact("CashBalance"),
            exact("AvailableFunds"),
            snapshot.positions,
            snapshot.open_order_refs,
            snapshot.executed_order_refs,
        )
        result.validate(self.config)
        return result

    def reference(self, symbol: str) -> ReferencePrice:
        return fetch_public_reference(symbol)

    def order_permission(self) -> dict:
        from ib_async import LimitOrder, Stock
        from ib_async.wrapper import RequestError

        self.account()
        try:
            matches = self.reader.ib.qualifyContracts(Stock("SPY", "SMART", "USD"))
        except RequestError as exc:
            raise QuantError(
                f"Permission contract lookup failed ({exc.code}): {exc.message}"
            ) from exc
        if len(matches) != 1:
            raise QuantError("The paper permission check requires one SPY contract.")
        reference = self.reference("SPY")
        price = experimental_order(reference, "BUY", "what_if", now_utc()).limit_price
        result = {
            "mode": "broker_what_if_only",
            "order_entry_available": False,
            "actual_orders_sent": 0,
            "error_codes": [],
        }

        def capture_error(request_id, code, message, contract):
            if code not in {2104, 2106, 2158}:
                result["error_codes"].append(code)

        timeout = self.reader.ib.RequestTimeout
        self.reader.ib.RequestTimeout = 5
        self.reader.ib.errorEvent += capture_error
        try:
            state = self.reader.ib.whatIfOrder(
                matches[0],
                LimitOrder(
                    "BUY", 1, price, account=self.config.account, tif="DAY", outsideRth=False
                ),
            )
            result["what_if_status"] = state.status
            result["warning"] = state.warningText
            result["order_entry_available"] = (
                state.status in {"PreSubmitted", "Submitted"}
                and not result["error_codes"]
                and not state.warningText
            )
        except (RequestError, asyncio.TimeoutError, ConnectionError) as exc:
            result["error"] = str(exc) or type(exc).__name__
        finally:
            self.reader.ib.errorEvent -= capture_error
            self.reader.ib.RequestTimeout = timeout
        return result

    def send(self, order: PlannedOrder) -> Outcome:
        from ib_async import Stock
        from ib_async.wrapper import RequestError

        if not self.authorized:
            raise QuantError("Paper lab order authority is disabled.")
        regular_session(now_utc())
        if Path(self.config.kill_switch_file).exists():
            raise QuantError("Kill switch exists; no new paper-lab order will be sent.")
        try:
            matches = self.reader.ib.qualifyContracts(Stock(order.symbol, "SMART", "USD"))
        except RequestError as exc:
            raise QuantError(
                f"Paper experiment contract failed ({exc.code}): {exc.message}"
            ) from exc
        if len(matches) != 1:
            raise QuantError("The experimental contract is ambiguous.")
        policy = replace(self.config, allow_submit=True, max_order_notional=MAX_TICKET)
        return submit_paper_ioc(self.reader.ib, policy, matches[0], order)


class ExperimentLedger:
    def __init__(self, path: Path, *, create: bool = False):
        if path.is_symlink() or (create and path.exists()):
            raise QuantError("Refusing to replace or follow a paper-lab ledger.")
        if not create and not path.is_file():
            raise QuantError("Explicitly authorize and initialize the paper lab first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if create:
            path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
        self.connection = sqlite3.connect(path, timeout=5)
        if create:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE authorization (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
                CREATE TABLE experiments (
                  id TEXT PRIMARY KEY, session_date TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
                  created_at TEXT NOT NULL, before_json TEXT NOT NULL, result_json TEXT
                );
                CREATE TABLE tickets (
                  order_ref TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                  instruction_json TEXT NOT NULL,
                  reference_json TEXT NOT NULL, status TEXT NOT NULL, outcome_json TEXT,
                  updated_at TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        self.connection.close()

    def authorize(self, config: PaperConfig, account: LabAccount) -> dict:
        account.validate(config)
        if account.positions or account.open_order_refs:
            raise QuantError("Initialize only a flat paper account; preserve unrelated holdings.")
        if account.funding_limit < CAPITAL:
            raise QuantError("The paper account cannot support the declared USD 10,000 sleeve.")
        metadata = {
            "authorized_at": utc_now(),
            "mode": "user_authorized_unqualified_paper_experiments",
            "authority": "Explicit user permission to test strategies in resettable simulation",
            "account": config.account,
            "paper_config_sha256": digest_json(asdict(config)),
            "engine_sha256": lab_fingerprint(),
            "experimental_capital_usd": CAPITAL,
            "expected_broker_equity_at_registration": account.net_liquidation,
            "funding_basis": "min(reported USD CashBalance, AvailableFunds); paper simulation only",
            "settled_cash_claimed": False,
            "max_ticket_usd": MAX_TICKET,
            "max_daily_loss_usd": MAX_DAILY_LOSS,
            "max_cumulative_loss_usd": MAX_CUMULATIVE_LOSS,
            "max_experiments_per_session": 1,
            "max_tickets_per_experiment": 2,
            "qualified_strategy_required": False,
            "live_account_authority": False,
            "broker_balance_reset_performed": False,
            "objective_verified": False,
            "expires_at": (now_utc() + pd.Timedelta(days=30)).isoformat(),
        }
        with self.connection:
            if self.connection.execute("SELECT COUNT(*) FROM authorization").fetchone()[0]:
                raise QuantError(
                    "The lab is already authorized; do not reset its performance history."
                )
            self.connection.execute(
                "INSERT INTO authorization VALUES (1, ?)", (json.dumps(metadata, sort_keys=True),)
            )
        public = {**metadata, "account": "DU***" + config.account[-4:]}
        return public

    def verified_authorization(
        self, config: PaperConfig, *, require_active: bool = True, require_engine: bool = True
    ) -> dict:
        row = self.connection.execute("SELECT body FROM authorization WHERE id=1").fetchone()
        if row is None:
            raise QuantError("Paper experimental permission is not recorded.")
        metadata = json.loads(row[0])
        if (
            metadata["account"] != config.account
            or metadata["paper_config_sha256"] != digest_json(asdict(config))
            or (require_engine and metadata["engine_sha256"] != lab_fingerprint())
            or (require_active and now_utc() > pd.Timestamp(metadata["expires_at"]))
        ):
            raise QuantError(
                "Paper experiment identity, limits, code, or authorization changed/expired."
            )
        return metadata

    def reconcile_rejection(
        self,
        config: PaperConfig,
        account: LabAccount,
        permission: dict,
        experiment: str,
        *,
        accept_code_update: bool = False,
        legacy_receipt: Path | None = None,
        legacy_private: Path | None = None,
    ) -> dict:
        metadata = self.verified_authorization(config, require_engine=False)
        account.validate(config)
        if (
            permission.get("mode") != "broker_what_if_only"
            or permission.get("order_entry_available") is not True
            or permission.get("actual_orders_sent") != 0
            or permission.get("error_codes") != []
            or permission.get("what_if_status") not in {"PreSubmitted", "Submitted"}
        ):
            raise QuantError("A successful non-executing broker permission check is required.")
        if (
            account.positions
            or account.open_order_refs
            or any(ref.startswith(experiment) for ref in account.execution_refs)
        ):
            raise QuantError("Broker exposure or matching executions prevent a no-fill resolution.")
        baseline = metadata["expected_broker_equity_at_registration"]
        if abs(account.net_liquidation - baseline) > max(CAPITAL, baseline * 0.05):
            raise QuantError(
                "Account reset or material equity change needs a separate operator audit."
            )
        if (
            metadata["experimental_capital_usd"] != CAPITAL
            or metadata["max_ticket_usd"] != MAX_TICKET
            or metadata["max_daily_loss_usd"] != MAX_DAILY_LOSS
            or metadata["max_cumulative_loss_usd"] != MAX_CUMULATIVE_LOSS
            or metadata["max_experiments_per_session"] != 1
            or metadata["max_tickets_per_experiment"] != 2
        ):
            raise QuantError("Reconciliation may not change experimental risk limits.")
        code_changed = metadata["engine_sha256"] != lab_fingerprint()
        if code_changed and not accept_code_update:
            raise QuantError("A reviewed code revision requires explicit --accept-code-update.")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            unfinished = self.connection.execute(
                "SELECT id, session_date, result_json FROM experiments "
                "WHERE status NOT IN ('completed', 'not_filled')"
            ).fetchall()
            if len(unfinished) != 1 or unfinished[0][0] != experiment:
                raise QuantError("Resolve exactly the one retained unsuccessful experiment.")
            _, day, previous_json = unfinished[0]
            if previous_json is None:
                raise QuantError("An unfinished attempt without a result needs an operator audit.")
            if day != now_utc().tz_convert("America/New_York").date().isoformat():
                raise QuantError(
                    "Older attempts require broker statements, not today's execution cache."
                )
            tickets = self.connection.execute(
                "SELECT order_ref, instruction_json, outcome_json FROM tickets "
                "WHERE experiment_id=?",
                (experiment,),
            ).fetchall()
            if len(tickets) != 1:
                raise QuantError(
                    "Automatic no-fill resolution only supports one rejected entry ticket."
                )
            reference, instruction_json, outcome_json = tickets[0]
            instruction = json.loads(instruction_json)
            if (
                reference != f"{experiment}_buy"
                or instruction.get("action") != "BUY"
                or instruction.get("quantity") != 1
                or instruction.get("symbol") != "SPY"
            ):
                raise QuantError("The retained ticket is not the experiment's entry order.")
            outcome = json.loads(outcome_json) if outcome_json else {}
            rejected = (
                outcome.get("status") == "ValidationError"
                and outcome.get("filled") == 0
                and outcome.get("permanent_id") == 0
                and 321 in outcome.get("error_codes", [])
            )
            legacy_hash = None
            if not rejected:
                if legacy_receipt is None or legacy_private is None:
                    raise QuantError(
                        "Unknown outcomes require original rejection evidence; never infer a fill."
                    )
                receipt, private = read_json(legacy_receipt), read_json(legacy_private)
                legacy_hash = digest_json(private)
                if receipt.get("private_original_sha256") != legacy_hash:
                    raise QuantError("The legacy rejection receipt fingerprint does not match.")
                message = private.get("error", "")
                message = re.sub(
                    r"\\+u([a-fA-F0-9]{4})", lambda match: chr(int(match.group(1), 16)), message
                )
                if (
                    private.get("exit_code") != 2
                    or f"orderRef='{reference}'" not in message
                    or f"account='{config.account}'" not in message
                    or "ValidationError" not in message
                    or "errorCode=321" not in message
                    or "permId=0" not in message
                    or "fills=[]" not in message
                    or not ("\u53ea\u8bfb" in message or "read-only" in message.lower())
                ):
                    raise QuantError(
                        "Legacy evidence does not prove this paper entry was rejected."
                    )
            previous = json.loads(previous_json)
            if previous.get("fills"):
                raise QuantError(
                    "An experiment with recorded fills cannot be resolved as unfilled."
                )
            result = {
                **previous,
                "status": "not_filled",
                "resolution": "broker_verified_rejected_no_fill",
                "realized_net_pnl": 0.0,
                "reconciled_at": utc_now(),
                "matching_executions": [],
                "broker_position_after": {},
                "same_day_retry_allowed": False,
            }
            revised = {**metadata, "engine_sha256": lab_fingerprint()}
            audit = {
                "recorded_at": utc_now(),
                "experiment": experiment,
                "previous_result": previous,
                "resolved_result": result,
                "previous_authorization": metadata,
                "revised_authorization": revised,
                "permission_check": permission,
                "broker_account": account.public_dict(),
                "legacy_rejection_sha256": legacy_hash,
                "order_submissions": 0,
                "risk_limits_changed": False,
            }
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS reconciliation_audit ("
                "id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL, body TEXT NOT NULL)"
            )
            self.connection.execute(
                "INSERT INTO reconciliation_audit (recorded_at, body) VALUES (?, ?)",
                (audit["recorded_at"], json.dumps(audit, sort_keys=True)),
            )
            self.connection.execute(
                "UPDATE experiments SET status='not_filled', result_json=? WHERE id=?",
                (json.dumps(result, sort_keys=True), experiment),
            )
            self.connection.execute(
                "UPDATE authorization SET body=? WHERE id=1",
                (json.dumps(revised, sort_keys=True),),
            )
        return {
            "experiment": experiment,
            "resolution": result["resolution"],
            "original_history_retained": True,
            "risk_limits_changed": False,
            "same_day_retry_allowed": False,
            "orders_sent": 0,
        }

    def reserve(self, config: PaperConfig, account: LabAccount, session: pd.Timestamp) -> str:
        metadata = self.verified_authorization(config)
        account.validate(config)
        if account.positions:
            raise QuantError(
                "A smoke test starts flat; never liquidate or reuse unrelated positions."
            )
        baseline = metadata["expected_broker_equity_at_registration"]
        if abs(account.net_liquidation - baseline) > max(CAPITAL, baseline * 0.05):
            raise QuantError(
                "Broker equity changed materially; an account reset needs explicit reconciliation."
            )
        day = session.date().isoformat()
        identifier = "lab_" + session.strftime("%Y%m%d")
        if any(reference.startswith(identifier) for reference in account.execution_refs):
            raise QuantError("The broker already recorded this experiment; never replay it.")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            unfinished = self.connection.execute(
                "SELECT id FROM experiments WHERE status NOT IN ('completed', 'not_filled') LIMIT 1"
            ).fetchone()
            if unfinished:
                raise QuantError(
                    "Unfinished or fee-pending paper experiment requires reconciliation."
                )
            previous = [
                json.loads(row[0])
                for row in self.connection.execute(
                    "SELECT result_json FROM experiments WHERE status='completed'"
                ).fetchall()
            ]
            cumulative = sum(row["realized_net_pnl"] for row in previous)
            if cumulative <= -MAX_CUMULATIVE_LOSS:
                raise QuantError("Paper sleeve loss limit has been reached.")
            try:
                self.connection.execute(
                    "INSERT INTO experiments VALUES (?, ?, 'reserved', ?, ?, NULL)",
                    (identifier, day, utc_now(), json.dumps(asdict(account), sort_keys=True)),
                )
            except sqlite3.IntegrityError as exc:
                raise QuantError(
                    "One paper experiment per session; refusing duplicate execution."
                ) from exc
        return identifier

    def ticket(self, experiment: str, order: PlannedOrder, reference: ReferencePrice) -> None:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            record = self.connection.execute(
                "SELECT status FROM experiments WHERE id=?", (experiment,)
            ).fetchone()
            if record is None or record[0] != "reserved":
                raise QuantError("Only the currently reserved experiment may submit a ticket.")
            if (
                order.symbol != reference.symbol
                or order.quantity != 1
                or order.action not in {"BUY", "SELL"}
                or order.order_ref != f"{experiment}_{order.action.lower()}"
            ):
                raise QuantError("Experimental ticket does not match its persistent intent.")
            count = self.connection.execute(
                "SELECT COUNT(*) FROM tickets WHERE experiment_id=?", (experiment,)
            ).fetchone()[0]
            if count >= 2:
                raise QuantError("The paper smoke test may send at most two orders.")
            self.connection.execute(
                "INSERT INTO tickets VALUES (?, ?, ?, ?, 'submitting', NULL, ?)",
                (
                    order.order_ref,
                    experiment,
                    json.dumps(asdict(order), sort_keys=True),
                    json.dumps(asdict(reference), sort_keys=True),
                    utc_now(),
                ),
            )

    def outcome(self, reference: str, outcome: Outcome) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE tickets SET status=?, outcome_json=?, updated_at=? WHERE order_ref=?",
                (outcome.status, json.dumps(asdict(outcome)), utc_now(), reference),
            )
            if cursor.rowcount != 1:
                raise QuantError("Experimental fill has no prior persistent order reservation.")

    def finish(self, experiment: str, status: str, result: dict) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE experiments SET status=?, result_json=? WHERE id=?",
                (status, json.dumps(result, sort_keys=True, allow_nan=False), experiment),
            )
            if cursor.rowcount != 1:
                raise QuantError("Unknown experiment completion.")

    def status(self, config: PaperConfig) -> dict:
        metadata = self.verified_authorization(config, require_active=False, require_engine=False)
        records = [
            {"id": row[0], "status": row[1], "result": json.loads(row[2]) if row[2] else None}
            for row in self.connection.execute(
                "SELECT id, status, result_json FROM experiments ORDER BY created_at"
            )
        ]
        return {
            "mode": metadata["mode"],
            "experimental_capital_usd": CAPITAL,
            "expected_broker_equity_at_registration": metadata[
                "expected_broker_equity_at_registration"
            ],
            "broker_balance_reset_performed": False,
            "experiments": records,
            "objective_verified": False,
            "live_account_authority": False,
            "qualification_evidence": False,
            "execution_code_matches_authorization": metadata["engine_sha256"] == lab_fingerprint(),
        }


def experimental_order(
    reference: ReferencePrice, action: str, identifier: str, now: pd.Timestamp
) -> PlannedOrder:
    reference.validate_for_experiment(now)
    if reference.symbol != "SPY" or action not in {"BUY", "SELL"}:
        raise QuantError("The first laboratory experiment is restricted to one SPY share.")
    multiplier = 1 + (LIMIT_COLLAR_BPS / 10000) * (1 if action == "BUY" else -1)
    rounding = math.ceil if action == "BUY" else math.floor
    price = rounding(reference.price * multiplier * 100) / 100
    if not 0 < price <= min(MAX_TICKET, CAPITAL * 0.1):
        raise QuantError("One share would exceed the experimental ticket or sleeve exposure limit.")
    return PlannedOrder("SPY", action, 1, price, f"{identifier}_{action.lower()}")


def validate_outcome(order: PlannedOrder, outcome: Outcome) -> None:
    if (
        outcome.status != "Filled"
        or outcome.filled != 1
        or not math.isfinite(outcome.average_price)
        or outcome.average_price <= 0
        or (order.action == "BUY" and outcome.average_price > order.limit_price + 1e-6)
        or (order.action == "SELL" and outcome.average_price < order.limit_price - 1e-6)
    ):
        raise QuantError(
            f"Paper fill requires reconciliation: status={outcome.status}, "
            f"filled={outcome.filled}, errors={outcome.error_codes}."
        )
    if outcome.commission is not None and (
        not math.isfinite(outcome.commission) or outcome.commission < 0
    ):
        raise QuantError("Invalid paper commission report.")


def run_smoke(
    config: PaperConfig, ledger: ExperimentLedger, broker: LabBroker, *, execute: bool
) -> dict:
    ledger.verified_authorization(config)
    session = regular_session(now_utc())
    if Path(config.kill_switch_file).exists():
        raise QuantError("Kill switch is active; experimental orders are disabled.")
    account = broker.account()
    account.validate(config)
    if account.positions:
        raise QuantError("Paper execution smoke test requires a flat account.")
    reference = broker.reference("SPY")
    order = experimental_order(reference, "BUY", "preview", now_utc())
    if account.funding_limit < CAPITAL or order.limit_price + 10 > account.funding_limit:
        raise QuantError("Reported paper cash cannot fund the bounded research sleeve.")
    if not execute:
        return {
            "mode": "paper_lab_dry_run",
            "orders_sent": 0,
            "order": asdict(order),
            "reference": asdict(reference),
            "experimental_capital_usd": CAPITAL,
            "objective_verified": False,
        }
    identifier = ledger.reserve(config, account, session)
    finished = False
    fills = []
    result = {
        "mode": "paper_execution_mechanics_only",
        "experiment_id": identifier,
        "experimental_capital_usd": CAPITAL,
        "orders_sent": 0,
        "fills": fills,
        "qualification_evidence": False,
        "objective_verified": False,
        "funding_basis": "reported USD cash for simulated orders, NOT claimed settled cash",
        "external_price_source": "Yahoo indicative complete 1m bar, not certified realtime NBBO",
    }
    try:
        for action in ("BUY", "SELL"):
            regular_session(now_utc())
            if Path(config.kill_switch_file).exists():
                raise QuantError("Kill switch activated; inspect and reconcile any paper position.")
            current = broker.account()
            current.validate(config)
            expected = {} if action == "BUY" else {"SPY": 1}
            if current.positions != expected:
                raise QuantError(
                    "Paper positions changed unexpectedly; preserve them and reconcile."
                )
            reference = broker.reference("SPY")
            order = experimental_order(reference, action, identifier, now_utc())
            if action == "BUY" and current.funding_limit < order.limit_price + 10:
                raise QuantError("Insufficient reported experimental paper cash.")
            ledger.ticket(identifier, order, reference)
            result["orders_sent"] += 1
            outcome = broker.send(order)
            ledger.outcome(order.order_ref, outcome)
            fills.append({"order": asdict(order), "outcome": asdict(outcome)})
            if (
                action == "BUY"
                and outcome.status in {"Cancelled", "ApiCancelled"}
                and outcome.filled == 0
            ):
                final = broker.account()
                final.validate(config)
                if final.positions:
                    raise QuantError("A supposedly unfilled entry changed account positions.")
                result.update(
                    {"status": "not_filled", "position_after": {}, "realized_net_pnl": 0.0}
                )
                ledger.finish(identifier, "not_filled", result)
                finished = True
                return result
            validate_outcome(order, outcome)
        final = broker.account()
        final.validate(config)
        if final.positions:
            raise QuantError("The test did not return the account to flat; reconcile the residual.")
        entry, exit_fill = (row["outcome"] for row in fills)
        gross_pnl = exit_fill["average_price"] - entry["average_price"]
        costs = (
            entry["commission"] + exit_fill["commission"]
            if entry["commission"] is not None and exit_fill["commission"] is not None
            else None
        )
        net_pnl = gross_pnl - costs if costs is not None else None
        status = "completed" if costs is not None else "completed_costs_pending"
        if net_pnl is not None and net_pnl <= -MAX_DAILY_LOSS:
            status = "risk_halted"
        result.update(
            {
                "status": status,
                "position_after": {},
                "gross_pnl": gross_pnl,
                "commission_usd": costs,
                "realized_net_pnl": net_pnl,
                "sleeve_equity_after": CAPITAL + net_pnl if net_pnl is not None else None,
            }
        )
        ledger.finish(identifier, status, result)
        finished = True
        return result
    finally:
        if not finished:
            result["status"] = "needs_reconciliation"
            ledger.finish(identifier, "needs_reconciliation", result)


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "paper-lab", help="Explicit bounded paper experiments; not qualification."
    )
    command.add_argument("stage", choices=["init", "status", "smoke", "reconcile"])
    command.add_argument("--paper-config", type=Path, default=Path("config/paper.json"))
    command.add_argument("--ledger", type=Path, default=Path("runtime/paper-lab.sqlite3"))
    command.add_argument("--authorize-experiments", action="store_true")
    command.add_argument("--execute", action="store_true")
    command.add_argument("--wait-seconds", type=int, default=0)
    command.add_argument("--experiment-id")
    command.add_argument("--confirm-rejected-no-fill", action="store_true")
    command.add_argument("--accept-code-update", action="store_true")
    command.add_argument("--legacy-rejection-receipt", type=Path)
    command.add_argument("--legacy-private-receipt", type=Path)


def dispatch_lab(args: argparse.Namespace) -> dict:
    config = load_paper_config(args.paper_config)
    if args.stage == "init":
        if not args.authorize_experiments or args.execute:
            raise QuantError(
                "Initialization needs explicit experimental permission and sends no orders."
            )
        with IBExperimentBroker(config, authorized=False) as broker:
            account = broker.account()
        ledger = ExperimentLedger(args.ledger, create=True)
        try:
            return ledger.authorize(config, account)
        finally:
            ledger.close()
    ledger = ExperimentLedger(args.ledger)
    try:
        if args.stage == "status":
            if args.execute or args.wait_seconds:
                raise QuantError("Status cannot execute or wait for an order.")
            return ledger.status(config)
        if args.stage == "reconcile":
            if args.execute or args.wait_seconds:
                raise QuantError("Reconciliation cannot submit an actual order.")
            ledger.verified_authorization(config, require_engine=False)
            with IBExperimentBroker(config, authorized=False) as broker:
                permission = broker.order_permission()
                account = broker.account()
            if not args.confirm_rejected_no_fill:
                return {
                    "mode": "read_only_reconciliation_check",
                    "account": account.public_dict(),
                    "permission": permission,
                    "ledger_changed": False,
                    "orders_sent": 0,
                }
            if not args.experiment_id:
                raise QuantError("Choose the retained experiment explicitly for reconciliation.")
            return ledger.reconcile_rejection(
                config,
                account,
                permission,
                args.experiment_id,
                accept_code_update=args.accept_code_update,
                legacy_receipt=args.legacy_rejection_receipt,
                legacy_private=args.legacy_private_receipt,
            )
        ledger.verified_authorization(config)
        wait_for_regular_window(args.wait_seconds)
        with IBExperimentBroker(config, authorized=args.execute) as broker:
            return run_smoke(config, ledger, broker, execute=args.execute)
    finally:
        ledger.close()
