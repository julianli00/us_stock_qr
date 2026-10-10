from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import rebalance
from us_quant.calendar import completed_session, is_month_end, market_calendar, next_session
from us_quant.config import QuantError, ResearchConfig
from us_quant.data import MarketData, fetch_dataset, load_market
from us_quant.metrics import acceptance, performance
from us_quant.research import verify_freeze
from us_quant.storage import digest_json, file_digest, read_json, utc_now
from us_quant.strategy import target_weights


def engine_fingerprint() -> str:
    return file_digest(Path(__file__))


class ShadowLedger:
    """Prospective reference accounting only; this module has no broker interface."""

    def __init__(self, path: Path, *, create: bool = False):
        if create and path.exists():
            raise QuantError("Refusing to replace an existing prospective ledger.")
        if not create and not path.is_file():
            raise QuantError("Initialize the prospective ledger explicitly before updating it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if create:
            path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
        self.connection = sqlite3.connect(path, timeout=5)
        if create:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE registration (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1), metadata_json TEXT NOT NULL
                );
                CREATE TABLE observations (
                  session_date TEXT PRIMARY KEY, observed_at TEXT NOT NULL,
                  observation_json TEXT NOT NULL
                );
                CREATE TABLE decisions (
                  signal_session TEXT PRIMARY KEY, execution_session TEXT UNIQUE NOT NULL,
                  created_at TEXT NOT NULL, weights_json TEXT NOT NULL
                );
                """
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def metadata(self) -> dict:
        row = self.connection.execute(
            "SELECT metadata_json FROM registration WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise QuantError("Prospective registration is absent; do not infer a historical start.")
        return json.loads(row[0])

    def verify(self, config: ResearchConfig, freeze: Path) -> dict:
        metadata = self.metadata()
        if (
            metadata["protocol_sha256"] != digest_json(config.to_dict())
            or metadata["freeze_sha256"] != file_digest(freeze)
            or metadata["engine_sha256"] != engine_fingerprint()
        ):
            raise QuantError("Prospective strategy, protocol, or accounting code changed.")
        if metadata.get("mode") != "prospective_shadow_reference_only":
            raise QuantError("This ledger cannot represent a broker account.")
        return metadata

    def _decision(
        self,
        day: pd.Timestamp,
        now: pd.Timestamp,
        data: MarketData,
        config: ResearchConfig,
        candidate_id: str | None,
    ) -> None:
        if not is_month_end(day):
            return
        execution = next_session(day)
        if now >= market_calendar().session_open(execution):
            raise QuantError("Missed decision deadline: do not backfill a signal after its open.")
        weights = target_weights(data.close.loc[:day], config.candidate(candidate_id), config)
        self.connection.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?)",
            (
                day.date().isoformat(),
                execution.date().isoformat(),
                now.isoformat(),
                json.dumps({key: float(value) for key, value in weights.items()}, sort_keys=True),
            ),
        )

    @staticmethod
    def _require_fresh(data: MarketData, now: pd.Timestamp, config: ResearchConfig) -> pd.Timestamp:
        data.validate()
        if tuple(data.close.columns) != config.symbols or config.execution_delay_sessions != 1:
            raise QuantError(
                "Shadow accounting requires the exact universe and next-open execution."
            )
        day = data.close.index[-1]
        if day != completed_session(now):
            raise QuantError("Prospective recording requires the latest fully completed session.")
        if now >= market_calendar().session_open(next_session(day)):
            raise QuantError("Record the close before the next open; no retrospective decisions.")
        return day

    def initialize(
        self,
        config: ResearchConfig,
        frozen: dict,
        freeze: Path,
        data: MarketData,
        snapshot_hash: str,
        now: pd.Timestamp,
    ) -> dict:
        day = self._require_fresh(data, now, config)
        metadata = {
            "mode": "prospective_shadow_reference_only",
            "registered_at": now.isoformat(),
            "baseline_session": day.date().isoformat(),
            "first_forward_session": next_session(day).date().isoformat(),
            "strategy_id": frozen["selected_candidate_id"],
            "protocol_sha256": digest_json(config.to_dict()),
            "freeze_sha256": file_digest(freeze),
            "engine_sha256": engine_fingerprint(),
            "initial_capital": config.initial_capital,
            "research_qualified": False,
            "broker_connected": False,
            "order_authority": False,
            "objective_verified": False,
            "warning": (
                "Failed historical baseline, retained only as a reference. "
                "Modeled next-open fills are not IBKR or exchange executions."
            ),
        }
        initial = {
            "session_date": day.date().isoformat(),
            "observed_at": now.isoformat(),
            "equity": config.initial_capital,
            "return": 0.0,
            "cash": config.initial_capital,
            "asset_values": {symbol: 0.0 for symbol in config.symbols},
            "benchmarks": {
                symbol: {
                    "equity": config.initial_capital,
                    "cash": config.initial_capital,
                    "asset_value": 0.0,
                    "return": 0.0,
                }
                for symbol in (config.primary_benchmark, config.secondary_benchmark)
            },
            "raw_close": {key: float(value) for key, value in data.raw_close.iloc[-1].items()},
            "risk_free": float(data.risk_free.iloc[-1]),
            "snapshot_manifest_sha256": snapshot_hash,
            "cost": 0.0,
            "turnover": 0.0,
            "synthetic_order_tickets": 0,
            "broker_orders_sent": 0,
            "baseline_only": True,
        }
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.connection.execute("SELECT COUNT(*) FROM registration").fetchone()[0]:
                raise QuantError("Prospective registration already exists.")
            self.connection.execute(
                "INSERT INTO registration VALUES (1, ?)", (json.dumps(metadata, sort_keys=True),)
            )
            self.connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?)",
                (initial["session_date"], now.isoformat(), json.dumps(initial, sort_keys=True)),
            )
            self._decision(day, now, data, config, metadata["strategy_id"])
        return self.status(config)

    def update(
        self,
        config: ResearchConfig,
        freeze: Path,
        data: MarketData,
        snapshot_hash: str,
        now: pd.Timestamp,
    ) -> dict:
        metadata = self.verify(config, freeze)
        day = self._require_fresh(data, now, config)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT session_date, observed_at, observation_json FROM observations "
                "ORDER BY session_date DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise QuantError("The prospective baseline observation is missing.")
            last_day, observed_at, raw = row
            if now < pd.Timestamp(observed_at):
                raise QuantError("Prospective observation time moved backwards.")
            prior = json.loads(raw)
            if day == pd.Timestamp(last_day):
                return self.status(config)
            if day != next_session(pd.Timestamp(last_day)):
                raise QuantError(
                    "Missed forward session: never backfill unobserved trading decisions."
                )
            prior_prices = data.raw_close.loc[last_day, list(config.symbols)].to_numpy()
            expected_prices = np.array([prior["raw_close"][symbol] for symbol in config.symbols])
            if not np.allclose(prior_prices, expected_prices, rtol=1e-6, atol=1e-6):
                raise QuantError(
                    "Source revised the prior raw close; audit before appending evidence."
                )
            previous_close = data.close.loc[last_day, list(config.symbols)].to_numpy()
            opening = data.open.loc[day, list(config.symbols)].to_numpy()
            closing = data.close.loc[day, list(config.symbols)].to_numpy()
            assets = np.array([prior["asset_values"][symbol] for symbol in config.symbols])
            assets = assets * opening / previous_close
            cash = float(prior["cash"])
            cost = turnover = 0.0
            tickets = 0
            decision = self.connection.execute(
                "SELECT signal_session, created_at, weights_json FROM decisions "
                "WHERE execution_session=?",
                (day.date().isoformat(),),
            ).fetchone()
            if decision is not None:
                signal_day, created_at, weights_json = decision
                if (
                    signal_day != last_day
                    or pd.Timestamp(created_at) >= market_calendar().session_open(day)
                    or pd.Timestamp(created_at) < pd.Timestamp(metadata["registered_at"])
                ):
                    raise QuantError("Prospective decision does not predate execution correctly.")
                weights = json.loads(weights_json)
                target = np.array([weights[symbol] for symbol in config.symbols])
                if (
                    not np.isfinite(target).all()
                    or (target < 0).any()
                    or target.sum() > 1 - config.cash_reserve + 1e-10
                ):
                    raise QuantError("Invalid prospective target allocation.")
                assets, cash, cost, turnover, tickets = rebalance(
                    assets, cash, target, config.cost_bps_per_side, config.commission_per_order
                )
            assets = assets * closing / opening
            equity = float(assets.sum() + cash)
            benchmarks = {}
            for symbol, position in prior["benchmarks"].items():
                location = config.symbols.index(symbol)
                value = position["asset_value"] * opening[location] / previous_close[location]
                benchmark_cash = position["cash"]
                if day.date().isoformat() == metadata["first_forward_session"]:
                    invested, benchmark_cash, _, _, _ = rebalance(
                        np.array([value]),
                        benchmark_cash,
                        np.array([1.0]),
                        config.cost_bps_per_side,
                        config.commission_per_order,
                    )
                    value = float(invested[0])
                value *= closing[location] / opening[location]
                benchmark_equity = float(value + benchmark_cash)
                benchmarks[symbol] = {
                    "equity": benchmark_equity,
                    "cash": benchmark_cash,
                    "asset_value": float(value),
                    "return": benchmark_equity / position["equity"] - 1,
                }
            observation = {
                "session_date": day.date().isoformat(),
                "observed_at": now.isoformat(),
                "equity": equity,
                "return": equity / prior["equity"] - 1,
                "cash": cash,
                "asset_values": dict(zip(config.symbols, (float(x) for x in assets), strict=True)),
                "benchmarks": benchmarks,
                "raw_close": {key: float(value) for key, value in data.raw_close.loc[day].items()},
                "risk_free": float(data.risk_free.loc[day]),
                "snapshot_manifest_sha256": snapshot_hash,
                "cost": cost,
                "turnover": turnover,
                "synthetic_order_tickets": tickets,
                "broker_orders_sent": 0,
                "baseline_only": False,
            }
            if not np.isfinite([equity, cash]).all() or equity <= 0 or cash < -1e-7:
                raise QuantError("Prospective accounting became nonfinite or insolvent.")
            self._decision(day, now, data, config, metadata["strategy_id"])
            self.connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?)",
                (
                    observation["session_date"],
                    now.isoformat(),
                    json.dumps(observation, sort_keys=True, allow_nan=False),
                ),
            )
        return self.status(config)

    def status(self, config: ResearchConfig) -> dict:
        metadata = self.metadata()
        if metadata["protocol_sha256"] != digest_json(config.to_dict()):
            raise QuantError("Status must use the original prospective configuration.")
        observations = [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT observation_json FROM observations ORDER BY session_date"
            ).fetchall()
        ]
        forward = [row for row in observations if not row["baseline_only"]]
        status = {
            "mode": metadata["mode"],
            "registered_at": metadata["registered_at"],
            "strategy_id": metadata["strategy_id"],
            "baseline_session": metadata["baseline_session"],
            "first_forward_session": metadata["first_forward_session"],
            "last_observed_session": observations[-1]["session_date"],
            "prospective_shadow_sessions": len(forward),
            "forward_paper_sessions": 0,
            "broker_orders_sent": 0,
            "objective_verified": False,
            "paper_submission_eligible": False,
            "minimum_observation_sessions": config.targets.minimum_forward_paper_sessions,
            "enough_forward_observations": len(forward)
            >= config.targets.minimum_forward_paper_sessions,
            "warning": metadata["warning"],
        }
        if len(forward) >= 1:
            nav = np.array([config.initial_capital, *[row["equity"] for row in forward]])
            status["observed_total_return"] = float(nav[-1] / nav[0] - 1)
            status["observed_max_drawdown"] = float(-(nav / np.maximum.accumulate(nav) - 1).min())
        minimum = config.targets.minimum_forward_paper_sessions
        status["annualized_metrics_withheld_for_short_history"] = len(forward) < max(minimum, 2)
        if len(forward) >= max(minimum, 2):
            dates = pd.to_datetime([row["session_date"] for row in forward])
            returns = pd.Series([row["return"] for row in forward], index=dates)
            risk_free = pd.Series([row["risk_free"] for row in forward], index=dates)
            metrics = performance(returns, risk_free)
            benchmarks = {
                symbol: performance(
                    pd.Series(
                        [row["benchmarks"][symbol]["return"] for row in forward], index=dates
                    ),
                    risk_free,
                )
                for symbol in (config.primary_benchmark, config.secondary_benchmark)
            }
            status["descriptive_metrics_not_a_forecast"] = metrics
            status["benchmark_metrics"] = benchmarks
            status["numeric_checks_not_qualification"] = acceptance(
                metrics,
                benchmarks[config.primary_benchmark],
                replace(config, targets=replace(config.targets, minimum_holdout_sessions=minimum)),
            )
        return status


def add_parser(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "shadow", help="Read-only prospective research; never broker orders."
    )
    command.add_argument("stage", choices=["init", "tick", "status"])
    command.add_argument("--ledger", type=Path, default=Path("runtime/shadow.sqlite3"))
    command.add_argument("--data", type=Path)
    command.add_argument("--data-root", type=Path, default=Path("data/shadow"))
    command.add_argument("--development", type=Path, default=Path("data/development"))
    command.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))


def dispatch_shadow(args: argparse.Namespace, config: ResearchConfig) -> dict:
    frozen = verify_freeze(config, args.development, args.freeze)
    if args.stage == "status":
        ledger = ShadowLedger(args.ledger)
        try:
            ledger.verify(config, args.freeze)
            return ledger.status(config)
        finally:
            ledger.close()
    now = pd.Timestamp.now(tz="UTC")
    snapshot = args.data or args.data_root / completed_session(now).date().isoformat()
    if not snapshot.exists():
        fetch_dataset(config, "forward", snapshot)
    data = load_market(config, snapshot, phase="forward")
    manifest = read_json(snapshot / "manifest.json")
    if pd.Timestamp(manifest["retrieved_at"]) > now:
        now = pd.Timestamp.now(tz="UTC")
    ShadowLedger._require_fresh(data, now, config)
    ledger = ShadowLedger(args.ledger, create=args.stage == "init")
    try:
        if args.stage == "init":
            result = ledger.initialize(
                config, frozen, args.freeze, data, file_digest(snapshot / "manifest.json"), now
            )
        else:
            result = ledger.update(
                config, args.freeze, data, file_digest(snapshot / "manifest.json"), now
            )
        return {**result, "checked_at": utc_now()}
    finally:
        ledger.close()
