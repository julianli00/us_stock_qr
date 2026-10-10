from __future__ import annotations

import argparse
import fcntl
import inspect
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from us_quant.bt_audit import independent_equity
from us_quant.calendar import market_calendar, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.metrics import performance
from us_quant.prospective_data import utc_now
from us_quant.prospective_target_observations import TargetJournal
from us_quant.research_program import safe_file
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/prospective-research-accounts.json"
ACCOUNT_IDS = ("strategy_base", "spy_base", "strategy_stress", "spy_stress")
SYMBOLS = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")


def validate_policy(policy: dict) -> None:
    constants = {
        "schema_version": 1,
        "experiment_id": "prospective_rejected_candidate_accounts_v1",
        "source_candidate_id": "macro_real_yield_factor_tilt",
        "source_historical_status": "rejected_historical",
        "target_policy": "config/prospective-target-observations.json",
        "target_journal": "data/prospective-target-observations-v1",
        "symbols": list(SYMBOLS),
        "capital_usd_per_account": 10000.0,
        "commission_per_order": 1.0,
        "minimum_observed_sessions_for_reported_sharpe": 63,
        "benchmark_symbol": "SPY",
        "price_encoding": "within_snapshot_ratio_chain",
        "scenarios": [
            {"id": "base", "cost_bps": 5.0, "delay_sessions": 1},
            {"id": "stress", "cost_bps": 20.0, "delay_sessions": 2},
        ],
    }
    if (
        any(policy.get(key) != value for key, value in constants.items())
        or any(
            policy.get(key) is not True
            for key in (
                "pause_on_unexplained_previous_raw_close_revision",
                "require_each_market_session_actual_input_receipt",
                "require_each_market_session_target_observation",
                "target_must_be_recorded_before_model_execution_open",
                "initial_capital_anchor_has_zero_return_and_no_interest",
            )
        )
        or any(
            policy.get(key) is not False
            for key in (
                "model_fills_are_broker_fills", "backfill_missing_observations",
                "automatic_live_deployment", "update_research_champion", "order_authority",
            )
        )
    ):
        raise QuantError("Prospective model accounts must preserve timing, costs and research-only scope.")


def aware(value) -> pd.Timestamp:
    try:
        observed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise QuantError("Research accounting requires a valid timezone-aware timestamp.") from exc
    if pd.isna(observed) or observed.tzinfo is None:
        raise QuantError("Research accounting requires a valid timezone-aware timestamp.")
    return observed


def chained_market(snapshots: dict[pd.Timestamp, MarketData]) -> MarketData:
    dates = pd.DatetimeIndex(sorted(snapshots))
    if len(dates) < 2 or not dates.equals(sessions(dates[0], dates[-1])):
        raise QuantError("Do not backfill a missing observed market session from later history.")
    columns = pd.Index(SYMBOLS)
    opening = pd.DataFrame(index=dates, columns=columns, dtype=float)
    closing = opening.copy()
    raw, volume = opening.copy(), opening.copy()
    rates = pd.Series(index=dates, name="risk_free", dtype=float)
    for i, day in enumerate(dates):
        data = snapshots[day]
        data.validate()
        if not set(columns) <= set(data.close.columns) or data.close.index[-1] != day:
            raise QuantError("Each model session needs its own complete contemporaneous input snapshot.")
        raw.loc[day] = data.raw_close.loc[day, columns]
        volume.loc[day] = data.volume.loc[day, columns]
        rates.loc[day] = data.risk_free.loc[day]
        if i == 0:
            opening.loc[day] = closing.loc[day] = 100.0
            continue
        prior = dates[i - 1]
        if prior not in data.close.index:
            raise QuantError("A daily return needs the previous close within the same actual snapshot.")
        if not np.allclose(
            data.raw_close.loc[prior, columns], raw.loc[prior], rtol=1e-9, atol=1e-6
        ):
            raise QuantError(
                "The previous raw close was revised; pause for source/corporate-action review."
            )
        denominator = data.close.loc[prior, columns]
        opening.loc[day] = closing.loc[prior] * data.open.loc[day, columns] / denominator
        closing.loc[day] = closing.loc[prior] * data.close.loc[day, columns] / denominator
    result = MarketData(opening, closing, raw, volume, rates)
    result.validate()
    return result


def target_signals(
    data: MarketData, instructions: list[dict], registration: dict, delay: int
) -> pd.DataFrame:
    signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    initial_day = pd.Timestamp(registration["baseline_session"])
    for instruction in instructions:
        day = pd.Timestamp(instruction["source_session"])
        if day < initial_day or day > data.close.index[-1]:
            continue
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if (
            aware(instruction["generated_at"]) >= market_calendar().session_open(execution)
            or aware(registration["registered_at"]) >= market_calendar().session_open(execution)
        ):
            raise QuantError("A model fill cannot use a target or experiment registered after that open.")
        target = instruction["weights"]
        if not isinstance(target, dict) or set(target) != set(SYMBOLS):
            raise QuantError("A model target must retain the complete frozen fund universe.")
        values = pd.Series(target, dtype=float).reindex(data.close.columns)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 0.98, rtol=0, atol=1e-12)
            or not signals.loc[day].isna().all()
        ):
            raise QuantError("A model target is invalid or would overwrite another dated instruction.")
        signals.loc[day] = values
    if signals.loc[initial_day].isna().any():
        raise QuantError("The prospective experiment requires its frozen initial target.")
    return signals


def calculate(
    data: MarketData, instructions: list[dict], registration: dict, policy: dict
) -> tuple[dict, dict]:
    validate_policy(policy)
    first, last = str(data.close.index[0].date()), str(data.close.index[-1].date())
    accounts, errors = {}, {}
    for scenario in policy["scenarios"]:
        strategy = target_signals(data, instructions, registration, scenario["delay_sessions"])
        spy = strategy.copy() * np.nan
        spy.loc[data.close.index[0]] = 0.0
        spy.loc[data.close.index[0], "SPY"] = 1.0
        for name, signals in (("strategy", strategy), ("spy", spy)):
            own = simulate(
                data, signals, first, last,
                initial_capital=policy["capital_usd_per_account"],
                cost_bps=scenario["cost_bps"],
                commission=policy["commission_per_order"],
                delay=scenario["delay_sessions"],
            )
            independent = independent_equity(
                data, signals, first, last,
                capital=policy["capital_usd_per_account"],
                cost_bps=scenario["cost_bps"],
                commission=policy["commission_per_order"],
                delay=scenario["delay_sessions"],
            )
            error = float(abs(own.frame["equity"] - independent["equity"]).max())
            if error > policy["capital_usd_per_account"] * 1e-8 or not np.allclose(
                own.frame["return"], independent["return"], rtol=0, atol=1e-10
            ):
                raise QuantError("Prospective accounting does not agree with the independent bt path.")
            identifier = f"{name}_{scenario['id']}"
            accounts[identifier] = own
            errors[identifier] = error
    return accounts, errors


class ResearchAccounts:
    def __init__(self, directory: Path, policy: dict, *, root: Path = ROOT):
        validate_policy(policy)
        directory = directory if directory.is_absolute() else root / directory
        target_directory = root / policy["target_journal"]
        parent_directory = root / "data/prospective-market-v1"
        if (
            directory.is_symlink()
            or not directory.resolve().is_relative_to(root.resolve())
            or any(
                directory.resolve().is_relative_to(path.resolve())
                or path.resolve().is_relative_to(directory.resolve())
                for path in (target_directory, parent_directory)
            )
        ):
            raise QuantError("Use a separate, explicitly named prospective research-account directory.")
        self.root, self.directory, self.policy = root, directory, policy
        self.targets = TargetJournal(
            target_directory, read_json(safe_file(root, policy["target_policy"])), root=root
        )
        self.parent = self.targets.parent

    def fingerprint(self) -> str:
        folder = Path(__file__).resolve().parent
        return digest_json(
            {
                "prospective_accounts": file_digest(Path(__file__)),
                "observer_dependencies": self.targets.fingerprint(),
                "funded_accounting": file_digest(folder / "cash_funded_accounting_v2.py"),
                "original_rebalance": file_digest(folder / "backtest.py"),
                "independent_accounting": file_digest(folder / "bt_audit.py"),
                "metrics": file_digest(folder / "metrics.py"),
                "path_validation": digest_json(inspect.getsource(safe_file)),
            }
        )

    def initialize(self, now=None) -> dict:
        now = aware(utc_now() if now is None else now)
        targets = self.targets.verify()
        if not targets:
            raise QuantError("Record the actual initial target before registering forward model accounts.")
        initial = targets[-1]
        manifest = read_json(
            safe_file(self.targets.directory, f"{initial['snapshot_path']}/manifest.json", initial["manifest_sha256"])
        )
        if (
            manifest["new_target_generated"] is not True
            or aware(initial["observed_at"]) > now
            or now >= market_calendar().session_open(next_session(pd.Timestamp(initial["session"])))
        ):
            raise QuantError("Register this experiment before the initial target's first future open.")
        if self.directory.exists():
            raise QuantError("Do not reset or replace an existing prospective research experiment.")
        self.directory.mkdir(parents=True, mode=0o700)
        registration = {
            "schema_version": 1,
            "registered_at": now.isoformat(),
            "policy_sha256": digest_json(self.policy),
            "accounting_sha256": self.fingerprint(),
            "baseline_session": initial["session"],
            "first_model_session": str(next_session(pd.Timestamp(initial["session"])).date()),
            "initial_target_receipt_sha256": file_digest(
                self.targets.directory / "receipts" / f"{initial['session']}.json"
            ),
            "target_journal_registration_sha256": file_digest(self.targets.directory / "registration.json"),
            "parent_registration_sha256": file_digest(self.parent.directory / "registration.json"),
            "mode": "prospective_research_simulation_only",
            "source_historical_status": "rejected_historical",
            "actual_orders_or_broker_account": False,
            "research_champion_updated": False,
            "order_authority": False,
        }
        write_json(self.directory / "registration.json", registration)
        write_json(self.directory / "head.json", {"last_session": None, "receipt_sha256": "0" * 64})
        self.verify()
        return registration

    def verify(self) -> list[dict]:
        self.targets.verify()
        registration = read_json(safe_file(self.directory, "registration.json"))
        if (
            registration.get("policy_sha256") != digest_json(self.policy)
            or registration.get("accounting_sha256") != self.fingerprint()
            or registration.get("target_journal_registration_sha256")
            != file_digest(self.targets.directory / "registration.json")
            or registration.get("parent_registration_sha256")
            != file_digest(self.parent.directory / "registration.json")
            or registration.get("source_historical_status") != "rejected_historical"
            or registration.get("mode") != "prospective_research_simulation_only"
            or registration.get("actual_orders_or_broker_account") is not False
            or registration.get("research_champion_updated") is not False
            or registration.get("order_authority") is not False
        ):
            raise QuantError("The prospective experiment or its frozen dependencies changed.")
        baseline = pd.Timestamp(registration["baseline_session"])
        if (
            pd.isna(baseline)
            or baseline.tzinfo is not None
            or not market_calendar().is_session(baseline)
            or registration["first_model_session"] != str(next_session(baseline).date())
            or aware(registration["registered_at"])
            >= market_calendar().session_open(next_session(baseline))
        ):
            raise QuantError("The registered prospective start or actual registration time changed.")
        initial_path = safe_file(
            self.targets.directory, f"receipts/{registration['baseline_session']}.json",
            registration["initial_target_receipt_sha256"],
        )
        initial = read_json(initial_path)
        if aware(initial["observed_at"]) > aware(registration["registered_at"]):
            raise QuantError("The initial target must actually exist before experiment registration.")
        rows, previous = [], "0" * 64
        expected = pd.Timestamp(registration["first_model_session"])
        last_processed = aware(registration["registered_at"])
        previous_equity = {identifier: 10000.0 for identifier in ACCOUNT_IDS}
        for path in sorted((self.directory / "receipts").glob("*.json")):
            record = read_json(safe_file(self.directory, path.relative_to(self.directory).as_posix()))
            if (
                record.get("session") != str(expected.date())
                or path.stem != record["session"]
                or record.get("previous_receipt_sha256") != previous
                or record.get("registration_sha256") != file_digest(self.directory / "registration.json")
                or aware(record["processed_at"]) < last_processed
                or set(record.get("accounts", {})) != set(ACCOUNT_IDS)
                or record.get("modelled_fills_are_actual_orders") is not False
                or record.get("source_strategy_historically_qualified") is not False
                or record.get("order_authority") is not False
            ):
                raise QuantError("Prospective account history is changed, missing or out of sequence.")
            for identifier, account in record["accounts"].items():
                required = {
                    "equity", "return", "cash", "gross_exposure",
                    "turnover", "cost", "orders", "risk_free",
                }
                if (
                    set(account) != required
                    or not np.isfinite(list(account.values())).all()
                    or account["equity"] <= 0
                    or account["return"] <= -1
                    or min(account["cash"], account["turnover"], account["cost"], account["orders"]) < 0
                    or account["orders"] != int(account["orders"])
                    or not np.isclose(
                        account["return"], account["equity"] / previous_equity[identifier] - 1,
                        rtol=0, atol=1e-10,
                    )
                ):
                    raise QuantError("A recorded prospective account violates its funding or return identity.")
                previous_equity[identifier] = account["equity"]
            for kind, archive in (("input_receipt", self.parent), ("target_receipt", self.targets)):
                reference = record[kind]
                receipt = read_json(safe_file(
                    archive.directory, f"receipts/{record['session']}.json", reference["sha256"]
                ))
                if (
                    receipt["manifest_sha256"] != reference["manifest_sha256"]
                    or receipt["observed_at"] != reference["observed_at"]
                    or aware(receipt["observed_at"]) > aware(record["processed_at"])
                ):
                    raise QuantError("An account record does not match its actual observed inputs.")
            rows.append(record)
            previous, last_processed = file_digest(path), aware(record["processed_at"])
            expected = next_session(expected)
        head = read_json(safe_file(self.directory, "head.json"))
        if head != {"last_session": rows[-1]["session"] if rows else None, "receipt_sha256": previous}:
            raise QuantError("The prospective account receipt head is missing or partially committed.")
        return rows

    def snapshot(self, receipt: dict) -> MarketData:
        manifest = read_json(safe_file(
            self.parent.directory, f"{receipt['snapshot_path']}/manifest.json",
            receipt["manifest_sha256"],
        ))
        folder = self.parent.directory / receipt["snapshot_path"]
        for name, digest in manifest["files"].items():
            safe_file(folder, name, digest)
        panels = {
            name: pd.read_csv(folder / f"{name}.csv", index_col=0, parse_dates=True)
            for name in ("open", "close", "raw_close", "volume", "risk_free")
        }
        return MarketData(
            panels["open"], panels["close"], panels["raw_close"], panels["volume"],
            panels["risk_free"]["risk_free"],
        )

    def replay(self, end: str, now: pd.Timestamp) -> tuple[dict, dict, dict, dict]:
        registration = read_json(self.directory / "registration.json")
        dates = sessions(registration["baseline_session"], end)
        parents = {record["session"]: record for record in self.parent.verify()}
        targets = {record["session"]: record for record in self.targets.verify()}
        snapshots, instructions = {}, []
        for day in dates:
            key = str(day.date())
            if key not in parents or key not in targets:
                raise QuantError(
                    f"Missing actual input/target observation for {key}; no retrospective backfill allowed."
                )
            if max(aware(parents[key]["observed_at"]), aware(targets[key]["observed_at"])) > now:
                raise QuantError("Model accounting cannot process inputs before their actual acquisition.")
            snapshots[day] = self.snapshot(parents[key])
            target_record = targets[key]
            manifest = read_json(safe_file(
                self.targets.directory, f"{target_record['snapshot_path']}/manifest.json",
                target_record["manifest_sha256"],
            ))
            if manifest["new_target_generated"]:
                target = read_json(safe_file(
                    self.targets.directory, f"{target_record['snapshot_path']}/target.json",
                    manifest["files"]["target.json"],
                ))
                instructions.append({
                    "source_session": key, "generated_at": target_record["observed_at"],
                    "weights": target["weights"],
                })
        accounts, errors = calculate(chained_market(snapshots), instructions, registration, self.policy)
        return accounts, errors, parents, targets

    def advance(self, now=None) -> dict:
        now = aware(utc_now() if now is None else now)
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self.verify()
            registration = read_json(self.directory / "registration.json")
            if now < aware(registration["registered_at"]):
                raise QuantError("Do not backdate prospective model processing.")
            if records and now < aware(records[-1]["processed_at"]):
                raise QuantError("Prospective processing cannot precede its already recorded history.")
            latest = self.parent.verify()[-1]["session"]
            prior = records[-1]["session"] if records else registration["baseline_session"]
            if latest <= prior:
                return {"action": "no_new_observed_session", **self.status()}
            attempt = self.directory / "attempts" / (
                now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
            )
            attempt.mkdir(parents=True, mode=0o700)
            try:
                accounts, errors, parents, targets = self.replay(latest, now)
                for record in records:
                    for identifier, account in accounts.items():
                        observed = account.frame.loc[record["session"]]
                        expected = record["accounts"][identifier]
                        for name in (
                            "equity", "return", "cash", "gross_exposure",
                            "cost", "turnover", "orders", "risk_free",
                        ):
                            if not np.isclose(observed[name], expected[name], rtol=0, atol=1e-8):
                                raise QuantError("A prior model account changed on frozen-source replay.")
            except QuantError as exc:
                write_json(attempt / "failure.json", {
                    "processed_at": now.isoformat(),
                    "attempted_last_session": latest,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "account_rows_committed": False,
                    "order_authority": False,
                })
                raise
            write_json(attempt / "independent-audit.json", {
                "processed_at": now.isoformat(),
                "last_session": latest,
                "maximum_equity_error_usd": errors,
                "previous_recorded_accounts_unchanged": True,
                "modelled_fills_are_actual_orders": False,
                "order_authority": False,
            })
            previous = read_json(self.directory / "head.json")["receipt_sha256"]
            for day in sessions(prior, latest)[1:]:
                key = str(day.date())
                record = {
                    "session": key,
                    "processed_at": now.isoformat(),
                    "registration_sha256": file_digest(self.directory / "registration.json"),
                    "previous_receipt_sha256": previous,
                    "accounts": {},
                    "independent_bt_maximum_equity_error_usd": errors,
                    "modelled_fills_are_actual_orders": False,
                    "source_strategy_historically_qualified": False,
                    "order_authority": False,
                }
                for kind, archive, source in (
                    ("input_receipt", self.parent, parents[key]),
                    ("target_receipt", self.targets, targets[key]),
                ):
                    record[kind] = {
                        "sha256": file_digest(archive.directory / "receipts" / f"{key}.json"),
                        "manifest_sha256": source["manifest_sha256"],
                        "observed_at": source["observed_at"],
                    }
                for identifier, account in accounts.items():
                    values = account.frame.loc[day]
                    record["accounts"][identifier] = {
                        name: float(values[name])
                        for name in (
                            "equity", "return", "cash", "gross_exposure",
                            "turnover", "cost", "orders", "risk_free",
                        )
                    }
                path = self.directory / "receipts" / f"{key}.json"
                if path.exists():
                    raise QuantError("Do not overwrite an existing prospective model session.")
                write_json(path, record)
                previous = file_digest(path)
                write_json(self.directory / "head.json", {"last_session": key, "receipt_sha256": previous})
            return {"action": "advanced", **self.status()}

    def status(self) -> dict:
        rows = self.verify()
        summaries = {}
        for identifier in ACCOUNT_IDS:
            if rows:
                frame = pd.DataFrame(
                    [row["accounts"][identifier] for row in rows],
                    index=pd.to_datetime([row["session"] for row in rows]),
                )
                metrics = performance(frame["return"], frame["risk_free"]) if len(frame) >= 2 else None
                summaries[identifier] = {
                    "equity_usd": float(frame["equity"].iloc[-1]),
                    "cost_paid_usd": float(frame["cost"].sum()),
                    "model_order_tickets": int(frame["orders"].sum()),
                    "total_return": float(frame["equity"].iloc[-1] / 10000 - 1),
                    "sharpe": metrics["sharpe"] if len(frame) >= 63 else None,
                    "cagr": metrics["cagr"] if len(frame) >= 63 else None,
                    "max_drawdown": float(
                        -np.min(
                            np.r_[10000.0, frame["equity"].to_numpy()]
                            / np.maximum.accumulate(np.r_[10000.0, frame["equity"].to_numpy()])
                            - 1
                        )
                    ),
                }
            else:
                summaries[identifier] = {
                    "equity_usd": 10000.0, "cost_paid_usd": 0.0,
                    "model_order_tickets": 0, "total_return": None, "sharpe": None,
                    "cagr": None, "max_drawdown": None,
                }
        registration = read_json(self.directory / "registration.json")
        failures = sorted((self.directory / "attempts").glob("*/failure.json"))
        return {
            "mode": "prospective_research_simulation_only",
            "source_candidate_id": self.policy["source_candidate_id"],
            "source_historical_status": "rejected_historical",
            "first_model_session": registration["first_model_session"],
            "observed_model_sessions": len(rows),
            "latest_model_session": rows[-1]["session"] if rows else None,
            "minimum_sessions_before_sharpe_report": 63,
            "accounts": summaries,
            "failed_advancement_attempts": len(failures),
            "latest_failure": (
                read_json(safe_file(self.directory, failures[-1].relative_to(self.directory).as_posix()))
                if failures else None
            ),
            "receipt_chain_head": read_json(self.directory / "head.json")["receipt_sha256"],
            "actual_orders_or_broker_account": False,
            "old_paused_forward_ledgers_modified": False,
            "new_historical_strategy_evaluations": 0,
            "research_champion_updated": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Prospective research model accounts, never broker orders.")
    parser.add_argument("action", choices=("init", "advance", "status"))
    parser.add_argument("--directory", type=Path, default=ROOT / "data/prospective-research-accounts-v1")
    args = parser.parse_args()
    try:
        study = ResearchAccounts(args.directory, read_json(POLICY))
        if args.action == "init":
            result = study.initialize()
        elif args.action == "advance":
            result = study.advance()
        else:
            result = study.status()
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, ValueError) as exc:
        parser.exit(2, f"Prospective model accounting blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
