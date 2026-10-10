from __future__ import annotations

import argparse
import fcntl
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import simulate
from us_quant.calendar import (
    completed_session,
    is_month_end,
    market_calendar,
    next_session,
    sessions,
)
from us_quant.config import QuantError, ResearchConfig
from us_quant.data import MarketData, fetch_dataset, load_market, verify_dataset
from us_quant.metrics import acceptance, block_bootstrap, performance
from us_quant.research import verify_freeze
from us_quant.storage import (
    digest_json,
    file_digest,
    implementation_fingerprint,
    read_json,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals, monthly_signals, target_weights

ACTIVE_STATES = ("awaiting_baseline", "observing")
GENE_FIELDS = frozenset(
    {"momentum_lookbacks", "trend_lookback", "volatility_lookback", "target_volatility", "top_k"}
)


def fingerprint() -> str:
    return digest_json(
        {
            "engine": file_digest(Path(__file__)),
            "frozen_research_core": implementation_fingerprint(),
        }
    )


def validate_genome(genome: dict) -> None:
    if not isinstance(genome, dict) or set(genome) != GENE_FIELDS:
        raise QuantError("Evolution may mutate only the registered signal parameters.")
    windows = genome["momentum_lookbacks"]
    if (
        not isinstance(windows, list)
        or len(windows) != 3
        or any(type(value) is not int or not 2 <= value <= 252 for value in windows)
        or windows != sorted(set(windows))
    ):
        raise QuantError("Momentum windows must be three unique increasing causal lookbacks.")
    for field in ("trend_lookback", "volatility_lookback"):
        if type(genome[field]) is not int or not 20 <= genome[field] <= 252:
            raise QuantError(f"Invalid bounded evolution window: {field}")
    if (
        type(genome["top_k"]) is not int
        or genome["top_k"] not in {3, 4}
        or type(genome["target_volatility"]) not in (int, float)
        or not math.isfinite(genome["target_volatility"])
        or not 0.10 <= genome["target_volatility"] <= 0.15
    ):
        raise QuantError(
            "Evolution cannot increase concentration or target volatility beyond policy."
        )


def validate_policy(policy: dict, base: ResearchConfig) -> None:
    required = {
        "schema_version",
        "seed_candidate",
        "capital_usd",
        "prior_trials",
        "max_candidates",
        "max_active_candidates",
        "proposal_interval_sessions",
        "minimum_forward_sessions",
        "max_cycle_sessions",
        "stop_after",
        "minimum_historical_score_improvement",
        "auto_order_submission",
        "parent_feedback",
        "mutations",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != required
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise QuantError("Invalid autonomous research policy schema.")
    limits = {
        "prior_trials": (14, 10000),
        "max_candidates": (1, 8),
        "max_active_candidates": (1, 3),
        "proposal_interval_sessions": (5, 252),
        "minimum_forward_sessions": (63, 252),
        "max_cycle_sessions": (1, 80),
    }
    if any(
        type(policy[key]) is not int or not low <= policy[key] <= high
        for key, (low, high) in limits.items()
    ):
        raise QuantError("Candidate, cadence, or observation budgets exceed bounded authorization.")
    if (
        policy["capital_usd"] != 10000.0
        or policy["auto_order_submission"] is not False
        or type(policy["minimum_historical_score_improvement"]) not in (int, float)
        or not math.isfinite(policy["minimum_historical_score_improvement"])
        or not 0.01 <= policy["minimum_historical_score_improvement"] <= 0.5
        or base.cash_reserve < 0.02
        or base.execution_delay_sessions != 1
        or base.targets.max_drawdown_at_most > 0.15
        or base.targets.cagr_strictly_above != 0.20
        or base.targets.sharpe_strictly_above != 1.0
        or base.primary_benchmark != "SPY"
        or policy["parent_feedback"] != "historical_plus_mature_forward_shortfall"
    ):
        raise QuantError("Evolution may not change capital, order authority, or risk safeguards.")
    try:
        stop = pd.Timestamp(policy["stop_after"])
    except (TypeError, ValueError) as exc:
        raise QuantError("Evolution stop_after must be an ISO date.") from exc
    if pd.isna(stop) or stop.tzinfo is not None or stop.date().isoformat() != policy["stop_after"]:
        raise QuantError("Evolution stop_after must be an ISO date.")
    seed = base.candidate(policy["seed_candidate"])
    if seed is None or seed.kind != "momentum" or seed.max_weight > 0.4 or seed.top_k != 3:
        raise QuantError("Evolution requires the registered diversified long-only momentum seed.")
    if not isinstance(policy["mutations"], list) or len(policy["mutations"]) != 7:
        raise QuantError("The first autonomous epoch contains exactly seven declared mutations.")
    identifiers = set()
    genome = seed_genome(base, policy)
    for mutation in policy["mutations"]:
        if (
            not isinstance(mutation, dict)
            or set(mutation) != {"id", "field", "value"}
            or not isinstance(mutation["id"], str)
            or not mutation["id"]
            or mutation["id"] in identifiers
            or not isinstance(mutation["field"], str)
            or mutation["field"] not in GENE_FIELDS
        ):
            raise QuantError("Invalid or duplicate mutation declaration.")
        identifiers.add(mutation["id"])
        validate_genome({**genome, mutation["field"]: mutation["value"]})


def seed_genome(base: ResearchConfig, policy: dict) -> dict:
    seed = base.candidate(policy["seed_candidate"])
    if seed is None:
        raise QuantError("Evolution seed is absent.")
    return {
        "momentum_lookbacks": list(base.momentum_lookbacks),
        "trend_lookback": base.trend_lookback,
        "volatility_lookback": base.volatility_lookback,
        "target_volatility": seed.target_volatility,
        "top_k": seed.top_k,
    }


def candidate_config(
    base: ResearchConfig, policy: dict, identifier: str, genome: dict
) -> ResearchConfig:
    validate_genome(genome)
    seed = base.candidate(policy["seed_candidate"])
    if seed is None:
        raise QuantError("The frozen seed candidate is absent.")
    candidate = replace(
        seed, id=identifier, top_k=genome["top_k"], target_volatility=genome["target_volatility"]
    )
    return replace(
        base,
        initial_capital=policy["capital_usd"],
        candidates=(candidate,),
        momentum_lookbacks=tuple(genome["momentum_lookbacks"]),
        trend_lookback=genome["trend_lookback"],
        volatility_lookback=genome["volatility_lookback"],
    )


def shortfall(metrics: dict, benchmark: dict, base: ResearchConfig) -> float:
    target = base.targets
    sharpe = metrics["sharpe"] if metrics["sharpe"] is not None else 0.0
    return float(
        max(0.0, target.cagr_strictly_above - metrics["cagr"]) / target.cagr_strictly_above
        + max(0.0, target.sharpe_strictly_above - sharpe) / target.sharpe_strictly_above
        + max(0.0, metrics["max_drawdown"] - target.max_drawdown_at_most)
        / target.max_drawdown_at_most
        + max(0.0, benchmark["cagr"] - metrics["cagr"]) / target.cagr_strictly_above
    )


def parent_priority(
    historical_score: float, forward: dict | None, base: ResearchConfig, minimum_sessions: int
) -> float:
    if forward is None or forward["forward_sessions"] < minimum_sessions:
        return historical_score
    if "metrics" not in forward or "benchmark" not in forward:
        raise QuantError("Mature forward feedback is missing its actual measured metrics.")
    return historical_score + shortfall(forward["metrics"], forward["benchmark"], base)


def historical_diagnostics(
    config: ResearchConfig, data: MarketData, identifier: str
) -> tuple[dict, dict[str, pd.DataFrame]]:
    signals = monthly_signals(data.close, config.candidate(identifier), config)
    records, frames, gaps = {}, {}, []
    for label, start, end in (
        ("development", config.simulation_start, config.development_end),
        ("reused_recent_history", config.holdout_start, config.as_of),
    ):
        benchmark = simulate(
            data,
            buy_and_hold_signals(data.close, config.primary_benchmark, start),
            start,
            end,
            initial_capital=config.initial_capital,
            cost_bps=config.cost_bps_per_side,
            commission=config.commission_per_order,
        )
        benchmark_metrics = performance(benchmark.frame["return"], benchmark.frame["risk_free"])
        for stressed in (False, True):
            name = label + ("_stress" if stressed else "")
            result = simulate(
                data,
                signals,
                start,
                end,
                initial_capital=config.initial_capital,
                cost_bps=config.stress.cost_bps_per_side if stressed else config.cost_bps_per_side,
                commission=config.commission_per_order,
                delay=config.execution_delay_sessions
                + (config.stress.extra_execution_delay_sessions if stressed else 0),
            )
            metrics = performance(result.frame["return"], result.frame["risk_free"])
            records[name] = {
                "strategy": metrics,
                "benchmark": benchmark_metrics,
                "gates": acceptance(metrics, benchmark_metrics, config),
                "total_modeled_cost": float(result.frame["cost"].sum()),
                "average_gross_exposure": float(result.frame["gross_exposure"].mean()),
            }
            gaps.append(shortfall(metrics, benchmark_metrics, config))
            frames[name] = result.frame
    return {
        "history_role": "previously_exposed_history_diagnostics_not_new_holdout",
        "periods": records,
        "objective_shortfall_score": float(np.mean(gaps)),
        "all_historical_numeric_gates_passed": all(
            all(value["gates"].values()) for value in records.values()
        ),
        "independent_forward_evidence": False,
        "paper_order_authority": False,
    }, frames


class EvolutionEngine:
    def __init__(self, path: Path, reports: Path, *, create: bool = False):
        if path.is_symlink() or (create and path.exists()):
            raise QuantError("Refusing to overwrite or follow an autonomous research ledger.")
        if not create and not path.is_file():
            raise QuantError("Initialize the bounded evolution epoch before running cycles.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if create:
            path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
        self.path, self.reports = path, reports
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.row_factory = sqlite3.Row
        if create:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
                CREATE TABLE candidates (
                  id TEXT PRIMARY KEY, genome_sha TEXT UNIQUE NOT NULL, genome TEXT NOT NULL,
                  parent_id TEXT, mutation_id TEXT UNIQUE NOT NULL,
                  created_cycle TEXT UNIQUE NOT NULL,
                  registered_at TEXT NOT NULL, baseline_session TEXT NOT NULL,
                  status TEXT NOT NULL, history TEXT, score REAL, forward_summary TEXT
                );
                CREATE TABLE decisions (
                  candidate_id TEXT NOT NULL, signal_date TEXT NOT NULL,
                  execution_date TEXT NOT NULL, created_at TEXT NOT NULL, weights TEXT NOT NULL,
                  PRIMARY KEY(candidate_id, signal_date)
                );
                CREATE TABLE observations (
                  candidate_id TEXT NOT NULL, session_date TEXT NOT NULL,
                  observed_at TEXT NOT NULL, body TEXT NOT NULL,
                  PRIMARY KEY(candidate_id, session_date)
                );
                CREATE TABLE cycles (
                  session_date TEXT PRIMARY KEY, snapshot_sha TEXT NOT NULL,
                  started_at TEXT NOT NULL, status TEXT NOT NULL, result TEXT
                );
                CREATE TABLE events (
                  id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL, kind TEXT NOT NULL,
                  candidate_id TEXT, body TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        self.connection.close()

    def event(self, now: pd.Timestamp, kind: str, candidate: str | None, body: dict) -> None:
        self.connection.execute(
            "INSERT INTO events (recorded_at,kind,candidate_id,body) VALUES (?,?,?,?)",
            (now.isoformat(), kind, candidate, json.dumps(body, sort_keys=True, allow_nan=False)),
        )

    def initialize(
        self, base: ResearchConfig, policy: dict, sources: dict, now: pd.Timestamp
    ) -> dict:
        validate_policy(policy, base)
        if now.tzinfo is None or now.date().isoformat() > policy["stop_after"]:
            raise QuantError("Evolution requires an unexpired deadline and timezone-aware clock.")
        now = now.tz_convert("UTC")
        body = {
            "registered_at": now.isoformat(),
            "base_protocol_sha": digest_json(base.to_dict()),
            "policy": policy,
            "policy_sha": digest_json(policy),
            "source_fingerprints": sources,
            "engine_sha": fingerprint(),
            "order_authority": False,
        }
        with self.connection:
            if self.connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]:
                raise QuantError("Do not reset a research epoch to erase trials or failures.")
            self.connection.execute("INSERT INTO metadata VALUES (1,?)", (json.dumps(body),))
            self.event(now, "epoch_registered", None, body)
        return self.status()

    def verify(
        self, base: ResearchConfig, policy: dict, sources: dict, *, require_engine: bool = True
    ) -> dict:
        row = self.connection.execute("SELECT body FROM metadata WHERE id=1").fetchone()
        if row is None:
            raise QuantError("Evolution registration is absent.")
        metadata = json.loads(row["body"])
        if (
            metadata["base_protocol_sha"] != digest_json(base.to_dict())
            or metadata["policy_sha"] != digest_json(policy)
            or metadata["source_fingerprints"] != sources
            or (require_engine and metadata["engine_sha"] != fingerprint())
        ):
            raise QuantError("Evolution rules, risk policy, code, or historical evidence changed.")
        validate_policy(policy, base)
        candidates = self.connection.execute("SELECT * FROM candidates").fetchall()
        registered = self.connection.execute(
            "SELECT candidate_id FROM events WHERE kind='candidate_registered_before_evaluation'"
        ).fetchall()
        if len(registered) != len(candidates) or {row["candidate_id"] for row in registered} != {
            row["id"] for row in candidates
        }:
            raise QuantError("Candidate history was removed or altered after registration.")
        for candidate in candidates:
            genome = json.loads(candidate["genome"])
            validate_genome(genome)
            if (
                candidate["genome_sha"] != digest_json(genome)
                or candidate["id"] != "evo_" + digest_json(genome)[:12]
            ):
                raise QuantError("A registered candidate's parameters changed.")
            if candidate["history"] is not None:
                report = json.loads(candidate["history"])
                directory = self.reports / "candidates" / candidate["id"]
                if read_json(directory / "history.json") != report:
                    raise QuantError("Published historical diagnostics changed.")
                for name, expected in report["equity_files"].items():
                    if file_digest(directory / name) != expected:
                        raise QuantError("A preserved candidate equity record changed.")
            observations = self.connection.execute(
                "SELECT COUNT(*) FROM observations WHERE candidate_id=?", (candidate["id"],)
            ).fetchone()[0]
            if candidate["forward_summary"] is not None:
                summary = json.loads(candidate["forward_summary"])
                if observations < 1 or summary["forward_sessions"] != observations - 1:
                    raise QuantError("Reported forward time does not match recorded observations.")
        return metadata

    def _write_status(self, result: dict) -> None:
        write_json(self.reports / "status.json", self.status())
        lines = [
            "# Autonomous research status",
            "",
            "**Research only: no order authority or verified investment objective.**",
            "",
            f"Cumulative candidate trials: {result['cumulative_trials']}.",
            "Previously viewed history is diagnostic, never relabeled as a new holdout.",
            "",
            "| Candidate | Parent | State | Historical shortfall | Forward sessions |",
            "|---|---|---|---:|---:|",
        ]
        for item in result["candidates"]:
            score = (
                "pending"
                if item["historical_shortfall"] is None
                else f"{item['historical_shortfall']:.3f}"
            )
            count = item["forward"]["forward_sessions"] if item["forward"] else 0
            lines.append(
                f"| {item['id']} | {item['parent_id'] or 'registered seed'} | "
                f"{item['state']} | {score} | {count} |"
            )
        lines.extend(
            [
                "",
                "Lower shortfall only prioritizes research; it is not a return forecast.",
                "Risk, cash reserve, capital, universe, costs, and next-open execution are locked.",
                "Missed or revised forward observations stop the cycle; no backfilling.",
                "Retired and rejected candidates remain in the ledger and total trial count.",
                "",
            ]
        )
        write_text_atomic(self.reports / "status.md", "\n".join(lines))

    def batch_screen(
        self,
        base: ResearchConfig,
        policy: dict,
        sources: dict,
        history: MarketData,
        now: pd.Timestamp,
        *,
        accept_code_update: bool = False,
    ) -> dict:
        if now.tzinfo is None:
            raise QuantError("Mutation screening requires a timezone-aware registration time.")
        now = now.tz_convert("UTC")
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a") as lock:
            lock_path.chmod(0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QuantError(
                    "Another evolution cycle is running; no concurrent screening."
                ) from exc
            metadata = self.verify(base, policy, sources, require_engine=False)
            if self.connection.execute(
                "SELECT 1 FROM events WHERE kind='batch_screen_failed_requires_review' LIMIT 1"
            ).fetchone():
                raise QuantError("A failed mutation screen is retained for operator review.")
            completed = self.connection.execute(
                "SELECT body FROM events WHERE kind='batch_screen_completed' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if completed:
                if metadata["engine_sha"] != fingerprint():
                    raise QuantError("Completed mutation evidence belongs to a different engine.")
                result = json.loads(completed["body"])
                self._write_status(result)
                return {**result, "repeated_batch_no_changes": True}
            rows = self.connection.execute(
                "SELECT * FROM candidates ORDER BY registered_at,id"
            ).fetchall()
            if len(rows) != 1 or rows[0]["mutation_id"] != "capital_matched_control":
                raise QuantError("Initial mutation screen requires exactly the registered control.")
            parent = rows[0]
            if parent["score"] is None or parent["history"] is None:
                raise QuantError("The control must finish historical diagnostics before screening.")
            if self.connection.execute(
                "SELECT 1 FROM cycles WHERE status='failed' LIMIT 1"
            ).fetchone():
                raise QuantError("A failed daily cycle must be reviewed before mutation screening.")
            if now < pd.Timestamp(metadata["registered_at"]):
                raise QuantError("Mutation registration time moved backwards.")
            history.validate()
            if (
                tuple(history.close.columns) != base.symbols
                or history.close.index[-1].date().isoformat() != base.as_of
            ):
                raise QuantError("Mutation screening requires the unchanged exposed-history panel.")
            if len(policy["mutations"]) + 1 != policy["max_candidates"]:
                raise QuantError(
                    "Every declared first-epoch mutation must be counted in this screen."
                )
            current_engine = fingerprint()
            if metadata["engine_sha"] != current_engine and not accept_code_update:
                raise QuantError(
                    "The reviewed screening engine needs explicit --accept-code-update."
                )
            control_genome = json.loads(parent["genome"])
            baseline = next_session(completed_session(now)).date().isoformat()
            day = completed_session(now).date().isoformat()
            registrations = []
            known_hashes = {parent["genome_sha"]}
            for index, mutation in enumerate(policy["mutations"], start=1):
                genome = {**control_genome, mutation["field"]: mutation["value"]}
                validate_genome(genome)
                genome_sha = digest_json(genome)
                if genome_sha in known_hashes:
                    raise QuantError("A declared mutation duplicates an already counted trial.")
                known_hashes.add(genome_sha)
                registrations.append(
                    {
                        "id": "evo_" + genome_sha[:12],
                        "genome_sha": genome_sha,
                        "genome": genome,
                        "parent_id": parent["id"],
                        "mutation_id": mutation["id"],
                        "created_cycle": f"{day}#{index:02d}",
                        "registered_at": (now + pd.Timedelta(microseconds=index)).isoformat(),
                        "baseline_session": baseline,
                    }
                )
            registration_artifact = {
                "registered_at": now.isoformat(),
                "mode": "all_mutations_registered_before_any_batch_result",
                "parent_id": parent["id"],
                "candidates": registrations,
                "source_fingerprints": sources,
                "policy_sha256": digest_json(policy),
                "previous_engine_sha256": metadata["engine_sha"],
                "screening_engine_sha256": current_engine,
                "risk_limits_changed": False,
                "order_authority": False,
            }
            path = self.reports / "batch-screens" / f"{day}-registration.json"
            if path.exists() and read_json(path) != registration_artifact:
                raise QuantError("A mutation registration artifact already differs.")
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS migrations ("
                    "id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL, "
                    "previous_engine TEXT NOT NULL, current_engine TEXT NOT NULL, "
                    "reason TEXT NOT NULL, risk_limits_changed INTEGER NOT NULL)"
                )
                if metadata["engine_sha"] != current_engine:
                    self.connection.execute(
                        "INSERT INTO migrations "
                        "(recorded_at,previous_engine,current_engine,reason,risk_limits_changed) "
                        "VALUES (?,?,?,?,0)",
                        (
                            now.isoformat(),
                            metadata["engine_sha"],
                            current_engine,
                            "Evaluate all mutations declared in the original epoch before results.",
                        ),
                    )
                    revised = {**metadata, "engine_sha": current_engine}
                    self.connection.execute(
                        "UPDATE metadata SET body=? WHERE id=1",
                        (json.dumps(revised, sort_keys=True),),
                    )
                for item in registrations:
                    self.connection.execute(
                        "INSERT INTO candidates VALUES "
                        "(?,?,?,?,?,?,?,?, 'registered',NULL,NULL,NULL)",
                        (
                            item["id"],
                            item["genome_sha"],
                            json.dumps(item["genome"]),
                            item["parent_id"],
                            item["mutation_id"],
                            item["created_cycle"],
                            item["registered_at"],
                            item["baseline_session"],
                        ),
                    )
                    self.event(
                        now,
                        "candidate_registered_before_evaluation",
                        item["id"],
                        {
                            "genome": item["genome"],
                            "mutation": item["mutation_id"],
                            "reason": "Predeclared first-epoch batch mutation.",
                            "baseline_session": baseline,
                            "history_is_already_exposed": True,
                        },
                    )
                self.event(
                    now,
                    "batch_screen_registered_before_evaluation",
                    None,
                    registration_artifact,
                )
            if not path.exists():
                write_json(path, registration_artifact)
            reports = {}
            try:
                for item in registrations:
                    directory = self.reports / "candidates" / item["id"]
                    if directory.exists():
                        raise QuantError("Partial mutation output exists; preserve it for review.")
                    config = candidate_config(base, policy, item["id"], item["genome"])
                    report, frames = historical_diagnostics(config, history, item["id"])
                    report.update(
                        {
                            "candidate_id": item["id"],
                            "registered_at": item["registered_at"],
                            "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                            "genome": item["genome"],
                            "score_used_only_for_research_priority": True,
                        }
                    )
                    for name, frame in frames.items():
                        write_text_atomic(
                            directory / f"{name}.csv",
                            frame.to_csv(float_format="%.12g"),
                        )
                    report["equity_files"] = {
                        f"{name}.csv": file_digest(directory / f"{name}.csv") for name in frames
                    }
                    write_json(directory / "history.json", report)
                    reports[item["id"]] = report
            except (QuantError, ValueError, OSError) as exc:
                with self.connection:
                    self.event(
                        pd.Timestamp.now(tz="UTC"),
                        "batch_screen_failed_requires_review",
                        None,
                        {"error": str(exc), "registered_trials": len(registrations)},
                    )
                raise
            active = self.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE status IN "
                "('awaiting_baseline','observing')"
            ).fetchone()[0]
            slots = max(0, policy["max_active_candidates"] - active)
            threshold = parent["score"] * (1 - policy["minimum_historical_score_improvement"])
            ranked = sorted(
                registrations,
                key=lambda item: (
                    reports[item["id"]]["objective_shortfall_score"],
                    item["id"],
                ),
            )
            eligible = [
                item
                for item in ranked
                if reports[item["id"]]["objective_shortfall_score"] <= threshold
                or reports[item["id"]]["all_historical_numeric_gates_passed"]
            ]
            admitted = {item["id"] for item in eligible[:slots]}
            evaluated = pd.Timestamp.now(tz="UTC")
            with self.connection:
                for item in registrations:
                    report = reports[item["id"]]
                    status = (
                        "awaiting_baseline" if item["id"] in admitted else "rejected_historical"
                    )
                    report["admission"] = status
                    write_json(
                        self.reports / "candidates" / item["id"] / "history.json",
                        report,
                    )
                    self.connection.execute(
                        "UPDATE candidates SET status=?,history=?,score=? WHERE id=?",
                        (
                            status,
                            json.dumps(report, sort_keys=True),
                            report["objective_shortfall_score"],
                            item["id"],
                        ),
                    )
                    self.event(
                        evaluated,
                        status,
                        item["id"],
                        {
                            "score": report["objective_shortfall_score"],
                            "historical_all_gates_passed": report[
                                "all_historical_numeric_gates_passed"
                            ],
                            "independent_forward_evidence": False,
                        },
                    )
                result = {
                    **self.status(),
                    "batch_action": "completed",
                    "batch_registered_trials": len(registrations),
                    "batch_admitted_candidates": sorted(admitted),
                    "batch_rejected_candidates": sorted(
                        item["id"] for item in registrations if item["id"] not in admitted
                    ),
                    "risk_limits_changed": False,
                }
                self.event(evaluated, "batch_screen_completed", None, result)
            write_json(self.reports / "batch-screens" / f"{day}-results.json", result)
            self._write_status(result)
            return result

    def _register_next(
        self, base: ResearchConfig, policy: dict, day: str, now: pd.Timestamp
    ) -> None:
        rows = self.connection.execute(
            "SELECT * FROM candidates ORDER BY registered_at,id"
        ).fetchall()
        if len(rows) >= policy["max_candidates"] or any(
            row["created_cycle"] == day for row in rows
        ):
            return
        if (
            rows
            and len(sessions(rows[-1]["created_cycle"], day)) - 1
            < policy["proposal_interval_sessions"]
        ):
            return
        active = sum(row["status"] in ACTIVE_STATES for row in rows)
        if active >= policy["max_active_candidates"]:
            return
        parent = None
        if not rows:
            genome, mutation = seed_genome(base, policy), "capital_matched_control"
            reason = "Re-evaluate the declared seed at the actual USD 10,000 experimental capital."
        else:
            evaluated = [row for row in rows if row["score"] is not None]
            if not evaluated:
                return
            parent = min(
                evaluated,
                key=lambda row: (
                    row["status"] == "retired_forward_risk",
                    parent_priority(
                        row["score"],
                        json.loads(row["forward_summary"])
                        if row["forward_summary"] and row["status"] != "retired_forward_risk"
                        else None,
                        base,
                        policy["minimum_forward_sessions"],
                    ),
                    row["id"],
                ),
            )
            used = {row["mutation_id"] for row in rows}
            hashes = {row["genome_sha"] for row in rows}
            selected = None
            for item in policy["mutations"]:
                if item["id"] in used:
                    continue
                proposal = {**json.loads(parent["genome"]), item["field"]: item["value"]}
                if digest_json(proposal) not in hashes:
                    selected = (proposal, item["id"])
                    break
            if selected is None:
                return
            genome, mutation = selected
            reason = (
                "One declared mutation of the lowest research shortfall parent. "
                "Mature forward feedback is included; this is not a return forecast."
            )
        validate_genome(genome)
        identifier = "evo_" + digest_json(genome)[:12]
        baseline = next_session(completed_session(now)).date().isoformat()
        with self.connection:
            self.connection.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?, 'registered',NULL,NULL,NULL)",
                (
                    identifier,
                    digest_json(genome),
                    json.dumps(genome),
                    parent["id"] if parent else None,
                    mutation,
                    day,
                    now.isoformat(),
                    baseline,
                ),
            )
            self.event(
                now,
                "candidate_registered_before_evaluation",
                identifier,
                {
                    "genome": genome,
                    "mutation": mutation,
                    "reason": reason,
                    "baseline_session": baseline,
                    "history_is_already_exposed": True,
                },
            )

    def _evaluate_registered(
        self, base: ResearchConfig, policy: dict, history: MarketData, now: pd.Timestamp
    ) -> None:
        rows = self.connection.execute(
            "SELECT * FROM candidates WHERE status='registered' ORDER BY registered_at,id"
        ).fetchall()
        for row in rows:
            config = candidate_config(base, policy, row["id"], json.loads(row["genome"]))
            report, frames = historical_diagnostics(config, history, row["id"])
            score = report["objective_shortfall_score"]
            parent = self.connection.execute(
                "SELECT score FROM candidates WHERE id=?", (row["parent_id"],)
            ).fetchone()
            improves = parent is None or score <= parent["score"] * (
                1 - policy["minimum_historical_score_improvement"]
            )
            admitted = improves or report["all_historical_numeric_gates_passed"]
            status = "awaiting_baseline" if admitted else "rejected_historical"
            report.update(
                {
                    "candidate_id": row["id"],
                    "registered_at": row["registered_at"],
                    "evaluated_at": now.isoformat(),
                    "genome": json.loads(row["genome"]),
                    "score_used_only_for_research_priority": True,
                    "admission": status,
                }
            )
            directory = self.reports / "candidates" / row["id"]
            for name, frame in frames.items():
                write_text_atomic(directory / f"{name}.csv", frame.to_csv(float_format="%.12g"))
            report["equity_files"] = {
                f"{name}.csv": file_digest(directory / f"{name}.csv") for name in frames
            }
            write_json(directory / "history.json", report)
            with self.connection:
                self.connection.execute(
                    "UPDATE candidates SET status=?,history=?,score=? WHERE id=?",
                    (status, json.dumps(report, sort_keys=True), score, row["id"]),
                )
                self.event(
                    now,
                    status,
                    row["id"],
                    {
                        "score": score,
                        "all_historical_numeric_gates_passed": report[
                            "all_historical_numeric_gates_passed"
                        ],
                        "independent_forward_evidence": False,
                    },
                )

    def _record_decision(
        self,
        config: ResearchConfig,
        candidate: str,
        data: MarketData,
        day: pd.Timestamp,
        now: pd.Timestamp,
    ) -> None:
        if not is_month_end(day):
            return
        execution = next_session(day)
        if now >= market_calendar().session_open(execution):
            raise QuantError("A forward decision cannot be created after its execution open.")
        weights = target_weights(data.close.loc[:day], config.candidate(candidate), config)
        self.connection.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?)",
            (
                candidate,
                day.date().isoformat(),
                execution.date().isoformat(),
                now.isoformat(),
                json.dumps(weights.to_dict(), sort_keys=True),
            ),
        )

    def _advance(
        self,
        row: sqlite3.Row,
        base: ResearchConfig,
        policy: dict,
        data: MarketData,
        snapshot_sha: str,
        now: pd.Timestamp,
    ) -> None:
        candidate = row["id"]
        config = candidate_config(base, policy, candidate, json.loads(row["genome"]))
        day, baseline = data.close.index[-1], pd.Timestamp(row["baseline_session"])
        if day < baseline:
            return
        observations = self.connection.execute(
            "SELECT * FROM observations WHERE candidate_id=? ORDER BY session_date", (candidate,)
        ).fetchall()
        expected = (
            baseline
            if not observations
            else next_session(pd.Timestamp(observations[-1]["session_date"]))
        )
        if observations and day == pd.Timestamp(observations[-1]["session_date"]):
            return
        if day != expected:
            raise QuantError(f"{candidate}: missed forward close; never backfill past decisions.")
        if now >= market_calendar().session_open(next_session(day)):
            raise QuantError("Forward observation is too late for the next-open decision boundary.")
        if pd.Timestamp(row["registered_at"]) >= market_calendar().session_close(baseline):
            raise QuantError("The prospective baseline must occur after candidate registration.")
        for old in observations:
            body = json.loads(old["body"])
            previous_close = data.raw_close.loc[old["session_date"]]
            if not np.allclose(
                previous_close.to_numpy(),
                [body["raw_close"][key] for key in data.close.columns],
                rtol=1e-6,
                atol=1e-6,
            ):
                raise QuantError(
                    "A previously observed source close was revised; require a data audit."
                )
        summary = {"forward_sessions": 0, "annualized_metrics_withheld": True}
        state = "observing"
        if not observations:
            body = {
                "equity": config.initial_capital,
                "return": 0.0,
                "cost": 0.0,
                "spy_equity": config.initial_capital,
                "spy_return": 0.0,
                "risk_free": float(data.risk_free.loc[day]),
                "baseline_only": True,
            }
        else:
            signals = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
            decisions = self.connection.execute(
                "SELECT * FROM decisions WHERE candidate_id=? ORDER BY signal_date", (candidate,)
            ).fetchall()
            for decision in decisions:
                signal_date, execution = (
                    pd.Timestamp(decision["signal_date"]),
                    pd.Timestamp(decision["execution_date"]),
                )
                created = pd.Timestamp(decision["created_at"])
                if (
                    execution != next_session(signal_date)
                    or not is_month_end(signal_date)
                    or created >= market_calendar().session_open(execution)
                    or created
                    < market_calendar().session_close(signal_date) + pd.Timedelta(minutes=30)
                    or created < pd.Timestamp(row["registered_at"])
                ):
                    raise QuantError("A prospective decision has invalid timing or lineage.")
                stored_weights = pd.Series(json.loads(decision["weights"])).reindex(
                    data.close.columns
                )
                recomputed = target_weights(
                    data.close.loc[:signal_date], config.candidate(candidate), config
                )
                if not np.allclose(stored_weights, recomputed, rtol=1e-8, atol=1e-10):
                    raise QuantError(
                        "A recorded decision changed or its historical inputs were revised."
                    )
                signals.loc[signal_date] = stored_weights
            result = simulate(
                data,
                signals,
                baseline.date().isoformat(),
                day.date().isoformat(),
                initial_capital=config.initial_capital,
                cost_bps=config.cost_bps_per_side,
                commission=config.commission_per_order,
            )
            spy = simulate(
                data,
                buy_and_hold_signals(
                    data.close, config.primary_benchmark, next_session(baseline).date().isoformat()
                ),
                baseline.date().isoformat(),
                day.date().isoformat(),
                initial_capital=config.initial_capital,
                cost_bps=config.cost_bps_per_side,
                commission=config.commission_per_order,
            )
            for old in observations:
                prior = json.loads(old["body"])
                date = pd.Timestamp(old["session_date"])
                if not np.allclose(
                    [prior["equity"], prior["spy_equity"]],
                    [result.frame.at[date, "equity"], spy.frame.at[date, "equity"]],
                    rtol=1e-10,
                    atol=1e-6,
                ):
                    raise QuantError(
                        "Recomputation changed a recorded forward value; do not rewrite it."
                    )
            final, benchmark = result.frame.iloc[-1], spy.frame.iloc[-1]
            body = {
                "equity": float(final["equity"]),
                "return": float(final["return"]),
                "cost": float(final["cost"]),
                "spy_equity": float(benchmark["equity"]),
                "spy_return": float(benchmark["return"]),
                "risk_free": float(final["risk_free"]),
                "baseline_only": False,
            }
            returns, risk_free = result.frame["return"].iloc[1:], result.frame["risk_free"].iloc[1:]
            wealth = result.frame["equity"].to_numpy()
            drawdown = float(-(wealth / np.maximum.accumulate(wealth) - 1).min())
            count = len(returns)
            summary = {
                "forward_sessions": count,
                "total_return": float(wealth[-1] / wealth[0] - 1),
                "max_drawdown": drawdown,
                "annualized_metrics_withheld": count < policy["minimum_forward_sessions"],
            }
            if drawdown > base.targets.max_drawdown_at_most:
                state = "retired_forward_risk"
            elif count >= policy["minimum_forward_sessions"]:
                metrics = performance(returns, risk_free)
                benchmark_metrics = performance(spy.frame["return"].iloc[1:], risk_free)
                forward_config = replace(
                    config,
                    targets=replace(
                        config.targets, minimum_holdout_sessions=policy["minimum_forward_sessions"]
                    ),
                )
                gates = acceptance(metrics, benchmark_metrics, forward_config)
                confidence = block_bootstrap(
                    returns,
                    spy.frame["return"].iloc[1:],
                    risk_free,
                    samples=config.stress.bootstrap_samples,
                    block=config.stress.bootstrap_block_sessions,
                    seed=config.stress.bootstrap_seed,
                )
                sharpe_interval = confidence["sharpe_95pct_interval"]
                excess_interval = confidence["excess_cagr_vs_spy_95pct_interval"]
                confidence_passed = bool(
                    sharpe_interval is not None
                    and sharpe_interval[0] > 0
                    and excess_interval is not None
                    and excess_interval[0] > 0
                )
                summary.update(
                    {
                        "metrics": metrics,
                        "benchmark": benchmark_metrics,
                        "gates": gates,
                        "conditional_bootstrap": confidence,
                        "positive_edge_interval": confidence_passed,
                        "multiple_testing_adjusted": False,
                    }
                )
                history_passed = json.loads(row["history"])["all_historical_numeric_gates_passed"]
                state = (
                    "paper_review_ready"
                    if history_passed and all(gates.values()) and confidence_passed
                    else "retired_forward"
                )
            else:
                state = "observing"
            if state != "observing":
                summary["observation_ended_without_assumed_liquidation"] = True
        body.update(
            {
                "raw_close": {key: float(value) for key, value in data.raw_close.loc[day].items()},
                "snapshot_sha": snapshot_sha,
                "broker_orders_sent": 0,
            }
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO observations VALUES (?,?,?,?)",
                (
                    candidate,
                    day.date().isoformat(),
                    now.isoformat(),
                    json.dumps(body, sort_keys=True),
                ),
            )
            if state == "observing":
                self._record_decision(config, candidate, data, day, now)
            self.connection.execute(
                "UPDATE candidates SET status=?,forward_summary=? WHERE id=?",
                (state, json.dumps(summary, sort_keys=True), candidate),
            )
            self.event(
                now,
                "forward_close_recorded",
                candidate,
                {
                    "session": day.date().isoformat(),
                    "state": state,
                    "summary": summary,
                    "paper_order_authority": False,
                },
            )

    def cycle(
        self,
        base: ResearchConfig,
        policy: dict,
        sources: dict,
        history: MarketData,
        data: MarketData,
        snapshot_sha: str,
        now: pd.Timestamp,
    ) -> dict:
        if now.tzinfo is None:
            raise QuantError("Evolution requires a timezone-aware observation time.")
        now = now.tz_convert("UTC")
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a") as lock:
            lock_path.chmod(0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QuantError(
                    "Another evolution cycle is running; no concurrent mutations."
                ) from exc
            metadata = self.verify(base, policy, sources)
            last = self.connection.execute("SELECT MAX(started_at) FROM cycles").fetchone()[0]
            if now < pd.Timestamp(metadata["registered_at"]) or (
                last is not None and now < pd.Timestamp(last)
            ):
                raise QuantError("Evolution clock moved backwards relative to recorded evidence.")
            if self.connection.execute(
                "SELECT 1 FROM cycles WHERE status='failed' LIMIT 1"
            ).fetchone():
                raise QuantError(
                    "A failed cycle is retained for review; no silent automatic retry."
                )
            if self.connection.execute(
                "SELECT 1 FROM events " "WHERE kind='batch_screen_failed_requires_review' LIMIT 1"
            ).fetchone():
                raise QuantError(
                    "A failed mutation screen is retained for review; no daily-cycle bypass."
                )
            data.validate()
            if (
                tuple(data.close.columns) != base.symbols
                or tuple(history.close.columns) != base.symbols
            ):
                raise QuantError(
                    "Evolution may not expand or reorder the registered market universe."
                )
            if now.tzinfo is None or data.close.index[-1] != completed_session(now):
                raise QuantError(
                    "Evolution needs the latest complete session, not stale/future data."
                )
            if history.close.index[-1].date().isoformat() != base.as_of:
                raise QuantError(
                    "Historical diagnostic data must end at the original exposed boundary."
                )
            if history.close.index[-1] > completed_session(now):
                raise QuantError("Historical diagnostics cannot include a future trading session.")
            day = data.close.index[-1].date().isoformat()
            if now.date().isoformat() > policy["stop_after"]:
                return {**self.status(), "cycle_action": "paused_deadline"}
            prior = self.connection.execute(
                "SELECT * FROM cycles WHERE session_date=?", (day,)
            ).fetchone()
            if prior:
                if prior["snapshot_sha"] != snapshot_sha:
                    raise QuantError("This session's immutable market snapshot changed.")
                if prior["status"] == "completed":
                    result = json.loads(prior["result"])
                    self.publish(day, result)
                    return {**result, "repeated_cycle_no_changes": True}
                if prior["status"] == "failed":
                    raise QuantError(
                        "A failed cycle is retained for review; no silent automatic retry."
                    )
            count = self.connection.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
            if prior is None and count >= policy["max_cycle_sessions"]:
                return {**self.status(), "cycle_action": "paused_cycle_budget"}
            with self.connection:
                if prior is None:
                    self.connection.execute(
                        "INSERT INTO cycles VALUES (?,?,?,'started',NULL)",
                        (day, snapshot_sha, now.isoformat()),
                    )
            try:
                active = self.connection.execute(
                    "SELECT * FROM candidates WHERE status IN ('awaiting_baseline','observing')"
                ).fetchall()
                for row in active:
                    self._advance(row, base, policy, data, snapshot_sha, now)
                self._register_next(base, policy, day, now)
                self._evaluate_registered(base, policy, history, now)
                result = {**self.status(), "cycle_session": day, "cycle_action": "completed"}
                with self.connection:
                    self.connection.execute(
                        "UPDATE cycles SET status='completed',result=? WHERE session_date=?",
                        (json.dumps(result, sort_keys=True), day),
                    )
                    self.event(now, "cycle_completed", None, {"session": day})
                self.publish(day, result)
                return result
            except (QuantError, ValueError, OSError, sqlite3.Error) as exc:
                with self.connection:
                    self.connection.execute(
                        "UPDATE cycles SET status='failed',result=? WHERE session_date=?",
                        (json.dumps({"error": str(exc)}), day),
                    )
                    self.event(
                        now,
                        "cycle_failed_requires_review",
                        None,
                        {"session": day, "error": str(exc)},
                    )
                raise

    def status(self) -> dict:
        row = self.connection.execute("SELECT body FROM metadata WHERE id=1").fetchone()
        if row is None:
            raise QuantError("Evolution policy has not been registered.")
        metadata = json.loads(row["body"])
        candidates = []
        for item in self.connection.execute("SELECT * FROM candidates ORDER BY registered_at,id"):
            candidates.append(
                {
                    "id": item["id"],
                    "parent_id": item["parent_id"],
                    "mutation": item["mutation_id"],
                    "registered_at": item["registered_at"],
                    "baseline_session": item["baseline_session"],
                    "state": item["status"],
                    "genome": json.loads(item["genome"]),
                    "historical_shortfall": item["score"],
                    "historical_all_gates_passed": json.loads(item["history"])[
                        "all_historical_numeric_gates_passed"
                    ]
                    if item["history"]
                    else False,
                    "forward": json.loads(item["forward_summary"])
                    if item["forward_summary"]
                    else None,
                }
            )
        return {
            "mode": "bounded_autonomous_research",
            "registered_at": metadata["registered_at"],
            "registered_engine_sha256": metadata["engine_sha"],
            "registered_policy_sha256": metadata["policy_sha"],
            "policy": metadata["policy"],
            "engine_matches_registration": metadata["engine_sha"] == fingerprint(),
            "new_trials": len(candidates),
            "cumulative_trials": metadata["policy"]["prior_trials"] + len(candidates),
            "candidates": candidates,
            "paper_review_candidates": [
                item["id"] for item in candidates if item["state"] == "paper_review_ready"
            ],
            "qualified_live_strategy": None,
            "broker_orders_sent": 0,
            "order_authority": False,
            "investment_objective_verified": False,
            "warning": (
                "Historical reuse and multiple testing remain disclosed. Research-stage promotion "
                "is not broker deployment, guaranteed returns, or the full investment objective."
            ),
        }

    def publish(self, day: str, result: dict) -> None:
        path = self.reports / "cycles" / f"{day}.json"
        if path.exists() and read_json(path) != result:
            raise QuantError("Refusing to rewrite a published autonomous cycle.")
        if not path.exists():
            write_json(path, result)
        write_json(self.reports / "status.json", self.status())
        lines = [
            "# Autonomous research status",
            "",
            "**Research only: no order authority or verified investment objective.**",
            "",
            f"Cumulative candidate trials: {result['cumulative_trials']}.",
            "Previously viewed history is diagnostic, never relabeled as a new holdout.",
            "",
            "| Candidate | Parent | State | Historical shortfall | Forward sessions |",
            "|---|---|---|---:|---:|",
        ]
        for item in result["candidates"]:
            score = (
                "pending"
                if item["historical_shortfall"] is None
                else f"{item['historical_shortfall']:.3f}"
            )
            count = item["forward"]["forward_sessions"] if item["forward"] else 0
            lines.append(
                f"| {item['id']} | {item['parent_id'] or 'registered seed'} | "
                f"{item['state']} | {score} | {count} |"
            )
        lines.extend(
            [
                "",
                "Lower shortfall only prioritizes research; it is not a return forecast.",
                "Risk, cash reserve, capital, universe, costs, and next-open execution are locked.",
                "Missed or revised forward observations stop the cycle; no backfilling.",
                "Retired and rejected candidates remain in the ledger and total trial count.",
                "",
            ]
        )
        write_text_atomic(self.reports / "status.md", "\n".join(lines))


def add_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "evolve", help="Bounded, auditable autonomous research; no orders."
    )
    parser.add_argument("stage", choices=["init", "cycle", "status", "batch-screen"])
    parser.add_argument("--policy", type=Path, default=Path("config/evolution.json"))
    parser.add_argument("--ledger", type=Path, default=Path("runtime/evolution.sqlite3"))
    parser.add_argument("--reports", type=Path, default=Path("reports/evolution"))
    parser.add_argument("--development", type=Path, default=Path("data/development"))
    parser.add_argument("--history", type=Path, default=Path("data/holdout"))
    parser.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))
    parser.add_argument(
        "--prior-trials", type=Path, default=Path("reports/expanded/development/results.json")
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--accept-code-update", action="store_true")


def dispatch_evolution(args: argparse.Namespace, base: ResearchConfig) -> dict:
    policy = read_json(args.policy)
    validate_policy(policy, base)
    if args.stage == "status":
        engine = EvolutionEngine(args.ledger, args.reports)
        try:
            return engine.status()
        finally:
            engine.close()
    verify_freeze(base, args.development, args.freeze)
    verify_dataset(args.history, base, "holdout")
    trials = read_json(args.prior_trials)
    if trials.get("cumulative_candidate_trials") != policy["prior_trials"]:
        raise QuantError("The evolution epoch must disclose every preceding candidate trial.")
    sources = {
        "development": file_digest(args.development / "manifest.json"),
        "exposed_history": file_digest(args.history / "manifest.json"),
        "seed_freeze": file_digest(args.freeze),
        "prior_trials": file_digest(args.prior_trials),
    }
    now = pd.Timestamp.now(tz="UTC")
    if args.stage == "init":
        if now.date().isoformat() > policy["stop_after"]:
            raise QuantError("Do not create an already-expired evolution epoch.")
        engine = EvolutionEngine(args.ledger, args.reports, create=True)
        try:
            return engine.initialize(base, policy, sources, now)
        finally:
            engine.close()
    engine = EvolutionEngine(args.ledger, args.reports)
    try:
        if args.stage == "batch-screen":
            history = load_market(base, args.development, args.history)
            return engine.batch_screen(
                base,
                policy,
                sources,
                history,
                now,
                accept_code_update=args.accept_code_update,
            )
        engine.verify(base, policy, sources)
        if now.date().isoformat() > policy["stop_after"]:
            return {**engine.status(), "cycle_action": "paused_deadline"}
        completed = completed_session(now).date().isoformat()
        count = engine.connection.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        if (
            count >= policy["max_cycle_sessions"]
            and not engine.connection.execute(
                "SELECT 1 FROM cycles WHERE session_date=?", (completed,)
            ).fetchone()
        ):
            return {**engine.status(), "cycle_action": "paused_cycle_budget"}
        snapshot = args.data or Path("data/shadow") / completed
        if not snapshot.exists():
            fetch_dataset(base, "forward", snapshot)
        data = load_market(base, snapshot, phase="forward")
        history = load_market(base, args.development, args.history)
        return engine.cycle(
            base,
            policy,
            sources,
            history,
            data,
            file_digest(snapshot / "manifest.json"),
            pd.Timestamp.now(tz="UTC"),
        )
    finally:
        engine.close()
