from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import completed_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.metrics import performance
from us_quant.storage import digest_json, file_digest, read_json, write_json
from us_quant.strategy import buy_and_hold_signals
from us_quant.strategy_replay import registered_targets, replay_dependencies_hash

ROOT = Path(__file__).resolve().parents[2]


def engine_fingerprint() -> str:
    return digest_json(
        {
            "program": file_digest(Path(__file__)),
            "replay_dependencies": replay_dependencies_hash(),
        }
    )


def timestamp(now: datetime | None = None) -> datetime:
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise QuantError("Research program clocks require an explicit timezone.")
    return now.astimezone(timezone.utc)


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("program_id") != "recurring_multifactor_research_v1"
        or policy.get("timezone") != "Asia/Shanghai"
        or policy.get("schedule") != {"interval": "weekly", "day": 6, "hour": 9}
        or policy.get("maximum_new_factors_per_cycle") != 2
        or policy.get("minimum_economic_factor_families") != 3
        or policy.get("prior_evaluated_configurations") != 84
        or policy.get("goals")
        != {
            "net_excess_sharpe_strictly_above": 1.0,
            "max_drawdown_at_most": 0.15,
            "beat_spy": True,
            "horizons_years": [10, 5],
            "scenarios": ["base", "stress"],
            "capital_usd": 10000.0,
            "base_cost_bps": 5.0,
            "stress_cost_bps": 20.0,
            "commission_per_order": 1.0,
            "base_delay_sessions": 1,
            "stress_delay_sessions": 2,
            "leveraged_products_allowed": False,
        }
        or policy.get("promotion")
        != {
            "scope": "research_version_only",
            "minimum_forward_sessions_for_trading_review": 63,
            "automatic_order_submission": False,
            "automatic_live_deployment": False,
            "backfill_forward_observations": False,
            "allow_relabelled_holdouts": False,
        }
        or policy.get("required_stock_data")
        != [
            "filing_time_fundamentals",
            "historical_security_master",
            "historical_membership",
            "delisted_total_returns",
            "adjusted_open_and_close",
        ]
    ):
        raise QuantError("The research program's cadence, objectives or authority changed.")
    seeds = policy.get("seed_factors", [])
    if {item.get("id") for item in seeds} != {
        "price_momentum",
        "value_exposure",
        "quality_exposure",
        "low_volatility_exposure",
    } or len(seeds) != 4:
        raise QuantError("The four previously studied factor families must remain in the catalog.")
    for factor in seeds:
        validate_factor(factor)


def safe_file(root: Path, relative: str, expected: str | None = None) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise QuantError("Research evidence paths must be relative to the isolated workbench.")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise QuantError("Research evidence is missing, external or a symlink.")
    if expected is not None and file_digest(path) != expected:
        raise QuantError("Research source or ledger hash changed.")
    return path


def validate_factor(factor: dict) -> str:
    for key in ("id", "family"):
        if not isinstance(factor.get(key), str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{2,80}", factor[key]
        ):
            raise QuantError("Factor identifiers and economic families must be stable identifiers.")
    if (
        not all(
            isinstance(factor.get(k), str) and factor[k].strip()
            for k in ("name", "definition", "rationale", "limitations")
        )
        or factor.get("direction") not in {"higher", "lower"}
        or not isinstance(factor.get("parameters"), dict)
        or not isinstance(factor.get("inputs"), list)
        or not factor["inputs"]
        or any(not isinstance(value, str) or not value for value in factor["inputs"])
        or not isinstance(factor.get("sources"), list)
        or not factor["sources"]
    ):
        raise QuantError("A factor needs a definition, direction, inputs, motivation and sources.")
    for url in factor["sources"]:
        if not isinstance(url, str):
            raise QuantError("Factor source URLs must be strings.")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise QuantError("Factor references must be public HTTPS URLs without credentials.")
    return digest_json(
        {
            "family": factor["family"],
            "definition": " ".join(factor["definition"].lower().split()),
            "direction": factor["direction"],
            "inputs": sorted(set(factor["inputs"])),
            "parameters": factor["parameters"],
        }
    )


def readiness_blockers(
    readiness: dict, policy: dict, root: Path, *, now: datetime | None = None
) -> list[str]:
    if readiness.get("schema_version") != 1 or not isinstance(readiness.get("capabilities"), dict):
        raise QuantError("Explicit data-capability evidence is required before a research cycle.")
    checked = pd.Timestamp(readiness.get("checked_at"))
    current = pd.Timestamp(timestamp(now))
    if (
        pd.isna(checked)
        or checked.tzinfo is None
        or checked > current
        or current - checked > pd.Timedelta(days=8)
    ):
        raise QuantError(
            "Data readiness must be freshly checked, timezone-aware and not future dated."
        )
    source = readiness.get("source_check")
    if source is not None:
        safe_file(root, source["path"], source["sha256"])
    scope = readiness.get("data_scope", "direct_stock")
    if scope == "direct_stock":
        required = policy["required_stock_data"]
    elif scope == "factor_etf_portfolio":
        required = [
            "factor_mandates",
            "actual_fund_history",
            "post_inception_and_actions",
            "unleveraged_fund_identity",
        ]
    else:
        raise QuantError("Unknown research data scope; do not relabel stock-data requirements.")
    blockers = []
    for name in required:
        item = readiness["capabilities"].get(name)
        if not isinstance(item, dict) or type(item.get("verified")) is not bool:
            raise QuantError(f"Data readiness is unspecified: {name}")
        if not item["verified"]:
            if not isinstance(item.get("reason"), str) or not item["reason"]:
                raise QuantError("An unavailable dataset requires an explicit reason.")
            blockers.append(f"{name}: {item['reason']}")
        else:
            safe_file(root, item["evidence_path"], item["evidence_sha256"])
    if scope == "factor_etf_portfolio" and not blockers:
        verified_etf_market(readiness, root)
    return blockers


def verified_etf_market(readiness: dict, root: Path) -> MarketData:
    from us_quant.multifactor_stability import load_market

    source = readiness.get("verified_etf_source", {})
    if source.get("adapter") == "growth_factor_satellite_20261011":
        from us_quant.growth_factor_satellite import verified_market

        policy = read_json(safe_file(root, source["policy"], source["policy_sha256"]))
        return verified_market(policy)
    if source.get("adapter") == "defensive_factor_rotation_20261011":
        from us_quant.defensive_factor_rotation import verified_market

        policy = read_json(safe_file(root, source["policy"], source["policy_sha256"]))
        return verified_market(policy)
    if source.get("adapter") == "factor_implementation_replication_20261010":
        from us_quant.factor_replication import load_replication_market

        policy = read_json(safe_file(root, source["policy"], source["policy_sha256"]))
        manifest = safe_file(root, source["factor_manifest"], source["factor_manifest_sha256"])
        return load_replication_market(policy, manifest.parent)
    if source.get("adapter") != "frozen_multifactor_etf_20261010":
        raise QuantError("ETF readiness needs the explicitly audited actual-fund adapter.")
    policy = read_json(safe_file(root, source["policy"], source["policy_sha256"]))
    registration = read_json(safe_file(root, source["registration"], source["registration_sha256"]))
    base_manifest = safe_file(root, source["base_manifest"], source["base_manifest_sha256"])
    factor_manifest = safe_file(root, source["factor_manifest"], source["factor_manifest_sha256"])
    return load_market(policy, registration, base_manifest.parent, factor_manifest.parent)


def review_market(bundle: dict, root: Path) -> MarketData:
    manifest = bundle.get("market")
    if not isinstance(manifest, dict) or set(manifest) != {
        "open",
        "close",
        "raw_close",
        "volume",
        "risk_free",
    }:
        raise QuantError(
            "Review requires the actual hashed market panels, not just an equity curve."
        )
    panels = {}
    for key, evidence in manifest.items():
        path = safe_file(root, evidence["path"], evidence["sha256"])
        panels[key] = pd.read_csv(path, index_col=0, parse_dates=True)
    if list(panels["risk_free"].columns) != ["risk_free"]:
        raise QuantError("The market panel needs one explicit lagged risk-free series.")
    data = MarketData(
        panels["open"],
        panels["close"],
        panels["raw_close"],
        panels["volume"],
        panels["risk_free"]["risk_free"],
    )
    data.validate()
    return data


class ResearchProgram:
    def __init__(
        self,
        path: Path,
        policy: dict,
        *,
        root: Path = ROOT,
        create: bool = False,
        expected_previous_engine: str | None = None,
        expected_previous_event: str | None = None,
    ):
        validate_policy(policy)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise QuantError("The program ledger must stay inside this isolated workbench.")
        if create:
            if path.exists():
                raise QuantError("Refusing to reset an existing research program.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o600, exist_ok=False)
        elif not path.is_file():
            raise QuantError("Initialize the research program explicitly before cycles.")
        path.chmod(0o600)
        if (expected_previous_engine is None) != (expected_previous_event is None) or (
            create and expected_previous_engine is not None
        ):
            raise QuantError("Engine migration requires both prior hashes and an existing ledger.")
        if expected_previous_engine is not None and any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (expected_previous_engine, expected_previous_event)
        ):
            raise QuantError("Migration anchors must be exact SHA256 values.")
        self.root, self.policy = root, policy
        self.previous_engine = expected_previous_engine
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        if create:
            with self.db:
                self.db.executescript("""
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE metadata (
                        id INTEGER PRIMARY KEY CHECK(id=1), policy_sha TEXT, engine_sha TEXT
                    );
                    CREATE TABLE events (
                        seq INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL,
                        body TEXT NOT NULL, previous_sha TEXT NOT NULL, sha TEXT NOT NULL
                    );
                    CREATE TABLE cycles (
                        week TEXT PRIMARY KEY, at TEXT NOT NULL, input_sha TEXT NOT NULL,
                        result TEXT NOT NULL
                    );
                    CREATE TABLE factors (
                        id TEXT PRIMARY KEY, semantic_sha TEXT UNIQUE NOT NULL,
                        registered_at TEXT NOT NULL, body TEXT NOT NULL
                    );
                    CREATE TABLE candidates (
                        id TEXT PRIMARY KEY, spec_sha TEXT UNIQUE NOT NULL,
                        semantic_sha TEXT UNIQUE NOT NULL,
                        registered_at TEXT NOT NULL, body TEXT NOT NULL
                    );
                    CREATE TABLE reviews (
                        candidate_id TEXT PRIMARY KEY, at TEXT NOT NULL,
                        evidence_sha TEXT NOT NULL, body TEXT NOT NULL
                    );
                """)
                now = timestamp()
                self.db.execute(
                    "INSERT INTO metadata VALUES (1,?,?)",
                    (digest_json(policy), engine_fingerprint()),
                )
                initial_factors = {}
                for factor in policy["seed_factors"]:
                    signature = validate_factor(factor)
                    self.db.execute(
                        "INSERT INTO factors VALUES (?,?,?,?)",
                        (
                            factor["id"],
                            signature,
                            now.isoformat(),
                            json.dumps(factor, sort_keys=True, allow_nan=False),
                        ),
                    )
                    initial_factors[factor["id"]] = signature
                self._event(
                    "initialized",
                    {
                        "policy_sha256": digest_json(policy),
                        "factor_records": initial_factors,
                    },
                    now,
                )
        try:
            self.verify()
            if expected_previous_engine is not None:
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.verify()
                    metadata = self.db.execute(
                        "SELECT engine_sha FROM metadata WHERE id=1"
                    ).fetchone()
                    head = self.db.execute(
                        "SELECT sha FROM events ORDER BY seq DESC LIMIT 1"
                    ).fetchone()
                    if (
                        metadata["engine_sha"] != expected_previous_engine
                        or head["sha"] != expected_previous_event
                        or expected_previous_engine == engine_fingerprint()
                    ):
                        raise QuantError(
                            "Engine migration preconditions changed; no ledger reset allowed."
                        )
                    self._event(
                        "engine_migrated",
                        {
                            "from_engine_sha256": expected_previous_engine,
                            "to_engine_sha256": engine_fingerprint(),
                            "prior_event_sha256": expected_previous_event,
                            "policy_unchanged": True,
                            "reason": "Reviewed engine update; preserve history, policy and goals.",
                            "fingerprint_scheme": "program_and_replay_dependencies",
                        },
                        timestamp(),
                    )
                    self.db.execute(
                        "UPDATE metadata SET engine_sha=? WHERE id=1",
                        (engine_fingerprint(),),
                    )
                self.previous_engine = None
                self.verify()
        except (QuantError, sqlite3.Error):
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def _event(self, kind: str, body: dict, now: datetime) -> None:
        row = self.db.execute("SELECT sha,at FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        if row and pd.Timestamp(now) < pd.Timestamp(row["at"]):
            raise QuantError("Research event timestamps may not move backwards.")
        previous = row["sha"] if row else "0" * 64
        value = {"at": now.isoformat(), "kind": kind, "body": body, "previous_sha": previous}
        self.db.execute(
            "INSERT INTO events (at,kind,body,previous_sha,sha) VALUES (?,?,?,?,?)",
            (
                value["at"],
                kind,
                json.dumps(body, sort_keys=True, allow_nan=False),
                previous,
                digest_json(value),
            ),
        )

    def verify(self) -> None:
        row = self.db.execute("SELECT policy_sha,engine_sha FROM metadata WHERE id=1").fetchone()
        if (
            row is None
            or row["policy_sha"] != digest_json(self.policy)
            or (
                row["engine_sha"] != engine_fingerprint()
                and (self.previous_engine is None or row["engine_sha"] != self.previous_engine)
            )
        ):
            raise QuantError(
                "Research policy/engine changed; preserve the ledger and review migration."
            )
        previous = "0" * 64
        bodies = {"cycle": {}, "candidate": {}, "review": {}}
        factor_records = {}
        for event in self.db.execute("SELECT * FROM events ORDER BY seq"):
            body = json.loads(event["body"])
            value = {
                "at": event["at"],
                "kind": event["kind"],
                "body": body,
                "previous_sha": previous,
            }
            if event["previous_sha"] != previous or event["sha"] != digest_json(value):
                raise QuantError("The append-only research event chain was altered.")
            previous = event["sha"]
            if event["kind"] in {"initialized", "cycle"}:
                factor_records.update(body.get("factor_records", {}))
            if event["kind"] in bodies:
                key = body["week"] if event["kind"] == "cycle" else body["candidate_id"]
                bodies[event["kind"]][key] = body
        for row in self.db.execute("SELECT * FROM factors"):
            factor = json.loads(row["body"])
            if (
                factor["id"] != row["id"]
                or validate_factor(factor) != row["semantic_sha"]
                or factor_records.get(row["id"]) != row["semantic_sha"]
            ):
                raise QuantError("A previously recorded factor was changed.")
        if self.db.execute("SELECT COUNT(*) FROM factors").fetchone()[0] != len(factor_records):
            raise QuantError("A factor definition was added or removed outside its recorded cycle.")
        for row in self.db.execute("SELECT * FROM candidates"):
            spec = json.loads(row["body"])
            event = bodies["candidate"].get(row["id"], {})
            semantic = digest_json({key: value for key, value in spec.items() if key != "id"})
            if (
                digest_json(spec) != row["spec_sha"]
                or semantic != row["semantic_sha"]
                or event.get("spec_sha256") != row["spec_sha"]
            ):
                raise QuantError("Candidate history was changed or detached from its registration.")
        for row in self.db.execute("SELECT * FROM reviews"):
            if json.loads(row["body"]) != bodies["review"].get(row["candidate_id"]):
                raise QuantError("A candidate review was changed after recording.")
        for row in self.db.execute("SELECT * FROM cycles"):
            if json.loads(row["result"]) != bodies["cycle"].get(row["week"]):
                raise QuantError("A completed weekly cycle was changed.")
        if self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] != len(
            bodies["candidate"]
        ):
            raise QuantError("A registered candidate was removed from the research history.")
        if self.db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] != len(bodies["review"]):
            raise QuantError("A recorded review was removed from the research history.")
        if self.db.execute("SELECT COUNT(*) FROM cycles").fetchone()[0] != len(bodies["cycle"]):
            raise QuantError("A completed research cycle was removed.")

    def cycle(self, proposals: dict, readiness: dict, *, now: datetime | None = None) -> dict:
        now = timestamp(now)
        week = now.astimezone(ZoneInfo(self.policy["timezone"])).strftime("%G-W%V")
        if proposals.get("schema_version") != 1 or not isinstance(proposals.get("factors"), list):
            raise QuantError("Factor discovery requires a versioned proposal list.")
        blockers = readiness_blockers(readiness, self.policy, self.root, now=now)
        factors = proposals["factors"]
        if len(factors) > self.policy["maximum_new_factors_per_cycle"]:
            raise QuantError("This cycle exceeds the bounded new-factor budget.")
        identities = [(factor, validate_factor(factor)) for factor in factors]
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.verify()
            existing = self.db.execute("SELECT result FROM cycles WHERE week=?", (week,)).fetchone()
            if existing:
                return {
                    "cycle_action": "already_recorded",
                    "week": week,
                    "prior_result": json.loads(existing["result"]),
                }
            added, duplicates = [], []
            for factor, signature in identities:
                same_id = self.db.execute(
                    "SELECT semantic_sha FROM factors WHERE id=?", (factor["id"],)
                ).fetchone()
                if same_id and same_id["semantic_sha"] != signature:
                    raise QuantError(
                        "A factor identifier cannot be reused for a changed definition."
                    )
                same = self.db.execute(
                    "SELECT id FROM factors WHERE semantic_sha=?", (signature,)
                ).fetchone()
                if same:
                    duplicates.append(same["id"])
                    continue
                self.db.execute(
                    "INSERT INTO factors VALUES (?,?,?,?)",
                    (
                        factor["id"],
                        signature,
                        now.isoformat(),
                        json.dumps(factor, sort_keys=True, allow_nan=False),
                    ),
                )
                added.append(factor["id"])
            result = {
                "week": week,
                "at": now.isoformat(),
                "new_factors": added,
                "duplicate_definitions": duplicates,
                "data_blockers": blockers,
                "data_ready": not blockers,
                "new_strategy_evaluations": 0,
                "state": "blocked_data" if blockers else "ready_for_preregistered_candidate",
                "readiness_sha256": digest_json(readiness),
                "proposal_sha256": digest_json(proposals),
                "factor_records": {
                    item["id"]: signature for item, signature in identities if item["id"] in added
                },
                "order_authority": False,
                "data_scope": readiness.get("data_scope", "direct_stock"),
            }
            self._event("cycle", result, now)
            self.db.execute(
                "INSERT INTO cycles VALUES (?,?,?,?)",
                (
                    week,
                    now.isoformat(),
                    digest_json({"proposals": proposals, "readiness": readiness}),
                    json.dumps(result, sort_keys=True, allow_nan=False),
                ),
            )
        return result

    def register_candidate(
        self, spec: dict, readiness: dict, *, now: datetime | None = None
    ) -> dict:
        now = timestamp(now)
        blockers = readiness_blockers(readiness, self.policy, self.root, now=now)
        if blockers:
            raise QuantError(
                "Candidate blocked by missing point-in-time data for its scope: "
                + "; ".join(blockers)
            )
        identifier = spec.get("id")
        scope = spec.get("data_scope", "direct_stock")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", identifier)
            or not isinstance(spec.get("factor_ids"), list)
            or len(spec["factor_ids"]) != len(set(spec["factor_ids"]))
            or spec.get("order_authority") is not False
            or spec.get("leveraged_products_allowed") is not False
            or spec.get("history_status") != "exposed_history_not_independent_holdout"
            or scope != readiness.get("data_scope", "direct_stock")
        ):
            raise QuantError("Candidate must declare its factors, data limitations and authority.")
        families = set()
        for identifier_factor in spec["factor_ids"]:
            row = self.db.execute(
                "SELECT body FROM factors WHERE id=?", (identifier_factor,)
            ).fetchone()
            if not row:
                raise QuantError("Candidate references an unregistered factor.")
            families.add(json.loads(row["body"])["family"])
        if len(families) < self.policy["minimum_economic_factor_families"]:
            raise QuantError(
                "Multiple versions of one factor do not make a diversified multifactor strategy."
            )
        if not isinstance(spec.get("frozen_files"), dict) or not spec["frozen_files"]:
            raise QuantError("Candidate code and parameters must be frozen before evaluation.")
        for relative, sha in spec["frozen_files"].items():
            safe_file(self.root, relative, sha)
        market = review_market({"market": spec.get("market")}, self.root)
        if scope == "factor_etf_portfolio":
            etf_market = verified_etf_market(readiness, self.root)
            if set(spec["factor_ids"]) != {
                "price_momentum",
                "value_exposure",
                "quality_exposure",
                "low_volatility_exposure",
            }:
                raise QuantError(
                    "ETF research must identify fund mandates, not missing stock signals."
                )
            for name in ("open", "close", "raw_close", "volume"):
                actual, verified = getattr(market, name), getattr(etf_market, name)
                if (
                    not actual.index.equals(verified.index)
                    or not actual.columns.equals(verified.columns)
                    or not np.allclose(actual, verified, rtol=1e-10, atol=1e-9)
                ):
                    raise QuantError("ETF market differs from the audited provider snapshot.")
            if not np.allclose(market.risk_free, etf_market.risk_free, rtol=0, atol=1e-12):
                raise QuantError("ETF review must retain the audited risk-free series.")
        end = pd.Timestamp(spec.get("evaluation_as_of"))
        if (
            pd.isna(end)
            or end.tzinfo is not None
            or spec["evaluation_as_of"] != end.date().isoformat()
            or end > completed_session(pd.Timestamp(now))
            or market.close.index[-1] != end
            or set(market.close.columns) != set(spec.get("asset_leverage", {}))
            or "SPY" not in market.close
            or any(
                type(value) not in (int, float) or value != 1
                for value in spec["asset_leverage"].values()
            )
        ):
            raise QuantError(
                "Freeze a completed evaluation date and actual unleveraged market inputs."
            )
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.verify()
            signature = digest_json(spec)
            semantic = digest_json({key: value for key, value in spec.items() if key != "id"})
            if self.db.execute(
                "SELECT 1 FROM candidates WHERE id=? OR spec_sha=? OR semantic_sha=?",
                (identifier, signature, semantic),
            ).fetchone():
                raise QuantError(
                    "Candidate already registered; do not erase or re-register its trials."
                )
            self.db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?)",
                (
                    identifier,
                    signature,
                    semantic,
                    now.isoformat(),
                    json.dumps(spec, sort_keys=True, allow_nan=False),
                ),
            )
            record = {
                "candidate_id": identifier,
                "registered_at": now.isoformat(),
                "spec_sha256": signature,
                "readiness_sha256": digest_json(readiness),
                "families": sorted(families),
                "data_scope": scope,
                "order_authority": False,
            }
            self._event("candidate", record, now)
        return record

    def review(
        self,
        candidate_id: str,
        bundle: dict,
        *,
        now: datetime | None = None,
        audit_existing: bool = False,
    ) -> dict:
        now = timestamp(now)
        self.verify()
        candidate = self.db.execute(
            "SELECT * FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if candidate is None:
            raise QuantError("A strategy cannot be evaluated before registration.")
        previous_review = self.db.execute(
            "SELECT body,evidence_sha FROM reviews WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if audit_existing and (
            previous_review is None or previous_review["evidence_sha"] != digest_json(bundle)
        ):
            raise QuantError("Read-only audit requires the unchanged original review evidence.")
        if previous_review is not None and not audit_existing:
            raise QuantError(
                "Preserve the earlier review; changed rules need a new registered version."
            )
        spec = json.loads(candidate["body"])
        for relative, sha in spec["frozen_files"].items():
            safe_file(self.root, relative, sha)
        finish = pd.Timestamp(bundle.get("completed_at"))
        end = pd.Timestamp(bundle.get("as_of"))
        if (
            bundle.get("candidate_spec_sha256") != candidate["spec_sha"]
            or bundle.get("as_of") != spec["evaluation_as_of"]
            or bundle.get("market") != spec["market"]
            or bundle.get("data_scope", "direct_stock") != spec.get("data_scope", "direct_stock")
            or pd.isna(finish)
            or finish.tzinfo is None
            or not pd.Timestamp(candidate["registered_at"]) <= finish <= pd.Timestamp(now)
            or pd.isna(end)
            or end.tzinfo is not None
            or end > completed_session(pd.Timestamp(now))
            or bundle.get("leveraged_products_allowed") is not False
            or bundle.get("order_authority") is not False
            or bundle.get("history_status") != "exposed_history_not_independent_holdout"
        ):
            raise QuantError("Evaluation timing, strategy identity or authority is invalid.")
        records = bundle.get("paths", [])
        pairs = {(item["years"], item["scenario"]) for item in records}
        if len(records) != 4 or pairs != {(y, s) for y in (10, 5) for s in ("base", "stress")}:
            raise QuantError("Review requires both exact horizons under base and stress execution.")
        data = review_market(bundle, self.root)
        if (
            data.close.index[-1] != end
            or set(data.close.columns) != set(spec.get("asset_leverage", {}))
            or "SPY" not in data.close.columns
        ):
            raise QuantError("Market panels differ from the registered universe or endpoint.")
        checked = []
        target_cache = {}
        for item in records:
            years, scenario = item["years"], item["scenario"]
            expected = sessions(end - pd.DateOffset(years=years) + pd.Timedelta(days=1), end)
            goals = self.policy["goals"]
            if (
                item.get("capital_usd") != goals["capital_usd"]
                or item.get("cost_bps") != goals[f"{scenario}_cost_bps"]
                or item.get("delay_sessions") != goals[f"{scenario}_delay_sessions"]
                or item.get("commission_per_order") != goals["commission_per_order"]
            ):
                raise QuantError(
                    "Reviewed costs, capital or execution latency differ from the program."
                )
            frames = {}
            for key in ("strategy", "strategy_bt", "spy", "spy_bt"):
                path = safe_file(self.root, item[key]["path"], item[key]["sha256"])
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                if not frame.index.equals(expected):
                    raise QuantError(
                        "Evaluation omitted sessions or changed the registered horizon."
                    )
                frames[key] = frame
            target_path = safe_file(self.root, item["targets"]["path"], item["targets"]["sha256"])
            targets = pd.read_csv(target_path, index_col=0, parse_dates=True)
            generated = registered_targets(
                spec,
                data,
                str(expected[0].date()),
                str(expected[-1].date()),
                item["cost_bps"],
                item["delay_sessions"],
                self.root,
                target_cache,
            )
            if (
                not targets.index.equals(generated.index)
                or not targets.columns.equals(generated.columns)
                or not targets.isna().equals(generated.isna())
                or not np.allclose(targets, generated, rtol=0, atol=1e-10, equal_nan=True)
            ):
                raise QuantError("Submitted targets do not match the frozen registered strategy.")
            replay = simulate(
                data,
                targets,
                str(expected[0].date()),
                str(expected[-1].date()),
                initial_capital=goals["capital_usd"],
                cost_bps=item["cost_bps"],
                commission=goals["commission_per_order"],
                delay=item["delay_sessions"],
            )
            independent_replay = independent_equity(
                data,
                targets,
                str(expected[0].date()),
                str(expected[-1].date()),
                capital=goals["capital_usd"],
                cost_bps=item["cost_bps"],
                commission=goals["commission_per_order"],
                delay=item["delay_sessions"],
            )
            benchmark_targets = buy_and_hold_signals(data.close, "SPY", str(expected[0].date()))
            benchmark_replay = simulate(
                data,
                benchmark_targets,
                str(expected[0].date()),
                str(expected[-1].date()),
                initial_capital=goals["capital_usd"],
                cost_bps=item["cost_bps"],
                commission=goals["commission_per_order"],
                delay=item["delay_sessions"],
            )
            independent_benchmark = independent_equity(
                data,
                benchmark_targets,
                str(expected[0].date()),
                str(expected[-1].date()),
                capital=goals["capital_usd"],
                cost_bps=item["cost_bps"],
                commission=goals["commission_per_order"],
                delay=item["delay_sessions"],
            )
            for key, actual_frame in (
                ("strategy", replay.frame),
                ("strategy_bt", independent_replay),
                ("spy", benchmark_replay.frame),
                ("spy_bt", independent_benchmark),
            ):
                if not np.allclose(
                    frames[key]["equity"], actual_frame["equity"], rtol=1e-10, atol=1e-6
                ):
                    raise QuantError(
                        "Results do not replay from market data, targets and actual costs."
                    )
            values, error = independent_metrics(
                frames["strategy"],
                frames["strategy_bt"],
                data.risk_free.loc[expected],
                goals["capital_usd"],
            )
            benchmark, benchmark_error = independent_metrics(
                frames["spy"],
                frames["spy_bt"],
                data.risk_free.loc[expected],
                goals["capital_usd"],
            )
            check_metrics(
                values, performance(frames["strategy"]["return"], frames["strategy"]["risk_free"])
            )
            weights_path = safe_file(self.root, item["weights"]["path"], item["weights"]["sha256"])
            weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
            if (
                not weights.index.equals(expected)
                or not np.isfinite(weights.to_numpy()).all()
                or (weights < 0).any().any()
                or (weights.sum(axis=1) > 1 + 1e-10).any()
                or set(weights.columns) != set(spec.get("asset_leverage", {}))
                or any(
                    type(value) not in (int, float) or value != 1
                    for value in spec["asset_leverage"].values()
                )
                or not np.isfinite(frames["strategy"][["cash", "gross_exposure"]].to_numpy()).all()
                or (frames["strategy"]["cash"] < -1e-8).any()
                or not np.allclose(
                    weights.sum(axis=1), frames["strategy"]["gross_exposure"], atol=1e-9, rtol=0
                )
                or not np.allclose(
                    weights.sum(axis=1) + frames["strategy"]["cash"] / frames["strategy"]["equity"],
                    1.0,
                    atol=1e-9,
                    rtol=0,
                )
                or not weights.columns.equals(replay.weights.columns)
                or not np.allclose(weights, replay.weights, rtol=0, atol=1e-10)
            ):
                raise QuantError(
                    "Strategy weights or declared product exposure are not unleveraged."
                )
            gates = {
                "sharpe_above_1": values["sharpe"] is not None and values["sharpe"] > 1,
                "drawdown_at_most_15pct": values["max_drawdown"] <= 0.15,
                "beats_spy": values["cagr"] > benchmark["cagr"],
            }
            checked.append(
                {
                    "years": years,
                    "scenario": scenario,
                    "metrics": values,
                    "benchmark": benchmark,
                    "gates": gates,
                    "max_independent_equity_error_usd": max(error, benchmark_error),
                    "market_and_target_replay_passed": True,
                    "independent_bt_reexecuted": True,
                    "registered_strategy_targets_regenerated": True,
                    "frozen_market_risk_free_used": True,
                }
            )
        for item in checked:
            if item["scenario"] == "stress":
                base = next(
                    row
                    for row in checked
                    if row["years"] == item["years"] and row["scenario"] == "base"
                )
                item["gates"]["also_beats_base_cost_spy"] = (
                    item["metrics"]["cagr"] > base["benchmark"]["cagr"]
                )
        qualified = all(all(item["gates"].values()) for item in checked)
        result = {
            "candidate_id": candidate_id,
            "at": now.isoformat(),
            "evidence_sha256": digest_json(bundle),
            "paths": checked,
            "status": "historical_qualified_awaiting_forward"
            if qualified
            else "rejected_historical",
            "historical_gates_passed": qualified,
            "independent_forward_validation": False,
            "order_authority": False,
            "live_strategy_update": False,
            "data_scope": spec.get("data_scope", "direct_stock"),
        }
        if audit_existing:
            original = json.loads(previous_review["body"])
            if (
                original["historical_gates_passed"] != result["historical_gates_passed"]
                or original["status"] != result["status"]
            ):
                raise QuantError(
                    "Reverification disagrees with the prior outcome; do not rewrite it."
                )
            prior_paths = {(row["years"], row["scenario"]): row for row in original["paths"]}
            for row in result["paths"]:
                prior = prior_paths[row["years"], row["scenario"]]
                check_metrics(row["metrics"], prior["metrics"])
                check_metrics(row["benchmark"], prior["benchmark"])
                if row["gates"] != prior["gates"]:
                    raise QuantError("A prior review gate disagrees with regenerated evidence.")
            return {**result, "read_only_reaudit": True, "original_record_unchanged": True}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.verify()
            if self.db.execute(
                "SELECT 1 FROM reviews WHERE candidate_id=?", (candidate_id,)
            ).fetchone():
                raise QuantError(
                    "Preserve the earlier review; changed rules need a new registered version."
                )
            self._event("review", result, now)
            self.db.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (
                    candidate_id,
                    now.isoformat(),
                    digest_json(bundle),
                    json.dumps(result, sort_keys=True, allow_nan=False),
                ),
            )
        return result

    def audit_reviews(self) -> dict:
        before = self.status()
        results = []
        for previous in before["candidate_reviews"]:
            candidate = previous["candidate_id"]
            bundles = sorted((self.root / "reports").glob(f"*/{candidate}/bundle.json"))
            original = None
            for path in bundles:
                safe_file(self.root, path.relative_to(self.root).as_posix())
                value = read_json(path)
                if digest_json(value) == previous["evidence_sha256"]:
                    original = value
                    break
            if original is None:
                raise QuantError(
                    "Original candidate bundle is unavailable for source-bound re-audit."
                )
            results.append(self.review(candidate, original, audit_existing=True))
        if self.status() != before:
            raise QuantError("Read-only re-audit unexpectedly changed the program ledger.")
        return {
            "schema_version": 1,
            "audited_at": timestamp().isoformat(),
            "engine_sha256": engine_fingerprint(),
            "event_chain_sha256": before["event_chain_sha256"],
            "reviewed_candidates": len(results),
            "regenerated_strategy_paths": sum(len(row["paths"]) for row in results),
            "total_evaluated_configurations": before["total_evaluated_configurations"],
            "ledger_unchanged": True,
            "reviews": results,
            "new_strategy_evaluations": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        }

    def evaluate_registered(self, candidate_id: str, output: Path) -> dict:
        from us_quant.storage import new_output_directory, write_text_atomic

        self.verify()
        candidate = self.db.execute(
            "SELECT body,spec_sha FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if candidate is None:
            raise QuantError("Register the frozen candidate before automatic evaluation.")
        existing = self.db.execute(
            "SELECT body FROM reviews WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if existing is not None:
            return {"evaluation_action": "already_reviewed", "review": json.loads(existing["body"])}
        spec = json.loads(candidate["body"])
        for relative, expected in spec["frozen_files"].items():
            safe_file(self.root, relative, expected)
        data = review_market({"market": spec["market"]}, self.root)
        output = output if output.is_absolute() else self.root / output
        directory = output / candidate_id
        if output.is_symlink() or not directory.resolve().is_relative_to(self.root.resolve()):
            raise QuantError("Automatic research outputs must remain inside the workbench.")
        new_output_directory(directory)
        cache, paths = {}, []
        goals = self.policy["goals"]
        end = pd.Timestamp(spec["evaluation_as_of"])
        for years in goals["horizons_years"]:
            dates = sessions(end - pd.DateOffset(years=years) + pd.Timedelta(days=1), end)
            first, last = str(dates[0].date()), str(dates[-1].date())
            for scenario in goals["scenarios"]:
                cost, delay = goals[f"{scenario}_cost_bps"], goals[f"{scenario}_delay_sessions"]
                targets = registered_targets(spec, data, first, last, cost, delay, self.root, cache)
                own = simulate(
                    data,
                    targets,
                    first,
                    last,
                    initial_capital=goals["capital_usd"],
                    cost_bps=cost,
                    commission=goals["commission_per_order"],
                    delay=delay,
                )
                independent = independent_equity(
                    data,
                    targets,
                    first,
                    last,
                    capital=goals["capital_usd"],
                    cost_bps=cost,
                    commission=goals["commission_per_order"],
                    delay=delay,
                )
                benchmark_targets = buy_and_hold_signals(data.close, "SPY", first)
                spy = simulate(
                    data,
                    benchmark_targets,
                    first,
                    last,
                    initial_capital=goals["capital_usd"],
                    cost_bps=cost,
                    commission=goals["commission_per_order"],
                    delay=delay,
                )
                spy_bt = independent_equity(
                    data,
                    benchmark_targets,
                    first,
                    last,
                    capital=goals["capital_usd"],
                    cost_bps=cost,
                    commission=goals["commission_per_order"],
                    delay=delay,
                )
                record = {
                    "years": years,
                    "scenario": scenario,
                    "capital_usd": goals["capital_usd"],
                    "cost_bps": cost,
                    "commission_per_order": goals["commission_per_order"],
                    "delay_sessions": delay,
                }
                for name, frame in (
                    ("strategy", own.frame),
                    ("strategy_bt", independent),
                    ("spy", spy.frame),
                    ("spy_bt", spy_bt),
                    ("targets", targets),
                    ("weights", own.weights),
                ):
                    path = directory / f"{years}y-{scenario}-{name}.csv"
                    write_text_atomic(path, frame.to_csv(float_format="%.17g"))
                    record[name] = {
                        "path": path.relative_to(self.root).as_posix(),
                        "sha256": file_digest(path),
                    }
                paths.append(record)
        bundle = {
            "candidate_spec_sha256": candidate["spec_sha"],
            "completed_at": timestamp().isoformat(),
            "as_of": spec["evaluation_as_of"],
            "market": spec["market"],
            "data_scope": spec.get("data_scope", "direct_stock"),
            "history_status": spec["history_status"],
            "leveraged_products_allowed": False,
            "order_authority": False,
            "paths": paths,
        }
        write_json(directory / "bundle.json", bundle)
        result = self.review(candidate_id, bundle)
        write_json(directory / "review.json", result)
        return result

    def status(self) -> dict:
        self.verify()
        factors = [
            json.loads(row["body"])
            for row in self.db.execute("SELECT body FROM factors ORDER BY registered_at,id")
        ]
        candidates = [
            json.loads(row["body"])
            for row in self.db.execute("SELECT body FROM candidates ORDER BY registered_at,id")
        ]
        reviews = [
            json.loads(row["body"])
            for row in self.db.execute("SELECT body FROM reviews ORDER BY at,candidate_id")
        ]
        latest = self.db.execute("SELECT result FROM cycles ORDER BY at DESC LIMIT 1").fetchone()
        tail = self.db.execute("SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        qualified = [row for row in reviews if row["historical_gates_passed"]]
        champion, best_score, versions = None, None, []
        for review in qualified:
            score = min(path["metrics"]["sharpe"] for path in review["paths"])
            if best_score is None or score > best_score:
                champion, best_score = review, score
                versions.append(
                    {
                        "version": len(versions) + 1,
                        "candidate_id": review["candidate_id"],
                        "reviewed_at": review["at"],
                        "minimum_historical_sharpe": score,
                        "scope": "research_only_awaiting_independent_forward_evidence",
                    }
                )
        return {
            "schema_version": 1,
            "program_id": self.policy["program_id"],
            "factor_proposals": factors,
            "registered_candidates": candidates,
            "candidate_reviews": reviews,
            "known_factor_definition_count": len(factors),
            "new_factor_proposal_count": len(factors) - len(self.policy["seed_factors"]),
            "economic_factor_family_count": len({factor["family"] for factor in factors}),
            "independent_factor_count_verified": False,
            "new_evaluated_strategy_count": len(reviews),
            "total_evaluated_configurations": 84 + len(reviews),
            "latest_cycle": json.loads(latest["result"]) if latest else None,
            "research_champion": champion["candidate_id"] if champion else None,
            "historical_qualified_candidate_count": len(qualified),
            "research_version_count": len(versions),
            "research_versions": versions,
            "policy_sha256": digest_json(self.policy),
            "event_count": tail["seq"],
            "event_chain_sha256": tail["sha"],
            "prior_high_risk_or_failed_candidates_not_promoted": True,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "automatic_live_deployment": False,
            "order_authority": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recurring evidence-driven factor research; no orders."
    )
    parser.add_argument(
        "action",
        choices=(
            "init",
            "cycle",
            "register-candidate",
            "review",
            "status",
            "migrate-engine",
            "audit-reviews",
            "evaluate-candidate",
        ),
    )
    parser.add_argument("--policy", type=Path, default=Path("config/research-program.json"))
    parser.add_argument("--ledger", type=Path, default=Path("runtime/research-program.sqlite3"))
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--expected-engine-sha")
    parser.add_argument("--expected-event-sha")
    parser.add_argument("--output", type=Path, default=Path("reports/program-evaluations"))
    args = parser.parse_args()
    program = None
    try:
        if args.action == "migrate-engine" and (
            not args.expected_engine_sha or not args.expected_event_sha
        ):
            raise QuantError("Migration needs exact previously observed engine and event hashes.")
        program = ResearchProgram(
            args.ledger,
            read_json(args.policy),
            create=args.action == "init",
            expected_previous_engine=args.expected_engine_sha
            if args.action == "migrate-engine"
            else None,
            expected_previous_event=args.expected_event_sha
            if args.action == "migrate-engine"
            else None,
        )
        if args.action == "cycle":
            if args.proposals is None or args.readiness is None:
                raise QuantError(
                    "Weekly cycles require explicit proposals and data-readiness evidence."
                )
            result = program.cycle(read_json(args.proposals), read_json(args.readiness))
        elif args.action == "register-candidate":
            if args.candidate is None or args.readiness is None:
                raise QuantError(
                    "Candidate registration requires a specification and readiness record."
                )
            result = program.register_candidate(
                read_json(args.candidate), read_json(args.readiness)
            )
        elif args.action == "review":
            if not args.candidate_id or args.bundle is None:
                raise QuantError(
                    "Review requires a registered candidate and hashed accounting bundle."
                )
            result = program.review(args.candidate_id, read_json(args.bundle))
        elif args.action == "audit-reviews":
            result = program.audit_reviews()
        elif args.action == "evaluate-candidate":
            if not args.candidate_id:
                raise QuantError("Automatic evaluation needs a registered candidate identifier.")
            result = program.evaluate_registered(args.candidate_id, args.output)
        else:
            result = program.status()
        if args.export is not None:
            state = result if args.action == "audit-reviews" else program.status()
            if args.export.exists() and read_json(args.export) != state:
                raise QuantError(
                    "A published research snapshot changed; choose a new versioned export."
                )
            if not args.export.exists():
                write_json(args.export, state)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (QuantError, sqlite3.Error) as exc:
        parser.exit(2, f"Research program blocked: {exc}\n")
    finally:
        if program is not None:
            program.close()


if __name__ == "__main__":
    main()
