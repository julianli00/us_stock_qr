from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, market_calendar, next_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.dollar_risk_guard import (
    QUARANTINED_DATE,
    parse_release,
    release_calendar,
    vintage_at,
)
from us_quant.dollar_risk_guard import (
    validate_policy as validate_source_policy,
)
from us_quant.growth_factor_satellite import composed_monthly_target
from us_quant.multifactor_stability import FACTORS, corporate_actions
from us_quant.prospective_data import (
    ProspectiveArchive,
    utc_now,
)
from us_quant.prospective_data import (
    fingerprint as parent_fingerprint,
)
from us_quant.prospective_factor_inputs import ExpandedArchive
from us_quant.prospective_factor_inputs import fingerprint as expanded_fingerprint
from us_quant.prospective_research_accounts import ResearchAccounts, aware
from us_quant.prospective_target_observations import TargetJournal
from us_quant.research_daily import run as daily_run
from us_quant.research_program import ResearchProgram, safe_file
from us_quant.storage import digest_json, file_digest, read_json, write_json, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/prospective-dollar-guard.json"
CANDIDATE = "semiconductor_dollar_half_guard"
SOURCE_SPEC_SHA = "1042e7e08f3e7c6de40fe88b509aa9bd8e596a1a92674eeb111f33ab58fb8960"
SYMBOLS = ("SPY", *FACTORS, "GLD", "IEF", "TLT", "BIL", "QQQ", "XLK", "SOXX")
SCENARIOS = [
    {"id": "base", "cost_bps": 5.0, "delay_sessions": 1},
    {"id": "stress", "cost_bps": 20.0, "delay_sessions": 2},
]


def validate_profile(profile: dict) -> None:
    required = {
        "schema_version": 1,
        "profile_id": "prospective_dollar_guard_v1",
        "source_candidate_id": CANDIDATE,
        "source_candidate_spec_sha256": SOURCE_SPEC_SHA,
        "source_registration": {
            "path": "evidence/dollar_risk_guard_20261011_registration.json",
            "sha256": "928cb2954a37db17347df6e7e520e72ebb30beeba606fa35f5905ddfc9bc46b9",
        },
        "required_original_review_status": "rejected_historical",
    }
    inputs = {
        "collector_id": "prospective_dollar_guard_inputs_v1",
        "directory": "data/prospective-dollar-inputs-v1",
        "parent_policy": "config/prospective-data.json",
        "parent_archive": "data/prospective-market-v1",
        "additional_symbols": ["QQQ", "XLK", "SOXX"],
        "additional_macro_series": ["DTWEXBGS"],
        "dollar_calendar_url": "https://www.federalreserve.gov/releases/h10/releaseDates.json",
        "earlier_releases_per_query": 2,
        "maximum_dollar_observation_age_days": 14,
        "inherit_only_matching_completed_session": True,
        "retain_parent_acquisition_times": True,
        "record_before_next_open": True,
        "first_snapshot_is_baseline_only": True,
        "backfill_missed_observations": False,
        "compute_strategy_returns": False,
        "order_authority": False,
    }
    targets = {
        "observer_id": "prospective_dollar_guard_targets_v1",
        "directory": "data/prospective-dollar-targets-v1",
        "initial_target_from_latest_known_completed_month": True,
        "update_only_for_new_completed_month": True,
        "retain_recorded_target_when_no_new_month": True,
        "target_investment_budget": 0.98,
        "hypothetical_base_delay_sessions": 1,
        "hypothetical_stress_delay_sessions": 2,
        "compute_strategy_returns": False,
        "initialize_portfolio": False,
        "submit_orders": False,
        "update_research_champion": False,
        "order_authority": False,
    }
    model = {
        "experiment_id": "prospective_dollar_guard_accounts_v1",
        "directory": "data/prospective-dollar-accounts-v1",
        "source_candidate_id": CANDIDATE,
        "source_historical_status": "rejected_historical",
        "symbols": list(SYMBOLS),
        "capital_usd_per_account": 10000.0,
        "commission_per_order": 1.0,
        "scenarios": SCENARIOS,
        "minimum_observed_sessions_for_reported_sharpe": 63,
        "benchmark_symbol": "SPY",
        "price_encoding": "within_snapshot_ratio_chain",
        "pause_on_unexplained_previous_raw_close_revision": True,
        "require_each_market_session_actual_input_receipt": True,
        "require_each_market_session_target_observation": True,
        "target_must_be_recorded_before_model_execution_open": True,
        "initial_capital_anchor_has_zero_return_and_no_interest": True,
        "model_fills_are_broker_fills": False,
        "backfill_missing_observations": False,
        "automatic_live_deployment": False,
        "update_research_champion": False,
        "order_authority": False,
    }
    if (
        any(profile.get(key) != value for key, value in required.items())
        or profile.get("inputs") != inputs
        or profile.get("targets") != targets
        or profile.get("model") != model
    ):
        raise QuantError(
            "The separate dollar forward profile, costs, sources or authority changed."
        )


def source_reference(profile: dict, root: Path) -> tuple[dict, dict]:
    validate_profile(profile)
    reference = profile["source_registration"]
    registration = read_json(safe_file(root, reference["path"], reference["sha256"]))
    matches = [row["spec"] for row in registration["candidates"] if row["spec"]["id"] == CANDIDATE]
    if len(matches) != 1 or digest_json(matches[0]) != profile["source_candidate_spec_sha256"]:
        raise QuantError("The exact previously registered dollar-only candidate is required.")
    spec = matches[0]
    for path, digest in spec["frozen_files"].items():
        safe_file(root, path, digest)
    policy = read_json(root / "config/dollar-risk-guard.json")
    validate_source_policy(policy)
    program = ResearchProgram(
        root / "runtime/research-program.sqlite3",
        read_json(root / "config/research-program.json"),
        root=root,
    )
    try:
        candidate = program.db.execute(
            "SELECT body FROM candidates WHERE id=?", (CANDIDATE,)
        ).fetchone()
        review = program.db.execute(
            "SELECT body FROM reviews WHERE candidate_id=?", (CANDIDATE,)
        ).fetchone()
        if (
            candidate is None
            or digest_json(json.loads(candidate["body"])) != digest_json(spec)
            or review is None
            or json.loads(review["body"])["status"] != "rejected_historical"
            or json.loads(review["body"])["historical_gates_passed"] is not False
        ):
            raise QuantError(
                "Forward observation cannot silently qualify or replace its source rule."
            )
    finally:
        program.close()
    return spec, policy


def fingerprint(profile: dict, root: Path) -> str:
    folder = Path(__file__).resolve().parent
    reference = profile["source_registration"]
    return digest_json(
        {
            "profile": digest_json(profile),
            "implementation": file_digest(Path(__file__)),
            "parent_collector": parent_fingerprint(),
            "expanded_archive": expanded_fingerprint(),
            "source_registration": file_digest(
                safe_file(root, reference["path"], reference["sha256"])
            ),
            "path_validation": digest_json(inspect.getsource(safe_file)),
            "reused_dependencies": {
                name: file_digest(folder / name)
                for name in (
                    "dollar_risk_guard.py",
                    "growth_factor_satellite.py",
                    "prospective_target_observations.py",
                    "prospective_research_accounts.py",
                    "cash_funded_accounting_v2.py",
                    "backtest.py",
                    "bt_audit.py",
                    "metrics.py",
                    "research_daily.py",
                )
            },
        }
    )


def separate_directory(directory: Path, root: Path, protected: list[Path]) -> Path:
    directory = directory if directory.is_absolute() else root / directory
    if (
        directory.is_symlink()
        or not directory.resolve().is_relative_to(root.resolve())
        or directory.resolve() == root.resolve()
        or any(
            directory.resolve().is_relative_to(path.resolve())
            or path.resolve().is_relative_to(directory.resolve())
            for path in protected
        )
    ):
        raise QuantError("Use a separate in-workbench dollar forward directory.")
    return directory


def read_market(folder: Path, manifest: dict) -> MarketData:
    for name, digest in manifest["files"].items():
        safe_file(folder, name, digest)
    frames = {
        name: pd.read_csv(folder / f"{name}.csv", index_col=0, parse_dates=True)
        for name in ("open", "close", "raw_close", "volume", "risk_free")
    }
    data = MarketData(
        frames["open"],
        frames["close"],
        frames["raw_close"],
        frames["volume"],
        frames["risk_free"]["risk_free"],
    )
    data.validate()
    if set(data.close.columns) != set(SYMBOLS):
        raise QuantError(
            "The prospective dollar panel must retain the complete twelve-fund universe."
        )
    return data


def latest_queries(index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    months = [(i, day) for i, day in enumerate(index) if i >= 63 and is_month_end(day)]
    if not months:
        raise QuantError("A complete risk warmup and completed source month are required.")
    position, month = months[-1]
    reference = index[position - 63]
    if reference <= pd.Timestamp("2019-06-24"):
        raise QuantError(
            "This separately registered forward profile cannot use old dollar backfill."
        )
    return month, reference


def required_releases(calendar: pd.DatetimeIndex, queries: tuple) -> set[pd.Timestamp]:
    selected = set()
    for day in queries:
        earlier = calendar[calendar < day]
        if len(earlier) < 2:
            raise QuantError("The actual H10 archive cannot cover the forward rule's risk queries.")
        selected.update(earlier[-2:])
    if pd.Timestamp(QUARANTINED_DATE) in selected:
        raise QuantError(
            "The quarantined misdated source is not admissible to this forward experiment."
        )
    return selected


def acquire_public(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now().isoformat()
    result = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "10",
            "--max-time",
            "90",
            "--output",
            str(destination),
            "--write-out",
            "%{json}",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=105,
        check=False,
    )
    response = json.loads(result.stdout) if result.stdout else None
    record = {
        "url": url,
        "started_at": started,
        "retrieved_at": utc_now().isoformat(),
        "curl_exit_code": result.returncode,
        "error": result.stderr,
        "response": response,
        "sha256": file_digest(destination) if destination.is_file() else None,
    }
    write_json(destination.with_suffix(destination.suffix + ".request.json"), record)
    if result.returncode or response is None or response["http_code"] != 200:
        raise QuantError("An official H10 request failed; its actual response remains retained.")
    return record


def snapshot_vintages(folder: Path, data: MarketData) -> tuple[pd.DataFrame, dict]:
    manifest = read_json(safe_file(folder, "h10/manifest.json"))
    calendar_record = manifest["calendar"]
    calendar_path = safe_file(folder, calendar_record["path"], calendar_record["sha256"])
    if calendar_record["url"] != "https://www.federalreserve.gov/releases/h10/releaseDates.json":
        raise QuantError("Forward vintages need the exact advertised official H10 calendar.")
    calendar = release_calendar(calendar_path.read_text())
    queries = latest_queries(data.close.index)
    expected = required_releases(calendar, queries)
    frames, dates = [], set()
    for record in manifest["releases"]:
        day = pd.Timestamp(record["date"])
        path = f"h10/{day:%Y%m%d}.html"
        if (
            record["path"] != path
            or record["url"] != f"https://www.federalreserve.gov/releases/h10/{day:%Y%m%d}/"
            or day in dates
            or day not in expected
        ):
            raise QuantError(
                "The dollar input must identify exactly the required official releases."
            )
        document = safe_file(folder, path, record["sha256"]).read_text()
        frame = parse_release(document, day, conservative_dates=True)
        frame["archive_path"], frame["archive_sha256"] = path, record["sha256"]
        frames.append(frame)
        dates.add(day)
    if dates != expected or manifest["rule_month"] != str(queries[0].date()):
        raise QuantError("A needed actual H10 vintage or the frozen query month is missing.")
    vintages = pd.concat(frames, ignore_index=True)
    selected = [vintage_at(vintages, query, "DTWEXBGS") for query in queries]
    if selected[0]["unit_regime"] != selected[1]["unit_regime"]:
        raise QuantError("A forward risk comparison cannot splice different dollar indexation.")
    return vintages, manifest


class DollarInputArchive(ExpandedArchive):
    def __init__(self, directory: Path, profile: dict, *, root: Path = ROOT):
        validate_profile(profile)
        policy = profile["inputs"]
        protected = [
            root / policy["parent_archive"],
            root / "data/prospective-factor-inputs-v2",
            root / "data/prospective-target-observations-v1",
            root / "data/prospective-research-accounts-v1",
        ]
        self.root, self.profile, self.policy = root, profile, policy
        self.directory = separate_directory(directory, root, protected)
        parent_policy = read_json(safe_file(root, policy["parent_policy"]))
        self.parent = ProspectiveArchive(root / policy["parent_archive"], parent_policy)
        self.journal = ProspectiveArchive(self.directory, parent_policy)

    def initialize(self, now=None) -> dict:
        source_reference(self.profile, self.root)
        self.parent.verify()
        record = self.journal.initialize(now)
        record.update(
            {
                "extension_policy_sha256": digest_json(self.policy),
                "extension_collector_sha256": expanded_fingerprint(),
                "parent_archive": self.policy["parent_archive"],
                "parent_registration_sha256": file_digest(
                    self.parent.directory / "registration.json"
                ),
                "extension_mode": "data_only_expanded_factor_inputs",
                "dollar_profile_sha256": fingerprint(self.profile, self.root),
                "source_candidate_id": CANDIDATE,
            }
        )
        write_json(self.directory / "registration.json", record)
        self.verify()
        return record

    def validate_extra_times(
        self, manifest: dict, day: pd.Timestamp, finished: pd.Timestamp
    ) -> None:
        quotes, macro = (
            manifest.get("additional_quote_sources"),
            manifest.get("additional_macro_source"),
        )
        if (
            not isinstance(quotes, list)
            or len(quotes) != 3
            or {row.get("symbol") for row in quotes} != set(self.policy["additional_symbols"])
            or not isinstance(macro, dict)
            or macro.get("series") != "DTWEXBGS"
            or not isinstance(macro.get("acquisitions"), list)
            or len(macro["acquisitions"]) < 3
        ):
            raise QuantError("Dollar inputs must retain three quotes and every actual H10 request.")
        registration = read_json(safe_file(self.directory, "registration.json"))
        earliest = max(
            aware(registration["registered_at"]),
            market_calendar().session_close(day) + pd.Timedelta(minutes=30),
        )
        for record in [*quotes, *macro["acquisitions"]]:
            acquired = aware(record.get("retrieved_at"))
            if not earliest <= acquired <= aware(finished):
                raise QuantError(
                    "An extra dollar input acquisition time is outside its actual window."
                )

    def verify(self) -> list[dict]:
        source_reference(self.profile, self.root)
        records = super().verify()
        registration = read_json(safe_file(self.directory, "registration.json"))
        if (
            registration.get("dollar_profile_sha256") != fingerprint(self.profile, self.root)
            or registration.get("source_candidate_id") != CANDIDATE
        ):
            raise QuantError("The separate dollar input profile or frozen implementation changed.")
        for record in records:
            manifest = read_json(
                safe_file(
                    self.directory,
                    f"{record['snapshot_path']}/manifest.json",
                    record["manifest_sha256"],
                )
            )
            folder = self.directory / record["snapshot_path"]
            data = read_market(folder, manifest)
            _, h10 = snapshot_vintages(folder, data)
            if manifest["additional_macro_source"]["acquisitions"] != [
                h10["calendar"],
                *h10["releases"],
            ]:
                raise QuantError("Recorded H10 requests differ from their actual source seal.")
        return records

    def status(self) -> dict:
        return {
            **super().status(),
            "mode": "data_only_dollar_guard_inputs",
            "source_candidate_id": CANDIDATE,
            "original_macro_experiment_replaced": False,
        }

    def acquire(self, policy: dict, day: pd.Timestamp, output: Path) -> dict:
        receipt, inherited, snapshot = self.parent_snapshot(day)
        reference = self.parent_reference(day)
        for name, digest in inherited["files"].items():
            source = safe_file(snapshot, name, digest)
            destination = output / "parent" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(destination) != digest:
                raise QuantError("An inherited parent input changed while copying.")
        frames = {
            name: pd.read_csv(output / "parent" / f"{name}.csv", index_col=0, parse_dates=True)
            for name in ("open", "close", "raw_close", "volume", "risk_free")
        }
        index = frames["close"].index
        start, end = str(index[0].date()), str(day.date())
        quotes = []
        with requests.Session() as client:
            client.headers["User-Agent"] = "us-stock-qr research (separate forward dollar inputs)"
            for symbol in policy["additional_symbols"]:
                try:
                    response = client.get(
                        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                        params={
                            "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
                            "period2": int(
                                (day.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()
                            ),
                            "interval": "1d",
                            "events": "div,splits",
                        },
                        timeout=(10, 60),
                    )
                    write_text_atomic(output / "extra" / f"{symbol}.json", response.text)
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    raise QuantError(
                        f"Forward quote failed for {symbol}: {type(exc).__name__}, HTTP {status}."
                    ) from exc
                observed = parse_chart(payload, symbol, start, end)
                actions = corporate_actions(payload, observed, symbol)
                if not observed.index.equals(index):
                    raise QuantError(
                        "The additional fund history differs from the inherited sessions."
                    )
                write_text_atomic(
                    output / "extra" / f"{symbol}.csv", observed.to_csv(float_format="%.17g")
                )
                for name, field in (
                    ("open", "adj_open"),
                    ("close", "adj_close"),
                    ("raw_close", "close"),
                    ("volume", "volume"),
                ):
                    frames[name][symbol] = observed[field]
                record = {
                    "symbol": symbol,
                    "url": response.url,
                    "retrieved_at": utc_now().isoformat(),
                    "sessions": len(observed),
                    **actions,
                }
                write_json(output / "extra" / f"{symbol}-request.json", record)
                quotes.append(record)
        month, risk_reference = latest_queries(index)
        calendar_path = output / "h10/calendar.json"
        calendar_request = acquire_public(policy["dollar_calendar_url"], calendar_path)
        calendar = release_calendar(calendar_path.read_text())
        calendar_record = {
            **calendar_request,
            "path": "h10/calendar.json",
            "sha256": file_digest(calendar_path),
        }
        releases = []
        for released in sorted(required_releases(calendar, (month, risk_reference))):
            url = f"https://www.federalreserve.gov/releases/h10/{released:%Y%m%d}/"
            path = output / "h10" / f"{released:%Y%m%d}.html"
            request = acquire_public(url, path)
            parse_release(path.read_text(), released, conservative_dates=True)
            releases.append(
                {
                    **request,
                    "date": str(released.date()),
                    "path": path.relative_to(output).as_posix(),
                    "sha256": file_digest(path),
                }
            )
        write_json(
            output / "h10/manifest.json",
            {
                "rule_month": str(month.date()),
                "reference_session": str(risk_reference.date()),
                "calendar": calendar_record,
                "releases": releases,
                "current_fred_csv_used": False,
            },
        )
        data = MarketData(
            frames["open"],
            frames["close"],
            frames["raw_close"],
            frames["volume"],
            frames["risk_free"]["risk_free"],
        )
        data.validate()
        snapshot_vintages(output, data)
        for name, frame in frames.items():
            write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
        return {
            "session": end,
            "first_price_session": start,
            "price_sessions": len(index),
            "symbols": list(data.close.columns),
            "parent_reference": reference,
            "inherited_quote_sources": inherited["quote_sources"],
            "inherited_macro_sources": inherited["macro_sources"],
            "inherited_option_index_sources": inherited["option_index_sources"],
            "additional_quote_sources": quotes,
            "additional_symbols": policy["additional_symbols"],
            "additional_macro_series": policy["additional_macro_series"],
            "additional_macro_source": {
                "series": "DTWEXBGS",
                "acquisitions": [calendar_record, *releases],
            },
            "parent_observed_at_not_relabelled": receipt["observed_at"],
            "all_required_sources_verified": True,
            "strategy_returns_calculated": False,
        }


def observed_target(data: MarketData, vintages: pd.DataFrame) -> tuple[str, dict, dict]:
    data.validate()
    if set(data.close.columns) != set(SYMBOLS):
        raise QuantError("The frozen dollar rule requires all original and admitted sector funds.")
    month, reference = latest_queries(data.close.index)
    current, prior = (vintage_at(vintages, query, "DTWEXBGS") for query in (month, reference))
    if current["unit_regime"] != prior["unit_regime"]:
        raise QuantError("The recorded target cannot compare different dollar indexation regimes.")
    history = data.close.loc[:month].drop(columns="QQQ").rename(columns={"SOXX": "QQQ"})
    target = composed_monthly_target(history, {"growth_share_of_equity": 0.5})
    target = target.rename(index={"QQQ": "SOXX"}).reindex(data.close.columns, fill_value=0.0)
    rising = current["value"] > prior["value"]
    assets = target.index.drop("BIL")
    if rising:
        removed = float(target.loc[assets].sum()) * 0.5
        target.loc[assets] *= 0.5
        target["BIL"] += removed
    if (target < 0).any() or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12):
        raise QuantError("The actual forward target violated the frozen cash-funded budget.")
    audit = {
        "reference_session": str(reference.date()),
        "dollar_rising": bool(rising),
        "current_release_date": str(current["release_date"].date()),
        "reference_release_date": str(prior["release_date"].date()),
        "current_archive_sha256": current["archive_sha256"],
        "reference_archive_sha256": prior["archive_sha256"],
        "source_month_is_not_actual_generation_time": True,
    }
    return str(month.date()), {name: float(value) for name, value in target.items()}, audit


class DollarTargetJournal(TargetJournal):
    def __init__(self, directory: Path, profile: dict, *, root: Path = ROOT):
        validate_profile(profile)
        self.root, self.profile = root, profile
        self.policy = {
            **profile["targets"],
            "source_candidate_spec_sha256": profile["source_candidate_spec_sha256"],
        }
        self.parent = DollarInputArchive(Path(profile["inputs"]["directory"]), profile, root=root)
        self.directory = separate_directory(
            directory,
            root,
            [self.parent.directory, root / "data/prospective-target-observations-v1"],
        )
        self.journal = ProspectiveArchive(self.directory, self.parent.journal.policy)

    def fingerprint(self) -> str:
        return fingerprint(self.profile, self.root)

    def source_reference(self) -> tuple[dict, dict]:
        return source_reference(self.profile, self.root)

    def validate_target(self, target: dict) -> None:
        weights = target.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(SYMBOLS):
            raise QuantError("The target must specify every instrument in the frozen dollar rule.")
        values = pd.Series(weights, dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 0.98, rtol=0, atol=1e-12)
            or any(values[name] <= 0 for name in (*FACTORS, "SOXX", "GLD"))
            or any(values[name] != 0 for name in ("SPY", "IEF", "TLT", "QQQ", "XLK"))
            or not np.isclose(values.loc[list(FACTORS)].sum(), values["SOXX"], atol=1e-12)
            or not np.allclose(values.loc[list(FACTORS)], values["SOXX"] / 4, atol=1e-12)
            or not (
                np.isclose(values["BIL"], 0, atol=1e-12)
                or np.isclose(values["BIL"], 0.49, atol=1e-12)
            )
        ):
            raise QuantError(
                "The target changed the frozen dollar guard, factor shares or funding."
            )

    def verify(self) -> list[dict]:
        self.source_reference()
        self.parent.verify()
        registration = read_json(safe_file(self.directory, "registration.json"))
        if (
            registration.get("observer_policy_sha256") != digest_json(self.policy)
            or registration.get("observer_mode") != "prospective_target_observations_only"
            or registration.get("observer_sha256") != self.fingerprint()
            or registration.get("parent_registration_sha256")
            != file_digest(self.parent.directory / "registration.json")
            or registration.get("source_candidate_spec_sha256")
            != self.profile["source_candidate_spec_sha256"]
            or registration.get("historical_gates_passed") is not False
        ):
            raise QuantError("The separate dollar target registration or its source rule changed.")
        records = self.journal.verify()
        previous_month, previous_target = None, None
        for record in records:
            manifest = read_json(
                safe_file(
                    self.directory,
                    f"{record['snapshot_path']}/manifest.json",
                    record["manifest_sha256"],
                )
            )
            parent = read_json(
                safe_file(
                    self.parent.directory,
                    f"receipts/{record['session']}.json",
                    manifest["parent_reference"]["receipt_sha256"],
                )
            )
            source = manifest["parent_reference"]
            if (
                source["observed_at"] != parent["observed_at"]
                or source["manifest_sha256"] != parent["manifest_sha256"]
                or aware(source["observed_at"]) > aware(record["observed_at"])
                or manifest["source_candidate_id"] != CANDIDATE
                or any(
                    manifest[name] is not False
                    for name in (
                        "historical_gates_passed",
                        "strategy_returns_calculated",
                        "orders_submitted",
                        "portfolio_initialized",
                    )
                )
            ):
                raise QuantError(
                    "Recorded dollar target provenance or research-only scope changed."
                )
            month, day = (
                pd.Timestamp(manifest["source_rule_signal_session"]),
                pd.Timestamp(record["session"]),
            )
            if (
                month > day
                or not is_month_end(month)
                or (previous_month is not None and month < previous_month)
            ):
                raise QuantError("An unobserved or earlier month cannot retune a forward target.")
            generated = previous_month is None or month > previous_month
            path = safe_file(
                self.directory,
                f"{record['snapshot_path']}/target.json",
                manifest["files"]["target.json"],
            )
            target = read_json(path)
            self.validate_target(target)
            if (
                manifest["new_target_generated"] != generated
                or target["source_rule_signal_session"] != str(month.date())
                or manifest["hypothetical_base_execution_session"]
                != (str(next_session(day).date()) if generated else None)
                or manifest["hypothetical_stress_execution_session"]
                != (str(next_session(next_session(day)).date()) if generated else None)
                or (not generated and file_digest(path) != previous_target)
            ):
                raise QuantError(
                    "The recorded target cadence, weights or execution intent changed."
                )
            if generated:
                parent_manifest = read_json(
                    safe_file(
                        self.parent.directory,
                        f"{parent['snapshot_path']}/manifest.json",
                        parent["manifest_sha256"],
                    )
                )
                source_folder = self.parent.directory / parent["snapshot_path"]
                data = read_market(source_folder, parent_manifest)
                vintages, _ = snapshot_vintages(source_folder, data)
                replayed_month, replayed_weights, replayed_audit = observed_target(data, vintages)
                if (
                    replayed_month != target["source_rule_signal_session"]
                    or target["weights"] != replayed_weights
                    or target.get("source_audit") != replayed_audit
                ):
                    raise QuantError("A target differs from its original captured-source replay.")
            previous_month, previous_target = month, file_digest(path)
        return records

    def status(self) -> dict:
        return {
            **super().status(),
            "source_candidate_id": CANDIDATE,
            "original_macro_experiment_replaced": False,
        }

    def collect(self, clock=utc_now, generator=observed_target) -> dict:
        pending = []

        def capture(_policy, day, output):
            records = self.verify()
            parent, folder = self.parent_snapshot(day)
            source = {
                "receipt_sha256": file_digest(
                    self.parent.directory / "receipts" / f"{day.date()}.json"
                ),
                "manifest_sha256": parent["manifest_sha256"],
                "observed_at": parent["observed_at"],
            }
            manifest = read_json(folder / "manifest.json")
            data = read_market(folder, manifest)
            latest, _ = latest_queries(data.close.index)
            previous = (
                read_json(self.directory / records[-1]["snapshot_path"] / "manifest.json")
                if records
                else None
            )
            if previous is not None and str(latest.date()) < previous["source_rule_signal_session"]:
                raise QuantError("A captured source month cannot move behind a recorded target.")
            generated = (
                previous is None or str(latest.date()) > previous["source_rule_signal_session"]
            )
            if generated:
                vintages, _ = snapshot_vintages(folder, data)
                month, weights, audit = generator(data, vintages)
                if month != str(latest.date()):
                    raise QuantError(
                        "The frozen dollar rule did not produce the latest known month."
                    )
                if (month, weights, audit) != observed_target(data, vintages):
                    raise QuantError("A submitted target differs from the frozen source rule.")
                target = {
                    "source_rule_signal_session": month,
                    "weights": weights,
                    "source_audit": audit,
                }
            else:
                target = read_json(self.directory / records[-1]["snapshot_path"] / "target.json")
            self.validate_target(target)
            write_json(output / "target.json", target)
            pending.append(source["observed_at"])
            return {
                "session": str(day.date()),
                "parent_reference": source,
                "source_candidate_id": CANDIDATE,
                "source_rule_signal_session": str(latest.date()),
                "new_target_generated": generated,
                "hypothetical_base_execution_session": str(next_session(day).date())
                if generated
                else None,
                "hypothetical_stress_execution_session": (
                    str(next_session(next_session(day)).date()) if generated else None
                ),
                "historical_gates_passed": False,
                "all_required_sources_verified": True,
                "strategy_returns_calculated": False,
                "orders_submitted": False,
                "portfolio_initialized": False,
            }

        def completed():
            current = clock()
            if pending and aware(pending.pop()) > aware(current):
                raise QuantError("The target cannot predate its actual captured dollar inputs.")
            return current

        result = self.journal.collect(capture, completed)
        return {"action": result["action"], **self.status()}


def chained_market(snapshots: dict[pd.Timestamp, MarketData]) -> MarketData:
    dates = pd.DatetimeIndex(sorted(snapshots))
    if len(dates) < 2 or not dates.equals(sessions(dates[0], dates[-1])):
        raise QuantError("Do not backfill a missing observed dollar-model market session.")
    columns = pd.Index(SYMBOLS)
    opening = pd.DataFrame(index=dates, columns=columns, dtype=float)
    closing, raw, volume = opening.copy(), opening.copy(), opening.copy()
    rates = pd.Series(index=dates, name="risk_free", dtype=float)
    for i, day in enumerate(dates):
        data = snapshots[day]
        data.validate()
        if set(data.close.columns) != set(SYMBOLS) or data.close.index[-1] != day:
            raise QuantError(
                "Each dollar-model day needs its own complete contemporaneous snapshot."
            )
        raw.loc[day], volume.loc[day] = (
            data.raw_close.loc[day, columns],
            data.volume.loc[day, columns],
        )
        rates.loc[day] = data.risk_free.loc[day]
        if i == 0:
            opening.loc[day] = closing.loc[day] = 100.0
            continue
        previous = dates[i - 1]
        if previous not in data.close.index or not np.allclose(
            data.raw_close.loc[previous, columns], raw.loc[previous], rtol=1e-9, atol=1e-6
        ):
            raise QuantError(
                "The previous raw close was revised or missing; pause the dollar model."
            )
        denominator = data.close.loc[previous, columns]
        opening.loc[day] = closing.loc[previous] * data.open.loc[day, columns] / denominator
        closing.loc[day] = closing.loc[previous] * data.close.loc[day, columns] / denominator
    result = MarketData(opening, closing, raw, volume, rates)
    result.validate()
    return result


def target_signals(
    data: MarketData, instructions: list[dict], registration: dict, delay: int
) -> pd.DataFrame:
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    baseline = pd.Timestamp(registration["baseline_session"])
    for instruction in instructions:
        day = pd.Timestamp(instruction["source_session"])
        if day < baseline or day > data.close.index[-1]:
            continue
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        if max(aware(instruction["generated_at"]), aware(registration["registered_at"])) >= (
            market_calendar().session_open(execution)
        ):
            raise QuantError("A dollar model fill cannot use a target recorded after that open.")
        weights = instruction["weights"]
        if not isinstance(weights, dict) or set(weights) != set(SYMBOLS):
            raise QuantError("A dollar model target must retain every frozen instrument.")
        values = pd.Series(weights, dtype=float).reindex(data.close.columns)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 0.98, rtol=0, atol=1e-12)
            or not targets.loc[day].isna().all()
        ):
            raise QuantError("A dollar model target violates funding or overwrites an instruction.")
        targets.loc[day] = values
    if targets.loc[baseline].isna().any():
        raise QuantError("The dollar model needs its genuinely recorded initial target.")
    return targets


def calculate(
    data: MarketData, instructions: list[dict], registration: dict, profile: dict
) -> tuple[dict, dict]:
    validate_profile(profile)
    policy = profile["model"]
    first, last = str(data.close.index[0].date()), str(data.close.index[-1].date())
    accounts, errors = {}, {}
    for scenario in policy["scenarios"]:
        strategy = target_signals(data, instructions, registration, scenario["delay_sessions"])
        spy = strategy.copy() * np.nan
        spy.loc[data.close.index[0]] = 0.0
        spy.loc[data.close.index[0], "SPY"] = 1.0
        for name, targets in (("strategy", strategy), ("spy", spy)):
            funded = simulate(
                data,
                targets,
                first,
                last,
                initial_capital=10000,
                cost_bps=scenario["cost_bps"],
                commission=1,
                delay=scenario["delay_sessions"],
            )
            independent = independent_equity(
                data,
                targets,
                first,
                last,
                capital=10000,
                cost_bps=scenario["cost_bps"],
                commission=1,
                delay=scenario["delay_sessions"],
            )
            error = float(abs(funded.frame["equity"] - independent["equity"]).max())
            if error > 10000 * 1e-8 or not np.allclose(
                funded.frame["return"], independent["return"], rtol=0, atol=1e-10
            ):
                raise QuantError(
                    "The dollar model does not agree with independent funded accounting."
                )
            identifier = f"{name}_{scenario['id']}"
            accounts[identifier], errors[identifier] = funded, error
    return accounts, errors


class DollarResearchAccounts(ResearchAccounts):
    def __init__(self, directory: Path, profile: dict, *, root: Path = ROOT):
        validate_profile(profile)
        self.root, self.profile, self.policy = root, profile, profile["model"]
        self.targets = DollarTargetJournal(
            Path(profile["targets"]["directory"]), profile, root=root
        )
        self.parent = self.targets.parent
        self.directory = separate_directory(
            directory,
            root,
            [
                self.targets.directory,
                self.parent.directory,
                root / "data/prospective-research-accounts-v1",
            ],
        )

    def fingerprint(self) -> str:
        return fingerprint(self.profile, self.root)

    def replay(self, end: str, now: pd.Timestamp) -> tuple[dict, dict, dict, dict]:
        registration = read_json(self.directory / "registration.json")
        parents = {record["session"]: record for record in self.parent.verify()}
        targets = {record["session"]: record for record in self.targets.verify()}
        snapshots, instructions = {}, []
        for day in sessions(registration["baseline_session"], end):
            key = str(day.date())
            if key not in parents or key not in targets:
                raise QuantError(
                    f"Missing actual dollar input/target receipt for {key}; no backfill."
                )
            if max(aware(parents[key]["observed_at"]), aware(targets[key]["observed_at"])) > now:
                raise QuantError(
                    "The dollar model cannot process sources before their actual acquisition."
                )
            snapshots[day] = self.snapshot(parents[key])
            record = targets[key]
            manifest = read_json(
                safe_file(
                    self.targets.directory,
                    f"{record['snapshot_path']}/manifest.json",
                    record["manifest_sha256"],
                )
            )
            if manifest["new_target_generated"]:
                target = read_json(
                    safe_file(
                        self.targets.directory,
                        f"{record['snapshot_path']}/target.json",
                        manifest["files"]["target.json"],
                    )
                )
                instructions.append(
                    {
                        "source_session": key,
                        "generated_at": record["observed_at"],
                        "weights": target["weights"],
                    }
                )
        accounts, errors = calculate(
            chained_market(snapshots), instructions, registration, self.profile
        )
        return accounts, errors, parents, targets


def components(root: Path, profile: dict) -> tuple:
    return (
        DollarInputArchive(Path(profile["inputs"]["directory"]), profile, root=root),
        DollarTargetJournal(Path(profile["targets"]["directory"]), profile, root=root),
        DollarResearchAccounts(Path(profile["model"]["directory"]), profile, root=root),
    )


def daily_steps(root: Path) -> list:
    profile = read_json(root / "config/prospective-dollar-guard.json")
    inputs, targets, model = components(root, profile)
    return [
        ("parent_data", inputs.parent.collect),
        ("expanded_data", inputs.collect),
        ("target_observations", targets.collect),
        ("research_model", model.advance),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate dollar-guard forward research; never orders."
    )
    parser.add_argument(
        "action",
        choices=(
            "init-inputs",
            "collect-inputs",
            "init-targets",
            "collect-targets",
            "init-model",
            "advance-model",
            "status",
            "daily",
        ),
    )
    parser.add_argument(
        "--origin",
        choices=("agent_continuation", "session_automation", "operator"),
        default="operator",
    )
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    try:
        profile = read_json(POLICY)
        inputs, targets, model = components(ROOT, profile)
        actions = {
            "init-inputs": inputs.initialize,
            "collect-inputs": inputs.collect,
            "init-targets": targets.initialize,
            "collect-targets": targets.collect,
            "init-model": model.initialize,
            "advance-model": model.advance,
            "status": lambda: {
                "inputs": inputs.status(),
                "targets": targets.status(),
                "model": model.status(),
                "investment_objective_verified": False,
                "order_authority": False,
            },
            "daily": lambda: daily_run(
                ROOT / "reports/daily-dollar-guard-operations",
                args.origin,
                root=ROOT,
                step_factory=daily_steps,
            ),
        }
        result = actions[args.action]()
        if args.export:
            export = args.export if args.export.is_absolute() else ROOT / args.export
            if (
                export.exists()
                or export.is_symlink()
                or not export.resolve().is_relative_to(ROOT.resolve())
            ):
                raise QuantError("Preserve an existing forward experiment checkpoint.")
            write_json(export, result)
        print(json.dumps(result, indent=2))
    except (QuantError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Dollar forward research blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
