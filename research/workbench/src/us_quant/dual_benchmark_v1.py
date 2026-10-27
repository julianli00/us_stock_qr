from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.backtest import BacktestResult
from us_quant.bt_audit import independent_equity
from us_quant.calendar import previous_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData, parse_chart
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.metrics import performance
from us_quant.multifactor_stability import corporate_actions
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy import buy_and_hold_signals

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/dual-benchmark-v1.json"
POLICY_SHA256 = "1b5b2572089ffa53746b01e72ff8303e00d6b5e5e3f815844f93834d0521e469"
DEADLINE = datetime(2026, 10, 31, 15, 59, tzinfo=timezone.utc)
REPORTS = ROOT / "reports/ssd-continuous-research-v1"
IDENTITY = ROOT / "data/dual-benchmark-identity-20261027"


def before_deadline(now: datetime) -> None:
    if now.tzinfo is None or now >= DEADLINE:
        raise QuantError("Research authorization expired or the actual time is ambiguous.")


def runtime_guard() -> None:
    before_deadline(datetime.now(timezone.utc))
    expected = os.environ.get("SSD_RESEARCH_ROOT")
    if expected is None:
        raise QuantError("Source the accepted SSD runtime environment before research.")
    base = Path(expected)
    if (
        not base.is_absolute()
        or base.resolve() != base
        or not os.path.ismount(base.parent)
        or ROOT != base / "workspace/research/workbench"
        or Path.cwd().resolve() != base / "workspace"
        or Path(sys.prefix).resolve() != ROOT / ".venv"
    ):
        raise QuantError("The actual SSD mount, workspace or isolated interpreter changed.")
    git = ["git", "--no-pager", "-C", str(base / "workspace")]
    branch = subprocess.check_output(git + ["branch", "--show-current"], text=True).strip()
    directory = subprocess.check_output(
        git + ["rev-parse", "--absolute-git-dir"], text=True
    ).strip()
    if branch != "julianli00-factor-research-handoff" or Path(directory).resolve() != (
        base / "workspace/.git"
    ):
        raise QuantError("Keep the accepted independent SSD Git directory and handoff branch.")


@contextmanager
def research_job(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent:
        raise QuantError("The research lock must remain in its actual private directory.")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QuantError("Another research computation holds the lock; resume it.") from exc
        record = {"pid": os.getpid(), "started_at": utc_now(), "state": "running"}

        def persist() -> None:
            # Replacing a locked inode would allow a second process to bypass its lock.
            stream.seek(0)
            json.dump(record, stream, sort_keys=True)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())

        persist()
        try:
            yield
        finally:
            record.update(state="released", released_at=utc_now())
            persist()
            fcntl.flock(stream, fcntl.LOCK_UN)


class IssuerParagraphs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self.parts: list[str] | None = None
        self.skipped = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.skipped += 1
        if tag == "p" and not self.skipped:
            self.parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skipped = max(0, self.skipped - 1)
        if tag == "p" and self.parts is not None:
            self.paragraphs.append(" ".join("".join(self.parts).split()))
            self.parts = None

    def handle_data(self, data: str) -> None:
        if self.parts is not None and not self.skipped:
            self.parts.append(data)


def issuer_identity(spy_text: str, qqq_html: str) -> dict:
    spy = " ".join(spy_text.replace("\u00ae", "").split())
    page = IssuerParagraphs()
    page.feed(qqq_html)
    if (
        "SPY" not in spy
        or "S&P 500" not in spy
        or "price and yield performance" not in spy
        or "Inception Date 01/22/1993" not in spy
        or not any(
            paragraph.startswith("Invesco QQQ ETF tracks the Nasdaq-100 Index")
            for paragraph in page.paragraphs
        )
    ):
        raise QuantError("Actual primary documents do not establish both benchmark identities.")
    return {
        "SPY": {"benchmark": "S&P 500", "issuer_reported_inception": "1993-01-22"},
        "QQQ": {"benchmark": "Nasdaq-100", "issuer_reported_inception": None},
        "current_primary_objective_verified": True,
        "full_historical_constituents_or_methodology_verified": False,
        "complete_absence_of_internal_fund_derivatives_claimed": False,
    }


def _identity_evidence() -> dict:
    from pypdf import PdfReader

    references = {}
    for symbol, document, receipt, url in (
        (
            "SPY",
            "SPY-factsheet.pdf",
            "SPY-factsheet-request.json",
            "https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spy.pdf",
        ),
        (
            "QQQ",
            "QQQ-issuer.html",
            "QQQ-issuer-request.json",
            "https://www.invesco.com/qqq-etf/en/about.html",
        ),
    ):
        request = read_json(IDENTITY / receipt)
        if (
            request["http_status"] != 200
            or request["curl_exit_code"] != 0
            or request["requested_url"] != url
            or request["effective_url"] != url
            or file_digest(IDENTITY / document) != request["body_sha256"]
        ):
            raise QuantError(f"The retained {symbol} primary-source response changed or failed.")
        references[symbol] = {
            "path": (IDENTITY / document).relative_to(ROOT).as_posix(),
            "sha256": request["body_sha256"],
            "url": request["requested_url"],
            "retrieved_at": request["completed_at_utc"],
        }
    if not (IDENTITY / "SPY-factsheet.pdf").read_bytes().startswith(b"%PDF-"):
        raise QuantError("An HTML page must not be admitted as a PDF factsheet.")
    spy_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(IDENTITY / "SPY-factsheet.pdf").pages
    )
    identity = issuer_identity(spy_text, (IDENTITY / "QQQ-issuer.html").read_text())
    return {
        **identity,
        "primary_sources": references,
        "qqq_factsheet_unavailable": (
            "The advertised PDF URL returned a Country Splash HTML page. It is not PDF "
            "evidence and no country/access gate was bypassed. The independently obtained "
            "official QQQ product page establishes the current Nasdaq-100 objective; "
            "the issuer's inception date is not inferred."
        ),
    }


def admit_source(freeze: dict, policy: dict) -> tuple[MarketData, dict]:
    reference = freeze["market_reference_registration"]
    packet = read_json(safe_file(ROOT, reference["path"], reference["sha256"]))
    entry = packet["candidates"][0]
    if entry["spec"]["market"] != freeze["existing_market_references_byte_verified"]:
        raise QuantError("The pre-outcome freeze references a different market panel.")
    ready = entry["readiness"]
    readiness = read_json(safe_file(ROOT, ready["path"], ready["sha256"]))
    actual = verified_etf_market(readiness, ROOT)
    data = review_market(entry["spec"], ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        left, right = getattr(actual, name), getattr(data, name)
        if (
            not left.index.equals(right.index)
            or not left.columns.equals(right.columns)
            or not np.allclose(left, right, rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("The frozen panel differs from its actual source reconstruction.")
    if not np.allclose(actual.risk_free, data.risk_free, rtol=0, atol=1e-12):
        raise QuantError("The original risk-free series changed.")
    identity = _identity_evidence()
    base = ROOT / "data/factor-round-20261007/base"
    manifest = read_json(base / "manifest.json")
    price_proofs = {}
    for symbol in policy["benchmarks"]["symbols"]:
        raw_name = f"raw/{symbol}.json"
        raw = read_json(
            safe_file(
                ROOT,
                (base / raw_name).relative_to(ROOT).as_posix(),
                manifest["files"][raw_name],
            )
        )
        frame = parse_chart(raw, symbol, manifest["data_start"], manifest["data_end"])
        cached = pd.read_csv(base / f"{symbol}.csv", index_col="date", parse_dates=True)
        if (
            not frame.index.equals(cached.index)
            or not frame.columns.equals(cached.columns)
            or not np.allclose(frame, cached, rtol=1e-12, atol=1e-9)
        ):
            raise QuantError("The benchmark raw response and retained price CSV disagree.")
        for field, column in (
            ("open", "adj_open"),
            ("close", "adj_close"),
            ("raw_close", "close"),
            ("volume", "volume"),
        ):
            if not np.allclose(
                getattr(data, field)[symbol],
                frame.loc[data.close.index, column],
                rtol=1e-10,
                atol=1e-9,
            ):
                raise QuantError("A benchmark was substituted inside the shared market panel.")
        first_trade = raw["chart"]["result"][0]["meta"]["firstTradeDate"]
        if isinstance(first_trade, bool) or not isinstance(first_trade, int):
            raise QuantError("Require the provider's actual first-trade metadata.")
        first_trade_date = pd.Timestamp(first_trade, unit="s", tz="UTC").date()
        if first_trade_date >= data.close.index[0].date():
            raise QuantError("A benchmark panel begins before its supported trading history.")
        price_proofs[symbol] = {
            "raw": {
                "path": (base / raw_name).relative_to(ROOT).as_posix(),
                "sha256": manifest["files"][raw_name],
            },
            "original_retrieved_at": manifest["retrieved_at"],
            "provider_reported_first_trade_date": str(first_trade_date),
            "provider_first_trade_date_is_issuer_inception_proof": False,
            "corporate_actions": corporate_actions(raw, frame, symbol),
            "raw_csv_and_shared_panel_parity": True,
        }
    for window in policy["windows"]:
        dates = data.close.loc[window["start"] : window["end"]].index
        if (
            not dates.equals(sessions(window["start"], window["end"]))
            or len(dates) != window["sessions"]
            or previous_session(dates[0]) not in data.close.index
        ):
            raise QuantError("A complete original benchmark window or causal seed is missing.")
    return data, {
        "recorded_at": utc_now(),
        "protocol_sha256": file_digest(POLICY),
        "market": entry["spec"]["market"],
        "readiness_reference": ready,
        "source_adapter": readiness["verified_etf_source"],
        "identity": identity,
        "benchmark_price_proofs": price_proofs,
        "base_price_manifest": {
            "path": (base / "manifest.json").relative_to(ROOT).as_posix(),
            "sha256": file_digest(base / "manifest.json"),
        },
        "sessions": len(data.close),
        "symbols": list(data.close.columns),
        "new_quote_requests": 0,
        "source_admission_precedes_accounts": True,
        "candidate_returns_or_metrics_computed": False,
        "historical_overlap_and_adaptive_selection": True,
        "order_authority": False,
    }


def execution_statistics(
    result: BacktestResult, data: MarketData, capital: float, cost_bps: float, commission: float
) -> dict:
    frame = result.frame
    dates = frame.index
    opening = data.open.loc[dates]
    units = result.weights.mul(frame["equity"], axis=0) / data.close.loc[dates]
    prior_units = units.shift(1).fillna(0.0)
    prior_cash = frame["cash"].shift(1).fillna(capital)
    pretrade_nav = (prior_units * opening).sum(axis=1) + prior_cash
    active = frame["orders"] > 0
    notional = ((units - prior_units).abs() * opening).sum(axis=1).where(active, 0.0)
    fixed = frame["orders"] * commission
    proportional = notional * cost_bps / 10000
    if (
        (pretrade_nav <= 0).any()
        or not np.allclose(frame["turnover"], notional / pretrade_nav, rtol=0, atol=1e-8)
        or not np.allclose(frame["cost"], fixed + proportional, rtol=0, atol=1e-7)
        or (frame["cash"] < -1e-7).any()
        or (frame["gross_exposure"] > 1 + 1e-10).any()
    ):
        raise QuantError("Actual positions, trade notional, costs or cash funding disagree.")
    elapsed_years = (dates[-1] - previous_session(dates[0])).days / 365.25
    return {
        "total_two_sided_turnover": float(frame["turnover"].sum()),
        "annualized_two_sided_turnover": float(frame["turnover"].sum() / elapsed_years),
        "executed_notional_usd": float(notional.sum()),
        "model_order_count": int(frame["orders"].sum()),
        "rebalance_session_count": int(active.sum()),
        "total_cost_usd": float(frame["cost"].sum()),
        "fixed_commission_usd": float(fixed.sum()),
        "proportional_cost_usd": float(proportional.sum()),
        "initial_deployment_cost_usd": float(frame.loc[active, "cost"].iloc[0])
        if active.any()
        else 0.0,
        "minimum_cash_usd": float(frame["cash"].min()),
        "maximum_gross_exposure": float(frame["gross_exposure"].max()),
        "model_orders_are_broker_orders": False,
    }


def benchmark_account(
    data: MarketData, symbol: str, window: dict, scenario: dict, capital: float
) -> tuple[dict, BacktestResult, pd.DataFrame]:
    signals = buy_and_hold_signals(data.close, symbol, window["start"])
    args = (window["start"], window["end"])
    cost, commission, delay = (
        scenario["cost_bps_per_side"],
        scenario["commission_usd_per_order"],
        scenario["delay_sessions"],
    )
    result = simulate(
        data,
        signals,
        *args,
        initial_capital=capital,
        cost_bps=cost,
        commission=commission,
        delay=delay,
    )
    independent = independent_equity(
        data, signals, *args, capital=capital, cost_bps=cost, commission=commission, delay=delay
    )
    metrics = performance(result.frame["return"], result.frame["risk_free"])
    checked, difference = independent_metrics(
        result.frame, independent, result.frame["risk_free"], capital
    )
    check_metrics(checked, metrics)
    trades = execution_statistics(result, data, capital, cost, commission)
    dates = result.frame.index
    execution = dates[delay - 1]
    units = (capital - commission) / (1 + cost / 10000) / data.open.loc[execution, symbol]
    expected = data.close.loc[dates, symbol] * units
    expected.loc[dates[: delay - 1]] = capital
    if (
        trades["model_order_count"] != 1
        or trades["rebalance_session_count"] != 1
        or result.frame.index[result.frame["orders"] > 0].tolist() != [execution]
        or not np.allclose(expected, result.frame["equity"], rtol=0, atol=1e-6)
    ):
        raise QuantError("The benchmark is not the frozen full-capital single-purchase account.")
    return (
        {
            "symbol": symbol,
            "years": window["years"],
            "scenario": scenario["id"],
            "metrics": metrics,
            "execution": trades,
            "first_execution_session": str(execution.date()),
            "independent_bt_max_equity_difference_usd": difference,
            "closed_form_full_capital_equity_verified": True,
            "benchmark_is_a_candidate_or_independent_forward_validation": False,
        },
        result,
        independent,
    )


def gates(own: dict, spy: dict, qqq: dict, base_spy: dict, scenario: str) -> dict[str, bool]:
    if scenario not in {"base", "stress"}:
        raise QuantError("A complete original cost scenario is required.")
    rows = (own, spy, qqq, base_spy)
    required = {"start", "end", "sessions", "cagr", "sharpe", "max_drawdown"}
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise QuantError("All actual benchmark and strategy metrics are required.")
    if type(own["sessions"]) is not int or (own["start"], own["end"], own["sessions"]) not in {
        ("2016-10-06", "2026-10-05", 2512),
        ("2021-10-06", "2026-10-05", 1254),
    }:
        raise QuantError("Acceptance requires a complete original formal window.")
    if any(
        tuple(row[key] for key in ("start", "end", "sessions"))
        != tuple(own[key] for key in ("start", "end", "sessions"))
        for row in rows
    ):
        raise QuantError("Do not compare different return calendars or windows.")
    for row in rows:
        for key in ("cagr", "max_drawdown", "sharpe"):
            value = row[key]
            if key == "sharpe" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise QuantError("Metric inputs must be actual finite numbers, not placeholders.")
        if not 0 <= row["max_drawdown"] <= 1:
            raise QuantError("Invalid drawdown metric.")
    result = {
        "sharpe_above_1": own["sharpe"] is not None and own["sharpe"] > 1,
        "cagr_above_12pct": own["cagr"] > 0.12,
        "drawdown_at_most_15pct": own["max_drawdown"] <= 0.15,
        "beats_spy": own["cagr"] > spy["cagr"],
        "beats_qqq": own["cagr"] > qqq["cagr"],
    }
    if scenario == "stress":
        result["also_beats_base_cost_spy"] = own["cagr"] > base_spy["cagr"]
    return result


def run_benchmarks(freeze_path: Path, output: Path) -> dict:
    runtime_guard()
    if (
        not output.resolve().is_relative_to(REPORTS)
        or output.resolve() == REPORTS
        or REPORTS.resolve() != REPORTS
        or not freeze_path.resolve().is_relative_to(
            Path(os.environ["SSD_RESEARCH_ROOT"]) / "migration"
        )
        or file_digest(POLICY) != POLICY_SHA256
    ):
        raise QuantError("Use the unchanged frozen protocol and a new private report directory.")
    freeze = read_json(freeze_path)
    if freeze["policy"] != {"path": "config/dual-benchmark-v1.json", "sha256": POLICY_SHA256}:
        raise QuantError("The recorded pre-outcome protocol freeze changed.")
    if (
        freeze["benchmark_accounts_computed"] is not False
        or freeze["new_candidate_accounts_computed"] is not False
        or datetime.fromisoformat(freeze["recorded_at_utc"]) > datetime.now(timezone.utc)
    ):
        raise QuantError("Keep the real protocol freeze before any new account outcomes.")
    for path, digest in freeze["frozen_original_implementation_references"].items():
        safe_file(ROOT, path, digest)
    policy = read_json(POLICY)
    ledger = ROOT / "runtime/research-program.sqlite3"
    original_ledger = file_digest(ledger)
    runner_sha = file_digest(Path(__file__))
    progress_path = REPORTS / "progress.json"
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        new_output_directory(output)
        progress = read_json(progress_path)
        progress.update(
            updated_at_utc=utc_now(),
            phase="admitting_benchmark_sources",
            active_compute_job={
                "pid": os.getpid(),
                "output": str(output),
                "runner_sha256": runner_sha,
            },
            completed_benchmark_paths=[],
        )
        write_json(progress_path, progress)
        completed = []
        try:
            data, admission = admit_source(freeze, policy)
            admission.update(
                runner_sha256=runner_sha, freeze_receipt_sha256=file_digest(freeze_path)
            )
            write_json(output / "admission.json", admission)
            for window in policy["windows"]:
                for scenario in policy["accounts"]["scenarios"]:
                    for symbol in policy["benchmarks"]["symbols"]:
                        runtime_guard()
                        if (
                            file_digest(Path(__file__)) != runner_sha
                            or file_digest(POLICY) != POLICY_SHA256
                        ):
                            raise QuantError(
                                "Benchmark implementation or policy changed during the job."
                            )
                        row, account, independent = benchmark_account(
                            data,
                            symbol,
                            window,
                            scenario,
                            policy["accounts"]["fresh_capital_usd_per_path"],
                        )
                        name = f"{symbol}-{window['years']}y-{scenario['id']}"
                        write_text_atomic(
                            output / f"{name}.csv", account.frame.to_csv(float_format="%.17g")
                        )
                        write_text_atomic(
                            output / f"{name}-bt.csv", independent.to_csv(float_format="%.17g")
                        )
                        write_json(output / f"{name}.json", row)
                        completed.append(row)
                        progress.update(
                            phase="building_dual_benchmarks",
                            updated_at_utc=utc_now(),
                            completed_benchmark_paths=[
                                f"{item['symbol']}-{item['years']}y-{item['scenario']}"
                                for item in completed
                            ],
                        )
                        write_json(progress_path, progress)
            if (
                len(completed) != 8
                or file_digest(ledger) != original_ledger
                or file_digest(POLICY) != POLICY_SHA256
                or file_digest(Path(__file__)) != runner_sha
            ):
                raise QuantError("Incomplete accounts or changed original ledger/policy/code.")
            result = {
                "completed_at": utc_now(),
                "protocol_sha256": POLICY_SHA256,
                "runner_sha256": runner_sha,
                "admission_sha256": file_digest(output / "admission.json"),
                "benchmarks": completed,
                "benchmark_accounts": 8,
                "new_strategy_registrations": 0,
                "new_strategy_evaluations": 0,
                "original_ledger_sha256_unchanged": original_ledger,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_json(output / "results.json", result)
            progress.update(
                phase="dual_benchmarks_complete",
                updated_at_utc=utc_now(),
                active_compute_job=None,
                benchmark_results=str(output / "results.json"),
                pending_steps=[
                    "first_economic_candidate_preregistration",
                    "market_data_only_client_startup_proof",
                ],
            )
            write_json(progress_path, progress)
            return result
        except (QuantError, OSError, ValueError, KeyError, TypeError) as exc:
            write_json(
                output / "failure.json",
                {
                    "recorded_at": utc_now(),
                    "reason": str(exc),
                    "completed_benchmark_paths": len(completed),
                    "partial_outputs_retained": True,
                    "original_ledger_sha256": file_digest(ledger),
                    "new_strategy_evaluations": 0,
                },
            )
            progress.update(
                phase="benchmark_job_failed",
                updated_at_utc=utc_now(),
                active_compute_job=None,
                failure=str(output / "failure.json"),
            )
            write_json(progress_path, progress)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="New source-bound SPY/QQQ benchmark accounts; no broker."
    )
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def offline(event: str, arguments: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("This frozen-source benchmark command must remain offline.")

    sys.addaudithook(offline)
    try:
        result = run_benchmarks(args.freeze, args.output)
        print(
            json.dumps(
                {
                    "benchmark_accounts": result["benchmark_accounts"],
                    "new_strategy_evaluations": 0,
                    "paths": [
                        {
                            key: row[key]
                            for key in ("symbol", "years", "scenario", "metrics", "execution")
                        }
                        for row in result["benchmarks"]
                    ],
                    "investment_objective_verified": False,
                },
                indent=2,
                allow_nan=False,
            )
        )
    except (
        QuantError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(2, f"Dual-benchmark research blocked: {exc}\n")


if __name__ == "__main__":
    main()
