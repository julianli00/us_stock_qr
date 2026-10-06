from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path

from us_quant.config import QuantError, load_config
from us_quant.data import fetch_dataset, load_market
from us_quant.paper import (
    IBPaperBroker,
    PaperLedger,
    discover_paper_identity,
    execute_plan,
    load_paper_config,
    now_utc,
    probe_paper_ports,
    save_readonly_paper_config,
)
from us_quant.research import (
    make_signal,
    run_development,
    run_holdout,
    verify_freeze,
    verify_qualification,
)
from us_quant.storage import digest_json, file_digest, read_json, write_json


def redact_account_ids(message: str) -> str:
    return re.sub(
        r"\bD?U[A-Z]?[0-9]{4,}\b",
        lambda match: re.sub(r"[0-9]+", "***" + match.group()[-4:], match.group()),
        message,
    )


class AccountLogRedactor(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        redacted = redact_account_ids(text)
        if redacted != text:
            record.msg, record.args = redacted, ()
        return True


def parser() -> argparse.ArgumentParser:
    from us_quant.dual_horizon import add_parser as add_dual_parser
    from us_quant.evolution import add_parser as add_evolution_parser
    from us_quant.expanded import add_parser
    from us_quant.free_quotes import add_parser as add_free_quote_parser
    from us_quant.fundamentals import add_parser as add_fundamentals_parser
    from us_quant.learned_allocation import add_parser as add_learned_parser
    from us_quant.legacy_horizons import add_parser as add_legacy_horizon_parser
    from us_quant.membership import add_parser as add_membership_parser
    from us_quant.online_reversion import add_parser as add_online_parser
    from us_quant.paper_lab import add_parser as add_lab_parser
    from us_quant.portfolio_protection import add_parser as add_protection_parser
    from us_quant.regime import add_parser as add_regime_parser
    from us_quant.shadow import add_parser as add_shadow_parser

    root = argparse.ArgumentParser(description="Auditable ETF research; IBKR paper only.")
    root.add_argument("--config", type=Path, default=Path("config/research.json"))
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download a fingerprinted, non-overwriting snapshot.")
    fetch.add_argument("--phase", choices=["development", "holdout", "forward"], required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--development", type=Path, default=Path("data/development"))
    fetch.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))
    develop = commands.add_parser(
        "develop", help="Walk-forward selection; freeze before holdout access."
    )
    develop.add_argument("--data", type=Path, default=Path("data/development"))
    develop.add_argument("--output", type=Path, default=Path("reports/development"))
    evaluate = commands.add_parser("evaluate", help="Evaluate the frozen rule without retuning.")
    evaluate.add_argument("--development", type=Path, default=Path("data/development"))
    evaluate.add_argument("--holdout", type=Path, default=Path("data/holdout"))
    evaluate.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))
    evaluate.add_argument("--output", type=Path, default=Path("reports/holdout"))
    signal = commands.add_parser(
        "signal", help="Create a fresh shadow or qualified month-end signal."
    )
    signal.add_argument("--data", type=Path, required=True)
    signal.add_argument("--development", type=Path, default=Path("data/development"))
    signal.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))
    signal.add_argument("--qualification", type=Path, default=Path("reports/holdout/results.json"))
    signal.add_argument("--output", type=Path, required=True)
    doctor = commands.add_parser(
        "doctor", help="Probe paper ports; optionally verify a read-only account."
    )
    doctor_mode = doctor.add_mutually_exclusive_group()
    doctor_mode.add_argument("--paper-config", type=Path)
    doctor_mode.add_argument(
        "--discover-paper",
        action="store_true",
        help="Verify only the identity handshake on listening paper ports; no account data.",
    )
    doctor_mode.add_argument(
        "--save-paper-config",
        type=Path,
        help="Bind one verified paper endpoint locally; leave order submission disabled.",
    )
    observe = commands.add_parser("observe", help="Record a read-only paper account observation.")
    observe.add_argument("--paper-config", type=Path, default=Path("config/paper.json"))
    paper = commands.add_parser(
        "paper", help="Dry-run by default; explicit gated paper submission only."
    )
    paper.add_argument("--paper-config", type=Path, default=Path("config/paper.json"))
    paper.add_argument("--signal", type=Path, default=Path("reports/holdout/latest_signal.json"))
    paper.add_argument("--development", type=Path, default=Path("data/development"))
    paper.add_argument("--freeze", type=Path, default=Path("reports/development/frozen.json"))
    paper.add_argument("--qualification", type=Path, default=Path("reports/holdout/results.json"))
    paper.add_argument(
        "--market-data", type=Path, help="Verified forward snapshot for this signal."
    )
    paper.add_argument("--submit-paper", action="store_true")
    paper.add_argument("--confirm-account")
    add_parser(commands)
    add_fundamentals_parser(commands)
    add_free_quote_parser(commands)
    add_membership_parser(commands)
    add_lab_parser(commands)
    add_shadow_parser(commands)
    add_evolution_parser(commands)
    add_regime_parser(commands)
    add_dual_parser(commands)
    add_legacy_horizon_parser(commands)
    add_learned_parser(commands)
    add_online_parser(commands)
    add_protection_parser(commands)
    return root


def brief_metrics(item: dict) -> dict:
    return {key: item[key] for key in ("start", "end", "cagr", "sharpe", "max_drawdown")}


def dispatch(args: argparse.Namespace) -> dict:
    if args.command == "protection":
        from us_quant.portfolio_protection import dispatch_protection

        return dispatch_protection(args)
    if args.command == "online-reversion":
        from us_quant.online_reversion import dispatch_online

        return dispatch_online(args)
    if args.command == "learned-allocation":
        from us_quant.learned_allocation import dispatch_learned

        return dispatch_learned(args)
    if args.command == "legacy-horizons":
        from us_quant.legacy_horizons import dispatch_legacy

        return dispatch_legacy(args)
    if args.command == "dual-horizon":
        from us_quant.dual_horizon import dispatch_dual

        return dispatch_dual(args)
    if args.command == "regime":
        from us_quant.regime import dispatch_regime

        report = dispatch_regime(args)
        return {
            key: report[key]
            for key in (
                "stage",
                "selected_candidate_id",
                "walk_forward",
                "diagnostic",
                "gates",
                "objective_verified",
                "order_authority",
            )
            if key in report
        }
    if args.command == "paper-lab":
        from us_quant.paper_lab import dispatch_lab

        return dispatch_lab(args)
    if args.command == "free-quote":
        from us_quant.free_quotes import dispatch_free_quote

        return dispatch_free_quote(args)
    if args.command == "observe":
        paper_config = load_paper_config(args.paper_config)
        ledger = PaperLedger(paper_config)
        try:
            with IBPaperBroker(paper_config, readonly=True) as broker:
                snapshot = broker.snapshot()
                observed_at = now_utc()
                ledger.observe(snapshot, observed_at)
                return {
                    "mode": "read_only_paper_observation",
                    "account": "DU***" + paper_config.account[-4:],
                    "observed_at": observed_at.isoformat(),
                    "net_liquidation": snapshot.net_liquidation,
                    "orders_sent": 0,
                    "note": "Observation only; not a daily return or verified track record.",
                }
        finally:
            ledger.close()
    if args.command == "doctor":
        result = probe_paper_ports()
        if args.save_paper_config:
            ports = [
                int(port)
                for port, listening in result["paper_ports_listening"].items()
                if listening
            ]
            if len(ports) != 1:
                raise QuantError("Binding requires one unambiguous listening paper API endpoint.")
            return save_readonly_paper_config(
                args.save_paper_config, Path("config/paper.example.json"), ports[0]
            )
        if args.discover_paper:
            identities = [
                discover_paper_identity(int(port))
                for port, listening in result["paper_ports_listening"].items()
                if listening
            ]
            result["paper_identity_checks"] = identities
            result["paper_identity_verified"] = bool(identities)
        if args.paper_config:
            paper_config = load_paper_config(args.paper_config)
            with IBPaperBroker(paper_config, readonly=True) as broker:
                snapshot = broker.snapshot()
                account = asdict(snapshot)
                account["account"] = "DU***" + paper_config.account[-4:]
                result.update({"account_verified": True, "snapshot": account})
                result["execution_blockers"] = []
                if snapshot.settled_cash is None:
                    result["execution_blockers"].append("settled_usd_cash_not_reported")
                if abs(snapshot.net_liquidation / paper_config.initial_equity_usd - 1) > 0.05:
                    result["execution_blockers"].append("equity_does_not_match_configured_baseline")
                result["configured_submission_enabled"] = paper_config.allow_submit
        result["status"] = (
            "paper_account_verified"
            if result["account_verified"]
            else "paper_identity_verified_account_not_synced"
            if result.get("paper_identity_verified")
            else "port_only_unverified"
            if any(result["paper_ports_listening"].values())
            else "blocked_no_local_paper_service"
        )
        return result
    if args.command == "membership":
        from us_quant.membership import write_membership_snapshot

        report = write_membership_snapshot(args.source, args.as_of, args.output)
        return {
            key: report[key]
            for key in (
                "mode",
                "requested_as_of",
                "effective_source_snapshot",
                "member_count",
                "source_completeness_independently_verified",
                "permanent_security_identity_verified",
                "delisted_price_coverage_verified",
                "stock_strategy_qualified",
                "order_authority",
            )
        }
    if args.command == "fundamentals":
        from us_quant.fundamentals import write_features

        report = write_features(args.facts, args.as_of, args.output)
        return {
            key: report[key]
            for key in (
                "mode",
                "cik",
                "entity",
                "as_of_date",
                "annual_period_end",
                "features",
                "strategy_qualified",
                "order_authority",
            )
        }
    config = load_config(args.config)
    if args.command == "evolve":
        from us_quant.evolution import dispatch_evolution

        return dispatch_evolution(args, config)
    if args.command == "shadow":
        from us_quant.shadow import dispatch_shadow

        return dispatch_shadow(args, config)
    if args.command == "expanded":
        from us_quant.expanded import dispatch_expanded

        report = dispatch_expanded(args, config)
        summary_keys = (
            "stage",
            "selected_candidate_id",
            "cumulative_candidate_trials",
            "gates",
            "historical_numeric_gates",
            "objective_verified",
            "paper_submission_eligible",
            "block_reasons",
        )
        if "stage" in report:
            return {key: report[key] for key in summary_keys if key in report}
        return {
            key: report[key]
            for key in (
                "registered_at",
                "cumulative_candidate_trials",
                "evidence_role",
                "start",
                "end",
            )
            if key in report
        }
    if args.command == "fetch":
        if args.phase in {"holdout", "forward"}:
            verify_freeze(config, args.development, args.freeze)
        result = fetch_dataset(config, args.phase, args.output)
        return {
            "output": str(args.output),
            "phase": args.phase,
            "start": result["start"],
            "end": result["end"],
            "source_count": len(result["sources"]),
        }
    if args.command == "develop":
        result = run_development(config, args.data, args.output)
        return {
            "output": str(args.output),
            "selected_candidate": result["final_selection"]["selected_candidate_id"],
            "walk_forward": brief_metrics(result["walk_forward"]),
            "gates": result["gates"],
        }
    if args.command == "evaluate":
        result = run_holdout(config, args.development, args.holdout, args.freeze, args.output)
        return {
            "output": str(args.output),
            "selected_candidate": result["selected_candidate_id"],
            "holdout": brief_metrics(result["holdout"]),
            "gates": result["gates"],
            "research_gates_passed": result["research_gates_passed"],
            "objective_verified": result["objective_verified"],
        }
    frozen = verify_freeze(config, args.development, args.freeze)
    qualification = verify_qualification(config, args.freeze, args.qualification)
    if args.command == "signal":
        if args.output.exists():
            raise QuantError("Refusing to overwrite an existing signal audit artifact.")
        data = load_market(config, args.data, phase="forward")
        result = make_signal(config, data, frozen, args.freeze, qualification)
        result["market_data_manifest_sha256"] = file_digest(args.data / "manifest.json")
        write_json(args.output, result)
        return result
    if args.command == "paper":
        signal = read_json(args.signal)
        if (
            signal.get("freeze_sha256") != file_digest(args.freeze)
            or signal.get("qualification_sha256") != digest_json(qualification)
            or signal.get("protocol_sha256") != digest_json(config.to_dict())
            or signal.get("strategy_id") != frozen["selected_candidate_id"]
            or signal.get("research_qualified") is not qualification["research_gates_passed"]
        ):
            raise QuantError("Signal is inconsistent with the frozen qualification evidence.")
        if signal.get("executable") is not True:
            raise QuantError(f"Paper order generation blocked: {signal.get('block_reasons')}.")
        if args.market_data is None:
            raise QuantError(
                "Paper execution requires the verified --market-data forward snapshot."
            )
        data = load_market(config, args.market_data, phase="forward")
        if signal.get("market_data_manifest_sha256") != file_digest(
            args.market_data / "manifest.json"
        ):
            raise QuantError("Signal market-data provenance does not match the provided snapshot.")
        recomputed = make_signal(config, data, frozen, args.freeze, qualification)
        for key in (
            "signal_date",
            "execution_session",
            "strategy_id",
            "mode",
            "research_qualified",
            "executable",
            "block_reasons",
            "weights",
            "cash_weight",
        ):
            if signal.get(key) != recomputed[key]:
                raise QuantError(
                    f"Signal {key} differs from recomputed frozen rules; refuse execution."
                )
        paper_config = load_paper_config(args.paper_config)
        ledger = PaperLedger(paper_config)
        try:
            with IBPaperBroker(paper_config, readonly=not args.submit_paper) as broker:
                return execute_plan(
                    paper_config,
                    broker,
                    ledger,
                    signal,
                    submit=args.submit_paper,
                    confirmation=args.confirm_account,
                )
        finally:
            ledger.close()
    raise QuantError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for handler in logging.getLogger().handlers:
        if not any(isinstance(item, AccountLogRedactor) for item in handler.filters):
            handler.addFilter(AccountLogRedactor())
    try:
        result = dispatch(parser().parse_args(argv))
        print(redact_account_ids(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)))
        return 0
    except (QuantError, OSError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        print(f"ERROR: {redact_account_ids(str(exc) or type(exc).__name__)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
