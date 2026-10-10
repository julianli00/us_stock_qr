from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.paths import ARTIFACT_DIR  # noqa: E402
from src.us_quant.slack_signal_notifier import (  # noqa: E402
    SlackSignalConfig,
    _as_float,
    _first_matching_row,
    _rank_single_stock_candidates,
    _single_action_text,
    _value_from,
    build_legacy_single_stock_artifact,
)


PREFIX = "personal_single_stock_signal_v1"
SIGNAL_PATH = ARTIFACT_DIR / "personal_signal_lifecycle_v1_latest_actions.csv"
STATUS_PATH = ARTIFACT_DIR / "personal_signal_lifecycle_v1_status.json"
OUTPUT_JSON = ARTIFACT_DIR / f"{PREFIX}_status.json"
OUTPUT_CSV = ARTIFACT_DIR / f"{PREFIX}_selected.csv"
RANKING_CSV = ARTIFACT_DIR / f"{PREFIX}_ranking.csv"
OUTPUT_MD = ARTIFACT_DIR / f"{PREFIX}_report.md"
OUTPUT_TXT = ARTIFACT_DIR / f"{PREFIX}_message.txt"
EXECUTION_PACKET_TICKETS = ARTIFACT_DIR / "personal_signal_execution_packet_v1_tickets.csv"
ORDER_PLAN_REVIEW_BOARD = ARTIFACT_DIR / "personal_signal_order_plan_v1_review_board.csv"
CATALYST_EVIDENCE_CURRENT = ARTIFACT_DIR / "personal_signal_catalyst_evidence_v1_current.csv"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"${number:,.2f}"


def _combined_guard_status(ticket: dict[str, Any], review_row: dict[str, Any]) -> str:
    statuses = [
        str(_value_from(ticket.get("catalyst_guard_status"), review_row.get("catalyst_guard_status"), "") or "").lower(),
        str(_value_from(ticket.get("entry_guard_status"), review_row.get("entry_guard_status"), "") or "").lower(),
    ]
    if "blocked" in statuses:
        return "blocked"
    return next((status for status in statuses if status), "n/a")


def _selection_reason(row: dict[str, Any], action_text: str, guard_status: Any) -> str:
    priority = _as_float(row.get("_single_priority"))
    if priority == 0:
        lead = "sell/exit has top delivery priority"
    elif priority == 1:
        lead = "trim has priority over new buy candidates"
    elif priority == 2:
        lead = "highest-ranked executable buy/observe candidate"
    elif priority == 3:
        lead = "buy candidate blocked by catalyst guard, sent as review-only"
    elif priority == 4:
        lead = "buy candidate awaiting execution evidence"
    elif priority == 5:
        lead = "highest-ranked hold/observe update"
    else:
        lead = "fallback-ranked candidate"
    return (
        f"{lead}; action={action_text}; guard={guard_status or 'n/a'}; "
        f"source_score={_as_float(row.get('source_signal_score'))}; "
        f"external_score={_as_float(row.get('external_catalyst_score'))}; "
        f"abs_delta={_as_float(row.get('_abs_delta'))}"
    )


def _build_ranking_board(language: str) -> pd.DataFrame:
    actions = _read_csv(SIGNAL_PATH)
    ranked = _rank_single_stock_candidates(actions)
    if ranked.empty:
        return pd.DataFrame()
    tickets = _read_csv(EXECUTION_PACKET_TICKETS)
    review = _read_csv(ORDER_PLAN_REVIEW_BOARD)
    evidence = _read_csv(CATALYST_EVIDENCE_CURRENT)
    rows: list[dict[str, Any]] = []
    for rank, (_, item) in enumerate(ranked.iterrows(), start=1):
        row = item.to_dict()
        ticker = str(row.get("ticker", "")).upper()
        ticket = _first_matching_row(tickets, ticker)
        review_row = _first_matching_row(review, ticker)
        evidence_row = _first_matching_row(evidence, ticker)
        side = _value_from(ticket.get("side"), review_row.get("side"), review_row.get("requested_side"), "NO_TRADE")
        guard_status = _combined_guard_status(ticket, review_row)
        action_text = _single_action_text(row.get("action"), side, guard_status, language)
        selection_reason = _selection_reason(row, action_text, guard_status)
        rows.append(
            {
                "single_rank": rank,
                "selected": rank == 1,
                "ticker": ticker,
                "action": row.get("action"),
                "action_text": action_text,
                "side": side,
                "order_type": _value_from(ticket.get("ibkr_order_type"), review_row.get("order_type"), "n/a"),
                "selection_priority": _as_float(row.get("_single_priority")),
                "selection_reason": selection_reason,
                "source_signal_score": _as_float(row.get("source_signal_score")),
                "external_catalyst_score": _as_float(row.get("external_catalyst_score")),
                "abs_delta_weight": _as_float(row.get("_abs_delta")),
                "target_weight": _as_float(_value_from(ticket.get("target_weight"), review_row.get("target_weight"), row.get("target_weight"))),
                "delta_weight": _as_float(_value_from(ticket.get("delta_weight"), review_row.get("delta_weight"), row.get("delta_weight"))),
                "notional": _as_float(_value_from(ticket.get("notional"), review_row.get("notional"), row.get("estimated_trade_notional"))),
                "quantity": _as_float(_value_from(ticket.get("quantity"), review_row.get("quantity"), row.get("estimated_qty_at_reference_close"))),
                "limit_price": _as_float(_value_from(ticket.get("limit_price"), review_row.get("reference_limit_price"), row.get("entry_high"), row.get("reference_close_price"))),
                "stop_price": _as_float(_value_from(ticket.get("protective_stop_price"), review_row.get("protective_stop"), row.get("trailing_stop"), row.get("initial_stop"))),
                "take_profit_1": _as_float(_value_from(ticket.get("take_profit_1_limit"), review_row.get("trim_zone_1"), row.get("trim_zone_1"))),
                "take_profit_2": _as_float(_value_from(ticket.get("take_profit_2_limit"), review_row.get("trim_zone_2"), row.get("trim_zone_2"))),
                "catalyst_guard_status": guard_status,
                "catalyst_evidence_status": _value_from(review_row.get("catalyst_evidence_status"), evidence_row.get("evidence_status"), "n/a"),
                "catalyst_recent_7d_count": _as_float(_value_from(review_row.get("catalyst_recent_7d_count"), evidence_row.get("recent_7d_count"))),
                "catalyst_fresh_1d_count": _as_float(_value_from(review_row.get("catalyst_fresh_1d_count"), evidence_row.get("fresh_1d_count"))),
                "catalyst_source_type": _value_from(evidence_row.get("top_source_type"), "n/a"),
                "catalyst_title": str(_value_from(review_row.get("catalyst_top_source_title"), evidence_row.get("top_source_title"), "n/a")).replace("\n", " "),
                "reason": _value_from(row.get("reason"), review_row.get("reason"), "n/a"),
                "source_reason": _value_from(row.get("source_reason"), review_row.get("source_reason"), "n/a"),
            }
        )
    return pd.DataFrame(rows)


def _write_report(status: dict[str, Any], message: str) -> None:
    ranking = _read_csv(RANKING_CSV)
    lines = [
        "# Personal Single Stock Signal V1",
        "",
        f"- Checked at UTC: {status['checked_at_utc']}",
        f"- Selected ticker: {status.get('selected_ticker', 'n/a')}",
        f"- Action: {status.get('selected_action_text', status.get('selected_action', 'n/a'))}",
        f"- Signal date: {status.get('signal_date', 'n/a')}",
        f"- Best quant reference: `{status.get('best_quant_reference', 'n/a')}`",
        f"- Backtest: CAGR {_fmt_pct(status.get('best_reference_cagr'))}, "
        f"MDD {_fmt_pct(status.get('best_reference_mdd'))}, "
        f"profit factor {status.get('best_reference_profit_factor', 'n/a')}",
        f"- Position: target {_fmt_pct(status.get('selected_target_weight'))}, "
        f"delta {_fmt_pct(status.get('selected_delta_weight'))}, "
        f"notional {_fmt_money(status.get('selected_notional'))}, "
        f"quantity {status.get('selected_quantity', 'n/a')}",
        f"- Prices: limit {_fmt_money(status.get('selected_limit_price'))}, "
        f"stop {_fmt_money(status.get('selected_stop_price'))}, "
        f"TP1 {_fmt_money(status.get('selected_take_profit_1'))}, "
        f"TP2 {_fmt_money(status.get('selected_take_profit_2'))}",
        f"- Guard: {status.get('selected_catalyst_guard_status', 'n/a')} - "
        f"{status.get('selected_catalyst_guard_reason', 'n/a')}",
        f"- Catalyst: {status.get('selected_catalyst_evidence_status', 'n/a')}, "
        f"recent_7d={status.get('selected_catalyst_recent_7d_count', 'n/a')}, "
        f"fresh_1d={status.get('selected_catalyst_fresh_1d_count', 'n/a')}, "
        f"{status.get('selected_catalyst_source_type', 'n/a')}: {status.get('selected_catalyst_title', 'n/a')}",
        f"- Reason: {status.get('selected_reason', 'n/a')}; {status.get('selected_source_reason', 'n/a')}",
        f"- Selection reason: {status.get('selected_selection_reason', 'n/a')}",
        f"- Ranking board: `{status.get('ranking_csv', RANKING_CSV)}`",
        f"- Live order allowed: {status.get('live_order_allowed', False)}",
        f"- Final decision: {status.get('final_decision', 'NO LIVE ORDER')}",
        "",
        "## Discord-Ready Message",
        "",
        "```text",
        message.strip(),
        "```",
    ]
    if not ranking.empty:
        lines.extend(["", "## Top Ranked Candidates", ""])
        lines.append("| Rank | Ticker | Action | Priority | Guard | Source | External | Reason |")
        lines.append("| ---: | --- | --- | ---: | --- | ---: | ---: | --- |")
        for _, row in ranking.head(8).iterrows():
            lines.append(
                "| "
                f"{int(row.get('single_rank'))} | "
                f"{row.get('ticker', '')} | "
                f"{row.get('action_text', '')} | "
                f"{row.get('selection_priority', '')} | "
                f"{row.get('catalyst_guard_status', '')} | "
                f"{row.get('source_signal_score', '')} | "
                f"{row.get('external_catalyst_score', '')} | "
                f"{str(row.get('selection_reason', '')).replace('|', '/')} |"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the single-stock signal artifact used by the Discord brief.")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--account-equity", type=float, default=20_000.0)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config = SlackSignalConfig(
        webhook_url="https://hooks.slack.com/services/DRY/RUN/SINGLE",
        signal_path=SIGNAL_PATH,
        status_path=STATUS_PATH,
        top_n=args.top_n,
        account_equity=args.account_equity,
        language=args.language,
        single_stock=True,
    )
    message, digest, meta = build_legacy_single_stock_artifact(config)
    ranking = _build_ranking_board(args.language)
    if not ranking.empty:
        ranking.to_csv(RANKING_CSV, index=False)
        selected_rank_row = ranking.iloc[0].to_dict()
        meta["selected_selection_reason"] = selected_rank_row.get("selection_reason")
        meta["selected_rank"] = int(selected_rank_row.get("single_rank") or 1)
    else:
        pd.DataFrame().to_csv(RANKING_CSV, index=False)
        meta["selected_selection_reason"] = "no_ranked_candidates"
        meta["selected_rank"] = None
    status = {
        "checked_at_utc": _now_utc(),
        "source_family": PREFIX,
        "best_version": f"{PREFIX}_on_{meta.get('best_quant_reference') or 'unknown'}",
        "message_digest": digest,
        **meta,
        "message_txt": str(OUTPUT_TXT),
        "selected_csv": str(OUTPUT_CSV),
        "ranking_csv": str(RANKING_CSV),
        "ranking_rows": int(len(ranking)),
        "report_md": str(OUTPUT_MD),
        "shadow_test_ready": True,
        "live_order_allowed": False,
        "final_decision": "NO LIVE ORDER",
    }
    OUTPUT_TXT.write_text(message.strip() + "\n", encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(_jsonable(status), indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_jsonable(status)]).to_csv(OUTPUT_CSV, index=False)
    _write_report(status, message)
    print(json.dumps(_jsonable(status), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
