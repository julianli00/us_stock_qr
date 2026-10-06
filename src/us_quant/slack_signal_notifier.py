from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from src.us_quant.paths import ARTIFACT_DIR, ROOT
from src.us_quant.research_delivery import (
    deliver_report,
    post_to_slack as post_research_report,
    validate_webhook,
)


DEFAULT_SIGNAL_PATH = ARTIFACT_DIR / "best_current_latest_signal.csv"
DEFAULT_STATUS_PATH = ARTIFACT_DIR / "user_ibkr_no_leverage_status.json"
DEFAULT_STATE_PATH = ARTIFACT_DIR / "slack_signal_state.json"
DEFAULT_PUSH_LOG_PATH = ARTIFACT_DIR / "slack_signal_push_log.csv"
DEFAULT_TIMEZONE = "America/New_York"
EXECUTION_PACKET_STATUS_PATH = ARTIFACT_DIR / "personal_signal_execution_packet_v1_status.json"
EXECUTION_PACKET_TICKETS_PATH = ARTIFACT_DIR / "personal_signal_execution_packet_v1_tickets.csv"
ORDER_PLAN_STATUS_PATH = ARTIFACT_DIR / "personal_signal_order_plan_v1_status.json"
ORDER_PLAN_REVIEW_BOARD_PATH = ARTIFACT_DIR / "personal_signal_order_plan_v1_review_board.csv"
RESEARCH_QUEUE_STATUS_PATH = ARTIFACT_DIR / "personal_signal_research_queue_v1_status.json"
RESEARCH_QUEUE_PATH = ARTIFACT_DIR / "personal_signal_research_queue_v1_queue.csv"
PRE_TAKEOFF_STATUS_PATH = ARTIFACT_DIR / "personal_signal_pre_takeoff_profile_v1_status.json"
PRE_TAKEOFF_PROFILE_PATH = ARTIFACT_DIR / "personal_signal_pre_takeoff_profile_v1_profile.csv"
QUALITY_TRACKER_STATUS_PATH = ARTIFACT_DIR / "personal_signal_quality_tracker_v1_status.json"
PORTFOLIO_TRACKER_STATUS_PATH = ARTIFACT_DIR / "personal_signal_portfolio_tracker_v1_status.json"
PORTFOLIO_TRACKER_CURRENT_PATH = ARTIFACT_DIR / "personal_signal_portfolio_tracker_v1_current_positions.csv"
EARLY_STATUS_PATH = ARTIFACT_DIR / "early_accumulation_signal_v1_status.json"
SOURCE_MIX_STATUS_PATH = ARTIFACT_DIR / "personal_signal_source_mix_v1_status.json"
SOURCE_MIX_FORWARD_STATUS_PATH = ARTIFACT_DIR / "personal_signal_source_mix_forward_v1_status.json"
CATALYST_EVIDENCE_STATUS_PATH = ARTIFACT_DIR / "personal_signal_catalyst_evidence_v1_status.json"
CATALYST_EVIDENCE_CURRENT_PATH = ARTIFACT_DIR / "personal_signal_catalyst_evidence_v1_current.csv"
BURIED_OPPORTUNITY_STATUS_PATH = ARTIFACT_DIR / "personal_signal_buried_opportunity_v1_status.json"
BURIED_OPPORTUNITY_ROWS_PATH = ARTIFACT_DIR / "personal_signal_buried_opportunity_v1_rows.csv"
BURIED_OPPORTUNITY_ARCHIVE_STATUS_PATH = ARTIFACT_DIR / "personal_signal_buried_opportunity_v1_archive_status.json"


@dataclass(frozen=True)
class SlackSignalConfig:
    webhook_url: str
    signal_path: Path = DEFAULT_SIGNAL_PATH
    status_path: Path = DEFAULT_STATUS_PATH
    state_path: Path = DEFAULT_STATE_PATH
    push_log_path: Path = DEFAULT_PUSH_LOG_PATH
    top_n: int = 12
    account_equity: float = 20_000.0
    timezone_name: str = DEFAULT_TIMEZONE
    include_paper_warning: bool = True
    language: str = "zh"
    single_stock: bool = False
    report_root: Path = ROOT
    channel_alias: str = "canonical-slack"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overwriting existing environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def webhook_from_env(env_var: str = "SLACK_WEBHOOK_URL") -> str:
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise RuntimeError(f"Missing Slack webhook. Set {env_var} in the environment or .env.local.")
    if not value.startswith("https://hooks.slack.com/services/"):
        raise RuntimeError("The Slack webhook URL does not look like a Slack incoming webhook.")
    return value


def is_regular_session(now: datetime | None = None, timezone_name: str = DEFAULT_TIMEZONE) -> bool:
    tz = ZoneInfo(timezone_name)
    local_now = (now or datetime.now(tz)).astimezone(tz)
    if local_now.weekday() >= 5:
        return False
    return time(9, 30) <= local_now.time() <= time(16, 0)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _fmt_price(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"${number:,.2f}"


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _action_counts_text(action_counts: dict[str, int], language: str = "zh") -> str:
    if not action_counts:
        return "none"
    labels = {
        "buy": "买入",
        "buy_watch": "埋伏候选",
        "sell": "卖出",
        "sell_watch": "卖出观察",
        "trim_watch": "减仓观察",
        "hold": "持有",
        "hold_watch": "继续观察",
        "watch": "观察",
    }
    priority = {
        "sell": 0,
        "sell_watch": 1,
        "trim_watch": 2,
        "buy": 3,
        "buy_watch": 4,
        "watch": 5,
        "hold": 6,
        "hold_watch": 7,
    }
    parts: list[str] = []
    for action, count in sorted(action_counts.items(), key=lambda item: priority.get(str(item[0]).lower(), 99)):
        action_key = str(action).lower()
        label = labels.get(action_key, action_key.upper()) if language == "zh" else action_key.upper()
        parts.append(f"{label} {count}")
    return "，".join(parts) if language == "zh" else ", ".join(parts)


def _paper_warning(status: dict[str, Any], language: str = "zh") -> str:
    shadow_ready = bool(status.get("shadow_test_ready"))
    live_ready = bool(status.get("live_trading_ready") or status.get("live_order_allowed"))
    decision = str(status.get("final_decision") or ("LIVE READY" if live_ready else "NO LIVE ORDER"))
    reason = str(status.get("no_live_order_reason") or "").strip()
    if language == "zh":
        mode = "可用于 shadow 跟踪" if shadow_ready else "诊断待验证"
        trade_state = "live-ready" if live_ready else "Paper-only"
        text = (
            f"风险提示：当前为{mode} / {trade_state} 信号，"
            f"shadow_test_ready={_bool_text(shadow_ready)}，"
            f"live_trading_ready={_bool_text(live_ready)}，final_decision={decision}。"
        )
        if not live_ready:
            text += "不要自动实盘下单。"
        if reason:
            text += f"原因：{reason}"
        return text
    text = (
        f"WARNING: current signal is {'shadow-test ready' if shadow_ready else 'diagnostic'} / "
        f"{'live-ready' if live_ready else 'paper-only'}; "
        f"shadow_test_ready={_bool_text(shadow_ready)}, "
        f"live_trading_ready={_bool_text(live_ready)}, final_decision={decision}."
    )
    if not live_ready:
        text += " Do not auto-live-trade."
    if reason:
        text += f" Reason: {reason}"
    return text


def _quality_by_horizon(status: dict[str, Any], horizon: str) -> dict[str, Any]:
    for row in status.get("signal_quality") or []:
        if str(row.get("horizon")) == horizon:
            return row
    return {}


def _metrics_line(status: dict[str, Any], language: str = "zh") -> str:
    if any(k in status for k in ["CAGR", "Sharpe", "MDD", "excess_Sharpe_vs_SPY", "excess_Sharpe_vs_QQQ"]):
        if language == "zh":
            return (
                f"核心指标：CAGR {_fmt_pct(status.get('CAGR'))}，Sharpe {_fmt_float(status.get('Sharpe'))}，"
                f"MDD {_fmt_pct(status.get('MDD'))}，超额 Sharpe vs SPY/QQQ "
                f"{_fmt_float(status.get('excess_Sharpe_vs_SPY'))}/{_fmt_float(status.get('excess_Sharpe_vs_QQQ'))}"
            )
        return (
            f"Metrics: CAGR {_fmt_pct(status.get('CAGR'))}, Sharpe {_fmt_float(status.get('Sharpe'))}, "
            f"MDD {_fmt_pct(status.get('MDD'))}, excess Sharpe SPY/QQQ "
            f"{_fmt_float(status.get('excess_Sharpe_vs_SPY'))}/{_fmt_float(status.get('excess_Sharpe_vs_QQQ'))}"
        )
    q14 = _quality_by_horizon(status, "T+14")
    q3 = _quality_by_horizon(status, "T+3")
    if q14 or q3:
        if language == "zh":
            return (
                "固定周期质量："
                f"T+14 样本 {q14.get('sample', 'n/a')}，胜率 {_fmt_pct(q14.get('win_rate'))}，"
                f"跑赢SPY {_fmt_pct(q14.get('beat_spy_rate'))}，平均超额 {_fmt_pct(q14.get('avg_excess'))}；"
                f"T+3 胜率 {_fmt_pct(q3.get('win_rate'))}，平均超额 {_fmt_pct(q3.get('avg_excess'))}"
            )
        return (
            "Fixed-horizon quality: "
            f"T+14 sample {q14.get('sample', 'n/a')}, win {_fmt_pct(q14.get('win_rate'))}, "
            f"beat SPY {_fmt_pct(q14.get('beat_spy_rate'))}, avg excess {_fmt_pct(q14.get('avg_excess'))}; "
            f"T+3 win {_fmt_pct(q3.get('win_rate'))}, avg excess {_fmt_pct(q3.get('avg_excess'))}"
        )
    return "核心指标：n/a" if language == "zh" else "Metrics: n/a"


def _diagnostic_line(status: dict[str, Any], language: str = "zh") -> str:
    if "candidate_rows_scanned" in status or "screen_pass_count" in status or "shadow_signal_count" in status:
        if language == "zh":
            return (
                f"扫描状态：市场={status.get('market_regime', 'n/a')}，"
                f"允许新仓={status.get('allow_new_positions', 'n/a')}，"
                f"扫描 {status.get('candidate_rows_scanned', 'n/a')}，"
                f"通过 {status.get('screen_pass_count', 'n/a')}，"
                f"信号 {status.get('shadow_signal_count', 'n/a')}"
            )
        return (
            f"Scan: market={status.get('market_regime', 'n/a')}, "
            f"allow_new_positions={status.get('allow_new_positions', 'n/a')}, "
            f"scanned {status.get('candidate_rows_scanned', 'n/a')}, "
            f"passed {status.get('screen_pass_count', 'n/a')}, "
            f"signals {status.get('shadow_signal_count', 'n/a')}"
        )
    if language == "zh":
        return (
            f"失败年份：年度 Sharpe 未过 `{status.get('failed_sharpe_years', 'n/a')}`；"
            f"跑输 SPY `{status.get('fail_spy_years', 'n/a')}`；跑输 QQQ `{status.get('fail_qqq_years', 'n/a')}`"
        )
    return (
        f"Failures: annual Sharpe years `{status.get('failed_sharpe_years', 'n/a')}`, "
        f"excess fail SPY `{status.get('fail_spy_years', 'n/a')}`, QQQ `{status.get('fail_qqq_years', 'n/a')}`"
    )


def _signal_hash(signal: pd.DataFrame, status: dict[str, Any]) -> str:
    tickets = _read_csv(EXECUTION_PACKET_TICKETS_PATH)
    execution_status = _read_json(EXECUTION_PACKET_STATUS_PATH)
    quality_status = _read_json(QUALITY_TRACKER_STATUS_PATH)
    source_mix_status = _read_json(SOURCE_MIX_STATUS_PATH)
    source_mix_forward_status = _read_json(SOURCE_MIX_FORWARD_STATUS_PATH)
    catalyst_status = _read_json(CATALYST_EVIDENCE_STATUS_PATH)
    research_queue_status = _read_json(RESEARCH_QUEUE_STATUS_PATH)
    research_queue = _read_csv(RESEARCH_QUEUE_PATH)
    pre_takeoff_status = _read_json(PRE_TAKEOFF_STATUS_PATH)
    pre_takeoff_profile = _read_csv(PRE_TAKEOFF_PROFILE_PATH)
    buried_status = _read_json(BURIED_OPPORTUNITY_STATUS_PATH)
    buried_archive_status = _read_json(BURIED_OPPORTUNITY_ARCHIVE_STATUS_PATH)
    buried_rows = _read_csv(Path(str(buried_status.get("rows_csv") or BURIED_OPPORTUNITY_ROWS_PATH)))
    portfolio_tracker_status = _read_json(PORTFOLIO_TRACKER_STATUS_PATH)
    portfolio_current = _read_csv(PORTFOLIO_TRACKER_CURRENT_PATH)
    if signal.empty:
        payload = {"signal": [], "status_version": status.get("best_version")}
    else:
        cols = [
            col
            for col in [
                "signal_date",
                "ticker",
                "action",
                "target_weight",
                "delta_weight",
                "estimated_trade_notional",
                "estimated_qty_at_reference_close",
                "reference_close_price",
            ]
            if col in signal.columns
        ]
        payload = {
            "signal": signal[cols].fillna("").to_dict(orient="records"),
            "status_version": status.get("best_version"),
            "shadow_test_ready": status.get("shadow_test_ready"),
        }
    if not tickets.empty:
        ticket_cols = [
            col
            for col in [
                "signal_date",
                "ticker",
                "side",
                "quantity",
                "limit_price",
                "protective_stop_price",
                "take_profit_1_limit",
                "take_profit_2_limit",
                "notional",
                "estimated_stop_risk",
                "live_order_allowed",
            ]
            if col in tickets.columns
        ]
        payload["execution_tickets"] = tickets[ticket_cols].fillna("").to_dict(orient="records")
    payload["execution_packet_version"] = execution_status.get("best_version")
    payload["execution_packet_ticket_count"] = execution_status.get("ticket_count")
    payload["quality_tracker"] = {
        key: quality_status.get(key)
        for key in [
            "checked_at_utc",
            "historical_core_ready",
            "t14_win_rate",
            "t14_avg_excess",
            "source_mix_forward_reason",
            "open_watchlist_rows",
        ]
    }
    payload["source_mix"] = source_mix_status.get("summary", {})
    payload["source_mix_forward"] = source_mix_forward_status.get("summary", {})
    payload["catalyst_evidence"] = catalyst_status.get("summary", {})
    payload["research_queue"] = {
        "status": {
            key: research_queue_status.get(key)
            for key in ["checked_at_utc", "queue_count", "top_queue_tickers", "catalyst_refresh_before_buy_count"]
        },
        "rows": research_queue.fillna("").head(12).to_dict(orient="records") if not research_queue.empty else [],
    }
    payload["pre_takeoff_profile"] = {
        "status": {
            key: pre_takeoff_status.get(key)
            for key in [
                "checked_at_utc",
                "profile_count",
                "top_pre_takeoff_tickers",
                "core_pre_takeoff_count",
                "chase_risk_count",
            ]
        },
        "rows": pre_takeoff_profile.fillna("").head(12).to_dict(orient="records") if not pre_takeoff_profile.empty else [],
    }
    payload["buried_opportunity"] = {
        "status": {
            key: buried_status.get(key)
            for key in [
                "checked_at_utc",
                "row_count",
                "decision_counts",
                "top_review_tickers",
                "top_starter_buy_review_tickers",
                "source_mix_decision_dates",
                "live_order_allowed",
                "final_decision",
            ]
        },
        "archive": {
            key: buried_archive_status.get(key)
            for key in [
                "checked_at_utc",
                "total_archive_rows",
                "new_archive_rows",
                "archive_decision_dates",
                "duplicate_archive_keys",
                "live_order_allowed",
                "final_decision",
            ]
        },
        "rows": buried_rows.fillna("").head(12).to_dict(orient="records") if not buried_rows.empty else [],
    }
    payload["portfolio_tracker"] = {
        "status": {
            key: portfolio_tracker_status.get(key)
            for key in [
                "checked_at_utc",
                "current_model_return",
                "same_weight_spy_return",
                "same_weight_alpha_vs_spy",
                "current_best_ticker",
                "current_worst_ticker",
                "historical_t14_avg_excess_vs_spy",
                "historical_t14_curve_total_return",
                "historical_t14_curve_spy_total_return",
                "historical_t14_curve_excess_multiple",
                "historical_t14_curve_max_drawdown",
            ]
        },
        "rows": portfolio_current.fillna("").head(12).to_dict(orient="records") if not portfolio_current.empty else [],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _top_signal_lines(signal: pd.DataFrame, top_n: int, language: str = "zh") -> list[str]:
    if signal.empty:
        return ["没有找到信号文件记录。"] if language == "zh" else ["No signal file rows found."]
    data = signal.copy()
    review = _read_csv(ORDER_PLAN_REVIEW_BOARD_PATH)
    review_by_ticker: dict[str, dict[str, Any]] = {}
    if not review.empty and "ticker" in review.columns:
        review_by_ticker = {
            str(raw.get("ticker") or "").upper(): raw.to_dict()
            for _, raw in review.iterrows()
            if str(raw.get("ticker") or "").strip()
        }
    if "ticker" in data.columns:
        data["_sort_notional"] = data.apply(
            lambda raw: _value_from(
                review_by_ticker.get(str(raw.get("ticker") or "").upper(), {}).get("notional"),
                raw.get("estimated_trade_notional"),
            ),
            axis=1,
        )
    else:
        data["_sort_notional"] = data.get("estimated_trade_notional", 0.0)
    data["_abs_notional"] = pd.to_numeric(data["_sort_notional"], errors="coerce").abs().fillna(0.0)
    if "action" in data.columns:
        priority = {
            "sell": 0,
            "sell_watch": 0,
            "trim_watch": 1,
            "buy": 2,
            "buy_watch": 2,
            "watch": 3,
            "hold": 4,
            "hold_watch": 4,
        }
        data["_action_priority"] = data["action"].map(priority).fillna(9)
    else:
        data["_action_priority"] = 9
    data = data.sort_values(["_action_priority", "_abs_notional"], ascending=[True, False]).head(top_n)

    lines: list[str] = []
    zh_action = {
        "buy": "买入",
        "buy_watch": "埋伏买入复核",
        "sell": "卖出",
        "sell_watch": "卖出观察",
        "trim_watch": "减仓观察",
        "hold": "持有",
        "hold_watch": "继续观察",
        "watch": "观察",
    }
    for _, row in data.iterrows():
        ticker = str(row.get("ticker", ""))
        review_row = review_by_ticker.get(ticker.upper(), {})
        raw_action = str(row.get("action", "n/a")).lower()
        action = zh_action.get(raw_action, raw_action.upper()) if language == "zh" else raw_action.upper()
        entry_status = str(review_row.get("entry_guard_status") or "").lower()
        catalyst_status = str(review_row.get("catalyst_guard_status") or "").lower()
        requested_side = str(review_row.get("requested_side") or "").upper()
        side = str(review_row.get("side") or "").upper()
        guard_blocked = "blocked" in {entry_status, catalyst_status}
        buy_like = raw_action in {"buy", "buy_watch"} or requested_side == "BUY"
        if buy_like and guard_blocked:
            if entry_status == "blocked":
                action = "等回踩" if language == "zh" else "WAIT PULLBACK"
            else:
                action = "补证据" if language == "zh" else "EVIDENCE REVIEW"
        elif raw_action == "buy_watch" and side == "BUY":
            action = "埋伏买入复核" if language == "zh" else "BUY REVIEW"
        target = _fmt_pct(_value_from(review_row.get("target_weight"), row.get("target_weight")))
        delta = _fmt_pct(_value_from(review_row.get("delta_weight"), row.get("delta_weight")))
        notional = _fmt_money(
            _value_from(
                review_row.get("notional"),
                row.get("estimated_trade_notional"),
            )
        )
        qty = _fmt_float(
            _value_from(
                review_row.get("quantity"),
                row.get("estimated_qty_at_reference_close"),
            ),
            4,
        )
        ref = _fmt_money(_value_from(review_row.get("reference_close_price"), row.get("reference_close_price")))
        guard_note = ""
        if buy_like and guard_blocked:
            reason = str(
                _value_from(
                    review_row.get("entry_guard_reason"),
                    review_row.get("catalyst_guard_reason"),
                    "",
                )
            )
            if reason:
                reason = reason.replace("\n", " ")
                if len(reason) > 74:
                    reason = reason[:73].rstrip() + "…"
                guard_note = f" 原因 {reason}" if language == "zh" else f" reason {reason}"
        if language == "zh":
            lines.append(f"{action:<6} {ticker:<6} 目标 {target:>7} 调仓 {delta:>8} 金额 {notional:>9} 股数 {qty} 参考价 {ref}{guard_note}")
        else:
            lines.append(f"{action:<13} {ticker:<6} target {target:>7} delta {delta:>8} notional {notional:>9} qty {qty} ref {ref}{guard_note}")
    return lines


def _ibkr_snapshot_summary(language: str = "zh") -> str:
    status = _read_json(ARTIFACT_DIR / "ibkr_paper_connection_status.json")
    if not status:
        return "IBKR Paper 快照：暂不可用。" if language == "zh" else "IBKR paper snapshot: not available yet."
    connected = status.get("connected")
    account = status.get("selected_account_masked", "")
    positions = status.get("positions", "n/a")
    open_orders = status.get("open_orders", "n/a")
    executions = status.get("recent_executions", "n/a")
    bidask = f"{status.get('market_snapshot_with_bid_ask', 0)}/{status.get('market_snapshot_symbols', 0)}"
    fill_verified = status.get("actual_broker_fill_verified", False)
    if language == "zh":
        return (
            f"IBKR Paper：连接={connected}，账户={account}，持仓数={positions}，"
            f"未成交订单={open_orders}，近期成交={executions}，bid/ask={bidask}，"
            f"paper成交对账={fill_verified}"
        )
    return (
        f"IBKR paper: connected={connected}, account={account}, positions={positions}, "
        f"open_orders={open_orders}, recent_exec={executions}, bid/ask={bidask}, "
        f"paper_fill_verified={fill_verified}"
    )


def _execution_packet_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(EXECUTION_PACKET_STATUS_PATH)
    if not status:
        text = "Paper执行票：暂未生成。" if language == "zh" else "Paper execution tickets: not generated yet."
        return text, {}
    ticket_count = status.get("ticket_count", "n/a")
    checks = status.get("pre_trade_check_count", "n/a")
    buy_count = status.get("buy_ticket_count", "n/a")
    sell_count = status.get("sell_ticket_count", "n/a")
    buy_notional = _fmt_money(status.get("buy_notional"))
    sell_notional = _fmt_money(status.get("sell_notional"))
    stop_risk = _fmt_money(status.get("total_estimated_stop_risk"))
    stop_risk_pct = _fmt_pct(status.get("total_estimated_stop_risk_pct_equity"))
    if language == "zh":
        text = (
            f"Paper执行票：{ticket_count} 张，盘前检查 {checks} 条；"
            f"buy={buy_count} / sell={sell_count}；买入 {buy_notional}，卖出 {sell_notional}；"
            f"估算止损风险 {stop_risk} ({stop_risk_pct})"
        )
    else:
        text = (
            f"Paper tickets: {ticket_count}, pre-trade checks {checks}; "
            f"buy={buy_count} / sell={sell_count}; buy {buy_notional}, sell {sell_notional}; "
            f"estimated stop risk {stop_risk} ({stop_risk_pct})"
        )
    return text, status


def _quality_tracker_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(QUALITY_TRACKER_STATUS_PATH)
    if not status:
        text = "质量追踪：暂未生成。" if language == "zh" else "Quality tracker: not generated yet."
        return text, {}
    source_mix_ready = bool(status.get("source_mix_forward_ready"))
    source_mix_state = "已验证" if source_mix_ready else "累积中"
    if language == "zh":
        text = (
            f"质量追踪：核心历史={status.get('historical_core_ready', 'n/a')}；"
            f"T+14 样本 {status.get('t14_sample', 'n/a')}，胜率 {_fmt_pct(status.get('t14_win_rate'))}，"
            f"平均超额 {_fmt_pct(status.get('t14_avg_excess'))}；"
            f"当前观察 {status.get('open_watchlist_rows', 'n/a')}，前向胜率 {_fmt_pct(status.get('current_forward_win_rate'))}；"
            f"source-mix {source_mix_state} ({status.get('source_mix_forward_reason', 'n/a')})"
        )
    else:
        text = (
            f"Quality tracker: core_ready={status.get('historical_core_ready', 'n/a')}; "
            f"T+14 sample {status.get('t14_sample', 'n/a')}, win {_fmt_pct(status.get('t14_win_rate'))}, "
            f"avg excess {_fmt_pct(status.get('t14_avg_excess'))}; "
            f"open watchlist {status.get('open_watchlist_rows', 'n/a')}, forward win {_fmt_pct(status.get('current_forward_win_rate'))}; "
            f"source-mix {source_mix_state} ({status.get('source_mix_forward_reason', 'n/a')})"
        )
    return text, status


def _portfolio_tracker_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(PORTFOLIO_TRACKER_STATUS_PATH)
    if not status:
        text = "组合追踪：暂未生成。" if language == "zh" else "Portfolio tracker: not generated yet."
        return text, {}
    if language == "zh":
        text = (
            f"组合追踪：当前模型贡献 {_fmt_pct(status.get('current_model_return'))}，"
            f"同权SPY {_fmt_pct(status.get('same_weight_spy_return'))}，"
            f"alpha {_fmt_pct(status.get('same_weight_alpha_vs_spy'))}；"
            f"gross {_fmt_pct(status.get('current_gross_target_weight'))}；"
            f"最好 {status.get('current_best_ticker', 'n/a')}，最弱 {status.get('current_worst_ticker', 'n/a')}；"
            f"T+14历史超额 {_fmt_pct(status.get('historical_t14_avg_excess_vs_spy'))}；"
            f"cohort曲线 {_fmt_pct(status.get('historical_t14_curve_total_return'))} vs SPY {_fmt_pct(status.get('historical_t14_curve_spy_total_return'))}，"
            f"回撤 {_fmt_pct(status.get('historical_t14_curve_max_drawdown'))}"
        )
    else:
        text = (
            f"Portfolio tracker: model contribution {_fmt_pct(status.get('current_model_return'))}, "
            f"same-weight SPY {_fmt_pct(status.get('same_weight_spy_return'))}, "
            f"alpha {_fmt_pct(status.get('same_weight_alpha_vs_spy'))}; "
            f"gross {_fmt_pct(status.get('current_gross_target_weight'))}; "
            f"best {status.get('current_best_ticker', 'n/a')}, worst {status.get('current_worst_ticker', 'n/a')}; "
            f"T+14 historical excess {_fmt_pct(status.get('historical_t14_avg_excess_vs_spy'))}; "
            f"cohort curve {_fmt_pct(status.get('historical_t14_curve_total_return'))} vs SPY {_fmt_pct(status.get('historical_t14_curve_spy_total_return'))}, "
            f"drawdown {_fmt_pct(status.get('historical_t14_curve_max_drawdown'))}"
        )
    return text, status


def _source_mix_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(SOURCE_MIX_STATUS_PATH)
    forward_status = _read_json(SOURCE_MIX_FORWARD_STATUS_PATH)
    summary = status.get("summary") or {}
    forward_summary = forward_status.get("summary") or {}
    if not summary and not forward_summary:
        text = "来源归因：暂未生成。" if language == "zh" else "Source mix: not generated yet."
        return text, {}
    added = ", ".join(summary.get("full_blend_added_vs_quant_only") or []) or "none"
    dropped = ", ".join(summary.get("quant_only_dropped_by_full_blend") or []) or "none"
    forward_ready = bool(forward_summary.get("forward_attribution_ready"))
    if language == "zh":
        text = (
            f"来源归因：完整混合较纯量化新增 {added}，挤出 {dropped}；"
            f"外部催化命中 {summary.get('final_external_catalyst_count', 'n/a')}，"
            f"v123重合 {summary.get('final_v123_reference_count', 'n/a')}；"
            f"前向归因 ready={forward_ready}，best={forward_summary.get('best_forward_variant') or 'insufficient'}"
        )
    else:
        text = (
            f"Source mix: full blend added {added}, dropped {dropped}; "
            f"external hits {summary.get('final_external_catalyst_count', 'n/a')}, "
            f"v123 overlap {summary.get('final_v123_reference_count', 'n/a')}; "
            f"forward ready={forward_ready}, best={forward_summary.get('best_forward_variant') or 'insufficient'}"
        )
    return text, {"source_mix": summary, "source_mix_forward": forward_summary}


def _catalyst_evidence_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(CATALYST_EVIDENCE_STATUS_PATH)
    summary = status.get("summary") or {}
    if not summary:
        text = "催化证据：暂未生成。" if language == "zh" else "Catalyst evidence: not generated yet."
        return text, {}
    missing_buy = ", ".join(summary.get("missing_recent_buy_external_evidence") or []) or "none"
    missing_all = ", ".join(summary.get("missing_recent_external_evidence") or []) or "none"
    if language == "zh":
        text = (
            f"催化证据：外部预期 {summary.get('external_expected_rows', 'n/a')}，"
            f"近7日 {summary.get('external_with_recent_evidence', 'n/a')}，"
            f"fresh {summary.get('external_with_fresh_evidence', 'n/a')}；"
            f"买入票近7日 {summary.get('buy_external_with_recent_evidence', 'n/a')}/"
            f"{summary.get('buy_external_expected_rows', 'n/a')}，缺口 {missing_buy}；"
            f"Reddit {summary.get('reddit_records', 'n/a')} / News {summary.get('news_records', 'n/a')}"
        )
    else:
        text = (
            f"Catalyst evidence: external expected {summary.get('external_expected_rows', 'n/a')}, "
            f"recent {summary.get('external_with_recent_evidence', 'n/a')}, "
            f"fresh {summary.get('external_with_fresh_evidence', 'n/a')}; "
            f"buy recent {summary.get('buy_external_with_recent_evidence', 'n/a')}/"
            f"{summary.get('buy_external_expected_rows', 'n/a')}, gaps {missing_buy}; "
            f"Reddit {summary.get('reddit_records', 'n/a')} / News {summary.get('news_records', 'n/a')}"
        )
    meta = dict(summary)
    meta["missing_recent_external_evidence_text"] = missing_all
    return text, meta


def _ticker_text(tickers: list[Any], limit: int = 5) -> str:
    clean = [str(ticker).upper().strip() for ticker in tickers if str(ticker or "").strip()]
    return ", ".join(dict.fromkeys(clean[:limit])) or "none"


def _buried_decision_tickers(rows: pd.DataFrame, decision: str, limit: int = 5) -> list[str]:
    if rows.empty or "ticker" not in rows.columns or "opportunity_decision" not in rows.columns:
        return []
    matched = rows[rows["opportunity_decision"].astype(str).eq(decision)].copy()
    if "rank" in matched.columns:
        matched["_rank"] = pd.to_numeric(matched["rank"], errors="coerce")
        matched = matched.sort_values("_rank")
    tickers = matched["ticker"].dropna().astype(str).str.upper().tolist()
    return list(dict.fromkeys(tickers))[:limit]


def _buried_opportunity_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(BURIED_OPPORTUNITY_STATUS_PATH)
    archive_status = _read_json(BURIED_OPPORTUNITY_ARCHIVE_STATUS_PATH)
    rows = _read_csv(Path(str(status.get("rows_csv") or BURIED_OPPORTUNITY_ROWS_PATH)))
    if not status:
        text = "早期埋伏复核：暂未生成。" if language == "zh" else "Early accumulation review: not generated yet."
        return text, {}
    starter = [str(item).upper() for item in (status.get("top_starter_buy_review_tickers") or [])]
    if not starter:
        starter = _buried_decision_tickers(rows, "STARTER_BUY_REVIEW")
    wait = _buried_decision_tickers(rows, "WAIT_PULLBACK_NO_CHASE")
    catalyst = _buried_decision_tickers(rows, "CATALYST_DILIGENCE")
    archive_dates = archive_status.get("archive_decision_dates", "n/a")
    archive_rows = archive_status.get("total_archive_rows", "n/a")
    if language == "zh":
        text = (
            f"早期埋伏复核：rows={status.get('row_count', 'n/a')}；"
            f"首仓复核 {_ticker_text(starter)} ({status.get('starter_buy_review_count', 'n/a')})；"
            f"等回踩 {_ticker_text(wait)} ({status.get('wait_pullback_no_chase_count', 'n/a')})；"
            f"催化尽调 {_ticker_text(catalyst)} ({status.get('catalyst_diligence_count', 'n/a')})；"
            f"归档 {archive_dates} 天 / {archive_rows} 条；final={status.get('final_decision', 'n/a')}"
        )
    else:
        text = (
            f"Early accumulation review: rows={status.get('row_count', 'n/a')}; "
            f"starter {_ticker_text(starter)} ({status.get('starter_buy_review_count', 'n/a')}); "
            f"pullback {_ticker_text(wait)} ({status.get('wait_pullback_no_chase_count', 'n/a')}); "
            f"catalyst diligence {_ticker_text(catalyst)} ({status.get('catalyst_diligence_count', 'n/a')}); "
            f"archive {archive_dates} days / {archive_rows} rows; final={status.get('final_decision', 'n/a')}"
        )
    return text, {
        "status": status,
        "archive": archive_status,
        "starter_tickers": starter,
        "wait_pullback_tickers": wait,
        "catalyst_diligence_tickers": catalyst,
    }


def _top_buried_opportunity_lines(top_n: int, language: str = "zh") -> list[str]:
    status = _read_json(BURIED_OPPORTUNITY_STATUS_PATH)
    rows = _read_csv(Path(str(status.get("rows_csv") or BURIED_OPPORTUNITY_ROWS_PATH)))
    if rows.empty:
        return []
    data = rows.copy()
    for col in ["rank", "buried_opportunity_score", "not_run_score", "source_conviction_score"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "rank" in data.columns:
        data = data.sort_values("rank")
    elif "buried_opportunity_score" in data.columns:
        data = data.sort_values("buried_opportunity_score", ascending=False)
    lines: list[str] = []
    for _, row in data.head(top_n).iterrows():
        ticker = str(row.get("ticker", ""))
        decision = str(row.get("opportunity_decision", "n/a"))
        score = _fmt_float(row.get("buried_opportunity_score"), 1)
        not_run = _fmt_float(row.get("not_run_score"), 2)
        source = _fmt_float(row.get("source_conviction_score"), 2)
        title = str(row.get("top_source_title") or "").replace("\n", " ")
        if len(title) > 58:
            title = title[:57].rstrip() + "…"
        lines.append(f"{ticker:<6} {decision:<24} score={score:<5} quiet={not_run:<4} src={source:<4} {title}")
    return lines


def _top_catalyst_lines(top_n: int, language: str = "zh") -> list[str]:
    evidence = _read_csv(CATALYST_EVIDENCE_CURRENT_PATH)
    if evidence.empty:
        return []
    data = evidence.copy()
    if "action" in data.columns:
        buy_rows = data[data["action"].astype(str).str.lower().isin(["buy", "buy_watch"])].copy()
        if not buy_rows.empty:
            data = buy_rows
    for col in ["external_catalyst_score", "source_signal_score", "latest_catalyst_age_days", "recent_7d_count", "fresh_1d_count"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    sort_cols = [col for col in ["external_catalyst_score", "fresh_1d_count", "recent_7d_count", "source_signal_score"] if col in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    data = data.head(top_n)
    lines: list[str] = []
    for _, row in data.iterrows():
        ticker = str(row.get("ticker", ""))
        status = str(row.get("evidence_status", "n/a"))
        source_type = str(row.get("top_source_type", "n/a"))
        age = row.get("latest_catalyst_age_days")
        age_text = "n/a" if pd.isna(age) else f"{int(float(age))}d"
        title = str(row.get("top_source_title") or "").replace("\n", " ")
        if len(title) > 84:
            title = title[:83].rstrip() + "…"
        score = _fmt_float(row.get("external_catalyst_score"), 3)
        if language == "zh":
            lines.append(f"{ticker:<6} {status:<30} ext={score:<5} age={age_text:<4} {source_type:<6} {title}")
        else:
            lines.append(f"{ticker:<6} {status:<30} ext={score:<5} age={age_text:<4} {source_type:<6} {title}")
    return lines


def _top_execution_ticket_lines(top_n: int, language: str = "zh") -> list[str]:
    tickets = _read_csv(EXECUTION_PACKET_TICKETS_PATH)
    if tickets.empty:
        return []
    data = tickets.copy()
    for col in [
        "quantity",
        "limit_price",
        "protective_stop_price",
        "take_profit_1_limit",
        "take_profit_2_limit",
        "notional",
        "estimated_stop_risk",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "notional" in data.columns:
        data["_abs_notional"] = data["notional"].abs()
    else:
        data["_abs_notional"] = 0.0
    if "side" in data.columns:
        priority = {"SELL": 0, "BUY": 1}
        data["_side_priority"] = data["side"].astype(str).str.upper().map(priority).fillna(9)
    else:
        data["_side_priority"] = 9
    data = data.sort_values(["_side_priority", "_abs_notional"], ascending=[True, False]).head(top_n)
    lines: list[str] = []
    for _, row in data.iterrows():
        side = str(row.get("side", "n/a")).upper()
        ticker = str(row.get("ticker", ""))
        qty = _fmt_float(row.get("quantity"), 4)
        limit_price = _fmt_money(row.get("limit_price"))
        stop = _fmt_money(row.get("protective_stop_price"))
        tp1 = _fmt_money(row.get("take_profit_1_limit"))
        tp2 = _fmt_money(row.get("take_profit_2_limit"))
        notional = _fmt_money(row.get("notional"))
        risk = _fmt_money(row.get("estimated_stop_risk"))
        if language == "zh":
            lines.append(
                f"{side:<4} {ticker:<6} qty {qty:<8} limit {limit_price:>7} stop {stop:>7} "
                f"TP1 {tp1:>7} TP2 {tp2:>7} 金额 {notional:>7} 风险 {risk:>7}"
            )
        else:
            lines.append(
                f"{side:<4} {ticker:<6} qty {qty:<8} limit {limit_price:>7} stop {stop:>7} "
                f"TP1 {tp1:>7} TP2 {tp2:>7} notional {notional:>7} risk {risk:>7}"
            )
    return lines


def _research_queue_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(RESEARCH_QUEUE_STATUS_PATH)
    if not status:
        text = "研究队列：暂未生成。" if language == "zh" else "Research queue: not generated yet."
        return text, {}
    tickers = ", ".join((status.get("top_queue_tickers") or [])[:6]) or "none"
    if language == "zh":
        text = (
            f"研究队列：{status.get('queue_count', 'n/a')} 个；"
            f"买入前补催化 {status.get('catalyst_refresh_before_buy_count', 'n/a')}，"
            f"外部证据刷新 {status.get('external_evidence_refresh_count', 'n/a')}；"
            f"优先 {tickers}"
        )
    else:
        text = (
            f"Research queue: {status.get('queue_count', 'n/a')}; "
            f"pre-buy catalyst refresh {status.get('catalyst_refresh_before_buy_count', 'n/a')}, "
            f"external-evidence refresh {status.get('external_evidence_refresh_count', 'n/a')}; "
            f"priority {tickers}"
        )
    return text, status


def _top_research_queue_lines(top_n: int, language: str = "zh") -> list[str]:
    queue = _read_csv(RESEARCH_QUEUE_PATH)
    if queue.empty:
        return []
    data = queue.copy()
    for col in ["priority_score", "requested_notional", "paper_notional", "catalyst_recent_7d_count", "catalyst_latest_age_days"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "rank" in data.columns:
        data = data.sort_values("rank")
    elif "priority_score" in data.columns:
        data = data.sort_values("priority_score", ascending=False)
    data = data.head(top_n)
    lines: list[str] = []
    for _, row in data.iterrows():
        ticker = str(row.get("ticker", ""))
        task = str(row.get("research_task", "n/a"))
        evidence = str(row.get("catalyst_evidence_status", "n/a"))
        recent = row.get("catalyst_recent_7d_count", "n/a")
        age = row.get("catalyst_latest_age_days", "n/a")
        note = str(row.get("manual_research_note") or "").replace("\n", " ")
        if len(note) > 46:
            note = note[:45].rstrip() + "…"
        if language == "zh":
            lines.append(
                f"{ticker:<6} {task:<30} recent={recent!s:<3} age={age!s:<4} {evidence:<32} {note}"
            )
        else:
            lines.append(
                f"{ticker:<6} {task:<30} recent={recent!s:<3} age={age!s:<4} {evidence:<32} {note}"
            )
    return lines


def _pre_takeoff_line(language: str = "zh") -> tuple[str, dict[str, Any]]:
    status = _read_json(PRE_TAKEOFF_STATUS_PATH)
    if not status:
        text = "起飞前评分：暂未生成。" if language == "zh" else "Pre-takeoff profile: not generated yet."
        return text, {}
    tickers = ", ".join((status.get("top_pre_takeoff_tickers") or [])[:6]) or "none"
    avg = _fmt_float(status.get("average_pre_takeoff_score"), 1)
    if language == "zh":
        text = (
            f"起飞前评分：{status.get('profile_count', 'n/a')} 个；"
            f"核心埋伏 {status.get('core_pre_takeoff_count', 'n/a')}，"
            f"追高警示 {status.get('chase_risk_count', 'n/a')}；"
            f"均分 {avg}；优先 {tickers}"
        )
    else:
        text = (
            f"Pre-takeoff profile: {status.get('profile_count', 'n/a')}; "
            f"core {status.get('core_pre_takeoff_count', 'n/a')}, "
            f"chase flags {status.get('chase_risk_count', 'n/a')}; "
            f"avg {avg}; priority {tickers}"
        )
    return text, status


def _top_pre_takeoff_lines(top_n: int, language: str = "zh") -> list[str]:
    profile = _read_csv(PRE_TAKEOFF_PROFILE_PATH)
    if profile.empty:
        return []
    data = profile.copy()
    for col in ["rank", "pre_takeoff_score", "mom_20d", "gap_to_52w_high", "rel_spy_20d"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "rank" in data.columns:
        data = data.sort_values("rank")
    else:
        data = data.sort_values("pre_takeoff_score", ascending=False)
    lines: list[str] = []
    for _, row in data.head(top_n).iterrows():
        ticker = str(row.get("ticker", ""))
        raw_action = str(row.get("action", "n/a")).lower()
        score = _fmt_float(row.get("pre_takeoff_score"), 1)
        stage = str(row.get("pre_takeoff_stage", "n/a"))
        chase = "Y" if bool(row.get("chase_risk_flag")) else "N"
        if stage == "wait_for_pullback_no_chase":
            action = "等回踩" if language == "zh" else "WAIT"
        elif raw_action == "buy_watch":
            action = "埋伏复核" if language == "zh" else "BUY_REVIEW"
        elif raw_action == "hold_watch":
            action = "继续观察" if language == "zh" else "OBSERVE"
        else:
            action = raw_action.upper()
        if language == "zh":
            lines.append(
                f"{ticker:<6} score={score:<5} {action:<10} chase={chase} "
                f"20d={_fmt_pct(row.get('mom_20d')):<8} gap={_fmt_pct(row.get('gap_to_52w_high')):<8} "
                f"relSPY={_fmt_pct(row.get('rel_spy_20d')):<8} {stage}"
            )
        else:
            lines.append(
                f"{ticker:<6} score={score:<5} {action:<10} chase={chase} "
                f"20d={_fmt_pct(row.get('mom_20d')):<8} gap={_fmt_pct(row.get('gap_to_52w_high')):<8} "
                f"relSPY={_fmt_pct(row.get('rel_spy_20d')):<8} {stage}"
            )
    return lines


def _first_matching_row(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if frame.empty or "ticker" not in frame.columns:
        return {}
    matched = frame[frame["ticker"].astype(str).str.upper().eq(str(ticker).upper())]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _value_from(*items: Any) -> Any:
    for item in items:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(item, str) and not item.strip():
            continue
        return item
    return None


def _rank_single_stock_candidates(signal: pd.DataFrame) -> pd.DataFrame:
    if signal.empty:
        return signal
    data = signal.copy()
    tickets = _read_csv(EXECUTION_PACKET_TICKETS_PATH)
    review = _read_csv(ORDER_PLAN_REVIEW_BOARD_PATH)
    executable = set()
    if not tickets.empty and "ticker" in tickets.columns:
        executable = set(tickets["ticker"].astype(str).str.upper())
    guard_status: dict[str, str] = {}
    if not review.empty and "ticker" in review.columns:
        for _, raw in review.iterrows():
            ticker_key = str(raw.get("ticker") or "").upper()
            catalyst_status = str(raw.get("catalyst_guard_status") or "").lower()
            entry_status = str(raw.get("entry_guard_status") or "").lower()
            if ticker_key:
                guard_status[ticker_key] = "blocked" if "blocked" in {catalyst_status, entry_status} else catalyst_status or entry_status
    for col in ["estimated_trade_notional", "source_signal_score", "external_catalyst_score", "delta_weight"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    action = data["action"].astype(str).str.lower() if "action" in data.columns else pd.Series("", index=data.index)
    ticker = data["ticker"].astype(str).str.upper() if "ticker" in data.columns else pd.Series("", index=data.index)
    base_priority = action.map(
        {
            "sell": 0,
            "sell_watch": 0,
            "trim_watch": 1,
            "buy": 2,
            "buy_watch": 2,
            "hold": 5,
            "hold_watch": 5,
            "watch": 6,
        }
    ).fillna(9)
    is_executable = ticker.isin(executable)
    is_blocked = ticker.map(lambda item: guard_status.get(item, "") == "blocked")
    buy_like = action.isin(["buy", "buy_watch"])
    data["_single_priority"] = base_priority
    data.loc[buy_like & is_executable, "_single_priority"] = 2
    data.loc[buy_like & ~is_executable & is_blocked, "_single_priority"] = 3
    data.loc[buy_like & ~is_executable & ~is_blocked, "_single_priority"] = 4
    data["_single_score"] = data.get("source_signal_score", pd.Series(0.0, index=data.index)).fillna(0.0)
    data["_single_external"] = data.get("external_catalyst_score", pd.Series(0.0, index=data.index)).fillna(0.0)
    data["_abs_delta"] = data.get("delta_weight", pd.Series(0.0, index=data.index)).abs().fillna(0.0)
    data["_abs_notional"] = data.get("estimated_trade_notional", pd.Series(0.0, index=data.index)).abs().fillna(0.0)
    return data.sort_values(
        ["_single_priority", "_single_score", "_single_external", "_abs_delta", "_abs_notional"],
        ascending=[True, False, False, False, False],
    )


def _single_action_text(action: Any, side: Any, guard_status: Any, language: str = "zh") -> str:
    raw_action = str(action or "").lower()
    raw_side = str(side or "").upper()
    blocked = str(guard_status or "").lower() == "blocked"
    if language != "zh":
        if raw_action in {"sell", "sell_watch"}:
            return "SELL"
        if raw_action == "trim_watch":
            return "TRIM"
        if raw_action in {"buy", "buy_watch"} and raw_side == "BUY" and not blocked:
            return "BUY WATCH / PAPER BUY"
        if raw_action in {"buy", "buy_watch"}:
            return "OBSERVE / BUY REVIEW"
        return "OBSERVE"
    if raw_action in {"sell", "sell_watch"}:
        return "SELL（卖出/退出观察）"
    if raw_action == "trim_watch":
        return "TRIM（减仓观察）"
    if raw_action in {"buy", "buy_watch"} and raw_side == "BUY" and not blocked:
        return "BUY（埋伏观察，Paper 限价票）"
    if raw_action in {"buy", "buy_watch"}:
        return "OBSERVE（买入候选，先复核）"
    return "OBSERVE（继续观察）"


def _combined_guard_status(ticket: dict[str, Any], review_row: dict[str, Any]) -> str:
    statuses = [
        str(_value_from(ticket.get("catalyst_guard_status"), review_row.get("catalyst_guard_status"), "") or "").lower(),
        str(_value_from(ticket.get("entry_guard_status"), review_row.get("entry_guard_status"), "") or "").lower(),
    ]
    if "blocked" in statuses:
        return "blocked"
    return next((status for status in statuses if status), "n/a")


def _best_strategy_meta(status: dict[str, Any]) -> dict[str, Any]:
    early_status = _read_json(EARLY_STATUS_PATH)
    quality_status = _read_json(QUALITY_TRACKER_STATUS_PATH)
    v123 = early_status.get("v123_reference") or {}
    best = v123.get("best_candidate") or quality_status.get("best_quant_reference") or status.get("best_version", "n/a")
    q14 = _quality_by_horizon(status, "T+14")
    return {
        "best_quant_reference": best,
        "best_reference_cagr": _as_float(v123.get("best_CAGR")),
        "best_reference_mdd": _as_float(v123.get("best_MDD")),
        "best_reference_profit_factor": _as_float(v123.get("best_profit_factor")),
        "best_reference_date": v123.get("reference_date"),
        "best_reference_stale_days": v123.get("reference_stale_days"),
        "strategy_quality_horizon": "T+14",
        "strategy_quality_sample": q14.get("sample"),
        "strategy_quality_win_rate": _as_float(q14.get("win_rate")),
        "strategy_quality_beat_spy_rate": _as_float(q14.get("beat_spy_rate")),
        "strategy_quality_avg_excess": _as_float(q14.get("avg_excess")),
    }


def _best_strategy_line(status: dict[str, Any], language: str = "zh") -> str:
    meta = _best_strategy_meta(status)
    best = meta.get("best_quant_reference")
    cagr = _fmt_pct(meta.get("best_reference_cagr"))
    mdd = _fmt_pct(meta.get("best_reference_mdd"))
    profit_factor = _fmt_float(meta.get("best_reference_profit_factor"))
    if language == "zh":
        return (
            f"策略依据：基于回测最佳版本 `{best}`；CAGR {cagr}，MDD {mdd}，profit factor {profit_factor}；"
            f"当前量化核心 T+14 样本 {meta.get('strategy_quality_sample', 'n/a')}，"
            f"胜率 {_fmt_pct(meta.get('strategy_quality_win_rate'))}，"
            f"平均超额 {_fmt_pct(meta.get('strategy_quality_avg_excess'))}。"
        )
    return (
        f"Strategy: based on backtested best version `{best}`; CAGR {cagr}, MDD {mdd}, profit factor {profit_factor}; "
        f"current quant core T+14 sample {meta.get('strategy_quality_sample', 'n/a')}, "
        f"win {_fmt_pct(meta.get('strategy_quality_win_rate'))}, "
        f"avg excess {_fmt_pct(meta.get('strategy_quality_avg_excess'))}."
    )


def _build_single_stock_signal_message(
    config: SlackSignalConfig,
    signal: pd.DataFrame,
    status: dict[str, Any],
    digest: str,
) -> tuple[str, str, dict[str, Any]]:
    now_et = datetime.now(ZoneInfo(config.timezone_name))
    signal_date = signal["signal_date"].max() if "signal_date" in signal.columns and not signal.empty else "n/a"
    ranked = _rank_single_stock_candidates(signal)
    execution_status = _read_json(EXECUTION_PACKET_STATUS_PATH)
    order_plan_status = _read_json(ORDER_PLAN_STATUS_PATH)
    tickets = _read_csv(EXECUTION_PACKET_TICKETS_PATH)
    review = _read_csv(ORDER_PLAN_REVIEW_BOARD_PATH)
    evidence = _read_csv(CATALYST_EVIDENCE_CURRENT_PATH)
    quality_status = _read_json(QUALITY_TRACKER_STATUS_PATH)
    source_mix_status = _read_json(SOURCE_MIX_STATUS_PATH)
    source_mix_forward_status = _read_json(SOURCE_MIX_FORWARD_STATUS_PATH)
    catalyst_status = _read_json(CATALYST_EVIDENCE_STATUS_PATH)

    if ranked.empty:
        text = "\n".join(
            [
                "*美股单股信号*" if config.language == "zh" else "*US Single-Stock Signal*",
                f"美东时间：{now_et:%Y-%m-%d %H:%M:%S}" if config.language == "zh" else f"Time ET: {now_et:%Y-%m-%d %H:%M:%S}",
                "当前没有可推送的单股信号。" if config.language == "zh" else "No single-stock signal is available.",
            ]
        )
        return text, digest, {"digest": digest, "signal_rows": int(len(signal)), "signal_date": str(signal_date), "selected_ticker": None}

    row = ranked.iloc[0].to_dict()
    ticker = str(row.get("ticker", "n/a")).upper()
    ticket = _first_matching_row(tickets, ticker)
    review_row = _first_matching_row(review, ticker)
    evidence_row = _first_matching_row(evidence, ticker)
    side = _value_from(ticket.get("side"), review_row.get("side"), review_row.get("requested_side"))
    guard_status = _combined_guard_status(ticket, review_row)
    action_text = _single_action_text(row.get("action"), side, guard_status, config.language)
    strategy_meta = _best_strategy_meta(status)

    target_weight = _fmt_pct(_value_from(ticket.get("target_weight"), review_row.get("target_weight"), row.get("target_weight")))
    delta_weight = _fmt_pct(_value_from(ticket.get("delta_weight"), review_row.get("delta_weight"), row.get("delta_weight")))
    notional = _fmt_money(_value_from(ticket.get("notional"), review_row.get("notional"), row.get("estimated_trade_notional")))
    qty = _fmt_float(_value_from(ticket.get("quantity"), review_row.get("quantity"), row.get("estimated_qty_at_reference_close")), 4)
    limit_price = _fmt_price(_value_from(ticket.get("limit_price"), review_row.get("reference_limit_price"), row.get("entry_high"), row.get("reference_close_price")))
    stop = _fmt_price(_value_from(ticket.get("protective_stop_price"), review_row.get("protective_stop"), row.get("trailing_stop"), row.get("initial_stop")))
    take_profit_1 = _fmt_price(_value_from(ticket.get("take_profit_1_limit"), review_row.get("trim_zone_1"), row.get("trim_zone_1")))
    take_profit_2 = _fmt_price(_value_from(ticket.get("take_profit_2_limit"), review_row.get("trim_zone_2"), row.get("trim_zone_2")))
    stop_risk = _fmt_money(ticket.get("estimated_stop_risk"))
    stop_risk_pct = _fmt_pct(ticket.get("estimated_stop_risk_pct_equity"))
    top_title = str(_value_from(review_row.get("catalyst_top_source_title"), evidence_row.get("top_source_title"), "n/a")).replace("\n", " ")
    if len(top_title) > 120:
        top_title = top_title[:119].rstrip() + "…"
    catalyst_evidence_status = _value_from(
        review_row.get("catalyst_evidence_status"),
        evidence_row.get("evidence_status"),
        "n/a",
    )
    catalyst_recent = _value_from(review_row.get("catalyst_recent_7d_count"), evidence_row.get("recent_7d_count"), "n/a")
    catalyst_fresh = _value_from(review_row.get("catalyst_fresh_1d_count"), evidence_row.get("fresh_1d_count"), "n/a")
    source_type = _value_from(evidence_row.get("top_source_type"), "news")
    reason = str(_value_from(row.get("reason"), review_row.get("reason"), "n/a"))
    source_reason = str(_value_from(row.get("source_reason"), review_row.get("source_reason"), "n/a"))
    guard_reason = str(_value_from(review_row.get("catalyst_guard_reason"), ticket.get("pre_trade_instruction"), "n/a"))
    source_mix_summary = source_mix_status.get("summary") or {}
    source_mix_forward = source_mix_forward_status.get("summary") or {}
    catalyst_summary = catalyst_status.get("summary") or {}
    if config.language == "zh":
        lines = [
            "*美股单股信号*",
            f"美东时间：{now_et:%Y-%m-%d %H:%M:%S}",
            "更新口径：事件驱动；价格数据、新闻、Reddit 或策略分数变化后触发。本次 Discord 只推 1 只股票。",
            f"标的：`{ticker}`",
            f"动作：{action_text}",
            f"信号日期：{signal_date}；计划执行：下一个 regular session。",
            _best_strategy_line(status, config.language),
            f"仓位：目标 {target_weight}，本次调仓 {delta_weight}，金额 {notional}，数量 {qty}。",
            f"价格：limit {limit_price}；止损 {stop}；止盈 TP1 {take_profit_1} / TP2 {take_profit_2}。",
            f"Paper执行票：side={side or 'NO_TRADE'}，guard={guard_status or 'n/a'}，单票估算止损风险 {stop_risk} ({stop_risk_pct})。",
            f"原因：{reason}；{source_reason}",
            f"催化证据：{catalyst_evidence_status}，近7日 {catalyst_recent}，fresh {catalyst_fresh}；{source_type}: {top_title}",
            f"执行复核：{guard_reason}",
            f"来源归因：外部催化命中 {source_mix_summary.get('final_external_catalyst_count', 'n/a')}，"
            f"v123重合 {source_mix_summary.get('final_v123_reference_count', 'n/a')}；"
            f"source-mix ready={source_mix_forward.get('forward_attribution_ready', False)} "
            f"({source_mix_forward.get('readiness_reason', quality_status.get('source_mix_forward_reason', 'n/a'))})。",
            f"全量候选：{len(signal)} 只；买入票近7日证据 {catalyst_summary.get('buy_external_with_recent_evidence', 'n/a')}/"
            f"{catalyst_summary.get('buy_external_expected_rows', 'n/a')}；Reddit {catalyst_summary.get('reddit_records', 'n/a')} / News {catalyst_summary.get('news_records', 'n/a')}。",
        ]
    else:
        lines = [
            "*US Single-Stock Signal*",
            f"Time ET: {now_et:%Y-%m-%d %H:%M:%S}",
            "Cadence: event-driven; price, news, Reddit, or strategy-score updates can trigger a push. This Discord message covers one ticker only.",
            f"Ticker: `{ticker}`",
            f"Action: {action_text}",
            f"Signal date: {signal_date}; planned execution: next regular session.",
            _best_strategy_line(status, config.language),
            f"Position: target {target_weight}, delta {delta_weight}, notional {notional}, quantity {qty}.",
            f"Prices: limit {limit_price}; stop {stop}; take-profit TP1 {take_profit_1} / TP2 {take_profit_2}.",
            f"Paper ticket: side={side or 'NO_TRADE'}, guard={guard_status or 'n/a'}, estimated stop risk {stop_risk} ({stop_risk_pct}).",
            f"Reason: {reason}; {source_reason}",
            f"Catalyst evidence: {catalyst_evidence_status}, recent 7d {catalyst_recent}, fresh {catalyst_fresh}; {source_type}: {top_title}",
            f"Execution review: {guard_reason}",
        ]
    if config.include_paper_warning:
        lines.append(_paper_warning(status, config.language))
    text = "\n".join(lines)
    meta = {
        "digest": digest,
        "signal_rows": int(len(signal)),
        "signal_date": str(signal_date),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_stock_message": True,
        "selected_ticker": ticker,
        "selected_action": str(row.get("action", "n/a")),
        "selected_action_text": action_text,
        "selected_side": str(side or "NO_TRADE"),
        "selected_order_type": str(_value_from(ticket.get("ibkr_order_type"), review_row.get("order_type"), "n/a")),
        "selected_quantity": _as_float(_value_from(ticket.get("quantity"), review_row.get("quantity"), row.get("estimated_qty_at_reference_close"))),
        "selected_notional": _as_float(_value_from(ticket.get("notional"), review_row.get("notional"), row.get("estimated_trade_notional"))),
        "selected_limit_price": _as_float(_value_from(ticket.get("limit_price"), review_row.get("reference_limit_price"))),
        "selected_stop_price": _as_float(_value_from(ticket.get("protective_stop_price"), review_row.get("protective_stop"))),
        "selected_take_profit_1": _as_float(_value_from(ticket.get("take_profit_1_limit"), review_row.get("trim_zone_1"))),
        "selected_take_profit_2": _as_float(_value_from(ticket.get("take_profit_2_limit"), review_row.get("trim_zone_2"))),
        "selected_target_weight": _as_float(_value_from(ticket.get("target_weight"), review_row.get("target_weight"), row.get("target_weight"))),
        "selected_delta_weight": _as_float(_value_from(ticket.get("delta_weight"), review_row.get("delta_weight"), row.get("delta_weight"))),
        "selected_source_signal_score": _as_float(row.get("source_signal_score")),
        "selected_external_catalyst_score": _as_float(row.get("external_catalyst_score")),
        "selected_priority": _as_float(row.get("_single_priority")),
        "selected_catalyst_guard_status": str(guard_status or "n/a"),
        "selected_catalyst_guard_reason": guard_reason,
        "selected_catalyst_evidence_status": str(catalyst_evidence_status),
        "selected_catalyst_recent_7d_count": _as_float(catalyst_recent),
        "selected_catalyst_fresh_1d_count": _as_float(catalyst_fresh),
        "selected_catalyst_source_type": str(source_type),
        "selected_catalyst_title": top_title,
        "selected_reason": reason,
        "selected_source_reason": source_reason,
        "selected_live_order_allowed": bool(_value_from(ticket.get("live_order_allowed"), review_row.get("live_order_allowed"), row.get("live_order_allowed"), False)),
        **strategy_meta,
        "execution_packet_ticket_count": execution_status.get("ticket_count"),
        "execution_packet_pre_trade_check_count": execution_status.get("pre_trade_check_count"),
        "execution_packet_total_estimated_stop_risk": execution_status.get("total_estimated_stop_risk"),
        "catalyst_guard_blocked_buy_count": order_plan_status.get("catalyst_guard_blocked_buy_count"),
        "catalyst_guard_blocked_tickers": order_plan_status.get("catalyst_guard_blocked_tickers") or [],
        "quality_tracker_historical_core_ready": quality_status.get("historical_core_ready"),
        "quality_tracker_t14_win_rate": quality_status.get("t14_win_rate"),
        "quality_tracker_t14_avg_excess": quality_status.get("t14_avg_excess"),
        "source_mix_forward_ready": source_mix_forward.get("forward_attribution_ready"),
        "source_mix_forward_reason": source_mix_forward.get("readiness_reason"),
        "catalyst_buy_external_with_recent_evidence": catalyst_summary.get("buy_external_with_recent_evidence"),
        "catalyst_buy_external_expected_rows": catalyst_summary.get("buy_external_expected_rows"),
    }
    return text, digest, meta


def build_signal_message(config: SlackSignalConfig) -> tuple[str, str, dict[str, Any]]:
    result = deliver_report(
        config.report_root,
        channel_alias=config.channel_alias,
        language=config.language,
        dry_run=True,
    )
    return result["text"], result["digest"], {key: value for key, value in result.items() if key != "text"}


def _legacy_build_signal_message(config: SlackSignalConfig) -> tuple[str, str, dict[str, Any]]:
    signal = _read_csv(config.signal_path)
    status = _read_json(config.status_path)
    digest = _signal_hash(signal, status)
    if config.single_stock:
        return _build_single_stock_signal_message(config, signal, status, digest)
    now_et = datetime.now(ZoneInfo(config.timezone_name))
    signal_date = signal["signal_date"].max() if "signal_date" in signal.columns and not signal.empty else "n/a"
    action_counts = signal["action"].value_counts().to_dict() if "action" in signal.columns and not signal.empty else {}
    action_counts_label = _action_counts_text(action_counts, config.language)
    buy_actions = {"buy", "buy_watch"}
    sell_actions = {"sell", "sell_watch", "trim_watch"}
    buy_notional = float(
        signal.loc[signal.get("action", pd.Series(dtype=str)).astype(str).str.lower().isin(buy_actions), "estimated_trade_notional"].sum()
    ) if {"action", "estimated_trade_notional"}.issubset(signal.columns) else 0.0
    sell_notional = float(
        signal.loc[signal.get("action", pd.Series(dtype=str)).astype(str).str.lower().isin(sell_actions), "estimated_trade_notional"].abs().sum()
    ) if {"action", "estimated_trade_notional"}.issubset(signal.columns) else 0.0
    execution_line, execution_status = _execution_packet_line(config.language)
    order_plan_status = _read_json(ORDER_PLAN_STATUS_PATH)
    quality_tracker_line, quality_tracker_status = _quality_tracker_line(config.language)
    portfolio_tracker_line, portfolio_tracker_status = _portfolio_tracker_line(config.language)
    source_mix_line, source_mix_meta = _source_mix_line(config.language)
    catalyst_evidence_line, catalyst_evidence_meta = _catalyst_evidence_line(config.language)
    research_queue_line, research_queue_status = _research_queue_line(config.language)
    pre_takeoff_line, pre_takeoff_status = _pre_takeoff_line(config.language)
    buried_opportunity_line, buried_opportunity_meta = _buried_opportunity_line(config.language)
    blocked_tickers = ", ".join(order_plan_status.get("catalyst_guard_blocked_tickers") or []) or "none"
    blocked_count = int(order_plan_status.get("catalyst_guard_blocked_buy_count") or 0)
    review_board = _read_csv(ORDER_PLAN_REVIEW_BOARD_PATH)
    entry_blocked_tickers: list[str] = []
    if not review_board.empty and {"ticker", "entry_guard_status"}.issubset(review_board.columns):
        entry_blocked_tickers = (
            review_board.loc[
                review_board["entry_guard_status"].astype(str).str.lower().eq("blocked"),
                "ticker",
            ]
            .astype(str)
            .str.upper()
            .tolist()
        )
    entry_blocked_count = int(order_plan_status.get("entry_guard_blocked_buy_count") or len(entry_blocked_tickers))
    entry_blocked_text = ", ".join(entry_blocked_tickers) or "none"
    paper_buy_notional = _fmt_money(execution_status.get("buy_notional"))
    paper_sell_notional = _fmt_money(execution_status.get("sell_notional"))

    if config.language == "zh":
        paper_warning = _paper_warning(status, config.language) if config.include_paper_warning else ""
        lines = [
            "*美股策略信号推送*",
            f"美东时间：{now_et:%Y-%m-%d %H:%M:%S}",
            f"当前最佳诊断版本：`{status.get('best_version', 'unknown')}`",
            f"信号日期：{signal_date}；计划执行：下一个 regular session 开盘",
            f"账户模型：IBKR Pro 阶梯佣金，margin 账户，不使用融资，允许碎股，基准资金 ${config.account_equity:,.0f}",
            _metrics_line(status, config.language),
            _diagnostic_line(status, config.language),
            f"动作统计：{action_counts_label}；信号候选买入 {_fmt_money(buy_notional)}，候选卖出 {_fmt_money(sell_notional)}；"
            f"Paper可执行买入 {paper_buy_notional}，Paper可执行卖出 {paper_sell_notional}；"
            f"证据不足转复核 {blocked_count} 个：{blocked_tickers}；"
            f"不追高等回踩 {entry_blocked_count} 个：{entry_blocked_text}",
            execution_line,
            quality_tracker_line,
            portfolio_tracker_line,
            source_mix_line,
            catalyst_evidence_line,
            pre_takeoff_line,
            buried_opportunity_line,
            research_queue_line,
            _ibkr_snapshot_summary(config.language),
        ]
    else:
        paper_warning = _paper_warning(status, config.language) if config.include_paper_warning else ""
        lines = [
            "*US Strategy Signal Push*",
            f"Time ET: {now_et:%Y-%m-%d %H:%M:%S}",
            f"Best diagnostic version: `{status.get('best_version', 'unknown')}`",
            f"Signal date: {signal_date}; planned execution: next regular session open",
            f"Account model: IBKR Pro Tiered, margin account, no leverage, fractional shares, ${config.account_equity:,.0f} base",
            _metrics_line(status, config.language),
            _diagnostic_line(status, config.language),
            f"Actions: {action_counts_label}; signal buy candidate {_fmt_money(buy_notional)}, signal sell candidate {_fmt_money(sell_notional)}; "
            f"Paper-executable buy {paper_buy_notional}, Paper-executable sell {paper_sell_notional}; "
            f"catalyst-review blocks {blocked_count}: {blocked_tickers}; "
            f"no-chase wait-pullback {entry_blocked_count}: {entry_blocked_text}",
            execution_line,
            quality_tracker_line,
            portfolio_tracker_line,
            source_mix_line,
            catalyst_evidence_line,
            pre_takeoff_line,
            buried_opportunity_line,
            research_queue_line,
            _ibkr_snapshot_summary(config.language),
        ]
    if paper_warning:
        lines.append(paper_warning)
    lines.append("")
    lines.append(f"按动作和金额排序的前 {config.top_n} 条研究信号：" if config.language == "zh" else f"Top {config.top_n} signals by action/notional:")
    lines.append("```")
    lines.extend(_top_signal_lines(signal, config.top_n, config.language))
    lines.append("```")
    catalyst_lines = _top_catalyst_lines(min(config.top_n, 5), config.language)
    if catalyst_lines:
        lines.append("")
        lines.append(
            f"买入催化证据前 {len(catalyst_lines)} 条："
            if config.language == "zh"
            else f"Top {len(catalyst_lines)} buy catalyst evidence rows:"
        )
        lines.append("```")
        lines.extend(catalyst_lines)
        lines.append("```")
    pre_takeoff_lines = _top_pre_takeoff_lines(min(config.top_n, 5), config.language)
    if pre_takeoff_lines:
        lines.append("")
        lines.append(
            f"起飞前埋伏评分前 {len(pre_takeoff_lines)} 条："
            if config.language == "zh"
            else f"Top {len(pre_takeoff_lines)} pre-takeoff profile rows:"
        )
        lines.append("```")
        lines.extend(pre_takeoff_lines)
        lines.append("```")
    buried_opportunity_lines = _top_buried_opportunity_lines(min(config.top_n, 5), config.language)
    if buried_opportunity_lines:
        lines.append("")
        lines.append(
            f"早期埋伏复核前 {len(buried_opportunity_lines)} 条："
            if config.language == "zh"
            else f"Top {len(buried_opportunity_lines)} early accumulation review rows:"
        )
        lines.append("```")
        lines.extend(buried_opportunity_lines)
        lines.append("```")
    research_queue_lines = _top_research_queue_lines(min(config.top_n, 5), config.language)
    if research_queue_lines:
        lines.append("")
        lines.append(
            f"待研究队列前 {len(research_queue_lines)} 条："
            if config.language == "zh"
            else f"Top {len(research_queue_lines)} research queue rows:"
        )
        lines.append("```")
        lines.extend(research_queue_lines)
        lines.append("```")
    ticket_lines = _top_execution_ticket_lines(config.top_n, config.language)
    if ticket_lines:
        lines.append("")
        lines.append(
            f"Paper执行票前 {min(config.top_n, len(ticket_lines))} 条："
            if config.language == "zh"
            else f"Top {min(config.top_n, len(ticket_lines))} Paper execution tickets:"
        )
        lines.append("```")
        lines.extend(ticket_lines)
        lines.append("```")
    text = "\n".join(lines)
    meta = {
        "digest": digest,
        "signal_rows": int(len(signal)),
        "action_counts": action_counts,
        "signal_date": str(signal_date),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_packet_ticket_count": execution_status.get("ticket_count"),
        "execution_packet_pre_trade_check_count": execution_status.get("pre_trade_check_count"),
        "execution_packet_total_estimated_stop_risk": execution_status.get("total_estimated_stop_risk"),
        "catalyst_guard_blocked_buy_count": blocked_count,
        "catalyst_guard_blocked_tickers": order_plan_status.get("catalyst_guard_blocked_tickers") or [],
        "research_queue_count": research_queue_status.get("queue_count"),
        "research_queue_top_tickers": research_queue_status.get("top_queue_tickers") or [],
        "research_queue_catalyst_refresh_before_buy_count": research_queue_status.get("catalyst_refresh_before_buy_count"),
        "research_queue_external_evidence_refresh_count": research_queue_status.get("external_evidence_refresh_count"),
        "pre_takeoff_profile_count": pre_takeoff_status.get("profile_count"),
        "pre_takeoff_top_tickers": pre_takeoff_status.get("top_pre_takeoff_tickers") or [],
        "pre_takeoff_core_count": pre_takeoff_status.get("core_pre_takeoff_count"),
        "pre_takeoff_chase_risk_count": pre_takeoff_status.get("chase_risk_count"),
        "buried_opportunity_rows": (buried_opportunity_meta.get("status") or {}).get("row_count"),
        "buried_opportunity_starter_buy_review_count": (buried_opportunity_meta.get("status") or {}).get("starter_buy_review_count"),
        "buried_opportunity_wait_pullback_no_chase_count": (buried_opportunity_meta.get("status") or {}).get("wait_pullback_no_chase_count"),
        "buried_opportunity_catalyst_diligence_count": (buried_opportunity_meta.get("status") or {}).get("catalyst_diligence_count"),
        "buried_opportunity_starter_tickers": buried_opportunity_meta.get("starter_tickers") or [],
        "buried_opportunity_wait_pullback_tickers": buried_opportunity_meta.get("wait_pullback_tickers") or [],
        "buried_opportunity_catalyst_diligence_tickers": buried_opportunity_meta.get("catalyst_diligence_tickers") or [],
        "buried_opportunity_archive_total_rows": (buried_opportunity_meta.get("archive") or {}).get("total_archive_rows"),
        "buried_opportunity_archive_decision_dates": (buried_opportunity_meta.get("archive") or {}).get("archive_decision_dates"),
        "quality_tracker_historical_core_ready": quality_tracker_status.get("historical_core_ready"),
        "quality_tracker_t14_win_rate": quality_tracker_status.get("t14_win_rate"),
        "quality_tracker_t14_avg_excess": quality_tracker_status.get("t14_avg_excess"),
        "portfolio_current_model_return": portfolio_tracker_status.get("current_model_return"),
        "portfolio_same_weight_spy_return": portfolio_tracker_status.get("same_weight_spy_return"),
        "portfolio_same_weight_alpha_vs_spy": portfolio_tracker_status.get("same_weight_alpha_vs_spy"),
        "portfolio_current_best_ticker": portfolio_tracker_status.get("current_best_ticker"),
        "portfolio_current_worst_ticker": portfolio_tracker_status.get("current_worst_ticker"),
        "portfolio_historical_t14_curve_total_return": portfolio_tracker_status.get("historical_t14_curve_total_return"),
        "portfolio_historical_t14_curve_spy_total_return": portfolio_tracker_status.get("historical_t14_curve_spy_total_return"),
        "portfolio_historical_t14_curve_excess_multiple": portfolio_tracker_status.get("historical_t14_curve_excess_multiple"),
        "portfolio_historical_t14_curve_max_drawdown": portfolio_tracker_status.get("historical_t14_curve_max_drawdown"),
        "source_mix_forward_ready": (source_mix_meta.get("source_mix_forward") or {}).get("forward_attribution_ready"),
        "source_mix_forward_reason": (source_mix_meta.get("source_mix_forward") or {}).get("readiness_reason"),
        "catalyst_buy_external_with_recent_evidence": catalyst_evidence_meta.get("buy_external_with_recent_evidence"),
        "catalyst_buy_external_expected_rows": catalyst_evidence_meta.get("buy_external_expected_rows"),
    }
    return text, digest, meta


def _read_state(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_push_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            existing = pd.DataFrame()
        if not existing.empty:
            columns = list(dict.fromkeys([*existing.columns, *new_row.columns]))
            pd.concat(
                [existing.reindex(columns=columns), new_row.reindex(columns=columns)],
                ignore_index=True,
            ).to_csv(path, index=False)
        else:
            new_row.to_csv(path, index=False)
    else:
        new_row.to_csv(path, index=False)


def post_to_slack(webhook_url: str, text: str, timeout: int = 12) -> dict[str, Any]:
    receipt = post_research_report(webhook_url, text, timeout=timeout)
    return {"ok": receipt.state == "SENT", "status_code": receipt.http_status, "reason": receipt.code}


def send_latest_signal(
    config: SlackSignalConfig,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    # Legacy payload paths, account options, and force never bypass the research contract.
    if not dry_run:
        validate_webhook(config.webhook_url)
    return deliver_report(
        config.report_root,
        channel_alias=config.channel_alias,
        language=config.language,
        dry_run=dry_run,
        transport=None if dry_run else lambda text: post_research_report(config.webhook_url, text),
    )
