from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo


CALENDAR_HASH = "ad549218da11ad3103c7027cf7b354b102839e471891f6cd973bc96b8717dd07"
INCUMBENT_ID = "user_nolev_top30_m126_invvol_m"
WATCH_ID = "early_accumulation_signal_v1"
SOURCE_ID = "us_quant_research_catalogue_52"
STREAM = "daily-research"
NY = ZoneInfo("America/New_York")
SHANGHAI = ZoneInfo("Asia/Shanghai")
PRIVATE_KEYS = {
    "account_id", "account_number", "holdings", "positions", "fills", "orders",
    "webhook_url", "authorization", "api_key", "access_token", "password",
    "account_equity", "net_liquidation", "shares", "quantity", "current_shares",
    "average_cost", "cost_basis", "unrealized_pnl",
}
SECRET_TEXT = re.compile(
    r"hooks\.slack(?:-gov)?\.com|discord(?:app)?\.com/api/webhooks"
    r"|xox[baprs]-[A-Za-z0-9-]+|gh[pousr]_[A-Za-z0-9]+"
    r"|(?:authorization|api[_-]?key|access[_-]?token|password)\s*[:=]",
    re.IGNORECASE,
)
BLOCKERS_ZH = {
    "top30_price_session_mismatch": "Top30行情仍停在历史截止日，未覆盖本次已收盘交易日。",
    "top30_signal_session_mismatch": "Top30信号尚未按本次交易日重新验证，不能当作今日新信号。",
    "early_watch_price_session_mismatch": "旧早期观察行情已过期，与本次交易日不一致。",
    "early_watch_signal_session_mismatch": "旧早期观察信号已过期，不能改写日期后重新推荐。",
    "early_watch_future_news_vs_signal": "资讯晚于旧信号日期，不能证明当时已知的催化。",
    "early_watch_known_at_unproven": "旧资讯缺少首次可得时间，点时证据尚未证明。",
    "no_hash_verified_watch_snapshot": "尚无通过来源哈希与同截止日校验的独立观察快照。",
    "source_archive_endpoint_mismatch": "导入研究的历史终点不是本次交易日，仍只作为历史证据。",
    "watch_input_missing": "独立观察所需输入缺失。",
    "watch_input_hash_mismatch": "独立观察输入与来源哈希不一致。",
    "watch_input_manifest_missing": "独立观察缺少完整的来源清单。",
    "watch_identity_or_session_mismatch": "独立观察的策略身份、交易日或权限声明不符合约定。",
    "watch_price_session_mismatch": "至少一项必要价格不属于本次交易日。",
    "watch_required_price_missing": "候选或基准标的缺少必要价格，不能用全表最新日期代替完整覆盖。",
    "watch_price_known_at_invalid": "价格首次可得时间不在规定的决策窗口内。",
    "watch_event_known_at_invalid": "事件发布时间或首次可得时间不满足点时约束。",
    "watch_universe_not_point_in_time": "历史股票池的点时身份或可得时间尚未证明。",
    "watch_decision_outside_cutoff": "观察决策必须在收盘数据缓冲后、下次开盘前，且不能晚于报告生成时间。",
    "watch_next_open_unavailable": "日历未覆盖下次开盘时间，暂不能验证观察决策窗口。",
    "watch_not_new_session": "观察不是本次交易日首次出现，不能作为新增想法。",
    "timestamp_missing": "必要的发布时间或首次可得时间缺失。",
    "timestamp_invalid": "必要时间字段格式无效。",
    "timestamp_timezone_missing": "必要时间字段缺少时区。",
    "private_fields_rejected": "输入含不应进入研究日报的私人字段，已拒绝使用。",
    "credential_text_rejected": "输入含敏感内容，已拒绝使用。",
}


class ReportError(RuntimeError):
    """A stable error code, never raw input, file content, or a transport URL."""


def canonical_hash(value: Any) -> str:
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError):
        raise ReportError("provenance_serialization_invalid") from None
    return hashlib.sha256(data).hexdigest()


def checked_public(value: Any) -> None:
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(str(key).lower() for key in value):
            raise ReportError("private_fields_rejected")
        for item in value.values():
            checked_public(item)
    elif isinstance(value, list):
        for item in value:
            checked_public(item)
    elif isinstance(value, str) and SECRET_TEXT.search(value):
        raise ReportError("credential_text_rejected")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ReportError("input_missing") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReportError("input_unreadable") from None
    if not isinstance(value, dict):
        raise ReportError("input_not_object")
    checked_public(value)
    return value


def instant(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ReportError("timestamp_missing")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReportError("timestamp_invalid") from None
    if result.tzinfo is None:
        raise ReportError("timestamp_timezone_missing")
    return result.astimezone(timezone.utc)


def session_date(value: Any) -> str:
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError:
        raise ReportError("session_date_invalid") from None
    return value


def public_text(value: Any, limit: int = 600) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReportError("public_text_invalid")
    checked_public(value)
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def positive_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError("number_invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ReportError("number_invalid")
    return number


def source_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ReportError("evidence_url_missing")
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise ReportError("evidence_url_rejected") from None
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
        or SECRET_TEXT.search(value) or any(char in value for char in "<>|\r\n")
        or any(re.search(r"token|secret|auth|key|signature", key, re.I) for key, _ in parse_qsl(parsed.query))
    ):
        raise ReportError("evidence_url_rejected")
    return value


@dataclass(frozen=True)
class MarketSession:
    day: str
    opened: datetime
    closed: datetime
    decision_not_before: datetime
    next_open: datetime | None
    report_after: datetime


class ExchangeCalendar:
    def __init__(self, payload: dict[str, Any]) -> None:
        if (
            payload.get("schema_version") != 1 or payload.get("calendar_id") != "XNYS"
            or payload.get("provider_version") != "4.13.2"
            or payload.get("sessions_sha256") != CALENDAR_HASH
            or payload.get("coverage_start") != "2024-01-01"
            or payload.get("coverage_end") != "2028-12-31"
            or payload.get("data_finalization_buffer_minutes") != 30
        ):
            raise ReportError("calendar_identity_invalid")
        rows = payload.get("sessions")
        if not isinstance(rows, list):
            raise ReportError("calendar_sessions_missing")
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if digest != CALENDAR_HASH:
            raise ReportError("calendar_hash_mismatch")
        self.start = date.fromisoformat(session_date(payload.get("coverage_start")))
        self.end = date.fromisoformat(session_date(payload.get("coverage_end")))
        self.sessions: dict[str, MarketSession] = {}
        previous = ""
        for index, row in enumerate(rows):
            day = session_date(row.get("session"))
            opened, closed = instant(row.get("open_utc")), instant(row.get("close_utc"))
            if day <= previous or opened >= closed or closed.astimezone(NY).date().isoformat() != day:
                raise ReportError("calendar_session_invalid")
            close_local = closed.astimezone(SHANGHAI)
            due = datetime.combine(close_local.date(), time(6), SHANGHAI)
            if due <= close_local:
                due += timedelta(days=1)
            next_open = instant(rows[index + 1].get("open_utc")) if index + 1 < len(rows) else None
            self.sessions[day] = MarketSession(
                day, opened, closed, closed + timedelta(minutes=30), next_open, due,
            )
            previous = day

    def latest_completed(self, now: datetime) -> MarketSession:
        if now.tzinfo is None:
            raise ReportError("clock_timezone_missing")
        local_day = now.astimezone(NY).date()
        if not self.start <= local_day <= self.end:
            raise ReportError("calendar_out_of_range")
        completed = [item for item in self.sessions.values() if item.closed <= now]
        if not completed:
            raise ReportError("calendar_no_completed_session")
        return completed[-1]


@dataclass(frozen=True)
class ReportInputs:
    registry: dict[str, Any]
    source: dict[str, Any]
    progress: dict[str, Any]
    watch: dict[str, Any] | None = None
    watch_error: str | None = None


def load_inputs(root: Path) -> ReportInputs:
    registry = read_json(root / "config/research_strategy_registry.json")
    source = read_json(root / "research/workbench/evidence/research_status.json")
    progress = read_json(root / "docs/research_progress.json")
    watch_dir = root / "artifacts/private/research_reporting/watch"
    snapshot_path = watch_dir / "snapshot.json"
    watch = None
    watch_error = None
    if snapshot_path.exists():
        try:
            if watch_dir.is_symlink() or snapshot_path.is_symlink():
                raise ReportError("watch_input_path_rejected")
            watch = read_json(snapshot_path)
            hashes = watch.get("input_sha256")
            if not isinstance(hashes, dict) or set(hashes) != {"prices", "events", "universe"}:
                raise ReportError("watch_input_manifest_missing")
            loaded = {}
            for name in ("prices", "events", "universe"):
                path = watch_dir / f"{name}.json"
                if path.is_symlink() or not path.resolve().is_relative_to(watch_dir.resolve()):
                    raise ReportError("watch_input_path_rejected")
                try:
                    raw = path.read_bytes()
                except OSError:
                    raise ReportError("watch_input_missing") from None
                if hashlib.sha256(raw).hexdigest() != hashes[name]:
                    raise ReportError("watch_input_hash_mismatch")
                loaded[name] = read_json(path)
            watch = {**watch, "inputs": loaded}
        except ReportError as exc:
            watch = None
            watch_error = str(exc)
    return ReportInputs(registry, source, progress, watch, watch_error)


def validate_watch(
    snapshot: dict[str, Any], session: MarketSession, seen: Iterable[str], *, now: datetime,
) -> list[dict[str, Any]]:
    checked_public(snapshot)
    if (
        snapshot.get("schema_version") != 1 or snapshot.get("origin") != "us_stock_qr"
        or snapshot.get("strategy_id") != WATCH_ID
        or snapshot.get("order_authority") is not False
        or snapshot.get("session") != session.day
    ):
        raise ReportError("watch_identity_or_session_mismatch")
    decision = instant(snapshot.get("decision_at"))
    if session.next_open is None:
        raise ReportError("watch_next_open_unavailable")
    if not session.decision_not_before <= decision < session.next_open or decision > now:
        raise ReportError("watch_decision_outside_cutoff")
    inputs = snapshot.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"prices", "events", "universe"}:
        raise ReportError("watch_inputs_missing")
    if not all(isinstance(value, dict) for value in inputs.values()):
        raise ReportError("watch_inputs_invalid")
    hashes = snapshot.get("input_sha256")
    if (
        not isinstance(hashes, dict) or set(hashes) != set(inputs)
        or any(canonical_hash(value) != hashes[name] for name, value in inputs.items())
    ):
        raise ReportError("watch_input_hash_mismatch")
    universe = inputs["universe"]
    if (
        universe.get("session") != session.day or universe.get("point_in_time") is not True
        or instant(universe.get("known_at")) > decision
    ):
        raise ReportError("watch_universe_not_point_in_time")
    members = universe.get("members")
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        raise ReportError("watch_universe_invalid")
    prices = {}
    price_rows = inputs["prices"].get("rows")
    event_rows = inputs["events"].get("rows")
    ideas = snapshot.get("ideas")
    if not all(isinstance(value, list) for value in (price_rows, event_rows, ideas)):
        raise ReportError("watch_rows_invalid")
    for row in price_rows:
        if not isinstance(row, dict):
            raise ReportError("watch_price_row_invalid")
        symbol = row.get("ticker")
        if not isinstance(symbol, str) or symbol in prices:
            raise ReportError("watch_price_identity_invalid")
        if row.get("session") != session.day:
            raise ReportError("watch_price_session_mismatch")
        if not session.closed <= instant(row.get("known_at")) <= decision:
            raise ReportError("watch_price_known_at_invalid")
        positive_number(row.get("close"))
        if row.get("basis") != "adjusted_close":
            raise ReportError("watch_price_basis_invalid")
        prices[symbol] = row
    required = {"SPY", "QQQ", "^VIX"}
    events = {}
    for event in event_rows:
        if not isinstance(event, dict):
            raise ReportError("watch_event_row_invalid")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in events:
            raise ReportError("watch_event_identity_invalid")
        published, known = instant(event.get("published_at")), instant(event.get("known_at"))
        if not decision - timedelta(days=10) <= published <= known <= decision:
            raise ReportError("watch_event_known_at_invalid")
        source_url(event.get("url"))
        events[event_id] = event
    selected = []
    seen_keys = set(seen)
    tickers = set()
    for idea in ideas:
        if not isinstance(idea, dict):
            raise ReportError("watch_idea_row_invalid")
        ticker = idea.get("ticker")
        signal_id = idea.get("signal_id")
        if (
            not isinstance(ticker, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker)
            or ticker in tickers or ticker not in members
            or not isinstance(signal_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", signal_id)
        ):
            raise ReportError("watch_idea_identity_invalid")
        tickers.add(ticker)
        required.add(ticker)
        if idea.get("signal_session") != session.day or idea.get("first_observed_session") != session.day:
            raise ReportError("watch_not_new_session")
        event_ids = idea.get("event_ids")
        if not isinstance(event_ids, list) or not event_ids or not all(isinstance(key, str) for key in event_ids):
            raise ReportError("watch_evidence_missing")
        if any(key not in events or events[key].get("ticker") != ticker for key in event_ids):
            raise ReportError("watch_evidence_identity_mismatch")
        rationale = public_text(idea.get("rationale"))
        key = f"us_stock_qr:{WATCH_ID}:{signal_id}"
        output = {"key": key, "ticker": ticker, "rationale": rationale, "url": source_url(events[event_ids[0]]["url"])}
        levels = idea.get("levels")
        if levels is not None:
            names = {"entry_low", "entry_high", "stop", "take_profit_1", "take_profit_2"}
            if not isinstance(levels, dict) or set(levels) != names:
                raise ReportError("watch_levels_invalid")
            numbers = {name: positive_number(levels[name]) for name in names}
            if not (
                numbers["stop"] < numbers["entry_low"] <= numbers["entry_high"]
                < numbers["take_profit_1"] <= numbers["take_profit_2"]
            ):
                raise ReportError("watch_levels_invalid")
            output["levels"] = numbers
        if key not in seen_keys:
            selected.append(output)
            seen_keys.add(key)
    if not required.issubset(prices):
        raise ReportError("watch_required_price_missing")
    return selected[:5]


@dataclass(frozen=True)
class ResearchReport:
    text: str
    session: str
    digest: str
    due: bool
    idea_keys: tuple[str, ...]
    blockers: tuple[str, ...]
    watch_data_ready: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "report_stream": STREAM,
            "session": self.session,
            "signal_date": self.session if self.idea_keys else None,
            "signal_rows": len(self.idea_keys),
            "new_watch_ideas": len(self.idea_keys),
            "status_only": not self.idea_keys,
            "blockers": list(self.blockers),
            "digest": self.digest,
            "due": self.due,
            "order_authority": False,
            "watch_data_ready": self.watch_data_ready,
        }


def build_report(
    inputs: ReportInputs,
    calendar: ExchangeCalendar,
    now: datetime,
    *,
    seen: Iterable[str] = (),
    language: str = "zh",
) -> ResearchReport:
    if language not in {"zh", "en"}:
        raise ReportError("language_invalid")
    session = calendar.latest_completed(now)
    registry, source, progress = inputs.registry, inputs.source, inputs.progress
    for payload in (registry, source, progress):
        checked_public(payload)
        if payload.get("schema_version") != 1:
            raise ReportError("report_schema_invalid")
    incumbent = registry.get("incumbent", {})
    early = registry.get("early_watch", {})
    if (
        incumbent.get("strategy_id") != INCUMBENT_ID or early.get("strategy_id") != WATCH_ID
        or source.get("strategy_id") != SOURCE_ID
        or source.get("order_authority") is not False
        or source.get("incumbent_stock_strategy_replacement_authorized") is not False
        or registry.get("order_authority") is not False
        or registry.get("automatic_retuning_enabled") is not False
        or source.get("role") != "research_archive_not_current_trade_recommendation"
    ):
        raise ReportError("report_identity_or_authority_invalid")
    blockers = []
    for lane, name in ((incumbent, "top30"), (early, "early_watch")):
        for field in ("price_session", "signal_session"):
            if session_date(lane.get(field)) != session.day:
                blockers.append(f"{name}_{field}_mismatch")
    if session_date(early.get("news_end")) > early["signal_session"]:
        blockers.append("early_watch_future_news_vs_signal")
    if early.get("published_and_known_at_proven") is not True:
        blockers.append("early_watch_known_at_unproven")
    ideas = []
    watch_ready = False
    if inputs.watch_error:
        blockers.append(inputs.watch_error)
    elif inputs.watch is None:
        blockers.append("no_hash_verified_watch_snapshot")
    else:
        try:
            ideas = validate_watch(inputs.watch, session, seen, now=now)
            watch_ready = True
        except ReportError as exc:
            blockers.append(str(exc))
    result = incumbent.get("historical_result", {})
    cagr = positive_number(result.get("cagr"))
    sharpe = positive_number(result.get("sharpe"))
    drawdown = positive_number(result.get("max_drawdown"))
    start, end = session_date(result.get("start")), session_date(result.get("end"))
    source_end = session_date(source.get("common_research_end"))
    source_count = source.get("trial_count")
    pass_count = source.get("joint_base_pass_count")
    if (
        not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < 0
        or not isinstance(pass_count, int) or isinstance(pass_count, bool) or not 0 <= pass_count <= source_count
        or source.get("current_broker_positions_included") is not False
        or source.get("current_trade_ideas") != []
    ):
        raise ReportError("source_archive_contract_invalid")
    if source_end != session.day:
        blockers.append("source_archive_endpoint_mismatch")
    zh = language == "zh"
    completed = progress.get("completed_zh" if zh else "completed")
    priorities = progress.get("next_zh" if zh else "next")
    if not isinstance(completed, list) or not isinstance(priorities, list):
        raise ReportError("progress_contract_invalid")
    completed_text = [public_text(item) for item in completed[-3:]]
    priorities_text = [public_text(item) for item in priorities[:3]]
    generated = now.astimezone(NY).strftime("%Y-%m-%d %H:%M %Z")
    close = session.closed.astimezone(NY).strftime("%Y-%m-%d %H:%M %Z")
    github = progress.get("github_progress", {})
    if not isinstance(github, dict):
        raise ReportError("progress_contract_invalid")
    progress_link = github.get("pull_request_url") or github.get("branch_url")
    if progress_link is not None:
        progress_link = source_url(progress_link)
        if not progress_link.startswith("https://github.com/julianli00/us_stock_qr/"):
            raise ReportError("canonical_progress_link_invalid")
    lines = [
        "*每日研究 / 策略状态*" if zh else "*Daily research / strategy status*",
        f"NYSE 已收盘交易日：{session.day}；收盘：{close}" if zh else f"Completed NYSE session: {session.day}; close: {close}",
        f"报告生成：{generated}（不是行情更新时间）" if zh else f"Generated: {generated} (not a market-data timestamp)",
        "",
        f"*主研究：Top30* `{INCUMBENT_ID}`" if zh else f"*Incumbent Top30* `{INCUMBENT_ID}`",
        "静态699只股票；126日动量前30；63日逆波动权重；月度调仓。QQQ/SPY150MA、VIX和回撤过滤，风险关闭转BIL；不融资、不做空。"
        if zh else "Static 699-stock universe; top30 126-day momentum; inverse 63-day volatility; monthly. QQQ/SPY150MA, VIX and drawdown filters; BIL risk-off; no leverage/shorts.",
        f"记录的行情/信号截止：{incumbent['price_session']} / {incumbent['signal_session']}。"
        if zh else f"Recorded price/signal cutoff: {incumbent['price_session']} / {incumbent['signal_session']}.",
        f"保存的历史结果 {start}–{end}：CAGR {cagr:.2%}，Sharpe {sharpe:.4f}，最大回撤 {drawdown:.2%}。"
        if zh else f"Saved historical result {start}–{end}: CAGR {cagr:.2%}, Sharpe {sharpe:.4f}, max drawdown {drawdown:.2%}.",
        "未满足15%回撤门槛；全样本选参，缺少同规则10年/5年独立评估。以上不是账户收益或样本外证明。"
        if zh else "Fails the 15% drawdown gate; full-sample selection, without matching-rule independent 10y/5y evaluation. Not account returns or OOS proof.",
        "",
        f"{'*早期观察独立产品*' if zh else '*Independent early watch*'} `{WATCH_ID}`",
        f"已记录的价格/信号截止：{early['price_session']} / {early['signal_session']}；新闻截至{early['news_end']}。9月事件不能验证8月信号。"
        if zh else f"Recorded price/signal cutoff: {early['price_session']} / {early['signal_session']}; news through {early['news_end']}. September events cannot validate August signals.",
        "",
        f"{'*独立导入研究*' if zh else '*Independent imported research*'} `{SOURCE_ID}`",
        f"历史研究截至{source_end}；{source_count}个配置，同规则10年/5年联合通过{pass_count}个。未取代Top30，也不继承其业绩。"
        if zh else f"Historical research through {source_end}; {source_count} configurations, {pass_count} joint matching-rule 10y/5y passes. Does not replace or inherit Top30.",
        "前瞻记录因连续性/数据缺口暂停；不回填，不自动重启。"
        if zh else "Forward observations remain paused for continuity/data gaps; no backfill or automatic restart.",
        "",
    ]
    if ideas:
        lines.append(f"*本次新增观察：{len(ideas)} / 最多5*" if zh else f"*NEW watch ideas: {len(ideas)} / maximum 5*")
        for idea in ideas:
            lines.append(f"- {idea['ticker']}：{idea['rationale']} <{idea['url']}|source>")
            if "levels" in idea:
                level = idea["levels"]
                lines.append(
                    f"  Research levels: entry {level['entry_low']:.2f}–{level['entry_high']:.2f}; "
                    f"stop {level['stop']:.2f}; targets {level['take_profit_1']:.2f} / {level['take_profit_2']:.2f}."
                )
        lines.append(f"独立观察数据截止：{session.day}；仅研究观察，不代表成交。" if zh else f"Independent watch cutoff: {session.day}; research only, not executions.")
    else:
        if watch_ready:
            lines.append("*本次新增观察：0。没有尚未发布且符合条件的新观察，不凑数。*" if zh else "*NEW watch ideas: 0. No unpublished qualifying ideas; no quota filling.*")
        else:
            lines.append("*本次新增观察：0。数据仍未就绪，不提供新的买卖或价格区间建议。*" if zh else "*NEW watch ideas: 0. Data not ready; no fresh trade or price-level guidance.*")
    lines.extend([
        "",
        "*数据阻断*" if zh else "*Data blockers*",
        "\n".join(
            "- " + (BLOCKERS_ZH.get(code, "独立观察输入校验未通过；详细原因保留在结构化状态中。") if zh else code)
            for code in blockers
        ) if blockers else ("无日期/证据阻断；仍仅研究。" if zh else "No date/evidence blockers; research only."),
        "",
        "*最新研究进展*" if zh else "*Research progress*",
        *[f"- {item}" for item in completed_text],
        "*下一步优化*" if zh else "*Next optimization priorities*",
        *[f"- {item}" for item in priorities_text],
        "",
        "手工持仓、成交和Discord通知独立维护，本报告不读取或改写。策略自动调参关闭。"
        if zh else "Manual holdings, executions and Discord remain separate and unread/unmodified. Automatic retuning is disabled.",
        "GitHub进展采用审阅后的显式提交；未配置每日自动推送。" if zh else "GitHub progress uses reviewed explicit commits; no automatic daily push is configured.",
    ])
    if progress_link:
        merged = github.get("merged") is True
        lines.append(
            f"GitHub进展：<{progress_link}|整合进展与PR>（{'已合并到main' if merged else '尚未合并到main'}）。"
            if zh else f"GitHub progress: <{progress_link}|integration progress and PR> ({'merged' if merged else 'not merged'} into main)."
        )
    keys = tuple(idea["key"] for idea in ideas)
    digest = canonical_hash({"session": session.day, "registry": registry, "source": source, "progress": progress, "ideas": ideas, "blockers": blockers})
    return ResearchReport("\n".join(lines), session.day, digest, now >= session.report_after, keys, tuple(blockers), watch_ready)


def report_from_root(
    root: Path, now: datetime, *, seen: Iterable[str] = (), language: str = "zh",
) -> ResearchReport:
    calendar = ExchangeCalendar(read_json(root / "config/nyse_calendar.json"))
    return build_report(load_inputs(root), calendar, now, seen=seen, language=language)
