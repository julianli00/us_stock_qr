"use strict";

const $ = (id) => document.getElementById(id);
let data = null;
let currentView = "overview";
let page = 0;
const pageSize = 15;
const methodNames = { hypothesis: "研究问题", rule: "固定规则", allocation: "资产配置", budget: "预算方法", target_rule: "目标规则", execution: "执行约定", limitations: "限制", risks: "风险" };
const keyNames = { sharpe_above_1: "夏普 >1", cagr_above_12pct: "净 CAGR >12%", drawdown_at_most_15pct: "回撤 ≤15%", beats_spy: "跑赢同成本 SPY", beats_qqq: "跑赢同成本 QQQ", also_beats_base_cost_spy: "压力跑赢基础 SPY" };
const titles = {
  equity_first_trend200_band025_v1: "权益优先：长趋势开关",
  equity_first_vix_variance_band025_v1: "权益优先：增加 VIX 方差上限",
  equity_first_month_end_commitment_v1: "权益优先：月末目标承诺",
  equity_first_21slot_commitment_v1: "权益优先：21 个交错目标",
  equity_channel63x21_goldtrend200_v1: "价格通道：权益 63/21 + 黄金趋势",
  equity_channel63x21_goldchannel63x21_v1: "价格通道：权益与黄金独立 63/21",
  equity_month_end_option_exit_latch_v1: "期权压力退出锁：月末恢复",
  equity_month_end_six_family_core_v1: "月末权益：六经济家族因子半仓",
  benchmark_opportunity_joint_tail_v1: "联合尾部风险：基准动量机会约束",
  benchmark_tail_budget_growth_v1: "基准尾部预算：机会优先",
  dollar_option_daily_variance_disagreement_v1: "月末股金：日频波动差异代理",
  dollar_option_symmetric_variance_disagreement_v1: "月末股金：对称日频差异代理",
  daily_proxy_nearest_band_execution_v1: "月末持仓带：最近可行点执行",
  daily_proxy_relative_equity_odds_v1: "月末股票内部：半导体/核心相对财富",
  dollar_option_two_segment_variation_v1: "月末股金：隔夜与开盘后两区间代理",
  dollar_guarded_simplex_wealth_average_v1: "月末组合：连续权重财富平均",
  two_segment_anfci_residual_confirmation_v1: "月末美元防御：宏观调整后金融紧张确认",
  ordered_path_growth_budget_v1: "有序路径组合 v1：准备失败，未运行账户",
  ordered_path_growth_budget_precision_v2: "有序路径组合：更严格求解精度 v2",
  two_segment_vvix_uncertainty_confirmation_v1: "两时段组合：VVIX季度上升确认美元防御",
  dollar_option_overnight_range_product_v1: "隔夜＋日内区间乘积：保留美元防御",
  two_segment_core_price_momentum_control_v1: "风格排序：纯价格动量对照",
  two_segment_core_activity_momentum_v1: "风格排序：报告成交活跃度加权",
  causal_two_state_growth_bil_v1: "因果波动状态：股票／BIL组合",
  hmm_core_close_momentum_control_v1: "HMM核心：全日收益动量对照",
  hmm_core_overnight_momentum_v1: "HMM核心：隔夜收益动量",
  hmm_core_intraday_momentum_v1: "HMM核心：日内收益动量",
  hmm_vxn_relative_uncertainty_v1: "VXN相对不确定性：半导体／风格核心",
  symmetric_garch_growth_bil_v1: "条件方差：对称GARCH股票／BIL对照",
  downside_gjr_growth_bil_v1: "条件方差：负收益冲击GJR股票／BIL",
  expert_pool_equal_weight_v1: "策略专家：固定三专家等权",
  expert_pool_rolling_pair_v1: "策略专家：滚动净表现／风险／成本择优",
  expert_pool_equal_weight_execution_v2: "策略专家等权：发布适配修正 v2",
  expert_pool_rolling_pair_execution_v2: "策略专家滚动择优：发布适配修正 v2",
  expert_pool_four_equal_weight_v1: "四专家：加入净值反馈防御的等权对照",
  expert_pool_four_rolling_pair_v1: "四专家：同窗净表现／风险／成本滚动择对",
  semiconductor_option_proxy_dollar_observed_band025: "隐含波动预算 + 美元防御 + 持仓偏差带",
  semiconductor_option_proxy_observed_band025: "隐含波动预算 + 持仓偏差带",
  macro_real_yield_factor_tilt: "实际利率因子倾斜",
  semiconductor_dollar_half_guard: "半导体 / 因子 + 美元防御",
  bounded_option_dollar_pdp_momentum_full: "动量实现：PDP 完全替换 MTUM",
  bounded_option_dollar_pdp_momentum_equal_mix: "动量实现：MTUM / PDP 等分",
  bounded_option_dollar_technology_full: "产业实现：科技 XLK 替换半导体 SOXX",
  bounded_option_dollar_nasdaq_full: "产业实现：Nasdaq-100 QQQ 替换半导体 SOXX",
  bounded_treasury_defense_ief: "防御包：中期美债 IEF + 真实股债波动预算",
  bounded_treasury_defense_tlt: "防御包：长期美债 TLT + 真实股债波动预算",
};
const words = { momentum: "动量", dollar: "美元", gold: "黄金", factor: "因子", semiconductor: "半导体", quality: "质量", risk: "风险", yield: "利率", trend: "趋势", volatility: "波动", equity: "股票", credit: "信用", unemployment: "失业", housing: "住房", growth: "成长" };

function el(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function append(parent, ...children) { children.forEach((child) => child && parent.append(child)); return parent; }
function pct(value) { return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—"; }
function decimal(value) { return typeof value === "number" ? value.toFixed(4) : "—"; }
function label(row) { return titles[row.id] || row.title; }
function date(value) { return value ? value.slice(0, 10) : "未评估"; }
function badge(row) {
  const className = row.status === "technical_failure" || row.status === "rejected_historical" ? "fail" : row.full_gate_pass ? "pass" : "wait";
  return el("span", `tag ${className}`, data.status_names[row.status] || row.status);
}
function evidence(path) {
  const link = el("a", "", "查看原始公开证据");
  link.href = `/evidence/${path}`; link.target = "_blank"; link.rel = "noopener noreferrer";
  return link;
}
function button(text, action, className = "primary") {
  const node = el("button", className, text); node.type = "button"; node.addEventListener("click", action); return node;
}
function gateTable(row) {
  const dual = row.scope === "ssd_v1";
  const wrap = el("div", "table-wrap");
  const table = el("table", "gate-table");
  const heading = el("tr");
  ["路径", "净年化 CAGR", "同成本 SPY", ...(dual ? ["同成本 QQQ"] : []), "净超额夏普", "最大回撤", dual ? "全部门槛" : "SPY 验收"].forEach((text) => heading.append(el("th", "", text)));
  append(table, append(el("thead"), heading));
  const body = el("tbody");
  row.paths.forEach((path) => {
    const tr = el("tr");
    tr.append(el("td", "", `${path.years} 年 / ${path.scenario === "base" ? "基础" : "压力"}`));
    if (!path.metrics) {
      const missing = el("td", "muted", "该路径未完成；不填充收益或通过指标"); missing.colSpan = dual ? 6 : 5; tr.append(missing);
    } else {
      tr.append(el("td", "", pct(path.metrics.cagr)));
      tr.append(el("td", "", pct(path.benchmark?.cagr)));
      if (dual) tr.append(el("td", "", pct(path.qqq_benchmark.cagr)));
      if (row.scope === "legacy") {
        tr.append(el("td", "", decimal(path.metrics.sharpe)), el("td", "", pct(path.metrics.max_drawdown)), el("td", "muted", "原协议，见证据"));
      } else {
        tr.append(append(el("td"), el("span", `gate ${path.gates.sharpe_above_1 ? "pass" : "fail"}`, decimal(path.metrics.sharpe))));
        tr.append(append(el("td"), el("span", `gate ${path.gates.drawdown_at_most_15pct ? "pass" : "fail"}`, pct(path.metrics.max_drawdown))));
        const passed = dual ? Object.values(path.gates).every(Boolean) : path.gates.beats_spy;
        tr.append(append(el("td"), el("span", `gate ${passed ? "pass" : "fail"}`, passed ? "通过" : "未通过")));
      }
    }
    body.append(tr);
  });
  append(table, body); wrap.append(table); return wrap;
}
function ssdOverview(ssd) {
  const section = el("section");
  section.append(el("p", "notice", `SSD 新口径：${ssd.registered} 项登记，${ssd.completed} 项完成，${ssd.pending ?? 0} 项待完成，${ssd.technical_failures ?? 0} 项技术失败，${ssd.qualified} 项通过全部历史门槛。净 CAGR >12%、夏普 >1、回撤 ≤15%，并同时跑赢 SPY / QQQ；不是独立前瞻证明。`));
  const catalog = ssd.factor_catalog_extension;
  if (catalog) {
    const notice = el("div", "notice");
    append(notice,
      el("p", "", `SSD 目录新增 ${catalog.new_definition_count} 个定义：${catalog.new_factors.map((row) => row.title).join("、")}。原 ${catalog.preserved_definition_count} 个定义不改写；合并为 ${catalog.combined_definition_count} 个定义、${catalog.combined_family_count} 个经济家族，不代表独立 alpha。`),
      el("p", "muted small", `仅追加 ${catalog.appended_symbols.join(" / ")} 历史数据，保留原 ${catalog.preserved_asset_count} 项资产与 RF 原值。此项准入没有评估策略，也没有增加合格冠军。`),
      evidence(catalog.evidence));
    section.append(notice);
  }
  ssd.candidates.forEach((row) => {
    const article = el("article", "conclusion");
    append(article, append(el("div", "conclusion-title"), append(el("div"), el("h2", "", label(row)), el("p", "candidate-id", row.id)), badge(row)), gateTable(row));
    const failed = row.paths.flatMap((path) => Object.entries(path.gates).filter(([, pass]) => !pass).map(([key]) => `${path.years}年${path.scenario === "base" ? "基础" : "压力"}：${keyNames[key]}`));
    const incomplete = row.status === "pending" || row.status === "technical_failure";
    article.append(el("p", "gate-note", incomplete
      ? `${row.status === "technical_failure" ? "技术失败，保留已经产生的路径和失败记录。" : "尚未完成全部账户及审阅，不计为完成或通过。"}${failed.length ? `已完成路径未通过：${failed.join("；")}。` : ""}`
      : failed.length ? `未通过：${failed.join("；")}。` : "四条历史路径通过；仍无独立前瞻证明或交易权限。"));
    const control = ssd.candidates.find((candidate) => candidate.id === "equity_first_month_end_commitment_v1");
    if (row.id === "equity_month_end_six_family_core_v1" && !incomplete && control) {
      const pairs = row.paths.map((path) => [path, control.paths.find((prior) => prior.years === path.years && prior.scenario === path.scenario)]);
      const complete = pairs.length === 4 && pairs.every(([current, prior]) => current.metrics && current.execution && prior?.metrics && prior.execution);
      if (complete && pairs.every(([current, prior]) =>
        current.metrics.cagr < prior.metrics.cagr && current.metrics.sharpe < prior.metrics.sharpe &&
        current.execution.annualized_two_sided_turnover > prior.execution.annualized_two_sided_turnover &&
        current.execution.model_order_count > prior.execution.model_order_count &&
        current.execution.total_cost_usd > prior.execution.total_cost_usd)) {
        const worseDrawdowns = pairs.filter(([current, prior]) => current.metrics.max_drawdown > prior.metrics.max_drawdown).length;
        article.append(el("p", "conclusion-description", `与已存月末来源相比，4 条路径的净 CAGR 与夏普均下降，换手、模型订单和费用均上升；${worseDrawdowns} 条路径的回撤也更差。新增家族未改善本次固定结构的整体表现。这包含执行差异，不是单资产收益归因或独立 alpha。`));
      }
    }
    const table = el("table"), heading = el("tr"), body = el("tbody");
    ["路径", "年化双边换手（含初始建仓）", "模型订单数", "累计费用 USD"].forEach((value) => heading.append(el("th", "", value)));
    table.append(append(el("thead"), heading));
    row.paths.filter((path) => path.execution).forEach((path) => body.append(append(el("tr"),
      el("td", "", `${path.years}年 / ${path.scenario === "base" ? "基础" : "压力"}`),
      el("td", "", `${path.execution.annualized_two_sided_turnover.toFixed(2)} 倍`),
      el("td", "", path.execution.model_order_count),
      el("td", "", path.execution.total_cost_usd.toFixed(2)))));
    table.append(body);
    if (body.childElementCount) article.append(append(el("div", "table-wrap"), table));
    section.append(article);
  });
  const references = el("p", "notice");
  const cohorts = ssd.cohorts || [ssd];
  const links = cohorts.flatMap((cohort, index) => [
    [cohort.registration_evidence, `批次 ${index + 1} 事前登记`],
    ...(cohort.results_evidence ? [[cohort.results_evidence, `批次 ${index + 1} 完整结果`]] : []),
  ]);
  links.push([ssd.benchmark_evidence, "全额持有双基准"]);
  links.forEach(([path, title], index) => {
    if (index) references.append(document.createTextNode(" / "));
    const link = evidence(path); link.textContent = title; references.append(link);
  });
  section.append(references);
  return section;
}
function progressLine(count, limit) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${limit} 6`); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("class", "progress-svg"); svg.setAttribute("aria-hidden", "true");
  [ [limit, "#d8e5ef"], [Math.min(count, limit), "#214e76"] ].forEach(([width, fill]) => {
    const rect = document.createElementNS(svg.namespaceURI, "rect");
    for (const [key, value] of Object.entries({ x: 0, y: 0, width, height: 6, rx: 1, fill })) rect.setAttribute(key, value);
    svg.append(rect);
  });
  return svg;
}
function overview() {
  const root = $("overview"); root.replaceChildren();
  const ssd = data.ssd_research;
  const intro = el("div", "overview-head");
  const headline = ssd ? "双基准研究，投资目标尚未验证。" : data.summary.qualified ? "历史达标，仍需独立前瞻" : "还没有达标冠军。";
  append(intro, append(el("div"), el("h1", "", headline), el("p", "", ssd ? "新 SSD 研究与迁移前记录分开核验。保留全部失败，不将较高收益或较低回撤拼成一个通过结果。" : "这是持续研究的完整账本。更低的回撤、更高的历史收益都只是单项表现，只有同一策略通过全部四条路径，才算历史候选。")));
  const budget = el("div", "budget");
  if (ssd) {
    const cutoff = new Date(ssd.deadline_beijing).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false });
    append(budget, el("strong", "", ssd.authorization_window_open ? "窗口内不限配置次数" : "已到研究截止时间"),
      el("p", "", `截止北京时间 ${cutoff}，之后不再启动研究。历史 ${data.budget.used}/${data.budget.cap} 配置轮次保持原记录，不是当前额度。`));
  } else {
    append(budget, el("strong", "", `${data.budget.used} / ${data.budget.cap} 个配置`), el("p", "", `本轮已用名额，剩余 ${data.budget.remaining} 个。耗尽后复盘，不无限调参。`));
    const track = el("div", "budget-track"); for (let i = 0; i < data.budget.cap; i++) track.append(el("span", i < data.budget.used ? "used" : ""));
    budget.append(track);
  }
  intro.append(budget); root.append(intro);
  if (ssd) append(root, ssdOverview(ssd), append(el("div", "section-heading"), el("h2", "", "迁移前研究记录，原口径保留")));
  const strip = el("div", "metrics-strip");
  [[data.summary.completed, ssd ? "原口径完成配置" : "完成配置"], [data.summary.reviews, ssd ? "原 SPY 协议审阅" : "当前协议审阅"], [data.summary.definitions, `${data.summary.economic_families} 个经济家族`], [data.summary.technical_failures, "保留的技术失败"], [data.summary.source_audit_paths, "独立来源审计路径"]].forEach(([value, title]) => strip.append(append(el("div"), el("strong", "", value), el("span", "", title))));
  root.append(strip);
  const leader = data.strategies.find((row) => row.id === data.leader_id);
  if (leader) {
    const conclusion = el("article", "conclusion");
    append(conclusion, append(el("div", "conclusion-title"), append(el("div"), el("h2", "", ssd ? "原口径低回撤研究对照" : "当前低回撤领先研究对照"), el("p", "candidate-id", leader.id)), badge(leader)));
    conclusion.append(el("p", "conclusion-description", data.ranking));
    if (leader.id === "semiconductor_option_proxy_dollar_observed_band025") conclusion.append(el("p", "conclusion-description", "股票内部为半导体 SOXX 一半、MTUM / VLUE / QUAL / USMV 各占其余一半的四分之一；月末用 GVZ 与 VIX 的相对隐含波动分配股票和黄金，美元走强时减半转 BIL。实际持仓偏离目标达到 2.5 个百分点才调仓；不交易期权。"));
    conclusion.append(gateTable(leader));
    const failed = leader.paths.flatMap((path) => Object.entries(path.gates).filter(([, pass]) => !pass).map(([key]) => `${path.years}年${path.scenario === "base" ? "基础" : "压力"}：${keyNames[key]}`));
    conclusion.append(el("p", "gate-note", failed.length ? `仍未达标：${failed.join("；")}。` : "全部历史门槛通过，但不是实盘资格或独立前瞻证明。"));
    append(conclusion, append(el("div", "conclusion-actions"), button("查看配置与四路径详情", () => showDetail(leader.id))));
    root.append(conclusion);
  } else root.append(el("div", "notice", "当前没有四路径完整且回撤均不超过15%的配置。未用其他协议或不完整结果替补。"));
  const below = el("div", "below");
  const forward = append(el("article", "panel"), el("h2", "", "原前瞻模型，保持独立"));
  data.forward.forEach((model) => {
    append(forward, append(el("div", "model"),
      append(el("div", "model-head"), el("strong", "", titles[model.id] || model.id), el("span", "tag", `${model.sessions} / ${model.minimum_sessions} 日`)),
      progressLine(model.sessions, model.minimum_sessions),
      el("p", "", `最新真实收益日 ${model.latest_session || "暂无"}。夏普 ${decimal(model.sharpe)}，年化 ${pct(model.cagr)}。不替换成回测赢家。`)));
  });
  const operations = append(el("article", "panel"), el("h2", "", "最近的输入与执行"));
  const sourceMessage = data.daily.reason?.includes("SPY contains missing") ? "SPY 收盘价及复权收盘价缺失，已停止后续流程。" : data.daily.reason || data.daily.status;
  append(operations, el("p", "", sourceMessage), el("p", "muted small", `尝试观察 ${data.daily.session}；最新接受 ${data.daily.latest_accepted_session}。新增真实日：${data.daily.new_sessions}。来源失败不等于策略亏损，也不补数据继续算。`), evidence(data.daily.evidence));
  append(below, forward, operations); root.append(below);
  if (data.round_review) {
    const review = append(el("article", "panel round-review"), el("h2", "", `${ssd ? "历史六配置轮次" : "本轮复盘"}：6 个配置，${data.round_review.qualified} 个历史达标`));
    review.append(el("p", "muted small", ssd ? "旧轮次已耗尽，不改写、不重置。新的 SSD 研究有独立授权与更严格口径，不受这六个历史名额限制。" : "三批固定假设已全部回测。名额用尽，不重置预算继续扫参数；失败记录完整保留。"));
    review.append(el("p", "muted small scroll-hint", "横向滑动表格查看收益、夏普与回撤。"));
    const table = el("table"), header = el("tr");
    ["固定配置", "十年基础 CAGR", "四路径最低夏普", "四路径最差回撤"].forEach((text) => header.append(el("th", "", text)));
    append(table, append(el("thead"), header));
    const body = el("tbody");
    data.round_review.rows.forEach((row) => body.append(append(el("tr"),
      append(el("td"), button(titles[row.candidate_id] || row.candidate_id, () => showDetail(row.candidate_id), "strategy-name")),
      el("td", "", pct(row.ten_year_base_cagr)), el("td", "", decimal(row.minimum_four_path_sharpe)),
      el("td", "", pct(row.maximum_four_path_drawdown)))));
    append(table, body); append(review, append(el("div", "table-wrap"), table), evidence(data.round_review.evidence)); root.append(review);
  }
  const boundary = el("div", "boundary"); data.warnings.forEach((warning) => boundary.append(el("p", "", warning))); root.append(boundary);
}
function filteredStrategies() {
  const query = $("search").value.toLowerCase().trim();
  const scope = $("scope").value, status = $("status").value;
  const rows = data.strategies.filter((row) => {
    const searchable = `${label(row)} ${row.id} ${row.assets.join(" ")} ${row.factor_ids.map((id) => data.factors.find((f) => f.id === id)?.title || id).join(" ")} ${Object.keys(words).filter((key) => row.id.includes(key)).map((key) => words[key]).join(" ")}`;
    return (!query || searchable.toLowerCase().includes(query)) && (scope === "all" || row.scope === scope)
      && (status === "all" || status === "lowdd" && row.eligible_for_current_ranking && row.max_drawdown <= .15
        || status === "qualified" && row.full_gate_pass || status === row.status);
  });
  if ($("sort").value === "sharpe") rows.sort((a, b) => (b.min_sharpe ?? -Infinity) - (a.min_sharpe ?? -Infinity) || a.id.localeCompare(b.id));
  else if ($("sort").value === "drawdown") rows.sort((a, b) => (a.max_drawdown ?? Infinity) - (b.max_drawdown ?? Infinity) || a.id.localeCompare(b.id));
  else rows.sort((a, b) => (b.reviewed_at || "9999").localeCompare(a.reviewed_at || "9999") || a.id.localeCompare(b.id));
  return rows;
}
function strategyTable() {
  const rows = filteredStrategies();
  const total = Math.max(1, Math.ceil(rows.length / pageSize)); page = Math.min(page, total - 1);
  $("strategy-rows").replaceChildren();
  rows.slice(page * pageSize, (page + 1) * pageSize).forEach((row) => {
    const name = button(label(row), () => showDetail(row.id), "strategy-name");
    name.append(el("span", "identifier", row.id));
    const base = row.paths.find((p) => p.years === 10 && p.scenario === "base");
    $("strategy-rows").append(append(el("tr"), append(el("td"), name), append(el("td"), badge(row)),
      el("td", "", decimal(row.min_sharpe)), el("td", "", pct(row.max_drawdown)),
      el("td", "", pct(base?.metrics?.cagr)), el("td", "", date(row.reviewed_at))));
  });
  $("result-count").textContent = `${rows.length} 条记录 / 共 ${data.strategies.length} 条`;
  $("empty").hidden = rows.length > 0; $("page-label").textContent = `${page + 1} / ${total}`;
  $("previous").disabled = page === 0; $("next").disabled = page + 1 >= total;
}
function factors() {
  const root = $("factors"); root.replaceChildren();
  root.append(append(el("div", "section-heading"), append(el("div"), el("h1", "", "定义不等于发现了 alpha"), el("p", "", "将经济家族、已登记定义、数据可得性和实际策略使用分开看。"))));
  const layout = el("div", "factor-layout");
  const catalog = data.ssd_research?.factor_catalog_extension;
  const summary = append(el("aside", "factor-summary"), el("strong", "", data.summary.definitions),
    el("span", "", data.ssd_research ? "原目录已登记定义" : "已登记定义"),
    el("p", "", `${data.summary.economic_families} 个经济家族。${data.ssd_research ? "原计划另有" : "另有"} ${data.queued.definitions} 个定义和 ${data.queued.strategies} 个配置排队，最早 ${data.queued.not_before}；${data.ssd_research ? "历史计划保持原记录" : "未提前评估"}。`));
  if (catalog) summary.append(el("p", "", `独立 SSD 目录已新增 ${catalog.new_definition_count} 个定义；合并 ${catalog.combined_definition_count} 个定义、${catalog.combined_family_count} 个经济家族。登记不等于完成策略评估。`));
  layout.append(summary);
  const list = el("div", "factor-list");
  data.factors.forEach((factor) => {
    const details = el("details", "factor");
    const registeredLocal = factor.ssd_registration
      ? new Date(factor.ssd_registration.registered_at).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false }) : null;
    const status = factor.ssd_registration
      ? `原目录排队；SSD 目录已于北京时间 ${registeredLocal} 登记`
      : factor.status === "queued" ? "来源已核验，尚未登记" : factor.strategy_count + " 个原目录配置引用";
    append(details, append(el("summary"), append(el("div"), el("h3", "", factor.title), el("span", "muted small", `${factor.family} / ${status}`))));
    const body = el("div", "factor-body"), definition = el("dl");
    [["定义", factor.definition], ["经济理由", factor.rationale], ["限制", factor.limitations], ["所需输入", factor.inputs.join(" / ")]].forEach(([title, value]) => append(definition, el("dt", "", title), el("dd", "", value)));
    body.append(definition);
    const links = el("div", "factor-links");
    factor.sources.forEach((url, i) => {
      if (!/^https:\/\/[^/]+/.test(url)) return;
      const link = el("a", "", `来源 ${i + 1}`); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; links.append(link);
    });
    if (factor.ssd_registration) append(body,
      el("p", "muted small", "此处标注独立 SSD 登记，不回写原目录，也不将定义登记计为策略回测。"),
      evidence(factor.ssd_registration.evidence));
    append(body, links, button(data.ssd_research ? "查看原目录相关策略" : "查看相关策略", () => {
      $("search").value = factor.title; $("scope").value = "recurring"; $("status").value = "all"; page = 0; switchView("strategies");
    }, ""));
    details.append(body); list.append(details);
  });
  append(layout, list); root.append(layout);
  root.append(el("p", "boundary", "现金流应计质量与保守资产增长仍缺完整个股点时数据；定义已登记不代表已经完成策略验证。"));
  data.fundamental_data_blockers.forEach((reason) => root.append(el("p", "muted small", reason)));
}
function history() {
  const root = $("history"); root.replaceChildren();
  root.append(append(el("div", "section-heading"), append(el("div"), el("h1", "", "迁移前研究日志"), el("p", "", "保留原协议回测、审阅、技术失败与采集记录。新 SSD 双基准结果与真实时间在「当前结论」单独展示。"))));
  const timeline = el("div", "timeline");
  const groups = new Map();
  data.history.forEach((entry) => {
    const key = date(entry.at); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(entry);
  });
  groups.forEach((entries, key) => {
    const row = el("div", "timeline-day");
    const items = el("div", "timeline-items");
    entries.forEach((entry) => {
      const item = el("article", "timeline-event");
      if (entry.kind === "strategy") {
        const strategy = data.strategies.find((s) => s.id === entry.id);
        append(item, button(label(strategy), () => showDetail(entry.id), ""), el("p", "", `${data.status_names[entry.status] || entry.status} / ${entry.id}`));
      } else append(item, el("strong", "", entry.title), el("p", "", `输入观察 ${entry.session}；${entry.status}`), evidence(entry.evidence));
      items.append(item);
    });
    append(row, el("time", "timeline-date", key), items); timeline.append(row);
  });
  root.append(timeline);
}
function showDetail(identifier) {
  const row = data.strategies.find((r) => r.id === identifier);
  if (!row) return;
  $("detail-title").textContent = label(row);
  const body = $("detail-body"); body.replaceChildren();
  body.append(el("p", "candidate-id", row.id));
  append(body, append(el("div", "detail-summary"), badge(row), el("span", "tag", row.scope === "legacy" ? "旧协议，不参与当前排名" : "当前四路径协议"), el("span", "tag", row.forward_validated ? "独立前瞻验证" : "未取得独立前瞻验证")));
  if (row.accounting) body.append(el("span", "tag", row.accounting === "funded-v2" ? "注册资金账务版本 2" : "保留的原账务版本"));
  if (row.failure_reason) body.append(el("div", "notice error", row.failure_reason));
  if (row.paths.length) body.append(gateTable(row));
  else body.append(el("p", "empty", row.status === "pending" ? "此配置已登记，但尚无完整评估结果。" : "技术失败记录保留；没有用不完整路径补齐结果。"));
  if (row.scope === "recurring") {
    body.append(el("p", "disclaimer", "四条路径分别使用独立 USD 10,000 账户；基础 5bp / 延迟1交易日，压力20bp / 延迟2交易日，每笔USD1。压力路径还必须跑赢基础成本SPY。回撤从初始资本高点计算；不拼指标、不把接近门槛四舍五入为通过。"));
    row.paths.forEach((path) => {
      const missed = Object.entries(path.gates).filter(([, pass]) => !pass).map(([key]) => keyNames[key]);
      body.append(el("p", "small muted", `${path.years}年${path.scenario === "base" ? "基础" : "压力"}：${missed.length ? "未通过 " + missed.join("、") : "全部通过"}`));
    });
  }
  Object.entries(row.description).forEach(([name, value]) => body.append(append(el("section", "rule"), el("h3", "", methodNames[name] || name), el("p", "method-text", value))));
  body.append(append(el("section", "rule"), el("h3", "", "固定配置"), el("p", "method-text", JSON.stringify(row.configuration, null, 2))));
  if (row.original_assessment) body.append(append(el("section", "rule"), el("h3", "", "原协议结论（不改写成当前门槛）"), el("p", "method-text", JSON.stringify(row.original_assessment, null, 2))));
  if (row.policy_evidence) body.append(append(el("div", "detail-evidence"), evidence(row.policy_evidence)));
  if (row.scope === "recurring") body.append(append(el("div", "detail-evidence"), evidence(data.source_snapshot)));
  $("detail").showModal();
}
function switchView(view) {
  currentView = view;
  document.querySelectorAll(".view").forEach((node) => { node.hidden = node.id !== view; });
  document.querySelectorAll("[data-view]").forEach((node) => {
    if (node.dataset.view === view) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  });
  if (data && view === "strategies") strategyTable();
}
async function load() {
  $("refresh").disabled = true;
  try {
    const response = await fetch("/api/research", { cache: "no-store" });
    if (!response.ok) {
      const error = await response.json(); throw new Error(error.error || `HTTP ${response.status}`);
    }
    data = await response.json();
    $("error").hidden = true; $("loading").hidden = true;
    $("freshness").textContent = `记录更新 ${data.updated_on} · 本地读取 ${new Date(data.read_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
    $("strategy-count").textContent = data.strategies.length; $("factor-count").textContent = `${data.summary.definitions} + ${data.queued.definitions}`;
    $("history-end").textContent = data.history_end;
    overview(); factors(); history(); switchView(currentView);
  } catch (error) {
    $("loading").hidden = true;
    $("error").textContent = `研究记录读取失败：${error.message}。请检查已发布的进度指针和证据；未使用虚构或空白数据替代。${data ? "下方保留的是上一次成功读取的记录，不代表已更新。" : ""}`;
    $("freshness").textContent = "记录刷新失败，未取得新状态";
    $("error").hidden = false;
  } finally { $("refresh").disabled = false; }
}
$("refresh").addEventListener("click", load);
document.querySelectorAll("[data-view]").forEach((node) => node.addEventListener("click", () => { if (data) switchView(node.dataset.view); }));
["search", "scope", "status", "sort"].forEach((id) => $(id).addEventListener(id === "search" ? "input" : "change", () => { page = 0; if (data) strategyTable(); }));
$("previous").addEventListener("click", () => { page--; strategyTable(); });
$("next").addEventListener("click", () => { page++; strategyTable(); });
$("close-detail").addEventListener("click", () => $("detail").close());
load();
setInterval(() => { if (!document.hidden && !$("refresh").disabled && !$("detail").open) load(); }, 60000);
