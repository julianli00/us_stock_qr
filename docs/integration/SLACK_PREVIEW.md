# Reviewed status-only Slack preview

Representative localized dry-run captured before the approved first delivery.
The research inputs and substantive warnings were reviewed; this is not a
channel-history export. Its then-current test count describes the first
checkpoint; the expanded final suite is recorded in `STATUS.md`.
The accepted receipt is separately recorded in `DELIVERY_RECEIPT.json`.

---

**每日研究 / 策略状态**

NYSE 已收盘交易日：2026-10-05；收盘：2026-10-05 16:00 EDT

报告生成：2026-10-06 12:14 EDT（不是行情更新时间）

**主研究：Top30** `user_nolev_top30_m126_invvol_m`

静态699只股票；126日动量前30；63日逆波动权重；月度调仓。QQQ/SPY150MA、VIX和回撤过滤，风险关闭转BIL；不融资、不做空。

记录的行情/信号截止：2026-05-08 / 2026-05-08。

保存的历史结果 2015-01-02–2026-05-08：CAGR 29.54%，Sharpe 1.1867，最大回撤 28.31%。

未满足15%回撤门槛；全样本选参，缺少同规则10年/5年独立评估。以上不是账户收益或样本外证明。

**早期观察独立产品** `early_accumulation_signal_v1`

已记录的价格/信号截止：2026-08-20 / 2026-08-20；新闻截至2026-09-21。9月事件不能验证8月信号。

**独立导入研究** `us_quant_research_catalogue_52`

历史研究截至2026-10-05；52个配置，同规则10年/5年联合通过0个。未取代Top30，也不继承其业绩。

前瞻记录因连续性/数据缺口暂停；不回填，不自动重启。

**本次新增观察：0。数据仍未就绪，不提供新的买卖或价格区间建议。**

**数据阻断**

- Top30行情仍停在历史截止日，未覆盖本次已收盘交易日。
- Top30信号尚未按本次交易日重新验证，不能当作今日新信号。
- 旧早期观察行情已过期，与本次交易日不一致。
- 旧早期观察信号已过期，不能改写日期后重新推荐。
- 资讯晚于旧信号日期，不能证明当时已知的催化。
- 旧资讯缺少首次可得时间，点时证据尚未证明。
- 尚无通过来源哈希与同截止日校验的独立观察快照。

**最新研究进展**

- 已将验证过的代码检查点推送至本仓库整合分支，并建立草稿PR，尚未合并。
- 独立研究环境357项测试通过，原有Python3.9策略包保持独立。
- 研究日报30项确定性保护测试通过，覆盖并发去重和发送结果不明的处理。

**下一步优化**

- 先取得同一截止日、可证明首次可得时间并通过来源哈希校验的观察输入，不改写旧信号日期。
- 使用相同窗口、成本与Sharpe定义比较策略，再讨论优化，不混用不同策略的历史业绩。
- 预注册有界研究假设，保留失败结果和暂停的前瞻记录；不回填、不自动调参。

手工持仓、成交和Discord通知独立维护，本报告不读取或改写。策略自动调参关闭。

GitHub进展采用审阅后的显式提交；未配置每日自动推送。

[GitHub整合进展与PR](https://github.com/julianli00/us_stock_qr/pull/1)（尚未合并到main）。
