# Canonical US equity research

`julianli00/us_stock_qr` is the canonical home for integration code, research
definitions, failed trials and reviewed progress. This is research software,
not authority to trade or to modify user-maintained holdings.

## Separate research lanes

| Lane | Preserved evidence | Boundary |
| --- | --- | --- |
| Incumbent Top30 | Static 699-stock, 126-day momentum, inverse-volatility monthly strategy; saved 2015-01-02 through 2026-05-08 result | CAGR 29.54%, Sharpe 1.1867, drawdown 28.31%; full-sample selection, not independent OOS; fails the 15% drawdown gate |
| Early watch | Price/volume plus optional event evidence; legacy prices/signals through 2026-08-20 | September evidence cannot validate August signals; historical news/Reddit coverage incomplete |
| [Isolated workbench](research/workbench/README.md) | 52 registered configurations through 2026-10-05; zero joint 10-year/5-year passes | Separate package/environment, different objectives and assumptions; forward record remains paused |
| Manual holdings | Private user-maintained facts | Never loaded, overwritten or published by the research reporter |

The checked baseline is in
[`config/research_strategy_registry.json`](config/research_strategy_registry.json).
It is a dated, curated record, **not a live market refresh**. Existing local
strategy and holdings material is preserved; only an audited subset is published.
No existing parameter search or broker workflow is automatically activated.
The [incumbent source slice](research/incumbent/README.md) includes the pure
Top30 algorithm, its original 699-stock configuration and cost engine, and the
distinct original early scanner. This is runnable strategy code, not only
descriptive metadata; the full legacy search/holdings application is not merged.

## Safe daily research report (Python 3.9+)

The local incumbent environment is retained. A new installation can use a
separate Python environment and `pip install -r requirements-reporting.txt`.
The reporting core itself uses only the standard library; the compatibility
module retains its pandas dependency.

```bash
.venv/bin/python -B scripts/test_research_reporting_guard.py
.venv/bin/python -B scripts/test_incumbent_baseline_guard.py
.venv/bin/python -B scripts/send_slack_signal_once.py --dry-run
```

Preview is the default. It reads no credentials, performs no HTTP requests, and
writes no outbox, legacy state, logs, positions or fills. Legacy Slack callers
and previews now use the same canonical bridge before the old mixed-date
builder/hash. Legacy `force`, signal paths and account options cannot opt back
into that trading payload. Discord behavior is unchanged.
Its existing single-stock selection artifact has an explicit legacy-only
compatibility builder so the canonical Slack formatter does not remove the
independent Discord selection metadata. The reporter never invokes that path.

Real delivery requires explicit `--send` and the locally configured
`SLACK_WEBHOOK_URL`; never put its value in a command, repository or issue.
The intended cadence is after 06:00 Asia/Shanghai, with a bounded 15-minute
catch-up invocation and one report per completed NYSE session. The pinned
calendar covers 2024-2028, including known early closes and ad-hoc closures;
missing, modified or out-of-range calendars fail closed. Future unexpected
exchange closures require a reviewed snapshot update.
The machine must be awake and online; 06:00 is the earliest eligible time,
not a promise of an exact wakeup. Missed polls coalesce into the latest
completed session, without backfilling forward research observations.

The private outbox is under `artifacts/private/research_reporting/`. Its lock
and atomic writes prevent ordinary repeat/restart/concurrent duplicate sends.
A timeout or crash after reservation is **UNKNOWN**, not success or permission
to retry. Incoming webhooks cannot prove remote exactly-once delivery after an
ambiguous acknowledgement; reconciliation is required. An HTTP 200/`ok`
receipt is not a claim that channel history was independently read.
The approved first receipt and actual scheduler verification are recorded in
[`docs/integration/DELIVERY_RECEIPT.json`](docs/integration/DELIVERY_RECEIPT.json).

The installed job runs from this in-place checkout. Keep the tested code and
environment available: changing branches, deleting the environment or moving
the checkout affects the job. It does not fetch, switch branches or auto-merge
the integration PR. After a reviewed deployment, the guarded installer can be
inspected with `scripts/install_research_report_launchd.py --dry-run`; `--install`
requires an already-confirmed current-session outbox receipt and persistent
containment of the old service before bootstrapping.

## Point-in-time watch input contract

Without an independently supplied, verified watch snapshot the report is
status-only and emits zero ideas. It remains useful: rules, dated historical
results, blockers, progress and optimization priorities. It never fills a
five-name quota or turns stale records into current price-level advice.

An optional local producer may supply four files under
`artifacts/private/research_reporting/watch/`:

- `snapshot.json`: schema 1, origin `us_stock_qr`, strategy ID
  `early_accumulation_signal_v1`, session, timezone-aware `decision_at`,
  `order_authority: false`, `input_sha256` and `ideas`.
- `prices.json`: `rows` with ticker, exact session, positive close,
  timezone-aware first `known_at`, and `basis: adjusted_close`. Every idea
  and SPY/QQQ/VIX must have the same completed-session cutoff.
- `events.json`: `rows` with event ID, ticker, original `published_at`,
  first `known_at`, and a public HTTPS source URL. Publication and availability
  must precede the decision; neither generated-at nor a backdated label proves
  point-in-time availability.
- `universe.json`: session, `point_in_time: true`, first `known_at`, and members.

The three input files use canonical JSON (`sort_keys=True`,
`separators=(",", ":")`, no trailing newline); their SHA256 values must match
`input_sha256`. The decision must be at or after exchange close plus the pinned
30-minute minimum data-finalization buffer and before the **next session open**,
and must not be later than report generation. The buffer is an earliest
availability time, not a deadline: a 17:30 ET post-close producer is valid.
An idea references matching event IDs, has
its original signal/first-observed session, a stable signal ID and rationale.
Optional research levels must form a valid entry/stop/target range. All inputs
are checked, not just a global maximum date. Published idea identities are
not treated as NEW again. Synthetic examples are in the guard test.

This contract does not create missing point-in-time data, independently prove
provider timestamps, or grant trading authority. There is deliberately no
automatic adapter from the old news CSVs, broker snapshots or holdings importer.

## Isolated imported runtime

`research/workbench` preserves the manifest-selected source, tests,
public configuration, dependency constraints, notices and sanitized evidence.
New research is added in separately allowlisted files; the imported manifest
and its original payload remain unchanged.
Its `us_quant` package requires Python 3.10+ and must not be installed into the
incumbent Python 3.9 environment or added globally to `PYTHONPATH`.

```bash
cd research/workbench
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[paper,dev,research]' -c requirements-research.lock
.venv/bin/python -I -B -m pytest -q
```

Paper adapter source is retained for reproducibility, not activated. No broker
configuration, raw cache or evolving forward ledger was imported. The reporter
reads only `evidence/research_status.json`, never invokes the workbench CLI.

## Preregistered factor research

The first October 7 round introduces eight fixed configurations covering
sector residual momentum, price-trend continuity, inverse-variance exposure
and a growth/gold diversification control. Definitions and paper attribution
are in `research/workbench/config/factor-research.json`; these ETF adaptations
are not claims to replicate the original papers' stock-level results.

The latest explicit primary objective is net excess-return Sharpe above one
and higher net CAGR than SPY, using the same rule in both original ten-year and
five-year windows and in both base and stressed execution. The earlier 20%
CAGR and 15% drawdown thresholds remain separately reported. QLD variants use
real embedded daily-reset leverage, not borrowing or synthetic pre-inception
prices. No result is promoted to live trading authority.

The second round adds six separately registered covariance/diversification
and trend-filter configurations. It was designed after the first results,
so its adaptive research status is explicitly disclosed in
`config/allocation-research.json`. It does not modify the first eight rules.
All six second-round configurations fail the joint primary goal; their
[complete results](research/workbench/evidence/allocation_round_20261007_results.json)
remain available. A third, separately registered six-configuration round
tests quarterly or target-band rebalancing, mechanically fixed weights and
equal mixtures of the already disclosed allocation components.

The published third-round registration can be used directly. Run from
`research/workbench` using its isolated environment:

```bash
.venv/bin/python -I -B -m pytest tests/test_factor_research.py -q
.venv/bin/python -I -B -m us_quant.factor_research stage-data \
  --policy config/implementation-research.json \
  --registration evidence/implementation_round_20261007_registration.json \
  --source-base /path/to/verified/dual-horizon/market \
  --source-supplement /path/to/verified/dual-horizon/legacy-supplement
.venv/bin/python -I -B -m us_quant.factor_research evaluate \
  --policy config/implementation-research.json \
  --registration evidence/implementation_round_20261007_registration.json \
  --output reports/implementation-round-20261007
```

Registration binds code, parameters and both existing snapshot manifests
before evaluation. Staging copies only verified market artifacts into ignored
local data directories and preserves their original retrieval timestamps.
Results, independent accounting paths and all failures stay in a new,
non-overwritten output directory. Sanitized summaries can be published
explicitly; raw prices are not part of the publication boundary.
The historical intervals were already exposed and overlap. Conditional
bootstrap intervals are not independent forward validation or a correction
for selecting the best of many configurations. The old paused forward
ledger and all broker, holdings and messaging services remain untouched.
Each registration is bound to its implementation revision. First-round
reproduction uses commit `3656e098`, and second-round reproduction uses
`56d95478`; changing a recorded hash to accept a later implementation is
not a valid reproduction. New experiments use the
`register` stage with a new policy and a new registration path.

The [third-round results](research/workbench/evidence/implementation_round_20261007_results.json)
retain all six implementation variants. Two pass the registered latest
two-metric target in both horizons and both base/stress scenarios:
`growth_gold_target_change_band` and
`equal_growth_gold_min_variance_ensemble`. They do not satisfy the earlier
15% drawdown cap and are not independently validated future strategies.

An independently specified validation plan checks every retained accounting
path and every success flag, then reports rolling endpoint sensitivity and
50bp/three-session execution diagnostics for **both** qualified candidates:

```bash
.venv/bin/python -I -B -m pytest tests/test_factor_validation.py -q
.venv/bin/python -I -B -m us_quant.factor_validation
```

This requires the retained local ledgers from all three implementation
revisions. It uses `bt` equity and independent `ffn`/excess-return statistics;
it does not change strategy parameters. Its stronger scenarios and all
failures are additional diagnostics, not substituted primary windows.

The [completed independent validation](research/workbench/evidence/factor_validation_20261007.json)
confirms all 20 new configurations, 138 original accounting paths and 24
additional paths. All 72 registered configurations, including the original
archive, remain disclosed. The latest requested **historical** two-metric
thresholds are met, but selection-adjusted alpha and future performance are
not established.

The stronger candidate is `equal_growth_gold_min_variance_ensemble`:
half of the 98% invested budget follows the QLD/GLD inverse-volatility
allocation; half follows the fixed six-month momentum selection and
minimum-variance allocation among QLD, GLD, TLT, IEF and DBC. Remaining
defensive allocation is actual BIL, with 2% idle cash. Targets combine before
trading one account. There is no account borrowing, but QLD has embedded
daily leverage; the observed approximate underlying gross exposure can
exceed one.

| Window/scenario | Net CAGR | Excess Sharpe | Maximum drawdown | Base-cost SPY CAGR |
| --- | ---: | ---: | ---: | ---: |
| Ten years, base | 20.37% | 1.1870 | 19.97% | 15.46% |
| Five years, base | 22.14% | 1.1045 | 20.06% | 14.04% |
| Ten years, 20bp/two-session stress | 19.58% | 1.1373 | 19.68% | 15.46% |
| Five years, 20bp/two-session stress | 20.59% | 1.0160 | 19.78% | 14.04% |

These are USD10,000 historical accounts ending October 5, 2026, before
personal tax, with fractional units, modeled costs and USD1 per order.
Base execution uses 5bp per side at the next session open. **Limitations are
material:** at 50bp per side the candidate's five-year Sharpe falls to
0.9674; only 19/45 rolling ten-year and 53/105 rolling five-year stress
windows meet the primary goal. Its old 15% drawdown target fails, and
conditional excess-CAGR confidence intervals include zero. The target-band
alternative also has rolling-window and stronger-stress failures. No live
signal, order, Slack post or restart of paused forward records resulted.

The [first-round evidence](research/workbench/evidence/factor_round_20261007_results.json)
contains all eight configurations. The growth/gold inverse-volatility control
passes the latest two primary gates in the base ten/five-year windows, but
its ten-year stressed Sharpe is 0.9913 and drawdown is approximately 25%.
It is **not** a base-plus-stress qualified result and fails the old drawdown
goal. The first non-overlapping five-year diagnostic also fails; no forward
performance or statistically established alpha is claimed.

## Portfolio-level refinement

The next bounded round preserves all earlier rules and tests six separately
registered portfolio controls in
`research/workbench/config/portfolio-refinement.json`: a 30% aggregate QLD
target cap, a 12% ex-ante portfolio-volatility ceiling, a 2.5-percentage-point
actual-holdings no-trade band, and the declared combinations. All controls
operate on the combined portfolio rather than separately on its two sleeves.
Reductions move to actual BIL, not fictitious cash interest.

The band compares current closing holdings with the desired target, unlike
the earlier target-change-only experiment. A breached risk limit overrides
the band at the monthly decision. This does **not** guarantee a continuous
weight limit or maximum drawdown: between-decision drift and opening gaps
remain possible. Fractional units, modeled costs and pre-tax returns are
unchanged assumptions; no broker account is accessed.

This round raises the disclosed count from 72 to 78. The unchanged
incumbent is also replayed, but is not counted as another independent trial.
The primary ten/five-year Sharpe and SPY conditions stay unchanged, including
20bp/delayed stress. To qualify as a research improvement, a new configuration
must additionally either satisfy the old 15% drawdown goal in all primary
paths, or pass both 50bp windows with at least 10% lower turnover. It may
not worsen drawdown by more than one percentage point in any primary path.
No rule is automatically promoted to trading or substituted for the incumbent.

From `research/workbench`, after preserving the original local ledgers:

```bash
.venv/bin/python -I -B -m pytest tests/test_portfolio_refinement.py -q
.venv/bin/python -I -B -m us_quant.portfolio_refinement evaluate
```

The evaluator checks all decisions through the original cash-funded simulator
and independent `bt` accounting, recomputes metrics, and requires the unchanged
control to reproduce the complete original ledgers. It retains every new
failure and conditionally adds rolling-window comparisons for qualified
improvements. The original snapshot remains dated October 5; another
historical experiment is not a new independent forward observation.

The [completed six-configuration evidence](research/workbench/evidence/portfolio_refinement_20261007_results.json)
retains every result and 72 independently audited paths, including six
full-ledger matches to the original incumbent. All six variants pass the
base-only ten/five-year primary target; two retain it under stress, but
**none qualifies as an upgrade under the frozen promotion contract**.
The incumbent remains unchanged.

The actual-holdings band gives a modest improvement, not a new validated
strategy. Its five-year results illustrate why order count and dollar
turnover must not be confused:

| Five-year measure | Unchanged incumbent | Actual-holdings 2.5pp band |
| --- | ---: | ---: |
| Base net CAGR | 22.14% | 22.33% |
| Base excess Sharpe | 1.1045 | 1.1118 |
| Base maximum drawdown | 20.06% | 19.78% |
| Base order tickets | 167 | 94 |
| 20bp delayed-stress Sharpe | 1.0160 | 1.0417 |
| 50bp delayed-stress Sharpe | 0.9674 | 0.9957 |
| 50bp annualized turnover | 2.5121 | 2.3860 |

The band removes about 44% of order tickets but only about 5% of traded-dollar
turnover. Sharpe 0.9957 is **below**, not equal to or above, the strict threshold.
The 12% volatility variants reduce some drawdowns to approximately 18%, not
15%, and increase turnover; all fail the five-year stressed Sharpe gate.
These results do not justify tuning the band until it crosses one, changing
cost assumptions, or treating the reused history as independent evidence.
There is no automatic baseline replacement, paper order or persistent
research/notification automation created by this round.

## Unleveraged multifactor stability research

The October 10 request prioritizes genuine multifactor exposure and lower
risk. The new research lane uses four actual, unleveraged US equity factor
ETFs: MTUM (momentum), VLUE (value), QUAL (quality) and USMV (low volatility).
Issuer URLs, economic definitions, inception dates and current benchmark
labels are recorded in
`research/workbench/config/multifactor-stability.json`.

This is explicitly a **multifactor ETF-sleeve portfolio, not direct
stock-level fundamental scoring**. Value and quality exposure comes through
the funds' actual historical portfolios. Today's holdings or financial ratios
are not used to manufacture historical selections. Existing SEC filing-date
utilities do not provide complete historical security identities, constituent
membership and delisted prices; that missing evidence still blocks a qualified
direct-stock backtest. Current issuer definitions do not establish unchanged
index rules throughout the fund's history.

Six fixed configurations compare equal factor exposure, a 60/20/20
factor-equity/IEF/GLD allocation, bounded inverse-volatility factor shares,
common equity-breadth defense, and a 10% portfolio-volatility target.
There is no QLD, leveraged product, shorting or account borrowing. The
four-factor equity control deliberately has no bond/gold defense: it tests
whether adding equity factors alone actually reduces risk.
Every nonzero equity allocation retains all four factor sleeves; defense
scales them together rather than selecting the historical winner.

The stability criterion requires maximum drawdown at most 15% and both lower
drawdown and lower volatility than the prior QLD-containing candidate in
the same ten/five-year base/stress windows. Sharpe above one and outperformance
of SPY remain separately reported; a lower-risk result cannot be silently
declared to satisfy those return objectives. Higher-cost and rolling-window
failures must remain visible. Correlations, market betas and the covariance
effective dimension are diagnostics, not a claim that four funds produce
four independent alpha sources or avoid overlapping stock holdings.

The registration preserves the previous 78 configurations and records six
new ones, for 84 total. From `research/workbench`:

```bash
.venv/bin/python -I -B -m pytest tests/test_multifactor_stability.py -q
.venv/bin/python -I -B -m us_quant.multifactor_stability fetch
.venv/bin/python -I -B -m us_quant.multifactor_stability evaluate
```

The initial 2014 warmup request was rejected before any candidate evaluation:
VLUE contains 27 flat zero-volume bars, the last on August 7, 2015. The
[quality audit](research/workbench/evidence/multifactor_data_quality_20261010.json)
and original registration remain preserved. Registration v2 starts all series
on August 10, 2015, the next real session, without inventing volume or changing
any candidate rule or either formal return window. There are still 290
sessions through the last pre-evaluation month end, enough for the registered
252-session warmup. Available rolling ten-year windows are consequently few;
this reduced diagnostic coverage must not be concealed.

The common comparison endpoint remains October 5, 2026. Original benchmark,
defensive-asset and risk-free snapshot hashes are preserved; the different
factor-fund download vintage is explicitly recorded. Raw prices and issuer
documents remain ignored local inputs. No old strategy is overwritten and
no trading, notification or persistent automation is activated.

The [six complete multifactor results](research/workbench/evidence/multifactor_stability_20261010_results.json)
contain 77 independently accounted paths. Two defense variants satisfy the
registered stability criteria, but **none satisfies the unchanged Sharpe/SPY
return goals**, and none qualifies for a full strategy upgrade.
The lowest worst-drawdown candidate selected by the frozen rule is
`four_factor_bounded_defense_vol10`:

| Window / base costs | Net CAGR | Excess Sharpe | Maximum drawdown | Annual volatility |
| --- | ---: | ---: | ---: | ---: |
| Ten years, prior QLD-containing comparison | 20.37% | 1.1870 | 19.97% | 14.48% |
| Ten years, unleveraged four-factor defense | 6.90% | 0.6063 | 13.93% | 7.42% |
| Five years, prior QLD-containing comparison | 22.14% | 1.1045 | 20.06% | 15.86% |
| Five years, unleveraged four-factor defense | 7.04% | 0.4374 | 13.05% | 7.46% |

This is a lower-risk research prototype, **not a recommendation to replace
the old strategy**. Its defensive assets, lower equity exposure and absence
of embedded leverage matter materially. The all-equity four-factor control
still suffers 34.38% ten-year maximum drawdown. The four factor funds'
ten-year daily-return correlations range from 0.736 to 0.894; their covariance
effective dimension is about 1.31, not four independent return sources.
These diagnostics do not measure historical constituent-level overlap.
There are only two complete rolling ten-year endpoints and 62 five-year
endpoints; overlapping windows are not independent confirmations.

A separately frozen
[attribution plan](research/workbench/evidence/multifactor_attribution_20261010_plan.json)
audits all original paths and replaces the four factor funds with SPY at
exactly the same aggregate equity target, preserving IEF/GLD/BIL allocations,
decision dates and costs. It includes both stability-qualified candidates,
not just the most favorable one:

```bash
.venv/bin/python -I -B -m us_quant.multifactor_attribution evaluate \
  --output reports/multifactor-attribution-20261010-replay
```

This is an equity-budget ablation, not equal-risk matching or an independent
SPY timing strategy. It distinguishes the contribution of factor fund
selection from simply reducing stock exposure; no candidate is retuned.

The [completed attribution audit](research/workbench/evidence/multifactor_attribution_20261010.json)
recomputes all 77 original paths, reproduces all six complete target matrices
and checks 12 budget-matched SPY control paths. The first attribution command
was interrupted; its partial output was not treated as a result. The unchanged
plan was rerun into the new `-replay` directory without overwriting evidence.

In **all 12 comparisons**, the factor portfolio has lower net CAGR and lower
Sharpe than its equity-budget-matched SPY control. For the selected low-risk
prototype under base costs:

| Window | Four-factor CAGR | Same-equity-budget SPY CAGR | Four-factor drawdown | Control drawdown |
| --- | ---: | ---: | ---: | ---: |
| Ten years | 6.90% | 7.54% | 13.93% | 13.82% |
| Five years | 7.04% | 7.82% | 13.05% | 13.16% |

This comparison includes actual differences in modeled order costs from
holding four equity funds rather than one. It does not prove statistical
underperformance or match market beta exactly, but it **does not establish
incremental net value from the factor-fund selection**. Lower equity exposure,
defensive allocation and removal of embedded leverage must not be presented
as newly discovered factor alpha. The research direction remains unleveraged
multifactor stability; the old leveraged strategy is preserved only as a
historical comparison, not promoted as a suitable solution to the new risk
preference. No replacement or live recommendation is authorized by these
results.

## Recurring factor discovery and research versions

The recurring program in `research/workbench/config/research-program.json`
separates new factor **definitions**, registered strategy **configurations**,
audited historical results and actual independent forward evidence. Its
weekly cadence is Saturday 09:00 Asia/Shanghai, with at most two new economic
definitions per cycle. The Copilot session automation performs the literature
and implementation work; the CLI below enforces the journal and review
boundaries. The CLI does not independently invent factors, download arbitrary
code or execute commands stored in a proposal.

The original four studied factor families are retained as references.
This first new discovery batch proposes cash-flow accrual quality and
conservative asset growth. Cash-flow accrual quality is related to the
already studied earnings-quality family, not a new independent source of
alpha. `fundamental_signals.py` reuses the filing-time, period-matched
financial helpers to compute both features without changing the frozen
imported module. Unit fixtures establish calculation behavior, not profitable
stock selection.

Actual public SEC Company Facts requests returned HTTP 403. The
[source availability evidence](research/workbench/evidence/factor_program_source_availability_20261010.json)
and [readiness record](research/workbench/evidence/research_program_data_readiness_20261010.json)
also disclose the missing historical security master, constituent membership,
delisting returns and matched stock prices. A two-company sample, current
stock list or fabricated data must not substitute for these requirements.
No new stock strategy has been evaluated from the new proposals.

From the isolated `research/workbench` environment:

```bash
# Initialize once; existing ledgers cannot be reset with this command.
.venv/bin/python -I -B -m us_quant.research_program init
.venv/bin/python -I -B -m us_quant.research_program cycle \
  --proposals config/factor-discovery-20261010.json \
  --readiness evidence/research_program_data_readiness_20261010.json
.venv/bin/python -I -B -m us_quant.research_program status
```

The private, ignored SQLite journal is `runtime/research-program.sqlite3`.
Transactions and ISO-week keys prevent concurrent/restarted cycles from
registering duplicates. The event hash chain, policy/engine binding and
reconstructed factor/candidate/review records detect altered or removed
history. Reusing a factor under another name does not create a new trial.
Proposals and data-blocked cycles do not increment the 84 previously
evaluated configurations. Readiness must be checked within eight days;
verified capabilities require hashed evidence, not a missing-data default.
These readiness attestations still require source/coverage review: a hash
does not itself establish point-in-time completeness.

Once data is genuinely available, `register-candidate --candidate <spec.json>
--readiness <readiness.json>` freezes at least three economic factor families,
code/configuration hashes, actual market panels, instrument leverage and the
evaluation cutoff **before** testing. A candidate specification supplies
`id`, `factor_ids`, `frozen_files`, `asset_leverage`, `evaluation_as_of`,
`market`, `history_status: exposed_history_not_independent_holdout`,
`leveraged_products_allowed: false` and `order_authority: false`.
`market` holds `open`, `close`, `raw_close`, `volume`, and `risk_free`
CSV references, each with a workbench-relative `path` and `sha256`.

`review --candidate-id <id> --bundle <bundle.json>` does not trust a submitted
Sharpe number or a success boolean. The bundle repeats the frozen market,
cutoff and candidate-spec hash and supplies four `paths`: ten/five years
under base/stress conditions. Each path includes hashed strategy, `strategy_bt`,
SPY, `spy_bt`, weights and target CSVs, plus the prescribed capital, costs,
commission and delay. The reviewer reexecutes the original simulator and
independent `bt` engine from market data/targets, reconstructs metrics and
weights, and checks the exact date ranges. Only net excess Sharpe above one,
drawdown at most 15% and outperformance of SPY in all four paths can qualify.
Stressed strategies must also exceed base-cost SPY.

Qualified results may advance the **research** version ranked by worst-path
Sharpe; rejected versions stay recorded. This is not trading promotion.
The program never enables brokerage submission, automatically deploys a
live strategy, invents forward observations or resumes old paused ledgers.
The recorded 63-session forward-review requirement is a future minimum,
not evidence already collected and not proof of long-run performance.
Versioned `--export` snapshots cannot overwrite changed earlier exports.

The app and host must be available for session automation to run; a saved
schedule is not proof of an actual future execution. Integrity errors must
pause automation for review; supplier access failures are recorded as data
blockers, never bypassed with fake prices or silently relaxed objectives.

The native weekly automation is now configured and read back successfully.
The [schedule receipt](research/workbench/evidence/research_program_automation_20261010.json)
records the next invocation observed at configuration time: **October 17,
2026, approximately 09:00 Asia/Shanghai**. It wakes this same session rather
than modifying the old Slack/Discord services. Pending registered candidates
are resumed under their original frozen inputs, not discarded or counted again.

The [first real cycle](research/workbench/evidence/research_program_2026-W41.json)
registered two new factor definitions and recorded five explicit stock-data
blockers. Together with four prior definitions, the catalog has six definitions
in five economic families; it does not prove five independent sources of
alpha. The number of evaluated strategy configurations is still **84**, with
zero new evaluations, zero qualified research versions and no research
champion. Repeating the same week leaves the event chain and counts unchanged.
**The requested Sharpe/robustness objective remains unfulfilled.** Neither
working automation nor passing synthetic tests constitutes investment evidence.

### Separate actual-fund data scope

The recurring reviewer also supports the explicitly labeled
`factor_etf_portfolio` scope. It validates the original four-factor fund
mandates, actual post-inception history, corporate-action evidence and
unleveraged product identities by loading the frozen multifactor provider
snapshot. Candidate panels must match that snapshot. This is **not** permission
to mark missing stock fundamentals, historical constituents or delistings
as available: the `direct_stock` readiness requirements remain unchanged.
Candidates and reviews record their scope, and an ETF candidate must use the
four actual fund-factor references rather than claiming direct stock signals.

Existing ledgers refuse a changed engine unless an explicit `migrate-engine`
operation supplies the exact previous engine hash and event-chain head.
The migration verifies existing history in a transaction, appends an engine
transition event and preserves the policy, all factors, blocked cycles,
candidates and reviews. It does not reset trial counts or change performance
or live-deployment gates. Migrations require a reviewed code change, not
automatic recovery from unexplained corruption.

The next fixed two-configuration study is in
`research/workbench/config/factor-gold-risk.json`. It preserves equal exposure
to the four factor funds within an equity sleeve and balances that sleeve
against GLD using trailing 63-session volatility. The two variants compare
monthly rebalancing with an additional daily 12% volatility target and
5-percentage-point trade band. Neither uses QLD, leverage or fixed bond
duration. Monthly requests are retained during modeled execution delays;
pending targets cannot be overwritten. A volatility target is not a loss
guarantee, and two allocation variants are not two newly discovered factors.

From the workbench, after the explicit engine migration:

```bash
.venv/bin/python -I -B -m us_quant.factor_gold_risk prepare
.venv/bin/python -I -B -m us_quant.factor_gold_risk register
.venv/bin/python -I -B -m us_quant.factor_gold_risk evaluate
```

Preparation verifies and freezes real inputs but does not compute strategy
performance. Registration goes through the actual recurring program before
evaluation. Outcomes go through its market/target/cost and independent `bt`
replay, not a parallel success flag. Already registered or reviewed work is
preserved and resumed without duplicating trials.

The explicit migration and both preregistrations are recorded in
[the scope snapshot](research/workbench/evidence/research_program_2026-W41_scope_v2.json)
and [candidate receipts](research/workbench/evidence/factor_gold_risk_20261010_registration.json).
The [actual evaluated outcomes](research/workbench/evidence/factor_gold_risk_20261010_results.json)
were submitted through the recurring reviewer, which regenerated all strategy
and SPY paths from market data/targets and independent `bt` accounting.
Both candidates were recorded as `rejected_historical`:

| Rule | Window | Base net CAGR | Base Sharpe | Base drawdown | Stress Sharpe |
| --- | --- | ---: | ---: | ---: | ---: |
| Four-factor/gold monthly risk balance | Ten years | 13.19% | 0.9055 | 20.20% | 0.8776 |
| Four-factor/gold monthly risk balance | Five years | 16.10% | 0.9602 | 18.22% | 0.9200 |
| Same baseline with daily volatility control | Ten years | 11.17% | 0.8367 | 18.19% | 0.7788 |
| Same baseline with daily volatility control | Five years | 13.96% | 0.8896 | 18.35% | 0.8523 |

The [updated program snapshot](research/workbench/evidence/research_program_2026-W41_results_v3.json)
has 86 evaluated configurations, two new completed strategy reviews, and
still no qualifying research champion or version update. The two new
fundamental-factor proposals and all five direct-stock data blockers are
unchanged. ETF research is no longer blocked merely because stock fundamentals
are missing, but working data access does not make either strategy successful.
Weekly automation instructions now distinguish both scopes and retain every
failure; no performance threshold, old ledger or trading permission changed.

### Factor implementation replication

The next controlled comparison changes fund implementations rather than
searching more allocation parameters. The initial intent attempted replacing
MTUM with SPMO and QUAL with SPHQ. Before any candidate performance calculation,
the provider's SPMO history failed the data gate: 238 zero-volume records,
including the formal ten-year interval. The
[source-quality audit](research/workbench/evidence/factor_replication_quality_audit_20261010.json)
and initial intent commit `be2aaaaf` remain preserved. These two initially
planned variants were not admitted as complete strategy backtests.

The data-only revision replaces QUAL quality with SPHQ and retains MTUM
momentum, VLUE value and USMV low volatility. It does not fill volume, shorten
the formal windows, or change the allocation rules after seeing outcomes.
The two variants retain the preceding study's exact monthly risk balance
and optional daily volatility control. Identical synthetic return inputs
produce identical complete targets and accounting in both implementations.
This is two new configurations of existing factor families, not a claim
of two new independent factors.

The current Invesco product pages identify SPMO and SPHQ. The old SPVU
link resolves to a Concentrated QVM product, so it is not used as a stable
value implementation. The public factsheet links returned HTML rather
than PDF; their contents are not claimed as directly verified. Current
product names and actual fund returns do not establish unchanged historical
index methodology. The protocol records the reported SPHQ benchmark change
and starts input history in July 2016; the original ten/five-year return
windows stay unchanged.

Definitions and limitations are in
`research/workbench/config/factor-implementation-replication.json`.
Before calculating returns, the source fingerprints and candidate receipts
must be recorded through the same recurring program:

```bash
.venv/bin/python -I -B -m us_quant.factor_replication fetch
.venv/bin/python -I -B -m us_quant.factor_replication prepare
.venv/bin/python -I -B -m us_quant.factor_replication register
.venv/bin/python -I -B -m us_quant.factor_replication evaluate
```

The adapter revalidates SPHQ's actual history and retains the previously
hashed momentum/value/low-volatility, defensive and benchmark data.
Old source and candidate modules remain byte-preserved; the small separate
replication runner avoids invalidating those frozen implementation hashes
and is tested against the original algorithm. No name substitution is
made in the actual market data or evaluated holdings.

The [quality implementation results](research/workbench/evidence/factor_replication_20261010_results.json)
are actual recurring-program reviews, not just standalone backtests.
Both fixed variants were rejected. Monthly risk balance has ten/five-year
base Sharpe 0.9054/0.9627 and stressed Sharpe 0.8780/0.9237; daily risk control
has base Sharpe 0.8397/0.8979 and stressed Sharpe 0.7869/0.8666. Drawdown remains
approximately 18%-20%. Changing the quality implementation did not solve the
target shortfall.

The [fifth weekly snapshot](research/workbench/evidence/research_program_2026-W41_results_v5.json)
retains 88 evaluated configurations and four rejected program candidates.
The source-rejected SPMO plans are separately disclosed but are not counted as
completed strategy evaluations. There is still no qualified research champion,
live deployment or new independent forward evidence.

### Causal adaptive factor-combination comparison

The next two fixed hypotheses use the original verified four factor funds
plus GLD and BIL. One control keeps 70% of the invested budget in equal factor
equities and 30% in gold. The adaptive method estimates trailing excess means
with a fixed exponential half-life, shrinks the mean/covariance estimates and
solves a constrained mean-variance problem with an explicit turnover penalty.
It updates only at completed month ends from already observed data.

The protocol is `research/workbench/config/adaptive-factor-allocation.json`.
Every factor fund retains at least 2.5% and at most 25% of total capital;
gold is capped at 50%, total investment at 98%, and no leverage is allowed.
All estimation windows, bounds and penalties are fixed before results.
The implementation uses an epigraph for the absolute turnover cost and
checks solver success and final bounds; it does not return equal weights
as a hidden fallback on optimization failure.
This is adaptive allocation of existing exposures, not new economic-factor
discovery and not a replication of Bayesian Dynamic Model Averaging.

```bash
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation prepare
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation register
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation evaluate
```

Both configurations must be registered in the recurring program before
evaluation and meet its unchanged actual-cost, independent-accounting and
dual-horizon performance gates. No historical data is relabeled as a new
independent holdout.

The [adaptive comparison results](research/workbench/evidence/adaptive_factors_20261010_results.json)
record two actual rejections. The static control has ten/five-year base
Sharpe 0.8367/0.8042. Adaptive allocation has base Sharpe 0.8294/0.8125 and
stress Sharpe 0.7658/0.6630, with stressed drawdown approximately 17.5%-17.9%.
It reduces some drawdown but does not meet the original objective.
SLSQP reported intermediate bound-clipping warnings; final optimizer success,
feasibility, actual targets and independent accounting were checked. No
fallback allocation or warning suppression converted a failure to success.
The [sixth weekly snapshot](research/workbench/evidence/research_program_2026-W41_results_v6.json)
retains 90 completed configurations and an empty research-champion slot.

### Option-implied term-structure risk information

The next two preregistered risk overlays use official Cboe daily VIX and
VIX3M observations. These indices measure different option-implied volatility
horizons; they are risk inputs, not tradable holdings or fabricated index
returns. The [source manifest](research/workbench/evidence/volatility_term_source_20261010.json)
records complete coverage of the existing ETF research sessions. The actual
downloaded VIX3M series starts September 18, 2009; no earlier history is assumed.

`config/volatility-term-risk.json` fixes the inversion boundary at VIX
greater than or equal to VIX3M. On inversion, the candidates either halve
or remove the equity sleeve of the unchanged four-factor/gold monthly risk
balance, moving the reduction to BIL while retaining gold. Month-end changes
and binary state transitions issue dated absolute targets for subsequent
opens. Costs and delayed-stress execution remain unchanged. No threshold
grid, index trading, same-close fill or missing-session forward fill is allowed.

```bash
.venv/bin/python -I -B -m us_quant.volatility_term_risk prepare
.venv/bin/python -I -B -m us_quant.volatility_term_risk register
.venv/bin/python -I -B -m us_quant.volatility_term_risk evaluate
```

This adds option-implied risk information rather than modifying the rejected
adaptive allocator. It is not yet evidence of incremental alpha. All
candidate outcomes must still go through the existing recurring reviewer.
Historical end-of-day index availability is not an independently proven
intraday release timestamp and cannot authorize a live decision.

The [actual term-risk outcomes](research/workbench/evidence/term_risk_20261010_results.json)
were also rejected by the recurring reviewer. Halving equity on inversion
gives ten/five-year base Sharpe 0.8692/0.8403 and stress Sharpe 0.7274/0.7688.
Exiting the equity sleeve gives base Sharpe 0.8270/0.7240 and stress
0.5663/0.6309. Transaction activity, delayed execution and missed recoveries
do not provide an improvement under these frozen rules; both outcomes remain
visible rather than being retuned to cross a threshold.

The [latest weekly research snapshot](research/workbench/evidence/research_program_2026-W41_results_v7.json)
contains 92 evaluated configurations in total and eight actual recurring
candidate reviews, all rejected. Two fundamental factor definitions remain
proposals with missing stock data, not completed strategy trials.
There is still no qualified unleveraged multifactor research champion or
independent forward result. The standing weekly process continues to seek
new justified evidence; it does not promise that repeated searches will
necessarily discover a strategy satisfying the requested return/risk goals.

### Fundamental-data feasibility checkpoint

The [official bulk-access record](research/workbench/evidence/sec_bulk_availability_20261010.json)
confirms that a historical SEC quarterly ZIP and the official format document
also returned HTTP 403. Each distribution received one normal request; no
proxy, identity impersonation or access-control bypass was used. The federal
Data.gov catalog was accessible, but its description cannot substitute for
the underlying filings or prove historical security/return coverage.

The [source assessment](research/workbench/evidence/fundamental_source_feasibility_20261010.json)
keeps actual provider statements separate from search summaries.
QuantConnect explicitly distinguishes its security master from the separately
licensed equity price data. Sharadar's free entry-level offering covers
current Dow30 companies, not a verified full historical universe. A search
summary describing HistPrice as complete US-stock PIT data was not confirmed:
the actual page demonstrated cryptocurrency data tooling. None of these
checks admits a complete new stock-factor dataset or a new performance result.

The existing ETF research scope remains available, but its completed candidates
have failed the retained performance goals. A new direct-stock fundamental
backtest requires an authorized and coverage-verified source; no subscription
purchase, account creation or current-stock-list substitution is implicit.
The strategy objective remains unfulfilled and the weekly research schedule
does not guarantee that a qualifying strategy exists.

### Public macro information without a subscription

With no authorization to buy data or open accounts, the next fixed study
uses two public FRED series: T10Y3M (ten-year minus three-month Treasury
constant-maturity yield spread) and DFII10 (ten-year inflation-indexed real
Treasury yield). The source pages identify the Federal Reserve Bank of
St. Louis and Federal Reserve Board H.15, respectively. Standard Python
HTTP requests had transport errors; `curl` using the existing network
configuration obtained the official CSV files. No access-control or
proxy bypass was used.

The [source manifest](research/workbench/evidence/macro_rate_source_20261010.json)
retains both CSV fingerprints and their real missing-release counts.
`config/macro-factor-tilt.json` preregisters two configurations: a fixed
factor-share tilt when lagged real yields rise, and the same tilt with
equity reduction during a nonpositive nominal yield-curve spread.
Allocation is evaluated only at completed month ends; the original aggregate
equity/gold risk balance is preserved before the declared defense.

An observation is usable only after two complete NYSE sessions strictly
following its date. The last available observation can be carried across
release holidays only while its actual observation date is at most seven
calendar days old; each decision's observation/availability dates are recorded.
Negative rates are preserved, missing values are not invented, and future
observations cannot change earlier signals. These are current historical
downloads, not vintage ALFRED proof of unrevised past data.

```bash
.venv/bin/python -I -B -m us_quant.macro_factor_tilt prepare
.venv/bin/python -I -B -m us_quant.macro_factor_tilt register
.venv/bin/python -I -B -m us_quant.macro_factor_tilt evaluate
```

Both configurations still require real recurring-program registration and
independent market/target/accounting review. Macro information is not a
substitute for the missing individual-stock financial dataset and does not
itself establish a profitable or independently verified factor.

The [two actual macro reviews](research/workbench/evidence/macro_tilt_20261010_results.json)
remain rejected. The factor tilt has ten/five-year base Sharpe 0.9420/0.9850
and stress Sharpe 0.9107/0.9553. The version with curve defense has base
Sharpe 0.9589/0.9143 and stress 0.9179/0.8556. Neither meets the strict
Sharpe threshold, ten-year SPY outperformance and 15% drawdown requirements.
The published availability audit covers 2,805 decision sessions per series;
the largest actual observation age is five calendar days, below the
registered seven-day maximum.

The [eighth program snapshot](research/workbench/evidence/research_program_2026-W41_results_v8.json)
preserves 94 completed configurations and ten rejected recurring-program
candidate reviews. The official free macro-data route is now working,
but the complete individual-stock fundamental dataset remains unavailable.
There is no qualified research champion or live strategy change.
This unresolved research goal is not converted to success by rounding
0.9850 upward, shortening windows, or hiding the risk/benchmark failures.

### Source-bound strategy review

Three synthetic regression cases exposed missing review boundaries: the
reviewer previously accepted a self-consistent but unrelated target matrix,
an unsupported frozen-code placeholder, or an altered submitted risk-free
series. None of these fixtures represents an actual market strategy result.
The corrected reviewer rejects all three before advancing a research version.

`strategy_replay.py` now reconstructs targets from the exact registered,
supported implementation and its frozen policy. It preserves the actual
cost-dependent feedback for daily-risk strategies and reuses the causal
indicator builders for adaptive, option-implied and macro rules. The complete
target matrix, including its missing/non-decision rows, must match submitted
targets. Unknown strategy generators cannot qualify merely by supplying an
equity curve; new implementations require an explicit reviewed replay adapter
and causality tests.

Sharpe is recalculated using the risk-free series in the **registered market
snapshot**, not a submitted return-table column. The program engine fingerprint
also binds the replay adapter and shared accounting/metric dependencies.
An explicit event-anchored migration preserves earlier records; an unexplained
engine/dependency change still fails verification.

Existing reviews can be rechecked without replacing or recounting them:

```bash
.venv/bin/python -I -B -m us_quant.research_program audit-reviews \
  --export evidence/research_program_source_bound_audit_v1.json
```

This action finds the original hashed bundles, regenerates targets and both
accounting paths, compares metrics and gates with the existing review, and
asserts that the ledger is unchanged. Disagreement is an error, not permission
to rewrite old results or promote a previously rejected candidate.

The [actual source-bound re-audit](research/workbench/evidence/research_program_source_bound_audit_v1.json)
regenerated all 40 strategy paths for the ten existing recurring-program
reviews. Every stored metric and gate agreed; all ten candidates remain
rejected. The
[migration snapshot](research/workbench/evidence/research_program_2026-W41_review_v9.json)
preserves all 94 evaluated configurations, prior factor definitions,
blocked-stock-data cycle and candidate histories. This audit adds zero new
strategy trials and no investment success claim.

The native weekly instructions now require a supported source-bound generator
for new candidate implementations and explain that engine identity includes
the replay/accounting dependencies. An exact stored `metadata.engine_sha`
and event-chain head are required for reviewed migrations; a single-file
checksum is no longer the complete engine identity.

### Defensive asset selection and a generic evaluation runner

The next fixed study keeps the four equity factor funds and selects defense
from GLD, TLT and IEF using 126-session returns above actual BIL. One variant
holds the highest positive-excess defensive fund; the other inverse-volatility
weights all qualifying defensive funds. If none qualify, defense is BIL.
The equity/defense budget uses the same 63-session risk balance and 30%-70%
equity limits as before. This is two allocation hypotheses, not a new
independent economic factor. Duration and gold risks remain explicit.

`config/defensive-factor-rotation.json` freezes the rules. TLT is loaded from
the same previously audited base snapshot, not a changed historical vendor
vintage. Every non-cash equity sleeve keeps all four factor funds equally
weighted, and no leverage, new market account or paid data is introduced.

The recurring program now has a generic `evaluate-candidate` operation.
It generates targets through the supported frozen implementation, runs
the exact market/cost and independent `bt` paths, creates the hashed bundle,
and sends that bundle through the stronger reviewer. Already reviewed
candidates are returned without another evaluation or trial increment.
This avoids writing a new bespoke accounting runner for each future study.

```bash
.venv/bin/python -I -B -m us_quant.defensive_factor_rotation
.venv/bin/python -I -B -m us_quant.research_program register-candidate \
  --candidate data/defensive-factor-rotation-20261011/four_factor_defense_momentum-spec.json \
  --readiness data/defensive-factor-rotation-20261011/readiness.json
.venv/bin/python -I -B -m us_quant.research_program evaluate-candidate \
  --candidate-id four_factor_defense_momentum
```

The second candidate uses `four_factor_defense_diversified` and its matching
specification. Both must be registered before any outcome is computed.
Interrupted output remains evidence; a new explicit output directory is
required for replay, and original candidate code/data cannot be changed.

Both defensive variants completed the real generic evaluator and were
rejected. The [complete results](research/workbench/evidence/defensive_rotation_20261011_results.json)
show ten/five-year base Sharpe 0.6922/0.7815 for the defensive winner and
0.5320/0.4078 for diversified defense, with lower stressed results.
The [updated snapshot](research/workbench/evidence/research_program_2026-W41_results_v11.json)
retains 96 completed configurations and twelve rejected recurring reviews.
No qualifying research champion exists.

### Prospective data acquisition, not simulated trading

`config/prospective-data.json` defines a separate data-only archive. Its
collector acquires actual ETF/IRX data and official FRED/Cboe risk inputs
after the latest NYSE close plus the existing data buffer, and before the
next open. It retains original source bytes and hashes, performs the same
market/calendar/release-lag checks, and records actual acquisition time.
Older rows in a downloaded history are **not** relabeled as observations
made on their historical dates.

```bash
.venv/bin/python -I -B -m us_quant.prospective_data init
.venv/bin/python -I -B -m us_quant.prospective_data collect
.venv/bin/python -I -B -m us_quant.prospective_data status
```

The first successful snapshot is a baseline only. Repeated calls for the
same completed session skip acquisition and do not increment counts.
Missing observed sessions are listed explicitly and never backfilled;
late or incomplete fetches do not create completed receipts. Concurrent
collectors serialize through a file lock. Receipt-chain/head checks and
the latest snapshot's input hashes detect changed or partially committed
records. Old failed attempts and old independent forward ledgers remain
unchanged.

Collection is configured for approximately 09:00 Asia/Shanghai daily, while
new-factor discovery remains Saturday 09:00. This archive contains no account, orders,
positions or strategy-return calculation. Its snapshot count is not a
forward Sharpe, a 63-session trading qualification or an investment success.
The app/host must be available, and real future observations require actual
market time to pass.

The [first acquisition receipt](research/workbench/evidence/prospective_data_baseline_receipt_v1.json)
confirms 14 real public sources and 35 retained input files for the October 9,
2026 closing-session baseline. Repeating collection skips network acquisition
and preserves the receipt-chain head. It records **zero** post-baseline
observations and zero strategy returns. The next new NYSE market session is
October 12; no later observation has been fabricated.

The [updated native schedule](research/workbench/evidence/prospective_data_automation_v1.json)
was saved and read back as daily. Its next invocation observed at configuration
time is October 11, 2026, approximately 09:01 Asia/Shanghai; weekend duplicate
sessions are skipped. On non-Saturdays, the agent collects data and handles
already registered work without launching new factor searches. On Saturdays,
the bounded weekly discovery/review process continues.
Scheduled agents may publish explicitly checked metadata checkpoints; the
standalone Slack reporter still performs no git, trading or deployment.

### Static-allocation feasibility diagnostic

The [recorded diagnostic](research/workbench/evidence/static_allocation_feasibility_v1.json)
uses the existing nine unleveraged assets, including SPY and defensive funds,
with 2% idle cash. It deliberately uses the entire historical window and
zero transaction costs to inspect a **fixed-weight, daily-close,
daily-rebalanced arithmetic-return class**. Each horizon is optimized
separately. This is not a causal strategy, a deployable backtest, or an
independent validation result.

The convex normalized-mean problem and a numerical dual certificate agree:
the specified class has an optimistic ten-year Sharpe ceiling about 0.93375
and five-year ceiling about 0.99439. Mean/covariance inputs and floating-point
tolerances are recorded; optimal portfolio weights are not presented as
trade recommendations. Analytical synthetic cases verify the diagnostic.

```bash
.venv/bin/python -I -B -m us_quant.allocation_feasibility \
  --output reports/static-allocation-feasibility-replay.json
```

The result adds **zero** strategy trials and cannot be submitted as a
qualifying strategy through the source-bound reviewer. It does not bound
dynamic strategies, different assets, different initial-entry assumptions
or every cost-bearing execution model. Its purpose is to discourage
unproductive static-mix tuning, not to prove that the overall research goal
is impossible. Credible progress needs predictive information, a justified
and verified broader universe, or genuinely new observations.
The [ordinary-client SEC recheck](research/workbench/evidence/sec_normal_client_check_v1.json)
still returned 403 under the same identity/network settings; no archive
download or access-control bypass occurred.

### Matched-risk growth exposure comparison

`config/growth-factor-satellite.json` introduces two fixed matched-risk
configurations. The control retains the original four-factor equity sleeve;
the other uses half that sleeve in the unleveraged Nasdaq-100 ETF QQQ and
half equally across the four factor funds. QQQ is explicitly growth/market
beta, **not a newly discovered factor or assumed independent alpha**.

Both use identical 63-session equity/gold risk balance, daily 12% predicted
volatility projection, a 5-percentage-point target-change band, 2% idle cash,
and the existing base/stress execution costs. The experiment does not raise
the risk target, remove the factor core or silently substitute leveraged
products. QQQ comes from the previously audited base-data manifest.
Actual drawdown, return and Sharpe gates remain required.

```bash
.venv/bin/python -I -B -m us_quant.growth_factor_satellite
# Register both matching spec files with research_program register-candidate,
# then use evaluate-candidate; no outcomes are computed by preparation.
```

The source-bound generator reproduces the actual composed sleeve risk, not
the volatility of the factor-only control. Any gains must be attributed
honestly to additional growth exposure rather than presented as factor
discovery. The prior static-mix diagnostic did not include QQQ and is not
a bound on this enlarged asset scope.

The [actual matched-risk results](research/workbench/evidence/growth_factor_satellite_20261011_results.json)
were both rejected. Growth exposure raises ten/five-year base Sharpe from
0.8347/0.8878 to 0.8865/0.9131, but drawdowns increase slightly to about
18.5%-18.7%; stressed Sharpe remains 0.8298/0.8739. This is a modest growth-beta
effect, not evidence that a new factor or qualifying strategy was found.
The [current program snapshot](research/workbench/evidence/research_program_2026-W41_results_v13.json)
retains 98 actual evaluations and fourteen rejected recurring reviews.

The enlarged-scope feasibility diagnostic initially failed its five-year
numerical certificate. A mathematically equivalent unit-volatility coordinate
transformation fixes numerical conditioning without changing the objective,
constraints or tolerances. A new disparate-volatility regression case passes,
and the [original-scope v2 result](research/workbench/evidence/static_allocation_feasibility_v2.json)
agrees with v1 within numerical tolerance; v1 remains preserved with its own
implementation hash.
The [QQQ-inclusive diagnostic](research/workbench/evidence/growth_scope_feasibility_v1.json)
gives zero-cost, full-sample fixed-weight numerical bounds of about 1.00333
for ten years and 0.99622 for five years. It remains a hindsight diagnostic,
not a strategy, not a joint shared-weight result, and not a bound on all
dynamic or cost-bearing execution. It adds zero strategy trials and cannot
qualify as investment success.

### Causally trained conditional forecasts

`config/conditional-factor-model.json` freezes a price-only pooled ridge model
and the same model augmented with the previously verified macro/option risk
inputs. Both predict each factor fund's and gold's next monthly open-to-open
return relative to BIL. A training label is eligible only after its exit open
has occurred; neither the current holding period nor a future label can enter
the training set. Training-window selection and the fixed ridge pipeline
reuse the imported workbench helpers.

The model uses at most 36 distinct months, training-only standardization,
and current-feature clipping to observed training ranges. The earliest fit
has only nine matured months due actual fund-history availability; five
cross-sectional observations per month are not five independent months.
The weak initial sample is disclosed rather than replaced with synthetic
pre-inception histories.

Forecasts rank the four factor funds into fixed 35/30/20/15% equity shares.
Equity and gold allocations require a fixed positive-excess forecast hurdle;
risk balance and BIL handle the declared active/inactive cases. Model
coefficients and allocations update chronologically, but this is not
discovery of new independent economic factors or a live strategy update.
Price-only predictions must remain unchanged when only macro/option data
are perturbed, and future prices/labels cannot change earlier predictions.

```bash
.venv/bin/python -I -B -m us_quant.conditional_factor_model
# Register both specs with the recurring program, then evaluate-candidate.
```

The source-bound reviewer regenerates the whole forecasting sequence before
checking actual costs and performance. No hyperparameter search, random split
or weakening of the original Sharpe/SPY/drawdown gates is part of this study.

The first conditional-model code version failed **before complete account
evaluation** because the residual BIL weight was negative by only
1.1102230246251565e-16. The
[original numeric diagnostic](research/workbench/evidence/conditional_model_numeric_diagnostic_v1.json)
is preserved. Revision 2 clamps only a cash-budget residue within 1e-12 of
zero and still rejects a material overspend. The
[complete-matrix correction check](research/workbench/evidence/conditional_model_numeric_fix_v2.json)
shows that training audits and every non-BIL weight are unchanged; the largest
target difference is the original roundoff magnitude. No economic parameter
or strategy outcome was used to select this correction.

The program supports an explicit `record-numeric-failure` lifecycle event
for this narrow, evidenced pre-accounting failure. It cannot erase a completed
review, overwrite an earlier failure or relabel a materially negative weight
as roundoff. Technically rejected code versions are retained separately from
completed strategy evaluations; they do not create a false return result.
Corrected specifications use new IDs and a `supersedes_candidate` link.
`pending_candidate_ids` excludes terminal technical failures, so future runs
do not repeatedly retry an immutable invalid version.

The [corrected model results](research/workbench/evidence/conditional_factor_model_20261011_results_v2.json)
completed actual source-bound review and were both rejected. Price-only
base Sharpe is 0.5460/0.6239 for ten/five years; macro/option augmentation
gives 0.5462/0.7765. Stressed Sharpe is lower and drawdown remains above 15%.
The added information improves the five-year comparison but does not satisfy
the target.
The [training audit](research/workbench/evidence/conditional_model_training_audit_v2.json)
records 121 fits per model, first fitted September 30, 2016 with nine
distinct matured months. All label-availability dates precede decisions,
and regenerated target matrices match the evaluated account inputs.

The [conditional-round program snapshot](research/workbench/evidence/research_program_2026-W41_results_v16.json)
had 100 complete strategy evaluations and 16 completed recurring reviews,
all rejected under the current full gates. Two separate, invalid first code
versions remain as technical failures, not hidden or falsely counted as
complete performance evaluations. There is still no qualifying research
champion or live strategy update.
The [conditional-round read-only source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v2.json)
regenerated all 64 paths for those 16 completed reviews and confirmed every
recorded metric/gate without changing the ledger. The two technical failures
remain outside the completed-performance count.

### Matched-budget total versus residual factor momentum

`config/factor-momentum-comparison.json` freezes two bounded allocation
hypotheses after the 100 completed configurations: total-return momentum
versus SPY-regression residual momentum in the same four actual factor ETFs.
Both inherit identical equity/gold totals from the unchanged monthly
factor/gold baseline, and rank funds into fixed 35/30/20/15% equity-sleeve
shares. All four economic exposures remain held; beta removal in the score
does not create a beta hedge in the long-only portfolio.

The unchanged original `factor_scores` helper fits trailing 252-session
daily excess returns and scores the last 126 sessions excluding the latest
21. This shorter ETF adaptation fits the available pre-account history
without moving either formal return window. It is not a stock-level
multifactor or academic long-short replication, and does not add an
independent factor definition. Source metadata verifies Ehsani and
Linnainmaa's journal article as **2022, DOI `10.1111/jofi.13131`**, distinct
from their 2019 [NBER working paper](https://www.nber.org/papers/w25551).
The residual-momentum motivation is the
[2011 Blitz/Huij/Martens article](https://repub.eur.nl/pub/22252/).

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_momentum_comparison
```

Preparation revalidates existing provider snapshots and reuses their hashed
panels; it computes no strategy returns. Both
[specifications were registered before outcomes](research/workbench/evidence/factor_momentum_comparison_20261011_registration.json)
after an exact-anchor engine migration preserved all 16 old reviews and both
technical failures. At registration both were pending and the completed
count was still 100, not 102. The previously exposed history is not an
independent holdout.

The [actual source-bound results](research/workbench/evidence/factor_momentum_comparison_20261011_results.json)
now reject both configurations:

| Factor ranking | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Total momentum | 0.8993 / 0.8590 | 0.9628 / 0.9101 | 20.52% / 18.04% |
| SPY-residual momentum | 0.9037 / 0.8567 | 0.9526 / 0.8907 | 20.52% / 17.69% |

Both beat SPY's five-year net CAGR, but fail the ten-year benchmark and all
Sharpe/drawdown paths. The
[paired attribution](research/workbench/evidence/factor_momentum_comparison_20261011_attribution_v2.json)
verifies identical target issue dates and equity/gold budgets within
numerical tolerance. Residual scoring raises dollar turnover despite the
same 605/305 ten/five-year order tickets. For five-year stress its traded
volume is about USD143,656 versus USD116,561 and costs USD592.31 versus
USD538.12, with lower Sharpe. Dollar turnover is independently reconstructed
from opening holdings/cash and checked against transaction charges; it is
not the dimensionless sum of the ledger's turnover ratios.

The [factor-momentum snapshot](research/workbench/evidence/research_program_2026-W41_results_v18.json)
retains **102 complete configurations, 18 recurring reviews, two separate
technical failures, no pending candidates and no qualified champion**.
The [read-only re-audit](research/workbench/evidence/research_program_source_bound_audit_v3.json)
regenerated every one of the 72 completed strategy paths without changing
history or conclusions. Do not tune these rejected lookbacks or shares
merely to cross one on the exposed sample. Daily collection again skipped
the existing October 9 baseline; there are still zero new prospective
observations and no strategy-return proof.

### Joint versus component trend risk gates

`config/trend-factor-guard.json` freezes two risk-management hypotheses
after 102 completed configurations, without changing factor rankings or
adding an economic factor definition. Both start from the unchanged
four-factor/gold monthly risk targets. At every completed close, a strict
200-session moving-average gate can disable risk; the removed weight goes
to actual BIL, never into a larger surviving risk sleeve.

The joint variant observes an **untraded signal index** compounded from
daily returns and the baseline target known at the preceding close. The
component variant separately gates an equal-weight four-factor return index
and GLD. These indices are not cost-free account returns or investment
evidence. Missing pre-target history is not filled with artificial cash
observations. Only complete 200-observation windows can issue decisions.
Both methods request orders on gate changes and month-end target updates;
unchanged daily states do not rebalance.

This daily adaptation is motivated by Faber's 2007 article,
DOI [`10.3905/jwm.2007.674809`](https://doi.org/10.3905/jwm.2007.674809),
not a replication of its monthly ten-month timing rule. Trend risk gates
can whipsaw, gap through a threshold and incur extra costs. They do not
guarantee a maximum drawdown or Sharpe above one.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.trend_factor_guard
```

Preparation revalidates and reuses the old hashed actual-fund snapshots;
it computes no strategy outcomes. Both
[fixed implementations are registered before outcomes](research/workbench/evidence/trend_factor_guard_20261011_registration.json).
At registration the completed count remains 102, with two pending
candidates. Exact-anchor migration preserved all 18 previous reviews and
both technical failures. The original dual windows, execution delays,
capital, costs and Sharpe/SPY/drawdown gates remain unchanged.

The joint guard completed its review, but the component guard's first
attempt stopped **before a completed review** on an independent accounting
disagreement. The
[preserved diagnostic](research/workbench/evidence/trend_guard_accounting_incident_v1.json)
reproduces a legacy solver corner: a rounding-only requested change of
about USD1.82e-12 enters the fixed-commission branch and manufactures a
USD1.00196 charge. The independent engine correctly performs no trade.
Do not count the interrupted attempt as a completed strategy evaluation.

`cash_funded_accounting_v2.py` keeps holdings and cash unchanged when every
pre-fee delta is below the **existing USD1e-6 order-notional threshold**.
Real trades reuse the original cash-budget solver, fees and execution
assumptions. This is a versioned accounting correction, not a different
portfolio rule or wider audit tolerance. The imported `backtest.py` and
all old frozen sources remain unchanged.

New reviews require an explicit, exact-source-hash
`cash_funded_noop_v2` bundle reference. Read-only audits of the previously
completed legacy bundles continue using their original engine; they are
not silently rescored. The original failed component output is retained,
and recovery must write to a fresh replay directory. At this incident
checkpoint there are **103 complete configurations, 19 completed reviews,
one pending candidate, and two separately indexed invalid strategy-code
versions**. The accounting incident is neither a third invalid strategy
definition nor an additional performance trial.
The [accounting-version registration](research/workbench/evidence/cash_funded_accounting_v2_registration.json)
pins the corrective source hash before a new replay; exact migration
preserved the original policy, candidate definitions, all 19 reviews and
both indexed strategy-code failures.

The [corrected component replay](research/workbench/evidence/trend_guard_accounting_recovery_v2.json)
now agrees with independent `bt` within USD1.06e-10 on all four account
paths. Every target CSV hash is identical to the interrupted attempt;
parameters, data, costs, delay and audit tolerance did not change. The
original failed bundle is retained. The completed
[two-candidate results](research/workbench/evidence/trend_factor_guard_20261011_results.json)
reject both risk gates:

| Risk gate | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Joint portfolio | 0.8878 / 0.7518 | 0.9076 / 0.7933 | 13.18% / 11.76% |
| Separate equity/gold | 0.6813 / 0.4131 | 0.7175 / 0.4168 | 15.78% / 14.69% |

The joint gate's base drawdown improves, but stress drawdowns are
15.26%/15.63% and neither horizon beats SPY's net CAGR. Component stress
drawdowns reach 21.61%/21.11%.
The [existing-control attribution](research/workbench/evidence/trend_factor_guard_20261011_attribution.json)
shows lower net CAGR and Sharpe versus the ungated core in every comparison.
Daily gate changes increase order tickets and charged turnover: component
five-year stress pays USD2,153.83 across 615 tickets. Risk reduction and
whipsaw costs are not new factor alpha or a guaranteed risk ceiling.

The [trend-guard ledger snapshot](research/workbench/evidence/research_program_2026-W41_results_v22.json)
retains **104 complete configurations, 20 completed reviews, two terminal
strategy-code failures, one separately preserved/resolved accounting
incident, no pending candidates and no qualified research champion**.
The [80-path mixed-version audit](research/workbench/evidence/research_program_source_bound_audit_v4.json)
preserves every old conclusion: 19 reviews use the original engine and the
corrected component review uses its frozen version-two reference. Do not
retune these windows or gate thresholds merely to cross one on this sample.

The [updated automation receipt](research/workbench/evidence/prospective_data_automation_v2.json)
keeps daily data acquisition and Saturday factor exploration unchanged,
while requiring explicit new-accounting references and preserved failure
evidence. No future scheduled execution is claimed. Duplicate collection
again leaves one October 9 baseline and zero post-baseline observations.

### Alternative implementation within the momentum family

`config/momentum-implementation.json` freezes two alternatives after 104
completed configurations: replace MTUM's original momentum-family budget
with PDP, or use equal MTUM/PDP shares within that same budget. Value,
quality, low-volatility and aggregate equity/gold target budgets stay
identical to the already evaluated original factor/gold control.
The second momentum fund is not a fifth independent economic factor.

PDP's actually retrieved official
[product page](https://www.invesco.com/us/en/financial-products/etfs/invesco-dorsey-wright-momentum-etf.html)
identifies US passive equity, ISIN US46137V8375, March 1, 2007 inception,
and the Dorsey Wright Technical Leaders mandate. The directly retrieved
[Nasdaq methodology](https://indexes.nasdaqomx.com/docs/Methodology_DorseyWrightTechnicalIndexes.pdf)
and [index page](https://indexes.nasdaq.com/Index/Overview/DWTL) confirm DWTL's
proprietary relative-strength selection. These are actual current primary
sources, not proof that all historical methodology versions were identical.
The issuer factsheet URL returned audience-selection HTML, not a PDF;
neither that gate nor Stooq's browser verification was bypassed.

The new actual PDP history has 2,805 complete sessions over the unchanged
factor-data interval, positive volume throughout, and explained adjustment
changes. The adapter reparses raw OHLCV and corporate actions, compares
the retained CSV and binds the actual document hashes. It never fills
SPMO's previously rejected zero-volume records or shortens either formal
return window. The
[source audit](research/workbench/evidence/momentum_implementation_source_audit_v1.json)
retains SPMO's 238 zero-volume rejection, Stooq's HTML verification response
and the issuer's non-PDF factsheet response. Current primary documents
support PDP's mandate without claiming a complete historical method archive.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.momentum_implementation
```

Preparation creates hashed panels but computes no strategy outcomes.
Both [fixed configurations were registered before outcomes](research/workbench/evidence/momentum_implementation_20261011_registration.json),
after exact provider-engine migration preserved all 104 completed
configurations and 20 reviews. At registration two candidates remain
pending; no new independent factor family or completed evaluation is
claimed. Use the generic versioned-accounting evaluator and retain all
costs, delay and full gates. Actual fund returns are not a recreated
proprietary stock-score strategy or independent forward evidence.

The [completed implementation results](research/workbench/evidence/momentum_implementation_20261011_results.json)
reject both alternatives:

| Momentum-family implementation | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| PDP only | 0.8594 / 0.8295 | 0.9089 / 0.8634 | 20.25% / 18.14% |
| Equal MTUM/PDP | 0.8764 / 0.8473 | 0.9277 / 0.8846 | 20.23% / 18.23% |

Both beat SPY's five-year CAGR but not its ten-year CAGR; every path misses
the net Sharpe and retained drawdown gates. The
[matched-budget attribution](research/workbench/evidence/momentum_implementation_20261011_attribution.json)
confirms identical target issue dates, momentum-family totals, other factor
budgets and equity/gold totals. Net Sharpe and CAGR are lower than the
already evaluated MTUM control in all eight comparisons.
PDP/MTUM daily return correlations are 0.931/0.907 for ten/five years;
the different implementation is not an independent return source.
Adding the second momentum fund raises five-year tickets from 305 to 366,
and five-year stress costs from USD434.31 for full PDP to USD496.29 for
the implementation mix. No mix optimization is justified by these outcomes.

The [momentum-round program state](research/workbench/evidence/research_program_2026-W41_results_v24.json)
retains **106 complete configurations, 22 reviews, two terminal invalid
strategy-code versions, the separately resolved accounting incident, no
pending candidates and no qualified research champion**.
The [88-path read-only audit](research/workbench/evidence/research_program_source_bound_audit_v5.json)
preserves all outcomes, including 19 legacy-engine reviews and three
explicit version-two reviews. The newly verified history does not create
future observations; daily collection still has one baseline and zero
post-baseline observations.

### Joint selection diagnostics on completed recurring research

`config/selection-validation.json` freezes a read-only diagnostic for the
22 completed recurring reviews. It does **not** pretend to cover the whole
106-configuration search: the earlier 84 trials remain outside joint
inference. Nor does it change qualification, create a strategy or provide
independent forward observations.

The auditor verifies each original hashed bundle and actual strategy/SPY
account path, including initial fees, cost scenarios and execution delays.
Within each horizon it simultaneously resamples all 66 comparisons:
base versus base SPY, stress versus stress SPY, and stress versus base SPY.
Identical sampled 21-session circular blocks preserve contemporaneous
dependence across candidates and cost comparisons. A least-favourable
mean-centered maximum statistic accounts for searching within this cohort,
rather than reporting a separately bootstrapped favourite.

The five-year window overlaps the ten-year window. Global 5% diagnostic
alpha is split across horizons by Bonferroni, not by assuming independence.
One-sided simultaneous bounds concern annualized **log-growth advantage**,
not CAGR percentage-point differences or future success probabilities.
The method adapts White's
[2000 data-snooping reality check](https://doi.org/10.1111/1468-0262.00152);
the finite-block bootstrap remains approximate and conditional on exposed
history, with stationarity and uncounted-selection limitations.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.selection_validation register
# Commit the fixed code/settings and registration before computing joint outcomes.
.venv/bin/python -I -B -m us_quant.selection_validation run
```

The audit fails on changed source hashes, a changed cohort/event head or
different settings. Both live journal and original review outcomes remain
unchanged. Even favourable scoped evidence cannot override Sharpe, SPY,
drawdown or absent independent-validation requirements.
The [fixed audit registration](research/workbench/evidence/selection_validation_20261011_registration.json)
pins code, settings, the 22-review cohort and source hashes before joint
resampling. At registration no statistical outcome, new strategy trial or
research-version update is claimed.

The [actual joint results](research/workbench/evidence/selection_validation_20261011_results.json)
find **no positive simultaneous lower outperformance bound**:

| Scope | Candidate/cost comparisons | Scoped omnibus p-value | Positive simultaneous lower bounds |
|---|---|---|---|
| Ten years | 66 | 1.0000 | 0 |
| Five years | 66 | 0.5881 | 0 |

The two-horizon Bonferroni omnibus p-value is 1.0. Some five-year curves
have positive historical net growth differences, but this least-favourable
joint diagnostic does not establish statistically significant advantage
within the declared cohort. It is conservative and conditional, **not a
probability of future failure or proof that profitable trading is
impossible**. The excluded 84 earlier trials and uncounted research choices
remain limitations; there is still no full-search selection-adjusted alpha
or independent future evidence.

The [reproducibility check](research/workbench/evidence/selection_validation_20261011_reproducibility.json)
replays all statistics exactly and independently reconstructs all 4,000
samples per horizon through direct indexing rather than the implemented
matrix-count shortcut. Omnibus p-values agree exactly; critical values
agree within 1e-15. Source hashes, journal events, 106 completed
configurations, 22 reviews and all prior qualification outcomes remain
unchanged. This audit adds **zero strategy evaluations**.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v3.json)
preserve daily collection and weekly discovery, explain the limited
selection scope, and require a newly registered statistical plan if the
cohort changes. The frozen 22-candidate result cannot be silently applied
to future candidates. No future scheduled execution is claimed.

### Credit-risk information and fixed recovery rules

`config/credit-factor-guard.json` freezes two risk-input experiments after
106 completed configurations. They use the previously evaluated static
70% equal-four-factor / 30% gold portfolio, not a newly optimized mix.
At completed month ends, the level guard enables equity only while the
lagged Baa/Treasury spread is below its trailing 252-session median.
The second rule also permits equity when that spread has declined over
21 sessions, a predefined early-recovery condition. Risk-off moves the
original equity budget to actual BIL; gold is never increased.

The initially considered
[ICE high-yield OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
now has only three years on FRED, so it cannot support the formal windows.
The study explicitly uses a **different credit proxy**:
[BAA10Y](https://fred.stlouisfed.org/series/BAA10Y), long-maturity seasoned
investment-grade Baa yield minus ten-year Treasury yield. It includes
duration/liquidity/quality effects and is not a high-yield option-adjusted
spread. The Treasury input provider changed in June 2019; this limitation
is retained, not erased. Proprietary input observations and derived
credit-value series stay private and are not published.

The actual public CSV covers the requested 2014-2026 range. The unchanged
macro availability helper delays every observation until the second
strictly later NYSE session and caps its age at seven calendar days.
The [source and timing audit](research/workbench/evidence/credit_factor_guard_source_20261011.json)
records 3,186 actual observations, 2,805 ETF decision sessions, 22 bounded
carry sessions and a maximum observation age of five days. The source
restriction and different proxy definition are preserved before outcomes.
Current historical extraction is not an ALFRED vintage or exact original
release-time archive. Credit is a risk input, not another independent
return factor, and the frozen earlier 22-candidate joint diagnostic does
not cover these new candidates.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.credit_factor_guard
```

Preparation revalidates the original ETF panels and actual credit timing
without computing strategy outcomes. Both
[fixed rules are registered before outcomes](research/workbench/evidence/credit_factor_guard_20261011_registration.json).
Exact-anchor engine migration preserved all 106 prior configurations,
22 completed reviews and the two terminal invalid code versions. The
two credit candidates are pending, not completed evaluations. Original
costs, execution delays, fresh
capital and Sharpe/SPY/drawdown goals stay unchanged. The existing data-only
prospective archive does not yet contain BAA10Y; no new independent
credit-strategy observations or future results are claimed.

The [completed credit results](research/workbench/evidence/credit_factor_guard_20261011_results.json)
reject both fixed hypotheses:

| Credit rule | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Below trailing median only | 0.8125 / 0.7649 | 0.6711 / 0.4926 | 15.65% / 15.80% |
| Also recover on compression | 0.8756 / 0.8472 | 0.6793 / 0.5100 | 16.67% / 16.90% |

Neither meets any full path's Sharpe/SPY/drawdown contract. The
[control attribution](research/workbench/evidence/credit_factor_guard_20261011_attribution.json)
verifies identical issue dates, unchanged gold, original family budgets
when equity is enabled, and actual BIL holding for every removed equity
budget. The strict guard reduces some drawdown but misses return.
Recovery improves ten-year Sharpe versus the original static control, but
still reduces CAGR by about 2.58 percentage points; its five-year Sharpe
and CAGR are lower. Both variants miss the index return goal.
This does not establish that every credit indicator is useless, and does
not justify tuning the rejected recovery/median windows on exposed history.

The [credit-round program snapshot](research/workbench/evidence/research_program_2026-W41_results_v26.json)
retains **108 complete configurations, 24 completed reviews, two terminal
invalid code versions, the separately resolved accounting incident, no
pending candidates and no qualified champion**.
The [96-path source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v6.json)
preserves all original outcomes, including 19 legacy and five explicit
version-two reviews. The earlier joint-selection audit still applies only
to its frozen 22-candidate cohort, not the two new credit candidates.
The original prospective archive stays unchanged, with zero new
observations and no Baa/credit-strategy forward-performance proof.

### Audited additional ETF families and a future-cycle queue

`config/factor-family-expansion.json` prepares two **new-to-this-catalog**
economic families: small-company exposure through actual IJR and net-share
reduction exposure through actual PKW. This is not a claim to discover
globally new anomalies or independently profitable factor returns.
The issuer confirms PKW's trailing-twelve-month net-share reduction
threshold of at least 5%; IJR tracks the S&P SmallCap 600, not academic
zero-cost SMB. IJR can use futures to offset cash for tracking, so it is
not described as derivatives-free.

Both actual fund panels cover the unchanged 2,805-session interval.
The [source preparation audit](research/workbench/evidence/factor_family_source_expansion_20261011.json)
preserves the actual issuer snapshots, acquisition times, positive volumes
and company-action checks for both funds.
The S&P methodology request returned 403; it is not bypassed or reported
as verified. Current issuer definitions do not prove historical screen
continuity. Raw current holdings, back-tested index returns and recreated
stock-level scores are not used as fund-price substitutes.

The factor-ETF registration path now accepts the complete definitions
bound by an explicitly audited source adapter, rather than always forcing
the same four fixed labels. Catalog bodies must match the audited
definitions, not just similarly named families. Legacy adapters retain
the original four definitions; unknown adapters and mislabeled direct-stock
signals still fail closed. Minimum three economic families, leverage,
capital, costs and performance objectives are unchanged.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_family_sources
```

Preparation creates source-verified panels, not strategy results or new
candidate registrations. The two definitions are queued with an explicit
**not-before October 17, 2026 at 09:00 Asia/Shanghai** constraint. This week's
two-definition allowance has already been used. `cycle` rejects a premature
queue rather than inventing a future timestamp or silently bypassing the
weekly limit. Existing factor counts and all 108 completed configurations
remain unchanged until a real eligible cycle occurs.

The original prospective archive does not include IJR or PKW. The separate
extension below adds only a new data baseline; no independent new-family
strategy performance is claimed.
The [real-clock queue check](research/workbench/evidence/factor_family_queue_receipt_v1.json)
actually rejected premature discovery and verified that the live catalog,
cycles, candidates, results and event head did not change after rejection.
The [history-preserving migration snapshot](research/workbench/evidence/research_program_2026-W41_families_v27.json)
still has **108 completed configurations, 24 reviews, six registered
definitions across five families and no qualified champion**. Two queued
definitions are not counted as registered discoveries.

The [latest 96-path source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v7.json)
reproduces all existing outcomes after the adapter change without modifying
the journal. The [native queue instructions](research/workbench/evidence/prospective_data_automation_v4.json)
refresh readiness only at a real eligible cycle and register definitions
before any candidate relying on them. They do not invent a future execution
or use a synthetic clock. Minimum economic-family, weekly-definition,
cost, risk and return criteria remain unchanged.

### Separate expanded prospective input archive

`config/prospective-factor-inputs.json` defines a new data-only archive for
IJR, PKW and BAA10Y in addition to the existing 14 public sources.
`prospective_factor_inputs.py` composes the unchanged parent archive's
receipt, lock, gap and deadline logic. It never changes the original
collector, parent registration, receipt chain or paused strategy ledgers.

The new collector requires a verified parent snapshot for the **same
completed session**. It copies the exact parent input bytes and retains
their actual acquisition times, then requests only the two extra fund
histories and one extra credit series. Expanded records have their own
collector hash and receipt chain, tied to the parent's registration and
same-session receipt hash. A missing parent, changed source, failed fetch
or next-open deadline miss cannot produce a completed expanded snapshot.

The first expanded snapshot is another **baseline**, not a second
independent market observation. Snapshot counts never imply a strategy
Sharpe, P&L or trading qualification. Capturing IJR/PKW data does not
register the queued factor definitions early; their weekly eligibility
still applies. The proprietary Baa observations and aligned values remain
private. All price/macro rows are current-vendor context, not proof they
were first available when those historical rows occurred.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs init
.venv/bin/python -I -B -m us_quant.prospective_data collect
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs collect
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs status
```

Initialize only the new explicitly named expanded archive, never the
existing parent. Daily execution collects the parent first and expanded
inputs second; same-session duplicates make no network requests.
Both archives preserve missed sessions without retroactive backfill.
The [extension registration](research/workbench/evidence/prospective_factor_inputs_v2_registration.json)
pins the new collector and policy before acquisition. It initializes only
`data/prospective-factor-inputs-v2`; the original parent's receipt head,
registration and source inputs remain unchanged. At registration the new
archive has zero snapshots and no independent strategy performance.

The [actual expanded baseline receipt](research/workbench/evidence/prospective_factor_inputs_v2_baseline_receipt.json)
records the October 9 session acquired at
**2026-10-10T21:22:03.497540Z**, before the next NYSE open.
The [capture audit](research/workbench/evidence/prospective_factor_inputs_v2_capture_audit.json)
verifies all 35 inherited input files and their acquisition timestamps,
unchanged parent receipt head, and the three actually acquired new sources.
The expanded archive has 17 source streams and 502 sessions of price context.
Those historical rows, and the two same-day parent/expanded snapshots, are
not extra independent market observations.

The [expanded status](research/workbench/evidence/prospective_factor_inputs_v2_initial_status.json)
has one baseline and **zero observations after baseline**, just like the
unchanged original archive. A duplicate call is verified to skip fetching
and leave both receipt hashes unchanged. No strategy account, P&L,
candidate evaluation or factor registration was added; all 108 research
configurations and the queued October 17 definitions remain unchanged.

The [latest native instructions](research/workbench/evidence/prospective_data_automation_v5.json)
collect the parent first and extension second at the existing daily cadence,
without reinitializing either archive. Future execution has not been
observed or claimed. The original collector fingerprint, old forward
ledgers and weekly research eligibility remain untouched.

### Frozen six-family candidate implementations for the eligible cycle

`config/six-factor-strategy.json` defines two complete candidate rules for
momentum, value, quality, low-volatility, size and net-share-reduction
exposures through the six actually verified ETFs. One uses equal equity
family shares; the other reuses the unchanged `bounded_factor_shares`
helper on trailing 63-session volatility, bounding every family between
half and one-and-a-half times the equal share.

Both inherit the original four-factor/gold baseline's identical monthly
equity/gold totals, cash reserve and issue dates. The comparison changes
only family composition, not benchmark costs or aggregate defensive
exposure. All six equity sleeves remain positive. Inverse marginal
volatility is **not full-covariance equal-risk contribution**, and six
economic characteristics are not six orthogonal alpha sources.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.six_factor_strategy prepare
# After the actual eligible weekly factor registration:
.venv/bin/python -I -B -m us_quant.six_factor_strategy register \
  --prepared data/six-factor-prepared-20261017 \
  --receipt evidence/six_factor_strategy_20261017_registration.json
```

Preparation creates source-bound specifications, not performance results.
Registration is explicitly blocked until October 17 at 09:00 Asia/Shanghai
and requires the new definitions already present in the real journal.
The source-bound replay registry supports both implementations; they
cannot qualify using arbitrary submitted curves. Costs, fresh capital,
execution delay, drawdown and Sharpe/SPY goals remain unchanged.
No real candidate outcome is computed merely because the code is prepared;
the earlier 108 trials and 22-candidate statistical diagnostic are not
rewritten or silently extended.

The [complete prepared specifications](research/workbench/evidence/six_factor_strategy_preparation_v1.json)
freeze both implementations and all source dependencies. On the real
2,805-session / ten-asset panel, each has 131 complete monthly targets.
The budget audit verifies positive six-family shares, fixed share limits,
98% target investment and unchanged aggregate equity/gold targets.
**No actual strategy-account performance was computed.**

The [actual eligibility checks](research/workbench/evidence/six_factor_strategy_eligibility_v1.json)
refused both source-backed candidates because their new definitions are not
yet registered; the independent real-clock gate also refused early execution.
These checks add no candidates or strategy trials. The
[history-preserving engine snapshot](research/workbench/evidence/research_program_2026-W41_six_factor_v28.json)
still has 108 completed configurations, 24 reviews and six registered
definitions. The [latest 96-path audit](research/workbench/evidence/research_program_source_bound_audit_v8.json)
reproduces every old metric and gate after adding the new generator.

The [native execution instructions](research/workbench/evidence/prospective_data_automation_v6.json)
prepare fresh evidence at the real eligible cycle, register definitions
before candidates, publish the actual candidate receipt, and only then
evaluate performance. They do not fabricate a registration receipt or
pretend prepared targets establish Sharpe above one. Both data baselines
and the earlier performance/selection limitations remain unchanged.

### Prospective target provenance, not execution or performance

`config/prospective-target-observations.json` freezes an independent
observation profile for the already registered `macro_real_yield_factor_tilt`
rule. This candidate **failed the historical qualification gates** and
remains rejected; recording its future targets does not promote it or claim
a profitable strategy.

The observer reuses the exact original rule and parameters with an actually
captured same-session parent ETF/macro snapshot. It records the real
generation time, original source receipt and most recent completed rule
month. The initial target is generated from the latest known completed
month, not backdated to that month's close. New input days cannot silently
retune an old monthly target; it is retained until a newly observed completed
month exists.

Hypothetical base/stress execution dates are later than the actual observed
market session. They are **not fills or submitted orders**. The journal
never creates positions, computes P&L, starts a portfolio or changes the
research champion. Data gaps stay explicit and cannot become manufactured
earlier signals or proof of continuous forward performance.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_target_observations init
.venv/bin/python -I -B -m us_quant.prospective_target_observations collect
.venv/bin/python -I -B -m us_quant.prospective_target_observations status
```

Initialize only this new named observation journal, never the existing
data archives or old paused strategy ledgers. Source code/parameters and
the original rejected status must verify. Capture is restricted to actual
known inputs before the next open; same-session duplicates do not regenerate
targets. The existing 108 historical configurations and queued six-family
definitions/candidates remain unchanged.
The [observer registration](research/workbench/evidence/prospective_target_observations_v1_registration.json)
pins the rule and recording policy before the first generated target.
The source candidate remains rejected, and registration itself creates
zero target observations, orders, portfolios or strategy-performance paths.

The [first actual intent record](research/workbench/evidence/prospective_target_observations_v1_first_target.json)
was generated at **2026-10-10T21:59:17.810517Z**, using the October 9 input
snapshot actually captured at 16:44:09.524784Z. The latest completed rule
month is September 30; the output is **not backdated to September 30**.
Hypothetical next-open/extra-delay dates October 12/13 are unfilled research
intent, not a claim to have traded or earned returns.

The [first receipt](research/workbench/evidence/prospective_target_observations_v1_first_receipt.json)
binds the target to the original source snapshot and observation-policy
registration. The [status](research/workbench/evidence/prospective_target_observations_v1_initial_status.json)
has one target observation and zero portfolios, orders, returns or research
champion updates. Same-session reruns are verified not to regenerate or
change the receipt. Full target weights are retained privately rather than
published as an actionable allocation. The source strategy remains
historically rejected; the 108 completed configurations are unchanged.

The [updated daily instructions](research/workbench/evidence/prospective_data_automation_v7.json)
collect parent inputs, then expanded inputs, then target observations.
They preserve the old monthly target until a new month is actually
available and never restart a paused forward account. Actual future
automated execution, strategy fills or Sharpe qualification is not claimed.

### Separately registered prospective research model accounts

`config/prospective-research-accounts.json` defines a new research-only
simulation for the already recorded, historically rejected macro-factor
candidate. It is not a live account, research champion or restart of any
paused shadow ledger. Four fresh USD10,000 model accounts compare the
strategy and SPY under original 5/20bp costs, USD1 ticket charges and
one/two-session delays. Registration must precede the first model open.

Each modelled session requires its own actually captured parent data receipt
and target-observation receipt. Targets must have been recorded before the
applicable open. A later download's historical rows cannot fill a missing
observation; missing inputs explicitly stop the model. Late processing of
already captured, on-time immutable inputs is not relabelled as earlier
processing or actual execution.

Adjusted total-return price levels may be renormalized in successive vendor
downloads. The model chains actual within-snapshot open/close ratios against
that snapshot's previous close, using **normalized model units rather than
broker shares**. This does not invent asset returns. A changed previous
raw-close quote pauses the experiment for source/corporate-action review,
rather than silently accepting a revision. Frozen funded-account simulation
and independent `bt` must agree before any new row is recorded; prior model
rows must also reproduce unchanged.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_research_accounts init
.venv/bin/python -I -B -m us_quant.prospective_research_accounts advance
.venv/bin/python -I -B -m us_quant.prospective_research_accounts status
```

Initialize only this new explicitly named model journal, never the existing
archives or old paused ledgers. The initial cash anchor has zero return and
no invented interest. Before 63 actual model sessions, no Sharpe estimate
is reported. Even later modelled returns are not broker performance,
independent selection-adjusted alpha or qualification of the historically
rejected source strategy. No orders or live deployment are authorized.

The [actual experiment registration](research/workbench/evidence/prospective_research_accounts_v1_registration.json)
occurred at **2026-10-10T22:10:59.950314Z**, before the first possible
October 12 model open, and binds the previously recorded target receipt.
The [initial status](research/workbench/evidence/prospective_research_accounts_v1_initial_status.json)
has **zero observed model sessions**. Four USD10,000 cash anchors are
bookkeeping initial capital, not four strategy discoveries or past returns.
Total return, Sharpe, CAGR and drawdown are unreported rather than
manufactured as a successful zero-risk performance history.

The [initial verification](research/workbench/evidence/prospective_research_accounts_v1_initial_verification.json)
actually calls advancement twice without new inputs and records no model
rows or costs. Parent/target receipt heads remain unchanged; the source
candidate is still historically rejected and 108 prior evaluations are
unchanged. Missing observations, revised raw quotes and independent-account
disagreement preserve failure attempts rather than produce success-shaped
fallbacks.

The [latest native daily sequence](research/workbench/evidence/prospective_data_automation_v8.json)
collects parent data, expanded data and target observations, then advances
only this separate registered research model. It does not trade, restart
old paused portfolios or claim that October 12/13 execution already occurred.
Future automatic execution and Sharpe qualification remain unverified.

### Descriptive factor-risk structure before the eligible new-family cycle

`config/factor-risk-structure.json` fixes a read-only study of actual asset
returns, **not the queued six-family strategies' performance**. The original
four equity funds, six equity-family funds and six-plus-GLD/IEF diagnostic
cohorts share the exact ten/five-year windows and verified source panels.
No allocation weights, portfolio returns, CAGR or Sharpe are calculated.

The report separates correlation-based and covariance-based effective risk
dimensions. Correlation normalizes asset volatilities; covariance retains
their actual observed scale. Neither dimension counts proven independent
economic alpha. The old four-factor helper is preserved and cross-checked;
its historically named `covariance_effective_dimension` field is in fact
correlation-based, not silently rewritten.

Full-sample SPY excess-return regressions describe beta and market R-squared.
Their residual dependence is an **in-sample, non-investable diagnostic**,
not a forecast or permitted beta hedge. Near-zero residual variance is
marked explicitly unestimable rather than assigned fake zero correlation.
Overlapping trailing 252-session month-end slices describe risk persistence,
not independent validation samples.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_risk_structure register
# Publish the frozen source/method registration before computing statistics.
.venv/bin/python -I -B -m us_quant.factor_risk_structure run
```

The sources, cohorts and event head are bound before results. Registered
factor definitions, the weekly allowance, queued October 17 eligibility,
108 completed evaluations and original outcomes are unchanged.
The [risk-study registration](research/workbench/evidence/factor_risk_structure_20261011_registration.json)
pins all sources and descriptive cohorts before statistics; it computes no
pending candidate performance or additional strategy evaluations.

The [actual dependence results](research/workbench/evidence/factor_risk_structure_20261011_results.json)
show that extra equity-family labels provide only a small increase in
volatility-normalized risk dimensions:

| Descriptive asset cohort | Ten/five-year correlation dimensions | Ten/five-year covariance dimensions |
|---|---|---|
| Original four equity families | 1.314 / 1.450 | 1.313 / 1.404 |
| Six equity families | 1.354 / 1.473 | 1.360 / 1.466 |
| Six equity plus GLD/IEF | 2.201 / 2.368 | 1.711 / 2.011 |

For the six equity funds, the first standardized component explains
85.57%/81.81% of observed ten/five-year variance. All six fund returns have
substantial in-sample market exposure; individual SPY R-squared ranges
are about 0.697-0.964 and 0.670-0.951. These are not forecasts, residual
alpha returns or proof of constituent overlap. GLD/IEF broaden the
descriptive risk space, but no capital allocation or portfolio performance
is computed from that comparison.

The [arithmetic/replay check](research/workbench/evidence/factor_risk_structure_20261011_reproducibility.json)
reproduces every full-window and overlapping rolling statistic exactly.
Independent singular-value decomposition of centered/raw and standardized
return matrices verifies the risk dimensions and leading components;
covariance slopes and squared Pearson correlations verify beta/R-squared
within 1e-12. The original four-asset helper also agrees unchanged.

This evidence does not retune the frozen equal/inverse-volatility
six-family candidates, register their queued definitions early, create a
historical trial or claim six independent return sources. All 108
completed configurations, 24 reviews and future-cycle eligibility remain
unchanged. Greater diversification must be verified in an actual
portfolio, not inferred simply by counting economic labels.

### Growth/factor/gold core with ratcheted portfolio protection

`config/growth-portfolio-protection.json` freezes two experiments using only
the four **already registered** families and actual unleveraged QQQ/GLD
inputs. They do not advance the queued size/share-issuance registration,
change weekly-definition eligibility or compute queued six-family outcomes.
QQQ is growth/market beta, not an independent new alpha family.

The monthly control reuses the frozen growth-satellite composition/risk
helper but does not impose its earlier daily 12% volatility cap. Half of
the equity sleeve goes to QQQ; the other half remains equally allocated to
the four factor funds. Equity/gold inverse-volatility balance and the
30%-70% equity bounds stay unchanged.

The second rule adds an unleveraged ratcheted-floor adaptation motivated by
Black and Perold's [1992 CPPI paper](https://doi.org/10.1016/0165-1889(92)90043-e).
The wealth floor is 85% of the highest actual closing account NAV,
including initial capital. Risk scaling is derived from the retained
15% drawdown target, capped at the monthly cash-funded budget; reduced
investment goes to actual BIL. There is no fitted cushion multiplier,
borrowed capital, synthetic interest or risk-ceiling guarantee.

Targets depend on fresh-window capital, incurred fees and execution delay,
so each path regenerates its own causal decisions. A five-percentage-point
actual-holdings band limits daily requests, pending targets are never
overwritten, and monthly updates remain queued until eligible. Gaps,
delayed execution, costs and cash lock-in can still defeat the drawdown or
return goals; this is not a paper replication or guaranteed insurance.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.growth_portfolio_protection
```

Preparation revalidates the original growth ETF snapshot and computes no
outcomes. Both [fixed rules were registered before outcomes](research/workbench/evidence/growth_portfolio_protection_20261011_registration.json).
Exact-anchor migration preserves all 108 prior configurations and 24
reviews, and adds no new factor definition. The two actual candidates
are pending; the queued six-family rules remain unregistered and unevaluated.
All original dual-window costs, Sharpe/SPY/drawdown criteria and paused
ledgers remain unchanged. The current prospective archive lacks QQQ;
no independent future result for this new study is claimed.

The [actual complete results](research/workbench/evidence/growth_portfolio_protection_20261011_results.json)
reject both fixed rules:

| Growth/factor/gold rule | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Monthly unprotected control | 0.9762 / 0.9470 | **1.000447** / 0.9548 | 19.16% / 19.31% |
| Ratcheted protection | 0.6890 / 0.4791 | 0.6973 / 0.5246 | 12.82% / 12.80% |

The control's single five-year base Sharpe barely exceeds one, but its
ten-year/stress Sharpe, ten-year SPY comparison and drawdown gates fail.
It is **not a qualified strategy**. Protection satisfies the modeled
drawdown gates in these four paths, with stress drawdown 13.51%/12.96%,
but has much lower net CAGR/Sharpe and misses SPY. These observed outcomes
do not guarantee protection against different gaps, costs or delays.

The [feedback/account attribution](research/workbench/evidence/growth_portfolio_protection_20261011_attribution.json)
checks each request against the actual closing NAV and high-water mark,
including fresh capital, incurred costs and original delay. Pending targets
are never overwritten. For five-year stress, protection pays USD2,422.68
across 1,622 tickets versus USD510.26 and 366 tickets for the control.
Its mean realized BIL share is about 29%, and CAGR is lower by about
8.22 percentage points. More frequent protection trades and lower risk
exposure are not incremental factor alpha.

The [growth-protection program snapshot](research/workbench/evidence/research_program_2026-W41_results_v30.json)
retains **110 completed configurations, 26 reviews, six registered
definitions, two terminal invalid code versions, the separately resolved
accounting incident, no pending registered candidates and no qualified
champion**. These two completed rules use only existing families; the
queued six-family strategies remain unregistered and unevaluated.
The [104-path read-only re-audit](research/workbench/evidence/research_program_source_bound_audit_v9.json)
confirms all 26 original outcomes, including 19 legacy and seven explicit
version-two reviews. The fixed 22-candidate selection diagnostic is not
silently extended to these later four candidates.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v9.json)
preserve data/target/model collection and the queued October 17 eligibility,
and do not treat the one barely passing base metric as success or authorize
post-result parameter tuning. The separately registered forward model still
has zero observed return sessions; no future scheduled execution, live
orders or objective completion is claimed.

### Two fixed downside-risk balances on the unchanged growth/factor core

`config/downside-risk-balance.json` compares downside semideviation and
5% expected-loss budgets while keeping the original 63-session window,
half-QQQ equity composition, four existing families, equity bounds and
funding/cost rules fixed. The already evaluated total-volatility monthly
control is reused, not counted again as a third new strategy.

Semideviation measures squared negative daily excess returns averaged over
all 63 observations. Expected loss uses the worst 5% empirical probability
mass, including a fractionally weighted boundary observation; simply taking
four full observations out of 63 would not be exactly a 5% tail. Risk-free
thresholds use the same frozen lagged market series as Sharpe evaluation.
Zero observed loss is not future risk-free proof: one zero sleeve estimate
uses the declared 30%-70% bounds; two zero estimates explicitly fail.

These are marginal risk-budget heuristics, not coherent joint-portfolio
expected-shortfall optimization, drawdown guarantees or new independent
factors. Reference metadata verifies Sortino/Price's
[1994 downside-risk paper](https://doi.org/10.3905/joi.3.3.59) and
Acerbi/Tasche's [2002 expected-shortfall paper](https://doi.org/10.1111/1468-0300.00091).
Current histories are exposed; no tail/window/growth-weight search or
post-outcome estimator revision is permitted in this version.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.downside_risk_balance
```

Preparation revalidates the original actual ETF snapshots and computes no
outcomes. Both [fixed rules are registered before outcomes](research/workbench/evidence/downside_risk_balance_20261011_registration.json).
Exact-anchor migration preserves 110 completed configurations and 26
reviews; the existing control is not recounted as a new experiment.
The two new candidates remain pending until source-bound evaluation under
unchanged net Sharpe/SPY/15% drawdown gates. Queued six-family definitions,
their October 17 eligibility and the independent forward model are not
modified or pre-evaluated.

The [complete actual results](research/workbench/evidence/downside_risk_balance_20261011_results.json)
reject both fixed downside estimators:

| Marginal risk estimator | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Semideviation | 0.9667 / 0.9323 | 0.9817 / 0.9297 | 19.59% / 19.74% |
| Exact 5% expected loss | 0.9672 / 0.9287 | 0.9841 / 0.9250 | 19.38% / 19.61% |

Neither improves the completed total-volatility control. The
[independent-risk/account attribution](research/workbench/evidence/downside_risk_balance_20261011_attribution.json)
reconstructs the estimators using scalar negative squares and explicit
probability weights rather than calling the production risk helpers.
All 131 monthly budgets per method and all eight full account target
matrices agree; dates, four-family/QQQ composition and funding constraints
remain fixed. Every comparison has lower net Sharpe/CAGR and higher
drawdown than the existing control. This failure does not establish that
every joint downside model is useless or justify a post-result tail grid.

The [current program snapshot](research/workbench/evidence/research_program_2026-W41_results_v32.json)
retains **112 completed configurations, 28 recurring reviews, six registered
definitions, two terminal invalid code versions, the resolved accounting
incident, no pending candidates and no qualified champion**.
The [112-path source-bound re-audit](research/workbench/evidence/research_program_source_bound_audit_v10.json)
reproduces all outcomes, including 19 legacy-engine and nine explicit
version-two reviews. The earlier fixed 22-candidate joint diagnostic is
not extended to the later six candidates. Neither queued six-family
strategy nor its October 17 registration eligibility changed.
The prospective model still has zero observed return sessions, with no
new live orders or research-champion promotion.
The [latest native instructions](research/workbench/evidence/prospective_data_automation_v10.json)
retain daily capture and eligible weekly exploration, the exact rejected
tail definition and the distinction between 112 completed studies and two
unregistered six-family plans. Future scheduled execution is not claimed.

### Dollar-risk source review without premature unleveraged admission

The [UUP source-admission audit](research/workbench/evidence/dollar_risk_source_admission_20261011.json)
investigates a currency driver rather than another equity-style label.
The actual Invesco product page identifies UUP, ISIN US46141D2036,
February 20, 2007 inception and long USDX futures against six currencies,
plus Treasury/money-market collateral income. Those facts do **not alone
prove a historical notional exposure cap no greater than NAV**.

The issuer's linked annual report, quarterly report and factsheet all
returned audience-selection HTML, not PDFs, under ordinary public requests.
No gate was bypassed, no shared or acquired credentials were used, and
search-generated assertions were not treated as primary leverage evidence.
The review therefore **does not admit UUP** into the unleveraged strategy
universe or silently fill `asset_leverage: 1`. This is an unverified
admission constraint, not a finding that UUP is necessarily a leveraged
product.

No price-history dataset, strategy specification, new factor registration
or complete backtest was added by this source check. All 112 completed
configurations, 28 reviews and original qualification conditions remain
unchanged. The actual four-step collection/target/model sequence again
returned same-session duplicate/no-new-data actions; it did not create a
new market observation, target, model return or future-execution claim.

### Receipt-backed daily operations and honest schedule provenance

`research_daily.py` runs the already registered parent-data, expanded-data,
target-observation and research-model stages in the fixed order. Each unique
operational receipt records source code, invocation start/end, per-stage
actual results and unchanged historical-research state. It does not create
factor definitions, candidate evaluations, archives or a trading account.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.research_daily --origin operator
```

A failed stage preserves completed acquisitions and stops later stages;
retries naturally use each component's existing duplicate/integrity guards.
A missed pre-open deadline is recorded as failure, not a successful
pipeline. There is no rollback of genuine previously acquired data and no
quiet catch that converts a source error into success.

The invocation origin is **declared, not authenticated scheduler proof**.
Even `--origin session_automation` does not establish that the app delivered
a native wakeup. A changed `nextRunAt` establishes schedule advancement,
not message delivery or agent execution. Actual command results verify the
pipeline separately; same-day duplicate success is not a new market
observation, portfolio return or Sharpe qualification.

The [native scheduling check](research/workbench/evidence/native_daily_scheduling_acceptance_20261011_v1.json)
observed the overdue schedule's metadata advance to October 12 at
01:00:02Z, with a 09:02 Asia/Shanghai metadata update. Available turn
history does not prove native message delivery. Its absence there also
does not prove delivery failed. This uncertainty is retained instead of
declaring the native automation end-to-end successful.

The [actual after-due execution receipt](research/workbench/evidence/research_daily_actual_run_20261011_v1.json)
records a **continuation-origin** run at 01:28:16.900797Z through
01:28:17.586734Z. All four stages completed: three existing-snapshot/target
duplicates and one no-new-observed-model-session result. No source hashes,
market observations, target counts, model returns or historical trials
changed. This verifies the pipeline separately, not the scheduler origin.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v11.json)
invoke the consolidated runner on future due turns and require actual
stage receipts. Origin remains declared rather than authenticated;
the daily cadence, queued weekly eligibility, original registrations and
112 completed strategy outcomes remain unchanged.

### Sector growth: stronger historical returns, unresolved drawdown

The [sector source-admission audit](research/workbench/evidence/sector_growth_source_admission_20261011.json)
verifies actual XLK and SOXX issuer identities, non-multiplied equity-index
mandates and 2,805 complete positive-volume sessions per fund. Primary
documents were actually retrieved and parsed; HTML PDF viewers remain
recorded as viewers rather than falsely reported PDFs. SOXX's disclosed
June 21, 2021 transition from the PHLX index to the NYSE Semiconductor Index
is retained. Current documents do not establish unchanged historical rules
or complete absence of fund-level derivatives and incidental borrowing.
Raw prices and proprietary document contents remain private.

The [two fixed specifications](research/workbench/evidence/sector_growth_balance_20261011_registration.json)
keep half of equity in the original four equal factor sleeves and replace
only the other half's QQQ exposure with either XLK or SOXX. The unchanged
growth-budget helper supplies monthly 63-session equity/gold risk balance,
30%-70% equity bounds and 2% target cash. Sector concentration is not a new
independent economic factor or proof of safer returns. Both candidates
retain the original capital, costs, execution delay, benchmark and
15% drawdown qualification policy.

```bash
cd research/workbench
.venv/bin/python -m pip install -r requirements-sector-research.txt
.venv/bin/python -I -B -m us_quant.sector_growth_balance
```

Preparation refuses an existing output directory and computes no strategy
outcomes. The PDF dependency is separate because the imported dependency
manifest remains immutable. Registration used the actual clock and an
exact prior-engine/event migration. At registration, all original files,
112 completed configurations, 28 reviews and technical failures remained
unchanged; the [preregistered snapshot](research/workbench/evidence/research_program_2026-W41_sector_v33.json)
recorded two pending candidates. Actual source/code/specifications and
registration receipts were committed and pushed as `86c486ee` before
account outcomes.
The IJR/PKW queue still waits for the real October 17 cycle, and no
prospective sector returns, live orders or qualified champion are claimed.

The [completed results](research/workbench/evidence/sector_growth_balance_20261011_results.json)
retain **both rejections** under the unchanged qualification contract:

| Candidate | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Four factors + XLK + gold | Ten years | 15.93% / 1.02045 / 18.97% | 15.51% / 0.99169 / 19.09% |
| Four factors + XLK + gold | Five years | 19.47% / 1.06563 / 19.12% | 18.71% / 1.01815 / 19.24% |
| Four factors + SOXX + gold | Ten years | 18.40% / 1.07665 / 20.49% | 17.98% / 1.05066 / 20.49% |
| Four factors + SOXX + gold | Five years | 23.45% / 1.14441 / 20.65% | 22.62% / 1.10042 / 20.64% |

SOXX passes strict Sharpe and same-window SPY return gates in all four
paths, including stress against base-cost SPY. **Its drawdown still exceeds
15% in every path**, so it is a return-leading research reference, not
a qualified stable strategy. XLK also fails drawdown and misses ten-year
stress Sharpe. No investment objective, independent future performance
or trading authority is established.

The [preregistered matched-budget attribution](research/workbench/evidence/sector_growth_balance_20261011_attribution.json)
independently reproduces all 131 monthly allocations per candidate within
1.11e-16 and reaccounts eight QQQ substitutions with identical equity/gold
budgets, dates, funding and costs. Every candidate path has higher net
CAGR and Sharpe than its QQQ comparison. For SOXX, the improvement comes
with roughly 1.6 percentage points more drawdown; the fund's descriptive
in-sample SPY beta is 1.55/1.80 across ten/five years. This is not proof
of independent sector alpha. The worst loss interval peaks March 8, 2022
and troughs October 14, 2022 in these model accounts.

The [sector-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v34.json)
recorded **114 completed configurations, 30 recurring reviews, no
pending candidates and no qualified champion**. The
[120-path read-only source audit](research/workbench/evidence/research_program_source_bound_audit_v11.json)
retains all 19 legacy and 11 version-two reviews and leaves the previous
28 conclusions unchanged. The fixed 22-review joint-selection inference
does not cover the later eight studies.

The [preservation receipt](research/workbench/evidence/sector_growth_baseline_preservation_20261011.json)
verifies unchanged imported files, all original prospective receipt heads
and zero observed model-return sessions. A real continuation-origin
daily run at 02:20:17Z completed four duplicate/no-new-data steps;
it is not independent native-delivery proof. The
[updated daily/weekly instructions](research/workbench/evidence/prospective_data_automation_v12.json)
retain the original cadence, source boundaries and October 17 queued-family
eligibility. They do not silently replace the frozen rejected macro
forward experiment with this historically better sector strategy.

### Dollar funding risk: lower drawdown, unresolved ten-year outperformance

The [dollar source audit](research/workbench/evidence/dollar_information_source_admission_20261011.json)
retains 299 actually acquired official H.10 weekly releases. **298 are
admitted; one page with a ten-day release-date conflict is quarantined.**
Twelve identified archive/page date conflicts retain both declared dates
and conservatively admit values only after the later date. This does not
authenticate true first-publication timestamps. The dated `pubtables`
template is accepted only with its exact H.10 table title and observation
headers. Original source-check failures and raw files remain private.

Dollar risk uses the original goods-only broad index through its announced
December 2019 retirement, then the goods/services replacement at both ends
of each comparison. The replacement's retroactively constructed 2006
history is never treated as pre-2019 information. No different index
methods or January 2006 indexation regimes are level-spliced. Actual
published snapshots differ from current FRED values in many observations;
those current CSVs are diagnostics only, not fallback trading inputs.
All 131 month-end comparisons are source-bound, with at most 13-day
dollar observation age and five-day real-yield age. DFII10 retains its
previous two-session lag and current-vintage limitation; this is not a
complete ALFRED-vintage macro study.

The [actual preregistration](research/workbench/evidence/dollar_risk_guard_20261011_registration.json)
freezes exactly two guards on the unchanged four-factor/SOXX/gold monthly
allocation. Positive 63-session dollar change triggers the first;
positive dollar **and** lagged real-yield changes trigger the second.
A trigger halves all original non-BIL targets and transfers the removed
budget to actual BIL, retaining positive factor sleeves, relative
equity/gold composition and 2% target idle cash. The original 63-session
risk window, 50% scale, dates, source code, funding and costs are not
retuned to the already observed drawdown. Sharpe, SPY and 15% drawdown
gates remain unchanged.

```bash
cd research/workbench
# Requires the sealed local source files; choose a new preparation directory.
.venv/bin/python -I -B -m us_quant.dollar_risk_guard --output data/dollar-risk-prepared-new
```

Preparation computes no portfolio outcomes. The separately preregistered
eight constant-scale attribution controls replace each guard with its
full-window average scheduled exposure; they are hindsight descriptive
controls, not causal candidates, added strategy trials or holdouts.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_dollar_v35.json)
recorded **114 completed configurations, 30 reviews and two pending
candidates**, with no qualified champion. The exact engine migration
preserves all old records and prospective registration/receipt bytes.
No new alpha-family definitions, early IJR/PKW admission, prospective
sector performance, UUP investment, orders or live deployment are claimed.
Actual source/code/specifications and registration receipts were
committed and pushed as `a5387f89` before computing the new outcomes.

The [completed results](research/workbench/evidence/dollar_risk_guard_20261011_results.json)
retain both rejected candidates:

| Guard on the four-factor/SOXX/gold core | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Dollar appreciation alone | Ten years | 14.25% / 1.06754 / 13.13% | 13.68% / 1.01924 / 13.26% |
| Dollar appreciation alone | Five years | 19.09% / 1.18889 / 13.15% | 18.10% / 1.11689 / 13.28% |
| Dollar and real-yield increase | Ten years | 16.03% / 1.06793 / 17.92% | 15.25% / 1.01243 / 17.74% |
| Dollar and real-yield increase | Five years | 21.34% / 1.25400 / 13.14% | 20.01% / 1.16687 / 13.28% |

**The dollar-only guard passes every Sharpe and 15% drawdown gate, but
fails ten-year SPY outperformance under both cost assumptions.** Its
worst-path Sharpe is 1.01924135 and maximum drawdown 13.28272896%.
Same-window SPY has 15.46% ten-year base CAGR and 15.41% stress CAGR;
the strategy cannot qualify by showing only the stronger five-year
results. It is a more stable research reference, not a qualified champion.
Requiring real-yield confirmation keeps more market exposure and raises
five-year returns, but its ten-year drawdown and stressed SPY gates fail.
Its largest ten-year loss occurs in February-March 2020, outside the
five-year window. Neither rule guarantees future drawdown or returns.

The [independent attribution](research/workbench/evidence/dollar_risk_guard_20261011_attribution.json)
reads the actual H.10 tables through a separate parser and selects
real-yield observations using independently calculated strictly later
sessions. All 131 risk changes reproduce exactly, and both complete
2,805-row target matrices agree within 2.23e-16. Eight preregistered
constant-scale controls also reconcile with independent accounting.
The dollar-only guard improves net CAGR and Sharpe over its same-average
scheduled-exposure controls in all four comparisons. The confirmed guard
has lower ten-year stress Sharpe than its control. These controls use
full-window mean exposure, not causal signals or matched realized risk;
positive differences do not establish selection-adjusted alpha.

The [dollar-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v36.json)
recorded **116 completed configurations, 32 reviews, no pending candidate
and no qualified champion**. The
[128-path source audit](research/workbench/evidence/research_program_source_bound_audit_v12.json)
reexecutes all 19 legacy and 13 version-two reviews without changing any
previous conclusion. The earlier fixed 22-review joint-selection
diagnostic excludes the later ten reviews and the original 84 trials.
The [preservation evidence](research/workbench/evidence/dollar_risk_baseline_preservation_20261011.json)
checks all 69 imported files, previous 30 reviews and original prospective
registration/receipt heads. A real continuation-origin daily run at
03:12:17Z completed all four duplicate/no-new-model-session stages;
there are still zero observed forward-model sessions.
The [updated automation receipt](research/workbench/evidence/prospective_data_automation_v13.json)
keeps daily input collection and the real October 17 weekly discovery
eligibility. Configuration is not proof of native delivery, and the
original rejected macro forward experiment is not replaced.

### Separate prospective dollar-guard experiment

The [new forward profile](research/workbench/config/prospective-dollar-guard.json)
observes the fixed `semiconductor_dollar_half_guard` research reference
without replacing the existing macro experiment. Its historical status
remains rejected for ten-year SPY underperformance. No new backtest,
alpha-family definition, account authorization or champion is created.

The [actual input registration](research/workbench/evidence/prospective_dollar_inputs_v1_registration.json)
freezes a separate archive that inherits only the matching parent
snapshot's original bytes and acquisition times, then genuinely acquires
QQQ, XLK, SOXX and the official H.10 vintages needed by the latest completed
month. Real source preflight verified all three quotes and four required
weekly releases. Its 502 historical context rows are **not** prospective
observations; at registration the new archive had no snapshots or returns.
Current FRED dollar CSVs are not substituted for archived publications.
The original collector, targets, model and imported files remain unchanged.

The versioned adapter reuses the frozen monthly budget helper and exact
dollar-only guard. Targets are generated at their actual observation
time, never backdated to the rule month, and cannot be retuned within an
already recorded month. New target records must reproduce from their
own captured inputs. The separate model uses the explicitly declared
twelve-fund universe with the existing funded and independent accounting,
unchanged capital/costs/delay, no missing-day backfill and no broker.
Original model-journal checks are reused rather than changing its sealed
eight-fund implementation. Neither an initial target nor a cash anchor
is a future return observation.

```bash
cd research/workbench
# All directories are now registered. Do not repeat any init action.
.venv/bin/python -I -B -m us_quant.research_daily --origin session_automation && \
  .venv/bin/python -I -B -m us_quant.prospective_dollar_guard daily --origin session_automation
```

All registration, source and failure artifacts are retained in their
separate directories. No existing archive is reset. The original
daily pipeline and October 17 factor-definition eligibility are unchanged.

The source/profile and actual input registration were committed and pushed
as `290aa686` before the
[real baseline capture](research/workbench/evidence/prospective_dollar_guard_v1_baseline.json).
The separate October 9 baseline was genuinely acquired October 11 at
03:44:03.696386Z, inheriting all 35 matching parent files with their original
October 10 acquisition times. Only the additional quotes and official
H.10 releases receive new acquisition timestamps. This is one baseline,
not a second or third independent market day.

The first target was actually recorded at 03:44:05.942057Z from the
September 30 rule month. **It is not backdated to September 30.**
Its own captured-source replay and independent scalar budget agree
exactly. The
[separate research model](research/workbench/evidence/prospective_dollar_guard_v1_model_registration.json)
was registered at 03:44:06.941079Z, before the earliest possible
October 12 13:30Z opening. Base/stress strategy and SPY model accounts
each anchor USD 10,000; all have **zero observed model sessions** and
null return, Sharpe, CAGR and drawdown. These are neither broker accounts
nor executed trades. The minimum reporting window remains 63 actually
observed model sessions, not downloaded historical rows.

The [preservation audit](research/workbench/evidence/prospective_dollar_guard_v1_preservation.json)
checks all original/new control heads, all 69 imported files and unchanged
116 completed historical configurations. The original macro and separate
dollar pipelines genuinely ran at 03:45Z with four duplicate/no-new-model
actions each; no extra data day or account return was invented.
The [updated automation instructions](research/workbench/evidence/prospective_data_automation_v14.json)
run the original pipeline first and the separate dollar pipeline only
after it succeeds. Actual native delivery remains unverified.
The first normal capture opportunity for an October 12 U.S. model
session is October 13 morning in China; October 12 morning still sees
the October 9 baseline. The historical SPY gate still fails, no champion
is promoted, and the investment objective remains incomplete.

### Financial-condition confirmation using actual historical vintages

The [NFCI source admission](research/workbench/evidence/financial_conditions_vintages_20261011_admission.json)
retains **225 actual ALFRED vintages** in six public-form downloads.
Every requested date appears in its archive README and CSV column;
future observations are missing rather than replaced with current values.
Three separately acquired probes reproduce the batch data. Native
publication precision and revisions are retained. The first available
ALFRED vintage is May 25, 2011, before every required query. The actual
Chicago Fed FAQ states Wednesday 08:30 **Eastern**, with early-week
federal holidays shifting release to Thursday. Search summaries claiming
Central Time were not used.

For both current and 63-session reference inputs, the
[frozen specifications](research/workbench/evidence/financial_conditions_guard_20261011_registration.json)
use the snapshot as of the preceding NYSE session. This is conservative
dated availability, not authentication of first-publication timestamps.
NFCI revisions include incoming data and changing estimated indicator
weights; a difference between as-published readings is not a pure measure
of an economic shock. Current revised histories are not fallback inputs.
The official weekly observation cutoff of October 2 does not change the
original October 5 daily return cutoff or either evaluation horizon.

An initial source audit found two year-end queries, December 29 and 30,
2022, with last observations from December 9. **The 14-day freshness
limit was not relaxed to admit those 20/21-day-old values.** The
unregistered strict-coverage failure and its exact source/code snapshots
remain retained. Before strategy outcomes, the data-only revision
explicitly froze standby: a candidate does not rebalance when its
required NFCI input is stale, records the source gap and retains actual
existing holdings. Every actual daily price return and cost stays in
the account; no period is dropped, zeroed or reconstructed.
Tightness pauses December 30, 2022; deterioration also pauses March 31,
2023 because its 63-session reference is stale. All consumed readings
are at most 13 days old. An ALFRED gap does not by itself prove that the
original publisher paused publication.

Exactly two fixed confirmations are registered. Dollar appreciation must
coincide with either NFCI strictly above its source-defined zero-average
threshold, or a strictly positive 63-session change in separately archived
NFCI readings. A confirmed state halves all original non-BIL targets;
otherwise the original four-factor/SOXX/gold allocation remains unchanged,
except for explicit source standby. All factor shares, equity/gold budget
helpers, costs, 2% target cash and qualification gates are preserved.
The index is information, not a traded or leveraged product, new alpha
family or proof of independent returns.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.financial_conditions_guard --output data/nfci-prepared-new
```

Preparation requires sealed local source artifacts, refuses overwrite and
computes no portfolio outcomes. Eight constant-exposure attribution
controls were also preregistered, retaining the same candidate-specific
source-pause dates; they use full-window mean exposure and are not causal
candidates, holdouts or added historical trials.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_nfci_v37.json)
recorded **116 completed configurations, 32 reviews and two pending
candidates**. Exact migration preserves every old record and all seven
existing prospective profiles. Neither source acquisition nor
preregistration establishes the investment objective.
The actual source, code and registration receipts were committed and
pushed as `87b96bf7` before calculating the new portfolio outcomes.

The [completed results](research/workbench/evidence/financial_conditions_guard_20261011_results.json)
retain both failures:

| NFCI confirmation | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Tighter than average | Ten years | 18.38% / 1.07598 / 20.49% | 17.96% / 1.04930 / 20.49% |
| Tighter than average | Five years | 23.43% / 1.14355 / 20.65% | 22.57% / 1.09831 / 20.64% |
| Deteriorating as-published readings | Ten years | 14.80% / 0.95987 / 17.92% | 13.55% / 0.86955 / 17.74% |
| Deteriorating as-published readings | Five years | 19.92% / 1.10007 / 13.94% | 17.69% / 0.95815 / 14.76% |

**The tightness confirmation never triggered at any of the 131 scheduled
month-ends.** Its 130 nonempty target rows differ from the original SOXX
core only through the December 2022 source standby, not better risk
information. It still fails drawdown in every path. The deterioration
confirmation reduces risk in 41 months and has 129 target rows, but fails
ten-year Sharpe, drawdown and SPY gates; five-year stress Sharpe also
falls below one. These outcomes do not improve the qualifying strategy
set. The dollar-only guard remains the more stable reference, with its
original ten-year SPY failure still exposed.

The [independent attribution](research/workbench/evidence/financial_conditions_guard_20261011_attribution.json)
uses a separate raw ZIP/CSV reader for historical NFCI snapshots and the
separate H.10 table parser. All source readings, missingness and 131 risk
changes reproduce exactly. Both full 2,805-row target matrices, including
standby rows, agree within 2.23e-16. Eight constant-exposure controls
reconcile with independent accounts and the identical source-pause
dates. Deterioration has lower Sharpe than its constant-exposure control
in all four paths; tightness is the same exposure as its control.
The controls are hindsight attribution, not new candidates, complete
search correction or matched realized risk.

The [NFCI-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v38.json)
recorded **118 completed configurations, 34 reviews, no pending
candidate and no qualified champion**. The
[136-path read-only audit](research/workbench/evidence/research_program_source_bound_audit_v13.json)
retains 19 legacy-engine and 15 version-two reviews. The
[baseline-preservation receipt](research/workbench/evidence/financial_conditions_baseline_preservation_20261011.json)
checks every previous record, all 69 imported files and all seven
prospective profiles. Both original macro and separate dollar daily
pipelines actually completed duplicate/no-new-model-session runs at
04:28Z; each model still has zero observed return sessions. The
[updated automation receipt](research/workbench/evidence/prospective_data_automation_v15.json)
retains daily data collection and the real October 17 weekly discovery
eligibility. NFCI is not silently added to either frozen forward model.
The old 22-review joint-selection inference excludes the subsequent
twelve reviews and all earlier 84 trials. The investment objective is
still unverified, with no live deployment or order authority.

### Timely sector risk: fixed option-term and dollar guards

The [source audit](research/workbench/evidence/sector_term_source_admission_20261011.json)
revalidates the unchanged official VIX and VIX3M histories previously
admitted for the factor-only study. Both cover all 2,805 required ETF
sessions without forward fill. The actual current Cboe dashboard and
advertised methodology PDF confirm VIX3M's 93-day horizon, regular
09:31-16:15 Eastern calculation interval and September 2017 ticker rename
from VXV. The current methodology's launch entry and first history date
are retained, not rewritten to imply complete historical rule continuity.
Current downloaded closes are not authenticated unrevised intraday
publication snapshots. No volatility index becomes a portfolio holding.

The [two preregistered rules](research/workbench/evidence/sector_term_guard_20261011_registration.json)
keep the exact four-factor/SOXX/gold month-end budgets. The first halves
all non-BIL exposure when the existing VIX/VIX3M ratio threshold reaches
one; the second uses that condition **or** the latest known monthly
dollar-risk state. Removed exposure goes to actual BIL, not synthetic
interest. Factor/gold relative budgets and 2% target idle cash remain
unchanged. Dollar information updates only at a completed month, not
with later daily observations.

Targets are emitted only for a month-end budget update or binary-state
change. Each absolute target remains in its dated execution queue:
a one-day inversion and reversal are two later-open instructions, not
an overwritten pending order. The original base/stress delay, costs,
independent accounting and full qualification contract stay fixed.
Neither threshold nor 50% reduction is selected from a new grid.
The earlier factor-only option-risk failures remain retained; this
separate sector composition is not an independent alpha-family discovery.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.sector_term_guard --output data/sector-term-prepared-new
```

Preparation requires sealed local sources, refuses overwrite and computes
no portfolio outcomes. Eight mean-exposure controls are preregistered
as hindsight attribution, with event-frequency differences disclosed;
they are not causal candidates, matched realized risk or new trial counts.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_sector_term_v39.json)
records **118 completed configurations, 34 reviews and two pending
candidates**, no qualified champion. Exact engine migration retains
every previous record and all seven forward profiles. No new prospective
target experiment, early October 17 family admission or live orders
are authorized by these registrations.

## Progress and publication

See [integration status](docs/integration/STATUS.md) and
[`docs/research_progress.json`](docs/research_progress.json). GitHub progress
uses reviewed explicit commits/PRs, not an unconditional git-push cron job.
The scheduled agent may create checked data/research checkpoints; the
standalone Slack job cannot run git or publish files.

```bash
.venv/bin/python -B scripts/check_publication.py
# Stage only the listed, reviewed paths; never git add .
.venv/bin/python -B scripts/check_publication.py --staged
```

The publication allowlist excludes raw artifacts, environments, account
configuration, holdings/importer modules and the original private handoff.
