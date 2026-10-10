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

## Progress and publication

See [integration status](docs/integration/STATUS.md) and
[`docs/research_progress.json`](docs/research_progress.json). GitHub progress
uses reviewed explicit commits/PRs, **not a configured daily automatic push**.
The Slack job cannot run git or publish files.

```bash
.venv/bin/python -B scripts/check_publication.py
# Stage only the listed, reviewed paths; never git add .
.venv/bin/python -B scripts/check_publication.py --staged
```

The publication allowlist excludes raw artifacts, environments, account
configuration, holdings/importer modules and the original private handoff.
