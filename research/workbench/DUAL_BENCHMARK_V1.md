# SSD dual-benchmark protocol, 2026-10-27

The separately frozen `config/dual-benchmark-v1.json` applies only to newly
registered research. It preserves the original four windows/cost paths and adds
net CAGR strictly above 12% and a full-capital, matched-cost QQQ total-return
benchmark alongside SPY. The old stress requirement to also beat base-cost SPY
remains. Sharpe must exceed 1 and drawdown must not exceed 15% on every path.
Turnover, modeled orders and fees are measured; there is no invented turnover
ceiling. Net returns are after modeled costs and before personal taxes.

The accepted private SSD environment, original inputs and frozen ledger are
reused, not reinstalled or reset. Source admission verifies the original raw
quotes, adjusted panels, corporate actions, calendar and current primary issuer
identity. A current issuer page is not proof of historical constituents, and an
HTML country-selection page returned by a PDF URL is not a factsheet. Such
failed document attempts remain visible and are not bypassed. QQQ's first-trade
date is explicitly vendor-reported, not an independently established issuer
inception date. Raw documents, prices and daily account ledgers remain private.

The original bundle's README and manifest remain byte-for-byte unchanged; this
separate document describes the new protocol without rewriting that history.

## Initial benchmark command

From the SSD workspace root:

```bash
. "/Volumes/Extreme SSD/us_stock_qr-research/migration/ssd-takeover-20261027/ssd-runtime-env.sh"
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.dual_benchmark_v1 \
  --freeze "$SSD_RESEARCH_ROOT/migration/research-activation-20261027/dual-benchmark-protocol-freeze.json" \
  --output "$SSD_WORKBENCH/reports/ssd-continuous-research-v1/benchmarks-20261027-v1"
```

The one-shot command refuses existing output, enforces the real SSD path and
research deadline, and holds a single-computation lock. It constructs eight
benchmark accounts with both funded accounting and independent `bt`, plus a
closed-form single-purchase check. Benchmark accounts are not strategy trials
or forward validation. Subsequent candidate accounts still require their own
actual preregistration and sanitized publication fence. Never rerun completed
accounts merely because a continuation message arrives.

## First measured benchmark hurdle

| Window | SPY base net CAGR | SPY stress net CAGR | QQQ base net CAGR | QQQ stress net CAGR |
|---|---:|---:|---:|---:|
| 2016-10-06 through 2026-10-05 | 15.4620% | 15.4056% | 21.1624% | 21.1239% |
| 2021-10-06 through 2026-10-05 | 14.0409% | 13.5283% | 17.0740% | 16.4707% |

Consequently, 12% or even 20% net CAGR alone cannot pass the ten-year
dual-benchmark gate. The source-bound four-path comparison, Sharpe and drawdown
requirements still apply together. Each benchmark account has one modeled
purchase, costing USD5.9970 in base or USD20.9581 in stress, funded from its own
fresh USD10,000. These are historical reference results, not a strategy success
or a promise of future returns.

The aggregate-only public reference evidence is
[`evidence/dual_benchmark_20261027_v1.json`](evidence/dual_benchmark_20261027_v1.json).
It includes scalar metrics, turnover, fees, source identities and hashes, not
raw issuer documents, market prices or daily account series.

## First registered equity-first pair

[`config/equity-first-regimes-v1.json`](config/equity-first-regimes-v1.json)
freezes a different capital-deployment architecture rather than another
volatility-target or sector-share grid. Both hypotheses retain half SOXX and
equal allocations to the four existing momentum/value/quality/low-volatility
style funds within equities, a 2% target cash reserve, the 200-observation trend
law and a 0.025 actual-holdings trade band. Eligible equity receives the
investable budget; the defense is separately trend-qualified GLD or actual BIL.
The second hypothesis caps equity exposure using prior observed VIX variance
relative to the current completed observation. It is not a new economic factor,
a fitted risk forecast, or a guaranteed drawdown floor.

The [actual registration packet](evidence/equity_first_20261027_registration_v1.json)
records both registrations at 2026-10-27 16:30 UTC (October 28 Beijing), before
real target preparation. Each complete 2,553-session target matrix matched a
separate sequential oracle and prefix-truncation checks. This is a causal-rule
proof, **not a return evaluation**. All source, policy, implementation and
benchmark references are bound to the packet; target matrices remain private.

The new immutable JSON journal is separate from the original frozen SQLite
ledger. The original 178 completed configurations, 94 recurring reviews,
96 recurring versions and two terminal technical failures remain unchanged.
The pair adds two registered hypotheses, not a new two-trial authorization cap.
All the history is already exposed and adaptively selected; even a four-path
point-gate pass would not prove selection-adjusted alpha or independent forward
performance.

After actual sanitized feature-branch publication, evaluate using its real
40-character commit SHA:

```bash
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_research_v1 \
  evaluate --commit "$PUBLISHED_COMMIT"
```

The publication fence checks the configured remote, explicitly allowlisted
extensions, unchanged pinned imports, registration identities and frozen bytes.
All four fresh accounts belong to the same candidate and parameters. Completed
path receipts are reused after an interruption; incomplete attempts and
technical failures remain recorded. Do not modify a registered implementation
to conceal an evaluation failure, rerun finished studies on a timer, or treat
offline modeled orders as permission to place brokerage orders.

## First pair outcome: both rejected

The preregistration commit `00b82e9e00cc777901689190083f0627c4ac4a48` was actually
observed on the configured remote before candidate evaluation. All eight paths
completed at `2026-10-27T16:52:00.173786+00:00`; maximum independent accounting
discrepancy was approximately USD1.68e-10. These are two new completed trials
and two reviews, not eight configurations. The original 178/94/96 ledger remains
unchanged, with 180 completed configurations across the two lanes.

| Candidate | Path | Net CAGR | Net Sharpe | Maximum drawdown | Modeled fees, USD |
|---|---|---:|---:|---:|---:|
| Equity-first trend | 10y base | 18.3806% | 0.8412 | 38.5221% | 1,803.77 |
| Equity-first trend | 10y stress | 18.8578% | 0.8578 | 31.2652% | 4,944.25 |
| Equity-first trend | 5y base | 17.7366% | 0.7037 | 39.1256% | 816.89 |
| Equity-first trend | 5y stress | 18.7128% | 0.7420 | 31.7827% | 1,890.88 |
| With VIX variance cap | 10y base | 13.5602% | 0.6891 | 36.0993% | 6,697.93 |
| With VIX variance cap | 10y stress | 14.4954% | 0.7334 | 26.4615% | 9,799.22 |
| With VIX variance cap | 5y base | 10.9685% | 0.4386 | 37.4171% | 3,000.66 |
| With VIX variance cap | 5y stress | 14.5809% | 0.6033 | 27.0843% | 4,005.40 |

Every path fails both the Sharpe and drawdown gates. The uncapped variant also
misses QQQ in both ten-year paths; the capped variant misses QQQ everywhere.
The cap raises ten-year base annualized two-sided turnover from 13.1837 to
24.9775 and modeled orders from 469 to 4,487. These measured comparisons do not
isolate fees from the changed risk exposure. Stress has a different original
execution delay and trade history, so a higher stress return is not evidence
that higher costs improved returns.

The [complete sanitized result](evidence/equity_first_20261027_results_v1.json)
contains all eight gate decisions, execution disclosures and evidence hashes.
No prospective observation or brokerage order was created. Diagnose saved
evidence before choosing a distinct economic hypothesis; do not conceal these
failures, replay finished accounts, or turn the failed cap into an unregistered
parameter grid.

## Separately frozen commitment-duration study

[`config/commitment-duration-v1.json`](config/commitment-duration-v1.json) keeps
the original uncapped equity-first signal and studies two execution-intent
architectures: carry the last completed XNYS month-end target, or average
21 rotating daily target commitments. Twenty-one comes from 252 sessions
divided by 12 months; it is not selected by a horizon search. Both need
21 complete source observations before issuing an aggregate intent.

The source is the already saved and proved target matrix, not a refit or replay
of the old strategy. Source files, old code, the 24-event pre-study journal
prefix, four economic families, 200-observation threshold, 98% target invested
budget and 0.025 actual-holdings band remain fixed. A calendar-aware month-end
test must not mistake the end of a truncated file for the end of a month.
An independent sequential/deque oracle and prefix truncations check both laws
before accounts. Only one funded account trades each aggregate target: these
are neither separately capitalized shadow portfolios nor averaged CAGRs.

The [actual packet](evidence/commitment_duration_20261027_registration_v1.json)
records both registrations at 2026-10-27 18:39:57 UTC, before target preparation.
Both have 2,533 defined sessions. The independent maximum target-weight
difference is zero for month-end commitment and approximately 2.22e-16 for the
21-slot mean; eight prefix checks include a non-month-end final date. No
candidate account was calculated by this registration or target proof.

Monthly commitment does not prohibit intra-month drift rebalancing under the
unchanged daily holdings-band executor. Both laws can retain declining
exposures, miss rebounds, or increase costs; neither promises a drawdown floor.
The economic question was chosen after 180 completed configurations and the
first pair's saved 2022 holdings/PnL diagnosis. All historical windows remain
exposed and adaptively selected. This is not a new independent test set.

The separate driver leaves the original registered runner unchanged:

```bash
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_commitment_research_v1 register
# Publish the exact sanitized registration, code and policy before accounts.
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_commitment_research_v1 \
  evaluate --commit "$PUBLISHED_COMMIT"
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_commitment_research_v1 publish-result
```

Registration is resumable, not permission to repeat completed studies. The
new packet has its own publication check in addition to the inherited source
and engine check. New candidate IDs receive distinct account directories
under the immutable executor's existing `equity-first-20261027-v1` namespace;
cohort results are separately stored under `commitment-duration-20261027-v1`.
No original account is overwritten or reclassified. Aggregate progress counts
all actual journal registrations, reviews, pending candidates and technical
failures, rather than resetting counters to the size of the latest pair.
Authorization expiry preserves partial work without inventing a completed
account or a technical failure.

The dashboard retains the original first-pair snapshot and derives separate
cohort/global totals. Pending and failed paths remain missing, never zero-return
successes. It uses only explicitly published scalar evidence, not private
prices, targets, daily accounts or the mutable private journal.

### Commitment outcome: lower notional turnover, still rejected

The exact preregistration/code commit
`56c3c9366bbbaf0249eb7990d500f9edc3aaa143` was verified on the configured remote
before accounts. All eight paths completed at
`2026-10-27T18:50:43.497706+00:00`. Maximum independent accounting discrepancy
was below USD4.08e-10. These are two additional completed hypotheses,
not eight new configurations: the SSD lane has four completed trials, the
preserved original lane has 178, and there are no new technical failures,
qualified champions or independent forward observations.

| Commitment | Path | Net CAGR | Net Sharpe | Maximum drawdown | Modeled fees, USD |
|---|---|---:|---:|---:|---:|
| Month-end | 10y base | 23.1015% | 0.9731 | 32.8762% | 666.64 |
| Month-end | 10y stress | 21.6486% | 0.9139 | 32.8762% | 1,856.87 |
| Month-end | 5y base | 27.3171% | 1.0755 | 22.7963% | 286.23 |
| Month-end | 5y stress | 24.3694% | 0.9619 | 22.7512% | 675.87 |
| 21 equal slots | 10y base | 20.6507% | 0.9157 | 31.2290% | 1,955.99 |
| 21 equal slots | 10y stress | 20.2712% | 0.8930 | 31.8730% | 2,290.67 |
| 21 equal slots | 5y base | 21.7387% | 0.8763 | 26.0663% | 1,124.95 |
| 21 equal slots | 5y stress | 21.0502% | 0.8475 | 25.8762% | 1,092.76 |

Month-end commitment beats matched-cost SPY and QQQ on all four paths, but
every drawdown exceeds 15%; only five-year base Sharpe exceeds 1. Its ten-year
base annualized two-sided turnover is 3.7742 versus the original daily
uncapped source's 13.1837, with 240 versus 469 modeled orders and USD666.64
versus USD1,803.77 in fees. Changed exposure timing and fees are inseparable
in this comparison; it is not a fee-only causal estimate.

The 21-slot law reduces ten-year base notional turnover further to 2.8535, yet
creates 1,653 modeled orders and USD1,955.99 in fees. Lower traded notional
does not guarantee fewer fixed-fee tickets. It misses QQQ in both ten-year
paths and fails all Sharpe/drawdown paths. Stress can have fewer tickets
because its original extra-session delay changes execution history; it is
not merely a surcharge on the same trades.

The [complete scalar evidence](evidence/commitment_duration_20261027_results_v1.json)
retains every gate and execution disclosure. None of the attractive CAGR,
turnover or single-path Sharpe numbers may be combined into an overall pass.
The next question must address the remaining risk mechanism rather than
searching more commitment horizons or claiming independent alpha.

## Fixed stateful price-channel study

The saved eight-path commitment diagnosis reconciles existing drawdown,
calendar returns, fees and ticket counts without creating new accounts or
strategy trials. All four ten-year commitment accounts first breach 15% on
2020-03-09, when the previous close's raw 200-observation equity intent is
still 98% stocks. Later losses also occur after that source is already off.
Thus merely accelerating exits from the unchanged slow source cannot be
claimed to address both mechanisms. These observations are descriptive,
not counterfactual returns or asset-level PnL attribution.

[`config/channel-regimes-v1.json`](config/channel-regimes-v1.json) freezes a
different state machine: enter strictly above the preceding 63 observed
closing levels, exit strictly below the preceding 21, and otherwise retain
state. Both barriers exclude the current close; equality retains state.
The first valid comparison initializes off unless an entry occurs.
State runs through genuine prehistory, without a reset at either formal
account start. The original 253-observation common target warmup remains.
Quarter/month horizons are fixed engineering choices, not fitted optima.

Both hypotheses use the original SOXX/four-style equity composition and
equity-first 98% budget. One retains the strict 200-observation GLD trend
defense; the other uses an independent gold channel, updated even while
stocks are eligible. Ineligible defense goes to actual BIL. The comparison
changes gold eligibility, not the channel horizons. The funded executor,
actual-holdings band, pending-order discipline, costs, delays, RF and all
four acceptance paths remain unchanged.

The [actual packet](evidence/channel_regimes_20261027_registration_v1.json)
records registrations at 2026-10-27 19:51:47 UTC before preparation of both
2,553-session target matrices. A separate sequential price oracle agrees
exactly, and eight prefix checks preserve every earlier target. This is
causal preparation, not a return result. At registration there are six SSD
registrations, four completed trials, two pending and 182 combined completed
configurations. No account is run before actual sanitized publication.

The separate `us_quant.ssd_channel_research_v1` driver reuses immutable
journal/receipt helpers and the original funded executor without modifying
old runners or their globals. Publication independently verifies the new
packet, original dependencies and prior cohort dependencies. All new code,
rules, inputs, pre-registration checks and diagnostic context are frozen.
After actual publication, use the actual commit, never a placeholder:

```bash
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_channel_research_v1 \
  evaluate --commit "$PUBLISHED_COMMIT"
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_channel_research_v1 publish-result
```

Completed accounts must be recovered, not replayed. This choice follows
182 exposed trials, so registration is not independent validation. Channel
breakouts can whipsaw, miss rebounds, retain sector concentration and lose
through gaps, delays or gold declines. No drawdown floor, cost reduction,
benchmark outperformance or future profitability is guaranteed.

### Channel outcome: all eight risk and QQQ paths fail

The actual preregistration commit
`b013af2a33c431f5ab386dff2643c86241979f5f` was verified before funded accounts.
All eight paths completed at `2026-10-27T19:56:54.739840+00:00`, with maximum
independent accounting discrepancy below USD7.64e-11. The
[complete scalar result](evidence/channel_regimes_20261027_results_v1.json)
preserves every path and rejection.

| Gold defense | Path | Net CAGR | Net Sharpe | Maximum drawdown | Modeled fees, USD |
|---|---|---:|---:|---:|---:|
| 200-observation trend | 10y base | 9.5344% | 0.4833 | 29.3793% | 1,973.47 |
| 200-observation trend | 10y stress | 6.0569% | 0.2866 | 34.0865% | 4,847.46 |
| 200-observation trend | 5y base | 14.7407% | 0.6292 | 23.5655% | 918.60 |
| 200-observation trend | 5y stress | 12.8371% | 0.5356 | 21.4330% | 2,504.04 |
| Independent channel | 10y base | 7.0490% | 0.3552 | 27.2817% | 1,534.77 |
| Independent channel | 10y stress | 4.2971% | 0.1896 | 34.5904% | 4,095.25 |
| Independent channel | 5y base | 12.5606% | 0.5440 | 19.7905% | 706.45 |
| Independent channel | 5y stress | 10.9207% | 0.4580 | 18.2080% | 2,039.03 |

Neither candidate passes any Sharpe, drawdown or QQQ path. The gold-trend
variant beats matched-cost SPY only in five-year base; this isolated gate is
not qualification. Ten-year base annualized two-sided turnover is 22.6986
and 17.0165, with 556 and 498 modeled tickets respectively. The channel law
does not establish the hoped-for implementation improvement. Changes in
exposure and execution remain inseparable; no fee-only causal claim is made.

The separate SSD journal now has six completed reviews, 24 account paths,
zero technical failures and zero full-gate passes. With the preserved
178 original configurations, the total is 184. No old record, source
refusal, prospective observation or acceptance threshold was changed.
Do not turn this failure into a channel-horizon grid or combine outcomes
from different candidates into a passing strategy.

## Fixed asymmetric option-exit latch

The completed saved-channel diagnosis finds the first ten-year base 15%
breaches on 2018-03-22, while both the prior-close channel and old trend
intents still hold stocks. The ten-year target histories have 115 and 86
asset-class intention changes. None of the four base paths has a
prior-close-channel-off/current-close-equity-held mismatch. Gold exposure also
appears at later first breaches. These are descriptive observations, not
opening-weight attribution, a fee-only effect or proof that a faster fill
would avoid the losses. The diagnosis adds no strategy trial or account.

[`config/option-exit-latch-v1.json`](config/option-exit-latch-v1.json) freezes
**one** new response/recovery clock after 184 exposed completed configurations.
It reads the original saved month-end commitment matrix, not a reconstruction
of its signal. At each completed close, the previously registered
`VIX >= VIX3M` condition latches all 98% invested capital into actual BIL,
including any gold allocation. Only a noninverted, actual XNYS month end clears
that state and restores the latest saved monthly intent. An inversion wins
on month end; a normal intervening day cannot restore risk.

The state begins off in genuine prehistory and is never reset for a formal
window. Source warmup remains undefined. The full original option-close
coverage, price/RF inputs, daily actual-holdings band, cash funding, pending
requests, costs and four-path gates remain unchanged. No pending request is
cancelled, and overnight gaps can still breach the drawdown limit. This is a
conservative portfolio response to an equity-option indicator, not proof it
forecasts gold, a new economic factor or a capital guarantee.

The [actual registration packet](evidence/option_exit_latch_20261027_registration_v1.json)
records `2026-10-27T20:31:30.392482+00:00`, before new target preparation.
All 2,533 defined target sessions match a separate sequential oracle exactly;
nine prefix checks include truncated non-month-end histories. The saved source
and original accounts are not recomputed. At registration there are seven SSD
registrations, six completed, one pending and still 184 combined completed.
No new candidate account had been evaluated at that point.

The packet-wide no-account declaration is explicit and required. A per-target
copy, when present, must agree. Missing or nonfalse packet-wide evidence and
conflicting per-target declarations are rejected, not defaulted to safety.
This keeps the actual registration and target-proof receipts unchanged.

The separate fixed driver checks its own packet plus all preserved publication
dependencies, without mutating earlier runners or their globals:

```bash
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_option_latch_research_v1 \
  evaluate --commit "$PUBLISHED_COMMIT"
"$SSD_WORKBENCH/.venv/bin/python" -I -B -m us_quant.ssd_option_latch_research_v1 publish-result
```

`PUBLISHED_COMMIT` must be the actual verified preregistration/code commit.
All four fresh funded paths remain required; no threshold, reentry-clock,
allocation-scope or source search follows from this registration. Rejected
outcomes and adaptive selection remain visible.

The actual preregistration commit
`c6452082acafcbffc6b7cf0fbce85ab1488abfaa` was remotely verified before
the four accounts completed at `2026-10-27T20:43:19.984335+00:00`.
The [complete scalar result](evidence/option_exit_latch_20261027_results_v1.json)
retains every gate and the fixed source/clock:

| Path | Net CAGR | Net Sharpe | Max DD | Annualized two-sided turnover | Model orders | Total costs (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10y base | 15.6961% | 0.8171 | 30.4238% | 11.5617 | 416 | 1,561.68 |
| 10y stress | 14.3425% | 0.7458 | 29.4229% | 11.5456 | 421 | 4,694.70 |
| 5y base | 13.9260% | 0.5856 | 30.5784% | 12.4900 | 241 | 591.88 |
| 5y stress | 13.0070% | 0.5446 | 29.6233% | 12.4532 | 236 | 1,610.68 |

The hypothesis is **rejected**. All four net-CAGR gates pass, but all four
Sharpe, drawdown and QQQ gates fail. Only ten-year base beats matched-cost
SPY; both stress paths also miss the preserved base-cost SPY gate.
The largest independent accounting difference is below USD 0.000000000073.
These are funded historical model orders, not broker orders. Stress changes
both costs and delay, so its return difference is not a fee-only effect.

The separate SSD journal now records seven completed/reviewed hypotheses,
28 account paths, zero pending, zero technical failures and no qualified
champion. Combined completed configurations are 185. Original ledger and
prospective models remain unchanged; no new source or broker request occurred.

The bounded saved-account diagnosis completed at 21:08:34 UTC. It reconciles
the four existing accounts without regenerating targets or calling an account
simulator. Both base paths first breach 15% on 2022-01-25: the previous-close
monthly source and latch still intend 98% stocks, with no option inversion.
Both stress paths first breach on 2022-01-27: the prior curve is inverted and
the saved target is BIL, but actual closing holdings still contain stocks
under the original extra delay. Neither event can be generalized into the
other mechanism.

The maximum drawdown starts from the 2021-11-18 peak. Its trough is 2022-12-28
for base and 2023-03-13 for stress. Observed prior-close-off/current-close-stock
sessions are zero for both base paths and 27/13 for ten-/five-year stress.
The 58/31 ten-/five-year target regime changes are intentions, not executed
order counts. Condition-labeled net log-return partitions describe the same
realized portfolio; they are not asset PnL attribution, conditional strategies
or evidence that a hypothetical earlier exit would satisfy the gates.

The private diagnostic receipt has SHA256
`db243e3aa7b59d1711d8f08e2467cfb0797576670120477938af1b4b47f0ce69`.
The 82-event journal and all original evidence remain unchanged. Next work
reviews distinct economic characteristics and their admission evidence,
not the failed option threshold, recovery clock or risk-cap parameters.
Historical queued definitions and prepared targets are not current
registrations; old catalog and account pipelines must not be resumed merely
because their planning dates have passed.

## Separate SSD factor catalog and source admission

The cached IJR and PKW source audit completed at 2026-10-27 21:42:25 UTC,
without new requests or target/account replay. Each fund retains 2,805 XNYS
sessions, 2015-08-10 through 2026-10-05. Raw/normalized prices, actions,
saved issuer identity and prepared-input hashes agree with the retained
evidence. The 102 pinned inputs remain unchanged.

The separately versioned source admission completed at 22:26:38.302827 UTC;
actual definition registration followed at 22:26:38.449697 UTC. These are
real **2026-W44** records, not a backdated completion of the W42 plans.
The [public catalog packet](evidence/ssd_factor_catalog_20261028_v1.json)
records two new definitions, IJR size and PKW net share reduction, with their
original semantic identities. The original six-definition/five-family
catalog is unchanged; together the catalogs contain eight definitions and
seven distinct economic families. This is not a claim of independent alpha.

The original SQLite is opened only with `mode=ro&immutable=1` and
`query_only=ON`; no original cycle, event, policy or candidate is changed.
The new source panel appends only IJR/PKW columns to the existing twelve
assets, retaining the original CSV cell text and exact numeric arrays.
The original RF file reference and array are reused without reserialization.
All eight existing benchmark accounts remain unchanged and are not rerun.

The original two-definition cycle limit and eight-day readiness-age rule
remain in force. Readiness uses the actual 21:42 cache audit, not a new
timestamp pasted onto the stale October 10 readiness document. Historical
quotes and issuer pages retain their original acquisition times. Historical
methodology continuity, independent factor returns and complete direct-stock
PIT data remain unproven; the S&P methodology403 is not bypassed.

The policy, admission implementation, tests, new input files and catalog
packet were frozen for this registration. At admission no eighth strategy
had been chosen and no new target or funded account had been computed.
No new observation was added to the original forward models. Counts were seven
SSD completions, 28 paths and 185 combined configurations, all without a
qualified champion. A subsequent fixed hypothesis needs its own actual
registration/publication fence under the unchanged dual-benchmark gates.

## Six-family factor-half composition: completed and rejected

The [actual registration packet](evidence/family_core_expansion_20261028_registration_v1.json)
records one fixed candidate, `equity_month_end_six_family_core_v1`, at
2026-10-27 23:58:51.410312 UTC. Its target-proof event follows at
23:58:51.721615 UTC and the packet at 23:58:51.722553 UTC. No candidate
funded accounts or performance metrics existed at this fence.

This tests economic composition rather than another failed risk threshold:
retain the actual saved `equity_first_month_end_commitment_v1` source,
including its warmup, SOXX half of equity, gold/BIL decisions and monthly
timing. Divide only the four-fund factor budget equally across MTUM, VLUE,
QUAL, USMV, IJR and PKW. Each gets one twelfth of an equity intent; SOXX keeps
one half. The total invested target remains 98%, with 2% idle cash.
The source's original five-fund equity signal is **not** reconstructed from
the new six-family portfolio.

The unchanged funded execution implementation supports fourteen assets.
Its original 0.025 actual-holdings band, single pending request, costs,
base/stress delays, fresh capital, RF and initial drawdown peak remain.
The sequential target oracle independently derives the factor budget from
the preserved SOXX sleeve and has zero difference. Eight truncation checks
preserve past targets, and both full formal windows have defined prior-close
anchors. Original targets, benchmark accounts and all old ledger rows are
not regenerated or rewritten.

The two new definitions were already registered and actually published in
`83514dd1186ff4fe612c5078a7787509919f80cb`; this study adds zero factor
definitions. It uses six actual source-supported families, not the
data-blocked direct-stock definitions. The old W42 two-portfolio plans used
a different aggregate risk budget and remain unchanged.

Selection follows 185 completed configurations and their exposed historical
diagnoses. No independent holdout, diversification benefit or qualification
is inferred. Shared equity beta, semiconductor concentration, small-cap
liquidity, buyback leverage, overlapping characteristics and additional
costs remain risks. This is one fixed composition, not a weighting or
lookback grid. The preregistration snapshot contained eight registrations,
seven completed, one pending, 28 paths and 185 combined completions.

Actual publication `ea997167c5121da2faa6a178e2b974b0950592be` was observed
before any account. All four paths finished at 2026-10-28 00:11:13.626966 UTC;
the [complete sanitized outcome](evidence/family_core_expansion_20261028_results_v1.json)
is rejected.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 22.5567% | 0.9489 | 33.8188% | 3.7843 | 342 | 764.49 |
| 10y stress | 21.1015% | 0.8907 | 33.8303% | 3.7614 | 335 | 1,935.42 |
| 5y base | 26.4066% | 1.0480 | 22.6943% | 4.7239 | 229 | 369.21 |
| 5y stress | 23.2902% | 0.9278 | 23.0772% | 4.6816 | 222 | 744.31 |

All four pass net CAGR>12% and matched-cost SPY, and all four also exceed the
20% return preference. That is **not** joint qualification: all drawdowns
fail, only five-year base passes Sharpe, and ten-year stress is below matched
QQQ by 0.0224083 percentage points. This strict failure is not rounded away.
Both stress paths pass the preserved extra base-cost-SPY gate.
The largest independent equity-accounting difference is
USD 1.0186340659856796e-10. Orders are historical model orders, not broker
orders; stress changes both costs and delay, not costs alone.

### Comparison with the actual saved monthly source

The correct control is `equity_first_month_end_commitment_v1`, not the
later option-exit latch. Its already saved
[four-path results](evidence/commitment_duration_20261027_results_v1.json)
use the same windows, protocol, capital and cost scenarios. A new bounded
comparison reads saved scalar outcomes only; no targets, accounts or metrics
are regenerated.

| Path | Source CAGR | New CAGR | CAGR change, percentage points | Additional model orders | Additional costs USD |
|---|---:|---:|---:|---:|---:|
| 10y base | 23.1015% | 22.5567% | -0.5448 | 102 | 97.85 |
| 10y stress | 21.6486% | 21.1015% | -0.5472 | 105 | 78.55 |
| 5y base | 27.3171% | 26.4066% | -0.9105 | 82 | 82.98 |
| 5y stress | 24.3694% | 23.2902% | -1.0792 | 75 | 68.44 |

Sharpe declines and annualized two-sided turnover rises in all four paths.
Drawdown worsens in three; five-year base improves by only 0.10192 percentage
points and still fails. Identical source-state decisions and non-core
targets do not imply identical executed non-core holdings: drift triggers,
pending requests, NAV and fees can differ. These are whole-model differences,
not causal asset-level factor PnL or independent alpha.
Private comparison SHA256:
`72426d90b826f4899b5960faa5498a17da4457f5dc84e0bf7ae844d6e04a59ad`.

The exact eight-candidate journal now has 94 events. SSD totals are eight
completed/reviewed, 32 paths, zero pending, zero technical failures and zero
qualifiers; combined completed configurations are 186. Original ledger,
eight benchmark accounts, source/catalog evidence and prospective models
remain unchanged. Diagnose saved risk before another economic question,
not a family-weight, clock or risk-cap sweep to rescue this configuration.

## Saved concentration, realized risk and construction prior art

A separate diagnostic completed at 2026-10-28 02:14:42.545687 UTC. It
examines the four existing family-core accounts and four existing monthly
source accounts, without generating targets, replaying accounts or
requalifying strategies. Its 147 input references, original SQLite and
94-event trial journal remain unchanged.

The accounting bridge infers adjusted model units from saved closing NAV
and weights. Previous units explain overnight PnL; post-open units explain
intraday PnL; saved fees are deducted separately. Daily cash, NAV change and
return components reconcile, including the initial cash-only USD10,000
anchor. The largest daily NAV error is USD3.1604940886609256e-11.
This is an accounting identity for modeled holdings, not a counterfactual
trading strategy or causal factor-alpha attribution.

| New account | Mean SOXX closing NAV weight | SOXX realized variance share | All-stock realized variance share | Stock-normalized holding HHI |
|---|---:|---:|---:|---:|
| 10y base | 40.87% | 64.16% | 93.31% | 0.2952 |
| 10y stress | 40.83% | 64.15% | 93.26% | 0.2951 |
| 5y base | 36.82% | 67.64% | 88.96% | 0.2931 |
| 5y stress | 36.76% | 67.48% | 88.77% | 0.2929 |

Variance shares are each realized additive return component's covariance
with the same net portfolio return, divided by that portfolio's variance.
They are signed ex-post statistics over each complete formal window, not
nominal weights, predictive covariance estimates or suggested new weights.
The new ten-year-base nominal HHI is lower than the source's 0.3147, but
stock risk remains dominant. Source/new saved net-return correlations are
0.9977--0.9980.

Stock-group variance share is **not** a single market factor's R-squared.
Separate in-sample excess-return regressions against the two fixed
benchmarks give new-account SPY R-squared of 26.25%--48.63% and QQQ
R-squared of 33.65%--50.26%. Beta is not capital leverage; residual series
are not tradable long-only accounts or independent validation.

### Different loss mechanisms must remain distinct

All four ten-year source/new base/stress drawdowns peak on 2020-02-19 and
bottom on 2020-03-20. Their first 15% breach is 2020-03-09, with an
equity intent and actual stock exposure. The new ten-year-base interval
loses USD5,927.99: SOXX gross PnL is -3,010.15 and the factor core -2,917.84,
with zero model orders and fees throughout that interval. Overnight gross
PnL is -4,910.69 and intraday gross PnL -1,017.30. This does not support a
fee-only explanation or a claim that SOXX alone caused the loss.

The new five-year base first breaches on 2022-05-10; stress first breaches
on 2022-04-27. Both have a gold intent and actual gold exposure near 98%,
zero stocks and no orders or fees on those days. They are not unexecuted
stock exits. The five-year peak is 2021-12-27, with a 2022-12-28 base trough
and 2023-03-13 stress trough. Earlier stock losses, subsequent defensive
asset moves and accumulated implementation costs have to be distinguished.
The original source's first five-year breach on 2022-02-23 was while in
stocks; it is not interchangeable with these later new-model events.

Identical target clocks do not imply identical execution. Source/new
ten-year-base rebalance sessions are 46/48, with only 27 shared dates.
The respective counts are 44/47 with 29 shared for ten-year stress, 28/32
with 16 shared for five-year base, and 28/31 with 20 shared for five-year
stress. Drift feedback and differing holdings matter even without changing
the declared band or execution delay.

Private diagnostic SHA256:
`04b71c57a5491acc6c10d4bde1e264f82536c20b149536a4580989ad9d6cbddc`.

### Existing construction research is not a new hypothesis by renaming

A saved-only inventory checked 28 candidates in six prior groups and their
original four-path results. Four recurring reviews also exactly match the
latest frozen original program snapshot. No account or goal was recomputed.

| Prior group | Existing methods | Preserved outcome and scope |
|---|---|---|
| [Allocation](evidence/allocation_round_20261007_results.json) | Equal, minimum variance, maximum diversification, rank and trend variants | Six tests; none passes the old joint primary goal. QLD exposure is not allowed in the current 1x lane. |
| [Implementation](evidence/implementation_round_20261007_results.json) | Quarterly/band controls and equal mixtures with covariance portfolios | Two of six pass old primary goals, none all old goals; embedded leverage prevents treating them as current 1x results. |
| [Refinement](evidence/portfolio_refinement_20261007_results.json) | QLD cap, covariance volatility limits and observed-weight bands | Six new controls; no qualified improvement under the original contract. A volatility target is not a drawdown guarantee. |
| [Unleveraged four-factor](evidence/multifactor_stability_20261010_results.json) | Balanced equity/bond/gold, breadth, inverse volatility and covariance cap | Two of six satisfy stability tests; none satisfies return/full goals. |
| [Equity/gold risk balance](evidence/factor_gold_risk_20261010_results.json) | Inverse 63-session volatility and daily 12% target variant | Both actual recurring reviews are rejected. |
| [Downside balance](evidence/downside_risk_balance_20261011_results.json) | Semideviation and empirical-tail-loss marginal risk budgets | Both reviews are rejected; marginal sleeve balancing is not joint portfolio expected-shortfall optimization. |

The [old unweighted factor-risk diagnostic](evidence/factor_risk_structure_20261011_results.json)
already recorded mean six-equity-fund correlations of 0.8253/0.7797 and
correlation participation dimensions of 1.3540/1.4735 for ten/five years.
It did not register the queued definitions, select weights or compute the
pending strategies. The two original W42 six-family allocation candidates
still do not occur in the preserved candidate/review registry.

Private prior-art review SHA256:
`2260ec62d87ce6b173b5c68663d6157c5df44ff873437dff360d4b6828052bcb`.
Old primary/SPY passes, legacy leverage and one-way-turnover fields are not
silently reinterpreted as the current dual-benchmark/1x/two-sided contract.

The fixed SOXX half-equity and equal-core proportions were experimental
controls for the completed eighth study, not permanent user requirements.
Evidence may motivate a separately preregistered different construction;
no new weights, window or ninth model has been selected. No global
ineffectiveness of size/buyback factors is inferred.

At the user's explicit request, the sole native continuation was changed
from 30 minutes to **one minute** on 2026-10-28 02:21:04.321 UTC and
read back successfully. Its complete prompt is byte-identical. The
single-active-job, no-replay/no-source-retry, original deadline and all
trading/data restrictions remain. The 30-minute market-data finalization
buffer and historical scheduling receipts were not changed.

## Joint portfolio tail loss with observed benchmark opportunity

The next fixed candidate is `benchmark_opportunity_joint_tail_v1`.
The [actual registration](evidence/joint_tail_construction_20261028_registration_v1.json)
occurred at 2026-10-28 03:56:56.804178 UTC, followed by its target-proof
event at 03:57:11.575373 UTC and packet at 03:57:11.576530 UTC.
No candidate account or performance metric existed at this fence.

The supplemental prior-art check explicitly includes the already tested
[unleveraged covariance constructions](evidence/risk_only_covariance_20261011_results.json):
`four_factor_covariance_risk_budget` and
`four_factor_covariance_max_diversification` were both rejected. Earlier
published and latest frozen metrics have tiny recorded numeric differences
(up to about 1.3e-13 in Sharpe), not bit-identical representations; both
versions and their unchanged rejection are preserved. The learned-allocation
policy also predates this work. This new model is neither a renamed
covariance-only objective nor a fitted return-ranking model.

At each actual completed exchange month end, use the latest 252 daily
adjusted-close return scenarios and 252-session observed total returns.
Treat the equal-six-family core as one linear scenario exposure, alongside
SOXX, GLD, IEF, TLT and actual BIL. No synthetic fund-price history or old
strategy target is created or replayed.

The observed opportunity score subtracts BIL's total return over the same
horizon. The hurdle is the greater SPY/QQQ score. If it is nonpositive, or
exceeds the analytical feasible maximum over the declared sleeve set, the
model explicitly chooses BIL. Otherwise the first linear program minimizes
empirical 5% expected shortfall of **joint portfolio** daily losses. The
second maximizes the same observed opportunity subject to the primary risk
optimum plus a frozen numerical slack. Missing data, infeasibility
inconsistent with the analytical bound, optimizer errors or failed
certificates stop the experiment; none silently becomes cash.

Weights are nonnegative and sum to one before the unchanged 98% investment
scale. The factor core is at least as large as the SOXX sleeve, a
model-specific thematic-concentration safeguard inherited without a ratio
grid, not a permanent user constraint or a fixed half-portfolio allocation.
Total equity and defensive allocations are jointly variable. All six
registered economic-family funds receive positive targets somewhere in
each complete formal window; source-blocked stock definitions do not count.

The registered target proof covers 122 monthly decisions: 75 optimized,
14 nonpositive-opportunity and 33 opportunity-unavailable observations.
Primary/secondary primal, dual, bound, stationarity and complementarity
certificates pass; a separate sorted-loss calculation retains fractional
tail mass. Independent scalar price/scenario reconstruction differs by
zero in returns and hurdles and at most 1.1102230246251565e-16 in scores.
Five prefix checks and both initial window anchors pass.

Historical momentum is not a forecast and a sampled tail loss is not a
maximum-drawdown guarantee. Solver tolerances do not relax any investment
gate. The same original funded simulator, 0.025 actual-holdings band, one
pending request, 5/20bp costs, USD1 tickets, one/two-session delays, RF,
initial USD10,000 peak and four complete windows remain. No borrowing,
shorting, new factor definition, benchmark replay or independent-forward
claim is introduced.

The study is selected after 186 exposed completed configurations.
The preregistration snapshot contained nine registrations, eight completed,
one pending, 32 paths and 186 combined completions.

### Completed joint-tail result: drawdowns pass, return goals fail

Actual preregistration publication
`4fc9ebe220203cf86e6b48ff963d84a6572a3631` was observed before the first
new account. All four paths completed at 2026-10-28 04:03:55.058480 UTC.
The [complete sanitized result](evidence/joint_tail_construction_20261028_results_v1.json)
is rejected.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 9.0968% | 0.7288 | 11.9830% | 5.5727 | 746 | 1,124.94 |
| 10y stress | 8.4278% | 0.6500 | 13.8192% | 5.5561 | 746 | 2,247.26 |
| 5y base | 11.2137% | 0.7971 | 12.0187% | 3.7283 | 339 | 448.68 |
| 5y stress | 9.8895% | 0.6471 | 13.8573% | 3.7162 | 339 | 760.48 |

Every path passes the original 15% drawdown ceiling. Every path fails net
CAGR>12%, Sharpe>1 and both matched-cost benchmarks; both stress paths also
fail the preserved base-cost-SPY comparison. The largest independent
equity-accounting difference is USD4.001776687800884e-11. There was no
borrowing, broker order or new market-data request.

The 122 optimizer decision records remain unchanged and are not executed
order counts. Their observed benchmark-opportunity constraints were not
future-return guarantees. Historical sample tail estimates did not create
independent alpha. Do not splice this model's drawdowns with the eighth
model's higher returns, lower the actual gates, or sweep the opportunity
hurdle, core bound or tail probability after this outcome.

The journal now has 106 events. SSD totals are nine completed/reviewed
hypotheses, 36 paths, zero pending, zero technical failures and zero
qualifiers; combined completed configurations are 187. All original ledger,
benchmark, source/catalog and prospective records remain unchanged.

## Saved joint-tail return, standby and implementation diagnosis

The next diagnostic reads only the completed ninth-model accounts,
registered targets/optimizer records and the original eight funded
benchmark accounts. It does not solve another LP, replay an account,
construct a zero-cost alternative, recompute primary CAGR/Sharpe gates or
select a new parameter.

Consecutive saved monthly decision dates, shifted by the original
base/stress delay, define diagnostic intervals. Actual account returns
are partitioned once across those intervals. Formal-window starts and
ends are explicitly clipped and marked: the ten-year window has 119
complete intervals and the five-year window 59. A decision-clock label
is not an assertion that its target executed exactly on that boundary;
overnight exposure, outstanding requests and actual holding drift remain.

| Path | Mean closing BIL | Mean closing stocks | Fixed commission share of cost | Mean model ticket notional USD | Complete optimized intervals beating QQQ |
|---|---:|---:|---:|---:|---:|
| 10y base | 53.74% | 28.18% | 66.31% | 1,015.93 | 27 / 72 |
| 10y stress | 53.72% | 28.23% | 33.20% | 1,006.20 | 29 / 72 |
| 5y base | 63.11% | 19.63% | 75.56% | 647.06 | 18 / 33 |
| 5y stress | 63.07% | 19.68% | 44.58% | 621.66 | 17 / 33 |

The recorded 122 monthly optimizer decisions are not 122 completed
future holding intervals in either funded window. Only complete
decision-clock intervals enter the last column. The recorded opportunity
score uses a preceding 252-session horizon, not the subsequent interval,
and the comparison counts are not independent forecast probabilities.

### The shortfall is not confined to unavailable-opportunity standby

The ten-year-base partition is additive in **net log-return units**:

| Saved decision label | Actual sessions | Model minus SPY log-return sum | Model minus QQQ log-return sum |
|---|---:|---:|---:|
| Joint-tail optimum | 1,536 | -0.3268 | -0.6649 |
| Nonpositive benchmark opportunity | 291 | -0.1035 | -0.2105 |
| Benchmark opportunity unavailable | 685 | -0.1366 | -0.1734 |
| Full formal window | 2,512 | -0.5670 | -1.0488 |

Both benchmark comparisons use their already saved same-window,
same-cost funded accounts. The label groups may contain disjoint dates:
their log sums reconcile to the full gap, but they are not reported as
tradable conditional-strategy returns or asset PnL attribution.

Even within optimizer-labeled intervals, mean BIL exposure is 25.60% in
ten-year base and 36.51% in five-year base. Optimized intervals have
negative total net log gaps to both benchmarks in all four paths.
This does not identify a unique causal fix or justify lowering the
opportunity hurdle after outcomes. The five-year-base nonpositive group
has a small positive SPY log gap but a negative QQQ gap; the two
benchmarks must not be conflated.

The full-window prior-close BIL intent versus non-BIL close-holdings
mismatch counts are 0/7/0/2 for ten-year base/stress and five-year
base/stress. Those are not the delay-shifted grouping labels. Holding
timestamps and actual execution must remain separate from intended
allocations.

Fixed tickets account for most base-case costs, especially with smaller
mean ticket notionals in the five-year account. Recorded fees are
reported, not retrospectively removed or compounded into an imaginary
fee-free NAV. The nonpositive-opportunity groups have no model orders
or fees in these saved accounts, yet they generally still trail QQQ.
Neither standby nor commission totals alone establish causal attribution.

Private diagnostic SHA256:
`357555f23bc6ad073e9743f1f7f60005c05145e2c0e9f33dc4ada242962c9730`.
Its 56 direct input references and all frozen study inputs remain
unchanged; the 106-event journal and original ledger are preserved.
Counts remain nine completed hypotheses, 36 paths, 187 combined
completions and zero qualifiers. No tenth model or new weights/window
has been selected.

## Growth-first construction with a benchmark-defined tail budget

One fixed reciprocal question is now registered:
`benchmark_tail_budget_growth_v1`. The
[actual preregistration packet](evidence/benchmark_tail_budget_20261028_registration_v1.json)
records registration at 2026-10-28 05:46:36.615636 UTC, target proof at
05:46:54.638470 UTC and packet creation at 05:46:54.639837 UTC.
No candidate account or realized performance metric existed at this fence.

The preceding diagnosis found substantial BIL exposure even in optimized
intervals of the risk-first rule. This version reverses economic priority:
maximize the same observed 252-session cash-relative opportunity score,
subject to a risk allowance derived from the smaller empirical 5% daily
tail loss of actual SPY/QQQ, floored at actual BIL tail loss. Then minimize
joint portfolio tail loss among primary opportunity optima.

The allowance uses only the completed decision-date observation window.
It is not the full-window drawdown of a benchmark and does not fit a risk
multiplier to previous outcomes. The unchanged core/SOXX/gold/IEF/TLT/BIL
scenario set, six actual core funds, 252-session lookback, 5% tail,
core-at-least-SOXX constraint, 98% invested target and execution law remain.
This is one priority/risk-anchor hypothesis, not a sequence of numerical
hurdle, tail, core or risk-budget changes to rescue the ninth result.

BIL is explicitly used when no positive observed opportunity exists, or
when a certified risk-feasible opportunity maximum is nonpositive. Its
own observed tail risk remains in the budget. Missing inputs, solver
failure or invalid certificates stop the study instead of becoming a
cash fallback. No sampled risk statistic guarantees the user's 15%
drawdown ceiling or future net benchmark outperformance.

The registered proof covers 122 completed monthly plans: 112 optimized
and ten nonpositive-opportunity standby plans. Both optimization stages'
primal/dual certificates and independently sorted fractional-tail
calculations pass. Independent sleeve/benchmark return differences are
zero and score differences at most 1.1102230246251565e-16. Five prefix
checks, both full-window initial anchors and actual positive participation
of all six registered core funds pass.

The original four funded paths, net Sharpe>1, CAGR>12%, drawdown<=15%,
matched-cost SPY/QQQ and extra stress-vs-base-SPY gates remain unchanged.
Past opportunity/risk observations are not independent validation.
The preregistration snapshot contained ten registrations, nine completed,
one pending, 36 paths and 187 combined completed configurations.

### Completed benchmark-tail-budget result

Actual publication `1074904478bfd0ba2a19d502c479ebd2de13c227` was verified
before candidate accounts. All four paths finished at
2026-10-28 05:54:28.534515 UTC. The
[complete sanitized outcome](evidence/benchmark_tail_budget_20261028_results_v1.json)
is rejected.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 12.9292% | 0.6781 | 26.4086% | 6.2167 | 744 | 1,187.36 |
| 10y stress | 11.4123% | 0.5922 | 27.0756% | 6.1955 | 737 | 2,380.56 |
| 5y base | 16.5627% | 0.7818 | 26.6875% | 6.4342 | 398 | 565.31 |
| 5y stress | 14.9118% | 0.6907 | 27.3277% | 6.4152 | 391 | 1,029.44 |

All four Sharpe, drawdown and QQQ gates fail. Net CAGR>12% passes except
ten-year stress; SPY comparisons pass only in the five-year window,
including its preserved stress-vs-base-cost-SPY gate. No parameter,
benchmark or numerical acceptance tolerance was changed after outcomes.
The largest independent NAV difference is USD1.014683221001178e-6,
below the unchanged `capital * 1e-8` verifier limit of USD0.0001;
recorded returns and independently recomputed scalar metrics also pass
their original checks.

The benchmark-derived historical tail allowance is not the user's
maximum-drawdown guarantee. Greater observed-opportunity participation
raises realised CAGR relative to the risk-first ninth study, but the
resulting drawdowns are 26.41%--27.33%, not 15%. Do not interpolate or
sweep an allowance multiplier, alter the 252/5%/core settings, or splice
the ninth model's drawdowns with this model's returns to fabricate a pass.

Totals are ten completed/reviewed SSD hypotheses, 40 funded paths and
188 combined completed configurations; zero pending, zero technical
failures and zero qualifiers. The 118-event journal, original ledger,
source evidence, benchmark accounts and prospective identities are
preserved. No eleventh model has been selected.

## Broader retained reference, not just the most recent weaker candidate

The saved scalar review compares the ninth and tenth constructions with
`semiconductor_option_proxy_dollar_observed_band025` from the existing
[implied-risk execution result](evidence/implied_execution_20261012_results.json).
The original policy and registered asset-leverage metadata confirm the
same full ten/five-year windows, fresh USD10,000, 5/20bp costs, one/two-session
execution delays, USD1 modeled tickets and 1x actual assets. No old account
or primary metric is recalculated.

| Existing dollar-guarded reference | Net CAGR | Sharpe | Max DD | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|
| 10y base | 14.722859% | 1.095380 | 13.572134% | 436 | 756.07 |
| 10y stress | 14.464138% | 1.073957 | 13.413710% | 442 | 1,710.97 |
| 5y base | 19.118758% | 1.168490 | 13.591322% | 264 | 407.13 |
| 5y stress | 18.577520% | 1.132536 | 13.433199% | 264 | 828.38 |

This reference is better than the tenth candidate on all five saved
dimensions in all four paths: higher CAGR/Sharpe and lower drawdown,
order count and total modeled cost. That counterexample is sufficient
to reject a claim of broader frontier improvement by the tenth study;
the three-candidate review is not an exhaustive frontier search.

The reference still falls below both SPY and QQQ in the ten-year window.
Its original rejected status and source evidence remain unchanged;
existing prospective identities are untouched. Better saved scalar results do not create a
new qualification or independent forward validation.

Same execution delays do not mean identical decision calendars or
information sets. The reference uses its original option/USD inputs and
monthly observed-weight execution; the new constructions use price-based
joint-tail optimization and their own feedback rule. This is not a
same-model causal ablation, a reason to force future work back into the
old architecture, or permission to scan failed parameters. Unavailable
turnover fields are not invented or substituted across definitions.

Private saved-review SHA256:
`ed0d1c9e146bb02ee6103147490e475872bead6869860e1598af6530192039cf`.
All three scalar panels, directional differences and unchanged old
benchmark comparisons are retained there. Counts remain ten completed
SSD hypotheses, 40 paths, 188 combined completions and zero qualifiers.

## Explicit daily variance-disagreement proxy

The [new registration](evidence/daily_variance_disagreement_20261028_registration_v1.json)
fixes `dollar_option_daily_variance_disagreement_v1`. Actual registration
at 2026-10-28 08:38:21.714785 UTC precedes target proof at
08:38:22.737987 UTC and packet creation at 08:38:22.739559 UTC.
No candidate account or realized return metric existed at this fence.

The [Federal Reserve working-paper abstract](https://www.federalreserve.gov/pubs/feds/2007/200711/200711abs.html)
for the variance-risk-premium research explicitly ties its empirical
quarterly-return result to high-frequency intraday realized variance,
not daily observations. The
[published paper identity](https://doi.org/10.1093/rfs/hhp008)
does not validate a daily SPY proxy or this ETF allocation. The source
review therefore rejected a **replication claim**, not every possible
separately specified daily-proxy experiment.

This single exploratory rule has its own hypothesis. At each original
month end, calculate annualized mean squared SPY daily log returns over
21 actual sessions, including that close. Multiply the source equity/gold
odds, GVZ/VIX, by VIX-squared divided by that backward-looking variance.
Equivalently, the new equity share is GVZ-times-VIX divided by
GVZ-times-VIX plus realized variance, with all volatility units converted
consistently and the original 30%--70% bounds retained. The unit tilt
coefficient and one-month window are fixed before targets and outcomes.
Zero/invalid variance or missing quotes stops the study, not a cash fallback.

Read the source's existing guarded target matrix instead of regenerating
the old pipeline. Preserve its exact 131 monthly dates, total non-BIL
allocation, BIL amount, original dollar half-risk decisions and relative
SOXX/four-style proportions. Only the division of that unchanged non-BIL
budget between equities and GLD changes. The four known nonmonthly GVZ
gaps remain unfilled; no intraday or full-daily source readiness is claimed.

Execution uses the original source's **monthly** observed-weight pure
0.025 band, not the newer daily-feedback allocator. Initial funding,
USD10,000, USD1 tickets, 5/20bp costs, one/two-session delays, fractional
1x cash funding, RF, initial drawdown peak and four complete windows remain.
Independent verification will check actual monthly holdings gaps,
issued targets, pending dates and funded-account parity.

The target proof retains all four genuine registered factor exposures
in both formal windows, original BIL/non-BIL scale and monthly dates.
The independent scalar target difference is at most
1.5432100042289676e-14; five prefix checks pass. No old strategy account
or historical target generator was rerun.

The proxy may have no predictive value and is not a true variance risk
premium, sector-specific implied variance, forecast guarantee or paper
replication. All joint net-CAGR/Sharpe/drawdown/SPY/QQQ gates and wider
reference comparisons remain. Counts at preregistration were eleven
registered, ten completed, one pending, 40 paths and 188 combined
completed configurations.

### Completed daily-proxy result: five-year passes, overall rejection

The actual preregistration publication
`352c21ae58bcfd0c5e32e617c958ba45c53a9593` preceded all four new funded
accounts. They completed at 2026-10-28 08:43:02.476900 UTC.
The [full sanitized result](evidence/daily_variance_disagreement_20261028_results_v1.json)
is rejected overall.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.8177% | 1.1696 | 15.6588% | 4.0982 | 617 | 1,050.08 |
| 10y stress | 16.3216% | 1.1295 | 14.6143% | 4.0904 | 603 | 2,293.53 |
| 5y base | 20.9426% | 1.2399 | 12.5061% | 4.4930 | 339 | 508.78 |
| 5y stress | 19.9205% | 1.1716 | 12.4731% | 4.4705 | 332 | 992.14 |

All four net-CAGR>12%, Sharpe>1 and matched-cost SPY gates pass, as do
both stress-vs-base-cost-SPY checks. Both five-year paths pass every
point gate. Both ten-year QQQ comparisons fail; ten-year-base drawdown
also fails. The 15.6588% drawdown is not rounded down, replaced by the
stress result or excused by the five-year performance.

Against the preserved stronger source, CAGR increases by approximately
2.095/1.857/1.824/1.343 percentage points and Sharpe is higher in all
four paths. Five-year drawdowns decline, but ten-year drawdowns rise
and total modeled costs increase throughout. The original source,
its prospective identities and its rejected status remain untouched.
This is an exposed historical model comparison, not proof of a true
variance risk premium or independent future alpha.

Independent verification checks 121 monthly-plus-initial decisions per
ten-year account and 61 per five-year account, including observed
holdings gaps, issued targets and original delayed fills. Issued request
counts are 92/90/50/49, distinct from the modeled order counts above.
The largest independent NAV difference is USD4.3655745685100555e-11.
Original accounting and metric tolerances are unchanged.

The journal now has 130 events. Totals are eleven completed/reviewed SSD
hypotheses, 44 funded paths, 189 combined completed configurations,
zero pending, zero technical failures and zero qualifiers. No twelfth
model or post-outcome coefficient/window/bound adjustment has been selected.

### Saved full-window failure diagnosis

This diagnostic uses only saved account/weight/decision/target files and
the original funded benchmark accounts. It partitions each actual ten-year
account at the fixed 2021-10-06 boundary; no earlier-period return is
inferred by dividing independently funded ten/five-year CAGRs.

| Actual account | Segment | Sessions | Net return | QQQ net return | Net log gap to QQQ |
|---|---|---:|---:|---:|---:|
| 10y base | 2016-10-06--2021-10-05 | 1,258 | 81.5884% | 212.6646% | -0.5434 |
| 10y base | 2021-10-06--2026-10-05 | 1,254 | 160.5584% | 118.0152% | +0.1783 |
| 10y stress | 2016-10-06--2021-10-05 | 1,258 | 79.9793% | 211.6710% | -0.5491 |
| 10y stress | 2021-10-06--2026-10-05 | 1,254 | 151.9359% | 118.0152% | +0.1446 |

The two gaps add to -0.365126 in ten-year base and -0.404496 in stress.
Later gains do not erase the earlier deficit. These are contiguous
segments of the same existing account and benchmarks, not new fresh
accounts, independently qualified subwindows or a replacement for the
original full-window gates.

Both ten-year maximum drawdowns run from the 2020-02-19 peak to the
2020-03-20 trough. Base reaches 15.6588% and stress 14.6143%; the latter
cannot replace the failed base path. The first base breach is 2020-03-19.
At the previous close, the latest known monthly target was 2020-02-28:
49% BIL, about 28.1052% stocks and 20.8948% gold. Actual prior-close
holdings were about 52.9929% BIL, 23.6961% stocks and 21.1525% gold.
There was no already issued pending request and no new model order or
fee on the breach date.

The base peak-to-trough NAV change is USD-2,284.92. Reconstructed adjusted
model-unit mark-to-market contributions are SOXX -1,059.45, four-style
core -983.06, gold -243.20 and BIL +14.35, less USD13.57 of actual fees
from seven orders in the full drawdown span. Overnight gross PnL is
-1,303.40 and intraday gross PnL -967.96. This is an accounting identity
on the actual executed holdings, not a zero-cost portfolio or causal
counterfactual exit.

The five-year maximum-drawdown episodes are different: both peak on
2026-01-29 and trough on 2026-03-30, with no model orders or fees in those
spans. Base NAV change is -2,837.26, comprising gold -1,185.30,
SOXX -1,132.12 and the style core -519.84. Stress is -2,744.52, comprising
gold -1,157.82, SOXX -1,076.65 and core -510.05. Both remain below 15%;
neither outcome creates a ten-year pass.

The maximum accounting-bridge NAV/cash differences across all four
paths are below USD1.4e-11. The initial USD10,000 peak, 56 direct input
references, all frozen study inputs, the original ledger and 130-event
journal are preserved. Private diagnostic SHA256:
`dc766169532c91551d65d9f93b7caa5d7cbb898f6d995fcf7dac199355a99f5e`.
No strategy, target, account or primary acceptance calculation was replayed.

## Symmetric stock/gold daily-disagreement information

The [twelfth registration](evidence/symmetric_variance_disagreement_20261028_registration_v1.json)
fixes `dollar_option_symmetric_variance_disagreement_v1`. Registration
at 2026-10-28 09:34:41.127535 UTC precedes proof at 09:34:42.277511 UTC
and packet creation at 09:34:42.278998 UTC. No account outcome existed
at this fence.

The eleventh rule used equity-side daily disagreement while treating gold
only through its implied quote. This extension adds GLD's backward
realized daily variance using the **same** 21-session window. Multiply
the original equity/gold odds by the equity implied/realized variance
ratio divided by the gold implied/realized ratio. The equivalent equity
fraction is VIX-times-RV_GLD divided by VIX-times-RV_GLD plus
GVZ-times-RV_SPY, in consistent decimal units and the original bounds.

When both disagreement ratios equal one, the original option allocation
is recovered. When RV_GLD equals GVZ squared, the eleventh one-sided
formula is recovered. These algebraic identities are not performance
promises. The experiment is a single information-set extension, not a
coefficient, lookback, boundary or cash-scale grid after outcomes.

Preserve the exact source BIL/non-BIL dollar scale, 131 monthly dates,
four-style/SOXX proportions, 30%--70% equity bounds, 98% target investment
and original monthly 0.025 observed-weight execution. Reuse the existing
parameterized monthly account adapter only for the new candidate's
targets; no old account is replayed. All costs, delays, RF, initial
capital peak and four-window joint qualification gates remain.

The proof checks both variances, all source invariants and four-family
coverage. Maximum independent scalar target difference is
2.1815882433884326e-14; five prefix checks pass. Missing or nonpositive
variance/quotes stop the study. Daily ratios are not true variance-risk
premia, exact implied portfolio risk, intraday-paper replication or
independent alpha.

Preregistration counts were twelve registered, eleven completed, one
pending, 44 funded paths and 189 combined completed configurations.

### Completed symmetric result: risk passes, benchmark failures remain

Actual preregistration publication
`631e0cd1b752feb88ff080796f07cb7c6c1d78da` preceded all four new accounts.
The [complete result](evidence/symmetric_variance_disagreement_20261028_results_v1.json)
finished at 2026-10-28 09:39:28.176934 UTC and is rejected overall.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 15.3379% | 1.1057 | 13.4553% | 4.4309 | 656 | 1,077.77 |
| 10y stress | 14.8105% | 1.0631 | 12.7279% | 4.4258 | 656 | 2,301.51 |
| 5y base | 19.9740% | 1.2248 | 12.5273% | 4.7045 | 344 | 514.99 |
| 5y stress | 19.0332% | 1.1609 | 12.5200% | 4.6951 | 344 | 1,012.46 |

Every path passes net CAGR>12%, Sharpe>1 and drawdown<=15%. Both five-year
paths pass all point gates. Both ten-year paths fail matched-cost SPY
and QQQ; ten-year stress also fails the preserved base-cost-SPY check.
The ten-year-base SPY gap is not rounded into a pass.

Relative to the eleventh model, CAGR is lower by about
1.480/1.511/0.969/0.887 percentage points and Sharpe is lower throughout.
Ten-year drawdowns improve, but five-year drawdowns increase slightly;
turnover, model order counts and total costs increase in all four paths.
This tradeoff is not uniform dominance or proof that symmetry improves
the broader research frontier.

The preserved monthly adapter independently checks 121/121/61/61
decision events and 98/98/51/51 issued requests. The maximum independent
NAV difference is USD5.093170329928398e-11. All original source, accounting,
timing and metric tolerances remain unchanged.

Counts are twelve completed/reviewed SSD hypotheses, 48 funded paths,
190 combined completed configurations, zero pending, zero technical
failures and zero qualifiers, with 142 journal events. No thirteenth
model, coefficient interpolation, window/bound scan or partial-pass
promotion has been selected.

### Saved symmetric-information assessment

The paired assessment reads both frozen monthly target matrices and the
twelfth recorded gold-disagreement ratios. On the 131 original month
ends, equity targets decrease on 88 dates, increase on 14 and are
unchanged on 29. The average change is -4.048703 percentage points of
total target capital. BIL targets and total target investment are exactly
preserved; these counts are not executed model orders.

The gold implied/backward-realized ratio is above one on 103 dates and
below one on 28. Target-change directions follow the fixed algebra;
unchanged targets can result from retained bounds. This is a check on
recorded calculations, not evidence that either ratio predicts returns.

Using the already saved accounts, ten-year-base mean closing stocks are
37.6904%, gold 32.6661% and BIL 27.6616%, versus approximately
41.9901%, 28.3833% and 27.6493% for the one-sided model. Identical BIL
targets do not force identical actual closing weights after price drift,
fees and different stock/gold holdings.

| Actual symmetric account | Fixed segment | Net log gap to QQQ | Change versus saved one-sided account |
|---|---|---:|---:|
| 10y base | 2016-10-06--2021-10-05 | -0.631526 | -0.088138 |
| 10y base | 2021-10-06--2026-10-05 | +0.138935 | -0.039327 |
| 10y stress | 2016-10-06--2021-10-05 | -0.643363 | -0.094256 |
| 10y stress | 2021-10-06--2026-10-05 | +0.108132 | -0.036478 |

Both fixed ten-year segments deteriorate relative to the one-sided
account. The existing eleventh diagnostic is read, not recomputed;
the fixed segments are not new fresh-capital accounts or a basis for
choosing a favorable subwindow.

The ten-year maximum-drawdown episode remains 2020-02-19--2020-03-20.
For ten-year base, actual stock mark-to-market losses are smaller, while
gold's loss is larger: SOXX -821.49, style core -780.93, gold -302.81
and BIL +14.16 USD, less 13.48 USD in fees, reconcile to NAV -1,904.55.
This saved-account identity does not certify a causal or future risk reduction.

Both symmetric five-year maximum drawdowns instead span
2022-03-08--2022-10-14, rather than the one-sided model's
2026-01-29--2026-03-30 episode. Base loses 1,345.72 USD: gold -958.36,
SOXX -301.25, core -81.62 and BIL +30.17, less 34.66 in fees from 28
orders. Stress loses 1,331.67 USD with 54.34 in fees and the same order
count. Neither path breaches 15%, but this is not the same risk event
as the one-sided model's similar headline drawdown.

Only frozen pure diagnostic functions are reused on the twelfth saved
account; no old diagnostic main, target generator or strategy account
is rerun. The 61 direct references, all frozen inputs, original ledger
and 142-event journal remain unchanged. Private assessment SHA256:
`479b0375309e09cdbae4efe83b21ff1a7bc49d3ac6150f08db385794939c9399`.
Counts remain twelve completed hypotheses, 48 paths, 190 combined
completions and zero qualifiers; no new parameter or model is selected.

## Nearest feasible execution point inside the unchanged band

The [thirteenth registration](evidence/band_projection_20261028_registration_v1.json)
fixes `daily_proxy_nearest_band_execution_v1`. Actual registration at
2026-10-28 10:41:17.151470 UTC precedes the source-only proof at
10:41:17.452347 UTC and packet at 10:41:17.453485 UTC.
No new path-dependent issued targets or accounts existed at this fence.

The desired source is the eleventh model's exact saved 131-month matrix.
Preserve its monthly calendar, initial fresh-center deployment, 98%
target investment, all original source evidence and the existing trigger:
maximum actual holding/desired gap plus 1e-12 reaches 0.025. Otherwise hold.
Only the destination of a triggered noninitial trade changes.

Project actual closing weights onto nonnegative weights summing to
0.98 within each source desired weight plus/minus 0.025. Assets with
both current and desired weights exactly zero remain zero. The unique
minimum-squared-weight-movement solution clips current weights minus
a common budget multiplier to those bounds. A fixed bisection solution,
convex optimality conditions and an independent piecewise-breakpoint
oracle must agree before a target can be issued.

This is **not a minimum-fee optimization**. A synthetic USD10,000
example has an algebraically consistent two-ticket cost of USD2.750316,
while the unchanged funding engine selects a three-ticket result costing
USD3.750431 for the same absolute target. This does not invalidate old
results; it prevents falsely pricing a new policy using a different
fixed-point selection. The engine is not modified and its actual fees
are always charged. Boundary execution may increase ticket frequency
or sacrifice return even when weight movement is smaller.

The source dollar/BIL scale is preserved in **desired** targets, not
claimed identical in executed projected weights. Actual destinations
can differ within the unchanged band, including relative factor shares
and BIL. Initial source-center deployment retains genuine factor
participation. Price gaps and delayed fills can move realized holdings
outside the band; it is not continuous drawdown insurance.

All USD1 tickets, 5/20bp costs, one/two-session delays, 1x fractional
cash funding, pending-request preservation, RF, initial capital peak
and four full-window gates remain. Synthetic projection, execution and
fee-contract checks precede registration; real holding-feedback
verification and independent accounts occur only after publication.

Preregistration counts were thirteen registered, twelve completed,
one pending, 48 funded paths and 190 combined completed configurations.

### Completed nearest-band result: lower costs, lower net returns

The actual preregistration publication
`0ac8889c004057a973665bd6ed9abd8c9b5b6ff8` preceded all new feedback
accounts. The [complete result](evidence/band_projection_20261028_results_v1.json)
finished at 2026-10-28 10:43:56.773479 UTC and is rejected overall.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.2845% | 1.1353 | 15.8479% | 3.4677 | 648 | 1,004.33 |
| 10y stress | 15.9323% | 1.1067 | 14.7694% | 3.4730 | 654 | 2,055.55 |
| 5y base | 20.3246% | 1.2136 | 12.3105% | 3.8264 | 327 | 469.49 |
| 5y stress | 19.6286% | 1.1682 | 12.7382% | 3.8219 | 327 | 886.59 |

All four CAGR>12%, Sharpe>1 and matched-cost SPY gates pass, including
both stress-vs-base-SPY checks. Both five-year paths pass every point gate.
Both ten-year QQQ comparisons fail, as does ten-year-base drawdown.
The source's historical outcome and its failed gates are not rewritten.

Against the eleventh center-rebalance source, fees decline by
USD45.75/237.98/39.28/105.55 and notional turnover declines throughout.
However, net CAGR declines by approximately
0.533/0.389/0.618/0.292 percentage points and Sharpe also declines
in every path. Ten-year orders increase from 617/603 to 648/654,
whereas five-year orders decrease from 339/332 to 327/327. Lower weight
movement or total fees does not imply fewer tickets or better net returns.

Actual monthly feedback, independent breakpoint projections and the
unchanged fee/funding engine are checked. Decision counts are
121/121/61/61, issued requests 93/94/47/47, and the maximum independent
projection difference is 1.1102230246251565e-16. The maximum independent
NAV difference is USD9.458744898438454e-11. No cost fixed-point choice,
threshold or metric tolerance was modified.

Counts are thirteen completed/reviewed SSD hypotheses, 52 funded paths,
191 combined completed configurations, zero pending, zero technical
failures and zero qualifiers, with 154 journal events. No fourteenth
model, bandwidth/norm scan, fee-free counterfactual or partial-pass
promotion has been selected.

### Saved execution accounting assessment

The assessment reads the actual source and projection account/holding
files and reconciles each to the unchanged price and fee records.
It does not replay either strategy or create a zero-fee account.

| Path | Total fee change USD | Fixed-ticket change USD | Proportional-cost change USD | Source mean ticket USD | Projection mean ticket USD |
|---|---:|---:|---:|---:|---:|
| 10y base | -45.75 | +31.00 | -76.75 | 1,403.83 | 1,099.79 |
| 10y stress | -237.98 | +51.00 | -288.98 | 1,401.77 | 1,071.52 |
| 5y base | -39.28 | -12.00 | -27.28 | 1,001.65 | 871.52 |
| 5y stress | -105.55 | -5.00 | -100.55 | 994.19 | 855.64 |

Ten-year-base fixed commissions represent 64.5204% of projection fees,
versus 58.7573% for the source. Lower notional turnover did not lower
the number of ten-year tickets. These are modeled orders and original
USD1 charges, not brokerage activity or a revised fee schedule.

For return attribution, divide each actual asset mark-to-market and
fee by that account's own previous NAV. Their daily differences sum
to the observed daily return difference. An ex-post logarithmic linking
coefficient then makes the components sum exactly to the actual paired
log-return/ending-wealth difference.

| Path | Actual net log-return difference | SOXX linked contribution | Gold linked contribution | Actual-fee linked contribution | Ending NAV difference USD |
|---|---:|---:|---:|---:|---:|
| 10y base | -0.045743 | -0.036397 | -0.009921 | -0.000163 | -2,115.55 |
| 10y stress | -0.033518 | -0.038616 | -0.007722 | +0.008256 | -1,494.61 |
| 5y base | -0.025615 | -0.023491 | -0.004936 | +0.002042 | -654.30 |
| 5y stress | -0.012183 | -0.020336 | -0.001237 | +0.006618 | -300.28 |

The complete private decomposition also includes the four-style core
and BIL. SOXX is the largest negative linked holding component on each
path. Dollar fee totals alone do not determine the sign of the fee-return
effect: ten-year base pays fewer total dollars but has a slightly negative
linked fee contribution because actual timing and prior NAV differ.
Three paths have positive linked fee contributions, insufficient to
offset the adverse holding contributions.

Linked values are log-return accounting contributions, **not CAGR
percentage-point changes**, asset standalone returns, causal alpha or
an investable fee-free counterfactual. The paired daily arithmetic
identity error is below 7.1e-16 and the linked sum error below 5e-15.
All 52 direct references, frozen source/study inputs, original ledger
and 154-event journal remain unchanged. Private assessment SHA256:
`eb1d84ddacdb934c41adfa0758a5af23ea39cd42933a1432f9378225aff47ad4`.
Counts remain thirteen completed hypotheses, 52 paths, 191 combined
completions and zero qualifiers. No new parameter or model is selected.

## Continuous semiconductor/core relative-wealth allocation

The [fourteenth registration](evidence/relative_equity_odds_20261028_registration_v1.json)
fixes `daily_proxy_relative_equity_odds_v1`. Actual registration at
2026-10-28 11:07:49.984648 UTC precedes proof at 11:07:53.000217 UTC and
packet creation at 11:07:53.001499 UTC. No account outcome existed at
this preregistration fence.

The source's total equity, GLD, BIL, other targets and monthly execution
remain unchanged. Only the division of equity between SOXX and the
four-style core changes. Source half/half composition is a model choice,
not a permanent user constraint.

At each eligible source month end, use prices at D-126 and D-21: the
inherited 126-observation lookback with a 21-session skip contains 105
actual returns. SOXX gross wealth is its endpoint price ratio. Core
gross wealth is the arithmetic mean of the four fund ratios, representing
equal **starting** capital, not a synthetic daily-rebalanced traded index.
The semiconductor share is SOXX gross wealth divided by SOXX plus core
gross wealth; the remaining stock budget stays equally in four styles.
Equal relative wealth recovers the original half/half composition.

This differs from old binary SOXX/SPY switching inside a fixed satellite
and old four-style ranking inside a fixed core. There is no fitted
temperature, satellite fraction/cap, subset, risk-budget or observation
window grid. Positive finite wealth factors keep actual style exposures,
but do not guarantee comparable risk, alpha or future outperformance.

The first three source month ends, 2015-11 through 2016-01, lack the
full inherited history and remain wholly undefined. They are genuine
prehistory warmup, not filled neutral signals or retroactively removed
losses. There are 128 eligible dates from 2016-02-29 through 2026-09-30.
Both original formal anchors are causally seeded; all 2,512/1,254
formal returns remain. Unknown source gaps are fatal.

The independent proof compounds individual daily ratios rather than
reusing the endpoint formula, with maximum target difference
4.440892098500626e-16. Five prefix checks, unchanged equity/defense budgets
and all four factor exposures in every formal desired state pass.
The old source generator or account is not replayed.

Original monthly observed-weight execution, USD10,000 fresh capital,
1x cash funding, USD1 tickets, 5/20bp costs, one/two-session delays, RF,
initial drawdown peak and all joint four-window gates remain. Counts
at preregistration were fourteen registered, thirteen completed, one
pending, 52 paths and 191 combined completed configurations.

### Completed relative-equity result: modest gain, full gates still fail

The actual preregistration publication
`d141f9caba55f8b1bd96a538e6b8ff260dc73eb4` preceded all four new accounts.
The [complete result](evidence/relative_equity_odds_20261028_results_v1.json)
finished at 2026-10-28 11:11:18.366772 UTC and is rejected overall.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 17.0480% | 1.1710 | 15.6817% | 4.1236 | 630 | 1,069.66 |
| 10y stress | 16.5771% | 1.1341 | 14.6562% | 4.1202 | 622 | 2,340.20 |
| 5y base | 21.1781% | 1.2393 | 12.6506% | 4.5198 | 338 | 509.03 |
| 5y stress | 20.1904% | 1.1761 | 12.6187% | 4.4971 | 338 | 1,003.19 |

All net-CAGR>12%, Sharpe>1 and SPY comparisons pass, including the
stress-vs-base-SPY checks. Both five-year paths pass every point gate.
Both ten-year QQQ comparisons and ten-year-base drawdown fail.
No subwindow, source drawdown or stress result substitutes for those failures.

Versus the source's fixed half-sector/half-core composition, net CAGR
increases by about 0.230/0.256/0.235/0.270 percentage points.
Drawdowns, turnover and fees increase in every path. Sharpe increases
in three paths but decreases slightly in five-year base; this is not
uniform risk-adjusted dominance or proof of persistent leadership.

The three undefined prehistory months stay undefined; all original
2,512/1,254 formal returns and causal seeds remain. Independent
monthly feedback checks 121/121/61/61 decisions and 94/93/50/50 issued
requests. The maximum independent NAV difference is
USD5.820766091346741e-11. No old account, risk threshold, source hash,
fee/timing rule or numerical acceptance tolerance was changed.

Counts are fourteen completed/reviewed SSD hypotheses, 56 funded paths,
192 combined completed configurations, zero pending, zero technical
failures and zero qualifiers, with 166 journal events. No fifteenth
model or post-outcome share/window adjustment has been selected.

### Saved relative-equity composition assessment

The assessment uses the 128 actual eligible target dates, not neutral
fills for the three insufficient prehistory dates. The source's total
equity, gold, BIL and unused targets remain unchanged on every matched
date. SOXX's share **within equity** is above half on 95 dates and below
half on 33, with no equal-half dates. Its range is 43.7753%--60.9286%
and mean 51.5813%; the mean SOXX shift in **total target capital** is
+0.780784 percentage points. These are source-target statistics, not
trade counts or equal-risk assertions.

The actual average closing SOXX weight rises by approximately
0.7622/0.7125/0.5664/0.4263 percentage points across ten-year base/stress
and five-year base/stress. Unchanged desired equity/defense totals do
not force identical actual closing weights after composition-dependent
returns, fees and observed-weight execution.

Frozen pure accounting functions are reused, with their new-account
output roles explicitly renamed for this composition study. No old
diagnostic main, target generator or portfolio is replayed.

| Path | Actual net log-return difference | SOXX linked contribution | Core linked contribution | Actual-fee linked contribution | Ending NAV difference USD |
|---|---:|---:|---:|---:|---:|
| 10y base | +0.019694 | +0.035736 | -0.014693 | -0.000876 | +941.07 |
| 10y stress | +0.021938 | +0.037867 | -0.014599 | -0.001801 | +1,005.75 |
| 5y base | +0.009724 | +0.016173 | -0.005629 | +0.000088 | +252.81 |
| 5y stress | +0.011243 | +0.016619 | -0.004857 | -0.000800 | +280.36 |

The complete private accounting includes gold and BIL and sums exactly
to the observed paired ending-wealth ratio. SOXX is the largest positive
linked holding component; the four-style core contributes negatively
to the paired difference. Total dollar fees increase throughout, but
five-year-base NAV-normalized fee contribution is slightly positive
because actual wealth and timing differ. Dollar-fee signs cannot simply
be copied into return attribution.

These are ex-post log-return accounting contributions, not CAGR
percentage-point changes, causal alpha, standalone asset returns or
zero-fee/equal-wealth counterfactuals. Daily arithmetic identity error
is below 7.5e-16 and linked sum error below 3.5e-15.
All 56 direct references, frozen inputs, original ledger and 166-event
journal remain unchanged. Private assessment SHA256:
`b9d1046562e362eeac04dd72b8ecc9911383281ed952eeb7cdfc76c598bddab1`.
Counts remain fourteen completed hypotheses, 56 paths, 192 combined
completions and zero qualifiers. No new parameter or model is selected.

## Separately frozen two-segment information study

[`config/split-session-variation-v1.json`](config/split-session-variation-v1.json)
asks whether an already admitted daily open adds useful price-path
information beyond the eleventh study's close-only squared returns.
It does not tune the failed composition shares, risk bounds, window,
coefficient or holding band.

Each of the same 21 completed sessions contributes two squared log
movements: previous adjusted close to adjusted open, and that open to
that session's adjusted close. Sum the two within each session, average
over 21 sessions and annualize by 252, **not 504**. The close-only
measure equals this measure plus twice the mean product of the two
movements, with the same annualization. A cancellation can hide
variation in the daily net return; persistent same-direction moves
can make the two-segment measure lower. This is not a guaranteed
conservative risk estimate.

Retain the eleventh unit-coefficient option-odds rule, original
30%--70% stock fraction bounds, half-SOXX/equal-four-style stock
composition, saved original dollar guard, BIL amount, 98% target
investment and every original monthly decision. This study does not
continue the fourteenth within-equity share rule. All original
USD10,000 accounts, funding, delayed opens, monthly 0.025 observed
holding band and four-path dual-benchmark gates remain.

The admitted panel contains 2,805 complete price sessions and all
131 source months have sufficient prior observations. Opens are
vendor-adjusted model prices under the existing raw-open times
adjusted-close/raw-close convention. Cross-day distribution
adjustments and weekend gaps follow that convention; these are not
authenticated auction prints or standalone overnight-trading returns.
Two coarse intervals are not high-frequency realized variance, true
variance premium or an imported literature forecast. A range-estimator
literature search returned a transport HTTP 504 and supplied no
verified paper or formula; its actual failure is retained privately.
No Yang-Zhang/Rogers-Satchell/high-low construction is claimed.

The [actual packet](evidence/split_session_variation_20261028_registration_v1.json)
records registration at 2026-10-28 11:45:21.052080 UTC, before target
proof at 11:45:22.071005 UTC and packet creation at 11:45:22.072293 UTC.
All 131 targets retain the source decision lattice. Independent
ratio-log reconstruction differs by at most 1.15e-14 in weight; the
squared-return identity differs by at most 1.39e-16. Five prefixes,
both full-window anchors, four actual registered economic families
and 24 synthetic/lifecycle cases are preserved in the evidence.
These are not candidate account metrics.

At this registration there are fifteen SSD registrations, fourteen
completions, one pending hypothesis, 56 completed funded paths,
192 combined completions and zero qualifiers. A real sanitized
preregistration publication must precede all new accounts; no old
account, target preparation or diagnostic main is replayed.

### Completed two-segment result

Actual preregistration commit
`cc4fc052bfaebace695f1f49668549d41b3c6fc1` was observed on the remote at
2026-10-28 11:48:07 UTC. All four fresh accounts completed once at
11:52:02.802874 UTC. The [full scalar result](evidence/split_session_variation_20261028_results_v1.json)
is **rejected overall**: every CAGR, Sharpe, drawdown, matched-cost SPY
and applicable stress-versus-base-SPY gate passes, but both ten-year
QQQ comparisons fail. Both five-year paths pass their point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.9172% | 1.1803 | 14.8656% | 4.1159 | 623 | 1,059.49 |
| 10y stress | 16.3809% | 1.1374 | 13.8994% | 4.1174 | 630 | 2,327.60 |
| 5y base | 20.9599% | 1.2426 | 12.5061% | 4.4246 | 324 | 492.23 |
| 5y stress | 20.0204% | 1.1802 | 12.4731% | 4.4057 | 324 | 979.41 |

The independent funded-account discrepancy is at most USD6.55e-11.
All 121/121/61/61 monthly-plus-initial feedback decisions were checked;
issued requests number 93/94/48/48. They are model requests and
623/630/324/324 model tickets, not broker activity. All 2,512/1,254
formal return sessions remain, without terminal liquidation or changed
cash interest, risk-free series, fees, initial peak or fill delay.

Saved comparisons use four retained references and all sixteen matching
path pairs, not just the immediately preceding candidate:

| Reference | Return/risk comparison | Implementation boundary |
|---|---|---|
| Eleventh close-only proxy | CAGR gains +0.0996/+0.0594/+0.0173/+0.0999 percentage points; Sharpe rises throughout; ten-year DD falls, five-year DD unchanged within numerical precision | Ten-year turnover, tickets and costs rise; five-year turnover, tickets and costs fall |
| Twelfth symmetric proxy | CAGR/Sharpe rise throughout; ten-year DD is higher, five-year DD is lower | Costs are lower in three paths, but higher in ten-year stress |
| Fourteenth relative-equity composition | CAGR is lower throughout; Sharpe rises and DD falls throughout | Turnover and costs fall throughout, but ten-year-stress ticket count rises |
| Older observed-band dollar reference | CAGR/Sharpe rise throughout; ten-year DD is higher, five-year DD lower | All four ticket counts and dollar costs are higher; no dominance claim |

Five-year DD differences against the eleventh are only about 1.11e-16
in decimal units, not an economically meaningful improvement.
That descriptive equality tolerance does not relax any acceptance
threshold. The actual ten-year QQQ CAGR deficits remain 4.2452 and
4.7429 percentage points. Neither a newly passing drawdown gate nor
two five-year CAGRs above 20% can substitute for those missing gates.

The comparisons read existing scalar files only; different information
and construction choices are not an isolated fee experiment, causal
alpha or independent forward validation. Private comparison SHA256:
`ddd4fd6664908009b79322231fcc4d41fcf87b8fefd669ff70feeb2e2f83c811`.
There are now fifteen completed SSD hypotheses, 60 funded paths,
193 combined completions and zero qualifiers. The SSD journal has
178 events; this is distinct from the original ledger's unchanged
178 completed configurations. No sixteenth model is selected.

### Saved two-segment information and holding assessment

All 131 paired monthly targets and saved audits remain aligned.
The split measure is higher on 64 dates and lower on 67, with no
equal-measure dates. Stock targets increase on 45 dates, decrease on
41 and are unchanged on 45 despite different measured variation.
The original clipping bounds limit how much new information reaches
the allocation. The average stock shift is **-0.0693 percentage
points of total target capital**, ranging from -9.3692 to +7.7432
percentage points. These are descriptive monthly targets, not
orders, predictive effects or a uniform reduction in exposure.

The saved source BIL/non-BIL budgets and relative stock composition
are unchanged. Actual daily-average stock weights change by about
-0.1471/-0.2826/-0.2283/-0.3419 percentage points across ten-year
base/stress and five-year base/stress. Monthly target averages and
formal-window daily holding averages use different clocks and
coverage; they are not interchangeable risk measures.

Ex-post linked accounting compares each actual account using its
own prior NAV. The complete private reconciliation includes BIL:

| Path | Paired net log-return difference | Gold component | SOXX component | Four-style core component | Actual-fee component |
|---|---:|---:|---:|---:|---:|
| 10y base | +0.008518 | +0.024844 | -0.015064 | -0.000201 | -0.000944 |
| 10y stress | +0.005101 | +0.026992 | -0.016519 | -0.002091 | -0.003232 |
| 5y base | +0.000715 | +0.011190 | -0.009697 | -0.002257 | +0.001623 |
| 5y stress | +0.004164 | +0.015178 | -0.009712 | -0.002661 | +0.001455 |

Gold is the largest positive holding component of this paired change,
partly offset by semiconductor and core components. These log-accounting
contributions are not CAGR percentage points, causal alpha, standalone
asset returns or a fee-free/equal-wealth counterfactual.

Both ten-year models' maximum drawdowns still run from 2020-02-19
through 2020-03-20. Each decline contains seven model orders in both
accounts. Base fees are USD13.56656 for the source and USD13.56603 for
the new model; the improvement is not a reduction in order count or
a switch to a different maximum-drawdown episode. The new base account
has no 15% breach, unlike the source's 2020-03-19 breach.

Both five-year maximum-drawdown episodes still run from 2026-01-29
through 2026-03-30. Neither account trades or pays fees in those
41-session declines. Maximum paired differences are below 1.12e-16
in closing weights, 4.45e-16 in daily returns and 3.34e-16 in
peak-normalized NAV. This explains the numerically unchanged maximum
drawdown: it does not mean equal dollar wealth, identical full
histories or independent validation.

The actual ten-year accounts retain the same fixed chronological split:

| Path | 2016-10-06--2021-10-05 QQQ log gap | 2021-10-06--2026-10-05 QQQ log gap | Full-window QQQ log gap |
|---|---:|---:|---:|
| 10y base | -0.533557 | +0.176949 | -0.356608 |
| 10y stress | -0.546663 | +0.147267 | -0.399395 |

The earlier shortfall outweighs the later gain. These are additive
segments of the **same actual ten-year account**, not CAGRs, new
fresh-capital accounts or a splice with the separate five-year result.
Neither segment is an untouched test set.

The assessment preserves all 85 direct references, frozen inputs,
178 SSD journal events and the original ledger. Nine synthetic cases
cover information identities, source-budget corruption and normalized
episode comparisons; old diagnostic mains and account/target generators
are not called. Private assessment SHA256:
`42df21cdccccbd4fcbb367c4c1afd4a76ec87c18dac3c21f0641abe3bb12edd4`.
Counts remain fifteen completed hypotheses, 60 paths, 193 combined
completions and zero qualifiers. No new model or parameter is selected.

## Universal-portfolio theorem and execution boundary

A distinct construction review checks Thomas M. Cover's **Universal
Portfolios**, *Mathematical Finance* 1(1), 1--29 (1991),
[DOI 10.1111/j.1467-9965.1991.tb00002.x](https://doi.org/10.1111/j.1467-9965.1991.tb00002.x).
[Crossref metadata](https://api.crossref.org/works/10.1111/j.1467-9965.1991.tb00002.x)
and the first page of the
[author-hosted scan](https://isl.stanford.edu/~cover/papers/paper93.pdf)
agree on its identity. The PDF has 29 image-only pages; local OCR and
reading were limited to pages 1--3. OCR mathematical glyphs are not
treated as exact formula transcription, and no full-paper or proof
read is claimed. An author-index URL returned HTTP 404; this is retained,
not interpreted as absence of the successfully retrieved paper.

The inspected model uses long-only period weights, gross price
relatives and performance weighting over constant-rebalanced portfolios.
Its asymptotic growth-rate comparator is the hindsight best constant
rebalanced portfolio, not a finite ten/five-year net benchmark victory.
Page 3 also mentions a finite-sample lower bound: this review does
**not** deny that result or claim to have verified its formula or proof.
A lower bound relative to a comparator is not an unconditional promise
to beat that comparator at every finite horizon.

Three predetermined synthetic checks clarify the application boundary:

| Synthetic check | Actual result | Meaning |
|---|---|---|
| Initial uniform weights; gross relatives 2 and 1 | Wealth factor 1.5 versus best constant portfolio 2 | Finite underperformance is compatible with the asymptotic comparator |
| Initial uniform weights; gross relatives 1 and 0.5 | Wealth factor 0.75; drawdown 25% | The growth comparator does not impose a 15% drawdown limit |
| One funded 98% target purchase at 5bp; USD1 fixed ticket | USD20,000 account ends at 19,989.205289; two USD10,000 accounts total 19,988.205779 | Fixed tickets break degree-one wealth scaling; their difference is USD0.99951024 |

For the same funding calculation with proportional costs only, the
scale difference is zero. This uses the **unchanged** pure funded
rebalance helper, not a new portfolio or a fee-free substitute for
qualification. USD20,000 is only a scale-test input; all formal
research accounts retain fresh USD10,000.

The inspected paper equations do not encode this repository's fixed
USD1 tickets, 5/20bp costs, 1/2-session delayed opens and monthly
actual-holding no-trade band. A gross mathematical wealth mixture is
not automatically the wealth of one executable funded account.
No Sharpe, drawdown, finite-window net CAGR/QQQ guarantee or new
economic-factor definition is imported.

This boundary does not prove that cost-aware universal methods are
impossible. Any future adaptation must first freeze its expert space
and prior, information/decision clocks, matured observations, numerical
approximation, funding and cost aggregation. It must then satisfy
the unchanged actual registration, publication and four-path gates.
No expert grid, weights, sixteenth model or historical return outcome
was selected by this review.

All 20 direct references, the 178-event SSD journal and original ledger
remain unchanged; the three checks use only synthetic inputs.
Private review SHA256:
`3d0f934e158f2b136ee98bdd99b89599e3ea4b03472e3fc8f81e7093650b28f4`.
Counts remain fifteen completed SSD hypotheses, 60 funded paths,
193 combined completions and zero qualifiers.

## Separately frozen continuous-simplex wealth learner

[`config/simplex-wealth-average-v1.json`](config/simplex-wealth-average-v1.json)
freezes one new construction after 193 exposed completed configurations.
It averages the complete three-sleeve simplex under a uniform
Dirichlet(1,1,1) prior, weighted by products of **already observed**
monthly gross relative scores. The sleeves are the four-style core,
SOXX and GLD. Core information uses the arithmetic mean of four fund
gross relatives from equal **starting monthly capital**, not a
daily-rebalanced index or a synthetic asset traded without tickets.

A normalized Dirichlet-moment recursion integrates this polynomial
exactly in algebra, evaluated in floating point. Its finite monomial
states are not sampled expert portfolios or a parameter grid. There
is no fitted prior, forgetting rate, temperature, lookback or historical
best weight. Independent Gauss-Legendre simplex integration uses the
degree of the observed polynomial to verify targets, never to choose
allocation parameters.

The actual first completed month end, 2015-08-31, is only an anchor.
The first learning interval ends 2015-09-30; no return is invented
for the initial partial month. At each original donor decision, only
completed intervals through that close are available, followed by
the unchanged one/two-session fill delay. Learning does not reset
when the independent five-year funded account starts.

The original saved dollar guard's BIL and total non-BIL budgets stay
fixed. The learned fractions distribute non-BIL capital among the
three sleeves, with the core divided equally among its four actual
funds. The old half-SOXX and 30%--70% stock/gold fraction bounds are
**explicitly replaced model choices**, not permanent user constraints
or modifications to old studies. Long-only exposure, full funding,
98% target investment and every user hard gate remain unchanged.

This is a gross-score learning signal coupled to **one** actual
net funded account. It is not a net wealth average of separately
capitalized experts, a transaction-cost universal-portfolio theorem,
predictive-likelihood DMA or a Sharpe/drawdown guarantee. Fees are
charged by the original monthly observed 0.025-band executor, not
silently omitted because the information score is gross.

The [actual packet](evidence/simplex_wealth_average_20261028_registration_v1.json)
records registration at 2026-10-28 12:48:33.777062 UTC, before target
proof at 12:48:37.998866 UTC and packet creation at 12:48:38.000534 UTC.
All 131 original decisions remain. The first and last targets use
3 and 133 completed learning intervals; maximum independent weight
difference is 3.34e-15. Five prefixes, both original formal anchors
and positive participation by all four registered factor families
pass. Twenty-four synthetic/lifecycle cases include an independent
140-period integration check and no-replay/publication/expiry guards.
These are not candidate account results.

At registration there are sixteen SSD hypotheses registered, fifteen
completed, one pending, 60 completed paths, 193 combined completions
and zero qualifiers. Actual sanitized publication must precede all
four fresh accounts. No old target generator, account or source
acquisition is replayed.

### Completed continuous-simplex result

Actual preregistration commit
`ec300d591caaf56395b7456d02dec5f5dc1fd405` was observed on the remote
at 2026-10-28 12:52:22 UTC. All four fresh funded accounts completed
once at 12:56:38.898423 UTC. The
[complete scalar result](evidence/simplex_wealth_average_20261028_results_v1.json)
is **rejected overall**: both ten-year drawdown and QQQ gates fail.
All CAGR, Sharpe, matched-cost SPY and applicable extra stress-SPY
gates pass; both five-year paths pass all point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 17.8953% | 1.0907 | 17.0942% | 2.9409 | 241 | 587.16 |
| 10y stress | 18.0751% | 1.1013 | 16.2357% | 2.9684 | 276 | 1,686.77 |
| 5y base | 21.4270% | 1.0969 | 14.6796% | 3.5702 | 158 | 294.14 |
| 5y stress | 21.3807% | 1.0977 | 14.8357% | 3.6005 | 179 | 726.59 |

The independent funded NAV discrepancy is at most USD7.28e-11.
All 121/121/61/61 monthly-plus-initial feedback decisions were checked;
issued requests number 35/40/23/26. The 241/276/158/179 tickets are
modeled orders, not broker activity. All 2,512/1,254 formal returns,
initial USD10,000 peaks, original RF, costs, delays and cash-funding
rules remain. The 2% idle-cash amount remains a target, not a
continuously enforced dollar floor.

Saved comparisons cover the eleventh, fourteenth, fifteenth and
older observed-band dollar reference, with all sixteen path pairs
retained and no account replay. Against the fifteenth, CAGR rises
by +0.9781/+1.6942/+0.4670/+1.3604 percentage points, but maximum
drawdown rises by +2.2286/+2.3363/+2.1735/+2.3626 percentage points
and Sharpe falls throughout. Turnover, tickets and costs fall in
all four paths. The implementation improvement does not substitute
for the failed risk and benchmark gates.

The broader older reference also has higher model-ticket counts and
fees throughout, while retaining lower drawdowns throughout and
better Sharpe in three of four paths. The eleventh and fourteenth
likewise retain better Sharpe and lower drawdown throughout.
This construction does not dominate the broader research frontier.
Its higher ten-year stress return cannot be attributed to higher fees:
the original stress delay and resulting holdings/feedback path differ.
No equal-risk or isolated cost counterfactual is claimed.

The frozen three-sleeve gross learning score is not a funded net
wealth mixture or a guarantee imported from the paper. Neither the
finite algebraic state count nor the independent quadrature is a
separately screened strategy grid. No forgetting rate, prior,
temperature, stock bound or source guard is retuned after outcomes.

Private saved-comparison SHA256:
`6edcadeb2ea690ce7ac2c109f8f4937b50ee55ffe0e1c35dcbeb780d9c2503fe`.
Counts are sixteen completed SSD hypotheses, 64 funded paths,
194 combined completions and zero qualifiers. The SSD journal now
has 190 events; the original ledger and prospective identities remain
unchanged. No seventeenth model is selected.

### Saved simplex learning and holding assessment

The assessment reads all 131 saved learning/target rows; it does not
rerun moment updates, quadrature or any account. All original
BIL/non-BIL budgets are preserved. Learned fractions are measured
**within non-BIL capital**, not total account NAV:

| Sleeve | First fraction | Mean fraction | Last fraction |
|---|---:|---:|---:|
| Four-style core | 33.5130% | 31.3692% | 28.1566% |
| SOXX | 33.9439% | 38.4893% | 44.2051% |
| GLD | 32.5431% | 30.1415% | 27.6383% |

Relative to the fifteenth model, total stock targets are higher on
98 dates and lower on 33. Mean changes in **total target capital**
are +7.1282 percentage points in stocks, comprising +6.0895 in SOXX
and +1.0387 in the four-style core, versus -7.1282 in gold.
These are not a change to the donor dollar guard or executed-order counts.
The consecutive monthly target L1 movement sum falls from 45.283863
to 32.847971; that intention statistic is **not executed turnover**.

Actual daily-average stock weights rise by
7.7360/7.7360/8.6879/8.5899 percentage points across ten-year base/stress
and five-year base/stress. Actual SOXX weights rise by
6.6911/6.6745/7.9527/7.8865 points. This is a material exposure shift,
not simply fewer commissions on the same risk.

Ex-post linked accounting uses each actual account's own prior NAV;
the complete private reconciliation also includes BIL:

| Path | Paired net log-return difference | SOXX component | Gold component | Four-style core component | Actual-fee component |
|---|---:|---:|---:|---:|---:|
| 10y base | +0.083299 | +0.215287 | -0.161392 | -0.000377 | +0.029876 |
| 10y stress | +0.144501 | +0.245520 | -0.154172 | +0.006653 | +0.046449 |
| 5y base | +0.019265 | +0.116000 | -0.099747 | -0.011377 | +0.014402 |
| 5y stress | +0.056345 | +0.141458 | -0.096574 | -0.007718 | +0.019013 |

The positive semiconductor and fee components are partly offset by
gold and most core components. These are log-accounting contributions,
not CAGR percentage points, causal alpha, standalone factor returns
or a fee-free/equal-risk counterfactual.

Both ten-year maximum drawdowns remain 2020-02-19--2020-03-20.
Both first exceed 15% on 2020-03-16. At the previous close, BIL already
represents 51.1567%/51.7007% of NAV, under the existing February 28
target with BIL 49%. Neither breach day has an already-issued pending
request, new model order or fee. The dollar guard had already acted;
its bill allocation is not a guaranteed cumulative drawdown floor.

Both five-year maxima instead move to **2022-03-29--2022-10-14**,
compared with the fifteenth model's 2026-01-29--2026-03-30 episode.
The new intervals contain 14/28 model tickets and USD19.57787/52.27759
fees. Both models' observed maximum intervals are retained in the
private paired analysis; comparing only a shared favorable episode
would miss this risk change.

The same actual ten-year accounts retain the fixed chronological split:

| Path | 2016-10-06--2021-10-05 QQQ log gap | 2021-10-06--2026-10-05 QQQ log gap | Full-window QQQ log gap |
|---|---:|---:|---:|
| 10y base | -0.463190 | +0.189881 | -0.273309 |
| 10y stress | -0.451767 | +0.196873 | -0.254894 |

The earlier shortfall still outweighs later gains. These additive
segments do not splice independent five-year accounts or constitute
untouched holdouts. All 85 direct references, frozen inputs, the
190-event SSD journal and original ledger are unchanged. Eight
synthetic cases cover fraction/budget/clock validation and corruption.
Private assessment SHA256:
`fa5a1646d1d213d20cd08049b328f053702e32a8ad079ac78f7fa35327f77d48`.
Counts remain sixteen completed SSD hypotheses, 64 paths,
194 combined completions and zero qualifiers. No new parameter or
seventeenth model is selected.

## Credit permission boundary and separate ANFCI source review

The proposed Baa--Aaa quality-spread question is not a reuse of the
old BAA10Y, which also contains the corporate/Treasury maturity
mismatch, or a replacement for restricted ICE high-yield OAS.
The primary [DAAA](https://fred.stlouisfed.org/series/DAAA) and
[DBAA](https://fred.stlouisfed.org/series/DBAA) pages identify daily,
not-seasonally-adjusted percent yields on bonds with maturities of
20 years and above. Their rating-basket difference is not automatically
a pure default-risk or duration-matched factor.

Both pages contain explicit proprietary copying/reuse restrictions.
No applicable new source-use permission was verified; historical
observation acquisition is therefore deferred. This is a permission
boundary, **not an HTTP refusal** or a claim that the earlier frozen
BAA10Y study had no permission. The old data, results and caveats
remain unchanged. No Moody's observations or restricted high-yield
series were acquired, reconstructed or obtained through another provider.

A separate question concerns the
[Chicago Fed Adjusted National Financial Conditions Index, ANFCI](https://fred.stlouisfed.org/series/ANFCI).
It is a weekly, Friday-ending, not-seasonally-adjusted **index**.
The source describes a financial-conditions component adjusted for
economic conditions; it is not the proposed credit-quality spread
or proof of independence from portfolio returns.
The [Chicago Fed's primary explanation](https://www.chicagofed.org/research/data/nfci/current-data)
describes 105 financial indicators, estimated weights and historical
revisions. ANFCI additionally responds to economic activity and inflation.
A current historical curve cannot stand in for past as-of observations.

The [public ALFRED form](https://alfred.stlouisfed.org/series/downloaddata?seid=ANFCI)
and [download help](https://alfred.stlouisfed.org/help/downloaddata)
were inspected without submitting a historical download. The actual
form advertises POST, plain index units `lin`, output type `2`
("Observations by Vintage Date, All Observations"), zipped CSV and
a 500-character entered-date limit. Its first listed vintage is
2011-05-25. This predates the research windows but does **not** prove
that every required vintage or fresh observation is available.

The saved prospective query plan reuses only the existing market
calendar: 131 original monthly decisions plus their 63-session
reference queries yield **225 distinct prior-session vintage dates**,
from 2015-08-28 through 2026-09-29. Six proposed batches contain at
most 40 dates each and satisfy the advertised text limit. These
are source-download batches, not strategy configurations or a restored
six-trial quota. The proposed weekly observation range ends
2026-10-02; the original ETF return cutoff remains 2026-10-05.

Before acquisition, the source contract and validation must be frozen
separately. Normal advertised-form requests, raw archive provenance,
README/column vintage identities, actual observation availability
and the proposed unchanged 14-calendar-day freshness limit still
need verification. Daily vintage cutoffs are not exact historical
intraday publication timestamps. Unknown gaps, stale observations
or failed requests cannot be replaced with current data, relaxed
ages or silent cash states.

**At that metadata-review stage, no ANFCI vintage archive had been
downloaded, no admission had passed, and no trading rule had been selected.**
The metadata and query plan are not strategy trials, performance
results or independent forward evidence. All 16 direct references, the 190-event SSD journal
and original ledger remain unchanged. Private source-review SHA256:
`7cc18b3faedf6b31cfadac9687c1912abccea0b4bbb96898a384cea45645a6e5`.
Counts remain sixteen completed SSD hypotheses, 64 funded paths,
194 combined completions and zero qualifiers.

### Actual ANFCI archives: complete coverage, failed freshness

The [actual source-readiness evidence](evidence/anfci_input_readiness_20261028_v2.json)
now covers all six archives and 225 requested own-vintage columns.
Every unique data POST returned HTTP 200. Archive identity, requested
vintage metadata, column identities and all 84,117 finite cells were
checked by independent parsers with zero numeric difference.
Raw observations and network artifacts remain private.

The unchanged full-query freshness contract is **not admitted**:

| Query session | Prior-session vintage | Latest actual observation | Age, calendar days |
|---|---|---|---:|
| 2022-12-29 | 2022-12-28 | 2022-12-09 | 20 |
| 2022-12-30 | 2022-12-29 | 2022-12-09 | 21 |

Only 223 of 225 queries meet the frozen 14-day limit. Complete
archive/vintage coverage is not complete fresh coverage. No age
limit is raised, missing observation filled, date removed or stale
reading converted into a safe trading state. The cause of the
historical stale snapshots is not inferred from holidays or current data.

The acquisition history preserves two distinct technical issues:

- The preferred owned-browser preflight timed out and its control
  channel retried navigation. Retained traffic shows five normal
  form GETs returning HTTP 200 and **zero data POSTs**, not a provider
  refusal. The owned browser was closed. Before any archive request,
  the collector was frozen as an anonymous, no-retry client of the
  same advertised public form, without custom UA, proxy, TLS or
  credential changes.
- The first actual archive returned HTTP 200 after the v1 source
  contract was frozen at 2026-10-28 16:13:05 UTC. The dot-only parser
  rejected native **empty CSV fields**. Its failed contract, code,
  raw response and failure receipt remain unchanged. The retained
  first archive contains 2,049 empty fields and 5,591 finite cells,
  with no finite future observations.

The separate
[`anfci-native-csv-v2.json`](config/anfci-native-csv-v2.json)
correction recognizes only empty and dot fields as absent. It does
not insert observations, change finite values or relax source,
calendar, vintage or freshness rules. Its code was frozen at
16:32:25 UTC after seven correction tests, before the five remaining
requests. The first request at 16:18:51 UTC predates that technical
correction and is explicitly reused, not redownloaded. Remaining
requests began at 16:33:34 UTC; the sixth unique POST began at
16:33:56 UTC. There were **six unique data POSTs total**, not seven.
The original client/parser contract also retains its 19 offline checks.

The v2 result completed at 2026-10-28 16:33:59 UTC with
`source_admitted: false`. Its native-format correction is not a
freshness repair. The original NFCI, BAA10Y, historical strategy
results and prospective identities remain unchanged. No source
collection or completed validation should be repeated.

Any future ANFCI use would need a separate explicit information and
missingness contract before outcomes, without claiming this
full-freshness contract passed. No such trading rule or seventeenth
candidate is selected. All source failures and raw archives remain
on the SSD; the original ledger and 190-event research journal are
unchanged. Counts remain sixteen completed SSD hypotheses, 64 funded
paths, 194 combined completions and zero qualifiers.

## Separately frozen fresh-only residual-stress decision policy

[`config/anfci-residual-confirmation-v1.json`](config/anfci-residual-confirmation-v1.json)
defines one new hypothesis after 194 exposed completed configurations.
The [primary Chicago Fed Letter 386](https://www.chicagofed.org/publications/chicago-fed-letter/2017/386)
explains positive ANFCI as financial conditions tighter than typically
suggested by economic activity and inflation. This supplies the
fixed zero boundary, not a threshold fitted to strategy outcomes.
The article also documents a 2017 change from two-step to simultaneous
estimation, additional macroeconomic adjustments and a longer history.
Actual own-vintage values are kept; today's method is not rebuilt
backward into earlier snapshots. Exact historical method-switch
timing and full methodology continuity are not authenticated.

The new policy reads the **saved fifteenth target matrix**, not its
generator. Its original BIL 0/49% identifies the previously verified
published-dollar 63-session guard. At each original month end, with
a fresh current ANFCI reading, retain the half-risk allocation only
when that dollar guard is active **and ANFCI is strictly above zero**.
Otherwise restore its non-BIL composition proportionally to 98%
and set BIL to zero. A donor already at full risk is unchanged.
Stock/gold proportions and half-SOXX/equal-four-style stock composition
are preserved. This can increase risk; no net return or drawdown
improvement is presumed.

The original **225-query full-freshness admission remains false**,
including both stale queries and every raw archive. This distinct
trading-information requirement uses current ANFCI levels, not
ANFCI 63-session changes. It does not silently omit an unfavorable
return date or claim that all source readings became fresh.

Before any strategy outcome, the following operational rule is fixed:
a current reading older than 14 calendar days produces an explicit
source-abstention record and **no new target/request**. It is never
interpreted as zero or a safe macro regime, even if the dollar
condition would otherwise be false. Actual holdings and idle cash
continue; previously queued fills are not canceled. All returns,
fees and mark-to-market dates remain. Unknown hashes, malformed
records, missing query rows or budget errors stop the study instead.
Both initial anchors must have an eligible latest scheduled target.

This follows the earlier repository's explicitly registered source-pause
pattern, but does not replay the older NFCI strategy or source audit.
The new ANFCI residual-stress channel and fifteenth price budget make
it a separate whole-model comparison, not isolated causal factor alpha.
No prior, temperature, forgetting rate, stock bound, source-age
limit, dollar window, reduction scale or trade band is swept.

The [actual packet](evidence/anfci_residual_confirmation_20261028_registration_v1.json)
records registration at 2026-10-28 17:37:56.489261 UTC, target proof
at 17:37:57.412628 UTC and packet creation at 17:37:57.414164 UTC.
All **131 monthly opportunities** are retained, with 130 fresh
targets and one generic-age-rule abstention on **2022-12-30**.
The independent target difference is at most 1.12e-16, with five
prefix checks, both original anchors and four positive registered
factor families. These are source/target proofs, not account results.

The unchanged funded monthly executor is paired with a separately
versioned independent verifier that checks all opportunities,
eligible decisions and abstentions. The old verifier's every-month
assumption is **not modified or weakened**. Twenty-five offline
cases include both original delays, retained return days and
rejection of a hidden request or removed date at an abstention.

Seventeen SSD hypotheses are registered, sixteen completed, one
pending, 64 funded paths, 194 combined completions and zero qualifiers.
Actual sanitized publication still must precede four fresh USD10,000
accounts under all original timing, costs, RF, initial-peak and
dual-benchmark gates. There is no trading or deployment authority.

### Completed residual-stress confirmation result

Actual preregistration commit
`94ff2e2d40e58e8194db06e936722524650b8d08` was verified on the remote
at 2026-10-28 17:40:22 UTC. All four fresh funded paths completed
once at 17:43:06.697811 UTC. The
[complete scalar result](evidence/anfci_residual_confirmation_20261028_results_v1.json)
is **rejected overall**.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 19.1908% | 1.039127 | 22.2677% | 2.6650 | 553 | 812.46 |
| 10y stress | 18.5577% | 0.999926 | 22.4891% | 2.6543 | 541 | 1,545.42 |
| 5y base | 24.5144% | 1.125942 | 19.5913% | 2.7219 | 287 | 377.14 |
| 5y stress | 23.3055% | 1.067553 | 19.9082% | 2.7153 | 287 | 636.35 |

All net CAGR, matched-cost SPY and applicable extra stress-SPY gates
pass. Both ten-year QQQ gates fail, and **all four drawdowns exceed
15%**. Ten-year stress Sharpe is **0.9999262771386388**, so its strict
greater-than-one gate fails; rounding to 1.0000 is not a pass.
The two five-year paths beat QQQ but fail drawdown. No path passes
all point gates and no metrics are spliced across configurations.

Independent funded NAV difference is at most USD4.37e-11.
The new verifier checks 121/121/61/61 monthly-plus-initial
opportunities, 120/120/60/60 eligible actual decisions and one
source abstention in each path. Issued requests are 90/88/47/47.
All four 2022-12-30 abstention records have no new request, zero
same-day model orders/fees and an actual retained return observation.
Every 2,512/1,254 formal return, original RF, initial USD10,000
peak, delay and cash-funding rule remains. The source's original
225-query full-freshness admission is still false.

Saved comparisons cover the fifteenth donor, sixteenth learner,
eleventh daily proxy and older observed-band dollar reference.
Against the fifteenth, CAGR rises by
2.2735/2.1768/3.5544/3.2852 percentage points, but maximum drawdown
rises by 7.4021/8.5897/7.0852/7.4351 points and Sharpe falls on
every path. Fees, model tickets and annualized two-sided turnover
are lower throughout. These are whole-model historical tradeoffs,
not causal ANFCI alpha or proof that lower costs caused the gain.

Against the sixteenth, turnover is lower throughout but model
tickets increase by 312/265/129/108. Base-path costs are higher,
while stress-path costs are lower. The broader older reference
retains higher Sharpe and lower drawdown on every path; no uniform
research-frontier dominance is claimed.

Private saved-comparison SHA256:
`b46a33df61443f11e8ad46025ea161e126ec3ad60d4bcc46009fb6125d336640`.
All sixteen scalar pairs and actual source-abstention proofs are
retained without replaying prior accounts, targets or source audits.
The 202-event SSD journal and original ledger are preserved.
Counts are seventeen completed SSD hypotheses, 68 funded paths,
195 combined completions and zero qualifiers. No threshold,
source age, dollar window, reduction scale, trade band or calendar
is tuned after these outcomes; no eighteenth candidate is selected.

### Saved residual guard and holding assessment

The diagnosis preserves all 131 source opportunities and the one
unavailable target. Among the **130 eligible paired targets**, the
new policy retains 11 original half-risk guards, releases 63, and
leaves 56 originally full-risk states unchanged. No target is
invented for 2022-12-30 and no account return is omitted.
Non-BIL relative composition remains unchanged.

Mean changes in **total target capital on eligible dates** are
+23.7462 percentage points in non-BIL assets, comprising +13.9667
in stocks and +9.7794 in gold, versus -23.7462 in BIL. SOXX alone
accounts for +6.9834 points of the stock change. These intention
statistics are not executed turnover or ticket counts.

Actual daily-average stock weights rise
14.1736/14.1800/15.1847/15.1421 percentage points across ten-year
base/stress and five-year base/stress. Gold weights rise
9.8949/9.9038/11.5225/11.5763 points, while BIL weights fall
24.0541/24.0705/26.6911/26.7034 points. The return increase therefore
accompanies materially more stock **and** gold exposure, not just
fewer trading costs.

The following exact ex-post log linkage uses each account's own
prior NAV. It is not CAGR percentage points, causal alpha,
standalone asset returns or an equal-risk/fee-free counterfactual:

| Path | Paired net log-return difference | Gold | SOXX | Four-style core | BIL | Actual-fee component |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | +0.192564 | +0.171611 | +0.050252 | +0.019026 | -0.060520 | +0.012194 |
| 10y stress | +0.185290 | +0.192473 | +0.007250 | +0.009937 | -0.060049 | +0.035679 |
| 5y base | +0.144789 | +0.116380 | +0.057643 | +0.010318 | -0.045776 | +0.006224 |
| 5y stress | +0.135000 | +0.131501 | +0.022891 | +0.007248 | -0.045285 | +0.018645 |

Gold is the largest positive linked component in all four paths;
the BIL component is negative throughout. This accounting result
does not identify a causal forecast advantage.

Both ten-year first 15% breaches occur on **2020-03-12**, after the
recorded 2020-02-28 guard release. Both five-year first breaches
follow the recorded 2022-08-31 release: 2022-09-15 for base and
2022-09-06 for stress. In every case the prior close and latest target
have **BIL zero**, and the breach day has no pending request, model
order or fee. Lack of positive ANFCI confirmation is not a bound
on subsequent asset losses.

Both ten-year maximum drawdowns remain
2020-02-19--2020-03-20. The five-year maxima move from the donor's
2026-01-29--2026-03-30 episode to
**2021-12-27--2022-10-14** (base) and
**2021-12-27--2022-09-26** (stress). Both models' observed maximum
intervals remain in the private paired assessment. All these new
maximum episodes end **before** the 2022-12-30 source abstention;
that later absence cannot cause the earlier observed losses.

The original fixed split of each actual ten-year account shows:

| Path | Early QQQ log gap, through 2021-10-05 | Later QQQ log gap, from 2021-10-06 | Full-window QQQ log gap |
|---|---:|---:|---:|
| 10y base | -0.485633 | +0.321589 | -0.164044 |
| 10y stress | -0.503414 | +0.289308 | -0.214106 |

The early deficit still exceeds the later gain. These segments
are not independent five-year accounts or untouched holdouts.

The first diagnostic attempt stopped on a metadata spelling mismatch:
the producer uses `stale_numeric_reading_used`, while the reader
expected a plural alias. Its script, intent and failed log remain
unchanged. A separate v2 uses a strict in-memory alias with no
source/value changes and seven regression cases built from the
real producer's **synthetic** audit schema. The failed main was not
replayed, and no real signal or account was regenerated.

All 96 direct references, frozen inputs, the 202-event SSD journal
and original ledger remain unchanged. Private v2 assessment SHA256:
`48107bab2fd7a8e2701a9cef4940aa1ce6d2076e266e7b8e30cd0cbb10d011d1`.
Counts remain seventeen completed SSD hypotheses, 68 paths,
195 combined completions and zero qualifiers. No new parameter,
source admission or eighteenth model is selected.

## Independent IBKR price-history source boundary

The prepared Gateway is authorized only for existing-entitlement market data,
not account synchronization or orders. `ib_async` 2.1.0's high-level
`IB.connectAsync(readonly=True)` still requests positions; the existing paper
handshake also reads account identities. Neither path is used here.

`us_quant.ibkr_market_history.MarketHistory` is a separate, single-use,
loopback-only adapter. Its transport permits only the protocol handshake,
start-API message 71, instrument identification 9, and bounded historical-price
request 20. The initial source probe is restricted to SPY/QQQ, USD/SMART, daily
three-month `TRADES` or `ADJUSTED_LAST`, regular hours and no live subscription.
It cannot request snapshots, accounts, positions, executions, orders or order
cancellation. Every actual send rechecks the accepted SSD runtime and deadline.

Unsolicited managed-account frames are discarded before SDK decoding, logging
or source capture. Readiness requires the actual negotiated protocol and
request-ID acknowledgement, not dummy account identifiers or a fabricated SDK
account-ready flag. It proves only the market-request handshake, not data
entitlement or a paper/live account type. Provider errors retain numeric codes,
not potentially identifying text, and fail the single attempt without retries.
Malformed messages, source-persistence failures, disconnects and timeouts cannot
be converted into empty successful datasets.

The isolated synthetic-transport suite has 68 passing cases, including
fragmented/coalesced private startup frames, forbidden outgoing operations,
timeouts, deadline enforcement and output failures. This offline evidence is
not proof that any real Gateway request or historical-data admission succeeded.
Each actual probe still needs its own source-bound receipts and permitted raw
responses, retained privately on SSD.

[IBKR's historical-bar documentation](https://interactivebrokers.github.io/tws-api/historical_bars.html)
distinguishes split-adjusted `TRADES` from split-and-dividend-adjusted
`ADJUSTED_LAST`. A current-time request can include an unfinished daily bar;
calendar, completed-session cutoff, actual coverage and adjustment behavior
must be checked separately before admission. Do not silently substitute
unadjusted prices, double-count dividends, replace registered inputs, infer
ten-year coverage from a short probe, or backfill missed forward observations.

The one actual probe stopped on unknown status 2157 before a historical
request; no history was admitted. A bounded documentation review obtained
the complete warning-code section of the
[official reference](https://interactivebrokers.github.io/tws-api/message_codes.html):
it includes 2158 but not 2157. A newer official entry failed at the transport
layer, without HTTP content. Neither observation proves a subscription denial,
document nonexistence or the absence of a definition in all official material.
The existing unknown-status shutdown remains; no whitelist change or
automatic reconnection followed.
