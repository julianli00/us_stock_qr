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
No candidate account or performance metric exists at this fence.

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
Current counts are nine SSD registrations, eight completed, one pending,
32 paths, zero qualifiers and 186 combined completed configurations.

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
