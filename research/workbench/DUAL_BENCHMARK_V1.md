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
No new candidate account has been evaluated.

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
