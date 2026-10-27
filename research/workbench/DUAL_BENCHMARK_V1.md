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
