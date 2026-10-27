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
