# Isolated research workbench

This directory imports the tested `us-quant-research` workbench into the canonical
`us_stock_qr` repository without replacing its existing `src.us_quant` package,
Top30 stock research, early-opportunity pipeline, or user-maintained holdings.

## Runtime boundary

The workbench requires Python 3.10 or newer. Create a **separate virtual
environment in this directory**, and invoke its interpreter explicitly. Do not
install it into the incumbent application's Python 3.9 environment or add this
source path globally.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[paper,dev,research]' -c requirements-research.lock
.venv/bin/python -m pytest
.venv/bin/python -m us_quant --help
```

The source and tests are copied with their original file bytes and provenance
hashes. The package's console entry point is also named `us-quant`; always use
this workbench's explicit `.venv/bin/us-quant`, not an ambiguous global command.

## Preserved evidence, not current recommendations

The snapshot covers 52 registered configurations. None passed the simultaneous
ten-year/five-year criteria in the source research: net CAGR above 20%,
excess-return Sharpe above 1, maximum drawdown at most 15%, and higher CAGR than
the same-period SPY benchmark.

The common research endpoint is 2026-10-05. The original ten-year window starts
2016-10-06; the five-year window starts 2021-10-06. The research capital is
$10,000 and returns are before personal taxes. Different methods disclose their
own modeled transaction costs. These values cannot be silently combined with
the incumbent project's $20,000 Top30 backtest, different cutoff, stock
universe, cost model, or Sharpe definition.

`evidence/research_status.json` is the small integration-facing research status
record. `evidence/all_52_summary.json` retains the per-configuration results.
Historical rejection, overlapping evaluation windows, incomplete stressed
paths, zero-cost diagnostics, and lack of prospective evidence remain visible.
An historical test result is not a current actionable signal.

## Deliberately excluded

No real/paper account configuration, positions, order/fill ledger, environment
file, webhook, access token, local service configuration, virtual environment,
raw market-data cache, user handoff holdings, or private broker log is imported.

Raw market data and original evolving runtime ledgers remain local to the old
workspace. Tests use synthetic local fixtures; running historical studies
requires separately obtaining appropriately licensed data and preserving the
declared provenance. Do not claim an old snapshot's fingerprint matches newly
downloaded data.

The paper adapter source is retained for its tested identity, order, and
idempotence checks, **not as authority to submit orders**. `paper.example.json`
is a nonworking placeholder with submission disabled. The canonical daily
reporter should consume research status, never call a paper submission command.

## Integration and continuing research

`us_stock_qr` is the authoritative code, documentation, and progress repository
from this integration onward. Keep strategy identities separate:

- The incumbent Top30 momentum/inverse-volatility strategy remains its own
  research lane, with its own universe and historical limitations.
- Early-opportunity ideas and news/Reddit evidence remain a distinct product;
  missing point-in-time event history must not inherit another strategy's
  performance.
- The imported research workbench provides explicit multi-window comparisons,
  calendar/data checks, independent accounting, and preserved failed trials.
- Actual user holdings and manual fills remain private facts, not automatically
  inferred from suggested orders or old screenshots.

Daily Slack should identify the strategy and distinguish source-data cutoff,
signal cutoff, and report-generation time. Stale critical inputs must result in
a status/data-gap message, not refreshed-looking buy/sell advice. Improvement
work should start with data freshness, comparable metrics, and independent
validation before expanding parameter searches or changing trading authority.

Existing source-forward records were paused for continuity/data failures.
This import does not backfill those observations or implicitly restart any old
automations.
