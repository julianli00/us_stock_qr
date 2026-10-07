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

The published second-round registration can be used directly. Run from
`research/workbench` using its isolated environment:

```bash
.venv/bin/python -I -B -m pytest tests/test_factor_research.py -q
.venv/bin/python -I -B -m us_quant.factor_research stage-data \
  --policy config/allocation-research.json \
  --registration evidence/allocation_round_20261007_registration.json \
  --source-base /path/to/verified/dual-horizon/market \
  --source-supplement /path/to/verified/dual-horizon/legacy-supplement
.venv/bin/python -I -B -m us_quant.factor_research evaluate \
  --policy config/allocation-research.json \
  --registration evidence/allocation_round_20261007_registration.json \
  --output reports/allocation-round-20261007
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
reproduction uses commit `3656e098`; changing its recorded hash to accept a
later implementation is not a valid reproduction. New experiments use the
`register` stage with a new policy and a new registration path.

The [first-round evidence](research/workbench/evidence/factor_round_20261007_results.json)
contains all eight configurations. The growth/gold inverse-volatility control
passes the latest two primary gates in the base ten/five-year windows, but
its ten-year stressed Sharpe is 0.9913 and drawdown is approximately 25%.
It is **not** a base-plus-stress qualified result and fails the old drawdown
goal. The first non-overlapping five-year diagnostic also fails; no forward
performance or statistically established alpha is claimed.

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
