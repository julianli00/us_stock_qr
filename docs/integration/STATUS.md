# Canonical integration status

Updated: 2026-10-06 UTC. Canonical repository: `julianli00/us_stock_qr`.

Integration is on the review branch, not merged into `main`:
[canonical integration PR](https://github.com/julianli00/us_stock_qr/pull/1).

## Verified first checkpoint

- Imported the byte-preserved 70-file workbench under `research/workbench`.
  Import manifest SHA256:
  `3b2ef6789348ac88b8fdf82805415f518da85bb72c1d652735e7aeaaa63666b6`.
- Preserved the incumbent Python 3.9 package/environment. Independently ran
  all 357 imported tests in a separate Python 3.12 environment with package
  resolution checked against the imported source.
- Added the pure research report, strict watch proof contract, NYSE calendar,
  locked/atomic session outbox and credential-free preview.
- The first checkpoint passed 30 deterministic reporting guards, including multi-process delivery
  races, interrupted/ambiguous sends and private-ledger nonreads.
- All existing Slack send/preview entrypoints use the canonical report before
  the legacy mixed-date payload/hash. Force cannot restore stale trading text.

## Final code and rollout checkpoint

- The expanded target suite passes **50 tests**: 43 reporting/scheduler/privacy/
  compatibility cases and seven incumbent-source cases. The imported suite
  separately passes all 357 tests.
- The Top30 algorithm is now a pure published module, not just documentation.
  Its four functions are AST-identical to the original; five synthetic complete
  weight matrices are exactly equal to the original pure reference functions.
  The 699-stock configuration, original cost engine and distinct early
  scanner are byte-preserved. No real-data search or account pipeline was run.
- The 30-minute close buffer is an **earliest** complete-data boundary.
  Decisions may occur after it, including 17:30 ET, but must be before the
  next session open and no later than report generation. Explicit early,
  late and future-decision cases pass.
- A directly coupled Discord artifact generator retains its original selected
  ticker metadata through an explicit compatibility builder; all Slack send/
  preview routes still enter the canonical gate. No Discord sender, holdings
  ledger or private importer was run or modified.
- Explicit publication checks cover staged paths, all tracked paths and
  preserved import hashes. CI runs reporting and workbench tests separately.

## Evidence is not a current trading signal

Top30 is `user_nolev_top30_m126_invvol_m`: a static 699-stock universe,
126-day momentum, top30, inverse 63-day volatility and monthly selection.
Risk filters use QQQ/SPY150MA, VIX and market drawdown; defensive allocation is
BIL. The original algorithm uses the last available monthly row, including an
incomplete final month, and permits missing VIX. No retuning is authorized by
the reporting job.

The saved 2015-01-02 through 2026-05-08 result is CAGR 29.5432%, Sharpe
1.1867 and maximum drawdown 28.3109%. It fails a 15% drawdown cap, was chosen
using the full sample, and lacks the imported research's matching-rule
independent 10-year/5-year evaluation. Do not merge its performance with the
early-watch product or imported ETF/ML research.

Early-watch prices/signals end 2026-08-20. The inspected external news dates
were September 7-21, with no separate first-available timestamp; the old loader
had no upper date bound. These events cannot establish August point-in-time
evidence. The legacy engine produces 12 rows, not a universal five-NEW-idea
limit. The new reporting boundary caps NEW, validated observations at five.

The imported workbench ends 2026-10-05, retains all 52 configurations and zero
joint base/stress passes, and has no current trade ideas. The original forward
record remains paused; two observed days, last observed session 2026-09-30.
The import does not backfill gaps or restart its automations.

## Delivery and scheduler

The reviewed first status-only report was accepted with **HTTP 200 and `ok`**
at **2026-10-06T16:49:58.130508Z**, for completed NYSE session **2026-10-05**.
It contained zero new watch ideas. The private outbox is SENT with one attempt.
A repeated `--send` was skipped, not treated as a second successful delivery.
See the [sanitized receipt](DELIVERY_RECEIPT.json) and
[reviewed representative preview](SLACK_PREVIEW.md).

The old `com.usstockqr.slack-signal-monitor` was persistently disabled and
booted out after exact-label/argv verification. Its original plist and a
restricted private rollback copy remain local. No PID-name kill or broad
service operation was used.

The new `com.usstockqr.daily-research-report` is installed and loaded. It is a
bounded 900-second invocation of the tested
`scripts/send_slack_signal_once.py --send`, internally gated after 06:00
Asia/Shanghai and deduplicated by destination alias/stream/NYSE session.
It has RunAtLoad, no KeepAlive, no broker refresh and no force-heartbeat flags.
The actual first launchd run exited 0 and returned `session_already_reported`;
the outbox attempt count remained one. The three other installed
Discord/collector job configurations were verified unchanged.

The outbox records SENT only after HTTP 200 with `ok`. Failures are not sent.
UNKNOWN/PENDING outcomes require reconciliation, not automatic retransmission.
Incoming webhooks alone cannot guarantee provider-side exactly-once delivery
after acknowledgement loss. No Slack channel-history access was assumed.
The job depends on this in-place checkout, its interpreter and network access.
It does not automatically update or merge the working branch.

Discord scheduling, manual holdings, FIFO and brokerage pipelines were not
changed. Their pre-existing independent issues remain out of this unit:
local-time schedules, stale cutoff derivation, daily reimport resetting watch
state, date-incomplete price coverage, and truncated holdings notifications.
Do not publish or run those private workflows as an integration shortcut.

## Next bounded work

1. Review the integration PR and keep future code, sanitized evidence and
   progress in this repository; do not mistake the review branch for `main`.
2. Obtain genuinely point-in-time, same-cutoff inputs before adding actionable
   watch content. Do not substitute refreshed file timestamps for availability.
3. Compare matched windows, cost assumptions, Sharpe definitions and universe
   construction before any new preregistered experiment. Preserve failures and
   paused forward provenance.

All subsequent integration changes and progress belong in this repository.
Reviewed explicit commits/PRs are the executable publication process.
**No automatic daily GitHub push is configured**, and the reporter has no git,
broker, holdings-import, Discord or strategy-search execution path.

## Exact acceptance commands

```bash
.venv/bin/python -B -m unittest scripts.test_research_reporting_guard scripts.test_incumbent_baseline_guard
.venv/bin/python -B scripts/check_publication.py
.venv/bin/python -B scripts/send_slack_signal_once.py --dry-run
.venv/bin/python -B scripts/install_research_report_launchd.py --dry-run
cd research/workbench
.venv/bin/python -I -B -m pytest -q
```

The first command includes calendar/17:30/next-open checks, concurrency and
UNKNOWN handling, exact scheduler arguments, receipt-before-bootstrap gates,
private-data nonreads, Discord artifact compatibility, the frozen strategy
hashes and synthetic engine/weight equivalence. No further live POST is
required to run these checks.
