# Canonical integration status

Updated: 2026-10-06. Canonical repository: `julianli00/us_stock_qr`.

## Verified first checkpoint

- Imported the byte-preserved 70-file workbench under `research/workbench`.
  Import manifest SHA256:
  `3b2ef6789348ac88b8fdf82805415f518da85bb72c1d652735e7aeaaa63666b6`.
- Preserved the incumbent Python 3.9 package/environment. Independently ran
  all 357 imported tests in a separate Python 3.12 environment with package
  resolution checked against the imported source.
- Added the pure research report, strict watch proof contract, NYSE calendar,
  locked/atomic session outbox and credential-free preview.
- Passed 30 deterministic reporting guards, including multi-process delivery
  races, interrupted/ambiguous sends and private-ledger nonreads.
- All existing Slack send/preview entrypoints use the canonical report before
  the legacy mixed-date payload/hash. Force cannot restore stale trading text.

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

At this checkpoint: **no new real Slack POST; new scheduler not installed**.
The old KeepAlive service still needs the approved exact-label cutover.
The replacement is a bounded 900-second invocation of the tested
`scripts/send_slack_signal_once.py --send`, internally gated after 06:00
Asia/Shanghai and deduplicated by destination alias/stream/NYSE session.

The outbox records SENT only after HTTP 200 with `ok`. Failures are not sent.
UNKNOWN/PENDING outcomes require reconciliation, not automatic retransmission.
Incoming webhooks alone cannot guarantee provider-side exactly-once delivery
after acknowledgement loss. No Slack channel-history access was assumed.

Discord scheduling, manual holdings, FIFO and brokerage pipelines were not
changed. Their pre-existing independent issues remain out of this unit:
local-time schedules, stale cutoff derivation, daily reimport resetting watch
state, date-incomplete price coverage, and truncated holdings notifications.
Do not publish or run those private workflows as an integration shortcut.

## Next bounded work

1. Complete final rollout/publication checks and the reviewed status-only
   Slack delivery; record receipt and scheduler state without credentials.
2. Obtain genuinely point-in-time, same-cutoff inputs before adding actionable
   watch content. Do not substitute refreshed file timestamps for availability.
3. Compare matched windows, cost assumptions, Sharpe definitions and universe
   construction before any new preregistered experiment. Preserve failures and
   paused forward provenance.

All subsequent integration changes and progress belong in this repository.
Reviewed explicit commits/PRs are the executable publication process.
**No automatic daily GitHub push is configured**, and the reporter has no git,
broker, holdings-import, Discord or strategy-search execution path.
