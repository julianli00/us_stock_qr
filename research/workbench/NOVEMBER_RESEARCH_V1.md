# November research lifecycle V1

The explicit November authorization is separate from the closed October window.
It starts no earlier than **2026-11-03T07:40:50Z** and ends at
**2026-11-30 23:59 Beijing / 2026-11-30T15:59:00Z**. It does not retroactively
authorize the October overrun or change the actual late cleanup receipts.
The private authorization record is pinned by
`6324d64f2c9ca99349e2b5e9f2502f0bc908091d02cb8cdede5d736c0bad6b4f`.

`us_quant.november_research_lifecycle_v1` is the new operational Interface.
New November orchestration calls its `runtime_guard()` before each new action
and uses its `research_job()` over the **same existing cross-version lock**.
The old October implementation and deadline are not patched or monkeypatched.
The complete original financial protocol remains byte-pinned by
`1b5b2572089ffa53746b01e72ff8303e00d6b5e5e3f815844f93834d0521e469`:
same full windows, capital, RF, initial drawdown peak, costs, delays, independent
accounting, matched SPY/QQQ benchmarks and strict per-path qualification gates.

Only the existing SSD handoff session owns the one-minute continuation.
At **2026-11-30T15:49:00Z / 23:49 Beijing**, new actions are blocked and that
same binding enters closeout-only mode. This ten-minute operational margin is
not a financial threshold or another timer. Closeout preserves partial work,
saves the summary/checkpoint, clears the owned native binding and reads it back.
Unavailable host/SSD/native permissions can interrupt execution; uninterrupted
24-hour operation or exactly on-time cleanup is not promised.

The new full continuation prompt is derived from the actual October prompt
`46d88b147d139df4fd238fbdbbbc40fbfc543a5250b05d0bfb7c0a73bb937b92`.
Only its explicit authority and three operational date occurrences change;
the original financial and safety text remains, with stricter November
ownership, time checks, terminal-source and closeout instructions appended.
Actual get-save-get receipts and native invocation origin are separate evidence:
configuration is not proof of a later scheduled call.

The recovered baseline is **675 events, 64 registrations, 56 completed reviews,
eight technical failures, zero pending, 224 funded paths, 234 combined
completions and zero qualifiers**. Seven experts are technically available.
This is not a reset, new financial result or investment-goal completion.
Term V1 remains terminal; V2 remains blocked. VIG/SDY429 and IBKR unknown2157
remain stopped. New observations are separately timestamped; missed forward
days and the two old single-day identities are never backfilled.
