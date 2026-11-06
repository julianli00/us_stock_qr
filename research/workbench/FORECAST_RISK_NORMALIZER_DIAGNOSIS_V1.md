# Signed forecast-score risk normalizer diagnosis V1

The latest complete portfolios pass their return, RF-Sharpe and dual-benchmark
gates but fail every drawdown gate. Before changing allocation or adding a
protective policy, this read-only study tests a narrower mechanical question:
does the **recorded score** sometimes reward greater risk because its predicted
post-cost active-rate numerator is negative?

This is not the earlier future-variance/ERC calibration diagnosis. It reads
only the retained source allocation audit underlying both completed forecast
portfolios. It does not fit risk, read future-month returns, generate alternative
pair rankings, change weights, reconstruct holdings or attribute main losses.

## Frozen identity and Interface

For each recorded pair let each numerator be its saved SPY/QQQ forecast mean
minus its saved nominal switch cost divided by21. Use the actual binary64
operations used by the frozen score, without a near-zero tolerance band.
Let `u` be the smaller numerator and let the two positive RF-excess pair risks
be the original training and validation standard deviations.

The minimum of the four recorded annualized scores is `sqrt252*u/risk`.
For positive`u`, the binding risk is the **larger** standard deviation and the
continuous score decreases as risk grows at fixed numerator. For negative`u`,
the binding risk is the **smaller** standard deviation and the continuous
score increases as risk grows at fixed numerator. At zero, both risk windows
are score-indifferent. Tied worst benchmarks/windows are retained as sets.

`diagnose` consumes saved audits and returns sign/identity classifications and
aggregate counts. Both real saved-record and synthetic-record Adapters cross
that Interface. No source selector, original score kernel, ridge fit, target
builder, account, benchmark or historical verifier entrypoint is invoked.
An independent50-digit Decimal implementation checks all four components,
not only the recorded selected pair. The fixed1e-10absolute agreement tolerance
is the original allocation proof's score tolerance, not a revised significance
test or variance floor.

Retain all121original dates, both models, all28mature menus and explicit
unscored/unscorable states. Preserve inputs before each classification and
verify exact prefix invariance. Negative values are not relabeled missing,
zero or safe. No alternative winner or new portfolio is produced.

## Interpretation limits

The derivative is a local property of the **continuous unrounded** score with
its numerator held fixed. The actual decision uses a12-decimal rounded key;
this diagnostic does not claim that infinitesimal risk changes alter that
discrete ranking or reconstruct another previous-allocation state.
Past risk estimates are not future risk or main drawdown guarantees.

Report selected and nonselected numerator signs, binding-risk behavior and
origins with all eligible numerators negative. A selected negative numerator
despite a nonnegative menu is reported without silently waiving possible
rounding ties. Counts across overlapping pairs, models and dates are not
independent economic trials.

Finding the algebraic incentive does **not** show it caused the recorded
20--26%drawdowns. Its absence among selected cases would not prove the
portfolio safe either. The study is proposed after285exposed completions,
keeps every financial failure and all source/PIT/RF restrictions, and changes
no acceptance gate, global expert roster or forward identity. New allocations,
financial trials and trading authority remain zero.
