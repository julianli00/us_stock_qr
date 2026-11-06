# Complete-expert active-return prediction diagnosis V1

This is a read-only prerequisite question, not another funded policy or a repair
of a rejected allocation. Can a fixed continuous price context help predict the
**amount** of complete experts' next-month assessment active returns, compared
with the same unconditional learner? The older categorical contest forecast
predicted winning credit among six experts. The older dual-relative selector
ranked historical active-risk scores. Neither is this monetary-rate regression.

The pool is exactly the original eight admitted complete policies in their
original order, not the most recently tested compact trio. All eight results are
retained; no expert or predictor is selected from this diagnosis. Mechanism
labels do not prove independent risks. Standalone financial rejections, actual
introduction/admission clocks, source availability, and global membership stay
unchanged. The existing admitted defensive assessment/recommendation views are
reused unchanged; this is not a claim that its native RF equals canonical RF.

## Frozen question and Interface

`analyze` consumes dated, full-precision source observations and known price
observations and returns assessment forecasts and descriptive error summaries.
It does not accept or return holdings, expert allocation weights, orders or
capital. The cached-source Adapter and synthetic input Adapter cross this same
Interface. The independent verifier uses batch NumPy linear algebra, not the
Implementation's exact-rational elimination.

The label is the arithmetic mean of daily **source NET minus benchmark NET**
returns in one full exchange month, for each of the eight experts and both
original full-capital tradable SPY/QQQ 10y-base books. These sixteen assessment
rates are not realized monthly main returns, compound returns, forecast wealth
or independent observations. Original source costs are already in source NET.
No new cost proxy, financial metric, account, shadow, RF view or source request
is introduced. Assessment-to-main divergence is not measured or waived here.

Both learners use every previously completed full month with equal month
weight. The intercept-only predictor uses one feature. The contextual predictor
adds the SPY/BIL and IEF/BIL relative-price returns over the same 126 sessions
already used by the earlier context study. It uses their continuous values, not
four context bins. Features are known at the preceding decision; the label is
unavailable until the next complete month ends.

The coefficient prior is fixed unit ridge on **every coefficient, including the
intercept**: solve `(I + X'X) b = X'y`. The zero prior means zero assessment active
rate, not a safe asset or calibrated economic belief. No standardization,
fitted penalty, rolling window search, variance floor, winsorization, solver
retry or hyperparameter selection is permitted. Recorded binary64 values are
treated as exact rationals in the primary Implementation. The independent
batch-double check has a fixed absolute tolerance of 1e-12 before real outputs.

## Chronology and missingness

Keep the original initial anchor and every genuine month end. The initial
partial October 2016 interval is never assimilated. At each full month end,
grade the **previously issued** forecast first, assimilate that outcome second,
then predict the following month from the newly known context. Warm-up learning
can accumulate genuine months, but no forecast is issued until all eight source
histories have 126 common matured observations. An unscored forecast is null,
not zero or success. A zero numeric prior exists only inside the declared model.

Every daily source/benchmark date must match. Missing/nonfinite inputs,
unavailable opinions, future clocks, skipped months or inconsistent RF views
raise explicit errors. No return day is cropped or padded. The final incomplete
October 2026 interval is not graded or backfilled. A pending final forecast is
not a new prospective account.

Actual inputs are saved before updates, with the resolved audit immediately
afterward. Synthetic future-poisoning checks and exact prefix truncations test
chronology; all nonselected sources are independently verified.

## Interpretation and reporting

Report equal-month RMSE, MAE, signed error and exact three-way sign accuracy
(positive/zero/negative), both pooled and for **every** source/benchmark/model.
The descriptive skill score is one minus forecast MSE divided by the zero-active
reference MSE. Zero denominator means undefined skill, explicitly null. This
reference is not an implemented strategy. Pooled correlated expert errors do
not multiply the number of independent months. No significance test,
confidence claim, acceptance threshold or predictor promotion follows.

The study is adaptively proposed after 283 exposed financial completions and
other diagnoses. Causal historical forecasts are not independent out-of-sample
or forward proof. No future financial rule or allocation is frozen by this
diagnosis. Any later funded study needs separately justified rules, actual new
registration/publication fences and all unchanged original four-path gates.
Current financial qualification and investment-objective status remain unmet.
