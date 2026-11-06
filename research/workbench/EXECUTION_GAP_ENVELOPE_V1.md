# Finite execution-gap reference V1

## Independent question, not a headroom retry

This source-only feasibility study asks what a hypothetical newly issued full
fractional rebalance could certify over the original two-session stress delay.
It is not a new portfolio policy, applied target, account or financial trial.
The old headroom V1 remains terminal: its 451 retained probes and missing failed
comparison operands do not establish an exact numerical root cause. Neither
its resolver, verifier nor preparation/evaluation pipeline is executed here.

The new reference separates the positions held before delivery, the scenario's
opening NAV and fee bound, and the first closing mark after delivery. This is
not the old close-only, next-review-horizon rule or a different 126-day setting.
All fully known historical two-transition windows are used without a lookback
scan. Data after the current information date cannot enter a scenario.

## Interface and finite assumptions

The `resolve` / `verify` Module accepts a known MainMark, one saved nominal
14-asset expert composition and a dated StressSample. The real known-price
archive Adapter and synthetic path Adapter cross the same Seam. It returns an
exact interval of represented sufficient inequalities, or an explicit failure
with no selected fraction. It never returns a target vector or silently routes
infeasibility to BIL.

Within each supplied scenario, original held quantities persist through the
next close and second open. A hypothetical full fill then interpolates all
non-BIL nominal weights by a fraction between zero and one; removed weight
goes to actual BIL. Its price-bearing return is not idle-cash interest.
The reference ends at that delivery session's close, not the next month.

At the scenario open, a conservative upper bound on original 20bp slippage and
USD1-per-changed-asset fees is 0.002 times opening gross assets plus nominal
ETF budget times opening NAV, plus 14 tickets. The post-cost NAV lower bound
uses that allowance; it is not an actual debit. Original no-op notional slack
and USD1e-6 money coherence reserve cover reference marking/rounding.
Long-only positive marks make this a conservative delivered-close inequality
conditional on the specified scenario and fractional cash-funding assumptions.

The first waiting close can already violate 85% of the current own-main peak.
No new delayed fill can change holdings at that earlier close. Current saved
states already outside the reference are labeled separately. An exhausted
conservative fee bound or empty postfill interval means only that this
sufficient test cannot certify a fraction; it is not proof that every actual
trade or future policy is physically impossible.

## Exact certificate and limitations

Intersect the represented binary64 affine inequalities with the unit interval
using exact Fractions. Retain lower, upper and zero-slope witnesses; valid tied
witnesses need not have identical indices. An independent scalar construction
checks all scenario coefficients under the original USD1e-6 money limit,
then verifies exact represented endpoints and witnesses separately.

Frozen inputs are all 364 saved initial/month-end decision states from the
four complete Prod books, their actual holdings/cash/NAV/own peaks, the original
saved unified nominal targets and source-admitted market panels. Fresh five-year
initialization uses fresh capital/peak, not the ten-year book's main wealth.
No source learner, shadow, account, benchmark or prior diagnosis is rerun.

This does not implement a new band, daily controller, request/queue policy,
fill instruction or standby allocation. Original actual decisions stay intact.
Historical windows overlap and are not independent samples or future support
bounds. Even a certified finite interval does not establish a future drawdown
guarantee, a maximal actual trade, financial qualification or forward profit.

Freeze criteria, inputs, code and successful synthetic evidence, save and
prospectively link an actual preanalysis checkpoint, then inspect the saved
states once. Retain exact scenario matrices and source references privately
before resolving each interval. Five prefixes and future-price perturbation
checks must preserve past reference results. Only sanitized aggregates are
published; no raw prices, holdings, nominal targets or dated case rows.
