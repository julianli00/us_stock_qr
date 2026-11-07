# Original AdaNormalHedge source-relative regret study

Study: `ssd_expert_adanormalhedge_20261107_v1`.
Candidate: `expert_pool_six_adanormalhedge_v1`.

The question holds the original six bounded source-loss labels fixed and replaces
the source learner with Algorithm 1 of arXiv `1502.05934v1`. It does not tune
another temperature, switching hazard, normalization, confidence or score window.
The original mathematical document, acquisition receipt and source catalogue were
captured before this Implementation; no mathematical document is fetched again.

## Module and Interface

The Module accepts six complete then-known policy opinions through `OpinionFeed`,
with an archive Adapter for admitted saved inputs and a memory Adapter for
synthetics. `build_targets` advances one causal source learner, returns one
underlying-asset target and retains the exact operands before each update.

An expert remains a complete causal, admitted, unleveraged cash-funded policy,
not an asset sleeve. An expert's availability, score maturity, source allocation
and the final main portfolio's qualification are different facts. Zero potential
weight is a decision in this source game, not a new technical admission decision.

The capital Interface is reused without replaying either previous source learner.
The source state is not a main account: each of the four formal paths starts with
fresh USD 10,000 and executes unified underlying holdings with the original actual
holding band and pending-ticket feedback.

## Fixed source law

For the previously played source distribution, source regret is played loss minus
each original expert loss. `R` sums signed regret; `C` sums absolute instantaneous
regret. The potential is `exp(max(R, 0)^2 / (3*C))`, with `Phi(0,0)=1`.
The next score is half the difference between the potentials at `(R+1,C+1)` and
`(R-1,C+1)`, weighted by the uniform six-expert prior and normalized.

Algorithm 1's footnote permits arbitrary prediction if every score is zero.
This study fixes the uniform source prior for that valid case and logs it.
It never resets `R/C`, excludes an admitted expert, clips regret or installs a
positive weight floor to repair failed arithmetic.

The primary Implementation uses 320-digit Decimal arithmetic and a stable positive
exponential increment. A separate 160-digit learner maintains its own played loss,
regrets and magnitudes; its scores use the positive polynomial-difference series.
Both stop explicitly if numerical convergence fails. The original 119 complete
source rounds bound every shifted exponent by 40. The 4096-term cap and arithmetic
precision are numerical safety choices, not a financial parameter search.

Played loss is evaluated as the common minimum plus the weighted differences from
that minimum. This is an arithmetic identity, not recoding the original labels;
it preserves exact zero regret for equal labels without a numerical floor.
Absolute Decimal operands use context-free magnitudes, not an implicit 28-digit
rounding in a caller.

Borrowed market and publication helpers are rebound at invocation with guarded
content readers. The archived Adapter does not expose legacy unchecked hashes;
removing a protected reference from one list is not sufficient.

The source loss forecast precedes the interval's labels. Only a complete matured
month is assimilated before forming the next target. The initial partial month is
not trained. Both original capital windows consume the same global source history,
but do not fabricate prior main-account wealth or refresh old native opinions.

## Financial and information fences

The original full 10y/5y base/stress windows, fresh capital, RF delivery, initial
drawdown peak, modeled ticket cost, execution delays, long-only cash funding,
98% target budget/2% idle reserve and actual BIL remain unchanged.
Every path must satisfy strict net RF Sharpe > 1, strict net CAGR > 12%, maximum
drawdown <= 15%, both matched-cost tradable full-capital SPY/QQQ benchmarks, and
the original additional stress/base-SPY gate. Actual turnover, modeled tickets and
fees are disclosed; no numerical turnover cap is invented.

Synthetic verification, actual code/input/package freeze, the actual successful
checkpoint and original receipt, one schema-3 registration, exact remote
preregistration and a prefunding checkpoint precede any real source `R/C`,
weights, targets or funded account. Partial/frozen failures are preserved, never
patched and rerun under the same identity.

Prefix checks cover source inputs, forecasts, states and targets, followed by
independent complete cash/position operands, BT, source initialization and capital
prefixes. Neither averaged expert CAGR/drawdown nor summed independent expert
capital is an acceptable main portfolio.

## Protected evidence and interpretation

`stored-main-cost-audit-v1.json` is excluded from both live protection lists.
Every new content-hash/read entry rejects its path before opening it. Its old
reference is archival only, never counted as a new live-validation success.
Sixteen reference/metadata items are distinct from fifteen permitted live-content
checks. No prior config, frozen code, failure, approval hash or source clock is
rewritten to conceal the earlier prohibited hash access.

The same six representatives, admissible nonselected variants, original source
clocks and exposed-history selection bias are retained. Existing completed studies
are not repeated. PIT/source refusals and missed-forward nonbackfill restrictions
remain; no market request, replacement RF view or broker synchronization is added.

Potential, regret and source mixture weights are not market probabilities, main
net wealth, leverage, risk limits or drawdown guarantees. This historical study is
not independent forward proof, an investment-goal completion or order authority.
