# Main-capital close-mark cost estimate V1

## Status and purpose

This is a new, local preparation stage, not a financial trial, expert admission,
pool promotion or evidence that the portfolio objective is met. It addresses a
specific missing input to the original rolling meta-strategy: its switching-cost
proxy compares two current expert-opinion targets at a nominal USD 10000. That
proxy is not the main capital account's actual holdings, cash or wealth.

The hypothesis for a later separately registered study is that pricing candidate
allocations from the known main capital state can alter cost-aware pair selection
without changing the original training, validation, risk or economic source
rules. This is not a new return forecast or a promise to meet drawdown gates.
No historical candidate allocation, capital path or performance is computed in
this preparation stage.

## Module, Interface, Seam and Adapters

`SwitchCostEstimator.estimate(target)` is the narrow Interface at the cost-input
Seam. It returns an immutable, validated `CostEstimate`, not an execution.

`TargetProxyCost` is an Adapter reproducing the exact original nominal target
proxy, including its 1e-12 changed-coordinate rule and USD 10000 denominator.
Its reference is the current known expert opinions combined using the previous
statistical expert weights. It is not yesterday's holdings.

`CloseMarkedCashCost` is an Adapter that copies nonnegative, current close-marked
asset dollar values and cash. It applies the original 2.5pp actual-holdings band,
including the original 1e-12 comparison tolerance. Inside the band it estimates
no execution. Otherwise it calls the unchanged cash-funded accounting solver
with the baseline five basis points per side and USD 1 model ticket, preserving
post-cost target funding and the engine's existing notional threshold.
An explicit capital initialization must start entirely in cash.

The Adapter also checks the returned pro-forma book against its full post-cost
target and capital conservation. The unchanged legacy bisection can encounter
a fixed-ticket discontinuity at very small wealth: a tiny lower feasible point
need not implement the desired portfolio. Such a result raises `QuantError`
here rather than becoming a successful cheap estimate. The legacy solver and
all historical accounts remain unchanged. The retained first synthetic
preflight exposed this at USD 0.50 initial cash with a USD 1 commission.

Both Adapters require exactly ordered named asset coordinates and complete 98%
ETF targets. They reject missing/nonfinite/negative inputs, silent reindexing,
renormalization, borrowing and unfundable charges. The cash-funded Adapter uses
actual input wealth as its denominator; it never substitutes USD 10000.
These are research model ticket counts, not brokerage messages or orders.

`choose_pair_with_cost` consumes the same Interface. It reuses the unchanged
original pair catalogue and its train-derived equal marginal variance weights,
mechanism exclusions, training correlation and marginal risk contributions.
It recomputes only the two net risk scores with the supplied cost fraction,
retaining 84/42 out of 126 matured sessions, 21-session amortization, the lower
score, twelve-decimal comparison and lexical pair tie break. Unmatured or
degenerate assessments remain explicitly unscored equal weights, not success.
Every scored row preserves its original target-proxy assessment separately.
Original modules and archived policy implementations are not edited.

The close-marked fraction prices the whole candidate rebalance, including a
native opinion refresh with no change in statistical expert weights. It is not
an incremental fee relative to a newly optimized alternative. Stored expert
NET returns already contain each shadow book's own frictions; the additional
score penalty may be conservative or double-penalize some behavior. It is not
a calibrated forecast or isolated causal fee/alpha attribution. This penalty
does not debit the main account: a future funded runner must charge its actual
modeled fills exactly once.

## Information clock and limits

The caller must supply the main account's legitimately known holdings and cash
at the decision clock, after the current close. The Module accepts no future
prices, pending future fills or prices at the later execution opening. It is a
close-marked baseline-cost estimate, not the actual cost that will be paid.
Market gaps, delayed execution, intervening holdings changes and future capital
can make the eventual fee, turnover and ticket count different.

The assessment's five-basis-point/USD 1 convention is fixed for both future
base and stress assessments, as in the old proxy. It must not be changed after
seeing outcomes. Actual execution, if later authorized through a complete
preregistered research packet, must still use each original path's own five or
twenty basis points, commission and one/two-session delay. The 2.5pp band is an
unchanged research choice, not a user-imposed turnover ceiling.

The Module does not establish that the caller supplied a causal main account.
No held values may be reconstructed from future prices or borrowed from an
expert's separate USD 10000 shadow book. A real integration must price candidate
targets from the main account then known, derive one unified underlying-asset
target, and issue it for later execution without anticipating that execution.

## Preparation evidence and future financial fence

Only deterministic synthetic fixtures are used here. Tests cross the same
Interface with both Adapters; check exact original-proxy scores and choices,
independent closed-form funding examples, fixed-commission capital scaling,
band edges, explicit invalid states, copied state, five prefix/future
perturbations and the absence of an account-execution entrypoint.
Synthetic prefix checks do not prove a future real capital runner is causal.
An independent full financial-account/BT comparison has not yet been performed.

Any later financial study needs a new explicitly versioned funded runner and
publication packet: precomputed target files cannot secretly depend on an
unpublished historical main-account simulation. Registration, exact economic
law/parameters, original eight-source pool, source identities and implementation
must be frozen and actually published before the first new capital computation.
The existing equal-eight and rolling-eight results are cached controls, not
permission to replay those accounts. The global rosters remain unchanged.

All four original full windows/cost paths, fresh USD 10000/initial peak, exact RF,
cash funding, independent accounting/BT, matched-cost full-capital SPY/QQQ and
all original joint gates remain binding. No financial criterion, source/PIT
scope or old failure is relaxed. FXY remains not admitted after its stopped
statutory-document source path; neither FXY nor credit/Macro/HMM variants are
automatically included. Actual introduction is November, not historical
availability; a future historical result will still be exposed and
counterfactual, not independent forward validation.
