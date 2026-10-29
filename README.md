# Canonical US equity research

`julianli00/us_stock_qr` is the canonical home for integration code, research
definitions, failed trials and reviewed progress. This is research software,
not authority to trade or to modify user-maintained holdings.

## Separate research lanes

| Lane | Preserved evidence | Boundary |
| --- | --- | --- |
| Incumbent Top30 | Static 699-stock, 126-day momentum, inverse-volatility monthly strategy; saved 2015-01-02 through 2026-05-08 result | CAGR 29.54%, Sharpe 1.1867, drawdown 28.31%; full-sample selection, not independent OOS; fails the 15% drawdown gate |
| Early watch | Price/volume plus optional event evidence; legacy prices/signals through 2026-08-20 | September evidence cannot validate August signals; historical news/Reddit coverage incomplete |
| [Isolated workbench](research/workbench/README.md) | Initial imported snapshot: 52 registered configurations through 2026-10-05; zero joint 10-year/5-year passes | Separate package/environment; original snapshot and subsequent historical protocols remain preserved |
| [SSD dual-benchmark research](research/workbench/DUAL_BENCHMARK_V1.md) | 178 legacy completions preserved; 23 SSD registrations, 22 completed/rejected and one retained preparation failure; 88 funded paths | Separate net-CAGR/SPY/QQQ criteria; zero qualifiers, no independent forward qualification or trading authority |
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

## Current SSD research authorization

The accepted SSD research stage runs through **2026-10-31 23:59 Beijing time**
(`2026-10-31T15:59:00Z`) without a configuration-count ceiling or ordinary
per-batch approval. The old exhausted six-configuration round remains historical;
it is not a limit on this separately authorized stage. Earlier instructions and
results below retain their original scope, not the current acceptance criteria.
No messaging, trading, holdings synchronization or production scheduler is
activated by this research authorization.

The [new frozen protocol](research/workbench/config/dual-benchmark-v1.json)
requires every original ten/five-year base/stress path to have net Sharpe above
1, net CAGR above 12%, drawdown at most 15%, and beat full-capital matched-cost
SPY **and QQQ**, retaining the original extra stress-versus-base-SPY gate.
The [eight measured reference accounts](research/workbench/evidence/dual_benchmark_20261027_v1.json)
are benchmarks, not additional strategy trials. The
[first two actual registrations](research/workbench/evidence/equity_first_20261027_registration_v1.json)
occurred at 2026-10-27 16:30 UTC, already October 28 in Beijing. Their source-bound
target proofs precede any candidate account; actual remote publication remains
a required fence. Preregistration commit `00b82e9e00cc777901689190083f0627c4ac4a48`
was genuinely published before all eight candidate paths completed at 16:52 UTC.
Both hypotheses were rejected; the
[complete scalar results](research/workbench/evidence/equity_first_20261027_results_v1.json)
retain every path, turnover and fee. The first pair brought the completed total
to 180 across the preserved legacy and separate SSD lanes, not a qualified champion.

The [next actual registration](research/workbench/evidence/commitment_duration_20261027_registration_v1.json)
occurred at 2026-10-27 18:39:57 UTC, before target preparation. It freezes
month-end intent commitment and 21 equal rotating commitments of the unchanged
uncapped source. Both 2,533-session target matrices matched independent causal
checks; these are not account results. Preregistration commit
`56c3c9366bbbaf0249eb7990d500f9edc3aaa143` was published before all eight new
paths completed at 18:50 UTC. The [complete commitment results](research/workbench/evidence/commitment_duration_20261027_results_v1.json)
reject both hypotheses: month-end commitment beats both return benchmarks on
every path but still fails the drawdown gate throughout, and most Sharpe paths
also fail. The 21-slot law fails every Sharpe/drawdown path and misses ten-year
QQQ. At that cohort's completion there were four completed/rejected new-lane
hypotheses, 182 completed configurations across both lanes, and no champion.

The [third cohort registration](research/workbench/evidence/channel_regimes_20261027_registration_v1.json)
occurred at 2026-10-27 19:51:47 UTC. Its fixed stateful price channel enters on a
new prior-quarter closing high and exits on a new prior-month closing low.
The pair compares the existing gold trend defense with an independently
maintained gold channel, without a horizon search. Both 2,553-session target
matrices match the separate sequential oracle and eight prefix checks.
No channel account was computed at registration. Actual preregistration commit
`b013af2a33c431f5ab386dff2643c86241979f5f` was published before all eight paths
completed at 19:56:54 UTC. The [complete channel results](research/workbench/evidence/channel_regimes_20261027_results_v1.json)
reject both hypotheses: every path fails Sharpe, drawdown and QQQ gates.
Ten-year base net CAGR is 9.5344% with gold trend defense and 7.0490% with
gold channel defense; annualized two-sided turnover rises to 22.6986 and
17.0165. At that cohort's completion there were six rejected SSD hypotheses,
24 new account paths, 184 combined completed configurations and no champion.

The [fourth cohort registration](research/workbench/evidence/option_exit_latch_20261027_registration_v1.json)
occurred at 2026-10-27 20:31:30 UTC, before target preparation. It contains
exactly one fixed asymmetric rule: option-term inversion exits the saved
monthly commitment into actual BIL, while only a noninverted completed exchange
month end can restore the latest monthly intent. It does not retune the failed
price channels or replay the original monthly accounts. All 2,533 defined
targets match a separate sequential oracle and nine prefix checks. At that
registration there were seven registered SSD hypotheses, six completed and
one pending; no new candidate account had run.

Actual preregistration commit `c6452082acafcbffc6b7cf0fbce85ab1488abfaa`
was published before all four new paths completed at 20:43:19 UTC.
The [complete option-latch results](research/workbench/evidence/option_exit_latch_20261027_results_v1.json)
reject the hypothesis: every path clears 12% net CAGR but fails Sharpe,
drawdown and QQQ gates. Ten-year base net CAGR is 15.6961%, Sharpe 0.8171
and maximum drawdown 30.4238%; annualized two-sided turnover is 11.5617
with 416 model orders and USD 1,561.68 total costs. Only that path beats
matched-cost SPY. That snapshot contained seven completed/rejected SSD hypotheses,
28 new account paths and 185 combined completed configurations, with no
qualified champion or new prospective observation.

The subsequent saved-account diagnosis separates two failures. Both base
paths first exceed 15% drawdown on 2022-01-25 while the previous-close intent
still holds stocks and the option curve is not inverted. Both stress paths
first breach on 2022-01-27 with a prior BIL intent but stocks still held under
the preserved extra execution delay. This adds no strategy trial and does
not justify tuning the option threshold, recovery clock or execution rules.

The [separate SSD factor catalog](research/workbench/evidence/ssd_factor_catalog_20261028_v1.json)
registered the existing IJR size and PKW net-share-reduction definitions at
2026-10-27 22:26:38 UTC, in the actual 2026-W44 cycle. This adds two definitions,
not two strategy trials: the original catalog remains six definitions across
five families; the combined catalogs contain eight definitions across seven
families. The new 14-asset historical panel preserves every original
12-asset price cell and the exact RF reference, adding only IJR and PKW.
Its fresh cache check is not newly retrieved prices or refreshed mandates.
That admission selected or evaluated no strategy; the original W42 plans,
SQLite ledger, benchmark accounts and prospective models remain unchanged.

The subsequent
[family-core preregistration](research/workbench/evidence/family_core_expansion_20261028_registration_v1.json)
freezes one new hypothesis at 2026-10-27 23:58:51 UTC. It preserves the actual
saved monthly equity-first source, half-SOXX stock allocation, risk/defense
decisions and execution. Only the other half of the stock allocation changes:
six equal economic-family funds replace four. The signal still refers to
the original stock proxy; it is not rebuilt using the new portfolio.
Independent target proof has zero difference, with eight prefix checks and
both full-window anchors defined. The
[complete result](research/workbench/evidence/family_core_expansion_20261028_results_v1.json)
followed actual preregistration publication `ea997167` and finished at
2026-10-28 00:11:13 UTC. It is **rejected**, despite all four net CAGRs exceeding
20%: every drawdown fails, three Sharpes fail, and ten-year stress misses QQQ.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 22.5567% | 0.9489 | 33.8188% | 3.7843 | 342 | 764.49 |
| 10y stress | 21.1015% | 0.8907 | 33.8303% | 3.7614 | 335 | 1,935.42 |
| 5y base | 26.4066% | 1.0480 | 22.6943% | 4.7239 | 229 | 369.21 |
| 5y stress | 23.2902% | 0.9278 | 23.0772% | 4.6816 | 222 | 744.31 |

The saved monthly source already had higher CAGR and Sharpe in **all four**
matching paths. The expansion increases turnover, model orders and costs in
all four; drawdown worsens in three. The slight five-year-base drawdown
improvement still leaves 22.69%, not the required 15%. This is a saved
whole-model comparison, including changed drift-triggered execution and fees,
not isolated asset PnL attribution. More economic labels did not improve
this fixed structure and are not independent alpha.

At that eighth-study snapshot there were eight SSD registrations/completions, zero pending,
32 paths and 186 combined completions, with no qualified champion.

The subsequent saved-holdings diagnosis separates nominal concentration
from realized risk. In the new ten-year base account, SOXX averages 40.87%
of closing NAV but contributes 64.16% of realized net daily-return variance;
all stock components contribute 93.31%. The stock-normalized holding HHI
falls from the source's 0.3147 to 0.2952, while source/new net returns remain
correlated above 0.9977. More labels did not produce independent portfolio
risk sources. These are signed ex-post covariance contributions, not
forecast risk budgets or causal alpha.

The 2020-02-19--2020-03-20 ten-year drawdown contains no model orders or fees:
losses come from held stock exposure, including both SOXX and the factor
core, largely overnight. In contrast, the new five-year first breaches
occur in 2022 while already holding gold, not while awaiting stock exits.
Gold and reduced stock exposure are not guarantees on cumulative drawdown.
The old covariance, diversification, volatility-cap, breadth and downside
risk-budget studies have now been checked before selecting another
construction; old QLD-based results are not current 1x evidence.

SOXX half-equity and equal factor-half allocation remain frozen **for that
completed controlled study**, not as permanent user allocation constraints.
At that diagnostic review no new construction had been selected. The configured
native continuation is now **one minute**, still with one active research
job and the original cutoff; it is not a per-minute backtest or data-download
instruction.

The [ninth preregistration](research/workbench/evidence/joint_tail_construction_20261028_registration_v1.json)
now fixes one different construction: minimize joint empirical 5% tail loss
using 252 known daily scenarios, subject to an observed momentum-opportunity
hurdle matching the stronger SPY/QQQ cash-relative score. The equal-family
core, SOXX, gold, intermediate/long Treasuries and BIL are allocated jointly;
SOXX is not a fixed half of the portfolio. This is not a future-return or
drawdown guarantee, or a reuse of full-sample risk shares as historical weights.
The supplemental review also records the previously rejected **unleveraged**
covariance risk-budget/max-diversification pair, not just the older QLD studies.

Actual registration at 2026-10-28 03:56:56 UTC precedes target preparation.
Of 122 completed monthly decisions, 75 have certified joint-tail optima,
14 have nonpositive benchmark opportunity and 33 have an analytically
unavailable opportunity hurdle. The latter two are explicit model conditions
for BIL, never a fallback on solver or data failure. All primary/secondary
LP certificates and independently sorted tail losses passed; five prefix
checks preserve causality. The preregistration snapshot had nine registrations,
eight completed and one pending, before any new account.

Actual preregistration `4fc9ebe220203cf86e6b48ff963d84a6572a3631` preceded
the [complete ninth result](research/workbench/evidence/joint_tail_construction_20261028_results_v1.json),
finished at 2026-10-28 04:03:55 UTC. The construction is **rejected**:
all four drawdowns satisfy 15%, but all four net CAGR, Sharpe, SPY and QQQ
gates fail.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 9.0968% | 0.7288 | 11.9830% | 5.5727 | 746 | 1,124.94 |
| 10y stress | 8.4278% | 0.6500 | 13.8192% | 5.5561 | 746 | 2,247.26 |
| 5y base | 11.2137% | 0.7971 | 12.0187% | 3.7283 | 339 | 448.68 |
| 5y stress | 9.8895% | 0.6471 | 13.8573% | 3.7162 | 339 | 760.48 |

The observed opportunity constraint and certified sample-tail optimum did
not guarantee future net return. Its low drawdowns cannot be combined with
the previous model's higher CAGR to manufacture a passing strategy.
Current totals are nine completed SSD hypotheses, 36 paths and 187 combined
completed configurations, with zero pending, zero technical failures and
no qualified champion. No original ledger or prospective model changed.

The saved-return diagnosis finds substantial BIL exposure even when an
optimizer solution exists. Mean closing BIL weights are 53.74%/53.72% in
ten-year base/stress and 63.11%/63.07% in five-year base/stress. In ten-year
base, optimizer-labeled intervals contribute a -0.6649 log-return gap to
QQQ, compared with -0.3839 from the two standby labels combined. The return
shortfall is therefore not confined to analytically unavailable-opportunity
periods. These are labels on existing returns, not counterfactual portfolios
or causal loss attribution.

Implementation also matters: fixed commissions are 66.31% of ten-year-base
costs and 75.56% of five-year-base costs, with mean modeled ticket notionals
of USD1,015.93 and USD647.06. This does not establish a zero-cost return or
show that fees alone explain the shortfall. Initial/final partial decision
periods are excluded from complete-period comparison counts. No optimizer,
target, account, primary metric or investment gate was rerun in that diagnosis.

The [tenth preregistration](research/workbench/evidence/benchmark_tail_budget_20261028_registration_v1.json)
fixes one reciprocal construction: maximize the same observed opportunity
scores under the lower observed SPY/QQQ tail-risk allowance, floored
at actual BIL tail risk. The secondary objective minimizes tail loss among
opportunity optima. No risk multiplier is fitted; the original 252-session
window, 5% tail, core constraint and funded execution remain unchanged.
It is not a forecast, a drawdown guarantee, or a sweep of the failed
opportunity hurdle.

Actual registration at 2026-10-28 05:46:36 UTC precedes target preparation.
All 122 monthly plans are independently certified: 112 optimized and ten
explicit nonpositive-opportunity BIL plans, with five prefix checks and
full-window factor coverage. That registration snapshot had no new funded
account.

Actual preregistration `1074904478bfd0ba2a19d502c479ebd2de13c227` preceded
the [complete tenth result](research/workbench/evidence/benchmark_tail_budget_20261028_results_v1.json),
finished at 2026-10-28 05:54:28 UTC. It is **rejected**: all four drawdown,
Sharpe and QQQ gates fail. Three CAGR paths pass 12%; only the two five-year
paths beat matched-cost SPY.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 12.9292% | 0.6781 | 26.4086% | 6.2167 | 744 | 1,187.36 |
| 10y stress | 11.4123% | 0.5922 | 27.0756% | 6.1955 | 737 | 2,380.56 |
| 5y base | 16.5627% | 0.7818 | 26.6875% | 6.4342 | 398 | 565.31 |
| 5y stress | 14.9118% | 0.6907 | 27.3277% | 6.4152 | 391 | 1,029.44 |

Growth-first raises returns relative to the ninth construction but loses
its drawdown control; neither meets the joint goal. Do not combine their
best metrics or tune a risk-budget multiplier after seeing this result.
Current totals are ten completed SSD hypotheses, 40 paths and 188 combined
completions, with zero pending, zero technical failures and no qualifier.

Improvement relative to the ninth construction is **not an improvement of
the broader retained research frontier**. The existing
`semiconductor_option_proxy_dollar_observed_band025` reference has higher
CAGR and Sharpe, lower drawdown, fewer modeled orders and lower total fees
than the tenth construction on all four comparable funded paths. It still
fails ten-year SPY and QQQ, so remains rejected and is not promoted.
Different construction, information sets and decision calendars prevent
interpreting this comparison as a single-factor or fee-only causal effect.
The [bounded reference review](research/workbench/DUAL_BENCHMARK_V1.md)
reads saved results only; it does not rerun or rewrite old accounts.

The [eleventh preregistration](research/workbench/evidence/daily_variance_disagreement_20261028_registration_v1.json)
defines a separate **coarse daily variance-disagreement proxy**, not a
true variance-risk-premium estimate or replication of intraday research.
It tilts the original equity/gold odds by the ratio of native VIX squared
to trailing 21-session SPY daily log-return variance. The stronger
reference's original dollar/BIL scale, 30%--70% equity bounds, relative
stock composition and **monthly** observed-weight execution are preserved.
Daily-proxy predictability is an unproven hypothesis, not a borrowed
literature result.

Registration at 2026-10-28 08:38:21 UTC preceded the 131 new monthly
targets; independent scalar/source-preservation and five prefix checks
passed before the new accounts. Actual preregistration
`352c21ae58bcfd0c5e32e617c958ba45c53a9593` precedes the
[complete eleventh result](research/workbench/evidence/daily_variance_disagreement_20261028_results_v1.json),
finished at 2026-10-28 08:43:02 UTC. It is **rejected overall**:
both five-year paths pass all point gates, but both ten-year QQQ
comparisons fail and ten-year-base drawdown is 15.6588%, above 15%.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.8177% | 1.1696 | 15.6588% | 4.0982 | 617 | 1,050.08 |
| 10y stress | 16.3216% | 1.1295 | 14.6143% | 4.0904 | 603 | 2,293.53 |
| 5y base | 20.9426% | 1.2399 | 12.5061% | 4.4930 | 339 | 508.78 |
| 5y stress | 19.9205% | 1.1716 | 12.4731% | 4.4705 | 332 | 992.14 |

CAGR, Sharpe and SPY pass in all four paths. The original reference has
lower ten-year drawdowns and lower fees, while this proxy has higher
CAGR/Sharpe across all four paths and lower five-year drawdowns. These
tradeoffs do not permit substituting the stress drawdown for the base
path, rounding a failed gate, or claiming independent alpha.
Current totals are eleven completed SSD hypotheses, 44 paths, 189
combined completions and zero qualified champions.

The saved full-window diagnosis locates the ten-year QQQ deficit in
the earlier fixed segment, not in a recomputed account: ten-year-base
net log gap is **-0.5434** over 2016-10-06--2021-10-05 and **+0.1783**
over 2021-10-06--2026-10-05, adding to **-0.3651**. The later segment
of that continuous account is not the independently funded five-year account.

The first 15% breach was 2020-03-19. The already known February target
had halved non-BIL capital to 49%; no pending request, new order or fee
existed on the breach date. Actual prior-close stocks were about 23.70%,
gold 21.15% and BIL 52.99%. The remaining failure cannot simply be
described as an unexecuted exit or a fee-only event. The original strict
gate remains failed; no target, account or parameter was rerun or changed.

The [twelfth preregistration](research/workbench/evidence/symmetric_variance_disagreement_20261028_registration_v1.json)
fixes one symmetric information-set extension: include GLD's realized
daily variance as well as SPY's, while retaining the same 21-session
window, source dollar/BIL scale, bounds, factor proportions and monthly
execution. It does not scan coefficients or claim true variance risk
premia. The original and one-sided laws are recovered in the specified
unit-disagreement cases.

Actual registration at 2026-10-28 09:34:41 UTC preceded 131 independently
checked monthly targets and five prefix checks. Preregistration
`631e0cd1b752feb88ff080796f07cb7c6c1d78da` preceded the
[complete twelfth result](research/workbench/evidence/symmetric_variance_disagreement_20261028_results_v1.json),
finished at 2026-10-28 09:39:28 UTC. It is **rejected overall**:
CAGR, Sharpe and drawdown pass throughout, but both ten-year SPY and
QQQ comparisons fail. Both five-year paths pass all point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 15.3379% | 1.1057 | 13.4553% | 4.4309 | 656 | 1,077.77 |
| 10y stress | 14.8105% | 1.0631 | 12.7279% | 4.4258 | 656 | 2,301.51 |
| 5y base | 19.9740% | 1.2248 | 12.5273% | 4.7045 | 344 | 514.99 |
| 5y stress | 19.0332% | 1.1609 | 12.5200% | 4.6951 | 344 | 1,012.46 |

The symmetric treatment lowers ten-year drawdown relative to the
one-sided proxy, but reduces return and Sharpe and increases costs
in all four paths. It does not create a full-goal winner or justify
interpolating between exposed model results. Current totals are
twelve completed SSD hypotheses, 48 paths, 190 combined completions
and zero qualified champions.

The saved information assessment finds that symmetry lowered equity
targets on **88 of 131** original month ends, raised them on 14 and left
29 unchanged; the mean change was **-4.0487 percentage points of total
target capital**, not an order count. Ten-year-base mean closing stocks
fell to 37.69% and gold rose to 32.67%. The QQQ log gap worsened in both
fixed chronological halves relative to the one-sided model.

The symmetric five-year maximum-drawdown episode moved to
**2022-03-08--2022-10-14**, unlike the one-sided model's early-2026
episode. Similar headline drawdowns do not mean the same loss mechanism.
The saved assessment introduced no new accounts, parameter search or
qualification, and did not rerun the eleventh diagnostic.

The [thirteenth preregistration](research/workbench/evidence/band_projection_20261028_registration_v1.json)
changes execution destination, not the eleventh desired signals or
2.5% band. After an unchanged monthly breach, project actual weights to
the nearest feasible point inside that band and the original 98%
investment budget; initial deployment still uses the source center.
The original fee/funding engine and delayed fills are unchanged.

This minimizes squared weight movement, **not fees**. A retained
synthetic witness shows why a cheaper algebraic fixed-fee solution
cannot simply replace the unchanged engine's actual charge.
Registration at 2026-10-28 10:41:17 UTC binds the original 131 desired
targets and synthetic projection proof before path-dependent issued
targets or accounts. Actual preregistration
`0ac8889c004057a973665bd6ed9abd8c9b5b6ff8` preceded the
[complete thirteenth result](research/workbench/evidence/band_projection_20261028_results_v1.json),
finished at 2026-10-28 10:43:56 UTC. It is **rejected overall**:
both ten-year QQQ comparisons fail and ten-year-base drawdown exceeds
15%; both five-year paths pass their point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.2845% | 1.1353 | 15.8479% | 3.4677 | 648 | 1,004.33 |
| 10y stress | 15.9323% | 1.1067 | 14.7694% | 3.4730 | 654 | 2,055.55 |
| 5y base | 20.3246% | 1.2136 | 12.3105% | 3.8264 | 327 | 469.49 |
| 5y stress | 19.6286% | 1.1682 | 12.7382% | 3.8219 | 327 | 886.59 |

Actual turnover and fees are lower in all four paths than the eleventh
source, but net CAGR and Sharpe are also lower. Ten-year ticket counts
rise despite lower notional turnover. This is not a cost optimum or
net-performance improvement. Counts are thirteen completed SSD hypotheses,
52 paths, 191 combined completions and zero qualified champions.

The saved execution assessment separates fee components rather than
assuming every cost reduction improves returns. Ten-year fixed-ticket
charges rise by **USD31/51**, while proportional charges fall by
**USD76.75/288.98**. Average modeled ticket size falls from about
USD1,404/1,402 to USD1,100/1,072.

An exact ex-post log-linked accounting decomposition finds the largest
negative holding contribution in SOXX on all four paths. Ten-year-base
ending NAV is USD2,115.55 below the source despite USD45.75 less total
fees. Its fee contribution, normalized to each actual account's prior
NAV and timing, is slightly negative rather than positive. This is not
a causal or zero-fee counterfactual, and no account was replayed.

The [fourteenth preregistration](research/workbench/evidence/relative_equity_odds_20261028_registration_v1.json)
tests a distinct **within-equity** construction. Preserve the eleventh
source's total stock, gold and BIL budgets, but allocate SOXX versus the
equal-start four-fund core continuously from their relative gross wealth
over the inherited 126-session window, excluding the latest 21 sessions.
This is not the old binary SOXX/SPY switch or within-core rank allocation.

Actual registration at 2026-10-28 11:07:49 UTC preceded the independent
128-target proof. Three insufficient prehistory months stay undefined;
both full ten/five-year windows retain every return and causal initial
anchor. Actual preregistration
`d141f9caba55f8b1bd96a538e6b8ff260dc73eb4` precedes the
[complete fourteenth result](research/workbench/evidence/relative_equity_odds_20261028_results_v1.json),
finished at 2026-10-28 11:11:18 UTC. It is **rejected overall**:
both ten-year QQQ comparisons fail and ten-year-base drawdown exceeds
15%, despite both five-year paths passing all point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 17.0480% | 1.1710 | 15.6817% | 4.1236 | 630 | 1,069.66 |
| 10y stress | 16.5771% | 1.1341 | 14.6562% | 4.1202 | 622 | 2,340.20 |
| 5y base | 21.1781% | 1.2393 | 12.6506% | 4.5198 | 338 | 509.03 |
| 5y stress | 20.1904% | 1.1761 | 12.6187% | 4.4971 | 338 | 1,003.19 |

CAGR improves modestly relative to the unchanged-budget source,
but drawdown, turnover and costs rise throughout; five-year-base
Sharpe declines slightly. Two historical five-year CAGRs above 20%
do not establish the full joint goal or independent forward performance.
Counts are fourteen completed SSD hypotheses, 56 paths, 192 combined
completions and zero qualified champions.

The saved composition assessment shows a modest semiconductor tilt:
SOXX's share of the unchanged equity budget averages **51.5813%** over
the 128 eligible month ends, above half on 95 dates and below half on 33.
The average shift is **+0.7808 percentage points of total target
capital**, not a change in total equity or an order count.

Ex-post accounting links the positive relative-wealth outcome mainly
to SOXX holding contributions, partly offset by lower core contributions
and fees. It does not establish a causal or forward return advantage.
The three undefined prehistory dates remain excluded from this
descriptive target comparison, not replaced with 50% observations.

The [fifteenth preregistration](research/workbench/evidence/split_session_variation_20261028_registration_v1.json)
tests additional daily-open information, not another share or lookback
choice. Split each of the same 21 completed SPY daily log returns into
previous-close-to-open and open-to-close movements, then sum their
squares before annualizing. Opposite movements no longer cancel before
squaring; same-direction movements can instead lower this proxy.
The source dollar guard, relative stock composition, 131 monthly
decisions, option-odds law, bounds and monthly execution stay unchanged.

Actual registration at 2026-10-28 11:45:21 UTC precedes its independent
target proof. This uses already admitted adjusted model prices, not
high-frequency observations, a range estimator or true variance-risk
premium. At preregistration, fifteen SSD hypotheses are registered,
fourteen completed and one pending; all 56 completed paths and 192
combined completions remained, with zero qualifiers. No new candidate
account had run at that registration.

Actual preregistration
`cc4fc052bfaebace695f1f49668549d41b3c6fc1` preceded the
[complete fifteenth result](research/workbench/evidence/split_session_variation_20261028_results_v1.json),
finished at 2026-10-28 11:52:02 UTC. It is **rejected overall**:
all four CAGR, Sharpe, drawdown and SPY gates pass, but both
ten-year paths still fail QQQ. Both five-year paths pass all point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.9172% | 1.1803 | 14.8656% | 4.1159 | 623 | 1,059.49 |
| 10y stress | 16.3809% | 1.1374 | 13.8994% | 4.1174 | 630 | 2,327.60 |
| 5y base | 20.9599% | 1.2426 | 12.5061% | 4.4246 | 324 | 492.23 |
| 5y stress | 20.0204% | 1.1802 | 12.4731% | 4.4057 | 324 | 979.41 |

Relative to the eleventh close-only study, CAGR and Sharpe increase
slightly on all paths; ten-year drawdowns fall, while five-year
drawdowns are numerically unchanged. Ten-year costs/turnover rise,
whereas five-year costs/turnover fall. This is not a uniform
implementation-efficiency gain.

The broader retained reference also matters: the older observed-band
model still has lower ten-year drawdowns and fewer orders/lower fees
throughout, while the fourteenth model still has higher CAGR throughout.
This result does not dominate every prior construction. Ten-year net
CAGR remains about **4.2452/4.7429 percentage points below QQQ**.
Counts are fifteen completed SSD hypotheses, 60 paths, 193 combined
completions and zero qualified champions; historical point passes
still do not establish independent forward performance.

The saved information assessment finds **64 higher and 67 lower**
two-segment variation observations across all 131 month ends. Stock
targets rise on 45 dates, fall on 41 and remain unchanged on 45;
their average change is only **-0.0693 percentage points of total
target capital**, not a uniform risk reduction or an order count.

Both ten-year maximum-drawdown intervals remain
2020-02-19--2020-03-20. Both five-year intervals remain
2026-01-29--2026-03-30; in those five-year declines the two models have
the same actual weights and normalized NAV path within numerical
precision, with **no orders or fees**. Their dollar wealth is different,
and their full histories are not identical.

The small full-window improvement over the eleventh model has a
positive gold holding component offset by negative semiconductor/core
components in ex-post accounting. It is not causal or standalone gold
alpha. The ten-year QQQ shortfall still comes from earlier losses
outweighing later gains; no strategy, parameter or trial count changes.

A subsequent bounded review examines Cover's
[Universal Portfolios](https://doi.org/10.1111/j.1467-9965.1991.tb00002.x)
as a different construction principle, not another volatility-window
variant. Crossref and the author-hosted scan agree on the 1991 paper;
only its first three pages were OCR-read, not the full proofs.
Its asymptotic growth comparator, and its mentioned finite-sample
lower bounds, do not certify our finite-window net-return or risk gates.

Synthetic witnesses show why: the initial uniform allocation can grow
1.5 times while the best constant portfolio grows 2 times, and a valid
two-asset period can produce 25% drawdown. The unchanged funded engine
also shows that USD1 fixed tickets break simple wealth scaling.
This does not rule out cost-aware adaptations; it prevents transferring
a gross mathematical guarantee to the existing delayed, fee-paying
executor. No sixteenth model, expert grid or historical account was
selected or evaluated by that review. Its counts stayed at 193
completed and zero qualifiers.

The [sixteenth preregistration](research/workbench/evidence/simplex_wealth_average_20261028_registration_v1.json)
now fixes one continuous-weight learning construction. A uniform
three-sleeve prior averages constant mixtures of the equal-start
four-style core, SOXX and GLD by their already observed monthly gross
growth scores. An exact normalized moment recursion avoids a sampled
expert grid or selection of the historical best mixture.

The saved dollar-guard BIL/non-BIL budget and monthly funded executor
remain. The old half-SOXX and 30%--70% stock split are explicitly
replaced as model architecture, not relaxed user risk gates.
Gross learning scores are not net account returns or a universal
performance guarantee; only one actual funded account will trade.

Actual registration at 2026-10-28 12:48:33 UTC precedes all 131 target
proofs. Independent polynomial integration differs by at most
3.34e-15 in target weight; five prefix checks and both complete formal
windows retain all four genuine factor families. Learning starts from
the first completed August 2015 month-end anchor, with the first
matured interval ending September 2015; it does not reset for the
five-year account. At registration sixteen hypotheses were registered,
fifteen completed and one pending; no new candidate account had run.

Actual preregistration
`ec300d591caaf56395b7456d02dec5f5dc1fd405` preceded the
[complete sixteenth result](research/workbench/evidence/simplex_wealth_average_20261028_results_v1.json),
finished at 2026-10-28 12:56:38 UTC. It is **rejected overall**:
both ten-year drawdown and QQQ gates fail, although every CAGR,
Sharpe and SPY gate passes and both five-year paths pass all point gates.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 17.8953% | 1.0907 | 17.0942% | 2.9409 | 241 | 587.16 |
| 10y stress | 18.0751% | 1.1013 | 16.2357% | 2.9684 | 276 | 1,686.77 |
| 5y base | 21.4270% | 1.0969 | 14.6796% | 3.5702 | 158 | 294.14 |
| 5y stress | 21.3807% | 1.0977 | 14.8357% | 3.6005 | 179 | 726.59 |

Relative to the fifteenth model, all four CAGRs rise and turnover,
tickets and fees fall, but Sharpe declines and drawdown increases
throughout. The older observed-band reference also has more tickets
and higher costs on every path, but still has lower drawdowns
throughout; this is not uniform dominance. Higher ten-year stress
return is not evidence that higher fees help: its original fill delay
and path-dependent holdings differ.

Counts are sixteen completed SSD hypotheses, 64 funded paths,
194 combined completions and zero qualifiers. Gross-score learning
did not establish a net-performance theorem or independent forward
proof, and no prior, forgetting rate or bounds are retuned to rescue it.

The saved learning assessment shows a material exposure change:
across 131 paired month ends, stock targets rise on 98 dates and fall
on 33. Their average increase is **7.1282 percentage points of total
target capital**, including **6.0895 points in SOXX**, offset by gold.
Lower costs are therefore not an equal-risk explanation of the return gain.

Both ten-year paths first breach 15% drawdown on **2020-03-16**,
despite already holding over 51% BIL at the previous close. Neither
has a pending request, new order or fee on that breach day.
The five-year maximum-drawdown episode moves to
2022-03-29--2022-10-14, rather than the fifteenth model's 2026 episode.
Saved ex-post accounting associates the gain mainly with a positive
SOXX component and lower fee effects, offset by negative gold
components; it is not causal or standalone alpha.

A subsequent source review distinguishes a proposed Baa--Aaa
credit-quality spread from the older BAA10Y and restricted high-yield
OAS inputs. The new Moody's series pages contain explicit copying
and reuse restrictions; no applicable new permission was verified,
so no historical yield observations were downloaded. This does not
reclassify the old frozen credit study.

The review separately examines Chicago Fed **ANFCI**, an adjusted
financial-conditions index, not a credit-spread substitute. Its
definition, revision caveats and public ALFRED form were checked.
A prospective plan covered 225 prior-session vintage cutoffs in six
bounded batches. At that metadata review, no archives had been
downloaded and no source admission had passed.

The [actual ANFCI validation](research/workbench/evidence/anfci_input_readiness_20261028_v2.json)
has now received six archives covering all 225 requested vintages,
but **full source admission still fails**: 223 queries meet the
unchanged 14-day freshness limit and two do not.
The 2022-12-29 and 2022-12-30 queries contain only the 2022-12-09
observation, aged 20 and 21 days.

A separately frozen technical correction recognizes the source's
native empty CSV fields as missing, not zero or filled values.
The failed dot-only parser and its raw response remain intact; that
first archive was reused without another request. All six unique
data POSTs succeeded, and 84,117 finite cells matched independent
parsing. This does not repair the two stale observations or authorize
a strategy using them. No seventeenth model is selected;
194 completed configurations and zero qualifiers remain.

The [seventeenth preregistration](research/workbench/evidence/anfci_residual_confirmation_20261028_registration_v1.json)
defines a **separate fresh-only decision policy**, not a repair of
ANFCI's failed full-source admission. On the frozen fifteenth
allocation, the original dollar half-risk guard is retained only
when fresh ANFCI is strictly positive. Otherwise the same non-BIL
composition is restored to the original 98% target budget.
This tests macro-adjusted financial tightness rather than tuning the
sixteenth learner; it can increase exposure and worsen drawdown.

Every monthly opportunity requires a reading no more than 14 days
old. Staleness means a logged **no new request**, not zero ANFCI or
a switch to cash. All 131 opportunities remain recorded: 130 have
targets and 2022-12-30 explicitly abstains. Actual existing holdings,
pending fills and every formal return day remain in the account policy.
Unknown integrity errors stop rather than become abstentions.

Actual registration at 2026-10-28 17:37:56 UTC precedes target proof.
Both original anchors, all four genuine factor families and five
prefix checks pass; independent target difference is at most
1.12e-16. Seventeen hypotheses are registered, sixteen completed
and one pending at registration, before any new account had run.

Actual preregistration
`94ff2e2d40e58e8194db06e936722524650b8d08` preceded the
[complete seventeenth result](research/workbench/evidence/anfci_residual_confirmation_20261028_results_v1.json).
It is **rejected overall**: every path exceeds 15% drawdown,
both ten-year QQQ comparisons fail, and ten-year stress Sharpe is
**0.9999262771**, strictly below 1 despite rounding close to it.

| Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 19.1908% | 1.039127 | 22.2677% | 2.6650 | 553 | 812.46 |
| 10y stress | 18.5577% | 0.999926 | 22.4891% | 2.6543 | 541 | 1,545.42 |
| 5y base | 24.5144% | 1.125942 | 19.5913% | 2.7219 | 287 | 377.14 |
| 5y stress | 23.3055% | 1.067553 | 19.9082% | 2.7153 | 287 | 636.35 |

Compared with the fifteenth donor, CAGR rises and fees, orders
and turnover fall throughout, but Sharpe falls and drawdown rises
throughout. Against the sixteenth, turnover is lower but ticket
counts are higher on every path; costs are not uniformly lower.
The older robust reference still has better Sharpe and lower drawdown.

The explicit 2022-12-30 source abstention issued no new request;
that actual return day remains in all four accounts. It does not
repair ANFCI's full-source admission. At that result snapshot, counts
were seventeen completed SSD hypotheses, 68 funded paths,
195 combined completions and zero qualifiers. No eighteenth model
or post-result threshold rescue had yet been selected.

The saved guard diagnosis shows that, among 130 fresh paired
opportunities, **63 original half-risk guards were released**, only
11 retained, and 56 already-full-risk targets stayed unchanged.
Mean non-BIL target capital rises **23.7462 percentage points**,
split between stocks (+13.9667) and gold (+9.7794), with an equal
reduction in BIL. These are eligible target averages, not order counts.

At the first 15% breaches, every path has **zero BIL** at the prior
close, with no pending request, same-day order or fee. The preceding
2020-02-28 and 2022-08-31 decisions are both recorded guard releases.
All observed maximum-drawdown intervals end before the later
2022-12-30 source abstention, so that missing-data event cannot
explain those earlier losses. Lower costs did not preserve the
donor's risk protection.

The [eighteenth registration and preparation failure](research/workbench/evidence/ordered_path_growth_20261028_preparation_failure_v1.json)
records a different price-only construction: maximize past terminal
wealth while constraining every ordered peak-to-later-trough decline
in a constant-unit training portfolio. The 252-return window,
15% training limit, 98% invested budget, existing stock basket and
14.7% minimum stock capital were frozen before any real optimization.
The training bound is not a forecast or actual net-account result.

Actual registration occurred at **2026-10-28 18:40:06 UTC**.
Preparation stopped at **18:40:20 UTC**, before a target-proof
packet or any funded account. A separate, fixed-parameter diagnostic
reproduced the first failure on 2026-04-30: raw GLD weight
`-2.9006786461321336e-10`, with the budget sum exactly `0.98`.
The original solver permits `1e-9` feasibility error, but the
unchanged nonnegative cleanup accepts only `1e-10`; it correctly
refused to emit that negative target.

This is a retained **technical failure**, not an evaluated portfolio
loss or established economic infeasibility. Version 1 is frozen and
will not be retried. Any numerical correction needs a separate
version and publication before accounts. At that failure publication,
no correction had yet been implemented or registered. Counts were **18 registered,
17 completed/reviewed, one technical failure, zero pending,
68 funded paths, 195 combined completions and zero qualifiers**.
The dashboard shows null metrics for the failed preparation, not
zero returns or a fabricated target proof.

The [separate precision-v2 registration](research/workbench/evidence/ordered_path_growth_20261028_registration_v2.json)
keeps that failure and changes only the primal and dual solver
feasibility tolerances from `1e-9` to `1e-10`. It does not widen
the cleanup guard, alter the economic rule or count as another
independent economic hypothesis. Frozen v1 pure helpers are reused;
allocation, target-building and accounting logic remain unchanged.

Actual v2 registration occurred at **2026-10-28 19:38:26 UTC**,
before the complete target proof at **19:39:13 UTC**. All 122
monthly decisions passed the original numerical checks, with zero
independent target difference, five prefix checks and all four style
funds present at both full formal windows. This is target preparation,
not financial-performance evidence or a future drawdown guarantee.
At that target-proof snapshot, counts were **19 registered, 17 completed/reviewed, one retained
technical failure and one pending version**, with 68 funded paths
and 195 combined completions unchanged. No v2 account had run;
actual remote publication was still required before the four accounts.

Actual preregistration commit
`005afed40e43fae63a581a4aa04f365c6b06136e` and the recorded remote
fence at **19:47:57 UTC** preceded all four accounts, which finished
at **19:52:55 UTC**. The [complete precision-v2 result](research/workbench/evidence/ordered_path_growth_20261028_results_v2.json)
is **rejected**: every Sharpe and drawdown gate fails, both ten-year
QQQ comparisons fail, and ten-year stress also fails matched-cost
SPY and the original extra base-cost SPY comparison.

| Path | Net CAGR | Net excess Sharpe | Maximum DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 15.8017% | 0.766248 | 26.4135% | 3.6472x | 259 | 587.51 |
| 10y stress | 15.0863% | 0.730357 | 28.3572% | 3.6344x | 253 | 1,499.90 |
| 5y base | 19.8129% | 0.812902 | 26.6640% | 4.7162x | 163 | 300.27 |
| 5y stress | 19.0616% | 0.780295 | 28.3785% | 4.7162x | 162 | 693.53 |

Every CAGR exceeds 12%, and both five-year paths beat SPY and QQQ,
but no path passes jointly. Against the retained older low-drawdown
reference, CAGR rises only 0.48--1.08 percentage points while Sharpe
falls on every path and drawdown rises 12.84--14.95 points. This
is not an equal-risk gain or causal alpha. The numerical repair
enabled preparation; it did not validate the economic construction.
Current counts are **19 registered, 18 completed/reviewed, one
retained technical failure, zero pending, 72 funded paths,
196 combined completions and zero qualifiers**.

The subsequent saved-risk diagnosis explains why the trailing
constraint should not be read as protection of the actual account:

| Path | First DD above 15% | Prior-close stocks | Prior-close gold | Latest known target | Its saved training DD |
|---|---|---:|---:|---|---:|
| 10y base | 2018-12-19 | 97.9100% | 0% | 2018-11-30 | 13.6143% |
| 10y stress | 2018-12-19 | 97.8620% | 0% | 2018-11-30 | 13.6143% |
| 5y base | 2022-05-02 | 13.5956% | 84.3546% | 2022-04-29 | 7.8731% |
| 5y stress | 2022-02-23 | 97.8296% | 0% | 2022-01-31 | 14.8345% |

All four first breaches have **no pending request at the preceding
close, no order and no fee on the breach day**. BIL is only a small
positive residue, at most about 0.000631% of total capital, not
strictly zero. The five-year base breach occurs in a gold-dominated
portfolio rather than a nearly all-stock one.

All maximum-drawdown episodes begin at the actual account peak
on **2021-12-27**, ending on **2023-11-10** in base and
**2023-10-26** in stress. The trailing training windows used near
those troughs start in September/October 2022, after that account
peak. Both stock and gold mark-to-market losses accumulate in the
same funded account; a locally bounded, normalized training path
does not bound that longer wealth history. This diagnosis uses
only saved artifacts and changes no primary result or trial count.

The subsequent prior-art review found that **actual-account wealth
cushion protection had already been tested** in
[`growth_portfolio_protection`](research/workbench/config/growth-portfolio-protection.json).
Its recorded protected drawdowns are 12.80--13.51%, but net CAGR
is only 6.24--10.64%, with Sharpe 0.479--0.697 and failed original
SPY gates. Those old outcomes and criteria remain unchanged.

The original [Hsieh--Barmish drawdown-modulation paper](https://arxiv.org/abs/1710.01503v1)
provides a useful theoretical comparison, not a ready-made guarantee
for this project. Its bare wealth-cushion factor is already present
in the older code. Its theorem uses explicit return-support bounds
and ideal cost-free, zero-interest dynamics with the selected exposure
applied before the next return. The lemma does **not** require IID
returns; the separate gain-optimization example does. Existing
fees, queued next-open execution and actual BIL exposure are not
covered merely by citing the lemma.

Exact **synthetic unit examples**, not new historical backtests,
show why the timing and cash distinctions matter. No candidate,
gain estimator or replacement strategy was selected, and the
paper's growth optimizer was not replicated. This review prevents
relabelling an already tested wealth-floor rule as new research;
it adds no strategy trial or investment qualification.

The next distinct information direction is **VVIX**, the option-implied
volatility of VIX, rather than another wealth-floor or VIX-horizon
parameter change. Its [separate source validation](research/workbench/evidence/vvix_input_readiness_20261028_v1.json)
completed at **2026-10-28 23:05 UTC**. One advertised historical CSV
provided 5,135 native observations; the existing 9,291-row VIX source
was reused without another request. Independent native parsing agrees
on all **14,426** values, and both series cover every original
**2,805 NYSE context session and 131 month-end query**, without
filling gaps or changing market dates.

Admission is limited to those **archival daily index values**. VVIX
uses standard monthly VIX options for a 30-calendar-day expectation,
not a tradable index return or a certified risk premium. Its published
regular-session calculation runs through 16:15 ET, so it cannot be
used for an earlier 16:00 stock-close trade. Full historical intraday
first-publication, revision and methodology continuity are not
authenticated. At source admission no VVIX ratio, signal, target or
strategy return had been computed; a candidate still required separate
frozen rules and actual registration/publication. Old source records, including their
historical `VVIX_admitted: false`, remain unchanged.

One [VVIX uncertainty-confirmation candidate](research/workbench/evidence/vvix_uncertainty_confirmation_20261029_registration_v1.json)
is now actually registered. At the original 131 month ends it retains
the fifteenth model's dollar-triggered half-risk budget only when the
current native VVIX level is strictly above its level 63 NYSE sessions
earlier. Otherwise it restores the same stock/gold proportions to
98% non-BIL capital. The quarter horizon is fixed, not searched;
equality does not confirm uncertainty, and an already full-risk
source target stays unchanged. This can increase drawdown.

Registration occurred at **2026-10-28T23:58:17.764688Z**, before
real VVIX states and target proof at **23:58:18.695807Z**.
The proof retains 35 of 74 original half-risk guards, releases 39,
and leaves 57 original full-risk targets unchanged. All 131 targets,
262 native quote queries, five prefix checks, original formal anchors
and four style funds are retained. No candidate account had run
at that snapshot; remote publication was still required before the
original four funded paths. Counts then were **20 registered, 18 completed,
one retained technical failure and one pending candidate**, with
72 funded paths and 196 combined completions unchanged.

Preregistration commit `399d0fd8475f9a3630dfaf0aa21a856fdd9474bb`
was actually published, with its remote fence recorded at
**2026-10-29T00:06:54.504696Z**, before any candidate account.
All four paths finished at **00:10:10.763397Z** and the
[complete VVIX outcome](research/workbench/evidence/vvix_uncertainty_confirmation_20261029_results_v1.json)
is **rejected**.

| Path | Net CAGR | Net excess Sharpe | Maximum DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 17.4069% | 1.039144 | 18.9454% | 5.3285x | 656 | 1,178.98 |
| 10y stress | 15.9311% | 0.944764 | 20.5419% | 5.3188x | 656 | 2,600.15 |
| 5y base | 20.8705% | 1.015963 | 19.2029% | 5.4512x | 317 | 498.38 |
| 5y stress | 18.8433% | 0.909462 | 20.7829% | 5.4283x | 317 | 1,006.53 |

Every drawdown gate fails, both stress Sharpe gates fail and both
ten-year QQQ comparisons fail. All CAGR and SPY gates pass, but
even the five-year-base CAGR above 20% is not joint qualification.
Against the preselected fifteenth donor, CAGR rises only on ten-year
base; all four Sharpe values worsen, drawdowns rise, annual turnover
increases by 1.02--1.21 times NAV and costs increase. Current counts
are **20 registered, 19 completed/reviewed, one retained technical
failure, zero pending, 76 funded paths, 197 combined completions
and zero qualifiers**.

The saved guard/cost diagnosis finds **33 source budget switches
versus 44 with VVIX confirmation**. Across the same 131 month-end
intentions, mean non-BIL target capital rises 14.5878 percentage
points: stocks 8.3131 and gold 6.2747. These are target-date
averages, not average daily executed holdings.

Every first 15% breach occurs after the VVIX condition releases an
otherwise active dollar half-guard. New prior-close BIL is zero,
while the source account still holds approximately 49--51% BIL:

| Path | First breach | Last known guard decision | Source DD that day | VVIX DD that day |
|---|---|---|---:|---:|
| 10y base | 2022-09-20 | 2022-08-31 | 9.1501% | 15.1543% |
| 10y stress | 2022-07-05 | 2022-06-30 | 8.6434% | 15.0582% |
| 5y base | 2022-09-15 | 2022-08-31 | 9.1106% | 15.0596% |
| 5y stress | 2022-07-05 | 2022-06-30 | 8.7570% | 15.2665% |

The base breaches have no pending request or breach-day order/fee.
The stress accounts do execute the June 30 request on July 5 after
the original two-session delay, but the VVIX request retains a
full non-BIL budget: it is not an unfilled protective half-exit.
Cost partitioning also finds higher costs on **shared execution
dates**, not only on additional trade days. The analysis reads saved
accounts and does not change the rejection, add trials or claim
causal alpha.

The subsequent [DBC asset-role review](research/workbench/evidence/dbc_asset_role_review_20261029_v1.json)
does **not admit DBC to a new strict-1x portfolio extension**.
Its 4,214-row historical archive already existed in earlier
macro/TAA research and remains unchanged; this is not a new ticker
discovery or a rerun of those strategies. The proposed question was
direct-commodity diversification of gold risk while preserving
core-equity and BIL budgets, not another VVIX parameter change.

The actual issuer-linked 2026 prospectus reports a dated futures
notional allocation of 102.65%, approximately 9% of NAV posted as
collateral, and an index-methodology change effective 2025-11-10.
These are different measures, not a constant leverage factor.
The no-borrowing disclosure and full payment for fund shares do not
by themselves establish the required derivative-exposure mapping.
No 1.0/1.0265 multiplier, new 13-asset panel, portfolio weights or
historical notional reconstruction is adopted. Old DBC research is
not reclassified, and the review does not label every DBC holding
as a 2x product.

The next [registered range-product candidate](research/workbench/evidence/overnight_range_product_20261029_registration_v1.json)
uses additional **already archived SPY high/low fields**, rather
than a new asset or another defense-permission filter. Its
[source proof](research/workbench/evidence/spy_ohlc_range_readiness_20261029_v1.json)
checks all 2,805 original context dates and 16,830 native fields
without replacing the registered opening, closing, volume or RF
series or requesting new quotes.

The fixed 21-session proxy adds the original overnight squared-log
component to a same-day OHLC range-product term, annualized by
252. It retains the fifteenth model's BIL/non-BIL budgets, stock/gold
allocation law and four style families. This explicitly defined
composite is not a full Yang–Zhang implementation or a claim of
unbiased variance/true variance-risk premium; the original publisher
request was blocked and was not retried or mirrored.

Actual registration **229** occurred at
**2026-10-29T03:16:44.121982Z**, before target proof **230**
at **03:16:45.609113Z**. All 131 targets, five prefix checks and
original full-window anchors are retained; independent measurement
is checked before clipping allocation weights. No account metrics
existed at that snapshot; remote publication was still required before
the four funded paths. Counts then were **21 registered, 19 completed,
one retained technical failure and one pending candidate**;
76 funded paths and 197 combined completions are unchanged.

The actual preregistration commit
`08fd9c898ffc993bfa202ad84b990267e74b203e` and its remote fence
at **2026-10-29T03:23:17.846126Z** preceded all four funded paths.
The [complete result](research/workbench/evidence/overnight_range_product_20261029_results_v1.json)
finished at **03:25:50.089090Z** and is **rejected overall**.

| Path | Net CAGR | Net excess Sharpe | Maximum DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---:|---:|---:|---:|---:|---:|
| 10y base | 16.6909% | 1.164101 | 15.2083% | 4.1115x | 637 | 1,067.53 |
| 10y stress | 16.2195% | 1.127461 | 14.1322% | 4.1240x | 651 | 2,338.67 |
| 5y base | 20.9247% | 1.236615 | 12.5061% | 4.4658x | 333 | 502.14 |
| 5y stress | 19.9845% | 1.175988 | 13.0814% | 4.4697x | 340 | 1,001.76 |

Both five-year paths pass their historical point gates, but both
ten-year QQQ comparisons fail and ten-year-base drawdown
**15.208306334934574%** strictly exceeds 15%. All CAGR, Sharpe
and SPY gates pass; this is still not four-path qualification.
Five-year stress remains below the ultimate 20% level,
not “above 20%” after rounding.

Against the preselected fifteenth donor, all four CAGR and Sharpe
values are lower and costs are higher. Five-year-base DD is
numerically unchanged; the other drawdowns increase. Current counts
are **21 registered, 20 completed/reviewed, one retained technical
failure, zero pending, 80 funded paths, 198 combined completions
and zero qualifiers**. No measurement coefficient or window was
changed to rescue the result.

The saved-only measurement diagnosis finds that all **131** proxy
values differ from the donor: 77 are higher and 54 lower, while
the overnight component is identical. Allocation bounds mask
35 differences—32 shared upper-bound and three shared lower-bound
targets—but **96 target rows do change**.

The sole 15% breach is ten-year base on **2020-03-20**, with the same
2020-02-19 peak as the donor. Both accounts still hold approximately
53% BIL beforehand, with no breach-day orders, fees or pending
request. Although the newer February 28 target has less equity,
its January 31 target before the peak had **4.5665 percentage points
more stock capital**. Current defensive holdings therefore do not
erase differences in the earlier wealth path.

On shared execution dates, ticket counts are identical in all four
pairs; net additional tickets come from different execution dates.
This is not the prior VVIX defense-release mechanism, nor proof that
breach-day fees caused the drawdown. No strategy was rerun and
the joint rejection remains unchanged.

The new [paired core-activity study](research/workbench/evidence/core_activity_momentum_20261029_registration_v1.json)
compares time-uniform mean log momentum with a version weighted
by each fund's own reported close-times-volume activity. It changes
only the four-style allocation inside the fifteenth model's existing
core budget. SOXX, GLD, BIL and every other noncore target remain
exact; the fixed 35/30/20/15 rank shares are inherited rather than fitted.

The [activity-source proof](research/workbench/evidence/core_activity_source_readiness_20261029_v1.json)
checks 44,880 native fields across four existing 2,805-session fund
histories. Reported volume matches exactly and no split is reported
in the retained context. This is not fund flow, actual traded-dollar
notional or constituent-stock turnover, and no new quotes were requested.

Both candidates were actually registered as **241/242** at
**2026-10-29T04:55:54.816435Z / .915319Z**, before any new
score or target. Proofs **243/244** follow at
**04:55:58.135417Z / .234466Z**. The original 126/21 window
leaves 128 eligible month ends and three undefined prehistory months;
all formal ten/five-year returns remain. No candidate account had
run at that snapshot: remote publication still had to precede all eight paths.
Counts then were **23 registered, 20 completed, one retained technical
failure, two pending, 80 funded paths and 198 combined completions**.

The [paired results](research/workbench/evidence/core_activity_momentum_20261029_results_v1.json)
are now complete. The actual `d35968631a624e1b9017c07608cce59ecf36d940`
publication and **2026-10-29T05:01:56.208367Z** fence preceded
all eight funded paths, which finished at **05:05:15.467939Z**.
**Both configurations are rejected:** every CAGR, Sharpe, drawdown
and SPY gate passes, but both configurations miss QQQ in both
ten-year paths. All four five-year cells pass their historical point
gates; they do not qualify either complete four-path configuration.

| Method | Path | Net CAGR | Sharpe | Max DD | Annual two-sided turnover | Model orders | Costs USD |
|---|---|---:|---:|---:|---:|---:|---:|
| Price control | 10y base | 17.0081% | 1.178584 | 14.9791% | 4.2534x | 649 | 1,098.87 |
| Price control | 10y stress | 16.4328% | 1.132890 | 14.0511% | 4.2553x | 656 | 2,402.38 |
| Price control | 5y base | 21.0873% | 1.243648 | 12.5565% | 4.5672x | 337 | 509.82 |
| Price control | 5y stress | 20.1328% | 1.180197 | 12.5305% | 4.5507x | 337 | 1,010.92 |
| Activity | 10y base | 16.9487% | 1.179328 | 14.9698% | 4.2732x | 661 | 1,112.37 |
| Activity | 10y stress | 16.3875% | 1.134337 | 13.9940% | 4.2783x | 668 | 2,421.75 |
| Activity | 5y base | 21.0051% | 1.243296 | 12.2902% | 4.5620x | 337 | 510.24 |
| Activity | 5y stress | 20.0548% | 1.179636 | 12.2553% | 4.5449x | 337 | 1,012.13 |

Activity weighting reduces DD but also reduces CAGR and increases
costs in every paired path. Its Sharpe improves slightly in ten-year
paths and declines slightly in five-year paths. Small growth gains
over the equal-core donor cannot be credited to activity alone:
the paired price-only control has the higher CAGR throughout.
Current totals are **23 registered, 22 completed/reviewed, one
retained technical failure, zero pending, 88 funded paths,
200 combined completions and zero qualifiers**.

The saved-only ranking diagnosis finds **109 different complete
rankings out of 128 month ends**, but only 58 different top choices.
Activity weighting shifts average target capital away from MTUM
and QUAL toward USMV and VLUE. The paired full-window accounting
shows negative incremental MTUM/QUAL log-return terms and
positive USMV/VLUE terms in all four paths; this is not causal alpha.

All four five-year accounts have their worst drawdown over
**2026-01-29--2026-03-30**, with **no orders or fees during that
interval**. The activity portfolios already hold more USMV and
less QUAL, so the lower drawdown must not be presented as a new
February trade or successful within-episode timing.

Shared-date ticket counts are identical in every pair. Different
execution dates explain the extra ten-year tickets. In five-year
paths, slightly higher total dollar fees coexist with a slightly
positive normalized fee-return difference because payment dates
and each account's prior NAV differ. These measures are not
interchangeable; the full strategy result remains rejected.

Current counts and the distinction from old quota records are in
[`docs/research_progress.json`](docs/research_progress.json).

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

At the October 7 stage, the primary objective was net excess-return Sharpe above one
and higher net CAGR than SPY, using the same rule in both original ten-year and
five-year windows and in both base and stressed execution. The earlier 20%
CAGR and 15% drawdown thresholds remain separately reported. QLD variants use
real embedded daily-reset leverage, not borrowing or synthetic pre-inception
prices. No result is promoted to live trading authority.

The second round adds six separately registered covariance/diversification
and trend-filter configurations. It was designed after the first results,
so its adaptive research status is explicitly disclosed in
`config/allocation-research.json`. It does not modify the first eight rules.
All six second-round configurations fail the joint primary goal; their
[complete results](research/workbench/evidence/allocation_round_20261007_results.json)
remain available. A third, separately registered six-configuration round
tests quarterly or target-band rebalancing, mechanically fixed weights and
equal mixtures of the already disclosed allocation components.

The published third-round registration can be used directly. Run from
`research/workbench` using its isolated environment:

```bash
.venv/bin/python -I -B -m pytest tests/test_factor_research.py -q
.venv/bin/python -I -B -m us_quant.factor_research stage-data \
  --policy config/implementation-research.json \
  --registration evidence/implementation_round_20261007_registration.json \
  --source-base /path/to/verified/dual-horizon/market \
  --source-supplement /path/to/verified/dual-horizon/legacy-supplement
.venv/bin/python -I -B -m us_quant.factor_research evaluate \
  --policy config/implementation-research.json \
  --registration evidence/implementation_round_20261007_registration.json \
  --output reports/implementation-round-20261007
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
reproduction uses commit `3656e098`, and second-round reproduction uses
`56d95478`; changing a recorded hash to accept a later implementation is
not a valid reproduction. New experiments use the
`register` stage with a new policy and a new registration path.

The [third-round results](research/workbench/evidence/implementation_round_20261007_results.json)
retain all six implementation variants. Two pass the registered latest
two-metric target in both horizons and both base/stress scenarios:
`growth_gold_target_change_band` and
`equal_growth_gold_min_variance_ensemble`. They do not satisfy the earlier
15% drawdown cap and are not independently validated future strategies.

An independently specified validation plan checks every retained accounting
path and every success flag, then reports rolling endpoint sensitivity and
50bp/three-session execution diagnostics for **both** qualified candidates:

```bash
.venv/bin/python -I -B -m pytest tests/test_factor_validation.py -q
.venv/bin/python -I -B -m us_quant.factor_validation
```

This requires the retained local ledgers from all three implementation
revisions. It uses `bt` equity and independent `ffn`/excess-return statistics;
it does not change strategy parameters. Its stronger scenarios and all
failures are additional diagnostics, not substituted primary windows.

The [completed independent validation](research/workbench/evidence/factor_validation_20261007.json)
confirms all 20 new configurations, 138 original accounting paths and 24
additional paths. All 72 registered configurations, including the original
archive, remain disclosed. The latest requested **historical** two-metric
thresholds are met, but selection-adjusted alpha and future performance are
not established.

The stronger candidate is `equal_growth_gold_min_variance_ensemble`:
half of the 98% invested budget follows the QLD/GLD inverse-volatility
allocation; half follows the fixed six-month momentum selection and
minimum-variance allocation among QLD, GLD, TLT, IEF and DBC. Remaining
defensive allocation is actual BIL, with 2% idle cash. Targets combine before
trading one account. There is no account borrowing, but QLD has embedded
daily leverage; the observed approximate underlying gross exposure can
exceed one.

| Window/scenario | Net CAGR | Excess Sharpe | Maximum drawdown | Base-cost SPY CAGR |
| --- | ---: | ---: | ---: | ---: |
| Ten years, base | 20.37% | 1.1870 | 19.97% | 15.46% |
| Five years, base | 22.14% | 1.1045 | 20.06% | 14.04% |
| Ten years, 20bp/two-session stress | 19.58% | 1.1373 | 19.68% | 15.46% |
| Five years, 20bp/two-session stress | 20.59% | 1.0160 | 19.78% | 14.04% |

These are USD10,000 historical accounts ending October 5, 2026, before
personal tax, with fractional units, modeled costs and USD1 per order.
Base execution uses 5bp per side at the next session open. **Limitations are
material:** at 50bp per side the candidate's five-year Sharpe falls to
0.9674; only 19/45 rolling ten-year and 53/105 rolling five-year stress
windows meet the primary goal. Its old 15% drawdown target fails, and
conditional excess-CAGR confidence intervals include zero. The target-band
alternative also has rolling-window and stronger-stress failures. No live
signal, order, Slack post or restart of paused forward records resulted.

The [first-round evidence](research/workbench/evidence/factor_round_20261007_results.json)
contains all eight configurations. The growth/gold inverse-volatility control
passes the latest two primary gates in the base ten/five-year windows, but
its ten-year stressed Sharpe is 0.9913 and drawdown is approximately 25%.
It is **not** a base-plus-stress qualified result and fails the old drawdown
goal. The first non-overlapping five-year diagnostic also fails; no forward
performance or statistically established alpha is claimed.

## Portfolio-level refinement

The next bounded round preserves all earlier rules and tests six separately
registered portfolio controls in
`research/workbench/config/portfolio-refinement.json`: a 30% aggregate QLD
target cap, a 12% ex-ante portfolio-volatility ceiling, a 2.5-percentage-point
actual-holdings no-trade band, and the declared combinations. All controls
operate on the combined portfolio rather than separately on its two sleeves.
Reductions move to actual BIL, not fictitious cash interest.

The band compares current closing holdings with the desired target, unlike
the earlier target-change-only experiment. A breached risk limit overrides
the band at the monthly decision. This does **not** guarantee a continuous
weight limit or maximum drawdown: between-decision drift and opening gaps
remain possible. Fractional units, modeled costs and pre-tax returns are
unchanged assumptions; no broker account is accessed.

This round raises the disclosed count from 72 to 78. The unchanged
incumbent is also replayed, but is not counted as another independent trial.
The primary ten/five-year Sharpe and SPY conditions stay unchanged, including
20bp/delayed stress. To qualify as a research improvement, a new configuration
must additionally either satisfy the old 15% drawdown goal in all primary
paths, or pass both 50bp windows with at least 10% lower turnover. It may
not worsen drawdown by more than one percentage point in any primary path.
No rule is automatically promoted to trading or substituted for the incumbent.

From `research/workbench`, after preserving the original local ledgers:

```bash
.venv/bin/python -I -B -m pytest tests/test_portfolio_refinement.py -q
.venv/bin/python -I -B -m us_quant.portfolio_refinement evaluate
```

The evaluator checks all decisions through the original cash-funded simulator
and independent `bt` accounting, recomputes metrics, and requires the unchanged
control to reproduce the complete original ledgers. It retains every new
failure and conditionally adds rolling-window comparisons for qualified
improvements. The original snapshot remains dated October 5; another
historical experiment is not a new independent forward observation.

The [completed six-configuration evidence](research/workbench/evidence/portfolio_refinement_20261007_results.json)
retains every result and 72 independently audited paths, including six
full-ledger matches to the original incumbent. All six variants pass the
base-only ten/five-year primary target; two retain it under stress, but
**none qualifies as an upgrade under the frozen promotion contract**.
The incumbent remains unchanged.

The actual-holdings band gives a modest improvement, not a new validated
strategy. Its five-year results illustrate why order count and dollar
turnover must not be confused:

| Five-year measure | Unchanged incumbent | Actual-holdings 2.5pp band |
| --- | ---: | ---: |
| Base net CAGR | 22.14% | 22.33% |
| Base excess Sharpe | 1.1045 | 1.1118 |
| Base maximum drawdown | 20.06% | 19.78% |
| Base order tickets | 167 | 94 |
| 20bp delayed-stress Sharpe | 1.0160 | 1.0417 |
| 50bp delayed-stress Sharpe | 0.9674 | 0.9957 |
| 50bp annualized turnover | 2.5121 | 2.3860 |

The band removes about 44% of order tickets but only about 5% of traded-dollar
turnover. Sharpe 0.9957 is **below**, not equal to or above, the strict threshold.
The 12% volatility variants reduce some drawdowns to approximately 18%, not
15%, and increase turnover; all fail the five-year stressed Sharpe gate.
These results do not justify tuning the band until it crosses one, changing
cost assumptions, or treating the reused history as independent evidence.
There is no automatic baseline replacement, paper order or persistent
research/notification automation created by this round.

## Unleveraged multifactor stability research

The October 10 request prioritizes genuine multifactor exposure and lower
risk. The new research lane uses four actual, unleveraged US equity factor
ETFs: MTUM (momentum), VLUE (value), QUAL (quality) and USMV (low volatility).
Issuer URLs, economic definitions, inception dates and current benchmark
labels are recorded in
`research/workbench/config/multifactor-stability.json`.

This is explicitly a **multifactor ETF-sleeve portfolio, not direct
stock-level fundamental scoring**. Value and quality exposure comes through
the funds' actual historical portfolios. Today's holdings or financial ratios
are not used to manufacture historical selections. Existing SEC filing-date
utilities do not provide complete historical security identities, constituent
membership and delisted prices; that missing evidence still blocks a qualified
direct-stock backtest. Current issuer definitions do not establish unchanged
index rules throughout the fund's history.

Six fixed configurations compare equal factor exposure, a 60/20/20
factor-equity/IEF/GLD allocation, bounded inverse-volatility factor shares,
common equity-breadth defense, and a 10% portfolio-volatility target.
There is no QLD, leveraged product, shorting or account borrowing. The
four-factor equity control deliberately has no bond/gold defense: it tests
whether adding equity factors alone actually reduces risk.
Every nonzero equity allocation retains all four factor sleeves; defense
scales them together rather than selecting the historical winner.

The stability criterion requires maximum drawdown at most 15% and both lower
drawdown and lower volatility than the prior QLD-containing candidate in
the same ten/five-year base/stress windows. Sharpe above one and outperformance
of SPY remain separately reported; a lower-risk result cannot be silently
declared to satisfy those return objectives. Higher-cost and rolling-window
failures must remain visible. Correlations, market betas and the covariance
effective dimension are diagnostics, not a claim that four funds produce
four independent alpha sources or avoid overlapping stock holdings.

The registration preserves the previous 78 configurations and records six
new ones, for 84 total. From `research/workbench`:

```bash
.venv/bin/python -I -B -m pytest tests/test_multifactor_stability.py -q
.venv/bin/python -I -B -m us_quant.multifactor_stability fetch
.venv/bin/python -I -B -m us_quant.multifactor_stability evaluate
```

The initial 2014 warmup request was rejected before any candidate evaluation:
VLUE contains 27 flat zero-volume bars, the last on August 7, 2015. The
[quality audit](research/workbench/evidence/multifactor_data_quality_20261010.json)
and original registration remain preserved. Registration v2 starts all series
on August 10, 2015, the next real session, without inventing volume or changing
any candidate rule or either formal return window. There are still 290
sessions through the last pre-evaluation month end, enough for the registered
252-session warmup. Available rolling ten-year windows are consequently few;
this reduced diagnostic coverage must not be concealed.

The common comparison endpoint remains October 5, 2026. Original benchmark,
defensive-asset and risk-free snapshot hashes are preserved; the different
factor-fund download vintage is explicitly recorded. Raw prices and issuer
documents remain ignored local inputs. No old strategy is overwritten and
no trading, notification or persistent automation is activated.

The [six complete multifactor results](research/workbench/evidence/multifactor_stability_20261010_results.json)
contain 77 independently accounted paths. Two defense variants satisfy the
registered stability criteria, but **none satisfies the unchanged Sharpe/SPY
return goals**, and none qualifies for a full strategy upgrade.
The lowest worst-drawdown candidate selected by the frozen rule is
`four_factor_bounded_defense_vol10`:

| Window / base costs | Net CAGR | Excess Sharpe | Maximum drawdown | Annual volatility |
| --- | ---: | ---: | ---: | ---: |
| Ten years, prior QLD-containing comparison | 20.37% | 1.1870 | 19.97% | 14.48% |
| Ten years, unleveraged four-factor defense | 6.90% | 0.6063 | 13.93% | 7.42% |
| Five years, prior QLD-containing comparison | 22.14% | 1.1045 | 20.06% | 15.86% |
| Five years, unleveraged four-factor defense | 7.04% | 0.4374 | 13.05% | 7.46% |

This is a lower-risk research prototype, **not a recommendation to replace
the old strategy**. Its defensive assets, lower equity exposure and absence
of embedded leverage matter materially. The all-equity four-factor control
still suffers 34.38% ten-year maximum drawdown. The four factor funds'
ten-year daily-return correlations range from 0.736 to 0.894; their covariance
effective dimension is about 1.31, not four independent return sources.
These diagnostics do not measure historical constituent-level overlap.
There are only two complete rolling ten-year endpoints and 62 five-year
endpoints; overlapping windows are not independent confirmations.

A separately frozen
[attribution plan](research/workbench/evidence/multifactor_attribution_20261010_plan.json)
audits all original paths and replaces the four factor funds with SPY at
exactly the same aggregate equity target, preserving IEF/GLD/BIL allocations,
decision dates and costs. It includes both stability-qualified candidates,
not just the most favorable one:

```bash
.venv/bin/python -I -B -m us_quant.multifactor_attribution evaluate \
  --output reports/multifactor-attribution-20261010-replay
```

This is an equity-budget ablation, not equal-risk matching or an independent
SPY timing strategy. It distinguishes the contribution of factor fund
selection from simply reducing stock exposure; no candidate is retuned.

The [completed attribution audit](research/workbench/evidence/multifactor_attribution_20261010.json)
recomputes all 77 original paths, reproduces all six complete target matrices
and checks 12 budget-matched SPY control paths. The first attribution command
was interrupted; its partial output was not treated as a result. The unchanged
plan was rerun into the new `-replay` directory without overwriting evidence.

In **all 12 comparisons**, the factor portfolio has lower net CAGR and lower
Sharpe than its equity-budget-matched SPY control. For the selected low-risk
prototype under base costs:

| Window | Four-factor CAGR | Same-equity-budget SPY CAGR | Four-factor drawdown | Control drawdown |
| --- | ---: | ---: | ---: | ---: |
| Ten years | 6.90% | 7.54% | 13.93% | 13.82% |
| Five years | 7.04% | 7.82% | 13.05% | 13.16% |

This comparison includes actual differences in modeled order costs from
holding four equity funds rather than one. It does not prove statistical
underperformance or match market beta exactly, but it **does not establish
incremental net value from the factor-fund selection**. Lower equity exposure,
defensive allocation and removal of embedded leverage must not be presented
as newly discovered factor alpha. The research direction remains unleveraged
multifactor stability; the old leveraged strategy is preserved only as a
historical comparison, not promoted as a suitable solution to the new risk
preference. No replacement or live recommendation is authorized by these
results.

## Recurring factor discovery and research versions

The recurring program in `research/workbench/config/research-program.json`
separates new factor **definitions**, registered strategy **configurations**,
audited historical results and actual independent forward evidence. Its
weekly cadence is Saturday 09:00 Asia/Shanghai, with at most two new economic
definitions per cycle. The Copilot session automation performs the literature
and implementation work; the CLI below enforces the journal and review
boundaries. The CLI does not independently invent factors, download arbitrary
code or execute commands stored in a proposal.

The original four studied factor families are retained as references.
This first new discovery batch proposes cash-flow accrual quality and
conservative asset growth. Cash-flow accrual quality is related to the
already studied earnings-quality family, not a new independent source of
alpha. `fundamental_signals.py` reuses the filing-time, period-matched
financial helpers to compute both features without changing the frozen
imported module. Unit fixtures establish calculation behavior, not profitable
stock selection.

Actual public SEC Company Facts requests returned HTTP 403. The
[source availability evidence](research/workbench/evidence/factor_program_source_availability_20261010.json)
and [readiness record](research/workbench/evidence/research_program_data_readiness_20261010.json)
also disclose the missing historical security master, constituent membership,
delisting returns and matched stock prices. A two-company sample, current
stock list or fabricated data must not substitute for these requirements.
No new stock strategy has been evaluated from the new proposals.

From the isolated `research/workbench` environment:

```bash
# Initialize once; existing ledgers cannot be reset with this command.
.venv/bin/python -I -B -m us_quant.research_program init
.venv/bin/python -I -B -m us_quant.research_program cycle \
  --proposals config/factor-discovery-20261010.json \
  --readiness evidence/research_program_data_readiness_20261010.json
.venv/bin/python -I -B -m us_quant.research_program status
```

The private, ignored SQLite journal is `runtime/research-program.sqlite3`.
Transactions and ISO-week keys prevent concurrent/restarted cycles from
registering duplicates. The event hash chain, policy/engine binding and
reconstructed factor/candidate/review records detect altered or removed
history. Reusing a factor under another name does not create a new trial.
Proposals and data-blocked cycles do not increment the 84 previously
evaluated configurations. Readiness must be checked within eight days;
verified capabilities require hashed evidence, not a missing-data default.
These readiness attestations still require source/coverage review: a hash
does not itself establish point-in-time completeness.

Once data is genuinely available, `register-candidate --candidate <spec.json>
--readiness <readiness.json>` freezes at least three economic factor families,
code/configuration hashes, actual market panels, instrument leverage and the
evaluation cutoff **before** testing. A candidate specification supplies
`id`, `factor_ids`, `frozen_files`, `asset_leverage`, `evaluation_as_of`,
`market`, `history_status: exposed_history_not_independent_holdout`,
`leveraged_products_allowed: false` and `order_authority: false`.
`market` holds `open`, `close`, `raw_close`, `volume`, and `risk_free`
CSV references, each with a workbench-relative `path` and `sha256`.

`review --candidate-id <id> --bundle <bundle.json>` does not trust a submitted
Sharpe number or a success boolean. The bundle repeats the frozen market,
cutoff and candidate-spec hash and supplies four `paths`: ten/five years
under base/stress conditions. Each path includes hashed strategy, `strategy_bt`,
SPY, `spy_bt`, weights and target CSVs, plus the prescribed capital, costs,
commission and delay. The reviewer reexecutes the original simulator and
independent `bt` engine from market data/targets, reconstructs metrics and
weights, and checks the exact date ranges. Only net excess Sharpe above one,
drawdown at most 15% and outperformance of SPY in all four paths can qualify.
Stressed strategies must also exceed base-cost SPY.

Qualified results may advance the **research** version ranked by worst-path
Sharpe; rejected versions stay recorded. This is not trading promotion.
The program never enables brokerage submission, automatically deploys a
live strategy, invents forward observations or resumes old paused ledgers.
The recorded 63-session forward-review requirement is a future minimum,
not evidence already collected and not proof of long-run performance.
Versioned `--export` snapshots cannot overwrite changed earlier exports.

The app and host must be available for session automation to run; a saved
schedule is not proof of an actual future execution. Integrity errors must
pause automation for review; supplier access failures are recorded as data
blockers, never bypassed with fake prices or silently relaxed objectives.

The native weekly automation is now configured and read back successfully.
The [schedule receipt](research/workbench/evidence/research_program_automation_20261010.json)
records the next invocation observed at configuration time: **October 17,
2026, approximately 09:00 Asia/Shanghai**. It wakes this same session rather
than modifying the old Slack/Discord services. Pending registered candidates
are resumed under their original frozen inputs, not discarded or counted again.

The [first real cycle](research/workbench/evidence/research_program_2026-W41.json)
registered two new factor definitions and recorded five explicit stock-data
blockers. Together with four prior definitions, the catalog has six definitions
in five economic families; it does not prove five independent sources of
alpha. The number of evaluated strategy configurations is still **84**, with
zero new evaluations, zero qualified research versions and no research
champion. Repeating the same week leaves the event chain and counts unchanged.
**The requested Sharpe/robustness objective remains unfulfilled.** Neither
working automation nor passing synthetic tests constitutes investment evidence.

### Separate actual-fund data scope

The recurring reviewer also supports the explicitly labeled
`factor_etf_portfolio` scope. It validates the original four-factor fund
mandates, actual post-inception history, corporate-action evidence and
unleveraged product identities by loading the frozen multifactor provider
snapshot. Candidate panels must match that snapshot. This is **not** permission
to mark missing stock fundamentals, historical constituents or delistings
as available: the `direct_stock` readiness requirements remain unchanged.
Candidates and reviews record their scope, and an ETF candidate must use the
four actual fund-factor references rather than claiming direct stock signals.

Existing ledgers refuse a changed engine unless an explicit `migrate-engine`
operation supplies the exact previous engine hash and event-chain head.
The migration verifies existing history in a transaction, appends an engine
transition event and preserves the policy, all factors, blocked cycles,
candidates and reviews. It does not reset trial counts or change performance
or live-deployment gates. Migrations require a reviewed code change, not
automatic recovery from unexplained corruption.

The next fixed two-configuration study is in
`research/workbench/config/factor-gold-risk.json`. It preserves equal exposure
to the four factor funds within an equity sleeve and balances that sleeve
against GLD using trailing 63-session volatility. The two variants compare
monthly rebalancing with an additional daily 12% volatility target and
5-percentage-point trade band. Neither uses QLD, leverage or fixed bond
duration. Monthly requests are retained during modeled execution delays;
pending targets cannot be overwritten. A volatility target is not a loss
guarantee, and two allocation variants are not two newly discovered factors.

From the workbench, after the explicit engine migration:

```bash
.venv/bin/python -I -B -m us_quant.factor_gold_risk prepare
.venv/bin/python -I -B -m us_quant.factor_gold_risk register
.venv/bin/python -I -B -m us_quant.factor_gold_risk evaluate
```

Preparation verifies and freezes real inputs but does not compute strategy
performance. Registration goes through the actual recurring program before
evaluation. Outcomes go through its market/target/cost and independent `bt`
replay, not a parallel success flag. Already registered or reviewed work is
preserved and resumed without duplicating trials.

The explicit migration and both preregistrations are recorded in
[the scope snapshot](research/workbench/evidence/research_program_2026-W41_scope_v2.json)
and [candidate receipts](research/workbench/evidence/factor_gold_risk_20261010_registration.json).
The [actual evaluated outcomes](research/workbench/evidence/factor_gold_risk_20261010_results.json)
were submitted through the recurring reviewer, which regenerated all strategy
and SPY paths from market data/targets and independent `bt` accounting.
Both candidates were recorded as `rejected_historical`:

| Rule | Window | Base net CAGR | Base Sharpe | Base drawdown | Stress Sharpe |
| --- | --- | ---: | ---: | ---: | ---: |
| Four-factor/gold monthly risk balance | Ten years | 13.19% | 0.9055 | 20.20% | 0.8776 |
| Four-factor/gold monthly risk balance | Five years | 16.10% | 0.9602 | 18.22% | 0.9200 |
| Same baseline with daily volatility control | Ten years | 11.17% | 0.8367 | 18.19% | 0.7788 |
| Same baseline with daily volatility control | Five years | 13.96% | 0.8896 | 18.35% | 0.8523 |

The [updated program snapshot](research/workbench/evidence/research_program_2026-W41_results_v3.json)
has 86 evaluated configurations, two new completed strategy reviews, and
still no qualifying research champion or version update. The two new
fundamental-factor proposals and all five direct-stock data blockers are
unchanged. ETF research is no longer blocked merely because stock fundamentals
are missing, but working data access does not make either strategy successful.
Weekly automation instructions now distinguish both scopes and retain every
failure; no performance threshold, old ledger or trading permission changed.

### Factor implementation replication

The next controlled comparison changes fund implementations rather than
searching more allocation parameters. The initial intent attempted replacing
MTUM with SPMO and QUAL with SPHQ. Before any candidate performance calculation,
the provider's SPMO history failed the data gate: 238 zero-volume records,
including the formal ten-year interval. The
[source-quality audit](research/workbench/evidence/factor_replication_quality_audit_20261010.json)
and initial intent commit `be2aaaaf` remain preserved. These two initially
planned variants were not admitted as complete strategy backtests.

The data-only revision replaces QUAL quality with SPHQ and retains MTUM
momentum, VLUE value and USMV low volatility. It does not fill volume, shorten
the formal windows, or change the allocation rules after seeing outcomes.
The two variants retain the preceding study's exact monthly risk balance
and optional daily volatility control. Identical synthetic return inputs
produce identical complete targets and accounting in both implementations.
This is two new configurations of existing factor families, not a claim
of two new independent factors.

The current Invesco product pages identify SPMO and SPHQ. The old SPVU
link resolves to a Concentrated QVM product, so it is not used as a stable
value implementation. The public factsheet links returned HTML rather
than PDF; their contents are not claimed as directly verified. Current
product names and actual fund returns do not establish unchanged historical
index methodology. The protocol records the reported SPHQ benchmark change
and starts input history in July 2016; the original ten/five-year return
windows stay unchanged.

Definitions and limitations are in
`research/workbench/config/factor-implementation-replication.json`.
Before calculating returns, the source fingerprints and candidate receipts
must be recorded through the same recurring program:

```bash
.venv/bin/python -I -B -m us_quant.factor_replication fetch
.venv/bin/python -I -B -m us_quant.factor_replication prepare
.venv/bin/python -I -B -m us_quant.factor_replication register
.venv/bin/python -I -B -m us_quant.factor_replication evaluate
```

The adapter revalidates SPHQ's actual history and retains the previously
hashed momentum/value/low-volatility, defensive and benchmark data.
Old source and candidate modules remain byte-preserved; the small separate
replication runner avoids invalidating those frozen implementation hashes
and is tested against the original algorithm. No name substitution is
made in the actual market data or evaluated holdings.

The [quality implementation results](research/workbench/evidence/factor_replication_20261010_results.json)
are actual recurring-program reviews, not just standalone backtests.
Both fixed variants were rejected. Monthly risk balance has ten/five-year
base Sharpe 0.9054/0.9627 and stressed Sharpe 0.8780/0.9237; daily risk control
has base Sharpe 0.8397/0.8979 and stressed Sharpe 0.7869/0.8666. Drawdown remains
approximately 18%-20%. Changing the quality implementation did not solve the
target shortfall.

The [fifth weekly snapshot](research/workbench/evidence/research_program_2026-W41_results_v5.json)
retains 88 evaluated configurations and four rejected program candidates.
The source-rejected SPMO plans are separately disclosed but are not counted as
completed strategy evaluations. There is still no qualified research champion,
live deployment or new independent forward evidence.

### Causal adaptive factor-combination comparison

The next two fixed hypotheses use the original verified four factor funds
plus GLD and BIL. One control keeps 70% of the invested budget in equal factor
equities and 30% in gold. The adaptive method estimates trailing excess means
with a fixed exponential half-life, shrinks the mean/covariance estimates and
solves a constrained mean-variance problem with an explicit turnover penalty.
It updates only at completed month ends from already observed data.

The protocol is `research/workbench/config/adaptive-factor-allocation.json`.
Every factor fund retains at least 2.5% and at most 25% of total capital;
gold is capped at 50%, total investment at 98%, and no leverage is allowed.
All estimation windows, bounds and penalties are fixed before results.
The implementation uses an epigraph for the absolute turnover cost and
checks solver success and final bounds; it does not return equal weights
as a hidden fallback on optimization failure.
This is adaptive allocation of existing exposures, not new economic-factor
discovery and not a replication of Bayesian Dynamic Model Averaging.

```bash
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation prepare
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation register
.venv/bin/python -I -B -m us_quant.adaptive_factor_allocation evaluate
```

Both configurations must be registered in the recurring program before
evaluation and meet its unchanged actual-cost, independent-accounting and
dual-horizon performance gates. No historical data is relabeled as a new
independent holdout.

The [adaptive comparison results](research/workbench/evidence/adaptive_factors_20261010_results.json)
record two actual rejections. The static control has ten/five-year base
Sharpe 0.8367/0.8042. Adaptive allocation has base Sharpe 0.8294/0.8125 and
stress Sharpe 0.7658/0.6630, with stressed drawdown approximately 17.5%-17.9%.
It reduces some drawdown but does not meet the original objective.
SLSQP reported intermediate bound-clipping warnings; final optimizer success,
feasibility, actual targets and independent accounting were checked. No
fallback allocation or warning suppression converted a failure to success.
The [sixth weekly snapshot](research/workbench/evidence/research_program_2026-W41_results_v6.json)
retains 90 completed configurations and an empty research-champion slot.

### Option-implied term-structure risk information

The next two preregistered risk overlays use official Cboe daily VIX and
VIX3M observations. These indices measure different option-implied volatility
horizons; they are risk inputs, not tradable holdings or fabricated index
returns. The [source manifest](research/workbench/evidence/volatility_term_source_20261010.json)
records complete coverage of the existing ETF research sessions. The actual
downloaded VIX3M series starts September 18, 2009; no earlier history is assumed.

`config/volatility-term-risk.json` fixes the inversion boundary at VIX
greater than or equal to VIX3M. On inversion, the candidates either halve
or remove the equity sleeve of the unchanged four-factor/gold monthly risk
balance, moving the reduction to BIL while retaining gold. Month-end changes
and binary state transitions issue dated absolute targets for subsequent
opens. Costs and delayed-stress execution remain unchanged. No threshold
grid, index trading, same-close fill or missing-session forward fill is allowed.

```bash
.venv/bin/python -I -B -m us_quant.volatility_term_risk prepare
.venv/bin/python -I -B -m us_quant.volatility_term_risk register
.venv/bin/python -I -B -m us_quant.volatility_term_risk evaluate
```

This adds option-implied risk information rather than modifying the rejected
adaptive allocator. It is not yet evidence of incremental alpha. All
candidate outcomes must still go through the existing recurring reviewer.
Historical end-of-day index availability is not an independently proven
intraday release timestamp and cannot authorize a live decision.

The [actual term-risk outcomes](research/workbench/evidence/term_risk_20261010_results.json)
were also rejected by the recurring reviewer. Halving equity on inversion
gives ten/five-year base Sharpe 0.8692/0.8403 and stress Sharpe 0.7274/0.7688.
Exiting the equity sleeve gives base Sharpe 0.8270/0.7240 and stress
0.5663/0.6309. Transaction activity, delayed execution and missed recoveries
do not provide an improvement under these frozen rules; both outcomes remain
visible rather than being retuned to cross a threshold.

The [latest weekly research snapshot](research/workbench/evidence/research_program_2026-W41_results_v7.json)
contains 92 evaluated configurations in total and eight actual recurring
candidate reviews, all rejected. Two fundamental factor definitions remain
proposals with missing stock data, not completed strategy trials.
There is still no qualified unleveraged multifactor research champion or
independent forward result. The standing weekly process continues to seek
new justified evidence; it does not promise that repeated searches will
necessarily discover a strategy satisfying the requested return/risk goals.

### Fundamental-data feasibility checkpoint

The [official bulk-access record](research/workbench/evidence/sec_bulk_availability_20261010.json)
confirms that a historical SEC quarterly ZIP and the official format document
also returned HTTP 403. Each distribution received one normal request; no
proxy, identity impersonation or access-control bypass was used. The federal
Data.gov catalog was accessible, but its description cannot substitute for
the underlying filings or prove historical security/return coverage.

The [source assessment](research/workbench/evidence/fundamental_source_feasibility_20261010.json)
keeps actual provider statements separate from search summaries.
QuantConnect explicitly distinguishes its security master from the separately
licensed equity price data. Sharadar's free entry-level offering covers
current Dow30 companies, not a verified full historical universe. A search
summary describing HistPrice as complete US-stock PIT data was not confirmed:
the actual page demonstrated cryptocurrency data tooling. None of these
checks admits a complete new stock-factor dataset or a new performance result.

The existing ETF research scope remains available, but its completed candidates
have failed the retained performance goals. A new direct-stock fundamental
backtest requires an authorized and coverage-verified source; no subscription
purchase, account creation or current-stock-list substitution is implicit.
The strategy objective remains unfulfilled and the weekly research schedule
does not guarantee that a qualifying strategy exists.

### Public macro information without a subscription

With no authorization to buy data or open accounts, the next fixed study
uses two public FRED series: T10Y3M (ten-year minus three-month Treasury
constant-maturity yield spread) and DFII10 (ten-year inflation-indexed real
Treasury yield). The source pages identify the Federal Reserve Bank of
St. Louis and Federal Reserve Board H.15, respectively. Standard Python
HTTP requests had transport errors; `curl` using the existing network
configuration obtained the official CSV files. No access-control or
proxy bypass was used.

The [source manifest](research/workbench/evidence/macro_rate_source_20261010.json)
retains both CSV fingerprints and their real missing-release counts.
`config/macro-factor-tilt.json` preregisters two configurations: a fixed
factor-share tilt when lagged real yields rise, and the same tilt with
equity reduction during a nonpositive nominal yield-curve spread.
Allocation is evaluated only at completed month ends; the original aggregate
equity/gold risk balance is preserved before the declared defense.

An observation is usable only after two complete NYSE sessions strictly
following its date. The last available observation can be carried across
release holidays only while its actual observation date is at most seven
calendar days old; each decision's observation/availability dates are recorded.
Negative rates are preserved, missing values are not invented, and future
observations cannot change earlier signals. These are current historical
downloads, not vintage ALFRED proof of unrevised past data.

```bash
.venv/bin/python -I -B -m us_quant.macro_factor_tilt prepare
.venv/bin/python -I -B -m us_quant.macro_factor_tilt register
.venv/bin/python -I -B -m us_quant.macro_factor_tilt evaluate
```

Both configurations still require real recurring-program registration and
independent market/target/accounting review. Macro information is not a
substitute for the missing individual-stock financial dataset and does not
itself establish a profitable or independently verified factor.

The [two actual macro reviews](research/workbench/evidence/macro_tilt_20261010_results.json)
remain rejected. The factor tilt has ten/five-year base Sharpe 0.9420/0.9850
and stress Sharpe 0.9107/0.9553. The version with curve defense has base
Sharpe 0.9589/0.9143 and stress 0.9179/0.8556. Neither meets the strict
Sharpe threshold, ten-year SPY outperformance and 15% drawdown requirements.
The published availability audit covers 2,805 decision sessions per series;
the largest actual observation age is five calendar days, below the
registered seven-day maximum.

The [eighth program snapshot](research/workbench/evidence/research_program_2026-W41_results_v8.json)
preserves 94 completed configurations and ten rejected recurring-program
candidate reviews. The official free macro-data route is now working,
but the complete individual-stock fundamental dataset remains unavailable.
There is no qualified research champion or live strategy change.
This unresolved research goal is not converted to success by rounding
0.9850 upward, shortening windows, or hiding the risk/benchmark failures.

### Source-bound strategy review

Three synthetic regression cases exposed missing review boundaries: the
reviewer previously accepted a self-consistent but unrelated target matrix,
an unsupported frozen-code placeholder, or an altered submitted risk-free
series. None of these fixtures represents an actual market strategy result.
The corrected reviewer rejects all three before advancing a research version.

`strategy_replay.py` now reconstructs targets from the exact registered,
supported implementation and its frozen policy. It preserves the actual
cost-dependent feedback for daily-risk strategies and reuses the causal
indicator builders for adaptive, option-implied and macro rules. The complete
target matrix, including its missing/non-decision rows, must match submitted
targets. Unknown strategy generators cannot qualify merely by supplying an
equity curve; new implementations require an explicit reviewed replay adapter
and causality tests.

Sharpe is recalculated using the risk-free series in the **registered market
snapshot**, not a submitted return-table column. The program engine fingerprint
also binds the replay adapter and shared accounting/metric dependencies.
An explicit event-anchored migration preserves earlier records; an unexplained
engine/dependency change still fails verification.

Existing reviews can be rechecked without replacing or recounting them:

```bash
.venv/bin/python -I -B -m us_quant.research_program audit-reviews \
  --export evidence/research_program_source_bound_audit_v1.json
```

This action finds the original hashed bundles, regenerates targets and both
accounting paths, compares metrics and gates with the existing review, and
asserts that the ledger is unchanged. Disagreement is an error, not permission
to rewrite old results or promote a previously rejected candidate.

The [actual source-bound re-audit](research/workbench/evidence/research_program_source_bound_audit_v1.json)
regenerated all 40 strategy paths for the ten existing recurring-program
reviews. Every stored metric and gate agreed; all ten candidates remain
rejected. The
[migration snapshot](research/workbench/evidence/research_program_2026-W41_review_v9.json)
preserves all 94 evaluated configurations, prior factor definitions,
blocked-stock-data cycle and candidate histories. This audit adds zero new
strategy trials and no investment success claim.

The native weekly instructions now require a supported source-bound generator
for new candidate implementations and explain that engine identity includes
the replay/accounting dependencies. An exact stored `metadata.engine_sha`
and event-chain head are required for reviewed migrations; a single-file
checksum is no longer the complete engine identity.

### Defensive asset selection and a generic evaluation runner

The next fixed study keeps the four equity factor funds and selects defense
from GLD, TLT and IEF using 126-session returns above actual BIL. One variant
holds the highest positive-excess defensive fund; the other inverse-volatility
weights all qualifying defensive funds. If none qualify, defense is BIL.
The equity/defense budget uses the same 63-session risk balance and 30%-70%
equity limits as before. This is two allocation hypotheses, not a new
independent economic factor. Duration and gold risks remain explicit.

`config/defensive-factor-rotation.json` freezes the rules. TLT is loaded from
the same previously audited base snapshot, not a changed historical vendor
vintage. Every non-cash equity sleeve keeps all four factor funds equally
weighted, and no leverage, new market account or paid data is introduced.

The recurring program now has a generic `evaluate-candidate` operation.
It generates targets through the supported frozen implementation, runs
the exact market/cost and independent `bt` paths, creates the hashed bundle,
and sends that bundle through the stronger reviewer. Already reviewed
candidates are returned without another evaluation or trial increment.
This avoids writing a new bespoke accounting runner for each future study.

```bash
.venv/bin/python -I -B -m us_quant.defensive_factor_rotation
.venv/bin/python -I -B -m us_quant.research_program register-candidate \
  --candidate data/defensive-factor-rotation-20261011/four_factor_defense_momentum-spec.json \
  --readiness data/defensive-factor-rotation-20261011/readiness.json
.venv/bin/python -I -B -m us_quant.research_program evaluate-candidate \
  --candidate-id four_factor_defense_momentum
```

The second candidate uses `four_factor_defense_diversified` and its matching
specification. Both must be registered before any outcome is computed.
Interrupted output remains evidence; a new explicit output directory is
required for replay, and original candidate code/data cannot be changed.

Both defensive variants completed the real generic evaluator and were
rejected. The [complete results](research/workbench/evidence/defensive_rotation_20261011_results.json)
show ten/five-year base Sharpe 0.6922/0.7815 for the defensive winner and
0.5320/0.4078 for diversified defense, with lower stressed results.
The [updated snapshot](research/workbench/evidence/research_program_2026-W41_results_v11.json)
retains 96 completed configurations and twelve rejected recurring reviews.
No qualifying research champion exists.

### Prospective data acquisition, not simulated trading

`config/prospective-data.json` defines a separate data-only archive. Its
collector acquires actual ETF/IRX data and official FRED/Cboe risk inputs
after the latest NYSE close plus the existing data buffer, and before the
next open. It retains original source bytes and hashes, performs the same
market/calendar/release-lag checks, and records actual acquisition time.
Older rows in a downloaded history are **not** relabeled as observations
made on their historical dates.

```bash
.venv/bin/python -I -B -m us_quant.prospective_data init
.venv/bin/python -I -B -m us_quant.prospective_data collect
.venv/bin/python -I -B -m us_quant.prospective_data status
```

The first successful snapshot is a baseline only. Repeated calls for the
same completed session skip acquisition and do not increment counts.
Missing observed sessions are listed explicitly and never backfilled;
late or incomplete fetches do not create completed receipts. Concurrent
collectors serialize through a file lock. Receipt-chain/head checks and
the latest snapshot's input hashes detect changed or partially committed
records. Old failed attempts and old independent forward ledgers remain
unchanged.

Collection is configured for approximately 09:00 Asia/Shanghai daily, while
new-factor discovery remains Saturday 09:00. This archive contains no account, orders,
positions or strategy-return calculation. Its snapshot count is not a
forward Sharpe, a 63-session trading qualification or an investment success.
The app/host must be available, and real future observations require actual
market time to pass.

The [first acquisition receipt](research/workbench/evidence/prospective_data_baseline_receipt_v1.json)
confirms 14 real public sources and 35 retained input files for the October 9,
2026 closing-session baseline. Repeating collection skips network acquisition
and preserves the receipt-chain head. It records **zero** post-baseline
observations and zero strategy returns. The next new NYSE market session is
October 12; no later observation has been fabricated.

The [updated native schedule](research/workbench/evidence/prospective_data_automation_v1.json)
was saved and read back as daily. Its next invocation observed at configuration
time is October 11, 2026, approximately 09:01 Asia/Shanghai; weekend duplicate
sessions are skipped. On non-Saturdays, the agent collects data and handles
already registered work without launching new factor searches. On Saturdays,
the bounded weekly discovery/review process continues.
Scheduled agents may publish explicitly checked metadata checkpoints; the
standalone Slack reporter still performs no git, trading or deployment.

### Static-allocation feasibility diagnostic

The [recorded diagnostic](research/workbench/evidence/static_allocation_feasibility_v1.json)
uses the existing nine unleveraged assets, including SPY and defensive funds,
with 2% idle cash. It deliberately uses the entire historical window and
zero transaction costs to inspect a **fixed-weight, daily-close,
daily-rebalanced arithmetic-return class**. Each horizon is optimized
separately. This is not a causal strategy, a deployable backtest, or an
independent validation result.

The convex normalized-mean problem and a numerical dual certificate agree:
the specified class has an optimistic ten-year Sharpe ceiling about 0.93375
and five-year ceiling about 0.99439. Mean/covariance inputs and floating-point
tolerances are recorded; optimal portfolio weights are not presented as
trade recommendations. Analytical synthetic cases verify the diagnostic.

```bash
.venv/bin/python -I -B -m us_quant.allocation_feasibility \
  --output reports/static-allocation-feasibility-replay.json
```

The result adds **zero** strategy trials and cannot be submitted as a
qualifying strategy through the source-bound reviewer. It does not bound
dynamic strategies, different assets, different initial-entry assumptions
or every cost-bearing execution model. Its purpose is to discourage
unproductive static-mix tuning, not to prove that the overall research goal
is impossible. Credible progress needs predictive information, a justified
and verified broader universe, or genuinely new observations.
The [ordinary-client SEC recheck](research/workbench/evidence/sec_normal_client_check_v1.json)
still returned 403 under the same identity/network settings; no archive
download or access-control bypass occurred.

### Matched-risk growth exposure comparison

`config/growth-factor-satellite.json` introduces two fixed matched-risk
configurations. The control retains the original four-factor equity sleeve;
the other uses half that sleeve in the unleveraged Nasdaq-100 ETF QQQ and
half equally across the four factor funds. QQQ is explicitly growth/market
beta, **not a newly discovered factor or assumed independent alpha**.

Both use identical 63-session equity/gold risk balance, daily 12% predicted
volatility projection, a 5-percentage-point target-change band, 2% idle cash,
and the existing base/stress execution costs. The experiment does not raise
the risk target, remove the factor core or silently substitute leveraged
products. QQQ comes from the previously audited base-data manifest.
Actual drawdown, return and Sharpe gates remain required.

```bash
.venv/bin/python -I -B -m us_quant.growth_factor_satellite
# Register both matching spec files with research_program register-candidate,
# then use evaluate-candidate; no outcomes are computed by preparation.
```

The source-bound generator reproduces the actual composed sleeve risk, not
the volatility of the factor-only control. Any gains must be attributed
honestly to additional growth exposure rather than presented as factor
discovery. The prior static-mix diagnostic did not include QQQ and is not
a bound on this enlarged asset scope.

The [actual matched-risk results](research/workbench/evidence/growth_factor_satellite_20261011_results.json)
were both rejected. Growth exposure raises ten/five-year base Sharpe from
0.8347/0.8878 to 0.8865/0.9131, but drawdowns increase slightly to about
18.5%-18.7%; stressed Sharpe remains 0.8298/0.8739. This is a modest growth-beta
effect, not evidence that a new factor or qualifying strategy was found.
The [current program snapshot](research/workbench/evidence/research_program_2026-W41_results_v13.json)
retains 98 actual evaluations and fourteen rejected recurring reviews.

The enlarged-scope feasibility diagnostic initially failed its five-year
numerical certificate. A mathematically equivalent unit-volatility coordinate
transformation fixes numerical conditioning without changing the objective,
constraints or tolerances. A new disparate-volatility regression case passes,
and the [original-scope v2 result](research/workbench/evidence/static_allocation_feasibility_v2.json)
agrees with v1 within numerical tolerance; v1 remains preserved with its own
implementation hash.
The [QQQ-inclusive diagnostic](research/workbench/evidence/growth_scope_feasibility_v1.json)
gives zero-cost, full-sample fixed-weight numerical bounds of about 1.00333
for ten years and 0.99622 for five years. It remains a hindsight diagnostic,
not a strategy, not a joint shared-weight result, and not a bound on all
dynamic or cost-bearing execution. It adds zero strategy trials and cannot
qualify as investment success.

### Causally trained conditional forecasts

`config/conditional-factor-model.json` freezes a price-only pooled ridge model
and the same model augmented with the previously verified macro/option risk
inputs. Both predict each factor fund's and gold's next monthly open-to-open
return relative to BIL. A training label is eligible only after its exit open
has occurred; neither the current holding period nor a future label can enter
the training set. Training-window selection and the fixed ridge pipeline
reuse the imported workbench helpers.

The model uses at most 36 distinct months, training-only standardization,
and current-feature clipping to observed training ranges. The earliest fit
has only nine matured months due actual fund-history availability; five
cross-sectional observations per month are not five independent months.
The weak initial sample is disclosed rather than replaced with synthetic
pre-inception histories.

Forecasts rank the four factor funds into fixed 35/30/20/15% equity shares.
Equity and gold allocations require a fixed positive-excess forecast hurdle;
risk balance and BIL handle the declared active/inactive cases. Model
coefficients and allocations update chronologically, but this is not
discovery of new independent economic factors or a live strategy update.
Price-only predictions must remain unchanged when only macro/option data
are perturbed, and future prices/labels cannot change earlier predictions.

```bash
.venv/bin/python -I -B -m us_quant.conditional_factor_model
# Register both specs with the recurring program, then evaluate-candidate.
```

The source-bound reviewer regenerates the whole forecasting sequence before
checking actual costs and performance. No hyperparameter search, random split
or weakening of the original Sharpe/SPY/drawdown gates is part of this study.

The first conditional-model code version failed **before complete account
evaluation** because the residual BIL weight was negative by only
1.1102230246251565e-16. The
[original numeric diagnostic](research/workbench/evidence/conditional_model_numeric_diagnostic_v1.json)
is preserved. Revision 2 clamps only a cash-budget residue within 1e-12 of
zero and still rejects a material overspend. The
[complete-matrix correction check](research/workbench/evidence/conditional_model_numeric_fix_v2.json)
shows that training audits and every non-BIL weight are unchanged; the largest
target difference is the original roundoff magnitude. No economic parameter
or strategy outcome was used to select this correction.

The program supports an explicit `record-numeric-failure` lifecycle event
for this narrow, evidenced pre-accounting failure. It cannot erase a completed
review, overwrite an earlier failure or relabel a materially negative weight
as roundoff. Technically rejected code versions are retained separately from
completed strategy evaluations; they do not create a false return result.
Corrected specifications use new IDs and a `supersedes_candidate` link.
`pending_candidate_ids` excludes terminal technical failures, so future runs
do not repeatedly retry an immutable invalid version.

The [corrected model results](research/workbench/evidence/conditional_factor_model_20261011_results_v2.json)
completed actual source-bound review and were both rejected. Price-only
base Sharpe is 0.5460/0.6239 for ten/five years; macro/option augmentation
gives 0.5462/0.7765. Stressed Sharpe is lower and drawdown remains above 15%.
The added information improves the five-year comparison but does not satisfy
the target.
The [training audit](research/workbench/evidence/conditional_model_training_audit_v2.json)
records 121 fits per model, first fitted September 30, 2016 with nine
distinct matured months. All label-availability dates precede decisions,
and regenerated target matrices match the evaluated account inputs.

The [conditional-round program snapshot](research/workbench/evidence/research_program_2026-W41_results_v16.json)
had 100 complete strategy evaluations and 16 completed recurring reviews,
all rejected under the current full gates. Two separate, invalid first code
versions remain as technical failures, not hidden or falsely counted as
complete performance evaluations. There is still no qualifying research
champion or live strategy update.
The [conditional-round read-only source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v2.json)
regenerated all 64 paths for those 16 completed reviews and confirmed every
recorded metric/gate without changing the ledger. The two technical failures
remain outside the completed-performance count.

### Matched-budget total versus residual factor momentum

`config/factor-momentum-comparison.json` freezes two bounded allocation
hypotheses after the 100 completed configurations: total-return momentum
versus SPY-regression residual momentum in the same four actual factor ETFs.
Both inherit identical equity/gold totals from the unchanged monthly
factor/gold baseline, and rank funds into fixed 35/30/20/15% equity-sleeve
shares. All four economic exposures remain held; beta removal in the score
does not create a beta hedge in the long-only portfolio.

The unchanged original `factor_scores` helper fits trailing 252-session
daily excess returns and scores the last 126 sessions excluding the latest
21. This shorter ETF adaptation fits the available pre-account history
without moving either formal return window. It is not a stock-level
multifactor or academic long-short replication, and does not add an
independent factor definition. Source metadata verifies Ehsani and
Linnainmaa's journal article as **2022, DOI `10.1111/jofi.13131`**, distinct
from their 2019 [NBER working paper](https://www.nber.org/papers/w25551).
The residual-momentum motivation is the
[2011 Blitz/Huij/Martens article](https://repub.eur.nl/pub/22252/).

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_momentum_comparison
```

Preparation revalidates existing provider snapshots and reuses their hashed
panels; it computes no strategy returns. Both
[specifications were registered before outcomes](research/workbench/evidence/factor_momentum_comparison_20261011_registration.json)
after an exact-anchor engine migration preserved all 16 old reviews and both
technical failures. At registration both were pending and the completed
count was still 100, not 102. The previously exposed history is not an
independent holdout.

The [actual source-bound results](research/workbench/evidence/factor_momentum_comparison_20261011_results.json)
now reject both configurations:

| Factor ranking | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Total momentum | 0.8993 / 0.8590 | 0.9628 / 0.9101 | 20.52% / 18.04% |
| SPY-residual momentum | 0.9037 / 0.8567 | 0.9526 / 0.8907 | 20.52% / 17.69% |

Both beat SPY's five-year net CAGR, but fail the ten-year benchmark and all
Sharpe/drawdown paths. The
[paired attribution](research/workbench/evidence/factor_momentum_comparison_20261011_attribution_v2.json)
verifies identical target issue dates and equity/gold budgets within
numerical tolerance. Residual scoring raises dollar turnover despite the
same 605/305 ten/five-year order tickets. For five-year stress its traded
volume is about USD143,656 versus USD116,561 and costs USD592.31 versus
USD538.12, with lower Sharpe. Dollar turnover is independently reconstructed
from opening holdings/cash and checked against transaction charges; it is
not the dimensionless sum of the ledger's turnover ratios.

The [factor-momentum snapshot](research/workbench/evidence/research_program_2026-W41_results_v18.json)
retains **102 complete configurations, 18 recurring reviews, two separate
technical failures, no pending candidates and no qualified champion**.
The [read-only re-audit](research/workbench/evidence/research_program_source_bound_audit_v3.json)
regenerated every one of the 72 completed strategy paths without changing
history or conclusions. Do not tune these rejected lookbacks or shares
merely to cross one on the exposed sample. Daily collection again skipped
the existing October 9 baseline; there are still zero new prospective
observations and no strategy-return proof.

### Joint versus component trend risk gates

`config/trend-factor-guard.json` freezes two risk-management hypotheses
after 102 completed configurations, without changing factor rankings or
adding an economic factor definition. Both start from the unchanged
four-factor/gold monthly risk targets. At every completed close, a strict
200-session moving-average gate can disable risk; the removed weight goes
to actual BIL, never into a larger surviving risk sleeve.

The joint variant observes an **untraded signal index** compounded from
daily returns and the baseline target known at the preceding close. The
component variant separately gates an equal-weight four-factor return index
and GLD. These indices are not cost-free account returns or investment
evidence. Missing pre-target history is not filled with artificial cash
observations. Only complete 200-observation windows can issue decisions.
Both methods request orders on gate changes and month-end target updates;
unchanged daily states do not rebalance.

This daily adaptation is motivated by Faber's 2007 article,
DOI [`10.3905/jwm.2007.674809`](https://doi.org/10.3905/jwm.2007.674809),
not a replication of its monthly ten-month timing rule. Trend risk gates
can whipsaw, gap through a threshold and incur extra costs. They do not
guarantee a maximum drawdown or Sharpe above one.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.trend_factor_guard
```

Preparation revalidates and reuses the old hashed actual-fund snapshots;
it computes no strategy outcomes. Both
[fixed implementations are registered before outcomes](research/workbench/evidence/trend_factor_guard_20261011_registration.json).
At registration the completed count remains 102, with two pending
candidates. Exact-anchor migration preserved all 18 previous reviews and
both technical failures. The original dual windows, execution delays,
capital, costs and Sharpe/SPY/drawdown gates remain unchanged.

The joint guard completed its review, but the component guard's first
attempt stopped **before a completed review** on an independent accounting
disagreement. The
[preserved diagnostic](research/workbench/evidence/trend_guard_accounting_incident_v1.json)
reproduces a legacy solver corner: a rounding-only requested change of
about USD1.82e-12 enters the fixed-commission branch and manufactures a
USD1.00196 charge. The independent engine correctly performs no trade.
Do not count the interrupted attempt as a completed strategy evaluation.

`cash_funded_accounting_v2.py` keeps holdings and cash unchanged when every
pre-fee delta is below the **existing USD1e-6 order-notional threshold**.
Real trades reuse the original cash-budget solver, fees and execution
assumptions. This is a versioned accounting correction, not a different
portfolio rule or wider audit tolerance. The imported `backtest.py` and
all old frozen sources remain unchanged.

New reviews require an explicit, exact-source-hash
`cash_funded_noop_v2` bundle reference. Read-only audits of the previously
completed legacy bundles continue using their original engine; they are
not silently rescored. The original failed component output is retained,
and recovery must write to a fresh replay directory. At this incident
checkpoint there are **103 complete configurations, 19 completed reviews,
one pending candidate, and two separately indexed invalid strategy-code
versions**. The accounting incident is neither a third invalid strategy
definition nor an additional performance trial.
The [accounting-version registration](research/workbench/evidence/cash_funded_accounting_v2_registration.json)
pins the corrective source hash before a new replay; exact migration
preserved the original policy, candidate definitions, all 19 reviews and
both indexed strategy-code failures.

The [corrected component replay](research/workbench/evidence/trend_guard_accounting_recovery_v2.json)
now agrees with independent `bt` within USD1.06e-10 on all four account
paths. Every target CSV hash is identical to the interrupted attempt;
parameters, data, costs, delay and audit tolerance did not change. The
original failed bundle is retained. The completed
[two-candidate results](research/workbench/evidence/trend_factor_guard_20261011_results.json)
reject both risk gates:

| Risk gate | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Joint portfolio | 0.8878 / 0.7518 | 0.9076 / 0.7933 | 13.18% / 11.76% |
| Separate equity/gold | 0.6813 / 0.4131 | 0.7175 / 0.4168 | 15.78% / 14.69% |

The joint gate's base drawdown improves, but stress drawdowns are
15.26%/15.63% and neither horizon beats SPY's net CAGR. Component stress
drawdowns reach 21.61%/21.11%.
The [existing-control attribution](research/workbench/evidence/trend_factor_guard_20261011_attribution.json)
shows lower net CAGR and Sharpe versus the ungated core in every comparison.
Daily gate changes increase order tickets and charged turnover: component
five-year stress pays USD2,153.83 across 615 tickets. Risk reduction and
whipsaw costs are not new factor alpha or a guaranteed risk ceiling.

The [trend-guard ledger snapshot](research/workbench/evidence/research_program_2026-W41_results_v22.json)
retains **104 complete configurations, 20 completed reviews, two terminal
strategy-code failures, one separately preserved/resolved accounting
incident, no pending candidates and no qualified research champion**.
The [80-path mixed-version audit](research/workbench/evidence/research_program_source_bound_audit_v4.json)
preserves every old conclusion: 19 reviews use the original engine and the
corrected component review uses its frozen version-two reference. Do not
retune these windows or gate thresholds merely to cross one on this sample.

The [updated automation receipt](research/workbench/evidence/prospective_data_automation_v2.json)
keeps daily data acquisition and Saturday factor exploration unchanged,
while requiring explicit new-accounting references and preserved failure
evidence. No future scheduled execution is claimed. Duplicate collection
again leaves one October 9 baseline and zero post-baseline observations.

### Alternative implementation within the momentum family

`config/momentum-implementation.json` freezes two alternatives after 104
completed configurations: replace MTUM's original momentum-family budget
with PDP, or use equal MTUM/PDP shares within that same budget. Value,
quality, low-volatility and aggregate equity/gold target budgets stay
identical to the already evaluated original factor/gold control.
The second momentum fund is not a fifth independent economic factor.

PDP's actually retrieved official
[product page](https://www.invesco.com/us/en/financial-products/etfs/invesco-dorsey-wright-momentum-etf.html)
identifies US passive equity, ISIN US46137V8375, March 1, 2007 inception,
and the Dorsey Wright Technical Leaders mandate. The directly retrieved
[Nasdaq methodology](https://indexes.nasdaqomx.com/docs/Methodology_DorseyWrightTechnicalIndexes.pdf)
and [index page](https://indexes.nasdaq.com/Index/Overview/DWTL) confirm DWTL's
proprietary relative-strength selection. These are actual current primary
sources, not proof that all historical methodology versions were identical.
The issuer factsheet URL returned audience-selection HTML, not a PDF;
neither that gate nor Stooq's browser verification was bypassed.

The new actual PDP history has 2,805 complete sessions over the unchanged
factor-data interval, positive volume throughout, and explained adjustment
changes. The adapter reparses raw OHLCV and corporate actions, compares
the retained CSV and binds the actual document hashes. It never fills
SPMO's previously rejected zero-volume records or shortens either formal
return window. The
[source audit](research/workbench/evidence/momentum_implementation_source_audit_v1.json)
retains SPMO's 238 zero-volume rejection, Stooq's HTML verification response
and the issuer's non-PDF factsheet response. Current primary documents
support PDP's mandate without claiming a complete historical method archive.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.momentum_implementation
```

Preparation creates hashed panels but computes no strategy outcomes.
Both [fixed configurations were registered before outcomes](research/workbench/evidence/momentum_implementation_20261011_registration.json),
after exact provider-engine migration preserved all 104 completed
configurations and 20 reviews. At registration two candidates remain
pending; no new independent factor family or completed evaluation is
claimed. Use the generic versioned-accounting evaluator and retain all
costs, delay and full gates. Actual fund returns are not a recreated
proprietary stock-score strategy or independent forward evidence.

The [completed implementation results](research/workbench/evidence/momentum_implementation_20261011_results.json)
reject both alternatives:

| Momentum-family implementation | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| PDP only | 0.8594 / 0.8295 | 0.9089 / 0.8634 | 20.25% / 18.14% |
| Equal MTUM/PDP | 0.8764 / 0.8473 | 0.9277 / 0.8846 | 20.23% / 18.23% |

Both beat SPY's five-year CAGR but not its ten-year CAGR; every path misses
the net Sharpe and retained drawdown gates. The
[matched-budget attribution](research/workbench/evidence/momentum_implementation_20261011_attribution.json)
confirms identical target issue dates, momentum-family totals, other factor
budgets and equity/gold totals. Net Sharpe and CAGR are lower than the
already evaluated MTUM control in all eight comparisons.
PDP/MTUM daily return correlations are 0.931/0.907 for ten/five years;
the different implementation is not an independent return source.
Adding the second momentum fund raises five-year tickets from 305 to 366,
and five-year stress costs from USD434.31 for full PDP to USD496.29 for
the implementation mix. No mix optimization is justified by these outcomes.

The [momentum-round program state](research/workbench/evidence/research_program_2026-W41_results_v24.json)
retains **106 complete configurations, 22 reviews, two terminal invalid
strategy-code versions, the separately resolved accounting incident, no
pending candidates and no qualified research champion**.
The [88-path read-only audit](research/workbench/evidence/research_program_source_bound_audit_v5.json)
preserves all outcomes, including 19 legacy-engine reviews and three
explicit version-two reviews. The newly verified history does not create
future observations; daily collection still has one baseline and zero
post-baseline observations.

### Joint selection diagnostics on completed recurring research

`config/selection-validation.json` freezes a read-only diagnostic for the
22 completed recurring reviews. It does **not** pretend to cover the whole
106-configuration search: the earlier 84 trials remain outside joint
inference. Nor does it change qualification, create a strategy or provide
independent forward observations.

The auditor verifies each original hashed bundle and actual strategy/SPY
account path, including initial fees, cost scenarios and execution delays.
Within each horizon it simultaneously resamples all 66 comparisons:
base versus base SPY, stress versus stress SPY, and stress versus base SPY.
Identical sampled 21-session circular blocks preserve contemporaneous
dependence across candidates and cost comparisons. A least-favourable
mean-centered maximum statistic accounts for searching within this cohort,
rather than reporting a separately bootstrapped favourite.

The five-year window overlaps the ten-year window. Global 5% diagnostic
alpha is split across horizons by Bonferroni, not by assuming independence.
One-sided simultaneous bounds concern annualized **log-growth advantage**,
not CAGR percentage-point differences or future success probabilities.
The method adapts White's
[2000 data-snooping reality check](https://doi.org/10.1111/1468-0262.00152);
the finite-block bootstrap remains approximate and conditional on exposed
history, with stationarity and uncounted-selection limitations.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.selection_validation register
# Commit the fixed code/settings and registration before computing joint outcomes.
.venv/bin/python -I -B -m us_quant.selection_validation run
```

The audit fails on changed source hashes, a changed cohort/event head or
different settings. Both live journal and original review outcomes remain
unchanged. Even favourable scoped evidence cannot override Sharpe, SPY,
drawdown or absent independent-validation requirements.
The [fixed audit registration](research/workbench/evidence/selection_validation_20261011_registration.json)
pins code, settings, the 22-review cohort and source hashes before joint
resampling. At registration no statistical outcome, new strategy trial or
research-version update is claimed.

The [actual joint results](research/workbench/evidence/selection_validation_20261011_results.json)
find **no positive simultaneous lower outperformance bound**:

| Scope | Candidate/cost comparisons | Scoped omnibus p-value | Positive simultaneous lower bounds |
|---|---|---|---|
| Ten years | 66 | 1.0000 | 0 |
| Five years | 66 | 0.5881 | 0 |

The two-horizon Bonferroni omnibus p-value is 1.0. Some five-year curves
have positive historical net growth differences, but this least-favourable
joint diagnostic does not establish statistically significant advantage
within the declared cohort. It is conservative and conditional, **not a
probability of future failure or proof that profitable trading is
impossible**. The excluded 84 earlier trials and uncounted research choices
remain limitations; there is still no full-search selection-adjusted alpha
or independent future evidence.

The [reproducibility check](research/workbench/evidence/selection_validation_20261011_reproducibility.json)
replays all statistics exactly and independently reconstructs all 4,000
samples per horizon through direct indexing rather than the implemented
matrix-count shortcut. Omnibus p-values agree exactly; critical values
agree within 1e-15. Source hashes, journal events, 106 completed
configurations, 22 reviews and all prior qualification outcomes remain
unchanged. This audit adds **zero strategy evaluations**.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v3.json)
preserve daily collection and weekly discovery, explain the limited
selection scope, and require a newly registered statistical plan if the
cohort changes. The frozen 22-candidate result cannot be silently applied
to future candidates. No future scheduled execution is claimed.

### Credit-risk information and fixed recovery rules

`config/credit-factor-guard.json` freezes two risk-input experiments after
106 completed configurations. They use the previously evaluated static
70% equal-four-factor / 30% gold portfolio, not a newly optimized mix.
At completed month ends, the level guard enables equity only while the
lagged Baa/Treasury spread is below its trailing 252-session median.
The second rule also permits equity when that spread has declined over
21 sessions, a predefined early-recovery condition. Risk-off moves the
original equity budget to actual BIL; gold is never increased.

The initially considered
[ICE high-yield OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
now has only three years on FRED, so it cannot support the formal windows.
The study explicitly uses a **different credit proxy**:
[BAA10Y](https://fred.stlouisfed.org/series/BAA10Y), long-maturity seasoned
investment-grade Baa yield minus ten-year Treasury yield. It includes
duration/liquidity/quality effects and is not a high-yield option-adjusted
spread. The Treasury input provider changed in June 2019; this limitation
is retained, not erased. Proprietary input observations and derived
credit-value series stay private and are not published.

The actual public CSV covers the requested 2014-2026 range. The unchanged
macro availability helper delays every observation until the second
strictly later NYSE session and caps its age at seven calendar days.
The [source and timing audit](research/workbench/evidence/credit_factor_guard_source_20261011.json)
records 3,186 actual observations, 2,805 ETF decision sessions, 22 bounded
carry sessions and a maximum observation age of five days. The source
restriction and different proxy definition are preserved before outcomes.
Current historical extraction is not an ALFRED vintage or exact original
release-time archive. Credit is a risk input, not another independent
return factor, and the frozen earlier 22-candidate joint diagnostic does
not cover these new candidates.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.credit_factor_guard
```

Preparation revalidates the original ETF panels and actual credit timing
without computing strategy outcomes. Both
[fixed rules are registered before outcomes](research/workbench/evidence/credit_factor_guard_20261011_registration.json).
Exact-anchor engine migration preserved all 106 prior configurations,
22 completed reviews and the two terminal invalid code versions. The
two credit candidates are pending, not completed evaluations. Original
costs, execution delays, fresh
capital and Sharpe/SPY/drawdown goals stay unchanged. The existing data-only
prospective archive does not yet contain BAA10Y; no new independent
credit-strategy observations or future results are claimed.

The [completed credit results](research/workbench/evidence/credit_factor_guard_20261011_results.json)
reject both fixed hypotheses:

| Credit rule | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Below trailing median only | 0.8125 / 0.7649 | 0.6711 / 0.4926 | 15.65% / 15.80% |
| Also recover on compression | 0.8756 / 0.8472 | 0.6793 / 0.5100 | 16.67% / 16.90% |

Neither meets any full path's Sharpe/SPY/drawdown contract. The
[control attribution](research/workbench/evidence/credit_factor_guard_20261011_attribution.json)
verifies identical issue dates, unchanged gold, original family budgets
when equity is enabled, and actual BIL holding for every removed equity
budget. The strict guard reduces some drawdown but misses return.
Recovery improves ten-year Sharpe versus the original static control, but
still reduces CAGR by about 2.58 percentage points; its five-year Sharpe
and CAGR are lower. Both variants miss the index return goal.
This does not establish that every credit indicator is useless, and does
not justify tuning the rejected recovery/median windows on exposed history.

The [credit-round program snapshot](research/workbench/evidence/research_program_2026-W41_results_v26.json)
retains **108 complete configurations, 24 completed reviews, two terminal
invalid code versions, the separately resolved accounting incident, no
pending candidates and no qualified champion**.
The [96-path source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v6.json)
preserves all original outcomes, including 19 legacy and five explicit
version-two reviews. The earlier joint-selection audit still applies only
to its frozen 22-candidate cohort, not the two new credit candidates.
The original prospective archive stays unchanged, with zero new
observations and no Baa/credit-strategy forward-performance proof.

### Audited additional ETF families and a future-cycle queue

`config/factor-family-expansion.json` prepares two **new-to-this-catalog**
economic families: small-company exposure through actual IJR and net-share
reduction exposure through actual PKW. This is not a claim to discover
globally new anomalies or independently profitable factor returns.
The issuer confirms PKW's trailing-twelve-month net-share reduction
threshold of at least 5%; IJR tracks the S&P SmallCap 600, not academic
zero-cost SMB. IJR can use futures to offset cash for tracking, so it is
not described as derivatives-free.

Both actual fund panels cover the unchanged 2,805-session interval.
The [source preparation audit](research/workbench/evidence/factor_family_source_expansion_20261011.json)
preserves the actual issuer snapshots, acquisition times, positive volumes
and company-action checks for both funds.
The S&P methodology request returned 403; it is not bypassed or reported
as verified. Current issuer definitions do not prove historical screen
continuity. Raw current holdings, back-tested index returns and recreated
stock-level scores are not used as fund-price substitutes.

The factor-ETF registration path now accepts the complete definitions
bound by an explicitly audited source adapter, rather than always forcing
the same four fixed labels. Catalog bodies must match the audited
definitions, not just similarly named families. Legacy adapters retain
the original four definitions; unknown adapters and mislabeled direct-stock
signals still fail closed. Minimum three economic families, leverage,
capital, costs and performance objectives are unchanged.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_family_sources
```

Preparation creates source-verified panels, not strategy results or new
candidate registrations. The two definitions are queued with an explicit
**not-before October 17, 2026 at 09:00 Asia/Shanghai** constraint. This week's
two-definition allowance has already been used. `cycle` rejects a premature
queue rather than inventing a future timestamp or silently bypassing the
weekly limit. Existing factor counts and all 108 completed configurations
remain unchanged until a real eligible cycle occurs.

The original prospective archive does not include IJR or PKW. The separate
extension below adds only a new data baseline; no independent new-family
strategy performance is claimed.
The [real-clock queue check](research/workbench/evidence/factor_family_queue_receipt_v1.json)
actually rejected premature discovery and verified that the live catalog,
cycles, candidates, results and event head did not change after rejection.
The [history-preserving migration snapshot](research/workbench/evidence/research_program_2026-W41_families_v27.json)
still has **108 completed configurations, 24 reviews, six registered
definitions across five families and no qualified champion**. Two queued
definitions are not counted as registered discoveries.

The [latest 96-path source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v7.json)
reproduces all existing outcomes after the adapter change without modifying
the journal. The [native queue instructions](research/workbench/evidence/prospective_data_automation_v4.json)
refresh readiness only at a real eligible cycle and register definitions
before any candidate relying on them. They do not invent a future execution
or use a synthetic clock. Minimum economic-family, weekly-definition,
cost, risk and return criteria remain unchanged.

### Separate expanded prospective input archive

`config/prospective-factor-inputs.json` defines a new data-only archive for
IJR, PKW and BAA10Y in addition to the existing 14 public sources.
`prospective_factor_inputs.py` composes the unchanged parent archive's
receipt, lock, gap and deadline logic. It never changes the original
collector, parent registration, receipt chain or paused strategy ledgers.

The new collector requires a verified parent snapshot for the **same
completed session**. It copies the exact parent input bytes and retains
their actual acquisition times, then requests only the two extra fund
histories and one extra credit series. Expanded records have their own
collector hash and receipt chain, tied to the parent's registration and
same-session receipt hash. A missing parent, changed source, failed fetch
or next-open deadline miss cannot produce a completed expanded snapshot.

The first expanded snapshot is another **baseline**, not a second
independent market observation. Snapshot counts never imply a strategy
Sharpe, P&L or trading qualification. Capturing IJR/PKW data does not
register the queued factor definitions early; their weekly eligibility
still applies. The proprietary Baa observations and aligned values remain
private. All price/macro rows are current-vendor context, not proof they
were first available when those historical rows occurred.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs init
.venv/bin/python -I -B -m us_quant.prospective_data collect
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs collect
.venv/bin/python -I -B -m us_quant.prospective_factor_inputs status
```

Initialize only the new explicitly named expanded archive, never the
existing parent. Daily execution collects the parent first and expanded
inputs second; same-session duplicates make no network requests.
Both archives preserve missed sessions without retroactive backfill.
The [extension registration](research/workbench/evidence/prospective_factor_inputs_v2_registration.json)
pins the new collector and policy before acquisition. It initializes only
`data/prospective-factor-inputs-v2`; the original parent's receipt head,
registration and source inputs remain unchanged. At registration the new
archive has zero snapshots and no independent strategy performance.

The [actual expanded baseline receipt](research/workbench/evidence/prospective_factor_inputs_v2_baseline_receipt.json)
records the October 9 session acquired at
**2026-10-10T21:22:03.497540Z**, before the next NYSE open.
The [capture audit](research/workbench/evidence/prospective_factor_inputs_v2_capture_audit.json)
verifies all 35 inherited input files and their acquisition timestamps,
unchanged parent receipt head, and the three actually acquired new sources.
The expanded archive has 17 source streams and 502 sessions of price context.
Those historical rows, and the two same-day parent/expanded snapshots, are
not extra independent market observations.

The [expanded status](research/workbench/evidence/prospective_factor_inputs_v2_initial_status.json)
has one baseline and **zero observations after baseline**, just like the
unchanged original archive. A duplicate call is verified to skip fetching
and leave both receipt hashes unchanged. No strategy account, P&L,
candidate evaluation or factor registration was added; all 108 research
configurations and the queued October 17 definitions remain unchanged.

The [latest native instructions](research/workbench/evidence/prospective_data_automation_v5.json)
collect the parent first and extension second at the existing daily cadence,
without reinitializing either archive. Future execution has not been
observed or claimed. The original collector fingerprint, old forward
ledgers and weekly research eligibility remain untouched.

### Frozen six-family candidate implementations for the eligible cycle

`config/six-factor-strategy.json` defines two complete candidate rules for
momentum, value, quality, low-volatility, size and net-share-reduction
exposures through the six actually verified ETFs. One uses equal equity
family shares; the other reuses the unchanged `bounded_factor_shares`
helper on trailing 63-session volatility, bounding every family between
half and one-and-a-half times the equal share.

Both inherit the original four-factor/gold baseline's identical monthly
equity/gold totals, cash reserve and issue dates. The comparison changes
only family composition, not benchmark costs or aggregate defensive
exposure. All six equity sleeves remain positive. Inverse marginal
volatility is **not full-covariance equal-risk contribution**, and six
economic characteristics are not six orthogonal alpha sources.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.six_factor_strategy prepare
# After the actual eligible weekly factor registration:
.venv/bin/python -I -B -m us_quant.six_factor_strategy register \
  --prepared data/six-factor-prepared-20261017 \
  --receipt evidence/six_factor_strategy_20261017_registration.json
```

Preparation creates source-bound specifications, not performance results.
Registration is explicitly blocked until October 17 at 09:00 Asia/Shanghai
and requires the new definitions already present in the real journal.
The source-bound replay registry supports both implementations; they
cannot qualify using arbitrary submitted curves. Costs, fresh capital,
execution delay, drawdown and Sharpe/SPY goals remain unchanged.
No real candidate outcome is computed merely because the code is prepared;
the earlier 108 trials and 22-candidate statistical diagnostic are not
rewritten or silently extended.

The [complete prepared specifications](research/workbench/evidence/six_factor_strategy_preparation_v1.json)
freeze both implementations and all source dependencies. On the real
2,805-session / ten-asset panel, each has 131 complete monthly targets.
The budget audit verifies positive six-family shares, fixed share limits,
98% target investment and unchanged aggregate equity/gold targets.
**No actual strategy-account performance was computed.**

The [actual eligibility checks](research/workbench/evidence/six_factor_strategy_eligibility_v1.json)
refused both source-backed candidates because their new definitions are not
yet registered; the independent real-clock gate also refused early execution.
These checks add no candidates or strategy trials. The
[history-preserving engine snapshot](research/workbench/evidence/research_program_2026-W41_six_factor_v28.json)
still has 108 completed configurations, 24 reviews and six registered
definitions. The [latest 96-path audit](research/workbench/evidence/research_program_source_bound_audit_v8.json)
reproduces every old metric and gate after adding the new generator.

The [native execution instructions](research/workbench/evidence/prospective_data_automation_v6.json)
prepare fresh evidence at the real eligible cycle, register definitions
before candidates, publish the actual candidate receipt, and only then
evaluate performance. They do not fabricate a registration receipt or
pretend prepared targets establish Sharpe above one. Both data baselines
and the earlier performance/selection limitations remain unchanged.

### Prospective target provenance, not execution or performance

`config/prospective-target-observations.json` freezes an independent
observation profile for the already registered `macro_real_yield_factor_tilt`
rule. This candidate **failed the historical qualification gates** and
remains rejected; recording its future targets does not promote it or claim
a profitable strategy.

The observer reuses the exact original rule and parameters with an actually
captured same-session parent ETF/macro snapshot. It records the real
generation time, original source receipt and most recent completed rule
month. The initial target is generated from the latest known completed
month, not backdated to that month's close. New input days cannot silently
retune an old monthly target; it is retained until a newly observed completed
month exists.

Hypothetical base/stress execution dates are later than the actual observed
market session. They are **not fills or submitted orders**. The journal
never creates positions, computes P&L, starts a portfolio or changes the
research champion. Data gaps stay explicit and cannot become manufactured
earlier signals or proof of continuous forward performance.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_target_observations init
.venv/bin/python -I -B -m us_quant.prospective_target_observations collect
.venv/bin/python -I -B -m us_quant.prospective_target_observations status
```

Initialize only this new named observation journal, never the existing
data archives or old paused strategy ledgers. Source code/parameters and
the original rejected status must verify. Capture is restricted to actual
known inputs before the next open; same-session duplicates do not regenerate
targets. The existing 108 historical configurations and queued six-family
definitions/candidates remain unchanged.
The [observer registration](research/workbench/evidence/prospective_target_observations_v1_registration.json)
pins the rule and recording policy before the first generated target.
The source candidate remains rejected, and registration itself creates
zero target observations, orders, portfolios or strategy-performance paths.

The [first actual intent record](research/workbench/evidence/prospective_target_observations_v1_first_target.json)
was generated at **2026-10-10T21:59:17.810517Z**, using the October 9 input
snapshot actually captured at 16:44:09.524784Z. The latest completed rule
month is September 30; the output is **not backdated to September 30**.
Hypothetical next-open/extra-delay dates October 12/13 are unfilled research
intent, not a claim to have traded or earned returns.

The [first receipt](research/workbench/evidence/prospective_target_observations_v1_first_receipt.json)
binds the target to the original source snapshot and observation-policy
registration. The [status](research/workbench/evidence/prospective_target_observations_v1_initial_status.json)
has one target observation and zero portfolios, orders, returns or research
champion updates. Same-session reruns are verified not to regenerate or
change the receipt. Full target weights are retained privately rather than
published as an actionable allocation. The source strategy remains
historically rejected; the 108 completed configurations are unchanged.

The [updated daily instructions](research/workbench/evidence/prospective_data_automation_v7.json)
collect parent inputs, then expanded inputs, then target observations.
They preserve the old monthly target until a new month is actually
available and never restart a paused forward account. Actual future
automated execution, strategy fills or Sharpe qualification is not claimed.

### Separately registered prospective research model accounts

`config/prospective-research-accounts.json` defines a new research-only
simulation for the already recorded, historically rejected macro-factor
candidate. It is not a live account, research champion or restart of any
paused shadow ledger. Four fresh USD10,000 model accounts compare the
strategy and SPY under original 5/20bp costs, USD1 ticket charges and
one/two-session delays. Registration must precede the first model open.

Each modelled session requires its own actually captured parent data receipt
and target-observation receipt. Targets must have been recorded before the
applicable open. A later download's historical rows cannot fill a missing
observation; missing inputs explicitly stop the model. Late processing of
already captured, on-time immutable inputs is not relabelled as earlier
processing or actual execution.

Adjusted total-return price levels may be renormalized in successive vendor
downloads. The model chains actual within-snapshot open/close ratios against
that snapshot's previous close, using **normalized model units rather than
broker shares**. This does not invent asset returns. A changed previous
raw-close quote pauses the experiment for source/corporate-action review,
rather than silently accepting a revision. Frozen funded-account simulation
and independent `bt` must agree before any new row is recorded; prior model
rows must also reproduce unchanged.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.prospective_research_accounts init
.venv/bin/python -I -B -m us_quant.prospective_research_accounts advance
.venv/bin/python -I -B -m us_quant.prospective_research_accounts status
```

Initialize only this new explicitly named model journal, never the existing
archives or old paused ledgers. The initial cash anchor has zero return and
no invented interest. Before 63 actual model sessions, no Sharpe estimate
is reported. Even later modelled returns are not broker performance,
independent selection-adjusted alpha or qualification of the historically
rejected source strategy. No orders or live deployment are authorized.

The [actual experiment registration](research/workbench/evidence/prospective_research_accounts_v1_registration.json)
occurred at **2026-10-10T22:10:59.950314Z**, before the first possible
October 12 model open, and binds the previously recorded target receipt.
The [initial status](research/workbench/evidence/prospective_research_accounts_v1_initial_status.json)
has **zero observed model sessions**. Four USD10,000 cash anchors are
bookkeeping initial capital, not four strategy discoveries or past returns.
Total return, Sharpe, CAGR and drawdown are unreported rather than
manufactured as a successful zero-risk performance history.

The [initial verification](research/workbench/evidence/prospective_research_accounts_v1_initial_verification.json)
actually calls advancement twice without new inputs and records no model
rows or costs. Parent/target receipt heads remain unchanged; the source
candidate is still historically rejected and 108 prior evaluations are
unchanged. Missing observations, revised raw quotes and independent-account
disagreement preserve failure attempts rather than produce success-shaped
fallbacks.

The [latest native daily sequence](research/workbench/evidence/prospective_data_automation_v8.json)
collects parent data, expanded data and target observations, then advances
only this separate registered research model. It does not trade, restart
old paused portfolios or claim that October 12/13 execution already occurred.
Future automatic execution and Sharpe qualification remain unverified.

### Descriptive factor-risk structure before the eligible new-family cycle

`config/factor-risk-structure.json` fixes a read-only study of actual asset
returns, **not the queued six-family strategies' performance**. The original
four equity funds, six equity-family funds and six-plus-GLD/IEF diagnostic
cohorts share the exact ten/five-year windows and verified source panels.
No allocation weights, portfolio returns, CAGR or Sharpe are calculated.

The report separates correlation-based and covariance-based effective risk
dimensions. Correlation normalizes asset volatilities; covariance retains
their actual observed scale. Neither dimension counts proven independent
economic alpha. The old four-factor helper is preserved and cross-checked;
its historically named `covariance_effective_dimension` field is in fact
correlation-based, not silently rewritten.

Full-sample SPY excess-return regressions describe beta and market R-squared.
Their residual dependence is an **in-sample, non-investable diagnostic**,
not a forecast or permitted beta hedge. Near-zero residual variance is
marked explicitly unestimable rather than assigned fake zero correlation.
Overlapping trailing 252-session month-end slices describe risk persistence,
not independent validation samples.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_risk_structure register
# Publish the frozen source/method registration before computing statistics.
.venv/bin/python -I -B -m us_quant.factor_risk_structure run
```

The sources, cohorts and event head are bound before results. Registered
factor definitions, the weekly allowance, queued October 17 eligibility,
108 completed evaluations and original outcomes are unchanged.
The [risk-study registration](research/workbench/evidence/factor_risk_structure_20261011_registration.json)
pins all sources and descriptive cohorts before statistics; it computes no
pending candidate performance or additional strategy evaluations.

The [actual dependence results](research/workbench/evidence/factor_risk_structure_20261011_results.json)
show that extra equity-family labels provide only a small increase in
volatility-normalized risk dimensions:

| Descriptive asset cohort | Ten/five-year correlation dimensions | Ten/five-year covariance dimensions |
|---|---|---|
| Original four equity families | 1.314 / 1.450 | 1.313 / 1.404 |
| Six equity families | 1.354 / 1.473 | 1.360 / 1.466 |
| Six equity plus GLD/IEF | 2.201 / 2.368 | 1.711 / 2.011 |

For the six equity funds, the first standardized component explains
85.57%/81.81% of observed ten/five-year variance. All six fund returns have
substantial in-sample market exposure; individual SPY R-squared ranges
are about 0.697-0.964 and 0.670-0.951. These are not forecasts, residual
alpha returns or proof of constituent overlap. GLD/IEF broaden the
descriptive risk space, but no capital allocation or portfolio performance
is computed from that comparison.

The [arithmetic/replay check](research/workbench/evidence/factor_risk_structure_20261011_reproducibility.json)
reproduces every full-window and overlapping rolling statistic exactly.
Independent singular-value decomposition of centered/raw and standardized
return matrices verifies the risk dimensions and leading components;
covariance slopes and squared Pearson correlations verify beta/R-squared
within 1e-12. The original four-asset helper also agrees unchanged.

This evidence does not retune the frozen equal/inverse-volatility
six-family candidates, register their queued definitions early, create a
historical trial or claim six independent return sources. All 108
completed configurations, 24 reviews and future-cycle eligibility remain
unchanged. Greater diversification must be verified in an actual
portfolio, not inferred simply by counting economic labels.

### Growth/factor/gold core with ratcheted portfolio protection

`config/growth-portfolio-protection.json` freezes two experiments using only
the four **already registered** families and actual unleveraged QQQ/GLD
inputs. They do not advance the queued size/share-issuance registration,
change weekly-definition eligibility or compute queued six-family outcomes.
QQQ is growth/market beta, not an independent new alpha family.

The monthly control reuses the frozen growth-satellite composition/risk
helper but does not impose its earlier daily 12% volatility cap. Half of
the equity sleeve goes to QQQ; the other half remains equally allocated to
the four factor funds. Equity/gold inverse-volatility balance and the
30%-70% equity bounds stay unchanged.

The second rule adds an unleveraged ratcheted-floor adaptation motivated by
Black and Perold's [1992 CPPI paper](https://doi.org/10.1016/0165-1889(92)90043-e).
The wealth floor is 85% of the highest actual closing account NAV,
including initial capital. Risk scaling is derived from the retained
15% drawdown target, capped at the monthly cash-funded budget; reduced
investment goes to actual BIL. There is no fitted cushion multiplier,
borrowed capital, synthetic interest or risk-ceiling guarantee.

Targets depend on fresh-window capital, incurred fees and execution delay,
so each path regenerates its own causal decisions. A five-percentage-point
actual-holdings band limits daily requests, pending targets are never
overwritten, and monthly updates remain queued until eligible. Gaps,
delayed execution, costs and cash lock-in can still defeat the drawdown or
return goals; this is not a paper replication or guaranteed insurance.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.growth_portfolio_protection
```

Preparation revalidates the original growth ETF snapshot and computes no
outcomes. Both [fixed rules were registered before outcomes](research/workbench/evidence/growth_portfolio_protection_20261011_registration.json).
Exact-anchor migration preserves all 108 prior configurations and 24
reviews, and adds no new factor definition. The two actual candidates
are pending; the queued six-family rules remain unregistered and unevaluated.
All original dual-window costs, Sharpe/SPY/drawdown criteria and paused
ledgers remain unchanged. The current prospective archive lacks QQQ;
no independent future result for this new study is claimed.

The [actual complete results](research/workbench/evidence/growth_portfolio_protection_20261011_results.json)
reject both fixed rules:

| Growth/factor/gold rule | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Monthly unprotected control | 0.9762 / 0.9470 | **1.000447** / 0.9548 | 19.16% / 19.31% |
| Ratcheted protection | 0.6890 / 0.4791 | 0.6973 / 0.5246 | 12.82% / 12.80% |

The control's single five-year base Sharpe barely exceeds one, but its
ten-year/stress Sharpe, ten-year SPY comparison and drawdown gates fail.
It is **not a qualified strategy**. Protection satisfies the modeled
drawdown gates in these four paths, with stress drawdown 13.51%/12.96%,
but has much lower net CAGR/Sharpe and misses SPY. These observed outcomes
do not guarantee protection against different gaps, costs or delays.

The [feedback/account attribution](research/workbench/evidence/growth_portfolio_protection_20261011_attribution.json)
checks each request against the actual closing NAV and high-water mark,
including fresh capital, incurred costs and original delay. Pending targets
are never overwritten. For five-year stress, protection pays USD2,422.68
across 1,622 tickets versus USD510.26 and 366 tickets for the control.
Its mean realized BIL share is about 29%, and CAGR is lower by about
8.22 percentage points. More frequent protection trades and lower risk
exposure are not incremental factor alpha.

The [growth-protection program snapshot](research/workbench/evidence/research_program_2026-W41_results_v30.json)
retains **110 completed configurations, 26 reviews, six registered
definitions, two terminal invalid code versions, the separately resolved
accounting incident, no pending registered candidates and no qualified
champion**. These two completed rules use only existing families; the
queued six-family strategies remain unregistered and unevaluated.
The [104-path read-only re-audit](research/workbench/evidence/research_program_source_bound_audit_v9.json)
confirms all 26 original outcomes, including 19 legacy and seven explicit
version-two reviews. The fixed 22-candidate selection diagnostic is not
silently extended to these later four candidates.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v9.json)
preserve data/target/model collection and the queued October 17 eligibility,
and do not treat the one barely passing base metric as success or authorize
post-result parameter tuning. The separately registered forward model still
has zero observed return sessions; no future scheduled execution, live
orders or objective completion is claimed.

### Two fixed downside-risk balances on the unchanged growth/factor core

`config/downside-risk-balance.json` compares downside semideviation and
5% expected-loss budgets while keeping the original 63-session window,
half-QQQ equity composition, four existing families, equity bounds and
funding/cost rules fixed. The already evaluated total-volatility monthly
control is reused, not counted again as a third new strategy.

Semideviation measures squared negative daily excess returns averaged over
all 63 observations. Expected loss uses the worst 5% empirical probability
mass, including a fractionally weighted boundary observation; simply taking
four full observations out of 63 would not be exactly a 5% tail. Risk-free
thresholds use the same frozen lagged market series as Sharpe evaluation.
Zero observed loss is not future risk-free proof: one zero sleeve estimate
uses the declared 30%-70% bounds; two zero estimates explicitly fail.

These are marginal risk-budget heuristics, not coherent joint-portfolio
expected-shortfall optimization, drawdown guarantees or new independent
factors. Reference metadata verifies Sortino/Price's
[1994 downside-risk paper](https://doi.org/10.3905/joi.3.3.59) and
Acerbi/Tasche's [2002 expected-shortfall paper](https://doi.org/10.1111/1468-0300.00091).
Current histories are exposed; no tail/window/growth-weight search or
post-outcome estimator revision is permitted in this version.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.downside_risk_balance
```

Preparation revalidates the original actual ETF snapshots and computes no
outcomes. Both [fixed rules are registered before outcomes](research/workbench/evidence/downside_risk_balance_20261011_registration.json).
Exact-anchor migration preserves 110 completed configurations and 26
reviews; the existing control is not recounted as a new experiment.
The two new candidates remain pending until source-bound evaluation under
unchanged net Sharpe/SPY/15% drawdown gates. Queued six-family definitions,
their October 17 eligibility and the independent forward model are not
modified or pre-evaluated.

The [complete actual results](research/workbench/evidence/downside_risk_balance_20261011_results.json)
reject both fixed downside estimators:

| Marginal risk estimator | Ten-year base/stress Sharpe | Five-year base/stress Sharpe | Ten/five-year base drawdown |
|---|---|---|---|
| Semideviation | 0.9667 / 0.9323 | 0.9817 / 0.9297 | 19.59% / 19.74% |
| Exact 5% expected loss | 0.9672 / 0.9287 | 0.9841 / 0.9250 | 19.38% / 19.61% |

Neither improves the completed total-volatility control. The
[independent-risk/account attribution](research/workbench/evidence/downside_risk_balance_20261011_attribution.json)
reconstructs the estimators using scalar negative squares and explicit
probability weights rather than calling the production risk helpers.
All 131 monthly budgets per method and all eight full account target
matrices agree; dates, four-family/QQQ composition and funding constraints
remain fixed. Every comparison has lower net Sharpe/CAGR and higher
drawdown than the existing control. This failure does not establish that
every joint downside model is useless or justify a post-result tail grid.

The [current program snapshot](research/workbench/evidence/research_program_2026-W41_results_v32.json)
retains **112 completed configurations, 28 recurring reviews, six registered
definitions, two terminal invalid code versions, the resolved accounting
incident, no pending candidates and no qualified champion**.
The [112-path source-bound re-audit](research/workbench/evidence/research_program_source_bound_audit_v10.json)
reproduces all outcomes, including 19 legacy-engine and nine explicit
version-two reviews. The earlier fixed 22-candidate joint diagnostic is
not extended to the later six candidates. Neither queued six-family
strategy nor its October 17 registration eligibility changed.
The prospective model still has zero observed return sessions, with no
new live orders or research-champion promotion.
The [latest native instructions](research/workbench/evidence/prospective_data_automation_v10.json)
retain daily capture and eligible weekly exploration, the exact rejected
tail definition and the distinction between 112 completed studies and two
unregistered six-family plans. Future scheduled execution is not claimed.

### Dollar-risk source review without premature unleveraged admission

The [UUP source-admission audit](research/workbench/evidence/dollar_risk_source_admission_20261011.json)
investigates a currency driver rather than another equity-style label.
The actual Invesco product page identifies UUP, ISIN US46141D2036,
February 20, 2007 inception and long USDX futures against six currencies,
plus Treasury/money-market collateral income. Those facts do **not alone
prove a historical notional exposure cap no greater than NAV**.

The issuer's linked annual report, quarterly report and factsheet all
returned audience-selection HTML, not PDFs, under ordinary public requests.
No gate was bypassed, no shared or acquired credentials were used, and
search-generated assertions were not treated as primary leverage evidence.
The review therefore **does not admit UUP** into the unleveraged strategy
universe or silently fill `asset_leverage: 1`. This is an unverified
admission constraint, not a finding that UUP is necessarily a leveraged
product.

No price-history dataset, strategy specification, new factor registration
or complete backtest was added by this source check. All 112 completed
configurations, 28 reviews and original qualification conditions remain
unchanged. The actual four-step collection/target/model sequence again
returned same-session duplicate/no-new-data actions; it did not create a
new market observation, target, model return or future-execution claim.

### Receipt-backed daily operations and honest schedule provenance

`research_daily.py` runs the already registered parent-data, expanded-data,
target-observation and research-model stages in the fixed order. Each unique
operational receipt records source code, invocation start/end, per-stage
actual results and unchanged historical-research state. It does not create
factor definitions, candidate evaluations, archives or a trading account.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.research_daily --origin operator
```

A failed stage preserves completed acquisitions and stops later stages;
retries naturally use each component's existing duplicate/integrity guards.
A missed pre-open deadline is recorded as failure, not a successful
pipeline. There is no rollback of genuine previously acquired data and no
quiet catch that converts a source error into success.

The invocation origin is **declared, not authenticated scheduler proof**.
Even `--origin session_automation` does not establish that the app delivered
a native wakeup. A changed `nextRunAt` establishes schedule advancement,
not message delivery or agent execution. Actual command results verify the
pipeline separately; same-day duplicate success is not a new market
observation, portfolio return or Sharpe qualification.

The [native scheduling check](research/workbench/evidence/native_daily_scheduling_acceptance_20261011_v1.json)
observed the overdue schedule's metadata advance to October 12 at
01:00:02Z, with a 09:02 Asia/Shanghai metadata update. Available turn
history does not prove native message delivery. Its absence there also
does not prove delivery failed. This uncertainty is retained instead of
declaring the native automation end-to-end successful.

The [actual after-due execution receipt](research/workbench/evidence/research_daily_actual_run_20261011_v1.json)
records a **continuation-origin** run at 01:28:16.900797Z through
01:28:17.586734Z. All four stages completed: three existing-snapshot/target
duplicates and one no-new-observed-model-session result. No source hashes,
market observations, target counts, model returns or historical trials
changed. This verifies the pipeline separately, not the scheduler origin.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v11.json)
invoke the consolidated runner on future due turns and require actual
stage receipts. Origin remains declared rather than authenticated;
the daily cadence, queued weekly eligibility, original registrations and
112 completed strategy outcomes remain unchanged.

### Sector growth: stronger historical returns, unresolved drawdown

The [sector source-admission audit](research/workbench/evidence/sector_growth_source_admission_20261011.json)
verifies actual XLK and SOXX issuer identities, non-multiplied equity-index
mandates and 2,805 complete positive-volume sessions per fund. Primary
documents were actually retrieved and parsed; HTML PDF viewers remain
recorded as viewers rather than falsely reported PDFs. SOXX's disclosed
June 21, 2021 transition from the PHLX index to the NYSE Semiconductor Index
is retained. Current documents do not establish unchanged historical rules
or complete absence of fund-level derivatives and incidental borrowing.
Raw prices and proprietary document contents remain private.

The [two fixed specifications](research/workbench/evidence/sector_growth_balance_20261011_registration.json)
keep half of equity in the original four equal factor sleeves and replace
only the other half's QQQ exposure with either XLK or SOXX. The unchanged
growth-budget helper supplies monthly 63-session equity/gold risk balance,
30%-70% equity bounds and 2% target cash. Sector concentration is not a new
independent economic factor or proof of safer returns. Both candidates
retain the original capital, costs, execution delay, benchmark and
15% drawdown qualification policy.

```bash
cd research/workbench
.venv/bin/python -m pip install -r requirements-sector-research.txt
.venv/bin/python -I -B -m us_quant.sector_growth_balance
```

Preparation refuses an existing output directory and computes no strategy
outcomes. The PDF dependency is separate because the imported dependency
manifest remains immutable. Registration used the actual clock and an
exact prior-engine/event migration. At registration, all original files,
112 completed configurations, 28 reviews and technical failures remained
unchanged; the [preregistered snapshot](research/workbench/evidence/research_program_2026-W41_sector_v33.json)
recorded two pending candidates. Actual source/code/specifications and
registration receipts were committed and pushed as `86c486ee` before
account outcomes.
The IJR/PKW queue still waits for the real October 17 cycle, and no
prospective sector returns, live orders or qualified champion are claimed.

The [completed results](research/workbench/evidence/sector_growth_balance_20261011_results.json)
retain **both rejections** under the unchanged qualification contract:

| Candidate | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Four factors + XLK + gold | Ten years | 15.93% / 1.02045 / 18.97% | 15.51% / 0.99169 / 19.09% |
| Four factors + XLK + gold | Five years | 19.47% / 1.06563 / 19.12% | 18.71% / 1.01815 / 19.24% |
| Four factors + SOXX + gold | Ten years | 18.40% / 1.07665 / 20.49% | 17.98% / 1.05066 / 20.49% |
| Four factors + SOXX + gold | Five years | 23.45% / 1.14441 / 20.65% | 22.62% / 1.10042 / 20.64% |

SOXX passes strict Sharpe and same-window SPY return gates in all four
paths, including stress against base-cost SPY. **Its drawdown still exceeds
15% in every path**, so it is a return-leading research reference, not
a qualified stable strategy. XLK also fails drawdown and misses ten-year
stress Sharpe. No investment objective, independent future performance
or trading authority is established.

The [preregistered matched-budget attribution](research/workbench/evidence/sector_growth_balance_20261011_attribution.json)
independently reproduces all 131 monthly allocations per candidate within
1.11e-16 and reaccounts eight QQQ substitutions with identical equity/gold
budgets, dates, funding and costs. Every candidate path has higher net
CAGR and Sharpe than its QQQ comparison. For SOXX, the improvement comes
with roughly 1.6 percentage points more drawdown; the fund's descriptive
in-sample SPY beta is 1.55/1.80 across ten/five years. This is not proof
of independent sector alpha. The worst loss interval peaks March 8, 2022
and troughs October 14, 2022 in these model accounts.

The [sector-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v34.json)
recorded **114 completed configurations, 30 recurring reviews, no
pending candidates and no qualified champion**. The
[120-path read-only source audit](research/workbench/evidence/research_program_source_bound_audit_v11.json)
retains all 19 legacy and 11 version-two reviews and leaves the previous
28 conclusions unchanged. The fixed 22-review joint-selection inference
does not cover the later eight studies.

The [preservation receipt](research/workbench/evidence/sector_growth_baseline_preservation_20261011.json)
verifies unchanged imported files, all original prospective receipt heads
and zero observed model-return sessions. A real continuation-origin
daily run at 02:20:17Z completed four duplicate/no-new-data steps;
it is not independent native-delivery proof. The
[updated daily/weekly instructions](research/workbench/evidence/prospective_data_automation_v12.json)
retain the original cadence, source boundaries and October 17 queued-family
eligibility. They do not silently replace the frozen rejected macro
forward experiment with this historically better sector strategy.

### Dollar funding risk: lower drawdown, unresolved ten-year outperformance

The [dollar source audit](research/workbench/evidence/dollar_information_source_admission_20261011.json)
retains 299 actually acquired official H.10 weekly releases. **298 are
admitted; one page with a ten-day release-date conflict is quarantined.**
Twelve identified archive/page date conflicts retain both declared dates
and conservatively admit values only after the later date. This does not
authenticate true first-publication timestamps. The dated `pubtables`
template is accepted only with its exact H.10 table title and observation
headers. Original source-check failures and raw files remain private.

Dollar risk uses the original goods-only broad index through its announced
December 2019 retirement, then the goods/services replacement at both ends
of each comparison. The replacement's retroactively constructed 2006
history is never treated as pre-2019 information. No different index
methods or January 2006 indexation regimes are level-spliced. Actual
published snapshots differ from current FRED values in many observations;
those current CSVs are diagnostics only, not fallback trading inputs.
All 131 month-end comparisons are source-bound, with at most 13-day
dollar observation age and five-day real-yield age. DFII10 retains its
previous two-session lag and current-vintage limitation; this is not a
complete ALFRED-vintage macro study.

The [actual preregistration](research/workbench/evidence/dollar_risk_guard_20261011_registration.json)
freezes exactly two guards on the unchanged four-factor/SOXX/gold monthly
allocation. Positive 63-session dollar change triggers the first;
positive dollar **and** lagged real-yield changes trigger the second.
A trigger halves all original non-BIL targets and transfers the removed
budget to actual BIL, retaining positive factor sleeves, relative
equity/gold composition and 2% target idle cash. The original 63-session
risk window, 50% scale, dates, source code, funding and costs are not
retuned to the already observed drawdown. Sharpe, SPY and 15% drawdown
gates remain unchanged.

```bash
cd research/workbench
# Requires the sealed local source files; choose a new preparation directory.
.venv/bin/python -I -B -m us_quant.dollar_risk_guard --output data/dollar-risk-prepared-new
```

Preparation computes no portfolio outcomes. The separately preregistered
eight constant-scale attribution controls replace each guard with its
full-window average scheduled exposure; they are hindsight descriptive
controls, not causal candidates, added strategy trials or holdouts.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_dollar_v35.json)
recorded **114 completed configurations, 30 reviews and two pending
candidates**, with no qualified champion. The exact engine migration
preserves all old records and prospective registration/receipt bytes.
No new alpha-family definitions, early IJR/PKW admission, prospective
sector performance, UUP investment, orders or live deployment are claimed.
Actual source/code/specifications and registration receipts were
committed and pushed as `a5387f89` before computing the new outcomes.

The [completed results](research/workbench/evidence/dollar_risk_guard_20261011_results.json)
retain both rejected candidates:

| Guard on the four-factor/SOXX/gold core | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Dollar appreciation alone | Ten years | 14.25% / 1.06754 / 13.13% | 13.68% / 1.01924 / 13.26% |
| Dollar appreciation alone | Five years | 19.09% / 1.18889 / 13.15% | 18.10% / 1.11689 / 13.28% |
| Dollar and real-yield increase | Ten years | 16.03% / 1.06793 / 17.92% | 15.25% / 1.01243 / 17.74% |
| Dollar and real-yield increase | Five years | 21.34% / 1.25400 / 13.14% | 20.01% / 1.16687 / 13.28% |

**The dollar-only guard passes every Sharpe and 15% drawdown gate, but
fails ten-year SPY outperformance under both cost assumptions.** Its
worst-path Sharpe is 1.01924135 and maximum drawdown 13.28272896%.
Same-window SPY has 15.46% ten-year base CAGR and 15.41% stress CAGR;
the strategy cannot qualify by showing only the stronger five-year
results. It is a more stable research reference, not a qualified champion.
Requiring real-yield confirmation keeps more market exposure and raises
five-year returns, but its ten-year drawdown and stressed SPY gates fail.
Its largest ten-year loss occurs in February-March 2020, outside the
five-year window. Neither rule guarantees future drawdown or returns.

The [independent attribution](research/workbench/evidence/dollar_risk_guard_20261011_attribution.json)
reads the actual H.10 tables through a separate parser and selects
real-yield observations using independently calculated strictly later
sessions. All 131 risk changes reproduce exactly, and both complete
2,805-row target matrices agree within 2.23e-16. Eight preregistered
constant-scale controls also reconcile with independent accounting.
The dollar-only guard improves net CAGR and Sharpe over its same-average
scheduled-exposure controls in all four comparisons. The confirmed guard
has lower ten-year stress Sharpe than its control. These controls use
full-window mean exposure, not causal signals or matched realized risk;
positive differences do not establish selection-adjusted alpha.

The [dollar-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v36.json)
recorded **116 completed configurations, 32 reviews, no pending candidate
and no qualified champion**. The
[128-path source audit](research/workbench/evidence/research_program_source_bound_audit_v12.json)
reexecutes all 19 legacy and 13 version-two reviews without changing any
previous conclusion. The earlier fixed 22-review joint-selection
diagnostic excludes the later ten reviews and the original 84 trials.
The [preservation evidence](research/workbench/evidence/dollar_risk_baseline_preservation_20261011.json)
checks all 69 imported files, previous 30 reviews and original prospective
registration/receipt heads. A real continuation-origin daily run at
03:12:17Z completed all four duplicate/no-new-model-session stages;
there are still zero observed forward-model sessions.
The [updated automation receipt](research/workbench/evidence/prospective_data_automation_v13.json)
keeps daily input collection and the real October 17 weekly discovery
eligibility. Configuration is not proof of native delivery, and the
original rejected macro forward experiment is not replaced.

### Separate prospective dollar-guard experiment

The [new forward profile](research/workbench/config/prospective-dollar-guard.json)
observes the fixed `semiconductor_dollar_half_guard` research reference
without replacing the existing macro experiment. Its historical status
remains rejected for ten-year SPY underperformance. No new backtest,
alpha-family definition, account authorization or champion is created.

The [actual input registration](research/workbench/evidence/prospective_dollar_inputs_v1_registration.json)
freezes a separate archive that inherits only the matching parent
snapshot's original bytes and acquisition times, then genuinely acquires
QQQ, XLK, SOXX and the official H.10 vintages needed by the latest completed
month. Real source preflight verified all three quotes and four required
weekly releases. Its 502 historical context rows are **not** prospective
observations; at registration the new archive had no snapshots or returns.
Current FRED dollar CSVs are not substituted for archived publications.
The original collector, targets, model and imported files remain unchanged.

The versioned adapter reuses the frozen monthly budget helper and exact
dollar-only guard. Targets are generated at their actual observation
time, never backdated to the rule month, and cannot be retuned within an
already recorded month. New target records must reproduce from their
own captured inputs. The separate model uses the explicitly declared
twelve-fund universe with the existing funded and independent accounting,
unchanged capital/costs/delay, no missing-day backfill and no broker.
Original model-journal checks are reused rather than changing its sealed
eight-fund implementation. Neither an initial target nor a cash anchor
is a future return observation.

```bash
cd research/workbench
# All directories are now registered. Do not repeat any init action.
.venv/bin/python -I -B -m us_quant.research_daily --origin session_automation && \
  .venv/bin/python -I -B -m us_quant.prospective_dollar_guard daily --origin session_automation
```

All registration, source and failure artifacts are retained in their
separate directories. No existing archive is reset. The original
daily pipeline and October 17 factor-definition eligibility are unchanged.

The source/profile and actual input registration were committed and pushed
as `290aa686` before the
[real baseline capture](research/workbench/evidence/prospective_dollar_guard_v1_baseline.json).
The separate October 9 baseline was genuinely acquired October 11 at
03:44:03.696386Z, inheriting all 35 matching parent files with their original
October 10 acquisition times. Only the additional quotes and official
H.10 releases receive new acquisition timestamps. This is one baseline,
not a second or third independent market day.

The first target was actually recorded at 03:44:05.942057Z from the
September 30 rule month. **It is not backdated to September 30.**
Its own captured-source replay and independent scalar budget agree
exactly. The
[separate research model](research/workbench/evidence/prospective_dollar_guard_v1_model_registration.json)
was registered at 03:44:06.941079Z, before the earliest possible
October 12 13:30Z opening. Base/stress strategy and SPY model accounts
each anchor USD 10,000; all have **zero observed model sessions** and
null return, Sharpe, CAGR and drawdown. These are neither broker accounts
nor executed trades. The minimum reporting window remains 63 actually
observed model sessions, not downloaded historical rows.

The [preservation audit](research/workbench/evidence/prospective_dollar_guard_v1_preservation.json)
checks all original/new control heads, all 69 imported files and unchanged
116 completed historical configurations. The original macro and separate
dollar pipelines genuinely ran at 03:45Z with four duplicate/no-new-model
actions each; no extra data day or account return was invented.
The [updated automation instructions](research/workbench/evidence/prospective_data_automation_v14.json)
run the original pipeline first and the separate dollar pipeline only
after it succeeds. Actual native delivery remains unverified.
The first normal capture opportunity for an October 12 U.S. model
session is October 13 morning in China; October 12 morning still sees
the October 9 baseline. The historical SPY gate still fails, no champion
is promoted, and the investment objective remains incomplete.

### Financial-condition confirmation using actual historical vintages

The [NFCI source admission](research/workbench/evidence/financial_conditions_vintages_20261011_admission.json)
retains **225 actual ALFRED vintages** in six public-form downloads.
Every requested date appears in its archive README and CSV column;
future observations are missing rather than replaced with current values.
Three separately acquired probes reproduce the batch data. Native
publication precision and revisions are retained. The first available
ALFRED vintage is May 25, 2011, before every required query. The actual
Chicago Fed FAQ states Wednesday 08:30 **Eastern**, with early-week
federal holidays shifting release to Thursday. Search summaries claiming
Central Time were not used.

For both current and 63-session reference inputs, the
[frozen specifications](research/workbench/evidence/financial_conditions_guard_20261011_registration.json)
use the snapshot as of the preceding NYSE session. This is conservative
dated availability, not authentication of first-publication timestamps.
NFCI revisions include incoming data and changing estimated indicator
weights; a difference between as-published readings is not a pure measure
of an economic shock. Current revised histories are not fallback inputs.
The official weekly observation cutoff of October 2 does not change the
original October 5 daily return cutoff or either evaluation horizon.

An initial source audit found two year-end queries, December 29 and 30,
2022, with last observations from December 9. **The 14-day freshness
limit was not relaxed to admit those 20/21-day-old values.** The
unregistered strict-coverage failure and its exact source/code snapshots
remain retained. Before strategy outcomes, the data-only revision
explicitly froze standby: a candidate does not rebalance when its
required NFCI input is stale, records the source gap and retains actual
existing holdings. Every actual daily price return and cost stays in
the account; no period is dropped, zeroed or reconstructed.
Tightness pauses December 30, 2022; deterioration also pauses March 31,
2023 because its 63-session reference is stale. All consumed readings
are at most 13 days old. An ALFRED gap does not by itself prove that the
original publisher paused publication.

Exactly two fixed confirmations are registered. Dollar appreciation must
coincide with either NFCI strictly above its source-defined zero-average
threshold, or a strictly positive 63-session change in separately archived
NFCI readings. A confirmed state halves all original non-BIL targets;
otherwise the original four-factor/SOXX/gold allocation remains unchanged,
except for explicit source standby. All factor shares, equity/gold budget
helpers, costs, 2% target cash and qualification gates are preserved.
The index is information, not a traded or leveraged product, new alpha
family or proof of independent returns.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.financial_conditions_guard --output data/nfci-prepared-new
```

Preparation requires sealed local source artifacts, refuses overwrite and
computes no portfolio outcomes. Eight constant-exposure attribution
controls were also preregistered, retaining the same candidate-specific
source-pause dates; they use full-window mean exposure and are not causal
candidates, holdouts or added historical trials.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_nfci_v37.json)
recorded **116 completed configurations, 32 reviews and two pending
candidates**. Exact migration preserves every old record and all seven
existing prospective profiles. Neither source acquisition nor
preregistration establishes the investment objective.
The actual source, code and registration receipts were committed and
pushed as `87b96bf7` before calculating the new portfolio outcomes.

The [completed results](research/workbench/evidence/financial_conditions_guard_20261011_results.json)
retain both failures:

| NFCI confirmation | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Tighter than average | Ten years | 18.38% / 1.07598 / 20.49% | 17.96% / 1.04930 / 20.49% |
| Tighter than average | Five years | 23.43% / 1.14355 / 20.65% | 22.57% / 1.09831 / 20.64% |
| Deteriorating as-published readings | Ten years | 14.80% / 0.95987 / 17.92% | 13.55% / 0.86955 / 17.74% |
| Deteriorating as-published readings | Five years | 19.92% / 1.10007 / 13.94% | 17.69% / 0.95815 / 14.76% |

**The tightness confirmation never triggered at any of the 131 scheduled
month-ends.** Its 130 nonempty target rows differ from the original SOXX
core only through the December 2022 source standby, not better risk
information. It still fails drawdown in every path. The deterioration
confirmation reduces risk in 41 months and has 129 target rows, but fails
ten-year Sharpe, drawdown and SPY gates; five-year stress Sharpe also
falls below one. These outcomes do not improve the qualifying strategy
set. The dollar-only guard remains the more stable reference, with its
original ten-year SPY failure still exposed.

The [independent attribution](research/workbench/evidence/financial_conditions_guard_20261011_attribution.json)
uses a separate raw ZIP/CSV reader for historical NFCI snapshots and the
separate H.10 table parser. All source readings, missingness and 131 risk
changes reproduce exactly. Both full 2,805-row target matrices, including
standby rows, agree within 2.23e-16. Eight constant-exposure controls
reconcile with independent accounts and the identical source-pause
dates. Deterioration has lower Sharpe than its constant-exposure control
in all four paths; tightness is the same exposure as its control.
The controls are hindsight attribution, not new candidates, complete
search correction or matched realized risk.

The [NFCI-stage snapshot](research/workbench/evidence/research_program_2026-W41_results_v38.json)
recorded **118 completed configurations, 34 reviews, no pending
candidate and no qualified champion**. The
[136-path read-only audit](research/workbench/evidence/research_program_source_bound_audit_v13.json)
retains 19 legacy-engine and 15 version-two reviews. The
[baseline-preservation receipt](research/workbench/evidence/financial_conditions_baseline_preservation_20261011.json)
checks every previous record, all 69 imported files and all seven
prospective profiles. Both original macro and separate dollar daily
pipelines actually completed duplicate/no-new-model-session runs at
04:28Z; each model still has zero observed return sessions. The
[updated automation receipt](research/workbench/evidence/prospective_data_automation_v15.json)
retains daily data collection and the real October 17 weekly discovery
eligibility. NFCI is not silently added to either frozen forward model.
The old 22-review joint-selection inference excludes the subsequent
twelve reviews and all earlier 84 trials. The investment objective is
still unverified, with no live deployment or order authority.

### Timely sector risk: fixed option-term and dollar guards

The [source audit](research/workbench/evidence/sector_term_source_admission_20261011.json)
revalidates the unchanged official VIX and VIX3M histories previously
admitted for the factor-only study. Both cover all 2,805 required ETF
sessions without forward fill. The actual current Cboe dashboard and
advertised methodology PDF confirm VIX3M's 93-day horizon, regular
09:31-16:15 Eastern calculation interval and September 2017 ticker rename
from VXV. The current methodology's launch entry and first history date
are retained, not rewritten to imply complete historical rule continuity.
Current downloaded closes are not authenticated unrevised intraday
publication snapshots. No volatility index becomes a portfolio holding.

The [two preregistered rules](research/workbench/evidence/sector_term_guard_20261011_registration.json)
keep the exact four-factor/SOXX/gold month-end budgets. The first halves
all non-BIL exposure when the existing VIX/VIX3M ratio threshold reaches
one; the second uses that condition **or** the latest known monthly
dollar-risk state. Removed exposure goes to actual BIL, not synthetic
interest. Factor/gold relative budgets and 2% target idle cash remain
unchanged. Dollar information updates only at a completed month, not
with later daily observations.

Targets are emitted only for a month-end budget update or binary-state
change. Each absolute target remains in its dated execution queue:
a one-day inversion and reversal are two later-open instructions, not
an overwritten pending order. The original base/stress delay, costs,
independent accounting and full qualification contract stay fixed.
Neither threshold nor 50% reduction is selected from a new grid.
The earlier factor-only option-risk failures remain retained; this
separate sector composition is not an independent alpha-family discovery.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.sector_term_guard --output data/sector-term-prepared-new
```

Preparation requires sealed local sources, refuses overwrite and computes
no portfolio outcomes. Eight mean-exposure controls are preregistered
as hindsight attribution, with event-frequency differences disclosed;
they are not causal candidates, matched realized risk or new trial counts.
The [pending snapshot](research/workbench/evidence/research_program_2026-W41_sector_term_v39.json)
recorded **118 completed configurations, 34 reviews and two pending
candidates**, no qualified champion. Exact engine migration retains
every previous record and all seven forward profiles. No new prospective
target experiment, early October 17 family admission or live orders
are authorized by these registrations.
The source, code and actual registration receipts were committed and
pushed as `64798241` before any new portfolio outcomes.

The [completed results](research/workbench/evidence/sector_term_guard_20261011_results.json)
retain both rejections:

| Risk guard on the original SOXX core | Window | Base CAGR / Sharpe / drawdown | Stress CAGR / Sharpe / drawdown |
|---|---|---|---|
| Daily option-term inversion alone | Ten years | 15.55% / 0.97503 / 21.36% | 12.85% / 0.79775 / 25.90% |
| Daily option-term inversion alone | Five years | 19.89% / 0.99827 / 21.70% | 17.66% / 0.88208 / 26.35% |
| Option-term inversion or monthly dollar risk | Ten years | 13.27% / 1.03573 / 14.64% | 12.26% / 0.94611 / 12.09% |
| Option-term inversion or monthly dollar risk | Five years | 17.43% / 1.13158 / 14.75% | 17.38% / 1.12756 / 11.90% |

**Faster risk observations did not improve the complete qualification
contract.** Term-only misses Sharpe and drawdown gates throughout, including
five-year base Sharpe 0.99827386, strictly below one. The combined guard
passes all drawdown gates but fails both ten-year SPY comparisons and
ten-year stress Sharpe. Its lower stressed drawdown is not evidence that
delayed execution improves risk generally. Neither scenario is dropped
or used to retune the threshold, delay, reduction scale or queued targets.
The more stable dollar-only reference remains historically rejected
on ten-year SPY returns.

The [independent attribution](research/workbench/evidence/sector_term_guard_20261011_attribution.json)
reproduces every raw Cboe close and all 131 monthly H.10 risk changes
without source-value error. Scalar risk calculations reproduce both
2,805-row target matrices within 2.23e-16, including 251 term-only and
173 combined dated instructions. There are 128/73 binary-state changes.
One-day inversions and reversals stay in the queue rather than silently
cancelling inconvenient later fills.

Eight preregistered mean-exposure controls reconcile with independent
accounting. Ten-year model tickets rise to 1,481 term-only and 1,078
combined, versus 847 for their monthly controls. Term-only has lower
net CAGR and Sharpe than its control in all four comparisons; combined
ten-year Sharpe also falls below its control. The different turnover,
exposure timing and execution delay are disclosed, not attributed solely
to commissions. These are hindsight descriptive controls, not matched
realized risk, independent alpha or additional strategy trials.

The [latest snapshot](research/workbench/evidence/research_program_2026-W41_results_v40.json)
records **120 completed configurations, 36 reviews, no pending candidate
and no qualified champion**. The
[144-path source audit](research/workbench/evidence/research_program_source_bound_audit_v14.json)
preserves all 19 legacy and 17 version-two reviews; the
[preservation receipt](research/workbench/evidence/sector_term_baseline_preservation_20261011.json)
checks every previous record, all 69 imported files and all seven
prospective profiles. Both daily pipelines actually completed duplicate
runs at 04:47Z; both registered models still have zero return observations.
The [updated automation receipt](research/workbench/evidence/prospective_data_automation_v16.json)
keeps the original daily/weekly cadence and October 17 family eligibility.
The 22-review joint-selection diagnostic does not cover these later
fourteen reviews or the original 84 trials. No forward rule is replaced,
no live trading is enabled, and the investment objective remains unmet.

### Expanded selection diagnostic: all 36 completed recurring reviews

The [separate preregistration](research/workbench/evidence/selection_expansion_20261011_registration.json)
extends the original 22-review diagnostic to all **36 completed recurring
reviews**, including the subsequent fourteen outcomes without dropping
failures. Each horizon contains 108 comparisons: base against base SPY,
stress against stress SPY, and stress against base SPY. These use all
144 original reviewed account paths, not a third cost rate or new
strategy backtests.

The [fixed policy](research/workbench/config/selection-expansion.json)
inherits the unchanged joint statistic, 4,000 synchronized 21-session
circular-block samples, seed and two-horizon Bonferroni split.
Strategies and overlapping horizons are not assumed independent.
Every original bundle is selected by its review digest, with account,
independent-account and frozen risk-free hashes and metrics revalidated.
The original 66-comparison prefix will be separately replayed against
its frozen results; old policy, code, registration and evidence are not
rewritten.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.selection_expansion run \
  --output reports/selection-expansion-20261011/new-replay
```

Choose a fresh output directory for each replay; existing registration
helpers, inputs and results are never overwritten.

After publishing the actual preregistration in `3cd53e1b`, the
[expanded results](research/workbench/evidence/selection_expansion_20261011_results.json)
remain **non-significant** under the frozen scoped family:

| Horizon | Comparisons | Scoped omnibus p-value | Positive simultaneous lower bounds |
|---|---:|---:|---:|
| Ten years | 108 | 0.500125 | 0 |
| Five years | 108 | 0.205699 | 0 |

The two-horizon Bonferroni omnibus p-value is **0.411397**, not a
probability of future profit. Even the highest observed log-growth
comparison, the unguarded semiconductor allocation against base SPY,
has a negative simultaneous lower bound in both windows. That
allocation's previously reported drawdown failure is not changed by
this diagnostic. Per-comparison annualized fields are 252 times the
daily log-growth means or bounds, not CAGR differences.

The [independent verification](research/workbench/evidence/selection_expansion_20261011_reproducibility.json)
separately reconstructs all account-based differences with scalar
logarithms, reproducing both full input matrices exactly. One-sample
direct-index circular-block arithmetic and explicit sorted critical
values verify every comparison mean, lower bound, sign and p-value
for both original and expanded scopes within `1.1e-18`.
Complete matrices of 2,512 by 108 and 1,254 by 108 replay
byte-identically, and the expanded maximum is never below its
original prefix on any shared sample. The old scoped p-values
`1.0 / 0.5881029743` and all four old selection artifacts are unchanged.

The original earlier 84 configurations remain explicitly outside this
inference. The full exposed 120-trial search, unknown research choices,
nonstationarity and current vendor revisions are not completely corrected.
A log-growth bound or omnibus p-value is not a Sharpe/drawdown gate,
an independent holdout, a probability of future profit or proof that
no viable strategy can exist. The
[preservation check](research/workbench/evidence/selection_expansion_20261011_preservation.json)
confirms all 36 prior reviews, six registered definitions, technical
failures, 69 imported files and seven prospective profiles remain
unchanged, with no engine migration or new strategy evaluation.
Both forward models still have zero return sessions; 120 completed
configurations and the absent champion remain unchanged. The
[updated native instructions](research/workbench/evidence/prospective_data_automation_v17.json)
retain daily 09:00 collection and the actual October 17 weekly
discovery gate. No new daily acquisition or independent native
delivery proof is claimed for this diagnostic checkpoint.

### Separately registered factor-exposure diagnosis

The [source admission](research/workbench/evidence/factor_exposure_source_20261011_admission.json)
records actual downloads from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The US daily five-factor and momentum archives both use the August 2026
CRSP cut and end on **August 31, 2026**. Their percent returns are
converted once to decimals; missing sentinels, duplicate dates and
interior coverage gaps are rejected, not carried or filled.
The current histories are not historical first-publication vintages.
Primary notes identify the January 2025 CRSP CIZ format change and
June 2024 Treasury-bill source change. Raw copyrighted archives and
documents remain private.

The [fixed plan](research/workbench/config/factor-exposure.json) and
[actual registration](research/workbench/evidence/factor_exposure_20261011_registration.json)
include all 36 completed recurring reviews, every rejected outcome
and all 144 original ten/five-year base/stress account paths.
Two descriptive models retain the **complete** 2,512/1,254-session
formal windows: SPY excess alone, and SPY excess plus the SOXX-minus-SPY
industry spread and GLD/IEF excess returns from frozen actual prices.
Each account uses its original frozen risk-free series.

A third model uses the six academic risk factors on only the complete
2,488/1,230-session overlap. Its 24 unavailable final sessions are
listed explicitly. This is **not** a shorter qualification window or
a replacement of account risk-free data. Only this descriptive
academic regression pairs returns with the published factor RF,
and reports its difference from the original account RF.

The models report in-sample coefficients, explained variance, residual
volatility and paired strategy-minus-cost-matched-SPY intercepts.
An intercept multiplied by 252 is an annualized **arithmetic**
quantity, not CAGR, a tradable residual, a predictive factor, or
independent alpha. No new significance or holdout claim is attached
to these already selected historical strategies. The dollar-only,
unguarded sector and original macro references are fixed highlights,
not chosen using the new regression outcomes.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.factor_exposure run
```

At preregistration portfolio-exposure outcomes are uncomputed.
There are no new candidate backtests, factor registrations, research
events, engine migrations or changes to the seven forward profiles.
All 120 completed configurations remain disclosed and no strategy
has satisfied the unchanged investment objective.

After publication of `76f27071`, the
[actual results](research/workbench/evidence/factor_exposure_20261011_results.json)
contain 432 paired fits across all 36 reviews, four formal account
paths and three fixed models. The frozen references show substantial
risk-model dependence:

| Reference | Horizon | SPY-only explained variance, base | Expanded explained variance, base | Expanded paired annualized arithmetic intercept, base / stress |
|---|---|---:|---:|---:|
| Dollar-only guard | Ten years | 32.28% | 86.57% | 1.28% / 0.81% |
| Dollar-only guard | Five years | 31.75% | 87.17% | 2.56% / 2.12% |
| Unguarded semiconductor allocation | Ten years | 40.95% | 97.16% | 0.95% / 0.61% |
| Unguarded semiconductor allocation | Five years | 39.26% | 96.88% | 2.05% / 1.74% |

Explained variance refers to daily excess-return variation, not a
fraction of cumulative wealth. For the dollar-only reference, the
paired annualized arithmetic intercept drops from 6.42%-10.02% under
SPY alone to 0.81%-2.56% after industry/gold/rate controls. Adding
predictors mechanically improves in-sample fit; neither this reduction,
the remaining positive intercepts, nor the incomplete academic
equity-factor model proves causal attribution or independent alpha.
The original ten-year SPY failure and sector drawdown failure remain.

The [independent verification](research/workbench/evidence/factor_exposure_20261011_reproducibility.json)
reconstructs original ZIP/account/RF inputs and direct price ratios,
then solves centered covariance normal equations instead of calling
the registered least-squares routine. Every fit's coefficients,
paired differences, arithmetic components, explained variance and
residual volatility agree: maximum coefficient error `2.0e-15` and
annualized-intercept error `2.56e-13`. Full replay keeps all numerical
results and source references identical.

The [preservation receipt](research/workbench/evidence/factor_exposure_20261011_preservation.json)
confirms all prior research records, both selection diagnostics,
69 imported files and seven forward profiles remain unchanged.
Neither model has any observed return sessions. The
[updated native instructions](research/workbench/evidence/prospective_data_automation_v18.json)
retain daily collection and the legitimate October 17 discovery
gate, without claiming new native delivery proof or treating these
academic archive downloads as prospective strategy observations.

### Historical real-yield availability: separately registered source comparison

The [new primary-source admission](research/workbench/evidence/real_yield_vintage_source_20261011_admission.json)
records 225 actual daily **DFII10** as-of vintages from the public
[ALFRED download form](https://alfred.stlouisfed.org/series/downloaddata?seid=DFII10).
Six bounded requests use at most 40 dates and 439 date-entry characters;
each archive's README and columns identify every requested vintage.
Three independently acquired probe dates reproduce the batch observations.
The native values are percent yields, not security returns.
Raw daily yield matrices, archives and documents remain private.

The [fixed source-only plan](research/workbench/config/real-yield-vintages.json)
and [actual registration](research/workbench/evidence/real_yield_vintage_20261011_registration.json)
retain all 131 original monthly/63-session comparison pairs on the
unchanged 2,805-session market index. Each query uses the actual
previous-NYSE-session snapshot **and** the original observation rule:
two NYSE sessions strictly after the observation date, with maximum
seven-calendar-day age. Missing or stale required data blocks the
diagnostic; it does not trigger a cash fallback, relaxed age limit or
new standby strategy.

The new adapter does not alter the frozen weekly NFCI parser or
the original macro helper. It reuses the unchanged lag/calendar logic,
while validating the distinct daily percent-yield archive schema.
Selection-date differences and same-observation revisions are reported
separately, followed by the original strict-positive real-yield states.
This is not a portfolio backtest or a search for a better lag.

The [publisher's current H.15 page](https://www.federalreserve.gov/releases/h15/)
describes weekday 4:15pm releases with holiday/closure exceptions.
Current schedule metadata and ALFRED date-level snapshots do not
authenticate every historical intraday first-publication timestamp.
The archived July 2019 title transition is not proof of unchanged
methodology, and an archive gap does not prove publisher non-delivery.

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.real_yield_vintages run
```

At registration the source-value and signal comparisons are uncomputed.
Old source bytes, strategy reviews, factor definitions, research events,
both forward rules and all investment qualification gates remain
unchanged. No new strategy evaluation or prospective model return
is claimed.

After publishing preregistration `c25051b7`, the
[verified source results](research/workbench/evidence/real_yield_vintage_20261011_results_v2.json)
find **no same-observation quote differences among the 225 matched
queries**, but three historical snapshots select an older available
observation than the current-history approximation:

| Query session | Current-history observation | Archived observation | Archived observation age |
|---|---|---|---:|
| 2015-08-31 | 2015-08-27 | 2015-08-26 | 5 days |
| 2024-12-26 | 2024-12-23 | 2024-12-20 | 6 days |
| 2025-12-29 | 2025-12-24 | 2025-12-22 | 7 days |

The maximum selected-yield difference is 0.05 percentage point.
One of the 131 strict-positive change states flips, on **2015-11-30**,
before either formal return window. This is an availability/selection
difference, not evidence that the 225 matched quotes were revised.
It does not prove the original publisher withheld a reading.

The [independent source verification](research/workbench/evidence/real_yield_vintage_20261011_reproducibility_v2.json)
uses separately parsed decimal quotes and scalar NYSE calendar
selection. All query dates, observation ages, values and monthly
states reproduce within `3.34e-16`; the complete private observation
and state tables replay byte-identically.

The [retained verification-scope failure](research/workbench/evidence/real_yield_vintage_20261011_verification_scope_failure_v1.json)
records an unpublished v1 proof that incorrectly described a return-only
target slice as including initialization. The frozen `seed_window`
places that signal on the **previous** NYSE session. V2 explicitly
checks the finite October 5, 2016 / October 5, 2021 seed rows and all
formal signal rows: all four original/source-replaced monthly target
matrices are byte-identical, with shapes **2,513 by 8 / 1,255 by 8**.
The actual return windows remain 2,512 / 1,254 sessions. No generator,
source, accounting engine or comparison tolerance was changed.
The retained verification omission is not another registered strategy
code failure or accounting incident.

This proof replaces only the audited monthly DFII10 query cells,
not all daily macro features or T10Y3M. It does not retroactively
make the conditional models or every prior study fully point-in-time.
The [preservation receipt](research/workbench/evidence/real_yield_vintage_20261011_preservation_v2.json)
confirms old source bytes, all 36 reviews, 69 imported files and seven
forward profiles remain unchanged. Both models still have zero
observed return sessions, and the original failures remain failures.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v19.json)
retain the daily/weekly cadence without claiming new native delivery
proof or promoting a strategy.

### UUP still excluded: current reported exposure is not a historical cap

The [new source evidence](research/workbench/evidence/uup_source_evidence_20261011.json)
records an actual request to the futures-holdings endpoint advertised
inside the issuer's UUP product page. The fund CUSIP, ISIN, long USD
contract and business date are checked, not inferred from a URL.
For **October 9, 2026**, the issuer's
`percentageOfTotalNetAssets` field reports a currency future at
**102.697813%**, or a reported-weight/NAV ratio of **1.02697813**.

This is a current issuer-reported percentage, not independently
authenticated notional terminology or a complete historical leverage
maximum. The snapshot is **after** the formal October 5 cutoff.
It cannot establish full-window compliance with the unleveraged
contract; absence of a 2x objective is not such a guarantee.
Collateral and futures are not added together as two NAV budgets.
UUP remains **excluded**, with no leverage value self-defaulted to one
and no new price-history or portfolio-performance evaluation.

Two additionally advertised reports, the 2024 annual and August 2026
monthly report, return HTTP 200 **country-selection HTML**, not PDF.
The prospectus and company-page requests to SEC return 403.
All three previous failed PDF attempts remain preserved. No country
cookies, browser impersonation, audience selection or access gate is
bypassed; search-generated prospectus quotes are not used as verified
primary text.

The dated parser verifies aggregate-versus-scoped holding values
and preserves the official pending-dividend cash adjustment's null
CUSIP and percentage rather than filling zero. Its first parser
attempt rejected this legitimate nullable row; that source/schema
failure and exact old code are retained. Only the observed adjustment
identity is recognized; unknown null identifiers, missing futures
weights, stale identities and inconsistent aggregates still stop.
This is not another registered strategy-code failure or accounting
incident, and a source percentage below 100% would still not
automatically establish a historical no-leverage cap.

Independent decimal arithmetic reproduces the reported future weight
and ratio. The
[preservation receipt](research/workbench/evidence/uup_source_preservation_20261011.json)
confirms all 120 configurations, 36 reviews, 69 imported files and
seven prospective profiles remain unchanged, with zero model returns.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v20.json)
retain daily collection and the actual October 17 discovery gate.
This source checkpoint does not constitute a qualified strategy.

### Inflation-compensation inputs prepared, not a new strategy

The [source-only plan](research/workbench/config/inflation-inputs.json) and
[readiness evidence](research/workbench/evidence/inflation_input_readiness_20261011.json)
add actual historical **T10YIE** as-of input provenance for possible
later research. Six official ALFRED batches contain all 225 original
month-end/reference-date snapshots for the 131 existing query pairs.
The unchanged 2,805-session index, previous-NYSE-session vintage
cutoff, two-session observation lag and seven-day age bound are
retained. Actual selected ages are at most **five days**.
Independent raw decimal/calendar reconstruction agrees exactly with
every selected value, date and age, and three separately acquired
probes reproduce their batch data.

TIP is **not new**: it already appears in the imported defensive
universe and the prior learned-allocation price features. No new TIP
price history, factor family or strategy trial is counted. The added
information is archived inflation-compensation data, not another
relabelled fund allocation.

The [primary series definition](https://fred.stlouisfed.org/series/T10YIE)
and [Federal Reserve interpretation](https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html)
do not make breakeven compensation pure expected inflation: inflation
risk and TIPS liquidity premiums also matter, and the series shares
rate information with the already researched real yield. The June
2019 underlying Treasury-source change, incomplete methodology
continuity and unauthenticated intraday first-publication times remain
explicit. Native percent quotes are not security returns.

No direction state, target, portfolio return, Sharpe, CAGR or factor
ranking is calculated. The inputs are not wired into any existing
strategy generator or forward profile. Any later hypothesis needs
separate actual eligible-cycle registration and publication before
performance evaluation; the October 17 IJR/PKW definitions and two
six-family candidates are not replaced or evaluated early.

The [preservation evidence](research/workbench/evidence/inflation_input_preservation_20261011.json)
keeps all 120 configurations, 36 reviews, six registered definitions,
69 imported files and seven forward profiles unchanged. Both models
still have zero return observations. The
[updated native instructions](research/workbench/evidence/prospective_data_automation_v21.json)
retain daily collection and the actual weekly discovery gate. Raw
archives, documents and the complete input matrix remain private;
source readiness is not investment-goal success.

### One fixed inflation-aware candidate implemented, not yet registered

The [frozen rule](research/workbench/config/inflation-factor-guard.json)
and [prepared plan](research/workbench/evidence/inflation_guard_20261011_plan.json)
extend the dollar-only reference with exactly one economic exception.
Dollar strength still halves the four-factor/semiconductor equities.
If the separately archived 63-session inflation-compensation change
is also positive, retain the original gold allocation; otherwise gold
is halved as before. Removed capital goes to actual BIL. The original
monthly budget, 50% reduction, four-family shares, sector share,
2% target idle cash, costs and formal windows are unchanged.

This is not a claim that gold reliably hedges inflation. The
[primary literature](https://www.nber.org/papers/w18706) describes
gold as an unreliable inflation hedge over practical horizons.
Breakeven compensation also contains risk/liquidity premiums.
The proposed exception may worsen risk or returns.

The [independent target check](research/workbench/evidence/inflation_guard_20261011_target_verification.json)
reconstructs actual ALFRED/H.10 inputs and scalar monthly volatility
budgets. Full targets have 2,805 rows and 12 assets, with 131 monthly
decisions and 39 conditional gold-retention months. Input/target
errors are at most `4.1e-16 / 2.23e-16`; all four source-bound
replays verify initial signals, every target weight and nondecision
row. These are target checks, **not portfolio returns**.

Both the dedicated wrapper and generic program registration reject
this candidate before **October 17, 2026 at 09:00 Asia/Shanghai**.
The coupled [typed-registration guard](research/workbench/evidence/inflation_guard_20261011_registration_type_guard.json)
also preserves explicit errors for malformed frozen-file manifests.
Two exact-anchor engine updates preserve all prior records and event
prefixes; [re-audit](research/workbench/evidence/research_program_source_bound_audit_v16.json)
regenerates all 36 existing reviews / 144 old paths without changing
their metrics or conclusions. No registered candidate, definition or
strategy evaluation is added.

At an eligible cycle, prepare into a fresh directory, check the exact
specification and publish actual registration **before** evaluation:

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.inflation_factor_guard prepare \
  --prepared data/inflation-guard-prepared-20261017
.venv/bin/python -I -B -m us_quant.inflation_factor_guard register \
  --prepared data/inflation-guard-prepared-20261017 \
  --receipt evidence/inflation_guard_20261017_actual_registration.json
```

The original IJR/PKW definitions and two six-family plans keep their
weekly gate; this adds one **prepared, unregistered** rule using the
existing four families, not a third new definition. All 120 completed
configurations, both frozen forward rules and zero observed model
returns remain unchanged. Older snapshot-bound diagnostics remain
bound to their original engine/cohort; use their matching frozen
environment rather than rewriting old hashes to force a new replay.
The [latest native instructions](research/workbench/evidence/prospective_data_automation_v23.json)
retain the daily/weekly cadence. A prepared strategy is not a
qualified strategy.

### Durable registration receipts without changing frozen strategies

The [separate receipt adapter](research/workbench/src/us_quant/candidate_receipts.py)
handles interruption **after** the candidate's transactional journal
registration but **before** its local receipt is saved. It matches the
exact stored specification, original readiness hash and verified event,
preserves the original registration timestamp, then writes the missing
receipt without a second admission. Repeated calls verify and retain
the existing bytes. Runtime locking and atomic no-overwrite publication
also cover concurrent calls.

`register` may create a first registration only through the original
data-admission and eligible-cycle gates. `recover` **never** registers
an unknown candidate. Neither mode evaluates accounts, promotes a
champion, changes a forward rule or submits orders. A new registration
still requires fresh readiness; recovering an old registration instead
requires its **original** readiness, not a refreshed replacement.
Changed sources, specifications, partial receipts, conflicting event
references and unsupported timestamps stop explicitly.

For the prepared inflation rule, use these exact paths only after its
legitimate cycle preparation; publish the receipt before evaluation:

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.candidate_receipts register \
  --spec data/inflation-guard-prepared-20261017/semiconductor_dollar_inflation_gold_guard-spec.json \
  --readiness data/inflation-guard-prepared-20261017/readiness.json \
  --receipt evidence/inflation_guard_20261017_actual_registration.json
# If registration committed but saving the receipt was interrupted:
.venv/bin/python -I -B -m us_quant.candidate_receipts recover \
  --spec data/inflation-guard-prepared-20261017/semiconductor_dollar_inflation_gold_guard-spec.json \
  --readiness data/inflation-guard-prepared-20261017/readiness.json \
  --receipt evidence/inflation_guard_20261017_actual_registration.json
```

Do not re-prepare or modify those inputs after a committed registration
to get recovery to pass. A failure may leave a valid candidate event;
inspect it and restore only its metadata receipt, not the trial count
or earlier result. Existing receipts must be matched, never overwritten.

The [actual recovery check](research/workbench/evidence/candidate_registration_recovery_20261011.json)
reproduces an already reviewed term-OR-dollar candidate's original
04:40:18 UTC registration in a new local metadata receipt and verifies
byte-identical repetition, with **zero** new registrations or reviews.
Both modes reject the current unregistered inflation plan.
Synthetic fault tests cover interrupted writes and concurrent retries;
they are not claims of actual disk, scheduler or missing-market-day
failures.

The [preservation evidence](research/workbench/evidence/candidate_registration_recovery_preservation_20261011.json)
keeps all 120 configurations, 36 reviews, 69 imported files, seven
forward profiles and the engine/event head unchanged. Frozen strategy
sources are untouched; no engine migration is needed. The
[updated native instructions](research/workbench/evidence/prospective_data_automation_v24.json)
retain daily collection and the October 17 weekly gate. This improves
the research workflow's recovery semantics, not its measured returns
or qualification.

### Recover complete review evidence, never missing portfolio paths

The [review receipt adapter](research/workbench/src/us_quant/review_receipts.py)
requires the original registered candidate, all four horizon/cost paths,
frozen market and source hashes, targets, weights and independently
accounted strategy/SPY curves. `record` may submit a complete, unreviewed
bundle through the original source-bound `ResearchProgram.review`.
`recover` requires an **already committed** matching review; it cannot
submit an unreviewed or unknown candidate.

For an existing review, both modes run the original **read-only**
source/target/account re-audit, then copy the stored review and unique
journal event. The new audit's timestamp does not replace the original
review time or outcome. Legacy reviews retain legacy accounting;
first reviews still require the explicitly versioned funded engine.
Atomic no-overwrite publication, runtime locking and exact receipt
matching prevent duplicate outcomes. An existing rejection remains a
rejection, not successful strategy qualification.

If evaluation was interrupted, inspect its original output first.
A complete `bundle.json` with no committed review may use `record`;
a committed review missing its new receipt may use `recover`.
Do **not** delete partial outputs, regenerate replacement curves,
fill missing horizons or invoke the evaluator blindly on an existing
directory. Changed or incomplete evidence stops explicitly. Choose a
dedicated `review-receipt.json`, not an existing legacy `review.json`.

These future paths apply only after actual eligible registration and
complete evaluation; the inflation candidate is still unregistered:

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.review_receipts record \
  --candidate-id semiconductor_dollar_inflation_gold_guard \
  --bundle reports/inflation-guard-20261017/semiconductor_dollar_inflation_gold_guard/bundle.json \
  --receipt reports/inflation-guard-20261017/semiconductor_dollar_inflation_gold_guard/review-receipt.json
# If that review committed but its receipt was interrupted:
.venv/bin/python -I -B -m us_quant.review_receipts recover \
  --candidate-id semiconductor_dollar_inflation_gold_guard \
  --bundle reports/inflation-guard-20261017/semiconductor_dollar_inflation_gold_guard/bundle.json \
  --receipt reports/inflation-guard-20261017/semiconductor_dollar_inflation_gold_guard/review-receipt.json
```

The [actual recovery evidence](research/workbench/evidence/candidate_review_recovery_20261011.json)
covers two previously rejected reviews: the legacy real-yield tilt
and versioned term-OR-dollar guard. Both retain their exact original
review bodies, accounting versions and timestamps; repeats keep receipt
bytes identical. `record` and `recover` both reject the unregistered
inflation candidate even when supplied with another complete bundle.
Synthetic interruptions exercise recovery semantics, not observed
disk failures or authenticated native scheduling.

The [preservation check](research/workbench/evidence/candidate_review_recovery_preservation_20261011.json)
keeps **120 complete configurations, 36 reviews, no champion and zero
forward return sessions**, all journal rows and metadata, 69 imported
files, seven forward profiles and the engine/head unchanged.
No strategy, threshold or forward model is changed and no migration
is performed. The [latest native instructions](research/workbench/evidence/prospective_data_automation_v25.json)
include complete-bundle recovery while retaining the daily/weekly
cadence. This is workflow reliability, **not a new investment result**.

### Employment input provenance with explicit historical coverage blocks

The [claims source policy](research/workbench/config/jobless-claims-inputs.json)
and [bounded validator](research/workbench/src/us_quant/jobless_claims_inputs.py)
add **source-only** ICSA preparation, not another factor or strategy.
Initial claims measure requests for unemployment-insurance eligibility
following employer separation; they are native **counts**, not the
unemployment rate or all unemployed people. Economic interpretation,
benefit coverage and seasonal adjustment do not establish predictive alpha.

The [actual verification](research/workbench/evidence/jobless_claims_input_readiness_20261011.json)
uses 225 ALFRED historical snapshots in six batches and 162 advertised
first-release vintages in five batches, covering 160 selected weekly
observations and all 131 original month/reference pairs. Each query uses
the snapshot from the **second previous NYSE session** and checks the
selected week's separate first-release date. A Saturday observation date
is not its publication date. Independent raw CSV/Decimal and calendar
checks reproduce every count, date, age and blocker exactly.

**Only 221 of 225 queries meet the frozen 14-day freshness limit.**
No date is removed or silently converted into a trading action:

| Query session | Latest source observation | Observation age |
|---|---|---:|
| 2018-03-29 | 2018-03-10 | 19 days |
| 2018-04-02 | 2018-03-10 | 23 days |
| 2025-10-29 | 2025-09-20 | 39 days |
| 2025-10-31 | 2025-09-20 | 41 days |

`inspect_queries` reports these four blocks on the complete grid;
`query_readiness` explicitly rejects full readiness. No interpolation,
current-history substitution, age-limit relaxation or automatic pause
rule is added. A later hypothesis would need a separately frozen
missing-data policy and actual registration **before** performance tests.
The existing October 17 IJR/PKW queue and three prepared candidates
are unchanged.

Originating DOL release/archive requests returned HTTP403 and were not
bypassed. Independently advertised Federal Reserve archives do not prove
the blocked DOL documents were read, or that an archive gap proves a
publisher failed to release. A preserved HTTP200 HTML response rejected
an arbitrary date for the initial-release-only format; corrected requests
use actual advertised release dates. This was a source-format failure,
not a registered strategy-code or accounting failure. Exact historical
intraday publication and full methodology continuity remain unverified.

The [preservation evidence](research/workbench/evidence/jobless_claims_input_preservation_20261011.json)
retains all **120 configurations, 36 reviews, no champion, zero forward
return sessions**, 69 imported files, seven forward profiles and the
unchanged journal/engine/head. No portfolio states, targets or returns
are calculated. The [updated native instructions](research/workbench/evidence/prospective_data_automation_v26.json)
retain these source blocks, daily collection and the actual weekly gate.
Source provenance is progress toward research, **not investment qualification**.

### One fixed labor-market confirmation, not a qualified strategy

The [frozen claims hypothesis](research/workbench/config/claims-factor-guard.json)
and [source-bound implementation](research/workbench/src/us_quant/claims_factor_guard.py)
prepare exactly one additional future candidate:
`semiconductor_dollar_claims_confirmation`.
It halves the original non-BIL targets only when the published dollar
has risen over 63 NYSE sessions **and** the four-week claims mean has
risen relative to its separately archived 63-session-earlier reading.
Equality does not trigger. Removed capital goes to real BIL; the four
factor families, half-equity SOXX satellite, gold sleeve, monthly risk
budgets, cash target and execution costs are unchanged.

Three same-vintage examples match the official IC4WSA smoothing
convention, not its full history or this rule's predictive accuracy.
Primary Fed research describes unstable claims level thresholds and a
late COVID-recession signal. We do not borrow its fitted thresholds,
Haver dataset or imply stock-return predictability.
Labor-market confirmation could worsen drawdown, timeliness or returns.

The [first source-window failure](research/workbench/evidence/claims_guard_20261011_source_window_failure_v1.json)
retains the unregistered policy/code and the actual 2018-11-26 query
missing its 2018-10-27 component. No strategy targets or portfolio
outcomes existed before an explicit **data-only** standby revision.
Together with the four previously frozen stale queries, this creates
five empty monthly target rows: 2018-03-29, 2018-06-29, 2019-02-28,
2025-10-31 and 2026-01-30. Existing holdings remain; all daily marks,
fees and return sessions still apply. No missing value is inserted,
the 14-day limit stays fixed, and any other incomplete window stops.
The original claims source itself remains only 221/225 fresh.

The [target verification](research/workbench/evidence/claims_guard_20261011_target_verification.json)
reproduces all source means/changes with zero error and all scalar
equity/gold/BIL budgets within 2.23e-16. The actual 2,805-by-12 matrix
contains 131 original months, 126 active targets, five standby months
and 26 joint-trigger reductions. Four complete target replays retain
the explicit seeds and 2,513/1,255-row seed-plus-return segments.
These are **target counts, not strategy returns**.

The [plan and actual early gate checks](research/workbench/evidence/claims_guard_20261011_plan.json)
retain the genuine October 17 eligibility. Dedicated and generic
registration both reject early admission; recovery cannot register an
unknown candidate. For the legitimate future cycle, prepare into a new
`data/claims-guard-prepared-20261017` directory, use `candidate_receipts`
to register the original spec/readiness and publish that actual receipt
**before** `evaluate-candidate`. If interrupted, recover only the same
registration or complete review evidence; never replace partial curves.
The original three queued strategies and IJR/PKW definitions are kept.
There are now **four prepared, unregistered strategies**, not four trials.

The [preservation check](research/workbench/evidence/claims_guard_20261011_preservation.json)
records one exact-anchor engine migration for the new finite replay and
admission gate, preserving every old record and event prefix.
The [36-review, 144-path audit](research/workbench/evidence/research_program_source_bound_audit_v17.json)
retains 19 legacy and 17 versioned accounting reviews, all 69 imported
files and seven forward profiles. The [current state](research/workbench/evidence/research_program_2026-W41_claims_plan_v43.json)
still has 120 complete configurations, no champion, no registered
pending candidates and zero forward return sessions. The
[latest native instructions](research/workbench/evidence/prospective_data_automation_v27.json)
keep daily collection and the actual weekly gate.
This hypothesis has **no measured portfolio performance or qualification yet**.

### Recoverable weekly queue with an actual publication fence

The [independent W42 queue](research/workbench/src/us_quant/research_weekly.py)
handles the four frozen candidates in order: equal six-family, bounded
inverse-volatility six-family, inflation/gold exception and claims
confirmation. It does **not** discover arbitrary factors, retune rules,
reset the ledger, submit orders or run Git commits/pushes. Future weekly
discovery still requires separately justified, preregistered hypotheses;
this is a bounded execution adapter, not a profitability guarantee.

After the real October 17 eligibility, complete the three original
preparation commands into their `*-prepared-20261017` directories and
the actual IJR/PKW factor cycle first. The queue verifies the exact
cycle, catalog definitions, original readiness and frozen specifications
before registering or recovering any candidate:

```bash
cd research/workbench
.venv/bin/python -I -B -m us_quant.research_weekly register \
  --origin agent_continuation
```

This produces the actual, immutable metadata packet
`evidence/weekly_research_2026-W42_registration.json` only after all four
registrations exist. That future file **does not exist yet**. Once
created, review its specifications/source references, explicitly add
its repo-relative path to the publication allowlist, then stage only
the reviewed metadata and allowlist and normally commit/push them.
Raw market/source/account files stay private.

Only then use the **full actual published commit SHA**, not `main`,
a local-only commit, an arbitrary earlier checkpoint or a fabricated
first-publication time:

```bash
.venv/bin/python -I -B -m us_quant.research_weekly evaluate \
  --origin agent_continuation \
  --published-commit FULL_ACTUALLY_PUBLISHED_40_CHARACTER_COMMIT_SHA
```

The queue queries this branch's configured remote, verifies ancestry
and exact published packet/code/policy bytes against the current frozen
files, checks the published and current approval boundaries, and saves
a complete no-overwrite publication fence **before** account evaluation.
Later unrelated approved commits do not replace that original fence.
If Git lacks the actual remote-head object locally, fetch the configured
ref explicitly; do not guess another branch or claim publication succeeded.
Git author/committer dates are not independently authenticated first
publication dates, and a declared invocation origin is not scheduler proof.

Registration interruptions reuse the original candidate events and
readiness. A complete bundle interrupted before review is submitted
through the original full source/account validation; a committed review
missing its receipt is recovered without recounting. Partial bundles,
missing paths, different or corrupted receipts, pre-publication
performance and unsupported scope stop explicitly. Original technical
rejections stay terminal. Earlier completed outputs remain; the first
failed stage stops later work. No partial curves are deleted or remade.

The [actual October 11 checks](research/workbench/evidence/weekly_research_queue_20261011_verification.json)
run both real CLI phases: each exits blocked before any registration,
publication query or evaluation step. Synthetic local Git/account tests
exercise future publication and recovery semantics, not an actual
October 17 publication or native delivery. The
[preservation proof](research/workbench/evidence/weekly_research_queue_20261011_preservation.json)
keeps all 120 configurations, 36 reviews, 69 imported files, seven forward
profiles, journal/metadata and engine/head unchanged, with no migration.
The [latest native instructions](research/workbench/evidence/prospective_data_automation_v28.json)
retain daily collection and the genuine weekly gate. No new investment
performance is claimed.

### Dividend-growth identities verified; price coverage still blocked

The [source-only dividend policy](research/workbench/config/dividend-growth-sources.json)
and [finite validator](research/workbench/src/us_quant/dividend_growth_sources.py)
inspect VIG and SDY as two implementations of dividend-policy
persistence, **not two independent factor families**. Their exposures
may overlap the existing quality, value and financing characteristics.
No new definition, allocation candidate or portfolio result is registered.

The [actual source evidence](research/workbench/evidence/dividend_growth_source_readiness_20261011.json)
retains the issuer identities and important differences:

| Fund | Inception | Verified current identity and limits |
|---|---|---|
| VIG | 2006-04-21 | Common-stock full replication; issuer footnote discloses Nasdaq Dividend Achievers Select through 2021-09-19, S&P U.S. Dividend Growers thereafter. |
| SDY | 2005-11-08 | S&P High Yield Dividend Aristocrats; current 20-year dividend-growth screen and indicated-yield weighting, cross-checked against its advertised factsheet. |

Current issuer objectives substantiate ordinary equity index exposure,
not derivatives-free holdings, independently certified historical gross
notional limits or complete historical constituent/screen archives.
VIG's exact consecutive-year screens across history remain unverified.
A search-generated 2021 announcement link redirected to a general
pressroom; that page is not treated as the original dated announcement.
We do not apply the newer benchmark's rules backwards to the older one.

**Complete fund histories are not ready.** Both ordinary VIG/SDY chart
requests returned actual HTTP429 rate-limit responses; there is no
complete in-workbench cache, and zero histories are admitted. NOBL's
issuer request returned HTTP403 and remains excluded. All failed
responses are preserved without switching headers, paths, transport or
endpoints to bypass the restrictions. Synthetic validator tests are not
actual fund history or corporate-action coverage.

`history_from_payload` requires the original 2,805-session calendar,
positive observed prices/volume, correct USD ETF identity and explained
distribution/split adjustment changes; missing history cannot become a
shorter window, a synthetic proxy or a success-shaped fallback. Future
ordinary lawful data availability needs separately preserved evidence
and full verification before any performance test. A separate dividend
hypothesis is not eligible for preparation-based admission before the
October 24 research cycle and still needs actual preregistration.
The October 17 IJR/PKW queue and four existing frozen candidates are unchanged.

The [preservation proof](research/workbench/evidence/dividend_growth_source_preservation_20261011.json)
keeps all 120 configurations, 36 reviews, 69 imported files, seven
forward profiles, exact journal/metadata and engine/head unchanged.
No target, account or performance metric is computed and no engine
migration is performed. The [latest native instructions](research/workbench/evidence/prospective_data_automation_v29.json)
retain these source blocks, daily collection and the original weekly
queue. This is verified source context with an unresolved data block,
**not investment qualification**.

### Preregistered equity-budget attribution across the full recurring cohort

The [descriptive budget-control policy](research/workbench/config/equity-budget-attribution.json)
and [adapter](research/workbench/src/us_quant/equity_budget_attribution.py)
generalize the earlier eight-asset SPY ablation to all **36 completed
recurring reviews**. At each original target, combine ordinary equity
ETF weights into SPY while preserving the exact BIL, GLD, IEF and TLT
weights, invested budget, original decisions/standby rows and horizon seeds.
This measures the combined equity sleeve, including any growth/sector
satellite, not an isolated individual factor or stock-level alpha.

Both legs use the same frozen funded-v2 engine and independent `bt`,
USD10,000 initialization, costs, delays and complete ten/five-year windows.
The original 19 legacy/17 versioned reviews are **not rewritten**;
differences from their preserved original metrics and ledgers are reported
separately so numerical rebasing cannot masquerade as economic value.
Unknown assets, partial targets or missing source paths stop explicitly.

The [actual registration](research/workbench/evidence/equity_budget_attribution_20261011_registration.json)
freezes the policy, code, complete source cohort and all original bundle
references **before** new control accounts are calculated. Execution
requires the exact remote-published registration/code/policy commit.
The [registration preservation check](research/workbench/evidence/equity_budget_attribution_20261011_registration_preservation.json)
retains all original research rows, journal/metadata, seven forward profiles
and engine/head unchanged.

These are **36 derived descriptive controls and 144 planned horizon/cost
pairs**, not 36 newly admitted trading strategies. Matching target equity
budgets does not match realized volatility, beta, turnover or costs, and
the controls inherit the original factor rule's risk schedule.
They are not independent holdouts, causal alpha or additions to the old
selection-inference cohort. Original 120 configurations/36 reviews,
the four W42 candidates, all failures and qualification goals remain.
At registration, no new control outcome or investment qualification
is claimed.

The [completed results](research/workbench/evidence/equity_budget_attribution_20261011_results.json)
contain all **144 paired comparisons**, with 144 reaccounted original
legs and 144 derived SPY-budget legs, each separately rebuilt in `bt`.
Original sleeves exceed the control's net CAGR in **66/144** comparisons
and Sharpe in **63/144**; only 15 of 36 source rules exceed control CAGR
on all four paths. These overlapping horizons/cost scenarios are not
144 independent observations or evidence of universally useful factors.

The preregistered dollar-guard reference illustrates a real but
limited equity-sleeve contribution:

| Window/cost | Source rule CAGR / Sharpe | Same-budget SPY control CAGR / Sharpe |
|---|---|---|
| Ten-year base | 14.25% / 1.0675 | 10.59% / 0.8897 |
| Ten-year stress | 13.68% / 1.0192 | 9.92% / 0.8214 |
| Five-year base | 19.09% / 1.1889 | 14.20% / 1.0019 |
| Five-year stress | 18.10% / 1.1169 | 13.05% / 0.8994 |

Its combined style/semiconductor sleeve adds approximately
3.66–5.05 percentage points of net CAGR versus this budget control,
but increases drawdown by approximately 1.05–1.07 points and incurs
more orders/fees. The unguarded SOXX reference also exceeds its budget
control but still fails the original drawdown limit. The real-yield
tilt's CAGR difference is much smaller: approximately −0.025 to
0.338 percentage points across the four paths.
This does not isolate a particular equity factor or prove causal alpha.
The dollar guard **still misses full-SPY ten-year outperformance**;
its better budget-control comparison does not change qualification.

The [independent verification](research/workbench/evidence/equity_budget_attribution_20261011_reproducibility.json)
checks every control weight with scalar Decimal sums and every account's
window, costs and metrics. Maximum weight/metric differences are
1.3e-16 / 1.06e-14; independent `bt` equity differences are below
1.82e-10 USD. The [CSV-read precision check](research/workbench/evidence/equity_budget_attribution_20261011_read_precision.json)
explains the runtime's at-most 7.28e-12 USD comparison difference:
the saved original and reaccounted equity paths agree exactly when
read with the same parser. This is not an accounting improvement
or a rewritten historical outcome.

The [preservation proof](research/workbench/evidence/equity_budget_attribution_20261011_preservation.json)
retains all 120 strategy configurations/36 reviews, 69 imported files,
seven forward profiles and the exact journal/engine/head.
All 36 new **descriptive controls** and 144 pairwise results are disclosed
separately; none is promoted or added to the trading-candidate catalog.
Original selection diagnostics do not correct these new controls,
and both forward models still have zero observed return sessions.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v30.json)
retain the original weekly queue, costs and goals. The investment
objective remains unmet.

## Preregistered style and satellite component attribution

The whole-equity budget diagnostic does not distinguish four-factor
style exposure from growth/technology/semiconductor satellites.
The [fixed component policy](research/workbench/config/equity-component-attribution.json)
and [actual registration](research/workbench/evidence/equity_component_attribution_20261011_registration.json),
recorded at **2026-10-11 10:09:25 UTC**, cover all 36 reviewed sources.
They specify six single-group replacements—momentum, value, quality,
low volatility, QQQ/XLK and SOXX—plus two aggregate replacements
for all styles and all satellites. This is **288 logical controls /
1,152 window-cost comparisons**, not 288 new trading strategies.

At each original target, only the selected group's allocation moves
to existing SPY. All other stock/defensive targets, invested budgets,
independent window seeds, decision dates and all-NaN pauses remain.
No volatility budget or macro gate is recomputed. The controls
therefore estimate differences **conditional on the original full
rule's risk budget**, not the result of rebuilding that rule without
the selected input. Unselected target weights remain equal, but
changed wealth can change their realized dollar holdings.

The [separate adapter](research/workbench/src/us_quant/equity_component_attribution.py)
reuses the 144 already verified funded-v2 source accounts and their
independent `bt` paths, after exact private artifact hash checks.
Every nonidentity control is rebuilt under the original 10-year /
5-year windows, 5/20 bp costs, USD 1/order and 1/2-session delays.
Groups with no actual source exposure retain explicit identity
comparisons and are excluded from active-control win denominators;
they are not simulated or counted as newly observed control returns.
All results, including unfavorable comparisons, must be disclosed.

These exposed-history, overlapping controls are **not additive
attribution, isolated factor alpha, equal-risk matches or independent
validation**. QQQ/XLK/SOXX are implementation satellites, not three
new factor families. No result may tune thresholds/weights, replace
the frozen forward rules, change the full-SPY qualification gate or
promote a candidate. Original selection bootstrap results do not
correct these new controls.

Forty [focused cases](research/workbench/tests/test_equity_component_attribution.py)
and 29 coupled budget cases pass. The [registration preservation
proof](research/workbench/evidence/equity_component_attribution_20261011_registration_preservation.json)
also records an actual run against the preceding published commit:
it stopped at the publication gate, before creating an output
directory or calculating any component account. All original
120 configurations, 36 reviews, 69 imported files and seven
forward profiles remain unchanged. Registration/code/policy must
be actually pushed before evaluation; at registration there are
**zero new component outcomes** and no investment qualification.

The [completed component results](research/workbench/evidence/equity_component_attribution_20261011_results.json)
now contain all **1,152 logical comparisons**: **824 newly computed
nonidentity controls**, independently rebuilt in `bt`, and **328
explicit identities** reusing the original verified source leg.
The following counts use only actual component exposure, not identities.

| Component replaced by SPY | Original CAGR higher | Original Sharpe higher |
|---|---|---|
| Momentum | 121/144 | 42/144 |
| Value | 140/144 | 139/144 |
| Quality | 11/144 | 11/144 |
| Low volatility | 1/144 | 9/144 |
| QQQ/XLK | 24/24 | 24/24 |
| SOXX | 28/28 | 28/28 |
| All styles jointly | 18/144 | 21/144 |
| All satellites jointly | 52/52 | 52/52 |

The current dollar-guard reference makes the limitation concrete.
Replacing its SOXX allocation with SPY reduces net CAGR by
**4.06–5.36 percentage points** and Sharpe by **0.215–0.242**, while
reducing drawdown by **1.21–1.42 points** across the four paths.
In contrast, replacing its four style funds jointly with SPY
increases net CAGR by **0.205–0.287 points** and Sharpe by
**0.016–0.024**, while increasing drawdown by **0.334–0.368 points**.
The reference's equity-budget advantage therefore cannot be
attributed to four broadly successful style exposures: its much
larger positive conditional difference comes from the semiconductor
satellite, with additional drawdown risk.

Within that same reference, value improves net CAGR by approximately
0.120–0.302 points and lowers drawdown by 0.172–0.177 points versus
its individual replacement. Low volatility lowers drawdown by
0.372–0.396 points but costs 0.261–0.388 points of net CAGR.
Momentum adds 0.160–0.222 points of CAGR with mixed/negligible Sharpe
differences; quality's net CAGR difference is approximately
−0.042 to −0.019 points. Single and aggregate differences **cannot
be summed**: covariance, wealth drift and trade costs differ.
All counts remain descriptive, correlated and conditioned on the
original rule; they do not establish causal or independent alpha.

The [independent verification](research/workbench/evidence/equity_component_attribution_20261011_reproducibility.json)
checks all 125,067 defined control rows with Decimal projection,
all 824 funded/`bt` controls and all 144 reused source accounts.
Maximum target/metric errors are 1.3e-16 / 1.22e-13, and maximum
independent equity discrepancy is 1.82e-10 USD. Both the exact
new artifact set and all 720 previous budget artifact hashes
are verified; 2,472 new CSV artifacts remain private.
The [preservation proof](research/workbench/evidence/equity_component_attribution_20261011_preservation.json)
retains every original journal row, engine/head, imported file,
forward profile and the four W42 plans. There are still 120
completed trading-strategy configurations, no champion and zero
observed return sessions in the two forward models.

No style is removed or weight fitted because of these results.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v31.json)
carry the distinction between sector-dependent historical returns
and independently supported multifactor stability into future
research. The frozen W42 queue and qualification gates remain;
the dollar reference still misses full-SPY ten-year outperformance.
**The investment objective remains unmet.**

## Preregistered covariance-only multifactor construction

The [fixed risk-construction policy](research/workbench/config/factor-covariance-risk.json)
tests two actual trading-strategy configurations, unlike the earlier
descriptive component controls: constrained risk budgeting and maximum
diversification. Both allocate among **MTUM, VLUE, QUAL, USMV, GLD and IEF**,
without SOXX/QQQ/XLK satellites, expected-return forecasts or new macro gates.
They retain four existing style families; covariance allocation methods
are not additional economic factors.

At completed month ends, use the last 252 observed daily returns and
10% diagonal covariance shrinkage, parameters already present in earlier
allocation research. Equity receives 30%–70% of the 98% invested budget;
each style receives 15%–35% of equity. Each of gold and intermediate
Treasuries receives 25%–75% of defense, a symmetric constraint frozen
before outcomes. The remaining 2% is target idle cash. SPY is only the
benchmark and BIL is not an optimized risky asset.

The [target implementation](research/workbench/src/us_quant/factor_covariance_risk.py)
solves a convex positive log-risk-budgeting problem or an equivalent
convex volatility-normalized maximum-diversification problem under
identical homogeneous bounds. Binding constraints can prevent exactly
equal risk contributions: actual signed contributions and deviations
are recorded, not advertised as six independent/equal risk sources.
No volatility/threshold grid, daily overlay, shorting or leverage is used.
Invalid covariance or optimizer failures stop explicitly, without an
equal-weight fallback.

The [source context](research/workbench/evidence/risk_only_covariance_20261011_source_context.json)
confirms the identities of Maillard/Roncalli/Teïletche (2010) and
Choueifaty/Coignard (2008) from primary Crossref metadata. It does **not**
claim access to their full text, abstracts or exact performance replication:
the author-PDF request failed in transport and the publisher DOI returned
403. The initial ASCII-only author-name validator failure is preserved;
the original accented name and raw response were not modified, and no
missing acquisition timestamp was invented. Raw metadata remains private.

The [independent target proof](research/workbench/evidence/risk_only_covariance_20261011_target_verification.json)
checks 244 fixed targets / 122 complete month ends per configuration,
reconstructing covariance from original CSV prices and checking convex
KKT stationarity/complementarity, positivity, funding and every group bound.
Maximum covariance/weight differences are 1.11e-16 and the maximum
stationarity residual is 8.68e-7. These are target/optimization checks,
not strategy returns or proof of the investment goal.

The [actual registration](research/workbench/evidence/risk_only_covariance_20261011_registration.json)
records both candidates at **2026-10-11 10:39:18 / 10:39:19 UTC**,
before any new strategy account. A [separate study runner](research/workbench/src/us_quant/risk_only_study.py)
requires the actual configured-remote publication of the exact registration,
code, policy and source/target proof before writing a durable account fence
and invoking the original four-path source-bound evaluator.
Local or unrelated commits and existing partial output directories stop.

One exact old-engine/head migration adds only the finite target generator.
The [preservation proof](research/workbench/evidence/risk_only_covariance_20261011_registration_preservation.json)
retains all prior 120 completed configurations, 36 reviews / 144 audited
paths, 19 legacy / 17 versioned accounts, 69 imported files and seven
forward profiles. Only the two actual candidate registrations are pending;
the four W42 plans and IJR/PKW definition queue remain unchanged.
146 coupled cases and the 52 root guards pass. An actual evaluation
against the preceding published commit stopped before output/accounts.

The [current registration snapshot](research/workbench/evidence/research_program_2026-W41_risk_registered_v44.json)
and [source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v18.json)
preserve every old outcome under the new replay engine.
Both new methods are formed on already exposed history; original selection
diagnostics do not include them. Risk diversification can lower CAGR,
Treasuries/gold can lose together and covariance bounds cannot guarantee
15% drawdown or Sharpe above 1. At registration there are **zero new
strategy outcomes**, no champion and no investment qualification.

The [completed risk-only results](research/workbench/evidence/risk_only_covariance_20261011_results.json)
now retain both candidates and every original base/stress path.
Neither candidate passes any of the Sharpe, full-SPY outperformance
or 15% drawdown tests on either horizon:

| Method/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Risk budgeting, ten-year | 7.50% / 0.6495 / 18.37% | 7.40% / 0.6386 / 18.39% |
| Risk budgeting, five-year | 7.78% / 0.4595 / 18.50% | 7.55% / 0.4349 / 18.53% |
| Maximum diversification, ten-year | 7.09% / 0.6361 / 17.99% | 6.96% / 0.6200 / 17.98% |
| Maximum diversification, five-year | 7.10% / 0.4046 / 18.12% | 6.86% / 0.3780 / 18.11% |

The worst drawdown runs from **2021-12-27 to 2022-09-27** on all
eight paths, with each window independently capitalized. These fixed
portfolios do not turn covariance diversification into incremental
return alpha or a guaranteed defensive allocation. Removing sector
satellites and avoiding mean forecasts did not solve the objective;
the observed 2022 drawdown still breaches the stability limit.
No bounds, covariance window or allocation method is revised after
these failures.

The [independent account proof](research/workbench/evidence/risk_only_covariance_20261011_reproducibility.json)
reconstructs all eight strategy/benchmark comparisons and 16 funded
plus 16 `bt` accounts, using the exact calendars, initial capital,
risk-free rates, costs, delays and initial-capital drawdown peaks.
Prepared versus seeded targets agree exactly; maximum scalar metric
difference is 1.28e-13 and independent equity discrepancy is
6.55e-11 USD. Strict joint gate decisions reproduce both rejections.

The [current result snapshot](research/workbench/evidence/research_program_2026-W41_risk_results_v45.json)
has **122 completed configurations = 84 prior + 38 recurring reviews**,
40 registered code versions including the two previously invalid
conditional versions, no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v19.json)
reexecutes all 38 reviews / 152 paths, preserving 19 legacy and
19 explicit funded-v2 outcomes.

The [final preservation proof](research/workbench/evidence/risk_only_covariance_20261011_preservation.json)
keeps every old journal row/event prefix, 69 imported files and
all seven prospective profile states/control bytes. Only the one
precise replay migration, two candidate registrations and two rejected
reviews are appended. All 720 old budget and 2,472 component artifact
hashes also remain intact; their frozen 36-source scope is not relabelled
as a 38-source diagnosis. The new two genuine strategy trials are
counted separately from those descriptive controls.

Six economic definitions / five catalog families, the original four
unregistered W42 plans, IJR/PKW quota and the two frozen forward models
are unchanged. Both forward models still have zero actual return
sessions; old selection diagnostics do not correct these two new trials.
The [updated native continuation](research/workbench/evidence/prospective_data_automation_v32.json)
preserves these rejected outcomes and the remaining weekly/forward
work. The more stable dollar-guard reference still misses full-SPY
ten-year outperformance. **The investment objective remains unmet.**

## Preregistered diversified-sector multifactor construction

The [sector-source policy](research/workbench/config/sector-basket-sources.json)
and [source readiness](research/workbench/evidence/sector_basket_source_readiness_20261011.json)
verify **all nine original Select Sector SPDR ETFs**:
XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY. Each actual issuer page's scoped
fund identity is cross-checked with its uniquely advertised PDF factsheet.
All 2,805 post-inception sessions per fund are revalidated from the
original registered raw/CSV cache, with independent Decimal price error
at most 5e-10 and no unexplained adjustment jumps. **No new price request**
is made; the original 2026-10-06 acquisition times remain.
Current issuer mandates are not complete historical sector definitions,
constituents or independently certified notional-cap archives.

The old first-round reader remains blocked because its implementation
fingerprint changed after that historical registration. Original policy,
windows, manifest, raw and CSV hashes still match. The [versioned source
adapter](research/workbench/src/us_quant/sector_basket_sources.py) therefore
checks **market prices only**, without altering the old reader/registration
or claiming to reproduce the original strategy returns.

The [fixed strategy policy](research/workbench/config/sector-basket-strategy.json)
keeps half of an equity bucket in all four equally weighted style ETFs;
the other half follows the original nine-sector, 252-minus-21-session
positive momentum / 200-day trend selection, top three, 63-day inverse
volatility and 40% standalone-NAV cap. Missing slots and cap residuals
remain actual BIL rather than being redistributed into surviving sectors.
This is another implementation of momentum, not nine new factor families.
The skip-only selector needs 252 returns, not the old residual model's
756-session regression warmup; neither old code nor old outcomes is changed.

The two configurations are the plain basket and the same basket with
the already frozen own-vintage 63-session positive-dollar **half-risk**
guard. Both use the original 63-session equity/gold risk balance,
30%–70% equity-bucket bounds and 98% total target investment, monthly
decisions only. A bucket can contain BIL when sector slots are missing,
so actual stock weight can be below 30%; it is not renormalized.
No single-sector winner is selected from new account results, and no
new macro confirmation or daily overlay is introduced.

The [first target proof](research/workbench/evidence/sector_basket_strategy_20261011_target_verification.json)
and [complete-dependency recheck](research/workbench/evidence/sector_basket_strategy_20261011_target_verification_v2.json)
independently verify 244 defined targets / 122 month ends per candidate.
Selected sector counts range from zero to three; 68 dollar-risk months
apply the second candidate's half scale. Scalar/Decimal target, momentum
and own-vintage dollar errors are at most 2.22e-16 / 3.45e-15 / 3.02e-16.
The later complete specifications retain every prior target byte.

Two unregistered preparation/admission failures are separately retained:
a rounding-only negative unit-BIL residual now clips only within 1e-12,
and the reused dollar helper initially omitted required transitive
bindings. Neither produced a registered terminal strategy version or
any portfolio return. The existing exact migration was **resumed, not
repeated or reset**, after completing those bindings. Original logs,
code, failed inputs and the first proof remain private/immutable.

The [actual registration](research/workbench/evidence/sector_basket_strategy_20261011_registration.json)
records the two strategies at **2026-10-11 11:29:47 / 11:29:50 UTC**.
The [publication-gated runner](research/workbench/src/us_quant/sector_basket_study.py)
requires the exact published packet, policy, code and source/target proof
on this branch's configured remote before a durable account fence and
the original four-path funded-v2 / independent-`bt` review.
An actual attempt with the preceding published commit stops before
output/accounts; existing partial directories cannot be overwritten.

The [registration preservation](research/workbench/evidence/sector_basket_strategy_20261011_registration_preservation.json)
and [current snapshot](research/workbench/evidence/research_program_2026-W41_sector_registered_v46.json)
retain **122 completed configurations / 38 reviews**, 42 registered
versions including the two old invalid ones, and only these two new
pending candidates. The [full old-cohort audit](research/workbench/evidence/research_program_source_bound_audit_v20.json)
reexecutes 38 reviews / 152 paths without changing any old outcome.
176 coupled cases and 52 root guards pass; 69 imported files and all
seven forward profile states/control bytes remain.
The four W42 plans and IJR/PKW queue are not replaced.

This source/construction work uses exposed history and correlated
style/sector stocks. Prior selection diagnostics do not cover these
two new strategies. At registration, **zero new strategy outcomes**
and no investment qualification are claimed; the full-SPY and
Sharpe/drawdown gates remain unchanged.

The [completed sector-basket results](research/workbench/evidence/sector_basket_strategy_20261011_results.json)
retain both configurations and all eight window/cost paths:

| Method/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Plain sector basket, ten-year | 11.70% / 0.8286 / 17.37% | 11.28% / 0.7918 / 16.45% |
| Plain sector basket, five-year | 14.37% / 0.8735 / 16.71% | 13.73% / 0.8228 / 16.62% |
| Dollar-half basket, ten-year | 9.02% / 0.7778 / 12.23% | 8.36% / 0.7012 / 10.60% |
| Dollar-half basket, five-year | 12.27% / 0.9264 / 9.24% | 11.28% / 0.8220 / 9.17% |

The plain basket exceeds cost-matched SPY only in the five-year
comparisons; its stress result still fails the additional base-cost
SPY hurdle. All its paths miss Sharpe above 1 and drawdown at most 15%.
The dollar-half basket meets the drawdown limit on all paths, but
none meets Sharpe or SPY outperformance. Thus a diversified sector
satellite lowers the previous single-industry dependence without
solving the joint return/stability objective. No sector, weight,
lookback or dollar scale is refitted after these rejections.

The [independent account proof](research/workbench/evidence/sector_basket_strategy_20261011_reproducibility.json)
checks eight complete strategy/SPY comparisons, 16 funded and
16 independent `bt` accounts, exact costs, delays, rates and
initial-capital drawdown peaks. Maximum prepared/seeded target,
scalar metric and independent equity differences are
3.34e-16 / 5.76e-14 / 6.55e-11 USD. Both strict joint rejections reproduce.
The [proof-scope note](research/workbench/evidence/sector_basket_strategy_20261011_proof_scope.json)
clarifies that the retained v2 target proof ran **after** the one
completed migration and **before** candidate registration: its
pre-migration-record preservation label does not mean that current
engine metadata or the migration event was unchanged. No original
proof/registration hash or strategy outcome is rewritten.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_sector_results_v47.json)
contains **124 completed configurations = 84 prior + 40 recurring reviews**,
42 registered code versions including the two old invalid ones,
no pending actual candidate and no champion. The [full source-bound
audit](research/workbench/evidence/research_program_source_bound_audit_v21.json)
reexecutes all 40 reviews / 160 paths, retaining 19 legacy and
21 funded-v2 accounts.
The [preservation proof](research/workbench/evidence/sector_basket_strategy_20261011_preservation.json)
retains every original journal row/event prefix, 69 imported files,
all seven forward profiles and all 3,192 old descriptive-control artifacts.

Six definitions / five catalog families and the original four
unregistered W42 plans / IJR-PKW queue remain. Both frozen forward
models still have zero actual return sessions. Original selection
diagnostics do not correct the new four risk-only/sector trials or
all prior searches. The [native continuation](research/workbench/evidence/prospective_data_automation_v33.json)
retains these failures without changing costs, windows, forward rules
or the full-SPY gate. **The investment objective remains unmet.**

## Preregistered semiconductor portfolio-floor protection

The [fixed protection policy](research/workbench/config/semiconductor-portfolio-protection.json)
transfers the **already frozen 15% closing-high-water ratchet and
5-percentage-point actual-holding trade band** to two original sources:
the four-style/SOXX/gold monthly rule and its dollar-only half-risk rule.
Asset shares, monthly 63-session risk balance, dollar availability and
all original costs/windows remain. There is no new protection threshold,
sector, lookback, economic factor or macro-confirmation grid.

The [adapter](research/workbench/src/us_quant/semiconductor_portfolio_protection.py)
reuses the unchanged `protect_target` law from the original growth
protection study. It preserves existing monthly BIL from the dollar
guard and moves only additional removed risky capital to BIL.
Each path starts from fresh USD 10,000 and its own closing high-water
mark; costs and delayed fills change subsequent protection decisions.
Only one modeled request can be pending, it is never overwritten,
and a month-end update remains requested while an older order is pending.
Gaps, fees, delays, the trade band and cash lock-in can still breach
15% drawdown or sacrifice return; this is not guaranteed insurance.

The [pre-account source/rule proof](research/workbench/evidence/semiconductor_protection_20261011_pre_account_verification.json)
checks all eight original source target paths including initialization.
Prepared versus original seeded sources agree exactly.
1,310 Decimal floor-law probes on the actual 131 monthly sources
reproduce the adapted risk/BIL targets within 2.22e-16; the probe NAVs
are synthetic examples, **not newly calculated strategy accounts**.
No new quotes, protected historical returns or forward observations
are computed at preparation.

A shared [registered-study runner](research/workbench/src/us_quant/registered_study.py)
reuses the original candidate receipts, finite target generator,
configured-remote publication verification and four-path evaluator.
It requires two exact existing-factor specifications and a pre-account
proof, verifies both original registration events, and requires actual
remote publication of the frozen code/policy/metadata before writing
a durable account fence and evaluating the candidates.
The runner itself does not perform Git publication or expose raw inputs.
It cannot register economic definitions, change strategy parameters,
skip an outcome or overwrite an existing partial account directory.
This replaces the need for another study-specific publication wrapper,
without changing previously frozen wrappers.

The [actual registration](research/workbench/evidence/semiconductor_protection_20261011_registration.json)
records the two candidates at **2026-10-11 11:57:43 / 11:57:46 UTC**,
before any protected account. The [registration preservation
proof](research/workbench/evidence/semiconductor_protection_20261011_registration_preservation.json)
retains 124 completed configurations / 40 reviews / 160 audited paths,
69 imported files and seven original forward profiles. One exact
old-engine/head migration adds the finite replay route; only the two
new registrations are pending, with no completed outcome or champion.
The original W42 four plans and IJR/PKW queue are unchanged.
206 coupled cases and 52 root guards pass; an actual evaluation using
the preceding published commit stops before output/accounts.

The [current registration snapshot](research/workbench/evidence/research_program_2026-W41_protection_registered_v48.json)
and [old-cohort source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v22.json)
preserve all old results. These two hypotheses were formed after
124 exposed-history configurations. Original selection diagnostics
do not cover them, and both frozen forward models still have zero
actual return sessions. Publication must precede account calculation;
**the investment objective remains unmet**.

The [completed protection results](research/workbench/evidence/semiconductor_protection_20261011_results.json)
retain both configurations and all eight original window/cost paths.
All paths meet the modeled 15% drawdown limit, but **none** meets
Sharpe above 1 or either required SPY outperformance comparison:

| Protected source/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Plain SOXX/factor/gold, ten-year | 9.76% / 0.7143 / 12.83% | 7.29% / 0.5310 / 13.78% |
| Plain SOXX/factor/gold, five-year | 12.83% / 0.7646 / 12.92% | 11.78% / 0.6748 / 13.06% |
| Dollar-half source, ten-year | 9.35% / 0.8001 / 10.56% | 8.42% / 0.7112 / 10.48% |
| Dollar-half source, five-year | 13.22% / 0.9288 / 10.70% | 12.13% / 0.8180 / 10.17% |

Protection lowers exposure and adds substantial trading. The plain
source's ten-year base account has 418 nonmonthly protection requests,
3,771 executed asset orders and USD 4,368.14 total modeled fees.
Its stress path has 304 nonmonthly requests, 2,974 asset orders and
USD 4,547.37 fees; risk falls as low as **9.46% of the already known
monthly risky baseline**, not 9.46% gross portfolio exposure.
The dollar-half source also adds protection requests and fees.
There are zero modeled closing-floor breaches on these eight paths,
but the reduced exposure, costs and delayed risk recovery do not
preserve the original return objective. Zero observed breaches are
not protection against future opening gaps or different market paths.

The [independent feedback/account proof](research/workbench/evidence/semiconductor_protection_20261011_reproducibility.json)
reconstructs **15,064 daily NAV/peak/floor/scale/pending-order states**
from recorded actual wealth and weights, without calling the protection
target generator. Every issued request, original BIL balance, monthly
update, 5-point band and delayed execution is checked against the
retained target matrix. All 16 funded strategy/SPY and 16 independent
`bt` accounts preserve costs, rates, fresh capital and complete calendars.
Maximum target/metric/equity differences are
1.34e-15 / 5.05e-14 / 1.71e-10 USD. Both strict rejections reproduce.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_protection_results_v49.json)
has **126 completed configurations = 84 prior + 42 recurring reviews**,
44 registered versions including the two old invalid versions,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v23.json)
reexecutes 42 reviews / 168 paths, preserving 19 legacy and 23
explicit funded-v2 reviews. The [preservation proof](research/workbench/evidence/semiconductor_protection_20261011_preservation.json)
keeps every old journal row/event prefix, 69 imported files,
seven original forward profiles and all 3,192 old control artifacts.

No floor, trade band, dollar threshold or asset share is retuned.
Six definitions / five catalog families, the original four unregistered
W42 plans and both frozen forward models remain; neither forward model
has an actual return session yet. Old selection diagnostics do not
correct these two new trials or the complete search.
The [native continuation](research/workbench/evidence/prospective_data_automation_v34.json)
retains the rejected protection paths and actual weekly/daily schedule.
**The investment objective remains unmet.**

## Preregistered joint dollar / option-risk confirmation

The [fixed conjunction policy](research/workbench/config/joint-risk-confirmation.json)
tests whether the already known positive 63-session dollar state
requires **simultaneous VIX ≥ VIX3M** confirmation before halving
the original four-style/SOXX/gold risk. It changes the earlier
option-only / OR logic to AND, not the original ratio 1.0,
strict positive-dollar threshold, 0.5 scale, asset shares or costs.
Dollar strength can reflect healthy relative growth; the conjunction
is a hypothesis, not evidence that this separates those states.

The two [target constructions](research/workbench/src/us_quant/joint_risk_confirmation.py)
are completed-month confirmation and daily option-state confirmation.
Both update dollar risk only at the original month end.
The first holds the joint decision until the next month; the second
reacts to daily option pressure using the **already recorded monthly
dollar regime**, not a newly observed daily dollar quote or relaxed
source-age rule. Targets appear only on a monthly allocation update
or, for the second rule, a change of its joint binary risk state.
All other rows remain empty and actual holdings continue.

All original official Cboe close CSVs/documents and H.10 vintages
are retained, with no new source request. Current Cboe methodology
and closing histories do not establish complete unrevised historical
first-publication timestamps; only later-open execution is modeled.
The original dollar availability, conservative conflicts, quarantine,
2019 unit regimes and 14-day observation age remain.
These risk observations are not portfolio index holdings or new
independent economic-factor families.

The [pre-account proof](research/workbench/evidence/joint_risk_confirmation_20261011_pre_account_verification.json)
independently reconstructs all raw option states, own-query dollar
comparisons and target truth-table events. Monthly confirmation has
131 targets / 6 risk-off targets. Daily confirmation has 209 targets /
44 risk-off targets, including 78 nonmonthly state changes.
Option, dollar and target errors are 0 / 3.02e-16 / 5.56e-17.
No candidate account or portfolio metric is computed before registration.

The [actual registration](research/workbench/evidence/joint_risk_confirmation_20261011_registration.json)
records both candidates at **2026-10-11 12:23:13 / 12:23:16 UTC**.
The shared registered-study runner requires exact actual remote
packet/code/policy/proof publication and a durable fence before
the original four-path funded-v2 / independent-`bt` evaluator.
An actual run against the preceding published commit stops before
creating an account/output directory.

The [registration preservation](research/workbench/evidence/joint_risk_confirmation_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_joint_registered_v50.json)
and [old-cohort audit](research/workbench/evidence/research_program_source_bound_audit_v24.json)
retain 126 completed configurations / 42 reviews / 168 audited paths,
19 legacy / 23 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact old-engine/head migration adds the finite
generator; only the two actual candidates are pending, among 46
registered versions including the two old invalid ones.
209 coupled cases and 52 root guards pass. The original W42 plans,
factor quota and frozen forward rules remain.

The two hypotheses use already exposed historical windows;
old selection diagnostics do not correct these new trials.
At registration there are **zero new strategy outcomes**, no champion,
and no independent future-alpha or investment qualification claim.

The [complete joint-confirmation results](research/workbench/evidence/joint_risk_confirmation_20261011_results.json)
retain both timings and all eight cost/window paths:

| Confirmation/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Monthly AND, ten-year | 17.99% / 1.1039 / 20.85% | 17.58% / 1.0761 / 20.80% |
| Monthly AND, five-year | 23.59% / 1.1612 / 21.00% | 22.62% / 1.1105 / 20.96% |
| Daily AND, ten-year | 16.57% / 1.0128 / 21.73% | 14.43% / 0.8796 / 25.18% |
| Daily AND, five-year | 21.66% / 1.0618 / 22.07% | 18.67% / 0.9121 / 25.67% |

The monthly rule passes **all four Sharpe and cost-matched SPY gates**,
including both stress versus base-SPY comparisons. Its minimum Sharpe
is 1.0761, but **every drawdown gate fails**, reaching 21.00% in
the five-year base path. Its worst episodes span 2022-03-08 to
2022-10-14. This meets the return/Sharpe portion, not the retained
15% stability requirement or independent future-validation requirement;
it remains a rejected historical candidate, not a champion.

Daily confirmation does not improve that stability. Both base
Sharpe/SPY comparisons pass, but stressed Sharpe falls below 1,
the ten-year stress SPY comparisons fail, and maximum drawdown
reaches 25.67%. The ten-year monthly stress account executes 737
asset orders / USD 1,449.22 fees; daily stress executes 1,209 /
USD 3,749.59 with more state reversals and different delayed fills.
More timely confirmation is not necessarily better risk control.
Neither ratio, dollar threshold, scale nor timing is tuned after
these outcomes, and the failures are not omitted.

The [independent proof](research/workbench/evidence/joint_risk_confirmation_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt` accounts,
all proven event dates/empty rows, monthly dollar regimes, independent
window seeds, costs, rates and original calendars. Prepared versus
actual seeded targets agree exactly. Maximum scalar metric / equity
differences are 3.17e-14 / 9.46e-11 USD; both strict joint rejections
reproduce despite the monthly candidate's partial goal passes.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_joint_results_v51.json)
contains **128 completed configurations = 84 prior + 44 recurring reviews**,
46 registered code versions including the two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v25.json)
reexecutes all 44 reviews / 176 paths, preserving 19 legacy and 25
explicit funded-v2 reviews. The [preservation proof](research/workbench/evidence/joint_risk_confirmation_20261011_preservation.json)
retains all old journal rows/event prefix, 69 imported files, seven
forward profiles and all 3,192 prior control artifacts.

The original four unregistered W42 plans and six definitions /
five catalog families remain. Both frozen forward models still
have zero actual return sessions. Original selection diagnostics
do not cover these new two candidates or the complete search.
The [updated native continuation](research/workbench/evidence/prospective_data_automation_v35.json)
records both the monthly return/Sharpe passes and the failed stability
limit without weakening any gate or replacing a forward rule.
**The investment objective remains unmet.**

## Preregistered transfer of the original volatility-control law

The [fixed transfer policy](research/workbench/config/semiconductor-volatility-target.json)
uses the **existing 63-session covariance, 12% predicted-volatility
target and 5-percentage-point last-issued-target band** on two
original monthly sources: plain four-style/SOXX/gold and the monthly
dollar/option AND-confirmation rule. These parameters and the
projection helper were frozen in the earlier growth-satellite study;
none is selected to turn the latest 21% drawdown into 15%.
The monthly AND source's exposed return/Sharpe passes motivate the
new experiment, not an independent holdout or permission to change
sector shares.

The [target adapter](research/workbench/src/us_quant/semiconductor_volatility_target.py)
reuses unchanged `cap_volatility` on the complete 12-asset trailing
sample covariance, including BIL. It only reduces non-BIL source
weights and transfers the remainder to BIL, preserving existing
bills and the 98% invested budget without borrowing or upscaling.
Initial/monthly source updates issue targets; between month ends,
the desired target must differ from the **last issued target** by
at least five points, including the original tiny comparison tolerance.
This is not an actual-holding band or continuously enforced risk limit:
drift, gaps, in-band changes and delayed fills can leave realized risk
above the prediction. Dated requests remain in the original 1/2-session
queues rather than being cancelled by a later close.

The [pre-account proof](research/workbench/evidence/volatility_transfer_20261011_pre_account_verification.json)
matches all eight original seeded source paths exactly and independently
checks **5,454 daily projections** using scalar portfolio-return
variances and the quadratic feasible-scale solution, without calling
the production covariance/projection helper.
Target/forecast errors are 2.78e-15 / 3.89e-16.
The plain source emits 197 targets; the confirmed source emits 190;
both retain 131 monthly updates. Existing source BIL is never removed.
A private verifier JSON-count serialization failure is separately
preserved; its integer fix changes no candidate code, target or rule.
No new quotes or strategy accounts are computed at preparation.

The [actual registration](research/workbench/evidence/volatility_transfer_20261011_registration.json)
records both candidates at **2026-10-11 12:53:00 / 12:53:03 UTC**.
The shared publication-gated runner requires the exact remote
packet/code/policy/proof and a durable fence before all four
funded-v2 / independent-`bt` paths. An actual attempt using the
preceding published commit stops before output/accounts.

The [registration preservation](research/workbench/evidence/volatility_transfer_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_vol_registered_v52.json)
and [old source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v26.json)
retain 128 completed configurations / 44 reviews / 176 paths,
19 legacy / 25 funded-v2 outcomes, 69 imported files and seven forward
profiles. One exact old-engine/head migration adds the finite generator;
only these two candidates are pending, among 48 registered versions
including the two old invalid ones. 190 coupled cases and 52 root
guards pass. Original W42 plans and frozen forward models remain.

Both hypotheses use already exposed history, and prior selection
diagnostics do not cover them or all search. A volatility forecast
is not independent alpha or guaranteed 15% drawdown protection.
At registration there are **zero new strategy outcomes** and no
investment qualification.

The [completed transfer results](research/workbench/evidence/volatility_transfer_20261011_results.json)
retain both configurations and all eight original cost/window paths:

| Source/window with original vol12 control | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Plain SOXX/factor/gold, ten-year | 13.59% / 0.9557 / 18.76% | 12.93% / 0.8975 / 18.77% |
| Plain SOXX/factor/gold, five-year | 16.98% / 1.0154 / 19.07% | 16.26% / 0.9652 / 19.06% |
| Monthly AND source, ten-year | 13.73% / 0.9981 / 18.98% | 13.42% / 0.9699 / 18.86% |
| Monthly AND source, five-year | 17.16% / 1.0408 / 19.27% | 16.40% / 0.9873 / 19.15% |

Both five-year base paths pass Sharpe and the five-year SPY comparisons;
five-year stress still beats SPY but misses Sharpe strictly above 1.
Neither ten-year path passes Sharpe or SPY, and **all drawdown paths
exceed 15%**. In particular, 0.9981 is not rounded into a passing
ten-year Sharpe. Risk reduction sacrifices the prior monthly AND
source's full return/Sharpe passes without bringing drawdown into range.
No 12% level, covariance window, target band or source weight is retuned.

The [risk-limit interpretation](research/workbench/evidence/volatility_transfer_20261011_risk_limit_diagnostics.json)
distinguishes desired forecasts, actual held-weight forecasts and
realized account volatility. The confirmed ten-year base account's
sample annualized net daily-return volatility is **11.08%**, yet its
drawdown is **18.98%**. Lower realized volatility does not guarantee
a 15% maximum loss. That account also has 616 / 2,512 closes where
the current held-weight, trailing-63-session risk estimate exceeds
12%, reaching 17.09%; target hysteresis, drift and delayed fills mean
the desired cap is not a continuously enforced position limit.
These complete eight-path descriptions are post-outcome observations,
not extra strategy trials, causal attribution or new optimization.

The [independent account proof](research/workbench/evidence/volatility_transfer_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt` accounts,
the previously proven 5,454 quadratic projections/event dates,
empty target rows, independent seeds and all original costs/rates/calendars.
Prepared versus actual seeded targets agree exactly; maximum
scalar metric / equity differences are 4.23e-14 / 6.92e-11 USD.
Both complete joint rejections reproduce.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_vol_results_v53.json)
contains **130 completed configurations = 84 prior + 46 recurring reviews**,
48 registered code versions including the two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v27.json)
reexecutes 46 reviews / 184 paths, retaining 19 legacy and 27
explicit funded-v2 outcomes. The [preservation proof](research/workbench/evidence/volatility_transfer_20261011_preservation.json)
keeps every old journal row/event prefix, 69 imported files, seven
forward profiles and all 3,192 original control artifacts.

Six definitions / five catalog families, original W42 four plans
and both frozen forward models remain; neither model has an actual
return session. Prior selection diagnostics do not include these
two new trials or the complete search.
The [native continuation](research/workbench/evidence/prospective_data_automation_v36.json)
retains both failed transfers and the distinction between variance
targeting and drawdown protection, without weakening the joint goals
or changing forward rules. **The investment objective remains unmet.**

## Preregistered transfer of fixed price-trend risk rules

The [fixed price-trend policy](research/workbench/config/semiconductor-trend-transfer.json)
transfers two already specified trend laws to the original
four-style/SOXX/gold monthly allocation: **daily 200-observation
component trend gating** and **monthly 63/126/252-session relative
trend fractions versus actual BIL**. No dollar/option confirmation,
new horizon, asset-weight grid, volatility target or net-wealth
floor is added. The half-equity SOXX / equal four-style remainder,
monthly 63-session risk balance, 30%–70% source equity budget,
98% investment and all original cost/window gates remain.

The [target implementation](research/workbench/src/us_quant/semiconductor_trend_transfer.py)
builds an **untraded signal index** from contemporaneous observed
fund returns: one-half SOXX and one-eighth each of the four styles.
It starts at one on the first actual source-price date. This is not
a fee-free account, a reconstructed ETF or independent market
observation; actual portfolios trade the original funds and pay fees.
Gold's signal uses the actual GLD adjusted price.

The daily method reuses the unchanged prior four-style component
gate helper, extending its equity risk state to SOXX. The entire
equity sleeve or gold can be disabled when its signal is not
strictly above its 200-observation mean; equality is risk-off.
The first fully informed target, state changes and original month
updates emit requests. The monthly method independently scales
each component by the fraction of the three fixed horizons where
its return strictly exceeds BIL's same-horizon return.
Removed risk goes only to BIL, without increasing the other
component or reimposing an equity minimum during risk-off.
The methods' horizons and clocks differ according to prior rules,
so their comparison is not isolated causal timing attribution.

The [pre-account proof](research/workbench/evidence/trend_transfer_20261011_pre_account_verification.json)
matches all four original source paths including initialization,
reconstructs the full composed signal from actual CSV prices and
checks every strict trend state / target with scalar Decimal
arithmetic. Source / index / target errors are
0 / 1.60e-14 / 1.11e-16. Daily gating emits 253 targets,
including 128 nonmonthly state changes; monthly relative trends
emit 122 targets. No new quote, candidate account or portfolio
metric is computed during preparation.

The [actual registration](research/workbench/evidence/trend_transfer_20261011_registration.json)
records both candidates at **2026-10-11 13:18:06 / 13:18:09 UTC**.
The shared registered-study runner requires exact configured-remote
publication of the packet/code/policy/proof, then a durable fence
before all four funded-v2 / independent-`bt` paths.
An actual attempt using the preceding published commit stops before
account calculation or output-directory creation.

The [registration preservation](research/workbench/evidence/trend_transfer_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_trend_registered_v54.json)
and [old source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v28.json)
retain 130 completed configurations / 46 reviews / 184 paths,
19 legacy / 27 funded-v2 outcomes, 69 imported files and seven
forward profiles. One precise old-engine/head migration adds the
finite generator; only these two candidates are pending, among
50 registered code versions including the two old invalid ones.
202 coupled cases and 52 root guards pass. Original W42 plans,
IJR/PKW quota and both frozen forward models remain.

These hypotheses use already exposed historical windows.
Trend timing can whipsaw, miss gaps or lose upside; prior selection
diagnostics do not include them or the complete search.
At registration there are **zero new strategy outcomes**, no
champion and no independent future-alpha or investment qualification.

The [completed trend-transfer results](research/workbench/evidence/trend_transfer_20261011_results.json)
retain both rules and all eight window/cost paths:

| Fixed trend rule/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Daily 200-observation components, ten-year | 13.10% / 0.8914 / 19.28% | 11.20% / 0.7436 / 18.38% |
| Daily 200-observation components, five-year | 17.07% / 0.9362 / 19.99% | 14.93% / 0.7969 / 18.73% |
| Monthly relative trends, ten-year | 14.55% / 0.9990 / 15.13% | 13.86% / 0.9479 / 14.68% |
| Monthly relative trends, five-year | 19.33% / 1.1090 / 13.15% | 18.39% / 1.0482 / 13.28% |

**Monthly relative trends pass every five-year base/stress gate**,
including stress versus base-cost SPY. They do not pass the joint
ten-year/five-year objective: ten-year CAGR remains below SPY
(14.55% / 13.86% versus 15.46% / 15.41%), both ten-year Sharpe
values remain below 1, and base drawdown is **15.1265%**, not at
most 15%. The exact base Sharpe **0.998955** is not rounded into
acceptance. The ten-year base worst episode is 2020-02-19 to
2020-03-19, distinct from the independently initialized five-year
worst episode. Only checking the latter would omit a real failure.

Daily 200-observation gating only exceeds SPY on the five-year
paths, while all its Sharpe and drawdown gates fail. Its ten-year
stress account has 1,450 asset orders / USD 5,367.10 modeled fees,
compared with 776 / USD 1,954.93 for monthly relative trends.
These two rules also differ in horizon/sign construction, so the
comparison does not isolate trading frequency as the cause.
Neither horizon, timing, component share nor risk threshold is
refitted after the results.

The [independent account proof](research/workbench/evidence/trend_transfer_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt` accounts,
the proven signal states/event dates, complete source/target matrices,
independent seeds, original costs/rates/calendars and initial-capital
drawdown peaks. Prepared versus actual seeded targets agree exactly.
Maximum scalar metric / equity errors are 3.96e-14 / 6.55e-11 USD.
Both full-window joint rejections reproduce despite the relative
trend rule's complete five-year passes.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_trend_results_v55.json)
contains **132 completed configurations = 84 prior + 48 recurring reviews**,
50 registered code versions including the two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v29.json)
reexecutes 48 reviews / 192 paths, retaining 19 legacy and 29
explicit funded-v2 reviews. The [preservation proof](research/workbench/evidence/trend_transfer_20261011_preservation.json)
keeps all old journal rows/event prefix, 69 imported files, seven
original forward profiles and all 3,192 prior control artifacts.

Six definitions / five catalog families, the original four W42
plans and both frozen forward models remain; neither model has
an actual return session. Prior selection diagnostics do not
correct these two trials or the complete search.
The [native continuation](research/workbench/evidence/prospective_data_automation_v37.json)
retains the five-year passes and ten-year failures without
relabelling the shorter window as independent validation or
weakening any gate. **The investment objective remains unmet.**

## Preregistered routing of relative-trend residuals to Treasuries

The [fixed residual-Treasury policy](research/workbench/config/residual-treasury-rotation.json)
keeps the preceding monthly relative-trend rule's entire
four-style/SOXX/gold target matrix and monthly decision dates.
Only its original BIL allocation may change: IEF/TLT qualify
when their actual 126-session adjusted return strictly exceeds
BIL's same-horizon return. One construction selects the strongest
eligible Treasury, with lexical ticker ties; the other uses all
eligible Treasuries with 63-session inverse sample-volatility
weights. No qualifying Treasury leaves BIL unchanged.

These selection methods and horizons come from the existing
defensive-rotation study, but this is explicitly a **Treasury-only
residual adapter**, not a replay of that study's entire portfolio.
Its old rule also selected gold and recalculated equity risk budgets.
Neither behavior is copied: original gold trend decisions stay
intact, and equity/gold are not renormalized. The 98% target
investment/2% idle cash, all windows, fees and joint gates remain.
Treasury duration can add drawdown, including during rising rates.

The [target implementation](research/workbench/src/us_quant/residual_treasury_rotation.py)
uses the original audited 12-asset market without a new provider
request. The [independent pre-account proof](research/workbench/evidence/residual_treasury_20261011_pre_account_verification.json)
matches the original four independently seeded source paths and
checks all 122 unchanged monthly decisions using scalar sample
volatility and Decimal funding. Original source error is zero;
share/target errors are 1.23e-15/2.23e-16. There are 61 states
with no qualifying Treasury, 14 with one and 47 with both;
only 31 states have a nonzero original residual routed to Treasuries.
Every original equity/gold target remains exactly equal.

The [actual registration](research/workbench/evidence/residual_treasury_20261011_registration.json)
records both fixed candidates at **2026-10-11 14:25:46 / 14:25:49 UTC**.
The shared runner requires exact configured-remote publication
of the packet/code/policy/proof before account computation.
An actual evaluation attempt with the previous published commit
stops before output-directory creation or any new account.

The [registration preservation](research/workbench/evidence/residual_treasury_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_residual_registered_v56.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v30.json)
retain **132 completed configurations / 48 reviews / 192 paths**,
19 legacy / 29 funded-v2 reviews, 69 imported files and seven
forward profiles. One exact engine/head migration and two
candidate events are added; 52 registered versions include the
two old invalid ones and these two pending candidates.
249 coupled cases and 52 root guards pass.

At registration there are **zero new outcomes**, no champion and
no actual forward return session. The original W42 plans,
IJR/PKW quota, both frozen forward models and qualification goals
remain. The parent rule's five-year passes motivated this
exposed-history hypothesis, not a new holdout; its ten-year
failures and all prior failed trials remain. Neither new
Treasury routing nor extra ETF names establish independent alpha.
**The investment objective remains unmet.**

The [completed residual-Treasury results](research/workbench/evidence/residual_treasury_20261011_results.json)
retain both fixed routes and every window/cost path:

| Residual-Treasury rule/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Strongest qualifying Treasury, ten-year | 14.15% / 0.9581 / 15.67% | 13.29% / 0.8950 / 15.12% |
| Strongest qualifying Treasury, five-year | 18.57% / 1.0526 / 13.15% | 17.48% / 0.9825 / 13.29% |
| All qualifying Treasuries, ten-year | 14.25% / 0.9720 / 15.41% | 13.37% / 0.9068 / 14.90% |
| All qualifying Treasuries, five-year | 18.49% / 1.0520 / 13.15% | 17.44% / 0.9837 / 13.29% |

**Neither route improves the original BIL-residual source: every
one of the eight comparisons has lower net CAGR and Sharpe and
higher drawdown.** Both pass all five-year base gates, but lose
the original source's five-year stress Sharpe pass
(0.982475/0.983735, strictly below 1). Both ten-year
return/Sharpe gates and base drawdown gates fail. The original
source's five-year complete passes and ten-year failures are
retained, not rewritten or rounded into overall acceptance.

Ten-year stress orders/modeled fees rise to 789/USD2,176.25
for the strongest route and 823/USD2,225.91 for diversification,
versus the original BIL source's 776/USD1,954.93. Changing
duration exposure also changes subsequent NAV and therefore
dollar holdings of unchanged targets; these observations do
not isolate fee causality, equal-risk performance or
independent Treasury alpha. The Treasury subset, eligibility,
126/63 horizons and parent trend rules are not refitted.

The [independent account proof](research/workbench/evidence/residual_treasury_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, all proven selector targets/event dates/window seeds,
original fees/rates/calendars and initial-capital peaks.
It also checks 30,128 account sessions from actual prices:
normalized quantities, opening target funding, order thresholds,
cash flows, costs, turnover and closing wealth, without calling
the production simulator. Adjusted-price units are internal
accounting quantities, not broker share quantities.
Prepared versus actual seeded targets agree exactly; maximum
scalar metric/equity/cash-flow errors are
4.11e-14/7.28e-11 USD/1.10e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_residual_results_v57.json)
contains **134 completed configurations = 84 prior + 50 recurring reviews**,
52 registered code versions including two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v31.json)
reexecutes 50 reviews/200 paths, retaining 19 legacy and 31
explicit funded-v2 reviews. The [preservation proof](research/workbench/evidence/residual_treasury_20261011_preservation.json)
keeps the exact old journal prefix/rows, 69 imported files,
seven forward profiles and all 3,192 prior control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v38.json)
retains these failed routes, original daily/weekly cadence,
all four W42 plans, IJR/PKW quota and both frozen forward
models. Both models still have zero actual return sessions.
Prior selection diagnostics do not cover these two trials
or the complete search. No research model is replaced and
**the investment objective remains unmet**.

## Preregistered original-sector ETF breadth risk rules

The [fixed sector-breadth policy](research/workbench/config/sector-breadth-risk.json)
uses all nine originally audited sector ETF price histories as
**signal-only market information**, not additional portfolio holdings.
At original completed month ends, each fund votes positive only
when its actual adjusted close is strictly above its own trailing
200-observation mean; equality is off. A strict majority of the
fixed nine means at least five positive funds.

The two constructions leave the original four-style/SOXX/gold
risk-balanced source unchanged in strong breadth. Weak breadth
**halves or disables only the original equity component**;
gold stays exact and removed equity budget goes only to actual BIL.
The half-SOXX/equal-four-style composition, source 63-session risk
balance, 30%–70% equity bounds, 98% investment, monthly decisions
and original execution/gates remain. No new sector selection,
Treasury rotation, dollar/option confirmation, volatility cap,
wealth floor, daily trigger or parameter grid is added.

The original sector policy already specifies the fixed nine funds
and strict 200-observation trend; half and zero risk scales also
exist. Their structural majority aggregation is a **new
existing-factor risk construction**, not a new economic family
or replication of the earlier four-style relative-return breadth.
This fixed-fund measure is not PIT stock breadth, a historical
constituent archive or the complete evolving eleven-sector system.

Source inspection before preparation found two independently
frozen XLK adjusted-price versions: original-sector signal versus
the unused XLK column in the original twelve-asset portfolio.
Maximum adjusted open/close differences are 3.85e-5/3.81e-5;
raw-close difference is 4.85e-10 and volume agrees. Both original
raw/CSV guards and quote versions remain; no tolerance is widened,
price is rewritten or later portfolio quote substituted into the
initially specified sector signal. Portfolio XLK targets must
remain exactly zero, and all eight shared core observations retain
their original strict identity checks. The unregistered
same-version assumption/code is preserved before preparation.

The [independent pre-account proof](research/workbench/evidence/sector_breadth_20261011_pre_account_verification.json)
matches all four original source paths including independent
initialization, checks 25,245 signal prices against original CSVs,
and reconstructs 2,606 fully informed mean rows with Decimal
arithmetic. Each construction emits 125 monthly targets:
100 strong / 25 weak states. Six original source warmup months
remain uninformed; all formal return dates remain.
Original source/target errors are zero; signal CSV error is 1e-14.
Every original gold/other nonequity target stays exactly equal.

The [actual registration](research/workbench/evidence/sector_breadth_20261011_registration.json)
records the pair at **2026-10-11 15:27:17 / 15:27:20 UTC**.
The shared registered-study runner requires exact
configured-remote publication of the packet/code/policy/proof
before account computation. An actual attempt using the previous
published commit stops before any new output or account.

The [registration preservation](research/workbench/evidence/sector_breadth_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_breadth_registered_v58.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v32.json)
retain **134 completed configurations / 50 reviews / 200 paths**,
19 legacy / 31 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact engine/head migration adds the
generator, followed only by the two candidate events.
54 registered versions include two old invalid ones and these
two pending candidates; 266 coupled cases and 52 root guards pass.

At registration there are **zero new strategy outcomes**, no
champion and no actual forward return session. Both frozen
forward rules, all four W42 plans, IJR/PKW quota and six catalog
definitions/five families remain. This exposed-history breadth
hypothesis may still fail; prior selection diagnostics do not
correct these trials or all searches.
**The investment objective remains unmet.**

The [completed breadth results](research/workbench/evidence/sector_breadth_20261011_results.json)
retain both constructions and all eight paths:

| Weak-breadth equity rule/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Half equity, ten-year | 17.30% / 1.0712 / 19.20% | 17.07% / 1.0543 / 18.90% |
| Half equity, five-year | 22.22% / 1.1242 / 19.38% | 21.58% / 1.0888 / 19.08% |
| Equity off, ten-year | 16.24% / 1.0306 / 18.21% | 16.18% / 1.0248 / 17.65% |
| Equity off, five-year | 21.08% / 1.0818 / 18.28% | 20.64% / 1.0572 / 17.72% |

**Every Sharpe/SPY gate and every stress-versus-base-cost-SPY
gate passes, but every drawdown gate fails.** Both constructions
remain rejected under the unchanged joint objective. A minimum
Sharpe of 1.0543/1.0248 does not erase maximum drawdowns of
19.38%/18.28% or establish independent future alpha.
No sector subset, 200-observation mean, majority threshold,
risk scale, source allocation or gold weight is refitted.

All eight worst episodes peak on 2022-03-08; half equity
troughs on 2022-10-20, equity off on 2022-09-26.
The [post-outcome drawdown attribution](research/workbench/evidence/sector_breadth_20261011_drawdown_attribution.json)
reconciles 1,184 observed post-peak account days. Prior normalized
quantities mark the prior-close-to-open move; post-trade
quantities mark open-to-close, with modeled fees separate.
Each day's NAV change and the full original peak-to-trough
loss reconcile without invoking the production simulator.

Across these fixed episodes, **gold mark losses contribute
14.28–14.40 percentage points of each path's own peak NAV**,
larger than equity losses of 2.99–4.63 points.
Modeled fees contribute 0.176–0.599 points; BIL contributes
a small gain. These are additive held-position loss fractions,
not sums of CAGR or isolated/equal-risk/causal factor alpha.
Daily/episode reconciliation errors are 6.37e-12/8.19e-12 USD.
The hindsight diagnosis does not prove a future gold gate
will work and does not add strategy trials or expand the
frozen 36-source control cohort.

The [independent account proof](research/workbench/evidence/sector_breadth_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, all proven sector signals, original targets/dates,
window seeds, costs/rates/calendars and initial-capital peaks.
It also checks 30,128 account sessions of actual-price
quantities, opening funding, orders, fees and cash flow.
Prepared/actual seeded target error is zero; maximum scalar
metric/equity/cash-flow errors are
3.74e-14/9.46e-11 USD/1.29e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_breadth_results_v59.json)
contains **136 completed configurations = 84 prior + 52 recurring reviews**,
54 registered versions including two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v33.json)
reexecutes 52 reviews/208 paths, retaining 19 legacy and 33
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/sector_breadth_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files, seven
forward profiles and 3,192 frozen prior control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v39.json)
retains these return passes and drawdown failures, the source
version limits, original daily/weekly cadence, W42 plans,
IJR/PKW quota and both frozen forward models.
Both models still have zero actual return sessions.
Any new risk hypothesis needs its own fixed rule,
source proof and published registration before accounts;
these results do not modify either forward experiment.
**The investment objective remains unmet.**

## Preregistered independent monthly gold trend on breadth sources

The [fixed breadth/gold policy](research/workbench/config/breadth-gold-trend.json)
transfers the existing strict 200-observation GLD price/mean
law to both completed equity-breadth constructions.
**Every original equity target and source monthly decision
date stays unchanged.** Only gold is removed when its own
actual adjusted close is not strictly above the trailing
200-observation mean; equality is off. The original gold
weight is restored only when that fixed condition passes.
Removed gold goes only to actual BIL through the unchanged
prior component-removal helper, without equity renormalization.

The preceding gold-loss diagnosis motivates this new combination,
but is not a prediction or proof of efficacy. The former
component-trend study also used independent daily state changes;
this new study deliberately keeps the original source monthly
clock. It reuses the gold law, not that old whole-portfolio
target matrix or return path. No sector/mean/majority/scale
selection, daily trigger, USD/option confirmation, Treasury
rotation, volatility cap, net-wealth floor or share grid is added.

The [independent pre-account proof](research/workbench/evidence/breadth_gold_20261011_pre_account_verification.json)
matches all eight original source/window/cost target matrices
including independent seeds and reconstructs all 2,606 informed
gold mean rows with Decimal arithmetic. Each construction
keeps 125 monthly targets: 90 gold-on / 35 gold-off states.
Original-source error is zero; maximum Decimal target error
is 1.12e-16. The [target implementation](research/workbench/src/us_quant/breadth_gold_trend.py)
retains actual original fund/risk-free observations and both
separately frozen XLK versions; zero new quote, account or
economic-factor definition is introduced during preparation.

The [actual registration](research/workbench/evidence/breadth_gold_20261011_registration.json)
records the pair at **2026-10-11 16:05:13 / 16:05:15 UTC**.
An actual evaluation attempt using the preceding published
commit stops before output creation or any account calculation.
Exact configured-remote publication of the packet/code/policy/
proof is required before the shared runner can calculate returns.

The [registration preservation](research/workbench/evidence/breadth_gold_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_breadth_gold_registered_v60.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v34.json)
retain **136 completed configurations / 52 reviews / 208 paths**,
19 legacy / 33 funded-v2 outcomes, 69 imported files and seven
original forward profiles. One exact engine/head migration
adds the finite generator, followed only by two candidate events.
56 registered versions include two old invalid ones and
these two pending candidates; 245 coupled cases and 52 root
guards pass.

At registration there are **zero new strategy outcomes**,
no champion and zero actual forward return sessions.
Both exposed windows, the gold diagnosis, all original W42
plans, IJR/PKW quota, catalog families and frozen forward
rules remain. A historical all-gate result would still not
establish independent future alpha or trading authority.
**The investment objective remains unmet.**

The [completed breadth/gold results](research/workbench/evidence/breadth_gold_20261011_results.json)
retain both fixed gold rules and every window/cost path:

| Original equity-breadth source + monthly gold gate | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Half equity source, ten-year | 14.82% / 0.9762 / 13.89% | 14.21% / 0.9322 / 13.57% |
| Half equity source, five-year | 19.66% / 1.0619 / 13.15% | 18.61% / 0.9991 / 13.64% |
| Equity off source, ten-year | 13.78% / 0.9245 / 13.13% | 13.35% / 0.8930 / 13.40% |
| Equity off source, five-year | 18.52% / 1.0048 / 13.15% | 17.69% / 0.9549 / 13.42% |

**All eight drawdown gates pass, but all ten-year Sharpe/SPY
gates and both five-year stress Sharpe gates fail.**
The half-source five-year stressed Sharpe is exactly
**0.9991004369**, not strictly above 1; it is not rounded
into acceptance. Both five-year base paths pass all gates,
but neither construction passes the complete joint objective.

Every paired source comparison has lower drawdown, CAGR and
Sharpe after the fixed monthly gold gate. This improves modeled
risk without retaining the preceding sources' all-window return
qualification. Ten-year stress orders/modeled fees are
763/USD1,977.18 for the half source and 688/USD2,051.10 for
equity off, versus the ungated sources' 760/USD1,406.51 and
685/USD1,548.23. The differences also include changed gold
exposure and account NAV, not isolated fee causality or
equal-risk independent gold alpha.
Neither gold mean, clock, scale nor source sector breadth/
equity shares are refitted; neither forward model is replaced.

The [independent account proof](research/workbench/evidence/breadth_gold_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, the proven gold states, original equity-breadth
targets/dates, independent seeds, costs/rates/calendars
and initial-capital peaks. It also checks 30,128 account
sessions of actual-price quantities, opening funding,
order thresholds, fees and cash flows.
Prepared versus actual seeded targets agree exactly;
maximum scalar metric/equity/cash-flow errors are
3.54e-14/6.55e-11 USD/1.21e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_breadth_gold_results_v61.json)
contains **138 completed configurations = 84 prior + 54 recurring reviews**,
56 registered versions including two old invalid ones,
no pending actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v35.json)
reexecutes 54 reviews/216 paths, retaining 19 legacy and 35
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/breadth_gold_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files,
seven original forward profiles and all 3,192 frozen controls.

The [native continuation](research/workbench/evidence/prospective_data_automation_v40.json)
retains both the preceding breadth rules' Sharpe/SPY passes
with failed drawdown gates and the new gold rules' drawdown
passes with failed return gates. These successes cannot be
spliced into an untested or implicitly fitted strategy.
Original W42 plans, IJR/PKW quota, daily/weekly cadence
and both frozen forward experiments remain; both experiments
still have zero actual return sessions.
**The investment objective remains unmet.**

## Preregistered prior relative-gold components on breadth sources

The [fixed graded-gold policy](research/workbench/config/breadth-gold-relative.json)
transfers the already registered 63/126/252-session GLD-versus-BIL
gold component to both original equity-breadth sources.
Three strict same-horizon return signs give equal-weight
gold fractions of zero, one-third, two-thirds or one; equality
is off. **The unchanged old generator's gold target is copied
directly**, not a newly fitted partial-gold percentage.

The [explicit adapter](research/workbench/src/us_quant/breadth_gold_relative.py)
checks that original pre-gate gold budgets agree exactly.
Every original nongold/non-BIL target remains unchanged:
the old relative law's equity component is not used.
Only removed gold goes to actual BIL without equity or
other-risk renormalization. Original sector means/majority/
half-zero scales, monthly risk budgets, prices and fees remain.
Both complete frozen source/helper graphs are preserved.

The prior relative law needs 252 actual return observations.
Three earlier breadth source months therefore remain
explicitly uninformed, rather than inventing long-horizon signs.
All 2,805 source-price dates, every formal ten-/five-year
return session and each window's complete seed remain.
The [independent pre-account proof](research/workbench/evidence/graded_gold_20261011_pre_account_verification.json)
checks all eight original seeded source matrices, four prior
gold component/seed paths, 366 strict Decimal return comparisons
per candidate and complete funding. Original source/gold
errors are zero; maximum Decimal target error is 1.12e-16.
Each pair has 122 informed monthly targets: 10 zero / 29
one-third / 30 two-thirds / 53 full-gold states.

The [actual registration](research/workbench/evidence/graded_gold_20261011_registration.json)
records both fixed candidates at **2026-10-11 16:29:23 / 16:29:26 UTC**.
An actual evaluation attempt using the preceding published
commit stops before new output or account creation.
Exact configured-remote publication of packet/code/policy/proof
must precede the shared runner's account calculations.

The [registration preservation](research/workbench/evidence/graded_gold_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_graded_gold_registered_v62.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v36.json)
retain **138 completed configurations / 54 reviews / 216 paths**,
19 legacy / 35 funded-v2 outcomes, 69 imported files and seven
original forward profiles. One exact engine/head migration
adds the finite adapter, followed only by two candidate events.
58 versions include two old invalid ones and these two
pending candidates; 256 coupled cases and 52 root guards pass.

The gold-loss diagnosis and failed binary gate motivated this
new exposed-history question, not an independent holdout
or a splice of partial performance passes.
No horizon, vote weight, threshold, timing or share grid
is introduced. At registration there are **zero new outcomes**,
no champion and no actual forward return session.
Original W42 plans, IJR/PKW quota and both frozen forward
experiments remain. **The investment objective remains unmet.**

The [completed graded-gold results](research/workbench/evidence/graded_gold_20261011_results.json)
retain both fixed source combinations and every required path:

| Original equity-breadth source + copied relative gold | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Half equity source, ten-year | 15.27% / 1.0300 / 13.89% | 14.67% / 0.9853 / 13.29% |
| Half equity source, five-year | 19.40% / 1.0688 / 13.15% | 18.44% / 1.0099 / 13.29% |
| Equity off source, ten-year | 14.22% / 0.9792 / 13.68% | 13.80% / 0.9475 / 13.27% |
| Equity off source, five-year | 18.25% / 1.0106 / 13.78% | 17.52% / 0.9651 / 13.29% |

**The half-source construction passes every five-year gate,
but neither candidate passes the joint objective.**
All eight drawdown gates pass. Ten-year half-source CAGR
is exactly 15.2749%/14.6668%, below SPY's 15.4620%/15.4056%;
its ten-year stressed Sharpe is 0.985286, not above 1.
The narrow base CAGR difference is not rounded into acceptance.
The equity-off source also fails both ten-year return/Sharpe
comparisons and five-year stressed Sharpe.

Compared with the prior binary 200-observation gold gate,
the fixed graded law improves the ten-year CAGR/Sharpe,
but that conditional comparison is not equal-risk or causal
alpha and does not erase ten-year failures.
Ten-year stress orders/modeled fees are 814/USD1,965.87
for half equity and 739/USD2,045.45 for equity off.
No gold horizon, vote weight, threshold, clock, equity
composition or modeled cost is retuned.

The [independent account proof](research/workbench/evidence/graded_gold_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, original targets/signals/month dates, own-window
seeds, fees/rates/calendars and initial-capital peaks.
It also checks 30,128 actual-price quantity/funding/order/
cash-flow sessions. Prepared versus actual seeded targets
agree exactly; maximum scalar metric/equity/cash-flow errors
are 3.49e-14/8.74e-11 USD/1.10e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_graded_gold_results_v63.json)
contains **140 completed configurations = 84 prior + 56 recurring reviews**,
58 versions including two old invalid ones, no pending
actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v37.json)
reexecutes 56 reviews/224 paths, retaining 19 legacy and 37
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/graded_gold_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files, seven
forward profiles and 3,192 frozen prior control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v41.json)
retains these five-year passes and ten-year failures, original
daily/weekly cadence, W42 plans, IJR/PKW quota and both
frozen forward models. Both models still have zero actual
return sessions. Further execution research requires its
own fixed rule, source proof and published registration;
it cannot change modeled fees or reuse only passing windows.
**The investment objective remains unmet.**

## Preregistered original observed-weight execution band

The [fixed execution policy](research/workbench/config/observed-weight-execution.json)
transfers the existing **2.5-percentage-point actual closing
holding-weight band** to both unchanged graded-gold sources.
Initial fresh-account deployment always issues the full
last-known target. At later original monthly source updates,
trade only when some desired-versus-actual closing fund weight,
including BIL, differs by at least 2.5 points, with the old
helper's existing 1e-12 rounding allowance.
This is not last-target hysteresis or a fitted optimal band.

The [execution adapter](research/workbench/src/us_quant/observed_weight_execution.py)
reuses the original unmodified pure-band decision helper
with both QLD cap and volatility target disabled.
No leveraged QLD holding or volatility override is introduced;
the helper's required observed covariance argument cannot
affect this pure-band decision. Original source signals,
desired equity/gold weights, 122 monthly updates,
USD10,000 capital, USD1/order and 5/20bp costs remain.
Skipping a small correction holds actual positions rather
than rewriting the desired target or giving a cost rebate.

Each independent window/cost path reconstructs actual funded-v2
positions, fees and closing weights. Dated 1/2-session queues
are preserved without cancellation/overwrite; fills precede
that day's closing-weight decision. End-of-window unfilled
requests are not retrospectively filled. Drift can worsen
drawdown or return, and 2% remains target idle cash, not a floor.
The old band study used a different leveraged portfolio:
only its decision helper is reused, not its old accounts
or promotion contract.

The [source-only pre-account proof](research/workbench/evidence/observed_band_20261011_pre_account_verification.json)
matches all eight original desired target/seed matrices exactly.
All 2,805 source dates, actual funds and frozen risk-free rates
remain. **Actual path-dependent issued targets and new accounts
are not calculated before publication.**
Synthetic tests cover observed-versus-last-target feedback,
funding, delays, initial deployment and unfilled final requests.
A synthetic future-price doubling correctly hit the unchanged
60% daily-return source guard; only that fixture was smoothed,
with the failed test note retained and no source rule changed.
255 coupled cases and 52 root guards pass.

The [actual registration](research/workbench/evidence/observed_band_20261011_registration.json)
records the pair at **2026-10-11 16:56:16 / 16:56:19 UTC**.
An actual attempt using the preceding published commit stops
before path-dependent targets, output creation or accounts.
The shared runner requires exact configured-remote publication
of packet/code/policy/proof before any such calculations.

The [registration preservation](research/workbench/evidence/observed_band_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_observed_band_registered_v64.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v38.json)
retain **140 completed configurations / 56 reviews / 224 paths**,
19 legacy / 37 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact engine/head migration adds the
dynamic adapter, followed only by two candidate events.
60 versions include two old invalid ones and these two
pending candidates.

At registration there are **zero new outcomes**, no champion
and no actual forward return sessions. Original W42 plans,
IJR/PKW quota, catalog families and both frozen forward rules
remain. No band, commission, latency, source vote or share grid
is introduced. **The investment objective remains unmet.**

The [completed observed-band results](research/workbench/evidence/observed_band_20261011_results.json)
retain both fixed execution constructions and every path:

| Original graded-gold source + actual 2.5pp band | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Half equity source, ten-year | 15.42% / 1.0410 / 13.77% | 14.84% / 0.9981 / 13.27% |
| Half equity source, five-year | 19.52% / 1.0786 / 12.96% | 18.51% / 1.0161 / 13.10% |
| Equity off source, ten-year | 14.31% / 0.9870 / 13.64% | 13.92% / 0.9568 / 13.08% |
| Equity off source, five-year | 18.28% / 1.0148 / 13.72% | 17.54% / 0.9684 / 13.10% |

**Every path has fewer asset orders, lower modeled costs and
drawdown within 15%, but neither full joint candidate qualifies.**
The half-source construction retains all five-year passes.
Its exact ten-year CAGR is 15.416509676%/14.842779716%,
still below SPY's 15.461997933%/15.405636614%;
stressed Sharpe is **0.9981430033**, not strictly above 1.
Neither close margin is rounded into acceptance.
The equity-off source also fails ten-year return/Sharpe
and five-year stressed Sharpe.

Ten-year half-source base orders/modeled costs fall from
814/USD1,109.48 to 622/USD910.81; stressed values fall
from 814/USD1,965.87 to 609/USD1,731.86.
The off-source stressed values fall from 739/USD2,045.45
to 580/USD1,866.04. Skipped corrections also change subsequent
holdings and NAV, so these are conditional execution comparisons,
not isolated fee or equal-risk causal alpha.
The original 2.5pp band, source parameters, costs and delays
are not changed after these near-threshold failures.

The [independent account and feedback proof](research/workbench/evidence/observed_band_20261011_reproducibility.json)
reconstructs all **728 monthly/initial decisions** from actual
closing weights, Decimal target gaps and the original source
matrix, independently of the production decision helper.
All requests, unchanged issued target values, own-window seeds,
delayed fills and funding reconcile. The half-source ten-year
base/stress paths skip 29/31 in-band updates; 8/6 requests are
triggered by actual drift even though last-target differences
remain inside the band. This verifies actual-holding feedback,
not mislabeled last-target hysteresis.

All 16 funded strategy/SPY and 16 independent `bt` accounts,
30,128 actual-price quantity/funding/order/cash-flow sessions,
original calendars/rates/fees and strict gates are checked.
Target/request error is zero; maximum scalar metric/equity/
cash-flow errors are 4.05e-14/1.17e-10 USD/1.28e-11 USD.
Internal adjusted-price quantities are not broker shares.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_observed_band_results_v65.json)
contains **142 completed configurations = 84 prior + 58 recurring reviews**,
60 versions including two old invalid ones, no pending
actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v39.json)
reexecutes 58 reviews/232 paths, retaining 19 legacy and 39
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/observed_band_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files,
seven forward profiles and all 3,192 frozen controls.

The [native continuation](research/workbench/evidence/prospective_data_automation_v42.json)
retains improved modeled execution, five-year passes and
ten-year failures without changing any qualification gate.
Daily/weekly cadence, W42 plans, IJR/PKW quota and both
frozen forward experiments remain; neither experiment has
an actual return session. These exposed-window gains do not
establish independent future performance or trading authority.
**The investment objective remains unmet.**

## Preregistered relative equity-leadership satellite rotation

The [fixed leadership policy](research/workbench/config/sector-leadership-rotation.json)
tests whether the original semiconductor satellite should
retain its equity budget only while it leads broad-market SPY.
One method uses the existing 126-session strict return-winner
principle; the other uses the existing 63/126/252-session
equal-weight strict relative-return signs. Equality goes to
SPY, not an invented positive-leadership state.

The [target adapter](research/workbench/src/us_quant/sector_leadership_rotation.py)
starts from the original graded-gold half-equity-breadth
source and changes **only SOXX/SPY**. The winner method moves
the full original satellite budget to SPY when SOXX's return
does not exceed SPY. Horizon voting splits that same budget
between SOXX and SPY by zero/one-third/two-thirds/one.
Every original style, gold, BIL and other fund target stays
exact; combined satellite/equity budgets remain.
Equity/gold risk balance is not recomputed after switching,
so unchanged budgets do not imply equal risk.

The existing unmodified 2.5pp actual-holding execution adapter
runs on each new desired matrix, with fresh positions for
every cost/window, original capital/fees/delays and forced
initial deployment. Its original candidate handle selects
the common pure-band rule; it does not claim these new
targets or positions replay the old whole strategy.
No new cash gate, fund, provider request, family, lookback,
vote weighting, sector/share or band grid is introduced.

The [independent pre-account proof](research/workbench/evidence/leadership_20261011_pre_account_verification.json)
matches all four original source/seed matrices, reconstructs
strict relative signs with Decimal actual prices, and checks
every unchanged style/gold/bill target and satellite budget.
Source error is zero; maximum desired-target error is 2.78e-17.
Both methods have 122 monthly decisions.
The winner chooses SPY in 30 months and SOXX in 92;
voting has 15 zero / 16 one-third / 27 two-thirds / 64
full-SOXX states. Actual path-dependent issued targets and
new accounts are not computed during preparation.

The [actual registration](research/workbench/evidence/leadership_20261011_registration.json)
records both fixed candidates at **2026-10-11 17:23:07 / 17:23:10 UTC**.
An actual evaluation attempt using the preceding published
commit stops before output/feedback-target/account creation.
Exact configured-remote publication of packet/code/policy/
proof must precede account calculation.

The [registration preservation](research/workbench/evidence/leadership_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_leadership_registered_v66.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v40.json)
retain **142 completed configurations / 58 reviews / 232 paths**,
19 legacy / 39 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact engine/head migration adds the
finite adapter, followed only by two actual candidate events.
62 versions include two old invalid ones and these two pending
candidates; 245 coupled cases and 52 root guards pass.

The source and both windows were already selected/exposed
after prior results; this is not an independent holdout,
old defensive-portfolio replication or proven sector alpha.
At registration there are **zero new outcomes**, no champion
and no actual forward return sessions. Original W42 plans,
IJR/PKW quota, frozen forward rules and goals remain.
**The investment objective remains unmet.**

The [completed leadership results](research/workbench/evidence/leadership_20261011_results.json)
retain both fixed rotation rules and all eight account paths:

| Relative sector leadership rule/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| 126-session winner, ten-year | 15.32% / 1.0562 / 13.77% | 14.79% / 1.0143 / 13.27% |
| 126-session winner, five-year | 19.27% / 1.0906 / 12.96% | 18.40% / 1.0336 / 13.10% |
| Three-horizon votes, ten-year | 14.48% / 1.0005 / 13.78% | 13.87% / 0.9529 / 13.27% |
| Three-horizon votes, five-year | 18.05% / 1.0225 / 12.97% | 17.16% / 0.9641 / 13.10% |

**The 126-session winner passes all four Sharpe/drawdown
gates and every five-year gate, but still fails ten-year SPY
outperformance.** Its exact ten-year CAGR is 15.3244%/14.7876%,
below SPY's 15.4620%/15.4056%. Minimum Sharpe of 1.0143
and maximum drawdown of 13.77% therefore do not qualify
the joint objective. The vote construction also fails
ten-year SPY and both stressed Sharpe gates.

Replacing a lagging industry allocation with broad equity
improves some conditional risk-adjusted metrics, while
the winner's CAGR remains lower than the preceding fixed
SOXX/observed-band reference. These comparisons are not
equal-risk or causal satellite alpha. Ten-year stress
orders/modeled fees are 649/USD1,920.16 for the winner
and 724/USD1,952.52 for voting. The satellite budget,
leadership horizons/vote weights, band, fees and clocks
are not changed after these outcomes.

The [independent account proof](research/workbench/evidence/leadership_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, all independently proven desired rotations,
728 actual-holding band decisions/initial seeds/dated
requests/fills, and 30,128 actual-price funding/order/
cash-flow sessions. All original rates, costs, calendars
and initial-capital peaks remain. Target/request error
is zero; maximum scalar metric/equity/cash-flow errors
are 4.36e-14/8.74e-11 USD/1.10e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_leadership_results_v67.json)
contains **144 completed configurations = 84 prior + 60 recurring reviews**,
62 versions including two old invalid ones, no pending
actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v41.json)
reexecutes 60 reviews/240 paths, retaining 19 legacy and 41
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/leadership_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files,
seven forward profiles and 3,192 frozen control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v43.json)
retains the winner's full-window Sharpe/drawdown passes
alongside the ten-year SPY failures, without reinterpreting
the benchmark or weakening any gate. Daily/weekly cadence,
W42 plans, IJR/PKW quota and both frozen forward experiments
remain. Both experiments still have zero actual return sessions.
**The investment objective remains unmet.**

## Preregistered bounded risk allocation inside the factor core

The [fixed factor-core policy](research/workbench/config/factor-core-risk-rotation.json)
tests two existing risk estimators inside the original four-style
budget of the 126-session leadership source: **63-session sample
volatility** and **63-session excess-return semideviation**.
The [implementation](research/workbench/src/us_quant/factor_core_risk_rotation.py)
reuses the unchanged bounded inverse-risk share helper and
its original 15%-35% per-factor limits. All four momentum,
value, quality and low-volatility funds remain; none is deleted.

The downside case reuses the existing scalar semideviation
helper on each style's actual daily return minus frozen lagged
risk-free rates, dividing squared negative parts by all 63
sessions. The older downside study balanced aggregate
growth/gold risk; only its estimator is adapted here.
Sample volatility retains its original raw-return definition,
so the comparison is not isolated causal tail-risk attribution.
Missing/nonfinite/zero risks fail rather than create default
equal shares or cash allocations.

Only four within-core target shares change. The exact core
budget, every SOXX/SPY/GLD/BIL/other target, original monthly
dates and 98% total funding remain. Bounds apply inside
the four-style core, not to whole equity/NAV.
The parent unmodified 2.5pp actual-holding execution path
is run on each new desired matrix with fresh positions and
original capital/fees/delays. No equity/gold risk balance,
sector/gold signals, band or costs are refitted.

The [independent source-only proof](research/workbench/evidence/core_risk_20261011_pre_account_verification.json)
matches the entire previously proven desired source matrix
and all four previous issued-target/initial-seed paths.
It reconstructs 976 scalar risk estimates, bounded shares
with 120-step Decimal bisection rather than the production
root solver, and all 122 target months per candidate.
Source/risk/share/target errors are
0/3.39e-17/4.08e-13/6.73e-14.
Actual feedback targets and new accounts are not calculated.

The [actual registration](research/workbench/evidence/core_risk_20261011_registration.json)
records both fixed hypotheses at **2026-10-11 17:47:37 / 17:47:40 UTC**.
An actual evaluation attempt using the preceding published
commit stops before output/feedback/account creation.
Exact configured-remote publication of packet/code/policy/
proof must precede the shared runner's account calculation.

The [registration preservation](research/workbench/evidence/core_risk_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_core_risk_registered_v68.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v42.json)
retain **144 completed configurations / 60 reviews / 240 paths**,
19 legacy / 41 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact engine/head migration adds the
finite adapter, followed only by two candidate events.
64 versions include two old invalid ones and these two
pending candidates; 232 coupled cases and 52 root guards pass.

At registration there are **zero new outcomes**, no champion
and no actual forward return sessions. Source selection/windows
were already exposed, prior selection diagnostics do not cover
these trials or all searches, and equal budgets are not equal risk.
No factor/bounds/window/share grid, new quote or economic
definition is introduced. Original W42 plans, IJR/PKW quota,
frozen forward experiments and goals remain.
**The investment objective remains unmet.**

The [completed factor-core results](research/workbench/evidence/core_risk_20261011_results.json)
retain both fixed risk estimators and every account path:

| Bounded style-core risk rule/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Sample volatility, ten-year | 15.20% / 1.0589 / 13.75% | 14.66% / 1.0160 / 13.23% |
| Sample volatility, five-year | 19.02% / 1.0910 / 12.81% | 18.16% / 1.0334 / 12.92% |
| Semideviation, ten-year | 15.26% / 1.0623 / 13.75% | 14.71% / 1.0190 / 13.25% |
| Semideviation, five-year | 19.09% / 1.0938 / 12.79% | 18.22% / 1.0359 / 12.88% |

**Both constructions pass every Sharpe/drawdown gate and
every five-year gate, but both fail ten-year SPY outperformance.**
All eight CAGR values are below their equal-core leadership
source. Minimum Sharpe improves modestly to 1.0160/1.0190
and maximum drawdown remains about 13.75%; risk allocation
has not closed the observed absolute-return gap.
SPY's ten-year base/stress CAGR remains 15.4620%/15.4056%.
Neither candidate qualifies the unchanged joint objective.

All four style exposures remain bounded within their
original core budget, and every nongroup desired target
stays exact. Ten-year stressed asset orders remain 649
for each rule; modeled fees are USD1,923.08/USD1,928.30.
Conditional risk-adjusted improvements are not isolated
or equal-risk factor alpha. No factor is removed and
no bounds, risk window, equity/gold share, band or fee
is retuned after these outcomes.

The [independent account proof](research/workbench/evidence/core_risk_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, original independently proven bounded targets,
728 actual-holding band/initial-seed/request/fill decisions,
and 30,128 actual-price funding/order/cash-flow sessions.
Original rates, costs, calendars and initial-capital
drawdown peaks remain. Target/request error is zero;
maximum scalar metric/equity/cash-flow errors are
4.27e-14/8.01e-11 USD/1.10e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_core_risk_results_v69.json)
contains **146 completed configurations = 84 prior + 62 recurring reviews**,
64 versions including two old invalid ones, no pending
actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v43.json)
reexecutes 62 reviews/248 paths, retaining 19 legacy and 43
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/core_risk_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files,
seven forward profiles and 3,192 frozen control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v44.json)
retains both full-window Sharpe/drawdown and five-year passes
alongside ten-year SPY failures. Original daily/weekly cadence,
W42 plans, IJR/PKW quota and frozen forward experiments remain;
both experiments still have zero actual return sessions.
Observed risk-allocation benefits do not establish independent
future performance or permit changing the return goal.
**The investment objective remains unmet.**

## Preregistered original return-aware factor-core ranking

The [fixed core-momentum policy](research/workbench/config/factor-core-momentum.json)
transfers two previously registered factor scores inside
the original positive four-style budget of the 126-leadership
source: **total momentum** and **SPY-regression residual momentum**.
The [implementation](research/workbench/src/us_quant/factor_core_momentum.py)
uses the unchanged original score helper and exact old
126-return/21-skip/252-fit settings. Rank all four funds,
with lexical ticker ties, into the original
35%/30%/20%/15% core shares.

Negative/zero scores do not delete a fund. Total momentum
compounds 105 observed returns from the fixed 126-minus-21
window. Residual momentum fits intercept/SPY excess-return
beta on 252 observed daily returns, then uses the same
105 residual observations divided by sample deviation.
The old helper's declared constant-residual zero rule
remains; rank-deficient/missing/nonfinite inputs fail.
Residual scoring does not hedge market beta in long-only
actual ETF holdings, or establish an independent family.

Only four within-core rank shares change. The exact core
budget, every SOXX/SPY/GLD/BIL/other target, 122 month dates,
original industry/gold rules and 98% funding remain.
The unmodified parent 2.5pp actual-holding execution law
runs on each new desired matrix with fresh positions
and original capital/fees/delays. No predictor, window,
rank share, factor subset or cost grid is introduced.

The [independent source-only proof](research/workbench/evidence/core_momentum_20261011_pre_account_verification.json)
matches the whole old desired source matrix and all four
previous issued-target/initial-seed paths.
It reconstructs 976 scalar scores using 105-return
compounding and centered Decimal OLS, not the production
least-squares routine. Every rank, tie-break and target
agrees; source/score/target errors are
0/9.22e-13/1.39e-17.
Actual feedback targets and accounts are not computed.

The [actual registration](research/workbench/evidence/core_momentum_20261011_registration.json)
records both fixed hypotheses at **2026-10-11 18:14:39 / 18:14:42 UTC**.
An actual evaluation attempt using the preceding published
commit stops before output/feedback/account creation.
Exact configured-remote publication of packet/code/policy/
proof must precede the shared runner's account calculations.

The [registration preservation](research/workbench/evidence/core_momentum_20261011_registration_preservation.json),
[current snapshot](research/workbench/evidence/research_program_2026-W41_core_momentum_registered_v70.json)
and [full prior source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v44.json)
retain **146 completed configurations / 62 reviews / 248 paths**,
19 legacy / 43 funded-v2 outcomes, 69 imported files and seven
forward profiles. One exact engine/head migration adds the
finite adapter, followed only by two actual candidate events.
66 versions include two old invalid ones and these two
pending candidates; 230 coupled cases and 52 root guards pass.

Only old pure score functions are reused, not the old whole
factor/gold portfolio or a legacy data loader with a changed
full fingerprint. The source/windows were already exposed,
prior selection diagnostics do not cover these trials/allsearch,
and no independent holdout/future alpha is established.
At registration there are **zero new outcomes**, no champion
and zero actual forward return sessions. Original W42 plans,
IJR/PKW quota, frozen forward experiments and goals remain.
**The investment objective remains unmet.**

The [completed core-momentum results](research/workbench/evidence/core_momentum_20261011_results.json)
retain both fixed score rules and every required path:

| Ranked four-factor core score/window | Base net CAGR / Sharpe / drawdown | Stress net CAGR / Sharpe / drawdown |
|---|---|---|
| Total momentum, ten-year | 15.35% / 1.0487 / 13.85% | 14.73% / 1.0024 / 13.38% |
| Total momentum, five-year | 19.40% / 1.0883 / 13.14% | 18.49% / 1.0287 / 13.29% |
| Residual momentum, ten-year | 15.38% / 1.0569 / 13.85% | 14.73% / 1.0080 / 13.37% |
| Residual momentum, five-year | 19.29% / 1.0877 / 12.91% | 18.35% / 1.0260 / 13.03% |

**Both constructions pass all four Sharpe/drawdown gates
and every five-year gate, but both fail ten-year SPY
outperformance.** Ten-year base/stress SPY CAGR remains
15.4620%/15.4056%. The higher residual base CAGR is
15.3805%, not above that benchmark; neither close base
margin nor stressed return gap is rounded into acceptance.
Minimum Sharpe of 1.0024/1.0080 does not erase the
remaining outperformance failures.

All four style exposures retain their fixed rank shares
inside the unchanged core budget, with every nongroup
target and source rule exact. Ten-year stressed orders/
modeled fees are 689/USD2,024.97 for total momentum
and 695/USD2,075.63 for residual ranking.
The comparison is not equal-risk or causal factor alpha,
and score beta-removal does not hedge beta in actual
long-only holdings. No score window, rank share, factor
subset, industry budget, band or fee is retuned.

The [independent account proof](research/workbench/evidence/core_momentum_20261011_reproducibility.json)
checks all 16 funded strategy/SPY and 16 independent `bt`
accounts, original independently proven score/rank targets,
728 actual-holding band/initial-seed/request/fill decisions,
and 30,128 actual-price funding/order/cash-flow sessions.
Original rates, costs, calendars and initial-capital peaks
remain. Target/request error is zero; maximum scalar
metric/equity/cash-flow errors are
4.20e-14/6.55e-11 USD/1.19e-11 USD.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_core_momentum_results_v71.json)
contains **148 completed configurations = 84 prior + 64 recurring reviews**,
66 versions including two old invalid ones, no pending
actual candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v45.json)
reexecutes 64 reviews/256 paths, retaining 19 legacy and 45
explicit funded-v2 reviews.
The [preservation proof](research/workbench/evidence/core_momentum_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imported files,
seven forward profiles and 3,192 frozen control artifacts.

The [native continuation](research/workbench/evidence/prospective_data_automation_v45.json)
retains the full-window Sharpe/drawdown and five-year passes
alongside ten-year SPY failures. Original daily/weekly cadence,
W42 plans, IJR/PKW quota and frozen forward experiments remain;
both experiments still have zero actual return sessions.
Observed return-score gains do not establish independent
future performance or permit changing the return goal.
**The investment objective remains unmet.**

## Existing energy and materials source preparation

The [source-only policy](research/workbench/config/producer-sector-sources.json)
and [implementation](research/workbench/src/us_quant/producer_sector_sources.py)
append actual XLE energy-company and XLB materials-sector fund prices to
the original twelve-asset portfolio. Each added fund has all 2,805 required
price sessions; no new quote request, portfolio target or account was produced.
The [independent source proof](research/workbench/evidence/producer_sector_20261011_source_readiness.json)
checks 137,445 original panel/rate observations without changing a value,
22,440 added raw/CSV price-volume observations, and each fund's scoped
issuer identity and advertised primary PDF. Original acquisition times,
corporate-action guards and both separate XLK price versions remain intact.

**This source-only checkpoint is data preparation, not a new strategy or factor family.**
Producer-company equities are not direct commodity futures or guaranteed
inflation hedges. At this checkpoint, a possible gold-trend-released-budget allocation required
separately frozen source/budget/selector rules, an explicit fourteen-asset
ResearchProgram adapter, actual registration and remote publication before
portfolio performance may be calculated. The source helper implements none
of that strategy admission and does not change stock-standby BIL targets.
The separate producer study below subsequently implements that admission.

The [preservation proof](research/workbench/evidence/producer_sector_20261011_preservation.json)
retains the exact old ledger/engine/head, seven forward profiles,
69 imported files and 3,192 frozen controls. Counts remain **148 completed
configurations, 64 recurring reviews, no qualified champion** at this source-only checkpoint.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v46.json)
keep daily/weekly timing, the original four W42 plans and both frozen
forward models; source coverage does not establish strategy qualification
or independent native message delivery.

## Preregistered producer equities for gold-released cash

The [separate policy](research/workbench/config/producer-gold-routing.json)
and [finite generator](research/workbench/src/us_quant/producer_gold_routing.py)
freeze two routes from the original half-breadth graded-gold
observed-band reference, not the later leadership or core-score variants.
At its 122 informed month ends, only the difference between the prior
ungraded gold budget and the graded GLD target may leave BIL.
Stock-standby BIL, GLD, the four-style core, SOXX and every other original
target remain unchanged.

XLE/XLB qualify only when their 126-session adjusted return strictly exceeds
actual BIL. One rule chooses the strongest with lexical ticker ties;
the other uses all eligible funds' 63-session inverse sample-volatility
shares. Total producer exposure is capped at the prior 40% of NAV
and the available gold-released budget. No eligible fund leaves source BIL
unchanged; capped residuals are not redistributed.
The original 2.5pp actual-holding band, delayed request queue, fresh capital
and transaction-cost assumptions remain fixed. **Added producer exposure
is equity risk, not a guaranteed commodity or inflation hedge.**

The [independent pre-account proof](research/workbench/evidence/producer_routing_20261011_pre_account_verification.json)
checks 244 Decimal selectors and four window seeds: original target error
is zero, maximum gold-budget/target/share errors are
7e-17/3.69e-16/1.13e-15. Source months have 22 no-eligible,
50 single-eligible and 50 dual-eligible states; each rule routes
57 months, with 19 cap-limited months. Lossless old observations are
separately checked; neither account parser was changed.

The [actual registration packet](research/workbench/evidence/producer_routing_20261011_registration.json)
records both candidates at 19:15 UTC.
The [registration preservation proof](research/workbench/evidence/producer_routing_20261011_registration_preservation.json)
confirms one exact old-engine/head migration, restored verification of
all seven unchanged semantic forward statuses, 69 imports and 3,192 controls.
An initial private verifier checked semantic statuses too early; its
original code/log remain, and only the check order was corrected.
The engine identity guard and prepared rules/data/specifications were not relaxed.

The [registered snapshot](research/workbench/evidence/research_program_2026-W41_producer_registered_v72.json)
retained 148 completed configurations, 64 reviews, 68 versions including
two old invalid ones, two actual pending candidates and no champion.
The [prior-path audit](research/workbench/evidence/research_program_source_bound_audit_v46.json)
reexecutes all 64 reviews/256 paths. An actual old-commit evaluation
stops before feedback targets or account creation.
**No new portfolio return had been calculated at registration:** the exact packet,
code, policy and proof must be approved and remotely published first.
Original W42 plans and both frozen forward models remain unchanged.

## Verified producer routes: return alone does not meet the risk goal

The [complete results](research/workbench/evidence/producer_routing_20261011_results.json)
follow actual remote publication of the original registration/code/proof
in `698a2490`. The recorded fence at 19:26:13.325601 UTC precedes
path-dependent targets/accounts from 19:26:13.326742 through
19:26:52.953434 UTC. Original capital, costs, dates, gold-only cash origin,
four-style/SOXX/GLD targets and the actual-holding band remain fixed.

| Producer route | Window | Base CAGR / excess Sharpe / drawdown | Stress CAGR / excess Sharpe / drawdown |
| --- | --- | --- | --- |
| Highest eligible return | Ten years | 14.7059% / 0.8820 / 20.8073% | 14.7804% / 0.8840 / 20.5168% |
| Highest eligible return | Five years | 18.9722% / 0.9476 / 20.8996% | 18.6082% / 0.9256 / 20.2987% |
| All eligible inverse volatility | Ten years | 15.9706% / 0.9673 / 20.7920% | 15.9952% / 0.9643 / 20.1869% |
| All eligible inverse volatility | Five years | 18.9721% / 0.9471 / 20.8987% | 18.5363% / 0.9195 / 20.2984% |

The diversified route beats the unchanged same-cost SPY and stress/base-SPY
return gates in every path. **All eight Sharpe and drawdown gates fail,
so both candidates are rejected.** Most worst episodes run from
March 8 through July 14, 2022; these descriptive dates do not establish a
causal inflation hedge or predict future protection.
Ten-year stress has 685 orders/USD2,109.70 for winner and
720/USD2,266.59 for diversified. Cost/delay scenarios produce different
actual feedback/fill paths, not a pure fee-only control: a higher stressed
CAGR here is not evidence that fees improve returns.
No cap, subset, source, lookback, band or cost is refitted to these outcomes.

The [independent reproduction](research/workbench/evidence/producer_routing_20261011_reproducibility.json)
checks 16 funded and 16 independent-bt accounts,
728 actual holding-band/seed/request/fill decisions and 30,128 funding days.
Maximum target/metric/equity/cash-flow errors are
0/5.22e-15/1.10e-10 USD/1.60e-11 USD.
The [current snapshot](research/workbench/evidence/research_program_2026-W41_producer_results_v73.json)
contains **150 completed configurations = 84 prior + 66 recurring reviews**,
68 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v47.json)
reexecutes all 66 reviews/264 paths, retaining 19 legacy and 47 explicit
funded-v2 reviews.
The [preservation proof](research/workbench/evidence/producer_routing_20261011_preservation.json)
retains every old journal row/prefix, 69 imports, seven forward profiles
and all 3,192 frozen controls.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v47.json)
retain the partial SPY success and full Sharpe/drawdown failures.
Original daily/weekly cadence, W42 plans and both frozen forward rules remain;
both forward models still have zero actual return sessions.
**The investment objective remains unmet.**

## Real-activity information: complete archives, freshness still blocked

The [source-only policy](research/workbench/config/activity-inputs.json)
and [monthly vintage helper](research/workbench/src/us_quant/activity_inputs.py)
prepare Chicago Fed CFNAIMA3 activity magnitude and CFNAIDIFF diffusion.
They are related views of one composite, not two independent factors.
The publisher's recession/expansion associations are historical and
conditional; neither is treated as an unconditional recession forecast
or a proven trading rule.

The [independent source proof](research/workbench/evidence/activity_input_readiness_20261011.json)
verifies 450 actual historical ALFRED snapshots in 12 bounded batches,
plus six separately acquired probes. All 131 original monthly/reference
pairs, 225 query dates per series and 2,805 NYSE context days remain.
Selection uses the actual second-previous-NYSE-date snapshot, not today's
revised history or an assumed publication on the first-of-month period label.
Every selected date, completed month, age and native index value is checked
independently; Decimal value error is zero.

**Only 440 of 450 selections meet the fixed source freshness contract.**
Each series has the following five blocked queries; full readiness remains false.

| Query session | Earlier snapshot | Latest observed month | Completed-period age |
| --- | --- | --- | --- |
| 2023-01-27 | 2023-01-25 | 2022-10 | 88 days |
| 2025-11-25 | 2025-11-21 | 2025-08 | 86 days |
| 2025-11-28 | 2025-11-25 | 2025-08 | 89 days |
| 2025-12-29 | 2025-12-24 | 2025-09 | 90 days |
| 2025-12-31 | 2025-12-29 | 2025-09 | 92 days |

The 62-day limit was fixed before acquisition and measures query date
minus the completed observation-month end. No limit relaxation,
current-history fill, query deletion or implicit standby strategy is used.
These archived snapshot gaps do not prove that the original publisher
failed to release data. Complete underlying component/weight vintages and
original intraday first-release times remain outside the verified scope.

The initial ordinary probe returned HTTP200 HTML with an explicit
observation-range error, not a ZIP. Its original policy, code,
request/response and log are retained privately. A new acquisition honors
the actual advertised `2026-08-01` monthly observation end; market as-of
`2026-10-05`, formal return windows, all historical query dates, snapshot
buffer, 62-day age limit, endpoint and transport are unchanged.
Primary series definitions and the actually advertised 14-page background
PDF are separately verified.

The [preservation proof](research/workbench/evidence/activity_input_preservation_20261011.json)
retains the exact research engine, ledger/event head, seven forward profiles,
69 imports and 3,192 frozen controls. **This source-only checkpoint added
no direction flags, portfolio targets, accounts, factor definitions or
strategy trials.** Its counts were 150 completed configurations,
66 reviews, 68 registered versions and no qualified champion;
the separate strategy study below does not relabel the blocked source as ready.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v48.json)
keep daily/weekly cadence, original W42 plans and both frozen forward models.
Source preparation is not investment qualification; the full objective
remains unmet.

## Preregistered low-and-deteriorating activity confirmations

The [separate policy](research/workbench/config/activity-factor-guard.json)
and [finite generator](research/workbench/src/us_quant/activity_factor_guard.py)
freeze two confirmations on the original four-style/SOXX/gold
sector risk-budget source. At each original monthly decision,
both require a positive own-vintage 63-session broad-dollar change
and an activity decline against its separately archived 63-session-earlier
reading. Magnitude additionally requires CFNAIMA3 strictly below -0.70;
diffusion requires CFNAIDIFF strictly below -0.35. Equality does not trigger.
A confirmed state halves all original non-BIL targets and moves the removed
budget to actual BIL; otherwise the original target is unchanged.
The published levels motivate a testable low/deteriorating-activity rule,
not an unconditional recession label or claimed independent alpha.

**A distinct, explicitly frozen missing-data rule preserves the strict source
failure.** The ten exact stale inputs map to no rebalance on
2023-04-28, 2025-11-28, 2025-12-31, 2026-02-27 and 2026-03-31.
Those entire target rows stay undefined, with recorded source reasons.
Existing holdings, mark-to-market, cash, fees and all 2,512/1,254
formal return days remain. Unknown gaps are fatal; no age relaxation,
current-history fill, cash fallback or synthetic zero return is introduced.
The original source policy/readiness remains blocked.

The [independent pre-account proof](research/workbench/evidence/activity_guard_20261011_pre_account_verification.json)
checks all 450 activity queries, 131 raw-published dollar comparisons,
scalar original budgets, both full target matrices and eight seeded
registered replays. Maximum input/budget/target/seed errors are
1.78e-15/2.22e-16/2.22e-16/0.
Each rule keeps 126 targets on the original 131-date grid,
with two magnitude and one diffusion half-risk months.
Thresholds, 63-session comparisons, 50% scale, source, costs and monthly
clock are fixed; no outcome-driven grid is used.

The [actual registration](research/workbench/evidence/activity_guard_20261011_registration.json)
records both candidates at 20:25 UTC.
The [preservation proof](research/workbench/evidence/activity_guard_20261011_registration_preservation.json)
verifies one exact old-engine/event-head migration and all seven original
semantic forward profiles, 69 imports and 3,192 controls.
The [registered snapshot](research/workbench/evidence/research_program_2026-W41_activity_registered_v74.json)
retained 150 completed configurations and 66 reviews, with 70 registered
versions including two old invalid ones, two actual pending candidates
and no champion.
The [prior-path audit](research/workbench/evidence/research_program_source_bound_audit_v48.json)
reexecutes all 66 reviews/264 paths. An actual old-commit attempt
stops before strategy account/output creation.
**No new account return had been calculated at registration.**
The exact original registration/code/policy/proof must be remotely published
before evaluation. Original W42 plans and both frozen forward models remain.

## Verified activity confirmations: Sharpe and returns pass, drawdown fails

The [complete outcomes](research/workbench/evidence/activity_guard_20261011_results.json)
follow actual remote publication of the original registration/code/proof
in `1a21a515`. The recorded fence at 20:33:50.018063 UTC precedes accounts
from 20:33:50.019121 through 20:34:12.150721 UTC.
Both rules retain the original monthly budgets, costs, capital and
explicit five-month no-rebalance law.

| Activity confirmation | Window | Base CAGR / excess Sharpe / drawdown | Stress CAGR / excess Sharpe / drawdown |
| --- | --- | --- | --- |
| Magnitude | Ten years | 17.6775% / 1.0388 / 20.5001% | 17.2800% / 1.0143 / 20.4942% |
| Diffusion | Ten years | 17.9416% / 1.0529 / 20.4960% | 17.5611% / 1.0293 / 20.4898% |
| Either rule | Five years | 22.9406% / 1.1203 / 20.6489% | 22.1626% / 1.0798 / 20.6435% |

**Every Sharpe/SPY return gate and every stressed-strategy/base-SPY gate
passes, but all eight drawdown gates fail. Both candidates are rejected.**
Worst episodes run from March 8 through October 14, 2022; these
descriptive dates do not establish causal risk protection.
Ten-year stress has 699 orders/USD1,126.81 for magnitude and
698/USD1,134.98 for diffusion.
No threshold, 63-session window, scale, source age or cost is refitted.

The [independent reproduction](research/workbench/evidence/activity_guard_20261011_reproducibility.json)
checks 16 funded and 16 independent-bt accounts, 30,128 funding days,
all seeded targets and 40 specific pause/no-rebalance/retained-holding cases.
Maximum target/metric/equity/cash-flow errors are
0/3.58e-14/1.10e-10 USD/1.24e-11 USD.
Both complete five-year seeded target segments and funded/bt account bytes
are identical, despite different full pre-window target contexts.
The five-year results are therefore not independent evidence for two alpha sources.
All actual return days remain; no stale source is filled or relabelled as repaired.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_activity_results_v75.json)
contains **152 completed configurations = 84 prior + 68 recurring reviews**,
70 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v49.json)
reexecutes all 68 reviews/272 paths, retaining 19 legacy and 49 explicit funded-v2
reviews. The [preservation proof](research/workbench/evidence/activity_guard_20261011_preservation.json)
keeps all old journal rows/prefix, 69 imports, seven forward profiles,
3,192 frozen controls and the original W42 queue.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v49.json)
retain Sharpe/SPY successes alongside the drawdown failures and unchanged
daily/weekly cadence. Both frozen forward models still have zero actual
return sessions. **The full investment objective remains unmet.**

## Preregistered existing monthly gold law on activity sources

The [actual held-loss attribution](research/workbench/evidence/activity_guard_20261011_drawdown_attribution.json)
reconciles all eight reviewed activity worst episodes and 1,224 post-peak
account days. Gold mark losses account for 13.8403–13.9779 percentage points
of peak NAV, versus equity losses of 6.2912–6.3785 points.
Maximum daily/episode mark-and-fee errors are 9.66e-12/3.64e-12 USD.
These overlapping, post-outcome loss decompositions are descriptive,
not causal alpha, a future hedge guarantee or new strategy trials.

The [separate frozen policy](research/workbench/config/activity-gold-trend.json)
and [generator](research/workbench/src/us_quant/activity_gold_trend.py)
reuse the original strict 200-observation GLD price/mean state and unchanged
component-removal helper on both original activity target matrices.
Only source GLD is set to zero and moved to actual BIL when its close is
not strictly above its trailing mean. All original equity/other targets,
the five macro no-request months, monthly clock, capital and costs remain.
There is no daily trigger, fitted mean, gold scale or stock-share grid.

The source has 126 original active updates. Each transferred rule has
120 issued targets, 85 gold-on and 35 gold-off decisions, after six
explicit pre-window gold warmup months (November 2015 through April 2016).
No new target is invented before 200 observations; no formal
2,512/1,254 return day is removed. Independent fresh-capital window seeds
and the original macro pauses remain.
The [pre-account proof](research/workbench/evidence/activity_gold_20261011_pre_account_verification.json)
checks 2,606 Decimal gold states, eight original seeded source paths
and eight new complete seeded target matrices: source/target/seed errors are zero.

The [actual registrations](research/workbench/evidence/activity_gold_20261011_registration.json)
record both candidates at 20:52 UTC.
The [registration preservation proof](research/workbench/evidence/activity_gold_20261011_registration_preservation.json)
verifies one exact old-engine/head migration, all seven original semantic
forward profiles, 69 imports and 3,192 frozen controls.
The [registered snapshot](research/workbench/evidence/research_program_2026-W41_activity_gold_registered_v76.json)
retained 152 completed configurations and 68 reviews, with 72 versions
including two old invalid ones, two pending candidates and no champion.
The [prior-path audit](research/workbench/evidence/research_program_source_bound_audit_v50.json)
reexecutes all 68 reviews/272 paths. An actual old-commit attempt stops
before new output/account creation.
**No new portfolio performance had been calculated at registration.**
The exact original packet, code, policy and proof must be remotely published
before evaluation. Original W42 plans and frozen forward rules remain.

## Verified gold transfer: lower drawdown is still not qualification

The [complete outcomes](research/workbench/evidence/activity_gold_20261011_results.json)
follow real remote publication of the original registration/code/proof
in `f194e5c5`. The recorded fence at 20:59:38.910863 UTC precedes accounts
from 20:59:38.911950 through 21:00:01.395626 UTC.
All original activity equity targets, macro pauses, capital, costs and
formal return calendars remain unchanged.

| Activity source plus gold gate | Window | Base CAGR / excess Sharpe / drawdown | Stress CAGR / excess Sharpe / drawdown |
| --- | --- | --- | --- |
| Magnitude | Ten years | 15.1886% / 0.9482 / 17.9171% | 14.4091% / 0.8966 / 17.7397% |
| Diffusion | Ten years | 15.4481% / 0.9635 / 17.9171% | 14.6845% / 0.9130 / 17.7397% |
| Either source | Five years | 20.3724% / 1.0662 / 15.3336% | 19.1627% / 0.9975 / 15.9742% |

Every drawdown is lower than the original activity-source result,
but **all eight drawdown gates still fail**. All ten-year Sharpe/SPY
gates fail, and five-year stress Sharpe is 0.9975121778, not above one.
Diffusion's ten-year base CAGR 15.4481% is still below SPY 15.4620%;
neither this gap nor 15.3336% drawdown is rounded into a pass.
All eight CAGR/Sharpe comparisons are lower than their original sources.
Both candidates are rejected; original Sharpe/return passes cannot be
combined with these different drawdowns.

The worst ten-year episodes now fall in February–March 2020,
while five-year worst episodes run from December 27, 2021 through
October 14, 2022. Descriptive episode shifts do not establish a robust hedge.
Ten-year stress has 718 orders/USD1,774.45 for magnitude and
717/USD1,795.34 for diffusion. The 200-observation rule, monthly clock,
gold scale, equity targets and costs are not refitted.

The [independent reproduction](research/workbench/evidence/activity_gold_20261011_reproducibility.json)
checks 16 funded and 16 independent-bt accounts, 30,128 funding days,
all full target/seed matrices and 40 macro-pause/no-rebalance/held-risk cases.
Maximum target/metric/equity/cash-flow errors are
0/3.72e-14/6.55e-11 USD/1.14e-11 USD.
Both entire five-year seeded target segments and account bytes are identical,
not independent evidence for two alpha sources.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_activity_gold_results_v77.json)
contains **154 completed configurations = 84 prior + 70 recurring reviews**,
72 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v51.json)
reexecutes all 70 reviews/280 paths, retaining 19 legacy and 51 explicit
funded-v2 reviews.
The [preservation proof](research/workbench/evidence/activity_gold_20261011_preservation.json)
retains every old journal row/prefix, 69 imports, seven forward profiles,
3,192 frozen controls and the original W42 plans.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v50.json)
retain lower drawdown alongside return failures and unchanged daily/weekly
cadence. Both frozen forward models still have zero actual return sessions.
**The full investment objective remains unmet.**

## Treasury curve vintages: verified source, not a new profitable factor

The [source-only policy](research/workbench/config/yield-curve-inputs.json)
and [vintage parser](research/workbench/src/us_quant/yield_curve_inputs.py)
prepare daily ten-year-minus-three-month (`T10Y3M`) and
ten-year-minus-two-year (`T10Y2Y`) Treasury constant-maturity spreads.
T10Y3M already appears in the registered macro tilt, conditional-model
features and original prospective inputs; it is not counted as a discovery.
Both spreads share their ten-year leg and are not two independent alpha sources.
Their native percent quotes express percentage-point yield differences,
including negative and zero values, not traded asset returns.

The [independent source proof](research/workbench/evidence/yield_curve_input_readiness_20261011.json)
verifies all 450 actual ALFRED snapshots in 12 bounded batches,
plus six separately acquired probe snapshots. All original 131 monthly/reference
pairs, 225 queries per series and 2,805 NYSE context days remain.
Each selected observation comes from its previous-NYSE-date vintage
with the unchanged two-strictly-later-NYSE-session lag and seven-day age limit.
**All 450 selected inputs are fresh; maximum actual age is five days,
and independent Decimal value error is zero.**

Every selected T10Y3M date and native level, and every same-observation value,
matches the original unchanged CSV on the 225-query cohort.
This supports that specific source-version comparison, not a claim that
all daily conditional-model inputs, all historical revisions or the old
portfolio's alpha have been validated.
Primary definitions disclose the June 21, 2019 underlying Treasury source
transition. Original intraday first-publication times and full methodology
continuity are not authenticated. The New York Fed page identifies the
yield-curve research topic; its model coefficients, monthly aggregation
and recession probabilities are not imported or replicated.

An initial local parser rejected an actual HTTP200 ZIP because its long title
wraps across the ALFRED metadata table beside real-time date columns.
The original code, synthetic fixture, logs and requests remain private.
The corrected parser reads scoped field values and continuation lines;
an independent regex-based date-column parser verifies the same complete
title/source/release/units/frequency/adjustment identities.
Three successful original document/probe requests are reused without refetch
or retiming. No source policy, quote, lag, age, window or transport is changed.

The [preservation proof](research/workbench/evidence/yield_curve_input_preservation_20261011.json)
retains the exact engine, ledger/event head, seven forward profiles,
69 imports and 3,192 frozen controls. **No inversion flags, portfolio targets,
accounts, factor definitions or strategy trials are added.**
Counts remain 154 completed configurations, 70 recurring reviews,
72 registered versions and no qualifying champion.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v51.json)
keep daily/weekly cadence, the original W42 queue and frozen forward models.
Any strategy using these versions needs separately frozen rules,
actual registration and remote publication before performance calculation.
Source readiness does not complete the investment objective.

## Treasury transition confirmations: return gates pass, drawdown fails

The [fixed policy](research/workbench/config/curve-transition-guard.json)
and [generator](research/workbench/src/us_quant/curve_transition_guard.py)
test two maturity views of one macro transition hypothesis.
At each original completed-month decision, halve all original non-BIL
targets only when published broad-dollar change over 63 NYSE sessions
is strictly positive, the selected Treasury spread is strictly negative,
and that spread has strictly increased from its own historical
63-session-earlier reading. Equality does not trigger.
Removed capital goes to actual BIL; otherwise the original
four-style/SOXX/gold allocation is issued unchanged.

This is not the old macro tilt's nonpositive-spread equity-only gate,
nor a new economic factor family, a Federal Reserve probability model
or a verified equity forecast. Both maturity views share the ten-year leg.
No thresholds, horizons, cash reserve, gold budget, factor ratios,
fees or trading clock are tuned after looking at candidate performance.
There is no activity gap policy, inversion-duration memory,
new holding band or current-history substitution.

The [pre-account proof](research/workbench/evidence/curve_transition_20261011_pre_account_verification.json)
independently reparses every one of the 450 Treasury vintage selections,
checks 131 actual published dollar pairs and scalar source budgets,
and reproduces all eight window/cost-seeded target paths.
Both retain all 131 original monthly requests and all formal return days;
the three-month and two-year views respectively reduce risk in 10 and eight
months. Source error is at most 3.02e-16, scalar budget/target error
7.22e-16 and seeded replay error zero. These are target-rule checks,
not portfolio performance.

The [actual registration packet](research/workbench/evidence/curve_transition_20261011_registration.json)
and [preservation proof](research/workbench/evidence/curve_transition_20261011_registration_preservation.json)
record two actual registrations after one exactly anchored engine migration.
All prior journal rows, seven forward profiles, 69 imports and 3,192
controls remain unchanged; an old-commit evaluation is blocked before
creating account output. The
[registered snapshot](research/workbench/evidence/research_program_2026-W41_curve_transition_registered_v78.json)
retains 154 completed configurations/70 reviews, with 74 versions including
two old invalid ones and exactly two pending candidates.
**No new candidate returns or metrics have been computed at registration.**
Actual remote publication
`c02cd6505c11a75fefffe46898aecd2851e5ce35` precedes the durable
21:36:44.098299UTC fence and both new account calculations.
The [independently verified results](research/workbench/evidence/curve_transition_20261011_results.json)
retain all four paths per view:

| View | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| 10y-minus-3m | 10y | Base | 18.0522% | 1.08043 | 20.4930% |
| 10y-minus-3m | 10y | Stress | 17.5918% | 1.05215 | 20.4869% |
| 10y-minus-3m | 5y | Base | 22.7229% | 1.14770 | 20.6489% |
| 10y-minus-3m | 5y | Stress | 21.7954% | 1.09871 | 20.6435% |
| 10y-minus-2y | 10y | Base | 17.7435% | 1.05566 | 20.4930% |
| 10y-minus-2y | 10y | Stress | 17.1622% | 1.01842 | 20.4869% |
| 10y-minus-2y | 5y | Base | 22.0779% | 1.10369 | 20.6489% |
| 10y-minus-2y | 5y | Stress | 20.9010% | 1.03904 | 20.6435% |

**All eight Sharpe and same-cost SPY gates pass**, as do all four stress
comparisons with base-cost SPY. **All eight 15% drawdown gates fail,
so both configurations remain rejected.**
Their worst drawdown spans March 8 to October 14, 2022.
The maturity confirmation does not authenticate an independent forecast,
and passing return gates cannot be combined with another candidate's
lower drawdown to create a fictitious qualified portfolio.

The [reproducibility proof](research/workbench/evidence/curve_transition_20261011_reproducibility.json)
checks all 16 funded strategy/SPY accounts and 16 independent bt accounts,
including 30,128 actual-price quantity/funding/cash-flow days.
Target, metric and account wealth errors are at most 1.11e-16,
3.42e-14 and USD1.10e-10. Both views have distinct seeded target and strategy
CSV bytes in all four matched window/cost comparisons; distinct paths
are still not independent alpha sources.
The [preservation proof](research/workbench/evidence/curve_transition_20261011_preservation.json)
retains every old journal row, seven profiles, 69 imports and 3,192 controls.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_curve_transition_results_v79.json)
contains **156 completed configurations = 84 prior + 72 recurring reviews**,
74 versions including two old invalid ones, no pending candidate and no champion.
The [complete source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v52.json)
reexecutes all 72 reviews/288 paths, with 19 legacy and 53 funded-v2 reviews,
without rewriting prior outcomes.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v52.json)
retain daily/weekly cadence, original W42 plans and both frozen forward
models, which still have zero actual return sessions.
No threshold, horizon, scale, source or execution rule is refitted.
**The full investment objective remains unmet.**

## Nine-day option confirmation: complete source, rejected strategies

The [source-only policy](research/workbench/config/near-term-volatility-inputs.json)
and [independent source proof](research/workbench/evidence/near_term_volatility_input_readiness_20261011.json)
add the actual Cboe VIX9D closing series.
The first ordinary request uses the exact `cdn-api.cboe.com` URL advertised
by the already acquired official history page, not a host fallback.
Original VIX/VIX3M CSVs, acquisition timestamps and primary documents
are reused unchanged. All 8,415 closing values across the original
2,805 NYSE context days are independently checked with zero numeric error;
the complete raw histories contain 17,547 closing rows.

The actual VIX9D CSV starts January 4, 2011.
The primary methodology distinguishes a January 2011 first-value month
from an October 2013 launch month. Its target is nine calendar days,
using SPX/end-of-week SPXW contracts, nearest-term selection, midpoint
quotes and the stated regular-session dissemination.
These annualized expected-volatility levels are not traded returns.
Current documents/downloads do not authenticate all historical revisions,
original intraday first-publication times or a complete methodology chain.
VVIX is not admitted.
The [source preservation proof](research/workbench/evidence/near_term_volatility_input_preservation_20261011.json)
records no strategy state, target, account, definition or engine migration
at that source-only checkpoint.

The separately [frozen strategy policy](research/workbench/config/near-term-risk-confirmation.json)
compares monthly and daily confirmation of the already recorded monthly
positive dollar regime using actual VIX9D greater than or equal to actual VIX.
Only that conjunction halves all original non-BIL four-style/SOXX/gold
targets into BIL. Original allocation construction, threshold one,
half scale, capital and fees remain. The daily method emits on monthly
updates or binary-state changes; no event means no request, not a new
daily allocation. Dated queued requests cannot be retrospectively replaced.

The [generator](research/workbench/src/us_quant/near_term_risk_confirmation.py)
reuses the immutable joint-confirmation event primitive.
Its legacy internal near/far labels explicitly map to VIX9D/VIX;
raw and prepared matrices retain genuine VIX9D/VIX/VIX3M names.
The real three-month series is an unchanged source comparator, not the
new far leg. This adapter neither rewrites old code nor relabels old results.
Independent checks compare raw named closes directly, without the adapter.

The [pre-account proof](research/workbench/evidence/near_term_confirmation_20261011_pre_account_verification.json)
verifies 8,415 named option inputs, 131 published H10 pairs,
131 scalar original budgets and eight complete window/cost-seeded paths.
Monthly confirmation has 131 targets/20 risk-off decisions;
daily confirmation has 345 targets, 409 observed risk-off sessions
and 214 non-monthly events. Source error is 3.02e-16,
budget/target error 7.22e-16 and seeded replay error zero.
These counts are signal/target diagnostics, not performance.

The [actual packet](research/workbench/evidence/near_term_confirmation_20261011_registration.json),
[registration preservation](research/workbench/evidence/near_term_confirmation_20261011_registration_preservation.json)
and [registered snapshot](research/workbench/evidence/research_program_2026-W41_near_term_registered_v80.json)
retain 156 completed configurations/72 reviews, 76 versions including
two old invalid ones and two pending candidates.
One exact engine migration preserves all old records, seven profiles,
69 imports, 3,192 controls and W42 plans.
An old-commit evaluation is blocked before account creation.
**No new candidate returns or metrics exist at registration;
remote publication must precede the funded/bt evaluation.**

Actual remote publication
`168baf59dc4b0f3f77bf32d54d74a3605a095fc4` precedes the durable
22:15:33.631239UTC fence and both account calculations.
The [independently verified results](research/workbench/evidence/near_term_confirmation_20261011_results.json)
retain all four paths for each timing:

| Timing | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Monthly | 10y | Base | 15.7875% | 0.99460 | 21.3078% |
| Monthly | 10y | Stress | 15.1371% | 0.95073 | 22.3830% |
| Monthly | 5y | Base | 20.5485% | 1.03539 | 21.5370% |
| Monthly | 5y | Stress | 19.2524% | 0.96476 | 22.6130% |
| Daily option | 10y | Base | 14.7096% | 0.93572 | 19.2682% |
| Daily option | 10y | Stress | 10.8437% | 0.67267 | 22.8786% |
| Daily option | 5y | Base | 19.5407% | 1.00900 | 19.8821% |
| Daily option | 5y | Stress | 15.8012% | 0.80002 | 23.2820% |

**All eight 15% drawdown gates fail and both configurations are rejected.**
All ten-year Sharpe gates fail; 0.9945966 cannot be rounded into a pass.
Both five-year base Sharpe gates pass, but neither stress Sharpe does.
Monthly confirmation beats ten-year base-cost SPY but misses the stress
comparison; daily confirmation misses both ten-year SPY comparisons.
The five-year return passes do not qualify either full four-path portfolio.

The [reproducibility proof](research/workbench/evidence/near_term_confirmation_20261011_reproducibility.json)
checks genuine named option closes, original H10 pairs, every dated risk
request and seed, 16 funded strategy/SPY accounts and 16 independent bt
accounts over 30,128 funding/cash-flow days.
Target, metric and account wealth errors are at most 1.11e-16,
3.38e-14 and USD9.46e-11.
The [descriptive comparison](research/workbench/evidence/near_term_confirmation_20261011_horizon_comparison.json)
reads eight already-frozen same-timing VIX/VIX3M accounts without
calculating new accounts or trials. New nine-day CAGR and Sharpe are lower
in all eight comparisons. Daily ten-year stress orders increase from
1,209 to 2,135 and actual fees from USD3,749.59 to USD7,690.85.
This is not an equal-risk or causal cost decomposition, nor a reason to
refit thresholds, horizons or fees after exposure.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_near_term_results_v81.json)
contains **158 completed configurations = 84 prior + 74 recurring reviews**,
76 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v53.json)
reexecutes all 74 reviews/296 paths, with 19 legacy and 55 funded-v2 reviews.
The [preservation proof](research/workbench/evidence/near_term_confirmation_20261011_preservation.json)
retains every prior row, seven forward profiles, 69 imports and 3,192 controls.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v53.json)
keep daily/weekly cadence, original W42 plans and frozen forward models.
Both forward models still have zero actual return sessions.
**The full investment objective remains unmet.**

## Gold/equity option risk budgets: stable guard still misses ten-year SPY

The [gold-option source policy](research/workbench/config/gold-volatility-inputs.json)
and [independent coverage proof](research/workbench/evidence/gold_volatility_input_readiness_20261011.json)
preserve one ordinary advertised GVZ CSV request and six new primary
document requests, with no replacement or refetch of the old VIX comparator.
GVZ has the actual native `DATE,GVZ` schema, not assumed OHLC fields.
All 13,580 raw GVZ/VIX quotes and 5,606 finite aligned daily inputs
match independent parsing exactly.

**Full daily GVZ readiness is false.** The advertised history is missing
December 3, 2018; February 20 and July 5, 2019; and October 16, 2020.
Those four cells remain missing in the full 2,805-NYSE-day context.
No filling, copied VIX quote, removed return day or silent standby repairs
the source. All 131 original completed-month queries nevertheless have
actual GVZ/VIX observations: all 262 monthly inputs are independently verified.
Monthly query availability is a narrower contract, not daily readiness.
The [source preservation proof](research/workbench/evidence/gold_volatility_input_preservation_20261011.json)
records no source-stage signal, target, account, event or migration.

Current primary methodology identifies GVZ as 30-calendar-day expected
GLD-option volatility, using monthly PM-settled third-Friday contracts,
nearest-term selection, NBBO and midpoint quotes.
Its revision is March 30, 2026, version 9.0.
The current table lists first values before 2008 and launch in 2008,
while the actual advertised CSV starts September 18, 2009.
The April 2011 circular concerns the planned option-contract launch,
not the spot index launch. The 2022 C1-to-NBBO consultation is a proposal,
not proof of an implementation date; the published 2024 filtering results
notice identifies July 8, 2024 as its effective date.
These documents do not authenticate the full historical methodology chain
or original intraday first publication. Spot index levels are not traded returns.

The separately [frozen strategy policy](research/workbench/config/implied-risk-balance.json)
uses those actual original month-end inputs.
Equity receives GVZ divided by GVZ plus VIX, clamped to the unchanged
30%-70% bounds; the remaining invested capital stays in GLD.
Preserve the original equal-four-style half of equity and SOXX's other half,
98% funded total, original monthly clock, capital and costs.
Compare the plain budget with the original monthly positive-dollar
half-risk guard. Neither config adds daily rules, covariance/beta fitting,
option trades, extra filters or new factor families.
VIX is SPX-option risk, not the concentrated stock sleeve's certified risk;
this is a proxy-budget hypothesis, not exact portfolio risk parity.

The [pre-account proof](research/workbench/evidence/implied_risk_20261011_pre_account_verification.json)
checks all 262 native monthly quotes, 131 actual H10 pairs,
131 independent original budgets and eight complete seeded paths.
Both retain 131 targets, zero lower-bound clamps and one upper-bound clamp;
the guarded view has 74 half-risk months.
Source, new-target and seeded errors are at most 3.02e-16, 1.11e-16
and zero. The four daily source gaps are not filled or used as target dates.

The [actual packet](research/workbench/evidence/implied_risk_20261011_registration.json),
[registration preservation](research/workbench/evidence/implied_risk_20261011_registration_preservation.json)
and [registered snapshot](research/workbench/evidence/research_program_2026-W41_implied_risk_registered_v82.json)
retain 158 completed configurations/74 reviews, 78 versions including two
old invalid ones and two pending candidates.
One exactly anchored engine migration preserves prior rows, seven forward
profiles, 69 imports, 3,192 controls and original W42 plans.
Old-commit evaluation is blocked before creating account output.
**No new candidate returns or metrics exist at registration;
remote publication must precede funded/bt evaluation.**

Actual remote publication
`20637ef0de241ee31b422b43dc3a35c776796854` precedes the durable
22:47:52.126776UTC fence and both new account calculations.
The [independently verified results](research/workbench/evidence/implied_risk_20261011_results.json)
retain all four paths for each budget:

| Budget | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Plain | 10y | Base | 18.4837% | 1.06079 | 21.5374% |
| Plain | 10y | Stress | 18.3435% | 1.05185 | 21.2044% |
| Plain | 5y | Base | 22.4983% | 1.06425 | 21.7038% |
| Plain | 5y | Stress | 22.1005% | 1.04400 | 21.4384% |
| Dollar guarded | 10y | Base | 14.5873% | 1.08430 | 13.5726% |
| Dollar guarded | 10y | Stress | 14.2969% | 1.05915 | 13.4142% |
| Dollar guarded | 5y | Base | 19.0952% | 1.16801 | 13.5916% |
| Dollar guarded | 5y | Stress | 18.5282% | 1.12745 | 13.4334% |

**The plain budget passes all Sharpe/SPY gates but fails all 15% drawdown
gates. The guarded budget passes every Sharpe/drawdown gate and both
five-year joint paths, but fails both ten-year SPY return gates.**
Ten-year guarded CAGR remains below same-cost SPY's 15.4620% base
and 15.4056% stress, and stress also misses base-cost SPY.
Both configurations remain rejected under the unchanged full objective.
Neither the passing shorter window nor a combination of plain returns
and guarded drawdown is an independently qualified portfolio.

The [reproducibility proof](research/workbench/evidence/implied_risk_20261011_reproducibility.json)
checks 262 actual monthly option inputs, 131 H10 pairs, every scalar
ratio/clamp/composition and full seed, 16 funded strategy/SPY accounts
and 16 independent bt accounts over 30,128 funding/cash-flow days.
Target, metric and wealth errors are at most 1.11e-16, 3.51e-14
and USD8.01e-11.
Sixteen actual source-gap-day account checks preserve daily returns and
risk holdings without creating new signals or filling GVZ gaps.
Valid fills of prior month-end signals on those days are not forbidden.

The [descriptive source comparison](research/workbench/evidence/implied_risk_20261011_source_comparison.json)
reads eight original frozen realized-budget accounts without calculating
new controls or trials.
The guarded option budget slightly raises CAGR on all four comparisons,
but raises Sharpe on only three and also slightly raises drawdown on all four.
Different budgets and implied-risk proxies are not equal-risk or causal alpha
evidence. No cap, proxy ratio, guard scale, clock or cost is refitted.

The [current snapshot](research/workbench/evidence/research_program_2026-W41_implied_risk_results_v83.json)
contains **160 completed configurations = 84 prior + 76 recurring reviews**,
78 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v54.json)
reexecutes all 76 reviews/304 paths, with 19 legacy and 57 funded-v2 reviews.
The [preservation proof](research/workbench/evidence/implied_risk_20261011_preservation.json)
retains every old row, seven profiles, 69 imports, 3,192 controls and W42 plans.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v54.json)
retain daily/weekly cadence and both unchanged forward models,
which still have zero actual return sessions.
**The full investment objective remains unmet.**

## Current selection audit: no positive simultaneous growth bound

The [new dated policy](research/workbench/config/selection-accounting-cohorts.json)
and [adapter](research/workbench/src/us_quant/selection_accounting_cohorts.py)
include all 76 actual recurring reviews without selecting by performance.
They split the records by original accounting identity:
19 legacy-original records and 57 explicit funded-v2 records.
Legacy results are described under their original engine; they are not
repaired, pooled into current funded performance or treated as fresh trials.
Two terminally invalid code versions have no accepted four-path accounts.
The earlier 84 configurations and unknown research choices remain outside
this inference.

Reuse the unchanged positive-part maximum mean daily log-growth advantage
against actual SPY accounts, with all three original base/base,
stress/stress and stress/base comparisons.
Each cohort/horizon cell has 57 or 171 columns.
There are four fixed cells across the overlapping ten-year/five-year windows.
The inherited 4,000 synchronous 21-session circular-block resamples and
seed 20261011/20261012 remain; both cohorts use the same sampled dates
within each horizon. No candidate, horizon or accounting-cell independence
is assumed. Allocate the 5% familywise error as 1.25% per cell,
and use the capped four-times-minimum omnibus p-value across cells.
This is conditional inference on existing net return paths, not a simulated
trading account, Sharpe/drawdown qualification or a future profit probability.

The [original registration](research/workbench/evidence/selection_accounting_20261011_registration.json)
freezes all original bundle/review digests and cohort IDs before joint
resampling. The
[corrected publication-adapter registration](research/workbench/evidence/selection_accounting_20261011_registration_v2.json)
retains the same policy, source bindings, cohort, statistic and settings.
The
[preservation proof](research/workbench/evidence/selection_accounting_20261011_registration_preservation.json)
retains 26 old selection code/policy/private/evidence artifacts,
every current journal row, engine/event head, seven forward profiles,
69 imports and 3,192 controls. An old-commit run is blocked before
matrix construction or resampling.
A private self-check initially expected different error wording;
its code/logs remain, and only that verifier expectation was corrected.
The publication guard, statistical method, sources and registered settings
were not changed.

**No new joint statistical outcomes have been calculated at registration.**
Remote publication of this packet, code, policy and source bindings
must precede the four-cell computation.
The first published-commit attempt stops at the unchanged publication guard:
three calendar/metric/storage helpers were already public through the
frozen 69-file import manifest but lacked this guard's literal allowlist entries.
The [boundary receipt](research/workbench/evidence/selection_accounting_20261011_publication_boundary.json)
preserves that failed attempt and original module hashes.
Attempting duplicate literal entries is separately rejected by the strict
root publication guard before any commit or push.
The [adapter recovery proof](research/workbench/evidence/selection_accounting_20261011_publication_recovery_v2.json)
instead reuses the existing manifest-aware imported-core verifier;
neither publication guard is changed or bypassed.
All eight cohort/source/statistical function ASTs remain identical.
Original code, packets and failed attempts are preserved; only the
publication adapter is corrected and registered as a new version before
any statistical outcome. A newly approved actual remote commit is required
before resampling; duplicate import approvals are not added.
Counts remain 160 completed configurations/76 recurring reviews,
78 versions, no pending strategy candidate and no qualified champion.
The old 22/36-candidate results retain their original scope and bytes;
this new study does not silently extend their p-values.
No new accounts, strategy/factor definitions, engine migration or forward
days are created, and the investment objective remains unmet.

The [actual corrected-adapter results](research/workbench/evidence/selection_accounting_20261011_results.json)
follow remote publication
`fe51ac294ee2fd82078a37ee44c4f541be0f277c`.
The actual 23:31:41.717163UTC fence precedes primary input/inference
processing at 23:31:41.718161UTC.
All four fixed cells are reported, including the non-significant outcomes:

| Original accounting cohort | Window | Comparisons | Scoped omnibus p-value | Positive simultaneous log-growth lower bounds |
|---|---|---:|---:|---:|
| Legacy original, not current funded performance | 10y | 57 | 1.000000 | 0 |
| Legacy original, not current funded performance | 5y | 57 | 0.579355 | 0 |
| Explicit cash-funded-v2 | 10y | 171 | 0.501125 | 0 |
| Explicit cash-funded-v2 | 5y | 171 | 0.225694 | 0 |

The four-cell Bonferroni scoped omnibus p-value is **0.9027743064**.
**No comparison has a strictly positive simultaneous log-growth lower
bound under this frozen diagnostic.**
This does not prove that successful strategies cannot exist or quantify
future profits. The inference still excludes the earlier 84 configurations,
unknown research choices and future-added trials; it is not full-search
selection-adjusted alpha or an independent holdout.
It does not change Sharpe, SPY or drawdown qualification.

The [reproducibility proof](research/workbench/evidence/selection_accounting_20261011_reproducibility.json)
reproduces all statistics and four private matrix hashes in a complete
production replay. Independent raw parsing checks 304 reviewed paths,
608 strategy/SPY account references and their 608 bt counterparts,
including 1,144,864 net account rows. Those checks are overlapping source
references, not independent trials.
Raw log-input, original metric and bt wealth errors are at most
2.09e-16, 1.45e-13 and USD1.82e-10.
Independent direct-index arithmetic verifies every one of the 4,000
samples per cell, with maximum bound error 7.70e-18 and exact p-values.

The [preservation proof](research/workbench/evidence/selection_accounting_20261011_preservation.json)
keeps all original rows, engine/head, 26 old selection artifacts,
seven profiles, 69 imports and 3,192 controls byte-identical.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v55.json)
retain these scope limits, daily/weekly cadence, original W42 plans
and both unchanged forward models with zero actual return sessions.
Counts remain **160 completed configurations/76 recurring reviews**,
no pending strategy candidate and no champion.
No cohort, statistic, block, seed or alpha is refitted to obtain significance.
**The full investment objective remains unmet.**

## SKEW tail-asymmetry source: verified definition, incomplete quotes

The [source-only policy](research/workbench/config/skew-inputs.json)
and [independent source proof](research/workbench/evidence/skew_input_readiness_20261011.json)
prepare actual Cboe SKEW index points alongside unchanged VIX closes.
The declared SKEW dashboard has one scoped static resource-HTML block
advertising the exact daily CSV and whitepaper URLs.
That data block is parsed without executing remote JavaScript.
One ordinary new CSV request and two primary-document requests are preserved;
the old VIX source bytes, URL and acquisition timestamp are not changed.

SKEW is the shifted/scaled risk-adjusted price of a standardized
third-moment payoff on 30-day S&P 500 log returns.
The primary definition maps the skewness price into SKEW with offset
100 and multiplier minus ten.
**Its points are not annualized volatility percentages, an asset price,
traded returns or physical crash probabilities.**
SKEW/VIX is not a volatility-term ratio.
The paper's probability examples are model-based risk-adjusted approximations;
a zero third moment does not generally establish Gaussian returns.
The publisher's typical 100-150 range is not a hard source bound.
Neither higher values nor a SKEW value of 100 authenticates a safe/risky
trading regime by itself.

Independent parsing verifies all 18,536 actual raw SKEW/VIX observations
and 6,412 finite aligned quotes with zero numeric error.
The 3,208-NYSE-day source context begins January 2, 2014;
all original 2,805 market days and 131 monthly queries remain.
**Full daily and monthly SKEW readiness are both false.**
The actual daily history lacks September 14, 2017; December 3, 2018;
July 5, 2019; and November 29, 2024.
The last is an original month-end query, so only 130 of 131 monthly SKEW
readings exist. Four daily missing cells and that month's explicit
unavailable/reason fields remain; no fill, neutral-value default,
cash/standby rule or deleted return day repairs them.

The advertised whitepaper filename refers to January 2011, the actual PDF
has 2010 copyright and historical 1990-2010 charts, and the actual CSV
starts January 2, 1990. These facts do not authenticate the exact live
launch day, full historical methodology chain, original intraday publication
or unrevised point-in-time quotes.
Any future trading hypothesis needs separately frozen rules,
source/missingness requirements, actual registration and remote publication
before performance calculation.

The [preservation proof](research/workbench/evidence/skew_input_preservation_20261011.json)
retains the exact 160-configuration/76-review journal, engine/head,
seven forward profiles, 69 imports, 3,192 controls and original W42 plans.
No new risk states, quantiles, targets, accounts, definitions, engine
migration or forward day are generated.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v56.json)
keep the source gaps, interpretation limits, fixed current-76 inference
scope and daily/weekly cadence.
Both frozen forward models still have zero actual returns.
**Source verification is not a new alpha factor or completed investment goal.**

## Monthly SKEW response scopes: both rejected under full gates

The [frozen policy](research/workbench/config/skew-factor-guard.json)
and [generator](research/workbench/src/us_quant/skew_factor_guard.py)
compare two responses to the same illustrative SKEW level of 130.
The primary linear mapping corresponds to skewness price minus three;
the source does not recommend this trading threshold or certify any
physical crash probability.
At each original completed-month decision, a quote at or above 130
halves the original four-style/SOXX equity sleeve for one candidate
and all original non-BIL exposure, including GLD, for the other.
Removed capital goes to actual BIL.
Equity-only transfer keeps GLD exact; both retain original stock ratios,
monthly risk budgets, 98% funded total, capital and costs.

The missing SKEW reading on November 29, 2024 is handled by a separately
frozen strategy law: no new monthly request, not a zero return or cash
portfolio. Actual holdings keep all marks, cash and fees.
All formal 2,512/1,254 return days remain and unknown missingness is fatal.
The source proof's full daily/monthly readiness remains false.
Neither another tail threshold nor a daily/dollar/trend/volatility overlay
is selected or tuned after new portfolio outcomes.

The [pre-account proof](research/workbench/evidence/skew_guard_20261012_pre_account_verification.json)
independently reads all 261 genuine SKEW/VIX monthly quote cells,
checks 131 scalar original budgets, four old seeded source paths
and all eight new seeded matrices.
Both issue 130 targets, with 84 half-risk months and the same exact
one-month no-request pause.
Quote, target and seeded errors are zero, at most 5.56e-17 and zero;
original scalar-budget error is at most 7.22e-16.
These are source/rule checks, not candidate returns.

The [actual registration](research/workbench/evidence/skew_guard_20261012_registration.json),
[registration preservation](research/workbench/evidence/skew_guard_20261012_registration_preservation.json)
and [registered snapshot](research/workbench/evidence/research_program_skew_guard_registered_v84.json)
record two actual configurations after one exactly anchored engine migration.
All old rows, seven forward profiles, 69 imports, 3,192 controls,
26 old selection artifacts and the original W42 queue remain.
The previous current-76 inference does not cover these new trials.
Counts remain 160 completed configurations/76 reviews,
with 80 versions including two old invalid ones and exactly two pending.
An old-commit evaluation stops before output or accounts.
**No new candidate portfolio metrics exist at registration;
actual remote publication must precede funded/bt evaluation.**

Actual remote publication
`a5b2221fa79d62a4f14a91eeda10241edd36a1e4` precedes the
00:19:55.233811UTC durable fence and both new account calculations.
The [independently verified results](research/workbench/evidence/skew_guard_20261012_results.json)
retain every path:

| Risk transfer scope | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Equity only | 10y | Base | 14.6443% | 1.00864 | 20.3247% |
| Equity only | 10y | Stress | 14.2217% | 0.97545 | 20.1440% |
| Equity only | 5y | Base | 17.7984% | 1.01614 | 20.4750% |
| Equity only | 5y | Stress | 17.1658% | 0.97315 | 20.2924% |
| All non-BIL | 10y | Base | 12.2044% | 1.02839 | 17.4188% |
| All non-BIL | 10y | Stress | 11.6995% | 0.97798 | 17.5561% |
| All non-BIL | 5y | Base | 13.7434% | 0.95625 | 17.5680% |
| All non-BIL | 5y | Stress | 13.0643% | 0.89407 | 17.7019% |

**All eight drawdown gates, all four stress Sharpe gates and every
ten-year SPY gate fail. Both configurations remain rejected.**
Equity-only transfer beats five-year SPY but still fails drawdown and
stress Sharpe; all-non-BIL transfer misses every SPY return gate.
The lower 17.70% drawdown is still above 15%, not a near-threshold pass.
No SKEW level, reduction scale, pause, asset scope, clock or cost is refitted.

The [reproducibility proof](research/workbench/evidence/skew_guard_20261012_reproducibility.json)
checks every source level/scope/target/seed, 16 funded strategy/SPY
accounts and 16 bt accounts over 30,128 funding/cash-flow days.
Target, metric and wealth errors are at most 5.56e-17, 5.21e-14
and USD7.28e-11.
Eight exact pause-due-day checks verify no rebalance/cost, continuing
risk holdings and every formal return day.
The [four matched scope comparisons](research/workbench/evidence/skew_guard_20261012_scope_comparison.json)
confirm identical issued stock weights and missing-month rows;
halving gold as well lowers CAGR and drawdown on all four comparisons.
Different NAV, actual quantities, cash and fees mean this is not equal
actual risk or causal gold alpha, and it creates no new accounts/trials.

The [current snapshot](research/workbench/evidence/research_program_skew_guard_results_v85.json)
contains **162 completed configurations = 84 prior + 78 recurring reviews**,
80 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v55.json)
reexecutes all 78 reviews/312 paths, retaining 19 legacy and 59 funded-v2 reviews.
The [preservation proof](research/workbench/evidence/skew_guard_20261012_preservation.json)
retains every old row, seven profiles, 69 imports, 3,192 controls,
26 old selection artifacts and the original W42 plans.
The fixed current-76 conditional inference does not cover these two new trials.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v57.json)
keep daily/weekly cadence and both frozen forward models with zero actual returns.
**The full investment objective remains unmet.**

## Housing pipeline vintages: source completeness remains blocked

The [frozen source policy](research/workbench/config/housing-inputs.json)
investigates housing authorization (`PERMIT`) and foundation-start
activity (`HOUST`) as related stages of an interest-sensitive economic
pipeline. They are **not two new independent factors**; the previously
studied CFNAI already includes housing information.
Official FRED notes and exact ALFRED metadata identify monthly
**thousands of units, seasonally adjusted annual rates**.
Permit-issuing-place coverage differs from all starts, so their
difference is not a measured backlog. Unit counts are not asset
prices, returns, volatility percentages or raw monthly counts.
The initial Census definitions request returned 403 and was not bypassed.
Complete historical survey/seasonal-methodology and intraday-release
chains are not certified.

The [source verifier](research/workbench/src/us_quant/housing_inputs.py)
reuses the original monthly calendar, bounded form-date helper and
62-calendar-day completed-month freshness selector without changing
them. Twelve ordinary public-form ZIP batches contain 450 actual
historical snapshots, plus six separately acquired probes and four
primary series/form documents.
The source period labels end at 2026-08-01; market as-of 2026-10-05,
all 2,805 context sessions, 131 monthly/reference pairs and 225
queries per series are unchanged.
Each snapshot is from the second-previous NYSE session.
A first-of-month period label is not an assumed first release.

The [independent source proof](research/workbench/evidence/housing_input_readiness_20261012.json)
reconstructs all 450 choices using raw CSV strings, Decimal values
and scalar NYSE dates, with zero unit/date/age differences.
**436 queries are fresh; 14 are stale, so full readiness remains false.**
Both series have the same seven blocked dates:

| Required query | Latest observed month | Age from completed month end |
|---|---|---:|
| 2025-11-25 | 2025-08 | 86 days |
| 2025-11-28 | 2025-08 | 89 days |
| 2025-12-29 | 2025-08 | 120 days |
| 2025-12-31 | 2025-08 | 122 days |
| 2026-01-29 | 2025-10 | 90 days |
| 2026-01-30 | 2025-10 | 91 days |
| 2026-04-30 | 2026-01 | 89 days |

No freshness relaxation, newer revised-history fallback, fill,
query deletion, cash/standby state or shortened return window is used.
These are source-availability failures, not proof about why the
publisher or archive lacks a newer observation.
There are no housing signals, ranks, targets, portfolios,
performance results, candidate/factor registrations or migrations.
Any later trading/missingness rule requires its own prior freeze,
actual registration and remote publication before accounts.

The first private independent metadata reader incorrectly assumed
dated `Real-Time End` fields; actual `Current` literals interrupted
its wrapped-title join. Its original code and traceback remain.
Only that private suffix normalization is corrected: the complete
raw-CSV numeric/date AST, production parser, source policy, cutoffs,
age limit and all acquired bytes remain unchanged, with no refetch.
160 coupled source cases and 52 root guards pass.

The [preservation proof](research/workbench/evidence/housing_input_preservation_20261012.json)
keeps **162 completed configurations / 78 reviews / 80 versions**,
six definitions / five families, no pending candidate and no champion.
Every old ledger row/head/engine, seven forward profiles, 69 imports,
3,192 controls and 26 old selection artifacts is preserved.
The current-76 selection scope is not extended to the two SKEW trials.
The original four W42 plans, IJR/PKW quota and both frozen forward
models remain, with zero actual model return days.
The [native update](research/workbench/evidence/prospective_data_automation_v58.json)
retains daily collection / Saturday research without claiming timer
delivery from configuration alone.
**This is new source research, not a qualified strategy; the full goal remains unmet.**

## Monthly housing pipeline confirmations: both rejected under full gates

The [frozen strategy policy](research/workbench/config/housing-factor-guard.json)
compares a decline in the earlier authorization stage with the same
decline requiring confirmation from actual foundation starts.
Both use the original four-style/SOXX/gold source, unchanged
63-session volatility / 30%-70% equity bounds / half-equity SOXX /
equal four-style remainder / 98% investment and original cash/cost rules.
No later option-budget, Dollar, trend or activity winner is substituted.

At each original month end, compare the latest actually available
`PERMIT` reading with its **own 63-NYSE-session-earlier vintage/query**
reading. The first candidate halves all original non-BIL weights
only when the current reading is strictly lower.
The second requires both this decline and a strictly lower `HOUST`
reading versus its own earlier query. Equal readings do not trigger.
Freed capital goes to actual BIL; no new scale, threshold grid,
daily timing, factor share, holding band or return-based optimization.
These differences can include revisions and methodology changes;
the housing stages are not two independent factors or recession labels.

The strict source remains false at its original 14 stale cells.
A **separate prior-frozen strategy law** issues no new monthly request
on the seven exact current/reference-pair affected months:
2025-11-28, 2025-12-31, 2026-01-30, 2026-02-27,
2026-03-31, 2026-04-30 and 2026-07-31.
Targets are all NaN there, not cash or zero-return positions.
Actual holdings, marks, cash, fees and all 2,512 / 1,254 formal
return days continue. Unknown or changed gap/date/age/hash is fatal,
not an expanded standby rule or source-readiness repair.

The [adapter](research/workbench/src/us_quant/housing_factor_guard.py)
reuses the unchanged housing archive/62-day monthly selector and
original sector target builder.
The [independent pre-account proof](research/workbench/evidence/housing_guard_20261012_pre_account_verification.json)
checks all 450 raw/Decimal housing selections, 131 scalar original
budgets, four original seeded source paths, both complete target
matrices and eight independently seeded replays.
Each candidate issues 124 targets; the permit-only / jointly confirmed
rules halve risk on 58 / 35 monthly requests.
Source/old-seed/new-seed errors are zero; budget and target errors
are at most 2.23e-16. No portfolio outcome is computed at this stage.
156 coupled cases and 52 root guards pass.

The [actual registration packet](research/workbench/evidence/housing_guard_20261012_registration.json)
records candidates at **2026-10-12 01:11:38 / 01:11:41 UTC**,
before any account.
The [registration preservation](research/workbench/evidence/housing_guard_20261012_registration_preservation.json)
proves one exact old-engine/head migration, every old row/event prefix,
all seven semantic profiles, 69 imports, 3,192 controls and 26
old selection artifacts.
An actual evaluation against the preceding source-only published
commit stops before output/accounts; the new packet/code/policy/proof
must be remotely published before funded-v2 / independent-bt evaluation.

The [registration snapshot](research/workbench/evidence/research_program_housing_guard_registered_v86.json)
retains **162 completed configurations / 78 reviews**, 82 registered
versions including two old invalid versions, and exactly these two
pending candidates. Six definitions / five families, the original
W42 plans / IJR-PKW quota and both frozen zero-return forward models
remain unchanged. Prior current-76 selection inference does not cover
these two housing or the two later SKEW trials, nor the full search.
**No housing performance or investment qualification is claimed at registration.**

Actual remote publication
`3350918b83e3d98ffd60e59b8e801d4e2a92783a` precedes the
01:16:36.404238UTC durable fence and both new account calculations.
The [independently verified complete results](research/workbench/evidence/housing_guard_20261012_results.json)
retain every path:

| Housing rule | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Permit decline | 10y | Base | 12.9429% | 0.95581 | 17.9170% |
| Permit decline | 10y | Stress | 12.3428% | 0.90459 | 17.7400% |
| Permit decline | 5y | Base | 15.4825% | 1.02307 | 14.4765% |
| Permit decline | 5y | Stress | 14.5691% | 0.94694 | 14.2807% |
| Permit + starts | 10y | Base | 14.6459% | 0.96919 | 17.9149% |
| Permit + starts | 10y | Stress | 13.3616% | 0.87763 | 19.4910% |
| Permit + starts | 5y | Base | 16.3599% | 0.91359 | 17.9745% |
| Permit + starts | 5y | Stress | 14.5895% | 0.79839 | 19.6538% |

**The permit-only five-year base path passes all three gates,
but both complete configurations remain rejected.**
Its five-year stress return/drawdown pass while Sharpe 0.94694 fails.
Every ten-year Sharpe, drawdown and SPY comparison fails for both rules.
Joint-stage confirmation misses every Sharpe/drawdown gate, despite
all five-year SPY return comparisons passing.
The complete permit-only / jointly confirmed worst drawdowns are
17.9170% / 19.6538%, with minimum Sharpe 0.90459 / 0.79839.
No stage, comparison horizon, half scale, source age, pause, clock or
fee is refitted, and partial successes are not spliced or promoted.

The independently identified ten-year base worst episode is
2020-02-19 through 2020-03-20; the permit-only stress peak is 2020-02-21.
The other worst episodes are 2022-03-08 through 2022-10-14.
Frozen inputs on 2020-02-28 still describe January housing and show
rising authorization and starts. Permit decline appears at the
March month end; both stages decline at the April month end,
after the March trough. This is descriptive timing evidence,
not a causal decomposition or justification to backdate a release
or refit a faster rule to the known crash.

The [account proof](research/workbench/evidence/housing_guard_20261012_reproducibility.json)
checks 16 funded strategy/SPY and 16 independent bt accounts,
30,128 actual-price funding/fee/cash-flow days and all 56
pause-due-day no-order/no-cost/positive-non-BIL-risk checks.
Seeded target error is zero; scalar metric, independent wealth
and cash-flow errors are at most 4.36e-14, USD6.55e-11 and USD1.10e-11.
All formal returns, fresh capital and initial-capital drawdown peaks remain.
The strict 436-fresh/14-stale source is still blocked, not repaired
by its separately frozen strategy no-request law.

The [current snapshot](research/workbench/evidence/research_program_housing_guard_results_v87.json)
contains **164 completed configurations = 84 prior + 80 recurring reviews**,
82 registered versions including two old invalid ones, no pending
candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v56.json)
reexecutes all 80 reviews / 320 paths, preserving 19 legacy and 61 funded-v2 reviews.
The [preservation proof](research/workbench/evidence/housing_guard_20261012_preservation.json)
retains every old row/event prefix, seven profiles, 69 imports,
3,192 controls, 26 old selection artifacts and the original W42 plans.
The fixed current-76 conditional inference does not cover the
two new housing or two later SKEW trials.
The [native update](research/workbench/evidence/prospective_data_automation_v59.json)
retains daily/weekly cadence and both unchanged zero-return forward models.
**The full investment objective remains unmet.**

## Bounded dividend-source recovery: registered, not yet requested

The [source-only recovery policy](research/workbench/config/dividend-source-recovery.json)
and [controller](research/workbench/src/us_quant/dividend_source_recovery.py)
prepare a lawful retry of the original blocked VIG/SDY histories,
not another risk threshold or a new strategy result.
Original issuer definitions, requests, HTTP429 responses and all
source hashes remain unchanged.
The same URL, GET method, default request headers/user agent,
curl transport, 2015-08-10 through 2026-10-05 range and 2,805-session
price/action validator are retained.
Saving response headers does not change request headers.
NOBL's403 refusal is not retried or bypassed.

The [actual source registration](research/workbench/evidence/dividend_source_recovery_20261012_registration.json)
freezes a conservative **24-hour minimum global cooldown** from the
latest original refusal, so the first new request cannot occur
before **2026-10-12 09:29:43.277441 UTC**.
The original response did not retain a `Retry-After` header;
this local limit is **not certified provider permission**.
A later directive may extend, never shorten, the cooldown.
Maximum one request per invocation and one per 24 hours;
VIG precedes SDY, each has at most one new attempt, total at most two.
Any refusal, transport failure, invalid/incomplete payload or
unverified partial attempt stops all further automated requests.
No changed host, endpoint, headers, browser challenge or current-history
substitute is used to bypass an access or rate restriction.

Actual remote publication of the exact registration, controller,
policy and original source contracts must precede a due request.
The [real-clock proof](research/workbench/evidence/dividend_source_recovery_20261012_verification.json)
shows the early invocation returns `not_due`, performs **zero**
provider requests and creates no attempt directory.
69 coupled controller/source-contract cases and 52 root guards pass;
synthetic tests are not price coverage or a successful provider retry.
A later acquired history still requires full raw/CSV/action inspection
before source admission. Acquisition is not factor registration,
portfolio return or a prospective market/model observation.

**No new dividend history is acquired at this checkpoint.**
164 completed configurations / 80 reviews / 82 versions, all old rows,
engine/head, seven profiles, 69 imports, 3,192 controls and 26 old
selection artifacts remain exact.
The W42 IJR/PKW quota and four plans are unchanged; any separate
dividend hypothesis remains ineligible before 2026-10-24.
There is no target, strategy account, new factor or investment qualification.

Actual remote publication
`039980c9a67f6be565569f5e595dbda1d29631a5` has been verified against
the exact six frozen public registration/controller/source-contract files;
this verification issues no quote request.
The [updated native instructions](research/workbench/evidence/prospective_data_automation_v60.json)
retain the original two daily pipelines in order.
Only after both succeed, inspect the independent source-recovery status:
`not_due` / `all_sources_acquired` sends no request;
an eligible state may run the registered controller using that exact
published commit. A failed/partial recovery stops source requests
without rolling back already completed prospective input steps.
This instruction update is not evidence that a timer ran or a provider
accepted a request. **The investment objective remains unmet.**

## Fixed observed-weight execution for original option budgets

The [frozen implementation policy](research/workbench/config/implied-risk-execution.json)
transfers the **already specified 2.5-percentage-point actual-holding
no-trade band** to both original plain / dollar-half option-proxy budgets.
It does not change their GVZ/VIX ratio, 30%-70% bounds, four-style/SOXX
composition, gold remainder, dollar regime, monthly desired targets,
costs or capital, nor assume that fewer trades will close the SPY gap.
This is execution research, not a new economic factor or return alpha.

The [adapter](research/workbench/src/us_quant/implied_risk_execution.py)
calls the unchanged original `observed_weight_execution.run`.
The old candidate handle selects only the pure-band law; it does not
claim reproduction of the old graded-gold inputs or historical account.
Initial fresh USD10,000 deployment is forced.
Later source updates compare the complete unchanged desired target,
including BIL, with **actual post-trade closing holdings**.
Only the original maximum-gap-at-least-0.025 rule, with its existing
1e-12 rounding allowance, issues a request.
No last-issued-target proxy, QLD cap, volatility override or extra
risk/source threshold is introduced.
Pending requests cannot be overwritten, canceled or filled early.

Desired targets remain at all 131 original month ends.
Actual issued rows, costs, drift and holdings depend on each window's
fresh capital, realized prices, fees and delay; they are calculated
**only after actual remote registration/code/proof publication**.
Skipping small corrections can worsen exposure, drawdown or return.
Identical desired budgets do not imply identical realized risk,
and two-percent cash remains a target rather than a continuous floor.

The [independent static proof](research/workbench/evidence/implied_execution_20261012_pre_account_verification.json)
checks 262 actual native monthly quotes, 131 published H.10 pairs,
131 Decimal budget/composition rows per source and all eight
original causal seeds against saved source targets.
Raw source / desired budget / old seeded target errors are at most
3.02e-16 / 1.12e-16 / zero.
Four nonmonthly GVZ gaps remain NaN with strict daily readiness false;
the separately complete monthly input scope is not source repair.
76 coupled allocation/execution/registration cases and 52 root guards pass.

The [actual registrations](research/workbench/evidence/implied_execution_20261012_registration.json)
occur at **2026-10-12 02:37:47 / 02:37:51 UTC**.
One exact anchored migration and the [preservation proof](research/workbench/evidence/implied_execution_20261012_registration_preservation.json)
retain all old rows/event prefix, seven semantic profiles,
69 imports, 3,192 controls and 26 old selection artifacts.
Source recovery keeps its exact fingerprint, cooldown and zero requests.
An actual attempt with the preceding published commit stops before
feedback target or account output.

The [registration snapshot](research/workbench/evidence/research_program_implied_execution_registered_v88.json)
retains **164 completed configurations / 80 reviews**, 84 registered
versions including two old invalid ones, and exactly these two pending
execution candidates. Six definitions / five families, original W42
plans / IJR-PKW quota and both zero-return forward models remain.
The fixed current-76 inference does not cover these, the later four
housing/SKEW trials or the complete search.
**No new feedback account, strategy performance or qualification exists at registration.**

Actual remote publication
`f50a7862ddcccff1f6ec2497f2a19111aeecaa92` precedes the
02:44:01.085360UTC durable fence and every new feedback account.
The [independently verified results](research/workbench/evidence/implied_execution_20261012_results.json)
retain both configurations and all eight paths:

| Execution source | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Plain option budget | 10y | Base | 18.5663% | 1.06507 | 21.4509% |
| Plain option budget | 10y | Stress | 18.4227% | 1.05677 | 21.0382% |
| Plain option budget | 5y | Base | 22.7199% | 1.07525 | 21.5450% |
| Plain option budget | 5y | Stress | 22.2029% | 1.05053 | 21.2000% |
| Dollar-half budget | 10y | Base | 14.7229% | 1.09538 | 13.5721% |
| Dollar-half budget | 10y | Stress | 14.4641% | 1.07396 | 13.4137% |
| Dollar-half budget | 5y | Base | 19.1188% | 1.16849 | 13.5913% |
| Dollar-half budget | 5y | Stress | 18.5775% | 1.13254 | 13.4332% |

**Both complete configurations remain rejected.**
Plain execution passes every Sharpe/SPY gate but every drawdown gate fails.
The guarded version passes every Sharpe/drawdown gate and both
five-year joint paths; minimum Sharpe is 1.07396 and worst drawdown
13.5913%. Its ten-year CAGRs still trail full-SPY 15.4620% / 15.4056%;
the stress path also misses the base-cost SPY hurdle.
Passing plain returns cannot be combined with guarded drawdown,
and no band, proxy ratio, dollar scale, budget, clock or fee is refitted.

All eight matched original-source comparisons have fewer orders,
lower total modeled dollar costs and higher CAGR/Sharpe.
Guarded ten-year stress tickets fall **809 to 442** and modeled fees
USD2,116.90 to USD1,710.97; CAGR improves from 14.2969% to 14.4641%,
which is still insufficient.
These are comparisons of actual saved accounts, not new controls,
independent samples, equal realized risk or causal fee-alpha estimates.
Skipped requests permit different drift/holdings/returns even though
the immutable desired source budgets are unchanged.

The [independent feedback/account proof](research/workbench/evidence/implied_execution_20261012_reproducibility.json)
checks **728 closing-holding gap/initial/queue decisions**, every
issued and held row, all original target values and 30,128 funding/
quantity/fee/cash-flow days. All 16 funded strategy/SPY and 16
independent bt accounts preserve complete windows and capital peaks.
Issued-target error is zero; scalar metric / wealth / cash-flow
errors are at most 4.51e-14 / USD6.55e-11 / USD1.44e-11.

The [current snapshot](research/workbench/evidence/research_program_implied_execution_results_v89.json)
contains **166 completed configurations = 84 prior + 82 recurring reviews**,
84 registered versions including two old invalid ones, no pending
candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v57.json)
reexecutes all 82 reviews / 328 paths, preserving 19 legacy and 63
funded-v2 reviews without changing any old outcome.
The [preservation proof](research/workbench/evidence/implied_execution_20261012_preservation.json)
keeps every old row/event prefix, seven profiles, 69 imports,
3,192 controls, 26 old selection artifacts, source-recovery fingerprint/
zero requests and the original W42 plans.
The current-76 conditional inference does not cover the six later trials.
The [native update](research/workbench/evidence/prospective_data_automation_v61.json)
retains daily/weekly cadence, bounded source recovery and both frozen
zero-return forward models. **The full investment objective remains unmet.**

## Quarterly profit/output vintages: declared source grid verified

The [frozen source policy](research/workbench/config/profit-inputs.json)
adds economic earnings information through **CPATAX**, aggregate
corporate profits after tax with inventory valuation / capital
consumption adjustments, and **GDP**, nominal domestic output.
Primary FRED definitions and exact ALFRED metadata identify BEA
accounts A551RC / A191RC and native quarterly billions of dollars
at seasonally adjusted annual rates.
The initially inspected `CP` series explicitly lacks those adjustments
and is not substituted for CPATAX; real GDP / growth-rate downloads
are not substituted for nominal levels.
These are related aggregate sources, not two independent factor
families, company EPS, filing-time stock fundamentals, cash flows
or tradable fund returns.

The [source-only verifier](research/workbench/src/us_quant/profit_inputs.py)
reuses the unchanged bounded vintage-date helper, original monthly/
63-session query cohort and scoped archive metadata parser.
Quarter-start labels identify observation periods rather than releases.
A **new quarterly contract**, frozen before numeric acquisition,
limits age to 186 calendar days from the completed quarter end
(two 93-day periods). It does not relax old monthly62 / weekly14
freshness, manufacture a quarter-end release or optimize a trading parameter.
The observation range 2014-01-01 through 2026-04-01 is authenticated
by the actual advertised forms; market as-of 2026-10-05 and all
original 2,805 context days / 131 pairs / 225 queries remain.

The [independent source proof](research/workbench/evidence/profit_input_readiness_20261012.json)
checks 12 ordinary public-form ZIP batches / **450 actual snapshots**,
six separately acquired probes and four primary series/form documents.
Raw CSV strings, Decimal values, scalar NYSE cutoffs and calendar
quarter ends reproduce all 450 selections with zero value/date/age errors.
**450 are fresh, none blocked; maximum age is 182 days, within 186.**
Full readiness is limited to this exact declared query cohort,
not every daily snapshot, complete historical NIPA/accounting
methodology or original intraday first-publication proof.

**87 of the 225 query dates have different independently latest
profit and output quarters.** Those levels are not silently mixed
into a profit margin or regime signal.
Aggregate populations also need not match company-level profitability.
Any later common-period join, ratio or portfolio rule requires its own
prior freeze, complete source-bound selection proof, actual registration
and remote publication before accounts.
No ratio, rank, recession state, target, account, performance,
new economic definition or candidate is computed here.

The [preservation proof](research/workbench/evidence/profit_input_preservation_20261012.json)
retains **166 completed configurations / 82 reviews / 84 versions**,
six definitions / five families, no pending candidate or champion.
Every old ledger row/engine/head, seven profiles, 69 imports,
3,192 controls, 26 old selection artifacts and original W42 plans remain.
Source recovery retains its exact fingerprint / cooldown / zero GET;
the fixed current-76 inference remains outside the six later trials.
170 coupled source cases and 52 root guards pass.
The [native update](research/workbench/evidence/prospective_data_automation_v62.json)
keeps daily/weekly cadence and both unchanged zero-return forward models.
**Verified new input coverage is not portfolio alpha or full investment success.**

## Common-quarter earnings confirmations: registered before accounts

The [frozen strategy policy](research/workbench/config/profit-factor-guard.json)
uses new earnings information to distinguish dollar strength with
healthy income from dollar strength accompanied by deteriorating income.
It compares a strictly falling common-quarter adjusted-profit level
with a strictly falling common-quarter aggregate-profit/output proxy.
The latter is **not a company profit margin or new independent factor**;
aggregate profits and domestic output can have different population scope.
Own-query differences can include revisions and tax/accounting changes.

The [selector](research/workbench/src/us_quant/profit_factor_guard.py)
uses each exact second-prior-NYSE CPATAX/GDP snapshot pair and selects
the **latest completed quarter actually present in both**.
Both values come from that same query's earlier snapshots and same
quarter, with the original 186-day completed-quarter freshness.
All 225 queries remain; no newer GDP quarter is mixed into an older
profit reading. Missing, stale, changed or unverified joins are fatal.
The original source proof's 87 independently latest-quarter differences
remain true; this is a separately frozen information-selection law.

At each original 131 month ends, compare the chosen measure with
the measure from its own 63-NYSE-session-earlier query/vintage.
Only original positive dollar change **and** a strictly lower
profit measure halves all original plain option-budget non-BIL
weights to actual BIL. Equal/rising measures do not trigger.
The source's GVZ/VIX risk budget, 30%-70% bounds, half-equity SOXX,
equal four-style remainder, gold and 98% funding are unchanged.
The unchanged original 2.5pp actual-closing-holding band, forced
fresh-capital deployment, dated queue and full costs are retained.
Actual feedback targets/accounts remain deferred until remote publication.

The [independent pre-account proof](research/workbench/evidence/profit_guard_20261012_pre_account_verification.json)
checks all 450 raw quarterly snapshots / 225 scalar common-quarter
joins, 262 raw option inputs, 131 H.10 pairs, 131 scalar original
budgets and eight static desired-seed contexts.
Join/level/proxy/comparison error is at most 4.44e-13;
option/dollar error 3.02e-16, original budget error 1.12e-16
and desired-target/seed error 2.23e-16.
Both retain 131 desired updates; level / output-share rules halve
risk on 22 / 32 months. No actual candidate wealth is calculated here.
109 coupled cases and 52 root guards pass.

The [actual registrations](research/workbench/evidence/profit_guard_20261012_registration.json)
occur at **2026-10-12 04:06:45 / 04:06:48 UTC**.
The [registration preservation](research/workbench/evidence/profit_guard_20261012_registration_preservation.json)
proves one exact anchored migration and retains every old row/event
prefix, seven semantic profiles, 69 imports, 3,192 controls,
26 old selection artifacts and source-recovery fingerprint/zero requests.
An actual run with the preceding source-only published commit stops
before feedback target/account output.

The [registration snapshot](research/workbench/evidence/research_program_profit_guard_registered_v90.json)
retains **166 completed configurations / 82 reviews**, 86 registered
versions including two old invalid versions, and exactly these two
pending candidates. Six definitions / five families, original W42 plans /
IJR-PKW quota and both frozen zero-return forward models remain.
The old current-76 inference does not cover these or the six later
trials, the full search or future alpha.
**New earnings portfolio outcomes do not exist at registration.**

Actual remote publication
`b188608ef8e9d01ac413746e9e65c606c1f7ab5d` precedes the
04:17:12.957439UTC durable fence and both new feedback-account calculations.
The [independently verified full results](research/workbench/evidence/profit_guard_20261012_results.json)
retain both rules and every original window/cost path:

| Earnings confirmation | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Profit level | 10y | Base | 17.7867% | 1.06145 | 17.8577% |
| Profit level | 10y | Stress | 17.1554% | 1.02224 | 18.5605% |
| Profit level | 5y | Base | 22.8911% | 1.12966 | 17.9531% |
| Profit level | 5y | Stress | 21.5506% | 1.05999 | 18.7239% |
| Income/output proxy | 10y | Base | 17.1115% | 1.08886 | 18.2181% |
| Income/output proxy | 10y | Stress | 16.7293% | 1.06476 | 18.5126% |
| Income/output proxy | 5y | Base | 21.1143% | 1.08660 | 18.2927% |
| Income/output proxy | 5y | Stress | 20.5215% | 1.05627 | 18.6544% |

**All eight Sharpe / cost-matched SPY gates and all four stress-versus-base-SPY
gates pass, but every drawdown gate fails. Both configurations remain rejected.**
Complete level / share worst drawdowns are **18.7239% / 18.6544%**,
not near-threshold passes at 15%; minimum Sharpes are 1.02224 / 1.05627.
Common-quarter selection avoids timing mistakes but does not establish
the required stability or independent predictive alpha.
No join, accounting period, measure, 186-day age, 63-session horizon,
sign, half scale, band, source budget, cost or clock is refitted.
The old dollar-band version's lower drawdown cannot be combined with
these higher earnings-rule returns.

The [independent account proof](research/workbench/evidence/profit_guard_20261012_reproducibility.json)
checks every frozen common-query/comparison hash, 728 actual closing
gap/initial/queue decisions, 16 funded strategy/SPY and 16 independent
bt accounts across 30,128 price/quantity/funding/fee/cash-flow days.
Issued-target error is zero; metric / wealth / cash-flow differences
are at most 3.47e-14 / USD1.10e-10 / USD1.43e-11.
Every original return session, cost, latency, independent capital
anchor and initial-capital drawdown peak remains.

The [current snapshot](research/workbench/evidence/research_program_profit_guard_results_v91.json)
contains **168 completed configurations = 84 prior + 84 recurring reviews**,
86 versions including two old invalid ones, no pending candidate and no champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v58.json)
reexecutes all 84 reviews / 336 paths, preserving 19 legacy and 65 funded-v2 reviews.
The [preservation proof](research/workbench/evidence/profit_guard_20261012_preservation.json)
retains every old row/event prefix, seven profiles, 69 imports,
3,192 controls, 26 old selection artifacts, W42 plans and source-recovery
fingerprint / cooldown / zero GET.
The fixed current-76 inference does not cover the eight later trials.
The [native update](research/workbench/evidence/prospective_data_automation_v63.json)
retains daily/weekly cadence and both unchanged zero-return forward models.
**The full investment objective remains unmet.**

## Fixed monthly earnings / option-pressure complementarity

The [frozen policy](research/workbench/config/profit-term-confirmation.json)
studies whether market-priced option pressure complements slower
quarterly income information. It uses both original earnings measures,
the already specified VIX at least VIX3M threshold1 and the same half scale,
not a new pressure cutoff or protection mask for a selected crash.
This is an exposed-history hypothesis, not independent alpha or
an economic-factor definition.

At each original 131 month end, retain the positive own-vintage dollar
condition and halve the original plain option-budget non-BIL weights
when **either** the selected common-quarter earnings measure declines
**or** actual month-end VIX is at least VIX3M.
If both branches trigger, the half scale is applied **once**.
Profit equality does not trigger; option equality does.
Otherwise the original full budget remains.
No daily regime events, changed quarter join/age63comparison,
asset share, proxy beta, fee, holding-band width or trade timing is added.

The [adapter](research/workbench/src/us_quant/profit_term_confirmation.py)
keeps actual 30-day VIX / current93-day VIX3M source closes,
full sealed primary documents / ticker rename / historical-methodology
limits, and original GVZ/VIX monthly allocation.
All quarter-selection, dollar-release, 98%funding, four-style/SOXX/gold
and fixed2.5pp actual-holding queue rules remain.
Actual feedback/account construction occurs only after remote publication.

The [independent pre-account proof](research/workbench/evidence/profit_term_20261012_pre_account_verification.json)
checks 225 common-quarter joins from450snapshots, 131H.10pairs,
262GVZ/VIX and262VIX/VIX3M monthly cells, scalar budgets and eight
static desired seeds. Join/proxy, raw source, original budget and
desired-target/seed differences are at most
1.51e-17 / 3.02e-16 / 1.12e-16 / 2.23e-16.
Level / income-output rules halve risk on **28 / 36 months**:
22 / 30 earnings-only, 6 / 4 option-only and 0 / 2 joint-branch months.
Both preserve131desiredupdates and no doublehalving.
85 coupled cases and52root guards pass.

The [actual registration](research/workbench/evidence/profit_term_20261012_registration.json)
records candidates at **2026-10-12 06:18:40 / 06:18:43 UTC**.
The [preservation proof](research/workbench/evidence/profit_term_20261012_registration_preservation.json)
retains every old row/event prefix, seven semantic profiles,
69imports,3,192controls,26selection artifacts and source-recovery
fingerprint/cooldown/zeroGET through one exact anchored migration.
The pre-existing immutable baseline is retained after its
duplicate-invocation guard safely refuses a rerun; it is not regenerated.
An actual run with the preceding earnings-only published commit
stops before feedback target or account output.

The [registration snapshot](research/workbench/evidence/research_program_profit_term_registered_v92.json)
contains **168 completed configurations /84reviews**,88registered
versions including two old invalid ones, and only these two pending candidates.
Six definitions/fivefamilies, originalW42plans/IJR-PKWquota and both
zero-return forward models remain.
Oldcurrent76conditionalinference does not cover these or the eight
later trials/fullsearch/futurealpha.
**No complementary portfolio outcomes exist at registration.**

Actual remote publication
`996f87a11e77e8980a9c6602132f5f1cb537352a` precedes the
06:50:35.034882UTC durable fence and every new feedback account.
The [independently verified complete results](research/workbench/evidence/profit_term_20261012_results.json)
retain both earnings/option constructions:

| Complementary rule | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
|---|---|---|---:|---:|---:|
| Level OR pressure | 10y | Base | 17.2162% | 1.07891 | 18.2241% |
| Level OR pressure | 10y | Stress | 16.6269% | 1.03861 | 18.5174% |
| Level OR pressure | 5y | Base | 22.9522% | 1.14330 | 18.2927% |
| Level OR pressure | 5y | Stress | 21.5935% | 1.07207 | 18.6544% |
| Share OR pressure | 10y | Base | 16.2046% | 1.04726 | 18.2258% |
| Share OR pressure | 10y | Stress | 15.7983% | 1.01964 | 18.5201% |
| Share OR pressure | 5y | Base | 21.2765% | 1.10015 | 18.2927% |
| Share OR pressure | 5y | Stress | 20.5432% | 1.06192 | 18.6544% |

**All eight Sharpe/cost-matched-SPY and four stress/base-SPY gates pass,
but every drawdown gate fails; both complete configurations remain rejected.**
Both worst drawdowns are 18.6544%, still above15%, with minimum
Sharpe1.03861/1.01964.
Adding four/six option-only monthly risk checks does not establish the
required stability. Identical worst-drawdown values do not make different
wealth paths, source decisions or risk exposures identical or independent.
No ORlogic,threshold1,scale,frequency,quarterage,own-query63horizon,
band,sourceallocation,clock or fee is refitted.
Passing high returns cannot be joined to a different old version's
13.59%drawdown to construct a qualified strategy.

The [independent account proof](research/workbench/evidence/profit_term_20261012_reproducibility.json)
checks all common-query/term-input hashes, 728 actual holding-gap/
initial/queue decisions, 16fundedstrategy/SPY and16independentbt
accounts across30,128 price/quantity/funding/fee/cash-flow days.
Issued-target error iszero; metric/wealth/cash-flow errors are at most
3.78e-14/USD6.55e-11/USD1.44e-11.
Every original capital anchor,initialpeak,returnday,fee and delayed
pending target remains.

The [current snapshot](research/workbench/evidence/research_program_profit_term_results_v93.json)
contains **170 completed configurations =84prior+86recurringreviews**,
88registeredversions including two old invalid ones,no pending candidate
andno champion.
The [full source-bound audit](research/workbench/evidence/research_program_source_bound_audit_v59.json)
reexecutes86reviews/344paths,retaining19legacy and67funded-v2 reviews
without altering any old outcome.
The [preservation proof](research/workbench/evidence/profit_term_20261012_preservation.json)
retains old rows/event prefix,7profiles,69imports,3,192controls,
26selection artifacts,W42plans andsource-recovery fingerprint/
cooldown/zeroGET.
The fixed current-76 inference excludes the ten later trials.
The [native update](research/workbench/evidence/prospective_data_automation_v64.json)
retains daily/weekly cadence and both frozen zero-return forward models.
**The full investment objective remains unmet.**

## Dividend source recovery: actual refusal, automated requests stopped

The registered source-only controller makes **one actual VIGrequest**
at 2026-10-12 09:30:24.608592–09:30:25.429726UTC,
after its original 09:29:43.277441UTC local24hour deadline.
The same previously frozen GET,provider URL,default request headers/
user agent,curl transport and full historical range are retained.
Actual source-registration publication
`039980c9a67f6be565569f5e595dbda1d29631a5` is verified before the request.
The [independent actual-attempt proof](research/workbench/evidence/dividend_source_recovery_20261012_attempt.json)
records **HTTP429**,transportexit0 and an unchanged effective URL.
Local cooldown compliance is not provider retry permission.

The raw refusal body,actual response headers,command diagnostics,
intent/request/result and their hashes remain private and preserved.
**All further automated source requests are now stopped.**
SDY is not requested; there is no retry,counter reset,partial deletion,
alternate host/headers/transport/browser challenge or API-key workaround.
The original controller's readonly state check confirms the terminal
failure rather than a valid-looking zero-price or successful history.
VIGCSV is absent: **zero complete new histories,zero admitted price rows**.
Old source manifests,issuer identities,original429receipts and earlier
zero-request checkpoint proofs are not overwritten.

The [preservation proof](research/workbench/evidence/dividend_source_recovery_20261012_attempt_preservation.json)
retains exact **170completed configurations/86reviews/88versions**,
engine/head,all old ledger rows,seven profiles,69imports,
3,192controls,26selection artifacts and originalW42plans.
This actual request is historical source acquisition,not a newfactor,
strategy evaluation,forward model day or proof of native timer delivery.
Any dividend hypothesis remains ineligible before2026-10-24.

The [updated native instructions](research/workbench/evidence/prospective_data_automation_v65.json)
keep the original macro/Dollar daily pipelines first and preserve weekly
discovery cadence. They report the saved terminal source blocker without
another acquisition or rollback of already collected prospective inputs.
69coupled source/recovery contract cases pass without newnetworkrequests;
synthetic tests are not provider price coverage.
**The dividend data remains blocked and the investment objective remains unmet.**

## Research session continuity

The [timer handoff receipt](research/workbench/evidence/prospective_data_automation_v66.json)
records removal and absence verification of the old session's native timer
before attaching the replacement to the receiving research session. The complete
original instructions are preserved byte-for-byte. Daily wakeups remain around
09:00 Asia/Shanghai, with Saturday factor discovery and the original October 17
eligibility. The next configured wakeup is October 13 around 09:00; configuration
is not evidence of future delivery, and the app and host must remain available.

The original conversation was natively compacted without discarding its raw
history. Research code, private inputs, prior reports and the ledger were
preserved, retaining **170 completed configurations / 86 reviews** and no
champion. Both unchanged daily pipelines completed in the receiving worktree
with already-collected inputs and no new observed model session. This verifies
continuity, not new returns, source coverage or a qualified strategy. The
terminal dividend-source stop and both frozen forward experiments remain.

## Point-in-time unemployment source, not a strategy result

The [frozen source policy](research/workbench/config/unemployment-inputs.json)
adds actual BLS household-survey U-3 unemployment, not weekly insured claims,
payrolls or a tradable return. Five primary documents, six bounded ALFRED
batches and a separately acquired three-date probe retain all 225 original
second-prior-NYSE snapshots and the original market/query windows.

The [independent source proof](research/workbench/evidence/unemployment_input_readiness_20261012.json)
checks each snapshot's latest observation and required 15-month history with
native CSV/Decimal values. **210 queries have complete history; 15 are blocked**
by the missing October 2025 observation. All latest-period ages are at most
61 days, within the unchanged 62-day limit: missing history, not staleness,
is the blocker. No value is filled, interpolated or replaced with today's
revised history, and complete source readiness remains false.

Two actual HTTP 200 responses initially exposed local title-validation errors:
the source changed from "Civilian Unemployment Rate" to "Unemployment Rate"
on September 6, 2019, and each archive includes only title intervals relevant
to its requested cohort. The exact dated identities and BLS source code are
now checked; original code, errors and responses remain private. Successful
probe/batch requests were not repeated, and the numeric/source contract
was not relaxed.

No unemployment rolling statistic, Sahm threshold, recession label, portfolio
target or account is computed here. A retrospectively available real-time
indicator is not proof that its later policy rule was deployed in earlier
years. A different latest-observation strategy would need a separately frozen
contract and independent verification; it cannot relabel this 15-month source
as complete. The [preservation proof](research/workbench/evidence/unemployment_input_preservation_20261012.json)
retains **170 configurations / 86 reviews**, no champion, the terminal dividend
stop, original weekly eligibility and both frozen forward experiments.

## Frozen latest-unemployment change hypotheses

The [separate strategy policy](research/workbench/config/unemployment-factor-guard.json)
requires only each query's actual latest completed UNRATE observation. It does
not calculate a Sahm statistic, consume the missing October 2025 value or repair
the separately blocked 15-month source contract.

Two fixed constructions compare the current unemployment level with the value
from its own 63-NYSE-session-earlier query and vintage. One halves the original
plain option-budget non-BIL weights on a strict unemployment increase; the other
also requires the original positive broad-dollar change. Equality does not
trigger. Both retain the original option budget, four factor ETFs, semiconductor
share, gold remainder, cash funding, 2.5-percentage-point actual-holding band,
costs, delayed orders and full ten-year/five-year return windows.

The [independent pre-account proof](research/workbench/evidence/unemployment_guard_20261012_pre_account_verification.json)
checks all 225 latest source values and cutoffs, 131 own-query comparisons,
262 original option quotes, scalar budgets and eight static seed contexts.
The two rules retain all 131 monthly targets and reduce risk on 38 / 20 months.
Maximum source/comparison/target differences are zero / 8.89e-16 / 1.12e-16.
Related labor information is not a new independently proven factor family.

[Actual registration](research/workbench/evidence/unemployment_guard_20261012_registration.json)
occurred at 2026-10-12 14:17:54 / 14:17:58 UTC.
The [preservation proof](research/workbench/evidence/unemployment_guard_20261012_registration_preservation.json)
retains all previous rows and seven forward profiles through one exact engine
migration. A real evaluation with the preceding source-only published commit
stops before any account output.
The [registration snapshot](research/workbench/evidence/research_program_unemployment_registered_v94.json)
has **170 completed configurations / 86 reviews**, 90 registered versions
including two old invalid ones, and exactly these two pending candidates.
Actual remote publication must precede new path-dependent targets and accounts;
**new performance is not yet computed at this registration checkpoint**.

## Completed unemployment change evaluations

Actual remote publication
`675bea84dea97ff84b553145641211cbfdc2ed82` and the
14:22:46.916425 UTC durable fence precede both new feedback-account calculations.
The [independently verified results](research/workbench/evidence/unemployment_guard_20261012_results.json)
retain both fixed rules and all original paths:

| Labor confirmation | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| Unemployment rise | 10y | Base | 14.5921% | 0.91155 | 21.0671% |
| Unemployment rise | 10y | Stress | 14.6223% | 0.91481 | 20.3350% |
| Unemployment rise | 5y | Base | 17.5577% | 0.91460 | 21.2009% |
| Unemployment rise | 5y | Stress | 17.6902% | 0.92305 | 20.4834% |
| Dollar-confirmed rise | 10y | Base | 17.2837% | 1.04427 | 21.0500% |
| Dollar-confirmed rise | 10y | Stress | 17.0739% | 1.03180 | 20.3241% |
| Dollar-confirmed rise | 5y | Base | 20.7519% | 1.04446 | 21.2009% |
| Dollar-confirmed rise | 5y | Stress | 20.3295% | 1.02155 | 20.4834% |

**Every drawdown gate fails and both configurations remain rejected.**
The dollar-confirmed rule passes all four Sharpe/SPY comparisons but its worst
drawdown is 21.2009%, not a near-threshold 15% pass. The unemployment-only rule
fails every Sharpe comparison and both ten-year SPY comparisons. Identical
five-year worst drawdowns are not independent evidence or identical wealth
paths. No source scope, sign, 63-session horizon, half scale, confirmation,
budget, band, cost or order clock is refitted after these outcomes.

The [independent account proof](research/workbench/evidence/unemployment_guard_20261012_reproducibility.json)
checks 728 actual closing-gap/initial/queue decisions, 16 funded and 16
independent bt strategy/SPY accounts over 30,128 funding sessions.
Issued-target error is zero; scalar metric, wealth and cash-flow errors are
at most 3.97e-14, USD 1.39e-10 and USD 1.10e-11 respectively.
The [full source audit](research/workbench/evidence/research_program_source_bound_audit_v60.json)
replays all 88 reviews / 352 paths, preserving 19 legacy and 69 funded-v2
reviews. The [preservation proof](research/workbench/evidence/unemployment_guard_20261012_preservation.json)
retains old journal rows, seven profiles, 69 imports, 3,192 controls,
26 selection artifacts, the terminal dividend stop and original W42 plans.

The [completed snapshot](research/workbench/evidence/research_program_unemployment_results_v95.json)
now contains **172 configurations = 84 prior + 88 recurring reviews**, 90
registered versions including two old invalid ones, no pending candidate and
no champion. The fixed old 76-review inference excludes the twelve later trials.
The [native instructions](research/workbench/evidence/prospective_data_automation_v67.json)
continue research in this receiving session with checkpoint recovery, the
unchanged daily/weekly cadence and both frozen zero-return forward models.
**The full investment objective remains unmet.**

## October 13 scheduled run: missing MTUM closing prices

The [actual daily receipt](research/workbench/evidence/research_daily_actual_run_20261013_v1.json)
records a scheduled message received at 01:01:22 UTC and an actual original
pipeline invocation at 01:06:23 UTC. The first `parent_data` stage fails because
the preserved MTUM response has **null close and adjusted-close values for
October 12**. The unchanged source validator independently reproduces that
failure; no price is filled, dropped or substituted.

The successful partial SPY data, original MTUM response and failure records
remain private and hash-bound. No complete new snapshot is accepted, later
original stages are not invoked, and the separate dollar pipeline is not
started. **At that failure checkpoint, both frozen forward models remained at zero observed return sessions**;
their Sharpe/CAGR values remain null. An acquisition attempt is not a new
observation or a strategy result.

The [preservation proof](research/workbench/evidence/research_daily_preservation_20261013_v1.json)
confirms the exact 172-configuration / 88-review historical state, all seven
accepted forward profiles, source/control hashes and terminal dividend stop
are unchanged. The scheduled message and timer readback were observed, but
this does not establish successful source collection, future delivery or
exactly-once execution. The unchanged timer's next observed wakeup is
October 14 around 09:00 Asia/Shanghai. If a session is not genuinely observed
before its next-open deadline, later downloaded history cannot backfill it.

## Approved bounded optimization and actual input recovery

The user confirmed a bounded next round: prioritize the ten-year return gap
versus SPY without changing the unleveraged four-path Sharpe/drawdown/cost
contract. US factor ETFs remain the core; the existing strategy is retained
as a comparison, and individual-stock data preparation stays separate.
The round admits **at most six fixed registered configurations, three batches
of at most two**. Failures remain recorded and exhausting the budget requires
review, not further parameter searching. Historical qualification is only a
candidate milestone; independent prospective observation remains necessary.

The [actual recovery receipt](research/workbench/evidence/research_daily_recovery_20261013_v1.json)
records one new ordinary original-pipeline invocation after the user's explicit
approval. At 07:50:28–07:50:43 UTC, the unchanged collector obtained valid
October 12 inputs and completed all four stages. The separate dollar pipeline
then completed at 07:51:58–07:52:10 UTC. Both acquisitions and observations
precede the genuine October 13 13:30 UTC next-open deadline.
The original missing-price response, partial SPY data and failed invocation
are preserved; no price was filled or substituted, and no endpoint, header,
source contract or validator was changed.

**Both original frozen forward models now contain one observed model session.**
Their identities and September 30 rule targets are unchanged; no new monthly
target was generated on October 12. Independent scalar checks reproduce all
eight first-day strategy/SPY accounts, including funded fees and the extra
stress delay, within USD 1.82e-12. Independent bt replay passed before append.
Sharpe and CAGR remain null until at least 63 actual sessions, and even that
minimum is not proof of long-term alpha. These are research model accounts,
not broker orders.

The [preservation proof](research/workbench/evidence/research_daily_recovery_preservation_20261013_v1.json)
retains every original journal prefix, registration, failed attempt and
historical research outcome. The [native continuation receipt](research/workbench/evidence/prospective_data_automation_v68.json)
preserves the daily/Saturday cadence, original W42 eligibility, source restrictions
and user-approved exploration cap. Historical totals remain
**172 configurations / 88 reviews, no champion**, with **zero** of the six new
configuration slots consumed by this data recovery.

## Bounded round, batch 1: momentum implementation only

The [frozen policy](research/workbench/config/option-momentum-implementation.json)
tests one return-source question inside the unchanged current guarded
option-budget/semiconductor architecture: replace only its MTUM momentum
sleeve with actual PDP, or use a fixed equal MTUM/PDP implementation mix.
Every other desired weight, month-end, dollar guard, option budget, actual
holding band and cost/delay rule remains unchanged.

PDP's already acquired actual fund history covers all 2,805 original context
sessions. Its issuer uses a different relative-strength implementation;
proprietary stock scores and full historical methodology are not reconstructed.
PDP and these fixed mixtures were already exposed in a different earlier
portfolio, so this is a **new current-architecture construction, not new
independent factor information or a holdout**. The previously acquired SPHQ
history is not extended or filled to fit this batch.

The [independent pre-account proof](research/workbench/evidence/option_momentum_20261013_pre_account_verification.json)
reproduces every native PDP price/volume cell, all original twelve-asset values
and risk-free inputs, 131 native option/dollar budgets, both complete desired
matrices and eight causal static seed contexts. Maximum price and target
differences are below 5.7e-14 and 1.12e-16 respectively.
57 coupled cases and 52 root boundary cases pass.

[Actual registration](research/workbench/evidence/option_momentum_20261013_registration.json)
occurred at 2026-10-13 08:56:26 / 08:56:31 UTC.
The [preservation proof](research/workbench/evidence/option_momentum_20261013_registration_preservation.json)
retains every old row and all seven original profiles, including their newly
accepted first observed day, through one exact engine migration.
A real evaluation using the preceding data-recovery published commit stops
before creating any new account output.

The [registration snapshot](research/workbench/evidence/research_program_option_momentum_registered_v96.json)
retains **172 completed configurations / 88 reviews**, 92 registered versions
including two old invalid ones, and exactly these two pending candidates.
**Two of the six approved slots are now registered; four remain.**
No new strategy performance exists at this registration checkpoint; actual
remote publication must precede path-dependent account calculation.

## Completed bounded batch 1: both momentum transfers rejected

Actual remote publication
`c3b26c7b2e1a5c8b3c226763c8b8ea87c92cc38e` and the
09:04:20.914120 UTC durable fence precede every new feedback account.
The [independently checked results](research/workbench/evidence/option_momentum_20261013_results.json)
retain both fixed constructions:

| Momentum implementation | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| PDP full | 10y | Base | 14.5438% | 1.08155 | 13.2660% |
| PDP full | 10y | Stress | 14.2757% | 1.05933 | 13.1131% |
| PDP full | 5y | Base | 18.8829% | 1.15181 | 13.2852% |
| PDP full | 5y | Stress | 18.3503% | 1.11630 | 13.1325% |
| MTUM/PDP equal | 10y | Base | 14.5901% | 1.08513 | 13.4241% |
| MTUM/PDP equal | 10y | Stress | 14.3227% | 1.06289 | 13.2685% |
| MTUM/PDP equal | 5y | Base | 18.9316% | 1.15558 | 13.4465% |
| MTUM/PDP equal | 5y | Stress | 18.3934% | 1.11963 | 13.2912% |

**Both remain rejected.** All eight Sharpe/drawdown gates and both five-year
joint paths pass, but both ten-year SPY comparisons and ten-year stress/base-SPY
comparisons fail. The lower drawdown comes with lower CAGR and Sharpe than
the unchanged current reference, not an improvement in the approved primary
ten-year return objective. No new mixture ratios, band widths, costs or
source windows are tried after these outcomes.

The [reproducibility proof](research/workbench/evidence/option_momentum_20261013_reproducibility.json)
checks 728 actual closing-gap/initial/queue decisions and 30,128 funding days
over 16 funded and 16 independent bt strategy/SPY accounts. Issued-target
error is zero, scalar metric error below 9.33e-15, and wealth error below
USD 1.17e-10. The [full source audit](research/workbench/evidence/research_program_source_bound_audit_v61.json)
replays all 90 reviews / 360 paths, preserving 19 legacy and 71 funded-v2
reviews. The [preservation proof](research/workbench/evidence/option_momentum_20261013_preservation.json)
retains every prior row, control/source hash, original first forward day and
weekly/source-access boundary.

The [completed snapshot](research/workbench/evidence/research_program_option_momentum_results_v97.json)
contains **174 completed configurations / 90 reviews**, 92 registered versions
including two old invalid ones, no pending candidate and no champion.
The round has used **two of six** slots; four remain for separately justified
questions, not further tuning of the failed momentum mixture.
The [actual native continuation](research/workbench/evidence/prospective_data_automation_v69.json)
preserves the same cadence and records this result and remaining budget.
**The full investment objective remains unmet.**

## October 14 scheduled run: missing SPY closing prices

The [actual daily receipt](research/workbench/evidence/research_daily_actual_run_20261014_v1.json)
records the scheduled message at 01:01:03 UTC and the one original-pipeline
invocation at 01:09:03 UTC. Its first `parent_data` stage fails because the
preserved SPY response has **null close and adjusted-close values for
October 13**. The unchanged validator independently rejects that exact response.
The partial raw response and failure receipt remain private and hash-bound;
no price is filled, deleted or substituted, and no automatic retry is made.

No complete new snapshot or model session is accepted. Later original stages,
the separate dollar pipeline and further optimization are not started in this
invocation. Both original forward models retain their one actual October 12
session and null Sharpe/CAGR. The prior MTUM failure and its user-approved
successful normal recovery remain intact; today's different SPY failure does
not erase either event.

The [preservation proof](research/workbench/evidence/research_daily_preservation_20261014_v1.json)
verifies all seven accepted profiles, fourteen registration/head files,
325 prior private files, 69 original imports, 3,192 control artifacts and
26 selection artifacts, with the exact historical ledger and terminal
dividend-source stop unchanged. Totals remain **174 configurations / 90 reviews,
no champion**, and the bounded round remains **two of six slots used, four left**.
The [read-only native receipt](research/workbench/evidence/prospective_data_automation_v70.json)
records the actual timer readback and its next observed wakeup,
October 15 around 09:00 Asia/Shanghai. The binding and prompt were not changed.
An observed wakeup is not source success or a future-delivery guarantee;
missing observations cannot later be backfilled after their acquisition deadline.

## Progress and publication

### Bounded historical backtests, batch 2 registered

The user directed that historical backtests be the main iteration loop rather
than waiting for future observations before doing more research. The original
63-session threshold only governs reporting prospective Sharpe/CAGR; it does
not delay historical research or turn exposed history into independent validation.

The [frozen sector implementation policy](research/workbench/config/option-sector-implementation.json)
replaces only the existing SOXX sleeve with actual XLK or actual QQQ inside the
unchanged guarded option-budget/actual-holding-band architecture. Both are
different industrial compositions already present in the complete cached
twelve-fund, 2,805-session source. Four factor sleeves, GLD/BIL, the original
monthly budgets, dollar guard, cash, costs, delays and formal windows stay fixed.
This is a new construction using exposed information, not a new economic factor
or independent holdout; VIX is not claimed to forecast exact technology-sleeve risk.

The [independent static proof](research/workbench/evidence/option_sector_20261014_pre_account_verification.json)
reproduces all 262 native option inputs, 131 published dollar comparisons, both
desired matrices and eight causal seed contexts within 1.12e-16. A durable
transactional guard counts every registered configuration against the user's
six-slot cap, even without a completed review. 117 coupled cases pass.
[Actual registration](research/workbench/evidence/option_sector_20261014_registration.json)
and the [preservation proof](research/workbench/evidence/option_sector_20261014_registration_preservation.json)
retain all old ledger rows, original one-day models and source restrictions.
The previous dashboard-only commit is rejected before creating account outputs.

At this [registration checkpoint](research/workbench/evidence/research_program_option_sector_registered_v98.json),
**174 configurations / 90 reviews remain complete, 94 versions are registered,
and exactly two new sector configurations are pending**. The bounded round has
used **four of six slots; two remain**. No new portfolio result is claimed before
actual remote publication of these rules and registrations.

### Completed bounded batch 2: both sector transfers rejected

The [completed batch 2 results](research/workbench/evidence/option_sector_20261014_results.json)
retain both sector transfers as rejected:

| Replacement | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| XLK | 10y | Base | 12.6484% | 1.04842 | 13.6905% |
| XLK | 10y | Stress | 12.3711% | 1.02126 | 13.4571% |
| XLK | 5y | Base | 16.0545% | 1.11746 | 13.7096% |
| XLK | 5y | Stress | 15.5095% | 1.07194 | 13.4763% |
| QQQ | 10y | Base | 11.7946% | 0.99570 | 12.8639% |
| QQQ | 10y | Stress | 11.4343% | 0.95947 | 12.6162% |
| QQQ | 5y | Base | 14.8141% | 1.05726 | 12.8826% |
| QQQ | 5y | Stress | 14.1441% | 0.99825 | 12.6351% |

Both ten-year benchmark comparisons fail. QQQ also fails both ten-year Sharpe
gates and the five-year stress Sharpe gate; values close to 1 are not rounded
into passes. XLK's five-year paths pass all gates, but this does not override
the ten-year failures. Both returns are below the unchanged reference; neither
replaces it and no sector share, band or cost is refitted after outcomes.

Actual remote publication `a292a41bf559f867fe9acdf663bae2314eceb59c`
and its 12:51:53.936388 UTC fence precede all accounts.
The [independent account proof](research/workbench/evidence/option_sector_20261014_reproducibility.json)
checks 728 holding decisions and 30,128 funding days across 16 funded and
16 bt strategy/SPY accounts: issued-target error zero, scalar metric error
below 5.16e-14 and wealth error below USD 7.28e-11.
The [complete source audit](research/workbench/evidence/research_program_source_bound_audit_v62.json)
replays all **92 reviews / 368 paths**, retaining 19 original and 73 funded-v2
reviews with every old record and original model unchanged.
Current totals are **176 configurations / 92 reviews, no champion**.
Four of six bounded slots are used; two remain for the final distinct question.

### Final bounded batch 3: Treasury packages registered

The [final batch 3 policy](research/workbench/config/treasury-defense-implementation.json)
compares actual IEF and TLT nominal Treasury defensive packages.
The equity composition stays half SOXX and half equally shared across the four
factor ETFs. The original 63-session realized sleeve-volatility helper, with
explicit local input-role aliases, sets the stock/Treasury budget within the
unchanged 30%–70% bounds. No market price or genuine ticker is overwritten.
The dollar half guard, BIL, funding, monthly clock, holding band, costs and
delays remain unchanged. **GVZ is not used to forecast Treasury risk.**
Both asset and risk proxy differ from the gold/option package, so this is not
an isolated causal asset-alpha comparison or an independent holdout.

The [independent static proof](research/workbench/evidence/treasury_defense_20261014_pre_account_verification.json)
checks all 33,660 cached close-price cells, 131 scalar realized-risk budgets
per candidate, 131 actual H10 pairs and eight causal seed contexts, with
maximum target error below 6.67e-16. 103 coupled cases pass.
[Actual registration](research/workbench/evidence/treasury_defense_20261014_registration.json)
and [preservation](research/workbench/evidence/treasury_defense_20261014_registration_preservation.json)
retain all prior results and original one-day models.
The preceding result commit is rejected before new account output.
At the [preregistered checkpoint](research/workbench/evidence/research_program_treasury_defense_registered_v100.json),
**176 configurations / 92 reviews remain complete, 96 versions are registered,
and only the two Treasury packages are pending**.
All **six of six slots are now consumed**; after their outcomes the round
requires review, not more parameter trials.

### Completed six-slot round: no qualified champion

The [final Treasury outcomes](research/workbench/evidence/treasury_defense_20261014_results.json)
also remain rejected:

| Defensive package | Window | Costs | CAGR | Net excess Sharpe | Maximum drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| IEF | 10y | Base | 6.5232% | 0.67779 | 13.0708% |
| IEF | 10y | Stress | 6.2212% | 0.63047 | 13.2448% |
| IEF | 5y | Base | 5.9644% | 0.31658 | 13.1844% |
| IEF | 5y | Stress | 5.4966% | 0.25311 | 13.3198% |
| TLT | 10y | Base | 6.8021% | 0.51876 | 19.8192% |
| TLT | 10y | Stress | 6.6478% | 0.50135 | 19.6438% |
| TLT | 5y | Base | 4.5818% | 0.11441 | 19.9149% |
| TLT | 5y | Stress | 4.0876% | 0.06645 | 19.7790% |

IEF passes the drawdown gates but fails every Sharpe and SPY comparison.
TLT also fails all drawdown gates; nominal Treasury duration is not a
guaranteed defensive asset. Actual publication
`780e047004cef9e87539bfed5f7f890e9d115392` and the
14:08:24.498419 UTC fence precede all eight paths.
[Independent account checks](research/workbench/evidence/treasury_defense_20261014_reproducibility.json)
retain 728 holding decisions / 30,128 funding days / 16 funded and 16 bt
accounts, with zero target error, scalar error below 7.93e-14 and wealth error
below USD 6.55e-11. The [final source audit](research/workbench/evidence/research_program_source_bound_audit_v63.json)
replays **94 reviews / 376 paths**, preserving all old records and original
one-day models.

The [six-configuration round review](research/workbench/evidence/bounded_optimization_round_20261014_review.json)
retains every outcome across all three batches: **six used, six completed,
zero qualified, zero slots remaining**. None improves the original reference's
ten-year base CAGR. This round checked 24 strategy paths, 2,184 actual holding
decisions and 90,384 funding days; it is not six independent alpha discoveries.
Further same-round parameter searching is stopped, and a new round needs a
new explicit budget. Current totals are **178 configurations / 94 reviews**,
96 registered versions including two old terminal failures, no pending
candidate and no champion. Historical research was completed without waiting
for future sessions; no retrospective pass is labelled forward validation.

The [actual native continuation receipt](research/workbench/evidence/prospective_data_automation_v71.json)
records the user-directed backtest priority and exhausted six-slot review.
The prompt alone changed at 14:32:14.176 UTC and was read back at
14:33:01.138 UTC; daily 09:00 Shanghai, Saturday discovery and the already
scheduled next wakeup remain unchanged. This is not proof of future delivery
or successful source collection. The failed October 13 observation was not
actually acquired before the October 14 next-open deadline; it is not
retrospectively filled and neither original model is reset.

### October 15 scheduled run: SPY HTTP access denial

The [actual scheduled attempt](research/workbench/evidence/research_daily_actual_run_20261015_v1.json)
started at 01:06:21 UTC to observe the completed October 14 market session.
The unchanged collector reported **SPY HTTP 403** at the first `parent_data`
stage; no later stage or separate Dollar pipeline ran. The one failed attempt
was retained without retrying, changing provider/headers, filling prices or
backfilling the missing October 13 observation.

This is an access failure, not another missing-price validation result.
The original collector did not retain the denied HTTP body or headers;
offline verification checks its saved exception and failure receipts, not
an independently repeated network rejection. The
[preservation receipt](research/workbench/evidence/research_daily_preservation_20261015_v1.json)
confirms all seven accepted profiles, fourteen registration/head hashes,
327 prior archive files, 69 imports, 3,192 controls and 26 selection artifacts
remain unchanged. Both original models still have one actual October 12
return day and unreported Sharpe/CAGR.

The [native readback](research/workbench/evidence/prospective_data_automation_v72.json)
matches the existing prompt without modifying its binding. It observed the
next wakeup at October 16 01:00:09 UTC, not a guarantee of future delivery or
source success. Research remains **178 completed configurations / 94 reviews,
no champion and zero slots remaining**; no new strategy trial was started.

### October 16 scheduled run: access failure retained

The [next day's actual scheduled attempt](research/workbench/evidence/research_daily_actual_run_20261016_v1.json)
started at 01:05:34 UTC for the completed October 15 market session.
The original first SPY request again reported HTTP 403. Only `parent_data`
started; no later stage, Dollar invocation, retry or alternate-source request
followed. No quote payload was accepted, and the denied HTTP body/headers
were not retained or independently requested again.

[Offline preservation](research/workbench/evidence/research_daily_preservation_20261016_v1.json)
checks all seven accepted profiles, fourteen registration/head hashes,
328 prior archive files, 69 imports, 3,192 controls and 26 selection artifacts.
Both original models retain one October 12 return day and null Sharpe/CAGR;
neither missing October 13 nor October 14 observation is backfilled or reset
away. There are zero new strategy trials or accepted observations.

The [unchanged native binding readback](research/workbench/evidence/prospective_data_automation_v73.json)
observed the next wakeup at October 17 01:00:24 UTC. The existing four W42
plans and two-definition quota still require actual October 17 Shanghai
09:00 eligibility and their original publication gates; no early registration
occurred. The exhausted six-configuration round remains stopped, and the
overall investment objective remains unmet.

### October 17: daily access failure and W42 eligibility

The [actual Saturday daily attempt](research/workbench/evidence/research_daily_actual_run_20261017_v1.json)
started at 01:08:24 UTC for the completed October 16 market session.
The first original SPY request again reported HTTP 403; no later daily stage,
Dollar invocation, retry, alternate-source request or accepted observation
followed. Neither original forward model was reset or advanced, and the
missed October 13-15 observations remain unfilled.

The [W42 gate snapshot](research/workbench/evidence/weekly_research_2026-W42_gate_status_20261017.json)
separates **time eligibility reached** from actual research execution.
This invocation stopped later work after the daily first failure: no weekly
preparation, factor cycle, definition/candidate admission, registration
packet or portfolio evaluation ran. The original four candidates and
two-definition quota remain pending, not cancelled or replaced by a new
optimization round. The dated three-group readiness files and actual cycle
are absent; time eligibility alone cannot substitute for their original
registration, approval and remote-publication gates.

[Offline preservation](research/workbench/evidence/research_daily_preservation_20261017_v1.json)
checks seven accepted profiles, fourteen registration/head hashes,
329 prior archive files, 69 imports, 3,192 controls, 26 selection artifacts
and both original early-blocked weekly CLI receipts. The
[native readback](research/workbench/evidence/prospective_data_automation_v74.json)
matches the unchanged prompt and observed October 18 01:01:53 UTC as the next
wakeup; it is not a source-success or delivery guarantee. Research remains
178 completed configurations / 94 reviews, no champion, zero new trials and
zero remaining slots in the completed six-configuration optimization round.

### October 18: no new session, no automatic source retry

The [Sunday preflight](research/workbench/evidence/research_daily_preflight_20261018_v1.json)
independently confirms that the latest completed NYSE session is still
October 16, with the next actual open on October 19 at 13:30 UTC.
That is the same session as the preserved October 17 SPY HTTP 403 attempt.
This wakeup therefore made **zero primary, Dollar, collector or weekly
invocations**, rather than automatically retrying the failed session.

This is explicitly `not_started_same_session_prior_failure_no_auto_retry`,
not `already_collected`, source recovery, another network failure or a
successful pipeline. The last actual pipeline receipt and dashboard
operation-history entry remain October 17; no October 18 actual-run receipt
or fake pipeline start time is created.

[Preservation](research/workbench/evidence/research_daily_preservation_20261018_v1.json)
checks seven unchanged profiles, fourteen anchors, 330 prior archive files,
69 imports, 3,192 controls and 26 selection artifacts. Both original models
still have one October 12 return day and null Sharpe/CAGR. The original W42
four-candidate/two-definition plan is still pending its full admission gates;
there is no new strategy trial, observation or qualification.

The [native readback](research/workbench/evidence/prospective_data_automation_v75.json)
records the genuine Sunday wakeup separately from the last actual pipeline,
without changing its binding. The next observed wakeup is October 19
01:00:09 UTC; this is not a source-success or delivery guarantee.

### October 20: genuinely new session, one actual failed attempt

The [actual October 20 pipeline](research/workbench/evidence/research_daily_actual_run_20261020_v1.json)
started at 01:26:44 UTC for the newly completed October 19 NYSE session.
Independent calendar selection confirms this is newer than the October 16
failed batch, unlike the preserved October 18-19 same-session no-retry
preflights. The one unchanged original SPY request again reported HTTP 403;
only `parent_data` started, and no later step, Dollar invocation, retry,
source/environment change or weekly rollover followed.

[Offline preservation](research/workbench/evidence/research_daily_preservation_20261020_v1.json)
verifies seven original profiles, fourteen anchors, 330 prior archive files,
69 imports, 3,192 controls and 26 selection artifacts. No input snapshot or
return day was accepted; neither original one-day model was reset or
advanced. The missed October 13-16 observations remain unfilled.
The denied HTTP body/headers were not retained, and verification did not
reissue the request.

The [native readback](research/workbench/evidence/prospective_data_automation_v76.json)
separates this real new-session attempt from the old no-attempt preflight.
The next observed wakeup is October 21 01:00:31 UTC, without changing the
binding or guaranteeing future source success. The actual operation history
now contains eight records, including six failures; preflight-only wakeups
are not counted as pipeline runs. Research remains 178 configurations /
94 reviews, no champion and no remaining optimization slots. The original
W42 four-candidate/two-definition plan stays pending its full gates, not
silently re-dated or replaced.

### October 21: original new-session request still denied

The [one actual scheduled attempt](research/workbench/evidence/research_daily_actual_run_20261021_v1.json)
ran at 01:07:24-25 UTC for the newly completed October 20 session.
The original first SPY request again reported HTTP 403; later daily steps,
Dollar and weekly work did not start. There was no retry, source/environment
change, old-session backfill or new accepted observation.
[Offline preservation](research/workbench/evidence/research_daily_preservation_20261021_v1.json)
checks all seven profiles, fourteen anchors, 331 prior archive files,
69 imports, 3,192 controls and 26 selection artifacts; both original models
remain one October 12 day with null Sharpe/CAGR. The missed October 19
observation is now also past its original acquisition deadline and is not
retroactively recorded.

The [unchanged native binding readback](research/workbench/evidence/prospective_data_automation_v77.json)
observed the next wakeup at October 22 01:01:38 UTC, not a source-success
guarantee. Actual operation history now has nine records / seven failures;
historical research remains 178 configurations / 94 reviews, no champion,
zero remaining optimization slots, and the original W42 plan remains pending.

### October 22: new-session source failure preserved

The [one original scheduled pipeline](research/workbench/evidence/research_daily_actual_run_20261022_v1.json)
ran at 01:09:06-07 UTC for the newly completed October 21 session.
The first SPY request again reported HTTP 403, stopping all later daily,
Dollar and weekly work. No retry, provider/environment change, historical
backfill or accepted observation occurred.
[Offline preservation](research/workbench/evidence/research_daily_preservation_20261022_v1.json)
checks seven profiles, fourteen anchors, 332 prior archive files, 69 imports,
3,192 controls and 26 selection artifacts. Original models remain one
October 12 return day with null Sharpe/CAGR; missed October 20 and earlier
observations are not retroactively created.

The [native readback](research/workbench/evidence/prospective_data_automation_v78.json)
matches the unchanged prompt and observed October 23 01:01:43 UTC as the next
wakeup, not a future source-success guarantee. Actual operation history is
ten records / eight failures, not ten research trials. Historical research
remains 178 configurations / 94 reviews, no champion, zero remaining
optimization slots and the original W42 plan pending its complete gates.

### October 23: original source gate remains blocked

The [one actual scheduled attempt](research/workbench/evidence/research_daily_actual_run_20261023_v1.json)
ran at 01:06:14-15 UTC for the newly completed October 22 session; the
original first SPY request reported HTTP 403 and stopped later daily,
Dollar and weekly work without retry, source changes or accepted data.
[Offline preservation](research/workbench/evidence/research_daily_preservation_20261023_v1.json)
checks seven profiles, fourteen anchors, 333 prior archive files, 69 imports,
3,192 controls and 26 selection artifacts. Both original models remain
one October 12 day with null Sharpe/CAGR; the missed October 21 observation
is past its original deadline and is not backfilled.
The [native readback](research/workbench/evidence/prospective_data_automation_v79.json)
matches the unchanged binding/prompt and observed October 24 01:00:49 UTC as
the next wakeup, not a source-success guarantee. Eleven actual operations /
nine failures are not additional strategy trials: research remains
178 configurations / 94 reviews, no champion, zero optimization slots,
and the original W42 plan pending all its original gates.

### October 24: original source gate remains blocked

The [one actual scheduled attempt](research/workbench/evidence/research_daily_actual_run_20261024_v1.json)
ran at 01:05:37-38 UTC for the newly completed October 23 session; the
original first SPY request again reported HTTP 403. Later daily, Dollar and
weekly work did not start; no retry, source change, backfill or accepted
observation followed.
[Offline preservation](research/workbench/evidence/research_daily_preservation_20261024_v1.json)
verifies seven profiles, fourteen anchors, 334 prior archive files, 69 imports,
3,192 controls and 26 selection artifacts. Both original models remain
one October 12 day with null Sharpe/CAGR; the missed October 22 observation
is past its original deadline and is not backfilled.
The [native readback](research/workbench/evidence/prospective_data_automation_v80.json)
matches the unchanged binding/prompt and observed October 25 01:00:20 UTC as
the next wakeup, not a source-success guarantee. Twelve actual operations /
ten failures are not new strategy trials: research remains 178 configurations /
94 reviews, no champion and zero optimization slots. The original W42 plan
remains pending, not rolled forward or executed after the daily failure.
The October 24 dividend time gate does not permit retrying terminal VIG/SDY sources.

### Local research dashboard

Run from this worktree, without installing a frontend framework:

```bash
research/workbench/.venv/bin/python -I -B src/us_quant/research_dashboard.py --check
research/workbench/.venv/bin/python -I -B src/us_quant/research_dashboard.py --port 8787
```

Open `http://127.0.0.1:8787`. The Chinese dashboard includes the factor catalog
and queued definitions, searchable strategy history, per-strategy four-path
metrics and failures, a research timeline, the original prospective models,
and the current low-drawdown research leader. The leader is ranked among
complete unleveraged recurring reviews with every drawdown at most 15%, using
the lowest net-excess Sharpe across all four paths. It is explicitly **not a
qualified champion or live recommendation**.

The 84 earlier configurations retain their original protocols and values,
separate from the recurring reviews and two terminal code failures. The
unchanged refinement baseline is a comparison, not an 85th new configuration.
Archived technical failures do not inherit the current corrected policy.

The server binds only to loopback and serves a finite set of frontend assets
and allowlisted published JSON; it never serves the repository directory,
private SQLite databases, source quotes, target matrices, broker state or
account rows. Refresh reads the current progress pointer and published
evidence, not market-data providers. It checks counts, frozen active policies,
budgets and strict unrounded gates; inconsistent or missing evidence produces
an explicit error instead of an empty or success-shaped result.
Each later reviewed research checkpoint updates the existing
`docs/research_progress.json` pointers and evidence; the dashboard automatically
reads those updates on refresh (once per minute while visible).

See [integration status](docs/integration/STATUS.md) and
[`docs/research_progress.json`](docs/research_progress.json). GitHub progress
uses reviewed explicit commits/PRs, not an unconditional git-push cron job.
The scheduled agent may create checked data/research checkpoints; the
standalone Slack job cannot run git or publish files.

```bash
.venv/bin/python -B scripts/check_publication.py
# Stage only the listed, reviewed paths; never git add .
.venv/bin/python -B scripts/check_publication.py --staged
```

The publication allowlist excludes raw artifacts, environments, account
configuration, holdings/importer modules and the original private handoff.
