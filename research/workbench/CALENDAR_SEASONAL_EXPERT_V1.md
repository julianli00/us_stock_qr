# Fixed calendar seasonal expert V1

This is one complete source-policy preparation for a complementary strategy
expert, not a new expert pool, technical admission or qualified portfolio.

At every genuine NYSE month end, after the existing admitted 16:15 ET information
clock, inspect the next calendar month. For November through April, recommend
equal capital in MTUM, VLUE, QUAL, USMV, IJR and PKW. For May through October,
recommend actual BIL. Both states invest 98% in ETFs and retain 2% idle cash.
April's opinion changes to BIL and October's opinion changes to the six-family
core. The original one/two-session later-open delay and actual 2.5 percentage-point
holding band must remain; this is not execution exactly at a month boundary.

The hypothesis is calendar-dependent equity participation, rather than price
trend, hedge covariance, option uncertainty, credit information or recent
expert-return ranking. Different allocation behavior does not establish an
independent economic risk source. The core holds the same existing equity ETFs;
winter drawdowns, loss of summer gains, BIL risk, delayed execution and costs
can defeat the hypothesis.

## Module and Interface

`calendar_seasonal_expert_v1.build_targets` constructs all native month-end
opinions and retains each calendar input before its resolved complete target.
It reuses the existing fourteen-asset market validator. Missing, nonpositive,
reordered or incomplete market inputs fail; BIL is a declared state, not a
success-shaped missing-data fallback.

`decision` needs a normalized genuine month-end timestamp, not prices or future
returns. Calendar availability therefore has no price-return warmup. This does
not make a NET assessment mature. The calendar must also cover the following
session; an unclassified holiday or its last known date is not guessed to be a
month end. At the existing `StrategyExpert.observe` Seam,
`ArchivedMonthlyExpert` carries the last announced seasonal opinion, whereas
`ArchivedDenseExpert` can expose a materially different daily source opinion.
Their synthetic Interface checks do not substitute for actual source books or
technical admission.

No historical source targets, account outputs or performance metrics are
computed in preparation. The prehistory, original four funded paths, initial
capital/peak, RF, costs, delays, cash funding and independent accounting must be
frozen and registered in a new financial implementation before financial
computation. Actual publication fences remain mandatory. A successful complete
expert review would still require a separate deliberate pool-inclusion study.

## Evidence and limitations

The November-April seasonal-participation hypothesis is fixed once, without a
month, cutoff, factor-weight, band, cost or execution search. A literature
reference motivates this hypothesis but does not establish its truth in this
market, replicate the referenced study, prove economic causality or admit data.

Bouman and Jacobsen (2002), *The Halloween Indicator, "Sell in May and Go Away":
Another Puzzle*, American Economic Review 92(5), 1618-1635,
[DOI 10.1257/000282802762024683](https://doi.org/10.1257/000282802762024683),
is the bibliographic motivation. The retained normal Crossref response verifies
that reference's identity only; the full article was not read and its method or
results were not independently replicated.

All 255 completed configurations are exposed. Native historical opinions assume
counterfactual availability of the newly defined policy. Actual definition,
registration and admission are never backdated. Prefix equality and future-price
perturbation checks are causality diagnostics, not independent out-of-sample or
prospective evidence.

The market freeze is reused only as an input contract. Its inherited completion
count 228 is stale, as recorded in the immutable prior clarification. This
preparation starts from authoritative J922 and 255 completed configurations.
It changes no old files, financial gates, global rosters or source-admission
scope and authorizes no broker action.
