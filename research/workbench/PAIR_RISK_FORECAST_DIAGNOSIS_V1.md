# Stored pair risk forecast diagnosis V1

This one fixed read-only diagnostic asks whether previously recorded training
source-pair covariance/risk contributions persist into the next fully matured
exchange month. It does not repair hierarchical allocation,the rejected
eight-expert rule,RF views,capital seeds or old numerical failures.

All121old decision states and all28pair records at each of114mature origins
are retained. Seven old immature origins are explicitly unscored. Pair weights,
correlation and marginal training risk come from the original audit,without
rerunning covariance/score/selection/target/account pipelines.
Summing the stored marginal fields is a new scalar diagnostic derivation,
not recovered unsaved original forecast memory.

Future sessions begin strictly after a genuine exchange-month end and end at
the following genuine exchange-month end,not a fixed21-row slice. The final
2026-09-30origin requires2026-10-30but original source books end2026-10-05.
Onlythreefuture sessions are available;all28final pair forecasts are retained
as incomplete and receive no risk evaluation or fill. An immutable pre-review
calendar clarification corrects the original design's unverified five-day
anticipation without changing this full-month requirement or original design.

The new statistical fixed-weight source NET-minus-RF mixture is not a funded
main account,tradeable counterfactual,added source wealth,CAGR/DD strategy or
transferred risk guarantee. Source shadow capital,feedback and costs differ
from actual unified main execution. Only covariance/variance/correlation and
relative marginal-risk mismatch are evaluated after complete future maturity.

NumPy and independent scalar fsum variance/covariance agreement are checked.
Exact constant future portfolios have undefined variance ratios and no safe
success claim;there is no variance floor or source normalization.
Every exact binary64 future source input is retained before its new
calculation,and resolved diagnostic fields afterward. Partial/failing records
are preserved instead of repeating or tuning the diagnostic.

All pairs and original selected/nonselected flags are compared descriptively.
Median,p90,max andratio-above-one counts are not confidence bounds,pvalues,
independent samples,main PNL attribution or a chosen new risk cap/window/rule.
The source expert introductions are historical counterfactuals and full
histories are exposed;this is not independent OOS or real forwardproof.
