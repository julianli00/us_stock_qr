# Lagged quarterly credit-relative input, V1

This source-consumer study is not a complete strategy, a new traded asset,
an admitted expert, or a portfolio performance trial. It reuses the immutable
HYG/LQD data-preparation V2 manifest and preserves the failed V1 serialization
and unfrozen V2 preflight evidence. No quote or issuer request is repeated.

The fixed input uses 63 NYSE price intervals ending at the previous NYSE
session. Each fund's adjusted close is divided by its own starting adjusted
close; the HYG gross change is then divided by the LQD gross change and one is
subtracted. A quarter is selected as an economic credit-conditions horizon,
not by inspecting resulting features or portfolio returns. There is no window
or lag search, fitted coefficient, state threshold, target weight, or portfolio.

The small Module Interface is `LaggedQuarterCreditInput.observe(session)`.
It accepts a complete ordered float64 HYG/LQD adjusted-close lattice and a real
timezone-aware consumer-introduction timestamp. It retains all source sessions,
returns explicit unavailable observations during the first 64 session labels,
and rejects malformed or uncovered inputs rather than filling or repairing them.
The first mature observation uses 64 preceding price observations and no price
from the observation's own session. The source endpoints are dates, not claims
about historical quote arrival. Introduction is recorded at the actual freeze;
historical observations are explicitly counterfactual research.

The admission review checks synthetic malformed inputs, cold history,
normalization, caller-state isolation, and exact causal-prefix equivalence.
Its actual review preserves every one of the 2805 source session labels,
checks five preregistered prefixes (2016-10-05, 2018-09-28, 2020-03-31,
2022-12-30, 2026-10-05), perturbs same-session and future prices without
changing earlier observations, and checks independent power-of-two price-unit
changes. These are input-construction checks, not return forecasts or economic
performance evidence. The first actual input calculation follows the frozen
implementation, test evidence, inputs, and review intent.

Any admission is restricted to this archival, lagged, within-fund-normalized
relative-price input. Cached quotes were acquired on 2026-10-06; issuer pages
were acquired on 2026-11-04. Neither is backdated. Current issuer identity does
not prove historical constituents, a complete methodology chain, historical
intraday publication, revision-free data, or licensing beyond the existing
private research scope. A one-session date lag does not establish those missing
facts. No broader PIT or source-quality gate is waived.

This is not a pure credit spread. Fund duration, interest rates, liquidity,
distributions and historical index changes can all matter. Price-unit
normalization does not remove these economic confounders. A subsequent complete
credit policy still needs its own economic rationale, law, availability and
prefix checks, real registration and publication fences, four fresh funded
paths, independent accounting, and all unchanged financial gates.
