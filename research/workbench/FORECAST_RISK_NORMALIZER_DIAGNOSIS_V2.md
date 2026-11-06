# Exact source scalar-key Adapter V2

V1 is a terminal partial diagnosis, not a financial failure or completed sign
survey. It preserved17input/result lines and stopped on strict key validation.
The source score is a NumPy float64 scalar produced by matrix arithmetic.
Its `round(score,12)` protocol can differ from `round(Python_float,12)` after
the same binary64 score is read from JSON. V1's synthetic plain-float examples
missed that source-type Interface fact.

The private first counterexample proves that the recorded source key agrees
bit-for-bit with both original NumPy rounding and an independently evaluated
scaled-binary64 nearest-even scalar calculation. V1's freeze, code, tests,
partial records and failure remain immutable. No source score, key, ranking,
target, account or financial result is corrected or rewritten.

## Narrow technical continuation

V2's Adapter restores only the known scalar-key protocol. It converts the
recorded binary64 score back to NumPy float64 for the original12decimal
rounding and requires exact `float.hex` equality with the recorded key.
Independently, it multiplies the binary64 score by10^12, applies scalar
nearest-even integer rounding, divides by10^12 and preserves signed zero.
This second route must also match **bit-exactly**. Overflow or mismatch stops;
there is no approximate key tolerance or fallback.

The frozen V1 sign identity, all four50-digit Decimal score checks, original
1e-10score agreement tolerance and aggregation remain unchanged. A locally
scoped function Adapter supplies the exact source-key reader; it does not
mutate the frozen V1 module or Python/NumPy global rounding behavior.
Both synthetic and saved-record inputs use the same Interface.

The V2 survey reports all numerator/risk-slope counts plus how often the
incorrect JSON/Python rounding assumption would disagree with actual saved
keys. It never uses Python's alternative keys to choose a winner. Existing
selected pairs remain only observed records, not reselected alternatives.

This is the same bounded mechanical hypothesis, not a new independent
economic research question. All original sign/derivative limits remain:
continuous unrounded risk slope at fixed numerator is not discrete ranking,
future risk, actual holdings, main drawdown attribution or a safety guarantee.
No new targets, funds, financial trials, source request or RF view follows.
