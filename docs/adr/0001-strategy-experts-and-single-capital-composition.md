# Compose strategy experts without rewriting their histories

The user changed the research mainline to a strategy expert pool and a rolling
meta-strategy on 2026-10-29. We separate expert validity from final portfolio
qualification, expose preserved recommendations through a small Interface at a
read-only Seam, and keep shadow assessment separate from one main capital account:
rerunning completed strategy entry points or summing their funded wealth would
rewrite evidence or misstate the capital and costs of the final portfolio.

The pool is selected after historical research has already been exposed. Causal
rolling scores do not remove that selection bias or establish independent forward
performance.
