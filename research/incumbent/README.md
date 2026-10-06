# Preserved incumbent strategy code

This is a runnable, credential-free research slice, not just a description of
the old strategy. Its package remains `src.us_quant` in the incumbent Python
3.9 environment, distinct from the imported workbench's `us_quant` package.

- `src/us_quant/incumbent_top30.py` extracts four pure weight-construction
  functions from `scripts/run_all_stock_pool_iteration.py`. Their ASTs are
  identical to the inspected original. It also exposes the selected execution
  assumptions without connecting to any broker.
- `config/incumbent_top30.json` preserves the saved 699-stock configuration
  byte-for-byte. This is a research universe, not user holdings.
- `src/us_quant/us_realistic_backtest_engine.py` preserves the original daily
  next-open simulation/cost engine byte-for-byte.
- `scripts/run_early_accumulation_signal_v1.py` preserves the distinct early
  scanner byte-for-byte. Its dependencies are numpy, pandas and `paths.py`;
  real execution additionally requires local, appropriately sourced research
  bars/universe and optional event/V123 artifacts. It is **not** invoked by
  the reporter or scheduler.

`config/incumbent_source_provenance.json` records original file/function
hashes. Synthetic fixtures were evaluated against the original four pure
functions without importing or executing its surrounding search pipeline.
The guards compare the extracted output to those frozen reference outputs.

```bash
.venv/bin/python -B scripts/test_incumbent_baseline_guard.py
```

For a new environment, install `research/incumbent/requirements.txt`; parquet
support is needed for the early scanner's explicit local-data reads.

The Top30 API is `load_config()`, `build_all_stock_weights(prices, config,
all_trade_cols)`, `execution_config()`, and the preserved engine's
`simulate_daily_target_weights(bars, weights, asset_types, config)`.
No network, broker or parameter-search entrypoint is needed to use this slice
on explicit research inputs.

This baseline deliberately retains last-available-row monthly rebalancing,
the partial-month tail, missing-VIX pass-through, static-universe assumptions,
and rebalance-only risk checks. They are limitations, not silently repaired
rules under the old ID. The early scanner's original missing upper event-date
bound and date-only external fields are also retained as historical source:
its output is **not trusted by the canonical reporter**. The new reporter
requires a separate hashed, point-in-time watch contract.

The full 12-configuration selection/reporting pipeline is not imported.
Its old plotting/annual-consistency/excess helpers and unrelated research
iterations remain local (`run_user_ibkr_no_leverage_iteration.py`,
`run_all_stock_annual_consistency_iteration.py`,
`run_leveraged_excess_iteration.py`, and the full orchestration in
`run_all_stock_pool_iteration.py`). They are not dependencies of this pure
slice. Private holdings/FIFO/importer modules are intentionally excluded.
No claim is made that the whole historical application or saved performance
has been independently reproduced.
