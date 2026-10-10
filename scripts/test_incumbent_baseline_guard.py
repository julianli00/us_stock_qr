from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant import incumbent_top30 as baseline  # noqa: E402
from src.us_quant.us_realistic_backtest_engine import simulate_daily_target_weights  # noqa: E402


def synthetic_case(name: str) -> tuple[pd.DataFrame, dict, list[str]]:
    dates = pd.bdate_range("2023-01-02", "2024-07-12")
    elapsed = np.arange(len(dates), dtype=float)
    equities = [f"T{number:03d}" for number in range(40)]
    prices = pd.DataFrame({
        ticker: 40.0 * np.exp((0.0007 + number * 0.00004) * elapsed + 0.01 * np.sin(elapsed / (7 + number / 8) + number))
        for number, ticker in enumerate(equities)
    }, index=dates)
    prices["SPY"] = 100.0 * np.exp(0.0005 * elapsed)
    prices["QQQ"] = 100.0 * np.exp(0.0008 * elapsed)
    prices["BIL"] = 100.0 * np.exp(0.00002 * elapsed)
    prices["^VIX"] = 20.0
    if name == "missing_vix":
        prices = prices.drop(columns="^VIX")
    elif name == "high_vix":
        prices["^VIX"] = 36.0
    elif name == "price_and_history_filters":
        prices["T039"] = 4.0
        prices.loc[dates[:-70], "T038"] = np.nan
    elif name == "drawdown_filter":
        prices.loc[dates[-80:], "QQQ"] *= np.linspace(1.0, 0.5, 80)
    elif name != "base":
        raise ValueError("Unknown synthetic case")
    config = baseline.load_config()
    config["equities"] = equities
    return prices, config, [*equities, "SPY", "QQQ", "BIL"]


def weights_hash(weights: pd.DataFrame) -> str:
    payload = {
        "dates": weights.index.strftime("%Y-%m-%d").tolist(),
        "columns": weights.columns.tolist(),
        "weights": weights.round(8).to_numpy().tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class IncumbentBaselineTests(unittest.TestCase):
    def test_extracted_ast_and_unmodified_sources_match_frozen_provenance(self) -> None:
        provenance = json.loads((ROOT / "config/incumbent_source_provenance.json").read_text())
        tree = ast.parse((ROOT / "src/us_quant/incumbent_top30.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and hasattr(node, "type_params"):
                node._fields = tuple(field for field in node._fields if field != "type_params")
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name, digest in provenance["original_function_ast_sha256"].items():
            actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, digest, name)
        for name, digest in provenance["byte_preserved_files"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_synthetic_results_equal_original_reference_outputs(self) -> None:
        fixture = json.loads((ROOT / "tests/fixtures/incumbent_equivalence.json").read_text())
        for name, expected in fixture["case_output_sha256"].items():
            with self.subTest(case=name):
                prices, config, columns = synthetic_case(name)
                original_prices = prices.copy(deep=True)
                original_config = json.dumps(config, sort_keys=True)
                weights = baseline.build_all_stock_weights(prices, config, columns)
                self.assertEqual(weights_hash(weights), expected)
                pd.testing.assert_frame_equal(prices, original_prices)
                self.assertEqual(json.dumps(config, sort_keys=True), original_config)
                self.assertTrue((weights.sum(axis=1) <= 1.0 + 1e-12).all())

    def test_saved_identity_and_699_stock_configuration(self) -> None:
        config = baseline.load_config()
        self.assertEqual(config["version_name"], baseline.STRATEGY_ID)
        self.assertEqual(len(config["equities"]), 699)
        self.assertEqual(config["lookback"], 126)
        self.assertEqual(config["top_n"], 30)
        self.assertEqual(config["weighting"], "inverse_vol")
        self.assertFalse(config["allow_margin"])

    def test_partial_month_rebalance_quirk_is_preserved(self) -> None:
        prices, _, _ = synthetic_case("base")
        self.assertEqual(baseline.rebalance_dates(prices, "M")[-1], pd.Timestamp("2024-07-12"))

    def test_missing_vix_passes_and_high_vix_goes_to_bil(self) -> None:
        frames = {}
        for name in ("base", "missing_vix", "high_vix"):
            prices, config, columns = synthetic_case(name)
            frames[name] = baseline.build_all_stock_weights(prices, config, columns)
        pd.testing.assert_frame_equal(frames["base"], frames["missing_vix"])
        self.assertEqual(frames["high_vix"].iloc[-1]["BIL"], 1.0)
        self.assertEqual(int((frames["base"].iloc[-1].loc[[f"T{i:03d}" for i in range(40)]] > 0).sum()), 30)

    def test_engine_is_runnable_next_open_costed_simulation_only(self) -> None:
        dates = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])
        bars = pd.DataFrame({
            "date": dates, "ticker": "SYNTHETIC", "adj_open": 100.0,
            "adj_close": 100.0, "adj_high": 101.0, "adj_low": 99.0, "volume": 1_000_000.0,
        })
        weights = pd.DataFrame({"SYNTHETIC": [0.5, 0.5, 0.0, 0.0]}, index=dates)
        original = bars.copy(deep=True)
        result = simulate_daily_target_weights(
            bars, weights, asset_types={"SYNTHETIC": "equity"}, config=baseline.execution_config(),
        )
        orders = result["orders"]
        self.assertFalse(orders.empty)
        self.assertEqual(pd.to_datetime(orders["submit_time"]).min(), dates[1])
        self.assertGreater(float(orders["commission"].sum()), 0)
        pd.testing.assert_frame_equal(bars, original)

    def test_early_scanner_import_does_not_run_pipeline(self) -> None:
        import scripts.run_early_accumulation_signal_v1 as early

        self.assertEqual(early.PREFIX, "early_accumulation_signal_v1")
        self.assertEqual(early.OUTPUT_SIGNAL_N, 12)
        self.assertEqual(early.V123_REFERENCE_MAX_STALE_DAYS, 7)


if __name__ == "__main__":
    unittest.main()
