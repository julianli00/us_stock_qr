from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from us_quant.config import QuantError
from us_quant.fundamentals import annual_quality_features
from us_quant.storage import file_digest, read_json, utc_now, write_json


def fundamental_signals(payload: dict, as_of: str) -> dict:
    evidence = annual_quality_features(payload, as_of)
    prior = evidence["provenance"]["prior_assets"]["value"]
    current = evidence["provenance"]["assets"]["value"]
    values = {
        "cashflow_accrual_quality_v1": evidence["features"][
            "cash_minus_income_over_average_assets"
        ],
        "conservative_asset_growth_v1": 1 - current / prior,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise QuantError("Annual factor values are not finite.")
    return {
        **evidence,
        "mode": "causal_fundamental_factor_preparation_not_strategy_backtest",
        "factor_values": values,
        "factor_direction": "higher values rank higher; no weights or orders generated",
        "independent_factor_count_verified": False,
        "full_historical_universe_verified": False,
        "strategy_qualified": False,
        "objective_verified": False,
        "order_authority": False,
        "limitations": [
            *evidence["limitations"],
            "Cash-flow accrual quality is a proxy, not the original balance-sheet replication.",
            "Conservative investment is negative asset growth, not a return guarantee.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal annual factor values; no stock recommendations."
    )
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise QuantError("Refusing to overwrite a dated factor evidence artifact.")
        result = fundamental_signals(read_json(args.facts), args.as_of)
        result.update(
            {
                "created_at": utc_now(),
                "source_sha256": file_digest(args.facts),
                "implementation_sha256": file_digest(Path(__file__)),
            }
        )
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "mode": result["mode"],
                    "decision_session": result["decision_session"],
                    "factor_values": result["factor_values"],
                    "objective_verified": False,
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Fundamental factor preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
