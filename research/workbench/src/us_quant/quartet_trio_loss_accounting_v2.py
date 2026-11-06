from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from us_quant import quartet_trio_loss_accounting_v1 as prior
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quartet-trio-loss-accounting-v2.json"
ASSETS, ROLES, PATHS, GROUPS = prior.ASSETS, prior.ROLES, prior.PATHS, prior.GROUPS
StoredBook, compare, marks, require = prior.StoredBook, prior.compare, prior.marks, prior.require
AUXILIARY = "reconstructed_net_change"
CANONICAL = "canonical_scalar_reconstructed_net_change"
PREFIX_CONTRACT = "all_primitive_money_marks_plus_exact_math_fsum_noncanonical_NumPy_auxiliary_v2"


def validate(policy):
    source = read_json(prior.POLICY)
    expected = {
        **source,
        "schema_version": 2,
        "study_id": "ssd_quartet_trio_loss_accounting_20261106_v2",
        "same_economic_question_continuation": True,
        "prefix_contract": PREFIX_CONTRACT,
        "only_noncanonical_auxiliary_column": AUXILIARY,
        "original_V1_all_column_prefix_contract_remains_failed": True,
        "original_auxiliary_vector_column_retained_unchanged": True,
        "original_monetary_cash_and_cached_DD_limits_unchanged": True,
        "completed_V1_case_arithmetic_reused_without_reexecution": True,
        "continuation_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/quartet-trio-loss-accounting-20261106-v2/"
                "continuation-inputs-v2.json"
            ),
            "sha256": "9ca226383a30fda8990f676e28d3e0afa137509e5c805dc348e1247b4b040a8d",
        },
    }
    require(
        isinstance(policy, dict)
        and policy == expected
        and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Version only the canonical prefix Interface; preserve V1 failure and all source limits.",
    )


def canonical_prefix_view(daily):
    primitive = ["equity", "cash", "cost", "orders", "observed_net_change"]
    marks_columns = [name for s in ASSETS for name in ("overnight::" + s, "intraday::" + s)]
    expected = {*primitive, *marks_columns, AUXILIARY}
    require(
        not daily.empty
        and daily.columns.is_unique
        and set(daily.columns) == expected
        and daily.index.is_unique
        and daily.index.is_monotonic_increasing
        and all(dtype.kind in "fiu" for dtype in daily.dtypes)
        and np.isfinite(daily.to_numpy()).all(),
        "Keep every finite primitive economic column and the original auxiliary vector sum.",
    )
    scalar = np.array(
        [
            math.fsum(
                [
                    *[float(daily["overnight::" + s].iloc[i]) for s in ASSETS],
                    *[float(daily["intraday::" + s].iloc[i]) for s in ASSETS],
                ]
            )
            - float(daily["cost"].iloc[i])
            for i in range(len(daily))
        ]
    )
    require(
        max(abs(scalar - daily[AUXILIARY].to_numpy())) <= marks.MONEY_TOLERANCE
        and max(abs(scalar - daily["observed_net_change"].to_numpy())) <= marks.MONEY_TOLERANCE,
        "Canonical scalar reconstruction cannot waive the original USD1e-6 identity checks.",
    )
    result = daily.loc[:, [*primitive, *marks_columns]].copy()
    result[CANONICAL] = scalar
    return result


def check_prefix(full, prefix):
    require(
        0 < len(prefix) <= len(full) and prefix.index.equals(full.index[: len(prefix)]),
        "Use an exact retained-calendar prefix, not a crop, fill or reset.",
    )
    expected = full.iloc[: len(prefix)]
    require(
        canonical_prefix_view(prefix).equals(canonical_prefix_view(expected)),
        "A primitive economic field or exact canonical scalar prefix changed.",
    )
    before = np.asarray(expected[AUXILIARY], dtype=np.float64)
    after = np.asarray(prefix[AUXILIARY], dtype=np.float64)
    return {
        "return_rows": len(prefix),
        "primitive_and_canonical_scalar_prefix_bit_exact": True,
        "original_auxiliary_vector_column_retained_unchanged": True,
        "auxiliary_NumPy_vector_bit_different_values": int(
            np.count_nonzero(before.view(np.uint64) != after.view(np.uint64))
        ),
        "maximum_auxiliary_NumPy_prefix_difference_USD": float(max(abs(before - after))),
        "original_V1_all_column_prefix_comparison": prefix.equals(expected),
        "original_monetary_and_cached_DD_limits_changed": False,
    }
