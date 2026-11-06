from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_quant import growth_macro_defense_hedge_v1 as quartet
from us_quant import macro_defense_hedge_trio_v1 as controls
from us_quant import saved_main_book_decomposition_v1 as marks
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quartet-trio-loss-accounting-v1.json"
ROLES = ("quartet", "cached_trio")
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
GROUPS = {
    "equity_ETFs": ("SPY", "MTUM", "VLUE", "QUAL", "USMV", "QQQ", "XLK", "SOXX", "IJR", "PKW"),
    "duration_ETFs": ("IEF", "TLT"),
    "gold_ETF": ("GLD",),
    "bill_ETF": ("BIL",),
}


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_quartet_trio_loss_accounting_20261106_v1",
        "actual_native_event_seq": 179917,
        "prior_journal_sequence": 1320,
        "prior_journal_head_sha256": (
            "ef00d3856e2cd2fa0d18f9d8614a0d7e2ece8f8e64c2872cb94e0e09214ba41d"
        ),
        "quartet_ids": [c["id"] for c in quartet.CANDIDATES],
        "matched_trio_ids": [c["id"] for c in controls.CANDIDATES],
        "path_keys": list(PATHS),
        "paired_cases": 8,
        "existing_full_books": 16,
        "asset_groups": {k: list(v) for k, v in GROUPS.items()},
        "monetary_reconciliation_tolerance_USD": marks.MONEY_TOLERANCE,
        "cached_drawdown_agreement_tolerance": marks.METRIC_TOLERANCE,
        "peak_convention": "latest_preceding_running_max_including_initial_USD10000",
        "trough_convention": "earliest_maximum_observed_drawdown",
        "both_own_episode_common_clocks_required": True,
        "source_data_reader": "unchanged_canonical_review_market",
        "new_strategy_target_account_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "fee_free_or_expert_causal_counterfactual_claimed": False,
        "financial_criteria_or_global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        isinstance(policy, dict)
        and set(policy) == set(expected) | {"source_inputs"}
        and all(type(policy.get(k)) is type(v) and policy.get(k) == v for k, v in expected.items()),
        "Freeze saved-book roles, original monetary limits, exact clocks and interpretation.",
    )


@dataclass(frozen=True)
class StoredBook:
    account: pd.DataFrame
    weights: pd.DataFrame
    cached_max_drawdown: float


def grouped(interval):
    if interval is None:
        return None
    require(
        set(interval["asset_mark_PnL_USD"]) == set(ASSETS),
        "Keep all fourteen asset marks, including zero and positive offsets.",
    )
    grouped_marks = {
        name: math.fsum(interval["asset_mark_PnL_USD"][symbol] for symbol in symbols)
        for name, symbols in GROUPS.items()
    }
    require(
        abs(
            math.fsum(grouped_marks.values())
            - interval["recorded_fees_USD"]
            - interval["actual_NAV_change_USD"]
        )
        <= marks.MONEY_TOLERANCE,
        "Asset grouping cannot change the whole saved interval's net PnL.",
    )
    return {
        **interval,
        "asset_group_mark_PnL_USD": grouped_marks,
        "asset_group_mark_fraction_of_own_start_NAV": {
            k: v / interval["start_NAV_USD"] for k, v in grouped_marks.items()
        },
        "asset_group_is_strategy_expert": False,
        "mechanical_PnL_identity_is_expert_causal_alpha": False,
    }


def compare(books: dict[str, StoredBook], opened, closed):
    require(
        set(books) == set(ROLES)
        and all(book.account.index.equals(books[ROLES[0]].account.index) for book in books.values())
        and tuple(opened.columns) == tuple(closed.columns) == tuple(ASSETS)
        and all(tuple(book.weights.columns) == tuple(ASSETS) for book in books.values()),
        "Pair only the two exact full-calendar books and original fourteen asset coordinates.",
    )
    daily, summaries = {}, {}
    for role, book in books.items():
        require(
            type(book.cached_max_drawdown) in (float, int)
            and math.isfinite(book.cached_max_drawdown)
            and 0 <= book.cached_max_drawdown < 1,
            "Keep the original explicit cached drawdown, not an inferred qualification.",
        )
        daily[role], summary = marks.decompose(book.account, book.weights, opened, closed)
        gap = abs(summary["maximum_observed_drawdown"] - book.cached_max_drawdown)
        require(gap <= marks.METRIC_TOLERANCE, "The retained initial-peak drawdown changed.")
        summaries[role] = {
            **summary,
            "cached_drawdown_agreement_error": gap,
            "own_maximum_drawdown_episode": grouped(summary["own_maximum_drawdown_episode"]),
        }
    common = []
    for origin in ROLES:
        episode = summaries[origin]["own_maximum_drawdown_episode"]
        common.append(
            {
                "episode_origin": origin,
                "episode_status": summaries[origin]["episode_status"],
                "both_stored_accounts_at_identical_clocks": {
                    role: grouped(
                        marks.interval(
                            daily[role], episode["start_position"], episode["end_position"]
                        )
                    )
                    for role in ROLES
                }
                if episode is not None
                else None,
                "new_capital_or_matched_window_financial_gate": False,
                "different_start_NAVs_are_equal_capital_counterfactuals": False,
            }
        )
    return daily, {
        "own_books": summaries,
        "common_clock_episode_views": common,
        "initial_capital_USD": 10000.0,
        "new_strategy_target_account_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "episode_selection_is_retrospective_not_predictive": True,
        "fee_free_or_expert_causal_counterfactual_claimed": False,
    }
