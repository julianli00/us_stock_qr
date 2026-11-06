from __future__ import annotations

from pathlib import Path
from types import FunctionType

import numpy as np

from us_quant import quartet_trio_loss_accounting_v1 as prior
from us_quant import quartet_trio_loss_accounting_v2 as canonical
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quintet-learning-loss-accounting-v1.json"
ASSETS, GROUPS, PATHS = prior.ASSETS, prior.GROUPS, prior.PATHS
ROLES = ("adaptive", "elapsed", "risk_units")
ROLE_IDS = {
    "adaptive": "expert_pool_quintet_adaptive_hedge_v1",
    "elapsed": "expert_pool_quintet_anytime_hedge_v1",
    "risk_units": "expert_pool_quintet_hedge_risk_units_v1",
}
StoredBook, marks, grouped, require = prior.StoredBook, prior.marks, prior.grouped, prior.require
PRIOR_HEAD = "b7f953c4190694b0b27751d7941599ddc0d243115cc190390224c13e9d930f2f"


def expected_policy():
    return {
        "schema_version": 1,
        "study_id": "ssd_quintet_learning_loss_accounting_20261106_v1",
        "actual_native_event_seq": 189600,
        "prior_journal_sequence": 1452,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-learning-loss-accounting-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "ca3914f24078279a6d19e7cfce70f3a63a8bc8aa50cc65aa8c1097bf5c1e5caf",
        },
        "role_ids": ROLE_IDS,
        "path_keys": list(PATHS),
        "stored_full_books": 12,
        "original_return_rows": 22596,
        "own_episode_slots": 12,
        "common_clock_interval_slots": 36,
        "adaptive_minus_control_comparison_slots": 24,
        "initial_capital_USD": 10000,
        "asset_groups": {key: list(value) for key, value in GROUPS.items()},
        "monetary_reconciliation_tolerance_USD": marks.MONEY_TOLERANCE,
        "cached_drawdown_agreement_tolerance": marks.METRIC_TOLERANCE,
        "peak_convention": "latest_preceding_running_max_including_initial_USD10000",
        "trough_convention": "earliest_maximum_observed_drawdown",
        "all_three_own_episode_common_clocks_required": True,
        "prefix_row_cuts": ["1", "21", "126", "half", "full"],
        "primitive_and_canonical_prefix_uint64_including_signed_zero": True,
        "original_numpy_auxiliary_retained_unchanged": True,
        "failed_V1_all_column_contract_repaired_or_waived": False,
        "source_data_reader": "unchanged_canonical_review_market",
        "source_RF_calendar_or_price_coordinates_changed": False,
        "new_strategy_target_account_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "retrospective_episodes_are_predictive_labels": False,
        "asset_groups_are_strategy_experts": False,
        "different_start_NAVs_are_equal_capital_counterfactuals": False,
        "fee_free_or_expert_causal_counterfactual_claimed": False,
        "raw_source_prices_units_targets_or_daily_rows_exported": False,
        "financial_criteria_or_global_pool_changed": False,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def validate(policy):
    fixed = expected_policy()
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze the twelve saved books,full clocks,old numerical limits and descriptive scope.",
    )


def compare(books, opened, closed):
    require(
        set(books) == set(ROLES),
        "Supply all three complete stored learning-law books,not a selected favorable pair.",
    )
    delegated = FunctionType(
        prior.compare.__code__,
        {**prior.compare.__globals__, **globals()},
        argdefs=prior.compare.__defaults__,
    )
    daily, original = delegated(books, opened, closed)
    common, contrasts = [], []
    for row in original["common_clock_episode_views"]:
        views = row["both_stored_accounts_at_identical_clocks"]
        common.append(
            {
                "episode_origin": row["episode_origin"],
                "episode_status": row["episode_status"],
                "all_three_stored_accounts_at_identical_clocks": views,
                "new_capital_or_matched_window_financial_gate": False,
                "different_start_NAVs_are_equal_capital_counterfactuals": False,
            }
        )
        for control in ROLES[1:]:
            comparison = None
            if views is not None:
                adaptive, cached = views["adaptive"], views[control]
                comparison = {
                    "stored_interval_net_change_difference": (
                        adaptive["actual_stored_interval_change"]
                        - cached["actual_stored_interval_change"]
                    ),
                    "asset_group_mark_fraction_own_start_NAV_difference": {
                        group: (
                            adaptive["asset_group_mark_fraction_of_own_start_NAV"][group]
                            - cached["asset_group_mark_fraction_of_own_start_NAV"][group]
                        )
                        for group in GROUPS
                    },
                    "recorded_fee_fraction_own_start_NAV_difference": (
                        adaptive["recorded_fees_USD"] / adaptive["start_NAV_USD"]
                        - cached["recorded_fees_USD"] / cached["start_NAV_USD"]
                    ),
                    "recorded_model_ticket_difference": (
                        adaptive["model_tickets"] - cached["model_tickets"]
                    ),
                    "is_equal_capital_causal_or_new_financial_result": False,
                }
            contrasts.append(
                {
                    "episode_origin": row["episode_origin"],
                    "control_role": control,
                    "episode_status": row["episode_status"],
                    "adaptive_minus_control": comparison,
                }
            )
    return daily, {
        "own_books": original["own_books"],
        "common_clock_episode_views": common,
        "adaptive_minus_control_comparisons": contrasts,
        "initial_capital_USD": marks.CAPITAL,
        "episode_selection_is_retrospective_not_predictive": True,
        "new_strategy_target_account_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "fee_free_or_expert_causal_counterfactual_claimed": False,
    }


def check_prefix(full, prefix):
    report = canonical.check_prefix(full, prefix)
    expected = canonical.canonical_prefix_view(full.iloc[: len(prefix)])
    actual = canonical.canonical_prefix_view(prefix)
    left = expected.to_numpy(dtype=np.float64)
    right = actual.to_numpy(dtype=np.float64)
    if not np.array_equal(left.view(np.uint64), right.view(np.uint64)):
        raise QuantError(
            "A primitive or canonical scalar binary64 prefix changed,including signed zero."
        )
    return {
        **report,
        "primitive_and_canonical_uint64_prefix_exact_including_signed_zero": True,
        "original_numpy_auxiliary_retained_unchanged": True,
        "failed_V1_all_column_contract_repaired_or_waived": False,
    }
