from pathlib import Path
from types import FunctionType

from us_quant import quartet_trio_loss_accounting_v1 as prior
from us_quant import quintet_learning_loss_accounting_v1 as canonical

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/equal-role-loss-accounting-v1.json"
ASSETS, GROUPS, PATHS = prior.ASSETS, prior.GROUPS, prior.PATHS
ROLES = ("equal_role", "flat")
ROLE_IDS = {
    "equal_role": "expert_pool_six_equal_role_projection_v2",
    "flat": "expert_pool_protective_six_adaptive_hedge_v1",
}
StoredBook, marks, grouped, require = prior.StoredBook, prior.marks, prior.grouped, prior.require
check_prefix = canonical.check_prefix


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_equal_role_loss_accounting_20261106_v1",
        "actual_native_event_seq": 192661,
        "prior_journal_sequence": 1478,
        "prior_journal_head_sha256": (
            "d0a7e820e4d503d82b18f88652b7a17c99ee60b99e4665fe7879d5167fd6f08d"
        ),
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/equal-role-loss-accounting-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "a618f47ed86ad392e081379dda6e24795891e976426c4f2fbd8bd9b79f428b3c",
        },
        "stored_full_books": 8,
        "original_return_rows": 15064,
        "own_episode_slots": 8,
        "common_clock_interval_slots": 16,
        "equal_role_minus_flat_comparison_slots": 8,
        "money_tolerance_USD": marks.MONEY_TOLERANCE,
        "cached_drawdown_tolerance": marks.METRIC_TOLERANCE,
        "prefix_row_cuts": ["1", "21", "126", "half", "full"],
        "primitive_canonical_prefix_uint64_including_signed_zero": True,
        "original_numpy_auxiliary_retained_unchanged": True,
        "failed_all_column_prefix_contract_repaired_or_waived": False,
        "source_reader": "unchanged_canonical_review_market",
        "new_strategy_target_account_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "causal_expert_fee_or_fee_free_counterfactual_claimed": False,
        "asset_groups_are_strategy_experts": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze the eight stored books and unchanged money,RF,prefix and scope contracts.",
    )


def compare(books, opened, closed):
    function = FunctionType(
        prior.compare.__code__,
        {**prior.compare.__globals__, **globals()},
        argdefs=prior.compare.__defaults__,
    )
    daily, summary = function(books, opened, closed)
    comparisons = []
    for row in summary["common_clock_episode_views"]:
        views = row["both_stored_accounts_at_identical_clocks"]
        differences = None
        if views is not None:
            left, right = views["equal_role"], views["flat"]
            differences = {
                "stored_interval_net_change_difference": (
                    left["actual_stored_interval_change"] - right["actual_stored_interval_change"]
                ),
                "asset_group_mark_fraction_own_start_NAV_difference": {
                    group: left["asset_group_mark_fraction_of_own_start_NAV"][group]
                    - right["asset_group_mark_fraction_of_own_start_NAV"][group]
                    for group in GROUPS
                },
                "recorded_fee_fraction_own_start_NAV_difference": (
                    left["recorded_fees_USD"] / left["start_NAV_USD"]
                    - right["recorded_fees_USD"] / right["start_NAV_USD"]
                ),
                "is_equal_capital_causal_or_new_financial_result": False,
            }
        comparisons.append(
            {
                "episode_origin": row["episode_origin"],
                "episode_status": row["episode_status"],
                "equal_role_minus_flat": differences,
            }
        )
    return daily, {**summary, "equal_role_minus_flat_comparisons": comparisons}
