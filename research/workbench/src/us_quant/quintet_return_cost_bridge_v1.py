from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.quartet_trio_loss_accounting_v1 import require
from us_quant.saved_main_book_decomposition_v1 import MONEY_TOLERANCE

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/quintet-return-cost-bridge-v1.json"
ROLES = ("opportunity", "cached_volatility")
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
INITIAL = 10000.0
TOLERANCE = 1e-12
COLUMNS = ("equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders", "risk_free")


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_quintet_return_cost_bridge_20261106_v1",
        "actual_native_event_seq": 186341,
        "prior_journal_sequence": 1416,
        "prior_journal_head_sha256": (
            "e50d5c627e252346930fc126c0a454494522a2b4a923a6e1fb9ad258cd573157"
        ),
        "prior_combined_completed": 297,
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-return-cost-bridge-20261106-v1/"
                "source-inputs-v1.json"
            ),
            "sha256": "104dbdb760da9ad9162106e85a2e4a6f4966cc1e8e3425dd9699994b91028f4b",
        },
        "paired_cases": 8,
        "existing_books": 16,
        "roles": list(ROLES),
        "path_keys": list(PATHS),
        "initial_capital_usd": 10000,
        "full_return_rows_10y": 2512,
        "full_return_rows_5y": 1254,
        "money_identity_tolerance_USD": MONEY_TOLERANCE,
        "return_and_log_identity_tolerance": TOLERANCE,
        "prefix_contract": "every_recorded_primitive_and_row_local_scalar_bridge_bit_exact",
        "prefix_positions": [1, 21, 126, "half", "full"],
        "net_log_component": "log(current_recorded_NAV/previous_recorded_NAV)",
        "recorded_current_cost_log_drag": "log1p(recorded_cost/current_recorded_NAV)",
        "gross_component": "math_fsum(net_log_component,recorded_current_cost_log_drag)",
        "independent_gross_check": (
            "log((current_recorded_NAV+recorded_cost)/previous_recorded_NAV)"
        ),
        "new_source_requests": 0,
        "new_strategy_targets_accounts_BT_or_financial_trials": 0,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
        "fee_free_funded_policy_counterfactual_claimed": False,
        "fee_or_expert_causal_alpha_attribution_claimed": False,
        "financial_criteria_or_global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze original recorded-NAV/cost identities and interpretation without source rescue.",
    )


def decompose(account):
    require(
        isinstance(account, pd.DataFrame)
        and not account.empty
        and isinstance(account.index, pd.DatetimeIndex)
        and account.index.is_unique
        and account.index.is_monotonic_increasing
        and tuple(account.columns) == COLUMNS
        and all(dtype.kind in "fiu" for dtype in account.dtypes)
        and np.isfinite(account.to_numpy()).all(),
        "Use the complete finite original recorded-book Interface without replacing columns.",
    )
    require(
        (account["equity"] > 0).all()
        and (account["return"] > -1).all()
        and (account["cash"] >= -MONEY_TOLERANCE).all()
        and (account["gross_exposure"] >= 0).all()
        and (account["gross_exposure"] <= 1 + TOLERANCE).all()
        and (account["cost"] >= 0).all()
        and (account["turnover"] >= 0).all()
        and (account["orders"] >= 0).all()
        and (account["orders"] == np.floor(account["orders"])).all(),
        "Recorded positive NAV, cash funding, nonnegative costs and integer tickets are required.",
    )
    previous, rows = INITIAL, []
    max_return, max_log, max_money = 0.0, 0.0, 0.0
    for _, row in account.iterrows():
        nav, cost, stored_return = float(row["equity"]), float(row["cost"]), float(row["return"])
        net_return = nav / previous - 1
        net_log = math.log(nav / previous)
        cost_log = math.log1p(cost / nav)
        gross_log = math.fsum((net_log, cost_log))
        observed_change = math.fsum((nav, -previous))
        gross_gain = math.fsum((nav, -previous, cost))
        return_gap = abs(net_return - stored_return)
        log_gap = max(
            abs(net_log - math.log1p(stored_return)),
            abs(gross_log - math.log((nav + cost) / previous)),
            abs(cost_log - math.log((nav + cost) / nav)),
        )
        money_gap = abs(math.fsum((gross_gain, -cost, -observed_change)))
        require(
            return_gap <= TOLERANCE and log_gap <= TOLERANCE and money_gap <= MONEY_TOLERANCE,
            "The recorded return or independent dollar/log identity disagrees.",
        )
        max_return, max_log, max_money = (
            max(max_return, return_gap),
            max(max_log, log_gap),
            max(max_money, money_gap),
        )
        rows.append(
            {
                **{name: float(row[name]) for name in COLUMNS},
                "previous_recorded_NAV_USD": previous,
                "observed_net_change_USD": observed_change,
                "recorded_path_gross_gain_USD": gross_gain,
                "net_log_component": net_log,
                "recorded_current_cost_log_drag": cost_log,
                "recorded_path_gross_log_component": gross_log,
            }
        )
        previous = nav
    daily = pd.DataFrame(rows, index=account.index)
    net_logs = math.fsum(daily["net_log_component"])
    cost_logs = math.fsum(daily["recorded_current_cost_log_drag"])
    gross_logs = math.fsum(daily["recorded_path_gross_log_component"])
    paid_costs = math.fsum(daily["cost"])
    gross_gain = math.fsum(daily["recorded_path_gross_gain_USD"])
    net_gain = math.fsum((previous, -INITIAL))
    require(
        abs(net_logs - math.log(previous / INITIAL)) <= TOLERANCE
        and abs(math.fsum((gross_logs, -cost_logs, -net_logs))) <= TOLERANCE
        and abs(math.fsum((gross_gain, -paid_costs, -net_gain))) <= MONEY_TOLERANCE,
        "Full-book dollar/log bridges must telescope from actual initial USD10000.",
    )
    return daily, {
        "initial_capital_USD": INITIAL,
        "recorded_start_session": str(account.index[0].date()),
        "recorded_end_session": str(account.index[-1].date()),
        "recorded_final_NAV_USD": previous,
        "recorded_net_gain_USD": net_gain,
        "recorded_paid_costs_USD": paid_costs,
        "recorded_path_gross_gain_USD": gross_gain,
        "net_log_component": net_logs,
        "recorded_current_cost_log_drag": cost_logs,
        "recorded_path_gross_log_component": gross_logs,
        "return_rows": len(account),
        "maximum_independent_return_difference": max_return,
        "maximum_independent_log_identity_difference": max_log,
        "maximum_independent_money_identity_difference_USD": max_money,
        "fee_free_funded_policy_counterfactual_claimed": False,
        "fee_or_expert_causal_alpha_attribution_claimed": False,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
    }


def check_prefix(full, prefix):
    expected = full.iloc[: len(prefix)]
    require(
        0 < len(prefix) <= len(full)
        and prefix.index.equals(full.index[: len(prefix)])
        and prefix.equals(expected)
        and np.array_equal(
            prefix.to_numpy(dtype="float64").view(np.uint64),
            expected.to_numpy(dtype="float64").view(np.uint64),
        ),
        "Every recorded primitive and row-local scalar bridge must be bit-exact on prefixes.",
    )
    return {"return_rows": len(prefix), "every_primitive_and_scalar_prefix_bit_exact": True}


def paired_summary(summaries):
    require(
        set(summaries) == set(ROLES)
        and all(
            summaries[ROLES[0]][field] == summaries[ROLES[1]][field]
            for field in (
                "initial_capital_USD",
                "return_rows",
                "recorded_start_session",
                "recorded_end_session",
            )
        ),
        "Retain both complete original book summaries for paired accounting.",
    )
    fields = (
        "recorded_final_NAV_USD",
        "recorded_paid_costs_USD",
        "recorded_path_gross_gain_USD",
        "net_log_component",
        "recorded_current_cost_log_drag",
        "recorded_path_gross_log_component",
    )
    difference = {
        name: math.fsum((summaries[ROLES[0]][name], -summaries[ROLES[1]][name])) for name in fields
    }
    require(
        abs(
            math.fsum(
                (
                    difference["recorded_path_gross_log_component"],
                    -difference["recorded_current_cost_log_drag"],
                    -difference["net_log_component"],
                )
            )
        )
        <= TOLERANCE
        and abs(
            math.fsum(
                (
                    difference["recorded_path_gross_gain_USD"],
                    -difference["recorded_paid_costs_USD"],
                    -difference["recorded_final_NAV_USD"],
                )
            )
        )
        <= MONEY_TOLERANCE,
        "Paired differences must preserve both actual dollar and log-growth identities.",
    )
    return {
        "both_original_books": summaries,
        "opportunity_minus_cached_volatility": difference,
        "new_strategy_targets_accounts_BT_or_financial_trials": 0,
        "fee_free_funded_policy_counterfactual_claimed": False,
        "fee_or_expert_causal_alpha_attribution_claimed": False,
        "source_CAGR_Sharpe_or_qualification_recomputed": False,
    }


def compare(books):
    require(
        set(books) == set(ROLES) and books[ROLES[0]].index.equals(books[ROLES[1]].index),
        "Pair only exact common-calendar original books, without cropping or filling.",
    )
    daily, summaries = {}, {}
    for role in ROLES:
        daily[role], summaries[role] = decompose(books[role])
    return daily, paired_summary(summaries)
