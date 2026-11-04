import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import archived_vxn_expert_eligibility_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def example():
    index = sessions("2015-08-10", "2026-10-05")
    columns = list(rules.UNIVERSE[:12])
    parent = pd.DataFrame(np.nan, index=index, columns=columns)
    dates = pd.DatetimeIndex(
        [d for d in index if d >= pd.Timestamp("2016-08-31") and is_month_end(d)]
    )
    parent.loc[dates] = 0.0
    parent.loc[dates, list(rules.STOCKS)] = 0.49 / 5
    parent.loc[dates, "BIL"] = 0.49
    child = parent.copy()
    child.loc[dates, "SOXX"] = 0.245
    child.loc[dates, ["MTUM", "VLUE", "QUAL", "USMV"]] = 0.245 / 4
    audits = [
        {
            "decision_session": str(d.date()),
            "source_equity_budget_preserved": 0.49,
            "VXN_quote_session": str(d.date()),
            "VIX_quote_session": str(d.date()),
            "information_clock_after_16_15_ET": True,
            "hmm_refitted_or_states_recomputed": False,
            "exact_portfolio_or_SOXX_implied_variance_claimed": False,
            "true_variance_risk_premium_claimed": False,
            "BIL_target_unchanged": 0.49,
            "GLD_target_unchanged": 0.0,
            "SOXX_share_of_stock_budget": 0.5,
            "observed_variation": {
                s: {"last_observation": str(d.date()), "return_sessions": 21}
                for s in ("SPY", "QQQ")
            },
        }
        for d in dates
    ]
    proof = {
        "monthly_decisions": 122,
        "original_calendar_opportunities": 131,
        "undefined_prehistory_months": 9,
        "additional_source_months_dropped": 0,
        "source_model_refitting": False,
        "original_source_signal_recomputed": False,
        "candidate_accounts_or_metrics_computed": False,
        "exact_nonstock_and_total_stock_budget_preserved": True,
        "prefix_truncations": list(dates[[0, 12, 36, 60, -1]].strftime("%Y-%m-%d")),
    }
    return parent, child, audits, proof


def test_full122_composition_preserves_parent_budget_and_all_nonstock_cells():
    result = rules.check_composition(*example())
    assert result["genuine_native_monthly_opinions"] == 122
    assert result["all_nonstock_cells_exact"]
    assert result["maximum_stock_budget_conservation_difference"] < 1e-12
    assert not result["HMM_variation_ratio_target_pipeline_replayed"]


@pytest.mark.parametrize(
    "kind", ["BIL", "stock_budget", "clock", "refit", "drop_month", "prehistory"]
)
def test_source_budget_clock_or_native_month_changes_cannot_be_repaired(kind):
    parent, child, audits, proof = example()
    day = child.dropna(how="all").index[0]
    if kind == "BIL":
        child.loc[day, "BIL"] += 0.01
        child.loc[day, "SOXX"] -= 0.01
    elif kind == "stock_budget":
        child.loc[day, "SOXX"] += 0.01
    elif kind == "clock":
        audits[0]["information_clock_after_16_15_ET"] = False
    elif kind == "refit":
        audits[0]["hmm_refitted_or_states_recomputed"] = True
    elif kind == "drop_month":
        child.loc[day] = np.nan
    else:
        proof["undefined_prehistory_months"] = 8
    with pytest.raises(QuantError):
        rules.check_composition(parent, child, audits, proof)


def test_original_archival_scope_not_upgraded_to_live_or_full_intraday_PIT():
    row = {
        "source_admitted": True,
        "admission_scope": "archival_daily_closes_and_original_query_coverage_only",
        "coverage": {"full_daily_coverage": True, "all_original_monthly_queries_covered": True},
        "ratios_strategy_states_targets_or_returns_computed": False,
        "full_intraday_or_historical_methodology_chain_verified": False,
    }
    assert not rules.check_scope(row, copy.deepcopy(row))[
        "source_scope_upgraded_or_new_query_executed"
    ]
    row["full_intraday_or_historical_methodology_chain_verified"] = True
    with pytest.raises(QuantError):
        rules.check_scope(row, copy.deepcopy(row))


def test_existing_monthly_Interface_structural_zeros_and125126_maturity_are_real():
    _, target, _, _ = example()
    dates = sessions("2016-10-06", "2026-10-05")
    returns = np.full(len(dates), 0.0001)
    nav = 10000 * np.cumprod(1 + returns)
    book = pd.DataFrame(
        {
            "equity": nav,
            "return": returns,
            "risk_free": 0.00001,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=dates,
    )
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(rules.CANDIDATE, "conditional_variance_growth", "2026-10-29T10:50:07Z"),
        target,
        book,
        rules.UNIVERSE,
    )
    initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
    assert initial.target_known_at == pd.Timestamp("2016-09-30")
    assert len(initial.net_returns) == 0 and not initial.score_mature
    assert initial.target[["IJR", "PKW"]].eq(0).all()
    assert not expert.observe(dates[124], 126).score_mature
    assert expert.observe(dates[125], 126).score_mature


def test_second_complete_or_partial_actual_review_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(rules.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(rules, "FOLDER", tmp_path)
    monkeypatch.setattr(rules, "RESULT", tmp_path / "result.json")
    (tmp_path / "review-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="never retry"):
        rules.review()
