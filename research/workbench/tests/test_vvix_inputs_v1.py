from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import vvix_inputs_v1 as inputs
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.storage import read_json


def test_native_quotes_remain_levels_not_returns(tmp_path):
    path = tmp_path / "VVIX.csv"
    path.write_text("DATE,VVIX\n01/02/2020,91.25\n01/03/2020,98.5\n")
    values = inputs.parse_history(path, "VVIX", pd.Timestamp("2020-01-03"))
    assert values.name == "VVIX" and values.tolist() == [91.25, 98.5]


@pytest.mark.parametrize(
    "text",
    [
        "DATE,CLOSE\n01/02/2020,91\n",
        "DATE,VVIX\n01/02/2020,91\n01/02/2020,92\n",
        "DATE,VVIX\n01/03/2020,91\n01/02/2020,92\n",
        "DATE,VVIX\n01/04/2020,91\n",
        "DATE,VVIX\n01/06/2020,91\n",
        "DATE,VVIX\n01/02/2020,0\n",
        "DATE,VVIX\n01/02/2020,-1\n",
        "DATE,VVIX\n01/02/2020,inf\n",
        "DATE,VVIX\n01/02/2020,\n",
    ],
)
def test_bad_native_data_is_not_filled_or_reinterpreted(tmp_path, text):
    path = tmp_path / "VVIX.csv"
    path.write_text(text)
    with pytest.raises(QuantError):
        inputs.parse_history(path, "VVIX", pd.Timestamp("2020-01-03"))


def synthetic_histories():
    index = sessions("2015-08-10", "2026-10-05")
    return index, {
        name: pd.Series(np.linspace(50, 100, len(index)), index=index, name=name)
        for name in inputs.SERIES
    }


@pytest.mark.parametrize("monthly_gap", [False, True])
def test_missingness_is_explicit_and_all_rows_remain(monthly_gap):
    index, histories = synthetic_histories()
    months = list(decision_pairs(index))
    gap = months[0] if monthly_gap else index[0]
    histories["VVIX"] = histories["VVIX"].drop(gap)
    aligned = inputs.align_closes(index, histories)
    result = inputs.coverage(aligned)
    assert len(aligned) == 2805 and pd.isna(aligned.loc[gap, "VVIX"])
    assert result["status"] == "incomplete_daily_coverage"
    assert result["full_daily_coverage"] is False
    assert result["all_original_monthly_queries_covered"] is not monthly_gap
    assert result["series"]["VVIX"]["missing_daily_dates"] == [str(gap.date())]
    assert result["series"]["VVIX"]["original_monthly_queries"] == 131


def test_complete_coverage_is_not_an_investment_or_fresh_quote_claim():
    index, histories = synthetic_histories()
    result = inputs.coverage(inputs.align_closes(index, histories))
    assert result["full_daily_coverage"] is True
    assert result["all_original_monthly_queries_covered"] is True
    assert result["strategy_eligibility_or_safe_state_inferred"] is False


def test_calendar_or_identity_substitution_stops():
    index, histories = synthetic_histories()
    with pytest.raises(QuantError):
        inputs.align_closes(index[1:], histories)
    histories["VVIX"].name = "VIX"
    with pytest.raises(QuantError):
        inputs.align_closes(index, histories)


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_market_context_sessions", 2804),
        ("original_monthly_query_count", 130),
        ("full_daily_coverage_required_for_full_admission", False),
        ("new_source_url", "https://unadvertised.invalid/VVIX.csv"),
        ("order_authority", True),
        ("old_near_term_VVIX_admitted_false_record_unchanged", False),
    ],
)
def test_source_scope_cannot_be_relaxed(field, value):
    policy = read_json(Path(__file__).resolve().parents[1] / "config/vvix-inputs-v1.json")
    inputs.validate_policy(policy)
    policy[field] = value
    with pytest.raises(QuantError):
        inputs.validate_policy(policy)
