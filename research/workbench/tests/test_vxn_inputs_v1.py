from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import vxn_inputs_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

POLICY = Path(__file__).resolve().parents[1] / "config/vxn-inputs-v1.json"


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-10-29T09:58:48.484974Z", "2026-10-28"),
        ("2026-10-29T21:00:00Z", "2026-10-29"),
        ("2026-10-10T15:21:41.648491Z", "2026-10-09"),
    ],
)
def test_source_ceiling_uses_actual_completed_ny_index_session(stamp, expected):
    assert rules.last_completed_session(stamp) == pd.Timestamp(expected)


@pytest.mark.parametrize("stamp", ["2026-10-29", "not-a-date"])
def test_unknown_acquisition_clock_is_not_a_current_date_default(stamp):
    with pytest.raises(QuantError):
        rules.last_completed_session(stamp)


def native(tmp_path, rows):
    path = tmp_path / "VXN.csv"
    path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n" + rows)
    return path


def test_vxn_uses_close_and_keeps_native_identity(tmp_path):
    path = native(tmp_path, "10/01/2026,1,3,1,2.25\n10/02/2026,2,4,1,3.50\n")
    values = rules.parse_history(path, "VXN", pd.Timestamp("2026-10-28"))
    assert values.name == "VXN"
    assert values.tolist() == [2.25, 3.5]


def test_wrong_header_is_not_a_close_column_fallback(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("DATE,VXN\n10/01/2026,20\n")
    with pytest.raises(QuantError, match="header"):
        rules.parse_history(path, "VXN", pd.Timestamp("2026-10-28"))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", ""])
def test_bad_native_close_is_not_filled(tmp_path, value):
    path = native(tmp_path, f"10/01/2026,1,3,1,{value}\n")
    with pytest.raises(QuantError):
        rules.parse_history(path, "VXN", pd.Timestamp("2026-10-28"))


@pytest.mark.parametrize(
    "rows",
    [
        "10/01/2026,1,3,1,2\n10/01/2026,1,3,1,2\n",
        "10/02/2026,1,3,1,2\n10/01/2026,1,3,1,2\n",
        "10/03/2026,1,3,1,2\n",
        "10/29/2026,1,3,1,2\n",
    ],
)
def test_duplicate_reordered_weekend_or_unclosed_dates_are_not_trimmed(tmp_path, rows):
    path = native(tmp_path, rows)
    with pytest.raises(QuantError):
        rules.parse_history(path, "VXN", pd.Timestamp("2026-10-28"))


def test_unknown_ticker_is_not_aliased_to_an_admitted_source(tmp_path):
    path = native(tmp_path, "10/01/2026,1,3,1,2\n")
    with pytest.raises(QuantError):
        rules.parse_history(path, "QQQ", pd.Timestamp("2026-10-28"))


@pytest.mark.parametrize("size", [0, 5_000_001])
def test_source_file_size_is_bounded(tmp_path, size):
    path = tmp_path / "bounded.csv"
    path.write_bytes(b" " * size)
    with pytest.raises(QuantError, match="bounded"):
        rules.parse_history(path, "VXN", pd.Timestamp("2026-10-28"))


def histories():
    index = sessions("2015-08-10", "2026-10-05")
    return index, {
        "VXN": pd.Series(20.0, index=index, name="VXN"),
        "VIX": pd.Series(15.0, index=index, name="VIX"),
    }


def test_original_daily_and_monthly_queries_are_complete():
    index, values = histories()
    actual = rules.coverage(rules.align_closes(index, values))
    assert actual["full_daily_coverage"] is True
    assert actual["all_original_monthly_queries_covered"] is True
    assert actual["series"]["VXN"]["context_sessions"] == 2805
    assert actual["series"]["VXN"]["original_monthly_queries"] == 131
    assert actual["strategy_eligibility_or_safe_state_inferred"] is False


def test_complete_monthly_queries_do_not_hide_a_daily_gap():
    index, values = histories()
    values["VXN"] = values["VXN"].drop(pd.Timestamp("2020-03-17"))
    aligned = rules.align_closes(index, values)
    actual = rules.coverage(aligned)
    assert len(aligned) == 2805 and pd.isna(aligned.loc["2020-03-17", "VXN"])
    assert actual["full_daily_coverage"] is False
    assert actual["all_original_monthly_queries_covered"] is True
    assert actual["series"]["VXN"]["missing_daily_dates"] == ["2020-03-17"]


def test_missing_month_end_is_retained_and_reported():
    index, values = histories()
    values["VXN"] = values["VXN"].drop(pd.Timestamp("2020-03-31"))
    actual = rules.coverage(rules.align_closes(index, values))
    assert actual["all_original_monthly_queries_covered"] is False
    assert actual["series"]["VXN"]["missing_monthly_dates"] == ["2020-03-31"]


@pytest.mark.parametrize("damage", ["duplicate", "nan", "negative", "name", "timezone"])
def test_alignment_cannot_launder_invalid_native_values(damage):
    index, values = histories()
    if damage == "duplicate":
        values["VXN"] = pd.concat([values["VXN"], values["VXN"].iloc[-1:]])
    elif damage == "nan":
        values["VXN"].iloc[2] = np.nan
    elif damage == "negative":
        values["VXN"].iloc[2] = -1
    elif damage == "name":
        values["VXN"].name = "VIX"
    else:
        values["VXN"].index = values["VXN"].index.tz_localize("UTC")
    with pytest.raises(QuantError):
        rules.align_closes(index, values)


@pytest.mark.parametrize(
    "key,value",
    [
        ("required_market_context_sessions", 2804),
        ("full_daily_coverage_required_for_full_admission", False),
        ("native_value_column", "OPEN"),
        ("order_authority", True),
    ],
)
def test_source_contract_cannot_be_weakened(key, value):
    policy = read_json(POLICY)
    rules.validate_policy(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate_policy(policy)
