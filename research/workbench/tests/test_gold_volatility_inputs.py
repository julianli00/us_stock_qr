from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import gold_volatility_inputs as inputs
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


def raw(tmp_path, dates=("09/18/2009", "09/21/2009"), values=("18.20", "19.10")):
    path = tmp_path / "GVZ.csv"
    path.write_text("DATE,GVZ\n" + "".join(f"{day},{value}\n" for day, value in zip(dates, values, strict=True)))
    return path


def histories():
    index = sessions("2015-08-10", "2026-10-05")
    result = {name:pd.Series(15.0, index=index, name=name) for name in inputs.SERIES}
    result["GVZ"] = result["GVZ"].drop(pd.to_datetime(list(inputs.MISSING)))
    return index, result


def test_exact_native_scalar_field_is_not_a_price_return_or_ohlc_default(tmp_path):
    observed = inputs.parse_history(raw(tmp_path), "GVZ", pd.Timestamp("2026-10-11"))
    assert observed.tolist() == [18.2, 19.1] and observed.name == "GVZ"
    assert observed.index.equals(pd.DatetimeIndex(["2009-09-18", "2009-09-21"]))


@pytest.mark.parametrize("problem", ["duplicate", "reordered", "weekend", "future", "nan", "zero",
                                   "negative", "infinite", "text", "wrong_schema", "empty"])
def test_invalid_native_history_stops_instead_of_inventing_closes(tmp_path, problem):
    dates, values = ["09/18/2009", "09/21/2009"], ["18.20", "19.10"]
    if problem == "duplicate":
        dates[1] = dates[0]
    elif problem == "reordered":
        dates.reverse()
    elif problem == "weekend":
        dates[1] = "09/19/2009"
    elif problem == "future":
        dates[1] = "10/12/2026"
    elif problem in ("nan", "zero", "negative", "infinite", "text"):
        values[1] = {"nan":"", "zero":"0", "negative":"-1", "infinite":"inf", "text":"missing"}[problem]
    path = raw(tmp_path, dates, values)
    if problem == "wrong_schema":
        path.write_text("DATE,CLOSE\n09/18/2009,18.20\n")
    elif problem == "empty":
        path.write_text("DATE,GVZ\n")
    with pytest.raises(QuantError):
        inputs.parse_history(path, "GVZ", pd.Timestamp("2026-10-11"))


def test_known_daily_gaps_remain_nan_but_original_monthly_actual_quotes_are_complete():
    index, source = histories()
    aligned = inputs.align_closes(index, source)
    assert len(aligned) == 2805 and aligned["GVZ"].isna().sum() == 4
    monthly = inputs.monthly_closes(aligned)
    assert len(monthly) == 131 and monthly.notna().all().all()
    assert not monthly.index.isin(pd.to_datetime(list(inputs.MISSING))).any()
    assert aligned.loc[list(inputs.MISSING), "GVZ"].isna().all()


@pytest.mark.parametrize("problem", ["unknown_gap", "filled_known", "missing_vix", "identity", "calendar"])
def test_unknown_gap_filled_gap_or_changed_scope_cannot_masquerade_as_readiness(problem):
    index, source = histories()
    if problem == "unknown_gap":
        source["GVZ"] = source["GVZ"].iloc[1:]
    elif problem == "filled_known":
        source["GVZ"] = pd.Series(15.0, index=index, name="GVZ")
    elif problem == "missing_vix":
        source["VIX"] = source["VIX"].iloc[1:]
    elif problem == "identity":
        source["GVZ"] = source["GVZ"].rename("VIX")
    else:
        index = index[1:]
    with pytest.raises(QuantError):
        inputs.align_closes(index, source)


@pytest.mark.parametrize("problem", ["filled_known", "bad_monthly_quote", "negative", "wrong_labels"])
def test_monthly_scope_does_not_repair_daily_gaps_or_accept_missing_monthly_inputs(problem):
    index, source = histories()
    aligned = inputs.align_closes(index, source)
    if problem == "filled_known":
        aligned.loc[inputs.MISSING[0], "GVZ"] = 15.0
    elif problem in ("bad_monthly_quote", "negative"):
        month = inputs.monthly_closes(aligned).index[0]
        aligned.loc[month, "GVZ"] = np.inf if problem == "bad_monthly_quote" else -1
    else:
        aligned = aligned.rename(columns={"GVZ":"GLD"})
    with pytest.raises(QuantError):
        inputs.monthly_closes(aligned)


@pytest.mark.parametrize("field,value", [("full_daily_readiness", True), ("known_missing_gvz_daily_dates", []),
                                        ("original_monthly_query_count", 130),
                                        ("strategy_states_targets_or_returns_computed", True),
                                        ("native_value_column", "CLOSE"), ("new_registered_candidates", 2)])
def test_fixed_source_scope_and_missingness_cannot_be_relaxed(field, value):
    policy = read_json(ROOT / "config/gold-volatility-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        inputs.validate_policy(policy)
