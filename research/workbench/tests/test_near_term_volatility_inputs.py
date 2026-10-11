from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import near_term_volatility_inputs as inputs
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


def history(tmp_path, dates=("01/03/2011", "01/04/2011"), closes=("13.21", "12.31")):
    path = tmp_path / "history.csv"
    path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n" + "".join(
        f"{day},1.0,30.0,1.0,{close}\n" for day, close in zip(dates, closes, strict=True)
    ))
    return path


def test_native_closing_field_is_used_not_open_or_high(tmp_path):
    observed = inputs.parse_history(history(tmp_path), "VIX9D", pd.Timestamp("2026-10-11"))
    assert observed.tolist() == [13.21, 12.31]
    assert observed.index.equals(pd.DatetimeIndex(["2011-01-03", "2011-01-04"]))
    assert observed.name == "VIX9D"


@pytest.mark.parametrize("problem", ["duplicate", "reordered", "weekend", "future", "wrong_schema",
                                   "missing", "text", "zero", "negative", "infinite", "empty"])
def test_invalid_or_success_shaped_sources_stop_explicitly(tmp_path, problem):
    dates, values = ["01/03/2011", "01/04/2011"], ["13.21", "12.31"]
    if problem == "duplicate":
        dates[1] = dates[0]
    elif problem == "reordered":
        dates.reverse()
    elif problem == "weekend":
        dates[1] = "01/08/2011"
    elif problem == "future":
        dates[1] = "10/12/2026"
    elif problem in ("missing", "text", "zero", "negative", "infinite"):
        values[1] = {"missing":"", "text":"not-data", "zero":"0", "negative":"-1", "infinite":"inf"}[problem]
    path = history(tmp_path, dates, values)
    if problem == "wrong_schema":
        path.write_text("DATE,VIX9D\n01/03/2011,13.21\n")
    elif problem == "empty":
        path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n")
    with pytest.raises(QuantError):
        inputs.parse_history(path, "VIX9D", pd.Timestamp("2026-10-11"))


def histories():
    index = sessions("2015-08-10", "2026-10-05")
    return index, {name:pd.Series(np.arange(len(index), dtype=float)+10, index=index, name=name)
                   for name in inputs.SERIES}


def test_exact_full_nyse_cohort_and_untransformed_input_values():
    index, raw = histories()
    frame = inputs.align_closes(index, raw)
    assert list(frame.columns) == list(inputs.SERIES)
    assert len(frame) == 2805
    for name in inputs.SERIES:
        pd.testing.assert_series_equal(frame[name], raw[name])


@pytest.mark.parametrize("problem", ["missing_day", "wrong_identity", "missing_series", "truncated_cohort", "inf"])
def test_no_fill_masquerading_or_dropped_calendar_sessions(problem):
    index, raw = histories()
    if problem == "missing_day":
        raw["VIX9D"] = raw["VIX9D"].iloc[1:]
    elif problem == "wrong_identity":
        raw["VIX9D"] = raw["VIX9D"].rename("VIX")
    elif problem == "missing_series":
        raw.pop("VIX3M")
    elif problem == "truncated_cohort":
        index = index[1:]
    else:
        raw["VIX9D"].iloc[0] = np.inf
    with pytest.raises(QuantError):
        inputs.align_closes(index, raw)


@pytest.mark.parametrize("field,value", [
    ("new_source_series", ["VIX9D", "VVIX"]), ("VVIX_admitted", True),
    ("new_source_field", "OPEN"), ("strategy_states_targets_or_returns_computed", True),
    ("new_registered_candidates", 2), ("existing_W42_or_forward_rules_changed", True),
    ("source_urls", {"VIX9D":"https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"}),
])
def test_source_scope_closing_field_authority_and_advertised_host_cannot_change(field, value):
    policy = read_json(ROOT / "config/near-term-volatility-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        inputs.validate_policy(policy)
