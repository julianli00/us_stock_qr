from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import skew_inputs as inputs
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


def raw(tmp_path, dates=("01/02/1990", "01/03/1990"), values=("98.5", "170.0")):
    path = tmp_path / "SKEW.csv"
    path.write_text("DATE,SKEW\n"+"".join(f"{day},{value}\n" for day, value in zip(dates, values, strict=True)))
    return path


def dashboard():
    return "var symbol = 'SKEW'; CTX[cSymbol] = '<h2>Cboe SKEW Index (SKEW)</h2><p>30-day horizon, out-of-the-money options</p>" \
        f'<a href="{inputs.URL}">Daily</a><a href="{inputs.WHITEPAPER}">Whitepaper</a>'+"';"


def histories():
    index = sessions("2014-01-02", "2026-10-05")
    result = {name:pd.Series(125.0 if name == "SKEW" else 20.0, index=index, name=name) for name in inputs.SERIES}
    result["SKEW"] = result["SKEW"].drop(pd.to_datetime(list(inputs.MISSING)))
    return result


def test_actual_native_skew_index_points_are_not_limited_to_typical_publisher_range(tmp_path):
    source = inputs.parse_history(raw(tmp_path), "SKEW", pd.Timestamp("2026-10-11"))
    assert source.tolist() == [98.5, 170.0]
    assert source.name == "SKEW" and source.index[0] == pd.Timestamp("1990-01-02")


@pytest.mark.parametrize("problem", ["duplicate", "reordered", "weekend", "future", "nan", "zero", "negative",
                                   "infinite", "text", "wrong_field", "empty"])
def test_invalid_or_misidentified_source_never_defaults_to_normality_or_no_risk(tmp_path, problem):
    dates, values = ["01/02/1990", "01/03/1990"], ["120.0", "130.0"]
    if problem == "duplicate":
        dates[1] = dates[0]
    elif problem == "reordered":
        dates.reverse()
    elif problem == "weekend":
        dates[1] = "01/06/1990"
    elif problem == "future":
        dates[1] = "10/12/2026"
    elif problem in ("nan", "zero", "negative", "infinite", "text"):
        values[1] = {"nan":"", "zero":"0", "negative":"-1", "infinite":"inf", "text":"missing"}[problem]
    path = raw(tmp_path, dates, values)
    if problem == "wrong_field":
        path.write_text("DATE,CLOSE\n01/02/1990,120.0\n")
    elif problem == "empty":
        path.write_text("DATE,SKEW\n")
    with pytest.raises(QuantError):
        inputs.parse_history(path, "SKEW", pd.Timestamp("2026-10-11"))


def test_scoped_resource_data_are_parsed_without_executing_javascript():
    assert inputs.dashboard_resources(dashboard()) == {"history":inputs.URL, "whitepaper":inputs.WHITEPAPER}


@pytest.mark.parametrize("problem", ["symbol", "duplicate_block", "missing_link", "changed_host", "horizon"])
def test_unscoped_or_changed_primary_advertisements_cannot_supply_a_guessed_endpoint(problem):
    html = dashboard()
    if problem == "symbol":
        html = html.replace("var symbol = 'SKEW'", "var symbol = 'VIX'")
    elif problem == "duplicate_block":
        html += html
    elif problem == "missing_link":
        html = html.replace(inputs.WHITEPAPER, "missing")
    elif problem == "changed_host":
        html = html.replace("cdn-api.cboe.com", "cdn.cboe.com")
    else:
        html = html.replace("30-day horizon", "weekly")
    with pytest.raises(QuantError):
        inputs.dashboard_resources(html)


def test_known_daily_and_monthly_missing_quotes_remain_explicit_not_a_safe_state():
    context = inputs.align_context(histories())
    monthly = inputs.monthly_queries(context)
    assert len(context) == 3208 and context["SKEW"].isna().sum() == 4
    assert len(monthly) == 131 and monthly["source_available"].sum() == 130
    assert pd.isna(monthly.loc["2024-11-29", "SKEW"])
    assert monthly.loc["2024-11-29", "source_available"] == False
    assert monthly.loc["2024-11-29", "source_reason"]


@pytest.mark.parametrize("problem", ["unknown_gap", "filled_known", "missing_vix", "identity"])
def test_unknown_or_filled_source_gap_cannot_be_relabelled_readiness(problem):
    values = histories()
    if problem == "unknown_gap":
        values["SKEW"] = values["SKEW"].iloc[1:]
    elif problem == "filled_known":
        values["SKEW"] = pd.Series(125.0, index=sessions("2014-01-02", "2026-10-05"), name="SKEW")
    elif problem == "missing_vix":
        values["VIX"] = values["VIX"].iloc[1:]
    else:
        values["SKEW"] = values["SKEW"].rename("VIX")
    with pytest.raises(QuantError):
        inputs.align_context(values)


@pytest.mark.parametrize("field,value", [("full_source_readiness", True), ("full_monthly_query_readiness", True),
                                        ("known_missing_skew_daily_dates", []), ("known_missing_skew_monthly_queries", []),
                                        ("physical_crash_probabilities_or_option_trading_returns_computed", True),
                                        ("strategy_states_targets_or_returns_computed", True)])
def test_exact_source_scope_cannot_create_success_shaped_probability_or_readiness(field, value):
    policy = read_json(ROOT / "config/skew-inputs.json")
    policy[field] = value
    with pytest.raises(QuantError):
        inputs.validate_policy(policy)
