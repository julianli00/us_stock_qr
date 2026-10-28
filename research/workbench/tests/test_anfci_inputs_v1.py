from __future__ import annotations

from urllib.parse import urlencode
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant import anfci_inputs_v1 as source
from us_quant.config import QuantError
from us_quant.storage import read_json

DATES = ["2015-08-28", "2015-09-30"]


def archive(tmp_path, *, series="ANFCI", body=None, extra=False):
    readme = (
        f"Series ID: {series}\nChicago Fed Adjusted National Financial Conditions Index\n"
        "Source: Federal Reserve Bank of Chicago\nUnits: Index\n"
        "Frequency: Weekly, Ending Friday\nSeasonal Adjustment: Not Seasonally Adjusted\n"
        "Output Format: Observations by Vintage Date, All Observations\n"
        "Vintage Dates Specified:\n----------\n2015-08-28 2015-09-30\n----------\n"
    )
    body = body or (
        "observation_date,ANFCI_20150828,ANFCI_20150930\n"
        "2015-08-21,-0.12,-0.11\n2015-09-25,.,0\n"
    )
    path = tmp_path / "source.zip"
    with ZipFile(path, "w") as stream:
        stream.writestr("README.txt", readme)
        stream.writestr("vintages_starting_2015-08-28.csv", body)
        if extra:
            stream.writestr("../unexpected.txt", "not allowed")
    return path


def test_actual_signed_values_and_future_missing_cells_are_preserved(tmp_path):
    frame, proof = source.parse_archive(archive(tmp_path), DATES, read_json(source.POLICY))
    assert frame.iloc[0, 0] == -0.12 and frame.iloc[1, 1] == 0
    assert np.isnan(frame.iloc[1, 0])
    assert proof["finite_cells_independently_compared"] == 3
    assert proof["maximum_independent_numeric_difference"] < 1e-12


@pytest.mark.parametrize(
    "kind", ["series", "extra", "column", "future", "weekday", "token", "infinity"]
)
def test_source_corruption_stops_instead_of_filling(tmp_path, kind):
    body = (
        "observation_date,ANFCI_20150828,ANFCI_20150930\n"
        "2015-08-21,-0.12,-0.11\n2015-09-25,.,0\n"
    )
    if kind == "column":
        body = body.replace("ANFCI_20150828", "NFCI_20150828")
    elif kind == "future":
        body = body.replace("2015-09-25,.,0", "2015-09-25,0.2,0")
    elif kind == "weekday":
        body = body.replace("2015-08-21", "2015-08-20")
    elif kind == "token":
        body = body.replace("-0.12", "NA")
    elif kind == "infinity":
        body = body.replace("-0.12", "inf")
    path = archive(
        tmp_path, series="NFCI" if kind == "series" else "ANFCI", body=body, extra=kind == "extra"
    )
    with pytest.raises(QuantError):
        source.parse_archive(path, DATES, read_json(source.POLICY))


def test_freshness_failure_retains_the_actual_stale_reading():
    policy = read_json(source.POLICY)
    values = {
        "2022-12-29": pd.Series([-0.5], index=pd.to_datetime(["2022-12-09"])),
        "2022-12-30": pd.Series([-0.4], index=pd.to_datetime(["2022-12-30"])),
    }
    result = source.assess_queries(values, ["2022-12-30", "2023-01-03"], policy)
    assert result["all_queries_fresh"] is False
    assert result["fresh_query_count"] == 1 and result["stale_query_count"] == 1
    assert result["query_readings"][0]["actual_latest_value"] == -0.5
    assert result["stale_queries"][0]["age_calendar_days"] == 21


def test_actual_post_fields_and_clock_required():
    policy = read_json(source.POLICY)
    row = {
        "started_at": "2026-10-28T15:10:01Z",
        "method": "POST",
        "url": policy["url"],
        "encoded_body": urlencode(source.form_fields(DATES, policy)),
        "status": 200,
        "headers": {},
    }
    receipt = {"exchanges": [row], "automatic_retries": 0}
    assert (
        source.verify_request(receipt, DATES, policy, "2026-10-28T15:10:00Z")["HTTP_status"] == 200
    )
    row["status"] = 403
    with pytest.raises(QuantError):
        source.verify_request(receipt, DATES, policy, "2026-10-28T15:10:00Z")


def test_only_same_origin_server_directed_get_redirect_allowed():
    policy = read_json(source.POLICY)
    first = {
        "started_at": "2026-10-28T15:10:01Z",
        "method": "POST",
        "url": policy["url"],
        "encoded_body": urlencode(source.form_fields(DATES, policy)),
        "status": 303,
        "headers": {"location": "/download/source.zip"},
    }
    final = {
        "started_at": "2026-10-28T15:10:02Z",
        "method": "GET",
        "url": "https://alfred.stlouisfed.org/download/source.zip",
        "encoded_body": None,
        "status": 200,
        "headers": {},
    }
    receipt = {"exchanges": [first, final], "automatic_retries": 0}
    assert (
        source.verify_request(receipt, DATES, policy, "2026-10-28T15:10:00Z")[
            "server_directed_GETs"
        ]
        == 1
    )
    final["url"] = "https://example.com/source.zip"
    with pytest.raises(QuantError):
        source.verify_request(receipt, DATES, policy, "2026-10-28T15:10:00Z")


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_observation_age_calendar_days", 21),
        ("automatic_request_retries", 1),
        ("series", "NFCI"),
        ("current_history_substitution_allowed", True),
    ],
)
def test_source_contract_is_not_relaxed(key, value):
    policy = read_json(source.POLICY)
    source.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        source.validate(policy)
