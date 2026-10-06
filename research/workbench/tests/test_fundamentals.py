from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.fundamentals import annual_quality_features, available_at, write_features
from us_quant.storage import read_json, write_json


def fact(value, end="2020-12-31", filed="2021-02-26", start=None, accession=None):
    row = {
        "val": value,
        "end": end,
        "filed": filed,
        "form": "10-K",
        "accn": accession or "0000000001-21-000001",
    }
    if start is not None:
        row["start"] = start
    return row


@pytest.fixture
def company():
    return {
        "cik": 1,
        "entityName": "Synthetic test issuer",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            fact(100, end="2019-12-31"),
                            fact(120),
                            fact(130, end="2021-03-31", filed="2021-05-07", start=None),
                        ]
                    }
                },
                "Liabilities": {"units": {"USD": [fact(60)]}},
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            fact(11, start="2020-01-01"),
                            fact(4, start="2020-10-01"),
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            fact(16.5, start="2020-01-01"),
                            fact(10, end="2020-09-30", start="2020-01-01"),
                        ]
                    }
                },
            }
        },
    }


def test_filing_date_not_reporting_end_controls_availability(company):
    with pytest.raises(QuantError, match="No available"):
        annual_quality_features(company, "2021-02-26")
    with pytest.raises(QuantError, match="No available"):
        annual_quality_features(company, "2021-02-28")
    result = annual_quality_features(company, "2021-03-01")
    assert result["features"] == pytest.approx(
        {
            "return_on_average_assets": 0.1,
            "cash_return_on_average_assets": 0.15,
            "cash_minus_income_over_average_assets": 0.05,
            "liabilities_to_assets": 0.5,
        }
    )
    cutoff = pd.Timestamp(result["decision_cutoff_utc"])
    assert all(pd.Timestamp(row["available_at"]) <= cutoff for row in result["provenance"].values())
    assert not result["strategy_qualified"] and not result["order_authority"]


def test_future_restatement_values_cannot_change_earlier_features(company):
    before = annual_quality_features(company, "2021-03-01")
    changed = deepcopy(company)
    for data in changed["facts"]["us-gaap"].values():
        later = deepcopy(data["units"]["USD"][-1])
        later.update({"filed": "2022-01-28", "accn": "0000000001-22-000001", "val": float("nan")})
        data["units"]["USD"].append(later)
    after = annual_quality_features(changed, "2021-03-01")
    assert before["features"] == after["features"]
    assert before["provenance"] == after["provenance"]
    assert after["record_audit"]["Assets"]["not_yet_available"] > 0


def test_unavailable_future_records_do_not_need_a_future_exchange_calendar(company):
    expected = annual_quality_features(company, "2021-03-01")
    company["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
        fact(float("nan"), start="2020-01-01", filed="2040-01-03")
    )
    actual = annual_quality_features(company, "2021-03-01")
    assert expected["features"] == actual["features"]
    assert expected["provenance"] == actual["provenance"]


def test_old_comparative_period_does_not_need_an_old_exchange_calendar(company):
    expected = annual_quality_features(company, "2021-03-01")
    company["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(
        fact(10, end="1999-12-31", filed="2009-03-02")
    )
    actual = annual_quality_features(company, "2021-03-01")
    assert expected["features"] == actual["features"]


def test_restatement_is_used_only_once_publication_lag_has_elapsed(company):
    records = company["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    records.append(
        fact(22, start="2020-01-01", filed="2021-03-05", accession="0000000001-21-000002")
    )
    before = annual_quality_features(company, "2021-03-05")
    after = annual_quality_features(company, "2021-03-08")
    assert before["features"]["return_on_average_assets"] == pytest.approx(0.1)
    assert after["features"]["return_on_average_assets"] == pytest.approx(0.2)


def test_quarter_and_ytd_flows_are_not_summed_with_annual_values(company):
    result = annual_quality_features(company, "2021-05-10")
    assert result["annual_period_start"] == "2020-01-01"
    assert result["provenance"]["net_income"]["value"] == 11
    assert result["provenance"]["operating_cash_flow"]["value"] == 16.5
    assert result["provenance"]["assets"]["value"] == 120


def test_holiday_release_waits_until_a_complete_following_session():
    assert available_at("2021-04-01") == pd.Timestamp("2021-04-05 20:00Z")
    assert available_at("2021-04-02") == pd.Timestamp("2021-04-05 20:00Z")


@pytest.mark.parametrize(
    "problem",
    [
        "missing_cash_flow",
        "wrong_unit",
        "unaligned_cash_flow",
        "missing_prior_assets",
        "conflicting_restatement",
        "invalid_value",
        "invalid_accession",
        "negative_assets",
    ],
)
def test_invalid_or_ambiguous_features_fail_explicitly(company, problem):
    tags = company["facts"]["us-gaap"]
    if problem == "missing_cash_flow":
        del tags["NetCashProvidedByUsedInOperatingActivities"]
    elif problem == "wrong_unit":
        tags["Liabilities"]["units"] = {"EUR": [fact(60)]}
    elif problem == "unaligned_cash_flow":
        tags["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"] = [
            fact(16.5, start="2020-01-02")
        ]
    elif problem == "missing_prior_assets":
        tags["Assets"]["units"]["USD"].pop(0)
    elif problem == "conflicting_restatement":
        tags["NetIncomeLoss"]["units"]["USD"].append(
            fact(50, start="2020-01-01", accession="0000000001-21-000003")
        )
    elif problem == "invalid_value":
        tags["Liabilities"]["units"]["USD"][0]["val"] = True
    elif problem == "invalid_accession":
        tags["Liabilities"]["units"]["USD"][0]["accn"] = "unknown"
    else:
        tags["Assets"]["units"]["USD"][1]["val"] = -120
    with pytest.raises(QuantError):
        annual_quality_features(company, "2021-03-01")


def test_identical_duplicate_facts_are_deterministic(company):
    expected = annual_quality_features(company, "2021-03-01")
    for data in company["facts"]["us-gaap"].values():
        data["units"]["USD"] *= 2
        data["units"]["USD"].reverse()
    actual = annual_quality_features(company, "2021-03-01")
    assert expected["features"] == actual["features"]
    assert expected["provenance"] == actual["provenance"]


def test_stale_financials_are_not_silently_carried_forward(company):
    with pytest.raises(QuantError, match="stale"):
        annual_quality_features(company, "2022-10-03")


def test_real_negative_earnings_are_not_replaced_with_zero(company):
    company["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"][0]["val"] = -11
    result = annual_quality_features(company, "2021-03-01")
    assert result["features"]["return_on_average_assets"] == pytest.approx(-0.1)


def test_large_finite_asset_values_do_not_overflow_the_average(company):
    tags = company["facts"]["us-gaap"]
    tags["Assets"]["units"]["USD"][0]["val"] = 1e308
    tags["Assets"]["units"]["USD"][1]["val"] = 1e308
    tags["NetIncomeLoss"]["units"]["USD"][0]["val"] = 1e307
    tags["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"][0]["val"] = 1.65e307
    result = annual_quality_features(company, "2021-03-01")
    assert result["features"]["return_on_average_assets"] == pytest.approx(0.1)
    assert result["features"]["cash_return_on_average_assets"] == pytest.approx(0.165)


def test_feature_artifact_preserves_provenance_and_cannot_be_overwritten(company, tmp_path):
    source, output = tmp_path / "facts.json", tmp_path / "features.json"
    write_json(source, company)
    first = write_features(source, "2021-03-01", output)
    assert read_json(output)["source_sha256"] == first["source_sha256"]
    assert not first["objective_verified"] and not first["order_authority"]
    with pytest.raises(QuantError, match="overwrite"):
        write_features(source, "2021-03-01", output)
