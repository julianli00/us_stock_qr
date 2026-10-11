from __future__ import annotations

from copy import deepcopy

import pytest

from us_quant.config import QuantError
from us_quant.uup_source_evidence import document_status, issuer_identity, snapshot_exposure


def snapshot():
    future = {
        "ticker": "DXZ6",
        "cusip": "DXZ6",
        "units": 4565,
        "marketValueBase": 465730430,
        "percentageOfTotalNetAssets": 102.697813,
        "assetClass": "DERIVATIVE",
        "securityTypeName": "Currency Future",
        "effectiveBusinessDate": "2026-10-09",
        "contractExpiryDate": "2026-12-14",
        "currency": "USD",
    }
    collateral = {
        "cusip": "825252885",
        "units": 455374155.86,
        "marketValueBase": 455374155.86,
        "percentageOfTotalNetAssets": 100.505959,
        "effectiveBusinessDate": "2026-10-09",
    }
    pending = {
        "ticker": "USDPDV",
        "issuerName": "USD Pending Dividends",
        "cusip": None,
        "assetClass": "CASH",
        "securityTypeName": "Currency",
        "units": 0,
        "marketValueBase": 0,
        "percentageOfTotalNetAssets": None,
        "effectiveBusinessDate": "2026-10-09",
    }
    return {
        "cusip": "46141D203",
        "effectiveDate": "2026-10-10",
        "effectiveBusinessDate": "2026-10-09",
        "holdingsFutures": [future],
        "holdingsCollateral": [collateral, pending],
        "holdings": [deepcopy(future), deepcopy(collateral), deepcopy(pending)],
    }


def test_current_reported_future_weight_is_not_added_to_collateral_or_promoted_to_history():
    actual = snapshot_exposure(snapshot())
    assert actual["reported_futures_percent_of_total_net_assets"] == 102.697813
    assert actual["reported_weight_to_NAV_ratio"] == pytest.approx(1.02697813)
    assert actual["reported_futures_weight_exceeds_100_percent"]
    assert not actual["collateral_weights_added_to_derivative_exposure"]
    assert not actual["reported_weight_is_independently_authenticated_notional"]
    assert not actual["current_snapshot_is_complete_historical_leverage_coverage"]
    assert not actual["snapshot_inside_frozen_qualification_windows"]
    assert not actual["asset_leverage_self_defaulted_to_one"]
    assert not actual["candidate_admitted"]
    assert actual["unreported_pending_dividend_weight_rows_preserved"] == 1
    assert not actual["null_pending_dividend_weight_assumed_zero"]


@pytest.mark.parametrize("weight", [50, 100, 102.697813])
def test_a_current_weight_at_or_below_NAV_is_not_a_historical_unleveraged_guarantee(weight):
    source = snapshot()
    source["holdingsFutures"][0]["percentageOfTotalNetAssets"] = weight
    source["holdings"][0]["percentageOfTotalNetAssets"] = weight
    result = snapshot_exposure(source)
    assert result["reported_futures_weight_exceeds_100_percent"] == (weight > 100)
    assert not result["candidate_admitted"]
    assert not result["historical_no_leverage_cap_verified"]


@pytest.mark.parametrize(
    "problem",
    [
        "identity",
        "date",
        "missing",
        "empty",
        "different_date",
        "wrong_contract",
        "mismatch",
        "duplicate",
        "bad_row",
        "missing_cusip",
        "collateral_duplicate",
    ],
)
def test_partial_wrong_or_inconsistent_public_holdings_raise_explicitly(problem):
    source = snapshot()
    if problem == "identity":
        source["cusip"] = "other"
    elif problem == "date":
        source["effectiveBusinessDate"] = "2026-10-08"
    elif problem == "missing":
        source.pop("holdingsCollateral")
    elif problem == "empty":
        source["holdingsFutures"] = []
    elif problem == "different_date":
        source["holdingsCollateral"][0]["effectiveBusinessDate"] = "2026-10-08"
    elif problem == "wrong_contract":
        source["holdingsFutures"][0]["currency"] = "EUR"
    elif problem == "mismatch":
        source["holdings"][0]["units"] += 1
    elif problem == "bad_row":
        source["holdingsCollateral"][0] = None
    elif problem == "missing_cusip":
        source["holdingsCollateral"][0].pop("cusip")
    elif problem == "collateral_duplicate":
        source["holdingsCollateral"].append(source["holdingsCollateral"][0])
    else:
        source["holdings"].append(source["holdings"][0])
    with pytest.raises(QuantError):
        snapshot_exposure(source)


@pytest.mark.parametrize("value", [None, True, "nan", "inf", "unknown", 0, -1])
def test_missing_or_invalid_future_weight_cannot_default_to_an_unleveraged_value(value):
    source = snapshot()
    source["holdingsFutures"][0]["percentageOfTotalNetAssets"] = value
    with pytest.raises(QuantError):
        snapshot_exposure(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticker", "unknown"),
        ("issuerName", "unrelated"),
        ("assetClass", "DERIVATIVE"),
        ("percentageOfTotalNetAssets", 0),
    ],
)
def test_unknown_null_identifier_or_weight_is_not_a_cash_adjustment_fallback(field, value):
    source = snapshot()
    source["holdingsCollateral"][-1][field] = value
    with pytest.raises(QuantError):
        snapshot_exposure(source)


def test_country_selector_or_PDF_signature_is_not_a_validated_statutory_report():
    blocked = document_status(
        b"<html><title>Country Splash | Invesco US</title></html>",
        "text/html;charset=utf-8",
    )
    assert blocked["status"] == "country_selection_page_not_statutory_report"
    assert not blocked["is_validated_statutory_report"]
    pdf = document_status(b"%PDF-1.7\nsynthetic", "application/pdf")
    assert pdf["status"] == "pdf_bytes_present_contents_not_yet_verified"
    assert not pdf["proves_historical_no_leverage_cap"]
    with pytest.raises(QuantError):
        document_status(b"%PDF-1.7\nsynthetic", "text/html")
    with pytest.raises(QuantError):
        document_status(b"<html><title>Unrelated</title></html>", "text/html")


def test_the_fund_identity_cannot_be_inferred_from_a_URL_or_different_fund_page():
    html = (
        "<html><title>Invesco DB US Dollar Index Bullish Fund | Invesco US</title>"
        '<meta name="ticker" content="UUP"><meta name="isin" content="US46141D2036">'
        '<meta name="shareClassInceptionDate" content="2007-02-20"></html>'
    )
    assert issuer_identity(html)["ticker"] == "UUP"
    with pytest.raises(QuantError):
        issuer_identity(html.replace('content="UUP"', 'content="UDN"'))
    with pytest.raises(QuantError):
        issuer_identity(html.replace("</html>", '<meta name="ticker" content="UUP"></html>'))
