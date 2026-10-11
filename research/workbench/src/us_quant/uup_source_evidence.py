from __future__ import annotations

from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser

from us_quant.config import QuantError

IDENTITY = {
    "ticker": "UUP",
    "isin": "US46141D2036",
    "shareClassInceptionDate": "2007-02-20",
}
EXPECTED_SECTIONS = ("holdingsFutures", "holdingsCollateral", "holdings")


class SourcePage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = {}
        self.title_parts = []
        self.in_title = False

    def handle_starttag(self, tag, attrs) -> None:
        values = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "meta" and values.get("name") in IDENTITY:
            key = values["name"]
            if key in self.metadata:
                raise QuantError("The primary fund identity metadata is duplicated.")
            self.metadata[key] = values.get("content")

    def handle_endtag(self, tag) -> None:
        if tag == "title":
            self.in_title = False

    def handle_data(self, value) -> None:
        if self.in_title:
            self.title_parts.append(value)

    @property
    def title(self) -> str:
        return " ".join("".join(self.title_parts).split())


def issuer_identity(document: str) -> dict:
    page = SourcePage()
    page.feed(document)
    if page.metadata != IDENTITY or page.title != (
        "Invesco DB US Dollar Index Bullish Fund | Invesco US"
    ):
        raise QuantError("The source does not identify the exact UUP issuer product.")
    return dict(page.metadata)


def document_status(content: bytes, content_type: str) -> dict:
    if content.startswith(b"%PDF-"):
        if "pdf" not in content_type.lower():
            raise QuantError("Document bytes and declared PDF content type disagree.")
        return {
            "status": "pdf_bytes_present_contents_not_yet_verified",
            "is_validated_statutory_report": False,
            "proves_historical_no_leverage_cap": False,
        }
    if "html" not in content_type.lower():
        raise QuantError("An unexpected statutory-source response needs explicit review.")
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise QuantError("The statutory-source HTML response cannot be decoded.") from exc
    page = SourcePage()
    page.feed(text)
    if page.title != "Country Splash | Invesco US":
        raise QuantError("An unrecognized HTML document cannot be treated as a financial report.")
    return {
        "status": "country_selection_page_not_statutory_report",
        "is_validated_statutory_report": False,
        "html_title": page.title,
        "proves_historical_no_leverage_cap": False,
    }


def numeric(value, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise QuantError(f"An explicit numeric {field} is required.")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise QuantError(f"Invalid numeric {field} in the issuer holdings.") from exc
    if not result.is_finite():
        raise QuantError(f"Nonfinite {field} cannot support a source admission.")
    return result


def holding_id(row: dict) -> str:
    if not isinstance(row, dict):
        raise QuantError("Each scoped or aggregate holding must be an explicit source object.")
    cusip = row.get("cusip")
    if isinstance(cusip, str) and cusip:
        return f"cusip:{cusip}"
    pending = {
        "ticker": "USDPDV",
        "issuerName": "USD Pending Dividends",
        "cusip": None,
        "assetClass": "CASH",
        "securityTypeName": "Currency",
        "percentageOfTotalNetAssets": None,
    }
    if all(key in row and row[key] == value for key, value in pending.items()):
        return "adjustment:USD_pending_dividends"
    raise QuantError(
        "An unidentified holding or unknown null-weight adjustment cannot be accepted."
    )


def snapshot_exposure(snapshot: dict) -> dict:
    if (
        snapshot.get("cusip") != "46141D203"
        or snapshot.get("effectiveDate") != "2026-10-10"
        or snapshot.get("effectiveBusinessDate") != "2026-10-09"
        or any(not isinstance(snapshot.get(key), list) for key in EXPECTED_SECTIONS)
        or not snapshot["holdingsFutures"]
        or not snapshot["holdingsCollateral"]
    ):
        raise QuantError("The fixed UUP snapshot identity, date or required sections changed.")
    derivatives = snapshot["holdingsFutures"]
    if len(derivatives) != 1:
        raise QuantError(
            "The registered source snapshot needs its complete single currency future."
        )
    future = derivatives[0]
    holding_id(future)
    expected = {
        "ticker": "DXZ6",
        "cusip": "DXZ6",
        "assetClass": "DERIVATIVE",
        "securityTypeName": "Currency Future",
        "effectiveBusinessDate": "2026-10-09",
        "contractExpiryDate": "2026-12-14",
        "currency": "USD",
    }
    if any(future.get(key) != value for key, value in expected.items()):
        raise QuantError("The source future is not the exact dated USD index contract.")
    units = numeric(future.get("units"), "units")
    value = numeric(future.get("marketValueBase"), "marketValueBase")
    weight = numeric(future.get("percentageOfTotalNetAssets"), "percentageOfTotalNetAssets")
    if min(units, value, weight) <= 0:
        raise QuantError("The current source must report a positive long futures position.")
    collateral_ids = [holding_id(row) for row in snapshot["holdingsCollateral"]]
    fund_rows = {holding_id(row): row for row in snapshot["holdings"]}
    if (
        len(fund_rows) != len(snapshot["holdings"])
        or len(set(collateral_ids)) != len(collateral_ids)
        or set(fund_rows) != {holding_id(future), *collateral_ids}
    ):
        raise QuantError(
            "The aggregate holdings omit or duplicate a derivative or collateral item."
        )
    nullable_weights = 0
    for row in [future, *snapshot["holdingsCollateral"]]:
        key = holding_id(row)
        if row.get("effectiveBusinessDate") != snapshot["effectiveBusinessDate"]:
            raise QuantError("A holding has a different effective date from the source snapshot.")
        aggregate = fund_rows[key]
        if aggregate.get("effectiveBusinessDate") != snapshot["effectiveBusinessDate"]:
            raise QuantError("An aggregate holding has an inconsistent effective date.")
        for field in ("units", "marketValueBase"):
            if numeric(row.get(field), field) != numeric(aggregate.get(field), field):
                raise QuantError("The fund aggregate disagrees with the scoped holdings sections.")
        if key == "adjustment:USD_pending_dividends":
            nullable_weights += 1
        elif numeric(
            row.get("percentageOfTotalNetAssets"), "percentageOfTotalNetAssets"
        ) != numeric(aggregate.get("percentageOfTotalNetAssets"), "percentageOfTotalNetAssets"):
            raise QuantError("The fund aggregate disagrees with a scoped percentage weight.")
    # Futures and collateral describe one funded exposure, not two additive NAV budgets.
    return {
        "symbol": "UUP",
        "snapshot_effective_date": snapshot["effectiveDate"],
        "snapshot_business_session": snapshot["effectiveBusinessDate"],
        "reported_currency_future_count": len(derivatives),
        "reported_futures_percent_of_total_net_assets": float(weight),
        "reported_futures_weight_exceeds_100_percent": weight > 100,
        "reported_weight_to_NAV_ratio": float(weight / 100),
        "unreported_pending_dividend_weight_rows_preserved": nullable_weights,
        "null_pending_dividend_weight_assumed_zero": False,
        "reported_weight_is_independently_authenticated_notional": False,
        "collateral_weights_added_to_derivative_exposure": False,
        "current_snapshot_is_complete_historical_leverage_coverage": False,
        "snapshot_inside_frozen_qualification_windows": False,
        "asset_leverage_self_defaulted_to_one": False,
        "historical_no_leverage_cap_verified": False,
        "candidate_admitted": False,
        "source_status": "blocked_historical_notional_cap_and_statutory_evidence_unverified",
        "new_strategy_evaluations": 0,
        "new_registered_factor_definitions": 0,
        "investment_objective_verified": False,
        "order_authority": False,
    }
