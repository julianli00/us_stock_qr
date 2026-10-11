from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, market_calendar
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.macro_factor_tilt import align_observations
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import review_market, safe_file, verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import (
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/dollar-risk-guard.json"
CONTROL = ROOT / "config/sector-growth-balance.json"
PRIOR = ROOT / "evidence/sector_growth_balance_20261011_registration.json"
SOURCE = ROOT / "data/dollar-information-source-20261011"
MACRO_SOURCE = ROOT / "data/macro-rate-access-20261010"
SOURCE_URLS = {
    "H10-release-dates.html": "https://www.federalreserve.gov/releases/h10/default.htm",
    "H10-about.html": "https://www.federalreserve.gov/releases/h10/about.htm",
    "H10-technical-qa.html": "https://www.federalreserve.gov/releases/h10/h10_technical_qa.htm",
    "H10-revisions.html": "https://www.federalreserve.gov/econres/notes/ifdp-notes/revisions-to-the-federal-reserve-dollar-indexes-20190115.htm",
    "DTWEXB-source.html": "https://fred.stlouisfed.org/series/DTWEXB",
    "DTWEXBGS-source.html": "https://fred.stlouisfed.org/series/DTWEXBGS",
    "DTWEXB.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXB&cosd=2014-01-01&coed=2026-10-05",
    "DTWEXBGS.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS&cosd=2014-01-01&coed=2026-10-05",
    "H10-releaseDates.json": "https://www.federalreserve.gov/releases/h10/releaseDates.json",
}
CANDIDATES = [
    {"id": "semiconductor_dollar_half_guard", "real_yield_confirmation": False},
    {"id": "semiconductor_dollar_real_yield_half_guard", "real_yield_confirmation": True},
]
CONSERVATIVE_DATE_CONFLICTS = {
    "2016-05-24": "2016-05-23",
    "2016-05-31": "2016-06-02",
    "2016-08-30": "2016-08-29",
    "2017-08-29": "2017-08-28",
    "2017-12-19": "2017-12-18",
    "2018-01-15": "2018-01-16",
    "2018-05-28": "2018-05-29",
    "2018-07-30": "2018-07-31",
    "2019-02-25": "2019-02-27",
    "2019-05-27": "2019-05-28",
    "2019-08-27": "2019-08-26",
    "2020-05-19": "2020-05-18",
}
QUARANTINED_DATE = "2020-01-20"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("data_scope") != "factor_etf_portfolio"
        or policy.get("data_start") != "2015-08-10"
        or policy.get("as_of") != "2026-10-05"
        or policy.get("new_configurations") != 2
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("factor_symbols") != list(FACTORS)
        or policy.get("factor_ids")
        != ["price_momentum", "value_exposure", "quality_exposure", "low_volatility_exposure"]
        or policy.get("original_control_policy") != CONTROL.relative_to(ROOT).as_posix()
        or policy.get("original_control_candidate") != "four_factor_semiconductor_sector50"
        or policy.get("risk_change_sessions") != 63
        or policy.get("risk_off_investment_scale") != 0.50
        or policy.get("cash_reserve") != 0.02
        or policy.get("candidates") != CANDIDATES
        or {
            key: policy.get("dollar_source", {}).get(key)
            for key in (
                "legacy_series",
                "replacement_series",
                "legacy_retirement_date",
                "first_archived_replacement_release",
                "replacement_reindex_release",
                "maximum_observation_age_days",
                "release_hour_new_york",
                "release_minute_new_york",
                "calendar_url",
            )
        }
        != {
            "legacy_series": "DTWEXB",
            "replacement_series": "DTWEXBGS",
            "legacy_retirement_date": "2019-12-31",
            "first_archived_replacement_release": "2019-02-05",
            "replacement_reindex_release": "2019-06-24",
            "maximum_observation_age_days": 14,
            "release_hour_new_york": 16,
            "release_minute_new_york": 15,
            "calendar_url": "https://www.federalreserve.gov/releases/h10/releaseDates.json",
        }
        or policy.get("real_yield_source", {}).get("series") != "DFII10"
        or policy.get("real_yield_source", {}).get("publication_delay_sessions") != 2
        or policy.get("real_yield_source", {}).get("maximum_observation_age_days") != 7
        or policy.get("dollar_source", {}).get("conservative_declared_date_conflicts")
        != CONSERVATIVE_DATE_CONFLICTS
        or policy.get("dollar_source", {}).get("quarantined_archive_dates") != [QUARANTINED_DATE]
        or policy.get("methodology", {}).get("order_authority") is not False
        or policy.get("methodology", {}).get("automatic_live_deployment") is not False
    ):
        raise QuantError("Frozen dollar-risk source, confirmation, timing or allocation changed.")


class H10Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.rows: list[list[dict]] = []
        self.row: list[dict] = []
        self.cell: dict | None = None
        self.table_depth = 0
        self.table_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self.table_depth:
                self.table_depth += 1
            elif "statistics" in (attributes.get("class") or "").split() or (
                "pubtables" in (attributes.get("class") or "").split()
                and attributes.get("title") == "Foreign Exchange Rates -- H.10 Weekly"
            ):
                self.table_depth = 1
                self.table_count += 1
        if self.table_depth and tag in ("th", "td"):
            if self.cell is not None:
                raise QuantError("Nested cells in the H10 index table are unsupported.")
            self.cell = {"id": attributes.get("id"), "text": ""}

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if self.table_depth and tag in ("th", "td") and self.cell is not None:
            self.cell["text"] = " ".join(self.cell["text"].split())
            self.row.append(self.cell)
            self.cell = None
        if self.table_depth and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = []
        if self.table_depth and tag == "table":
            self.table_depth -= 1


def parse_release(
    document: str, expected_date: pd.Timestamp, *, conservative_dates: bool = False
) -> pd.DataFrame:
    parser = H10Parser()
    parser.feed(document)
    text = " ".join(" ".join(parser.text).split())
    matches = re.findall(r"Release Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    if len(matches) != 1 or parser.table_count != 1:
        raise QuantError("The actual H10 release date and unique index table are required.")
    release = pd.Timestamp(matches[0])
    if release != expected_date:
        if not conservative_dates or CONSERVATIVE_DATE_CONFLICTS.get(
            str(expected_date.date())
        ) != str(release.date()):
            raise QuantError("The publisher page release date differs from its archive URL.")
    reported_date = release
    release = max(release, expected_date)
    headers = {
        cell["id"]: cell["text"]
        for row in parser.rows
        for cell in row
        if cell["id"] in {f"a{i}" for i in range(3, 8)}
    }
    if set(headers) != {f"a{i}" for i in range(3, 8)}:
        raise QuantError("The H10 table is missing its five actual observation dates.")
    dates = []
    for i in range(3, 8):
        try:
            day = pd.Timestamp(f"{headers[f'a{i}'].replace('.', '')} {release.year}")
        except ValueError as exc:
            raise QuantError("An H10 observation date cannot be parsed.") from exc
        if day >= release:
            day -= pd.DateOffset(years=1)
        if not 0 < (release - day).days <= 14:
            raise QuantError("H10 observation dates do not describe the bounded preceding week.")
        dates.append(day)
    if pd.DatetimeIndex(dates).has_duplicates or dates != sorted(dates):
        raise QuantError("H10 observation dates must be distinct and chronological.")
    available = market_calendar().date_to_session(release + pd.Timedelta(days=1), direction="next")
    released_at = (release + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York")
    records, present = [], set()
    for row in parser.rows:
        if len(row) != 7:
            continue
        label = re.sub(r"^\d+\)\s*", "", row[0]["text"]).upper()
        unit = row[1]["text"]
        if label not in ("BROAD", "BROAD - GOODS ONLY"):
            continue
        if unit == "JAN97=100":
            series, regime = "DTWEXB", "goods_january_1997"
        elif unit == "JAN06=100" and label == "BROAD":
            series = "DTWEXBGS"
            regime = (
                "goods_services_january_2006_month"
                if release >= pd.Timestamp("2019-06-24")
                else "goods_services_january_2006_day"
            )
            if release < pd.Timestamp("2019-02-05"):
                raise QuantError("The replacement index cannot precede its actual introduction.")
        else:
            raise QuantError("An unrecognized broad-dollar mandate or indexation was published.")
        if series in present:
            raise QuantError("The actual H10 release contains duplicate broad-index rows.")
        present.add(series)
        for day, cell in zip(dates, row[2:], strict=True):
            value = cell["text"]
            if value in ("", "ND"):
                continue
            try:
                number = float(value)
            except ValueError as exc:
                raise QuantError("A dollar value is neither numeric nor ND.") from exc
            if not np.isfinite(number) or number <= 0:
                raise QuantError("A published broad-dollar value must be finite and positive.")
            records.append(
                {
                    "series": series,
                    "observation_date": day,
                    "release_date": release,
                    "reported_release_date": reported_date,
                    "advertised_archive_date": expected_date,
                    "date_conflict": reported_date != expected_date,
                    "released_at": released_at.isoformat(),
                    "available_session": available,
                    "unit_regime": regime,
                    "value": number,
                }
            )
    required = {"DTWEXB"} if release < pd.Timestamp("2019-02-05") else {"DTWEXBGS"}
    if release <= pd.Timestamp("2019-12-31"):
        required.add("DTWEXB")
    if not required.issubset(present) or not records:
        raise QuantError("The declared broad-dollar index is absent from the actual release.")
    return pd.DataFrame(records)


def decision_pairs(index: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    if not index.is_unique or not index.is_monotonic_increasing:
        raise QuantError("Dollar risk comparisons need distinct chronological sessions.")
    return {day: index[i - 63] for i, day in enumerate(index) if i >= 63 and is_month_end(day)}


def release_calendar(document: str) -> pd.DatetimeIndex:
    body = json.loads(document)
    if not isinstance(body, list):
        raise QuantError("The official H10 archive requires its actual year/month/date list.")
    dates = []
    for year in body:
        for month in year["Months"]:
            for day in month["Dates"]:
                if not re.fullmatch(r"\d{8}", day) or day[:6] != month["MonthValue"]:
                    raise QuantError("The H10 archive has an inconsistent advertised date.")
                if day[:4] != year["yearValue"]:
                    raise QuantError("The H10 archive year differs from its release dates.")
                dates.append(pd.Timestamp(day))
    result = pd.DatetimeIndex(sorted(dates))
    if result.empty or result.has_duplicates:
        raise QuantError("The official H10 archive dates must be unique and nonempty.")
    return result


def load_vintages(index: pd.DatetimeIndex, policy: dict) -> tuple[pd.DataFrame, dict]:
    validate_policy(policy)
    manifest = read_json(safe_file(SOURCE, "verified-manifest.json"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy_sha256") != file_digest(POLICY)
        or manifest.get("raw_inputs_not_to_be_published") is not True
        or manifest.get("strategy_outcomes_computed") is not False
        or manifest.get("current_fred_values_used_for_targets") is not False
        or manifest.get("source_urls") != SOURCE_URLS
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(manifest.get("releases"), list)
        or not isinstance(manifest.get("quarantined_releases"), list)
        or not (set(SOURCE_URLS) | {"acquisition.json", "archive-acquisition.json"}).issubset(
            manifest.get("files", {})
        )
    ):
        raise QuantError("Dollar vintages need the sealed, pre-outcome source-admission manifest.")
    for name, digest in manifest["files"].items():
        safe_file(SOURCE, name, digest)
    acquisition = read_json(SOURCE / "acquisition.json")
    if len(acquisition["sources"]) != len(SOURCE_URLS) - 1:
        raise QuantError("The original dollar-source acquisition records are incomplete.")
    for record in acquisition["sources"]:
        if (
            record.get("url") != SOURCE_URLS.get(record["path"])
            or record.get("sha256") != manifest["files"].get(record["path"])
            or record.get("http_success") is not True
        ):
            raise QuantError("An actual dollar-source request or source hash was changed.")
    about = (SOURCE / "H10-about.html").read_text()
    published = (SOURCE / "H10-release-dates.html").read_text()
    if (
        "past releases are not revised" not in about
        or "On Mondays at 4:15 p.m." not in published
        or "If Monday falls on a Federal Holiday" not in published
        or 'getJSON("releaseDates.json"' not in published
    ):
        raise QuantError("Publisher vintage, release-time and holiday documentation is required.")
    calendar = release_calendar((SOURCE / "H10-releaseDates.json").read_text())
    required = set()
    for day in set(decision_pairs(index)) | set(decision_pairs(index).values()):
        earlier = calendar[calendar < day]
        if len(earlier) < 2:
            raise QuantError("The official archive cannot cover a required risk comparison.")
        required.update(earlier[-2:])
    releases, frames = set(), []
    for record in manifest["releases"]:
        release = pd.Timestamp(record["date"])
        name = f"releases/{release:%Y%m%d}.html"
        if (
            record.get("path") != name
            or record.get("url") != f"https://www.federalreserve.gov/releases/h10/{release:%Y%m%d}/"
            or record.get("sha256") != manifest["files"].get(name)
            or release not in calendar
            or release in releases
        ):
            raise QuantError("A dollar vintage is not a unique advertised official release.")
        releases.add(release)
        frame = parse_release((SOURCE / name).read_text(), release, conservative_dates=True)
        frame["archive_path"] = name
        frame["archive_sha256"] = record["sha256"]
        frames.append(frame)
    quarantined = manifest["quarantined_releases"]
    if len(quarantined) != 1 or (
        quarantined[0].get("date") != QUARANTINED_DATE
        or quarantined[0].get("reported_release_date") != "2020-01-30"
        or quarantined[0].get("path") != "releases/20200120.html"
        or quarantined[0].get("sha256") != manifest["files"].get("releases/20200120.html")
        or pd.Timestamp(QUARANTINED_DATE) in releases
    ):
        raise QuantError("The one excluded, misdated H10 source must remain exactly quarantined.")
    if not required.issubset(releases | {pd.Timestamp(QUARANTINED_DATE)}):
        raise QuantError("A required actual vintage is missing; do not substitute current history.")
    return pd.concat(frames, ignore_index=True), manifest


def vintage_at(vintages: pd.DataFrame, day: pd.Timestamp, series: str) -> pd.Series:
    eligible = vintages[(vintages["series"] == series) & (vintages["available_session"] <= day)]
    if eligible.empty:
        raise QuantError("No actually released vintage is available before the risk query.")
    selected = eligible.sort_values(["observation_date", "release_date"]).iloc[-1]
    if not 0 <= (day - selected["observation_date"]).days <= 14:
        raise QuantError("The published dollar observation is stale; unbounded carry is forbidden.")
    return selected


def dollar_comparisons(
    vintages: pd.DataFrame, index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.DataFrame]:
    values, provenance = {}, []
    for day, reference in decision_pairs(index).items():
        series = "DTWEXB" if day <= pd.Timestamp("2019-12-31") else "DTWEXBGS"
        current = vintage_at(vintages, day, series)
        prior = vintage_at(vintages, reference, series)
        if current["unit_regime"] != prior["unit_regime"]:
            raise QuantError("Risk comparisons cannot splice different dollar-indexation regimes.")
        values[day] = float(current["value"] / prior["value"] - 1)
        record = {"decision_session": day, "reference_session": reference, "series": series}
        for prefix, selected, query in (("current", current, day), ("reference", prior, reference)):
            record.update({f"{prefix}_{name}": value for name, value in selected.items()})
            record[f"{prefix}_age_calendar_days"] = int((query - selected["observation_date"]).days)
        provenance.append(record)
    if not values:
        raise QuantError("A complete prior risk window and monthly decision are required.")
    return pd.Series(values, name="dollar_change"), pd.DataFrame(provenance).set_index(
        "decision_session"
    )


def load_inputs(
    index: pd.DatetimeIndex, policy: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vintages, _ = load_vintages(index, policy)
    dollar, provenance = dollar_comparisons(vintages, index)
    manifest = read_json(safe_file(MACRO_SOURCE, "verified-manifest.json"))
    rows = [row for row in manifest["sources"] if row["series"] == "DFII10"]
    if len(rows) != 1 or rows[0]["url"] != (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&cosd=2015-01-01&coed=2026-10-05"
    ):
        raise QuantError("Reuse the original exact, official real-yield source.")
    path = safe_file(MACRO_SOURCE, rows[0]["path"], rows[0]["sha256"])
    frame = pd.read_csv(path, na_values=["."])
    if list(frame.columns) != ["observation_date", "DFII10"]:
        raise QuantError("The confirmation source must be the declared actual daily real yield.")
    raw = pd.Series(
        pd.to_numeric(frame["DFII10"], errors="raise").to_numpy(),
        index=pd.to_datetime(frame["observation_date"]),
        name="DFII10",
    )
    real_yield, real_provenance = align_observations(raw, index)
    comparisons = pd.DataFrame(
        {
            "dollar_change": dollar,
            "real_yield_change": (real_yield - real_yield.shift(63)).loc[dollar.index],
        }
    )
    if not np.isfinite(comparisons.to_numpy()).all():
        raise QuantError("Both causal risk changes must be available at every monthly decision.")
    return comparisons, provenance, real_provenance


def build_from_inputs(
    data: MarketData, comparisons: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate_policy(policy)
    original = sector_targets(data, read_json(CONTROL))[policy["original_control_candidate"]]
    if (
        not comparisons.index.equals(original.dropna(how="all").index)
        or set(comparisons.columns) != {"dollar_change", "real_yield_change"}
        or not np.isfinite(comparisons.to_numpy()).all()
    ):
        raise QuantError("Dollar targets need complete risk changes at exactly the original dates.")
    outputs = {}
    assets = [symbol for symbol in original.columns if symbol != "BIL"]
    for candidate in policy["candidates"]:
        targets = original.copy()
        for day in comparisons.index:
            risk_off = comparisons.loc[day, "dollar_change"] > 0
            if candidate["real_yield_confirmation"]:
                risk_off = risk_off and comparisons.loc[day, "real_yield_change"] > 0
            if risk_off:
                before = float(targets.loc[day, assets].sum())
                targets.loc[day, assets] *= policy["risk_off_investment_scale"]
                targets.loc[day, "BIL"] += before * (1 - policy["risk_off_investment_scale"])
        active = targets.dropna(how="all")
        if (active < 0).any().any() or not np.allclose(
            active.sum(axis=1), 0.98, rtol=0, atol=1e-12
        ):
            raise QuantError("Dollar defense cannot short, borrow cash or alter target investment.")
        outputs[candidate["id"]] = targets
    return outputs


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    comparisons, _, _ = load_inputs(data.close.index, policy)
    return build_from_inputs(data, comparisons, policy)


def prepare(output: Path) -> dict:
    policy = read_json(POLICY)
    validate_policy(policy)
    prior = read_json(PRIOR)
    original = prior["candidates"][0]["spec"]
    data = review_market({"market": original["market"]}, ROOT)
    actual = verified_etf_market(prior["readiness"], ROOT)
    for name in ("open", "close", "raw_close", "volume"):
        if (
            not getattr(data, name).index.equals(getattr(actual, name).index)
            or not getattr(data, name).columns.equals(getattr(actual, name).columns)
            or not np.allclose(getattr(data, name), getattr(actual, name), rtol=1e-10, atol=1e-9)
        ):
            raise QuantError("Dollar research must retain the audited sector/factor market.")
    if not np.allclose(data.risk_free, actual.risk_free, rtol=0, atol=1e-12):
        raise QuantError("Dollar research cannot substitute the original frozen risk-free data.")
    comparisons, provenance, real_provenance = load_inputs(data.close.index, policy)
    output = output if output.is_absolute() else ROOT / output
    if output.is_symlink() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Dollar research evidence must stay in the isolated workbench.")
    new_output_directory(output)
    for name, frame in (
        ("risk-comparisons", comparisons),
        ("dollar-vintage-provenance", provenance),
        ("real-yield-availability", real_provenance),
    ):
        write_text_atomic(output / f"{name}.csv", frame.to_csv(float_format="%.17g"))
    manifest = read_json(SOURCE / "verified-manifest.json")
    frozen = {
        **original["frozen_files"],
        POLICY.relative_to(ROOT).as_posix(): file_digest(POLICY),
        Path(__file__).relative_to(ROOT).as_posix(): file_digest(Path(__file__)),
        "src/us_quant/macro_factor_tilt.py": file_digest(
            ROOT / "src/us_quant/macro_factor_tilt.py"
        ),
        (SOURCE / "verified-manifest.json").relative_to(ROOT).as_posix(): file_digest(
            SOURCE / "verified-manifest.json"
        ),
        (MACRO_SOURCE / "verified-manifest.json").relative_to(ROOT).as_posix(): file_digest(
            MACRO_SOURCE / "verified-manifest.json"
        ),
        (MACRO_SOURCE / "DFII10-curl-response.csv").relative_to(ROOT).as_posix(): file_digest(
            MACRO_SOURCE / "DFII10-curl-response.csv"
        ),
        PRIOR.relative_to(ROOT).as_posix(): file_digest(PRIOR),
    }
    frozen.update(
        {
            (SOURCE / name).relative_to(ROOT).as_posix(): digest
            for name, digest in manifest["files"].items()
        }
    )
    readiness = {**prior["readiness"], "checked_at": utc_now()}
    specs = []
    for candidate in policy["candidates"]:
        spec = {
            **original,
            "id": candidate["id"],
            "configuration": candidate,
            "frozen_files": frozen,
        }
        write_json(output / f"{candidate['id']}-spec.json", spec)
        specs.append(spec)
    audit = {
        "decision_count": len(comparisons),
        "actual_h10_archives": len(manifest["releases"]),
        "quarantined_h10_archives": len(manifest["quarantined_releases"]),
        "conservative_date_conflict_queries": int(
            provenance[["current_date_conflict", "reference_date_conflict"]].to_numpy().sum()
        ),
        "source_series_counts": provenance["series"].value_counts().to_dict(),
        "maximum_dollar_observation_age_days": int(
            provenance[["current_age_calendar_days", "reference_age_calendar_days"]]
            .to_numpy()
            .max()
        ),
        "maximum_real_yield_observation_age_days": int(real_provenance["age_calendar_days"].max()),
        "all_vintages_available_no_later_than_their_query": bool(
            (provenance["current_available_session"] <= provenance.index).all()
            and (provenance["reference_available_session"] <= provenance["reference_session"]).all()
        ),
        "all_releases_strictly_before_their_query": bool(
            (provenance["current_release_date"] < provenance.index).all()
            and (provenance["reference_release_date"] < provenance["reference_session"]).all()
        ),
        "cross_method_or_indexation_level_comparisons": int(
            (provenance["current_unit_regime"] != provenance["reference_unit_regime"]).sum()
        ),
        "current_fred_values_used_for_targets": False,
        "dollar_vintage_provenance_sha256": file_digest(output / "dollar-vintage-provenance.csv"),
        "risk_comparisons_sha256": file_digest(output / "risk-comparisons.csv"),
        "real_yield_availability_sha256": file_digest(output / "real-yield-availability.csv"),
        "real_yield_current_historical_download_not_vintage_proof": True,
        "existing_prospective_archives_changed": False,
        "raw_or_aligned_macro_values_published": False,
        "strategy_outcomes_computed": False,
    }
    write_json(output / "availability-summary.json", audit)
    write_json(output / "readiness.json", readiness)
    return {"specs": specs, "readiness": readiness, "availability_audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare two fixed vintage-aware dollar-risk guards."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/dollar-risk-prepared-20261011")
    args = parser.parse_args()
    try:
        result = prepare(args.output)
        print(
            json.dumps(
                {
                    "prepared": [spec["id"] for spec in result["specs"]],
                    "availability_audit": result["availability_audit"],
                    "strategy_outcomes_computed": False,
                }
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Dollar research blocked: {exc}\n")


if __name__ == "__main__":
    main()
