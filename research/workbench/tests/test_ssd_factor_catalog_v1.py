from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from us_quant import ssd_factor_catalog_v1 as catalog
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import validate_factor
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


def csv_panel(symbols, dates=("2020-01-02", "2020-01-03"), number="1.0000000000000001"):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["", *symbols])
    for day in dates:
        writer.writerow([day, *[number for _ in symbols]])
    return output.getvalue()


def test_extension_preserves_original_number_text_and_appends_only_two_columns():
    original = csv_panel(catalog.EXISTING)
    additions = csv_panel((*catalog.CORE, *catalog.ADDED), number="2.2222222222222222")
    result = list(csv.reader(io.StringIO(catalog.append_csv(original, additions))))
    assert result[0] == ["", *catalog.EXISTING, *catalog.ADDED]
    for row, prior in zip(result[1:], list(csv.reader(io.StringIO(original)))[1:], strict=True):
        assert row[:-2] == prior
        assert row[-2:] == ["2.2222222222222222"] * 2


@pytest.mark.parametrize("kind", ["columns", "dates", "short", "long", "ragged", "empty"])
def test_extension_rejects_changed_schema_or_calendar(kind):
    original = csv_panel(catalog.EXISTING)
    additions = csv_panel((*catalog.CORE, *catalog.ADDED))
    if kind == "columns":
        original = csv_panel((*catalog.CORE, "TLT", "QQQ", "XLK", "OTHER"))
    elif kind == "dates":
        additions = additions.replace("2020-01-03", "2020-01-06")
    elif kind == "short":
        additions = csv_panel((*catalog.CORE, *catalog.ADDED), dates=("2020-01-02",))
    elif kind == "long":
        additions += additions.splitlines()[-1] + "\n"
    elif kind == "ragged":
        additions = additions.replace("2020-01-02,", "2020-01-02,extra,")
    else:
        original = csv_panel(catalog.EXISTING, dates=())
        additions = csv_panel((*catalog.CORE, *catalog.ADDED), dates=())
    with pytest.raises(QuantError):
        catalog.append_csv(original, additions)


def test_actual_freshness_not_relabelled_old_readiness():
    catalog.fresh_time("2026-10-27T21:42:25Z", "2026-10-28T00:00:00Z")
    catalog.fresh_time("2026-10-20T00:00:00Z", "2026-10-28T00:00:00Z")
    for checked in ("2026-10-10T21:31:12Z", "2026-10-29T00:00:00Z", "2026-10-27", "NaT"):
        with pytest.raises(QuantError):
            catalog.fresh_time(checked, "2026-10-28T00:00:00Z")


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_readiness_age_days", 20),
        ("maximum_new_definitions_this_cycle", 3),
        ("configuration_count_limit_before_deadline", 6),
        ("schema_version", True),
        ("new_market_requests", True),
        ("order_authority", 0),
        ("strategy_accounts_or_metrics", True),
        ("rewrite_legacy_ledger_or_plans", True),
        ("reuse_exact_original_risk_free_reference", False),
    ],
)
def test_policy_keeps_original_guards_without_reinstating_strategy_quota(field, value):
    policy = read_json(ROOT / "config/ssd-factor-catalog-v1.json")
    catalog.validate_policy(policy)
    policy[field] = value
    with pytest.raises(QuantError):
        catalog.validate_policy(policy)


def test_definitions_use_actual_clock_and_original_semantic_identity():
    proposals = read_json(ROOT / "config/factor-family-expansion.json")
    legacy = {"factors": []}
    result = catalog.definition_records(proposals, legacy, "2026-10-27T22:30:00Z")
    assert len(result) == 2
    assert {row["factor"]["family"] for row in result} == {"size", "share_issuance"}
    assert all(row["registered_at"] == "2026-10-27T22:30:00Z" for row in result)
    assert {row["factor"]["id"]: row["semantic_sha256"] for row in result} == catalog.SEMANTICS
    with pytest.raises(QuantError):
        catalog.definition_records(proposals, legacy, "2026-10-10T00:00:00Z")
    with pytest.raises(QuantError):
        catalog.definition_records(proposals, legacy, "2026-10-28")


@pytest.mark.parametrize("duplicate", ["id", "semantic"])
def test_catalog_does_not_count_existing_definitions_twice(duplicate):
    proposals = read_json(ROOT / "config/factor-family-expansion.json")
    identifier = next(iter(catalog.SEMANTICS))
    legacy = {
        "factors": [
            {
                "id": identifier if duplicate == "id" else "other_existing_id",
                "semantic_sha256": catalog.SEMANTICS[identifier]
                if duplicate == "semantic"
                else "0" * 64,
            }
        ]
    }
    with pytest.raises(QuantError):
        catalog.definition_records(proposals, legacy, "2026-10-27T22:30:00Z")


def sample_market(symbols, value=1.0):
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame(value, index=dates, columns=symbols)
    return MarketData(
        frame.copy(),
        frame.copy(),
        frame.copy(),
        frame.copy(),
        pd.Series(0.0001, index=dates, name="risk_free"),
    )


def test_original_prices_and_risk_free_must_match_exactly_not_just_within_tolerance():
    original = sample_market(catalog.EXISTING)
    additions = sample_market((*catalog.CORE, *catalog.ADDED))
    extended = sample_market((*catalog.EXISTING, *catalog.ADDED))
    catalog.exact_arrays(original, additions, extended)
    for field in (*catalog.FIELDS, "risk_free"):
        changed = copy.deepcopy(extended)
        values = getattr(changed, field)
        if field == "risk_free":
            values.iloc[0] += 1e-16
        else:
            values.iloc[0, 0] += 1e-14
        with pytest.raises(QuantError):
            catalog.exact_arrays(original, additions, changed)
    changed = copy.deepcopy(extended)
    changed.close.loc[:, "IJR"] = 2.0
    with pytest.raises(QuantError):
        catalog.exact_arrays(original, additions, changed)


def test_new_files_are_atomic_private_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog.common, "output_guard", lambda path: None)
    path = tmp_path / "source.csv"
    catalog.new_text(path, "a,b\n1,2\n")
    assert path.read_text() == "a,b\n1,2\n"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(QuantError):
        catalog.new_text(path, "replacement")
    assert path.read_text() == "a,b\n1,2\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["source.csv"]


def original_factors():
    policy = read_json(ROOT / "config/research-program.json")
    discovery = read_json(ROOT / "config/factor-discovery-20261010.json")
    return [*policy["seed_factors"], *discovery["factors"]]


def test_legacy_catalog_opens_only_read_only_immutable_sqlite(tmp_path, monkeypatch):
    rows = [
        {
            "id": factor["id"],
            "semantic_sha": validate_factor(factor),
            "registered_at": "2026-10-10T08:30:10+00:00",
            "body": json.dumps(factor),
        }
        for factor in original_factors()
    ]
    connection = Mock()
    connection.execute.side_effect = [
        Mock(),
        rows,
        Mock(
            fetchone=lambda: {
                "seq": 246,
                "sha": "bb157fa3b5ebcaca94e736137211af58a198790dd92302c38da8f4e25463e65f",
            }
        ),
        Mock(fetchone=lambda: None),
    ]
    connect = Mock(return_value=connection)
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "file_digest", lambda path: catalog.base.LEGACY_SHA256)
    monkeypatch.setattr(catalog, "engine_fingerprint", lambda: catalog.base.LEGACY_ENGINE_SHA256)
    monkeypatch.setattr(catalog.sqlite3, "connect", connect)
    result = catalog.legacy_catalog("2026-W44")
    assert result["definition_count"] == 6 and result["family_count"] == 5
    assert result["written_or_migrated"] is False
    connect.assert_called_once_with(
        (tmp_path / "runtime/research-program.sqlite3").as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    assert [call.args[0] for call in connection.execute.call_args_list] == [
        "PRAGMA query_only=ON",
        "SELECT id,semantic_sha,registered_at,body FROM factors ORDER BY id",
        "SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1",
        "SELECT 1 FROM cycles WHERE week=?",
    ]
    connection.close.assert_called_once()


def test_nonempty_original_wal_stops_before_sqlite_open(tmp_path, monkeypatch):
    wal = tmp_path / "runtime/research-program.sqlite3-wal"
    wal.parent.mkdir()
    wal.write_text("uncommitted")
    connect = Mock()
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "file_digest", lambda path: catalog.base.LEGACY_SHA256)
    monkeypatch.setattr(catalog, "engine_fingerprint", lambda: catalog.base.LEGACY_ENGINE_SHA256)
    monkeypatch.setattr(catalog.sqlite3, "connect", connect)
    with pytest.raises(QuantError, match="WAL"):
        catalog.legacy_catalog("2026-W44")
    connect.assert_not_called()
    assert wal.read_text() == "uncommitted"


def test_catalog_totals_count_economic_families_not_just_definitions():
    legacy = {"factors": [{"family": factor["family"]} for factor in original_factors()]}
    proposals = read_json(ROOT / "config/factor-family-expansion.json")
    added = [{"factor": factor} for factor in proposals["factors"]]
    assert catalog.catalog_totals(legacy, added) == {
        "combined_definition_count": 8,
        "combined_family_count": 7,
    }


@pytest.mark.parametrize("arguments,status", [([], 2), (["--help"], 0), (["register"], 2)])
def test_cli_requires_explicit_admission_action(arguments, status, monkeypatch):
    admit = Mock()
    monkeypatch.setattr(catalog, "admit", admit)
    monkeypatch.setattr("sys.argv", ["ssd-factor-catalog", *arguments])
    with pytest.raises(SystemExit) as exc:
        catalog.main()
    assert exc.value.code == status
    admit.assert_not_called()
