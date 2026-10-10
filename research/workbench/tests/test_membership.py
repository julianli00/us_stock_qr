from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from us_quant.config import QuantError
from us_quant.membership import load_membership, write_membership_snapshot
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic


def reference(root: Path, rows, *, license_name="MIT", headers=("date", "tickers")):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    write_text_atomic(root / "membership.csv", buffer.getvalue())
    write_text_atomic(root / "LICENSE.txt", "MIT License\nSynthetic test fixture.\n")
    write_text_atomic(root / "UPSTREAM_README.txt", "Synthetic source with unverified coverage.\n")
    write_json(
        root / "manifest.json",
        {
            "repository": "fja05680/sp500",
            "commit_sha": "a" * 40,
            "license": license_name,
            "files": {
                name: {"sha256": file_digest(root / name)}
                for name in ("membership.csv", "LICENSE.txt", "UPSTREAM_README.txt")
            },
        },
    )
    return root


def test_asof_uses_only_the_latest_effective_snapshot(tmp_path):
    root = reference(
        tmp_path / "source",
        [
            ("2020-12-18", "AAPL,FB,SIVB"),
            ("2020-12-21", "AAPL,FB,SIVB,TSLA"),
            ("2022-06-09", "AAPL,META,SIVB,TSLA"),
        ],
    )
    history = load_membership(root)
    assert "TSLA" not in history.on("2020-12-20")["source_labels"]
    assert "TSLA" in history.on("2020-12-21")["source_labels"]
    before = history.on("2021-12-31")
    assert before["source_labels"] == ["AAPL", "FB", "SIVB", "TSLA"]
    assert "META" not in before["source_labels"]
    assert not before["permanent_security_identity_verified"]
    assert not before["stock_strategy_qualified"] and not before["order_authority"]


def test_current_list_is_never_backfilled_into_past_dates(tmp_path):
    root = reference(tmp_path / "source", [("2025-01-02", "AAPL,META,TSLA")])
    with pytest.raises(QuantError, match="precedes"):
        load_membership(root).on("2021-12-31")


def test_stale_last_snapshot_is_not_extended_to_the_present(tmp_path):
    root = reference(tmp_path / "source", [("2021-12-31", "AAPL,FB")])
    with pytest.raises(QuantError, match="stale history"):
        load_membership(root).on("2022-01-03")


def test_later_composition_changes_do_not_change_earlier_members(tmp_path):
    rows = [("2020-01-02", "AAPL,FB,SIVB"), ("2021-01-04", "AAPL,FB,TSLA")]
    first = load_membership(reference(tmp_path / "first", rows))
    later = load_membership(
        reference(tmp_path / "later", rows + [("2022-06-09", "AAPL,META,TSLA")])
    )
    assert first.on("2020-06-01")["source_labels"] == later.on("2020-06-01")["source_labels"]


@pytest.mark.parametrize(
    "rows",
    [
        [("2020-01-02", "AAPL,AAPL")],
        [("2020-01-02", "AAPL"), ("2020-01-02", "MSFT")],
        [("2021-01-04", "AAPL"), ("2020-01-02", "MSFT")],
        [("2020-01-02", "")],
        [("2020-01-02", "AAPL,not a symbol")],
        [("01/02/2020", "AAPL")],
        [],
    ],
)
def test_malformed_or_ambiguous_history_fails_closed(tmp_path, rows):
    root = reference(tmp_path / "source", rows)
    with pytest.raises(QuantError):
        load_membership(root)


def test_current_constituent_file_schema_is_not_accepted_as_history(tmp_path):
    root = reference(tmp_path / "source", [("AAPL", "Apple")], headers=("Symbol", "Security"))
    with pytest.raises(QuantError, match="current ticker list"):
        load_membership(root)


def test_source_tampering_or_missing_attribution_is_detected(tmp_path):
    root = reference(tmp_path / "source", [("2020-01-02", "AAPL,FB")])
    with (root / "membership.csv").open("a") as stream:
        stream.write("2021-01-04,TSLA\n")
    with pytest.raises(QuantError, match="fingerprint"):
        load_membership(root)
    second = reference(tmp_path / "unlicensed", [("2020-01-02", "AAPL")], license_name="unknown")
    with pytest.raises(QuantError, match="license"):
        load_membership(second)


def test_snapshot_preserves_source_and_refuses_overwrite(tmp_path):
    root = reference(
        tmp_path / "source",
        [
            ("2020-01-02", "AAPL,FB"),
            ("2022-06-09", "AAPL,META,TSLA"),
        ],
    )
    output = tmp_path / "query.json"
    result = write_membership_snapshot(root, "2021-12-31", output)
    assert result["source_summary"]["unique_source_labels"] == 4
    assert result["source_labels"] == ["AAPL", "FB"]
    assert read_json(output)["source_commit_sha"] == "a" * 40
    assert not result["source_completeness_independently_verified"]
    with pytest.raises(QuantError, match="overwrite"):
        write_membership_snapshot(root, "2021-12-31", output)
