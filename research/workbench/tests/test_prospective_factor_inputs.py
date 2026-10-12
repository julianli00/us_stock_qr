from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.prospective_data import ProspectiveArchive, fingerprint as parent_fingerprint
from us_quant.prospective_factor_inputs import ExpandedArchive, validate_policy
from us_quant.storage import file_digest, read_json, write_json, write_text_atomic

ROOT = Path(__file__).parents[1]


def moment(value):
    return datetime.fromisoformat(value)


@pytest.fixture
def policy():
    return read_json(ROOT / "config/prospective-factor-inputs.json")


@pytest.fixture
def parent(tmp_path):
    policy = read_json(ROOT / "config/prospective-data.json")
    write_json(tmp_path / "config/prospective-data.json", policy)
    archive = ProspectiveArchive(tmp_path / "data/prospective-market-v1", policy)
    archive.initialize(moment("2026-10-10T08:00:00+00:00"))
    return archive


def parent_fetch(policy, day, output):
    write_json(output / "synthetic.json", {"session": str(day.date()), "synthetic_test_only": True})
    return {
        "session": str(day.date()),
        "all_required_sources_verified": True,
        "strategy_returns_calculated": False,
    }


def extra_fetch(archive):
    def fetch(policy, day, output):
        reference = archive.parent_reference(day)
        acquired = max(
            pd.Timestamp(reference["observed_at"]),
            pd.Timestamp(read_json(archive.directory / "registration.json")["registered_at"]),
        ).isoformat()
        write_json(output / "extra.json", {"session": str(day.date()), "synthetic_test_only": True})
        return {
            "session": str(day.date()),
            "parent_reference": reference,
            "additional_symbols": policy["additional_symbols"],
            "additional_macro_series": policy["additional_macro_series"],
            "additional_quote_sources": [
                {"symbol": symbol, "retrieved_at": acquired}
                for symbol in policy["additional_symbols"]
            ],
            "additional_macro_source": {"series": "BAA10Y", "retrieved_at": acquired},
            "all_required_sources_verified": True,
            "strategy_returns_calculated": False,
        }

    return fetch


def initialized(tmp_path, policy, parent):
    parent.collect(parent_fetch, lambda: moment("2026-10-10T08:00:00+00:00"))
    archive = ExpandedArchive(tmp_path / "data/expanded", policy, root=tmp_path)
    archive.initialize(moment("2026-10-10T09:00:00+00:00"))
    return archive


def test_expanded_baseline_and_duplicate_are_not_new_observations_or_parent_changes(
    tmp_path, policy, parent
):
    archive = initialized(tmp_path, policy, parent)
    before, original_code = parent.status(), parent_fingerprint()
    now = moment("2026-10-10T09:00:00+00:00")
    result = archive.collect(extra_fetch(archive), lambda: now)
    assert result["baseline_session"] == "2026-10-09"
    assert result["complete_snapshots"] == 1 and result["observations_after_baseline"] == 0
    assert not result["expanded_baseline_is_independent_strategy_evidence"]
    assert not result["new_catalog_definitions_registered_by_collection"]
    duplicate = archive.collect(lambda *args: pytest.fail("must not refetch"), lambda: now)
    assert duplicate["action"] == "already_collected"
    assert duplicate["receipt_chain_head"] == result["receipt_chain_head"]
    assert parent.status() == before and parent_fingerprint() == original_code


def test_mismatched_session_parent_blocks_new_capture_without_fetching(tmp_path, policy, parent):
    archive = initialized(tmp_path, policy, parent)
    with pytest.raises(QuantError, match="matching completed-session"):
        archive.collect(
            lambda *args: pytest.fail("parent is stale"),
            lambda: moment("2026-10-13T01:00:00+00:00"),
        )
    assert archive.status()["complete_snapshots"] == 0
    failures = list((archive.directory / "attempts").glob("*/failure.json"))
    assert len(failures) == 1 and not read_json(failures[0])["observation_recorded"]


def test_missing_expanded_session_is_preserved_not_backfilled(tmp_path, policy, parent):
    archive = initialized(tmp_path, policy, parent)
    archive.collect(extra_fetch(archive), lambda: moment("2026-10-10T09:00:00+00:00"))
    later = moment("2026-10-14T01:00:00+00:00")
    parent.collect(parent_fetch, lambda: later)
    result = archive.collect(extra_fetch(archive), lambda: later)
    assert result["latest_session"] == "2026-10-13"
    assert result["missed_sessions"] == ["2026-10-12"]
    assert result["observations_after_baseline"] == 1
    assert not result["contiguous_since_baseline"]
    assert not (archive.directory / "receipts/2026-10-12.json").exists()


@pytest.mark.parametrize("problem", ["parent_hash", "strategy_returns", "symbols"])
def test_wrong_provenance_or_success_shaped_scope_does_not_record_a_snapshot(
    tmp_path, policy, parent, problem
):
    archive = initialized(tmp_path, policy, parent)
    before = parent.status()

    def bad_fetch(*args):
        result = extra_fetch(archive)(*args)
        if problem == "parent_hash":
            result["parent_reference"]["receipt_sha256"] = "0" * 64
        elif problem == "strategy_returns":
            result["strategy_returns_calculated"] = True
        else:
            result["additional_symbols"] = ["SPY"]
        return result

    with pytest.raises(QuantError, match="bind the verified parent"):
        archive.collect(bad_fetch, lambda: moment("2026-10-10T09:00:00+00:00"))
    assert archive.status()["complete_snapshots"] == 0
    assert parent.status() == before


def test_updated_parent_or_extension_source_is_detected(tmp_path, policy, parent):
    archive = initialized(tmp_path, policy, parent)
    archive.collect(extra_fetch(archive), lambda: moment("2026-10-10T09:00:00+00:00"))
    extra = next((archive.directory / "attempts").glob("*/extra.json"))
    old = extra.read_bytes()
    write_text_atomic(extra, "{}")
    with pytest.raises(QuantError, match="revised"):
        archive.status()
    extra.write_bytes(old)
    registration = archive.directory / "registration.json"
    saved = read_json(registration)
    changed = {**saved, "extension_collector_sha256": "0" * 64}
    write_json(registration, changed)
    with pytest.raises(QuantError, match="policy/code"):
        archive.status()


def test_future_extra_acquisition_is_rejected_before_receipt_commit_and_can_retry(
    tmp_path, policy, parent
):
    archive = initialized(tmp_path, policy, parent)
    now = moment("2026-10-10T09:00:00+00:00")

    def future_source(*args):
        result = extra_fetch(archive)(*args)
        result["additional_macro_source"]["retrieved_at"] = "2026-10-10T10:00:00+00:00"
        return result

    with pytest.raises(QuantError, match="acquisition time"):
        archive.collect(future_source, lambda: now)
    assert archive.status()["complete_snapshots"] == 0
    assert not (archive.directory / "receipts/2026-10-09.json").exists()
    assert archive.collect(extra_fetch(archive), lambda: now)["complete_snapshots"] == 1


def test_expanded_capture_cannot_finish_after_next_open(tmp_path, policy, parent):
    archive = initialized(tmp_path, policy, parent)
    before = moment("2026-10-12T13:29:00+00:00")
    after = moment("2026-10-12T13:31:00+00:00")
    times = iter((before, after, after))
    with pytest.raises(QuantError, match="deadline"):
        archive.collect(extra_fetch(archive), lambda: next(times))
    assert archive.status()["complete_snapshots"] == 0


def test_late_capture_skips_all_fetches_and_concurrent_same_day_only_fetches_once(
    tmp_path, policy, parent
):
    archive = initialized(tmp_path, policy, parent)
    late = moment("2026-10-12T15:00:00+00:00")
    result = archive.collect(lambda *args: pytest.fail("deadline"), lambda: late)
    assert result["action"] == "missed_preopen_deadline"
    now = moment("2026-10-10T09:00:00+00:00")
    calls = []

    def fetch(*args):
        calls.append("called")
        return extra_fetch(archive)(*args)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: archive.collect(fetch, lambda: now), range(2)))
    assert len(calls) == 1
    assert {result["action"] for result in results} == {"collected", "already_collected"}


def test_no_archive_reset_or_parent_nested_directory(tmp_path, policy, parent):
    archive = initialized(tmp_path, policy, parent)
    with pytest.raises(QuantError, match="reset"):
        archive.initialize()
    with pytest.raises(QuantError, match="separate"):
        ExpandedArchive(parent.directory / "expanded", policy, root=tmp_path)
    with pytest.raises(QuantError, match="separate"):
        ExpandedArchive(tmp_path / "data", policy, root=tmp_path)


@pytest.mark.parametrize("change", ["symbols", "parent", "backfill", "trading"])
def test_expanded_policy_cannot_change_inputs_or_enable_trading(policy, change):
    if change == "symbols":
        policy["additional_symbols"] = ["QLD", "PKW"]
    elif change == "parent":
        policy["parent_archive"] = "data/other"
    elif change == "backfill":
        policy["backfill_missed_observations"] = True
    else:
        policy["order_authority"] = True
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_actual_acquisition_adapter_inherits_bytes_and_only_requests_three_new_sources(
    tmp_path, policy, parent, market_factory, monkeypatch
):
    data = market_factory("2024-10-09", "2026-10-09", tuple(parent.policy["symbols"]))
    now = moment("2026-10-10T08:00:00+00:00")

    def full_parent_fetch(_policy, day, output):
        for name in ("open", "close", "raw_close", "volume"):
            write_text_atomic(output / f"{name}.csv", getattr(data, name).to_csv())
        write_text_atomic(output / "risk_free.csv", data.risk_free.to_frame().to_csv())
        return {
            "session": str(day.date()),
            "quote_sources": [
                {"source": symbol, "retrieved_at": now.isoformat()}
                for symbol in (*parent.policy["symbols"], "^IRX")
            ],
            "macro_sources": [{"series": name} for name in ("T10Y3M", "DFII10")],
            "option_index_sources": [{"symbol": name} for name in ("VIX", "VIX3M")],
            "all_required_sources_verified": True,
            "strategy_returns_calculated": False,
        }

    parent.collect(full_parent_fetch, lambda: now)
    archive = ExpandedArchive(tmp_path / "data/expanded", policy, root=tmp_path)
    archive.initialize(moment("2026-10-10T09:00:00+00:00"))
    extra = market_factory("2024-10-09", "2026-10-09", ("IJR", "PKW"))
    calls = []

    class Response:
        def __init__(self, symbol, url):
            self.url = url
            self.payload = {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "meta": {"symbol": symbol, "currency": "USD", "instrumentType": "ETF"},
                            "timestamp": [
                                int((day.tz_localize("UTC") + pd.Timedelta(hours=16)).timestamp())
                                for day in extra.close.index
                            ],
                            "indicators": {
                                "quote": [{
                                    "open": extra.open[symbol].tolist(),
                                    "close": extra.close[symbol].tolist(),
                                    "high": (np.maximum(extra.open[symbol], extra.close[symbol]) + 1).tolist(),
                                    "low": (np.minimum(extra.open[symbol], extra.close[symbol]) - 1).tolist(),
                                    "volume": extra.volume[symbol].tolist(),
                                }],
                                "adjclose": [{"adjclose": extra.close[symbol].tolist()}],
                            },
                            "events": {},
                        }
                    ],
                }
            }
            self.text = json.dumps(self.payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, **kwargs):
            symbol = url.rsplit("/", 1)[-1]
            calls.append(symbol)
            assert symbol in {"IJR", "PKW"}
            return Response(symbol, url)

    def macro_fetch(url, output):
        calls.append("BAA10Y")
        assert "id=BAA10Y" in url
        dates = pd.bdate_range("2024-09-25", "2026-10-09")
        frame = pd.DataFrame({"observation_date": dates, "BAA10Y": np.linspace(3, 2, len(dates))})
        write_text_atomic(output, frame.to_csv(index=False))

    monkeypatch.setattr("us_quant.prospective_factor_inputs.requests.Session", Client)
    monkeypatch.setattr("us_quant.prospective_factor_inputs.curl_csv", macro_fetch)
    before = parent.status()
    captured = moment("2026-10-10T09:00:00+00:00")
    monkeypatch.setattr("us_quant.prospective_factor_inputs.utc_now", lambda: captured)
    result = archive.collect(clock=lambda: captured)
    assert calls == ["IJR", "PKW", "BAA10Y"]
    assert result["observations_after_baseline"] == 0
    receipt = archive.verify()[0]
    snapshot = archive.directory / receipt["snapshot_path"]
    manifest = read_json(snapshot / "manifest.json")
    for relative, digest in read_json(
        parent.directory / parent.verify()[0]["snapshot_path"] / "manifest.json"
    )["files"].items():
        assert file_digest(snapshot / "parent" / relative) == digest
    expanded = pd.read_csv(snapshot / "close.csv", index_col=0, parse_dates=True)
    assert list(expanded.columns) == [*parent.policy["symbols"], "IJR", "PKW"]
    assert manifest["parent_reference"]["observed_at"] == now.isoformat()
    assert manifest["inherited_quote_sources"][0]["retrieved_at"] == now.isoformat()
    assert parent.status() == before
