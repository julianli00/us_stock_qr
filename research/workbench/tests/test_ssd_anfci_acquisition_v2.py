from __future__ import annotations

from urllib.parse import urlencode

import pytest

from us_quant import ssd_anfci_acquisition_v2 as collection
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


def test_first_archive_reuse_never_calls_transport(tmp_path, monkeypatch):
    policy = read_json(collection.original.POLICY)
    dates = ["2015-08-28"]
    monkeypatch.setattr(collection.base, "ROOT", tmp_path)
    monkeypatch.setattr(
        collection.transport, "fetch_one", lambda *args: pytest.fail("No first archive redownload")
    )
    response = tmp_path / "response.bin"
    response.write_bytes(b"retained")
    response_ref = collection.base.reference(response)
    attempt = tmp_path / "attempt.json"
    write_json(attempt, {"created_at": "2026-10-28T16:00:00Z", "dates": dates})
    receipt = tmp_path / "request.json"
    write_json(
        receipt,
        {
            "automatic_retries": 0,
            "exchanges": [
                {
                    "started_at": "2026-10-28T16:00:01Z",
                    "method": "POST",
                    "url": policy["url"],
                    "encoded_body": urlencode(collection.original.form_fields(dates, policy)),
                    "status": 200,
                    "headers": {},
                    "response_body": response_ref,
                }
            ],
        },
    )
    correction = {
        "retained_attempt": collection.base.reference(attempt),
        "retained_request": collection.base.reference(receipt),
        "retained_response": response_ref,
    }
    saved, _ = collection.retained_first(correction, dates, policy)
    assert saved["dates"] == dates
    with pytest.raises(QuantError):
        collection.retained_first(correction, ["2015-09-30"], policy)
