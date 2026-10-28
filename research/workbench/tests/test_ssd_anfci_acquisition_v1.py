from __future__ import annotations

import io

import pytest

from us_quant import ssd_anfci_acquisition_v1 as acquisition
from us_quant.config import QuantError
from us_quant.storage import read_json


class Response(io.BytesIO):
    def __init__(self, status, body=b"source bytes", headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.get_method(), request.full_url, request.data, timeout))
        return next(self.responses)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "runtime_guard", lambda: None)
    monkeypatch.setattr(acquisition.base, "ROOT", tmp_path)
    directory = tmp_path / "attempt"
    directory.mkdir()
    return directory, read_json(acquisition.source.POLICY)


def test_refusal_is_one_submission_with_retained_body_and_no_retry(setup):
    directory, policy = setup
    opener = Opener([Response(403, b"denied")])
    with pytest.raises(QuantError, match="HTTP403"):
        acquisition.fetch_one(directory, ["2015-08-28"], policy, opener)
    assert len(opener.calls) == 1
    assert (directory / "response-00.bin").read_bytes() == b"denied"
    assert read_json(directory / "exchange-00.json")["status"] == 403


def test_normal_source_redirect_is_get_not_repeated_post(setup):
    directory, policy = setup
    opener = Opener(
        [
            Response(303, headers={"Location": "/download/source.zip"}),
            Response(200, b"archive"),
        ]
    )
    result = acquisition.fetch_one(directory, ["2015-08-28"], policy, opener)
    assert [row[0] for row in opener.calls] == ["POST", "GET"]
    assert result["automatic_retries"] == 0
    assert result["exchanges"][-1]["status"] == 200


def test_cross_origin_redirect_is_not_followed(setup):
    directory, policy = setup
    opener = Opener([Response(303, headers={"Location": "https://example.com/archive"})])
    with pytest.raises(QuantError, match="secure origin"):
        acquisition.fetch_one(directory, ["2015-08-28"], policy, opener)
    assert len(opener.calls) == 1


def test_response_size_limit_preserves_partial_without_retry(setup):
    directory, policy = setup
    policy["maximum_archive_bytes"] = 3
    opener = Opener([Response(200, b"oversized")])
    with pytest.raises(QuantError, match="byte limit"):
        acquisition.fetch_one(directory, ["2015-08-28"], policy, opener)
    assert len(opener.calls) == 1
    assert (directory / "response-00.bin").read_bytes() == b"oversized"
