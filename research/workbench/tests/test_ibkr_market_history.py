from __future__ import annotations

import asyncio
import json
import logging
import struct
from datetime import datetime, timezone

import pytest
from ib_async import Stock

from us_quant import ibkr_market_history as market
from us_quant.config import QuantError

PRIVATE = "SYNTHETIC_ACCOUNT_NEVER_RETAIN"


def wire(*fields: object) -> bytes:
    return market.frame(("\0".join(str(value) for value in fields) + "\0").encode())


def contract_reply(request_id: int) -> bytes:
    return wire(
        10,
        request_id,
        "SPY",
        "STK",
        "",
        0,
        "",
        "SMART",
        "USD",
        "SPY",
        "ARCA",
        "SPY",
        756733,
        0.01,
        "",
        "MKT,LMT",
        "SMART,ARCA",
        1,
        0,
        "Synthetic ETF",
        "ARCA",
        "",
        "",
        "",
        "",
        "US/Eastern",
        "20261026:0930-1600",
        "20261026:0930-1600",
        "",
        0,
        0,
        0,
        "",
        "",
        "26",
        "",
        "ETF",
        0.001,
        0.001,
        0.001,
    )


def history_reply(request_id: int) -> bytes:
    return wire(
        17,
        request_id,
        "20260727",
        "20261027",
        2,
        "20261023",
        100,
        103,
        99,
        102,
        12000,
        101,
        100,
        "20261026",
        102,
        104,
        100,
        103,
        14000,
        102,
        120,
    )


class FixtureTransport:
    def __init__(self, owner: market.MarketHistory, chunks: bool = False):
        self.owner = owner
        self.chunks = chunks
        self.closed = False
        self.written: list[bytes] = []
        self.acknowledge = True
        self.answer_history = True
        self.history_error = False
        self.capture_history_id: int | None = None

    def deliver(self, packet: bytes) -> None:
        loop = asyncio.get_running_loop()
        if self.chunks:
            for byte in packet:
                loop.call_soon(self.owner.client.conn.data_received, bytes([byte]))
        else:
            loop.call_soon(self.owner.client.conn.data_received, packet)

    def write(self, packet: bytes) -> None:
        assert not self.closed
        self.written.append(packet)
        if packet.startswith(b"API\0"):
            self.deliver(wire(178, "20261027 17:00:00 UTC"))
            return
        fields = packet[4:-1].decode().split("\0")
        if fields[0] == "71" and self.acknowledge:
            self.deliver(
                wire(15, 1, PRIVATE + ",")
                + wire(9, 1, 7)
                + wire(15, 1, PRIVATE + ",")
                + wire(4, 2, -1, 2106, PRIVATE, PRIVATE)
            )
        elif fields[0] == "9":
            request_id = int(fields[2])
            self.deliver(contract_reply(request_id) + wire(52, 1, request_id))
        elif fields[0] == "20":
            self.capture_history_id = int(fields[1])
            if self.history_error:
                self.deliver(wire(4, 2, self.capture_history_id, 162, PRIVATE, PRIVATE))
            elif self.answer_history:
                self.deliver(history_reply(self.capture_history_id))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def isolated_clock_and_no_real_network(monkeypatch):
    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 27, 17, 0, tzinfo=timezone.utc)

    async def forbidden_network(*args, **kwargs):
        raise AssertionError("This protocol proof must never open a real connection.")

    monkeypatch.setattr(market, "datetime", FrozenClock)
    monkeypatch.setattr(market.Connection, "connectAsync", forbidden_network)
    monkeypatch.setattr(market, "runtime_guard", lambda: None)


def fixture_session(monkeypatch, *, chunks=False):
    captures: list[tuple[int, bytes, str]] = []
    session = market.MarketHistory(271027, lambda *args: captures.append(args))
    transport = FixtureTransport(session, chunks)

    async def connect(host, port):
        assert (host, port) == ("127.0.0.1", 4002)
        session.client.conn.transport = transport

    monkeypatch.setattr(session.client.conn, "connectAsync", connect)
    return session, transport, captures


@pytest.mark.parametrize("chunks", [False, True])
@pytest.mark.parametrize("what", ["ADJUSTED_LAST", "TRADES"])
def test_actual_acknowledgements_without_account_sync_or_private_logging(
    monkeypatch, caplog, chunks, what
):
    async def exercise():
        session, transport, captures = fixture_session(monkeypatch, chunks=chunks)
        with caplog.at_level(logging.DEBUG):
            await session.connect()
            details = await session.identify("SPY")
            bars = await session.history(details.contract, what)
            await asyncio.sleep(0)
        assert session.client.isReady()
        assert session.client._apiReady is False
        assert session.client._accounts == []
        assert session.audit["discarded_managed_account_frames"] == 2
        assert [row["message"] for row in session.audit["outbound"]] == ["API", "71", "9", "20"]
        assert [bar.date for bar in bars] == ["20261023", "20261026"]
        assert {code for code, _, _ in captures} == {10, 17, 52}
        assert len(transport.written) == 4
        assert PRIVATE not in json.dumps(session.audit)
        assert PRIVATE not in repr(captures)
        assert PRIVATE not in caplog.text
        with pytest.raises(QuantError, match="Account access"):
            session.client.getAccounts()
        session.close()
        assert transport.closed
        assert not session.client.isReady()

    asyncio.run(exercise())


@pytest.mark.parametrize("client_id", [0, -1, True, 2**31, "17"])
def test_nonzero_separate_identity_is_required_before_any_socket(client_id):
    with pytest.raises(QuantError, match="nonzero"):
        market.MarketHistory(client_id, lambda *args: None)


@pytest.mark.parametrize(
    "code",
    [1, 3, 4, 5, 6, 7, 8, 12, 14, 15, 16, 17, 18, 19, 21, 25, 49, 61, 62, 63, 74, 76, 99],
)
def test_every_non_whitelisted_operation_is_blocked_before_queue_and_transport(monkeypatch, code):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        count = len(transport.written)
        with pytest.raises(QuantError):
            session.client.sendMsg(f"{code}\0{PRIVATE}\0")
        with pytest.raises(QuantError):
            session.client.conn.sendMsg(wire(code, PRIVATE))
        assert len(transport.written) == count
        assert not session.client._msgQ
        assert PRIVATE not in json.dumps(session.audit)
        session.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("code", [3, 6, 7, 11, 59, 61, 63, 71, 73, 90, 99, 107, 999])
def test_unexpected_private_or_streaming_frames_stop_without_logging(monkeypatch, caplog, code):
    async def exercise():
        session, transport, captures = fixture_session(monkeypatch)
        await session.connect()
        with caplog.at_level(logging.DEBUG):
            session.client.conn.data_received(wire(code, PRIVATE) + wire(15, 1, PRIVATE))
        assert session.failure is not None
        assert transport.closed
        assert not session.client.isReady()
        assert PRIVATE not in json.dumps(session.audit)
        assert PRIVATE not in repr(captures)
        assert PRIVATE not in caplog.text

    asyncio.run(exercise())


def test_provider_refusal_is_terminal_and_error_text_is_not_retained(monkeypatch, caplog):
    async def exercise():
        session, transport, captures = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        transport.history_error = True
        with caplog.at_level(logging.DEBUG), pytest.raises(QuantError, match="ibkr_status_162"):
            await session.history(details.contract, "ADJUSTED_LAST")
        assert transport.closed
        assert PRIVATE not in repr(captures) + json.dumps(session.audit) + caplog.text
        before = len(transport.written)
        with pytest.raises(QuantError, match="ibkr_status_162"):
            await session.history(details.contract, "TRADES")
        assert len(transport.written) == before

    asyncio.run(exercise())


def test_missing_real_acknowledgement_times_out_instead_of_faking_readiness(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        transport.acknowledge = False
        with pytest.raises(QuantError, match="TimeoutError"):
            await session.connect(timeout=0.01)
        assert transport.closed
        assert not session.client.isReady()
        assert "market_request_ready_at" not in session.audit
        assert session.client._accounts == []
        before = len(transport.written)
        with pytest.raises(QuantError, match="one bounded"):
            await session.connect()
        assert len(transport.written) == before

    asyncio.run(exercise())


def test_history_timeout_is_failure_not_an_empty_success_or_retry(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        transport.answer_history = False
        with pytest.raises(QuantError, match="timeout_no_retry"):
            await session.history(details.contract, "ADJUSTED_LAST", timeout=0.01)
        assert transport.closed
        assert [row["message"] for row in session.audit["outbound"]].count("20") == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "payload",
    [
        struct.pack(">I", market.MAX_FRAME_BYTES + 1),
        struct.pack(">I", 0),
        market.frame(b"not-a-valid-frame"),
        wire(9, 1, PRIVATE),
        wire(4, PRIVATE),
        wire(17, 123, "20261026", "20261027", 2, PRIVATE),
    ],
)
def test_malformed_frames_fail_without_eventkit_argument_logging(monkeypatch, caplog, payload):
    async def exercise():
        session, transport, captures = fixture_session(monkeypatch)
        await session.connect()
        with caplog.at_level(logging.DEBUG):
            session.client.conn.data_received(payload + wire(15, 1, PRIVATE))
        assert session.failure is not None
        assert transport.closed
        assert PRIVATE not in json.dumps(session.audit) + repr(captures) + caplog.text

    asyncio.run(exercise())


def test_unsolicited_history_and_second_outstanding_request_are_not_accepted(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        transport.answer_history = False
        request = asyncio.create_task(session.history(details.contract, "ADJUSTED_LAST"))
        await asyncio.sleep(0)
        before = len(transport.written)
        with pytest.raises(QuantError, match="one market-data request"):
            await session.history(details.contract, "TRADES")
        assert len(transport.written) == before
        session.client.conn.data_received(history_reply(9999))
        with pytest.raises(QuantError):
            await request
        assert transport.closed

    asyncio.run(exercise())


def test_cannot_enable_streaming_paid_snapshots_or_other_history_types(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        before = len(transport.written)
        with pytest.raises(QuantError):
            session.client.reqMktData(10, details.contract, "", True, True, [])
        with pytest.raises(QuantError):
            session.client.reqHistoricalData(
                10, details.contract, "", "3 M", "1 day", "TRADES", True, 1, True, []
            )
        with pytest.raises(QuantError):
            await session.history(details.contract, "FEE_RATE")
        with pytest.raises(QuantError):
            await session.identify("VIG")
        assert len(transport.written) == before
        session.close()

    asyncio.run(exercise())


def test_disconnect_fails_pending_request_and_never_reconnects(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        transport.answer_history = False
        request = asyncio.create_task(session.history(details.contract, "TRADES"))
        await asyncio.sleep(0)
        before = len(transport.written)
        session.client.conn.connection_lost(None)
        with pytest.raises(QuantError, match="disconnected"):
            await request
        assert not session.client.isReady()
        assert len(transport.written) == before

    asyncio.run(exercise())


def test_deadline_is_enforced_at_the_actual_transport_boundary(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()

        class ExpiredClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 31, 15, 59, tzinfo=timezone.utc)

        monkeypatch.setattr(market, "datetime", ExpiredClock)
        before = len(transport.written)
        with pytest.raises(QuantError, match="expired"):
            session.client.reqContractDetails(100, Stock("SPY", "SMART", "USD"))
        assert len(transport.written) == before
        session.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("timeout", [0, -1, None, float("inf"), float("nan"), True, 61])
def test_invalid_timeout_is_rejected_before_a_request(monkeypatch, timeout):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        before = len(transport.written)
        with pytest.raises(QuantError, match="timeout"):
            await session.history(details.contract, "TRADES", timeout=timeout)
        assert len(transport.written) == before
        assert session.pending is not None and session.pending.done()
        session.close()

    asyncio.run(exercise())


def test_history_requires_an_actual_unchanged_instrument_resolution(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        before = len(transport.written)
        with pytest.raises(QuantError, match="resolved instrument"):
            await session.history(Stock("SPY", "SMART", "USD"), "TRADES")
        assert len(transport.written) == before
        details = await session.identify("SPY")
        details.contract.conId += 1
        before = len(transport.written)
        with pytest.raises(QuantError, match="resolved instrument"):
            await session.history(details.contract, "TRADES")
        assert len(transport.written) == before
        session.close()

    asyncio.run(exercise())


def test_ssd_guard_failure_never_becomes_a_sent_request(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")
        before = len(transport.written)

        def absent_ssd():
            raise QuantError("Synthetic absent SSD")

        monkeypatch.setattr(market, "runtime_guard", absent_ssd)
        with pytest.raises(QuantError, match="not_sent"):
            await session.history(details.contract, "TRADES")
        assert len(transport.written) == before
        assert transport.closed
        assert session.pending is not None and session.pending.done()

    asyncio.run(exercise())


def test_source_capture_failure_is_not_a_successful_history(monkeypatch):
    async def exercise():
        session, transport, _ = fixture_session(monkeypatch)
        await session.connect()
        details = await session.identify("SPY")

        def failed_capture(*args):
            raise OSError("Synthetic full output volume")

        session.capture = failed_capture
        with pytest.raises(QuantError, match="wire_rejected: OSError"):
            await session.history(details.contract, "TRADES")
        assert transport.closed
        assert session.failure is not None

    asyncio.run(exercise())
