from __future__ import annotations

import asyncio
import logging
import math
import re
import struct
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version

from ib_async import BarData, Contract, ContractDetails, Stock
from ib_async.client import Client
from ib_async.connection import Connection
from ib_async.decoder import Decoder

from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import before_deadline, runtime_guard
from us_quant.storage import utc_now

PROTOCOL_VERSION = 178
MAX_FRAME_BYTES = 4 * 1024 * 1024
SYMBOLS = frozenset({"SPY", "QQQ"})
INFORMATIONAL_STATUS = frozenset({2104, 2106, 2107, 2108, 2158})
Capture = Callable[[int, bytes, str], None]


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def bounded_timeout(value: float, maximum: float) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
        raise QuantError("Require a finite, positive, bounded market-data timeout.")


class _HistoryConnection(Connection):
    def __init__(self, owner: MarketHistory):
        super().__init__()
        self.owner = owner
        self.hello_sent = False
        self.api_started = False

    def validate(self, message: bytes) -> str:
        before_deadline(datetime.now(timezone.utc))
        if message.startswith(b"API\0"):
            if self.hello_sent or message != b"API\0" + frame(b"v178..178"):
                raise QuantError("Unapproved or repeated market-history handshake.")
            return "API"
        if (
            not self.hello_sent
            or len(message) < 5
            or struct.unpack(">I", message[:4])[0] != len(message) - 4
            or not message.endswith(b"\0")
        ):
            raise QuantError("Malformed outbound market-history frame.")
        fields = message[4:-1].split(b"\0")
        code = fields[0]
        if code == b"71":
            expected = [b"71", b"2", str(self.owner.client_id).encode(), b""]
            if (
                self.api_started
                or fields != expected
                or self.owner.client.serverVersion() != PROTOCOL_VERSION
                or not self.owner.client.isConnected()
            ):
                raise QuantError("Unapproved API startup or client identity.")
        elif not self.api_started or not self.owner.client.isReady():
            raise QuantError("Market requests require an acknowledged real handshake.")
        elif code == b"9":
            if (
                len(fields) != 19
                or fields[1] != b"8"
                or fields[4].decode("ascii") not in SYMBOLS
                or fields[5] != b"STK"
                or fields[10] != b"SMART"
                or fields[12] != b"USD"
                or fields[15:] != [b"0", b"", b"", b""]
            ):
                raise QuantError("Only scoped USD ETF contract identification is permitted.")
        elif code == b"20":
            if (
                len(fields) != 23
                or fields[3].decode("ascii") not in SYMBOLS
                or fields[4] != b"STK"
                or fields[9] != b"SMART"
                or fields[11] != b"USD"
                or fields[14:19] != [b"0", b"", b"1 day", b"3 M", b"1"]
                or fields[19] not in {b"TRADES", b"ADJUSTED_LAST"}
                or fields[20:] != [b"1", b"0", b""]
            ):
                raise QuantError("Only finite three-month daily price history is permitted.")
        else:
            raise QuantError("Outbound operation is outside the market-history authorization.")
        return code.decode("ascii")

    def sendMsg(self, message: bytes) -> None:
        code = self.validate(message)
        runtime_guard()
        if self.transport is None or self.transport.is_closing():
            raise QuantError("No live transport; a request was not sent.")
        super().sendMsg(message)
        self.hello_sent |= code == "API"
        self.api_started |= code == "71"
        self.owner.audit["outbound"].append({"at": utc_now(), "message": code})

    def disconnect(self) -> None:
        if self.transport is not None:
            self.transport.close()


class _MarketDecoder(Decoder):
    def interpret(self, fields: list[str]) -> None:
        # Do not use the SDK's catch-and-log dispatcher on untrusted wire fields.
        code = int(fields[0])
        if code in {9, 52}:
            if len(fields) != 3:
                raise QuantError("Malformed market acknowledgement.")
            if code == 9:
                self.wrapper.nextValidId(int(fields[2]))
            else:
                self.wrapper.contractDetailsEnd(int(fields[2]))
        elif code == 10:
            self.contractDetails(fields)
        elif code == 17:
            if len(fields) < 5 or not 1 <= int(fields[4]) <= 1024:
                raise QuantError("Empty or oversized historical response.")
            if len(fields) != 5 + 8 * int(fields[4]):
                raise QuantError("Incomplete historical response.")
            self.historicalData(fields)
        else:
            raise QuantError("Unapproved decoder message.")


class _HistoryClient(Client):
    MinClientVersion = PROTOCOL_VERSION
    MaxClientVersion = PROTOCOL_VERSION

    def __init__(self, owner: MarketHistory):
        self._wire = bytearray()
        self._market_ready = False
        self._closing = False
        self._attempted = False
        super().__init__(owner)
        self.conn = _HistoryConnection(owner)
        self.conn.hasData += self._onSocketHasData
        self.conn.disconnected += self._onSocketDisconnected
        self.decoder = _MarketDecoder(owner, 0)
        self._logger = logging.Logger("us_quant.market_history.transport")
        self._logger.addHandler(logging.NullHandler())
        self._logger.propagate = False
        self._logger.disabled = True

    def reset(self) -> None:
        super().reset()
        self._wire.clear()
        self._market_ready = False

    def isReady(self) -> bool:
        return self._market_ready and self.isConnected()

    def getAccounts(self) -> list[str]:
        raise QuantError("Account access is outside the market-history authorization.")

    async def connectAsync(self, host, port, clientId, timeout=5.0):
        bounded_timeout(timeout, 10)
        if (
            self._attempted
            or host != "127.0.0.1"
            or type(port) is not int
            or port != 4002
            or type(clientId) is not int
            or clientId != self.wrapper.client_id
            or not 1 <= clientId < 2**31
        ):
            raise QuantError("Require one bounded loopback attempt with a nonzero client ID.")
        runtime_guard()
        before_deadline(datetime.now(timezone.utc))
        self._attempted = True
        self.host, self.port, self.clientId = host, port, clientId
        self.connState = Client.CONNECTING
        try:
            await asyncio.wait_for(self.conn.connectAsync(host, port), timeout)
            self.wrapper.ready = asyncio.get_running_loop().create_future()
            self.conn.sendMsg(b"API\0" + frame(b"v178..178"))
            await asyncio.wait_for(self.wrapper.ready, timeout)
            self.wrapper.require_healthy()
        except (OSError, TimeoutError, QuantError) as exc:
            error = self.wrapper.fail(f"market_connection_failed_{type(exc).__name__}")
            if self.wrapper.ready is not None and self.wrapper.ready.done():
                if not self.wrapper.ready.cancelled():
                    self.wrapper.ready.exception()
            raise error from None

    def sendMsg(self, message) -> None:
        if message:
            self.conn.validate(frame(message.encode()))
        super().sendMsg(message)

    def disconnect(self) -> None:
        self._closing = True
        super().disconnect()

    def _onSocketDisconnected(self, message) -> None:
        if not self._closing:
            self.wrapper.fail("market_transport_disconnected")
        self.reset()

    def _onSocketHasData(self, data: bytes) -> None:
        if self._closing or self.wrapper.failure is not None:
            return
        try:
            self._wire.extend(data)
            if len(self._wire) > MAX_FRAME_BYTES + 65536:
                raise QuantError("Inbound market-history buffer limit exceeded.")
            while len(self._wire) >= 4:
                length = struct.unpack(">I", self._wire[:4])[0]
                if not 1 <= length <= MAX_FRAME_BYTES:
                    raise QuantError("Invalid inbound market-history frame size.")
                if len(self._wire) < length + 4:
                    return
                packet = bytes(self._wire[: length + 4])
                del self._wire[: length + 4]
                self._receive(packet)
                if self.wrapper.failure is not None:
                    return
        except (
            QuantError,
            ValueError,
            TypeError,
            LookupError,
            AttributeError,
            RuntimeError,
            OSError,
            StopIteration,
            struct.error,
        ) as exc:
            # Eventkit otherwise logs callback arguments, including coalesced private frames.
            reason = str(exc) if isinstance(exc, QuantError) else type(exc).__name__
            self.wrapper.fail(f"market_wire_rejected: {reason}")

    def _receive(self, packet: bytes) -> None:
        payload = packet[4:]
        if not payload.endswith(b"\0"):
            raise QuantError("Unterminated inbound market-history frame.")
        if not self._serverVersion:
            fields = payload[:-1].split(b"\0")
            if len(fields) != 2 or fields[0] != b"178":
                raise QuantError("Unverified negotiated wire version.")
            super()._onSocketHasData(packet)
            self.wrapper.audit["negotiated_version"] = self.serverVersion()
            return
        token = payload.partition(b"\0")[0]
        if not token.isdigit():
            raise QuantError("Non-numeric inbound message type.")
        code = int(token)
        if code == 15:
            # Discard before decoding, SDK account collection, logging or source capture.
            self.wrapper.audit["discarded_managed_account_frames"] += 1
            return
        if code == 4:
            fields = payload.split(b"\0", 4)
            if len(fields) != 5:
                raise QuantError("Malformed provider status.")
            self.wrapper.provider_status(int(fields[2]), int(fields[3]))
            return
        if code not in {9, 10, 17, 52}:
            raise QuantError(f"Unexpected inbound message type {code}.")
        if code == 9:
            fields = payload[:-1].split(b"\0")
            if (
                len(fields) != 3
                or not fields[2].isdigit()
                or int(fields[2]) >= 2**31
                or self._market_ready
            ):
                raise QuantError("Invalid or repeated request-ID acknowledgement.")
        super()._onSocketHasData(packet)
        if code == 9:
            if not self._hasReqId or not self.conn.api_started or self._accounts:
                raise QuantError("Market readiness lacks an actual clean acknowledgement.")
            self._market_ready = True
            self.wrapper.audit["market_request_ready_at"] = utc_now()
            if self.wrapper.ready is None or self.wrapper.ready.done():
                raise QuantError("Unsolicited market readiness.")
            self.wrapper.ready.set_result(None)
        elif code in {10, 17, 52}:
            self.wrapper.capture(code, packet, utc_now())


class MarketHistory:
    """Single-use price-history connection; it never constructs the high-level IB object."""

    def __init__(self, client_id: int, capture: Capture):
        if type(client_id) is not int or not 1 <= client_id < 2**31:
            raise QuantError("A dedicated nonzero integer client ID is required.")
        if version("ib_async") != "2.1.0":
            raise QuantError("Prove the installed SDK version before any connection.")
        self.client_id = client_id
        self.capture = capture
        self.audit: dict = {
            "outbound": [],
            "provider_status": [],
            "discarded_managed_account_frames": 0,
            "readiness_scope": "actual_market_handshake_not_account_sync_or_entitlement",
        }
        self.failure: QuantError | None = None
        self.ready: asyncio.Future[None] | None = None
        self.pending: asyncio.Future[None] | None = None
        self.request_id: int | None = None
        self.request_kind: str | None = None
        self.contracts: list[ContractDetails] = []
        self.bars: list[BarData] = []
        self.identified_contracts: dict[int, dict] = {}
        self.client = _HistoryClient(self)

    def fail(self, reason: str) -> QuantError:
        error = self.failure
        if error is None:
            error = self.failure = QuantError(reason)
            self.audit["failure"] = reason
        for future in (self.ready, self.pending):
            if future is not None and not future.done():
                future.set_exception(error)
        self.client.disconnect()
        return error

    def require_healthy(self) -> None:
        if self.failure is not None:
            raise self.failure

    def connectAck(self) -> None:
        pass

    def nextValidId(self, request_id: int) -> None:
        if request_id < 0:
            raise QuantError("Invalid request-ID acknowledgement.")

    def provider_status(self, request_id: int, code: int) -> None:
        self.audit["provider_status"].append(
            {"at": utc_now(), "request_id": request_id, "code": code, "text_retained": False}
        )
        if code not in INFORMATIONAL_STATUS or request_id != -1:
            self.fail(f"ibkr_status_{code}")

    def _begin(self, kind: str) -> int:
        self.require_healthy()
        if self.pending is not None and not self.pending.done():
            raise QuantError("Only one market-data request may be outstanding.")
        self.request_id = self.client.getReqId()
        self.request_kind = kind
        self.pending = asyncio.get_running_loop().create_future()
        self.contracts = []
        self.bars = []
        return self.request_id

    def _match(self, request_id: int, kind: str) -> None:
        if (
            request_id != self.request_id
            or self.request_kind != kind
            or self.pending is None
            or self.pending.done()
        ):
            raise QuantError("Unexpected or duplicate market response identity.")

    async def _wait(self, timeout: float) -> None:
        if self.pending is None:
            raise QuantError("No actual market request is pending.")
        try:
            await asyncio.wait_for(self.pending, timeout)
        except TimeoutError:
            raise self.fail("market_request_timeout_no_retry") from None
        self.require_healthy()

    def _send_request(self, operation: Callable[[], None]) -> None:
        try:
            operation()
        except (QuantError, OSError, ValueError, TypeError) as exc:
            error = self.fail(f"market_request_not_sent_{type(exc).__name__}")
            if self.pending is not None and self.pending.done() and not self.pending.cancelled():
                self.pending.exception()
            raise error from None

    async def connect(self, timeout: float = 5.0) -> None:
        await self.client.connectAsync("127.0.0.1", 4002, self.client_id, timeout)

    async def identify(self, symbol: str, timeout: float = 15.0) -> ContractDetails:
        bounded_timeout(timeout, 30)
        if symbol not in SYMBOLS:
            raise QuantError("Instrument is outside this bounded source probe.")
        request_id = self._begin("contract")
        self._send_request(
            lambda: self.client.reqContractDetails(request_id, Stock(symbol, "SMART", "USD"))
        )
        await self._wait(timeout)
        if len(self.contracts) != 1:
            raise QuantError("Require a unique actual instrument, not a guessed first match.")
        details = self.contracts[0]
        contract = details.contract
        if (
            contract.symbol != symbol
            or contract.secType != "STK"
            or contract.currency != "USD"
            or contract.exchange != "SMART"
            or contract.conId <= 0
        ):
            raise QuantError("Resolved instrument does not match the permitted ETF.")
        self.identified_contracts[contract.conId] = asdict(contract)
        return details

    async def history(self, contract: Contract, what: str, timeout: float = 45.0) -> list[BarData]:
        bounded_timeout(timeout, 60)
        self.require_healthy()
        if what not in {"TRADES", "ADJUSTED_LAST"}:
            raise QuantError("Only the two specified historical price fields are permitted.")
        if self.identified_contracts.get(contract.conId) != asdict(contract):
            raise QuantError("History requires this connection's unchanged resolved instrument.")
        request_id = self._begin("history")
        self._send_request(
            lambda: self.client.reqHistoricalData(
                request_id, contract, "", "3 M", "1 day", what, True, 1, False, []
            )
        )
        await self._wait(timeout)
        return list(self.bars)

    def contractDetails(self, request_id: int, details: ContractDetails) -> None:
        self._match(request_id, "contract")
        if len(self.contracts) >= 8:
            raise QuantError("Unexpectedly broad instrument response.")
        self.contracts.append(details)

    def contractDetailsEnd(self, request_id: int) -> None:
        self._match(request_id, "contract")
        if self.pending is None:
            raise QuantError("Missing contract completion future.")
        self.pending.set_result(None)

    def historicalData(self, request_id: int, bar: BarData) -> None:
        self._match(request_id, "history")
        if not isinstance(bar.date, str) or re.fullmatch(r"[0-9]{8}", bar.date) is None:
            raise QuantError("Expected an actual daily session label.")
        datetime.strptime(bar.date, "%Y%m%d")
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (
            not all(math.isfinite(value) and value > 0 for value in prices)
            or bar.low > min(bar.open, bar.close)
            or bar.high < max(bar.open, bar.close)
            or not math.isfinite(bar.volume)
            or not math.isfinite(bar.average)
            or bar.volume < -1
            or bar.average < -1
            or bar.barCount < -1
            or (self.bars and str(self.bars[-1].date) >= bar.date)
        ):
            raise QuantError("Invalid, duplicated or unordered historical price bar.")
        self.bars.append(bar)

    def historicalDataEnd(self, request_id: int, start: str, end: str) -> None:
        self._match(request_id, "history")
        if self.pending is None or not self.bars:
            raise QuantError("An empty history cannot be a successful source.")
        self.pending.set_result(None)

    def close(self) -> None:
        if self.pending is not None and not self.pending.done():
            self.fail("market_connection_closed_with_request_pending")
        else:
            self.client.disconnect()
