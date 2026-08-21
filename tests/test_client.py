"""Tests for the asynchronous FH3X Modbus client."""

from __future__ import annotations

from collections import deque
from typing import Any
import unittest

from pylontech_fh3x import (
    FH3XConnectionError,
    FH3XModbusClient,
    FH3XProtocolError,
)
from pylontech_fh3x.constants import BMS_DEVICE_ID, PCS_DEVICE_ID
from pylontech_fh3x.protocol import (
    BMS_CELL_TEMPERATURE_OFFSET,
    BMS_CELL_VOLTAGE_OFFSET,
    BMS_MODULE_TEMPERATURE_OFFSET,
    BMS_MODULE_VOLTAGE_OFFSET,
    BMS_PILE_ADDRESS,
    BMS_SYSTEM_ADDRESS,
    BMS_SYSTEM_COUNT,
    PCS_ACTIVE_POWER_CONTROL_ADDRESS,
    PCS_DEVICE_INFO_EXT_ADDRESS,
    PCS_IDENTITY_ADDRESS,
)


class FakeResponse:
    """Minimal pymodbus-compatible response."""

    def __init__(
        self, registers: list[int] | None = None, *, error: bool = False
    ) -> None:
        self.registers = registers if registers is not None else []
        self._error = error

    def isError(self) -> bool:
        """Return whether this is a Modbus exception response."""
        return self._error


class InvalidResponse:
    """Response with no register payload."""

    def isError(self) -> bool:
        """Return a successful status with an invalid body."""
        return False


class FakeTransport:
    """Scriptable transport used by client tests."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_results: deque[bool | Exception] = deque([True])
        self.holding_results: deque[Any] = deque()
        self.holding_by_address: dict[int, deque[Any]] = {}
        self.input_results: deque[Any] = deque()
        self.write_results: deque[Any] = deque()
        self.connect_calls = 0
        self.close_calls = 0
        self.holding_calls: list[tuple[int, int, int]] = []
        self.input_calls: list[tuple[int, int, int]] = []
        self.write_calls: list[tuple[int, list[int], int]] = []

    @staticmethod
    def _take(results: deque[Any], default: Any) -> Any:
        result = results.popleft() if results else default
        if isinstance(result, Exception):
            raise result
        return result

    async def connect(self) -> bool:
        """Connect using the next scripted result."""
        self.connect_calls += 1
        result = self._take(self.connect_results, True)
        self.connected = bool(result)
        return self.connected

    def close(self) -> None:
        """Close the fake transport."""
        self.close_calls += 1
        self.connected = False

    async def read_holding_registers(
        self, address: int, *, count: int, device_id: int
    ) -> Any:
        """Return the next holding-register response."""
        self.holding_calls.append((address, count, device_id))
        if results := self.holding_by_address.get(address):
            return self._take(results, FakeResponse([0] * count))
        return self._take(self.holding_results, FakeResponse([0] * count))

    async def read_input_registers(
        self, address: int, *, count: int, device_id: int
    ) -> Any:
        """Return the next input-register response."""
        self.input_calls.append((address, count, device_id))
        return self._take(self.input_results, FakeResponse([0] * count))

    async def write_register(
        self, address: int, value: int, *, device_id: int
    ) -> Any:
        """Record a single-register write."""
        self.write_calls.append((address, [value], device_id))
        return self._take(self.write_results, FakeResponse())

    async def write_registers(
        self, address: int, values: list[int], *, device_id: int
    ) -> Any:
        """Record a multi-register write."""
        self.write_calls.append((address, list(values), device_id))
        return self._take(self.write_results, FakeResponse())


def make_client(
    transport: FakeTransport,
    *,
    read_retries: int = 0,
    detailed_bms: bool = False,
) -> FH3XModbusClient:
    """Create a client with all artificial pacing disabled."""
    return FH3XModbusClient(
        "192.0.2.10",
        502,
        timeout=1,
        request_interval=0,
        device_switch_delay=0,
        read_retries=read_retries,
        read_retry_delay=0,
        detailed_bms=detailed_bms,
        transport=transport,
    )


class TestFH3XModbusClient(unittest.IsolatedAsyncioTestCase):
    """Verify transport, retry and write behavior."""

    async def test_holding_error_falls_back_to_input_registers(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(FakeResponse(error=True))
        transport.input_results.append(FakeResponse([10, 20]))
        client = make_client(transport)

        registers = await client._async_read_registers(100, 2, 2)

        self.assertEqual(registers, [10, 20])
        self.assertEqual(transport.holding_calls, [(100, 2, 2)])
        self.assertEqual(transport.input_calls, [(100, 2, 2)])

    async def test_transient_read_reconnects_and_retries(self) -> None:
        transport = FakeTransport()
        transport.connect_results.extend([True])
        transport.holding_results.extend(
            [OSError("connection reset"), FakeResponse([7])]
        )
        client = make_client(transport, read_retries=1)

        registers = await client._async_read_registers(200, 1, 1)

        self.assertEqual(registers, [7])
        self.assertEqual(transport.connect_calls, 2)
        self.assertEqual(transport.close_calls, 1)

    async def test_connection_failure_is_reported(self) -> None:
        transport = FakeTransport()
        transport.connect_results = deque([False])
        client = make_client(transport)

        with self.assertRaises(FH3XConnectionError):
            await client._async_read_registers(100, 1, 2)

    async def test_short_response_is_rejected_and_closed(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(FakeResponse([1]))
        client = make_client(transport)

        with self.assertRaises(FH3XProtocolError):
            await client._async_read_registers(100, 2, 2)

        self.assertEqual(transport.close_calls, 1)

    async def test_response_without_registers_is_rejected(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(InvalidResponse())
        client = make_client(transport)

        with self.assertRaises(FH3XProtocolError):
            await client._async_read_registers(100, 1, 2)

        self.assertEqual(transport.close_calls, 1)

    async def test_single_register_write_is_verified(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(FakeResponse([42]))
        client = make_client(transport)

        await client.async_write_u16(40904, 42)

        self.assertEqual(transport.write_calls, [(40904, [42], 2)])
        self.assertEqual(transport.holding_calls, [(40904, 1, 2)])

    async def test_multi_register_write_encodes_signed_value(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(FakeResponse([0xFFFF, 0xFFFE]))
        client = make_client(transport)

        await client.async_write_s32(40901, -2)

        self.assertEqual(
            transport.write_calls,
            [(40901, [0xFFFF, 0xFFFE], 2)],
        )

    async def test_write_readback_mismatch_is_rejected(self) -> None:
        transport = FakeTransport()
        transport.holding_results.append(FakeResponse([41]))
        client = make_client(transport)

        with self.assertRaises(FH3XProtocolError):
            await client.async_write_u16(40904, 42)

    async def test_bit_update_preserves_other_bits(self) -> None:
        transport = FakeTransport()
        transport.holding_results.extend(
            [FakeResponse([0b1001]), FakeResponse([0b1011])]
        )
        client = make_client(transport)

        await client.async_update_u16_bit(40908, 1, True)

        self.assertEqual(transport.write_calls, [(40908, [0b1011], 2)])

    async def test_close_resets_transport(self) -> None:
        transport = FakeTransport()
        transport.connected = True
        client = make_client(transport)

        await client.async_close()

        self.assertFalse(transport.connected)
        self.assertEqual(transport.close_calls, 1)

    async def test_snapshot_groups_pcs_reads_before_bms_reads(self) -> None:
        transport = FakeTransport()
        client = make_client(transport)

        snapshot = await client.async_read_snapshot()

        device_ids = [device_id for _, _, device_id in transport.holding_calls]
        first_bms_read = device_ids.index(BMS_DEVICE_ID)
        self.assertTrue(
            all(device_id == PCS_DEVICE_ID for device_id in device_ids[:first_bms_read])
        )
        self.assertTrue(
            all(device_id == BMS_DEVICE_ID for device_id in device_ids[first_bms_read:])
        )
        self.assertEqual(snapshot.identity.serial, "modbus-192.0.2.10:502")

    async def test_snapshot_caches_identity_and_device_information(self) -> None:
        transport = FakeTransport()
        client = make_client(transport)

        await client.async_read_snapshot()
        await client.async_read_snapshot()

        addresses = [address for address, _, _ in transport.holding_calls]
        self.assertEqual(addresses.count(PCS_IDENTITY_ADDRESS), 1)
        self.assertEqual(addresses.count(PCS_DEVICE_INFO_EXT_ADDRESS), 1)

    async def test_optional_control_failure_does_not_break_snapshot(self) -> None:
        transport = FakeTransport()
        transport.holding_by_address[PCS_ACTIVE_POWER_CONTROL_ADDRESS] = deque(
            [FakeResponse(error=True)]
        )
        client = make_client(transport)

        snapshot = await client.async_read_snapshot()

        self.assertEqual(snapshot.identity.serial, "modbus-192.0.2.10:502")
        self.assertIn(
            PCS_ACTIVE_POWER_CONTROL_ADDRESS,
            [address for address, _, _ in transport.holding_calls],
        )

    async def test_detailed_bms_uses_reported_module_and_cell_counts(self) -> None:
        transport = FakeTransport()
        bms_registers = [0] * BMS_SYSTEM_COUNT
        bms_registers[54] = 2
        bms_registers[55] = 3
        transport.holding_by_address[BMS_SYSTEM_ADDRESS] = deque(
            [FakeResponse(bms_registers)]
        )
        client = make_client(transport, detailed_bms=True)

        snapshot = await client.async_read_snapshot()

        calls = {(address, count) for address, count, _ in transport.holding_calls}
        self.assertIn((BMS_PILE_ADDRESS + BMS_MODULE_VOLTAGE_OFFSET, 2), calls)
        self.assertIn((BMS_PILE_ADDRESS + BMS_MODULE_TEMPERATURE_OFFSET, 2), calls)
        self.assertIn((BMS_PILE_ADDRESS + BMS_CELL_VOLTAGE_OFFSET, 3), calls)
        self.assertIn((BMS_PILE_ADDRESS + BMS_CELL_TEMPERATURE_OFFSET, 3), calls)
        self.assertIn("bms_module_02_voltage", snapshot.values)
        self.assertIn("bms_cell_003_temperature", snapshot.values)


if __name__ == "__main__":
    unittest.main()
