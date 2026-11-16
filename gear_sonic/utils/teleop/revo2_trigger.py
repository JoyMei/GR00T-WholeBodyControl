"""Continuous PICO trigger control for bilateral BrainCo Revo2 hands.

The manager owns input and this module owns the Revo2 serial devices in a
separate process. Trigger zero holds the measured startup pose; increasing the
Trigger adds a bounded per-channel delta. This avoids a startup jump because
Revo2 thumb and finger channels do not share the same neutral value.
"""

from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import queue
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gear_sonic.utils.teleop.revo2_data import (
    REVO2_MOTOR_CHANNELS,
    REVO2_TOUCH_CAPACITIVE,
    REVO2_TOUCH_NONE,
    REVO2_TOUCH_PRESSURE_SUMMARY,
    decode_capacitive_batches,
    decode_motor_status,
    decode_pressure_detailed_batches,
    decode_pressure_summary_batches,
    empty_revo2_state,
)

REVO2_CHANNEL_NAMES = REVO2_MOTOR_CHANNELS
REVO2_CHANNEL_COUNT = len(REVO2_CHANNEL_NAMES)
DEFAULT_THUMB_DELTA = 0.15
DEFAULT_FINGER_DELTA = 0.20


def trigger_to_revo2_deltas(
    trigger: float,
    *,
    thumb_delta: float = DEFAULT_THUMB_DELTA,
    finger_delta: float = DEFAULT_FINGER_DELTA,
) -> np.ndarray:
    """Map one Trigger to bounded deltas for the six Revo2 channels."""

    value = float(trigger)
    thumb_limit = float(thumb_delta)
    finger_limit = float(finger_delta)
    if not math.isfinite(value):
        raise ValueError("trigger must be finite")
    if not math.isfinite(thumb_limit) or not 0.0 < thumb_limit <= 0.5:
        raise ValueError("thumb_delta must be finite and within (0, 0.5]")
    if not math.isfinite(finger_limit) or not 0.0 < finger_limit <= 0.5:
        raise ValueError("finger_delta must be finite and within (0, 0.5]")
    value = min(1.0, max(0.0, value))
    profile = np.asarray(
        [thumb_limit, thumb_limit, finger_limit, finger_limit, finger_limit, finger_limit],
        dtype=np.float64,
    )
    return value * profile


def bilateral_trigger_deltas(
    left_trigger: float,
    right_trigger: float,
    *,
    thumb_delta: float = DEFAULT_THUMB_DELTA,
    finger_delta: float = DEFAULT_FINGER_DELTA,
) -> np.ndarray:
    """Return independent left/right delta profiles with shape ``(2, 6)``."""

    return np.stack(
        (
            trigger_to_revo2_deltas(
                left_trigger,
                thumb_delta=thumb_delta,
                finger_delta=finger_delta,
            ),
            trigger_to_revo2_deltas(
                right_trigger,
                thumb_delta=thumb_delta,
                finger_delta=finger_delta,
            ),
        )
    )


@dataclass(frozen=True)
class Revo2Device:
    side: str
    port: str
    serial: str
    slave_id: int

    @classmethod
    def from_mapping(cls, value: Any) -> "Revo2Device":
        if not isinstance(value, dict):
            raise ValueError("each Revo2 device must be a JSON object")
        side = value.get("side")
        port = value.get("port")
        serial = value.get("serial")
        slave_id = value.get("slave_id")
        if side not in {"left", "right"}:
            raise ValueError("Revo2 side must be 'left' or 'right'")
        if not isinstance(port, str) or not port:
            raise ValueError("Revo2 port must be a non-empty string")
        if not isinstance(serial, str) or not serial:
            raise ValueError("Revo2 serial must be a non-empty string")
        if type(slave_id) is not int or not 0 <= slave_id <= 247:
            raise ValueError("Revo2 slave_id must be an integer in [0, 247]")
        return cls(side=side, port=port, serial=serial, slave_id=slave_id)


def load_revo2_devices(path: str | Path) -> tuple[Revo2Device, Revo2Device]:
    """Load exactly one left and one right Revo2, ordered left then right."""

    config_path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("Revo2 device config must contain exactly two devices")
    devices = tuple(Revo2Device.from_mapping(item) for item in raw)
    by_side = {device.side: device for device in devices}
    if set(by_side) != {"left", "right"}:
        raise ValueError("Revo2 device config must contain one left and one right hand")
    if len({device.port for device in devices}) != 2:
        raise ValueError("left and right Revo2 devices must use distinct ports")
    if len({device.serial for device in devices}) != 2:
        raise ValueError("left and right Revo2 devices must use distinct serial identities")
    if len({device.slave_id for device in devices}) != 2:
        raise ValueError("left and right Revo2 devices must use distinct slave IDs")
    return by_side["left"], by_side["right"]


def _bounded_slew(previous: np.ndarray, target: np.ndarray, dt: float, rate: float) -> np.ndarray:
    step = rate * min(max(dt, 0.0), 0.05)
    return previous + np.clip(target - previous, -step, step)


async def _close_client(sdk: Any, client: Any) -> None:
    import inspect

    completion = sdk.modbus_close(client)
    if inspect.isawaitable(completion):
        await asyncio.wait_for(completion, timeout=1.0)


async def _maybe_await(value: Any) -> Any:
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value


def _device_supports_touch(device_info: Any) -> bool:
    for attribute in ("is_touch", "uses_revo2_touch_api"):
        value = getattr(device_info, attribute, None)
        if value is None:
            continue
        if bool(value() if callable(value) else value):
            return True
    return False


async def _create_data_collector(
    sdk: Any,
    client: Any,
    device: Revo2Device,
    device_info: Any,
    *,
    frequency: int,
    touch_setup: bool,
) -> dict[str, Any]:
    """Create the official BrainCo collector on the existing serial owner."""

    motor_buffer = sdk.MotorStatusBuffer(max_size=max(128, frequency * 10))
    touch_type = REVO2_TOUCH_NONE
    touch_buffer = None
    pressure_detailed_buffer = None

    if _device_supports_touch(device_info):
        uses_pressure = bool(
            await _maybe_await(client.uses_pressure_touch_api(device.slave_id))
        )
        for api_name, label in (
            ("uses_array_pressure_touch_api", "ArrayPressure"),
            ("uses_force3d_touch_api", "Force3D"),
        ):
            detector = getattr(client, api_name, None)
            if callable(detector) and bool(
                await _maybe_await(detector(device.slave_id))
            ):
                raise RuntimeError(
                    f"Revo2 {label} touch requires a dedicated OXT schema; "
                    "refusing to decode it as capacitive touch"
                )
        required_mask = 0x3F if uses_pressure else 0x1F
        enabled_mask = int(
            await asyncio.wait_for(
                client.get_touch_sensor_enabled(device.slave_id), timeout=3.0
            )
        )
        if touch_setup and enabled_mask & required_mask != required_mask:
            await asyncio.wait_for(
                client.touch_sensor_setup(device.slave_id, required_mask), timeout=3.0
            )
            await asyncio.sleep(1.0)
            enabled_mask = int(
                await asyncio.wait_for(
                    client.get_touch_sensor_enabled(device.slave_id), timeout=3.0
                )
            )
        if enabled_mask & required_mask != required_mask:
            raise RuntimeError(
                f"Revo2 {device.side} touch sensors are not enabled "
                f"(mask=0x{enabled_mask:02x}, required=0x{required_mask:02x}); "
                "enable them explicitly before capture or use --revo2-touch-setup"
            )

        if uses_pressure:
            touch_type = REVO2_TOUCH_PRESSURE_SUMMARY
            touch_buffer = sdk.PressureSummaryBuffer(
                max_size=max(128, frequency * 10)
            )
            if hasattr(sdk, "PressureDetailedBuffer") and hasattr(
                sdk.DataCollector, "new_pressure_hybrid"
            ):
                pressure_detailed_buffer = sdk.PressureDetailedBuffer(
                    max_size=max(128, frequency * 10)
                )
                collector = sdk.DataCollector.new_pressure_hybrid(
                    ctx=client,
                    motor_buffer=motor_buffer,
                    pressure_summary_buffer=touch_buffer,
                    pressure_detailed_buffer=pressure_detailed_buffer,
                    slave_id=device.slave_id,
                    motor_frequency=frequency,
                    summary_frequency=frequency,
                    detailed_frequency=min(10, frequency),
                    enable_stats=True,
                )
            else:
                collector = sdk.DataCollector.new_pressure_summary(
                    ctx=client,
                    motor_buffer=motor_buffer,
                    pressure_summary_buffer=touch_buffer,
                    slave_id=device.slave_id,
                    motor_frequency=frequency,
                    touch_frequency=frequency,
                    enable_stats=True,
                )
        else:
            touch_type = REVO2_TOUCH_CAPACITIVE
            touch_buffer = sdk.TouchStatusBuffer(max_size=max(128, frequency * 10))
            collector = sdk.DataCollector.new_capacitive(
                ctx=client,
                motor_buffer=motor_buffer,
                touch_buffer=touch_buffer,
                slave_id=device.slave_id,
                motor_frequency=frequency,
                touch_frequency=frequency,
                enable_stats=True,
            )
    else:
        collector = sdk.DataCollector.new_basic(
            ctx=client,
            motor_buffer=motor_buffer,
            slave_id=device.slave_id,
            motor_frequency=frequency,
            enable_stats=True,
        )

    return {
        "collector": collector,
        "motor_buffer": motor_buffer,
        "touch_buffer": touch_buffer,
        "pressure_detailed_buffer": pressure_detailed_buffer,
        "touch_type": touch_type,
    }


def _replace_latest(queue_: Any, value: dict[str, Any]) -> None:
    """Publish latest-only state without ever blocking the hardware owner."""

    try:
        queue_.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        queue_.get_nowait()
    except queue.Empty:
        pass
    try:
        queue_.put_nowait(value)
    except queue.Full:
        pass


async def _run_revo2_owner(
    devices: Sequence[Revo2Device],
    targets: Any,
    source_triggers: Any,
    source_received_ns: Any,
    sequence: Any,
    enabled: Any,
    stop: Any,
    status: Any,
    state_queue: Any,
    deadman_s: float,
    slew_per_s: float,
    command_period_s: float,
    command_speed: float,
    capture_data: bool,
    data_rate_hz: float,
    touch_setup: bool,
) -> None:
    # Hardware-only dependencies stay in the child process so mapping/tests run
    # on Windows and on machines without the BrainCo SDK.
    import fcntl

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()

    clients: list[Any] = []
    locks: list[Any] = []
    collector_resources: list[dict[str, Any]] = []
    observed_devices: list[dict[str, Any]] = []
    previous: list[np.ndarray] = []
    state = empty_revo2_state()
    was_active = False
    motion_may_be_active = False
    last_sequence = -1
    last_report = float("-inf")

    async def bounded(awaitable: Any, timeout: float = 0.25) -> Any:
        return await asyncio.wait_for(awaitable, timeout=timeout)

    async def stop_motion() -> None:
        if not clients:
            return
        results = await asyncio.gather(
            *[
                bounded(client.set_finger_speeds(device.slave_id, [0] * REVO2_CHANNEL_COUNT))
                for client, device in zip(clients, devices[: len(clients)], strict=True)
            ],
            return_exceptions=True,
        )
        errors = [repr(result) for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("Revo2 zero-speed stop failed: " + "; ".join(errors))

    try:
        for side_index, device in enumerate(devices):
            resolved_port = Path(device.port).resolve(strict=True)
            lock_path = Path("/tmp") / f"g1-revo2-{resolved_port.name}.lock"
            lock_handle = lock_path.open("a", encoding="utf-8")
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock_handle)

            owners = subprocess.run(
                ["fuser", str(resolved_port)], capture_output=True, text=True, check=False
            )
            if owners.returncode != 1:
                raise RuntimeError(
                    f"Revo2 port is occupied or ownership check failed: {resolved_port}"
                )

            for attempt in range(2):
                client = None
                try:
                    client = await asyncio.wait_for(
                        sdk.modbus_open(str(resolved_port), sdk.Baudrate.Baud460800),
                        timeout=3.0,
                    )
                    info = await asyncio.wait_for(
                        client.get_device_info(device.slave_id), timeout=3.0
                    )
                    unit_mode = await asyncio.wait_for(
                        client.get_finger_unit_mode(device.slave_id), timeout=3.0
                    )
                    expected_hand = getattr(sdk.HandType, device.side.title())
                    if str(info.serial_number) != device.serial or info.hand_type != expected_hand:
                        raise ValueError(f"unexpected Revo2 identity on {resolved_port}: {info}")
                    if unit_mode != sdk.FingerUnitMode.Normalized:
                        raise ValueError(f"expected Normalized Revo2 mode, found {unit_mode}")
                    motor_status = await asyncio.wait_for(
                        client.get_motor_status(device.slave_id), timeout=3.0
                    )
                    decoded_motor = decode_motor_status(motor_status)
                    measured = decoded_motor["position_raw"].astype(np.float64) / 1000.0
                    if measured.shape != (REVO2_CHANNEL_COUNT,) or not np.isfinite(measured).all():
                        raise ValueError("invalid measured Revo2 startup positions")
                    if np.any((measured < 0.0) | (measured > 1.0)):
                        raise ValueError("measured Revo2 startup positions are outside [0, 1]")
                    resource = None
                    if capture_data:
                        resource = await _create_data_collector(
                            sdk,
                            client,
                            device,
                            info,
                            frequency=int(round(data_rate_hz)),
                            touch_setup=touch_setup,
                        )
                except Exception:
                    if client is not None:
                        try:
                            await _close_client(sdk, client)
                        except Exception:
                            pass
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    raise
                else:
                    clients.append(client)
                    observed_devices.append(
                        {
                            "side": device.side,
                            "serial": str(info.serial_number),
                            "slave_id": device.slave_id,
                            "firmware_version": str(
                                getattr(info, "firmware_version", "") or ""
                            ),
                            "hardware_type": str(
                                getattr(info, "hardware_type", "unknown")
                            ),
                            "hand_type": str(getattr(info, "hand_type", "unknown")),
                        }
                    )
                    previous.append(measured)
                    now_ns = time.monotonic_ns()
                    state["motor_position_raw"][side_index] = decoded_motor["position_raw"]
                    state["motor_speed_raw"][side_index] = decoded_motor["speed_raw"]
                    state["motor_current_raw"][side_index] = decoded_motor["current_raw"]
                    state["motor_state_code"][side_index] = decoded_motor["state_code"]
                    state["motor_ns"][side_index] = now_ns
                    state["motor_valid"][side_index] = True
                    state["command_position_raw"][side_index] = decoded_motor["position_raw"]
                    if resource is not None:
                        state["touch_type"][side_index] = resource["touch_type"]
                        collector_resources.append(resource)
                    break

        previous_array = np.asarray(previous, dtype=np.float64)
        baseline_array = previous_array.copy()
        for resource in collector_resources:
            resource["collector"].start()

        sdk_version = getattr(sdk, "__version__", None)
        get_sdk_version = getattr(sdk, "get_sdk_version", None)
        if callable(get_sdk_version):
            try:
                sdk_version = get_sdk_version()
            except Exception:
                pass
        status.send(
            (
                "ready",
                {
                    "capture_data": bool(capture_data),
                    "data_rate_hz": float(data_rate_hz),
                    "sdk_version": str(sdk_version) if sdk_version is not None else None,
                    "touch_type": state["touch_type"].astype(int).tolist(),
                    "devices": observed_devices,
                },
            )
        )
        next_tick = time.monotonic()
        next_state_publish = next_tick
        state_period_s = 1.0 / data_rate_hz
        timeout_ns = int(deadman_s * 1e9)

        while not stop.is_set():
            now_ns = time.monotonic_ns()
            for side_index, resource in enumerate(collector_resources):
                motor_batch = resource["motor_buffer"].pop_all()
                if motor_batch:
                    decoded_motor = decode_motor_status(motor_batch[-1])
                    measured = decoded_motor["position_raw"].astype(np.float64) / 1000.0
                    if np.any((measured < 0.0) | (measured > 1.0)):
                        raise RuntimeError("measured Revo2 positions are outside [0, 1]")
                    previous_array[side_index] = measured
                    state["motor_position_raw"][side_index] = decoded_motor["position_raw"]
                    state["motor_speed_raw"][side_index] = decoded_motor["speed_raw"]
                    state["motor_current_raw"][side_index] = decoded_motor["current_raw"]
                    state["motor_state_code"][side_index] = decoded_motor["state_code"]
                    state["motor_ns"][side_index] = now_ns
                    state["motor_valid"][side_index] = True
                    state["motor_coalesced"][side_index] += max(
                        0, len(motor_batch) - 1
                    )

                touch_buffer = resource["touch_buffer"]
                touch_type = resource["touch_type"]
                if touch_buffer is not None:
                    touch_batches = touch_buffer.pop_all()
                    if any(touch_batches):
                        if touch_type == REVO2_TOUCH_CAPACITIVE:
                            values, coalesced = decode_capacitive_batches(touch_batches)
                            state["capacitive_raw"][side_index] = values
                        elif touch_type == REVO2_TOUCH_PRESSURE_SUMMARY:
                            values, coalesced = decode_pressure_summary_batches(
                                touch_batches
                            )
                            state["pressure_summary_raw"][side_index] = values
                        else:
                            raise RuntimeError(f"unsupported Revo2 touch type: {touch_type}")
                        state["touch_ns"][side_index] = now_ns
                        state["touch_valid"][side_index] = True
                        state["touch_coalesced"][side_index] += coalesced

                pressure_detailed_buffer = resource["pressure_detailed_buffer"]
                if pressure_detailed_buffer is not None:
                    detailed_batches = pressure_detailed_buffer.pop_all()
                    if any(detailed_batches):
                        values, coalesced = decode_pressure_detailed_batches(
                            detailed_batches
                        )
                        state["pressure_detail_raw"][side_index] = values
                        state["pressure_detail_ns"][side_index] = now_ns
                        state["pressure_detail_valid"][side_index] = True
                        state["pressure_detail_coalesced"][side_index] += coalesced

            with source_received_ns.get_lock():
                received_ns = int(source_received_ns.value)
            active = (
                enabled.is_set() and received_ns > 0 and 0 <= now_ns - received_ns <= timeout_ns
            )

            if enabled.is_set() and received_ns > 0 and now_ns - received_ns > timeout_ns:
                enabled.clear()
                if was_active:
                    await stop_motion()
                    was_active = False
                    motion_may_be_active = False
                raise RuntimeError("Revo2 trigger input timed out; zero speed sent")

            if active:
                with targets.get_lock():
                    target_delta = np.asarray(targets[:], dtype=np.float64).reshape(2, 6)
                target = np.clip(baseline_array + target_delta, 0.0, 1.0)
                with sequence.get_lock():
                    current_sequence = int(sequence.value)
                limited = _bounded_slew(previous_array, target, command_period_s, slew_per_s)
                speed_values = [int(round(command_speed * 1000.0))] * REVO2_CHANNEL_COUNT
                motion_may_be_active = True
                await asyncio.gather(
                    *[
                        bounded(
                            client.set_finger_positions_and_speeds(
                                device.slave_id,
                                np.rint(values * 1000.0).astype(int).tolist(),
                                speed_values,
                            )
                        )
                        for client, device, values in zip(clients, devices, limited, strict=True)
                    ]
                )
                previous_array = limited
                state["command_position_raw"] = np.rint(limited * 1000.0).astype(
                    np.float32
                )
                was_active = True
                now = time.monotonic()
                if current_sequence != last_sequence and now - last_report >= 1.0:
                    print(
                        "[Revo2Trigger] "
                        f"seq={current_sequence} left={target[0].round(3).tolist()} "
                        f"right={target[1].round(3).tolist()}",
                        flush=True,
                    )
                    last_sequence = current_sequence
                    last_report = now
            elif was_active:
                await stop_motion()
                was_active = False
                motion_may_be_active = False

            if capture_data and time.monotonic() >= next_state_publish:
                with source_triggers.get_lock():
                    state["source_trigger"] = np.asarray(
                        source_triggers[:], dtype=np.float32
                    )
                with sequence.get_lock():
                    state["sequence"] = int(sequence.value)
                state["source_received_ns"] = received_ns
                state["publish_ns"] = time.monotonic_ns()
                _replace_latest(
                    state_queue,
                    {
                        key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in state.items()
                    },
                )
                next_state_publish += state_period_s
                if next_state_publish < time.monotonic() - state_period_s:
                    next_state_publish = time.monotonic() + state_period_s

            next_tick += command_period_s
            now = time.monotonic()
            if next_tick < now:
                missed_deadlines = int((now - next_tick) // command_period_s) + 1
                state["cmd_coalesced"] += missed_deadlines
                next_tick += missed_deadlines * command_period_s
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        cleanup_errors: list[str] = []
        try:
            if motion_may_be_active:
                try:
                    await stop_motion()
                except Exception as exc:
                    cleanup_errors.append(f"stop_motion={exc!r}")
        finally:
            for resource in collector_resources:
                try:
                    resource["collector"].stop()
                except Exception as exc:
                    cleanup_errors.append(f"collector_stop={exc!r}")
            for resource in collector_resources:
                wait = getattr(resource["collector"], "wait", None)
                if callable(wait):
                    try:
                        await asyncio.wait_for(asyncio.to_thread(wait), timeout=2.0)
                    except Exception as exc:
                        cleanup_errors.append(f"collector_wait={exc!r}")
            for client in clients:
                try:
                    await _close_client(sdk, client)
                except Exception as exc:
                    cleanup_errors.append(f"client_close={exc!r}")
            for lock_handle in locks:
                try:
                    lock_handle.close()
                except Exception as exc:
                    cleanup_errors.append(f"lock_close={exc!r}")
        if cleanup_errors:
            raise RuntimeError("Revo2 owner cleanup failed: " + "; ".join(cleanup_errors))


def _revo2_owner_main(*args: Any) -> None:
    status = args[7]
    try:
        asyncio.run(_run_revo2_owner(*args))
        status.send(("closed", None))
    except BaseException as exc:
        try:
            status.send(("error", repr(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        status.close()


class Revo2TriggerOutput:
    """Parent-side lifecycle for the isolated bilateral Revo2 owner."""

    def __init__(
        self,
        devices: Sequence[Revo2Device],
        *,
        deadman_s: float = 0.3,
        slew_per_s: float = 0.2,
        command_period_s: float = 0.04,
        thumb_delta: float = DEFAULT_THUMB_DELTA,
        finger_delta: float = DEFAULT_FINGER_DELTA,
        command_speed: float = 0.25,
        startup_timeout_s: float = 45.0,
        capture_data: bool = False,
        data_rate_hz: float = 20.0,
        touch_setup: bool = False,
    ) -> None:
        if tuple(device.side for device in devices) != ("left", "right"):
            raise ValueError("Revo2 devices must be ordered left then right")
        if not 0.1 <= deadman_s <= 2.0:
            raise ValueError("Revo2 deadman must be within [0.1, 2.0] seconds")
        if not 0.01 <= slew_per_s <= 0.5:
            raise ValueError("Revo2 slew must be within [0.01, 0.5] per second")
        if not 0.01 <= command_period_s <= 1.0:
            raise ValueError("Revo2 command period must be within [0.01, 1.0] seconds")
        if not math.isfinite(thumb_delta) or not 0.0 < thumb_delta <= 0.5:
            raise ValueError("Revo2 thumb_delta must be finite and within (0, 0.5]")
        if not math.isfinite(finger_delta) or not 0.0 < finger_delta <= 0.5:
            raise ValueError("Revo2 finger_delta must be finite and within (0, 0.5]")
        if not math.isfinite(command_speed) or not 0.01 <= command_speed <= 0.3:
            raise ValueError("Revo2 command_speed must be finite and within [0.01, 0.3]")
        if slew_per_s > command_speed:
            raise ValueError("Revo2 slew_per_s must not exceed command_speed")
        if not math.isfinite(data_rate_hz) or not 1.0 <= data_rate_hz <= 50.0:
            raise ValueError("Revo2 data_rate_hz must be finite and within [1, 50]")
        if not float(data_rate_hz).is_integer():
            raise ValueError("Revo2 data_rate_hz must be an integer frequency")

        self.devices = tuple(devices)
        self.thumb_delta = float(thumb_delta)
        self.finger_delta = float(finger_delta)
        self.deadman_ns = int(deadman_s * 1e9)
        self.startup_timeout_s = float(startup_timeout_s)
        self.capture_data = bool(capture_data)
        self.data_rate_hz = int(data_rate_hz)
        self.touch_setup = bool(touch_setup)
        ctx = mp.get_context("spawn")
        self._targets = ctx.Array("d", 2 * REVO2_CHANNEL_COUNT)
        self._source_triggers = ctx.Array("d", 2)
        self._source_received_ns = ctx.Value("q", 0)
        self._sequence = ctx.Value("q", 0)
        self._enabled = ctx.Event()
        self._stop = ctx.Event()
        self._status, child_status = ctx.Pipe(duplex=False)
        self._state_queue = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_revo2_owner_main,
            name="revo2-trigger-owner",
            args=(
                self.devices,
                self._targets,
                self._source_triggers,
                self._source_received_ns,
                self._sequence,
                self._enabled,
                self._stop,
                child_status,
                self._state_queue,
                float(deadman_s),
                float(slew_per_s),
                float(command_period_s),
                float(command_speed),
                self.capture_data,
                self.data_rate_hz,
                self.touch_setup,
            ),
        )
        self._child_status = child_status
        self._started = False
        self._closed = False
        self._error: str | None = None
        self.metadata: dict[str, Any] = {}
        self._latest_state: dict[str, Any] | None = None

    def start(self) -> "Revo2TriggerOutput":
        if self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner cannot be restarted")
        self._process.start()
        self._started = True
        self._child_status.close()
        if not self._status.poll(self.startup_timeout_s):
            self._stop.set()
            self._process.join(15.0)
            if self._process.is_alive():
                raise TimeoutError(
                    "Revo2 trigger owner startup timed out and did not acknowledge shutdown; "
                    "do not start another serial owner"
                )
            raise TimeoutError("Revo2 trigger owner startup timed out")
        kind, detail = self._status.recv()
        if kind != "ready":
            self._error = detail or "Revo2 trigger owner failed before ready"
            self._process.join(5.0)
            raise RuntimeError(self._error)
        self.metadata = dict(detail or {})
        return self

    def set_enabled(self, enabled: bool) -> None:
        if not self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner is not active")
        self.check_health()
        if enabled:
            with self._source_received_ns.get_lock():
                source_received_ns = int(self._source_received_ns.value)
                source_age_ns = time.monotonic_ns() - source_received_ns
                if source_received_ns <= 0 or not 0 <= source_age_ns <= self.deadman_ns:
                    raise RuntimeError("cannot enable Revo2 without a fresh trigger target")
            self._enabled.set()
        else:
            self._enabled.clear()

    def submit_triggers(
        self,
        left_trigger: float,
        right_trigger: float,
        *,
        source_sequence: int,
        source_received_monotonic_ns: int,
    ) -> None:
        if not self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner is not active")
        self.check_health()
        if source_received_monotonic_ns <= 0:
            raise ValueError("source receive timestamp must be positive")
        source_age_ns = time.monotonic_ns() - source_received_monotonic_ns
        if not 0 <= source_age_ns <= self.deadman_ns:
            raise ValueError("source receive timestamp is stale or in the future")
        if source_sequence <= 0:
            raise ValueError("source sequence must be positive")
        target = bilateral_trigger_deltas(
            left_trigger,
            right_trigger,
            thumb_delta=self.thumb_delta,
            finger_delta=self.finger_delta,
        )
        with self._targets.get_lock():
            self._targets[:] = target.reshape(-1).tolist()
        with self._source_triggers.get_lock():
            self._source_triggers[:] = [
                min(1.0, max(0.0, float(left_trigger))),
                min(1.0, max(0.0, float(right_trigger))),
            ]
        with self._source_received_ns.get_lock():
            self._source_received_ns.value = int(source_received_monotonic_ns)
        with self._sequence.get_lock():
            self._sequence.value = int(source_sequence)

    def get_latest_state(self) -> dict[str, Any] | None:
        """Return the newest fixed-shape state snapshot without blocking."""

        if not self.capture_data:
            return None
        self.check_health()
        while True:
            try:
                self._latest_state = self._state_queue.get_nowait()
            except queue.Empty:
                break
        return self._latest_state

    def check_health(self) -> None:
        if not self._started or self._closed:
            return
        while self._status.poll():
            try:
                kind, detail = self._status.recv()
            except EOFError:
                self._error = self._error or "Revo2 trigger owner status pipe closed"
                break
            if kind == "error":
                self._error = detail or "Revo2 trigger owner failed"
            elif kind == "closed" and not self._stop.is_set():
                self._error = "Revo2 trigger owner exited unexpectedly"
        if not self._process.is_alive() and not self._stop.is_set():
            self._error = self._error or "Revo2 trigger owner exited unexpectedly"
        if self._error:
            raise RuntimeError(self._error)

    def close(self) -> None:
        if self._closed:
            return
        self._enabled.clear()
        self._stop.set()
        if self._started:
            self._process.join(5.0)
            if self._process.is_alive():
                raise RuntimeError(
                    "Revo2 trigger owner did not acknowledge shutdown; do not start another owner"
                )
        else:
            self._child_status.close()
        self._status.close()
        self._state_queue.close()
        self._state_queue.join_thread()
        self._closed = True
