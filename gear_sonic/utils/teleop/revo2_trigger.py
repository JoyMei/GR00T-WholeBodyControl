"""Continuous PICO trigger control for bilateral BrainCo Revo2 hands.

The manager owns input and this module owns the Revo2 serial devices in a
separate process. Each mode combines a fixed per-channel offset with a bounded
Trigger-scaled delta, both relative to a calibrated open pose when configured,
or the measured startup pose as a compatibility fallback. Modes with nonzero
offsets intentionally move on selection.
"""

from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REVO2_CHANNEL_NAMES = (
    "thumb_flex",
    "thumb_rotation",
    "index",
    "middle",
    "ring",
    "pinky",
)
REVO2_CHANNEL_COUNT = len(REVO2_CHANNEL_NAMES)
DEFAULT_THUMB_DELTA = 0.15
DEFAULT_FINGER_DELTA = 0.20
REVO2_MODE_NAME_BYTES = 32


@dataclass(frozen=True)
class Revo2TriggerMode:
    """One named Trigger-to-Revo2 six-channel mapping profile."""

    name: str
    channels: np.ndarray
    offsets: np.ndarray = field(
        default_factory=lambda: np.zeros(REVO2_CHANNEL_COUNT)
    )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        channels = np.asarray(self.channels, dtype=np.float64)
        offsets = np.asarray(self.offsets, dtype=np.float64)
        if not name:
            raise ValueError("Revo2 trigger mode name must be non-empty")
        if len(name.encode("utf-8")) > REVO2_MODE_NAME_BYTES:
            raise ValueError(
                f"Revo2 trigger mode name must fit within {REVO2_MODE_NAME_BYTES} UTF-8 bytes"
            )
        if channels.shape != (REVO2_CHANNEL_COUNT,):
            raise ValueError(
                f"Revo2 trigger mode channels must have shape ({REVO2_CHANNEL_COUNT},)"
            )
        if not np.isfinite(channels).all():
            raise ValueError("Revo2 trigger mode channels must be finite")
        if np.any((channels < 0.0) | (channels > 1.0)):
            raise ValueError("Revo2 trigger mode channels must be within [0, 1]")
        if offsets.shape != (REVO2_CHANNEL_COUNT,):
            raise ValueError(
                f"Revo2 trigger mode offsets must have shape ({REVO2_CHANNEL_COUNT},)"
            )
        if not np.isfinite(offsets).all():
            raise ValueError("Revo2 trigger mode offsets must be finite")
        if np.any((offsets < 0.0) | (offsets > 1.0)):
            raise ValueError("Revo2 trigger mode offsets must be within [0, 1]")
        channels = np.array(channels, dtype=np.float64, copy=True)
        offsets = np.array(offsets, dtype=np.float64, copy=True)
        channels.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "offsets", offsets)

    @classmethod
    def from_mapping(cls, value: Any) -> "Revo2TriggerMode":
        if not isinstance(value, dict):
            raise ValueError("each Revo2 trigger mode must be a JSON object")
        return cls(
            name=value.get("name", ""),
            channels=value.get("channels", []),
            offsets=value.get("offsets", np.zeros(REVO2_CHANNEL_COUNT)),
        )


def default_revo2_trigger_modes(
    *,
    thumb_delta: float = DEFAULT_THUMB_DELTA,
    finger_delta: float = DEFAULT_FINGER_DELTA,
) -> tuple[Revo2TriggerMode, ...]:
    """Return the built-in all-fingers, pinch, and point profiles."""

    return (
        Revo2TriggerMode(
            "all_fingers",
            [thumb_delta, thumb_delta, finger_delta, finger_delta, finger_delta, finger_delta],
        ),
        Revo2TriggerMode(
            "pinch",
            [thumb_delta, thumb_delta, finger_delta, 0.0, 0.0, 0.0],
        ),
        Revo2TriggerMode(
            "point",
            [thumb_delta, thumb_delta, 0.0, finger_delta, finger_delta, finger_delta],
        ),
    )


def load_revo2_trigger_modes(
    path: str | Path | None = None,
    *,
    thumb_delta: float = DEFAULT_THUMB_DELTA,
    finger_delta: float = DEFAULT_FINGER_DELTA,
) -> tuple[Revo2TriggerMode, ...]:
    """Load named profiles, or return the three built-in profiles."""

    if path is None:
        modes = default_revo2_trigger_modes(
            thumb_delta=thumb_delta,
            finger_delta=finger_delta,
        )
    else:
        config_path = Path(path).expanduser().resolve(strict=True)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("modes"), list):
            raise ValueError("Revo2 trigger mode config must contain a 'modes' list")
        modes = tuple(Revo2TriggerMode.from_mapping(value) for value in raw["modes"])
    if not modes:
        raise ValueError("Revo2 trigger mode config must contain at least one mode")
    names = [mode.name for mode in modes]
    if len(set(names)) != len(names):
        raise ValueError("Revo2 trigger mode names must be unique")
    return modes


def encode_revo2_mode_name(name: str) -> np.ndarray:
    """Encode a profile name as fixed-width UTF-8 int32 values for ZMQ/LeRobot."""

    encoded = str(name).encode("utf-8")
    if len(encoded) > REVO2_MODE_NAME_BYTES:
        raise ValueError(
            f"Revo2 trigger mode name must fit within {REVO2_MODE_NAME_BYTES} UTF-8 bytes"
        )
    result = np.zeros(REVO2_MODE_NAME_BYTES, dtype=np.int32)
    result[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int32)
    return result


def revo2_triggers_released(
    left_trigger: float,
    right_trigger: float,
    *,
    threshold: float = 0.05,
) -> bool:
    """Return whether both Triggers are finite and below the switch threshold."""

    values = np.asarray([left_trigger, right_trigger], dtype=np.float64)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Trigger release threshold must be finite and within [0, 1]")
    return bool(np.isfinite(values).all() and np.all(values <= threshold))


def trigger_to_revo2_deltas(
    trigger: float,
    *,
    thumb_delta: float = DEFAULT_THUMB_DELTA,
    finger_delta: float = DEFAULT_FINGER_DELTA,
    channels: Sequence[float] | np.ndarray | None = None,
    offsets: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Map one Trigger to fixed plus proportional six-channel deltas."""

    value = float(trigger)
    thumb_limit = float(thumb_delta)
    finger_limit = float(finger_delta)
    if not math.isfinite(value):
        raise ValueError("trigger must be finite")
    if not math.isfinite(thumb_limit) or not 0.0 < thumb_limit <= 1.0:
        raise ValueError("thumb_delta must be finite and within (0, 1]")
    if not math.isfinite(finger_limit) or not 0.0 < finger_limit <= 1.0:
        raise ValueError("finger_delta must be finite and within (0, 1]")
    value = min(1.0, max(0.0, value))
    mode = Revo2TriggerMode(
        "mapping",
        (
            [thumb_limit, thumb_limit, finger_limit, finger_limit, finger_limit, finger_limit]
            if channels is None
            else channels
        ),
        np.zeros(REVO2_CHANNEL_COUNT) if offsets is None else offsets,
    )
    return mode.offsets + value * mode.channels


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
    open_positions: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "Revo2Device":
        if not isinstance(value, dict):
            raise ValueError("each Revo2 device must be a JSON object")
        side = value.get("side")
        port = value.get("port")
        serial = value.get("serial")
        slave_id = value.get("slave_id")
        open_positions_value = value.get("open_positions")
        if side not in {"left", "right"}:
            raise ValueError("Revo2 side must be 'left' or 'right'")
        if not isinstance(port, str) or not port:
            raise ValueError("Revo2 port must be a non-empty string")
        if not isinstance(serial, str) or not serial:
            raise ValueError("Revo2 serial must be a non-empty string")
        if type(slave_id) is not int or not 0 <= slave_id <= 247:
            raise ValueError("Revo2 slave_id must be an integer in [0, 247]")
        open_positions = None
        if open_positions_value is not None:
            values = np.asarray(open_positions_value, dtype=np.float64)
            if values.shape != (REVO2_CHANNEL_COUNT,):
                raise ValueError(
                    f"Revo2 open_positions must have shape ({REVO2_CHANNEL_COUNT},)"
                )
            if not np.isfinite(values).all():
                raise ValueError("Revo2 open_positions must be finite")
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError("Revo2 open_positions must be within [0, 1]")
            open_positions = tuple(float(position) for position in values)
        return cls(
            side=side,
            port=port,
            serial=serial,
            slave_id=slave_id,
            open_positions=open_positions,
        )


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


def _freeze_locked_targets(
    previous: np.ndarray,
    target: np.ndarray,
    locked: np.ndarray,
    was_locked: np.ndarray,
    held: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze newly locked hands at their last slew-limited command."""

    newly_locked = locked & ~was_locked
    held = held.copy()
    held[newly_locked] = previous[newly_locked]
    effective_target = target.copy()
    effective_target[locked] = held[locked]
    return effective_target, held


class Revo2GripGesture:
    """Recognize one Grip press/release without stealing a modifier chord."""

    def __init__(self, press_threshold: float = 0.6, release_threshold: float = 0.3) -> None:
        if not 0.0 <= release_threshold < press_threshold <= 1.0:
            raise ValueError("Grip thresholds must satisfy 0 <= release < press <= 1")
        self.press_threshold = float(press_threshold)
        self.release_threshold = float(release_threshold)
        self.pressed = False
        self.consumed = False

    def update(self, value: float, *, consume: bool = False) -> bool:
        """Return True once when an unconsumed Grip press is released."""

        try:
            grip = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(grip):
            return False

        if not self.pressed:
            if grip >= self.press_threshold:
                self.pressed = True
                self.consumed = bool(consume)
            return False

        self.consumed = self.consumed or bool(consume)
        if grip > self.release_threshold:
            return False

        should_toggle = not self.consumed
        self.pressed = False
        self.consumed = False
        return should_toggle


async def _close_client(sdk: Any, client: Any) -> None:
    import inspect

    completion = sdk.modbus_close(client)
    if inspect.isawaitable(completion):
        await asyncio.wait_for(completion, timeout=1.0)


async def _run_revo2_owner(
    devices: Sequence[Revo2Device],
    targets: Any,
    source_received_ns: Any,
    sequence: Any,
    enabled: Any,
    stop: Any,
    status: Any,
    deadman_s: float,
    slew_per_s: float,
    command_period_s: float,
    command_speed: float,
    hand_locks: Any,
) -> None:
    # Hardware-only dependencies stay in the child process so mapping/tests run
    # on Windows and on machines without the BrainCo SDK.
    import fcntl

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()

    clients: list[Any] = []
    locks: list[Any] = []
    previous: list[np.ndarray] = []
    open_targets: list[np.ndarray] = []
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
        for device in devices:
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
                    measured = np.asarray(motor_status.positions, dtype=np.float64) / 1000.0
                    if measured.shape != (REVO2_CHANNEL_COUNT,) or not np.isfinite(measured).all():
                        raise ValueError("invalid measured Revo2 startup positions")
                    if np.any((measured < 0.0) | (measured > 1.0)):
                        raise ValueError("measured Revo2 startup positions are outside [0, 1]")
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
                    previous.append(measured)
                    open_targets.append(
                        measured.copy()
                        if device.open_positions is None
                        else np.asarray(device.open_positions, dtype=np.float64)
                    )
                    break

        previous_array = np.asarray(previous, dtype=np.float64)
        baseline_array = np.asarray(open_targets, dtype=np.float64)
        if not np.allclose(previous_array, baseline_array, rtol=0.0, atol=0.02):
            print(
                "[Revo2Trigger] returning both hands to calibrated open positions...",
                flush=True,
            )
            homing_deadline = time.monotonic() + (
                1.0 / min(slew_per_s, command_speed) + 3.0
            )
            homing_next_tick = time.monotonic()
            while True:
                limited = _bounded_slew(
                    previous_array,
                    baseline_array,
                    command_period_s,
                    slew_per_s,
                )
                speed_values = [
                    int(round(command_speed * 1000.0))
                ] * REVO2_CHANNEL_COUNT
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
                        for client, device, values in zip(
                            clients, devices, limited, strict=True
                        )
                    ]
                )
                previous_array = limited
                if np.allclose(
                    previous_array, baseline_array, rtol=0.0, atol=1e-9
                ):
                    motor_statuses = await asyncio.gather(
                        *[
                            bounded(client.get_motor_status(device.slave_id), timeout=0.5)
                            for client, device in zip(clients, devices, strict=True)
                        ]
                    )
                    measured_array = np.asarray(
                        [motor_status.positions for motor_status in motor_statuses],
                        dtype=np.float64,
                    ) / 1000.0
                    if np.allclose(
                        measured_array, baseline_array, rtol=0.0, atol=0.02
                    ):
                        await stop_motion()
                        motion_may_be_active = False
                        print(
                            "[Revo2Trigger] both hands are at calibrated open positions",
                            flush=True,
                        )
                        break
                if time.monotonic() >= homing_deadline:
                    raise TimeoutError(
                        "Revo2 startup opening did not reach calibrated positions"
                    )
                homing_next_tick += command_period_s
                await asyncio.sleep(max(0.0, homing_next_tick - time.monotonic()))

        status.send(("ready", None))
        next_tick = time.monotonic()
        timeout_ns = int(deadman_s * 1e9)
        held_array = previous_array.copy()
        previous_lock_state = np.zeros(2, dtype=bool)

        while not stop.is_set():
            now_ns = time.monotonic_ns()
            with source_received_ns.get_lock():
                received_ns = int(source_received_ns.value)
            active = (
                enabled.is_set()
                and received_ns > 0
                and 0 <= now_ns - received_ns <= timeout_ns
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
                with hand_locks.get_lock():
                    lock_state = np.asarray(hand_locks[:], dtype=bool)
                target, held_array = _freeze_locked_targets(
                    previous_array,
                    target,
                    lock_state,
                    previous_lock_state,
                    held_array,
                )
                previous_lock_state = lock_state
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

            next_tick += command_period_s
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        try:
            if motion_may_be_active:
                await stop_motion()
        finally:
            for client in clients:
                await _close_client(sdk, client)
            for lock_handle in locks:
                lock_handle.close()


def _revo2_owner_main(*args: Any) -> None:
    status = args[6]
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
        slew_per_s: float = 0.4,
        command_period_s: float = 0.04,
        thumb_delta: float = DEFAULT_THUMB_DELTA,
        finger_delta: float = DEFAULT_FINGER_DELTA,
        command_speed: float = 0.25,
        startup_timeout_s: float = 45.0,
        modes: Sequence[Revo2TriggerMode] | None = None,
    ) -> None:
        if tuple(device.side for device in devices) != ("left", "right"):
            raise ValueError("Revo2 devices must be ordered left then right")
        if not 0.1 <= deadman_s <= 2.0:
            raise ValueError("Revo2 deadman must be within [0.1, 2.0] seconds")
        if not 0.01 <= slew_per_s <= 0.5:
            raise ValueError("Revo2 slew must be within [0.01, 0.5] per second")
        if not 0.01 <= command_period_s <= 1.0:
            raise ValueError("Revo2 command period must be within [0.01, 1.0] seconds")
        if not math.isfinite(thumb_delta) or not 0.0 < thumb_delta <= 1.0:
            raise ValueError("Revo2 thumb_delta must be finite and within (0, 1]")
        if not math.isfinite(finger_delta) or not 0.0 < finger_delta <= 1.0:
            raise ValueError("Revo2 finger_delta must be finite and within (0, 1]")
        if not math.isfinite(command_speed) or not 0.01 <= command_speed <= 0.3:
            raise ValueError("Revo2 command_speed must be finite and within [0.01, 0.3]")

        self.devices = tuple(devices)
        self.thumb_delta = float(thumb_delta)
        self.finger_delta = float(finger_delta)
        self.modes = (
            tuple(modes)
            if modes is not None
            else (
                Revo2TriggerMode(
                    "all_fingers",
                    [
                        thumb_delta,
                        thumb_delta,
                        finger_delta,
                        finger_delta,
                        finger_delta,
                        finger_delta,
                    ],
                ),
            )
        )
        if not self.modes:
            raise ValueError("Revo2 trigger output requires at least one mode")
        if not all(isinstance(mode, Revo2TriggerMode) for mode in self.modes):
            raise TypeError("modes must contain Revo2TriggerMode values")
        if len({mode.name for mode in self.modes}) != len(self.modes):
            raise ValueError("Revo2 trigger mode names must be unique")
        self._mode_index = 0
        self.startup_timeout_s = max(
            float(startup_timeout_s),
            1.0 / min(float(slew_per_s), float(command_speed)) + 5.0,
        )
        ctx = mp.get_context("spawn")
        self._targets = ctx.Array("d", 2 * REVO2_CHANNEL_COUNT)
        self._hand_locks = ctx.Array("b", 2)
        self._source_received_ns = ctx.Value("q", 0)
        self._sequence = ctx.Value("q", 0)
        self._enabled = ctx.Event()
        self._stop = ctx.Event()
        self._status, child_status = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_revo2_owner_main,
            name="revo2-trigger-owner",
            args=(
                self.devices,
                self._targets,
                self._source_received_ns,
                self._sequence,
                self._enabled,
                self._stop,
                child_status,
                float(deadman_s),
                float(slew_per_s),
                float(command_period_s),
                float(command_speed),
                self._hand_locks,
            ),
        )
        self._child_status = child_status
        self._started = False
        self._closed = False
        self._error: str | None = None

    def start(self) -> "Revo2TriggerOutput":
        if self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner cannot be restarted")
        self._process.start()
        self._started = True
        self._child_status.close()
        if not self._status.poll(self.startup_timeout_s):
            self._stop.set()
            self._process.join(5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(5.0)
            raise TimeoutError("Revo2 trigger owner startup timed out")
        kind, detail = self._status.recv()
        if kind != "ready":
            self._error = detail or "Revo2 trigger owner failed before ready"
            self._process.join(5.0)
            raise RuntimeError(self._error)
        return self

    def set_enabled(self, enabled: bool) -> None:
        if not self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner is not active")
        self.check_health()
        if enabled:
            with self._source_received_ns.get_lock():
                if self._source_received_ns.value <= 0:
                    raise RuntimeError("cannot enable Revo2 without a fresh trigger target")
            self._enabled.set()
        else:
            self._enabled.clear()

    def submit_triggers(
        self,
        left_trigger: float,
        right_trigger: float,
        *,
        source_received_monotonic_ns: int,
    ) -> None:
        if not self._started or self._closed:
            raise RuntimeError("Revo2 trigger owner is not active")
        self.check_health()
        if source_received_monotonic_ns <= 0:
            raise ValueError("source receive timestamp must be positive")
        mode = self.current_mode
        target = np.stack(
            (
                trigger_to_revo2_deltas(
                    left_trigger,
                    channels=mode.channels,
                    offsets=mode.offsets,
                ),
                trigger_to_revo2_deltas(
                    right_trigger,
                    channels=mode.channels,
                    offsets=mode.offsets,
                ),
            )
        )
        with self._targets.get_lock():
            self._targets[:] = target.reshape(-1).tolist()
        with self._source_received_ns.get_lock():
            self._source_received_ns.value = int(source_received_monotonic_ns)
        with self._sequence.get_lock():
            self._sequence.value += 1

    @property
    def current_mode_index(self) -> int:
        return self._mode_index

    @property
    def current_mode(self) -> Revo2TriggerMode:
        return self.modes[self._mode_index]

    @property
    def hand_locks(self) -> tuple[bool, bool]:
        with self._hand_locks.get_lock():
            return bool(self._hand_locks[0]), bool(self._hand_locks[1])

    @property
    def any_hand_locked(self) -> bool:
        return any(self.hand_locks)

    def set_hand_locked(self, side: str, locked: bool) -> None:
        """Lock one hand at its last slew-limited command position."""

        if self._closed:
            raise RuntimeError("Revo2 trigger owner is closed")
        try:
            index = ("left", "right").index(side)
        except ValueError as exc:
            raise ValueError("Revo2 hand side must be 'left' or 'right'") from exc
        with self._hand_locks.get_lock():
            self._hand_locks[index] = bool(locked)

    def cycle_mode(self) -> Revo2TriggerMode:
        """Select and return the next mode, wrapping after the final mode."""

        if self._closed:
            raise RuntimeError("Revo2 trigger owner is closed")
        if self.any_hand_locked:
            raise RuntimeError("cannot change Revo2 mode while a hand is locked")
        self._mode_index = (self._mode_index + 1) % len(self.modes)
        return self.current_mode

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
        self._closed = True
