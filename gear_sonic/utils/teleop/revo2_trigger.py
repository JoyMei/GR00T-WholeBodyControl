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
import subprocess
import time
from dataclasses import dataclass
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
) -> None:
    # Hardware-only dependencies stay in the child process so mapping/tests run
    # on Windows and on machines without the BrainCo SDK.
    import fcntl

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()

    clients: list[Any] = []
    locks: list[Any] = []
    previous: list[np.ndarray] = []
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
                    break

        previous_array = np.asarray(previous, dtype=np.float64)
        baseline_array = previous_array.copy()
        status.send(("ready", None))
        next_tick = time.monotonic()
        timeout_ns = int(deadman_s * 1e9)

        while not stop.is_set():
            now_ns = time.monotonic_ns()
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

        self.devices = tuple(devices)
        self.thumb_delta = float(thumb_delta)
        self.finger_delta = float(finger_delta)
        self.startup_timeout_s = float(startup_timeout_s)
        ctx = mp.get_context("spawn")
        self._targets = ctx.Array("d", 2 * REVO2_CHANNEL_COUNT)
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
        target = bilateral_trigger_deltas(
            left_trigger,
            right_trigger,
            thumb_delta=self.thumb_delta,
            finger_delta=self.finger_delta,
        )
        with self._targets.get_lock():
            self._targets[:] = target.reshape(-1).tolist()
        with self._source_received_ns.get_lock():
            self._source_received_ns.value = int(source_received_monotonic_ns)
        with self._sequence.get_lock():
            self._sequence.value += 1

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
