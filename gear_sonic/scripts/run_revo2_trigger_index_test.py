#!/usr/bin/env python3
"""Drive selected Revo2 fingers from the matching PICO Trigger.

This isolated commissioning test does not use SONIC, G1, or ZMQ.  Trigger zero
holds the measured startup position and trigger one adds only a bounded delta.
Stale XR timestamps stop the finger, and cleanup returns it to startup.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

DEVICES = {
    "left": {
        "port": "/dev/ttyUSB0",
        "slave_id": 126,
        "serial": "BCXTL2206J260000B",
    },
    "right": {
        "port": "/dev/ttyUSB1",
        "slave_id": 127,
        "serial": "BCXTR2212J2600027",
    },
}
INDEX_CHANNEL = 2
FOUR_FINGER_CHANNELS = (2, 3, 4, 5)
THUMB_CHANNELS = (0, 1)
ALL_CHANNELS = (0, 1, 2, 3, 4, 5)
CHANNEL_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--fingers",
        choices=("index", "four", "thumb", "all"),
        default="index",
        help="Control index, four non-thumb fingers, both thumbs, or all six",
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--max-delta",
        type=int,
        default=50,
        help="Maximum closing increment in normalized 0..1000 units",
    )
    parser.add_argument(
        "--thumb-delta",
        type=int,
        default=100,
        help="Thumb increment when controlling thumb or all channels",
    )
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--deadman-s", type=float, default=0.3)
    parser.add_argument("--xrt-ready-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--confirm-move",
        action="store_true",
        help="Required acknowledgement that this command moves real hardware",
    )
    return parser.parse_args()


async def await_result(value: Any, timeout_s: float = 3.0) -> Any:
    if inspect.isawaitable(value):
        return await asyncio.wait_for(value, timeout=timeout_s)
    return value


def xrt_service_is_listening() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", 60061)) == 0


async def run(args: argparse.Namespace) -> None:
    import fcntl

    import xrobotoolkit_sdk as xrt

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()

    device = DEVICES[args.side]
    client = None
    lock_handle = None
    xrt_initialized = False
    command_attempted = False
    baseline = None
    completed_normally = False
    try:
        resolved_port = Path(device["port"]).resolve(strict=True)
        lock_path = Path("/tmp") / f"g1-revo2-{resolved_port.name}.lock"
        lock_handle = lock_path.open("a", encoding="utf-8")
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = subprocess.run(
            ["fuser", str(resolved_port)],
            capture_output=True,
            text=True,
            check=False,
        )
        if owners.returncode != 1:
            detail = (owners.stdout + owners.stderr).strip() or "ownership check failed"
            raise RuntimeError(f"serial port is occupied: {detail}")

        client = await asyncio.wait_for(
            sdk.modbus_open(str(resolved_port), sdk.Baudrate.Baud460800),
            timeout=3.0,
        )
        info = await await_result(client.get_device_info(device["slave_id"]))
        if str(info.serial_number) != device["serial"]:
            raise RuntimeError(
                f"identity mismatch: expected {device['serial']}, got {info.serial_number}"
            )
        expected_hand = getattr(sdk.HandType, args.side.title())
        if info.hand_type != expected_hand:
            raise RuntimeError(
                f"hand side mismatch: expected {expected_hand}, got {info.hand_type}"
            )
        unit_mode = await await_result(client.get_finger_unit_mode(device["slave_id"]))
        if unit_mode != sdk.FingerUnitMode.Normalized:
            raise RuntimeError(f"expected Normalized unit mode, got {unit_mode}")
        status = await await_result(client.get_motor_status(device["slave_id"]))
        baseline = [int(value) for value in status.positions]
        if len(baseline) != CHANNEL_COUNT or not all(0 <= value <= 1000 for value in baseline):
            raise RuntimeError(f"invalid baseline positions: {baseline}")
        controlled_channels = {
            "index": (INDEX_CHANNEL,),
            "four": FOUR_FINGER_CHANNELS,
            "thumb": THUMB_CHANNELS,
            "all": ALL_CHANNELS,
        }[args.fingers]
        deltas = {
            channel: (args.thumb_delta if channel in THUMB_CHANNELS else args.max_delta)
            for channel in controlled_channels
        }
        finger_ids_by_channel = (
            sdk.FingerId.Thumb,
            sdk.FingerId.ThumbAux,
            sdk.FingerId.Index,
            sdk.FingerId.Middle,
            sdk.FingerId.Ring,
            sdk.FingerId.Pinky,
        )
        controlled_finger_ids = tuple(
            finger_ids_by_channel[channel] for channel in controlled_channels
        )
        if any(baseline[channel] + deltas[channel] > 1000 for channel in controlled_channels):
            raise RuntimeError(
                f"baseline {baseline} cannot accept configured channel deltas {deltas}"
            )

        if xrt_service_is_listening():
            print("[trigger-index] reusing RoboticsService on port 60061", flush=True)
        else:
            print("[trigger-index] starting RoboticsService...", flush=True)
            subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
            service_deadline = time.monotonic() + 10.0
            while time.monotonic() < service_deadline:
                if xrt_service_is_listening():
                    break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("RoboticsService did not listen on port 60061")
        xrt.init()
        xrt_initialized = True
        max_target = list(baseline)
        for channel in controlled_channels:
            max_target[channel] += deltas[channel]
        print(
            f"[trigger-index] verified {args.side} hand; fingers={args.fingers} "
            f"baseline={baseline} max_target={max_target}",
            flush=True,
        )
        print(
            "[trigger-index] SONIC/G1/ZMQ disabled; squeeze and release only the "
            f"{args.side} Trigger",
            flush=True,
        )

        getter = xrt.get_left_trigger if args.side == "left" else xrt.get_right_trigger
        print(
            "[trigger-index] waiting for live XR timestamps and a released Trigger...",
            flush=True,
        )
        ready_deadline = time.monotonic() + args.xrt_ready_timeout_s
        ready_stamp = None
        timestamp_updates = 0
        release_warning_printed = False
        while time.monotonic() < ready_deadline:
            source_stamp = int(xrt.get_time_stamp_ns())
            if source_stamp > 0 and source_stamp != ready_stamp:
                ready_stamp = source_stamp
                timestamp_updates += 1
            trigger = min(1.0, max(0.0, float(getter())))
            if timestamp_updates >= 2 and trigger <= 0.05:
                break
            if timestamp_updates >= 2 and trigger > 0.05 and not release_warning_printed:
                print(
                    f"[trigger-index] release the {args.side} Trigger to arm control",
                    flush=True,
                )
                release_warning_printed = True
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("XR input did not become live with the matching Trigger released")
        print(
            f"[trigger-index] XR ready after {timestamp_updates} timestamp updates",
            flush=True,
        )

        period = 1.0 / args.hz
        deadline = time.monotonic() + args.duration_s
        next_tick = time.monotonic()
        next_report = float("-inf")
        last_source_stamp = ready_stamp
        last_source_received = time.monotonic()
        last_target = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            source_stamp = int(xrt.get_time_stamp_ns())
            if source_stamp != last_source_stamp:
                last_source_stamp = source_stamp
                last_source_received = now
            elif now - last_source_received > args.deadman_s:
                for finger_id in controlled_finger_ids:
                    await await_result(client.set_finger_speed(device["slave_id"], finger_id, 0))
                raise RuntimeError("XR input timestamp timed out; finger stop sent")

            trigger = min(1.0, max(0.0, float(getter())))
            target = list(baseline)
            for channel in controlled_channels:
                target[channel] += round(trigger * deltas[channel])
            target_key = tuple(target[channel] for channel in controlled_channels)
            if target_key != last_target:
                command_attempted = True
                if args.fingers == "index":
                    await await_result(
                        client.set_finger_position_with_speed(
                            device["slave_id"],
                            sdk.FingerId.Index,
                            target[INDEX_CHANNEL],
                            args.speed,
                        )
                    )
                else:
                    await await_result(
                        client.set_finger_positions_and_speeds(
                            device["slave_id"], target, [args.speed] * CHANNEL_COUNT
                        )
                    )
                last_target = target_key

            if now >= next_report:
                current = await await_result(client.get_motor_status(device["slave_id"]))
                target_values = [target[channel] for channel in controlled_channels]
                measured_values = [current.positions[channel] for channel in controlled_channels]
                print(
                    f"[trigger-index] trigger={trigger:.3f} "
                    f"target={target_values} measured={measured_values}",
                    flush=True,
                )
                next_report = now + 0.25

            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        completed_normally = True
    finally:
        if client is not None:
            if command_attempted and baseline is not None:
                try:
                    if args.fingers == "index":
                        await await_result(
                            client.set_finger_position_with_speed(
                                device["slave_id"],
                                sdk.FingerId.Index,
                                baseline[INDEX_CHANNEL],
                                args.speed,
                            )
                        )
                    else:
                        await await_result(
                            client.set_finger_positions_and_speeds(
                                device["slave_id"],
                                baseline,
                                [args.speed] * CHANNEL_COUNT,
                            )
                        )
                    await asyncio.sleep(1.0)
                    restored = await await_result(client.get_motor_status(device["slave_id"]))
                    restored_values = [
                        restored.positions[channel] for channel in controlled_channels
                    ]
                    print(
                        f"[trigger-index] restored={restored_values}",
                        flush=True,
                    )
                finally:
                    for finger_id in controlled_finger_ids:
                        await await_result(
                            client.set_finger_speed(device["slave_id"], finger_id, 0)
                        )
                    print(
                        f"[trigger-index] {args.fingers}-finger zero-speed stop sent",
                        flush=True,
                    )
            await await_result(sdk.modbus_close(client), timeout_s=1.0)
            print("[trigger-index] serial closed", flush=True)
        if xrt_initialized:
            xrt.close()
            print("[trigger-index] XRoboToolkit closed", flush=True)
        if lock_handle is not None:
            lock_handle.close()
        if completed_normally:
            print("[trigger-index] test complete", flush=True)


def main() -> None:
    args = parse_args()
    if not args.confirm_move:
        raise SystemExit("refusing to move hardware without --confirm-move")
    if not 1.0 <= args.duration_s <= 60.0:
        raise SystemExit("--duration-s must be within [1, 60]")
    if not 5.0 <= args.hz <= 50.0:
        raise SystemExit("--hz must be within [5, 50]")
    if not 1 <= args.max_delta <= 200:
        raise SystemExit("--max-delta must be within [1, 200]")
    if not 1 <= args.thumb_delta <= 200:
        raise SystemExit("--thumb-delta must be within [1, 200]")
    if not 1 <= args.speed <= 300:
        raise SystemExit("--speed must be within [1, 300]")
    if not 0.1 <= args.deadman_s <= 1.0:
        raise SystemExit("--deadman-s must be within [0.1, 1.0]")
    if not 1.0 <= args.xrt_ready_timeout_s <= 60.0:
        raise SystemExit("--xrt-ready-timeout-s must be within [1, 60]")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[trigger-index] interrupted by user", flush=True)
    except RuntimeError as exc:
        raise SystemExit(f"[trigger-index] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
