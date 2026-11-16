#!/usr/bin/env python3
"""Perform one low-speed Revo2 finger nudge and return to start.

This is an isolated hardware commissioning test.  It verifies the expected
device identity and normalized unit mode, changes either the index or the four
non-thumb fingers by a small bounded amount, returns to the measured starting
position, and sends a zero-speed stop during cleanup.  It does not use SONIC,
G1, XR, or ZMQ.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import subprocess
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
CHANNEL_COUNT = 6
INDEX_CHANNEL = 2
FOUR_FINGER_CHANNELS = (2, 3, 4, 5)
THUMB_CHANNELS = (0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--fingers",
        choices=("index", "four", "thumb"),
        default="index",
        help="Move index, four non-thumb fingers, or both thumb channels",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=20,
        help="Positive position increment in normalized 0..1000 units",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=50,
        help="Finger speed in normalized 0..1000 units",
    )
    parser.add_argument("--hold-s", type=float, default=1.0)
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


async def main_async(args: argparse.Namespace) -> None:
    import fcntl

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()

    device = DEVICES[args.side]
    client = None
    lock_handle = None
    command_attempted = False
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
        start = [int(value) for value in status.positions]
        if len(start) != CHANNEL_COUNT or not all(0 <= value <= 1000 for value in start):
            raise RuntimeError(f"invalid starting positions: {start}")
        controlled_channels = {
            "index": (INDEX_CHANNEL,),
            "four": FOUR_FINGER_CHANNELS,
            "thumb": THUMB_CHANNELS,
        }[args.fingers]
        if any(start[channel] + args.delta > 1000 for channel in controlled_channels):
            raise RuntimeError(f"starting positions {start} plus delta {args.delta} exceed 1000")

        target = list(start)
        for channel in controlled_channels:
            target[channel] += args.delta
        print(
            f"[nudge] verified {args.side} hand serial={info.serial_number} "
            f"port={device['port']} slave_id={device['slave_id']}",
            flush=True,
        )
        print(
            f"[nudge] start={start} target={target} speed={args.speed} " f"fingers={args.fingers}",
            flush=True,
        )

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
        await asyncio.sleep(args.hold_s)
        nudged = await await_result(client.get_motor_status(device["slave_id"]))
        print(f"[nudge] reached={list(nudged.positions)}", flush=True)

        if args.fingers == "index":
            await await_result(
                client.set_finger_position_with_speed(
                    device["slave_id"],
                    sdk.FingerId.Index,
                    start[INDEX_CHANNEL],
                    args.speed,
                )
            )
        else:
            await await_result(
                client.set_finger_positions_and_speeds(
                    device["slave_id"], start, [args.speed] * CHANNEL_COUNT
                )
            )
        await asyncio.sleep(args.hold_s)
        restored = await await_result(client.get_motor_status(device["slave_id"]))
        print(f"[nudge] restored={list(restored.positions)}", flush=True)
    finally:
        if client is not None:
            if command_attempted:
                try:
                    finger_ids = {
                        "index": (sdk.FingerId.Index,),
                        "four": (
                            sdk.FingerId.Index,
                            sdk.FingerId.Middle,
                            sdk.FingerId.Ring,
                            sdk.FingerId.Pinky,
                        ),
                        "thumb": (sdk.FingerId.Thumb, sdk.FingerId.ThumbAux),
                    }[args.fingers]
                    for finger_id in finger_ids:
                        await await_result(
                            client.set_finger_speed(device["slave_id"], finger_id, 0)
                        )
                    print(
                        f"[nudge] {args.fingers}-finger zero-speed stop sent",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[nudge] WARNING: zero-speed stop failed: {exc}", flush=True)
            await await_result(sdk.modbus_close(client), timeout_s=1.0)
            print("[nudge] serial closed", flush=True)
        if lock_handle is not None:
            lock_handle.close()


def main() -> None:
    args = parse_args()
    if not args.confirm_move:
        raise SystemExit("refusing to move hardware without --confirm-move")
    if not 1 <= args.delta <= 200:
        raise SystemExit("--delta must be within [1, 200]")
    if not 1 <= args.speed <= 300:
        raise SystemExit("--speed must be within [1, 300]")
    if not 0.2 <= args.hold_s <= 3.0:
        raise SystemExit("--hold-s must be within [0.2, 3.0]")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
