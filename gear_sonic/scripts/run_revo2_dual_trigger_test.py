#!/usr/bin/env python3
"""Drive both Revo2 hands from the matching PICO Triggers without G1/SONIC."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

DEVICES = (
    {
        "side": "left",
        "port": "/dev/ttyUSB0",
        "slave_id": 126,
        "serial": "BCXTL2206J260000B",
    },
    {
        "side": "right",
        "port": "/dev/ttyUSB1",
        "slave_id": 127,
        "serial": "BCXTR2212J2600027",
    },
)
CHANNEL_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--finger-delta", type=int, default=200)
    parser.add_argument("--thumb-delta", type=int, default=150)
    parser.add_argument("--speed", type=int, default=250)
    parser.add_argument("--deadman-s", type=float, default=0.3)
    parser.add_argument("--xrt-ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--confirm-move", action="store_true")
    return parser.parse_args()


async def await_result(value: Any, timeout_s: float = 3.0) -> Any:
    if inspect.isawaitable(value):
        return await asyncio.wait_for(value, timeout=timeout_s)
    return value


def xrt_service_is_listening() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", 60061)) == 0


async def stop_hands(clients: list[Any], sdk: Any) -> None:
    finger_ids = (
        sdk.FingerId.Thumb,
        sdk.FingerId.ThumbAux,
        sdk.FingerId.Index,
        sdk.FingerId.Middle,
        sdk.FingerId.Ring,
        sdk.FingerId.Pinky,
    )
    for client, device in zip(clients, DEVICES, strict=True):
        for finger_id in finger_ids:
            await await_result(client.set_finger_speed(device["slave_id"], finger_id, 0))


async def run(args: argparse.Namespace) -> None:
    import fcntl

    import xrobotoolkit_sdk as xrt

    from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

    sdk = load_brainco_sdk()
    clients: list[Any] = []
    locks: list[Any] = []
    baselines: list[list[int]] = []
    xrt_initialized = False
    command_attempted = False
    completed_normally = False
    deltas = (
        args.thumb_delta,
        args.thumb_delta,
        args.finger_delta,
        args.finger_delta,
        args.finger_delta,
        args.finger_delta,
    )

    try:
        for device in DEVICES:
            resolved_port = Path(device["port"]).resolve(strict=True)
            lock_path = Path("/tmp") / f"g1-revo2-{resolved_port.name}.lock"
            lock_handle = lock_path.open("a", encoding="utf-8")
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock_handle)
            owners = subprocess.run(
                ["fuser", str(resolved_port)],
                capture_output=True,
                text=True,
                check=False,
            )
            if owners.returncode != 1:
                detail = (owners.stdout + owners.stderr).strip() or "ownership check failed"
                raise RuntimeError(f"serial port is occupied: {resolved_port}: {detail}")

            client = await asyncio.wait_for(
                sdk.modbus_open(str(resolved_port), sdk.Baudrate.Baud460800),
                timeout=3.0,
            )
            clients.append(client)
            info = await await_result(client.get_device_info(device["slave_id"]))
            expected_hand = getattr(sdk.HandType, device["side"].title())
            if str(info.serial_number) != device["serial"] or info.hand_type != expected_hand:
                raise RuntimeError(f"identity mismatch on {resolved_port}: {info}")
            unit_mode = await await_result(client.get_finger_unit_mode(device["slave_id"]))
            if unit_mode != sdk.FingerUnitMode.Normalized:
                raise RuntimeError(f"expected Normalized mode on {resolved_port}")
            status = await await_result(client.get_motor_status(device["slave_id"]))
            baseline = [int(value) for value in status.positions]
            if len(baseline) != CHANNEL_COUNT or any(not 0 <= value <= 1000 for value in baseline):
                raise RuntimeError(f"invalid baseline on {resolved_port}: {baseline}")
            if any(value + delta > 1000 for value, delta in zip(baseline, deltas)):
                raise RuntimeError(f"baseline cannot accept deltas on {resolved_port}")
            baselines.append(baseline)

        if xrt_service_is_listening():
            print("[dual-trigger] reusing RoboticsService on port 60061", flush=True)
        else:
            print("[dual-trigger] starting RoboticsService...", flush=True)
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
        print(
            f"[dual-trigger] verified both hands; baselines={baselines} deltas={deltas}",
            flush=True,
        )
        print(
            "[dual-trigger] waiting for live XR timestamps and both Triggers released...",
            flush=True,
        )
        ready_deadline = time.monotonic() + args.xrt_ready_timeout_s
        ready_stamp = None
        timestamp_updates = 0
        while time.monotonic() < ready_deadline:
            source_stamp = int(xrt.get_time_stamp_ns())
            if source_stamp > 0 and source_stamp != ready_stamp:
                ready_stamp = source_stamp
                timestamp_updates += 1
            left_trigger = min(1.0, max(0.0, float(xrt.get_left_trigger())))
            right_trigger = min(1.0, max(0.0, float(xrt.get_right_trigger())))
            if timestamp_updates >= 2 and max(left_trigger, right_trigger) <= 0.05:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("XR did not become live with both Triggers released")
        print("[dual-trigger] XR ready; squeeze Triggers independently", flush=True)

        period = 1.0 / args.hz
        deadline = time.monotonic() + args.duration_s
        next_tick = time.monotonic()
        next_report = float("-inf")
        last_source_stamp = ready_stamp
        last_source_received = time.monotonic()
        last_targets: list[tuple[int, ...] | None] = [None, None]

        while time.monotonic() < deadline:
            now = time.monotonic()
            source_stamp = int(xrt.get_time_stamp_ns())
            if source_stamp != last_source_stamp:
                last_source_stamp = source_stamp
                last_source_received = now
            elif now - last_source_received > args.deadman_s:
                await stop_hands(clients, sdk)
                raise RuntimeError("XR input timed out; both hands stopped")

            triggers = (
                min(1.0, max(0.0, float(xrt.get_left_trigger()))),
                min(1.0, max(0.0, float(xrt.get_right_trigger()))),
            )
            targets = [
                [
                    base + round(trigger * delta)
                    for base, delta in zip(baseline, deltas, strict=True)
                ]
                for baseline, trigger in zip(baselines, triggers, strict=True)
            ]
            writes = []
            for index, (client, device, target) in enumerate(
                zip(clients, DEVICES, targets, strict=True)
            ):
                target_key = tuple(target)
                if target_key != last_targets[index]:
                    writes.append(
                        await_result(
                            client.set_finger_positions_and_speeds(
                                device["slave_id"], target, [args.speed] * CHANNEL_COUNT
                            )
                        )
                    )
                    last_targets[index] = target_key
            if writes:
                command_attempted = True
                await asyncio.gather(*writes)

            if now >= next_report:
                statuses = await asyncio.gather(
                    *[
                        await_result(client.get_motor_status(device["slave_id"]))
                        for client, device in zip(clients, DEVICES, strict=True)
                    ]
                )
                print(
                    f"[dual-trigger] triggers=({triggers[0]:.3f},{triggers[1]:.3f}) "
                    f"left={list(statuses[0].positions)} "
                    f"right={list(statuses[1].positions)}",
                    flush=True,
                )
                next_report = now + 0.25

            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        completed_normally = True
    finally:
        if clients:
            if command_attempted and len(clients) == 2 and len(baselines) == 2:
                try:
                    await asyncio.gather(
                        *[
                            await_result(
                                client.set_finger_positions_and_speeds(
                                    device["slave_id"],
                                    baseline,
                                    [args.speed] * CHANNEL_COUNT,
                                )
                            )
                            for client, device, baseline in zip(
                                clients, DEVICES, baselines, strict=True
                            )
                        ]
                    )
                    await asyncio.sleep(1.0)
                    statuses = await asyncio.gather(
                        *[
                            await_result(client.get_motor_status(device["slave_id"]))
                            for client, device in zip(clients, DEVICES, strict=True)
                        ]
                    )
                    print(
                        f"[dual-trigger] restored left={list(statuses[0].positions)} "
                        f"right={list(statuses[1].positions)}",
                        flush=True,
                    )
                finally:
                    await stop_hands(clients, sdk)
                    print("[dual-trigger] both hands zero-speed stop sent", flush=True)
            for client in clients:
                await await_result(sdk.modbus_close(client), timeout_s=1.0)
            print("[dual-trigger] serial ports closed", flush=True)
        if xrt_initialized:
            xrt.close()
            print("[dual-trigger] XRoboToolkit closed", flush=True)
        for lock_handle in locks:
            lock_handle.close()
        if completed_normally:
            print("[dual-trigger] test complete", flush=True)


def main() -> None:
    args = parse_args()
    if not args.confirm_move:
        raise SystemExit("refusing to move hardware without --confirm-move")
    if not 1.0 <= args.duration_s <= 60.0:
        raise SystemExit("--duration-s must be within [1, 60]")
    if not 5.0 <= args.hz <= 50.0:
        raise SystemExit("--hz must be within [5, 50]")
    if not 1 <= args.finger_delta <= 250 or not 1 <= args.thumb_delta <= 200:
        raise SystemExit("finger/thumb deltas exceed commissioning limits")
    if not 1 <= args.speed <= 300:
        raise SystemExit("--speed must be within [1, 300]")
    if not 0.1 <= args.deadman_s <= 1.0:
        raise SystemExit("--deadman-s must be within [0.1, 1.0]")
    if not 1.0 <= args.xrt_ready_timeout_s <= 60.0:
        raise SystemExit("--xrt-ready-timeout-s must be within [1, 60]")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[dual-trigger] interrupted by user", flush=True)
    except RuntimeError as exc:
        raise SystemExit(f"[dual-trigger] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
