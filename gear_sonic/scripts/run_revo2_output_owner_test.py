#!/usr/bin/env python3
"""Exercise the production Revo2TriggerOutput without starting SONIC or G1."""

from __future__ import annotations

import argparse
import socket
import subprocess
import time

from gear_sonic.utils.teleop.revo2_trigger import Revo2Device, Revo2TriggerOutput

DEVICES = (
    Revo2Device(
        side="left",
        port="/dev/ttyUSB0",
        serial="BCXTL2206J260000B",
        slave_id=126,
    ),
    Revo2Device(
        side="right",
        port="/dev/ttyUSB1",
        serial="BCXTR2212J2600027",
        slave_id=127,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--thumb-delta", type=float, default=0.15)
    parser.add_argument("--finger-delta", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--deadman-s", type=float, default=0.3)
    parser.add_argument("--xrt-ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--confirm-move", action="store_true")
    return parser.parse_args()


def service_is_listening() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", 60061)) == 0


def main() -> None:
    args = parse_args()
    if not args.confirm_move:
        raise SystemExit("refusing to move hardware without --confirm-move")
    if not 1.0 <= args.duration_s <= 60.0 or not 5.0 <= args.hz <= 50.0:
        raise SystemExit("duration/hz outside commissioning limits")
    if not 1.0 <= args.xrt_ready_timeout_s <= 60.0:
        raise SystemExit("XR readiness timeout outside commissioning limits")

    import xrobotoolkit_sdk as xrt

    output = None
    xrt_initialized = False
    output_enabled = False
    try:
        if service_is_listening():
            print("[owner-test] reusing RoboticsService on port 60061", flush=True)
        else:
            print("[owner-test] starting RoboticsService...", flush=True)
            subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
            service_deadline = time.monotonic() + 10.0
            while time.monotonic() < service_deadline and not service_is_listening():
                time.sleep(0.1)
            if not service_is_listening():
                raise RuntimeError("RoboticsService did not listen on port 60061")

        xrt.init()
        xrt_initialized = True
        print("[owner-test] waiting for live XR and both Triggers released...", flush=True)
        ready_deadline = time.monotonic() + args.xrt_ready_timeout_s
        ready_stamp = None
        updates = 0
        while time.monotonic() < ready_deadline:
            stamp = int(xrt.get_time_stamp_ns())
            if stamp > 0 and stamp != ready_stamp:
                ready_stamp = stamp
                updates += 1
            left = float(xrt.get_left_trigger())
            right = float(xrt.get_right_trigger())
            if updates >= 2 and max(left, right) <= 0.05:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("XR did not become live with both Triggers released")

        output = Revo2TriggerOutput(
            DEVICES,
            deadman_s=args.deadman_s,
            slew_per_s=0.4,
            command_period_s=0.04,
            thumb_delta=args.thumb_delta,
            finger_delta=args.finger_delta,
            command_speed=args.speed,
        ).start()
        output.submit_triggers(
            0.0,
            0.0,
            source_received_monotonic_ns=time.monotonic_ns(),
        )
        output.set_enabled(True)
        output_enabled = True
        print(
            "[owner-test] production owner ready; test left, right, then both Triggers",
            flush=True,
        )

        deadline = time.monotonic() + args.duration_s
        period = 1.0 / args.hz
        next_tick = time.monotonic()
        next_report = float("-inf")
        last_stamp = ready_stamp
        last_received = time.monotonic()
        left = 0.0
        right = 0.0
        while time.monotonic() < deadline:
            output.check_health()
            now = time.monotonic()
            stamp = int(xrt.get_time_stamp_ns())
            if stamp != last_stamp:
                last_stamp = stamp
                last_received = now
                left = min(1.0, max(0.0, float(xrt.get_left_trigger())))
                right = min(1.0, max(0.0, float(xrt.get_right_trigger())))
                output.submit_triggers(
                    left,
                    right,
                    source_received_monotonic_ns=time.monotonic_ns(),
                )
            elif now - last_received > args.deadman_s:
                raise RuntimeError("XR timestamp stopped; owner deadman will stop both hands")

            if now >= next_report:
                print(f"[owner-test] triggers=({left:.3f},{right:.3f})", flush=True)
                next_report = now + 0.25
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[owner-test] interrupted by user", flush=True)
    finally:
        if output is not None:
            if output_enabled:
                try:
                    output.submit_triggers(
                        0.0,
                        0.0,
                        source_received_monotonic_ns=time.monotonic_ns(),
                    )
                    time.sleep(1.0)
                    output.set_enabled(False)
                except RuntimeError as exc:
                    print(f"[owner-test] owner cleanup warning: {exc}", flush=True)
            output.close()
            print("[owner-test] production owner closed", flush=True)
        if xrt_initialized:
            xrt.close()
            print("[owner-test] XRoboToolkit closed", flush=True)


if __name__ == "__main__":
    main()
