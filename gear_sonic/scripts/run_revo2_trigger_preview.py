"""Preview PICO Controller Trigger values without controlling hardware.

This utility initializes only XRoboToolkit input.  It does not open Revo2
serial ports, start SONIC, connect to the G1, or publish ZMQ commands.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time

from gear_sonic.utils.teleop.revo2_trigger import bilateral_trigger_deltas


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def bounded_delta(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("must be finite and within (0, 0.5]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only PICO Trigger to Revo2 target preview")
    parser.add_argument("--duration-s", type=positive_float, default=30.0)
    parser.add_argument("--hz", type=positive_float, default=20.0)
    parser.add_argument(
        "--thumb-delta",
        type=bounded_delta,
        default=0.15,
    )
    parser.add_argument(
        "--finger-delta",
        type=bounded_delta,
        default=0.20,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise SystemExit(
            "xrobotoolkit_sdk is unavailable; run this inside the PC2 teleop environment"
        ) from exc

    print("[preview] read-only mode", flush=True)
    print(
        "[preview] no Revo2 serial, SONIC, G1, or ZMQ output is used",
        flush=True,
    )
    print(
        f"[preview] thumb_delta={args.thumb_delta:.3f} " f"finger_delta={args.finger_delta:.3f}",
        flush=True,
    )

    service = subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    del service
    initialized = False
    try:
        xrt.init()
        initialized = True
        deadline = time.monotonic() + args.duration_s
        period = 1.0 / args.hz
        next_sample = time.monotonic()
        next_report = float("-inf")
        previous: tuple[float, float] | None = None

        while time.monotonic() < deadline:
            left = float(xrt.get_left_trigger())
            right = float(xrt.get_right_trigger())
            targets = bilateral_trigger_deltas(
                left,
                right,
                thumb_delta=args.thumb_delta,
                finger_delta=args.finger_delta,
            )
            now = time.monotonic()
            current = (left, right)
            changed = (
                previous is None or max(abs(left - previous[0]), abs(right - previous[1])) >= 0.01
            )
            if changed or now >= next_report:
                print(
                    "[preview] "
                    f"left_trigger={left:.3f} right_trigger={right:.3f} "
                    f"left_delta={targets[0].round(3).tolist()} "
                    f"right_delta={targets[1].round(3).tolist()}",
                    flush=True,
                )
                next_report = now + 0.25
                previous = current

            next_sample += period
            time.sleep(max(0.0, next_sample - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[preview] stopped by user", flush=True)
    finally:
        if initialized:
            xrt.close()
        print("[preview] XRoboToolkit closed; no control output was sent", flush=True)


if __name__ == "__main__":
    main()
