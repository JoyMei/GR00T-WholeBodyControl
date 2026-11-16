#!/usr/bin/env python3
"""Read Revo2 identities and status without sending actuator commands.

The probe performs Modbus auto-detection and read-only queries.  It never calls
finger position, finger speed, hardware configuration, or enable methods.
Run it only while no other process owns the Revo2 serial ports.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ports",
        nargs="+",
        default=("/dev/ttyUSB0", "/dev/ttyUSB1"),
        help="Revo2 serial ports to inspect",
    )
    parser.add_argument("--detect-timeout-s", type=float, default=10.0)
    parser.add_argument("--read-timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--direct-defaults",
        action="store_true",
        help="Skip auto-detect and read slave IDs 126/127 at 460800 baud",
    )
    return parser.parse_args()


async def await_result(value: Any, timeout_s: float) -> Any:
    if inspect.isawaitable(value):
        return await asyncio.wait_for(value, timeout=timeout_s)
    return value


async def close_client(sdk: Any, client: Any) -> None:
    await await_result(sdk.modbus_close(client), 1.0)


def describe(value: Any) -> str:
    fields = {}
    for name in (
        "serial_number",
        "hand_type",
        "firmware_version",
        "positions",
        "speeds",
        "currents",
    ):
        if hasattr(value, name):
            fields[name] = getattr(value, name)
    return repr(fields) if fields else repr(value)


async def probe_port(
    sdk: Any,
    requested_port: str,
    detect_timeout_s: float,
    read_timeout_s: float,
    direct_defaults: bool,
) -> None:
    import fcntl

    resolved_port = Path(requested_port).resolve(strict=True)
    lock_path = Path("/tmp") / f"g1-revo2-probe-{resolved_port.name}.lock"
    lock_handle = lock_path.open("a", encoding="utf-8")
    client = None
    try:
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

        print(f"[probe] port={resolved_port}", flush=True)
        if direct_defaults:
            client = await asyncio.wait_for(
                sdk.modbus_open(str(resolved_port), sdk.Baudrate.Baud460800),
                timeout=read_timeout_s,
            )
            slave_ids = (126, 127)
            print(
                "[probe] direct mode baud=460800 candidate_slave_ids=[126, 127]",
                flush=True,
            )
        else:
            protocol, detected_port, baud, slave_id = await asyncio.wait_for(
                sdk.auto_detect_modbus_revo2(str(resolved_port), True),
                timeout=detect_timeout_s,
            )
            print(
                "[probe] detected "
                f"protocol={protocol!r} port={detected_port!r} "
                f"baud={baud!r} slave_id={slave_id!r}",
                flush=True,
            )
            client = await asyncio.wait_for(
                sdk.modbus_open(detected_port, baud), timeout=read_timeout_s
            )
            slave_ids = (slave_id,)

        for slave_id in slave_ids:
            try:
                device_info = await await_result(client.get_device_info(slave_id), read_timeout_s)
            except Exception as exc:
                print(
                    f"[probe] slave_id={slave_id} no_response " f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                continue
            if device_info is None:
                print(f"[probe] slave_id={slave_id} no_response", flush=True)
                continue

            print(
                f"[probe] slave_id={slave_id} device_info={describe(device_info)}",
                flush=True,
            )
            reads = (
                ("finger_unit_mode", client.get_finger_unit_mode),
                ("motor_status", client.get_motor_status),
            )
            for label, method in reads:
                try:
                    value = await await_result(method(slave_id), read_timeout_s)
                except Exception as exc:
                    print(
                        f"[probe] slave_id={slave_id} {label}=UNAVAILABLE "
                        f"({type(exc).__name__}: {exc})",
                        flush=True,
                    )
                else:
                    print(
                        f"[probe] slave_id={slave_id} {label}={describe(value)}",
                        flush=True,
                    )
    finally:
        if client is not None:
            await close_client(sdk, client)
        lock_handle.close()


async def run(args: argparse.Namespace) -> None:
    try:
        from gear_sonic.utils.teleop.brainco_sdk_loader import load_brainco_sdk

        sdk = load_brainco_sdk()
    except ImportError as exc:
        raise SystemExit("bc_stark_sdk is unavailable; set BC_STARK_SDK_PACKAGE_DIR") from exc

    print(
        "[probe] READ-ONLY: no position, speed, configuration, or enable command is sent",
        flush=True,
    )
    for port in args.ports:
        try:
            await probe_port(
                sdk,
                port,
                args.detect_timeout_s,
                args.read_timeout_s,
                args.direct_defaults,
            )
        except Exception as exc:
            print(f"[probe] port={port} ERROR: {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    args = parse_args()
    if args.detect_timeout_s <= 0.0 or args.read_timeout_s <= 0.0:
        raise SystemExit("timeouts must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
