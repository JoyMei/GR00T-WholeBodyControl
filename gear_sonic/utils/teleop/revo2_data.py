"""Stable Revo2 state contract shared by the hardware owner and exporter.

The BrainCo SDK stays inside the Revo2 owner process.  This module only deals
with plain Python objects and NumPy arrays so schema and buffer decoding can be
tested on machines without the native SDK or hardware.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

REVO2_SIDES = ("left", "right")
REVO2_MOTOR_CHANNELS = (
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
)
REVO2_TOUCH_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
REVO2_PRESSURE_PARTS = (*REVO2_TOUCH_FINGERS, "palm")
REVO2_PRESSURE_SENSOR_COUNTS = (9, 9, 9, 9, 9, 46)
REVO2_CAPACITIVE_FIELDS = (
    "status",
    "normal_force1",
    "normal_force2",
    "normal_force3",
    "tangential_force1",
    "tangential_force2",
    "tangential_force3",
    "tangential_direction1",
    "tangential_direction2",
    "tangential_direction3",
    "self_proximity1",
    "self_proximity2",
    "mutual_proximity",
)

REVO2_HAND_COUNT = len(REVO2_SIDES)
REVO2_MOTOR_COUNT = len(REVO2_MOTOR_CHANNELS)
REVO2_CAPACITIVE_VALUES_PER_HAND = len(REVO2_TOUCH_FINGERS) * len(
    REVO2_CAPACITIVE_FIELDS
)
REVO2_PRESSURE_VALUES_PER_HAND = len(REVO2_PRESSURE_PARTS)
REVO2_PRESSURE_DETAILED_VALUES_PER_HAND = sum(REVO2_PRESSURE_SENSOR_COUNTS)

REVO2_TOUCH_NONE = 0
REVO2_TOUCH_CAPACITIVE = 1
REVO2_TOUCH_PRESSURE_SUMMARY = 2
REVO2_LEROBOT_SCHEMA_REVISION = "revo2-lerobot-v1"


def empty_revo2_state() -> dict[str, Any]:
    """Return a fixed-shape, validity-masked bilateral state snapshot."""

    return {
        "sequence": 0,
        "publish_ns": 0,
        "source_received_ns": 0,
        "source_trigger": np.zeros(REVO2_HAND_COUNT, dtype=np.float32),
        "command_position_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_MOTOR_COUNT), dtype=np.float32
        ),
        "motor_position_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_MOTOR_COUNT), dtype=np.float32
        ),
        "motor_speed_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_MOTOR_COUNT), dtype=np.float32
        ),
        "motor_current_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_MOTOR_COUNT), dtype=np.float32
        ),
        "motor_state_code": np.full(
            (REVO2_HAND_COUNT, REVO2_MOTOR_COUNT), -1, dtype=np.int32
        ),
        "motor_ns": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "motor_valid": np.zeros(REVO2_HAND_COUNT, dtype=bool),
        "motor_coalesced": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "touch_type": np.zeros(REVO2_HAND_COUNT, dtype=np.int32),
        "capacitive_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_CAPACITIVE_VALUES_PER_HAND), dtype=np.float32
        ),
        "pressure_summary_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_PRESSURE_VALUES_PER_HAND), dtype=np.float32
        ),
        "pressure_detail_raw": np.zeros(
            (REVO2_HAND_COUNT, REVO2_PRESSURE_DETAILED_VALUES_PER_HAND),
            dtype=np.float32,
        ),
        "pressure_detail_ns": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "pressure_detail_valid": np.zeros(REVO2_HAND_COUNT, dtype=bool),
        "pressure_detail_coalesced": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "touch_ns": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "touch_valid": np.zeros(REVO2_HAND_COUNT, dtype=bool),
        "touch_coalesced": np.zeros(REVO2_HAND_COUNT, dtype=np.int64),
        "cmd_coalesced": 0,
    }


def motor_state_code(value: Any) -> int:
    """Convert an SDK motor enum to a stable numeric code when possible."""

    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, (int, np.integer)):
        return int(raw_value)
    if isinstance(value, (int, np.integer)):
        return int(value)

    name = str(getattr(value, "name", value)).split(".")[-1].lower()
    aliases = {
        "idle": 0,
        "running": 1,
        "moving": 1,
        "stall": 2,
        "overcurrent": 3,
        "over_current": 3,
        "overtemperature": 4,
        "over_temperature": 4,
        "fault": 5,
        "calibrating": 6,
    }
    return aliases.get(name, -1)


def decode_motor_status(status: Any) -> dict[str, np.ndarray]:
    """Decode one SDK ``MotorStatusData`` without importing the SDK."""

    return {
        "position_raw": _finite_vector(status.positions, REVO2_MOTOR_COUNT, "positions"),
        "speed_raw": _finite_vector(status.speeds, REVO2_MOTOR_COUNT, "speeds"),
        "current_raw": _finite_vector(status.currents, REVO2_MOTOR_COUNT, "currents"),
        "state_code": np.asarray(
            [motor_state_code(value) for value in status.states], dtype=np.int32
        ).reshape(REVO2_MOTOR_COUNT),
    }


def decode_capacitive_batches(
    batches: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, int]:
    """Return latest five-finger values and samples coalesced into that snapshot."""

    if len(batches) != len(REVO2_TOUCH_FINGERS):
        raise ValueError("TouchStatusBuffer must return five finger batches")

    values = np.zeros(REVO2_CAPACITIVE_VALUES_PER_HAND, dtype=np.float32)
    coalesced = 0
    offset = 0
    for batch in batches:
        coalesced += max(0, len(batch) - 1)
        if batch:
            latest = batch[-1]
            for field_index, field in enumerate(REVO2_CAPACITIVE_FIELDS):
                value = float(getattr(latest, field))
                if not np.isfinite(value):
                    raise ValueError(f"non-finite Revo2 capacitive field: {field}")
                values[offset + field_index] = value
        offset += len(REVO2_CAPACITIVE_FIELDS)
    return values, coalesced


def decode_pressure_summary_batches(
    batches: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, int]:
    """Return latest pressure summaries and samples coalesced into the snapshot."""

    if len(batches) != len(REVO2_PRESSURE_PARTS):
        raise ValueError("PressureSummaryBuffer must return six part batches")

    values = np.zeros(REVO2_PRESSURE_VALUES_PER_HAND, dtype=np.float32)
    coalesced = 0
    for index, batch in enumerate(batches):
        coalesced += max(0, len(batch) - 1)
        if batch:
            value = float(batch[-1])
            if not np.isfinite(value):
                raise ValueError("non-finite Revo2 pressure summary value")
            values[index] = value
    return values, coalesced


def decode_pressure_detailed_batches(
    batches: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, int]:
    """Flatten latest pressure taxels and count samples coalesced into the snapshot."""

    if len(batches) != len(REVO2_PRESSURE_PARTS):
        raise ValueError("PressureDetailedBuffer must return six part batches")

    values = np.zeros(REVO2_PRESSURE_DETAILED_VALUES_PER_HAND, dtype=np.float32)
    coalesced = 0
    offset = 0
    for part, expected_count, batch in zip(
        REVO2_PRESSURE_PARTS, REVO2_PRESSURE_SENSOR_COUNTS, batches, strict=True
    ):
        coalesced += max(0, len(batch) - 1)
        if batch:
            latest = batch[-1]
            sensors = (
                latest.get_sensors()
                if callable(getattr(latest, "get_sensors", None))
                else latest.sensors
            )
            sensor_values = np.asarray(list(sensors), dtype=np.float32)
            if sensor_values.shape != (expected_count,):
                raise ValueError(
                    f"Revo2 pressure {part} must have {expected_count} taxels, "
                    f"found {sensor_values.shape}"
                )
            if not np.isfinite(sensor_values).all():
                raise ValueError(f"non-finite Revo2 pressure detailed values for {part}")
            values[offset : offset + expected_count] = sensor_values
        offset += expected_count
    return values, coalesced


def revo2_state_to_wire(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert a state snapshot to arrays accepted by ``pack_pose_message``."""

    return {
        key: np.asarray(value).copy()
        for key, value in state.items()
        if key
        not in {"sequence", "publish_ns", "source_received_ns"}
    } | {
        "sequence": np.array([state["sequence"]], dtype=np.int64),
        "publish_ns": np.array([state["publish_ns"]], dtype=np.int64),
        "source_received_ns": np.array([state["source_received_ns"]], dtype=np.int64),
    }


def revo2_feature_names() -> dict[str, list[str]]:
    """Return deterministic flattened feature labels for LeRobot metadata."""

    motor = [f"{side}_{channel}" for side in REVO2_SIDES for channel in REVO2_MOTOR_CHANNELS]
    capacitive = [
        f"{side}_{finger}_{field}"
        for side in REVO2_SIDES
        for finger in REVO2_TOUCH_FINGERS
        for field in REVO2_CAPACITIVE_FIELDS
    ]
    pressure = [
        f"{side}_{part}" for side in REVO2_SIDES for part in REVO2_PRESSURE_PARTS
    ]
    pressure_detailed = [
        f"{side}_{part}_taxel_{taxel_index:02d}"
        for side in REVO2_SIDES
        for part, count in zip(
            REVO2_PRESSURE_PARTS, REVO2_PRESSURE_SENSOR_COUNTS, strict=True
        )
        for taxel_index in range(count)
    ]
    return {
        "motor": motor,
        "capacitive": capacitive,
        "pressure": pressure,
        "pressure_detailed": pressure_detailed,
        "sides": list(REVO2_SIDES),
    }


def revo2_capture_metadata(devices: Sequence[Any]) -> dict[str, Any]:
    """Build stable provenance metadata for the opt-in LeRobot capture schema."""

    return {
        "schema_revision": REVO2_LEROBOT_SCHEMA_REVISION,
        "devices": [
            {
                "side": str(device.side),
                "port": str(device.port),
                "serial": str(device.serial),
                "slave_id": int(device.slave_id),
                "open_positions": (
                    list(device.open_positions)
                    if device.open_positions is not None
                    else None
                ),
            }
            for device in devices
        ],
        "motor_channel_order": list(REVO2_MOTOR_CHANNELS),
        "touch_finger_order": list(REVO2_TOUCH_FINGERS),
        "capacitive_field_order": list(REVO2_CAPACITIVE_FIELDS),
        "pressure_part_order": list(REVO2_PRESSURE_PARTS),
        "pressure_sensor_counts": list(REVO2_PRESSURE_SENSOR_COUNTS),
        "touch_type_codes": {
            "none": REVO2_TOUCH_NONE,
            "capacitive": REVO2_TOUCH_CAPACITIVE,
            "pressure_summary": REVO2_TOUCH_PRESSURE_SUMMARY,
        },
        "units": {
            "motor_position": "normalized_0_to_1_from_sdk_0_to_1000",
            "motor_speed_raw": "sdk_raw",
            "motor_current_raw": "sdk_raw",
            "capacitive_raw": "sdk_raw",
            "pressure_summary_raw": "sdk_raw",
            "pressure_detailed_raw": "sdk_raw",
            "timestamps": "host_monotonic_ns",
        },
        "sampling": {
            "frame_alignment": "latest_sample_hold",
            "coalesced_counters": "samples intentionally collapsed before each snapshot",
            "inactive_touch_fields": "zero with touch_type/validity mask",
        },
        "oxt_immutable_raw_bundle": False,
    }


def revo2_state_freshness(
    state: dict[str, Any],
    *,
    now_monotonic_ns: int,
    max_age_ns: int,
    require_tactile: bool,
) -> tuple[bool, str]:
    """Check bilateral validity/freshness before one LeRobot frame is accepted."""

    publish_ns = _scalar_int(state.get("publish_ns", 0))
    if publish_ns <= 0 or not 0 <= now_monotonic_ns - publish_ns <= max_age_ns:
        return False, "publisher state is missing or stale"

    source_sequence = _scalar_int(state.get("sequence", 0))
    source_received_ns = _scalar_int(state.get("source_received_ns", 0))
    if source_sequence <= 0 or source_received_ns <= 0:
        return False, "trigger source is missing"
    if not 0 <= now_monotonic_ns - source_received_ns <= max_age_ns:
        return False, "trigger source is stale"

    motor_valid = np.asarray(state.get("motor_valid"), dtype=bool).reshape(-1)
    motor_ns = np.asarray(state.get("motor_ns"), dtype=np.int64).reshape(-1)
    if motor_valid.shape != (REVO2_HAND_COUNT,) or not motor_valid.all():
        return False, "bilateral motor state is not valid"
    if np.any(motor_ns <= 0) or np.any(now_monotonic_ns - motor_ns > max_age_ns):
        return False, "bilateral motor state is stale"

    if require_tactile:
        touch_type = np.asarray(state.get("touch_type"), dtype=np.int32).reshape(-1)
        touch_valid = np.asarray(state.get("touch_valid"), dtype=bool).reshape(-1)
        touch_ns = np.asarray(state.get("touch_ns"), dtype=np.int64).reshape(-1)
        if touch_type.shape != (REVO2_HAND_COUNT,) or np.any(touch_type == REVO2_TOUCH_NONE):
            return False, "bilateral tactile hardware was not detected"
        if touch_valid.shape != (REVO2_HAND_COUNT,) or not touch_valid.all():
            return False, "bilateral tactile state is not valid"
        if np.any(touch_ns <= 0) or np.any(now_monotonic_ns - touch_ns > max_age_ns):
            return False, "bilateral tactile state is stale"

    return True, "ready"


def revo2_state_to_features(
    state: dict[str, Any], *, now_monotonic_ns: int
) -> dict[str, np.ndarray]:
    """Convert one fresh wire state into the optional LeRobot frame fields."""

    motor_ns = np.asarray(state["motor_ns"], dtype=np.int64).reshape(
        REVO2_HAND_COUNT
    )
    touch_ns = np.asarray(state["touch_ns"], dtype=np.int64).reshape(
        REVO2_HAND_COUNT
    )
    pressure_detailed_ns = np.asarray(
        state["pressure_detail_ns"], dtype=np.int64
    ).reshape(REVO2_HAND_COUNT)
    motor_valid = np.asarray(state["motor_valid"], dtype=bool).reshape(REVO2_HAND_COUNT)
    touch_valid = np.asarray(state["touch_valid"], dtype=bool).reshape(REVO2_HAND_COUNT)
    pressure_detailed_valid = np.asarray(
        state["pressure_detail_valid"], dtype=bool
    ).reshape(REVO2_HAND_COUNT)
    motor_age_ms = np.where(
        motor_valid,
        np.maximum(0, now_monotonic_ns - motor_ns) / 1e6,
        -1.0,
    ).astype(np.float32)
    touch_age_ms = np.where(
        touch_valid,
        np.maximum(0, now_monotonic_ns - touch_ns) / 1e6,
        -1.0,
    ).astype(np.float32)
    pressure_detailed_age_ms = np.where(
        pressure_detailed_valid,
        np.maximum(0, now_monotonic_ns - pressure_detailed_ns) / 1e6,
        -1.0,
    ).astype(np.float32)

    return {
        "observation.revo2.motor_position": (
            np.asarray(state["motor_position_raw"], dtype=np.float32).reshape(-1) / 1000.0
        ),
        "observation.revo2.motor_speed_raw": np.asarray(
            state["motor_speed_raw"], dtype=np.float32
        ).reshape(-1),
        "observation.revo2.motor_current_raw": np.asarray(
            state["motor_current_raw"], dtype=np.float32
        ).reshape(-1),
        "observation.revo2.motor_state_code": np.asarray(
            state["motor_state_code"], dtype=np.int32
        ).reshape(-1),
        "action.revo2.command_position": (
            np.asarray(state["command_position_raw"], dtype=np.float32).reshape(-1) / 1000.0
        ),
        "teleop.revo2.source_trigger": np.asarray(
            state["source_trigger"], dtype=np.float32
        ).reshape(REVO2_HAND_COUNT),
        "observation.revo2.capacitive_raw": np.asarray(
            state["capacitive_raw"], dtype=np.float32
        ).reshape(-1),
        "observation.revo2.pressure_summary_raw": np.asarray(
            state["pressure_summary_raw"], dtype=np.float32
        ).reshape(-1),
        "observation.revo2.pressure_detailed_raw": np.asarray(
            state["pressure_detail_raw"], dtype=np.float32
        ).reshape(-1),
        "observation.revo2.touch_type": np.asarray(
            state["touch_type"], dtype=np.int32
        ).reshape(REVO2_HAND_COUNT),
        "observation.revo2.motor_valid": motor_valid.astype(np.int32),
        "observation.revo2.touch_valid": touch_valid.astype(np.int32),
        "observation.revo2.pressure_detailed_valid": pressure_detailed_valid.astype(
            np.int32
        ),
        "observation.revo2.motor_monotonic_ns": motor_ns,
        "observation.revo2.touch_monotonic_ns": touch_ns,
        "observation.revo2.pressure_detailed_monotonic_ns": pressure_detailed_ns,
        "observation.revo2.motor_age_ms": motor_age_ms,
        "observation.revo2.touch_age_ms": touch_age_ms,
        "observation.revo2.pressure_detailed_age_ms": pressure_detailed_age_ms,
        "observation.revo2.motor_samples_coalesced": np.asarray(
            state["motor_coalesced"], dtype=np.int64
        ).reshape(REVO2_HAND_COUNT),
        "observation.revo2.touch_samples_coalesced": np.asarray(
            state["touch_coalesced"], dtype=np.int64
        ).reshape(REVO2_HAND_COUNT),
        "observation.revo2.pressure_detailed_samples_coalesced": np.asarray(
            state["pressure_detail_coalesced"], dtype=np.int64
        ).reshape(REVO2_HAND_COUNT),
        "observation.revo2.command_deadlines_coalesced": np.array(
            [_scalar_int(state["cmd_coalesced"])], dtype=np.int64
        ),
        "observation.revo2.publish_monotonic_ns": np.array(
            [_scalar_int(state["publish_ns"])], dtype=np.int64
        ),
        "teleop.revo2.source_received_monotonic_ns": np.array(
            [_scalar_int(state["source_received_ns"])], dtype=np.int64
        ),
        "teleop.revo2.source_sequence": np.array(
            [_scalar_int(state["sequence"])], dtype=np.int64
        ),
    }


def _finite_vector(values: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"Revo2 {label} must have shape ({size},), found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Revo2 {label} contains non-finite values")
    return array


def _scalar_int(value: Any) -> int:
    array = np.asarray(value).reshape(-1)
    return int(array[0]) if array.size else 0
