from types import SimpleNamespace

import numpy as np

from gear_sonic.data.features_sonic_vla import get_revo2_features
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
from gear_sonic.utils.teleop.revo2_data import (
    REVO2_CAPACITIVE_FIELDS,
    REVO2_TOUCH_CAPACITIVE,
    decode_capacitive_batches,
    decode_motor_status,
    decode_pressure_detailed_batches,
    decode_pressure_summary_batches,
    empty_revo2_state,
    revo2_state_freshness,
    revo2_state_to_features,
    revo2_state_to_wire,
)


def _touch_item(seed: int):
    return SimpleNamespace(
        **{field: seed + index for index, field in enumerate(REVO2_CAPACITIVE_FIELDS)}
    )


def test_decode_motor_status_preserves_raw_values():
    status = SimpleNamespace(
        positions=range(6),
        speeds=range(10, 16),
        currents=range(-6, 0),
        states=[SimpleNamespace(name="Idle", value=0)] * 6,
    )

    decoded = decode_motor_status(status)

    np.testing.assert_array_equal(decoded["position_raw"], np.arange(6))
    np.testing.assert_array_equal(decoded["speed_raw"], np.arange(10, 16))
    np.testing.assert_array_equal(decoded["current_raw"], np.arange(-6, 0))
    np.testing.assert_array_equal(decoded["state_code"], np.zeros(6, dtype=np.int32))


def test_decode_capacitive_batches_uses_latest_and_counts_coalesced_samples():
    batches = [
        [_touch_item(finger * 100), _touch_item(finger * 100 + 50)]
        for finger in range(5)
    ]

    values, coalesced = decode_capacitive_batches(batches)

    assert values.shape == (5 * len(REVO2_CAPACITIVE_FIELDS),)
    assert coalesced == 5
    assert values[0] == 50
    assert values[len(REVO2_CAPACITIVE_FIELDS)] == 150


def test_decode_pressure_summary_batches_preserves_five_fingers_and_palm():
    values, coalesced = decode_pressure_summary_batches(
        [[1, 2], [3], [4], [5], [6], [7, 8, 9]]
    )

    np.testing.assert_array_equal(values, np.array([2, 3, 4, 5, 6, 9]))
    assert coalesced == 3


def test_decode_pressure_detailed_batches_preserves_fixed_taxel_layout():
    counts = (9, 9, 9, 9, 9, 46)
    batches = [
        [SimpleNamespace(sensors=list(range(count)))] for count in counts
    ]

    values, coalesced = decode_pressure_detailed_batches(batches)

    assert values.shape == (91,)
    assert coalesced == 0
    np.testing.assert_array_equal(values[:9], np.arange(9))
    np.testing.assert_array_equal(values[-46:], np.arange(46))


def test_revo2_wire_roundtrip_is_fresh_and_matches_feature_schema():
    now_ns = 5_000_000_000
    state = empty_revo2_state()
    state.update(
        sequence=7,
        publish_ns=now_ns - 1_000_000,
        source_received_ns=now_ns - 2_000_000,
    )
    state["source_trigger"][:] = [0.25, 0.75]
    state["command_position_raw"][:] = np.arange(12).reshape(2, 6) + 100
    state["motor_position_raw"][:] = np.arange(12).reshape(2, 6) + 200
    state["motor_speed_raw"][:] = 3
    state["motor_current_raw"][:] = 4
    state["motor_state_code"][:] = 1
    state["motor_ns"][:] = now_ns - 3_000_000
    state["motor_valid"][:] = True
    state["touch_type"][:] = REVO2_TOUCH_CAPACITIVE
    state["capacitive_raw"][:] = 5
    state["touch_ns"][:] = now_ns - 4_000_000
    state["touch_valid"][:] = True

    wire = revo2_state_to_wire(state)
    assert pack_pose_message(wire, topic="revo2_state").startswith(b"revo2_state")
    ready, reason = revo2_state_freshness(
        wire, now_monotonic_ns=now_ns, max_age_ns=20_000_000, require_tactile=True
    )
    assert ready, reason

    frame = revo2_state_to_features(wire, now_monotonic_ns=now_ns)
    schema = get_revo2_features()
    assert set(frame) == set(schema)
    for key, value in frame.items():
        assert value.shape == tuple(schema[key]["shape"])
    np.testing.assert_allclose(
        frame["observation.revo2.motor_position"][:6],
        np.arange(6, dtype=np.float32) / 1000.0 + 0.2,
    )


def test_revo2_freshness_rejects_missing_tactile_without_inventing_zero():
    now_ns = 8_000_000_000
    state = empty_revo2_state()
    state["publish_ns"] = now_ns
    state["sequence"] = 1
    state["source_received_ns"] = now_ns
    state["motor_ns"][:] = now_ns
    state["motor_valid"][:] = True

    ready, reason = revo2_state_freshness(
        state, now_monotonic_ns=now_ns, max_age_ns=1_000_000, require_tactile=True
    )

    assert not ready
    assert "tactile" in reason


def test_revo2_freshness_rejects_stale_trigger_source():
    now_ns = 9_000_000_000
    state = empty_revo2_state()
    state["publish_ns"] = now_ns
    state["sequence"] = 2
    state["source_received_ns"] = now_ns - 2_000_000
    state["motor_ns"][:] = now_ns
    state["motor_valid"][:] = True

    ready, reason = revo2_state_freshness(
        state, now_monotonic_ns=now_ns, max_age_ns=1_000_000, require_tactile=False
    )

    assert not ready
    assert "trigger source" in reason
