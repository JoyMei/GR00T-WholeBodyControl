import json

import numpy as np
import pytest

from gear_sonic.utils.teleop.revo2_trigger import (
    REVO2_CHANNEL_COUNT,
    Revo2TriggerOutput,
    bilateral_trigger_deltas,
    load_revo2_devices,
    trigger_to_revo2_deltas,
)


@pytest.mark.parametrize("trigger", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_trigger_maps_linearly_to_per_channel_deltas(trigger):
    positions = trigger_to_revo2_deltas(trigger)

    assert positions.shape == (REVO2_CHANNEL_COUNT,)
    np.testing.assert_allclose(positions, [0.15 * trigger] * 2 + [0.20 * trigger] * 4)


def test_trigger_mapping_clamps_controller_edge_noise():
    np.testing.assert_array_equal(trigger_to_revo2_deltas(-0.01), np.zeros(6))
    np.testing.assert_allclose(trigger_to_revo2_deltas(1.01), [0.15] * 2 + [0.20] * 4)
    with pytest.raises(ValueError, match="finite"):
        trigger_to_revo2_deltas(float("nan"))


def test_trigger_mapping_respects_channel_deltas():
    np.testing.assert_allclose(
        trigger_to_revo2_deltas(1.0, thumb_delta=0.1, finger_delta=0.25),
        [0.1] * 2 + [0.25] * 4,
    )
    with pytest.raises(ValueError, match="thumb_delta"):
        trigger_to_revo2_deltas(1.0, thumb_delta=0.0)
    with pytest.raises(ValueError, match="finger_delta"):
        trigger_to_revo2_deltas(1.0, finger_delta=0.51)


def test_bilateral_mapping_keeps_left_and_right_independent():
    targets = bilateral_trigger_deltas(0.2, 0.8)

    assert targets.shape == (2, 6)
    np.testing.assert_allclose(targets[0], [0.03] * 2 + [0.04] * 4)
    np.testing.assert_allclose(targets[1], [0.12] * 2 + [0.16] * 4)


def test_device_config_is_ordered_left_then_right(tmp_path):
    path = tmp_path / "revo2_devices.json"
    path.write_text(
        json.dumps(
            [
                {"side": "right", "port": "/dev/right", "serial": "R", "slave_id": 127},
                {"side": "left", "port": "/dev/left", "serial": "L", "slave_id": 126},
            ]
        ),
        encoding="utf-8",
    )

    devices = load_revo2_devices(path)

    assert [device.side for device in devices] == ["left", "right"]
    assert [device.slave_id for device in devices] == [126, 127]


def test_device_config_rejects_shared_identity(tmp_path):
    path = tmp_path / "revo2_devices.json"
    path.write_text(
        json.dumps(
            [
                {"side": "left", "port": "/dev/hands", "serial": "same", "slave_id": 126},
                {"side": "right", "port": "/dev/hands", "serial": "same", "slave_id": 126},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct"):
        load_revo2_devices(path)


def test_output_is_opt_in_and_validates_safety_limits(tmp_path):
    path = tmp_path / "revo2_devices.json"
    path.write_text(
        json.dumps(
            [
                {"side": "left", "port": "/dev/left", "serial": "L", "slave_id": 126},
                {"side": "right", "port": "/dev/right", "serial": "R", "slave_id": 127},
            ]
        ),
        encoding="utf-8",
    )
    devices = load_revo2_devices(path)

    with pytest.raises(ValueError, match="deadman"):
        Revo2TriggerOutput(devices, deadman_s=3.0)
    with pytest.raises(ValueError, match="slew"):
        Revo2TriggerOutput(devices, slew_per_s=1.0)
    with pytest.raises(ValueError, match="thumb_delta"):
        Revo2TriggerOutput(devices, thumb_delta=0.0)
    with pytest.raises(ValueError, match="finger_delta"):
        Revo2TriggerOutput(devices, finger_delta=0.6)
    with pytest.raises(ValueError, match="command_speed"):
        Revo2TriggerOutput(devices, command_speed=0.5)

    output = Revo2TriggerOutput(devices)
    with pytest.raises(RuntimeError, match="not active"):
        output.submit_triggers(0.2, 0.8, source_received_monotonic_ns=1)
    output.close()
    output.close()
