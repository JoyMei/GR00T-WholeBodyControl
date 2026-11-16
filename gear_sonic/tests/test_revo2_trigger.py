import json

import numpy as np
import pytest

from gear_sonic.utils.teleop.revo2_trigger import (
    REVO2_CHANNEL_COUNT,
    Revo2TriggerMode,
    Revo2TriggerOutput,
    bilateral_trigger_deltas,
    default_revo2_trigger_modes,
    encode_revo2_mode_name,
    load_revo2_devices,
    load_revo2_trigger_modes,
    revo2_triggers_released,
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
    np.testing.assert_allclose(
        trigger_to_revo2_deltas(1.0, thumb_delta=0.6, finger_delta=1.0),
        [0.6] * 2 + [1.0] * 4,
    )
    with pytest.raises(ValueError, match="finger_delta"):
        trigger_to_revo2_deltas(1.0, finger_delta=1.01)


def test_bilateral_mapping_keeps_left_and_right_independent():
    targets = bilateral_trigger_deltas(0.2, 0.8)

    assert targets.shape == (2, 6)
    np.testing.assert_allclose(targets[0], [0.03] * 2 + [0.04] * 4)
    np.testing.assert_allclose(targets[1], [0.12] * 2 + [0.16] * 4)


def test_default_modes_have_expected_profiles():
    modes = default_revo2_trigger_modes()

    assert [mode.name for mode in modes] == ["all_fingers", "pinch", "point"]
    np.testing.assert_allclose(modes[0].channels, [0.15, 0.15, 0.20, 0.20, 0.20, 0.20])
    np.testing.assert_allclose(modes[1].channels, [0.15, 0.15, 0.20, 0.00, 0.00, 0.00])
    np.testing.assert_allclose(modes[2].channels, [0.15, 0.15, 0.00, 0.20, 0.20, 0.20])


def test_trigger_mapping_accepts_mode_profile():
    mode = Revo2TriggerMode("pinch", [0.15, 0.15, 0.20, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(
        trigger_to_revo2_deltas(0.5, channels=mode.channels),
        [0.075, 0.075, 0.10, 0.0, 0.0, 0.0],
    )


def test_trigger_mapping_combines_fixed_offsets_and_channels():
    mode = Revo2TriggerMode(
        "three_finger_pinch",
        [0.61, 0.61, 0.96, 0.96, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.96, 0.96],
    )

    np.testing.assert_allclose(
        trigger_to_revo2_deltas(
            0.0,
            channels=mode.channels,
            offsets=mode.offsets,
        ),
        [0.0, 0.0, 0.0, 0.0, 0.96, 0.96],
    )
    np.testing.assert_allclose(
        trigger_to_revo2_deltas(
            1.0,
            channels=mode.channels,
            offsets=mode.offsets,
        ),
        [0.61, 0.61, 0.96, 0.96, 0.96, 0.96],
    )


def test_mode_config_loads_and_rejects_invalid_profiles(tmp_path):
    path = tmp_path / "modes.json"
    path.write_text(
        json.dumps(
            {
                "modes": [
                    {"name": "one", "channels": [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]},
                    {"name": "two", "channels": [0.1, 0.1, 0.2, 0.0, 0.0, 0.0]},
                ]
            }
        ),
        encoding="utf-8",
    )

    modes = load_revo2_trigger_modes(path)
    assert [mode.name for mode in modes] == ["one", "two"]
    np.testing.assert_array_equal(modes[0].offsets, np.zeros(6))

    path.write_text(
        json.dumps({"modes": [{"name": "bad", "channels": [0.1]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shape"):
        load_revo2_trigger_modes(path)


@pytest.mark.parametrize(
    ("channels", "offsets", "match"),
    [
        ([1.01] * 6, [0.0] * 6, "channels"),
        ([0.0] * 6, [0.0] * 5, "offsets.*shape"),
        ([0.0] * 6, [1.01] * 6, "offsets"),
    ],
)
def test_mode_rejects_out_of_range_channels_and_offsets(channels, offsets, match):
    with pytest.raises(ValueError, match=match):
        Revo2TriggerMode("bad", channels, offsets)


def test_mode_name_encoding_is_fixed_width_and_reversible():
    encoded = encode_revo2_mode_name("all_fingers")

    assert encoded.shape == (32,)
    assert encoded.dtype == np.int32
    decoded = bytes(value for value in encoded if value).decode("utf-8")
    assert decoded == "all_fingers"


def test_mode_switch_requires_both_triggers_released():
    assert revo2_triggers_released(0.0, 0.05)
    assert not revo2_triggers_released(0.051, 0.0)
    assert not revo2_triggers_released(0.0, float("nan"))


def test_device_config_is_ordered_left_then_right(tmp_path):
    path = tmp_path / "revo2_devices.json"
    path.write_text(
        json.dumps(
            [
                {"side": "right", "port": "/dev/right", "serial": "R", "slave_id": 127},
                {
                    "side": "left",
                    "port": "/dev/left",
                    "serial": "L",
                    "slave_id": 126,
                    "open_positions": [0.4, 0.4, 0.05, 0.05, 0.05, 0.05],
                },
            ]
        ),
        encoding="utf-8",
    )

    devices = load_revo2_devices(path)

    assert [device.side for device in devices] == ["left", "right"]
    assert [device.slave_id for device in devices] == [126, 127]
    assert devices[0].open_positions == (0.4, 0.4, 0.05, 0.05, 0.05, 0.05)
    assert devices[1].open_positions is None


@pytest.mark.parametrize(
    "open_positions",
    ([0.1] * 5, [1.1] * 6, [float("nan")] * 6),
)
def test_device_config_rejects_invalid_open_positions(tmp_path, open_positions):
    path = tmp_path / "revo2_devices.json"
    path.write_text(
        json.dumps(
            [
                {
                    "side": "left",
                    "port": "/dev/left",
                    "serial": "L",
                    "slave_id": 126,
                    "open_positions": open_positions,
                },
                {"side": "right", "port": "/dev/right", "serial": "R", "slave_id": 127},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="open_positions"):
        load_revo2_devices(path)


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
        Revo2TriggerOutput(devices, finger_delta=1.1)
    with pytest.raises(ValueError, match="command_speed"):
        Revo2TriggerOutput(devices, command_speed=0.5)

    output = Revo2TriggerOutput(devices)
    with pytest.raises(RuntimeError, match="not active"):
        output.submit_triggers(0.2, 0.8, source_received_monotonic_ns=1)
    output.close()
    output.close()


def test_output_cycles_modes_and_wraps(tmp_path):
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
    output = Revo2TriggerOutput(
        load_revo2_devices(path),
        modes=default_revo2_trigger_modes(),
    )

    assert output.current_mode.name == "all_fingers"
    assert output.cycle_mode().name == "pinch"
    assert output.cycle_mode().name == "point"
    assert output.cycle_mode().name == "all_fingers"
    output.close()
