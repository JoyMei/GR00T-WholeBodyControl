import asyncio
from types import SimpleNamespace

from gear_sonic.utils.teleop.revo2_data import (
    REVO2_TOUCH_CAPACITIVE,
    REVO2_TOUCH_PRESSURE_SUMMARY,
    revo2_capture_metadata,
)
from gear_sonic.utils.teleop.revo2_trigger import (
    Revo2Device,
    Revo2TriggerOutput,
    _create_data_collector,
    _device_supports_touch,
)


class _Buffer:
    def __init__(self, *, max_size):
        self.max_size = max_size


class _CollectorFactory:
    calls = []

    @classmethod
    def _new(cls, mode, **kwargs):
        result = SimpleNamespace(mode=mode, kwargs=kwargs)
        cls.calls.append(result)
        return result

    @classmethod
    def new_basic(cls, **kwargs):
        return cls._new("basic", **kwargs)

    @classmethod
    def new_capacitive(cls, **kwargs):
        return cls._new("capacitive", **kwargs)

    @classmethod
    def new_pressure_summary(cls, **kwargs):
        return cls._new("pressure_summary", **kwargs)

    @classmethod
    def new_pressure_hybrid(cls, **kwargs):
        return cls._new("pressure_hybrid", **kwargs)


class _FakeSdk:
    MotorStatusBuffer = _Buffer
    TouchStatusBuffer = _Buffer
    PressureSummaryBuffer = _Buffer
    PressureDetailedBuffer = _Buffer
    DataCollector = _CollectorFactory


class _FakeClient:
    def __init__(self, *, pressure, enabled_mask):
        self.pressure = pressure
        self.enabled_mask = enabled_mask
        self.setup_masks = []

    async def uses_pressure_touch_api(self, slave_id):
        return self.pressure

    async def get_touch_sensor_enabled(self, slave_id):
        return self.enabled_mask

    async def touch_sensor_setup(self, slave_id, mask):
        self.setup_masks.append(mask)
        self.enabled_mask = mask


def _device(side="left"):
    return Revo2Device(
        side=side,
        port=f"/dev/revo2-{side}",
        serial=f"serial-{side}",
        slave_id=126 if side == "left" else 127,
    )


def test_capacitive_collector_reuses_control_client_without_configuration_write():
    client = _FakeClient(pressure=False, enabled_mask=0x1F)

    resource = asyncio.run(
        _create_data_collector(
            _FakeSdk,
            client,
            _device(),
            SimpleNamespace(is_touch=lambda: True),
            frequency=20,
            touch_setup=False,
        )
    )

    assert resource["touch_type"] == REVO2_TOUCH_CAPACITIVE
    assert resource["collector"].mode == "capacitive"
    assert resource["collector"].kwargs["ctx"] is client
    assert resource["collector"].kwargs["motor_frequency"] == 20
    assert resource["collector"].kwargs["touch_frequency"] == 20
    assert client.setup_masks == []


def test_pressure_collector_preserves_summary_and_detailed_streams():
    client = _FakeClient(pressure=True, enabled_mask=0x3F)

    resource = asyncio.run(
        _create_data_collector(
            _FakeSdk,
            client,
            _device(),
            SimpleNamespace(uses_revo2_touch_api=lambda: True),
            frequency=20,
            touch_setup=False,
        )
    )

    assert resource["touch_type"] == REVO2_TOUCH_PRESSURE_SUMMARY
    assert resource["collector"].mode == "pressure_hybrid"
    assert resource["collector"].kwargs["summary_frequency"] == 20
    assert resource["collector"].kwargs["detailed_frequency"] == 10
    assert resource["pressure_detailed_buffer"] is not None


def test_touch_detection_checks_both_supported_sdk_properties():
    info = SimpleNamespace(
        is_touch=lambda: False,
        uses_revo2_touch_api=lambda: True,
    )

    assert _device_supports_touch(info)


def test_touch_setup_requires_explicit_opt_in():
    client = _FakeClient(pressure=False, enabled_mask=0)

    try:
        asyncio.run(
            _create_data_collector(
                _FakeSdk,
                client,
                _device(),
                SimpleNamespace(is_touch=True),
                frequency=20,
                touch_setup=False,
            )
        )
    except RuntimeError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("disabled touch sensors must fail without explicit setup")

    resource = asyncio.run(
        _create_data_collector(
            _FakeSdk,
            client,
            _device(),
            SimpleNamespace(is_touch=True),
            frequency=20,
            touch_setup=True,
        )
    )
    assert resource["collector"].mode == "capacitive"
    assert client.setup_masks == [0x1F]


def test_slew_cannot_exceed_command_speed():
    try:
        Revo2TriggerOutput(
            (_device("left"), _device("right")),
            slew_per_s=0.3,
            command_speed=0.25,
        )
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("unsafe slew/command speed mismatch must be rejected")


def test_data_rate_requires_integer_sdk_frequency():
    try:
        Revo2TriggerOutput(
            (_device("left"), _device("right")),
            data_rate_hz=20.5,
        )
    except ValueError as exc:
        assert "integer frequency" in str(exc)
    else:
        raise AssertionError("fractional SDK collection frequency must be rejected")


def test_capture_metadata_records_device_identity_and_oxt_boundary():
    metadata = revo2_capture_metadata((_device("left"), _device("right")))

    assert metadata["schema_revision"] == "revo2-lerobot-v1"
    assert [item["serial"] for item in metadata["devices"]] == [
        "serial-left",
        "serial-right",
    ]
    assert metadata["pressure_sensor_counts"] == [9, 9, 9, 9, 9, 46]
    assert "validity mask" in metadata["sampling"]["inactive_touch_fields"]
    assert metadata["oxt_immutable_raw_bundle"] is False
