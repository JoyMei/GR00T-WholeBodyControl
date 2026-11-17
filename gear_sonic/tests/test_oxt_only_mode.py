import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from gear_sonic.scripts import run_data_exporter as exporter_script
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector,
    SonicDataExporterConfig,
    validate_capture_mode_config,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState


class FakeKeyboard:
    def __init__(self, *keys):
        self.keys = iter(keys)

    def read_msg(self):
        return next(self.keys, None)


class FakeOxtCapture:
    def __init__(self):
        self.events = []

    def begin_episode(self, *, episode_index, task):
        self.events.append(("begin", episode_index, task))

    def finish_episode(self, event, *, episode_index, metadata=None):
        self.events.append((event, episode_index, metadata))

    def close(self):
        return "/tmp/oxt-session"


def make_oxt_only_collector(*keys):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.text_to_speech = None
    collector.data_exporter = None
    collector.oxt_capture = FakeOxtCapture()
    collector.oxt_task_prompt = "move robot hand"
    collector.oxt_only = True
    collector._oxt_episode_index = 0
    collector._episode_state = EpisodeState()
    collector._keyboard_listener = FakeKeyboard(*keys)
    collector._manager_toggle_da = False
    collector._manager_toggle_dc = False
    collector._initial_yaw = None
    return collector


class OxtOnlyModeTest(unittest.TestCase):
    def test_native_lerobot_default_config_remains_valid(self):
        validate_capture_mode_config(SonicDataExporterConfig())

    def test_episode_start_and_complete_do_not_need_lerobot_exporter(self):
        collector = make_oxt_only_collector("c", "c")

        collector._check_recording_commands()
        self.assertEqual(collector._episode_state.get_state(), collector._episode_state.RECORDING)
        collector._check_recording_commands()

        self.assertEqual(
            collector.oxt_capture.events,
            [("begin", 0, "move robot hand"), ("episode_complete", 0, None)],
        )
        self.assertEqual(collector.current_episode_index, 1)
        self.assertEqual(collector._episode_state.get_state(), collector._episode_state.IDLE)

    def test_discard_finishes_raw_episode_and_advances_index(self):
        collector = make_oxt_only_collector("c", "x")
        collector._check_recording_commands()
        collector._check_recording_commands()

        self.assertEqual(
            collector.oxt_capture.events,
            [("begin", 0, "move robot hand"), ("episode_discard", 0, None)],
        )
        self.assertIsNone(collector.data_exporter)
        self.assertEqual(collector.current_episode_index, 1)

    def test_oxt_only_config_requires_an_independent_raw_output_root(self):
        config = SonicDataExporterConfig(oxt_only=True)

        with self.assertRaisesRegex(ValueError, "oxt-raw-output-dir is required"):
            validate_capture_mode_config(config)

    def test_dual_capture_is_rejected_unless_explicitly_enabled(self):
        config = SonicDataExporterConfig(oxt_raw_output_dir="/tmp/oxt")

        with self.assertRaisesRegex(ValueError, "not latency-qualified"):
            validate_capture_mode_config(config)

    def test_dual_capture_requires_a_raw_output_root(self):
        config = SonicDataExporterConfig(allow_unbenchmarked_dual_capture=True)

        with self.assertRaisesRegex(ValueError, "requires --oxt-raw-output-dir"):
            validate_capture_mode_config(config)

    def test_dual_capture_can_only_be_enabled_explicitly(self):
        config = SonicDataExporterConfig(
            oxt_raw_output_dir="/tmp/oxt",
            allow_unbenchmarked_dual_capture=True,
        )

        validate_capture_mode_config(config)

    def test_oxt_only_rejects_lerobot_specific_options(self):
        options = (
            ("record_revo2", "LeRobot snapshot option"),
            ("record_wrist_cameras", "LeRobot schema option"),
        )
        for option, message in options:
            with self.subTest(option=option):
                config = SonicDataExporterConfig(
                    oxt_only=True,
                    oxt_raw_output_dir="/tmp/oxt",
                    **{option: True},
                )
                with self.assertRaisesRegex(ValueError, message):
                    validate_capture_mode_config(config)

    def test_main_oxt_only_does_not_construct_lerobot_components(self):
        with TemporaryDirectory() as output_dir:
            config = SonicDataExporterConfig(
                oxt_only=True,
                oxt_raw_output_dir=output_dir,
                text_to_speech=False,
            )
            runtime = mock.Mock(session_root=Path(output_dir) / "raw" / "session")
            runtime.start.return_value = runtime
            collector = mock.Mock()

            with (
                mock.patch.object(exporter_script, "poll_robot_config_zmq", return_value={}),
                mock.patch.object(exporter_script, "get_g1_robot_model") as get_robot_model,
                mock.patch.object(exporter_script.Gr00tDataExporter, "create") as create_exporter,
                mock.patch.object(
                    exporter_script, "GrootDataCollector", return_value=collector
                ) as create_collector,
                mock.patch(
                    "gear_sonic.utils.data_collection.oxt_raw.OxtCaptureRuntime",
                    return_value=runtime,
                ),
            ):
                exporter_script.main(config)

            get_robot_model.assert_not_called()
            create_exporter.assert_not_called()
            collector_kwargs = create_collector.call_args.kwargs
            self.assertIsNone(collector_kwargs["data_exporter"])
            self.assertIsNone(collector_kwargs["robot_model"])
            self.assertTrue(collector_kwargs["oxt_only"])
            collector.run.assert_called_once_with()

    def test_oxt_only_cleanup_closes_raw_session_without_lerobot_exporter(self):
        collector = make_oxt_only_collector()
        collector._state_subscriber = None
        collector._sonic_zmq_socket = None
        collector._sonic_zmq_ctx = None

        collector.save_and_cleanup()

        self.assertIsNone(collector.data_exporter)


if __name__ == "__main__":
    unittest.main()
