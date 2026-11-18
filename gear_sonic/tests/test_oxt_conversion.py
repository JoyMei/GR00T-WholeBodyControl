import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from gear_sonic.utils.data_collection.oxt_conversion import (
    MAPPING_REVISION,
    _validate_mapping,
    convert_session,
)
from gear_sonic.utils.data_collection.oxt_raw import OxtRawSession


class _FakeReplayBuffer:
    episodes = []

    @classmethod
    def create_from_path(cls, path, mode="w"):
        Path(path).mkdir(parents=True)
        cls.episodes = []
        return cls()

    def add_episode(self, arrays, compressors=None):
        self.episodes.append({key: np.array(value, copy=True) for key, value in arrays.items()})


def _mapping():
    return {
        "schema_revision": MAPPING_REVISION,
        "canonical_rate_hz": 10,
        "submission_metadata": {
            "name": "synthetic_robot_fixture",
            "type": "dexhand",
            "description": "Synthetic converter test fixture; not a dataset submission.",
        },
        "fields": [
            {
                "output_key": "camera_ego_rgb",
                "stream": "camera_ego_rgb",
                "kind": "rgb",
                "max_age_ms": 60,
            },
            {
                "output_key": "right_hand_joints",
                "stream": "robot_state",
                "value_path": ["right_hand_q_measured"],
                "select_indices": [0, 1],
                "dtype": "float32",
                "shape": [2],
                "max_age_ms": 60,
            },
            {
                "output_key": "right_hand_joints_idx",
                "mode": "constant",
                "value": [1, 2],
                "dtype": "int32",
                "shape": [2],
            },
            {
                "output_key": "right_tactile_data_fixture",
                "stream": "revo2_native_sample_batch",
                "payload_encoding": "json_topic_prefix",
                "topic_prefix": "revo2_raw",
                "items_path": ["samples"],
                "value_path": ["sample", "pressure"],
                "reshape": [1, 2],
                "sample_time": {
                    "path": ["sample", "source_timestamp", "value"],
                    "unit": "ns",
                    "clock_domain": "host_monotonic",
                },
                "dtype": "float32",
                "shape": [1, 2],
                "max_age_ms": 60,
            },
            {
                "output_key": "right_tactile_area_fixture",
                "mode": "constant",
                "value": [0],
                "dtype": "int32",
                "shape": [1],
            },
            {
                "output_key": "right_tactile_sensor_fixture",
                "mode": "constant",
                "value": "SyntheticSensor",
                "dtype": "<U15",
                "shape": [],
            },
            {
                "output_key": "right_tactile_type_fixture",
                "mode": "constant",
                "value": "state",
                "dtype": "<U5",
                "shape": [],
            },
        ],
    }


def _make_session(dataset_root: Path, *, stale: bool = False) -> Path:
    session = OxtRawSession(dataset_root, session_id="fixture")
    session.append_event("session_start")
    session.append_event(
        "episode_start",
        episode_id="episode-000001",
        episode_index=0,
        metadata={"task": "move the robot hand"},
    )
    if stale:
        session.submit_stream(
            "camera_ego_rgb",
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        session.submit_stream(
            "robot_state",
            {"right_hand_q_measured": [0.1, 0.2]},
        )
        sample_time_ns = time.monotonic_ns()
        raw_event = {
            "samples": [
                {
                    "sample_index": 0,
                    "sample": {
                        "pressure": [[0.1, 0.2]],
                        "source_timestamp": {"field": "timestamp_ns", "value": sample_time_ns},
                    },
                }
            ]
        }
        session.submit_stream(
            "revo2_native_sample_batch",
            b"revo2_raw" + json.dumps(raw_event).encode("utf-8"),
            received_monotonic_ns=sample_time_ns,
        )
        time.sleep(0.3)
    else:
        for index in range(13):
            session.submit_stream(
                "camera_ego_rgb",
                np.full((8, 8, 3), index, dtype=np.uint8),
            )
            session.submit_stream(
                "robot_state",
                {"right_hand_q_measured": [index, index + 1]},
            )
            sample_time_ns = time.monotonic_ns()
            raw_event = {
                "samples": [
                    {
                        "sample_index": index,
                        "sample": {
                            "pressure": [[index, index + 1]],
                            "source_timestamp": {
                                "field": "timestamp_ns",
                                "value": sample_time_ns,
                            },
                        },
                    }
                ]
            }
            session.submit_stream(
                "revo2_native_sample_batch",
                b"revo2_raw" + json.dumps(raw_event).encode("utf-8"),
                received_monotonic_ns=sample_time_ns,
            )
            time.sleep(0.025)
    session.append_event(
        "episode_complete", episode_id="episode-000001", episode_index=0
    )
    session.append_event("session_close")
    return session.close()


class OxtConversionTest(unittest.TestCase):
    def test_mapping_rejects_incomplete_tactile_four_piece_group(self):
        mapping = _mapping()
        mapping["fields"].append(
            {
                "output_key": "right_tactile_data_pad",
                "stream": "touch",
                "dtype": "float32",
                "shape": [1, 2],
                "max_age_ms": 60,
            }
        )
        with self.assertRaisesRegex(ValueError, "incomplete tactile group"):
            _validate_mapping(mapping)

    def test_conversion_writes_candidate_and_keeps_unqualified_status_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root = _make_session(root / "dataset")
            output = root / "candidate"
            result = convert_session(
                session_root,
                _mapping(),
                output,
                root / "unused-oxt-qc",
                replay_buffer_class=_FakeReplayBuffer,
            )

            self.assertEqual(result, output)
            self.assertTrue((output / "dataset.zarr").is_dir())
            self.assertTrue((output / "metadata.json").is_file())
            report = json.loads((output / "processing_report.json").read_text())
            self.assertFalse(report["source_qualified_for_oxt"])
            self.assertFalse(report["release_ready"])
            self.assertEqual(report["episodes"][0]["task"], "move the robot hand")
            self.assertGreater(report["total_frames"], 1)
            self.assertTrue((output / "alignment" / "episode-000000.npz").is_file())

            episode = _FakeReplayBuffer.episodes[0]
            self.assertEqual(episode["camera_ego_rgb"].dtype, np.uint8)
            self.assertEqual(episode["camera_ego_rgb"].shape[1:], (8, 8, 3))
            self.assertEqual(episode["right_hand_joints"].shape[1:], (2,))
            self.assertEqual(episode["right_tactile_data_fixture"].shape[1:], (1, 2))
            self.assertTrue(np.all(np.diff(episode["timestamps"]) > 0))
            self.assertEqual(len(set(episode["sub_task_instruction"].tolist())), 1)

            with self.assertRaises(FileExistsError):
                convert_session(
                    session_root,
                    _mapping(),
                    output,
                    root / "unused-oxt-qc",
                    replay_buffer_class=_FakeReplayBuffer,
                )

    def test_conversion_fails_closed_on_stale_fields_without_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root = _make_session(root / "dataset", stale=True)
            output = root / "candidate"
            with self.assertRaisesRegex(ValueError, "exceeds max_age_ms"):
                convert_session(
                    session_root,
                    _mapping(),
                    output,
                    root / "unused-oxt-qc",
                    replay_buffer_class=_FakeReplayBuffer,
                )
            self.assertFalse(output.exists())

    def test_field_selection_and_reshape_are_explicit(self):
        mapping = _mapping()
        mapping["fields"][1]["select_indices"] = [0]
        mapping["fields"][1]["shape"] = [1]
        mapping["fields"][3]["reshape"] = [1, 2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root = _make_session(root / "dataset")
            output = root / "candidate"
            convert_session(
                session_root,
                mapping,
                output,
                root / "unused-oxt-qc",
                replay_buffer_class=_FakeReplayBuffer,
            )
            episode = _FakeReplayBuffer.episodes[0]
            self.assertEqual(episode["right_hand_joints"].shape[1:], (1,))
            self.assertEqual(episode["right_tactile_data_fixture"].shape[1:], (1, 2))

    @unittest.skipUnless(os.environ.get("OXT_QC_ROOT"), "set OXT_QC_ROOT for integration test")
    def test_official_replay_buffer_and_oxt_qc_precheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root = _make_session(root / "dataset")
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(_mapping()), encoding="utf-8")
            output = root / "candidate"
            oxt_qc_root = Path(os.environ["OXT_QC_ROOT"])
            converter = Path(__file__).parents[1] / "scripts" / "convert_oxt_raw.py"
            cli_result = subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(session_root),
                    str(mapping_path),
                    str(output),
                    "--oxt-qc-root",
                    str(oxt_qc_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            )
            self.assertEqual(cli_result.returncode, 0, cli_result.stdout + cli_result.stderr)
            precheck = oxt_qc_root / "precheck_metadata.py"
            qc_dir = output / "qc"
            result = subprocess.run(
                [
                    sys.executable,
                    str(precheck),
                    str(output / "dataset.zarr"),
                    "--output-dir",
                    str(qc_dir),
                    "--no-metadata",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((qc_dir / "precheck_report_result.json").read_text())
            self.assertEqual(report["overall_status"], "pass")


if __name__ == "__main__":
    unittest.main()
