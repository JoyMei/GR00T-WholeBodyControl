import hashlib
import json
from contextlib import nullcontext
import queue
import struct
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from gear_sonic.utils.data_collection.oxt_raw import OxtCaptureRuntime, OxtRawSession
from gear_sonic.scripts.validate_oxt_raw import validate_session
from gear_sonic.utils.data_collection.oxt_raw import (
    _decode_robot_config,
    _decode_g1_robot_state_action,
    _decode_revo2_robot_state_action,
)


class OxtRawSessionTest(unittest.TestCase):
    def test_session_writes_immutable_payloads_events_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = OxtRawSession(temp_dir, session_id="unit-test")
            session.append_event("session_start")
            session.append_clock(context={"test": True})
            self.assertTrue(
                session.submit_stream(
                    "motor_state",
                    {"position": np.float32(1.5), "valid": True},
                    source_timestamp_ns=123,
                )
            )
            self.assertTrue(session.submit_stream("wire", b"abc"))
            session.append_event("session_close")
            path = session.close()

            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "closed")
            self.assertTrue(manifest["valid"])
            self.assertTrue(manifest["storage_integrity_valid"])
            self.assertIn("episode_events.jsonl", manifest["file_integrity"])
            self.assertIn("frame_clock.jsonl", manifest["file_integrity"])
            self.assertIn("streams/motor_state.jsonl", manifest["file_integrity"])
            self.assertEqual(manifest["queue_overflow"], 0)
            self.assertEqual(manifest["streams"]["motor_state"]["records"], 1)
            self.assertEqual(manifest["streams"]["wire"]["records"], 1)

            motor_record = json.loads(
                (path / "streams" / "motor_state.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(motor_record["source_timestamp_ns"], 123)
            self.assertEqual(motor_record["payload"]["value"]["position"], 1.5)
            wire_record = json.loads(
                (path / "streams" / "wire.jsonl").read_text().splitlines()[0]
            )
            blob_path = path / wire_record["payload"]["path"]
            self.assertEqual(blob_path.read_bytes(), b"abc")
            self.assertEqual(
                wire_record["payload"]["sha256"],
                hashlib.sha256(b"abc").hexdigest(),
            )
            self.assertEqual(
                len((path / "episode_events.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                len((path / "frame_clock.jsonl").read_text().splitlines()), 1
            )

            with self.assertRaises(FileExistsError):
                OxtRawSession(temp_dir, session_id="unit-test")

            valid, errors = validate_session(path)
            self.assertTrue(valid, errors)

    def test_queue_overflow_is_recorded_and_invalidates_storage(self):
        original_write_item = OxtRawSession._write_item
        entered = threading.Event()
        release = threading.Event()

        def blocked_write_item(instance, item):
            if not entered.is_set():
                entered.set()
                release.wait(timeout=2.0)
            original_write_item(instance, item)

        OxtRawSession._write_item = blocked_write_item
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                session = OxtRawSession(temp_dir, queue_size=8)
                self.assertTrue(session.submit_stream("blocked", {"n": 0}))
                self.assertTrue(entered.wait(timeout=1.0))
                for index in range(8):
                    session.submit_stream("blocked", {"n": index + 1})
                self.assertFalse(session.submit_stream("blocked", {"n": 99}))
                release.set()
                path = session.close(timeout=5.0)
                manifest = json.loads((path / "manifest.json").read_text())
                self.assertGreater(manifest["queue_overflow"], 0)
                self.assertFalse(manifest["valid"])
                self.assertFalse(manifest["storage_integrity_valid"])
        finally:
            OxtRawSession._write_item = original_write_item

    def test_byte_budget_is_enforced_and_invalidates_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = OxtRawSession(
                temp_dir,
                session_id="byte-budget",
                queue_size=8,
                max_pending_bytes=8,
            )
            self.assertFalse(session.submit_stream("wire", b"123456789"))
            path = session.close()
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["byte_overflow"], 1)
            self.assertFalse(manifest["storage_integrity_valid"])


class OxtCaptureRuntimeTest(unittest.TestCase):
    def test_runtime_records_episode_and_camera_without_zmq(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = OxtCaptureRuntime(temp_dir, session_id="runtime-test", start_zmq=False)
            runtime.start()
            episode_id = runtime.begin_episode(episode_index=3, task="touch the object")
            self.assertEqual(episode_id, "episode-000001")
            runtime.capture_camera_message(
                {
                    "images": {"ego view": np.zeros((2, 3, 3), dtype=np.uint8)},
                    "timestamps": {"ego view": 12.5},
                },
                received_monotonic_ns=456,
            )
            runtime.finish_episode("episode_complete", episode_index=3)
            path = runtime.close()

            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "closed")
            self.assertFalse(manifest["oxt_submission_ready"])
            self.assertEqual(
                manifest["metadata"]["capture_mode"],
                "nonblocking_robot_state_action_camera_relay",
            )
            records = (path / "streams" / "camera_ego_view_rgb.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            record = json.loads(records[0])
            self.assertEqual(record["received_monotonic_ns"], 456)
            self.assertEqual(record["source_timestamp_ns"], 12_500_000_000)

            events = [
                json.loads(line)
                for line in (path / "episode_events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["session_start", "episode_start", "episode_complete", "session_close"],
            )
            clock_records = [
                json.loads(line)
                for line in (path / "frame_clock.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["metadata"]["kind"] for record in clock_records],
                ["session_start", "session_close"],
            )

    def test_camera_is_not_written_outside_machine_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = OxtCaptureRuntime(temp_dir, session_id="episode-gate", start_zmq=False)
            runtime.start()
            image = np.zeros((2, 3, 3), dtype=np.uint8)
            runtime.capture_camera_message({"images": {"ego": image}})
            runtime.begin_episode(episode_index=0, task="move robot hand")
            runtime.capture_camera_message({"images": {"ego": image}})
            runtime.finish_episode("episode_complete", episode_index=0)
            path = runtime.close()

            records = (path / "streams" / "camera_ego_rgb.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)

    def test_topic_matching_prefers_longest_known_topic(self):
        self.assertEqual(
            OxtCaptureRuntime._topic_for(b"revo2_state" + b"payload"), "revo2_state"
        )
        self.assertEqual(OxtCaptureRuntime._topic_for(b"not-a-topic"), None)


class OxtRawValidationTest(unittest.TestCase):
    def test_validator_rejects_modified_control_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = OxtRawSession(temp_dir, session_id="tamper-test")
            session.append_event("session_start")
            session.submit_stream("motor_state", {"position": 1})
            session.append_event("session_close")
            path = session.close()

            stream_path = path / "streams" / "motor_state.jsonl"
            record = json.loads(stream_path.read_text().splitlines()[0])
            record["payload"]["value"]["position"] = 2
            stream_path.write_text(json.dumps(record) + "\n")

            valid, errors = validate_session(path)
            self.assertFalse(valid)
            self.assertTrue(
                any(
                    "sha256 mismatch for control file streams/motor_state.jsonl" in error
                    for error in errors
                ),
                errors,
            )

    def test_validator_rejects_unindexed_files_and_orphan_blobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = OxtRawSession(temp_dir, session_id="orphan-test")
            session.append_event("session_start")
            session.submit_stream("wire", b"referenced")
            session.append_event("session_close")
            path = session.close()
            (path / "notes.txt").write_text("unindexed control file")
            orphan = path / "blobs" / "wire" / "orphan.bin"
            orphan.write_bytes(b"unreferenced blob")

            valid, errors = validate_session(path)
            self.assertFalse(valid)
            self.assertTrue(any("control file set differs" in error for error in errors), errors)
            self.assertTrue(any("blob file set differs" in error for error in errors), errors)

    def test_validator_rejects_blob_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = OxtRawSession(temp_dir, session_id="escape-test")
            session.append_event("session_start")
            session.submit_stream("wire", b"abc")
            session.append_event("session_close")
            path = session.close()
            record_path = path / "streams" / "wire.jsonl"
            record = json.loads(record_path.read_text().splitlines()[0])
            record["payload"]["path"] = "../outside.bin"
            record_path.write_text(json.dumps(record) + "\n")

            valid, errors = validate_session(path)
            self.assertFalse(valid)
            self.assertTrue(any("invalid blob path" in error for error in errors))


class _DropCounter:
    value = 0

    def get_lock(self):
        return nullcontext()


class _FakeSample:
    positions = np.array([1, 2, 3], dtype=np.int16)
    speeds = np.array([4, 5, 6], dtype=np.int16)
    currents = np.array([7, 8, 9], dtype=np.int16)
    states = np.array([0, 1, 2], dtype=np.int16)
    timestamp_ns = 123


class Revo2RawSerializationTest(unittest.TestCase):
    def test_serialization_is_json_safe_and_queue_drop_is_counted(self):
        from gear_sonic.utils.teleop.revo2_trigger import (
            Revo2Device,
            _serialize_motor_sample,
            _submit_raw_event,
        )

        sample = _serialize_motor_sample(_FakeSample())
        self.assertEqual(sample["positions"], [1, 2, 3])
        self.assertEqual(sample["source_timestamp"], {"field": "timestamp_ns", "value": 123})
        json.dumps(sample, allow_nan=False)

        raw_queue = queue.Queue(maxsize=1)
        dropped = _DropCounter()
        event = {
            "stream": "revo2_motor_status",
            "side": "left",
            "samples": [],
        }
        _submit_raw_event(raw_queue, dropped, event)
        _submit_raw_event(raw_queue, dropped, event)
        self.assertEqual(dropped.value, 1)

    def test_owner_close_preserves_pending_raw_events(self):
        from gear_sonic.utils.teleop.revo2_trigger import Revo2Device, Revo2TriggerOutput

        output = Revo2TriggerOutput(
            (
                Revo2Device("left", "/dev/left", "L", 126),
                Revo2Device("right", "/dev/right", "R", 127),
            ),
            capture_raw_data=True,
        )
        pending = {"stream": "revo2_motor_status", "samples": [{"position": 1}]}
        output._raw_queue.put_nowait(pending)
        output.close()

        self.assertEqual(output.get_raw_samples(), [pending])


class RobotOnlyAllowlistTest(unittest.TestCase):
    def test_robot_config_drops_source_file_paths(self):
        import msgpack

        payload = msgpack.packb(
            {
                "control_frequency": 50.0,
                "planner_frequency": 20.0,
                "is_using_encoder": True,
                "model_path": "/home/operator/private/model.onnx",
                "reference_motion_path": "/home/operator/private/motion.npz",
                "planner_path": "/home/operator/private/planner.json",
            },
            use_bin_type=True,
        )
        result = _decode_robot_config(b"robot_config" + payload)

        self.assertEqual(result["control_frequency"], 50.0)
        self.assertEqual(result["planner_frequency"], 20.0)
        self.assertTrue(result["is_using_encoder"])
        self.assertNotIn("model_path", result)
        self.assertNotIn("reference_motion_path", result)
        self.assertNotIn("planner_path", result)

    def test_g1_message_drops_vr_and_operator_derived_fields(self):
        import msgpack

        payload = msgpack.packb(
            {
                "ros_timestamp": 123.5,
                "body_q": [0.1, 0.2],
                "last_action": [0.3, 0.4],
                "vr_3point_position": [9.0, 9.0, 9.0],
                "vr_3point_orientation": [8.0] * 12,
                "delta_heading": 1.25,
                "controller_serial": "person-specific-value",
            },
            use_bin_type=True,
        )
        result, timestamp_ns = _decode_g1_robot_state_action(b"g1_debug" + payload)

        self.assertEqual(timestamp_ns, 123_500_000_000)
        self.assertEqual(result["body_q"], [0.1, 0.2])
        self.assertEqual(result["last_action"], [0.3, 0.4])
        self.assertNotIn("vr_3point_position", result)
        self.assertNotIn("vr_3point_orientation", result)
        self.assertNotIn("delta_heading", result)
        self.assertNotIn("controller_serial", result)

    def test_revo2_snapshot_drops_trigger_input_but_keeps_robot_command(self):
        header = {
            "v": 3,
            "endian": "le",
            "count": 1,
            "fields": [
                {"name": "source_trigger", "dtype": "f32", "shape": [2]},
                {"name": "source_received_ns", "dtype": "i64", "shape": [1]},
                {"name": "sequence", "dtype": "i64", "shape": [1]},
                {"name": "command_position_raw", "dtype": "f32", "shape": [2, 6]},
                {"name": "motor_position_raw", "dtype": "f32", "shape": [2, 6]},
                {"name": "publish_ns", "dtype": "i64", "shape": [1]},
            ],
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        payload = b"".join(
            (
                struct.pack("<ff", 0.2, 0.8),
                struct.pack("<q", 111),
                struct.pack("<q", 3),
                struct.pack("<12f", *range(12)),
                struct.pack("<12f", *range(12, 24)),
                struct.pack("<q", 222),
            )
        )
        raw = b"revo2_state" + header_bytes.ljust(1280, b"\0") + payload
        result, timestamp_ns = _decode_revo2_robot_state_action(raw)

        self.assertEqual(timestamp_ns, 222)
        self.assertIn("command_position_raw", result)
        self.assertIn("motor_position_raw", result)
        self.assertNotIn("source_trigger", result)
        self.assertNotIn("source_received_ns", result)
        self.assertNotIn("sequence", result)

    def test_human_input_topics_are_not_subscribed(self):
        for topic in ("pose", "planner", "manager_state"):
            self.assertNotIn(topic, OxtCaptureRuntime._TOPICS)


if __name__ == "__main__":
    unittest.main()
