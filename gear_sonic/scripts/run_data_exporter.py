"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --dataset-name my_session
    python gear_sonic/scripts/run_data_exporter.py --oxt-only --oxt-raw-output-dir /data/oxt/demo
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_revo2_features,
    get_revo2_modality_config,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)
from gear_sonic.utils.teleop.revo2_data import (
    revo2_capture_metadata,
    revo2_state_freshness,
    revo2_state_to_features,
)


def install_shutdown_signal_handlers() -> None:
    """Make background exporter jobs run their normal KeyboardInterrupt cleanup."""

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = "demo"
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""


    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    record_revo2: bool = False
    """Record measured/command/tactile Revo2 fields from the manager's revo2_state topic."""

    revo2_devices: str | None = None
    """Same verified bilateral device JSON used by the manager; stored as provenance."""

    revo2_max_age_ms: float = 200.0
    """Maximum accepted Revo2 publisher, motor, and tactile sample age."""

    require_revo2_tactile: bool = True
    """Require fresh bilateral tactile data whenever Revo2 recording is enabled."""

    oxt_raw_output_dir: str | None = None
    """Optional independent OXT raw-session root; disabled by default."""

    oxt_only: bool = False
    """Write OXT raw only; do not create or write a LeRobot dataset."""

    allow_unbenchmarked_dual_capture: bool = False
    """Explicitly allow OXT raw and LeRobot output in one process before latency qualification."""

    oxt_session_id: str | None = None
    """Optional OXT session ID. Existing session paths are never overwritten."""

    oxt_raw_queue_size: int = 8192
    """Bounded OXT writer queue; overflow is recorded as a capture fault."""

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


def validate_capture_mode_config(config: SonicDataExporterConfig) -> None:
    if config.oxt_only:
        if not config.oxt_raw_output_dir:
            raise ValueError("--oxt-raw-output-dir is required with --oxt-only")
        if config.allow_unbenchmarked_dual_capture:
            raise ValueError(
                "--allow-unbenchmarked-dual-capture cannot be combined with --oxt-only"
            )
        if config.record_revo2:
            raise ValueError(
                "--record-revo2 is a LeRobot snapshot option; OXT raw is captured separately"
            )
        if config.record_wrist_cameras:
            raise ValueError(
                "--record-wrist-cameras is a LeRobot schema option and is unavailable in OXT-only mode"
            )
    elif config.oxt_raw_output_dir and not config.allow_unbenchmarked_dual_capture:
        raise ValueError(
            "OXT + LeRobot dual capture is not latency-qualified; use --oxt-only or "
            "explicitly pass --allow-unbenchmarked-dual-capture after reviewing the risk"
        )
    elif config.allow_unbenchmarked_dual_capture and not config.oxt_raw_output_dir:
        raise ValueError("--allow-unbenchmarked-dual-capture requires --oxt-raw-output-dir")


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][1280-byte JSON header][concatenated binary fields]
    """
    HEADER_SIZE = 1280

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        record_revo2: bool = False,
        revo2_max_age_ms: float = 200.0,
        require_revo2_tactile: bool = True,
        oxt_capture=None,
        oxt_task_prompt: str = "demo",
        oxt_only: bool = False,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.record_revo2 = bool(record_revo2)
        if not np.isfinite(revo2_max_age_ms) or not 20.0 <= revo2_max_age_ms <= 2000.0:
            raise ValueError("revo2_max_age_ms must be finite and within [20, 2000]")
        self.revo2_max_age_ns = int(revo2_max_age_ms * 1e6)
        self.require_revo2_tactile = bool(require_revo2_tactile)
        self.oxt_capture = oxt_capture
        self.oxt_task_prompt = str(oxt_task_prompt)
        self.oxt_only = bool(oxt_only)
        self._oxt_episode_index = 0
        if self.oxt_only and (self.oxt_capture is None or data_exporter is not None):
            raise ValueError("OXT-only mode requires an OXT capture and no LeRobot exporter")
        if not self.oxt_only and data_exporter is None:
            raise ValueError("LeRobot exporter is required unless OXT-only mode is enabled")

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()

        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None
        self.latest_revo2_msg = None

        self.current_stream_mode = 0
        self.current_revo2_mode_index = -1
        self.current_revo2_mode_name_utf8 = np.zeros(32, dtype=np.int32)
        self.current_revo2_mode_channels = np.zeros(6, dtype=np.float32)
        self.current_revo2_mode_offsets = np.zeros(6, dtype=np.float32)
        self.current_revo2_left_locked = False
        self.current_revo2_right_locked = False

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._state_subscriber = (
            None
            if self.oxt_only
            else ZMQStateSubscriber(host=state_zmq_host, port=state_zmq_port)
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        try:
            self._sonic_zmq_ctx = zmq.Context()
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
            if not self.oxt_only:
                self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
                self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
            if self.record_revo2:
                self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "revo2_state")
            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {sonic_data_zmq_host}:{sonic_data_zmq_port}")
            topics = ["manager_state"]
            if not self.oxt_only:
                topics.extend(["pose", "planner"])
            if self.record_revo2:
                topics.append("revo2_state")
            print(f"[Sonic] Subscribed to: {', '.join(topics)}")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3, reset_timeout_sec=5, time_delta=0.1
        )

        self._last_latency_log_time = 0.0
        self._last_revo2_wait_log_time = 0.0
        self._initial_yaw = None

        if self.oxt_only:
            print("[OXT] OXT-only mode; LeRobot output is disabled")
        else:
            print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        if self.data_exporter is None:
            return self._oxt_episode_index
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        if self._state_subscriber is None:
            return
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return

        if msg.get("ros_timestamp", 0.0) == 0.0:
            msg["ros_timestamp"] = time.time()

        self.latest_proprio_msg = msg

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                if self.oxt_capture is not None:
                    self.oxt_capture.begin_episode(
                        episode_index=self.current_episode_index,
                        task=self.oxt_task_prompt,
                    )
                self._print_and_say(
                    (
                        f"Started OXT raw episode {self.current_episode_index}"
                        if self.oxt_only
                        else f"Started recording {self.current_episode_index}"
                    ),
                    blocking=False,
                )
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                if self.oxt_only:
                    self.oxt_capture.finish_episode(
                        "episode_complete",
                        episode_index=self.current_episode_index,
                    )
                    self._print_and_say(
                        f"Finished OXT raw episode {self.current_episode_index}", say=False
                    )
                    self._oxt_episode_index += 1
                    self._initial_yaw = None
                    self._episode_state.reset_state()
                    return
                if self.oxt_capture is not None:
                    buffer_size = self.data_exporter.episode_buffer.get("size", 0)
                    self.oxt_capture.finish_episode(
                        "episode_complete" if buffer_size > 0 else "episode_discard",
                        episode_index=self.current_episode_index,
                    )
                self._print_and_say("Stopping recording, preparing to save", blocking=False)
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state", blocking=False)
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                if self.oxt_capture is not None:
                    self.oxt_capture.finish_episode(
                        "episode_discard",
                        episode_index=self.current_episode_index,
                    )
                if self.data_exporter is not None:
                    self.data_exporter.save_episode_as_discarded()
                if self.oxt_only:
                    self._print_and_say(
                        f"Discarded OXT raw episode {self.current_episode_index}", say=False
                    )
                    self._oxt_episode_index += 1
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._print_and_say("Discarded episode", blocking=False)

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for pose, planner, and manager_state messages (non-blocking)."""
        if self._sonic_zmq_socket is None:
            return

        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"revo2_state"):
                self._handle_revo2_state(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            self.current_stream_mode = int(data["stream_mode"].flat[0])
        if "revo2_mode_index" in data:
            self.current_revo2_mode_index = int(data["revo2_mode_index"].flat[0])
        if "revo2_mode_name_utf8" in data and data["revo2_mode_name_utf8"].size == 32:
            self.current_revo2_mode_name_utf8 = (
                data["revo2_mode_name_utf8"].flatten().astype(np.int32)
            )
        if "revo2_mode_channels" in data and data["revo2_mode_channels"].size == 6:
            self.current_revo2_mode_channels = (
                data["revo2_mode_channels"].flatten().astype(np.float32)
            )
        if "revo2_mode_offsets" in data and data["revo2_mode_offsets"].size == 6:
            self.current_revo2_mode_offsets = (
                data["revo2_mode_offsets"].flatten().astype(np.float32)
            )
        if "revo2_left_locked" in data:
            self.current_revo2_left_locked = self._extract_bool(data, "revo2_left_locked")
        if "revo2_right_locked" in data:
            self.current_revo2_right_locked = self._extract_bool(data, "revo2_right_locked")

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_revo2_state(self, raw: bytes) -> None:
        try:
            self.latest_revo2_msg = unpack_pose_message(raw, topic="revo2_state")
        except Exception as exc:
            print(f"[Revo2] Error unpacking state message: {exc}")

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][0],
                "smpl_pose": smpl_pose,
                "body_quat_w": (
                    pose_data["body_quat_w"][0] if "body_quat_w" in pose_data else None
                ),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "receive_timestamp": time.time(),
            }
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    @staticmethod
    def _extract_hand_joints(pose_data: dict, key: str) -> np.ndarray:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(7, dtype=np.float32)

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                self._print_and_say("Finished saving episode")
            else:
                self._print_and_say("Skipping save: no frames collected", say=False)
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            self._print_and_say(
                f"Waiting for message. "
                f"Avail msg: proprio {self.latest_proprio_msg is not None} | "
                f"image {self.latest_image_msg is not None}",
                say=False,
            )
            return False

        if self.record_revo2:
            if self.latest_revo2_msg is None:
                self._log_revo2_wait("no revo2_state message received")
                return False
            ready, reason = revo2_state_freshness(
                self.latest_revo2_msg,
                now_monotonic_ns=time.monotonic_ns(),
                max_age_ns=self.revo2_max_age_ns,
                require_tactile=self.require_revo2_tactile,
            )
            if not ready:
                self._log_revo2_wait(reason)
                return False

        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        return self._add_data_frame_sonic(t_start)

    def _log_revo2_wait(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_revo2_wait_log_time >= 1.0:
            self._last_revo2_wait_log_time = now
            self._print_and_say(f"Waiting for Revo2 data: {reason}", say=False)

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["body_q"],
            left_hand_actuated_joint_values=proprio["left_hand_q"],
            right_hand_actuated_joint_values=proprio["right_hand_q"],
        )
        whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["last_action"],
            left_hand_actuated_joint_values=proprio["last_left_hand_action"],
            right_hand_actuated_joint_values=proprio["last_right_hand_action"],
        )

        self.robot_model.cache_forward_kinematics(whole_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        if self.record_revo2:
            assert self.latest_revo2_msg is not None
            frame_data.update(
                revo2_state_to_features(
                    self.latest_revo2_msg, now_monotonic_ns=time.monotonic_ns()
                )
            )

        self._add_images_to_frame_data(frame_data)

        self._log_latency_periodic(sonic_latency_ms)

        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            frame_data["observation.root_orientation"] = base_quat
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
        else:
            frame_data["observation.root_orientation"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            frame_data["observation.projected_gravity"] = np.array(
                [0.0, 0.0, -1.0], dtype=np.float64
            )
            frame_data["observation.cpp_rotation_offset"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float64
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(self, frame_data: dict) -> float | None:
        """Add teleop features based on current stream mode."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)
        frame_data["teleop.revo2_mode_index"] = np.array(
            [self.current_revo2_mode_index], dtype=np.int32
        )
        frame_data["teleop.revo2_mode_name_utf8"] = self.current_revo2_mode_name_utf8.copy()
        frame_data["teleop.revo2_mode_channels"] = self.current_revo2_mode_channels.copy()
        frame_data["teleop.revo2_mode_offsets"] = self.current_revo2_mode_offsets.copy()
        frame_data["teleop.revo2_left_locked"] = np.array(
            [self.current_revo2_left_locked], dtype=np.int32
        )
        frame_data["teleop.revo2_right_locked"] = np.array(
            [self.current_revo2_right_locked], dtype=np.int32
        )

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode == 5 and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = (
            smpl_msg if self.current_stream_mode in (1, 4) and smpl_msg is not None
            else planner_msg if planner_msg is not None
            else smpl_msg
        )
        frame_data["teleop.left_hand_joints"] = (
            hand_msg["left_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("left_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )
        frame_data["teleop.right_hand_joints"] = (
            hand_msg["right_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("right_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return sonic_latency_ms

    def _compute_target_body_orientation(
        self, body_quat_w: np.ndarray, frame_data: dict
    ) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def save_and_cleanup(self):
        if self.oxt_capture is not None:
            try:
                session_path = self.oxt_capture.close()
                self._print_and_say(f"OXT raw session closed: {session_path}", say=False)
            except Exception as exc:
                self._print_and_say(f"OXT raw session close failed: {exc}", say=False)

        if self.data_exporter is not None:
            try:
                self._print_and_say("saving episode done", blocking=False)
                buffer_size = self.data_exporter.episode_buffer.get("size", 0)
                if buffer_size > 0:
                    self.data_exporter.save_episode()
                self._print_and_say(
                    f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
                )
            except Exception as e:
                self._print_and_say(f"Error saving episode: {e}", blocking=True)

        if self._state_subscriber is not None:
            try:
                self._state_subscriber.close()
            except Exception:
                pass
        image_subscriber = getattr(self, "_image_subscriber", None)
        if image_subscriber is not None:
            try:
                image_subscriber.close()
            except Exception:
                pass
        for sock in [self._sonic_zmq_socket]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        for ctx in [self._sonic_zmq_ctx]:
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass

        shutdown_message = (
            "Shutting down OXT raw collector..."
            if self.oxt_only
            else "Shutting down data exporter..."
        )
        self._print_and_say(shutdown_message, say=False)

    def run(self):
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    if not self.oxt_only:
                        with self.telemetry.timer("poll_state"):
                            self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    with self.telemetry.timer("poll_image"):
                        img_msg = self._image_subscriber.read()
                        if img_msg is not None:
                            is_new_image_message = self.latest_image_msg is not img_msg
                            self.latest_image_msg = img_msg
                            if is_new_image_message and self.oxt_capture is not None:
                                self.oxt_capture.capture_camera_message(
                                    img_msg,
                                    received_monotonic_ns=time.monotonic_ns(),
                                )

                    if not self.oxt_only:
                        with self.telemetry.timer("add_frame"):
                            self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print(
                "OXT raw collector terminated by user"
                if self.oxt_only
                else "Data exporter terminated by user"
            )
            if self.oxt_capture is not None:
                self.oxt_capture.finish_episode(
                    "episode_abort",
                    episode_index=self.current_episode_index,
                    metadata={"reason": "collector_keyboard_interrupt"},
                )
            buffer_size = (
                self.data_exporter.episode_buffer.get("size", 0)
                if self.data_exporter is not None
                else 0
            )
            if self.data_exporter is not None and buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: SonicDataExporterConfig):
    validate_capture_mode_config(config)

    g1_rm = None
    data_exporter = None
    revo2_metadata = None
    text_to_speech = None
    if not config.oxt_only:
        g1_rm = get_g1_robot_model()
        dataset_features = get_features_sonic_vla(g1_rm)
        modality_config = get_modality_config_sonic_vla(g1_rm)

        if config.record_wrist_cameras:
            print("[Camera] Wrist cameras enabled — adding to dataset schema")
            dataset_features.update(get_wrist_camera_features())
            wrist_modality = get_wrist_camera_modality_config()
            for key, value in wrist_modality.items():
                if key in modality_config:
                    modality_config[key].update(value)
                else:
                    modality_config[key] = value

        if config.record_revo2:
            if not config.revo2_devices:
                raise ValueError("revo2_devices is required when Revo2 recording is enabled")
            from gear_sonic.utils.teleop.revo2_trigger import load_revo2_devices

            revo2_metadata = revo2_capture_metadata(
                load_revo2_devices(config.revo2_devices)
            )
            print("[Revo2] Measured motor, command, and tactile fields enabled")
            dataset_features.update(get_revo2_features())
            revo2_modality = get_revo2_modality_config()
            for key, value in revo2_modality.items():
                if key in modality_config:
                    modality_config[key].update(value)
                else:
                    modality_config[key] = value

        text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )

    if not config.oxt_only:
        data_exporter = Gr00tDataExporter.create(
            save_root=f"{config.root_output_dir}/{config.dataset_name}",
            fps=config.data_collection_frequency,
            features=dataset_features,
            modality_config=modality_config,
            task=config.task_prompt,
            script_config={
                **robot_config,
                "record_wrist_cameras": config.record_wrist_cameras,
                "record_revo2": config.record_revo2,
                "revo2_capture": revo2_metadata,
                "revo2_max_age_ms": config.revo2_max_age_ms,
                "require_revo2_tactile": config.require_revo2_tactile,
            },
        )

    oxt_capture = None
    if config.oxt_raw_output_dir:
        from gear_sonic.utils.data_collection.oxt_raw import OxtCaptureRuntime

        oxt_capture = OxtCaptureRuntime(
            config.oxt_raw_output_dir,
            session_id=config.oxt_session_id,
            sonic_zmq_host=config.sonic_zmq_host,
            sonic_zmq_port=config.sonic_zmq_port,
            state_zmq_host=config.state_zmq_host,
            state_zmq_port=config.state_zmq_port,
            queue_size=config.oxt_raw_queue_size,
            metadata={
                "task_prompt_default": config.task_prompt,
                "record_revo2_snapshot_requested": config.record_revo2,
                "oxt_only": config.oxt_only,
            },
        ).start()
        print(f"[OXT] Raw capture session: {oxt_capture.session_root}")

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        record_revo2=config.record_revo2,
        revo2_max_age_ms=config.revo2_max_age_ms,
        require_revo2_tactile=config.require_revo2_tactile,
        oxt_capture=oxt_capture,
        oxt_task_prompt=config.task_prompt,
        oxt_only=config.oxt_only,
    )
    data_collector.run()


if __name__ == "__main__":
    install_shutdown_signal_handlers()
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
