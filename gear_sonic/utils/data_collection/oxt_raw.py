"""Append-only OXT raw capture primitives.

The writer is deliberately independent from LeRobot.  It records immutable
source observations and machine episode/clock events into a session directory while
the caller stays on a non-blocking hot path.  Device owners may feed this
writer directly, or a separate process may use :class:`OxtCaptureRuntime` to
relay raw ZMQ messages and camera samples.

This module does not claim that a transformed SONIC or Revo2 latest snapshot
is raw hardware state.  The manifest records those source-contract limits so
an offline converter cannot silently promote them to an OXT submission.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Mapping
import uuid

import numpy as np


OXT_RAW_SCHEMA_REVISION = "oxt-raw-v1"
OXT_RAW_RECORD_REVISION = "oxt-raw-record-v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_G1_ROBOT_STATE_ACTION_FIELDS = frozenset(
    {
        "control_loop_type",
        "index",
        "ros_timestamp",
        "base_quat",
        "base_ang_vel",
        "body_torso_quat",
        "body_torso_ang_vel",
        "body_q",
        "body_dq",
        "left_hand_q",
        "left_hand_dq",
        "right_hand_q",
        "right_hand_dq",
        "last_action",
        "last_left_hand_action",
        "last_right_hand_action",
        "motor_temperature",
        "base_trans_target",
        "base_quat_target",
        "body_q_target",
        "base_quat_measured",
        "body_q_measured",
        "left_hand_q_measured",
        "right_hand_q_measured",
    }
)
_REVO2_ROBOT_STATE_ACTION_FIELDS = frozenset(
    {
        "command_position_raw",
        "motor_position_raw",
        "motor_speed_raw",
        "motor_current_raw",
        "motor_state_code",
        "motor_ns",
        "motor_valid",
        "motor_coalesced",
        "touch_type",
        "capacitive_raw",
        "pressure_summary_raw",
        "pressure_detail_raw",
        "pressure_detail_ns",
        "pressure_detail_valid",
        "pressure_detail_coalesced",
        "touch_ns",
        "touch_valid",
        "touch_coalesced",
        "publish_ns",
        "cmd_coalesced",
    }
)
_ROBOT_CONFIG_FIELDS = frozenset(
    {
        "control_frequency",
        "planner_frequency",
        "is_using_encoder",
        "policy_fp16",
        "planner_fp16",
    }
)
_PACKED_DTYPE = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bool": np.dtype("?"),
}


def _safe_component(value: str, *, fallback: str = "unnamed") -> str:
    cleaned = _SAFE_COMPONENT.sub("_", str(value).strip()).strip("._")
    return cleaned or fallback


def _jsonable(value: Any) -> Any:
    """Convert scalar NumPy values and small mappings to strict JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _unpack_packed_array_message(raw: bytes, topic: str) -> dict[str, Any]:
    """Decode the GR00T fixed-header array wire format without importing exporter code."""

    topic_bytes = topic.encode("utf-8")
    header_size = 1280
    header_start = len(topic_bytes)
    header_end = header_start + header_size
    if not raw.startswith(topic_bytes) or len(raw) < header_end:
        raise ValueError(f"invalid {topic} packed message")
    header_bytes = raw[header_start:header_end].split(b"\x00", 1)[0]
    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields")
    if not isinstance(fields, list):
        raise ValueError(f"{topic} message header has no fields list")
    offset = header_end
    result: dict[str, Any] = {}
    for field in fields:
        dtype = _PACKED_DTYPE.get(field.get("dtype"))
        shape = field.get("shape")
        if dtype is None or not isinstance(shape, list):
            raise ValueError(f"unsupported {topic} field descriptor: {field!r}")
        dimensions = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in dimensions):
            raise ValueError(f"negative {topic} field shape: {field!r}")
        size = int(np.prod(dimensions, dtype=np.int64))
        end = offset + size * dtype.itemsize
        if end > len(raw):
            raise ValueError(f"truncated {topic} field: {field.get('name')!r}")
        result[str(field["name"])] = np.frombuffer(
            raw[offset:end], dtype=dtype, count=size
        ).reshape(dimensions).copy()
        offset = end
    return result


def _filter_robot_state_action_fields(
    values: Mapping[str, Any], allowed_fields: frozenset[str]
) -> dict[str, Any]:
    return {
        str(key): _jsonable(value)
        for key, value in values.items()
        if key in allowed_fields
    }


def _decode_g1_robot_state_action(raw: bytes) -> tuple[dict[str, Any], int]:
    import msgpack

    prefix = b"g1_debug"
    if not raw.startswith(prefix):
        raise ValueError("invalid g1_debug message prefix")
    values = msgpack.unpackb(raw[len(prefix) :], raw=False, strict_map_key=False)
    if not isinstance(values, Mapping):
        raise ValueError("g1_debug payload is not a map")
    filtered = _filter_robot_state_action_fields(values, _G1_ROBOT_STATE_ACTION_FIELDS)
    source_time = filtered.get("ros_timestamp", 0.0)
    source_timestamp_ns = int(float(source_time) * 1e9) if float(source_time) > 0 else 0
    return filtered, source_timestamp_ns


def _decode_revo2_robot_state_action(raw: bytes) -> tuple[dict[str, Any], int]:
    values = _unpack_packed_array_message(raw, "revo2_state")
    filtered = _filter_robot_state_action_fields(values, _REVO2_ROBOT_STATE_ACTION_FIELDS)
    publish_ns = filtered.get("publish_ns", [0])
    source_timestamp_ns = int(np.asarray(publish_ns).reshape(-1)[0]) if np.size(publish_ns) else 0
    return filtered, source_timestamp_ns


def _decode_robot_config(raw: bytes) -> dict[str, Any]:
    import msgpack

    prefix = b"robot_config"
    if not raw.startswith(prefix):
        raise ValueError("invalid robot_config message prefix")
    values = msgpack.unpackb(raw[len(prefix) :], raw=False, strict_map_key=False)
    if not isinstance(values, Mapping):
        raise ValueError("robot_config payload is not a map")
    return _filter_robot_state_action_fields(values, _ROBOT_CONFIG_FIELDS)


def _utc_wall_time_ns() -> int:
    return time.time_ns()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class OxtRawSessionInfo:
    """Stable paths and identity for one raw capture session."""

    dataset_root: Path
    session_root: Path
    session_id: str


class OxtRawSession:
    """Non-blocking append-only writer for one OXT raw session.

    ``submit_*`` methods only enqueue work and never wait for storage.  A full
    queue is a recorded capture fault and returns ``False``.  ``close`` is the
    only operation that waits for the writer to drain and fsync its files.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        queue_size: int = 1024,
        max_pending_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if queue_size < 8:
            raise ValueError("OXT raw queue_size must be at least 8")
        if max_pending_bytes < 1:
            raise ValueError("max_pending_bytes must be positive")
        dataset_path = Path(dataset_root).expanduser().resolve()
        resolved_session_id = _safe_component(session_id or uuid.uuid4().hex)
        if not resolved_session_id:
            raise ValueError("session_id must not be empty")

        session_root = dataset_path / "raw" / resolved_session_id
        if session_root.exists():
            raise FileExistsError(f"refusing to overwrite OXT raw session: {session_root}")
        (session_root / "streams").mkdir(parents=True, exist_ok=False)
        (session_root / "blobs").mkdir(parents=True, exist_ok=False)

        self.info = OxtRawSessionInfo(dataset_path, session_root, resolved_session_id)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._sequence_by_stream: Counter[str] = Counter()
        self._stream_counts: Counter[str] = Counter()
        self._queue_overflow = 0
        self._byte_overflow = 0
        self._pending_bytes = 0
        self._peak_pending_bytes = 0
        self._max_pending_bytes = int(max_pending_bytes)
        self._writer_errors: list[str] = []
        self._closed = False
        self._close_requested = False
        self._sentinel_queued = False
        self._writer_failed = False
        self._handles: dict[str, Any] = {}
        self._metadata = _jsonable(dict(metadata or {}))
        self._manifest_path = session_root / "manifest.json"
        self._manifest = {
            "schema_revision": OXT_RAW_SCHEMA_REVISION,
            "session_id": resolved_session_id,
            "status": "open",
            "valid": False,
            "created_wall_time_ns": _utc_wall_time_ns(),
            "created_monotonic_ns": time.monotonic_ns(),
            "source_host": platform.node(),
            "metadata": self._metadata,
            "streams": {},
            "queue_size": queue_size,
            "queue_overflow": 0,
            "byte_overflow": 0,
            "max_pending_bytes": self._max_pending_bytes,
            "peak_pending_bytes": 0,
            "writer_errors": [],
            "close": None,
            "file_integrity": None,
            "storage_integrity_valid": False,
            "oxt_submission_ready": False,
            "qualification_status": "not_qualified",
            "qualification_reasons": [
                "raw stream contract and hardware-owner taps are not yet complete"
            ],
        }
        _atomic_write_json(self._manifest_path, self._manifest)

        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"oxt-raw-writer-{resolved_session_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def session_id(self) -> str:
        return self.info.session_id

    @property
    def session_root(self) -> Path:
        return self.info.session_root

    @property
    def queue_overflow(self) -> int:
        with self._lock:
            return self._queue_overflow

    @property
    def writer_errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._writer_errors)

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        """Merge capture facts into the manifest without touching stream data."""

        if not isinstance(values, Mapping):
            raise TypeError("manifest metadata must be a mapping")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot update metadata after the session is closed")
            self._metadata.update(_jsonable(dict(values)))
            self._manifest["metadata"] = self._metadata
            snapshot = dict(self._manifest)
        _atomic_write_json(self._manifest_path, snapshot)

    def submit_stream(
        self,
        stream: str,
        payload: Any,
        *,
        source_timestamp_ns: int = 0,
        received_monotonic_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        sequence: int | None = None,
    ) -> bool:
        """Enqueue one source record without waiting for disk I/O."""

        stream_name = _safe_component(stream)
        with self._lock:
            if self._closed or self._close_requested or self._writer_failed:
                return False
            if sequence is None:
                self._sequence_by_stream[stream_name] += 1
                sequence = self._sequence_by_stream[stream_name]
            else:
                sequence = int(sequence)
                self._sequence_by_stream[stream_name] = max(
                    self._sequence_by_stream[stream_name], sequence
                )

        item = {
            "kind": "stream",
            "stream": stream_name,
            "sequence": sequence,
            "source_timestamp_ns": int(source_timestamp_ns),
            "received_monotonic_ns": int(
                time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
            ),
            "received_wall_time_ns": _utc_wall_time_ns(),
            "payload": payload,
            "metadata": dict(metadata or {}),
        }
        return self._submit(item)

    def append_event(
        self,
        event: str,
        *,
        episode_id: str | None = None,
        episode_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Enqueue an OXT machine episode/session event."""

        item = {
            "kind": "event",
            "event": _safe_component(event),
            "episode_id": episode_id,
            "episode_index": episode_index,
            "received_monotonic_ns": time.monotonic_ns(),
            "received_wall_time_ns": _utc_wall_time_ns(),
            "metadata": dict(metadata or {}),
        }
        return self._submit(item)

    def append_clock(self, *, context: Mapping[str, Any] | None = None) -> bool:
        """Enqueue a host clock anchor for later cross-device alignment."""

        item = {
            "kind": "clock",
            "sequence": None,
            "received_monotonic_ns": time.monotonic_ns(),
            "received_wall_time_ns": _utc_wall_time_ns(),
            "metadata": dict(context or {}),
        }
        return self._submit(item)

    def _submit(self, item: dict[str, Any]) -> bool:
        item_bytes = self._estimate_item_bytes(item)
        with self._lock:
            if self._closed or self._close_requested or self._writer_failed:
                return False
            if self._pending_bytes + item_bytes > self._max_pending_bytes:
                self._queue_overflow += 1
                self._byte_overflow += 1
                return False
            item["_reserved_bytes"] = item_bytes
            self._pending_bytes += item_bytes
            self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_bytes)
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self._pending_bytes -= item_bytes
                self._queue_overflow += 1
                return False

    @staticmethod
    def _estimate_item_bytes(item: Mapping[str, Any]) -> int:
        payload = item.get("payload")
        if isinstance(payload, np.ndarray):
            payload_bytes = int(payload.nbytes)
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            payload_bytes = len(payload)
        else:
            try:
                serialized = json.dumps(
                    _jsonable(item),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError):
                serialized = repr(item).encode("utf-8", errors="replace")
            payload_bytes = len(serialized)
        return max(1, payload_bytes)

    def close(self, *, timeout: float = 10.0) -> Path:
        """Drain and close the session, returning its immutable session path."""

        with self._lock:
            if self._closed:
                return self.session_root
            self._close_requested = True

        deadline = time.monotonic() + max(0.1, float(timeout))
        while not self._sentinel_queued and time.monotonic() < deadline:
            with self._lock:
                try:
                    self._queue.put_nowait(None)
                    self._sentinel_queued = True
                except queue.Full:
                    pass
            if not self._sentinel_queued:
                time.sleep(0.01)

        if self._sentinel_queued:
            self._thread.join(max(0.0, deadline - time.monotonic()))

        if self._thread.is_alive():
            with self._lock:
                if not self._sentinel_queued:
                    self._writer_errors.append("close sentinel could not be queued")
                else:
                    self._writer_errors.append("writer thread did not drain before close timeout")
                self._writer_failed = True
                self._manifest.update(
                    {
                        "status": "closing_incomplete",
                        "queue_overflow": self._queue_overflow,
                        "byte_overflow": self._byte_overflow,
                        "max_pending_bytes": self._max_pending_bytes,
                        "peak_pending_bytes": self._peak_pending_bytes,
                        "writer_errors": list(self._writer_errors),
                        "close": {"drained": False},
                    }
                )
                snapshot = dict(self._manifest)
            _atomic_write_json(self._manifest_path, snapshot)
            return self.session_root

        try:
            file_integrity = {
                path.relative_to(self.session_root).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in sorted(self.session_root.rglob("*"))
                if path.is_file()
                and path != self._manifest_path
                and path.relative_to(self.session_root).parts[0] != "blobs"
            }
        except Exception as exc:
            file_integrity = None
            with self._lock:
                self._writer_errors.append(f"file integrity indexing failed: {exc!r}")
                self._writer_failed = True

        with self._lock:
            errors = list(self._writer_errors)
            overflow = self._queue_overflow
            streams = {
                stream: {"records": count} for stream, count in sorted(self._stream_counts.items())
            }
            storage_integrity_valid = (
                file_integrity is not None
                and not errors
                and overflow == 0
                and not self._writer_failed
            )
            self._manifest.update(
                {
                    "status": "closed",
                    "valid": storage_integrity_valid,
                    "storage_integrity_valid": storage_integrity_valid,
                    "closed_wall_time_ns": _utc_wall_time_ns(),
                    "closed_monotonic_ns": time.monotonic_ns(),
                    "streams": streams,
                    "queue_overflow": overflow,
                    "byte_overflow": self._byte_overflow,
                    "max_pending_bytes": self._max_pending_bytes,
                    "peak_pending_bytes": self._peak_pending_bytes,
                    "writer_errors": errors,
                    "close": {"drained": True},
                    "file_integrity": file_integrity,
                }
            )
            self._closed = True
            snapshot = dict(self._manifest)
        _atomic_write_json(self._manifest_path, snapshot)
        return self.session_root

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._close_writer_handles()
                    return
                self._write_item(item)
            except Exception as exc:
                with self._lock:
                    self._writer_errors.append(repr(exc))
                    self._writer_failed = True
            finally:
                if item is not None:
                    with self._lock:
                        self._pending_bytes = max(
                            0, self._pending_bytes - int(item.get("_reserved_bytes", 0))
                        )
                self._queue.task_done()

    def _close_writer_handles(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except Exception as exc:  # pragma: no cover - defensive close path
                with self._lock:
                    self._writer_errors.append(f"stream close failed: {exc!r}")
                    self._writer_failed = True
        self._handles.clear()

    def _write_item(self, item: Mapping[str, Any]) -> None:
        kind = item["kind"]
        if kind == "stream":
            stream = str(item["stream"])
            record = {
                "record_revision": OXT_RAW_RECORD_REVISION,
                "session_id": self.session_id,
                "stream": stream,
                "sequence": int(item["sequence"]),
                "source_timestamp_ns": int(item["source_timestamp_ns"]),
                "received_monotonic_ns": int(item["received_monotonic_ns"]),
                "received_wall_time_ns": int(item["received_wall_time_ns"]),
                "payload": self._write_payload(
                    stream,
                    int(item["sequence"]),
                    item["payload"],
                ),
                "metadata": _jsonable(item["metadata"]),
            }
            self._append_jsonl(self.session_root / "streams" / f"{stream}.jsonl", record)
            with self._lock:
                self._stream_counts[stream] += 1
            return

        if kind == "event":
            self._append_jsonl(
                self.session_root / "episode_events.jsonl",
                {
                    "session_id": self.session_id,
                    "event": item["event"],
                    "episode_id": item["episode_id"],
                    "episode_index": item["episode_index"],
                    "received_monotonic_ns": int(item["received_monotonic_ns"]),
                    "received_wall_time_ns": int(item["received_wall_time_ns"]),
                    "metadata": _jsonable(item["metadata"]),
                },
            )
            return

        if kind == "clock":
            self._append_jsonl(
                self.session_root / "frame_clock.jsonl",
                {
                    "session_id": self.session_id,
                    "received_monotonic_ns": int(item["received_monotonic_ns"]),
                    "received_wall_time_ns": int(item["received_wall_time_ns"]),
                    "metadata": _jsonable(item["metadata"]),
                },
            )
            return

        raise ValueError(f"unknown OXT raw queue item kind: {kind!r}")

    def _write_payload(self, stream: str, sequence: int, payload: Any) -> dict[str, Any]:
        if isinstance(payload, bytes):
            return self._write_blob(stream, sequence, payload, suffix=".bin", kind="bytes")
        if isinstance(payload, bytearray):
            return self._write_blob(stream, sequence, bytes(payload), suffix=".bin", kind="bytes")
        if isinstance(payload, np.ndarray):
            relative = Path("blobs") / stream / f"{sequence:012d}.npy"
            path = self.session_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, np.ascontiguousarray(payload), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return {
                "kind": "numpy",
                "path": relative.as_posix(),
                "dtype": str(payload.dtype),
                "shape": list(payload.shape),
                "sha256": _sha256_file(path),
            }
        try:
            return {"kind": "json", "value": _jsonable(payload)}
        except TypeError:
            encoded = repr(payload).encode("utf-8", errors="replace")
            return self._write_blob(stream, sequence, encoded, suffix=".repr", kind="repr")

    def _write_blob(
        self,
        stream: str,
        sequence: int,
        payload: bytes,
        *,
        suffix: str,
        kind: str,
    ) -> dict[str, Any]:
        relative = Path("blobs") / stream / f"{sequence:012d}{suffix}"
        path = self.session_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return {
            "kind": kind,
            "path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": _sha256_file(path),
        }

    def _append_jsonl(self, path: Path, record: Mapping[str, Any]) -> None:
        handle = self._handles.get(str(path))
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self._handles[str(path)] = handle
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()


class OxtCaptureRuntime:
    """Optional raw relay for the GR00T runtime.

    The relay subscribes to raw ZMQ payloads without CONFLATE, so it does not
    change the existing LeRobot subscribers.  Native Revo2 batches are accepted
    on the separate ``revo2_raw`` topic when the single device owner is started
    with its explicit raw-capture flag.
    """

    _TOPICS = (
        "g1_debug",
        "robot_config",
        "revo2_state",
        "revo2_raw_drop",
        "revo2_raw",
    )
    _STREAMS = {
        "g1_debug": "g1_robot_state_action",
        "robot_config": "robot_config_message",
        "revo2_state": "revo2_state_snapshot",
        "revo2_raw_drop": "revo2_native_capture_drop",
        "revo2_raw": "revo2_native_sample_batch",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        session_id: str | None = None,
        zmq_host: str = "localhost",
        sonic_zmq_port: int = 5556,
        state_zmq_port: int = 5557,
        sonic_zmq_host: str | None = None,
        state_zmq_host: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        queue_size: int = 2048,
        zmq_hwm: int = 10000,
        start_zmq: bool = True,
    ) -> None:
        raw_taps_missing = ["revo2_native_buffer_samples"]
        source_metadata = {
            "capture_mode": "nonblocking_robot_state_action_camera_relay",
            "lossless_device_samples": False,
            "raw_taps_missing": raw_taps_missing,
            "source_gaps": ["native_g1_lowstate_not_captured_by_this_adapter"],
            "capture_allowlist_revision": "robot-state-action-visual-tactile-v1",
            "excluded_source_data": [
                "PICO/XRT body and hand tracking packets",
                "pose topic",
                "planner fields derived from controller/body tracking",
                "manager_state controller and keyboard inputs",
                "G1 VR 3-point visualization fields",
                "Revo2 source trigger values and trigger sequence",
                "robot_config model/reference/planner/encoder file paths",
            ],
            "stream_contracts": {
                "revo2_state_snapshot": "derived_latest_snapshot",
                "revo2_native_sample_batch": "single_owner_native_buffer_batch_if_observed",
                "g1_robot_state_action": "g1_debug_msgpack_robot_field_allowlist",
                "robot_config_message": "filtered_machine_runtime_config",
            },
            **dict(metadata or {}),
        }
        self.session = OxtRawSession(
            dataset_root,
            session_id=session_id,
            metadata=source_metadata,
            queue_size=queue_size,
        )
        self.sonic_zmq_host = sonic_zmq_host or zmq_host
        self.state_zmq_host = state_zmq_host or zmq_host
        self.sonic_zmq_port = int(sonic_zmq_port)
        self.state_zmq_port = int(state_zmq_port)
        self.zmq_hwm = int(zmq_hwm)
        self.start_zmq = bool(start_zmq)
        self._ctx: Any = None
        self._sockets: list[Any] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._episode_counter = 0
        self._episode_id: str | None = None
        self._camera_sequence: Counter[str] = Counter()
        self._observed_streams: set[str] = set()
        self._revo2_raw_drop_messages = 0
        self._closed = False
        self._started = False

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def session_root(self) -> Path:
        return self.session.session_root

    def start(self) -> "OxtCaptureRuntime":
        if self._started:
            raise RuntimeError("OXT capture runtime cannot be started twice")
        self._started = True
        self.session.append_event("session_start", metadata={"session_id": self.session_id})
        self.record_clock(context={"kind": "session_start"})
        if self.start_zmq:
            import zmq

            self._ctx = zmq.Context()
            self._open_socket(
                self.sonic_zmq_host,
                self.sonic_zmq_port,
                (
                    "revo2_raw_drop",
                    "revo2_raw",
                    "revo2_state",
                ),
            )
            self._open_socket(
                self.state_zmq_host,
                self.state_zmq_port,
                ("g1_debug", "robot_config"),
            )
            self._thread = threading.Thread(
                target=self._zmq_loop,
                name=f"oxt-zmq-relay-{self.session_id}",
                daemon=True,
            )
            self._thread.start()
        return self

    def begin_episode(self, *, episode_index: int, task: str) -> str:
        if not self._started:
            raise RuntimeError("OXT capture runtime has not been started")
        if self._closed:
            raise RuntimeError("OXT capture runtime is closed")
        if self._episode_id is not None:
            raise RuntimeError("an OXT episode is already active")
        self._episode_counter += 1
        self._episode_id = f"episode-{self._episode_counter:06d}"
        self.session.append_event(
            "episode_start",
            episode_id=self._episode_id,
            episode_index=int(episode_index),
            metadata={"task": str(task)},
        )
        return self._episode_id

    def finish_episode(
        self,
        event: str,
        *,
        episode_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._episode_id is None:
            return
        terminal_events = {"episode_complete", "episode_discard", "episode_abort"}
        if event not in terminal_events:
            raise ValueError(f"unsupported OXT episode terminal event: {event!r}")
        self.session.append_event(
            event,
            episode_id=self._episode_id,
            episode_index=episode_index,
            metadata=metadata,
        )
        self._episode_id = None

    def capture_camera_message(
        self,
        message: Mapping[str, Any],
        *,
        received_monotonic_ns: int | None = None,
    ) -> None:
        if self._episode_id is None:
            return
        images = message.get("images", {})
        timestamps = message.get("timestamps", {})
        if not isinstance(images, Mapping):
            return
        for name, image in images.items():
            if not isinstance(image, np.ndarray):
                continue
            camera_name = _safe_component(str(name))
            self._camera_sequence[camera_name] += 1
            source_timestamp_ns = 0
            if isinstance(timestamps, Mapping) and name in timestamps:
                source_timestamp_ns = int(float(timestamps[name]) * 1e9)
            self.session.submit_stream(
                f"camera_{camera_name}_rgb",
                np.array(image, copy=True),
                source_timestamp_ns=source_timestamp_ns,
                received_monotonic_ns=received_monotonic_ns,
                metadata={
                    "camera_name": str(name),
                    "source_clock": "unix_seconds" if source_timestamp_ns else "unknown",
                    "dtype": str(image.dtype),
                    "shape": list(image.shape),
                },
                sequence=self._camera_sequence[camera_name],
            )

    def record_clock(self, *, context: Mapping[str, Any] | None = None) -> None:
        self.session.append_clock(context=context)

    def close(self, *, timeout: float = 10.0) -> Path:
        if self._closed:
            return self.session_root
        self._stop.set()
        if self._thread is not None:
            self._thread.join(max(0.1, float(timeout)))
            if self._thread.is_alive():
                for socket in self._sockets:
                    try:
                        socket.close(linger=0)
                    except Exception:
                        pass
                self._thread.join(1.0)
                if self._thread.is_alive():
                    self.session.append_event("capture_shutdown_timeout")
        for socket in self._sockets:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        self._sockets.clear()
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        observed_streams = sorted(self._observed_streams)
        revo2_observed = "revo2_native_sample_batch" in self._observed_streams
        raw_taps_missing: list[str] = []
        if not revo2_observed or self._revo2_raw_drop_messages:
            raw_taps_missing.append("revo2_native_buffer_samples")
        self.session.update_metadata(
            {
                "observed_streams": observed_streams,
                "revo2_raw_drop_messages": self._revo2_raw_drop_messages,
                "revo2_native_tap_observed": revo2_observed,
                "raw_taps_missing": raw_taps_missing,
            }
        )
        if self._episode_id is not None:
            self.finish_episode(
                "episode_abort",
                metadata={"reason": "session_closed_without_terminal_event"},
            )
        self.record_clock(context={"kind": "session_close"})
        self.session.append_event("session_close")
        path = self.session.close(timeout=timeout)
        self._closed = True
        return path

    def _open_socket(self, host: str, port: int, topics: tuple[str, ...]) -> None:
        import zmq

        socket = self._ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, self.zmq_hwm)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        for topic in topics:
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        socket.connect(f"tcp://{host}:{port}")
        self._sockets.append(socket)

    def _zmq_loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        for socket in self._sockets:
            poller.register(socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(100))
            except zmq.ZMQError:
                return
            for socket in self._sockets:
                if not events.get(socket, 0) & zmq.POLLIN:
                    continue
                try:
                    raw = socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                topic = self._topic_for(raw)
                if topic is None:
                    continue
                if topic != "robot_config" and self._episode_id is None:
                    continue
                stream = self._STREAMS[topic]
                self._observed_streams.add(stream)
                if topic == "revo2_raw_drop":
                    self._revo2_raw_drop_messages += 1
                if topic == "g1_debug":
                    try:
                        payload, source_timestamp_ns = _decode_g1_robot_state_action(raw)
                    except Exception as exc:
                        self.session.append_event(
                            "source_decode_error",
                            metadata={"stream": stream, "error": repr(exc)},
                        )
                        continue
                    self.session.submit_stream(
                        stream,
                        payload,
                        source_timestamp_ns=source_timestamp_ns,
                        metadata={
                            "topic": topic,
                            "source": "allowlisted_robot_state_and_action_fields",
                            "excluded_g1_visual_fields": [
                                "vr_3point_position",
                                "vr_3point_orientation",
                                "vr_3point_compliance",
                                "delta_heading",
                            ],
                        },
                    )
                    continue
                if topic == "revo2_state":
                    try:
                        payload, source_timestamp_ns = _decode_revo2_robot_state_action(raw)
                    except Exception as exc:
                        self.session.append_event(
                            "source_decode_error",
                            metadata={"stream": stream, "error": repr(exc)},
                        )
                        continue
                    self.session.submit_stream(
                        stream,
                        payload,
                        source_timestamp_ns=source_timestamp_ns,
                        metadata={
                            "topic": topic,
                            "source": "allowlisted_robot_state_and_action_fields",
                            "excluded_fields": [
                                "source_trigger",
                                "source_received_ns",
                                "sequence",
                            ],
                        },
                    )
                    continue
                if topic == "robot_config":
                    try:
                        payload = _decode_robot_config(raw)
                    except Exception as exc:
                        self.session.append_event(
                            "source_decode_error",
                            metadata={"stream": stream, "error": repr(exc)},
                        )
                        continue
                    self.session.submit_stream(
                        stream,
                        payload,
                        metadata={
                            "topic": topic,
                            "source": "allowlisted_machine_runtime_config",
                            "excluded_fields": [
                                "model_path",
                                "reference_motion_path",
                                "planner_path",
                                "obs_config_path",
                                "encoder_file",
                            ],
                        },
                    )
                    continue
                self.session.submit_stream(
                    stream,
                    raw,
                    metadata={
                        "topic": topic,
                        "source": "zmq_raw_payload",
                        "payload_encoding": "topic_prefix_plus_payload",
                    },
                )

    @classmethod
    def _topic_for(cls, raw: bytes) -> str | None:
        for topic in sorted(cls._TOPICS, key=len, reverse=True):
            if raw.startswith(topic.encode("utf-8")):
                return topic
        return None


__all__ = ["OXT_RAW_SCHEMA_REVISION", "OxtCaptureRuntime", "OxtRawSession"]
