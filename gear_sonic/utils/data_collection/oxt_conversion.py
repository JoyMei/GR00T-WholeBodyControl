"""Offline conversion of an audited OXT raw session to FTP-1 Zarr staging."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.scripts.validate_oxt_raw import validate_session


MAPPING_REVISION = "oxt-zarr-mapping-v1"
_TIME_UNITS_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
_TACTILE_KEY = re.compile(
    r"^(?P<side>.+)_tactile_(?P<member>data|area|sensor|type)_(?P<group>.+)$"
)


@dataclass(frozen=True)
class _Sample:
    timestamp_ns: int
    source_index: int
    value: np.ndarray | None
    array_path: Path | None = None
    array_sha256: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def _path_value(value: Any, path: Sequence[str | int], *, label: str) -> Any:
    current = value
    for part in path:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{label}: path {list(path)!r} is missing at {part!r}") from exc
    return current


def _decode_payload(
    record: Mapping[str, Any], session_root: Path, field: Mapping[str, Any]
) -> Any:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("raw stream record has no payload descriptor")
    kind = payload.get("kind")
    if kind == "json":
        return payload.get("value")
    if kind == "numpy":
        path = _safe_blob_path(session_root, payload)
        return np.load(path, allow_pickle=False)
    if kind == "bytes":
        path = _safe_blob_path(session_root, payload)
        encoding = field.get("payload_encoding")
        if encoding != "json_topic_prefix":
            raise ValueError(
                "binary raw payload needs payload_encoding='json_topic_prefix' "
                "in the mapping"
            )
        prefix = field.get("topic_prefix")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("json_topic_prefix payload requires a non-empty topic_prefix")
        raw = path.read_bytes()
        prefix_bytes = prefix.encode("utf-8")
        if not raw.startswith(prefix_bytes):
            raise ValueError(f"binary payload does not start with topic {prefix!r}")
        try:
            return json.loads(raw[len(prefix_bytes) :].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"binary payload is not topic-prefixed JSON: {exc}") from exc
    raise ValueError(f"unsupported raw payload kind: {kind!r}")


def _safe_blob_path(
    session_root: Path, payload: Mapping[str, Any], *, verify_hash: bool = True
) -> Path:
    relative = payload.get("path")
    if not isinstance(relative, str):
        raise ValueError("blob payload has no relative path")
    path = (session_root / relative).resolve()
    try:
        path.relative_to(session_root)
    except ValueError as exc:
        raise ValueError(f"raw payload path escapes session: {relative!r}") from exc
    if not path.is_file():
        raise ValueError(f"raw payload blob is missing: {relative!r}")
    expected = payload.get("sha256")
    if verify_hash and expected and _sha256(path) != expected:
        raise ValueError(f"raw payload hash changed after validation: {relative!r}")
    return path


def _source_timestamp_ns(
    record: Mapping[str, Any],
    value: Any,
    item: Any,
    field: Mapping[str, Any],
    *,
    is_item: bool,
) -> int:
    sample_time = field.get("sample_time")
    if sample_time is None:
        if is_item:
            raise ValueError(
                "batched samples require sample_time mapping; one batch receive time "
                "cannot timestamp every native sample"
            )
        return int(record["received_monotonic_ns"])

    if not isinstance(sample_time, Mapping):
        raise ValueError("sample_time must be an object")
    path = sample_time.get("path")
    if not isinstance(path, list) or not path:
        raise ValueError("sample_time.path must be a non-empty JSON path")
    source = item if is_item else value
    raw_time = _path_value(source, path, label="sample_time")
    if isinstance(raw_time, Mapping) and "value" in raw_time:
        raw_time = raw_time["value"]
    try:
        numeric_time = float(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_time is not numeric: {raw_time!r}") from exc
    if not math.isfinite(numeric_time):
        raise ValueError("sample_time must be finite")

    unit = sample_time.get("unit")
    if unit not in _TIME_UNITS_NS:
        raise ValueError(f"sample_time.unit must be one of {sorted(_TIME_UNITS_NS)}")
    timestamp_ns = numeric_time * _TIME_UNITS_NS[unit]
    clock_domain = sample_time.get("clock_domain")
    if clock_domain == "host_monotonic":
        scale = sample_time.get("scale", 1.0)
        offset = sample_time.get("offset_ns", 0.0)
    else:
        if not isinstance(clock_domain, str) or not clock_domain:
            raise ValueError("sample_time.clock_domain must identify the source clock")
        if "scale" not in sample_time or "offset_ns" not in sample_time:
            raise ValueError(
                "non-host sample clocks require an explicit scale and offset_ns "
                "mapping into host monotonic time"
            )
        scale = sample_time["scale"]
        offset = sample_time["offset_ns"]
    try:
        aligned_ns = timestamp_ns * float(scale) + float(offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_time scale/offset must be numeric") from exc
    if not math.isfinite(aligned_ns):
        raise ValueError("mapped sample_time must be finite")
    return int(round(aligned_ns))


def _field_value(
    value: Any, item: Any, field: Mapping[str, Any], *, is_item: bool
) -> Any:
    path = field.get("value_path", [])
    if not isinstance(path, list):
        raise ValueError("value_path must be a JSON path list")
    source = item if is_item else value
    if not path:
        selected = source
    else:
        selected = _path_value(source, path, label="value")
    select_indices = field.get("select_indices")
    if select_indices is not None:
        if not isinstance(select_indices, list) or not select_indices:
            raise ValueError("select_indices must be a non-empty list")
        try:
            selected = np.asarray(selected)[select_indices]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("select_indices cannot be applied to the source value") from exc
    return selected


def _array_value(value: Any, field: Mapping[str, Any]) -> np.ndarray:
    if field.get("kind") == "rgb":
        result = np.asarray(value)
        expected_shape = field.get("shape")
        if expected_shape is not None and list(result.shape) != expected_shape:
            raise ValueError(
                f"sample shape {list(result.shape)} does not match configured shape "
                f"{expected_shape}"
            )
        if result.dtype != np.uint8:
            raise ValueError(f"RGB must use uint8, got {result.dtype}")
        return result
    try:
        result = np.asarray(value, dtype=np.dtype(field["dtype"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot convert source value using dtype {field.get('dtype')!r}") from exc
    if result.dtype.kind == "O":
        raise ValueError("object arrays are not allowed in OXT Zarr output")
    reshape = field.get("reshape")
    if reshape is not None:
        if not isinstance(reshape, list) or not reshape:
            raise ValueError("reshape must be a non-empty list")
        try:
            result = result.reshape(tuple(int(dimension) for dimension in reshape))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sample cannot be reshaped to {reshape!r}") from exc
    expected_shape = field.get("shape")
    if not isinstance(expected_shape, list):
        raise ValueError("sample fields require an explicit shape list")
    if list(result.shape) != expected_shape:
        raise ValueError(
            f"sample shape {list(result.shape)} does not match configured shape {expected_shape}"
        )
    if result.dtype.kind in "fiu" and not np.isfinite(result).all():
        raise ValueError("sample contains non-finite numeric values")
    return result


def _load_field_samples(
    session_root: Path, field: Mapping[str, Any]
) -> list[_Sample]:
    stream = field.get("stream")
    if not isinstance(stream, str) or not stream:
        raise ValueError("sample field requires a stream name")
    records = _read_jsonl(session_root / "streams" / f"{stream}.jsonl")
    samples: list[_Sample] = []
    source_index = 0
    for record in records:
        payload = record.get("payload", {})
        lazy_array_path: Path | None = None
        lazy_array_hash: str | None = None
        if (
            field.get("kind") == "rgb"
            and isinstance(payload, Mapping)
            and payload.get("kind") == "numpy"
            and not field.get("value_path")
            and not field.get("items_path")
        ):
            lazy_array_path = _safe_blob_path(session_root, payload, verify_hash=False)
            lazy_array_hash = str(payload.get("sha256", "")) or None
            value = None
        else:
            value = _decode_payload(record, session_root, field)
        items_path = field.get("items_path")
        if items_path is None:
            items = [value]
            is_item = False
        else:
            if not isinstance(items_path, list) or not items_path:
                raise ValueError("items_path must be a non-empty JSON path list")
            items = _path_value(value, items_path, label="items")
            if not isinstance(items, list):
                raise ValueError("items_path must resolve to a JSON list")
            is_item = True
        for item in items:
            timestamp_ns = _source_timestamp_ns(
                record, value, item, field, is_item=is_item
            )
            if lazy_array_path is not None:
                samples.append(
                    _Sample(
                        timestamp_ns=timestamp_ns,
                        source_index=source_index,
                        value=None,
                        array_path=lazy_array_path,
                        array_sha256=lazy_array_hash,
                    )
                )
            else:
                raw_value = _field_value(value, item, field, is_item=is_item)
                samples.append(
                    _Sample(
                        timestamp_ns=timestamp_ns,
                        source_index=source_index,
                        value=_array_value(raw_value, field),
                    )
                )
            source_index += 1
    if not samples:
        raise ValueError(f"source stream {stream!r} has no samples")
    if any(
        samples[index].timestamp_ns > samples[index + 1].timestamp_ns
        for index in range(len(samples) - 1)
    ):
        raise ValueError(f"timestamps in stream {stream!r} are not monotonic")
    return samples


def _episode_events(session_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = _read_jsonl(session_root / "episode_events.jsonl")
    complete: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    terminal_events = {"episode_complete", "episode_discard", "episode_abort"}
    for event in events:
        name = event.get("event")
        if name == "episode_start":
            if active is not None:
                raise ValueError("episode events overlap")
            active = event
            continue
        if name not in terminal_events:
            continue
        if active is None or event.get("episode_id") != active.get("episode_id"):
            raise ValueError(f"terminal event {name!r} has no matching start")
        episode = {
            "episode_id": active.get("episode_id"),
            "episode_index": active.get("episode_index"),
            "task": active.get("metadata", {}).get("task"),
            "start_monotonic_ns": int(active["received_monotonic_ns"]),
            "end_monotonic_ns": int(event["received_monotonic_ns"]),
            "terminal_event": name,
        }
        if episode["end_monotonic_ns"] <= episode["start_monotonic_ns"]:
            raise ValueError(f"episode {episode['episode_id']!r} has non-positive duration")
        if name == "episode_complete":
            if not isinstance(episode["task"], str) or not episode["task"].strip():
                raise ValueError(f"episode {episode['episode_id']!r} has no fixed task prompt")
            complete.append(episode)
        else:
            skipped.append(episode)
        active = None
    if active is not None:
        raise ValueError(f"episode {active.get('episode_id')!r} has no terminal event")
    if not complete:
        raise ValueError("session has no completed episodes to convert")
    return complete, skipped


def _validate_mapping(mapping: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    if mapping.get("schema_revision") != MAPPING_REVISION:
        raise ValueError(f"mapping schema_revision must be {MAPPING_REVISION!r}")
    try:
        rate_hz = float(mapping["canonical_rate_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mapping needs a numeric canonical_rate_hz") from exc
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("canonical_rate_hz must be finite and positive")
    fields = mapping.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("mapping fields must be a non-empty list")

    keys: set[str] = {"timestamps", "sub_task_instruction"}
    rgb_count = 0
    tactile: dict[tuple[str, str], set[str]] = {}
    for field in fields:
        if not isinstance(field, Mapping):
            raise ValueError("each mapping field must be an object")
        key = field.get("output_key")
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError(f"invalid output_key: {key!r}")
        if key in keys:
            raise ValueError(f"duplicate/reserved output key: {key!r}")
        keys.add(key)
        mode = field.get("mode", "sample")
        if mode == "constant":
            _array_value(field.get("value"), field)
        elif mode == "sample":
            if field.get("alignment", "nearest") != "nearest":
                raise ValueError("only explicit nearest-sample alignment is implemented")
            try:
                max_age_ms = float(field["max_age_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{key}: max_age_ms is required") from exc
            if not math.isfinite(max_age_ms) or max_age_ms <= 0:
                raise ValueError(f"{key}: max_age_ms must be positive and finite")
            if field.get("kind") == "rgb":
                rgb_count += 1
                if not key.endswith("_rgb"):
                    raise ValueError("RGB output keys must end in '_rgb'")
            elif not field.get("dtype"):
                raise ValueError(f"{key}: sample array requires dtype")
        else:
            raise ValueError(f"{key}: unsupported mapping mode {mode!r}")

        match = _TACTILE_KEY.match(key)
        if match:
            group_id = (match.group("side"), match.group("group"))
            tactile.setdefault(group_id, set()).add(match.group("member"))
    if rgb_count == 0:
        raise ValueError("at least one kind='rgb' field is required")
    for group_id, members in tactile.items():
        if members != {"data", "area", "sensor", "type"}:
            raise ValueError(
                f"incomplete tactile group {group_id}: has {sorted(members)}, "
                "requires data/area/sensor/type"
            )
    metadata = mapping.get("submission_metadata")
    if not isinstance(metadata, Mapping) or not metadata.get("name"):
        raise ValueError("mapping requires submission_metadata.name")
    return rate_hz, fields


def _align_field(
    field: Mapping[str, Any],
    samples: list[_Sample],
    episode: Mapping[str, Any],
    target_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_samples = [
        sample
        for sample in samples
        if episode["start_monotonic_ns"]
        <= sample.timestamp_ns
        <= episode["end_monotonic_ns"]
    ]
    if not selected_samples:
        raise ValueError(f"{field['output_key']}: no samples in episode interval")
    source_times = np.asarray([sample.timestamp_ns for sample in selected_samples], dtype=np.int64)
    right = np.searchsorted(source_times, target_ns, side="left")
    left = np.clip(right - 1, 0, len(source_times) - 1)
    right = np.clip(right, 0, len(source_times) - 1)
    choose_right = np.abs(source_times[right] - target_ns) < np.abs(
        source_times[left] - target_ns
    )
    indices = np.where(choose_right, right, left)
    age_ns = np.abs(source_times[indices] - target_ns)
    max_age_ns = int(float(field["max_age_ms"]) * 1_000_000)
    if np.any(age_ns > max_age_ns):
        worst_ms = float(age_ns.max()) / 1_000_000
        raise ValueError(
            f"{field['output_key']}: nearest sample age {worst_ms:.3f} ms exceeds "
            f"max_age_ms={max_age_ns / 1_000_000:.3f}"
        )
    selected = [selected_samples[int(index)] for index in indices]
    materialized: dict[Path, np.ndarray] = {}
    selected_values: list[np.ndarray] = []
    for sample in selected:
        if sample.array_path is not None:
            image = materialized.get(sample.array_path)
            if image is None:
                if sample.array_sha256 and _sha256(sample.array_path) != sample.array_sha256:
                    raise ValueError(f"raw RGB blob changed after validation: {sample.array_path}")
                image = np.load(sample.array_path, allow_pickle=False)
                materialized[sample.array_path] = image
            selected_values.append(_array_value(image, field))
        elif sample.value is not None:
            selected_values.append(sample.value)
        else:  # pragma: no cover - defensive invariant
            raise ValueError(f"{field['output_key']}: sample has no stored value")
    values = np.stack(selected_values, axis=0)
    if field.get("kind") == "rgb":
        if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError(
                f"{field['output_key']}: RGB must have dtype uint8 and shape (T,H,W,3), "
                f"got {values.dtype} {values.shape}"
            )
        if not field["output_key"].endswith("_rgb"):
            raise ValueError("RGB output keys must end in '_rgb' for OXT-QC recognition")
    return (
        values,
        np.asarray([sample.source_index for sample in selected], dtype=np.int64),
        age_ns.astype(np.int64),
    )


def _load_replay_buffer(replay_buffer_root: Path) -> Any:
    module_path = replay_buffer_root.expanduser().resolve() / "common" / "replay_buffer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"OXT-QC ReplayBuffer not found: {module_path}")
    spec = importlib.util.spec_from_file_location("oxt_qc_replay_buffer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OXT-QC ReplayBuffer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReplayBuffer


def convert_session(
    session_root: str | Path,
    mapping: Mapping[str, Any],
    output_candidate: str | Path,
    replay_buffer_root: str | Path,
    *,
    replay_buffer_class: Any | None = None,
) -> Path:
    """Convert completed machine episodes; the raw session is read-only."""

    source = Path(session_root).expanduser().resolve()
    output = Path(output_candidate).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output}")
    storage_valid, errors = validate_session(source)
    if not storage_valid:
        raise ValueError("raw session validation failed: " + "; ".join(errors))

    rate_hz, fields = _validate_mapping(mapping)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes, skipped_episodes = _episode_events(source)
    loaded_samples: dict[str, list[_Sample]] = {}
    for field in fields:
        if field.get("mode", "sample") == "sample":
            key = str(field["output_key"])
            loaded_samples[key] = _load_field_samples(source, field)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent))
    )
    try:
        zarr_path = temporary / "dataset.zarr"
        buffer_cls = replay_buffer_class or _load_replay_buffer(Path(replay_buffer_root))
        replay_buffer = buffer_cls.create_from_path(str(zarr_path), mode="w")
        alignment_dir = temporary / "alignment"
        alignment_dir.mkdir()
        episode_ends: list[int] = []
        frame_count = 0
        aligned_summaries: list[dict[str, Any]] = []

        for output_index, episode in enumerate(episodes):
            duration_ns = (
                episode["end_monotonic_ns"] - episode["start_monotonic_ns"]
            )
            frame_count_episode = int(math.ceil(duration_ns / 1e9 * rate_hz))
            if frame_count_episode < 1:
                raise ValueError(f"episode {episode['episode_id']!r} is shorter than one frame")
            offsets_ns = np.rint(
                np.arange(frame_count_episode, dtype=np.float64) * (1e9 / rate_hz)
            ).astype(np.int64)
            target_ns = episode["start_monotonic_ns"] + offsets_ns
            arrays: dict[str, np.ndarray] = {
                "timestamps": (offsets_ns.astype(np.float64) / 1e9),
                "sub_task_instruction": np.full(
                    (frame_count_episode,), episode["task"], dtype=f"<U{max(1, len(episode['task']))}"
                ),
            }
            alignment: dict[str, np.ndarray] = {}
            field_summary: dict[str, Any] = {}
            for field in fields:
                key = str(field["output_key"])
                if field.get("mode", "sample") == "constant":
                    value = _array_value(field["value"], field)
                    arrays[key] = np.repeat(value[np.newaxis, ...], frame_count_episode, axis=0)
                    alignment[f"{key}__source_index"] = np.full(
                        frame_count_episode, -1, dtype=np.int64
                    )
                    alignment[f"{key}__age_ns"] = np.zeros(frame_count_episode, dtype=np.int64)
                    field_summary[key] = {"mode": "constant", "age_ms": 0.0}
                    continue
                values, source_indices, age_ns = _align_field(
                    field, loaded_samples[key], episode, target_ns
                )
                arrays[key] = values
                alignment[f"{key}__source_index"] = source_indices
                alignment[f"{key}__age_ns"] = age_ns
                field_summary[key] = {
                    "mode": "nearest",
                    "source_stream": field["stream"],
                    "max_age_ms": float(field["max_age_ms"]),
                    "observed_max_age_ms": float(age_ns.max(initial=0)) / 1_000_000,
                    "observed_mean_age_ms": float(age_ns.mean()) / 1_000_000,
                }

            replay_buffer.add_episode(arrays, compressors="disk")
            np.savez_compressed(alignment_dir / f"episode-{output_index:06d}.npz", **alignment)
            frame_count += frame_count_episode
            episode_ends.append(frame_count)
            aligned_summaries.append(
                {
                    "output_episode_index": output_index,
                    "source_episode_id": episode["episode_id"],
                    "source_episode_index": episode["episode_index"],
                    "task": episode["task"],
                    "frames": frame_count_episode,
                    "terminal_event": episode["terminal_event"],
                    "fields": field_summary,
                }
            )

        metadata = mapping["submission_metadata"]
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        source_qualified = (
            manifest.get("oxt_submission_ready") is True
            and manifest.get("qualification_status") == "qualified"
        )
        report = {
            "conversion_revision": MAPPING_REVISION,
            "source_session": str(source),
            "source_session_id": manifest.get("session_id"),
            "source_manifest_sha256": _sha256(manifest_path),
            "mapping_sha256": hashlib.sha256(
                json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "source_storage_valid": True,
            "source_qualified_for_oxt": source_qualified,
            "source_qualification_reasons": manifest.get("qualification_reasons", []),
            "release_ready": False,
            "release_blockers": [
                "OXT-QC, visual review, and FTP-1 loader gates are separate release steps",
                *([] if source_qualified else ["raw source contract is not qualified"]),
            ],
            "alignment_policy": "nearest sample with per-field max_age_ms; no interpolation",
            "canonical_rate_hz": rate_hz,
            "episodes": aligned_summaries,
            "skipped_episodes": skipped_episodes,
            "episode_ends": episode_ends,
            "total_frames": frame_count,
            "zarr_path": "dataset.zarr",
            "metadata_path": "metadata.json",
            "alignment_path": "alignment/",
        }
        (temporary / "processing_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["MAPPING_REVISION", "convert_session"]
