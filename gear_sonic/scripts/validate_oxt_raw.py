"""Validate the storage integrity of one OXT raw capture session.

This is deliberately not an OXT-QC Zarr validator.  It checks that the raw
session was closed, drained, hashed, and free of writer overflow/errors.  The
session manifest still controls whether the source contract is qualified for
offline OXT conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_session(session_root: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    session_root = session_root.expanduser().resolve()
    if not session_root.is_dir():
        return False, [f"missing session directory: {session_root}"]
    manifest_path = session_root / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid manifest JSON: {exc}"]

    if manifest.get("session_id") != session_root.name:
        errors.append(
            f"manifest session_id={manifest.get('session_id')!r} does not match "
            f"directory name {session_root.name!r}"
        )

    if manifest.get("status") != "closed":
        errors.append(f"session status is {manifest.get('status')!r}, expected 'closed'")
    if not manifest.get("close", {}).get("drained", False):
        errors.append("writer did not report a drained close")
    if not manifest.get("storage_integrity_valid", False):
        errors.append("storage_integrity_valid is false")
    if int(manifest.get("queue_overflow", 0)) != 0:
        errors.append(f"queue_overflow={manifest.get('queue_overflow')}")
    if int(manifest.get("byte_overflow", 0)) != 0:
        errors.append(f"byte_overflow={manifest.get('byte_overflow')}")
    if int(manifest.get("peak_pending_bytes", 0)) > int(
        manifest.get("max_pending_bytes", 0)
    ):
        errors.append("peak_pending_bytes exceeds max_pending_bytes")
    if manifest.get("writer_errors"):
        errors.append(f"writer_errors={manifest['writer_errors']!r}")

    file_integrity = manifest.get("file_integrity")
    if not isinstance(file_integrity, dict):
        errors.append("missing file_integrity index; raw session must be recaptured with the current writer")
        file_integrity = {}
    actual_control_files: set[str] = set()
    for path in sorted(session_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(session_root)
        if relative.parts and relative.parts[0] == "blobs":
            continue
        if path.is_symlink():
            errors.append(f"control file must not be a symlink: {relative.as_posix()}")
            continue
        actual_control_files.add(relative.as_posix())

    if set(file_integrity) != actual_control_files:
        errors.append(
            "control file set differs: "
            f"manifest={sorted(file_integrity)!r}, files={sorted(actual_control_files)!r}"
        )
    for relative_text, expected in file_integrity.items():
        try:
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts or relative_text == "manifest.json":
                raise ValueError("path is not a session-relative control file")
            if not isinstance(expected, dict):
                raise ValueError("integrity entry is not an object")
            path = (session_root / relative).resolve()
            path.relative_to(session_root)
            if not path.is_file():
                errors.append(f"indexed control file is missing: {relative_text}")
                continue
            if path.stat().st_size != int(expected["bytes"]):
                errors.append(f"byte count mismatch for control file {relative_text}")
            if _sha256(path) != expected["sha256"]:
                errors.append(f"sha256 mismatch for control file {relative_text}")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"invalid file_integrity entry {relative_text!r}: {exc}")

    stream_counts: dict[str, int] = {}
    referenced_blob_paths: set[str] = set()
    streams_dir = session_root / "streams"
    if not streams_dir.is_dir():
        errors.append(f"missing streams directory: {streams_dir}")
    for stream_path in sorted(streams_dir.glob("*.jsonl")):
        count = 0
        previous_sequence: int | None = None
        for line_number, line in enumerate(stream_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
                payload = record["payload"]
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append(f"{stream_path.name}:{line_number}: invalid record: {exc}")
                continue
            if record.get("session_id") != manifest.get("session_id"):
                errors.append(f"{stream_path.name}:{line_number}: session_id mismatch")
            if record.get("stream") != stream_path.stem:
                errors.append(f"{stream_path.name}:{line_number}: stream name mismatch")
            try:
                sequence = int(record["sequence"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{stream_path.name}:{line_number}: invalid sequence: {exc}")
                sequence = None
            if sequence is not None:
                if previous_sequence is not None and sequence <= previous_sequence:
                    errors.append(
                        f"{stream_path.name}:{line_number}: sequence is not increasing "
                        f"({previous_sequence} -> {sequence})"
                    )
                previous_sequence = sequence
            count += 1
            if not isinstance(payload, dict):
                errors.append(f"{stream_path.name}:{line_number}: payload is not an object")
                continue
            if payload.get("kind") in {"bytes", "numpy", "repr"}:
                try:
                    relative = Path(payload["path"])
                    if not relative.parts or relative.parts[0] != "blobs":
                        raise ValueError("blob path must be under blobs/")
                    blob = (session_root / relative).resolve()
                    blob.relative_to(session_root)
                except (KeyError, ValueError) as exc:
                    errors.append(f"{stream_path.name}:{line_number}: invalid blob path: {exc}")
                    continue
                if not blob.is_file():
                    errors.append(f"{stream_path.name}:{line_number}: missing blob {blob}")
                elif _sha256(blob) != payload.get("sha256"):
                    errors.append(f"{stream_path.name}:{line_number}: sha256 mismatch for {blob}")
                else:
                    referenced_blob_paths.add(blob.relative_to(session_root).as_posix())
        stream_counts[stream_path.stem] = count

    actual_blob_paths = {
        path.relative_to(session_root).as_posix()
        for path in (session_root / "blobs").rglob("*")
        if path.is_file()
    } if (session_root / "blobs").is_dir() else set()
    if actual_blob_paths != referenced_blob_paths:
        errors.append(
            "blob file set differs: "
            f"referenced={sorted(referenced_blob_paths)!r}, files={sorted(actual_blob_paths)!r}"
        )

    expected_counts = {
        name: int(info.get("records", -1))
        for name, info in manifest.get("streams", {}).items()
    }
    if stream_counts != expected_counts:
        errors.append(f"stream counts differ: manifest={expected_counts!r}, files={stream_counts!r}")

    events_path = session_root / "episode_events.jsonl"
    if not events_path.is_file():
        errors.append("missing episode_events.jsonl")
    else:
        events = []
        for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"episode_events.jsonl:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(event, dict):
                errors.append(f"episode_events.jsonl:{line_number}: event is not an object")
                continue
            events.append(event)
        names = [event.get("event") for event in events]
        if not names or names[0] != "session_start":
            errors.append("episode events do not start with session_start")
        if not names or names[-1] != "session_close":
            errors.append("episode events do not end with session_close")
        active_episode: str | None = None
        terminal_events = {"episode_complete", "episode_discard", "episode_abort"}
        for event in events:
            name = event.get("event")
            episode_id = event.get("episode_id")
            if name == "episode_start":
                if active_episode is not None:
                    errors.append("episode_start occurred while another episode was active")
                if not isinstance(episode_id, str) or not episode_id:
                    errors.append("episode_start is missing episode_id")
                active_episode = episode_id
            elif name in terminal_events:
                if active_episode is None:
                    errors.append(f"{name} occurred without an active episode")
                elif episode_id != active_episode:
                    errors.append(f"{name} episode_id does not match episode_start")
                active_episode = None
        if active_episode is not None:
            errors.append(f"episode {active_episode!r} has no terminal event")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="path to raw/<session-id>")
    args = parser.parse_args()
    ok, errors = validate_session(args.session.expanduser().resolve())
    session_root = args.session.expanduser().resolve()
    manifest_path = session_root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    print(
        json.dumps(
            {
                "storage_valid": ok,
                "oxt_submission_ready": manifest.get("oxt_submission_ready", False),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
