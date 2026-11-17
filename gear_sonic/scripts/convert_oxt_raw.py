"""Build an FTP-1/OXT Zarr candidate from one closed OXT raw session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear_sonic.utils.data_collection.oxt_conversion import convert_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="closed raw/<session-id> directory")
    parser.add_argument("mapping", type=Path, help="reviewed OXT mapping JSON")
    parser.add_argument("output", type=Path, help="new candidate output directory; must not exist")
    parser.add_argument(
        "--oxt-qc-root",
        required=True,
        type=Path,
        help="OXT-QC/OXT-QC checkout containing common/replay_buffer.py",
    )
    args = parser.parse_args()
    try:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("mapping root must be a JSON object")
        output = convert_session(
            args.session,
            mapping,
            args.output,
            args.oxt_qc_root,
        )
    except Exception as exc:
        print(f"OXT conversion failed: {exc}", file=sys.stderr)
        return 1
    report = json.loads((output / "processing_report.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "candidate": str(output),
                "zarr": str(output / "dataset.zarr"),
                "episodes": len(report["episodes"]),
                "frames": report["total_frames"],
                "source_qualified_for_oxt": report["source_qualified_for_oxt"],
                "release_ready": report["release_ready"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
