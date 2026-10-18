#!/usr/bin/env python3
"""Recover a Project Genie native export stranded in the browser download folder.

The source is accepted only after the native video validates against the run's
observed browser frames and action trace. The source file is copied, never moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.project_genie_alignment import (
    align_native_video,
    update_receipt_with_alignment,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover(source: Path, run_dir: Path, endpoint_mode: str) -> dict[str, object]:
    source = source.resolve()
    run_dir = run_dir.resolve()
    receipt_path = run_dir / "receipt.json"
    observed = run_dir / "observed_frames.json"
    actions = run_dir / "action_trace.json"
    for path in (source, receipt_path, observed, actions):
        if not path.is_file():
            raise FileNotFoundError(path)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_hash = sha256(source)
    existing_hash = receipt.get("native_video_sha256")
    if existing_hash not in {None, source_hash}:
        raise RuntimeError(
            "run already names a different native video; refusing to overwrite it"
        )

    # Validate the proposed association before changing the run directory.
    with tempfile.TemporaryDirectory(prefix="wmeval_genie_recovery_") as temporary:
        align_native_video(
            source,
            observed,
            actions,
            Path(temporary),
            endpoint_mode=endpoint_mode,
        )

    native = run_dir / "native.mp4"
    shutil.copy2(source, native)
    report = align_native_video(
        native,
        observed,
        actions,
        run_dir,
        endpoint_mode=endpoint_mode,
    )
    updated = update_receipt_with_alignment(receipt_path, report)
    updated["native_video_status"] = "recovered_filesystem_download"
    updated["native_video_recovery"] = {
        "method": "explicit_source_plus_visual_trajectory_alignment",
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": source_hash,
    }
    receipt_path.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-mode",
        choices=("action_trace_end", "closed_loop_visual_return"),
        required=True,
    )
    args = parser.parse_args()
    receipt = recover(args.source, args.run_dir, args.endpoint_mode)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "native_video": receipt["native_video"],
                "native_video_sha256": receipt["native_video_sha256"],
                "native_video_status": receipt["native_video_status"],
                "native_video_timeline_alignment": receipt[
                    "native_video_timeline_alignment"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
