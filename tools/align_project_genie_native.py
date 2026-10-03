#!/usr/bin/env python3
"""Validate and trim a Project Genie native video to its browser trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.project_genie_alignment import (
    align_native_video,
    update_receipt_with_alignment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--native-sample-fps", type=float, default=4.0)
    parser.add_argument("--no-update-receipt", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = align_native_video(
        run_dir / "native.mp4",
        run_dir / "observed_frames.json",
        run_dir / "action_trace.json",
        run_dir,
        sample_count=args.sample_count,
        native_sample_fps=args.native_sample_fps,
    )
    if not args.no_update_receipt:
        update_receipt_with_alignment(run_dir / "receipt.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
