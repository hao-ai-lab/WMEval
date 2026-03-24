#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.runner import run_world_model_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute RAFT-based controllability metrics by comparing flow derived "
            "from a Unity reference video against a world-model video."
        )
    )
    ap.add_argument("--video", required=True, help="Path to the generated / candidate video")
    ap.add_argument("--ref_video", required=True, help="Path to the Unity reference video")
    ap.add_argument(
        "--text_prompt",
        default="",
        help="Optional text prompt stored in the JSON report for context",
    )
    ap.add_argument(
        "--output_json",
        default="reports/world_model_controllability/report.json",
        help="Where to write the JSON report",
    )
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Timestamp sampling rate")
    ap.add_argument("--max_pairs", type=int, default=32, help="Maximum aligned frame pairs")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    ap.add_argument(
        "--disable_control_camera_compensation",
        action="store_true",
        help="Disable camera-motion compensation",
    )
    ap.add_argument(
        "--control_global_motion_mode",
        choices=["affine_ransac", "median_translation", "mean_translation"],
        default="median_translation",
        help="Global camera-motion estimator",
    )
    ap.add_argument(
        "--control_motion_mask_threshold",
        type=float,
        default=0.5,
        help="Reference-flow magnitude threshold for dynamic-pixel metrics",
    )
    ap.add_argument(
        "--control_motion_mask_percentile",
        type=float,
        default=75.0,
        help="Adaptive percentile for selecting dynamic pixels",
    )
    ap.add_argument(
        "--control_bad_pixel_thresholds",
        default="1.0,3.0",
        help="Comma-separated EPE thresholds for bad-pixel rates",
    )

    args = ap.parse_args()
    control_bad_pixel_thresholds = [
        float(value.strip())
        for value in args.control_bad_pixel_thresholds.split(",")
        if value.strip()
    ]

    report = run_world_model_evaluation(
        video_path=args.video,
        ref_video_path=args.ref_video,
        text_prompt=args.text_prompt,
        modules=["control_motion"],
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        device=args.device,
        control_camera_compensation=not args.disable_control_camera_compensation,
        control_global_motion_mode=args.control_global_motion_mode,
        control_motion_mask_threshold=args.control_motion_mask_threshold,
        control_motion_mask_percentile=args.control_motion_mask_percentile,
        control_bad_pixel_thresholds=control_bad_pixel_thresholds,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote controllability report to: {out_path}")


if __name__ == "__main__":
    main()
