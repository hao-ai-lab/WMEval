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
            "Compute RAFT-based controllability metrics (Unity ref vs world-model), "
            "optionally plus Longuet-Higgins synthetic flow (from actions or inferred "
            "from RAFT on the reference) vs RAFT on the candidate — metrics aligned "
            "with ptlflow eval_flow_divergence synthetic mode."
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
    ap.add_argument(
        "--synthetic_calibration",
        default=None,
        help=(
            "Path to calibration.json (e.g. from ptlflow repo). Enables synthetic-vs-RAFT "
            "block in the report when combined with --synthetic_action_file or "
            "--synthetic_infer_actions."
        ),
    )
    ap.add_argument(
        "--synthetic_action_file",
        default=None,
        help="Optional .npy with item()['keyboard'] and ['mouse'] (Unity-style log)",
    )
    ap.add_argument(
        "--synthetic_infer_actions",
        action="store_true",
        help=(
            "Infer continuous action vectors by fitting synthetic flow to RAFT flow on "
            "the reference (proxy when no action npy is available; slower)."
        ),
    )
    ap.add_argument(
        "--synthetic_use_depth",
        action="store_true",
        help="Use Depth Anything via transformers for translation term (heavier deps).",
    )
    ap.add_argument(
        "--ptlflow_root",
        default=None,
        help="Directory containing synthetic_flow.py (default: ../ptlflow or /workspace/ptlflow)",
    )
    ap.add_argument(
        "--synthetic_grid_size",
        type=int,
        default=8,
        help="Grid size for flow_divergence-style grid_epe metrics",
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
        synthetic_calibration=args.synthetic_calibration,
        synthetic_action_file=args.synthetic_action_file,
        synthetic_infer_actions=args.synthetic_infer_actions,
        synthetic_use_depth=args.synthetic_use_depth,
        ptlflow_root=args.ptlflow_root,
        synthetic_grid_size=args.synthetic_grid_size,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote controllability report to: {out_path}")


if __name__ == "__main__":
    main()
