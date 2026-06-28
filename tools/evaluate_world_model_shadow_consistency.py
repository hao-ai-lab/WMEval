#!/usr/bin/env python3
"""Shadow-consistency evaluation for world-model videos.

Runs GroundingDINO + SAM 2 object tracking and computes six lighting /
shadow-consistency metrics without requiring a per-frame Unity reference.

Example::

    python tools/evaluate_world_model_shadow_consistency.py \\
        --video path/to/generated.mp4 \\
        --text_prompt "table. chair. lamp." \\
        --anchor_label table \\
        --output_json reports/shadow_consistency.json

Metrics produced:
    SRR   Scene Retention Rate (%)
    SDR   Shadow Detection Rate (%)
    MOSD  Multi-Object Sun Disagreement (deg)
    SDJ   Shadow Direction Jitter (deg/frame)
    SAS   Shadow Area Stability (%)
    SSD   Shadow Shape Drift (Hu L2)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.registry import get_available_modules
from evaluation.runner import run_world_model_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shadow-consistency evaluation (GroundingDINO + SAM 2 + Hu moments)",
    )
    ap.add_argument("--video", required=True, help="Path to the world-model video")
    ap.add_argument(
        "--ref_video",
        default=None,
        help="Optional Unity reference video (not required for shadow_consistency)",
    )
    ap.add_argument("--text_prompt", default="table. chair. lamp.",
                     help="GroundingDINO text prompt for object detection")
    ap.add_argument("--anchor_label", default="table",
                     help="Object class used for Scene Retention Rate")
    ap.add_argument("--hud_preset", default="none",
                     choices=["none", "lingbot", "mg2", "mg3", "hy", "genie"],
                     help="Mask preset for model-specific HUD overlays")
    ap.add_argument("--output_json", default="reports/shadow_consistency.json",
                     help="Where to write the JSON report")
    ap.add_argument("--eval_fps", type=float, default=24.0)
    ap.add_argument("--max_pairs", type=int, default=0,
                     help="Max frame pairs (0 = all frames)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # The shadow_consistency module only needs candidate_video; we pass
    # ref_video if provided, else use the video itself as a dummy ref
    # to satisfy EvaluationContext construction.
    ref = args.ref_video or args.video

    report = run_world_model_evaluation(
        video_path=args.video,
        ref_video_path=ref,
        text_prompt=args.text_prompt,
        modules=["shadow_consistency"],
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        device=args.device,
    )

    # Inject the shadow-specific artifacts before they are needed
    # (these are passed via context.artifacts in the module)
    result = report.get("results", {}).get("shadow_consistency", {})

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Shadow Consistency: {args.video} ===")
    metric_keys = [
        "scene_retention_rate_pct",
        "shadow_detection_rate_pct",
        "multi_object_sun_disagreement_deg",
        "shadow_direction_jitter_deg_per_frame",
        "shadow_area_stability_pct",
        "shadow_shape_drift",
    ]
    for k in metric_keys:
        v = result.get(k)
        print(f"  {k:48s} {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
