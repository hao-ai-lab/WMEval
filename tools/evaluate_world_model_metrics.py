#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.registry import get_available_modules
from evaluation.runner import run_world_model_evaluation


def main() -> None:
    available_modules = get_available_modules()
    ap = argparse.ArgumentParser(
        description=(
            "Compute world-model video metrics with a selectable --modules list "
            "and a structured JSON report."
        )
    )
    ap.add_argument("--video", required=True, help="Path to the generated / candidate video")
    ap.add_argument("--ref_video", default=None, help="Optional reference video")
    ap.add_argument(
        "--text_prompt",
        default="",
        help="Optional text description / prompt for future text-conditioned metrics",
    )
    ap.add_argument(
        "--modules",
        default="lpips",
        help=f"Comma-separated subset of {{{', '.join(available_modules)}}}",
    )
    ap.add_argument(
        "--output_json",
        default="reports/world_model_metrics/report.json",
        help="Where to write the JSON report",
    )
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Timestamp sampling rate")
    ap.add_argument("--max_pairs", type=int, default=32, help="Maximum aligned frame pairs")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--unity_log_json", default=None, help="Unity per-frame log JSON with light/object states")
    ap.add_argument("--camera_intrinsics_json", default=None, help="Optional camera intrinsics JSON")
    ap.add_argument("--unity_flow_dir", default=None, help="Optional Unity optical-flow ground-truth directory")
    ap.add_argument("--unity_shadow_dir", default=None, help="Optional Unity shadow mask directory")
    ap.add_argument("--diffusionlight_repo", default="/workspace/unity/DiffusionLight", help="DiffusionLight repo path")
    ap.add_argument("--run_diffusionlight", action="store_true", help="Run DiffusionLight pipeline on sampled frames")
    ap.add_argument("--diffusionlight_output_root", default=None, help="Optional output root for DiffusionLight runs")
    ap.add_argument("--diffusionlight_ref_hdr_dir", default=None, help="Precomputed DiffusionLight HDR dir for reference video")
    ap.add_argument("--diffusionlight_candidate_hdr_dir", default=None, help="Precomputed DiffusionLight HDR dir for candidate video")
    ap.add_argument("--sam2_model", default="facebook/sam2.1-hiera-small", help="SAM2 model ID for shadow module")
    ap.add_argument(
        "--disable_control_camera_compensation",
        action="store_true",
        help="Disable camera-motion compensation for control_motion",
    )
    ap.add_argument(
        "--control_global_motion_mode",
        choices=["affine_ransac", "median_translation", "mean_translation"],
        default="median_translation",
        help="Global camera-motion estimator for control_motion",
    )
    ap.add_argument(
        "--control_motion_mask_threshold",
        type=float,
        default=0.5,
        help="Reference-flow magnitude threshold for dynamic-pixel control metrics",
    )
    ap.add_argument(
        "--control_motion_mask_percentile",
        type=float,
        default=75.0,
        help="Adaptive percentile for selecting dynamic pixels in control_motion",
    )
    ap.add_argument(
        "--control_bad_pixel_thresholds",
        default="1.0,3.0",
        help="Comma-separated bad-pixel EPE thresholds for control_motion",
    )
    ap.add_argument(
        "--synthetic_calibration",
        default=None,
        help="calibration.json path for control_motion synthetic-vs-RAFT block",
    )
    ap.add_argument("--synthetic_action_file", default=None, help="Optional action .npy")
    ap.add_argument(
        "--synthetic_infer_actions",
        action="store_true",
        help="Infer actions from RAFT on reference for synthetic flow",
    )
    ap.add_argument(
        "--synthetic_use_depth",
        action="store_true",
        help="Depth Anything for synthetic translation term",
    )
    ap.add_argument("--ptlflow_root", default="/workspace/ptlflow")
    ap.add_argument("--synthetic_grid_size", type=int, default=8)
    ap.add_argument(
        "--lpips_sampling",
        default="temporal",
        choices=["temporal", "causal_vlm_collision"],
        help="temporal: uniform time samples; causal_vlm_collision: VLM collision events plus temporal LPIPS baseline",
    )
    ap.add_argument("--lpips_vlm_model", default="gpt-4o", help="OpenAI vision model for causal LPIPS")
    ap.add_argument(
        "--lpips_vlm_max_grid_frames",
        type=int,
        default=24,
        help="Max keyframes sent to the VLM per video",
    )
    ap.add_argument(
        "--lpips_vlm_prompt_version",
        default="causal_collision_v2",
        help="Prompt version tag (stored in report; v2 adds front/back contact labels)",
    )
    ap.add_argument(
        "--causal_match_mode",
        default="strict_fallback_order",
        choices=["strict_fallback_order", "strict", "order", "side_agnostic", "auto"],
        help="Pairing for causal LPIPS: auto is an alias for strict_fallback_order",
    )

    args = ap.parse_args()

    modules = [module for module in args.modules.split(",") if module.strip()]
    control_bad_pixel_thresholds = [
        float(value.strip())
        for value in args.control_bad_pixel_thresholds.split(",")
        if value.strip()
    ]

    causal_match_mode = (
        "strict_fallback_order" if args.causal_match_mode == "auto" else args.causal_match_mode
    )

    report = run_world_model_evaluation(
        video_path=args.video,
        ref_video_path=args.ref_video,
        text_prompt=args.text_prompt,
        modules=modules,
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        device=args.device,
        unity_log_json=args.unity_log_json,
        camera_intrinsics_json=args.camera_intrinsics_json,
        unity_flow_dir=args.unity_flow_dir,
        unity_shadow_dir=args.unity_shadow_dir,
        diffusionlight_repo=args.diffusionlight_repo,
        run_diffusionlight=args.run_diffusionlight,
        diffusionlight_output_root=args.diffusionlight_output_root,
        diffusionlight_ref_hdr_dir=args.diffusionlight_ref_hdr_dir,
        diffusionlight_candidate_hdr_dir=args.diffusionlight_candidate_hdr_dir,
        sam2_model=args.sam2_model,
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
        lpips_sampling=args.lpips_sampling,
        lpips_vlm_model=args.lpips_vlm_model,
        lpips_vlm_max_grid_frames=args.lpips_vlm_max_grid_frames,
        lpips_vlm_prompt_version=args.lpips_vlm_prompt_version,
        causal_match_mode=causal_match_mode,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote analysis report to: {out_path}")


if __name__ == "__main__":
    main()
