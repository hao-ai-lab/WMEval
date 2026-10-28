#!/usr/bin/env python3
"""LiquidContainer-scoped CLI for the WMEval pipeline.

Exposes only the universally-relevant flags plus the `--liquid_*` namespace
so the LiquidContainer entry point stays inspectable. The runner core
(`evaluation.runner.run_world_model_evaluation`) is shared with the other
scene-specific tools; every kwarg this script does not pass falls back to
its module-level default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.registry import get_available_modules
from evaluation.runner import run_world_model_evaluation


def main() -> None:
    available_modules = get_available_modules()
    ap = argparse.ArgumentParser(
        description=(
            "Run the LiquidContainer (sealed-bottle sloshing) world-model eval. "
            "Wraps run_world_model_evaluation with only the flags that matter "
            "for the four sloshing modules + cross-cutting perceptual modules."
        )
    )
    ap.add_argument("--video", required=True, help="Path to the candidate (world-model) video")
    ap.add_argument("--ref_video", required=True, help="Path to the Unity reference recording")
    ap.add_argument("--text_prompt", default="", help="Optional text prompt (passed through to the runner)")
    ap.add_argument(
        "--modules",
        default="liquid_velocity,liquid_volume_conservation,liquid_surface_orientation,slosh_spectrum",
        help=f"Comma-separated subset of {{{', '.join(available_modules)}}}",
    )
    ap.add_argument(
        "--output_json",
        default="reports/world_model_liquid/report.json",
        help="Where to write the JSON report",
    )
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Timestamp sampling rate")
    ap.add_argument(
        "--max_pairs", type=int, default=512,
        help=(
            "Maximum aligned frame pairs sampled into the EvaluationContext. "
            "Liquid modules also read --liquid_max_pairs to bound their own decode. "
            "Default 512 covers a 20s clip at up to 25 fps; bump if you analyse longer clips."
        ),
    )
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")

    g_liquid = ap.add_argument_group("liquid_* (LiquidContainer scene)")
    g_liquid.add_argument(
        "--liquid_cache_root", type=str, default="runs/liquid_cache",
        help="Directory for SAM2/PIV pickled caches (shared across trajectories)",
    )
    g_liquid.add_argument(
        "--liquid_sam2_model", type=str, default="facebook/sam2.1-hiera-small",
        help="HuggingFace SAM2 video predictor checkpoint id",
    )
    g_liquid.add_argument(
        "--liquid_sam2_prompt", type=str, default=None,
        help="Text prompt used to seed the SAM2 mask track (default: 'the liquid inside the bottle')",
    )
    g_liquid.add_argument(
        "--liquid_bottle_prompt", type=str, default=None,
        help=(
            "Text prompt for the bottle-track SAM2 pass used to estimate bottle bulk "
            "motion (default: 'the glass bottle'). Active only when "
            "--liquid_camera_compensation is on."
        ),
    )
    g_liquid.add_argument(
        "--liquid_piv_window_size", type=int, default=32,
        help="OpenPIV interrogation window edge (px)",
    )
    g_liquid.add_argument(
        "--liquid_piv_overlap", type=int, default=16,
        help="OpenPIV interrogation window overlap (px)",
    )
    g_liquid.add_argument(
        "--liquid_max_pairs", type=int, default=0,
        help="Per-module pair cap for liquid modules (0 = inherit global --max_pairs)",
    )
    g_liquid.add_argument(
        "--liquid_camera_compensation", action=argparse.BooleanOptionalAction, default=True,
        help="Subtract estimated camera motion from PIV velocities",
    )
    g_liquid.add_argument(
        "--liquid_camera_regime", type=str, default="c2_static_side",
        help="Camera regime label (e.g. c1_handheld, c2_static_side)",
    )
    g_liquid.add_argument(
        "--liquid_flow_estimator", type=str, default="raft", choices=("raft", "openpiv"),
        help=(
            "Velocity estimator inside the liquid mask. 'raft' (default) uses learned "
            "dense optical flow and works on opaque/uniform fluids. 'openpiv' uses "
            "cross-correlation; only useful when the liquid has rich texture."
        ),
    )
    g_liquid.add_argument(
        "--liquid_flow_min_mask_frac", type=float, default=0.5,
        help=(
            "Min fraction of a PIV cell's window that must be inside the liquid mask "
            "for the cell to count as valid (default 0.5). Lower values let mask-boundary "
            "cells leak in, where on partially-transparent renders the flow signal is "
            "actually background-through-glass apparent motion."
        ),
    )

    g_lpips = ap.add_argument_group("lpips (cross-cutting perceptual)")
    g_lpips.add_argument(
        "--lpips_vlm_scene_key", default="liquid_container",
        help="Scene registry key for VLM judge instructions when lpips is enabled",
    )

    args = ap.parse_args()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    report = run_world_model_evaluation(
        video_path=args.video,
        ref_video_path=args.ref_video,
        text_prompt=args.text_prompt,
        modules=modules,
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        device=args.device,
        lpips_vlm_scene_key=args.lpips_vlm_scene_key,
        liquid_cache_root=args.liquid_cache_root,
        liquid_sam2_model=args.liquid_sam2_model,
        liquid_sam2_prompt=args.liquid_sam2_prompt,
        liquid_bottle_prompt=args.liquid_bottle_prompt,
        liquid_piv_window_size=args.liquid_piv_window_size,
        liquid_piv_overlap=args.liquid_piv_overlap,
        liquid_max_pairs=args.liquid_max_pairs,
        liquid_camera_compensation=args.liquid_camera_compensation,
        liquid_camera_regime=args.liquid_camera_regime,
        liquid_flow_estimator=args.liquid_flow_estimator,
        liquid_flow_min_mask_frac=args.liquid_flow_min_mask_frac,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote analysis report to: {out_path}")


if __name__ == "__main__":
    main()
