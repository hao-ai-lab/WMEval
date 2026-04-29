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
    ap.add_argument(
        "--lpips_sampling",
        choices=["temporal", "dense", "causal_vlm_collision"],
        default="temporal",
        help="LPIPS anchor mode: temporal fractions, dense time-aligned pairs, or VLM collision events",
    )
    ap.add_argument(
        "--lpips_temporal_fractions",
        default="0,0.5,1",
        help='Comma-separated fractions in [0,1] for temporal mode (e.g. "0,0.5,1")',
    )
    ap.add_argument(
        "--lpips_event_frames_before",
        type=int,
        default=2,
        help="Causal mode: frames before contact for LPIPS offsets (if --lpips_event_offsets not set)",
    )
    ap.add_argument(
        "--lpips_event_frames_after",
        type=int,
        default=2,
        help="Causal mode: frames after contact for LPIPS offsets (if --lpips_event_offsets not set)",
    )
    ap.add_argument(
        "--lpips_event_offsets",
        default=None,
        help='Optional comma-separated integer offsets around contact (overrides before/after), e.g. "-2,-1,0,1,2"',
    )
    ap.add_argument("--lpips_vlm_model", default="gpt-4o", help="OpenAI vision model for causal_vlm_collision")
    ap.add_argument(
        "--lpips_vlm_coarse_model",
        default=None,
        help="Optional model override for the coarse pass when using coarse_to_fine",
    )
    ap.add_argument(
        "--lpips_vlm_max_grid_frames",
        type=int,
        default=48,
        help="Max keyframes sent to the VLM per video in causal_vlm_collision (~40–56 typical)",
    )
    ap.add_argument(
        "--lpips_vlm_sampling_strategy",
        choices=["single_grid", "coarse_to_fine", "windowed_coarse_to_fine"],
        default="single_grid",
        help="Judge-frame selection strategy for causal_vlm_collision",
    )
    ap.add_argument(
        "--lpips_vlm_coarse_grid_frames",
        type=int,
        default=64,
        help="Coarse pass frame cap when --lpips_vlm_sampling_strategy coarse_to_fine is used",
    )
    ap.add_argument(
        "--lpips_vlm_disable_side_relabel",
        action="store_true",
        help="Disable the local second-pass side relabel step for detected contact episodes",
    )
    ap.add_argument(
        "--lpips_vlm_side_model",
        default=None,
        help="Optional model override for the local side relabel pass",
    )
    ap.add_argument(
        "--lpips_vlm_side_window_sec",
        type=float,
        default=0.75,
        help="Seconds of local context on each side of a detected event for side relabeling",
    )
    ap.add_argument(
        "--lpips_vlm_side_max_frames",
        type=int,
        default=12,
        help="Maximum frames sampled for the local side relabel pass per detected event",
    )
    ap.add_argument(
        "--lpips_vlm_parallel_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run reference and candidate causal extraction in parallel when event_source=both",
    )
    ap.add_argument(
        "--lpips_vlm_side_relabel_concurrency",
        type=int,
        default=2,
        help="Concurrency cap for local side relabel requests",
    )
    ap.add_argument(
        "--lpips_vlm_coarse_window_concurrency",
        type=int,
        default=1,
        help="Reserved concurrency cap for coarse window calls (kept at 1 by default)",
    )
    ap.add_argument(
        "--lpips_vlm_coarse_reasoning_effort",
        default="minimal",
        help="Reasoning effort for coarse causal judge calls on GPT-5 models",
    )
    ap.add_argument(
        "--lpips_vlm_fine_reasoning_effort",
        default="medium",
        help="Reasoning effort for fine causal judge calls on GPT-5 models",
    )
    ap.add_argument(
        "--lpips_vlm_side_reasoning_effort",
        default="medium",
        help="Reasoning effort for side relabel calls on GPT-5 models",
    )
    ap.add_argument(
        "--lpips_vlm_window_sec",
        type=float,
        default=2.5,
        help="Window length in seconds for windowed_coarse_to_fine detection",
    )
    ap.add_argument(
        "--lpips_vlm_window_overlap_sec",
        type=float,
        default=1.25,
        help="Window overlap in seconds for windowed_coarse_to_fine detection",
    )
    ap.add_argument(
        "--lpips_vlm_window_max_frames",
        type=int,
        default=24,
        help="Maximum sampled frames per detection window in windowed_coarse_to_fine mode",
    )
    ap.add_argument(
        "--lpips_vlm_window_merge_gap_frames",
        type=int,
        default=18,
        help="Frame gap used to merge nearby events after windowed detection",
    )
    ap.add_argument(
        "--lpips_vlm_event_split_gap_frames",
        type=int,
        default=12,
        help="Split an event into multiple episodes when its evidence frames contain larger temporal gaps",
    )
    ap.add_argument(
        "--lpips_vlm_event_split_min_cluster_size",
        type=int,
        default=2,
        help="Minimum evidence frames required for a split sub-episode to be kept",
    )
    ap.add_argument(
        "--lpips_vlm_refine_window_sec",
        type=float,
        default=1.0,
        help="Seconds of context added around coarse event evidence for the fine pass",
    )
    ap.add_argument(
        "--lpips_vlm_refine_max_frames_per_window",
        type=int,
        default=24,
        help="Maximum sampled frames per refined event window",
    )
    ap.add_argument(
        "--lpips_vlm_prompt_version",
        default="causal_collision_v2",
        help="Version tag stored in JSON for causal judge prompts",
    )
    ap.add_argument(
        "--lpips_vlm_max_reference_events",
        type=int,
        default=None,
        help=(
            "Unity VLM only: if set (e.g. 1), append a request for at most that many "
            "'events' in the judge JSON (soft constraint; not post-filtered)"
        ),
    )
    ap.add_argument(
        "--lpips_vlm_reference_trim_sec",
        type=float,
        default=None,
        help=(
            "Reference-only VLM: only sample keyframes from the first N seconds of the "
            "reference clip (indices stay real file frame numbers). Use e.g. 2 to focus "
            "on an early collision."
        ),
    )
    ap.add_argument(
        "--lpips_vlm_judge_hint",
        default="",
        help=(
            "Optional short hint appended to the reference VLM judge prompt (ignored if "
            "empty). Hints are soft: model may disregard if frames disagree."
        ),
    )
    ap.add_argument(
        "--lpips_vlm_scene_key",
        default=None,
        help="Optional scene registry key for judge instructions; inferred from video paths when omitted",
    )
    ap.add_argument(
        "--lpips_causal_event_source",
        choices=["both", "reference", "unity"],
        default="both",
        help=(
            "For causal_vlm_collision: 'reference' (or 'unity') = VLM only on the "
            "reference video; candidate instants follow the same elapsed times as Unity. "
            "'both' = legacy: VLM on both videos + greedy event matching."
        ),
    )
    ap.add_argument(
        "--lpips_causal_lpips_pairing",
        choices=["auto", "wall_clock", "per_video_contact"],
        default="auto",
        help=(
            "causal_vlm_collision: how to pick ref/cand frames per offset. auto = "
            "per_video_contact when event_source is both, else wall_clock. "
            "per_video_contact = (e_ref+Δ, e_cand+Δ) on each clip for Genie-friendly timing."
        ),
    )
    ap.add_argument(
        "--save_inspection_artifacts",
        action="store_true",
        help="Write QA frames under inspection_root / run_id / sample_id",
    )
    ap.add_argument(
        "--inspection_root",
        default="reports/world_model_lpips/qa",
        help="Base directory for optional LPIPS inspection outputs",
    )
    ap.add_argument(
        "--inspection_run_id",
        default="single_run",
        help="Run id subdirectory for inspection artifacts",
    )
    ap.add_argument(
        "--inspection_sample_id",
        default="sample",
        help="Sample id subdirectory for inspection artifacts",
    )
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
    ap.add_argument(
        "--ptlflow_root",
        default=None,
        help="Optional PTLFlow checkout; auto-resolves ../ptlflow then /workspace/ptlflow",
    )
    ap.add_argument("--synthetic_grid_size", type=int, default=8)
    ap.add_argument(
        "--lpips_vlm_use_flow_proposals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RAFT-guided motion proposals before causal VLM detection",
    )
    ap.add_argument(
        "--lpips_vlm_flow_proposal_stride",
        type=int,
        default=12,
        help="Frame stride for RAFT proposal sampling on each video",
    )
    ap.add_argument(
        "--lpips_vlm_flow_peak_threshold",
        type=float,
        default=0.35,
        help="Relative threshold against max RAFT saliency for keeping proposal peaks",
    )
    ap.add_argument(
        "--lpips_vlm_flow_topk_windows",
        type=int,
        default=8,
        help="Maximum number of RAFT-guided proposal windows per video",
    )
    ap.add_argument(
        "--lpips_vlm_flow_window_sec",
        type=float,
        default=1.25,
        help="Half-window size in seconds around each RAFT-guided proposal anchor",
    )
    ap.add_argument(
        "--lpips_vlm_flow_camera_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Camera-compensate RAFT flow before proposal saliency scoring",
    )
    ap.add_argument(
        "--lpips_vlm_scene_hints",
        default=None,
        help="Optional extra judge instructions (appended under Scene-specific instructions)",
    )
    ap.add_argument(
        "--lpips_vlm_contact_types",
        default=None,
        help="Comma-separated event types to keep (default: collision,impact,contact)",
    )
    ap.add_argument(
        "--lpips_vlm_directional_sides",
        default=None,
        help="Comma-separated sides for strict pairing (default: left,right,front,back)",
    )
    ap.add_argument(
        "--lpips_vlm_system_prompt",
        default=None,
        help="Override VLM system message (default: generic physics-style judge)",
    )
    ap.add_argument(
        "--lpips_vlm_judge_user_prompt_template",
        default=None,
        help="Override user prompt; must contain {idx_str} and {extra_instructions}",
    )
    ap.add_argument(
        "--causal_match_mode",
        default="monotonic_chronology",
        choices=[
            "monotonic_chronology",
            "monotonic_side_time",
            "strict_fallback_order",
            "strict",
            "order",
            "side_agnostic",
            "auto",
        ],
        help="Pairing for causal LPIPS: auto is an alias for monotonic_chronology",
    )
    ap.add_argument(
        "--lpips_vlm_pair_accept_cost",
        type=float,
        default=None,
        help="Maximum monotonic chronology match cost before a pair is left unmatched",
    )
    ap.add_argument(
        "--lpips_vlm_skip_cost",
        type=float,
        default=None,
        help="Cost of leaving an event unmatched in monotonic chronology matching",
    )

    args = ap.parse_args()

    modules = [module for module in args.modules.split(",") if module.strip()]
    temporal_fractions = [
        float(value.strip())
        for value in args.lpips_temporal_fractions.split(",")
        if value.strip()
    ]
    event_offsets = None
    if args.lpips_event_offsets:
        event_offsets = [
            int(value.strip())
            for value in args.lpips_event_offsets.split(",")
            if value.strip()
        ]
    control_bad_pixel_thresholds = [
        float(value.strip())
        for value in args.control_bad_pixel_thresholds.split(",")
        if value.strip()
    ]

    causal_match_mode = (
        "monotonic_chronology" if args.causal_match_mode == "auto" else args.causal_match_mode
    )

    report = run_world_model_evaluation(
        video_path=args.video,
        ref_video_path=args.ref_video,
        text_prompt=args.text_prompt,
        modules=modules,
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        lpips_sampling=args.lpips_sampling,
        lpips_temporal_fractions=temporal_fractions,
        lpips_event_frames_before=args.lpips_event_frames_before,
        lpips_event_frames_after=args.lpips_event_frames_after,
        lpips_event_offsets=event_offsets,
        lpips_vlm_model=args.lpips_vlm_model,
        lpips_vlm_coarse_model=args.lpips_vlm_coarse_model,
        lpips_vlm_max_grid_frames=args.lpips_vlm_max_grid_frames,
        lpips_vlm_sampling_strategy=args.lpips_vlm_sampling_strategy,
        lpips_vlm_coarse_grid_frames=args.lpips_vlm_coarse_grid_frames,
        lpips_vlm_side_relabel=not args.lpips_vlm_disable_side_relabel,
        lpips_vlm_side_model=args.lpips_vlm_side_model,
        lpips_vlm_side_window_sec=args.lpips_vlm_side_window_sec,
        lpips_vlm_side_max_frames=args.lpips_vlm_side_max_frames,
        lpips_vlm_parallel_videos=args.lpips_vlm_parallel_videos,
        lpips_vlm_side_relabel_concurrency=args.lpips_vlm_side_relabel_concurrency,
        lpips_vlm_coarse_window_concurrency=args.lpips_vlm_coarse_window_concurrency,
        lpips_vlm_coarse_reasoning_effort=args.lpips_vlm_coarse_reasoning_effort,
        lpips_vlm_fine_reasoning_effort=args.lpips_vlm_fine_reasoning_effort,
        lpips_vlm_side_reasoning_effort=args.lpips_vlm_side_reasoning_effort,
        lpips_vlm_use_flow_proposals=args.lpips_vlm_use_flow_proposals,
        lpips_vlm_flow_proposal_stride=args.lpips_vlm_flow_proposal_stride,
        lpips_vlm_flow_peak_threshold=args.lpips_vlm_flow_peak_threshold,
        lpips_vlm_flow_topk_windows=args.lpips_vlm_flow_topk_windows,
        lpips_vlm_flow_window_sec=args.lpips_vlm_flow_window_sec,
        lpips_vlm_flow_camera_compensation=args.lpips_vlm_flow_camera_compensation,
        lpips_vlm_window_sec=args.lpips_vlm_window_sec,
        lpips_vlm_window_overlap_sec=args.lpips_vlm_window_overlap_sec,
        lpips_vlm_window_max_frames=args.lpips_vlm_window_max_frames,
        lpips_vlm_window_merge_gap_frames=args.lpips_vlm_window_merge_gap_frames,
        lpips_vlm_event_split_gap_frames=args.lpips_vlm_event_split_gap_frames,
        lpips_vlm_event_split_min_cluster_size=args.lpips_vlm_event_split_min_cluster_size,
        lpips_vlm_refine_window_sec=args.lpips_vlm_refine_window_sec,
        lpips_vlm_refine_max_frames_per_window=args.lpips_vlm_refine_max_frames_per_window,
        lpips_vlm_prompt_version=args.lpips_vlm_prompt_version,
        lpips_vlm_max_reference_events=args.lpips_vlm_max_reference_events,
        lpips_vlm_reference_trim_sec=args.lpips_vlm_reference_trim_sec,
        lpips_vlm_judge_hint=args.lpips_vlm_judge_hint,
        lpips_vlm_scene_key=args.lpips_vlm_scene_key,
        lpips_causal_event_source=args.lpips_causal_event_source,
        lpips_causal_lpips_pairing=(
            None
            if args.lpips_causal_lpips_pairing == "auto"
            else args.lpips_causal_lpips_pairing
        ),
        save_inspection_artifacts=args.save_inspection_artifacts,
        inspection_root=args.inspection_root,
        inspection_run_id=args.inspection_run_id,
        inspection_sample_id=args.inspection_sample_id,
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
        lpips_vlm_scene_hints=args.lpips_vlm_scene_hints,
        lpips_vlm_contact_types=args.lpips_vlm_contact_types,
        lpips_vlm_directional_sides=args.lpips_vlm_directional_sides,
        lpips_vlm_system_prompt=args.lpips_vlm_system_prompt,
        lpips_vlm_judge_user_prompt_template=args.lpips_vlm_judge_user_prompt_template,
        causal_match_mode=causal_match_mode,
        lpips_vlm_pair_accept_cost=args.lpips_vlm_pair_accept_cost,
        lpips_vlm_skip_cost=args.lpips_vlm_skip_cost,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote analysis report to: {out_path}")


if __name__ == "__main__":
    main()
