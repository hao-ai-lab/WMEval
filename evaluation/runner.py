from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from .context import EvaluationContext
from .path_resolution import resolve_ptlflow_root
from .registry import normalize_modules, run_modules
from .scene_prompt_registry import resolve_scene_config, resolve_scene_hints
from .video_io import build_evaluation_context


def _merge_lpips_artifacts(
    context: EvaluationContext,
    *,
    lpips_sampling: str,
    lpips_temporal_fractions: Optional[Sequence[float]],
    lpips_temporal_labels: Optional[Sequence[str]],
    lpips_event_frames_before: int,
    lpips_event_frames_after: int,
    lpips_event_offsets: Optional[Sequence[int]],
    lpips_vlm_model: str,
    lpips_vlm_coarse_model: Optional[str],
    lpips_vlm_max_grid_frames: int,
    lpips_vlm_sampling_strategy: str,
    lpips_vlm_coarse_grid_frames: int,
    lpips_vlm_side_relabel: bool,
    lpips_vlm_side_model: Optional[str],
    lpips_vlm_side_window_sec: float,
    lpips_vlm_side_max_frames: int,
    lpips_vlm_window_sec: float,
    lpips_vlm_window_overlap_sec: float,
    lpips_vlm_window_max_frames: int,
    lpips_vlm_window_merge_gap_frames: int,
    lpips_vlm_event_split_gap_frames: int,
    lpips_vlm_event_split_min_cluster_size: int,
    lpips_vlm_refine_window_sec: float,
    lpips_vlm_refine_max_frames_per_window: int,
    lpips_vlm_prompt_version: str,
    lpips_vlm_max_reference_events: Optional[int],
    lpips_vlm_reference_trim_sec: Optional[float],
    lpips_vlm_judge_hint: str,
    lpips_vlm_scene_key: Optional[str],
    lpips_causal_event_source: str,
    lpips_causal_lpips_pairing: Optional[str],
    lpips_vlm_pair_accept_cost: Optional[float],
    lpips_vlm_skip_cost: Optional[float],
    lpips_vlm_parallel_videos: bool,
    lpips_vlm_side_relabel_concurrency: int,
    lpips_vlm_coarse_window_concurrency: int,
    lpips_vlm_coarse_reasoning_effort: Optional[str],
    lpips_vlm_fine_reasoning_effort: Optional[str],
    lpips_vlm_side_reasoning_effort: Optional[str],
    lpips_vlm_use_flow_proposals: bool,
    lpips_vlm_flow_proposal_stride: int,
    lpips_vlm_flow_peak_threshold: float,
    lpips_vlm_flow_topk_windows: int,
    lpips_vlm_flow_window_sec: float,
    lpips_vlm_flow_camera_compensation: bool,
    save_inspection_artifacts: bool,
    inspection_root: str,
    inspection_run_id: str,
    inspection_sample_id: Optional[str],
) -> None:
    context.artifacts["lpips_sampling"] = lpips_sampling.strip().lower()
    if lpips_temporal_fractions is not None:
        context.artifacts["lpips_temporal_fractions"] = list(lpips_temporal_fractions)
    if lpips_temporal_labels is not None:
        context.artifacts["lpips_temporal_labels"] = list(lpips_temporal_labels)
    context.artifacts["lpips_event_frames_before"] = int(lpips_event_frames_before)
    context.artifacts["lpips_event_frames_after"] = int(lpips_event_frames_after)
    if lpips_event_offsets is not None:
        context.artifacts["lpips_event_offsets"] = list(lpips_event_offsets)
    context.artifacts["lpips_vlm_model"] = lpips_vlm_model
    if lpips_vlm_coarse_model is not None and str(lpips_vlm_coarse_model).strip():
        context.artifacts["lpips_vlm_coarse_model"] = str(lpips_vlm_coarse_model).strip()
    context.artifacts["lpips_vlm_max_grid_frames"] = int(lpips_vlm_max_grid_frames)
    context.artifacts["lpips_vlm_sampling_strategy"] = lpips_vlm_sampling_strategy
    context.artifacts["lpips_vlm_coarse_grid_frames"] = int(lpips_vlm_coarse_grid_frames)
    context.artifacts["lpips_vlm_side_relabel"] = bool(lpips_vlm_side_relabel)
    if lpips_vlm_side_model is not None and str(lpips_vlm_side_model).strip():
        context.artifacts["lpips_vlm_side_model"] = str(lpips_vlm_side_model).strip()
    context.artifacts["lpips_vlm_side_window_sec"] = float(lpips_vlm_side_window_sec)
    context.artifacts["lpips_vlm_side_max_frames"] = int(lpips_vlm_side_max_frames)
    context.artifacts["lpips_vlm_window_sec"] = float(lpips_vlm_window_sec)
    context.artifacts["lpips_vlm_window_overlap_sec"] = float(lpips_vlm_window_overlap_sec)
    context.artifacts["lpips_vlm_window_max_frames"] = int(lpips_vlm_window_max_frames)
    context.artifacts["lpips_vlm_window_merge_gap_frames"] = int(
        lpips_vlm_window_merge_gap_frames
    )
    context.artifacts["lpips_vlm_event_split_gap_frames"] = int(
        lpips_vlm_event_split_gap_frames
    )
    context.artifacts["lpips_vlm_event_split_min_cluster_size"] = int(
        lpips_vlm_event_split_min_cluster_size
    )
    context.artifacts["lpips_vlm_refine_window_sec"] = float(lpips_vlm_refine_window_sec)
    context.artifacts["lpips_vlm_refine_max_frames_per_window"] = int(
        lpips_vlm_refine_max_frames_per_window
    )
    context.artifacts["lpips_vlm_prompt_version"] = lpips_vlm_prompt_version
    if lpips_vlm_max_reference_events is not None:
        context.artifacts["lpips_vlm_max_reference_events"] = int(
            lpips_vlm_max_reference_events
        )
    if lpips_vlm_reference_trim_sec is not None:
        context.artifacts["lpips_vlm_reference_trim_sec"] = float(
            lpips_vlm_reference_trim_sec
        )
    hint = (lpips_vlm_judge_hint or "").strip()
    if hint:
        context.artifacts["lpips_vlm_judge_hint"] = hint
    if lpips_vlm_scene_key is not None and str(lpips_vlm_scene_key).strip():
        context.artifacts["lpips_vlm_scene_key"] = str(lpips_vlm_scene_key).strip().lower()
    context.artifacts["lpips_causal_event_source"] = lpips_causal_event_source.strip().lower()
    if lpips_causal_lpips_pairing is not None:
        raw = str(lpips_causal_lpips_pairing).strip().lower()
        if raw and raw != "auto":
            context.artifacts["lpips_causal_lpips_pairing"] = raw
    if lpips_vlm_pair_accept_cost is not None:
        context.artifacts["lpips_vlm_pair_accept_cost"] = float(lpips_vlm_pair_accept_cost)
    if lpips_vlm_skip_cost is not None:
        context.artifacts["lpips_vlm_skip_cost"] = float(lpips_vlm_skip_cost)
    context.artifacts["lpips_vlm_parallel_videos"] = bool(lpips_vlm_parallel_videos)
    context.artifacts["lpips_vlm_side_relabel_concurrency"] = int(lpips_vlm_side_relabel_concurrency)
    context.artifacts["lpips_vlm_coarse_window_concurrency"] = int(lpips_vlm_coarse_window_concurrency)
    context.artifacts["lpips_vlm_use_flow_proposals"] = bool(lpips_vlm_use_flow_proposals)
    context.artifacts["lpips_vlm_flow_proposal_stride"] = int(lpips_vlm_flow_proposal_stride)
    context.artifacts["lpips_vlm_flow_peak_threshold"] = float(lpips_vlm_flow_peak_threshold)
    context.artifacts["lpips_vlm_flow_topk_windows"] = int(lpips_vlm_flow_topk_windows)
    context.artifacts["lpips_vlm_flow_window_sec"] = float(lpips_vlm_flow_window_sec)
    context.artifacts["lpips_vlm_flow_camera_compensation"] = bool(
        lpips_vlm_flow_camera_compensation
    )
    if lpips_vlm_coarse_reasoning_effort is not None:
        context.artifacts["lpips_vlm_coarse_reasoning_effort"] = str(lpips_vlm_coarse_reasoning_effort).strip().lower()
    if lpips_vlm_fine_reasoning_effort is not None:
        context.artifacts["lpips_vlm_fine_reasoning_effort"] = str(lpips_vlm_fine_reasoning_effort).strip().lower()
    if lpips_vlm_side_reasoning_effort is not None:
        context.artifacts["lpips_vlm_side_reasoning_effort"] = str(lpips_vlm_side_reasoning_effort).strip().lower()
    context.artifacts["save_inspection_artifacts"] = bool(save_inspection_artifacts)
    context.artifacts["inspection_root"] = inspection_root
    context.artifacts["inspection_run_id"] = inspection_run_id
    if inspection_sample_id is not None:
        context.artifacts["inspection_sample_id"] = inspection_sample_id


def _merge_liquid_artifacts(
    context: EvaluationContext,
    *,
    liquid_cache_root: str,
    liquid_sam2_model: str,
    liquid_sam2_prompt: Optional[str],
    liquid_bottle_prompt: Optional[str],
    liquid_piv_window_size: int,
    liquid_piv_overlap: int,
    liquid_max_pairs: int,
    liquid_camera_compensation: bool,
    liquid_camera_regime: str,
    liquid_flow_estimator: str,
    liquid_flow_min_mask_frac: float,
) -> None:
    context.artifacts["liquid_cache_root"] = str(liquid_cache_root)
    context.artifacts["liquid_sam2_model"] = str(liquid_sam2_model)
    if liquid_sam2_prompt is not None and str(liquid_sam2_prompt).strip():
        context.artifacts["liquid_sam2_prompt"] = str(liquid_sam2_prompt).strip()
    if liquid_bottle_prompt is not None and str(liquid_bottle_prompt).strip():
        context.artifacts["liquid_bottle_prompt"] = str(liquid_bottle_prompt).strip()
    context.artifacts["liquid_piv_window_size"] = int(liquid_piv_window_size)
    context.artifacts["liquid_piv_overlap"] = int(liquid_piv_overlap)
    context.artifacts["liquid_max_pairs"] = int(liquid_max_pairs)
    context.artifacts["liquid_camera_compensation"] = bool(liquid_camera_compensation)
    context.artifacts["liquid_camera_regime"] = str(liquid_camera_regime).strip().lower()
    context.artifacts["liquid_flow_estimator"] = str(liquid_flow_estimator).strip().lower()
    context.artifacts["liquid_flow_min_mask_frac"] = float(liquid_flow_min_mask_frac)


def ensure_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_world_model_evaluation(
    video_path: str,
    ref_video_path: Optional[str],
    text_prompt: str = "",
    modules: Iterable[str] = ("lpips",),
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    candidate_fps_override: Optional[float] = None,
    lpips_sampling: str = "temporal",
    lpips_temporal_fractions: Optional[Sequence[float]] = None,
    lpips_temporal_labels: Optional[Sequence[str]] = None,
    lpips_event_frames_before: int = 2,
    lpips_event_frames_after: int = 2,
    lpips_event_offsets: Optional[Sequence[int]] = None,
    lpips_vlm_model: str = "gpt-4o",
    lpips_vlm_coarse_model: Optional[str] = None,
    lpips_vlm_max_grid_frames: int = 48,
    lpips_vlm_sampling_strategy: str = "single_grid",
    lpips_vlm_coarse_grid_frames: int = 64,
    lpips_vlm_side_relabel: bool = True,
    lpips_vlm_side_model: Optional[str] = None,
    lpips_vlm_side_window_sec: float = 0.75,
    lpips_vlm_side_max_frames: int = 12,
    lpips_vlm_window_sec: float = 2.5,
    lpips_vlm_window_overlap_sec: float = 1.25,
    lpips_vlm_window_max_frames: int = 24,
    lpips_vlm_window_merge_gap_frames: int = 18,
    lpips_vlm_event_split_gap_frames: int = 12,
    lpips_vlm_event_split_min_cluster_size: int = 2,
    lpips_vlm_refine_window_sec: float = 1.0,
    lpips_vlm_refine_max_frames_per_window: int = 24,
    lpips_vlm_prompt_version: str = "causal_collision_v2",
    lpips_vlm_max_reference_events: Optional[int] = None,
    lpips_vlm_reference_trim_sec: Optional[float] = None,
    lpips_vlm_judge_hint: str = "",
    lpips_vlm_scene_key: Optional[str] = None,
    lpips_causal_event_source: str = "both",
    lpips_causal_lpips_pairing: Optional[str] = None,
    lpips_vlm_pair_accept_cost: Optional[float] = None,
    lpips_vlm_skip_cost: Optional[float] = None,
    lpips_vlm_parallel_videos: bool = True,
    lpips_vlm_side_relabel_concurrency: int = 2,
    lpips_vlm_coarse_window_concurrency: int = 1,
    lpips_vlm_coarse_reasoning_effort: Optional[str] = "low",
    lpips_vlm_fine_reasoning_effort: Optional[str] = "medium",
    lpips_vlm_side_reasoning_effort: Optional[str] = "medium",
    lpips_vlm_use_flow_proposals: bool = True,
    lpips_vlm_flow_proposal_stride: int = 12,
    lpips_vlm_flow_peak_threshold: float = 0.35,
    lpips_vlm_flow_topk_windows: int = 8,
    lpips_vlm_flow_window_sec: float = 1.25,
    lpips_vlm_flow_camera_compensation: bool = True,
    save_inspection_artifacts: bool = False,
    inspection_root: str = "reports/world_model_lpips/qa",
    inspection_run_id: str = "run",
    inspection_sample_id: Optional[str] = None,
    device: Optional[str] = None,
    unity_log_json: Optional[str] = None,
    camera_intrinsics_json: Optional[str] = None,
    unity_flow_dir: Optional[str] = None,
    unity_shadow_dir: Optional[str] = None,
    diffusionlight_repo: Optional[str] = None,
    run_diffusionlight: bool = False,
    diffusionlight_output_root: Optional[str] = None,
    diffusionlight_ref_hdr_dir: Optional[str] = None,
    diffusionlight_candidate_hdr_dir: Optional[str] = None,
    sam2_model: Optional[str] = None,
    control_camera_compensation: bool = True,
    control_global_motion_mode: str = "median_translation",
    control_motion_mask_threshold: float = 0.5,
    control_motion_mask_percentile: float = 75.0,
    control_bad_pixel_thresholds: Optional[Sequence[float]] = None,
    synthetic_calibration: Optional[str] = None,
    synthetic_action_file: Optional[str] = None,
    synthetic_infer_actions: bool = False,
    synthetic_use_depth: bool = False,
    ptlflow_root: Optional[str] = None,
    synthetic_grid_size: int = 8,
    lpips_vlm_scene_hints: Optional[str] = None,
    lpips_vlm_contact_types: Optional[str] = None,
    lpips_vlm_directional_sides: Optional[str] = None,
    lpips_vlm_system_prompt: Optional[str] = None,
    lpips_vlm_judge_user_prompt_template: Optional[str] = None,
    causal_match_mode: str = "monotonic_chronology",
    causal_step_qa_dataset: Optional[str] = None,
    causal_step_video_id: Optional[str] = None,
    causal_step_answers_cache_dir: Optional[str] = None,
    causal_step_vlm_model: str = "gpt-5.4-mini",
    causal_step_vlm_prompt_version: str = "v1",
    causal_step_vlm_num_frames: int = 4,
    causal_step_wm_full_video: bool = False,
    causal_step_causal_fail_advance: bool = True,
    causal_step_judge: str = "openai",
    causalstep_root: Optional[str] = None,
    liquid_cache_root: str = "runs/liquid_cache",
    liquid_sam2_model: str = "facebook/sam2.1-hiera-small",
    liquid_sam2_prompt: Optional[str] = None,
    liquid_bottle_prompt: Optional[str] = None,
    liquid_piv_window_size: int = 32,
    liquid_piv_overlap: int = 16,
    liquid_max_pairs: int = 0,
    liquid_camera_compensation: bool = True,
    liquid_camera_regime: str = "c2_static_side",
    liquid_flow_estimator: str = "raft",
    liquid_flow_min_mask_frac: float = 0.5,
    shadow_text_prompt: str = "table. chair. lamp.",
    shadow_anchor_label: str = "table",
    shadow_hud_preset: str = "none",
    eoc_text_prompt: str = "ball. chair. table.",
    eoc_character_prompt: str = "character. player.",
    eoc_tau_det: float = 0.30,
    eoc_tau_mask: float = 0.50,
    eoc_tau_id: float = 0.50,
    eoc_max_objects: int = 6,
    eoc_depth_model: Optional[str] = None,
    eoc_gd_model: Optional[str] = None,
    eoc_unity_object_transforms_json: Optional[str] = None,
    eoc_unity_poses_npy: Optional[str] = None,
    eoc_unity_intrinsics_npy: Optional[str] = None,
    eoc_unity_object_diameters_json: Optional[str] = None,
    kin_text_prompt: str = "ball. car. object.",
    kin_static_anchor_labels: str = "",
    kin_track_thresh: float = 0.5,
    kin_track_buffer: int = 30,
    kin_match_thresh: float = 0.8,
    kin_min_track_len: int = 4,
    kin_gd_threshold: float = 0.30,
    kin_gd_text_threshold: float = 0.25,
    kin_gd_model: Optional[str] = None,
    kin_unity_object_transforms_json: Optional[str] = None,
    kin_unity_poses_npy: Optional[str] = None,
    kin_unity_intrinsics_npy: Optional[str] = None,
    kin_unity_object_masses_json: Optional[str] = None,
    kin_color_filter: bool = False,
    kin_color_hue_tolerance_deg: float = 25.0,
    kin_depth_lift: bool = False,
    kin_depth_lift_window: int = 6,
    kin_depth_model: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_device = ensure_device(device)
    normalized_modules = normalize_modules(modules)

    report: Dict[str, Any] = {
        "video": video_path,
        "ref_video": ref_video_path,
        "text_prompt": text_prompt,
        "device": resolved_device,
        "modules_requested": normalized_modules,
        "results": {},
    }

    if not ref_video_path:
        # clip4clip does not need ref_video; run it with candidate as both ref and candidate
        clip4clip_only = set(normalized_modules) <= {"clip4clip"}
        if clip4clip_only:
            context = build_evaluation_context(
                reference_video=video_path,
                candidate_video=video_path,
                eval_fps=eval_fps,
                max_pairs=max_pairs,
                text_prompt=text_prompt,
            )
            _merge_lpips_artifacts(
                context,
                lpips_sampling=lpips_sampling,
                lpips_temporal_fractions=lpips_temporal_fractions,
                lpips_temporal_labels=lpips_temporal_labels,
                lpips_event_frames_before=lpips_event_frames_before,
                lpips_event_frames_after=lpips_event_frames_after,
                lpips_event_offsets=lpips_event_offsets,
                lpips_vlm_model=lpips_vlm_model,
                lpips_vlm_coarse_model=lpips_vlm_coarse_model,
                lpips_vlm_max_grid_frames=lpips_vlm_max_grid_frames,
                lpips_vlm_sampling_strategy=lpips_vlm_sampling_strategy,
                lpips_vlm_coarse_grid_frames=lpips_vlm_coarse_grid_frames,
                lpips_vlm_side_relabel=lpips_vlm_side_relabel,
                lpips_vlm_side_model=lpips_vlm_side_model,
                lpips_vlm_side_window_sec=lpips_vlm_side_window_sec,
                lpips_vlm_side_max_frames=lpips_vlm_side_max_frames,
                lpips_vlm_window_sec=lpips_vlm_window_sec,
                lpips_vlm_window_overlap_sec=lpips_vlm_window_overlap_sec,
                lpips_vlm_window_max_frames=lpips_vlm_window_max_frames,
                lpips_vlm_window_merge_gap_frames=lpips_vlm_window_merge_gap_frames,
                lpips_vlm_event_split_gap_frames=lpips_vlm_event_split_gap_frames,
                lpips_vlm_event_split_min_cluster_size=lpips_vlm_event_split_min_cluster_size,
                lpips_vlm_refine_window_sec=lpips_vlm_refine_window_sec,
                lpips_vlm_refine_max_frames_per_window=lpips_vlm_refine_max_frames_per_window,
                lpips_vlm_prompt_version=lpips_vlm_prompt_version,
                lpips_vlm_max_reference_events=lpips_vlm_max_reference_events,
                lpips_vlm_reference_trim_sec=lpips_vlm_reference_trim_sec,
                lpips_vlm_judge_hint=lpips_vlm_judge_hint,
                lpips_vlm_scene_key=lpips_vlm_scene_key,
                lpips_causal_event_source=lpips_causal_event_source,
                lpips_causal_lpips_pairing=lpips_causal_lpips_pairing,
                lpips_vlm_pair_accept_cost=lpips_vlm_pair_accept_cost,
                lpips_vlm_skip_cost=lpips_vlm_skip_cost,
                lpips_vlm_parallel_videos=lpips_vlm_parallel_videos,
                lpips_vlm_side_relabel_concurrency=lpips_vlm_side_relabel_concurrency,
                lpips_vlm_coarse_window_concurrency=lpips_vlm_coarse_window_concurrency,
                lpips_vlm_coarse_reasoning_effort=lpips_vlm_coarse_reasoning_effort,
                lpips_vlm_fine_reasoning_effort=lpips_vlm_fine_reasoning_effort,
                lpips_vlm_side_reasoning_effort=lpips_vlm_side_reasoning_effort,
                lpips_vlm_use_flow_proposals=lpips_vlm_use_flow_proposals,
                lpips_vlm_flow_proposal_stride=lpips_vlm_flow_proposal_stride,
                lpips_vlm_flow_peak_threshold=lpips_vlm_flow_peak_threshold,
                lpips_vlm_flow_topk_windows=lpips_vlm_flow_topk_windows,
                lpips_vlm_flow_window_sec=lpips_vlm_flow_window_sec,
                lpips_vlm_flow_camera_compensation=lpips_vlm_flow_camera_compensation,
                save_inspection_artifacts=save_inspection_artifacts,
                inspection_root=inspection_root,
                inspection_run_id=inspection_run_id,
                inspection_sample_id=inspection_sample_id,
            )
            report.update(
                {
                    "reference_video": context.reference_video,
                    "candidate_video": context.candidate_video,
                    "reference_metadata": context.reference.metadata.to_dict(),
                    "candidate_metadata": context.candidate.metadata.to_dict(),
                    "shared_duration_sec": context.shared_duration_sec,
                    "eval_fps": context.eval_fps,
                    "max_pairs": context.max_pairs,
                    "alignment_mode": context.alignment_mode,
                    "num_pairs_sampled": context.num_pairs,
                    "artifacts": context.artifacts,
                }
            )
            report["results"] = run_modules(
                context=context,
                modules=normalized_modules,
                device=resolved_device,
            )
            return report
        report["results"] = {
            module_name: {
                "module": module_name,
                "error": "ref_video is required for this module",
            }
            for module_name in normalized_modules
        }
        return report

    context = build_evaluation_context(
        reference_video=ref_video_path,
        candidate_video=video_path,
        eval_fps=eval_fps,
        max_pairs=max_pairs,
        text_prompt=text_prompt,
        candidate_fps_override=candidate_fps_override,
    )
    _merge_lpips_artifacts(
        context,
        lpips_sampling=lpips_sampling,
        lpips_temporal_fractions=lpips_temporal_fractions,
        lpips_temporal_labels=lpips_temporal_labels,
        lpips_event_frames_before=lpips_event_frames_before,
        lpips_event_frames_after=lpips_event_frames_after,
        lpips_event_offsets=lpips_event_offsets,
        lpips_vlm_model=lpips_vlm_model,
        lpips_vlm_coarse_model=lpips_vlm_coarse_model,
        lpips_vlm_max_grid_frames=lpips_vlm_max_grid_frames,
        lpips_vlm_sampling_strategy=lpips_vlm_sampling_strategy,
        lpips_vlm_coarse_grid_frames=lpips_vlm_coarse_grid_frames,
        lpips_vlm_side_relabel=lpips_vlm_side_relabel,
        lpips_vlm_side_model=lpips_vlm_side_model,
        lpips_vlm_side_window_sec=lpips_vlm_side_window_sec,
        lpips_vlm_side_max_frames=lpips_vlm_side_max_frames,
        lpips_vlm_window_sec=lpips_vlm_window_sec,
        lpips_vlm_window_overlap_sec=lpips_vlm_window_overlap_sec,
        lpips_vlm_window_max_frames=lpips_vlm_window_max_frames,
        lpips_vlm_window_merge_gap_frames=lpips_vlm_window_merge_gap_frames,
        lpips_vlm_event_split_gap_frames=lpips_vlm_event_split_gap_frames,
        lpips_vlm_event_split_min_cluster_size=lpips_vlm_event_split_min_cluster_size,
        lpips_vlm_refine_window_sec=lpips_vlm_refine_window_sec,
        lpips_vlm_refine_max_frames_per_window=lpips_vlm_refine_max_frames_per_window,
        lpips_vlm_prompt_version=lpips_vlm_prompt_version,
        lpips_vlm_max_reference_events=lpips_vlm_max_reference_events,
        lpips_vlm_reference_trim_sec=lpips_vlm_reference_trim_sec,
        lpips_vlm_judge_hint=lpips_vlm_judge_hint,
        lpips_vlm_scene_key=lpips_vlm_scene_key,
        lpips_causal_event_source=lpips_causal_event_source,
        lpips_causal_lpips_pairing=lpips_causal_lpips_pairing,
        lpips_vlm_pair_accept_cost=lpips_vlm_pair_accept_cost,
        lpips_vlm_skip_cost=lpips_vlm_skip_cost,
        lpips_vlm_parallel_videos=lpips_vlm_parallel_videos,
        lpips_vlm_side_relabel_concurrency=lpips_vlm_side_relabel_concurrency,
        lpips_vlm_coarse_window_concurrency=lpips_vlm_coarse_window_concurrency,
        lpips_vlm_coarse_reasoning_effort=lpips_vlm_coarse_reasoning_effort,
        lpips_vlm_fine_reasoning_effort=lpips_vlm_fine_reasoning_effort,
        lpips_vlm_side_reasoning_effort=lpips_vlm_side_reasoning_effort,
        lpips_vlm_use_flow_proposals=lpips_vlm_use_flow_proposals,
        lpips_vlm_flow_proposal_stride=lpips_vlm_flow_proposal_stride,
        lpips_vlm_flow_peak_threshold=lpips_vlm_flow_peak_threshold,
        lpips_vlm_flow_topk_windows=lpips_vlm_flow_topk_windows,
        lpips_vlm_flow_window_sec=lpips_vlm_flow_window_sec,
        lpips_vlm_flow_camera_compensation=lpips_vlm_flow_camera_compensation,
        save_inspection_artifacts=save_inspection_artifacts,
        inspection_root=inspection_root,
        inspection_run_id=inspection_run_id,
        inspection_sample_id=inspection_sample_id,
    )
    _merge_liquid_artifacts(
        context,
        liquid_cache_root=liquid_cache_root,
        liquid_sam2_model=liquid_sam2_model,
        liquid_sam2_prompt=liquid_sam2_prompt,
        liquid_bottle_prompt=liquid_bottle_prompt,
        liquid_piv_window_size=liquid_piv_window_size,
        liquid_piv_overlap=liquid_piv_overlap,
        liquid_max_pairs=liquid_max_pairs,
        liquid_camera_compensation=liquid_camera_compensation,
        liquid_camera_regime=liquid_camera_regime,
        liquid_flow_estimator=liquid_flow_estimator,
        liquid_flow_min_mask_frac=liquid_flow_min_mask_frac,
    )
    context.artifacts.update(
        {
            "shadow_text_prompt": shadow_text_prompt,
            "shadow_anchor_label": shadow_anchor_label,
            "shadow_hud_preset": shadow_hud_preset,
            "eoc_text_prompt": eoc_text_prompt,
            "eoc_character_prompt": eoc_character_prompt,
            "eoc_tau_det": eoc_tau_det,
            "eoc_tau_mask": eoc_tau_mask,
            "eoc_tau_id": eoc_tau_id,
            "eoc_max_objects": eoc_max_objects,
            "eoc_depth_model": eoc_depth_model,
            "eoc_gd_model": eoc_gd_model,
            "eoc_unity_object_transforms_json": eoc_unity_object_transforms_json,
            "eoc_unity_poses_npy": eoc_unity_poses_npy,
            "eoc_unity_intrinsics_npy": eoc_unity_intrinsics_npy,
            "eoc_unity_object_diameters_json": eoc_unity_object_diameters_json,
            "kin_text_prompt": kin_text_prompt,
            "kin_static_anchor_labels": kin_static_anchor_labels,
            "kin_track_thresh": kin_track_thresh,
            "kin_track_buffer": kin_track_buffer,
            "kin_match_thresh": kin_match_thresh,
            "kin_min_track_len": kin_min_track_len,
            "kin_gd_threshold": kin_gd_threshold,
            "kin_gd_text_threshold": kin_gd_text_threshold,
            "kin_gd_model": kin_gd_model,
            "kin_unity_object_transforms_json": kin_unity_object_transforms_json,
            "kin_unity_poses_npy": kin_unity_poses_npy,
            "kin_unity_intrinsics_npy": kin_unity_intrinsics_npy,
            "kin_unity_object_masses_json": kin_unity_object_masses_json,
            "kin_color_filter": kin_color_filter,
            "kin_color_hue_tolerance_deg": kin_color_hue_tolerance_deg,
            "kin_depth_lift": kin_depth_lift,
            "kin_depth_lift_window": kin_depth_lift_window,
            "kin_depth_model": kin_depth_model,
            "unity_log_json": unity_log_json,
            "camera_intrinsics_json": camera_intrinsics_json,
            "unity_flow_dir": unity_flow_dir,
            "unity_shadow_dir": unity_shadow_dir,
            "diffusionlight_repo": diffusionlight_repo,
            "run_diffusionlight": run_diffusionlight,
            "diffusionlight_output_root": diffusionlight_output_root,
            "diffusionlight_ref_hdr_dir": diffusionlight_ref_hdr_dir,
            "diffusionlight_candidate_hdr_dir": diffusionlight_candidate_hdr_dir,
            "sam2_model": sam2_model,
            "control_camera_compensation": control_camera_compensation,
            "control_global_motion_mode": control_global_motion_mode,
            "control_motion_mask_threshold": control_motion_mask_threshold,
            "control_motion_mask_percentile": control_motion_mask_percentile,
            "control_bad_pixel_thresholds": (
                list(control_bad_pixel_thresholds)
                if control_bad_pixel_thresholds is not None
                else [1.0, 3.0]
            ),
            "synthetic_calibration": synthetic_calibration,
            "synthetic_action_file": synthetic_action_file,
            "synthetic_infer_actions": synthetic_infer_actions,
            "synthetic_use_depth": synthetic_use_depth,
            "ptlflow_root": resolve_ptlflow_root(ptlflow_root),
            "synthetic_grid_size": synthetic_grid_size,
            "lpips_sampling": lpips_sampling,
            "lpips_vlm_model": lpips_vlm_model,
            "lpips_vlm_max_grid_frames": lpips_vlm_max_grid_frames,
            "lpips_vlm_prompt_version": lpips_vlm_prompt_version,
            "lpips_event_frames_before": lpips_event_frames_before,
            "lpips_event_frames_after": lpips_event_frames_after,
            "lpips_event_offsets": lpips_event_offsets,
            "causal_match_mode": causal_match_mode,
            "lpips_vlm_pair_accept_cost": lpips_vlm_pair_accept_cost,
            "lpips_vlm_skip_cost": lpips_vlm_skip_cost,
            "save_inspection_artifacts": save_inspection_artifacts,
            "inspection_root": inspection_root,
            "inspection_run_id": inspection_run_id,
            "inspection_sample_id": inspection_sample_id,
        }
    )
    resolved_scene_key, resolved_scene_hints = resolve_scene_hints(
        scene_key=lpips_vlm_scene_key,
        reference_video=context.reference_video,
        candidate_video=context.candidate_video,
        explicit_hints=lpips_vlm_scene_hints,
    )
    _, resolved_scene_config = resolve_scene_config(
        scene_key=resolved_scene_key,
        reference_video=context.reference_video,
        candidate_video=context.candidate_video,
    )
    if resolved_scene_key is not None:
        context.artifacts["lpips_vlm_scene_key"] = resolved_scene_key
    if resolved_scene_config:
        context.artifacts["lpips_vlm_scene_config"] = resolved_scene_config

    for _k, _v in (
        ("lpips_vlm_scene_hints", resolved_scene_hints),
        ("lpips_vlm_contact_types", lpips_vlm_contact_types),
        ("lpips_vlm_directional_sides", lpips_vlm_directional_sides),
        ("lpips_vlm_system_prompt", lpips_vlm_system_prompt),
        ("lpips_vlm_judge_user_prompt_template", lpips_vlm_judge_user_prompt_template),
        ("causal_step_qa_dataset", causal_step_qa_dataset),
        ("causal_step_video_id", causal_step_video_id),
        ("causal_step_answers_cache_dir", causal_step_answers_cache_dir),
        ("causalstep_root", causalstep_root),
    ):
        if _v is not None:
            context.artifacts[_k] = _v
    context.artifacts["causal_step_vlm_model"] = causal_step_vlm_model
    context.artifacts["causal_step_vlm_prompt_version"] = causal_step_vlm_prompt_version
    context.artifacts["causal_step_vlm_num_frames"] = int(causal_step_vlm_num_frames)
    context.artifacts["causal_step_wm_full_video"] = bool(causal_step_wm_full_video)
    context.artifacts["causal_step_causal_fail_advance"] = bool(
        causal_step_causal_fail_advance
    )
    context.artifacts["causal_step_judge"] = causal_step_judge

    report.update(
        {
            "reference_video": context.reference_video,
            "candidate_video": context.candidate_video,
            "reference_metadata": context.reference.metadata.to_dict(),
            "candidate_metadata": context.candidate.metadata.to_dict(),
            "shared_duration_sec": context.shared_duration_sec,
            "eval_fps": context.eval_fps,
            "max_pairs": context.max_pairs,
            "alignment_mode": context.alignment_mode,
            "num_pairs_sampled": context.num_pairs,
            "artifacts": context.artifacts,
        }
    )
    report["results"] = run_modules(
        context=context,
        modules=normalized_modules,
        device=resolved_device,
    )
    return report
