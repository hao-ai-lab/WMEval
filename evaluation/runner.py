from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from .context import EvaluationContext
from .registry import normalize_modules, run_modules
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
    lpips_vlm_max_grid_frames: int,
    lpips_vlm_prompt_version: str,
    lpips_vlm_max_reference_events: Optional[int],
    lpips_vlm_reference_trim_sec: Optional[float],
    lpips_vlm_judge_hint: str,
    lpips_causal_event_source: str,
    lpips_causal_lpips_pairing: Optional[str],
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
    context.artifacts["lpips_vlm_max_grid_frames"] = int(lpips_vlm_max_grid_frames)
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
    context.artifacts["lpips_causal_event_source"] = lpips_causal_event_source.strip().lower()
    if lpips_causal_lpips_pairing is not None:
        raw = str(lpips_causal_lpips_pairing).strip().lower()
        if raw and raw != "auto":
            context.artifacts["lpips_causal_lpips_pairing"] = raw
    context.artifacts["save_inspection_artifacts"] = bool(save_inspection_artifacts)
    context.artifacts["inspection_root"] = inspection_root
    context.artifacts["inspection_run_id"] = inspection_run_id
    if inspection_sample_id is not None:
        context.artifacts["inspection_sample_id"] = inspection_sample_id


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
    lpips_sampling: str = "temporal",
    lpips_temporal_fractions: Optional[Sequence[float]] = None,
    lpips_temporal_labels: Optional[Sequence[str]] = None,
    lpips_event_frames_before: int = 2,
    lpips_event_frames_after: int = 2,
    lpips_event_offsets: Optional[Sequence[int]] = None,
    lpips_vlm_model: str = "gpt-4o",
    lpips_vlm_max_grid_frames: int = 48,
    lpips_vlm_prompt_version: str = "causal_collision_v2",
    lpips_vlm_max_reference_events: Optional[int] = None,
    lpips_vlm_reference_trim_sec: Optional[float] = None,
    lpips_vlm_judge_hint: str = "",
    lpips_causal_event_source: str = "both",
    lpips_causal_lpips_pairing: Optional[str] = None,
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
    causal_match_mode: str = "strict_fallback_order",
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
                lpips_vlm_max_grid_frames=lpips_vlm_max_grid_frames,
                lpips_vlm_prompt_version=lpips_vlm_prompt_version,
                lpips_vlm_max_reference_events=lpips_vlm_max_reference_events,
                lpips_vlm_reference_trim_sec=lpips_vlm_reference_trim_sec,
                lpips_vlm_judge_hint=lpips_vlm_judge_hint,
                lpips_causal_event_source=lpips_causal_event_source,
                lpips_causal_lpips_pairing=lpips_causal_lpips_pairing,
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
        lpips_vlm_max_grid_frames=lpips_vlm_max_grid_frames,
        lpips_vlm_prompt_version=lpips_vlm_prompt_version,
        lpips_vlm_max_reference_events=lpips_vlm_max_reference_events,
        lpips_vlm_reference_trim_sec=lpips_vlm_reference_trim_sec,
        lpips_vlm_judge_hint=lpips_vlm_judge_hint,
        lpips_causal_event_source=lpips_causal_event_source,
        lpips_causal_lpips_pairing=lpips_causal_lpips_pairing,
        save_inspection_artifacts=save_inspection_artifacts,
        inspection_root=inspection_root,
        inspection_run_id=inspection_run_id,
        inspection_sample_id=inspection_sample_id,
    )
    context.artifacts.update(
        {
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
            "ptlflow_root": ptlflow_root if ptlflow_root is not None else "/workspace/ptlflow",
            "synthetic_grid_size": synthetic_grid_size,
            "lpips_sampling": lpips_sampling,
            "lpips_vlm_model": lpips_vlm_model,
            "lpips_vlm_max_grid_frames": lpips_vlm_max_grid_frames,
            "lpips_vlm_prompt_version": lpips_vlm_prompt_version,
            "lpips_event_frames_before": lpips_event_frames_before,
            "lpips_event_frames_after": lpips_event_frames_after,
            "lpips_event_offsets": lpips_event_offsets,
            "causal_match_mode": causal_match_mode,
            "save_inspection_artifacts": save_inspection_artifacts,
            "inspection_root": inspection_root,
            "inspection_run_id": inspection_run_id,
            "inspection_sample_id": inspection_sample_id,
        }
    )
    for _k, _v in (
        ("lpips_vlm_scene_hints", lpips_vlm_scene_hints),
        ("lpips_vlm_contact_types", lpips_vlm_contact_types),
        ("lpips_vlm_directional_sides", lpips_vlm_directional_sides),
        ("lpips_vlm_system_prompt", lpips_vlm_system_prompt),
        ("lpips_vlm_judge_user_prompt_template", lpips_vlm_judge_user_prompt_template),
    ):
        if _v is not None:
            context.artifacts[_k] = _v

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
