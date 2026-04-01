from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from .registry import normalize_modules, run_modules
from .video_io import build_evaluation_context


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
        }
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
