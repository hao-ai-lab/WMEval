"""Adapter that drives registry metric modules via the monolithic runner.

Translates one ``(dimension, case, candidate)`` work item into a single
:func:`evaluation.runner.run_world_model_evaluation` call. All metric
configuration flows through an explicit whitelist of runner kwargs —
``METRIC_CONFIG_KEYS`` — so a typo or an unsupported key fails loudly instead
of being silently dropped (which is how PR #16's three-key passthrough lost
configuration).

The metric modules themselves (``evaluation/modules/*``) are untouched; this
layer only assembles kwargs and extracts results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from ..case_loader import LiteCase
from ..candidate_discovery import CandidateRow
from ..suite_loader import DimensionSpec, Suite
from .base import STATUS_COMPLETED, STATUS_ERROR, CaseResult

# Runner kwargs a suite/case/candidate is sanctioned to configure. Anything
# else in metric_config/module_params/candidate params is an error — extend
# this set deliberately when the runner grows new supported knobs.
METRIC_CONFIG_KEYS = frozenset(
    {
        # generic sampling
        "eval_fps",
        "max_pairs",
        "candidate_fps_override",
        # Candidate-specific semantic prompt for reconstructed camera/control
        # regimes that intentionally reuse a Lite reference case.
        "text_prompt",
        # lpips / causal-VLM collision sampling
        "lpips_sampling",
        "lpips_causal_event_source",
        "lpips_causal_lpips_pairing",
        "lpips_reference_events_json",
        "lpips_vlm_model",
        "lpips_vlm_coarse_model",
        "lpips_vlm_prompt_version",
        "lpips_vlm_sampling_strategy",
        "lpips_vlm_max_reference_events",
        "lpips_vlm_reference_trim_sec",
        "lpips_vlm_judge_hint",
        "lpips_vlm_scene_key",
        "lpips_vlm_scene_hints",
        "lpips_vlm_use_flow_proposals",
        "lpips_vlm_flow_camera_compensation",
        # controllability (control_motion)
        "control_camera_compensation",
        "control_global_motion_mode",
        "control_motion_mask_threshold",
        "control_motion_mask_percentile",
        "control_bad_pixel_thresholds",
        "synthetic_calibration",
        "synthetic_use_depth",
        "synthetic_grid_size",
        "ptlflow_root",
        # causal step
        "causal_step_vlm_model",
        "causal_step_vlm_prompt_version",
        "causal_step_judge",
        "causal_step_causal_fail_advance",
        "causal_step_answers_cache_dir",
        "causal_step_video_id",
        "causal_step_vlm_num_frames",
        "causal_step_wm_full_video",
        "causal_step_judge_repeats",
        "causal_match_mode",
        "causalstep_root",
        # liquid family
        "liquid_cache_root",
        "liquid_sam2_model",
        "liquid_sam2_prompt",
        "liquid_bottle_prompt",
        "liquid_piv_window_size",
        "liquid_piv_overlap",
        "liquid_max_pairs",
        "liquid_camera_compensation",
        "liquid_camera_regime",
        "liquid_flow_estimator",
        "liquid_flow_min_mask_frac",
        # shadow consistency
        "shadow_text_prompt",
        "shadow_anchor_label",
        "shadow_hud_preset",
        # shared model knobs
        "sam2_model",
        # environment_object_consistency (paper appendix B.1)
        "eoc_text_prompt",
        "eoc_character_prompt",
        "eoc_tau_det",
        "eoc_tau_mask",
        "eoc_tau_id",
        "eoc_max_objects",
        "eoc_fixed_anchor_labels",
        "eoc_depth_model",
        "eoc_gd_model",
        "eoc_unity_object_transforms_json",
        "eoc_unity_poses_npy",
        "eoc_unity_intrinsics_npy",
        "eoc_unity_object_diameters_json",
        # kinematics_mechanics (paper appendix B.3)
        "kin_text_prompt",
        "kin_static_anchor_labels",
        "kin_track_thresh",
        "kin_track_buffer",
        "kin_match_thresh",
        "kin_min_track_len",
        "kin_gd_threshold",
        "kin_gd_text_threshold",
        "kin_gd_model",
        "kin_unity_object_transforms_json",
        "kin_unity_poses_npy",
        "kin_unity_intrinsics_npy",
        "kin_unity_object_masses_json",
        "kin_plane_calibration_json",
        "kin_color_filter",
        "kin_color_hue_tolerance_deg",
        "kin_track_point",
        "kin_depth_lift",
        "kin_depth_lift_window",
        "kin_depth_model",
        "kin_mechanics_estimator",
        "kin_event_alignment_tolerance_frames",
        "kin_event_min_neighbor_ratio",
    }
)

# Reference-artifact names understood by this adapter (case.yaml
# reference.artifacts keys).
ARTIFACT_SYNTHETIC_ACTION_DIR = "synthetic_action_dir"
ARTIFACT_ACTIONS_FILE = "actions_file"
ARTIFACT_QA_DATASET = "qa_dataset"
# Optional: pre-recorded reference-side judge answers, several independent
# passes per video. Present => the reference side replays them instead of
# paying the live judge (see _causal_step_recorded_judge).
ARTIFACT_REFERENCE_ANSWERS = "judge_answers"


class MetricConfigError(ValueError):
    pass


def merge_metric_config(
    dimension: DimensionSpec,
    case: LiteCase,
    candidate: CandidateRow,
) -> dict[str, Any]:
    """Merge dimension metric_config < case module_params < candidate params.

    Every key must be inside :data:`METRIC_CONFIG_KEYS`.
    """
    merged: dict[str, Any] = dict(dimension.metric_config)
    for module_name in dimension.modules:
        params = case.module_params.get(module_name)
        if params:
            merged.update(dict(params))
    if candidate.params:
        merged.update(dict(candidate.params))

    unknown = sorted(set(merged) - METRIC_CONFIG_KEYS)
    if unknown:
        raise MetricConfigError(
            f"{dimension.dimension_id}/{case.case_id}: metric configuration has "
            f"keys outside the sanctioned runner whitelist: {unknown}"
        )
    return merged


def _resolve_synthetic_action_file(case: LiteCase) -> Path | None:
    """Mirror PR #16's lookup: <synthetic_action_dir>/action.npy first, then
    the raw actions_file."""
    action_dir = case.reference_artifacts.get(ARTIFACT_SYNTHETIC_ACTION_DIR)
    if action_dir is not None:
        action_npy = Path(action_dir) / "action.npy"
        if action_npy.exists():
            return action_npy
    actions_file = case.reference_artifacts.get(ARTIFACT_ACTIONS_FILE)
    if actions_file is not None:
        return Path(actions_file)
    return None


def build_runner_kwargs(
    *,
    dimension: DimensionSpec,
    case: LiteCase,
    candidate: CandidateRow,
    candidate_video: Path,
    device: str | None,
) -> dict[str, Any]:
    merged_config = merge_metric_config(dimension, case, candidate)

    # Reference-free dimensions reuse the candidate as a dummy reference to
    # satisfy EvaluationContext construction (PR #18's CLI convention); the
    # output row records reference_video=null so downstream analysis cannot
    # mistake it for a real reference.
    reference_video = case.reference_video if dimension.requires_reference else None
    ref_video_path = str(reference_video) if reference_video else str(candidate_video)

    kwargs: dict[str, Any] = {
        "video_path": str(candidate_video),
        "ref_video_path": ref_video_path,
        "text_prompt": case.prompt or "",
        "modules": tuple(dimension.modules),
        "device": device,
    }

    if "control_motion" in dimension.modules:
        synthetic_action_file = _resolve_synthetic_action_file(case)
        kwargs["synthetic_action_file"] = (
            str(synthetic_action_file) if synthetic_action_file is not None else None
        )
        kwargs["synthetic_infer_actions"] = synthetic_action_file is None

    if "causal_step" in dimension.modules:
        qa_dataset = case.reference_artifacts.get(ARTIFACT_QA_DATASET)
        if qa_dataset is None:
            raise MetricConfigError(
                f"{dimension.dimension_id}/{case.case_id}: causal_step requires a "
                f"reference.artifacts.{ARTIFACT_QA_DATASET} entry in case.yaml"
            )
        kwargs["causal_step_qa_dataset"] = str(qa_dataset)
        recorded = case.reference_artifacts.get(ARTIFACT_REFERENCE_ANSWERS)
        if recorded is not None:
            kwargs["causal_step_reference_answers"] = str(recorded)

    kwargs.update(merged_config)
    return kwargs


class RegistryAdapter:
    """DimensionEvaluator backed by run_world_model_evaluation."""

    def __init__(self, runner: Callable[..., Mapping[str, Any]] | None = None) -> None:
        self._runner = runner

    def _resolve_runner(self) -> Callable[..., Mapping[str, Any]]:
        if self._runner is not None:
            return self._runner
        from evaluation.runner import run_world_model_evaluation

        return run_world_model_evaluation

    def evaluate(
        self,
        *,
        suite: Suite,
        dimension: DimensionSpec,
        case: LiteCase,
        candidate: CandidateRow,
        candidate_video: Path,
        device: str | None = None,
    ) -> CaseResult:
        kwargs = build_runner_kwargs(
            dimension=dimension,
            case=case,
            candidate=candidate,
            candidate_video=candidate_video,
            device=device,
        )
        runner = self._resolve_runner()
        report = dict(runner(**kwargs))

        results = report.get("results")
        if not isinstance(results, Mapping):
            return CaseResult(
                status=STATUS_ERROR,
                error="runner report has no results mapping",
                full_report=report,
            )
        module_errors = {
            module_name: str(module_result["error"])
            for module_name, module_result in results.items()
            if isinstance(module_result, Mapping) and "error" in module_result
        }
        if module_errors:
            return CaseResult(
                status=STATUS_ERROR,
                error="; ".join(
                    f"{module_name}: {message}"
                    for module_name, message in sorted(module_errors.items())
                ),
                full_report=report,
            )

        from ..aggregate import extract_metrics

        return CaseResult(
            status=STATUS_COMPLETED,
            metrics=extract_metrics(dimension, results),
            full_report=report,
        )
