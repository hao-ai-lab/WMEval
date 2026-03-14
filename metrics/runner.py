from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from .registry import normalize_modules, run_metric_modules
from .video_io import build_evaluation_context


def ensure_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_world_model_metrics(
    video_path: str,
    ref_video_path: Optional[str],
    text_prompt: str = "",
    modules: Iterable[str] = ("lpips",),
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    device: Optional[str] = None,
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
    report["results"] = run_metric_modules(
        context=context,
        modules=normalized_modules,
        device=resolved_device,
    )
    return report
