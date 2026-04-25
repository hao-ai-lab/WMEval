"""CausalStep evaluation module.

Runs the CausalStep stepwise evaluation twice on the same QA dataset — once
with the Unity GT video as reference, once with the world-model candidate —
using a shared VLM judge, and returns `{unity, wm, delta}` per-metric.

## Causal-failure restart semantics

The CausalStep paper (arXiv 2507.16878) is internally inconsistent about what
happens when a causal QA is answered incorrectly:

- **Prose** (Restart Mechanism section): "the current reasoning chain is
  interrupted, and it must restart from the descriptive understanding QA of
  the **same segment (Sk)**".
- **Pseudocode** (Algorithm 1, causal-fail else branch): `i ← i + 1`, i.e.
  advance to the next segment.

We expose this via the `causal_step_causal_fail_advance` artifact (default
`True`, matching the pseudocode). The effective value is echoed back in the
result dict so downstream analysis can distinguish the two regimes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..context import EvaluationContext
from ._causal_step_delta import compute_delta
from ._causal_step_io import ensure_causalstep_importable
from ._causal_step_judge import Judge, JudgeRequest
from ._causal_step_runner import run_causal_step_for_video

logger = logging.getLogger(__name__)


class _AlwaysAJudge:
    """Debug judge: always returns 'A'. Used in unit tests, never in real runs."""

    def answer(self, request: JudgeRequest) -> str:
        return "A"


def _read_frames_uint8(path: str) -> "np.ndarray":
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames).astype("uint8")


def _build_judge(context: EvaluationContext) -> Judge:
    judge_kind = _judge_kind(context)
    if judge_kind == "fake_always_a":
        logger.warning(
            "causal_step_judge=fake_always_a: using a debug judge that always "
            "returns 'A'. Metrics will NOT reflect video content. Only use this "
            "for plumbing/identity tests."
        )
        return _AlwaysAJudge()
    if judge_kind == "openai":
        from openai import OpenAI
        from ._causal_step_openai_judge import OpenAIJudge

        cache_dir = Path(context.artifacts["causal_step_answers_cache_dir"]) / "vlm_cache"
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model = str(context.artifacts.get("causal_step_vlm_model", "gpt-4o"))
        return OpenAIJudge(
            client=client,
            model=model,
            cache_dir=cache_dir,
            frames_provider=_read_frames_uint8,
        )
    raise ValueError(f"unknown causal_step_judge kind: {judge_kind!r}")


def _judge_kind(context: EvaluationContext) -> str:
    return str(context.artifacts.get("causal_step_judge", "openai")).strip().lower()


def analyze_causal_step_from_context(
    *,
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    del device
    arts = context.artifacts

    required = [
        "causal_step_qa_dataset",
        "causal_step_video_id",
        "causal_step_answers_cache_dir",
    ]
    missing = [k for k in required if k not in arts]
    if missing:
        return {"module": "causal_step", "error": f"missing artifacts: {missing}"}

    if not context.reference_video or not context.candidate_video:
        return {
            "module": "causal_step",
            "error": "both reference_video and candidate_video required",
        }

    qa_path = Path(arts["causal_step_qa_dataset"])
    video_id = str(arts["causal_step_video_id"])
    out_root = Path(arts["causal_step_answers_cache_dir"])
    causal_fail_advance = bool(arts.get("causal_step_causal_fail_advance", True))
    causalstep_root = arts.get("causalstep_root")
    ensure_causalstep_importable(str(causalstep_root) if causalstep_root else None)

    judge = _build_judge(context)

    unity_upper = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id=video_id,
        video_path=context.reference_video,
        judge=judge,
        answers_out_dir=out_root / "unity",
        causal_fail_advance=causal_fail_advance,
        fps=float(context.reference.metadata.fps),
    )
    wm_upper = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id=video_id,
        video_path=context.candidate_video,
        judge=judge,
        answers_out_dir=out_root / "wm",
        causal_fail_advance=causal_fail_advance,
        fps=float(context.candidate.metadata.fps),
    )

    unity_lower = {k.lower(): v for k, v in unity_upper.items()}
    wm_lower = {k.lower(): v for k, v in wm_upper.items()}
    delta = compute_delta(unity=unity_lower, wm=wm_lower)

    return {
        "module": "causal_step",
        "video_id": video_id,
        "unity": unity_upper,
        "wm": wm_upper,
        "delta": delta,
        "delta_convention": "positive = WM worse than Unity",
        "vlm_model": str(arts.get("causal_step_vlm_model", "gpt-4o")),
        "vlm_prompt_version": str(arts.get("causal_step_vlm_prompt_version", "v1")),
        "judge_kind": _judge_kind(context),
        "causal_fail_advance": causal_fail_advance,
    }
