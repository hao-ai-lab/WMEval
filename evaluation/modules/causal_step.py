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

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..context import EvaluationContext
from ._causal_step_delta import DELTA_CONVENTION_STRING, compute_delta
from ._causal_step_io import ensure_causalstep_importable
from ._causal_step_judge import Judge, JudgeRequest
from ._causal_step_recorded_judge import RecordedJudge, load_recorded_answers
from ._causal_step_runner import run_causal_step_for_video

logger = logging.getLogger(__name__)


class _AlwaysAJudge:
    """Debug judge: always returns 'A'. Used in unit tests, never in real runs."""

    def answer(self, request: JudgeRequest) -> str:
        return "A"


def _read_window_uint8(
    path: str,
    *,
    start_frame: int,
    end_frame: int,
    num_samples: int,
) -> "np.ndarray":
    """Decode and return ``num_samples`` evenly-spaced frames from
    ``[start_frame, end_frame)`` without buffering the whole video.

    We seek to ``start_frame`` and read sequentially up to the last sampled
    index, keeping only the picks, rather than buffering the clip (this is
    called once per QA). Out-of-range windows are clamped
    so a slightly-too-long Unity reference still yields valid frames; a
    window entirely past the last available frame raises (silently sampling
    from the video tail would feed the judge frames from the wrong segment).
    """
    import cv2
    import numpy as np

    if end_frame <= start_frame:
        raise ValueError(f"bad segment: {start_frame}..{end_frame}")

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Some codecs report unreliable frame counts; only enforce the
        # entirely-past-end check when total looks trustworthy.
        if total > 0 and start_frame >= total:
            raise ValueError(
                f"segment [{start_frame}, {end_frame}) is entirely outside video of length {total}"
            )
        s = max(start_frame, 0)
        e = min(end_frame, total) if total > 0 else end_frame
        idx = np.linspace(s, e - 1, num_samples).round().astype(int)
        wanted = sorted(set(int(i) for i in idx))
        picks: dict[int, np.ndarray] = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        cur = s
        last_needed = wanted[-1]
        while cur <= last_needed:
            ok, bgr = cap.read()
            if not ok:
                break
            if cur in picks or cur < wanted[0]:
                cur += 1
                continue
            if cur in wanted:
                picks[cur] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cur += 1

        if not picks:
            raise RuntimeError(
                f"no frames read from {path} in window [{start_frame}, {end_frame})"
            )

        # Map every requested index to the nearest decoded frame so a slightly
        # short clip (last_needed past EOF) still returns `num_samples` frames.
        nearest = sorted(picks.keys())
        out_frames = []
        for i in idx:
            i = int(i)
            if i in picks:
                out_frames.append(picks[i])
            else:
                # find nearest decoded index
                nearest_i = min(nearest, key=lambda k: abs(k - i))
                out_frames.append(picks[nearest_i])
        return np.stack(out_frames).astype("uint8")
    finally:
        cap.release()


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
        model = str(context.artifacts.get("causal_step_vlm_model", "gpt-5.4-mini"))
        prompt_version = str(context.artifacts.get("causal_step_vlm_prompt_version", "v1"))
        # Frame count is part of the cache key (see OpenAIJudge._cache_key), so
        # bumping it invalidates prior answers — that's intentional: at higher
        # sampling the judge often picks differently on tight segments.
        num_frame_samples = int(
            context.artifacts.get("causal_step_vlm_num_frames", 4)
        )
        return OpenAIJudge(
            client=client,
            model=model,
            cache_dir=cache_dir,
            frames_provider=_read_window_uint8,
            num_frame_samples=num_frame_samples,
            prompt_version=prompt_version,
        )
    raise ValueError(f"unknown causal_step_judge kind: {judge_kind!r}")


def _same_video_content(reference: str, candidate: str) -> bool:
    """True when both paths hold the same bytes.

    Path equality is not enough: a self-pair run discovers the candidate from a
    candidates root, so it is usually a copy of the reference at a different
    path. Sizes are compared first so differing files cost one stat call.
    """
    try:
        ref, cand = Path(reference), Path(candidate)
        if ref.resolve() == cand.resolve():
            return True
        if ref.stat().st_size != cand.stat().st_size:
            return False
        return _file_digest(ref) == _file_digest(cand)
    except OSError:
        return False


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    # App. C.5's published protocol: the candidate is scored over its whole clip
    # while the reference stays segment-windowed, because world-model rollouts
    # are not frame-aligned to the Unity log. Default to that.
    wm_full_video = bool(arts.get("causal_step_wm_full_video", True))
    wm_full_video_reason = None
    if wm_full_video and _same_video_content(
        context.reference_video, context.candidate_video
    ):
        # Asking the two sides different questions is only defensible when
        # they are different videos: on an identical pair every delta must be
        # 0 by construction.
        wm_full_video = False
        wm_full_video_reason = (
            "candidate is byte-identical to the reference; segment windowing "
            "forced so the pair measures as the identity it is"
        )
        logger.info("causal_step: %s", wm_full_video_reason)
    causalstep_root = arts.get("causalstep_root")
    ensure_causalstep_importable(str(causalstep_root) if causalstep_root else None)

    judge = _build_judge(context)

    # The reference side is a fixture, not a computation: same video, same QA,
    # pinned judge settings. When the case bundle ships a recording we replay
    # it instead of re-judging, so the reference numbers are reproducible.
    recorded = None
    recorded_path = arts.get("causal_step_reference_answers")
    if recorded_path:
        recorded = load_recorded_answers(recorded_path)

    repeats = arts.get("causal_step_judge_repeats")
    if repeats is None:
        # A recording's pass count is a property of the fixture, so it sets the
        # default rather than being silently truncated to a single pass.
        repeats = recorded.num_passes if recorded is not None else 1
    repeats = int(repeats)

    unity_judge = RecordedJudge(recorded) if recorded is not None else judge

    unity = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id=video_id,
        video_path=context.reference_video,
        judge=unity_judge,
        answers_out_dir=out_root / "unity",
        causal_fail_advance=causal_fail_advance,
        fps=float(context.reference.metadata.fps),
        repeats=repeats,
    )
    wm = run_causal_step_for_video(
        qa_dataset_path=qa_path,
        video_id=video_id,
        video_path=context.candidate_video,
        judge=judge,
        answers_out_dir=out_root / "wm",
        causal_fail_advance=causal_fail_advance,
        fps=float(context.candidate.metadata.fps),
        force_full_video=wm_full_video,
        repeats=repeats,
    )

    delta = compute_delta(unity=unity, wm=wm)

    parse_failures = _aggregate_parse_failures(judge)
    result = {
        "module": "causal_step",
        "video_id": video_id,
        "unity": unity,
        "wm": wm,
        "delta": delta,
        "delta_convention": DELTA_CONVENTION_STRING,
        "vlm_model": str(arts.get("causal_step_vlm_model", "gpt-5.4-mini")),
        "vlm_prompt_version": str(arts.get("causal_step_vlm_prompt_version", "v1")),
        "judge_kind": _judge_kind(context),
        "causal_fail_advance": causal_fail_advance,
        "wm_full_video": wm_full_video,
        "judge_repeats": repeats,
    }
    if wm_full_video_reason is not None:
        result["wm_full_video_reason"] = wm_full_video_reason
    if recorded is not None:
        result["reference_answers_passes"] = recorded.num_passes
        result["reference_answers_judge"] = dict(recorded.judge_settings)
    if parse_failures is not None:
        result["judge_parse_failures"] = parse_failures
    return result


def _aggregate_parse_failures(judge: Judge) -> Optional[int]:
    """Read a parse-failure counter off the judge if it exposes one.

    `OpenAIJudge` exposes `parse_failures` (incremented when the model returns
    text we can't map to a choice letter); the debug `_AlwaysAJudge` and
    `FakeJudge` don't, so the field is omitted from the result for those.
    """
    return getattr(judge, "parse_failures", None)
