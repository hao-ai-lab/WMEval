from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ._causal_step_io import ensure_causalstep_importable, write_answers
from ._causal_step_judge import Judge, JudgeRequest


def _frame_window(start_sec: float, end_sec: float, fps: float) -> tuple[int, int]:
    """Convert a [start_sec, end_sec) window into half-open frame indices [start, end).

    Rounds to nearest integer frame. Guarantees at least one frame in the window so
    downstream samplers never see end <= start.
    """
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    if start_frame < 0:
        start_frame = 0
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def run_causal_step_for_video(
    *,
    qa_dataset_path: Path,
    video_id: str,
    video_path: str,
    judge: Judge,
    answers_out_dir: Path,
    causal_fail_advance: bool,
    fps: float,
    causalstep_root: Path | None = None,
    force_full_video: bool = False,
    repeats: int = 1,
) -> Dict[str, Any]:
    """Judge every QA for `video_id`, run CausalStep, return per-video metrics dict.

    `fps` is required so segment time windows can be converted to the half-open frame
    index range `[segment_frames_start, segment_frames_end)` that frame-consuming
    judges (e.g. `OpenAIJudge`) need to slice the correct window out of the video.

    When ``force_full_video`` is True, every JudgeRequest's frame window is replaced
    with the whole video span. The QA still references segment_index, but the judge
    sees the full clip for every question — used for world-model videos that are not
    temporally aligned with the Unity reference (e.g. Project Genie's manually-driven
    rollouts), where Unity-derived segment windows would slice the wrong frames.

    ``repeats`` is how many independent judge passes to run over the same video.
    CSR is all-or-nothing per video, so a single pass can only ever report 0 or
    100; the published per-shape figures (e.g. Unity CSR 60 on `glancing`) are
    means over five passes. Every pass is handed to ``compute_all_metrics`` as a
    separate result, which is what turns CSR into successes/repeats and AMCL into
    a mean. Each pass carries its own ``repeat_index`` so caching judges treat it
    as a distinct request.

    Returns keys: csr, amcl, mcl, rf, ws, dua, icra (lowercase, matching
    causalstep.metrics.compute_all_metrics output and `compute_delta` input).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    ensure_causalstep_importable(str(causalstep_root) if causalstep_root else None)

    from causalstep.schema import load_dataset
    from causalstep.model_interface import PrecomputedModel
    from causalstep.evaluator import (
        evaluate_chain,
        evaluate_descriptive_isolated,
        evaluate_causal_isolated,
    )
    from causalstep.metrics import compute_all_metrics

    videos = load_dataset(str(qa_dataset_path))
    try:
        video = next(v for v in videos if v.video_id == video_id)
    except StopIteration as exc:
        raise ValueError(f"video_id {video_id!r} not in {qa_dataset_path}") from exc

    full_start_frame = 0
    full_end_frame = 0
    full_start_sec = 0.0
    full_end_sec = 0.0
    if force_full_video:
        import cv2  # local import: only paid when the flag is on

        cap = cv2.VideoCapture(video_path)
        try:
            full_end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        finally:
            cap.release()
        full_end_sec = full_end_frame / fps if fps else float(full_end_frame)

    def _judge_all_qas(repeat_index: int) -> dict[tuple[str, int, str], str]:
        """One full judge pass over every QA in this video."""
        answers_map: dict[tuple[str, int, str], str] = {}
        for seg in video.segments:
            if force_full_video:
                start_sec, end_sec = full_start_sec, full_end_sec
                start_frame, end_frame = full_start_frame, full_end_frame
            else:
                start_sec = float(seg.start_time) if seg.start_time is not None else 0.0
                end_sec = float(seg.end_time) if seg.end_time is not None else 0.0
                start_frame, end_frame = _frame_window(start_sec, end_sec, fps)

            desc_qa = video.descriptive_qas.get(seg.index)
            if desc_qa is not None:
                letter = judge.answer(
                    JudgeRequest(
                        video_path=video_path,
                        segment_index=seg.index,
                        qa_type="descriptive",
                        question=desc_qa.question,
                        choices=desc_qa.choices,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        segment_frames_start=start_frame,
                        segment_frames_end=end_frame,
                        repeat_index=repeat_index,
                    )
                )
                answers_map[(video_id, seg.index, "descriptive")] = letter
            causal_qa = video.causal_qas.get(seg.index)
            if causal_qa is not None:
                letter = judge.answer(
                    JudgeRequest(
                        video_path=video_path,
                        segment_index=seg.index,
                        qa_type="causal",
                        question=causal_qa.question,
                        choices=causal_qa.choices,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        segment_frames_start=start_frame,
                        segment_frames_end=end_frame,
                        repeat_index=repeat_index,
                    )
                )
                answers_map[(video_id, seg.index, "causal")] = letter

        return answers_map

    answers_out_dir = Path(answers_out_dir)
    answers_out_dir.mkdir(parents=True, exist_ok=True)

    chain_results = []
    isolated_descs = []
    isolated_causals = []
    for repeat_index in range(repeats):
        answers_map = _judge_all_qas(repeat_index)
        suffix = "" if repeat_index == 0 else f".r{repeat_index}"
        answers_path = answers_out_dir / f"{video_id}_answers{suffix}.json"
        write_answers(str(answers_path), answers_map, video_id=video_id)

        model = PrecomputedModel(answers=answers_map, video_id=video_id)
        chain_results.append(
            evaluate_chain(video, model, causal_fail_advance=causal_fail_advance)
        )
        isolated_descs.append(evaluate_descriptive_isolated(video, model))
        isolated_causals.append(evaluate_causal_isolated(video, model))

    agg = compute_all_metrics(chain_results, isolated_descs, isolated_causals)
    return dict(agg.to_dict()["metrics"])
