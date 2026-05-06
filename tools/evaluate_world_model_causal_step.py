"""Run CausalStep evaluation across multiple BeamDomino trajectories.

Each line in --manifest is a JSON object with keys:
    video_id, unity_video, wm_video, qa_dataset

Writes {per_video, aggregate} JSON to --out and prints the aggregate block
to stdout. The aggregate dict contains macro-averaged unity/wm/delta metrics
across all trajectories that evaluated without error.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CausalStep is added to sys.path lazily by ensure_causalstep_importable, which
# honors the optional `causalstep_root` artifact. Don't insert it here too —
# that would let an out-of-tree --causalstep-root override silently lose to the
# default path depending on insertion order.

from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata
from evaluation.registry import run_modules


def _probe_video(path: str) -> tuple[float, int]:
    """Read fps + total frame count off the file via cv2.

    Both Unity (60 fps) and Genie (20 fps) clips coexist in the same manifest,
    so the runner cannot assume a global fps when converting segment seconds
    to frame indices. Returns ``(fps, frame_count)``; falls back to ``(30.0, 1)``
    if cv2 reports nonsense (some codecs do).
    """
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 30.0, 1
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0:
            fps = 30.0
        if n <= 0:
            n = 1
        return fps, n
    finally:
        cap.release()


def _stub_frames(path: str) -> SampledVideoFrames:
    """Minimal SampledVideoFrames stub — causal_step does not consume the
    pixel data, but the runner *does* read ``metadata.fps`` to convert segment
    seconds to frame indices, so the fps must be the file's real rate."""
    fps, n = _probe_video(path)
    duration = n / fps if fps else 1.0
    meta = VideoMetadata(
        path=path, fps=fps, frame_count=n, width=64, height=64, duration_sec=duration
    )
    arr = np.zeros((64, 64, 3), dtype="uint8")
    return SampledVideoFrames(
        metadata=meta, timestamps_sec=[0.0], frame_indices=[0], frames=[arr]
    )


def _run_one(
    entry: dict[str, Any],
    cache_root: Path,
    judge: str,
    causalstep_root: Path | None,
    wm_full_video: bool = False,
    vlm_model: str | None = None,
    vlm_num_frames: int | None = None,
) -> dict[str, Any]:
    ctx = EvaluationContext(
        reference_video=entry["unity_video"],
        candidate_video=entry["wm_video"],
        reference=_stub_frames(entry["unity_video"]),
        candidate=_stub_frames(entry["wm_video"]),
        shared_duration_sec=1.0,
        eval_fps=30.0,
        max_pairs=1,
        artifacts={
            "causal_step_qa_dataset": entry["qa_dataset"],
            "causal_step_video_id": entry["video_id"],
            "causal_step_answers_cache_dir": str(cache_root / entry["video_id"]),
            "causal_step_causal_fail_advance": True,
            "causal_step_wm_full_video": wm_full_video,
            "causal_step_judge": judge,
            **(
                {"causal_step_vlm_model": vlm_model}
                if vlm_model is not None
                else {}
            ),
            **(
                {"causal_step_vlm_num_frames": vlm_num_frames}
                if vlm_num_frames is not None
                else {}
            ),
            **(
                {"causalstep_root": str(causalstep_root)}
                if causalstep_root is not None
                else {}
            ),
        },
    )
    return run_modules(context=ctx, modules=["causal_step"])["causal_step"]


def _run_manifest(
    *,
    manifest_path: Path,
    cache_root: Path,
    judge: str,
    causalstep_root: Path | None,
    wm_full_video: bool = False,
    vlm_model: str | None = None,
    vlm_num_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every entry in the manifest, catching per-entry failures.

    A failing trajectory is reported as ``{"video_id": ..., "error": str(exc)}``
    (matching the shape the downstream ``"error" not in r`` filter expects)
    so one bad trajectory does not halt the whole batch.
    """
    per_video: list[dict[str, Any]] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            per_video.append({"error": f"bad manifest line: {exc}", "line": line})
            continue
        try:
            result = _run_one(entry, cache_root, judge, causalstep_root, wm_full_video, vlm_model, vlm_num_frames)
            if "trajectory" in entry and isinstance(result, dict):
                result["trajectory"] = entry["trajectory"]
        except Exception as exc:  # noqa: BLE001 — one bad video must not kill the batch
            result = {"video_id": entry.get("video_id"), "error": f"{type(exc).__name__}: {exc}"}
        per_video.append(result)
    return per_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL file")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--judge", default="openai", choices=["openai", "fake_always_a"]
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--causalstep-root",
        type=Path,
        default=None,
        help="Override CausalStep package location (defaults to WMEval/imports/CausalStep).",
    )
    parser.add_argument(
        "--wm-full-video",
        action="store_true",
        help=(
            "Skip segment-windowing on the world-model side; the judge sees the whole "
            "WM clip for every QA. Used when WM frames aren't time-aligned to the "
            "Unity reference (e.g. manually-driven Project Genie rollouts)."
        ),
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help=(
            "Override the OpenAI model id used by the judge. If omitted, the module "
            "default (gpt-5.4-mini) is used. Cache keys include the model id so "
            "switching here does not collide with prior cached answers."
        ),
    )
    parser.add_argument(
        "--vlm-num-frames",
        type=int,
        default=None,
        help=(
            "Number of evenly-spaced frames the judge sees per QA. The module "
            "default is 4. Bumping this helps on tight segments (e.g. 0.12s "
            "cascades on the direct_strike shape) where 4 frames undersample "
            "the chain reaction. Cache keys include this number, so changing it "
            "invalidates prior answers."
        ),
    )
    args = parser.parse_args()

    per_video = _run_manifest(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        judge=args.judge,
        causalstep_root=args.causalstep_root,
        wm_full_video=args.wm_full_video,
        vlm_model=args.vlm_model,
        vlm_num_frames=args.vlm_num_frames,
    )

    ok = [r for r in per_video if "error" not in r]
    if not ok:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_video": per_video, "aggregate": None}, indent=2))
        raise SystemExit("all trajectories errored; see per-video results in --out")

    metric_keys = list(ok[0]["delta"].keys())

    def _stat(values: list[float], op: str) -> float:
        if op == "mean":
            return statistics.fmean(values)
        if op == "std":
            return statistics.pstdev(values) if len(values) > 1 else 0.0
        raise ValueError(op)

    agg: dict[str, Any] = {
        "unity_mean": {k: _stat([r["unity"][k] for r in ok], "mean") for k in metric_keys},
        "wm_mean": {k: _stat([r["wm"][k] for r in ok], "mean") for k in metric_keys},
        "delta_mean": {k: _stat([r["delta"][k] for r in ok], "mean") for k in metric_keys},
        "n_trajectories": len(ok),
        "n_errors": len(per_video) - len(ok),
        "delta_convention": ok[0].get(
            "delta_convention", "positive = WM worse than Unity"
        ),
    }

    by_traj: dict[str, list[dict[str, Any]]] = {}
    for r in ok:
        t = r.get("trajectory")
        if t is not None:
            by_traj.setdefault(str(t), []).append(r)
    if by_traj:
        agg["by_trajectory"] = {
            t: {
                "n_runs": len(rs),
                "delta_mean": {k: _stat([r["delta"][k] for r in rs], "mean") for k in metric_keys},
                "delta_std":  {k: _stat([r["delta"][k] for r in rs], "std")  for k in metric_keys},
                "wm_mean":    {k: _stat([r["wm"][k]    for r in rs], "mean") for k in metric_keys},
                "wm_std":     {k: _stat([r["wm"][k]    for r in rs], "std")  for k in metric_keys},
                "unity_mean": {k: _stat([r["unity"][k] for r in rs], "mean") for k in metric_keys},
            }
            for t, rs in by_traj.items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_video": per_video, "aggregate": agg}, indent=2)
    )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
