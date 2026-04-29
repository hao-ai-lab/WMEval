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


def _stub_frames(path: str) -> SampledVideoFrames:
    """Minimal SampledVideoFrames stub — causal_step does not consume these
    fields; it reads frames lazily from disk inside the judge when needed."""
    meta = VideoMetadata(
        path=path, fps=30.0, frame_count=1, width=64, height=64, duration_sec=1.0
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
            "causal_step_judge": judge,
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
            result = _run_one(entry, cache_root, judge, causalstep_root)
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
    args = parser.parse_args()

    per_video = _run_manifest(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        judge=args.judge,
        causalstep_root=args.causalstep_root,
    )

    ok = [r for r in per_video if "error" not in r]
    if not ok:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_video": per_video, "aggregate": None}, indent=2))
        raise SystemExit("all trajectories errored; see per-video results in --out")

    metric_keys = list(ok[0]["delta"].keys())
    agg = {
        "unity_mean": {
            k: statistics.fmean(r["unity"][k] for r in ok) for k in metric_keys
        },
        "wm_mean": {
            k: statistics.fmean(r["wm"][k] for r in ok) for k in metric_keys
        },
        "delta_mean": {
            k: statistics.fmean(r["delta"][k] for r in ok) for k in metric_keys
        },
        "n_trajectories": len(ok),
        "n_errors": len(per_video) - len(ok),
        "delta_convention": ok[0].get(
            "delta_convention", "positive = WM worse than Unity"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_video": per_video, "aggregate": agg}, indent=2)
    )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
