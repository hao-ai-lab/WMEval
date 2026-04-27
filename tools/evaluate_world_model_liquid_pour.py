"""Run liquid-pour evaluation across multiple LiquidContainer trajectories.

Each line in --manifest is a JSON object with keys:
    video_id, unity_video, wm_video, unity_log

Optionally: prompt, scene_key, camera_regime.

Writes {per_video, aggregate, config} JSON to --out and prints the aggregate
block to stdout. Per-trajectory errors land in `per_video` with an `error` key
so a single bad trajectory does not halt the batch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata  # noqa: E402
from evaluation.registry import run_modules  # noqa: E402
from evaluation.video_io import build_evaluation_context  # noqa: E402


_DEFAULT_MODULES = ("liquid_state", "liquid_velocity", "liquid_coupling")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT), check=False
        )
        sha = out.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def _build_context(entry: dict[str, Any], args: argparse.Namespace) -> EvaluationContext:
    context = build_evaluation_context(
        reference_video=entry["unity_video"],
        candidate_video=entry["wm_video"],
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        text_prompt=entry.get("prompt", ""),
    )
    context.artifacts.update(
        {
            "liquid_state_unity_log_path": entry["unity_log"],
            "liquid_cache_root": str(args.liquid_cache_root),
            "liquid_sam2_model": args.liquid_sam2_model,
            "liquid_sam2_prompt": args.liquid_sam2_prompt,
            "liquid_piv_window_size": int(args.liquid_piv_window_size),
            "liquid_piv_overlap": int(args.liquid_piv_overlap),
            "liquid_max_pairs": int(args.liquid_max_pairs),
            "liquid_pour_threshold_deg": float(args.liquid_pour_threshold_deg),
            "liquid_k_pour": float(args.liquid_k_pour),
            "liquid_bottle_local_volume_ml": float(args.liquid_bottle_local_volume_ml),
            "liquid_camera_compensation": bool(args.liquid_camera_compensation),
            "liquid_camera_regime": str(args.liquid_camera_regime),
            "liquid_coupling_max_lag_sec": float(args.liquid_coupling_max_lag_sec),
        }
    )
    return context


def _validate_entry(entry: dict[str, Any]) -> str | None:
    for k in ("video_id", "unity_video", "wm_video", "unity_log"):
        if k not in entry:
            return f"manifest entry missing key {k!r}"
    for k in ("unity_video", "wm_video", "unity_log"):
        if not Path(entry[k]).exists():
            return f"file not found ({k}): {entry[k]}"
    return None


def _stub_frames(path: str) -> SampledVideoFrames:
    import numpy as np
    meta = VideoMetadata(path=path, fps=30.0, frame_count=1, width=64, height=64, duration_sec=1.0)
    arr = np.zeros((64, 64, 3), dtype="uint8")
    return SampledVideoFrames(metadata=meta, timestamps_sec=[0.0], frame_indices=[0], frames=[arr])


def _build_stub_context(entry: dict[str, Any], args: argparse.Namespace) -> EvaluationContext:
    """Used when video files are missing — produces a context whose modules will return errors."""
    ctx = EvaluationContext(
        reference_video=entry["unity_video"],
        candidate_video=entry["wm_video"],
        reference=_stub_frames(entry["unity_video"]),
        candidate=_stub_frames(entry["wm_video"]),
        shared_duration_sec=1.0, eval_fps=args.eval_fps, max_pairs=args.max_pairs,
        artifacts={
            "liquid_state_unity_log_path": entry.get("unity_log", ""),
            "liquid_cache_root": str(args.liquid_cache_root),
        },
    )
    return ctx


def _run_one(entry: dict[str, Any], args: argparse.Namespace, modules: list[str]) -> dict[str, Any]:
    err = _validate_entry(entry)
    if err is not None:
        return {"video_id": entry.get("video_id"), "error": err}
    try:
        context = _build_context(entry, args)
    except Exception as exc:  # noqa: BLE001
        return {"video_id": entry["video_id"], "error": f"{type(exc).__name__}: {exc}"}
    results = run_modules(context=context, modules=modules)
    return {
        "video_id": entry["video_id"],
        "unity_video": entry["unity_video"],
        "wm_video": entry["wm_video"],
        **results,
    }


def _aggregate(per_video: list[dict[str, Any]], modules: list[str]) -> dict[str, Any]:
    ok = [r for r in per_video if "error" not in r]
    agg: dict[str, Any] = {
        "n_trajectories": len(ok),
        "n_errors": len(per_video) - len(ok),
        "delta_convention": "positive = WM worse than Unity",
    }
    if not ok:
        return agg

    def _collect(path: list[str]) -> list[float]:
        import math
        vals: list[float] = []
        for r in ok:
            cur: Any = r
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
                vals.append(float(cur))
        return vals

    paths_per_module = {
        "liquid_state": [
            ["liquid_state", "fill", "curve_correlation"],
            ["liquid_state", "fill", "mae"],
            ["liquid_state", "pour", "onset_offset_sec"],
            ["liquid_state", "pour", "rate_curve_dtw"],
            ["liquid_state", "spill", "final_volume_error_ml"],
            ["liquid_state", "surface", "height_trace_correlation"],
        ],
        "liquid_velocity": [
            ["liquid_velocity", "epe", "mean"],
            ["liquid_velocity", "epe", "median"],
            ["liquid_velocity", "epe", "p90"],
            ["liquid_velocity", "cosine_similarity"],
            ["liquid_velocity", "divergence_residual"],
            ["liquid_velocity", "temporal_consistency"],
        ],
        "liquid_coupling": [
            ["liquid_coupling", "tilt_pour_correlation", "unity"],
            ["liquid_coupling", "tilt_pour_correlation", "wm"],
            ["liquid_coupling", "tilt_pour_correlation", "delta"],
        ],
    }
    for mod in modules:
        for path in paths_per_module.get(mod, []):
            vals = _collect(path)
            if vals:
                key = ".".join(path[1:])
                agg.setdefault(mod, {})[f"{key}_mean"] = statistics.fmean(vals)
                agg[mod][f"{key}_n"] = len(vals)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL of trajectory entries")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--liquid-cache-root", required=True, type=Path)
    parser.add_argument("--modules", type=str, default=",".join(_DEFAULT_MODULES),
                        help="Comma-separated module names")
    parser.add_argument("--also-run", type=str, default="",
                        help="Comma-separated additional modules (e.g. lpips,control_motion)")
    parser.add_argument("--eval-fps", dest="eval_fps", type=float, default=10.0)
    parser.add_argument("--max-pairs", dest="max_pairs", type=int, default=120)
    parser.add_argument("--liquid_sam2_model", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--liquid_sam2_prompt", type=str, default="the liquid inside the bottle")
    parser.add_argument("--liquid_piv_window_size", type=int, default=32)
    parser.add_argument("--liquid_piv_overlap", type=int, default=16)
    parser.add_argument("--liquid_max_pairs", type=int, default=0)
    parser.add_argument("--liquid_pour_threshold_deg", type=float, default=30.0)
    parser.add_argument("--liquid_k_pour", type=float, default=0.05)
    parser.add_argument("--liquid_bottle_local_volume_ml", type=float, default=500.0)
    parser.add_argument("--liquid_camera_compensation", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--liquid_camera_regime", type=str, default="c2_static_side")
    parser.add_argument("--liquid_coupling_max_lag_sec", type=float, default=1.0)
    args = parser.parse_args()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    extras = [m.strip() for m in args.also_run.split(",") if m.strip()]
    modules = modules + [m for m in extras if m not in modules]

    per_video: list[dict[str, Any]] = []
    for line in args.manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            per_video.append({"error": f"bad manifest line: {exc}", "line": line})
            continue
        per_video.append(_run_one(entry, args, modules))

    agg = _aggregate(per_video, modules)
    payload = {
        "per_video": per_video,
        "aggregate": agg,
        "config": {
            "modules": modules,
            "sam2_model": args.liquid_sam2_model,
            "piv_window_size": args.liquid_piv_window_size,
            "piv_overlap": args.liquid_piv_overlap,
            "pour_threshold_deg": args.liquid_pour_threshold_deg,
            "k_pour": args.liquid_k_pour,
            "git_sha": _git_sha(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
