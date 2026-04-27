"""Run liquid-pour evaluation across multiple LiquidContainer trajectories.

Each line in --manifest is a JSON object with keys:
    video_id, unity_video, wm_video, unity_log

Optionally: prompt, scene_key, camera_regime.

Writes {per_video, aggregate, config} JSON to --out and prints a small status
JSON to stdout. Per-trajectory errors land in `per_video` with an `error` key
so a single bad trajectory does not halt the batch.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.context import EvaluationContext  # noqa: E402
from evaluation.registry import run_modules  # noqa: E402
from evaluation.video_io import build_evaluation_context  # noqa: E402


_DEFAULT_MODULES = ("liquid_state", "liquid_velocity", "liquid_coupling")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT), check=False
        )
    except (FileNotFoundError, OSError):
        return None
    sha = out.stdout.strip()
    return sha if sha else None


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


def _run_one(entry: dict[str, Any], args: argparse.Namespace, modules: list[str]) -> dict[str, Any]:
    err = _validate_entry(entry)
    if err is not None:
        return {"video_id": entry.get("video_id"), "error": err}
    try:
        context = _build_context(entry, args)
        results = run_modules(context=context, modules=modules)
    except Exception as exc:  # noqa: BLE001 — batch resilience: keep iterating across trajectories
        return {"video_id": entry["video_id"], "error": f"{type(exc).__name__}: {exc}"}
    return {
        "video_id": entry["video_id"],
        "unity_video": entry["unity_video"],
        "wm_video": entry["wm_video"],
        **results,
    }


def _collect_path(rows: list[dict[str, Any]], path: list[str]) -> list[float]:
    vals: list[float] = []
    for r in rows:
        cur: Any = r
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
            vals.append(float(cur))
    return vals


def _aggregate(per_video: list[dict[str, Any]], modules: list[str]) -> dict[str, Any]:
    ok = [r for r in per_video if "error" not in r]
    agg: dict[str, Any] = {
        "n_trajectories": len(ok),
        "n_errors": len(per_video) - len(ok),
        "delta_convention": "positive = WM worse than Unity",
    }
    if not ok:
        return agg

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
            vals = _collect_path(ok, path)
            if vals:
                key = ".".join(path[1:])
                bucket = agg.setdefault(mod, {})
                bucket[f"{key}_mean"] = statistics.fmean(vals)
                bucket[f"{key}_median"] = statistics.median(vals)
                bucket[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                bucket[f"{key}_n"] = len(vals)
    return agg


def _iter_manifest_entries(path: Path):
    """Yield (line_no, entry, parse_error) tuples. Streams to keep memory bounded."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"line {line_no}: bad JSON: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL of trajectory entries")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the {per_video, aggregate, config} JSON report")
    parser.add_argument("--liquid_cache_root", required=True, type=Path,
                        help="Directory for SAM2/PIV pickled caches (shared across trajectories)")
    parser.add_argument("--modules", type=str, default=",".join(_DEFAULT_MODULES),
                        help="Comma-separated module names")
    parser.add_argument("--also-run", type=str, default="",
                        help="Comma-separated additional modules (e.g. lpips,control_motion)")
    parser.add_argument("--eval-fps", dest="eval_fps", type=float, default=10.0,
                        help="Frame rate for paired sampling between unity and wm video")
    parser.add_argument("--max-pairs", dest="max_pairs", type=int, default=120,
                        help="Cap on paired frame count per trajectory (0 = no cap)")
    parser.add_argument("--liquid_sam2_model", type=str, default="facebook/sam2.1-hiera-small",
                        help="HuggingFace SAM2 video predictor checkpoint id")
    parser.add_argument("--liquid_sam2_prompt", type=str, default="the liquid inside the bottle",
                        help="Text prompt used to seed the SAM2 mask track")
    parser.add_argument("--liquid_piv_window_size", type=int, default=32,
                        help="OpenPIV interrogation window edge (px)")
    parser.add_argument("--liquid_piv_overlap", type=int, default=16,
                        help="OpenPIV interrogation window overlap (px)")
    parser.add_argument("--liquid_max_pairs", type=int, default=0,
                        help="Per-module pair cap (0 = inherit --max-pairs)")
    parser.add_argument("--liquid_pour_threshold_deg", type=float, default=30.0,
                        help="Tilt threshold above which pour is considered active")
    parser.add_argument("--liquid_k_pour", type=float, default=0.05,
                        help="Volumetric pour-rate coefficient (matches Unity model)")
    parser.add_argument("--liquid_bottle_local_volume_ml", type=float, default=500.0,
                        help="Bottle internal volume (ml), used to scale pour rate")
    parser.add_argument("--liquid_camera_compensation", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Subtract estimated camera motion from PIV velocities")
    parser.add_argument("--liquid_camera_regime", type=str, default="c2_static_side",
                        help="Camera regime label (e.g. c1_handheld, c2_static_side)")
    parser.add_argument("--liquid_coupling_max_lag_sec", type=float, default=1.0,
                        help="Maximum lag (sec) considered for tilt→pour-rate correlation")
    args = parser.parse_args()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    extras = [m.strip() for m in args.also_run.split(",") if m.strip()]
    modules = modules + [m for m in extras if m not in modules]

    per_video: list[dict[str, Any]] = []
    for line_no, entry, parse_error in _iter_manifest_entries(args.manifest):
        if parse_error is not None:
            per_video.append({"error": parse_error, "manifest_line": line_no})
            continue
        assert entry is not None  # paired with parse_error: exactly one is non-None
        row = _run_one(entry, args, modules)
        row["manifest_line"] = line_no
        per_video.append(row)

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
    status = {
        "out": str(args.out),
        "n_trajectories": agg["n_trajectories"],
        "n_errors": agg["n_errors"],
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
