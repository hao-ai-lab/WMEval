from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import imageio.v3 as iio
import numpy as np

from ..context import EvaluationContext
from .common_unity import (
    angular_error_deg,
    extract_light_vectors_and_intensity,
    get_unity_log_path,
    load_unity_log,
    sample_series_by_count,
)


def _extract_frames_to_dir(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames, start=1):
        h, w = frame.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = frame
        resized = cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}.png"), resized)


def _run_diffusionlight_pipeline(
    diffusionlight_repo: str,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    repo = Path(diffusionlight_repo)
    if not repo.exists():
        raise FileNotFoundError(f"DiffusionLight repo does not exist: {diffusionlight_repo}")
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ["python", "inpaint.py", "--dataset", str(input_dir), "--output_dir", str(output_dir)],
        [
            "python",
            "ball2envmap.py",
            "--ball_dir",
            str(output_dir / "square"),
            "--envmap_dir",
            str(output_dir / "envmap"),
        ],
        [
            "python",
            "exposure2hdr.py",
            "--input_dir",
            str(output_dir / "envmap"),
            "--output_dir",
            str(output_dir / "hdr"),
        ],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=str(repo), check=True)
    return output_dir / "hdr"


def _find_hdr_files(hdr_dir: Path) -> List[Path]:
    files = sorted(hdr_dir.glob("*.exr"))
    if files:
        return files
    return sorted(hdr_dir.glob("*.hdr"))


def _envmap_path_to_direction(path: Path) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    hdr = iio.imread(str(path), plugin="opencv")
    if hdr.ndim != 3 or hdr.shape[2] < 3:
        raise ValueError(f"Unexpected envmap shape for {path}: {hdr.shape}")

    lum = 0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]
    h, w = lum.shape
    y_idx, x_idx = np.unravel_index(np.argmax(lum), lum.shape)

    az = (x_idx / max(w - 1, 1)) * 2.0 * np.pi - np.pi
    el = (0.5 - y_idx / max(h - 1, 1)) * np.pi

    vec = np.asarray(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def _summarize(values: List[float], prefix: str) -> Dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def analyze_lighting_from_context(context: EvaluationContext) -> Dict[str, Any]:
    if context.num_pairs == 0:
        return {"module": "lighting", "error": "no overlapping timestamps available"}

    artifacts = context.artifacts or {}
    ref_hdr_dir = artifacts.get("diffusionlight_ref_hdr_dir")
    cand_hdr_dir = artifacts.get("diffusionlight_candidate_hdr_dir")
    run_diffusionlight = bool(artifacts.get("run_diffusionlight", False))
    diffusionlight_repo = str(artifacts.get("diffusionlight_repo") or "/workspace/unity/DiffusionLight")
    diffusionlight_output_root = artifacts.get("diffusionlight_output_root")

    with tempfile.TemporaryDirectory(prefix="wmeval_diffusionlight_") as tmp:
        tmp_path = Path(tmp)

        if run_diffusionlight:
            if diffusionlight_output_root:
                base = Path(str(diffusionlight_output_root))
            else:
                base = tmp_path / "outputs"
            base.mkdir(parents=True, exist_ok=True)

            ref_frames_dir = tmp_path / "frames_ref"
            cand_frames_dir = tmp_path / "frames_candidate"
            _extract_frames_to_dir(context.reference.frames[: context.num_pairs], ref_frames_dir)
            _extract_frames_to_dir(context.candidate.frames[: context.num_pairs], cand_frames_dir)

            ref_out = base / "reference"
            cand_out = base / "candidate"
            ref_hdr_dir = str(_run_diffusionlight_pipeline(diffusionlight_repo, ref_frames_dir, ref_out))
            cand_hdr_dir = str(_run_diffusionlight_pipeline(diffusionlight_repo, cand_frames_dir, cand_out))

        if not ref_hdr_dir or not cand_hdr_dir:
            return {
                "module": "lighting",
                "error": (
                    "Need DiffusionLight HDR directories or enable --run_diffusionlight. "
                    "Provide diffusionlight_ref_hdr_dir and diffusionlight_candidate_hdr_dir."
                ),
            }

        ref_hdr_files = _find_hdr_files(Path(str(ref_hdr_dir)))
        cand_hdr_files = _find_hdr_files(Path(str(cand_hdr_dir)))
        if not ref_hdr_files or not cand_hdr_files:
            return {
                "module": "lighting",
                "error": "No HDR files found in provided DiffusionLight output directories",
                "reference_hdr_dir": ref_hdr_dir,
                "candidate_hdr_dir": cand_hdr_dir,
            }

        n = min(len(ref_hdr_files), len(cand_hdr_files), context.num_pairs)
        ref_dirs = [_envmap_path_to_direction(ref_hdr_files[i]) for i in range(n)]
        cand_dirs = [_envmap_path_to_direction(cand_hdr_files[i]) for i in range(n)]

        dmc_angular_error = [angular_error_deg(cand_dirs[i], ref_dirs[i]) for i in range(n)]
        dmc_score = [max(0.0, 1.0 - (err / 180.0)) for err in dmc_angular_error]

        output: Dict[str, Any] = {
            "module": "lighting",
            "estimator": "DiffusionLight HDR envmap peak direction",
            "reference_hdr_dir": str(ref_hdr_dir),
            "candidate_hdr_dir": str(cand_hdr_dir),
            "num_pairs": n,
            "timestamps_sec": context.reference.timestamps_sec[:n],
            "dmc_angular_error_deg_per_frame": dmc_angular_error,
            "dmc_score_per_frame": dmc_score,
            **_summarize(dmc_angular_error, "dmc_angular_error_deg"),
            **_summarize(dmc_score, "dmc_score"),
        }

        log_path = get_unity_log_path(artifacts)
        if log_path:
            gt_light_vectors_full, gt_intensity_full = extract_light_vectors_and_intensity(load_unity_log(log_path))
            gt_light_vectors = sample_series_by_count(gt_light_vectors_full, n)
            gt_intensity = sample_series_by_count(gt_intensity_full, n)
            unity_json_angular_error = [angular_error_deg(cand_dirs[i], gt_light_vectors[i]) for i in range(n)]
            output.update(
                {
                    "unity_log_json": log_path,
                    "gt_light_intensity": gt_intensity,
                    "unity_json_angular_error_deg_per_frame": unity_json_angular_error,
                    **_summarize(unity_json_angular_error, "unity_json_angular_error_deg"),
                }
            )

        return output
