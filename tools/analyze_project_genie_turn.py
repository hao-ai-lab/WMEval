#!/usr/bin/env python3
"""Validate Project Genie yaw-cycle timing from calibration screenshots.

The tool deliberately validates *visual periodicity*, not absolute camera pose.
When a single yaw key is held continuously and the same ordered scene views
recur, one visual period is the best observable proxy for a 360-degree turn in
the hosted UI.  Cross-renderer Unity pose alignment remains a separate gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _feature(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read {path}")
    height, width = image.shape
    # Exclude header/progress and on-screen movement controls. The center is
    # the generated viewport and is stable enough for periodic correlation.
    crop = image[int(0.09 * height) : int(0.78 * height), int(0.05 * width) : int(0.95 * width)]
    small = cv2.resize(crop, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    norm = float(np.linalg.norm(small))
    if norm <= 1e-9:
        raise ValueError(f"uninformative calibration screenshot: {path}")
    return (small / norm).ravel()


def analyze(samples_path: Path, *, max_lag: int | None = None) -> dict[str, Any]:
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or len(samples) < 8:
        raise ValueError("at least eight calibration samples are required")
    features = np.stack([_feature(Path(row["path"])) for row in samples])
    times = np.asarray([float(row["elapsed_sec"]) for row in samples], dtype=np.float64)
    limit = min(max_lag or len(samples) // 2, len(samples) - 2)
    lag_rows: list[dict[str, Any]] = []
    for lag in range(1, limit + 1):
        pair_scores = np.sum(features[:-lag] * features[lag:], axis=1)
        periods = times[lag:] - times[:-lag]
        lag_rows.append(
            {
                "lag_samples": lag,
                "median_period_sec": float(np.median(periods)),
                "mean_similarity": float(pair_scores.mean()),
                "median_similarity": float(np.median(pair_scores)),
            }
        )

    candidates: list[dict[str, Any]] = []
    for index in range(1, len(lag_rows) - 1):
        row = lag_rows[index]
        score = row["mean_similarity"]
        left = lag_rows[index - 1]["mean_similarity"]
        right = lag_rows[index + 1]["mean_similarity"]
        if score > left and score > right:
            candidate = dict(row)
            candidate["local_prominence"] = float(score - max(left, right))
            candidates.append(candidate)
    plausible = [
        row
        for row in candidates
        if row["lag_samples"] >= 3 and row["mean_similarity"] >= 0.35
        and row["local_prominence"] >= 0.03
    ]
    if not plausible:
        ranked = sorted(candidates, key=lambda row: row["mean_similarity"], reverse=True)[:8]
        raise ValueError(f"no reliable full-yaw correlation peak found; local peaks={ranked}")
    # The first strong local peak is a candidate turn; later peaks must confirm
    # it.  This rejects a single coincidental return view caused by generative
    # drift or a repeated texture.
    best = min(plausible, key=lambda row: row["lag_samples"])
    base_lag = int(best["lag_samples"])
    harmonic_rows: list[dict[str, Any]] = []
    for multiple in (2, 3):
        target = base_lag * multiple
        nearby = [
            row
            for row in lag_rows
            if abs(int(row["lag_samples"]) - target) <= 1
        ]
        if nearby:
            harmonic = max(nearby, key=lambda row: row["mean_similarity"])
            harmonic_rows.append({"multiple": multiple, **harmonic})
    recurrence_scores = np.sum(features[:-base_lag] * features[base_lag:], axis=1)
    adjacent_scores = np.sum(features[:-1] * features[1:], axis=1)
    completed_periods = (len(samples) - 1) / base_lag
    confirmations = [
        row
        for row in harmonic_rows
        if row["mean_similarity"] >= 0.28
        and row["mean_similarity"] >= float(adjacent_scores.mean()) + 0.06
    ]
    validation_reasons: list[str] = []
    if completed_periods < 3.0:
        validation_reasons.append(
            f"only {completed_periods:.2f} candidate periods were observed; need at least 3"
        )
    if len(confirmations) < 2:
        validation_reasons.append(
            f"only {len(confirmations)} of 2 required harmonic recurrences were confirmed"
        )
    recurrence_fraction = float(np.mean(recurrence_scores >= 0.25))
    if recurrence_fraction < 0.60:
        validation_reasons.append(
            f"only {recurrence_fraction:.3f} of one-period pairs recur above 0.25"
        )
    diversity_gap = float(best["mean_similarity"] - adjacent_scores.mean())
    if diversity_gap < 0.10:
        validation_reasons.append(
            f"cycle/adjacent similarity gap {diversity_gap:.3f} is below 0.10"
        )
    if validation_reasons:
        raise ValueError(
            "candidate visual period failed validation: "
            + "; ".join(validation_reasons)
        )
    period = float(best["median_period_sec"])
    receipt_path = samples_path.parent / "receipt.json"
    receipt: dict[str, Any] = {}
    if receipt_path.is_file():
        candidate = json.loads(receipt_path.read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            receipt = candidate
    return {
        "schema_version": 1,
        "status": "validated_periodicity",
        "claim_scope": "visual_yaw_cycle_timing",
        "absolute_pose_status": "not_validated_by_this_tool",
        "source": str(samples_path.resolve()),
        "sample_count": len(samples),
        "key": receipt.get("key"),
        "view": receipt.get("view"),
        "page_url": receipt.get("page_url"),
        "full_turn_duration_sec": period,
        "yaw_rate_deg_sec": 360.0 / period,
        "duration_45_deg_sec": period / 8.0,
        "completed_periods_observed": completed_periods,
        "one_period_pair_count": int(len(recurrence_scores)),
        "one_period_recurrence_fraction": recurrence_fraction,
        "mean_adjacent_similarity": float(adjacent_scores.mean()),
        "cycle_adjacent_similarity_gap": diversity_gap,
        "selected_peak": best,
        "harmonic_confirmations": confirmations,
        "lag_scores": lag_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-lag", type=int)
    args = parser.parse_args()
    report = analyze(args.samples.resolve(), max_lag=args.max_lag)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
