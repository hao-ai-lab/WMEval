#!/usr/bin/env python3
"""Estimate visual yaw response, hold drift, and first-frame identity from video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    return gray[: int(height * 0.88)]


def yaw_between(a: np.ndarray, b: np.ndarray, focal_px: float) -> tuple[float | None, int]:
    first = _gray(a)
    second = _gray(b)
    points = cv2.goodFeaturesToTrack(first, maxCorners=800, qualityLevel=0.01, minDistance=6, blockSize=7)
    if points is None or len(points) < 12:
        return None, 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        first, second, points, None, winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    )
    if moved is None or status is None:
        return None, 0
    old = points.reshape(-1, 2)[status.ravel() == 1]
    new = moved.reshape(-1, 2)[status.ravel() == 1]
    if len(old) < 12:
        return None, len(old)
    displacement = np.linalg.norm(new - old, axis=1)
    median = float(np.median(displacement))
    mad = float(np.median(np.abs(displacement - median)))
    keep = displacement <= median + max(4.0, 3.5 * mad)
    old = old[keep]
    new = new[keep]
    center = (first.shape[1] - 1) / 2.0
    old_bearing = np.arctan((old[:, 0] - center) / focal_px)
    new_bearing = np.arctan((new[:, 0] - center) / focal_px)
    # Static scene bearings move opposite to camera yaw.
    yaw = -float(np.median(new_bearing - old_bearing))
    return float(np.rad2deg(yaw)), len(old)


def normalized_correlation(frame: np.ndarray, image: np.ndarray) -> float:
    resized = cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    a = _gray(frame).astype(np.float32)
    b = _gray(resized).astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 1e-9 else 0.0


def analyze(video: Path, conditioning: Path, intrinsics: Path, turn_end_sec: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 3 or fps <= 0:
        raise ValueError(f"invalid video: {video}")
    image = cv2.imread(str(conditioning), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read conditioning image: {conditioning}")
    values = np.load(intrinsics, allow_pickle=False)
    focal_px = float(values[0, 0] if values.ndim == 2 else values[0])
    focal_px *= frames[0].shape[1] / 832.0
    increments = []
    tracks = []
    for first, second in zip(frames[:-1], frames[1:]):
        yaw, count = yaw_between(first, second, focal_px)
        increments.append(yaw)
        tracks.append(count)
    turn_end = min(len(increments), max(1, int(round(turn_end_sec * fps))))
    valid_turn = [value for value in increments[:turn_end] if value is not None]
    valid_hold = [value for value in increments[turn_end:] if value is not None]
    estimated = float(sum(valid_turn))
    hold_drift = float(sum(valid_hold))
    return {
        "schema_version": 1,
        "status": "measured" if len(valid_turn) >= max(2, turn_end // 2) else "insufficient_tracks",
        "video": str(video.resolve()),
        "conditioning_image": str(conditioning.resolve()),
        "intrinsics": str(intrinsics.resolve()),
        "frame_count": len(frames),
        "fps": fps,
        "focal_px_scaled": focal_px,
        "turn_end_sec": turn_end_sec,
        "turn_frame_intervals": turn_end,
        "estimated_visual_yaw_deg": estimated,
        "hold_drift_deg": hold_drift,
        "hold_drift_deg_per_sec": hold_drift / max((len(increments) - turn_end) / fps, 1e-9),
        "first_frame_normalized_correlation": normalized_correlation(frames[0], image),
        "median_track_count": float(np.median(tracks)),
        "valid_turn_increments": len(valid_turn),
        "valid_hold_increments": len(valid_hold),
        "per_frame_yaw_deg": increments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--turn-end-sec", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.video.resolve(), args.conditioning.resolve(), args.intrinsics.resolve(), args.turn_end_sec)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame_yaw_deg"}, indent=2))
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
