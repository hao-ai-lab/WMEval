#!/usr/bin/env python3
"""Held-out check of the Table 4 kinematics plane calibration on Unity frames.

Each kinematics case maps image points to metric ground-plane coordinates with a
homography fitted to visual anchors (``plane_calibration.json``: frame-0 pixel
``conditioning_uv`` <-> Unity ``world_xz_m``). This tool refits the homography
with each anchor held out in turn (leave-one-out, plain DLT on the remaining
anchors) and reports the metric error at the held-out anchor -- an estimate of
the calibration's own error floor on the Unity reference, independent of any
world model. No hosted calls, no GPU.

    python tools/kinematics_plane_holdout.py --output reports/wmeval_lite/unity_calibration/plane_holdout.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval_sets/wmeval_lite_v1/cases"
DEFAULT_CASES = ("RacingCar/kinematics_third", "BilliardGame/kinematics_first")


def holdout(case_id: str) -> dict:
    calibration = json.loads((CASES / case_id / "reference/artifacts/plane_calibration.json").read_text())
    anchors = calibration["anchors"]
    uv = np.asarray([a["conditioning_uv"] for a in anchors], dtype=np.float64)
    xz = np.asarray([a["world_xz_m"] for a in anchors], dtype=np.float64)
    errors = []
    for index in range(len(anchors)):
        keep = np.arange(len(anchors)) != index
        matrix, _ = cv2.findHomography(uv[keep], xz[keep], 0)
        if matrix is None:
            continue
        point = cv2.perspectiveTransform(uv[index].reshape(1, 1, 2), matrix).reshape(2)
        errors.append(float(np.linalg.norm(point - xz[index])))
    errors_arr = np.asarray(errors)
    spread = float(np.linalg.norm(np.ptp(xz, axis=0)))
    return {
        "case_id": case_id,
        "anchors": len(anchors),
        "anchor_extent_m": round(spread, 3),
        "holdout_error_m": {
            "median": float(f"{np.median(errors_arr):.3e}"),
            "p90": float(f"{np.percentile(errors_arr, 90):.3e}"),
            "max": float(f"{errors_arr.max():.3e}"),
        },
        "per_anchor_error_m": [float(f"{e:.3e}") for e in errors],
        "method": "leave-one-out DLT homography on frame-0 anchors (Unity reference)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "cases": [holdout(case_id) for case_id in args.cases]}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
