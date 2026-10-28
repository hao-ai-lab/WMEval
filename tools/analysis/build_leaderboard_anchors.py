#!/usr/bin/env python3
"""Build the per-cell anchors for anchored leaderboard scores (tools/build_leaderboard.py --anchors).

Every (dimension, cell, metric) gets
  upper -- the Unity reference scored against itself (the best attainable value), and
  lower -- a defined failure: a frozen first-frame video scored by the same evaluator where
           that is a real failure (Control, Kinematics, Fluid), else the metric's chance or
           floor value (Consistency 0 %, Prompt rubric 0, Lighting 90 deg = independent
           random sun per object). A frozen video is not used for Consistency, Lighting or
           Prompt: it keeps identity, shadows and the opening scene perfectly and would
           score near the top.
Rows are mapped onto cells with build_leaderboard.lite_cells, so anchors line up with the
cells the leaderboard ranks. RacingCar FDE is in road coordinates, as the leaderboard's
values are (evaluation/modules/road_trajectory.py): upper = the module run on the Unity
reference video against Unity's logged car path; lower = a car that never moves, whose
final error is the logged start-to-end displacement (a frozen video measures exactly that).

    python tools/analysis/build_leaderboard_anchors.py --output experiments/leaderboard/anchors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_leaderboard import lite_cells  # noqa: E402

UNITY_SELFMATCH = ROOT / "reports/wmeval_lite/unity_selfmatch/per_case_results.jsonl"
JUDGE_FLOOR = ROOT / "lite/judge_noise_floor.json"   # Hub-hosted; see _judge_floor()


def _judge_floor() -> Path:
    """The judge noise floor: written locally by measure_judge_floor.py, otherwise fetched from the Hub."""
    if JUDGE_FLOOR.is_file():
        return JUDGE_FLOOR
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("lmgame/WMEval-Lite", JUDGE_FLOOR.name, repo_type="dataset"))
ANCHOR_RUNS = [ROOT / f"reports/wmeval_lite_modal/anchors_{d}/per_case_results.jsonl"
               for d in ("action_controllability", "kinematics_mechanics", "lighting_shading", "fluid_mechanics")]
ANCHOR_RUNS.append(ROOT / "reports/wmeval_lite_modal/ctrl_comp_v2/per_case_results.jsonl")   # Unity Control self-match
FIXED_LOWER = {("environment_object_consistency", "char"): (0.0, "0 % (identity never kept)"),
               ("prompt_scene_consistency", "overall"): (0.0, "rubric minimum 0"),
               ("lighting_shading", "mosd"): (90.0, "90 deg: mean angle between independent random sun directions")}
MEASURED_LOWER = {"action_controllability", "kinematics_mechanics", "fluid_mechanics"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def _cells(rows: list[dict]) -> dict:
    out: dict = {}
    for dim, cells in lite_cells(rows).items():
        for cell, metrics in cells.items():
            for metric, vals in metrics.items():
                if vals:
                    out[(dim, cell, metric)] = sum(vals) / len(vals)
    return out


def _judge_rows() -> list[dict]:
    # Unity prompt self-scores (K judge passes per case) as Lite-shaped rows.
    floor = json.loads(_judge_floor().read_text())["selfmatch"].get("prompt_scene_consistency", {})
    return [{"dimension": "prompt_scene_consistency", "case_id": case, "model": "unity_self", "seed": "run0",
             "status": "completed", "metrics": {k: v["mean"] for k, v in d["metrics"].items()}}
            for case, d in floor.items() if not case.startswith("synthetic/")]


def racing_car_road_fde() -> tuple[float, float]:
    """(Unity self-match, never-moving car) RacingCar FDE in road coordinates, metres."""
    import cv2
    import numpy as np
    from evaluation.modules import road_trajectory as rt
    case = ROOT / "lite/cases/RacingCar/kinematics_third"
    art = case / "reference/artifacts"
    pos = np.asarray(json.loads((art / "object_transforms.json").read_text())["objects"]["car"]["positions_world"], float)
    gt = pos[:, [0, 2]] - pos[0, [0, 2]]
    plane = json.loads((art / "plane_calibration.json").read_text())["plane"]["value_m"]
    rig = rt.Rig.from_unity(np.load(art / "intrinsics.npy"), np.load(art / "poses.npy")[0], plane, (1280, 720))
    cap, frames = cv2.VideoCapture(str(case / "reference/reference.mp4")), []
    while (ok_frame := cap.read())[0]:
        frames.append(ok_frame[1])
    hat = rt.measure(frames, rig)["car_xz"]
    hat = hat - hat[0]
    n = min(len(hat), len(gt))
    return float(np.linalg.norm(hat[n - 1] - gt[n - 1])), float(np.linalg.norm(gt[n - 1]))


def build() -> dict:
    anchor_rows = [r for p in ANCHOR_RUNS for r in _rows(p)]
    unity = _cells([r for r in anchor_rows if r["model"] == "unity_self"] + _rows(UNITY_SELFMATCH) + _judge_rows())
    frozen = _cells([r for r in anchor_rows if r["model"] == "frozen_baseline"])
    anchors: dict = {"_sources": {
        "upper": "Unity reference scored against itself: reports/wmeval_lite/unity_selfmatch (consistency, kinematics), "
                 "judge_noise_floor.json selfmatch (prompt), reports/wmeval_lite_modal/anchors_* (control, lighting, fluid)",
        "lower": {f"{d}:{m}": why for (d, m), (_, why) in FIXED_LOWER.items()}
                 | {d: "frozen first-frame video scored by the same evaluator (reports/wmeval_lite_modal/anchors_*)"
                    for d in MEASURED_LOWER}}}
    for (dim, cell, metric), hi in sorted(unity.items()):
        if dim == "kinematics_mechanics":      # road-coordinate anchors below replace the legacy reprojection
            continue
        if (dim, metric) in FIXED_LOWER:
            lo = FIXED_LOWER[(dim, metric)][0]
        elif dim in MEASURED_LOWER and (dim, cell, metric) in frozen:
            lo = frozen[(dim, cell, metric)]
        else:
            continue
        if hi == lo:
            continue
        anchors.setdefault(dim, {}).setdefault(metric, {})[cell] = {"upper": round(hi, 6), "lower": round(lo, 6)}
    hi, lo = racing_car_road_fde()
    anchors.setdefault("kinematics_mechanics", {})["fde"] = {"racing_car/third": {"upper": round(hi, 6), "lower": round(lo, 6)}}
    anchors["_sources"]["kinematics_mechanics:fde"] = ("road coordinates: upper = road_trajectory on the Unity reference vs "
                                                      "Unity's logged car path; lower = never-moving car (logged displacement)")
    return anchors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "experiments/leaderboard/anchors.json")
    args = ap.parse_args()
    anchors = build()
    args.output.write_text(json.dumps(anchors, indent=1) + "\n")
    for dim, metrics in anchors.items():
        if not dim.startswith("_"):
            for metric, cells in metrics.items():
                print(dim, metric, {c: (a["lower"], a["upper"]) for c, a in cells.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
