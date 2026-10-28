#!/usr/bin/env python3
"""Prepare short LingBot camera-gain calibration paths and batch jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_lingbot_table7_inputs import pose_yaw_source_indices, proper_opencv_c2w


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "runs/unity/garden_bridge_gt/evaluation_artifacts"
IMAGE = ROOT / "experiments/protocol/conditioning_frames/garden_inward.png"
# Put the expected response-compensated region first so an interrupted batch
# still resolves the production-critical gain before the diagnostic points.
TARGETS = (-115.0, 115.0, -135.0, 135.0, 0.0, -45.0, 45.0, -90.0, 90.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tables_4_7/calibration/lingbot_world_cam")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    poses = np.load(SOURCE_ROOT / "poses.npy", allow_pickle=False)
    intrinsics = np.load(SOURCE_ROOT / "intrinsics.npy", allow_pickle=False)
    jobs = []
    for target in TARGETS:
        probe = "no_action" if target == 0 else f"yaw_{target:+g}".replace("+", "pos").replace("-", "neg")
        offsets = np.zeros(41, dtype=np.float64)
        offsets[:21] = np.linspace(0.0, target, 21)
        offsets[21:] = target
        indices = pose_yaw_source_indices(poses, start_frame=0, offsets_deg=offsets)
        selected, conversion = proper_opencv_c2w(poses[indices])
        input_root = args.output / "inputs" / probe
        input_root.mkdir(parents=True, exist_ok=True)
        np.save(input_root / "poses.npy", selected)
        np.save(input_root / "intrinsics.npy", intrinsics[indices])
        receipt = {
            "schema_version": 1,
            "probe": probe,
            "commanded_input_yaw_deg": target,
            "frame_count": 41,
            "turn_end_sec": 1.25,
            "pose_handedness_conversion": conversion,
            "poses_sha256": sha256(input_root / "poses.npy"),
            "intrinsics_sha256": sha256(input_root / "intrinsics.npy"),
            "conditioning_image_sha256": sha256(IMAGE),
        }
        (input_root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        for repeat_index in range(args.repeats):
            output = args.output / probe / f"run{repeat_index}" / "candidate.mp4"
            jobs.append({
                "probe": probe,
                "repeat": f"run{repeat_index}",
                "image": str(IMAGE),
                "action_path": str(input_root),
                "prompt": "A first-person view from a fixed standing position in a richly textured traditional Japanese garden courtyard, with trees, stone lanterns, rocks, and wooden buildings preserved during a controlled camera yaw.",
                "seed": 46000 + repeat_index,
                "frame_num": 41,
                "save_file": str(output),
                "commanded_input_yaw_deg": target,
            })
    jobs_path = args.output / "calibration_jobs.jsonl"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs), encoding="utf-8")
    print(json.dumps({"status": "prepared", "probe_count": len(TARGETS), "job_count": len(jobs), "jobs": str(jobs_path), "jobs_sha256": sha256(jobs_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
