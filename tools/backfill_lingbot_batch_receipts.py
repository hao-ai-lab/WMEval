#!/usr/bin/env python3
"""Backfill receipts for finalized videos from a still-running LingBot batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs/tables_4_7/model_inputs/lingbot_world_cam/generation_queue.jsonl"
SOURCE_REVISION = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
CHECKPOINT_REVISION = "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_tool(name: str) -> str:
    system = shutil.which(name)
    if system:
        return system
    local = ROOT / ".local" / "bin" / name
    if local.is_file():
        return str(local)
    return name


def probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            media_tool("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,duration,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if count <= 0 or duration <= 0:
        raise ValueError(f"{path}: ffprobe did not report a complete video")
    return {
        "frame_count": count,
        "duration_sec": duration,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream.get("r_frame_rate") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--minimum-age-sec", type=float, default=60.0)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    jobs = [
        json.loads(line)
        for line in jobs_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    queue = {
        (str(item["case_id"]), str(item["generation_repeat"])): item
        for line in QUEUE.read_text(encoding="utf-8").splitlines()
        if line
        for item in [json.loads(line)]
    }
    now = time.time()
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    for job in jobs:
        key = (str(job["case_id"]), str(job["repeat"]))
        candidate = Path(job["save_file"])
        receipt_path = candidate.parent / "generation_receipt.json"
        if receipt_path.exists():
            skipped.append({"candidate": str(candidate), "reason": "receipt_exists"})
            continue
        if not candidate.is_file():
            skipped.append({"candidate": str(candidate), "reason": "candidate_missing"})
            continue
        if now - candidate.stat().st_mtime < args.minimum_age_sec:
            skipped.append({"candidate": str(candidate), "reason": "candidate_too_new"})
            continue
        item = queue.get(key)
        if item is None:
            raise ValueError(f"{key}: queue item is missing")
        video = probe_video(candidate)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "model": "lingbot_world_cam",
            "case_id": key[0],
            "generation_repeat": key[1],
            "seed": int(job["seed"]),
            "source_revision": SOURCE_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "execution_mode": "single_gpu_cpu_offload",
            "world_size": 1,
            "conditioning_image_sha256": sha256(Path(job["image"])),
            "poses_sha256": sha256(ROOT / str(item["poses"])),
            "intrinsics_sha256": sha256(ROOT / str(item["intrinsics"])),
            "batch_jobs_sha256": sha256(jobs_path),
            "receipt_source": "stable_video_mid_batch_backfill",
            "candidate_video": str(candidate),
            "candidate_video_sha256": sha256(candidate),
            "candidate_video_bytes": candidate.stat().st_size,
            "candidate_video_probe": video,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(str(receipt_path))
    report = {
        "schema_version": 1,
        "status": "complete",
        "jobs": str(jobs_path),
        "written": written,
        "skipped": skipped,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
