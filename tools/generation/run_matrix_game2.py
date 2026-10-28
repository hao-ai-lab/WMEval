#!/usr/bin/env python3
"""Run deterministic calibration or scored Matrix-Game 2 generations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
SOURCE = Path("/workspace/model_sources/Matrix-Game/Matrix-Game-2")
CHECKPOINT = Path("/workspace/model_checkpoints/Matrix-Game-2.0")
PYTHON = Path("/workspace/model_envs/matrix-game-2/bin/python")
QUEUE = ROOT / "runs/tables_4_7/model_inputs/matrix_game_2/generation_queue.jsonl"
CALIBRATION = ROOT / "runs/tables_4_7/model_inputs/matrix_game_2/calibration/manifest.json"
SOURCE_REVISION = "756776d516233027631008761d4d40ddaea3e23a"
CHECKPOINT_REVISION = "08c42d1a99e57e79df60a964bb815618cdf460f8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_complete(candidate: Path) -> bool:
    receipt_path = candidate.parent / "generation_receipt.json"
    if not candidate.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("status") == "complete"
        and receipt.get("candidate_video_sha256") == sha256(candidate)
    )


def command(schedule: Path, image: Path, output: Path, latent_frames: int, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "tools/generation/model_adapters/matrix_game2_deterministic.py"),
        "--source", str(SOURCE),
        "--schedule", str(schedule),
        "--config", str(SOURCE / "configs/inference_yaml/inference_universal.yaml"),
        "--checkpoint", str(CHECKPOINT / "base_distilled_model/base_distill.safetensors"),
        "--pretrained-model", str(CHECKPOINT),
        "--image", str(image),
        "--output", str(output),
        "--latent-frames", str(latent_frames),
        "--seed", str(seed),
    ]


def execute(spec: dict[str, Any], dry_run: bool, gpu_index: int) -> dict[str, Any]:
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / "candidate.mp4"
    cmd = command(Path(spec["schedule"]), Path(spec["image"]), output, int(spec["latent_frames"]), int(spec["seed"]))
    (output / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "dry_run", "output": str(candidate), "command": cmd}
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{SOURCE}:{ROOT}"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(cmd, cwd=SOURCE, env=environment, stdout=stdout, stderr=stderr, check=False)
    raw = output / "demo.mp4"
    if completed.returncode == 0 and raw.is_file():
        shutil.copy2(raw, candidate)
    receipt = {
        "schema_version": 1,
        "status": "complete" if candidate.is_file() and completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "model": "matrix_game_2",
        "case_id": spec.get("case_id"),
        "probe": spec.get("probe"),
        "generation_repeat": spec["repeat"],
        "seed": spec["seed"],
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "schedule": str(spec["schedule"]),
        "schedule_sha256": sha256(Path(spec["schedule"])),
        "conditioning_image_sha256": sha256(Path(spec["image"])),
        "elapsed_sec": time.time() - started,
    }
    if candidate.is_file():
        receipt["candidate_video"] = str(candidate)
        receipt["candidate_video_sha256"] = sha256(candidate)
        receipt["candidate_video_bytes"] = candidate.stat().st_size
    (output / "generation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def calibration_specs(probe: str | None, repeat: str | None) -> list[dict[str, Any]]:
    manifest = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    image = ROOT / "lite/cases/WanderingGarden/scene_perturb_in/reference/ref.png"
    rows = []
    for entry in manifest["probes"]:
        if probe and entry["probe"] != probe:
            continue
        for run in manifest["repeats"]:
            if repeat and run != repeat:
                continue
            index = int(run.removeprefix("run"))
            rows.append({
                "probe": entry["probe"],
                "repeat": run,
                "schedule": ROOT / entry["schedule"],
                "image": image,
                "latent_frames": manifest["latent_frames"],
                "seed": 43000 + index,
                "output": ROOT / f"runs/tables_4_7/calibration/matrix_game_2/{entry['probe']}/{run}",
            })
    return rows


def scored_specs(case: str | None, repeat: str | None) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in QUEUE.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for item in items:
        if case and item["case_id"] != case:
            continue
        if repeat and item["generation_repeat"] != repeat:
            continue
        run = item["generation_repeat"]
        rows.append({
            "case_id": item["case_id"],
            "repeat": run,
            "schedule": ROOT / item["schedule"],
            "image": ROOT / item["conditioning_image"],
            "latent_frames": int(action_backends.backend("matrix_game_2")["latent_frames"]),
            "seed": 44000 + int(run.removeprefix("run")),
            "output": (ROOT / item["candidate_video"]).parent,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration", action="store_true")
    mode.add_argument("--scored", action="store_true")
    parser.add_argument("--probe")
    parser.add_argument("--case")
    parser.add_argument("--repeat")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    specs = calibration_specs(args.probe, args.repeat) if args.calibration else scored_specs(args.case, args.repeat)
    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("--shard-index and --shard-count must be supplied together")
    if args.shard_count is not None:
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("invalid shard index/count")
        specs = [spec for index, spec in enumerate(specs) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit("no runs matched")
    for spec in specs:
        candidate = Path(spec["output"]) / "candidate.mp4"
        if generation_complete(candidate) and not args.dry_run:
            print(json.dumps({"status": "already_complete", "output": str(candidate)}), flush=True)
            continue
        result = execute(spec, args.dry_run, args.gpu_index)
        print(json.dumps(result), flush=True)
        if result["status"] == "failed" and not args.continue_on_error:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
