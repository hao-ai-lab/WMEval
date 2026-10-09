#!/usr/bin/env python3
"""Run deterministic five-repeat Table 7 generations with Matrix-Game 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/workspace/model_sources/Matrix-Game/Matrix-Game-3")
CHECKPOINT = Path("/workspace/model_checkpoints/Matrix-Game-3.0")
PYTHON = Path("/workspace/model_envs/matrix-game-3/bin/python")
QUEUE = ROOT / "runs/tables_4_7/model_inputs/matrix_game_3/generation_queue.jsonl"
SOURCE_REVISION = "756776d516233027631008761d4d40ddaea3e23a"
CHECKPOINT_REVISION = "382aa382a03cec057360755aa72c7735025770bb"


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


def build_command(item: dict[str, Any], gpus: int, gpu_index: int) -> list[str]:
    run = str(item["generation_repeat"])
    output = (ROOT / str(item["candidate_video"])).parent
    return [
        str(PYTHON), "-m", "torch.distributed.run",
        "--master_addr=127.0.0.1", f"--master_port={29513 + gpu_index}", f"--nproc_per_node={gpus}",
        str(ROOT / "tools/model_adapters/matrix_game3_deterministic.py"),
        "--source", str(SOURCE),
        "--schedule", str(ROOT / str(item["schedule"])),
        "--checkpoint", str(CHECKPOINT),
        "--image", str(ROOT / str(item["conditioning_image"])),
        "--prompt", str(item["environment_prompt"]),
        "--output", str(output),
        "--seed", str(45000 + int(run.removeprefix("run"))),
    ]


def run_item(item: dict[str, Any], dry_run: bool, gpus: int, gpu_index: int) -> dict[str, Any]:
    output = (ROOT / str(item["candidate_video"])).parent
    candidate = output / "candidate.mp4"
    output.mkdir(parents=True, exist_ok=True)
    command = build_command(item, gpus, gpu_index)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "dry_run", "output": str(candidate), "command": command}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / '.local/model_shims'}:{SOURCE}:{ROOT}"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_index + offset) for offset in range(gpus))
    started = time.time()
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=SOURCE, env=environment, stdout=stdout, stderr=stderr, check=False)
    receipt = {
        "schema_version": 1,
        "status": "complete" if completed.returncode == 0 and candidate.is_file() else "failed",
        "returncode": completed.returncode,
        "model": "matrix_game_3",
        "case_id": item["case_id"],
        "generation_repeat": item["generation_repeat"],
        "seed": 45000 + int(str(item["generation_repeat"]).removeprefix("run")),
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "schedule_sha256": sha256(ROOT / str(item["schedule"])),
        "conditioning_image_sha256": sha256(ROOT / str(item["conditioning_image"])),
        "elapsed_sec": time.time() - started,
    }
    if candidate.is_file():
        receipt["candidate_video"] = str(candidate)
        receipt["candidate_video_sha256"] = sha256(candidate)
        receipt["candidate_video_bytes"] = candidate.stat().st_size
    (output / "generation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--repeat")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    items = [json.loads(line) for line in QUEUE.read_text(encoding="utf-8").splitlines() if line]
    if args.case:
        items = [item for item in items if item["case_id"] == args.case]
    if args.repeat:
        items = [item for item in items if item["generation_repeat"] == args.repeat]
    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("--shard-index and --shard-count must be supplied together")
    if args.shard_count is not None:
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("invalid shard index/count")
        items = [item for index, item in enumerate(items) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no runs matched")
    for item in items:
        candidate = ROOT / str(item["candidate_video"])
        if generation_complete(candidate) and not args.dry_run:
            print(json.dumps({"status": "already_complete", "output": str(candidate)}), flush=True)
            continue
        result = run_item(item, args.dry_run, args.gpus, args.gpu_index)
        print(json.dumps(result), flush=True)
        if result["status"] == "failed" and not args.continue_on_error:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
