#!/usr/bin/env python3
"""Run the remaining LingBot Table 7 queue with one persistent model load."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/workspace/model_sources/lingbot-world")
CHECKPOINT = Path("/workspace/model_checkpoints/lingbot-world-base-cam")
PYTHON = Path("/workspace/model_envs/lingbot/bin/python")
QUEUE = ROOT / "runs/tables_4_7/model_inputs/lingbot_world_cam/generation_queue.jsonl"
BATCHES = ROOT / "runs/tables_4_7/batches/lingbot_world_cam"
SOURCE_REVISION = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
CHECKPOINT_REVISION = "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd"


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


def early_batch_reservations(batch_root: Path) -> set[tuple[str, str]]:
    """Return rows reserved by opportunistic batches across shard boundaries."""
    reservations: set[tuple[str, str]] = set()
    for path in sorted(batch_root.glob("world1_early_adaptive_*/jobs.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            case_id = job.get("case_id")
            repeat = job.get("repeat")
            if not isinstance(case_id, str) or not isinstance(repeat, str):
                raise ValueError(f"invalid early-batch reservation in {path}")
            key = (case_id, repeat)
            if key in reservations:
                raise ValueError(f"duplicate early-batch reservation {key} in {path}")
            reservations.add(key)
    return reservations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repeat")
    parser.add_argument(
        "--job-key",
        action="append",
        default=[],
        metavar="CASE_ID::RUN",
        help="select an exact case/repeat row; repeat for a tail-helper batch",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--world-size", type=int, choices=(1, 8), default=8)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument(
        "--ignore-early-reservations",
        action="store_true",
        help="recovery override: allow full-plan shards to claim early-batch rows",
    )
    parser.add_argument(
        "--batch-label",
        help=(
            "optional isolated batch-directory label; useful for safe "
            "opportunistic prelaunches without overwriting a shard log"
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.gpu_index < 8:
        raise SystemExit("--gpu-index must be in [0, 7]")
    if args.world_size == 8 and args.gpu_index != 0:
        raise SystemExit("8-GPU mode must start at --gpu-index 0")
    # Validate the label before touching the queue or any batch directory.
    if args.batch_label is not None:
        if not args.batch_label or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in args.batch_label
        ):
            raise SystemExit(
                "--batch-label must contain only ASCII letters, digits, '_' or '-'"
            )
    items = [json.loads(line) for line in QUEUE.read_text(encoding="utf-8").splitlines() if line]
    if args.job_key:
        if args.batch_label is None:
            raise SystemExit("--job-key requires an isolated --batch-label")
        if args.case or args.repeat or args.shard_index is not None or args.shard_count is not None:
            raise SystemExit(
                "--job-key cannot be combined with --case, --repeat, or sharding"
            )
        selected_keys: set[tuple[str, str]] = set()
        for value in args.job_key:
            try:
                case_id, repeat = value.rsplit("::", 1)
            except ValueError as error:
                raise SystemExit("--job-key must use CASE_ID::RUN") from error
            if not case_id or not repeat.startswith("run") or not repeat[3:].isdigit():
                raise SystemExit("--job-key must use CASE_ID::RUN")
            key = (case_id, repeat)
            if key in selected_keys:
                raise SystemExit(f"duplicate --job-key: {value}")
            selected_keys.add(key)
        available_keys = {
            (str(item["case_id"]), str(item["generation_repeat"])) for item in items
        }
        missing_keys = selected_keys - available_keys
        if missing_keys:
            missing = ", ".join(f"{case_id}::{repeat}" for case_id, repeat in sorted(missing_keys))
            raise SystemExit(f"unknown --job-key: {missing}")
        items = [
            item
            for item in items
            if (str(item["case_id"]), str(item["generation_repeat"]))
            in selected_keys
        ]
    if args.case:
        selected_cases = set(args.case)
        items = [item for item in items if item["case_id"] in selected_cases]
    if args.repeat:
        items = [item for item in items if item["generation_repeat"] == args.repeat]
    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("--shard-index and --shard-count must be supplied together")
    if args.shard_count is not None:
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("invalid shard index/count")
        items = [item for index, item in enumerate(items) if index % args.shard_count == args.shard_index]
    reserved_excluded = 0
    if args.batch_label is None and not args.ignore_early_reservations:
        reservations = early_batch_reservations(BATCHES)
        before = len(items)
        items = [
            item
            for item in items
            if (str(item["case_id"]), str(item["generation_repeat"]))
            not in reservations
        ]
        reserved_excluded = before - len(items)
    items = [
        item
        for item in items
        if not generation_complete(ROOT / item["candidate_video"])
    ]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print(json.dumps({
            "status": "already_complete",
            "generation_count": 0,
            "reserved_excluded": reserved_excluded,
        }))
        return 0

    if args.batch_label is not None:
        shard_label = args.batch_label
    else:
        shard_label = (
            "all"
            if args.shard_count is None
            else f"shard{args.shard_index}-of-{args.shard_count}"
        )
    batch_root = BATCHES / f"world{args.world_size}_{shard_label}"
    batch_root.mkdir(parents=True, exist_ok=True)
    jobs_path = batch_root / "jobs.jsonl"
    jobs: list[dict[str, Any]] = []
    for item in items:
        repeat_index = int(str(item["generation_repeat"]).removeprefix("run"))
        job = {
            "case_id": item["case_id"],
            "repeat": item["generation_repeat"],
            "image": str(ROOT / item["conditioning_image"]),
            "action_path": str((ROOT / item["poses"]).parent),
            "prompt": item["environment_prompt"],
            "seed": 42000 + repeat_index,
            "save_file": str(ROOT / item["candidate_video"]),
        }
        if not args.dry_run:
            job["receipt_static"] = {
                "schema_version": 1,
                "model": "lingbot_world_cam",
                "case_id": item["case_id"],
                "generation_repeat": item["generation_repeat"],
                "seed": 42000 + repeat_index,
                "source_revision": SOURCE_REVISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "execution_mode": "single_gpu_cpu_offload" if args.world_size == 1 else "eight_gpu_fsdp_sp",
                "world_size": args.world_size,
                "conditioning_image_sha256": sha256(ROOT / item["conditioning_image"]),
                "poses_sha256": sha256(ROOT / item["poses"]),
                "intrinsics_sha256": sha256(ROOT / item["intrinsics"]),
            }
        jobs.append(job)
    jobs_path.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs), encoding="utf-8")
    adapter_command = [
        str(ROOT / "tools/generation/model_adapters/lingbot_batch.py"),
        "--source", str(SOURCE), "--checkpoint", str(CHECKPOINT), "--jobs", str(jobs_path),
    ]
    if args.world_size == 8:
        command = [
            str(PYTHON), "-m", "torch.distributed.run",
            "--master_addr=127.0.0.1", f"--master_port={29511 + args.gpu_index}",
            "--nproc_per_node=8", *adapter_command,
        ]
    else:
        command = [str(PYTHON), *adapter_command]
    (batch_root / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "generation_count": len(jobs),
            "reserved_excluded": reserved_excluded,
            "command": command,
        }))
        return 0
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{SOURCE}:{ROOT}"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(args.gpu_index + offset) for offset in range(args.world_size)
    )
    started = time.time()
    with (batch_root / "stdout.log").open("w", encoding="utf-8") as stdout, (batch_root / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=SOURCE, env=environment, stdout=stdout, stderr=stderr, check=False)
    failures = []
    by_key = {(item["case_id"], item["generation_repeat"]): item for item in items}
    for job in jobs:
        item = by_key[(job["case_id"], job["repeat"])]
        candidate = Path(job["save_file"])
        status = "complete" if candidate.is_file() else "failed"
        receipt = {
            "schema_version": 1,
            "status": status,
            "batch_returncode": completed.returncode,
            "model": "lingbot_world_cam",
            "case_id": job["case_id"],
            "generation_repeat": job["repeat"],
            "seed": job["seed"],
            "source_revision": SOURCE_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "execution_mode": "single_gpu_cpu_offload" if args.world_size == 1 else "eight_gpu_fsdp_sp",
            "world_size": args.world_size,
            "conditioning_image_sha256": sha256(Path(job["image"])),
            "poses_sha256": sha256(ROOT / item["poses"]),
            "intrinsics_sha256": sha256(ROOT / item["intrinsics"]),
            "batch_jobs_sha256": sha256(jobs_path),
            "batch_elapsed_sec": time.time() - started,
        }
        if candidate.is_file():
            receipt.update(candidate_video=str(candidate), candidate_video_sha256=sha256(candidate), candidate_video_bytes=candidate.stat().st_size)
        else:
            failures.append(str(candidate))
        receipt_path = candidate.parent / "generation_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "complete" if not failures and completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "generation_count": len(jobs),
        "reserved_excluded": reserved_excluded,
        "failures": failures,
    }
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
