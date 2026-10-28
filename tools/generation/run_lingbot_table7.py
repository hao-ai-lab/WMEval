#!/usr/bin/env python3
"""Run the prepared five-repeat Table 7 queue with LingBot Base Cam."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
DEFAULT_QUEUE = ROOT / "runs/tables_4_7/model_inputs/lingbot_world_cam/generation_queue.jsonl"
DEFAULT_SOURCE = Path("/workspace/model_sources/lingbot-world")
DEFAULT_CHECKPOINT = Path("/workspace/model_checkpoints/lingbot-world-base-cam")
DEFAULT_PYTHON = Path("/workspace/model_envs/lingbot/bin/python")
SOURCE_REVISION = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
CHECKPOINT_REVISION = "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queue(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def seed_for(item: dict[str, Any]) -> int:
    """The queue's explicit seed, else the Table 7 default for the repeat."""
    if "seed" in item:
        return int(item["seed"])
    return 42000 + int(str(item["generation_repeat"]).removeprefix("run"))


def build_command(
    item: dict[str, Any], source: Path, checkpoint: Path, python: Path
) -> list[str]:
    output = ROOT / str(item["candidate_video"])
    action_path = (ROOT / str(item["poses"])).parent
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--master_addr=127.0.0.1",
        "--master_port=29511",
        "--nproc_per_node=8",
        str(ROOT / "tools/generation/model_adapters/lingbot_no_ui.py"),
        "--task", "i2v-A14B",
        "--size", "480*832",
        "--ckpt_dir", str(checkpoint),
        "--image", str(ROOT / str(item["conditioning_image"])),
        "--action_path", str(action_path),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", "8",
        "--frame_num", str(action_backends.backend("lingbot_world_cam")["frames"]),
        "--base_seed", str(seed_for(item)),
        "--prompt", str(item["environment_prompt"]),
        "--save_file", str(output),
    ]


def run_item(
    item: dict[str, Any], source: Path, checkpoint: Path, python: Path, dry_run: bool
) -> dict[str, Any]:
    output = ROOT / str(item["candidate_video"])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(item, source, checkpoint, python)
    command_record = output.parent / "command.json"
    command_record.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "dry_run", "output": str(output), "command": command}
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment["LINGBOT_SOURCE"] = str(source)
    with (output.parent / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output.parent / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if completed.returncode == 0 and output.is_file() else "failed",
        "returncode": completed.returncode,
        "model": "lingbot_world_cam",
        "case_id": item["case_id"],
        "generation_repeat": item["generation_repeat"],
        "seed": seed_for(item),
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "conditioning_image_sha256": sha256(ROOT / str(item["conditioning_image"])),
        "poses_sha256": sha256(ROOT / str(item["poses"])),
        "intrinsics_sha256": sha256(ROOT / str(item["intrinsics"])),
        "elapsed_sec": time.time() - started,
        "command_record": str(command_record),
        "candidate_video": str(output),
    }
    if output.is_file():
        receipt["candidate_video_sha256"] = sha256(output)
        receipt["candidate_video_bytes"] = output.stat().st_size
    receipt_path = output.parent / "generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--case")
    parser.add_argument("--repeat")
    parser.add_argument("--index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    items = load_queue(args.queue.resolve())
    if args.case:
        items = [item for item in items if item["case_id"] == args.case]
    if args.repeat:
        items = [item for item in items if item["generation_repeat"] == args.repeat]
    if args.index is not None:
        items = [items[args.index]]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no queue items matched")

    results = []
    for item in items:
        output = ROOT / str(item["candidate_video"])
        if output.is_file() and not args.dry_run:
            results.append({"status": "already_complete", "output": str(output)})
            continue
        result = run_item(item, args.source, args.checkpoint, args.python, args.dry_run)
        results.append(result)
        print(json.dumps(result), flush=True)
        if result["status"] == "failed" and not args.continue_on_error:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
