#!/usr/bin/env python3
"""Load LingBot Base Cam once and run a JSONL batch without UI overlays."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

import torch
import torch.distributed as dist
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_complete(candidate: Path, job: dict | None = None) -> bool:
    """Return true only for a candidate with a matching complete receipt.

    Batch job lists are intentionally rechecked at execution time.  This lets
    a newly freed GPU finish rows from a slower shard while the original shard
    safely skips those rows when it reaches them.
    """
    receipt_path = candidate.parent / "generation_receipt.json"
    if not candidate.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        receipt.get("status") != "complete"
        or receipt.get("candidate_video_sha256") != sha256(candidate)
    ):
        return False
    if job is None:
        return True
    expected = dict(job.get("receipt_static") or {})
    expected.update(
        case_id=job.get("case_id"),
        generation_repeat=job.get("repeat"),
    )
    return all(receipt.get(key) == value for key, value in expected.items())


def acquire_generation_claim(candidate: Path, *, blocking: bool = True) -> TextIO:
    """Acquire the process-scoped lock for one candidate output."""
    candidate.parent.mkdir(parents=True, exist_ok=True)
    handle = (candidate.parent / "generation_claim.lock").open(
        "a+", encoding="utf-8"
    )
    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(handle.fileno(), operation)
    except BaseException:
        handle.close()
        raise
    return handle


def release_generation_claim(handle: TextIO) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    jobs_path = args.jobs.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)

    import wan  # noqa: PLC0415
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: PLC0415
    from wan.distributed.util import init_distributed_group  # noqa: PLC0415
    from wan.utils.utils import save_video  # noqa: PLC0415

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {1, 8}:
        raise ValueError(f"LingBot batch requires either 1 or 8 processes, got {world_size}")
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", local_rank),
        )
        init_distributed_group()
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.checkpoint.resolve()),
        device_id=local_rank,
        rank=rank,
        t5_fsdp=distributed,
        dit_fsdp=distributed,
        use_sp=distributed,
        # This is the memory-saving single-GPU path used by the upstream
        # generate.py entrypoint. Distributed FSDP keeps T5 on CUDA.
        t5_cpu=not distributed,
        convert_model_dtype=False,
    )
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line]
    for index, job in enumerate(jobs, 1):
        output = Path(job["save_file"])
        claim_handle = None
        if rank == 0:
            logging.info("Waiting to claim %s (%s/%s)", output, index, len(jobs))
            claim_handle = acquire_generation_claim(output)
        skip_complete = generation_complete(output, job) if rank == 0 else False
        if distributed:
            skip_message = [skip_complete]
            dist.broadcast_object_list(skip_message, src=0)
            skip_complete = bool(skip_message[0])
        if skip_complete:
            if rank == 0:
                logging.info("Skipping receipt-complete %s (%s/%s)", output, index, len(jobs))
                release_generation_claim(claim_handle)
            continue
        if rank == 0:
            logging.info("Generating %s (%s/%s)", output, index, len(jobs))
        try:
            if distributed:
                dist.barrier()
            image = Image.open(job["image"]).convert("RGB")
            video = pipeline.generate(
                job["prompt"],
                image,
                action_path=job["action_path"],
                max_area=MAX_AREA_CONFIGS["480*832"],
                frame_num=int(job.get("frame_num", 161)),
                shift=cfg.sample_shift,
                sample_solver="unipc",
                sampling_steps=cfg.sample_steps,
                guide_scale=cfg.sample_guide_scale,
                seed=int(job["seed"]),
                offload_model=not distributed,
                vis_ui=False,
            )
            if rank == 0:
                save_video(
                    tensor=video[None],
                    save_file=str(output),
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
            del video
            torch.cuda.synchronize()
            if rank == 0:
                receipt = dict(job.get("receipt_static") or {})
                receipt.update(
                    status="complete",
                    incremental_batch_receipt=True,
                    candidate_video=str(output),
                    candidate_video_sha256=sha256(output),
                    candidate_video_bytes=output.stat().st_size,
                )
                (output.parent / "generation_receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if distributed:
                dist.barrier()
        finally:
            if rank == 0 and claim_handle is not None:
                release_generation_claim(claim_handle)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
