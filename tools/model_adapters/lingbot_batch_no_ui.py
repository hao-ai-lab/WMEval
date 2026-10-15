#!/usr/bin/env python3
"""Run independent fixed seeds with one frozen LingBot model allocation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import lingbot_no_ui as wrapper


def enable_sequential_checkpoint_reads(checkpoint_directory: str) -> None:
    # Diffusers 0.33.1 exposes disable_mmap, but its sharded dispatch omits
    # forwarding it. Scope this official load path to our local checkpoint.
    import diffusers.models.modeling_utils as loading
    original = loading.load_state_dict
    root = Path(checkpoint_directory).resolve()

    def sequential(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).suffix == '.safetensors' and Path(path).resolve().is_relative_to(root):
            kwargs['disable_mmap'] = True
        return original(path, *args, **kwargs)

    loading.load_state_dict = sequential


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-spec', type=Path, required=True)
    extra, native = parser.parse_known_args()
    upstream = wrapper.upstream
    import sys
    sys.argv = [sys.argv[0], *native]
    args = upstream._parse_args()
    enable_sequential_checkpoint_reads(args.ckpt_dir)
    spec = json.loads(extra.batch_spec.read_text())
    original_constructor = upstream.wan.WanI2V
    original_init = upstream.dist.init_process_group
    original_sp_init = upstream.init_distributed_group
    original_destroy = upstream.dist.destroy_process_group
    cache = None
    constructor_kwargs = None
    sp_initialized = False

    def model_once(**kwargs):
        nonlocal cache, constructor_kwargs
        if cache is None:
            constructor_kwargs = kwargs.copy()
            cache = original_constructor(**kwargs)
        elif kwargs != constructor_kwargs:
            raise ValueError('model constructor settings changed between seeds')
        return cache

    def init_once(*a, **kw):
        if not upstream.dist.is_initialized():
            original_init(*a, **kw)

    def sp_init_once():
        nonlocal sp_initialized
        if not sp_initialized:
            original_sp_init()
            sp_initialized = True

    upstream.wan.WanI2V = model_once
    upstream.dist.init_process_group = init_once
    upstream.init_distributed_group = sp_init_once
    upstream.dist.destroy_process_group = lambda: None
    rank = int(upstream.os.getenv('RANK', '0'))
    try:
        for item in spec['runs']:
            settings = copy.copy(args)
            settings.base_seed = item['seed']
            settings.save_file = item['video']
            path = Path(item['receipt'])
            receipt = json.loads(path.read_text())
            started = time.monotonic()
            if rank == 0:
                receipt.update(status='running', started_at_utc=datetime.now(timezone.utc).isoformat())
                path.write_text(json.dumps(receipt, indent=2)+'\n')
                print(f"WMEVAL_BATCH_SEED_START {item['seed']}", flush=True)
            try:
                # Native generate creates fresh noise, scheduler and seed_g for
                # each call. Only immutable eval-mode model weights are reused.
                upstream.generate(settings)
                if rank == 0:
                    video = Path(item['video'])
                    with video.open('rb') as handle:
                        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                    receipt.update(status='completed', returncode=0, video_sha256=digest,
                                   elapsed_sec=time.monotonic()-started,
                                   lighting_time_mapping={'schema_version':1, 'native_fps':16,
                                                          'canonical_zero_frame':0, 'video_sha256':digest})
                    path.write_text(json.dumps(receipt, indent=2)+'\n')
                    print(f"WMEVAL_BATCH_SEED_DONE {item['seed']}", flush=True)
            except BaseException as error:
                if rank == 0:
                    receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                                   elapsed_sec=time.monotonic()-started)
                    path.write_text(json.dumps(receipt, indent=2)+'\n')
                raise
    finally:
        if upstream.dist.is_initialized():
            original_destroy()


if __name__ == '__main__':
    main()
