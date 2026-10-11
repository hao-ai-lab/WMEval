#!/usr/bin/env python3
"""Supply the canonical orbit to pinned MG3's native camera conditioning."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--batch-spec', type=Path)
    args, native_args = parser.parse_known_args()
    data = np.load(args.conditioning, allow_pickle=False)
    poses, keyboard, mouse = (data[name] for name in ('poses', 'keyboard', 'mouse'))
    if poses.shape != (177, 4, 4) or keyboard.shape != (177, 6) or mouse.shape != (177, 2):
        raise ValueError('requires the complete 177-frame native orbit conditioning')
    seed_k = data['seed_intrinsics']
    sys.path.insert(0, str(args.source.resolve()))
    os.chdir(args.source)
    import pipeline.inference_pipeline as pipeline
    import generate as upstream

    def get_data(frames, height, width, image, device=None, dtype=None):
        if frames != 177 or image.size != (1024, 1024):
            raise ValueError('native clip count or square conditioning image changed')
        # The native latent canvas is rounded down to a multiple of 32.
        side = int(np.sqrt(height * width) // 32 * 32)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255
        pixels = functional.interpolate(pixels, size=(side, side), mode='bilinear',
                                        align_corners=False, antialias=True)
        pixels = (2 * pixels - 1).unsqueeze(2).to(device=device, dtype=dtype)
        return (pixels, torch.from_numpy(poses.copy()),
                torch.from_numpy(keyboard.copy())[None].to(device=device, dtype=dtype),
                torch.from_numpy(mouse.copy())[None].to(device=device, dtype=dtype))

    def get_intrinsics(height, width):
        if height != width:
            raise ValueError('square seed must keep its aspect ratio')
        return torch.tensor([seed_k[0, 0] * width / 1024, seed_k[1, 1] * height / 1024,
                             seed_k[0, 2] * width / 1024, seed_k[1, 2] * height / 1024])

    def process_video(video, output, controls, mouse_icon, **kwargs):
        from diffusers.utils import export_to_video
        export_to_video([frame.astype(np.float32) / 255 for frame in video], output, fps=17)

    pipeline.get_data = get_data
    pipeline.get_intrinsics = get_intrinsics
    pipeline.process_video = process_video
    sys.argv = [str(args.source / 'generate.py'), *native_args]
    parsed = upstream._parse_args()
    if parsed.interactive or parsed.num_iterations != 4:
        raise ValueError('requires the native pose pipeline with four iterations')
    if parsed.use_int8 or parsed.num_inference_steps != 3 or parsed.use_base_model:
        raise ValueError('requires the unquantized three-step distilled model')
    if args.batch_spec is None:
        upstream.generate(parsed)
        return
    spec = json.loads(args.batch_spec.read_text())
    original_constructor = upstream.MatrixGame3Pipeline
    original_init, original_destroy = upstream.dist.init_process_group, upstream.dist.destroy_process_group
    original_sp_init = upstream.init_distributed_group
    cached, fixed_kwargs, sp_initialized = None, None, False

    def model_once(**kwargs):
        nonlocal cached, fixed_kwargs
        immutable = {k: v for k, v in kwargs.items() if k != 'args'}
        if cached is None:
            fixed_kwargs = immutable
            cached = original_constructor(**kwargs)
        elif immutable != fixed_kwargs:
            raise ValueError('model settings changed between fixed seeds')
        return cached

    def init_once(*a, **kw):
        if not upstream.dist.is_initialized():
            original_init(*a, **kw)

    def sp_once():
        nonlocal sp_initialized
        if not sp_initialized:
            original_sp_init(); sp_initialized = True

    def sha256(path):
        with Path(path).open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()

    upstream.MatrixGame3Pipeline = model_once
    upstream.dist.init_process_group = init_once
    upstream.dist.destroy_process_group = lambda: None
    upstream.init_distributed_group = sp_once
    # Use the official Diffusers byte loader for our local derived checkpoint.
    import diffusers.models.modeling_utils as loading
    original_load = loading.load_state_dict
    model_root = Path(parsed.ckpt_dir).resolve()
    def sequential(path, *a, **kw):
        if isinstance(path, (str, Path)) and Path(path).suffix == '.safetensors' and Path(path).resolve().is_relative_to(model_root):
            kw['disable_mmap'] = True
        return original_load(path, *a, **kw)
    loading.load_state_dict = sequential
    rank = int(os.environ.get('RANK', '0'))
    try:
        for item in spec['runs']:
            settings = copy.copy(parsed)
            settings.seed = item['seed']; settings.save_name = f'seed_{item["seed"]}'
            receipt_path = Path(item['receipt'])
            if rank == 0:
                receipt = json.loads(receipt_path.read_text())
                receipt.update(status='running', started_at_utc=datetime.now(timezone.utc).isoformat())
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            started = time.monotonic()
            try:
                upstream.generate(settings)
                if rank == 0:
                    video = Path(item['video'])
                    video.parent.mkdir(parents=True, exist_ok=True)
                    (Path(settings.output_dir) / f'{settings.save_name}.mp4').replace(video)
                    receipt.update(status='completed', elapsed_sec=time.monotonic() - started,
                                   video_sha256=sha256(video))
                    receipt['lighting_time_mapping'] = {'schema_version': 1, 'native_fps': 17,
                        'canonical_zero_frame': 0, 'video_sha256': receipt['video_sha256']}
                    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                    print(json.dumps({'seed': item['seed'], 'status': 'completed',
                                      'elapsed_sec': receipt['elapsed_sec']}), flush=True)
            except BaseException as error:
                if rank == 0:
                    receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                                   elapsed_sec=time.monotonic() - started)
                    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                raise
            if upstream.dist.is_initialized():
                upstream.dist.barrier()
    finally:
        if upstream.dist.is_initialized():
            original_destroy()


if __name__ == '__main__':
    main()
