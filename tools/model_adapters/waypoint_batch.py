#!/usr/bin/env python3
"""Run pinned Waypoint controls with native four-frame steps and cache resets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--batch-spec', type=Path, required=True)
    p.add_argument('--compile-profile', choices=['native', 'bounded_startup'], default='native')
    args = p.parse_args()
    source_manifest = json.loads((args.source / 'source_provenance.json').read_text())
    if source_manifest['revision'] != 'b3f1e725b222679a517632918cc78bba0c9fa433':
        raise ValueError('wrong pinned WorldEngine source')
    for name, expected in source_manifest['files'].items():
        if sha256(args.source / name) != expected:
            raise ValueError(f'changed WorldEngine source: {name}')
    model_manifest = json.loads((args.model / 'checkpoint_provenance.json').read_text())
    if model_manifest['revision'] != '391f92827075edcf4a8b3c8a2ddae010698f8636':
        raise ValueError('wrong pinned Waypoint model')
    for name, entry in model_manifest['files'].items():
        if sha256(args.model / name) != entry['stored_sha256']:
            raise ValueError(f'changed Waypoint checkpoint: {name}')
    if sha256(args.vae) != '806f78a06266ba58bb982d8680add1a249aefcc96237e8a0aa60298617744682':
        raise ValueError('wrong pinned TAEHV1.5 checkpoint')
    # Keep the native model and FlexAttention graph, while avoiding the long
    # exhaustive matmul autotuning pass on a fresh GPU/network-volume cache.
    native_compile = torch.compile
    compile_options = {'max_autotune': False, 'coordinate_descent_tuning': False,
                       'triton.cudagraphs': False}
    if args.compile_profile == 'bounded_startup':
        def bounded_compile(model=None, **kwargs):
            kwargs.pop('mode', None)
            kwargs['options'] = {**kwargs.get('options', {}), **compile_options}
            return native_compile(model, **kwargs)
        torch.compile = bounded_compile
    try:
        from world_engine import WorldEngine, CtrlInput
    finally:
        torch.compile = native_compile
    import world_engine
    installed = Path(world_engine.__file__).resolve().parent
    for name, expected in source_manifest['files'].items():
        if name.startswith('src/') and sha256(installed / name[4:]) != expected:
            raise ValueError(f'installed WorldEngine differs from pinned source: {name}')
    engine = WorldEngine(str(args.model), quant=None, device='cuda',
                         model_config_overrides={'ae_uri': str(args.vae)})
    if engine.model_cfg.inference_fps != 60 or engine.model_cfg.temporal_compression != 4:
        raise ValueError('native timing changed')
    if engine.model_cfg.prompt_conditioning is not None:
        raise ValueError('this frozen model uses image plus controls, without text input')
    spec = json.loads(args.batch_spec.read_text())
    dependencies = {name: importlib.metadata.version(name) for name in
                    ('torch', 'triton', 'transformers', 'huggingface-hub', 'tensordict', 'world_engine', 'taehv')}
    for run in spec['runs']:
        output = Path(run['output']); output.mkdir(parents=True, exist_ok=True)
        receipt_path = output / 'receipt.json'
        if receipt_path.exists():
            raise FileExistsError(f'preserve prior run: {output}')
        controls = np.load(run['schedule'], allow_pickle=False)
        keys, mouse = controls['keyboard'], controls['mouse']
        if len(keys) != len(mouse) or keys.shape[1] != 4 or mouse.shape[1] != 2:
            raise ValueError('invalid per-step controls')
        image = np.asarray(Image.open(run['image']).convert('RGB'))
        if image.shape != (720, 1280, 3):
            raise ValueError('native seed must use recorded aspect-preserving 1280x720 canvas')
        seed_x4 = torch.from_numpy(np.repeat(image[None], 4, axis=0))
        torch.manual_seed(run['seed']); np.random.seed(run['seed']); random.seed(run['seed'])
        engine.reset()
        video = output / 'video.mp4'
        writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '60', '-i', '-',
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
            stdin=subprocess.PIPE)
        receipt = {'schema_version': 1, 'status': 'running', 'model': 'Waypoint',
            'seed': run['seed'], 'run_kind': spec['run_kind'], 'adapter': spec['adapter'],
            'source_revision': source_manifest['revision'], 'model_revision': model_manifest['revision'],
            'vae_model_revision': 'a0253886b13b9c4c3bd224bd479be03f5988a3df', 'vae_sha256': sha256(args.vae),
            'schedule_sha256': sha256(run['schedule']), 'seed_canvas_sha256': sha256(run['image']),
            'wrapper_sha256': sha256(Path(__file__)), 'batch_spec_sha256': sha256(args.batch_spec),
            'dependency_versions': dependencies, 'native_fps': 60, 'frames_per_step': 4,
            'canonical_zero_frame': 4, 'native_frames': 4 + 4 * len(keys),
            'native_scheduler_sigmas': list(engine.model_cfg.scheduler_sigmas), 'quantization': None,
            'compile_profile': args.compile_profile,
            'compile_option_overrides': compile_options if args.compile_profile == 'bounded_startup' else {},
            'text_conditioning': None, 'model_state': 'Native reset clears KV/streaming VAE/timestamps per seed; eval weights reused.'}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        start = time.monotonic()
        try:
            engine.append_frame(seed_x4)
            writer.stdin.write(seed_x4.numpy().tobytes())
            button_ids = [87, 83, 65, 68]
            for i, (key, motion) in enumerate(zip(keys, mouse)):
                buttons = {button_ids[j] for j, active in enumerate(key) if active}
                frames = engine.gen_frame(ctrl=CtrlInput(button=buttons, mouse=tuple(motion.tolist()))).cpu().numpy()
                if frames.shape != (4, 720, 1280, 3):
                    raise ValueError('native decoded frame count or dimensions changed')
                writer.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
                if i % 15 == 0:
                    print(json.dumps({'seed': run['seed'], 'step': i, 'steps': len(keys),
                                      'elapsed_sec': time.monotonic() - start}), flush=True)
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError('native video encoder failed')
            receipt.update(status='completed', video_sha256=sha256(video))
            receipt['lighting_time_mapping'] = {'schema_version': 1, 'native_fps': 60,
                'canonical_zero_frame': 4, 'video_sha256': receipt['video_sha256']}
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            if writer.poll() is None:
                writer.terminate(); writer.wait()
            raise
        finally:
            receipt['elapsed_sec'] = time.monotonic() - start
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'seed': run['seed'], 'status': receipt['status'],
                              'elapsed_sec': receipt['elapsed_sec']}), flush=True)


if __name__ == '__main__':
    main()
