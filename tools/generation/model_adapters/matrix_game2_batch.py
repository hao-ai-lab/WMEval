#!/usr/bin/env python3
"""Reuse MG2 eval weights, retaining native per-run noise and cache resets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--batch-spec', type=Path, required=True)
    args = p.parse_args()
    source, model = args.source.resolve(), args.model.resolve()
    manifest = json.loads((source.parent / 'source_provenance.json').read_text())
    if manifest['revision'] != '756776d516233027631008761d4d40ddaea3e23a':
        raise ValueError('wrong Matrix source revision')
    for name, expected in manifest['files'].items():
        if sha256(source.parent / name) != expected:
            raise ValueError(f'changed upstream source: {name}')
    provenance = json.loads((model / 'checkpoint_provenance.json').read_text())
    if provenance['revision'] != '08c42d1a99e57e79df60a964bb815618cdf460f8':
        raise ValueError('wrong MG2 model revision')
    spec = json.loads(args.batch_spec.read_text())
    sys.path.insert(0, str(source))
    os.chdir(source)
    import inference as upstream
    from diffusers.utils import export_to_video
    from safetensors.torch import load

    # Official safetensors byte loader avoids network-volume mmap page faults.
    upstream.load_file = lambda filename: load(Path(filename).read_bytes())

    def export(video, output, controls, mouse_icon, **kwargs):
        # Skip the duplicate diagnostic HUD file; preserve original native fps.
        if not kwargs.get('process_icon', True):
            export_to_video([f.astype(np.float32) / 255 for f in video], output, fps=12)
    upstream.process_video = export
    first = spec['runs'][0]
    native_args = SimpleNamespace(config_path=str(source / 'configs/inference_yaml/inference_universal.yaml'),
        checkpoint_path=str(model / 'base_distilled_model/base_distill.safetensors'),
        pretrained_model_path=str(model), img_path=first['image'], output_folder=first['output'],
        num_output_frames=first['latent_frames'], seed=first['seed'])
    upstream.set_seed(first['seed'])
    engine = upstream.InteractiveGameInference(native_args)
    for run in spec['runs']:
        out = Path(run['output']); out.mkdir(parents=True, exist_ok=True)
        receipt_path = out / 'receipt.json'
        if receipt_path.exists():
            raise FileExistsError(f'preserve previous run: {receipt_path}')
        controls = np.load(run['schedule'], allow_pickle=False)
        frames = 4 * (run['latent_frames'] - 1) + 1
        keyboard, mouse = controls['keyboard'], controls['mouse']
        if keyboard.shape != (frames, 4) or mouse.shape != (frames, 2):
            raise ValueError('schedule shape does not match native frame count')
        def schedule(requested, num_samples_per_action=4):
            if requested != frames:
                raise ValueError('native action duration changed')
            return {'keyboard_condition': torch.from_numpy(keyboard.copy()),
                    'mouse_condition': torch.from_numpy(mouse.copy())}
        upstream.Bench_actions_universal = schedule
        upstream.set_seed(run['seed'])
        engine.args.img_path = run['image']; engine.args.output_folder = str(out)
        engine.args.num_output_frames = run['latent_frames']; engine.args.seed = run['seed']
        engine.config['mode'] = 'universal'  # Native generate pops this scalar.
        receipt = {'schema_version': 1, 'status': 'running', 'model': 'Matrix-Game2',
            'seed': run['seed'], 'run_kind': spec['run_kind'], 'adapter': spec.get('adapter', {}),
            'source_revision': manifest['revision'], 'model_revision': provenance['revision'],
            'schedule_sha256': sha256(run['schedule']), 'seed_image_sha256': sha256(run['image']),
            'wrapper_sha256': sha256(Path(__file__)), 'batch_spec_sha256': sha256(args.batch_spec),
            'native_fps': 12, 'native_frames': frames, 'native_steps': [1000, 666, 333],
            'model_state': 'Native inference resets KV, cross-attention and streaming VAE caches per call.',
            'io': 'Official safetensors.torch.load(bytes), identical stored tensors; native compiled VAE retained.'}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        start = time.monotonic()
        try:
            engine.generate_videos()
            video = out / 'demo.mp4'
            receipt.update(status='completed', video_sha256=sha256(video))
            receipt['lighting_time_mapping'] = {'schema_version': 1, 'native_fps': 12,
                'canonical_zero_frame': 0, 'video_sha256': receipt['video_sha256']}
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['elapsed_sec'] = time.monotonic() - start
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'seed': run['seed'], 'output': str(out), 'status': receipt['status'],
                              'elapsed_sec': receipt['elapsed_sec']}), flush=True)


if __name__ == '__main__':
    main()
