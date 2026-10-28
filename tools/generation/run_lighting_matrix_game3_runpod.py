#!/usr/bin/env python3
"""Generate MG3's five fixed lighting orbits with one pinned model allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml

from prepare_lighting_backend_inputs import prepare, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/workspace/WMEval-lighting-orbit'))
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2, 3, 4])
    args = p.parse_args()
    root = args.workspace.resolve()
    if not root.is_relative_to('/workspace'):
        p.error('all remote files must remain under /workspace')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be distinct')
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        p.error('deadline must include a timezone')
    import torch
    if torch.cuda.device_count() != 1:
        p.error('select exactly one 96GB GPU for this recipe using CUDA_VISIBLE_DEVICES')
    source = root / 'model_sources/matrix-game'
    src = json.loads((source / 'source_provenance.json').read_text())
    if src['revision'] != '756776d516233027631008761d4d40ddaea3e23a':
        raise ValueError('wrong pinned Matrix source')
    for name, expected in src['files'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'changed upstream source: {name}')
    model = root / 'models/matrix-game3'
    provenance = json.loads((model / 'checkpoint_provenance.json').read_text())
    if provenance['revision'] != '382aa382a03cec057360755aa72c7735025770bb':
        raise ValueError('wrong MG3 model revision')
    if 'models_t5_umt5-xxl-enc-bf16.pth' not in provenance['files']:
        raise ValueError('materialize the checksum-identical pinned T5 and its provenance first')
    for name, item in provenance['files'].items():
        if sha256(model / name) != item['stored_sha256']:
            raise ValueError(f'checkpoint checksum mismatch: {name}')
    inputs = root / 'inputs/mg3-orbit'
    adapter = prepare(args.case, inputs, 'matrix-game3')
    case = yaml.safe_load((args.case / 'case.yaml').read_text())
    parent = root / 'outputs/lighting_orbit/Matrix-Game3'
    parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        output = parent / f'seed_{seed}'; output.mkdir(exist_ok=True)
        receipt, video = output / 'receipt.json', output / 'video.mp4'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['status'] == 'completed' and old['adapter'] == adapter and video.is_file() and sha256(video) == old['video_sha256']:
                continue
            raise FileExistsError(f'preserve and diagnose incomplete run: {output}')
        runs.append({'seed': seed, 'receipt': str(receipt), 'video': str(video)})
    if not runs:
        return
    batch = parent / ('batch_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    batch.mkdir(); spec = batch / 'batch_spec.json'
    spec.write_text(json.dumps({'schema_version': 1, 'runs': runs}, indent=2) + '\n')
    wrapper = Path(__file__).parent / 'model_adapters/matrix_game3_orbit.py'
    command = [sys.executable, str(wrapper), '--source', str(source / 'Matrix-Game-3'),
        '--conditioning', str(inputs / 'conditioning.npz'), '--batch-spec', str(spec),
        '--ckpt_dir', str(model), '--image', str(args.case / case['reference']['first_frame']),
        '--prompt', case['prompt'], '--size', '704*1280', '--num_iterations', '4',
        '--num_inference_steps', '3', '--fa_version', '2', '--vae_type', 'mg_lightvae_v2',
        '--seed', str(runs[0]['seed']), '--output_dir', str(parent)]
    dependencies = {name: importlib.metadata.version(name) for name in
                    ('torch', 'flash-attn', 'diffusers', 'transformers', 'numpy', 'pandas', 'trimesh')}
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True)
    for item in runs:
        Path(item['receipt']).write_text(json.dumps({'schema_version': 1, 'status': 'queued',
            'model': 'Matrix-Game3', 'seed': item['seed'], 'adapter': adapter,
            'source_revision': src['revision'], 'model_revision': provenance['revision'],
            'checkpoint_manifest_sha256': sha256(model / 'checkpoint_provenance.json'),
            'runner_sha256': sha256(Path(__file__)), 'wrapper_sha256': sha256(wrapper),
            'batch_spec_sha256': sha256(spec), 'command': command, 'dependency_versions': dependencies,
            'hardware': hardware, 'native_steps': 3, 'quantization': None,
            'vae': 'Official MG-LightVAE_v2 with Wan2.2 encoder',
            'model_state': 'Eval weights reused; native seeded generator, memory and streaming VAE caches are local to each generate call.',
            'runtime': 'Pinned source executed in the verified Torch2.8 environment; dependency freeze recorded, separate from upstream requirements.',
            'checkpoint_io': 'Official Diffusers disable_mmap byte loader for local checkpoint, identical stored tensors.',
            'generation_log': str(batch / 'generation.log')}, indent=2) + '\n')
    process = None
    try:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 60:
            raise TimeoutError('global deadline reached')
        with (batch / 'generation.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source / 'Matrix-Game-3', stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=remaining - 30)
        if code:
            raise RuntimeError(f'MG3 generation failed: {batch}/generation.log')
        for item in runs:
            receipt = json.loads(Path(item['receipt']).read_text())
            if receipt['status'] != 'completed' or receipt['video_sha256'] != sha256(Path(item['video'])):
                raise ValueError('incomplete formal receipt')
            print(json.dumps({'seed': item['seed'], 'status': 'completed', 'elapsed_sec': receipt['elapsed_sec']}), flush=True)
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for item in runs:
            path = Path(item['receipt']); receipt = json.loads(path.read_text())
            if receipt['status'] in {'running', 'queued'}:
                receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
                path.write_text(json.dumps(receipt, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
