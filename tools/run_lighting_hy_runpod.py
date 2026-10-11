#!/usr/bin/env python3
"""Run five pinned HY lighting orbits, terminating workers at a UTC deadline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from prepare_lighting_backend_inputs import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/workspace/WMEval-lighting-orbit'))
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    root, case = args.workspace.resolve(), args.case.resolve()
    if not root.is_relative_to('/workspace') or not case.is_relative_to(root):
        parser.error('workspace and case must stay inside the designated /workspace task directory')
    for relative in ('models/hunyuan-video-1.5/vision_encoder/siglip/image_encoder/model.safetensors',
                     'models/hunyuan-video-1.5/vision_encoder/siglip/feature_extractor/preprocessor_config.json',
                     'models/hunyuan-video-1.5/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt'):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        parser.error('deadline must include a timezone')
    import torch
    count = torch.cuda.device_count()
    if count not in (1, 2) or any(torch.cuda.get_device_properties(i).total_memory < 90e9 for i in range(count)):
        parser.error('this resident native recipe requires one or two GPUs with at least 90GB each')
    conditioning = root / 'backend_inputs/hy-worldplay'
    prepare(case, conditioning, 'hy-worldplay')
    parent = root / 'outputs/lighting_orbit_v2/HY-WorldPlay'
    runs = [{'seed': seed, 'output': str(parent / f'seed_{seed}')} for seed in (42, 1, 2, 3, 4)]
    if any((Path(run['output']) / 'receipt.json').exists() for run in runs):
        raise FileExistsError('preserve previous formal receipts; diagnose before any new batch')
    batch = parent / ('batch_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    batch.mkdir(parents=True)
    spec = batch / 'batch_spec.json'
    spec.write_text(json.dumps({'schema_version': 1, 'runs': runs}, indent=2) + '\n')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={count}', str(Path(__file__).parent / 'model_adapters/hy_worldplay_batch.py'),
        '--source', str(root / 'model_sources/hy-worldplay'),
        '--base-model', str(root / 'models/hunyuan-video-1.5'),
        '--action-model', str(root / 'models/hy-worldplay-action'),
        '--conditioning', str(conditioning), '--case', str(case), '--batch-spec', str(spec)]
    (batch / 'launch.json').write_text(json.dumps({'command': command,
        'deadline_utc': deadline.isoformat(), 'visible_gpu_count': count}, indent=2) + '\n')
    process = None
    try:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 60:
            raise TimeoutError('generation deadline reached')
        with (batch / 'generation.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=root / 'model_sources/hy-worldplay')
            if process.wait(timeout=remaining - 30):
                raise RuntimeError(f'HY native generation failed: {batch}/generation.log')
        for run in runs:
            receipt = json.loads((Path(run['output']) / 'receipt.json').read_text())
            if receipt['status'] != 'completed':
                raise ValueError('incomplete formal HY receipt')
        print(json.dumps({'status': 'completed', 'seeds': [r['seed'] for r in runs]}), flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    main()
