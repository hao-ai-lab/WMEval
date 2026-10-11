#!/usr/bin/env python3
"""Run fixed LingBot seeds on a prepared two-GPU Runpod network workspace.

This runner never creates or leaves a Pod running. The operator must provision
and terminate it separately. --deadline-utc bounds the entire generation batch;
the parent always kills the torchrun process group when time expires.
"""
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
import time

import yaml

from prepare_lighting_backend_inputs import prepare, sha256

SOURCE_REVISION = 'a43bec7f8091c83e9b30b16b912f6fc906236fa6'
MODEL_REVISION = '6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd'
DEPENDENCIES = {'torch':'2.8.0', 'flash-attn':'2.8.3', 'transformers':'4.51.3',
                'diffusers':'0.33.1', 'numpy':'1.26.4', 'scipy':'1.15.3'}


def run_batch(args, root, source, model, case, adapter, hardware, dependencies, deadline):
    """Keep weights resident while native generate resets seed-local state."""
    parent = root/'outputs/lighting_orbit_v2/LingBot'
    pending = []
    for seed in args.seeds:
        output = parent/f'seed_{seed}'
        output.mkdir(parents=True, exist_ok=True)
        path, video = output/'receipt.json', output/'video.mp4'
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get('status') == 'completed' and prior.get('adapter') == adapter and video.is_file() and prior.get('video_sha256') == sha256(video):
                print(json.dumps({'seed':seed, 'status':'verified_existing'}), flush=True)
                continue
            raise FileExistsError(f'existing incomplete run requires explicit diagnosis: {output}')
        pending.append({'seed':seed, 'receipt':str(path), 'video':str(video)})
    if not pending:
        return
    batch = parent/('batch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    batch.mkdir()
    spec = batch/'batch_spec.json'
    spec.write_text(json.dumps({'schema_version':1, 'runs':pending}, indent=2)+'\n')
    wrapper = Path(__file__).parent/'model_adapters/lingbot_batch_no_ui.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', str(wrapper),
               '--batch-spec',str(spec), '--task','i2v-A14B', '--size','480*832', '--ckpt_dir',str(model),
               '--image',str(args.case/case['reference']['first_frame']), '--action_path',str(root/'inputs/lingbot-orbit-v2'),
               '--dit_fsdp','--t5_fsdp','--ulysses_size','2','--frame_num','161','--sample_steps','20',
               '--base_seed',str(pending[0]['seed']), '--prompt',case['prompt']]
    for item in pending:
        path = Path(item['receipt'])
        path.write_text(json.dumps({'schema_version':1, 'status':'queued', 'seed':item['seed'],
            'model':'LingBot', 'source_revision':SOURCE_REVISION, 'model_revision':MODEL_REVISION,
            'adapter':adapter, 'command':command, 'hardware':hardware, 'dependency_versions':dependencies,
            'python_version':sys.version, 'checkpoint_manifest_sha256':sha256(model/'checkpoint_provenance.json'),
            'runner_sha256':sha256(Path(__file__)), 'wrapper_sha256':sha256(wrapper),
            'no_ui_wrapper_sha256':sha256(wrapper.with_name('lingbot_no_ui.py')),
            'batch_spec_sha256':sha256(spec), 'shared_model_allocation':True,
            'checkpoint_io':'Official Diffusers disable_mmap path, scoped shim for 0.33.1 sharded dispatch; identical stored tensors.',
            'seed_state':'Native generate recreates noise generator and scheduler per call; eval weights reused.',
            'generation_log':str(batch/'generation.log')}, indent=2)+'\n')
        (path.parent/'generation.log').symlink_to(Path('..')/batch.name/'generation.log')
    process = None
    try:
        remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
        if remaining <= 60:
            raise TimeoutError('generation deadline reached; release the Pod')
        with (batch/'generation.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=remaining-30)
        if code != 0:
            raise RuntimeError(f'batched generation failed: {batch}/generation.log')
        for item in pending:
            receipt = json.loads(Path(item['receipt']).read_text())
            if receipt['status'] != 'completed' or receipt['video_sha256'] != sha256(Path(item['video'])):
                raise ValueError(f'incomplete or mismatched batch receipt: {item["seed"]}')
            print(json.dumps({'seed':item['seed'], 'status':'completed', 'elapsed_sec':receipt['elapsed_sec']}), flush=True)
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for item in pending:
            path = Path(item['receipt']); receipt = json.loads(path.read_text())
            if receipt['status'] in {'running','queued'}:
                receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
                path.write_text(json.dumps(receipt, indent=2)+'\n')
        raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/workspace/WMEval-lighting-orbit'))
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,1,2,3,4])
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if deadline.tzinfo is None:
        parser.error('deadline must include a timezone')
    root=args.workspace.resolve()
    if not root.is_relative_to('/workspace'):
        parser.error('all remote files must be on /workspace')
    import torch
    dependencies={name:importlib.metadata.version(name) for name in DEPENDENCIES}
    if any(dependencies[name].split('+')[0]!=version for name,version in DEPENDENCIES.items()):
        raise ValueError(f'generation dependency versions changed: {dependencies}')
    if torch.cuda.device_count()!=2:
        parser.error('this frozen recipe requires exactly two GPUs')
    model=root/'models/lingbot-world-base-cam'
    provenance=json.loads((model/'checkpoint_provenance.json').read_text())
    if provenance['revision']!=MODEL_REVISION:
        raise ValueError('wrong pinned checkpoint revision')
    # Confirm every indexed shard is present, including final metadata updates.
    for directory in ['high_noise_model','low_noise_model']:
        index=json.loads((model/directory/'diffusion_pytorch_model.safetensors.index.json').read_text())
        for shard in set(index['weight_map'].values()):
            if not (model/directory/shard).is_file():
                raise FileNotFoundError(shard)
    source=root/'model_sources/lingbot-world'
    source_manifest=json.loads((source/'source_provenance.json').read_text())
    if source_manifest['revision']!=SOURCE_REVISION:
        raise ValueError('wrong pinned upstream source revision')
    for name, expected in source_manifest['files'].items():
        if sha256(source/name)!=expected:
            raise ValueError(f'upstream source changed: {name}')
    inputs=root/'inputs/lingbot-orbit-v2'
    adapter=prepare(args.case,inputs,'lingbot')
    case=yaml.safe_load((args.case/'case.yaml').read_text())
    wrapper=Path(__file__).parent/'model_adapters/lingbot_no_ui.py'
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True)
    os.environ['LINGBOT_SOURCE']=str(source)
    os.environ['OMP_NUM_THREADS']='4'
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct')
    if len(args.seeds) > 1:
        run_batch(args, root, source, model, case, adapter, hardware, dependencies, deadline)
        return
    for seed in args.seeds:
        remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
        if remaining<=60:
            raise TimeoutError('generation deadline reached; release the Pod')
        output=root/'outputs/lighting_orbit_v2/LingBot'/f'seed_{seed}'
        output.mkdir(parents=True,exist_ok=True)
        receipt_path=output/'receipt.json'
        video=output/'video.mp4'
        if receipt_path.exists():
            prior=json.loads(receipt_path.read_text())
            if prior.get('status')=='completed' and prior.get('adapter')==adapter and video.exists() and prior.get('video_sha256')==sha256(video):
                print(json.dumps({'seed':seed,'status':'verified_existing'}),flush=True)
                continue
            raise FileExistsError(f'existing incomplete run requires explicit diagnosis: {output}')
        command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',str(wrapper),
                 '--task','i2v-A14B','--size','480*832','--ckpt_dir',str(model),
                 '--image',str(args.case/case['reference']['first_frame']),
                 '--action_path',str(inputs),'--dit_fsdp','--t5_fsdp','--ulysses_size','2',
                 '--frame_num','161','--sample_steps','20','--base_seed',str(seed),
                 '--prompt',case['prompt'],'--save_file',str(video)]
        receipt={'schema_version':1,'status':'running','seed':seed,'model':'LingBot',
                 'source_revision':SOURCE_REVISION,'model_revision':MODEL_REVISION,
                 'adapter':adapter,'command':command,'hardware':hardware,
                 'dependency_versions':dependencies,'python_version':sys.version,
                 'checkpoint_manifest_sha256':sha256(model/'checkpoint_provenance.json'),
                 'runner_sha256':sha256(Path(__file__)),'wrapper_sha256':sha256(wrapper),
                 'started_at_utc':datetime.now(timezone.utc).isoformat()}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        started=time.monotonic()
        process=None
        try:
            with (output/'generation.log').open('w') as log:
                process=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=process.wait(timeout=remaining-30)
            receipt['returncode']=code
            if code!=0 or not video.exists():
                raise RuntimeError(f'generation failed: {output}/generation.log')
            receipt.update(status='completed',video_sha256=sha256(video))
            receipt['lighting_time_mapping']={'schema_version':1,'native_fps':16,
                                              'canonical_zero_frame':0,'video_sha256':receipt['video_sha256']}
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
            receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['elapsed_sec']=time.monotonic()-started
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({'seed':seed,'status':receipt['status'],'elapsed_sec':receipt['elapsed_sec']}),flush=True)


if __name__=='__main__':
    main()
