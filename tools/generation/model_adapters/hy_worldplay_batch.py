#!/usr/bin/env python3
"""Run pinned HY's native four-step AR pipeline once for five orbit seeds.

The complete action checkpoint replaces every base transformer tensor under
strict=True upstream. Allocate that same architecture on meta and assign the
complete action state instead of downloading/loading discarded base weights.
Missing keys, wrong shapes, or unmaterialized tensors remain fatal.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

SOURCE_REVISION = '1588e1336e842b03b0a7860c654ebd7c46bb065e'
MODEL_REVISION = 'f4c29235647707b571479a69b569e4166f9f5bf8'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_spec(spec):
    seeds = [r['seed'] for r in spec['runs']]
    if seeds != [42, 1, 2, 3, 4]:
        raise ValueError('requires the five fixed formal seeds in order')
    if len({r['output'] for r in spec['runs']}) != 5:
        raise ValueError('each independent seed requires its own output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--action-model', type=Path, required=True)
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--batch-spec', type=Path, required=True)
    parser.add_argument('--load-only', action='store_true')
    args = parser.parse_args()
    import torch
    import yaml
    from accelerate import init_empty_weights
    from safetensors.torch import load, load_file

    source = args.source.resolve()
    manifest = json.loads((source / 'source_manifest.json').read_text())
    if manifest['revision'] != SOURCE_REVISION:
        raise ValueError('unreviewed HY source revision')
    for name, digest in manifest['files'].items():
        if sha256(source / name) != digest:
            raise ValueError(f'HY source changed: {name}')
    action_root = args.action_model.resolve()
    action = action_root / 'ar_distilled_action_model/model.safetensors'
    provenance = json.loads((action_root / 'checkpoint_provenance.json').read_text())
    entry = provenance['files']['ar_distilled_action_model/model.safetensors']
    if provenance['revision'] != MODEL_REVISION or sha256(action) != entry['stored_sha256']:
        raise ValueError('action checkpoint pin/hash mismatch')
    spec = json.loads(args.batch_spec.read_text())
    validate_spec(spec)
    case = yaml.safe_load((args.case / 'case.yaml').read_text())
    conditioning = json.loads((args.conditioning / 'adapter_receipt.json').read_text())
    poses = args.conditioning / 'poses.json'
    if len(json.loads(poses.read_text())) != 64:
        raise ValueError('requires 64 native latent poses for 253 frames')
    sys.path.insert(0, str(source))
    os.chdir(source)
    import hyvideo.pipelines.worldplay_video_pipeline as native_pipeline
    transformer_class = native_pipeline.HunyuanVideo_1_5_DiffusionTransformer
    expected_base = (args.base_model / 'transformer/480p_i2v').resolve()
    native_from_pretrained = transformer_class.from_pretrained
    state_audit = {}

    def architecture_only(cls, path, **kwargs):
        if Path(path).resolve() != expected_base or kwargs.get('torch_dtype') != torch.bfloat16:
            raise ValueError('meta allocation is limited to the audited BF16 base architecture')
        config = cls.load_config(str(expected_base))
        with init_empty_weights():
            model = cls.from_config(config)
        model = model.to(dtype=torch.bfloat16)
        native_state_loader = model.load_state_dict

        def assign_complete(self, state_dict, strict=True, **kw):
            if not strict:
                raise ValueError('complete action state must load strictly')
            result = native_state_loader(state_dict, strict=True, assign=True)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError('incomplete action checkpoint')
            remaining = [n for n, v in list(self.named_parameters()) + list(self.named_buffers()) if v.is_meta]
            if remaining:
                raise ValueError(f'unmaterialized transformer tensors: {remaining[:8]}')
            state_audit.update(strict=True, missing_keys=[], unexpected_keys=[],
                remaining_meta_tensors=0, state_tensors=len(state_dict),
                parameter_count=sum(p.numel() for p in self.parameters()),
                base_weights_discarded_upstream=True)
            return result

        model.load_state_dict = types.MethodType(assign_complete, model)
        return model

    transformer_class.from_pretrained = classmethod(architecture_only)
    # Read the one pinned action state sequentially on network storage.
    import safetensors.torch as safe_loading
    def sequential_action(path, *a, **kw):
        if Path(path).resolve() == action:
            return load(action.read_bytes())
        return load_file(path, *a, **kw)
    safe_loading.load_file = sequential_action
    if args.load_only:
        model = transformer_class.from_pretrained(str(expected_base), torch_dtype=torch.bfloat16)
        model.add_action_parameters()
        model.load_state_dict(sequential_action(action), strict=True)
        model.to(torch.bfloat16)
        print(json.dumps({'status': 'strict_action_load_verified', **state_audit}), flush=True)
        return

    import hyvideo.generate as upstream
    native_create = upstream.HunyuanVideo_1_5_Pipeline.create_pipeline
    cached, fixed_kwargs = None, None
    def pipeline_once(**kwargs):
        nonlocal cached, fixed_kwargs
        if cached is None:
            fixed_kwargs = copy.copy(kwargs)
            cached = native_create(**kwargs)
        elif kwargs != fixed_kwargs:
            raise ValueError('native model settings changed between seeds')
        return cached
    upstream.HunyuanVideo_1_5_Pipeline.create_pipeline = pipeline_once
    rank = int(os.environ.get('RANK', '0'))
    dependencies = {p: importlib.metadata.version(p) for p in
                    ('torch', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'numpy')}
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                        '--format=csv,noheader'], text=True)
    runtime_manifests = {}
    for name in ('', 'text_encoder/llm', 'text_encoder/byt5-small',
                 'text_encoder/Glyph-SDXL-v2', 'vision_encoder/siglip'):
        path = args.base_model / name / 'checkpoint_provenance.json'
        runtime_manifests[name or 'base'] = json.loads(path.read_text())
    for run in spec['runs']:
        output = Path(run['output']); output.mkdir(parents=True, exist_ok=True)
        receipt_path = output / 'receipt.json'
        if receipt_path.exists():
            raise FileExistsError(f'preserve prior run: {output}')
        receipt = {'schema_version': 1, 'status': 'running', 'model': 'HY-WorldPlay',
            'seed': run['seed'], 'run_kind': 'formal', 'case_id': case['case_id'],
            'source_revision': SOURCE_REVISION, 'model_revision': MODEL_REVISION,
            'wrapper_sha256': sha256(Path(__file__)), 'batch_spec_sha256': sha256(args.batch_spec),
            'conditioning_receipt': conditioning, 'action_checkpoint': entry,
            'dependency_versions': dependencies, 'native_fps': 24, 'native_frames': 253,
            'hardware': hardware, 'runtime_checkpoint_manifests': runtime_manifests,
            'resolution': [640, 640], 'num_inference_steps': 4,
            'model_type': 'ar_distilled', 'quantization': None, 'torch_compile': False,
            'super_resolution': False, 'prompt_rewrite': False,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'seed_sha256': sha256(args.case / case['reference']['first_frame']),
            'poses_sha256': sha256(poses), 'prompt': case['prompt']}
        if rank == 0:
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        sys.argv = [str(source / 'hyvideo/generate.py'), '--model_path', str(args.base_model),
            '--action_ckpt', str(action), '--pose', str(poses), '--prompt', case['prompt'],
            '--image_path', str(args.case / case['reference']['first_frame']),
            '--output_path', str(output), '--seed', str(run['seed']), '--resolution', '480p',
            '--aspect_ratio', '1:1', '--width', '640', '--height', '640',
            '--video_length', '253', '--num_inference_steps', '4', '--few_step', 'true',
            '--model_type', 'ar', '--dtype', 'bf16', '--offloading', 'false',
            '--group_offloading', 'false', '--sr', 'false', '--rewrite', 'false',
            '--with-ui', 'false', '--enable_torch_compile', 'false', '--use_fp8_gemm', 'false',
            '--use_sageattn', 'false', '--use_vae_parallel', 'false',
            '--transformer_resident_ar_rollout', 'true']
        receipt['native_command'] = sys.argv.copy()
        start = time.monotonic()
        try:
            upstream.main()
            if rank == 0:
                video = output / 'gen.mp4'
                receipt.update(status='completed', video_sha256=sha256(video), state_load=state_audit)
                receipt['lighting_time_mapping'] = {'schema_version': 1, 'native_fps': 24,
                    'canonical_zero_frame': 0, 'video_sha256': receipt['video_sha256']}
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if rank == 0:
                receipt['elapsed_sec'] = time.monotonic() - start
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                print(json.dumps({'seed': run['seed'], 'status': receipt['status'],
                                  'elapsed_sec': receipt['elapsed_sec']}), flush=True)
    transformer_class.from_pretrained = native_from_pretrained


if __name__ == '__main__':
    main()
