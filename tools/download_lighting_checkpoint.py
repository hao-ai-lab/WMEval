#!/usr/bin/env python3
"""Fetch pinned checkpoint files, optionally storing native BF16 DiT weights.

FP32-to-BF16 conversion matches LingBot's explicit torch_dtype=bfloat16 load;
it is recorded as a derived asset, not represented as the original HF blob.
Download and convert one shard at a time to bound disk/RAM on a cheap Pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def finalize_shard_indices(output: Path, manifest: dict) -> None:
    """Retain source provenance while declaring derived shards' real size."""
    for name, provenance in manifest['files'].items():
        if not name.endswith('.safetensors.index.json'):
            continue
        path = output / name
        index = json.loads(path.read_text())
        prefix = str(Path(name).parent)
        shards = sorted(set(index['weight_map'].values()))
        entries = [manifest['files'][f'{prefix}/{shard}'] for shard in shards]
        if not all(entry['conversion'] == 'float32_to_bfloat16' for entry in entries):
            continue
        index.setdefault('metadata', {})['total_size'] = sum(entry['tensor_data_bytes'] for entry in entries)
        path.write_text(json.dumps(index, indent=2) + '\n')
        provenance.update(stored_sha256=sha256(path), stored_bytes=path.stat().st_size,
                          conversion='native_bfloat16_shard_index_metadata')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temporary', type=Path, required=True)
    parser.add_argument('--native-bf16-dit', action='store_true')
    args = parser.parse_args()
    output, temporary = args.output.resolve(), args.temporary.resolve()
    for path in (output, temporary):
        if not path.is_relative_to('/workspace'):
            parser.error('all remote checkpoint and temporary paths must be under /workspace')
    if output == temporary or output.is_relative_to(temporary) or temporary.is_relative_to(output):
        parser.error('output and temporary directories must be separate')
    rows = json.loads(args.inventory.read_text())
    row = next(r for r in rows if r['repo_id'] == args.repo)
    revision = row['revision']
    if len(revision) != 40:
        parser.error('requires a pinned 40-character HF revision')
    if args.native_bf16_dit and args.repo != 'robbyant/lingbot-world-base-cam':
        parser.error('native BF16 conversion has only been audited for LingBot Cam')
    # Only select runtime inputs; do not download demo media or other models.
    files = [f for f in row['files'] if f['path'].endswith(('.json', '.pth', '.safetensors', '.model'))]
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'checkpoint_provenance.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'schema_version': 1, 'repo_id': args.repo, 'revision': revision, 'files': {},
        'conversion': 'FP32 DiT to native BF16, matching upstream from_pretrained(torch_dtype=torch.bfloat16)' if args.native_bf16_dit else 'none'}
    if manifest['repo_id'] != args.repo or manifest['revision'] != revision:
        raise ValueError('existing output belongs to a different checkpoint revision')
    os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
    from huggingface_hub import hf_hub_download
    for info in files:
        name = info['path']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe checkpoint path')
        destination = output / name
        done = manifest['files'].get(name)
        if done and destination.is_file() and sha256(destination) == done['stored_sha256']:
            print(json.dumps({'file': name, 'status': 'verified_existing'}), flush=True)
            continue
        start = time.monotonic()
        print(json.dumps({'file': name, 'source_bytes': info['bytes'], 'status': 'downloading'}), flush=True)
        source = Path(hf_hub_download(args.repo, name, revision=revision, local_dir=temporary))
        original_hash = sha256(source)
        if info.get('sha256') and original_hash != info['sha256']:
            raise ValueError(f'published HF LFS checksum mismatch: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        convert = args.native_bf16_dit and name.endswith('.safetensors') and name.startswith(('high_noise_model/', 'low_noise_model/'))
        tensor_bytes = None
        if convert:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
            with safe_open(source, framework='pt', device='cpu') as handle:
                metadata = handle.metadata()
                values = {}
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    values[key] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
                tensor_bytes = sum(t.numel() * t.element_size() for t in values.values())
                save_file(values, str(destination), metadata=metadata)
            del values
            source.unlink()  # Only this invocation's original public download.
        else:
            source.replace(destination)
        manifest['files'][name] = {
            'source_sha256': original_hash, 'source_bytes': info['bytes'],
            'stored_sha256': sha256(destination), 'stored_bytes': destination.stat().st_size,
            'conversion': 'float32_to_bfloat16' if convert else 'none',
            'tensor_data_bytes': tensor_bytes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'file': name, 'status': 'completed', 'elapsed_sec': round(time.monotonic()-start, 2), 'stored_bytes': destination.stat().st_size}), flush=True)
    finalize_shard_indices(output, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'repo_id': args.repo, 'revision': revision,
                      'stored_bytes': sum(r['stored_bytes'] for r in manifest['files'].values())}), flush=True)


if __name__ == '__main__':
    main()
