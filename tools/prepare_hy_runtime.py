#!/usr/bin/env python3
"""Fetch only the pinned runtime files for the native HY orbit recipe.

Hugging Face uses its normal authorized login/HF_TOKEN environment. This tool
does not read credentials from a Pod or change account access. All downloaded
files and temporary originals stay in the designated task workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/workspace/WMEval-lighting-orbit'))
    args = p.parse_args()
    root = args.workspace.resolve()
    if not root.is_relative_to('/workspace'):
        p.error('all runtime assets must stay under /workspace')
    assets = Path(__file__).resolve().parents[1] / 'experiments/lighting_orbit'
    inventory = assets / 'hy_runtime_inventory.json'
    paths = {'tencent/HY-WorldPlay': root / 'models/hy-worldplay-action',
        'tencent/HunyuanVideo-1.5': root / 'models/hunyuan-video-1.5',
        'Qwen/Qwen2.5-VL-7B-Instruct': root / 'models/hunyuan-video-1.5/text_encoder/llm',
        'google/byt5-small': root / 'models/hunyuan-video-1.5/text_encoder/byt5-small',
        'black-forest-labs/FLUX.1-Redux-dev': root / 'models/hunyuan-video-1.5/vision_encoder/siglip'}
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    for row in json.loads(inventory.read_text()):
        repo = row['repo_id']
        command = [sys.executable, str(Path(__file__).parent / 'download_lighting_checkpoint.py'),
            '--inventory', str(inventory), '--repo', repo, '--output', str(paths[repo]),
            '--temporary', str(root / 'tmp/hy-download' / repo.replace('/', '_'))]
        if repo == 'tencent/HY-WorldPlay':
            command.append('--native-bf16-dit')
        subprocess.run(command, check=True)
    row = json.loads((assets / 'hy_glyph_inventory.json').read_text())
    if row['repo_id'] != 'AI-ModelScope/Glyph-SDXL-v2' or row['revision'] != '4bfc170a902feb27c44009fd3ea0b6a9525dbb2f':
        raise ValueError('unreviewed Glyph runtime inventory')
    output = root / 'models/hunyuan-video-1.5/text_encoder/Glyph-SDXL-v2'
    manifest = {'repo_id': row['repo_id'], 'revision': row['revision'], 'files': {}}
    for entry in row['files']:
        name = Path(entry['Path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('unsafe Glyph asset path')
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or digest(target) != entry['Sha256']:
            url = 'https://modelscope.cn/api/v1/models/' + row['repo_id'] + '/repo?' + urllib.parse.urlencode(
                {'Revision': row['revision'], 'FilePath': str(name)})
            temporary = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if digest(temporary) != entry['Sha256'] or temporary.stat().st_size != entry['Size']:
                raise ValueError(f'Glyph published checksum/size mismatch: {name}')
            temporary.replace(target)
        manifest['files'][str(name)] = {'sha256': entry['Sha256'], 'bytes': target.stat().st_size}
        print(json.dumps({'model': 'Glyph-SDXL-v2', 'file': str(name), 'status': 'verified'}), flush=True)
    (output / 'checkpoint_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('HY_RUNTIME_DOWNLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
