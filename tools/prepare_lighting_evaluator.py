#!/usr/bin/env python3
"""Populate a dedicated, pinned HF cache for offline lighting evaluation.

The existing evaluator takes model IDs rather than revisions. Bind only this
task's cache's main refs to the locked revisions, verify the actual weight
checksums, and set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 when scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,required=True)
    args=parser.parse_args()
    cache=Path(os.environ['HF_HUB_CACHE']).resolve()
    if not cache.is_relative_to('/workspace/WMEval-lighting-orbit/cache'):
        parser.error('ref bindings require this task\'s dedicated /workspace cache')
    from huggingface_hub import snapshot_download
    lock=json.loads(args.lock.read_text())
    verified={}
    for name,asset in lock['assets'].items():
        if len(asset['revision'])!=40:
            raise ValueError('requires immutable model revisions')
        snapshot=Path(snapshot_download(asset['model_id'],revision=asset['revision'],
                        cache_dir=cache,allow_patterns=['*.json','*.txt','*.model',asset['weight']]))
        weight=snapshot/asset['weight']
        with weight.open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=asset['sha256']:
            raise ValueError(f'evaluator weight checksum mismatch: {name}')
        refs=snapshot.parent.parent/'refs'
        refs.mkdir(exist_ok=True)
        (refs/'main').write_text(asset['revision'])
        verified[name]={'model_id':asset['model_id'],'revision':asset['revision'],
                        'weight_sha256':digest,'snapshot':str(snapshot)}
        print(json.dumps({'asset':name,'status':'checksum_verified'}),flush=True)
    destination=cache.parent/'lighting_evaluator_provenance.json'
    destination.write_text(json.dumps({'schema_version':1,'offline_required':True,'assets':verified},indent=2)+'\n')


if __name__=='__main__':
    main()
