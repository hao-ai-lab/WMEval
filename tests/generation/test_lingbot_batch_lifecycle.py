"""Check model reuse, distributed lifetime and preservation of run receipts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('fail_seed', [None, 2])
def test_batch_keeps_one_model_and_preserves_completed_receipts(tmp_path, monkeypatch, fail_seed):
    counts = {'model':0, 'init':0, 'sp':0, 'destroy':0}
    state = {'initialized':False}
    seen = []

    def init(*a, **kw):
        counts['init'] += 1
        state['initialized'] = True

    def destroy():
        counts['destroy'] += 1
        state['initialized'] = False

    def constructor(**kw):
        counts['model'] += 1
        return object()

    dist = SimpleNamespace(init_process_group=init, destroy_process_group=destroy,
                           is_initialized=lambda:state['initialized'])
    native = SimpleNamespace(os=os, dist=dist, wan=SimpleNamespace(WanI2V=constructor),
                             _parse_args=lambda:argparse.Namespace(base_seed=0, ckpt_dir=str(tmp_path)))
    native.init_distributed_group = lambda:counts.__setitem__('sp', counts['sp']+1)

    def generate(settings):
        native.dist.init_process_group(backend='nccl')
        native.init_distributed_group()
        model = native.wan.WanI2V(config={'frozen':True}, rank=0)
        seen.append((settings.base_seed, model))
        if settings.base_seed == fail_seed:
            raise RuntimeError('intentional worker failure')
        Path(settings.save_file).write_bytes(str(settings.base_seed).encode())
        native.dist.destroy_process_group()

    native.generate = generate
    monkeypatch.setitem(sys.modules, 'lingbot_no_ui', SimpleNamespace(upstream=native))
    source = Path(__file__).resolve().parents[2]/'tools/generation/model_adapters/lingbot_batch_no_ui.py'
    loader = importlib.util.spec_from_file_location('batch_lifecycle_test', source)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    monkeypatch.setattr(module, 'enable_sequential_checkpoint_reads', lambda root:None)
    runs = []
    for seed in [1,2,3]:
        receipt = tmp_path/f'{seed}.json'
        receipt.write_text(json.dumps({'status':'queued','seed':seed}))
        runs.append({'seed':seed,'receipt':str(receipt),'video':str(tmp_path/f'{seed}.mp4')})
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps({'runs':runs}))
    monkeypatch.setattr(sys, 'argv', [str(source), '--batch-spec',str(spec)])
    monkeypatch.setenv('RANK','0')
    if fail_seed:
        with pytest.raises(RuntimeError, match='intentional worker failure'):
            module.main()
    else:
        module.main()
    assert counts == {'model':1,'init':1,'sp':1,'destroy':1}
    assert all(model is seen[0][1] for seed,model in seen)
    first = json.loads((tmp_path/'1.json').read_text())
    assert first['status'] == 'completed'
    assert first['video_sha256'] == hashlib.sha256(b'1').hexdigest()
    assert json.loads((tmp_path/'2.json').read_text())['status'] == ('failed' if fail_seed else 'completed')
    assert json.loads((tmp_path/'3.json').read_text())['status'] == ('queued' if fail_seed else 'completed')
