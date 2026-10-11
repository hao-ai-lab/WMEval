#!/usr/bin/env python3
"""Prepare independent MG2 calibration probes or a frozen full-orbit schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(case, output, remote_root, gains=None):
    output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((case / 'case.yaml').read_text())
    if raw['case_id'] != 'SunRisesAndSets/lighting_orbit_v2':
        raise ValueError('this calibration recipe is specific to the frozen lighting orbit')
    seed = case / raw['reference']['first_frame']
    image = Image.open(seed).convert('RGB')
    if image.size != (1024, 1024):
        raise ValueError('requires the frozen square lighting seed')
    canvas = Image.new('RGB', (640, 352))
    canvas.paste(image.resize((352, 352), Image.Resampling.BILINEAR), (144, 0))
    canvas.save(output / 'seed.png')
    receipt = {'original_seed_sha256': sha256(seed), 'native_seed_sha256': sha256(output / 'seed.png'),
        'scale': 352 / 1024, 'offset_px': [144, 0], 'native_canvas': [640, 352],
        'resize': 'PIL bilinear; preserve square view inside native 16:9 canvas', 'vertical_fov_deg': 60,
        'case_sha256': sha256(case / 'case.yaml'),
        'trajectory_sha256': sha256(case / raw['reference']['artifacts']['trajectory'])}
    (output / 'seed_preprocessing.json').write_text(json.dumps(receipt, indent=2) + '\n')
    runs = []
    root = remote_root.rstrip('/')
    if gains is None:
        for s in (1001, 1002):
            for kind in ('strafe', 'yaw'):
                keys = np.zeros((33, 4), np.float32); mouse = np.zeros((33, 2), np.float32)
                if kind == 'strafe':
                    keys[1:25, 3] = 1
                else:
                    mouse[1:25, 1] = .1
                name = f'{kind}_{s}'
                np.savez_compressed(output / f'{name}.npz', keyboard=keys, mouse=mouse)
                runs.append({'seed': s, 'latent_frames': 9, 'image': root + '/inputs/mg2-calibration/seed.png',
                    'schedule': root + f'/inputs/mg2-calibration/{name}.npz',
                    'output': root + f'/outputs/calibration/Matrix-Game2/{name}'})
        kind = 'independent_calibration'
    else:
        gain_data = json.loads(gains.read_text())
        if gain_data['status'] != 'calibrated' or gain_data['calibration_seeds'] != [1001, 1002]:
            raise ValueError('requires accepted independent calibration, not formal output tuning')
        speed = float(gain_data['translation_m_per_key_frame'])
        yaw_gain = float(gain_data['yaw_deg_per_mouse_unit'])
        duty = (2 * np.pi * 2 / 10 / 12) / speed
        if not np.isfinite([duty, yaw_gain]).all() or not 0 < duty <= 1 or yaw_gain == 0:
            raise ValueError('calibration cannot express this nominal speed; retain pending status')
        keys = np.zeros((129, 4), np.float32); mouse = np.zeros((129, 2), np.float32)
        index = np.arange(120)
        keys[:120, 3] = np.floor((index + 1) * duty) - np.floor(index * duty)
        mouse[:120, 1] = -36 / 12 / yaw_gain
        np.savez_compressed(output / 'orbit.npz', keyboard=keys, mouse=mouse)
        receipt.update(calibration_sha256=sha256(gains), translation_duty_cycle=duty,
            yaw_mouse_units_per_frame=float(mouse[0, 1]), target_radius_m=2,
            target_orbit_deg=360, target_look_at_m=[0, 1.2735698223114014, 0],
            terminal_hold_after_sec=10, native_translation_response='Empirical calibration; no output correction.')
        for s in (42, 1, 2, 3, 4):
            runs.append({'seed': s, 'latent_frames': 33, 'image': root + '/inputs/mg2-orbit/seed.png',
                'schedule': root + '/inputs/mg2-orbit/orbit.npz',
                'output': root + f'/outputs/lighting_orbit_v2/Matrix-Game2/seed_{s}'})
        kind = 'formal'
    spec = {'schema_version': 1, 'run_kind': kind, 'runs': runs, 'adapter': receipt,
            'control_window_sec': [1 / 12, 25 / 12] if gains is None else [0, 10]}
    (output / 'batch_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', default='/workspace/WMEval-lighting-orbit')
    p.add_argument('--gains', type=Path)
    args = p.parse_args()
    result = prepare(args.case, args.output, args.remote_root, args.gains)
    print(json.dumps({'run_kind': result['run_kind'], 'runs': len(result['runs'])}))


if __name__ == '__main__':
    main()
