#!/usr/bin/env python3
"""Prepare independent key/mouse probes or a frozen full-orbit schedule."""
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


def prepare(case, output, remote_root, gains=None, backend='matrix-game2'):
    output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((case / 'case.yaml').read_text())
    if raw['case_id'] != 'SunRisesAndSets/lighting_orbit_v2':
        raise ValueError('this calibration recipe is specific to the frozen lighting orbit')
    seed = case / raw['reference']['first_frame']
    image = Image.open(seed).convert('RGB')
    if image.size != (1024, 1024):
        raise ValueError('requires the frozen square lighting seed')
    waypoint = backend == 'waypoint'
    model_name = 'Waypoint' if waypoint else 'Matrix-Game2'
    tag = 'waypoint' if waypoint else 'mg2'
    width, height = (1280, 720) if waypoint else (640, 352)
    offset = (width - height) // 2
    native_fps, control_fps, prefix = (60, 15, 4) if waypoint else (12, 12, 0)
    mouse_column = 0 if waypoint else 1
    canvas = Image.new('RGB', (width, height))
    canvas.paste(image.resize((height, height), Image.Resampling.BILINEAR), (offset, 0))
    canvas.save(output / 'seed.png')
    receipt = {'original_seed_sha256': sha256(seed), 'native_seed_sha256': sha256(output / 'seed.png'),
        'scale': height / 1024, 'offset_px': [offset, 0], 'native_canvas': [width, height],
        'native_fps': native_fps, 'control_fps': control_fps, 'canonical_zero_frame': prefix,
        'resize': 'PIL bilinear; preserve square view inside native 16:9 canvas', 'vertical_fov_deg': 60,
        'case_sha256': sha256(case / 'case.yaml'),
        'trajectory_sha256': sha256(case / raw['reference']['artifacts']['trajectory'])}
    (output / 'seed_preprocessing.json').write_text(json.dumps(receipt, indent=2) + '\n')
    runs = []
    root = remote_root.rstrip('/')
    if gains is None:
        for s in (1001, 1002):
            for kind in ('strafe', 'yaw'):
                count = 30 if waypoint else 33
                keys = np.zeros((count, 4), np.float32); mouse = np.zeros((count, 2), np.float32)
                moving = slice(None) if waypoint else slice(1, 25)
                if kind == 'strafe':
                    keys[moving, 3] = 1
                else:
                    mouse[moving, mouse_column] = .1
                name = f'{kind}_{s}'
                np.savez_compressed(output / f'{name}.npz', keyboard=keys, mouse=mouse)
                runs.append({'seed': s, 'latent_frames': 9 if not waypoint else None,
                    'image': root + f'/inputs/{tag}-calibration/seed.png',
                    'schedule': root + f'/inputs/{tag}-calibration/{name}.npz',
                    'output': root + f'/outputs/calibration/{model_name}/{name}'})
        kind = 'independent_calibration'
    else:
        gain_data = json.loads(gains.read_text())
        if gain_data['status'] != 'calibrated' or gain_data['calibration_seeds'] != [1001, 1002] or gain_data['model'] != model_name:
            raise ValueError('requires accepted independent calibration, not formal output tuning')
        speed = float(gain_data['translation_m_per_key_frame'])
        yaw_gain = float(gain_data['yaw_deg_per_mouse_unit'])
        duty = (2 * np.pi * 2 / 10 / native_fps) / speed
        if not np.isfinite([duty, yaw_gain]).all() or not 0 < duty <= 1 or yaw_gain == 0:
            raise ValueError('calibration cannot express this nominal speed; retain pending status')
        moving_count = 10 * control_fps
        count = moving_count if waypoint else 129
        keys = np.zeros((count, 4), np.float32); mouse = np.zeros((count, 2), np.float32)
        index = np.arange(moving_count)
        keys[:moving_count, 3] = np.floor((index + 1) * duty) - np.floor(index * duty)
        mouse[:moving_count, mouse_column] = -36 / native_fps / yaw_gain
        np.savez_compressed(output / 'orbit.npz', keyboard=keys, mouse=mouse)
        receipt.update(calibration_sha256=sha256(gains), translation_duty_cycle=duty,
            yaw_mouse_units_per_control=float(mouse[0, mouse_column]), target_radius_m=2,
            target_orbit_deg=360, target_look_at_m=[0, 1.2735698223114014, 0],
            terminal_hold_after_sec=None if waypoint else 10,
            native_translation_response='Empirical metres per native decoded key frame; no output correction.')
        for s in (42, 1, 2, 3, 4):
            runs.append({'seed': s, 'latent_frames': None if waypoint else 33,
                'image': root + f'/inputs/{tag}-orbit/seed.png',
                'schedule': root + f'/inputs/{tag}-orbit/orbit.npz',
                'output': root + f'/outputs/lighting_orbit_v2/{model_name}/seed_{s}'})
        kind = 'formal'
    spec = {'schema_version': 1, 'run_kind': kind, 'runs': runs, 'adapter': receipt,
            'control_window_sec': ([0, 2] if waypoint else [1 / 12, 25 / 12]) if gains is None else [0, 10]}
    (output / 'batch_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', default='/workspace/WMEval-lighting-orbit')
    p.add_argument('--gains', type=Path)
    p.add_argument('--backend', choices=['matrix-game2', 'waypoint'], default='matrix-game2')
    args = p.parse_args()
    result = prepare(args.case, args.output, args.remote_root, args.gains, args.backend)
    print(json.dumps({'run_kind': result['run_kind'], 'runs': len(result['runs'])}))


if __name__ == '__main__':
    main()
