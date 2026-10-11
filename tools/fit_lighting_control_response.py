#!/usr/bin/env python3
"""Freeze independent MG2 strafe/yaw gains, preserving unresolved fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fit(rows, cumulative, kind):
    valid = [r for r in rows if r['pose'] is not None and r['frame'] <= 24]
    if len(valid) < 12:
        return {'status': 'unresolved', 'reason': 'fewer than 12 valid geometric frames'}
    poses = np.array([r['pose']['c2w'] for r in valid])
    x = cumulative[[r['frame'] for r in valid]]
    if kind == 'strafe':
        y = (poses[:, :3, 3] - poses[0, :3, 3]) @ poses[0, :3, 0]
    else:
        forward = poses[:, :3, 2]
        y = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], -forward[:, 2])))
        y -= y[0]
    if np.ptp(x) <= 0:
        return {'status': 'unresolved', 'reason': 'no observed control span'}
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    r2 = float(1 - np.sum(residual ** 2) / max(1e-12, np.sum((y - y.mean()) ** 2)))
    ok = np.isfinite(slope) and (slope > 0 if kind == 'strafe' else abs(slope) > 1e-6)
    ok = ok and r2 >= .9 and rms <= (.04 if kind == 'strafe' else 2)
    return {'status': 'measured' if ok else 'unresolved', 'gain': float(slope),
            'fit_r2': r2, 'residual_rms_m' if kind == 'strafe' else 'residual_rms_deg': rms,
            'valid_frames': len(valid), 'last_valid_frame': valid[-1]['frame'],
            'observed_response_span': float(np.ptp(y)), 'intercept': float(intercept)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-root', type=Path, required=True)
    p.add_argument('--schedule-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    measurements, hashes = {}, {}
    for seed in (1001, 1002):
        for kind in ('strafe', 'yaw'):
            name = f'{kind}_{seed}'
            path = args.probe_root / name / 'pose_measurement.json'
            data = json.loads(path.read_text())
            controls = np.load(args.schedule_root / f'{name}.npz', allow_pickle=False)
            values = controls['keyboard'][:, 3] if kind == 'strafe' else controls['mouse'][:, 1]
            measurements[name] = fit(data['frames'], np.cumsum(values), kind)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = all(m['status'] == 'measured' for m in measurements.values())
    gains = {}
    if ok:
        for kind in ('strafe', 'yaw'):
            values = [measurements[f'{kind}_{seed}']['gain'] for seed in (1001, 1002)]
            relative_range = abs(values[0] - values[1]) / abs(np.mean(values))
            if np.sign(values[0]) != np.sign(values[1]):
                ok = False
            if relative_range > .25:
                ok = False
            gains[kind] = float(np.mean(values))
            gains[f'{kind}_relative_range'] = float(relative_range)
    result = {'schema_version': 1, 'status': 'calibrated' if ok else 'unresolved',
        'model': 'Matrix-Game2', 'calibration_seeds': [1001, 1002],
        'translation_m_per_key_frame': gains.get('strafe'),
        'yaw_deg_per_mouse_unit': gains.get('yaw'), 'measurements': measurements,
        'measurement_sha256': hashes, 'consistency': gains,
        'method': 'Independent mesh-anchored camera response regression; R2>=0.9, translation RMS<=0.04m, yaw RMS<=2deg, seed relative range<=25%.',
        'unresolved_semantics': 'Pending calibration, not interface unsupported and not a zero score.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
