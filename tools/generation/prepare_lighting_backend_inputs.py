#!/usr/bin/env python3
"""Compile the case's analytic orbit at a backend's actual control timestamps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.camera_orbit import camera_intrinsics, orbit_poses
from evaluation.trajectories import validate_trajectory


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(case_directory: Path, output: Path, backend: str) -> dict:
    case = yaml.safe_load((case_directory / 'case.yaml').read_text())
    trajectory_path = case_directory / case['reference']['artifacts']['trajectory']
    raw = yaml.safe_load(trajectory_path.read_text())
    validate_trajectory(raw)
    if raw['kind'] != 'camera_look_at_orbit' or raw['duration_sec'] != 10:
        raise ValueError('requires the complete 10-second lighting orbit protocol')
    output.mkdir(parents=True, exist_ok=True)
    if backend == 'lingbot':
        fps, frames = 16, 161
        times = np.arange(frames, dtype=np.float64) / fps
        poses = orbit_poses(raw, times)
        # Upstream always interprets input K as 832x480 before independently
        # scaling its axes to the square seed. Compensate to preserve 60° FOV.
        k = camera_intrinsics(raw, 1024, 1024)
        intrinsics = np.tile([832*k[0,0]/1024, 480*k[1,1]/1024, 416, 240], (frames,1))
        np.save(output / 'poses.npy', poses.astype(np.float32), allow_pickle=False)
        np.save(output / 'intrinsics.npy', intrinsics.astype(np.float32), allow_pickle=False)
        files = ['poses.npy', 'intrinsics.npy']
        convention = 'OpenCV c2w; K encoded in upstream nominal 832x480 canvas, square seed compensation'
    elif backend == 'hy-worldplay':
        fps, frames = 24, 253
        times = np.arange((frames-1)//4+1, dtype=np.float64) * 4 / fps
        # Native 4n+1 frame count exceeds the scoring window. Hold the endpoint
        # after t=10; these frames are explicitly excluded from scoring.
        poses = orbit_poses(raw, np.minimum(times, 10))
        k = camera_intrinsics(raw, 1024, 1024)
        (output / 'poses.json').write_text(json.dumps({str(i): {'extrinsic': p.tolist(), 'K': k.tolist()}
                                                     for i,p in enumerate(poses)}, indent=2)+'\n')
        files = ['poses.json']
        convention = 'OpenCV c2w; K from square seed, normalized by upstream pose_to_input'
    elif backend == 'matrix-game3':
        fps, frames = 17, 177  # Four native clips: 57 + 3 * 40.
        times = np.arange(frames, dtype=np.float64) / fps
        poses = orbit_poses(raw, np.minimum(times, 10))
        # The native world uses XY as its floor and Z up, with det(R)=-1.
        # A fixed world-basis change preserves every camera-relative transform.
        basis = np.array([[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]])
        native = poses.copy()
        native[:, :3, :3] = basis @ poses[:, :3, :3]
        native[:, :3, 3] = (basis @ (poses[:, :3, 3] - poses[0, :3, 3]).T).T
        keyboard = np.zeros((frames, 6), dtype=np.float32)
        mouse = np.zeros((frames, 2), dtype=np.float32)
        moving = times < 10
        rate = sum(float(s['values']['orbit_phase_rate_deg_s']) *
                   (float(s['end_sec']) - float(s['start_sec'])) for s in raw['segments']) / 10
        # Native D moves 0.1235 m per frame; binary auxiliary keys use a
        # deterministic duty cycle. The explicit poses remain authoritative.
        duty = abs(np.deg2rad(rate) * raw['geometry']['radius_m'] / fps) / 0.1235
        if duty > 1:
            raise ValueError('orbit speed exceeds the audited native auxiliary key speed')
        index = np.arange(frames)
        keyboard[moving, 3 if rate >= 0 else 2] = (
            np.floor((index[moving] + 1) * duty) - np.floor(index[moving] * duty))
        mouse[moving, 1] = -rate / fps / 15.0
        np.savez_compressed(output / 'conditioning.npz', poses=native,
                            keyboard=keyboard, mouse=mouse,
                            seed_intrinsics=camera_intrinsics(raw, 1024, 1024),
                            world_basis=basis)
        files = ['conditioning.npz']
        convention = 'Native XY-floor/Z-up c2w, det(R)=-1; fixed reflected world basis from canonical OpenCV c2w'
    else:
        raise ValueError(f'no verified numeric-pose interface: {backend}')
    (output / 'native_control_times.json').write_text(json.dumps(times.tolist())+'\n')
    files.append('native_control_times.json')
    receipt = {'schema_version': 1, 'backend': backend, 'trajectory_id': raw['trajectory_id'],
               'trajectory_sha256': sha256(trajectory_path), 'case_sha256': sha256(case_directory/'case.yaml'),
               'conditioning_sha256': sha256(case_directory/case['reference']['first_frame']),
               'native_fps': fps, 'native_frames': frames, 'canonical_zero_frame': 0,
               'scoring_window_sec': [0,10], 'control_duration_sec': 10,
               'last_control_timestamp_sec': float(times[-1]), 'coordinate_convention': convention,
               'seed_preprocessing': 'preserve square aspect ratio; upstream resolution resize only',
               'terminal_hold_after_sec': 10 if times[-1]>10 else None,
               'files': {name: sha256(output/name) for name in files}}
    if backend == 'lingbot':
        receipt['native_translation_normalization'] = (
            'Pinned upstream samples every fourth pose, computes frame-relative transforms, '
            'and divides translations by their maximum norm. Input geometry remains in '
            'metres; encoded translation uses upstream normalized units. No response correction.')
    if backend == 'matrix-game3':
        receipt['native_translation_normalization'] = (
            'Pinned upstream interpolates latent poses, computes frame-relative transforms, '
            'and divides translations by their maximum norm; no response correction.')
        receipt['auxiliary_controls'] = {'native_translation_m_per_key_frame': 0.1235,
                                        'native_yaw_deg_per_mouse_unit': 15.0,
                                        'translation_duty_cycle': duty,
                                        'explicit_poses_authoritative': True}
    (output / 'adapter_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['lingbot','hy-worldplay','matrix-game3'], required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.case, args.output, args.backend), indent=2))


if __name__ == '__main__':
    main()
