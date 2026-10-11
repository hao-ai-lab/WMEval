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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    (output / 'adapter_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['lingbot','hy-worldplay'], required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.case, args.output, args.backend), indent=2))


if __name__ == '__main__':
    main()
