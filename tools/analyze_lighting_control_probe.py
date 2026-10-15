#!/usr/bin/env python3
"""Measure control response using seed mesh points, optical flow and robust PnP.

Calibration runs are independent of formal seeds. Failed geometric fits stay
unresolved; they do not imply that an action interface is unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def estimate_pose(world, pixels, intrinsics):
    """Return camera-to-world only for a sufficiently constrained robust fit."""
    if len(world) < 12:
        return None
    cv2.setRNGSeed(0)
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        np.asarray(world, np.float64), np.asarray(pixels, np.float64),
        np.asarray(intrinsics, np.float64), None, iterationsCount=200,
        reprojectionError=3.0, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12 or len(inliers) / len(world) < .35:
        return None
    i = inliers[:, 0]
    rotation, translation = cv2.solvePnPRefineLM(world[i], pixels[i], intrinsics,
                                              None, rotation, translation)
    projected, _ = cv2.projectPoints(world[i], rotation, translation, intrinsics, None)
    rms = float(np.sqrt(np.mean(np.sum((projected[:, 0] - pixels[i]) ** 2, axis=1))))
    matrix, _ = cv2.Rodrigues(rotation)
    c2w = np.eye(4)
    c2w[:3, :3] = matrix.T
    c2w[:3, 3] = -matrix.T @ translation[:, 0]
    if rms > 3 or not np.isfinite(c2w).all():
        return None
    return {'c2w': c2w.tolist(), 'inliers': len(i), 'tracked_points': len(world),
            'reprojection_rms_px': rms, 'inlier_indices': i.tolist()}


def analyze(video, geometry, seed_image, canvas_size):
    data = json.loads(geometry.read_text())
    world = np.array([[p['worldM'][c] for c in ('x', 'y', 'z')] for p in data['points']], np.float64)
    pixels = np.array([[p['u'], p['v']] for p in data['points']], np.float32)
    width, height = canvas_size
    scale = min(width / data['width'], height / data['height'])
    offset = np.array([(width - scale * data['width']) / 2,
                       (height - scale * data['height']) / 2])
    pixels = (pixels * scale + offset).astype(np.float32)
    focal = data['height'] / (2 * np.tan(np.pi / 6)) * scale
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
    previous = cv2.imread(str(seed_image), cv2.IMREAD_GRAYSCALE)
    if previous is None or previous.shape != (height, width):
        raise ValueError('seed image must be the recorded native letterboxed canvas')
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError('cannot decode calibration video')
    indices = np.arange(len(world))
    rows = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != previous.shape:
            raise ValueError('generated native geometry differs from seed canvas')
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row = {'frame': len(rows), 'time_sec': len(rows) / fps, 'pose': None}
        if len(pixels) >= 12:
            forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, pixels[:, None], None,
                winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
            backward, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
                winSize=(31, 31), maxLevel=4)
            good = (status[:, 0] != 0) & (reverse[:, 0] != 0)
            good &= np.linalg.norm(backward[:, 0] - pixels, axis=1) <= 1.5
            good &= np.isfinite(forward[:, 0]).all(axis=1)
            pixels, indices = forward[good, 0], indices[good]
            row['pose'] = estimate_pose(world[indices], pixels.astype(np.float64), k)
            if row['pose'] is not None:
                # Once rejected, do not propagate a confidently tracked but
                # geometrically incorrect surface correspondence into later fits.
                inliers = np.array(row['pose'].pop('inlier_indices'), dtype=int)
                pixels, indices = pixels[inliers], indices[inliers]
        rows.append(row)
        previous = current
    capture.release()
    return {'schema_version': 1, 'video_sha256': sha256(video),
            'geometry_sha256': sha256(geometry), 'seed_canvas_sha256': sha256(seed_image),
            'native_fps': fps, 'intrinsics': k.tolist(), 'frames': rows,
            'valid_pose_frames': sum(r['pose'] is not None for r in rows),
            'status': 'measured' if sum(r['pose'] is not None for r in rows) >= 12 else 'unresolved',
            'limitations': 'Rigid seed mesh correspondences; reject drift/occlusion with forward-backward flow and PnP RANSAC. No output correction.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--seed-image', type=Path, required=True)
    p.add_argument('--canvas-size', type=int, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.video, args.geometry, args.seed_image, args.canvas_size)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}))


if __name__ == '__main__':
    main()
