from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(img: np.ndarray) -> np.ndarray:
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        return img[y0 : y0 + h, x0 : x0 + w]

    return crop(a), crop(b)


def to_rgb_tchw(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    # RAFT's OpticalFlow preset maps [0, 1] to [-1, 1] and leaves float input
    # unscaled, so 8-bit frames must be brought to [0, 1] here; fed 0..255 RAFT
    # returns hundreds of px of flow on a static scene.
    frame_rgb = frame_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def flow_raft(
    model: torch.nn.Module,
    transform,
    prev_bgr: np.ndarray,
    next_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    t0 = to_rgb_tchw(prev_bgr, device)
    t1 = to_rgb_tchw(next_bgr, device)
    t0, t1 = transform(t0, t1)

    h, w = t0.shape[-2:]
    target_h = max(128, int(np.ceil(h / 8.0) * 8))
    target_w = max(128, int(np.ceil(w / 8.0) * 8))
    if target_h != h or target_w != w:
        t0 = F.interpolate(t0, size=(target_h, target_w), mode="bilinear", align_corners=False)
        t1 = F.interpolate(t1, size=(target_h, target_w), mode="bilinear", align_corners=False)

    with torch.no_grad():
        flow_list = model(t0, t1)
    flow_t = flow_list[-1][0:1]

    if flow_t.shape[-2:] != (h, w):
        flow_t = F.interpolate(flow_t, size=(h, w), mode="bilinear", align_corners=False)
        flow_t[:, 0] *= float(w) / float(target_w)
        flow_t[:, 1] *= float(h) / float(target_h)

    flow = flow_t[0].detach().cpu().permute(1, 2, 0).numpy()
    return flow.astype(np.float32)


# Explicit rather than ``Raft_Large_Weights.DEFAULT``, an alias torchvision may
# repoint; recorded with the checkpoint hash in
# experiments/protocol/evaluator_assets.lock.json.
RAFT_WEIGHTS_NAME = "C_T_SKHT_V2"


def load_raft_model(device: torch.device) -> Tuple[torch.nn.Module, Any]:
    weights = Raft_Large_Weights[RAFT_WEIGHTS_NAME]
    transform = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device).eval()
    return model, transform


def estimate_global_motion(flow: np.ndarray, mode: str) -> np.ndarray:
    flat = flow.reshape(-1, 2)
    if mode == "mean_translation":
        return flat.mean(axis=0).astype(np.float32)
    if mode == "median_translation":
        return np.median(flat, axis=0).astype(np.float32)
    raise ValueError(
        f"Unsupported control global motion mode: {mode!r}. "
        "Use 'affine_ransac', 'median_translation', or 'mean_translation'."
    )


def fit_affine_motion(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = flow.shape[:2]
    stride_y = max(1, height // 64)
    stride_x = max(1, width // 64)

    ys = np.arange(0, height, stride_y, dtype=np.float32)
    xs = np.arange(0, width, stride_x, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    src = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    sampled_flow = flow[::stride_y, ::stride_x].reshape(-1, 2).astype(np.float32)
    dst = src + sampled_flow

    matrix, _ = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None:
        translation = np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)
        motion_field = np.broadcast_to(
            translation.reshape(1, 1, 2), flow.shape
        ).astype(np.float32)
        return motion_field, None

    full_x, full_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    homogeneous = np.stack(
        [full_x.reshape(-1), full_y.reshape(-1), np.ones(height * width, dtype=np.float32)],
        axis=1,
    )
    dst_full = homogeneous @ matrix.T
    motion_field = dst_full.reshape(height, width, 2) - np.stack([full_x, full_y], axis=2)
    return motion_field.astype(np.float32), matrix.astype(np.float32)


def estimate_camera_motion(
    flow: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if mode == "affine_ransac":
        motion_field, matrix = fit_affine_motion(flow)
        if matrix is None:
            translation = np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)
            return motion_field, {
                "type": "translation_fallback",
                "vector": [float(translation[0]), float(translation[1])],
            }
        return motion_field, {
            "type": "affine_ransac",
            "matrix_2x3": [[float(v) for v in row] for row in matrix.tolist()],
        }

    translation = estimate_global_motion(flow, mode)
    motion_field = np.broadcast_to(translation.reshape(1, 1, 2), flow.shape).astype(np.float32)
    return motion_field, {
        "type": mode,
        "vector": [float(translation[0]), float(translation[1])],
    }


def compensate_flow(flow: np.ndarray, *, mode: str) -> tuple[np.ndarray, Dict[str, Any]]:
    motion_field, camera_model = estimate_camera_motion(flow, mode)
    return flow - motion_field, camera_model


def compute_flow_transition_saliency(
    flows: Sequence[np.ndarray],
    *,
    camera_compensation: bool,
    global_motion_mode: str,
) -> List[float]:
    if not flows:
        return []

    processed: List[np.ndarray] = []
    for flow in flows:
        current = flow
        if camera_compensation:
            current, _ = compensate_flow(flow, mode=global_motion_mode)
        processed.append(current)

    scores: List[float] = []
    prev: np.ndarray | None = None
    for current in processed:
        mag = float(np.linalg.norm(current, axis=2).mean())
        delta = 0.0 if prev is None else float(np.linalg.norm(current - prev, axis=2).mean())
        scores.append(mag + 0.5 * delta)
        prev = current
    return scores


def build_flow_proposal_windows_from_scores(
    sample_indices: Sequence[int],
    scores: Sequence[float],
    *,
    frame_count: int,
    fps: float,
    window_sec: float,
    max_frames_per_window: int,
    topk_windows: int,
    peak_threshold: float,
) -> tuple[List[List[int]], List[Dict[str, Any]]]:
    if frame_count <= 0 or fps <= 0 or len(sample_indices) < 2 or not scores:
        return [], []

    radius_frames = max(1, int(round(window_sec * fps)))
    candidate_peaks: List[Tuple[float, int, int]] = []
    max_score = max(float(s) for s in scores) if scores else 0.0
    min_score = max(0.0, float(peak_threshold)) * max_score
    for idx, score in enumerate(scores):
        score_f = float(score)
        if score_f < min_score:
            continue
        left = float(scores[idx - 1]) if idx > 0 else -1.0
        right = float(scores[idx + 1]) if idx + 1 < len(scores) else -1.0
        if score_f < left or score_f < right:
            continue
        lo_idx = int(sample_indices[idx])
        hi_idx = int(sample_indices[idx + 1])
        anchor = int(round((lo_idx + hi_idx) / 2.0))
        candidate_peaks.append((score_f, anchor, idx))

    candidate_peaks.sort(key=lambda item: (-item[0], item[1]))
    windows: List[List[int]] = []
    logs: List[Dict[str, Any]] = []
    anchors_kept: List[int] = []
    min_anchor_gap = max(1, radius_frames // 2)
    for score_f, anchor, score_index in candidate_peaks:
        if len(windows) >= max(1, int(topk_windows)):
            break
        if any(abs(anchor - prev_anchor) < min_anchor_gap for prev_anchor in anchors_kept):
            continue
        lo = max(0, anchor - radius_frames)
        hi = min(frame_count - 1, anchor + radius_frames)
        span = hi - lo + 1
        sample_count = min(max(2, int(max_frames_per_window)), span)
        raw = np.linspace(lo, hi, num=sample_count)
        group = sorted(set(min(max(int(round(float(x))), 0), frame_count - 1) for x in raw))
        if not group:
            continue
        anchors_kept.append(anchor)
        windows.append(group)
        logs.append(
            {
                "anchor_frame": anchor,
                "score": score_f,
                "score_index": score_index,
                "indices": group,
            }
        )
    return windows, logs
