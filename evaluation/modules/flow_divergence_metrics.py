"""Flow comparison metrics aligned with ptlflow eval_flow_divergence (no matplotlib).

See: https://github.com/shaoxiongduan/ptlflow/blob/main/eval_flow_divergence.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def estimate_foe(
    flow: np.ndarray, step: int = 8, min_mag: float = 0.5
) -> tuple[float, float, float]:
    H, W = flow.shape[:2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.ravel()
    xx = xx.ravel()
    uu = flow[yy, xx, 0]
    vv = flow[yy, xx, 1]

    mag = np.sqrt(uu**2 + vv**2)
    valid = mag > min_mag
    if valid.sum() < 10:
        return W / 2.0, H / 2.0, float("inf")

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    uu = uu[valid].astype(np.float64)
    vv = vv[valid].astype(np.float64)

    A = np.column_stack([vv, -uu])
    b = vv * xx - uu * yy

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    fit_err = np.abs(A @ result - b)
    mean_residual = float(fit_err.mean())

    return float(result[0]), float(result[1]), mean_residual


def compute_flow_kl_2d(
    flow_a: np.ndarray,
    flow_b: np.ndarray,
    n_angle_bins: int = 36,
    n_mag_bins: int = 20,
    min_mag: float = 0.5,
) -> float:
    def _build_hist(flow: np.ndarray):
        u, v = flow[:, :, 0].ravel(), flow[:, :, 1].ravel()
        mag = np.sqrt(u**2 + v**2)
        angle = np.degrees(np.arctan2(v, u)) % 360

        valid = mag >= min_mag
        if valid.sum() < 10:
            return None
        mag = mag[valid]
        angle = angle[valid]

        mag_max = max(mag.max(), min_mag + 1.0)
        mag_edges = np.logspace(np.log10(min_mag), np.log10(mag_max), n_mag_bins + 1)
        angle_edges = np.linspace(0, 360, n_angle_bins + 1)

        hist, _, _ = np.histogram2d(angle, mag, bins=[angle_edges, mag_edges])
        return hist

    hist_a = _build_hist(flow_a)
    hist_b = _build_hist(flow_b)

    if hist_a is None or hist_b is None:
        return 0.0

    eps = 1.0
    p = (hist_a + eps) / (hist_a + eps).sum()
    q = (hist_b + eps) / (hist_b + eps).sum()
    kl = float((p * np.log(p / q)).sum())
    return kl


def compute_frame_metrics(
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    grid_size: int = 8,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    gt_mag_map = np.linalg.norm(flow_gt, axis=2)
    gen_mag_map = np.linalg.norm(flow_gen, axis=2)

    max_mag_map = np.maximum(gt_mag_map, gen_mag_map)
    mag_hi = np.percentile(max_mag_map, max_mag_pct)
    mag_mask = (max_mag_map >= min_mag) & (max_mag_map <= mag_hi)
    n_valid = mag_mask.sum()
    metrics["mag_filter_kept"] = float(n_valid / max(mag_mask.size, 1))

    if n_valid > 0:
        gt_filtered = flow_gt[mag_mask]
        gen_filtered = flow_gen[mag_mask]
        mean_gt = gt_filtered.mean(axis=0)
        mean_gen = gen_filtered.mean(axis=0)
    else:
        mean_gt = flow_gt.reshape(-1, 2).mean(axis=0)
        mean_gen = flow_gen.reshape(-1, 2).mean(axis=0)

    metrics["mf_epe"] = float(np.linalg.norm(mean_gt - mean_gen))

    mag_gt = np.linalg.norm(mean_gt)
    mag_gen = np.linalg.norm(mean_gen)
    if mag_gt > 1e-6 and mag_gen > 1e-6:
        cos_sim = np.dot(mean_gt, mean_gen) / (mag_gt * mag_gen)
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        metrics["mf_angle_err"] = float(np.degrees(np.arccos(cos_sim)))
        metrics["mf_cosine"] = cos_sim
    else:
        metrics["mf_angle_err"] = 0.0
        metrics["mf_cosine"] = 1.0

    metrics["mf_mag_ratio"] = float(mag_gen / mag_gt) if mag_gt > 1e-6 else 1.0

    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=2)
    if n_valid > 0:
        metrics["pixel_epe_mean"] = float(epe_map[mag_mask].mean())
        metrics["pixel_epe_max"] = float(epe_map[mag_mask].max())
    else:
        metrics["pixel_epe_mean"] = float(epe_map.mean())
        metrics["pixel_epe_max"] = float(epe_map.max())

    valid = mag_mask & (gt_mag_map > 0.5) & (gen_mag_map > 0.5)
    if valid.sum() > 0:
        dot = (
            flow_gt[:, :, 0] * flow_gen[:, :, 0] + flow_gt[:, :, 1] * flow_gen[:, :, 1]
        )
        cos_map = np.clip(dot / (gt_mag_map * gen_mag_map + 1e-8), -1.0, 1.0)
        angle_map = np.degrees(np.arccos(cos_map))
        metrics["px_angle_rmse"] = float(np.sqrt((angle_map[valid] ** 2).mean()))
    else:
        metrics["px_angle_rmse"] = 0.0

    H, W = epe_map.shape
    gh = H // grid_size
    gw = W // grid_size
    grid_epe_values: List[float] = []
    for gi in range(grid_size):
        for gj in range(grid_size):
            cell_mask = mag_mask[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw]
            cell_epe = epe_map[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw]
            if cell_mask.sum() > 0:
                grid_epe_values.append(float(cell_epe[cell_mask].mean()))
            else:
                grid_epe_values.append(float(cell_epe.mean()))
    metrics["grid_epe_mean"] = float(np.mean(grid_epe_values))
    metrics["grid_epe_max"] = float(np.max(grid_epe_values))

    if n_valid > 0:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map) & mag_mask
        metrics["fl_all"] = float(outlier.sum() / n_valid)
    else:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map)
        metrics["fl_all"] = float(outlier.mean())

    foe_gt_x, foe_gt_y, foe_gt_res = estimate_foe(flow_gt)
    foe_gen_x, foe_gen_y, foe_gen_res = estimate_foe(flow_gen)
    metrics["foe_gt_x"] = foe_gt_x
    metrics["foe_gt_y"] = foe_gt_y
    metrics["foe_gen_x"] = foe_gen_x
    metrics["foe_gen_y"] = foe_gen_y
    metrics["foe_dist"] = float(
        np.sqrt((foe_gt_x - foe_gen_x) ** 2 + (foe_gt_y - foe_gen_y) ** 2)
    )
    metrics["foe_gt_residual"] = foe_gt_res
    metrics["foe_gen_residual"] = foe_gen_res

    metrics["flow_kl_2d"] = compute_flow_kl_2d(flow_gt, flow_gen)

    return metrics


def compute_temporal_metrics(frame_metrics_list: List[Dict[str, float]]) -> Dict:
    n_frames = len(frame_metrics_list)
    if n_frames == 0:
        return {}

    metric_names = list(frame_metrics_list[0].keys())
    series = {name: np.array([m[name] for m in frame_metrics_list]) for name in metric_names}

    summary: Dict = {"n_frames": n_frames}

    for name, vals in series.items():
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())
        summary[f"{name}_max"] = float(vals.max())
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        summary[f"{name}_auc"] = float(trap(vals) / max(n_frames - 1, 1))

    epe_series = series["pixel_epe_mean"]
    window = min(5, n_frames)
    if n_frames >= window:
        baseline = float(np.median(epe_series[:window]))
        threshold = max(baseline * 2.0, 1.0)
        kernel = np.ones(window) / window
        smoothed = np.convolve(epe_series, kernel, mode="valid")
        divergence_frame = None
        for i, val in enumerate(smoothed):
            if val > threshold:
                divergence_frame = i
                break
        summary["divergence_onset_frame"] = divergence_frame
        summary["divergence_threshold"] = threshold
    else:
        summary["divergence_onset_frame"] = None
        summary["divergence_threshold"] = None

    return summary


def json_safe_floats(obj: Dict) -> Dict:
    """Replace non-finite floats so json.dumps works."""
    out: Dict = {}
    for k, v in obj.items():
        if isinstance(v, float):
            if not np.isfinite(v):
                out[k] = None
            else:
                out[k] = v
        else:
            out[k] = v
    return out
