"""Shadow-consistency evaluation (GroundingDINO + SAM 2 + Hu moments).

Six internal-consistency metrics per video, computed **without** a per-frame
Unity ground-truth reference.  The pipeline:

  1. Detect anchor objects in frame 0 with GroundingDINO (typed prompt).
  2. Track them through the video with SAM 2's video predictor.
  3. For every frame, extract cast shadows via a luminance gate inside a
     dilated halo around each tracked object.
  4. Compute six headline metrics from the per-frame track:
       - Scene Retention Rate (SRR)
       - Shadow Detection Rate (SDR)
       - Multi-Object Sun Disagreement (MOSD)
       - Shadow Direction Jitter (SDJ)
       - Shadow Area Stability (SAS)
       - Shadow Shape Drift (SSD)

Hu moments provide translation/rotation/scale-invariant shadow-shape
signatures so smooth perspective changes do not register; only genuine
morphing does.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None  # type: ignore[assignment,misc]

try:
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    AutoModelForZeroShotObjectDetection = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]

from ..context import EvaluationContext

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAM2_DEFAULT_HF = "facebook/sam2.1-hiera-large"
SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GD_DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

# Per-WM HUD masks: (x_frac, y_frac, w_frac, h_frac) rectangles zeroed out
# before detection.  Calibrated from frame-0 visual inspection.
HUD_PRESETS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "none": [],
    "lingbot": [(0.04, 0.78, 0.27, 0.20), (0.69, 0.78, 0.27, 0.20)],
    "mg3": [(0.31, 0.74, 0.45, 0.24), (0.42, 0.43, 0.18, 0.10)],
    "mg2": [(0.0, 0.92, 1.0, 0.08)],
    "hy": [],
    "genie": [(0.42, 0.92, 0.16, 0.07)],
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _apply_hud_mask(
    frame_bgr: np.ndarray,
    rects: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    if not rects:
        return frame_bgr
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    floor_strip = out[: max(1, h // 10), :, :].reshape(-1, 3)
    fill = np.median(floor_strip, axis=0).astype(np.uint8)
    for xf, yf, wf, hf in rects:
        x0, y0 = int(xf * w), int(yf * h)
        x1, y1 = min(w, int((xf + wf) * w)), min(h, int((yf + hf) * h))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = fill
    return out


def _detect_objects_groundingdino(
    frame_bgr: np.ndarray,
    text_prompt: str,
    model_id: str,
    threshold: float,
    text_threshold: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = [str(lb).strip().lower() for lb in results["labels"]]
    return boxes, scores, labels


def _bottom_centre(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(round(float(np.mean(xs)))), int(round(float(np.max(ys))))]


def _shadow_for_object(
    frame_bgr: np.ndarray,
    obj_mask: np.ndarray,
    other_obj_masks: List[np.ndarray],
    anchor_uv: List[int],
    dilate_px: int,
    min_area_px: int,
) -> Optional[Dict[str, Any]]:
    obj_u8 = obj_mask.astype(np.uint8)
    iters = max(1, dilate_px // 3)
    dilated = cv2.dilate(obj_u8, np.ones((3, 3), np.uint8), iterations=iters).astype(bool)

    other = np.zeros_like(obj_mask, dtype=bool)
    for m in other_obj_masks:
        other |= m.astype(bool)

    below = np.zeros_like(obj_mask, dtype=bool)
    below[max(0, anchor_uv[1] - 5) :, :] = True
    candidate = dilated & ~obj_mask & ~other & below
    if candidate.sum() < 30:
        return None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    threshold = float(np.percentile(v[candidate], 25.0))
    shadow = candidate & (v <= threshold)
    if shadow.sum() < min_area_px:
        return None

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        shadow.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return None
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(component_areas >= min_area_px)[0]
    if len(valid) == 0:
        return None

    best_label, best_area = None, -1
    for k in valid:
        cmp_label = k + 1
        ys, xs = np.where(labels == cmp_label)
        d_min = float(np.min(np.hypot(xs - anchor_uv[0], ys - anchor_uv[1])))
        if d_min > dilate_px * 1.5:
            continue
        area = int(component_areas[k])
        if area > best_area:
            best_area, best_label = area, cmp_label
    if best_label is None:
        return None

    component_mask = labels == best_label
    ys, xs = np.where(component_mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    dists = np.linalg.norm(pts - np.array(anchor_uv, dtype=np.float32), axis=1)
    shadow_far = pts[int(np.argmax(dists))]

    moments = cv2.moments(component_mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu_log = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).tolist()

    return {
        "shadow_far_uv": [int(shadow_far[0]), int(shadow_far[1])],
        "shadow_mask_area_px": int(component_mask.sum()),
        "shadow_hu_log": hu_log,
    }


def _build_sam2_predictor(sam2_config: str, sam2_hf: str, device: str):
    try:
        from sam2.build_sam import build_sam2_video_predictor_hf

        return build_sam2_video_predictor_hf(sam2_hf, device=device)
    except Exception:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=sam2_hf, filename="sam2.1_hiera_large.pt")
        return build_sam2_video_predictor(sam2_config, ckpt, device=device)


# ---------------------------------------------------------------------------
# Circular-statistics helpers
# ---------------------------------------------------------------------------
def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    rads = np.radians(angles_deg)
    return float(np.degrees(math.atan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))))


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    rads = np.radians(angles_deg)
    R = math.sqrt(float(np.mean(np.sin(rads))) ** 2 + float(np.mean(np.cos(rads))) ** 2)
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(math.sqrt(-2.0 * math.log(R))))


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _run_tracking(
    video_path: str,
    text_prompt: str,
    anchor_label: str,
    hud_rects: List[Tuple[float, float, float, float]],
    device: str,
    gd_model_id: str = GD_DEFAULT_MODEL,
    gd_threshold: float = 0.30,
    gd_text_threshold: float = 0.25,
    sam2_config: str = SAM2_DEFAULT_CONFIG,
    sam2_hf: str = SAM2_DEFAULT_HF,
    max_objects: int = 6,
    dilate_px: int = 80,
    min_shadow_area_px: int = 50,
    anchor_min_area_frac: float = 0.005,
) -> Dict[str, Any]:
    """Run GroundingDINO + SAM 2 tracking and shadow extraction."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_frame_raw = cap.read()
    cap.release()
    if not ok:
        return {"error": f"cannot read first frame of {video_path}"}

    first_frame = _apply_hud_mask(first_frame_raw, hud_rects)
    boxes, scores, labels = _detect_objects_groundingdino(
        first_frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
    )
    if len(boxes) == 0:
        return {
            "n_frames_total": n_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "first_frame_objects": [],
            "frames": [],
            "error": "GroundingDINO found 0 objects on frame 0",
        }

    order = np.argsort(-scores)[: max_objects]
    boxes, scores, labels = boxes[order], scores[order], [labels[i] for i in order]
    first_frame_objects = [
        {"id": i + 1, "label": labels[i], "score": float(scores[i]),
         "box": [float(b) for b in boxes[i]]}
        for i in range(len(boxes))
    ]

    predictor = _build_sam2_predictor(sam2_config, sam2_hf, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        cap = cv2.VideoCapture(video_path)
        for f_idx in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break
            fr_clean = _apply_hud_mask(fr, hud_rects)
            cv2.imwrite(
                os.path.join(tmpdir, f"{f_idx:06d}.jpg"),
                fr_clean,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
        cap.release()

        state = predictor.init_state(video_path=tmpdir)
        for entry, box in zip(first_frame_objects, boxes):
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0,
                obj_id=entry["id"], box=box.astype(np.float32),
            )

        masks_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
        for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            per_obj: Dict[int, np.ndarray] = {}
            for k, oid in enumerate(obj_ids):
                m = (mask_logits[k] > 0.0).cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                per_obj[int(oid)] = m.astype(bool)
            masks_per_frame[int(f_idx)] = per_obj

    anchor_min_area_px = int(anchor_min_area_frac * width * height)
    anchor_id = None
    for entry in first_frame_objects:
        if anchor_label.lower() in entry["label"].lower():
            anchor_id = entry["id"]
            break
    if anchor_id is None and first_frame_objects:
        anchor_id = first_frame_objects[0]["id"]

    cap = cv2.VideoCapture(video_path)
    frames_out: List[Dict[str, Any]] = []
    for f_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_clean = _apply_hud_mask(frame, hud_rects)
        per_obj = masks_per_frame.get(f_idx)
        if per_obj is None:
            continue
        anchor_mask = per_obj.get(anchor_id) if anchor_id else None
        anchor_area = int(anchor_mask.sum()) if anchor_mask is not None else 0
        scene_retained = bool(anchor_area >= anchor_min_area_px)

        objects_out: List[Dict[str, Any]] = []
        for entry in first_frame_objects:
            mask = per_obj.get(entry["id"])
            if mask is None or not mask.any():
                continue
            anchor = _bottom_centre(mask)
            if anchor is None:
                continue
            others = [m for oid, m in per_obj.items() if oid != entry["id"]]
            shadow = _shadow_for_object(
                frame_clean, mask, others, anchor,
                dilate_px=dilate_px, min_area_px=min_shadow_area_px,
            )
            shadow_far = shadow["shadow_far_uv"] if shadow else None
            shadow_vec = (
                [shadow_far[0] - anchor[0], shadow_far[1] - anchor[1]]
                if shadow_far
                else None
            )
            objects_out.append({
                "id": entry["id"],
                "label": entry["label"],
                "obj_anchor_uv": anchor,
                "obj_mask_area_px": int(mask.sum()),
                "shadow_far_uv": shadow_far,
                "shadow_vector_2d": shadow_vec,
                "shadow_mask_area_px": shadow["shadow_mask_area_px"] if shadow else 0,
                "shadow_hu_log": shadow.get("shadow_hu_log") if shadow else None,
            })
        frames_out.append({
            "frame": f_idx,
            "scene_retained": scene_retained,
            "anchor_area_px": anchor_area,
            "objects": objects_out,
        })
    cap.release()

    return {
        "n_frames_total": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "anchor_label": anchor_label,
        "anchor_id": anchor_id,
        "first_frame_objects": first_frame_objects,
        "frames": frames_out,
    }


def _compute_metrics(track: Dict[str, Any]) -> Dict[str, Any]:
    """Compute 6 shadow-consistency metrics from a per-frame track."""
    frames = track.get("frames", [])
    n_total = track.get("n_frames_total") or len(frames)

    if not frames:
        return {
            "scene_retention_rate_pct": 0.0,
            "shadow_detection_rate_pct": None,
            "multi_object_sun_disagreement_deg": None,
            "shadow_direction_jitter_deg_per_frame": None,
            "shadow_area_stability_pct": None,
            "shadow_shape_drift": None,
        }

    n_retained = sum(1 for f in frames if f.get("scene_retained", False))
    srr_pct = 100.0 * n_retained / max(1, n_total)

    azimuth_means: List[float] = []
    azimuth_spreads: List[float] = []
    n_shadows_per_frame: List[int] = []
    area_hist: Dict[int, List[Tuple[int, int]]] = {}
    hu_hist: Dict[int, List[Tuple[int, List[float]]]] = {}

    for fr in frames:
        if not fr.get("scene_retained", False):
            n_shadows_per_frame.append(0)
            azimuth_means.append(float("nan"))
            azimuth_spreads.append(float("nan"))
            continue

        per_obj_az: List[float] = []
        for o in fr["objects"]:
            sv = o.get("shadow_vector_2d")
            if sv is None:
                continue
            dx, dy = float(sv[0]), float(sv[1])
            if dx * dx + dy * dy < 4.0:
                continue
            per_obj_az.append(math.degrees(math.atan2(dy, dx)))
            oid = o["id"]
            area_hist.setdefault(oid, []).append(
                (fr["frame"], int(o.get("shadow_mask_area_px") or 0))
            )
            hu = o.get("shadow_hu_log")
            if hu is not None:
                hu_hist.setdefault(oid, []).append((fr["frame"], list(hu)))

        n_shadows_per_frame.append(len(per_obj_az))
        if not per_obj_az:
            azimuth_means.append(float("nan"))
            azimuth_spreads.append(float("nan"))
        elif len(per_obj_az) == 1:
            azimuth_means.append(per_obj_az[0])
            azimuth_spreads.append(float("nan"))
        else:
            arr = np.asarray(per_obj_az)
            azimuth_means.append(_circular_mean_deg(arr))
            azimuth_spreads.append(_circular_std_deg(arr))

    n_with_shadow = sum(
        1 for f, n in zip(frames, n_shadows_per_frame)
        if f.get("scene_retained", False) and n >= 1
    )
    sdr_pct = 100.0 * n_with_shadow / max(1, n_retained) if n_retained else None

    spread_arr = np.asarray(azimuth_spreads, dtype=np.float64)
    valid_spread = spread_arr[~np.isnan(spread_arr)]
    mosd = float(np.mean(valid_spread)) if valid_spread.size else None

    azim = np.asarray(azimuth_means, dtype=np.float64)
    steps = [
        abs(_angle_diff_deg(b, a))
        for a, b in zip(azim[:-1], azim[1:])
        if not (math.isnan(a) or math.isnan(b))
    ]
    sdj = float(np.median(steps)) if steps else None

    area_changes: List[float] = []
    for history in area_hist.values():
        history.sort()
        for (f0, a0), (f1, a1) in zip(history[:-1], history[1:]):
            if f1 - f0 != 1 or a0 <= 0:
                continue
            area_changes.append(abs(a1 - a0) / a0)
    sas = float(np.mean(area_changes) * 100.0) if area_changes else None

    hu_steps: List[float] = []
    for history in hu_hist.values():
        history.sort()
        for (f0, h0), (f1, h1) in zip(history[:-1], history[1:]):
            if f1 - f0 != 1:
                continue
            v0, v1 = np.asarray(h0, dtype=np.float64), np.asarray(h1, dtype=np.float64)
            if v0.shape != v1.shape:
                continue
            hu_steps.append(float(np.linalg.norm(v1 - v0)))
    ssd = float(np.mean(hu_steps)) if hu_steps else None

    return {
        "scene_retention_rate_pct": round(srr_pct, 2),
        "shadow_detection_rate_pct": round(sdr_pct, 2) if sdr_pct is not None else None,
        "multi_object_sun_disagreement_deg": round(mosd, 2) if mosd is not None else None,
        "shadow_direction_jitter_deg_per_frame": round(sdj, 2) if sdj is not None else None,
        "shadow_area_stability_pct": round(sas, 2) if sas is not None else None,
        "shadow_shape_drift": round(ssd, 4) if ssd is not None else None,
        "n_frames_total": n_total,
        "n_frames_scene_retained": n_retained,
        "n_frames_with_shadow": n_with_shadow,
    }


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------
def analyze_shadow_consistency_from_context(
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point called from :pymod:`evaluation.registry`."""
    if torch is None:
        return {"module": "shadow_consistency", "error": "torch is not installed"}
    if build_sam2_video_predictor is None:
        return {"module": "shadow_consistency", "error": "sam2 is not installed"}
    if AutoProcessor is None:
        return {"module": "shadow_consistency", "error": "transformers is not installed"}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts = context.artifacts or {}
    text_prompt = artifacts.get("shadow_text_prompt", "table. chair. lamp.")
    anchor_label = artifacts.get("shadow_anchor_label", "table")
    hud_preset = str(artifacts.get("shadow_hud_preset", "none"))
    hud_rects = HUD_PRESETS.get(hud_preset, [])

    video_path = context.candidate_video
    if not video_path:
        return {"module": "shadow_consistency", "error": "candidate_video is required"}

    track = _run_tracking(
        video_path=video_path,
        text_prompt=text_prompt,
        anchor_label=anchor_label,
        hud_rects=hud_rects,
        device=device,
    )

    if "error" in track and not track.get("frames"):
        return {"module": "shadow_consistency", "error": track["error"]}

    metrics = _compute_metrics(track)

    return {
        "module": "shadow_consistency",
        "video": video_path,
        "device": device,
        "text_prompt": text_prompt,
        "anchor_label": anchor_label,
        "hud_preset": hud_preset,
        "first_frame_objects": track.get("first_frame_objects", []),
        **metrics,
    }
