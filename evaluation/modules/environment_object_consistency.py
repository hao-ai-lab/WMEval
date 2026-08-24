"""Environment Object and Character Consistency evaluation (paper appendix B.1).

Pipeline, mirroring :mod:`evaluation.modules.shadow_consistency`'s GroundingDINO
+ SAM 2 tracking scaffold:

  1. Detect a fixed set of key objects (``eoc_text_prompt``) plus the main
     character (``eoc_character_prompt``) on frame 0 with GroundingDINO.
  2. Track every detected object through the candidate rollout with SAM 2's
     video predictor to get per-frame masks/centroids.
  3. Independently re-run GroundingDINO on every frame (not just the tracked
     boxes) to obtain a genuine per-frame detection confidence signal
     ``s_{t,o}`` for Object Permanence and Character Consistency, since SAM 2's
     video propagator does not expose a per-frame detection score.
  4. Feed a monocular depth estimator (optional transformers dependency) at
     each tracked object's mask centroid to get a candidate-side depth
     estimate, and back-project centroids with pinhole intrinsics to get
     candidate-side pairwise 3D distances.
  5. Feed cropped character masks through DINOv2 to get per-frame identity
     embeddings for Character Consistency.

Four metrics (paper appendix B.1):

  - ``depth_absrel_pct``            -- needs Unity-side analytical ground truth.
  - ``pairwise_distance_mae_pct``   -- needs Unity-side analytical ground truth.
  - ``object_permanence_pct``       -- reference-free (candidate only).
  - ``character_consistency_pct``   -- reference-free (candidate only).

Ground-truth sourcing (depth / pairwise distance): the paper specifies these
should be computed *analytically* from Unity's known per-object 3D world
transforms plus camera pose/intrinsics -- not by re-running detectors on
Unity's own reference video. This module therefore looks for three artifacts
on ``context.artifacts``:

  - ``eoc_unity_object_transforms_json``: per-frame per-object 3D world
    positions (schema documented in :func:`_load_unity_object_transforms`).
  - ``eoc_unity_poses_npy``: ``(T, 4, 4)`` camera-to-world matrices (same
    convention as the ``poses.npy`` artifact already bundled in
    e.g. ``cases/BallRoller/ctrl/reference/artifacts/poses.npy``).
  - ``eoc_unity_intrinsics_npy``: ``(T, 4)`` ``[fx, fy, cx, cy]`` per frame
    (same convention as the bundled ``intrinsics.npy``).

As of this writing no case bundle ships ``eoc_unity_object_transforms_json``
(see ``eval_sets/wmeval_lite_v1/PENDING.md``), so when it is absent this
module reports ``ground_truth_status: "pending"`` and leaves
``depth_absrel_pct`` / ``pairwise_distance_mae_pct`` as ``None`` rather than
inventing ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )
except ImportError:
    AutoModelForZeroShotObjectDetection = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]

from ..context import EvaluationContext
from .shadow_consistency import (
    GD_DEFAULT_MODEL,
    SAM2_DEFAULT_CONFIG,
    SAM2_DEFAULT_HF,
    _build_sam2_predictor,
    _detect_objects_groundingdino,
)

EPS = 1e-6
DEPTH_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ---------------------------------------------------------------------------
# Ground-truth artifact loading
# ---------------------------------------------------------------------------
def _load_unity_object_transforms(path: str) -> Dict[str, Any]:
    """Load Unity-side analytical object ground truth.

    Expected schema::

        {
          "objects": {
            "<label>": {
              "frames": [0, 1, 2, ...],
              "positions_world": [[x, y, z], [x, y, z], ...]
            },
            ...
          }
        }

    ``<label>`` should match the noun phrase GroundingDINO is asked to
    detect (case-insensitive) so candidate detections and Unity ground truth
    can be paired by label.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"unity object transforms artifact does not exist: {path}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "objects" not in payload:
        raise ValueError(f"unity object transforms artifact must be a dict with an 'objects' key: {path}")
    return payload


def _load_camera_poses(path: str) -> np.ndarray:
    poses = np.load(path, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"unity poses artifact must have shape (T, 4, 4), got {poses.shape}: {path}")
    return poses.astype(np.float64)


def _load_camera_intrinsics(path: str) -> np.ndarray:
    intr = np.load(path, allow_pickle=False)
    if intr.ndim != 2 or intr.shape[1] != 4:
        raise ValueError(f"unity intrinsics artifact must have shape (T, 4), got {intr.shape}: {path}")
    return intr.astype(np.float64)


def _camera_position_at_frame(poses: np.ndarray, frame_idx: int) -> np.ndarray:
    idx = min(max(frame_idx, 0), poses.shape[0] - 1)
    return poses[idx, :3, 3]


def _unity_ground_truth_depths_and_distances(
    labels: Sequence[str],
    gt: Dict[str, Any],
    poses: np.ndarray,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], List[str]]:
    """Analytical Unity depth (camera-to-object range) and pairwise 3D
    distance, both averaged over the frames the ground truth provides.

    Returns (depth_by_label, distance_by_label_pair, missing_labels).
    """
    objects = gt.get("objects", {})
    lower_index = {str(k).strip().lower(): v for k, v in objects.items()}

    positions_by_label: Dict[str, Dict[int, np.ndarray]] = {}
    missing: List[str] = []
    for label in labels:
        entry = lower_index.get(label.strip().lower())
        if entry is None:
            missing.append(label)
            continue
        frames = entry.get("frames", [])
        positions = entry.get("positions_world", [])
        positions_by_label[label] = {
            int(f): np.asarray(pos, dtype=np.float64)
            for f, pos in zip(frames, positions)
        }

    depth_by_label: Dict[str, float] = {}
    for label, per_frame in positions_by_label.items():
        ranges = []
        for f, pos_w in per_frame.items():
            cam_pos = _camera_position_at_frame(poses, f)
            ranges.append(float(np.linalg.norm(pos_w - cam_pos)))
        if ranges:
            depth_by_label[label] = float(np.mean(ranges))

    distance_by_pair: Dict[Tuple[str, str], float] = {}
    present_labels = list(positions_by_label.keys())
    for i in range(len(present_labels)):
        for j in range(i + 1, len(present_labels)):
            li, lj = present_labels[i], present_labels[j]
            common_frames = sorted(set(positions_by_label[li]) & set(positions_by_label[lj]))
            if not common_frames:
                continue
            dists = [
                float(np.linalg.norm(positions_by_label[li][f] - positions_by_label[lj][f]))
                for f in common_frames
            ]
            distance_by_pair[(li, lj)] = float(np.mean(dists))

    return depth_by_label, distance_by_pair, missing


# ---------------------------------------------------------------------------
# Candidate-side tracking
# ---------------------------------------------------------------------------
def _track_objects(
    video_path: str,
    text_prompt: str,
    character_prompt: str,
    device: str,
    gd_model_id: str,
    gd_threshold: float,
    gd_text_threshold: float,
    sam2_config: str,
    sam2_hf: str,
    max_objects: int,
) -> Dict[str, Any]:
    """GroundingDINO (frame 0 seed) + SAM 2 video tracking of key objects and
    the main character. Returns per-frame masks/centroids plus independent
    per-frame GroundingDINO re-detections used for detection confidence."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        return {"error": f"cannot read first frame of {video_path}"}

    obj_boxes, obj_scores, obj_labels = _detect_objects_groundingdino(
        first_frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
    )
    char_boxes, char_scores, char_labels = _detect_objects_groundingdino(
        first_frame, character_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
    )

    if len(obj_boxes) == 0 and len(char_boxes) == 0:
        return {
            "n_frames_total": n_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "objects": [],
            "character_id": None,
            "frames": [],
            "detections_by_frame": [],
            "error": "GroundingDINO found 0 objects and 0 characters on frame 0",
        }

    order = np.argsort(-obj_scores)[:max_objects] if len(obj_scores) else np.array([], dtype=int)
    boxes = obj_boxes[order] if len(order) else np.empty((0, 4), dtype=np.float32)
    scores = obj_scores[order] if len(order) else np.empty((0,), dtype=np.float32)
    labels = [obj_labels[i] for i in order]

    seed_boxes: List[np.ndarray] = list(boxes)
    seed_labels: List[str] = list(labels)
    seed_scores: List[float] = [float(s) for s in scores]
    character_id: Optional[int] = None
    if len(char_boxes) > 0:
        best = int(np.argmax(char_scores))
        seed_boxes.append(char_boxes[best])
        seed_labels.append(char_labels[best])
        seed_scores.append(float(char_scores[best]))
        character_id = len(seed_boxes)  # 1-indexed id assigned below

    objects_meta = [
        {"id": i + 1, "label": seed_labels[i], "seed_score": seed_scores[i],
         "box": [float(v) for v in seed_boxes[i]], "is_character": (i + 1) == character_id}
        for i in range(len(seed_boxes))
    ]

    predictor = _build_sam2_predictor(sam2_config, sam2_hf, device)

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cap = cv2.VideoCapture(video_path)
        for f_idx in range(n_frames):
            ok, fr = cap.read()
            if not ok:
                break
            cv2.imwrite(
                os.path.join(tmpdir, f"{f_idx:06d}.jpg"),
                fr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
        cap.release()

        state = predictor.init_state(video_path=tmpdir)
        for entry, box in zip(objects_meta, seed_boxes):
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

    frames_out: List[Dict[str, Any]] = []
    detections_by_frame: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(video_path)
    for f_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # Independent re-detection (not tied to SAM 2 IDs) for a genuine
        # per-frame confidence signal.
        det_boxes, det_scores, det_labels = _detect_objects_groundingdino(
            frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
        )
        cdet_boxes, cdet_scores, cdet_labels = _detect_objects_groundingdino(
            frame, character_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
        )
        detections_by_frame.append({
            "frame": f_idx,
            "objects": [
                {"label": det_labels[i], "score": float(det_scores[i])}
                for i in range(len(det_boxes))
            ],
            "character": [
                {"label": cdet_labels[i], "score": float(cdet_scores[i])}
                for i in range(len(cdet_boxes))
            ],
        })

        per_obj = masks_per_frame.get(f_idx, {})
        objects_out: List[Dict[str, Any]] = []
        for entry in objects_meta:
            mask = per_obj.get(entry["id"])
            if mask is None or not mask.any():
                objects_out.append({
                    "id": entry["id"], "label": entry["label"],
                    "is_character": entry["is_character"],
                    "mask_area_px": 0, "centroid_uv": None,
                })
                continue
            ys, xs = np.where(mask)
            centroid = [float(np.mean(xs)), float(np.mean(ys))]
            objects_out.append({
                "id": entry["id"],
                "label": entry["label"],
                "is_character": entry["is_character"],
                "mask_area_px": int(mask.sum()),
                "centroid_uv": centroid,
                "bbox_xyxy": [
                    float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()),
                ],
            })
        frames_out.append({"frame": f_idx, "objects": objects_out})
    cap.release()

    return {
        "n_frames_total": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "objects": objects_meta,
        "character_id": character_id,
        "frames": frames_out,
        "detections_by_frame": detections_by_frame,
    }


# ---------------------------------------------------------------------------
# Candidate-side depth / distance estimation
# ---------------------------------------------------------------------------
def _load_depth_estimator(model_id: str, device: str):
    from transformers import pipeline

    return pipeline("depth-estimation", model=model_id, device=(0 if device == "cuda" else -1))


def _estimate_candidate_depths_and_distances(
    video_path: str,
    track: Dict[str, Any],
    depth_model_id: str,
    device: str,
    intrinsics: Optional[np.ndarray],
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Monocular depth estimate at each object's mask centroid (relative
    depth units, per the depth model's own scale -- see module docstring),
    back-projected with pinhole intrinsics into candidate-side pairwise
    3D distances."""
    from PIL import Image

    depth_pipe = _load_depth_estimator(depth_model_id, device)

    labels = [o["label"] for o in track["objects"]]
    depth_series: Dict[str, List[float]] = {label: [] for label in labels}
    point3d_series: Dict[str, List[np.ndarray]] = {label: [] for label in labels}

    cap = cv2.VideoCapture(video_path)
    frame_lookup = {fr["frame"]: fr for fr in track["frames"]}
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fr = frame_lookup.get(f_idx)
        if fr is None:
            continue
        objects_here = [o for o in fr["objects"] if o.get("centroid_uv")]
        if not objects_here:
            continue

        image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_out = depth_pipe(image_rgb)
        depth_map = np.asarray(depth_out["predicted_depth"])
        if depth_map.shape != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        if intrinsics is not None:
            row = intrinsics[min(f_idx, intrinsics.shape[0] - 1)]
            fx, fy, cx, cy = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        else:
            fx = fy = float(max(frame.shape[1], frame.shape[0]))
            cx, cy = frame.shape[1] / 2.0, frame.shape[0] / 2.0

        for o in objects_here:
            u, v = o["centroid_uv"]
            ui, vi = int(round(u)), int(round(v))
            ui = min(max(ui, 0), depth_map.shape[1] - 1)
            vi = min(max(vi, 0), depth_map.shape[0] - 1)
            d = float(depth_map[vi, ui])
            depth_series[o["label"]].append(d)
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            point3d_series[o["label"]].append(np.asarray([x, y, d], dtype=np.float64))
    cap.release()

    depth_by_label = {
        label: float(np.mean(vals)) for label, vals in depth_series.items() if vals
    }

    distance_by_pair: Dict[Tuple[str, str], float] = {}
    present = [label for label in labels if point3d_series.get(label)]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            li, lj = present[i], present[j]
            n = min(len(point3d_series[li]), len(point3d_series[lj]))
            if n == 0:
                continue
            dists = [
                float(np.linalg.norm(point3d_series[li][k] - point3d_series[lj][k]))
                for k in range(n)
            ]
            distance_by_pair[(li, lj)] = float(np.mean(dists))

    return depth_by_label, distance_by_pair


# ---------------------------------------------------------------------------
# Reference-free metrics
# ---------------------------------------------------------------------------
def _object_permanence(
    track: Dict[str, Any],
    tau_det: float,
) -> Optional[float]:
    n_total = track.get("n_frames_total") or len(track.get("frames", []))
    anchor_labels = [o["label"] for o in track["objects"] if not o.get("is_character")]
    if not anchor_labels or n_total == 0:
        return None

    hits = 0
    total = 0
    for det in track.get("detections_by_frame", []):
        best_by_label: Dict[str, float] = {}
        for d in det["objects"]:
            lbl = d["label"]
            best_by_label[lbl] = max(best_by_label.get(lbl, 0.0), d["score"])
        for label in anchor_labels:
            total += 1
            if best_by_label.get(label, 0.0) >= tau_det:
                hits += 1
    if total == 0:
        return None
    return 100.0 * hits / total


def _load_dinov2(device: str):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


def _character_embedding(model, frame_bgr: np.ndarray, bbox_xyxy: List[float], device: str) -> np.ndarray:
    import torch

    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_bgr.shape[1], max(x1, x0 + 1)), min(frame_bgr.shape[0], max(y1, y0 + 1))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_bgr
    crop = cv2.resize(crop, (224, 224))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop_rgb = (crop_rgb - mean) / std
    tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        feat = model(tensor)
    return feat[0].detach().cpu().numpy()


def _character_consistency(
    video_path: str,
    track: Dict[str, Any],
    tau_det: float,
    tau_mask: float,
    tau_id: float,
    device: str,
) -> Optional[Dict[str, Any]]:
    character_id = track.get("character_id")
    if character_id is None:
        return None

    if torch is None:
        return {"error": "torch is not installed"}

    try:
        model = _load_dinov2(device)
    except Exception as exc:
        return {"error": f"failed to load DINOv2 via torch.hub: {exc}"}

    frame_lookup = {fr["frame"]: fr for fr in track["frames"]}
    det_lookup = {d["frame"]: d for d in track.get("detections_by_frame", [])}

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    embeddings: Dict[int, np.ndarray] = {}
    areas: Dict[int, int] = {}
    scores: Dict[int, float] = {}

    first_area: Optional[int] = None
    first_embedding: Optional[np.ndarray] = None

    for f_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fr = frame_lookup.get(f_idx)
        char_obj = None
        if fr is not None:
            for o in fr["objects"]:
                if o["id"] == character_id:
                    char_obj = o
                    break
        det = det_lookup.get(f_idx, {"character": []})
        best_score = max((d["score"] for d in det.get("character", [])), default=0.0)
        scores[f_idx] = best_score

        if char_obj is None or not char_obj.get("centroid_uv") or char_obj["mask_area_px"] <= 0:
            continue
        areas[f_idx] = char_obj["mask_area_px"]
        try:
            embeddings[f_idx] = _character_embedding(model, frame, char_obj["bbox_xyxy"], device)
        except Exception:
            continue
        if first_area is None:
            first_area = char_obj["mask_area_px"]
            first_embedding = embeddings[f_idx]
    cap.release()

    if first_embedding is None or first_area is None:
        return {"error": "character never segmented; cannot compute identity similarity"}

    hits = 0
    total = 0
    per_frame: List[Dict[str, Any]] = []
    for f_idx in range(n_frames):
        total += 1
        s_t = scores.get(f_idx, 0.0)
        area = areas.get(f_idx)
        a_t = (area / first_area) if area else 0.0
        a_t = min(a_t, 1.0)
        if f_idx in embeddings:
            phi_t = embeddings[f_idx]
            denom = np.linalg.norm(phi_t) * np.linalg.norm(first_embedding) + EPS
            cos_sim = float(np.dot(phi_t, first_embedding) / denom)
        else:
            cos_sim = None
        passed = bool(
            s_t >= tau_det and a_t >= tau_mask and cos_sim is not None and cos_sim >= tau_id
        )
        if passed:
            hits += 1
        per_frame.append({
            "frame": f_idx, "detection_score": s_t, "mask_quality": a_t,
            "identity_cosine": cos_sim, "passed": passed,
        })

    return {
        "character_consistency_pct": round(100.0 * hits / total, 2) if total else None,
        "per_frame": per_frame,
    }


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------
def analyze_environment_object_consistency_from_context(
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point called from :pymod:`evaluation.registry`."""
    if torch is None:
        return {"module": "environment_object_consistency", "error": "torch is not installed"}
    if build_sam2_video_predictor is None:
        return {"module": "environment_object_consistency", "error": "sam2 is not installed"}
    if AutoProcessor is None:
        return {"module": "environment_object_consistency", "error": "transformers is not installed"}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts = context.artifacts or {}
    text_prompt = str(artifacts.get("eoc_text_prompt", "ball. chair. table."))
    character_prompt = str(artifacts.get("eoc_character_prompt", "character. player."))
    tau_det = float(artifacts.get("eoc_tau_det", 0.30))
    tau_mask = float(artifacts.get("eoc_tau_mask", 0.50))
    tau_id = float(artifacts.get("eoc_tau_id", 0.50))
    max_objects = int(artifacts.get("eoc_max_objects", 6))
    depth_model_id = str(artifacts.get("eoc_depth_model", DEPTH_DEFAULT_MODEL))

    video_path = context.candidate_video
    if not video_path:
        return {"module": "environment_object_consistency", "error": "candidate_video is required"}

    track = _track_objects(
        video_path=video_path,
        text_prompt=text_prompt,
        character_prompt=character_prompt,
        device=device,
        gd_model_id=str(artifacts.get("eoc_gd_model") or GD_DEFAULT_MODEL),
        gd_threshold=0.30,
        gd_text_threshold=0.25,
        sam2_config=SAM2_DEFAULT_CONFIG,
        sam2_hf=str(artifacts.get("sam2_model") or SAM2_DEFAULT_HF),
        max_objects=max_objects,
    )
    if "error" in track and not track.get("frames"):
        return {"module": "environment_object_consistency", "error": track["error"]}

    result: Dict[str, Any] = {
        "module": "environment_object_consistency",
        "video": video_path,
        "device": device,
        "text_prompt": text_prompt,
        "character_prompt": character_prompt,
        "objects": track.get("objects", []),
    }

    # -- Reference-free metrics -------------------------------------------
    result["object_permanence_pct"] = _object_permanence(track, tau_det)
    char_result = _character_consistency(video_path, track, tau_det, tau_mask, tau_id, device)
    if char_result is not None and "error" not in char_result:
        result["character_consistency_pct"] = char_result["character_consistency_pct"]
        result["character_consistency_detail"] = char_result["per_frame"]
    elif char_result is not None:
        result["character_consistency_pct"] = None
        result["character_consistency_error"] = char_result["error"]
    else:
        result["character_consistency_pct"] = None
        result["character_consistency_note"] = "no character detected on frame 0"

    # -- Unity-grounded metrics --------------------------------------------
    gt_json_path = artifacts.get("eoc_unity_object_transforms_json")
    poses_path = artifacts.get("eoc_unity_poses_npy")
    intrinsics_path = artifacts.get("eoc_unity_intrinsics_npy")

    if not gt_json_path or not poses_path:
        result["ground_truth_status"] = "pending"
        result["ground_truth_note"] = (
            "eoc_unity_object_transforms_json and/or eoc_unity_poses_npy artifacts "
            "are not configured for this case; see eval_sets/wmeval_lite_v1/PENDING.md. "
            "depth_absrel_pct / pairwise_distance_mae_pct cannot be computed without "
            "Unity-side analytical 3D object transforms."
        )
        result["depth_absrel_pct"] = None
        result["pairwise_distance_mae_pct"] = None
        return result

    try:
        gt = _load_unity_object_transforms(str(gt_json_path))
        poses = _load_camera_poses(str(poses_path))
        intrinsics = _load_camera_intrinsics(str(intrinsics_path)) if intrinsics_path else None
    except Exception as exc:
        result["ground_truth_status"] = "error"
        result["ground_truth_note"] = f"failed to load Unity ground-truth artifacts: {exc}"
        result["depth_absrel_pct"] = None
        result["pairwise_distance_mae_pct"] = None
        return result

    labels = [o["label"] for o in track["objects"] if not o.get("is_character")]
    gt_depth, gt_distance, missing_labels = _unity_ground_truth_depths_and_distances(
        labels, gt, poses
    )
    if missing_labels:
        result["ground_truth_missing_labels"] = missing_labels

    try:
        cand_depth, cand_distance = _estimate_candidate_depths_and_distances(
            video_path, track, depth_model_id, device, intrinsics
        )
    except Exception as exc:
        result["ground_truth_status"] = "error"
        result["ground_truth_note"] = f"candidate depth estimation failed: {exc}"
        result["depth_absrel_pct"] = None
        result["pairwise_distance_mae_pct"] = None
        return result

    result["ground_truth_status"] = "available"

    common_labels = [l for l in gt_depth if l in cand_depth]
    if common_labels:
        rel_errs = [
            abs(cand_depth[l] - gt_depth[l]) / (gt_depth[l] + EPS) for l in common_labels
        ]
        result["depth_absrel_pct"] = round(100.0 * float(np.mean(rel_errs)), 2)
    else:
        result["depth_absrel_pct"] = None

    common_pairs = [p for p in gt_distance if p in cand_distance]
    if common_pairs:
        rel_errs = [
            abs(cand_distance[p] - gt_distance[p]) / (gt_distance[p] + EPS) for p in common_pairs
        ]
        result["pairwise_distance_mae_pct"] = round(100.0 * float(np.mean(rel_errs)), 2)
    else:
        result["pairwise_distance_mae_pct"] = None

    return result
