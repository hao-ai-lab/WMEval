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
# "-Small-hf" (no "-Metric-" prefix) is a RELATIVE depth model: its output has
# an arbitrary per-image scale/shift, not real metric units, so comparing it
# directly to Unity's metric-scale depth (as depth_absrel_pct does) is a
# scale-mismatched comparison by construction -- empirically confirmed this
# was responsible for most of the ~55-99% depth_absrel_pct self-match error
# seen across every case before this fix (see PENDING.md).
DEPTH_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_METRIC_INDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# Candidate checkpoints _auto_select_depth_model chooses between when a
# case doesn't explicitly override eoc_depth_model/kin_depth_model. A
# human manually picking indoor vs. outdoor per scene is a silent-wrong-
# by-default failure mode (BilliardGame needed indoor: 6-24% error vs.
# 64-95%+ outdoor -- a large, silent accuracy hit for anyone who forgot to
# set the override); auto-selecting by testing both against a real,
# known Unity ground-truth depth removes the guesswork entirely.
DEPTH_AUTO_SELECT_CANDIDATES = (DEPTH_DEFAULT_MODEL, DEPTH_METRIC_INDOOR_MODEL)


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


def _load_flat_label_json(path: str, description: str) -> Dict[str, float]:
    """Shared loader for every "known physical constant per label" artifact
    (masses, diameters, and any future one): a flat {label: value} JSON
    file, resolved to an absolute path at case-load time, keyed by a
    case-insensitive label lookup. Both kinematics_mechanics.py's
    _load_object_masses and this module's _load_object_diameters delegate
    here instead of each re-implementing the identical
    exists/parse/validate/lowercase logic."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"unity {description} artifact does not exist: {path}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"unity {description} artifact must be a flat {{label: value}} dict: {path}")
    return {str(k).strip().lower(): float(v) for k, v in payload.items()}


def _load_object_diameters(path: str) -> Dict[str, float]:
    """Real per-label object diameter in meters, a known physical constant
    like object_masses.json's masses -- same file-artifact convention
    (flat {label: value} JSON, resolved to an absolute path at case-load
    time, loaded with the same case-insensitive-label lookup), rather than
    a second, differently-shaped mechanism (an inline JSON string in
    case.yaml) for the same kind of "known ground truth constant" data."""
    return _load_flat_label_json(path, "object diameters")


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
_DEGENERATE_BOX_AREA_FRACTION = 0.6


def _dedupe_seed_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    frame_width: int,
    frame_height: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Drop degenerate whole-frame boxes and collapse duplicate detections
    that share the same text label to their single highest-scoring instance.

    GroundingDINO with a multi-noun prompt (e.g. "ramp. dome.") can emit a
    spurious box spanning nearly the entire frame under a merged label (e.g.
    "ramp dome"), and/or multiple distinct boxes under the same single-noun
    label. Downstream depth/distance aggregation buckets purely by label
    string, so unfiltered duplicates silently average a real detection with
    a spurious one. We assume one tracked instance per label, matching how
    ground-truth object_transforms.json is keyed.
    """
    frame_area = float(frame_width * frame_height)
    keep_idx = [
        i for i in range(len(boxes))
        if frame_area <= 0 or _box_area(boxes[i]) / frame_area < _DEGENERATE_BOX_AREA_FRACTION
    ]
    if not keep_idx:
        keep_idx = list(range(len(boxes)))  # all degenerate: keep as-is rather than drop everything

    best_by_label: Dict[str, int] = {}
    for i in keep_idx:
        label = labels[i]
        if label not in best_by_label or scores[i] > scores[best_by_label[label]]:
            best_by_label[label] = i

    final_idx = sorted(best_by_label.values())
    return boxes[final_idx], scores[final_idx], [labels[i] for i in final_idx]


def _box_area(box: np.ndarray) -> float:
    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _character_prompt_is_disabled(prompt: str) -> bool:
    """Recognize the suite's explicit no-character sentinel.

    WanderingGarden has no persistent visible character by paper definition.
    Sending its sentinel through GroundingDINO on every frame adds a second
    full detector pass without defining a metric, and can create spurious
    character boxes.  Empty prompts and the established sentinel therefore
    disable that branch explicitly.
    """
    normalized = prompt.strip().lower()
    return not normalized or normalized.startswith("no_character_configured_for_this_case")


def _empty_detections() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        [],
    )


def _detect_prompt_groups(
    frame: np.ndarray,
    prompt: str,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Detect pipe-delimited prompts independently, then concatenate them.

    GroundingDINO can emit one merged phrase for a period-delimited prompt
    such as ``"yellow ball. red ball."``. That merged label cannot be joined
    to per-object Unity ground truth. A pipe explicitly requests independent
    detector calls while retaining the legacy single-prompt behavior.
    """
    groups = [item.strip() for item in prompt.split("|") if item.strip()]
    if not groups:
        return _empty_detections()
    detections = [
        _detect_objects_groundingdino(
            frame,
            group,
            model_id,
            box_threshold,
            text_threshold,
            device,
        )
        for group in groups
    ]
    boxes = [item[0] for item in detections if len(item[0])]
    scores = [item[1] for item in detections if len(item[1])]
    labels = [label for item in detections for label in item[2]]
    return (
        np.concatenate(boxes, axis=0) if boxes else np.empty((0, 4), dtype=np.float32),
        np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32),
        labels,
    )


def _anchor_semantic_class(label: str) -> Optional[str]:
    normalized = label.strip().lower()
    if "sphere" in normalized or "ball" in normalized:
        return "sphere"
    if "tree" in normalized:
        return "tree"
    if "lamp" in normalized or "lantern" in normalized:
        return "lamp"
    if "rock" in normalized:
        return "rock"
    return None


def _project_unity_anchor(
    label: str,
    *,
    gt: Dict[str, Any],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    frame_idx: int,
) -> Optional[Tuple[float, float, float]]:
    """Project one Unity anchor into its aligned candidate frame.

    ``poses`` is camera-to-world with Unity's image-down Y convention, which
    is already the convention used by the bundled intrinsics.  The returned
    tuple is ``(u, v, camera_z)``; non-positive Z is behind the camera.
    """
    objects = gt.get("objects", {})
    lower_index = {str(key).strip().lower(): value for key, value in objects.items()}
    entry = lower_index.get(label.strip().lower())
    if not isinstance(entry, dict):
        return None
    frames = [int(value) for value in entry.get("frames", [])]
    positions = entry.get("positions_world", [])
    if not frames or not positions:
        return None
    nearest = min(range(len(frames)), key=lambda index: abs(frames[index] - frame_idx))
    position = np.asarray(positions[nearest], dtype=np.float64)
    pose = poses[min(max(frame_idx, 0), poses.shape[0] - 1)]
    camera_point = np.linalg.inv(pose) @ np.asarray([*position, 1.0], dtype=np.float64)
    z = float(camera_point[2])
    if z <= EPS:
        return None
    fx, fy, cx, cy = intrinsics[min(max(frame_idx, 0), intrinsics.shape[0] - 1)]
    u = float(fx * camera_point[0] / z + cx)
    v = float(fy * camera_point[1] / z + cy)
    return u, v, z


def _assign_fixed_anchor_detections(
    *,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    fixed_anchor_labels: Sequence[str],
    gt: Dict[str, Any],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    frame_idx: int,
    frame_width: int,
    frame_height: int,
    max_normalized_distance: float = 0.30,
) -> List[Dict[str, Any]]:
    """Assign generic GroundingDINO detections to the paper's fixed anchors.

    The paper names two lamp posts and two rockery ornaments, while open-vocab
    detectors generally emit only generic ``lamp post``/``rock`` labels.  The
    aligned Unity pose provides the missing identity: detections are matched
    one-to-one to visible analytical projections within the same semantic
    class.  This preserves the five-object denominator and exact Unity labels
    instead of silently collapsing several anchors into one generic label.
    """
    diagonal = max(float(np.hypot(frame_width, frame_height)), EPS)
    valid_detections: List[Tuple[int, str, float, List[float], float, float]] = []
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        semantic_class = _anchor_semantic_class(str(label))
        if semantic_class is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(
            float(frame_width * frame_height), EPS
        )
        # A close landmark can legitimately occupy most of the frame (the
        # Garden main tree is ~60%).  The stricter 0.60 generic-dedup guard is
        # therefore inappropriate once semantic class + Unity projection are
        # available to disambiguate it; reject only near-whole-frame boxes.
        if area_fraction >= 0.90:
            continue
        valid_detections.append(
            (index, semantic_class, float(score), [x1, y1, x2, y2], (x1 + x2) / 2.0, (y1 + y2) / 2.0)
        )

    pairings: List[Tuple[float, str, int]] = []
    for anchor_label in fixed_anchor_labels:
        projected = _project_unity_anchor(
            anchor_label,
            gt=gt,
            poses=poses,
            intrinsics=intrinsics,
            frame_idx=frame_idx,
        )
        if projected is None:
            continue
        u, v, _ = projected
        # Do not let an off-screen Unity anchor steal a visually similar box
        # from a different in-frame object.  A projected pivot must itself be
        # inside the aligned frame to participate in assignment.
        if not (0.0 <= u < frame_width and 0.0 <= v < frame_height):
            continue
        anchor_class = _anchor_semantic_class(anchor_label)
        for detection_index, detection_class, _score, _box, center_x, center_y in valid_detections:
            if detection_class != anchor_class:
                continue
            distance = float(np.hypot(center_x - u, center_y - v)) / diagonal
            if distance <= max_normalized_distance:
                pairings.append((distance, anchor_label, detection_index))

    assigned_anchors: set[str] = set()
    assigned_detections: set[int] = set()
    output: List[Dict[str, Any]] = []
    detection_lookup = {entry[0]: entry for entry in valid_detections}
    for distance, anchor_label, detection_index in sorted(pairings):
        if anchor_label in assigned_anchors or detection_index in assigned_detections:
            continue
        _index, _kind, score, box, _center_x, _center_y = detection_lookup[detection_index]
        output.append(
            {
                "label": anchor_label,
                "score": score,
                "box": box,
                "assignment_distance_normalized": distance,
            }
        )
        assigned_anchors.add(anchor_label)
        assigned_detections.add(detection_index)
    return output


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
    fixed_anchor_labels: Optional[Sequence[str]] = None,
    unity_gt: Optional[Dict[str, Any]] = None,
    unity_poses: Optional[np.ndarray] = None,
    unity_intrinsics: Optional[np.ndarray] = None,
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

    obj_boxes, obj_scores, obj_labels = _detect_prompt_groups(
        first_frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
    )
    character_disabled = _character_prompt_is_disabled(character_prompt)
    char_boxes, char_scores, char_labels = (
        _empty_detections()
        if character_disabled
        else _detect_objects_groundingdino(
            first_frame,
            character_prompt,
            gd_model_id,
            gd_threshold,
            gd_text_threshold,
            device,
        )
    )
    fixed_anchors = [str(label) for label in (fixed_anchor_labels or [])]
    if fixed_anchors:
        if unity_gt is None or unity_poses is None or unity_intrinsics is None:
            return {
                "n_frames_total": n_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "objects": [],
                "character_id": None,
                "frames": [],
                "detections_by_frame": [],
                "error": "fixed anchors require Unity transforms, poses, and intrinsics",
            }
        seed_anchor_detections = _assign_fixed_anchor_detections(
            boxes=obj_boxes,
            scores=obj_scores,
            labels=obj_labels,
            fixed_anchor_labels=fixed_anchors,
            gt=unity_gt,
            poses=unity_poses,
            intrinsics=unity_intrinsics,
            frame_idx=0,
            frame_width=width,
            frame_height=height,
        )
    else:
        obj_boxes, obj_scores, obj_labels = _dedupe_seed_detections(
            obj_boxes, obj_scores, obj_labels, frame_width=width, frame_height=height,
        )
        seed_anchor_detections = []

    if not fixed_anchors and len(obj_boxes) == 0 and len(char_boxes) == 0:
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

    seed_prompts: List[Tuple[int, np.ndarray]] = []
    if fixed_anchors:
        by_label = {entry["label"]: entry for entry in seed_anchor_detections}
        objects_meta = []
        for object_id, label in enumerate(fixed_anchors, start=1):
            detection = by_label.get(label)
            box = detection.get("box") if detection else None
            objects_meta.append(
                {
                    "id": object_id,
                    "label": label,
                    "seed_score": float(detection["score"]) if detection else 0.0,
                    "box": box,
                    "is_character": False,
                }
            )
            if box is not None:
                seed_prompts.append((object_id, np.asarray(box, dtype=np.float32)))
    else:
        order = np.argsort(-obj_scores)[:max_objects] if len(obj_scores) else np.array([], dtype=int)
        boxes = obj_boxes[order] if len(order) else np.empty((0, 4), dtype=np.float32)
        scores = obj_scores[order] if len(order) else np.empty((0,), dtype=np.float32)
        labels = [obj_labels[i] for i in order]
        objects_meta = [
            {
                "id": i + 1,
                "label": labels[i],
                "seed_score": float(scores[i]),
                "box": [float(v) for v in boxes[i]],
                "is_character": False,
            }
            for i in range(len(boxes))
        ]
        seed_prompts.extend(
            (entry["id"], np.asarray(entry["box"], dtype=np.float32))
            for entry in objects_meta
        )

    character_id: Optional[int] = None
    if len(char_boxes) > 0:
        best = int(np.argmax(char_scores))
        character_id = len(objects_meta) + 1
        char_box = [float(v) for v in char_boxes[best]]
        objects_meta.append(
            {
                "id": character_id,
                "label": char_labels[best],
                "seed_score": float(char_scores[best]),
                "box": char_box,
                "is_character": True,
            }
        )
        seed_prompts.append((character_id, np.asarray(char_box, dtype=np.float32)))

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

        masks_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
        if seed_prompts:
            state = predictor.init_state(video_path=tmpdir)
            for object_id, box in seed_prompts:
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0,
                    obj_id=object_id, box=box.astype(np.float32),
                )

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
        det_boxes, det_scores, det_labels = _detect_prompt_groups(
            frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
        )
        cdet_boxes, cdet_scores, cdet_labels = (
            _empty_detections()
            if character_disabled
            else _detect_objects_groundingdino(
                frame,
                character_prompt,
                gd_model_id,
                gd_threshold,
                gd_text_threshold,
                device,
            )
        )
        if fixed_anchors:
            canonical_detections = _assign_fixed_anchor_detections(
                boxes=det_boxes,
                scores=det_scores,
                labels=det_labels,
                fixed_anchor_labels=fixed_anchors,
                gt=unity_gt,
                poses=unity_poses,
                intrinsics=unity_intrinsics,
                frame_idx=f_idx,
                frame_width=width,
                frame_height=height,
            )
        else:
            canonical_detections = [
                {
                    "label": det_labels[i],
                    "score": float(det_scores[i]),
                    "box": [float(v) for v in det_boxes[i]],
                }
                for i in range(len(det_boxes))
            ]
        detections_by_frame.append({
            "frame": f_idx,
            "objects": canonical_detections,
            "character": [
                {"label": cdet_labels[i], "score": float(cdet_scores[i])}
                for i in range(len(cdet_boxes))
            ],
        })

        per_obj = masks_per_frame.get(f_idx, {})
        detection_by_label = {entry["label"]: entry for entry in canonical_detections}
        objects_out: List[Dict[str, Any]] = []
        for entry in objects_meta:
            mask = per_obj.get(entry["id"])
            detection = detection_by_label.get(entry["label"])
            if fixed_anchors and not entry["is_character"] and detection is None:
                mask = None
            if mask is None or not mask.any():
                if fixed_anchors and not entry["is_character"] and detection is not None:
                    x1, y1, x2, y2 = detection["box"]
                    objects_out.append(
                        {
                            "id": entry["id"],
                            "label": entry["label"],
                            "is_character": False,
                            "mask_area_px": int(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                            "centroid_uv": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "geometry_source": "groundingdino_box_no_sam_seed",
                        }
                    )
                    continue
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
        "fixed_anchor_labels": fixed_anchors,
        "fixed_anchor_assignment": "nearest_unity_projection_by_semantic_class" if fixed_anchors else None,
    }


# ---------------------------------------------------------------------------
# Candidate-side depth / distance estimation
# ---------------------------------------------------------------------------
_DEPTH_ESTIMATOR_CACHE: Dict[Tuple[str, str], Any] = {}


def _load_depth_estimator(model_id: str, device: str):
    """Load (and cache) a depth-estimation pipeline, mirroring
    shadow_consistency.py's _load_groundingdino cache. Without this,
    _auto_select_depth_model reloads up to 2 full checkpoints per case
    (one per candidate) and analyze_environment_object_consistency_from_
    context reloads a 3rd for the actual estimation -- the same
    "uncached per-call model loading is the actual reason evaluation runs
    took tens of minutes" problem this session already fixed for
    GroundingDINO, just missed one function over in this same file."""
    key = (model_id, device)
    cached = _DEPTH_ESTIMATOR_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model_id, device=(0 if device == "cuda" else -1))
    _DEPTH_ESTIMATOR_CACHE[key] = pipe
    return pipe


def _depth_map_for_frame(depth_pipe: Any, frame_bgr: np.ndarray) -> np.ndarray:
    """Run a loaded depth-estimation pipeline on one BGR video frame,
    returning a depth map resized to the frame's own resolution. Shared by
    every depth-lifting call site (environment_object_consistency.py's
    candidate-side depth/distance estimation and kinematics_mechanics.py's
    metric-velocity lifting) so the resize/units convention can't drift
    between them."""
    from PIL import Image

    image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    depth_out = depth_pipe(image_rgb)
    depth_map = np.asarray(depth_out["predicted_depth"])
    if depth_map.shape != frame_bgr.shape[:2]:
        depth_map = cv2.resize(depth_map, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return depth_map


def _read_frame_verified(cap: Any, f_idx: int) -> Optional[np.ndarray]:
    """Read frame ``f_idx``, verifying the seek actually landed there.

    ``CAP_PROP_POS_FRAMES`` seeking is not guaranteed frame-exact for
    compressed video with a non-trivial keyframe interval -- OpenCV/ffmpeg
    can silently land on the nearest keyframe instead of the requested
    frame, and ``cap.read()`` still reports ``ok=True``. Reading
    ``CAP_PROP_POS_FRAMES`` again *after* the read (not just after the
    seek) reports where the decoder's bookkeeping actually ended up, which
    is a real verification, not an echo of the requested value -- if it
    doesn't match, we know the seek was inexact rather than just hoping it
    wasn't. On mismatch, fall back to a sequential decode from frame 0
    (the one position with no seek ambiguity at all): slower, but always
    exact by construction, since ``cap.read()`` always advances exactly
    one frame regardless of codec. This only pays that cost for the
    specific frames where the fast seek path is actually wrong, not for
    every read -- for a well-formed, densely-keyframed video (true of
    every case captured in this repo) the fast path verifies every time.

    Shared by kinematics_mechanics.py's depth-lifting path and this
    module's _auto_select_depth_model, rather than one of them using a
    raw, unverified seek while the other has the safety net.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    if landed == f_idx:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = None
    for _ in range(f_idx + 1):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def _unproject_pixel(
    u: float, v: float, depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Pinhole-unproject one pixel to a metric 3D point using a depth map
    sampled at that pixel (nearest, clamped to the map bounds)."""
    ui = min(max(int(round(u)), 0), depth_map.shape[1] - 1)
    vi = min(max(int(round(v)), 0), depth_map.shape[0] - 1)
    d = float(depth_map[vi, ui])
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return np.asarray([x, y, d], dtype=np.float64)


def _auto_select_depth_model(
    video_path: str,
    track: Dict[str, Any],
    gt: Dict[str, Any],
    poses: np.ndarray,
    intrinsics: Optional[np.ndarray],
    device: str,
    candidate_model_ids: Sequence[str] = DEPTH_AUTO_SELECT_CANDIDATES,
) -> str:
    """Pick the depth-estimation checkpoint that's empirically most accurate
    for THIS specific case, by testing each candidate against one real,
    known Unity ground-truth depth -- instead of a human guessing indoor
    vs. outdoor per scene and silently getting it wrong by default (see
    DEPTH_AUTO_SELECT_CANDIDATES's docstring). Finds the first (frame,
    label) pair with both a real detected pixel centroid and a real Unity
    3D position at that exact frame, computes the true camera-to-object
    depth there, then measures each candidate model's error at that pixel
    against that real number. Falls back to the first candidate if no
    valid calibration point exists in this case's data (e.g. no tracked
    object has both a real trajectory entry and a detected centroid at
    the same frame) -- this can only happen if depth_absrel_pct itself
    would also come back None -- but this is a real, disclosed
    approximation, not a guarantee: this function requires only a
    NEARBY (not necessarily frame-exact) ground-truth sample for its
    single calibration point (see FRAME_TOLERANCE below), while
    depth_absrel_pct averages each label's GT independently over all its
    own available frames with no such requirement -- so a sparsely-
    sampled GT track can, in principle, still fail calibration even
    though depth_absrel_pct itself succeeds. That case falls back to
    candidate_model_ids[0] rather than silently fabricating a pick."""
    # How many frames away a GT sample may be from the candidate detection
    # it's paired with for calibration. This trades a small amount of
    # position staleness (real for a moving object across that many
    # frames) for a much higher chance of finding a usable calibration
    # point at all when GT is only sampled sparsely -- calibration only
    # needs to be informative about *relative* model accuracy, not
    # pixel-perfect, so this approximation is acceptable here even though
    # it would not be for the actual depth_absrel_pct/pairwise_distance_
    # mae_pct metrics themselves (which use exact per-frame GT).
    FRAME_TOLERANCE = 5
    gt_objects = gt.get("objects", {})
    lower_index = {str(k).strip().lower(): v for k, v in gt_objects.items()}
    cap = cv2.VideoCapture(video_path)
    try:
        for fr in track["frames"]:
            f_idx = fr["frame"]
            for obj in fr["objects"]:
                centroid = obj.get("centroid_uv")
                if not centroid:
                    continue
                entry = lower_index.get(str(obj["label"]).strip().lower())
                if entry is None:
                    continue
                frames_list = entry.get("frames", [])
                if not frames_list:
                    continue
                nearest_gt_frame = min(frames_list, key=lambda gf: abs(gf - f_idx))
                if abs(nearest_gt_frame - f_idx) > FRAME_TOLERANCE:
                    continue
                pos_world = np.asarray(
                    entry["positions_world"][frames_list.index(nearest_gt_frame)],
                    dtype=np.float64,
                )
                cam_pos = _camera_position_at_frame(poses, f_idx)
                calib_depth = float(np.linalg.norm(pos_world - cam_pos))
                if calib_depth <= EPS:
                    continue

                frame = _read_frame_verified(cap, f_idx)
                if frame is None:
                    continue
                u, v = centroid

                best_model, best_rel_err = candidate_model_ids[0], float("inf")
                for model_id in candidate_model_ids:
                    try:
                        depth_map = _depth_map_for_frame(_load_depth_estimator(model_id, device), frame)
                        ui = min(max(int(round(u)), 0), depth_map.shape[1] - 1)
                        vi = min(max(int(round(v)), 0), depth_map.shape[0] - 1)
                        predicted = float(depth_map[vi, ui])
                    except Exception:
                        continue
                    rel_err = abs(predicted - calib_depth) / (calib_depth + EPS)
                    if rel_err < best_rel_err:
                        best_rel_err, best_model = rel_err, model_id
                return best_model
    finally:
        cap.release()
    return candidate_model_ids[0]


def _estimate_depth_via_apparent_size(
    track: Dict[str, Any],
    real_diameter_m: Dict[str, float],
    intrinsics: Optional[np.ndarray],
    frame_shape: Tuple[int, int],
) -> Tuple[Dict[str, List[float]], Dict[str, List[np.ndarray]]]:
    """Classic photogrammetric depth: depth = real_diameter_m * fx /
    apparent_diameter_px, using each object's already-tracked SAM2 mask
    bounding box (``bbox_xyxy``) -- no depth model, no video re-read needed.

    For a known-geometry rigid object (a sphere, in practice) this sidesteps
    monocular depth estimation entirely, which matters because reflective/
    mirror surfaces confuse depth models structurally (they see a reflection
    of the environment, not the object's own surface): empirically ~18-27%
    error here vs. ~79-91% for every monocular depth model tried on
    MirrorRollingBall's chrome sphere (see suite.yaml/PENDING.md).

    ``real_diameter_m`` must be a real physical constant, not something
    re-derived per eval run -- callers should calibrate it once (e.g. from
    a case's known first-frame Unity ground truth) and store it in the
    case's artifacts, the same way object_masses.json stores real,
    Unity-read masses rather than recomputing them live."""
    height, width = frame_shape
    depth_series: Dict[str, List[float]] = {label: [] for label in real_diameter_m}
    point3d_series: Dict[str, List[np.ndarray]] = {label: [] for label in real_diameter_m}
    for fr in track["frames"]:
        f_idx = fr["frame"]
        if intrinsics is not None:
            row = intrinsics[min(f_idx, intrinsics.shape[0] - 1)]
            fx, fy, cx, cy = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        else:
            fx = fy = float(max(width, height))
            cx, cy = width / 2.0, height / 2.0
        for o in fr["objects"]:
            label = o["label"]
            bbox = o.get("bbox_xyxy")
            diam_m = real_diameter_m.get(label)
            if bbox is None or diam_m is None:
                continue
            x1, y1, x2, y2 = bbox
            diam_px = ((x2 - x1) + (y2 - y1)) / 2.0
            if diam_px < EPS:
                continue
            d = diam_m * fx / diam_px
            u, v = o["centroid_uv"]
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            depth_series[label].append(d)
            point3d_series[label].append(np.asarray([x, y, d], dtype=np.float64))
    return depth_series, point3d_series


def _pairwise_distances_from_point3d(
    point3d_series: Dict[str, List[np.ndarray]], labels: Sequence[str]
) -> Dict[Tuple[str, str], float]:
    """Mean pairwise 3D distance between every pair of labels with a
    non-empty point3d series, paired positionally (k-th point of label A
    against k-th point of label B) -- shared by the monocular-depth path
    and the caller merging in apparent-size points for specific labels."""
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
    return distance_by_pair


def _estimate_candidate_depths_and_distances(
    video_path: str,
    track: Dict[str, Any],
    depth_model_id: str,
    device: str,
    intrinsics: Optional[np.ndarray],
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[str, List[np.ndarray]]]:
    """Monocular depth estimate at each object's mask centroid (metric units
    if a metric depth model is configured -- see DEPTH_DEFAULT_MODEL),
    back-projected with pinhole intrinsics into candidate-side pairwise
    3D distances. Also returns the raw per-label point3d series so a caller
    can substitute a different depth source (e.g. apparent-size) for
    specific labels and recompute pairwise distances consistently."""
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

        depth_map = _depth_map_for_frame(depth_pipe, frame)

        if intrinsics is not None:
            row = intrinsics[min(f_idx, intrinsics.shape[0] - 1)]
            fx, fy, cx, cy = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        else:
            fx = fy = float(max(frame.shape[1], frame.shape[0]))
            cx, cy = frame.shape[1] / 2.0, frame.shape[0] / 2.0

        for o in objects_here:
            u, v = o["centroid_uv"]
            point3d = _unproject_pixel(u, v, depth_map, fx, fy, cx, cy)
            depth_series[o["label"]].append(float(point3d[2]))
            point3d_series[o["label"]].append(point3d)
    cap.release()

    depth_by_label = {
        label: float(np.mean(vals)) for label, vals in depth_series.items() if vals
    }
    distance_by_pair = _pairwise_distances_from_point3d(point3d_series, labels)
    return depth_by_label, distance_by_pair, point3d_series


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
    fixed_anchor_value = artifacts.get("eoc_fixed_anchor_labels") or []
    if isinstance(fixed_anchor_value, str):
        fixed_anchor_labels = [
            item.strip() for item in fixed_anchor_value.split("|") if item.strip()
        ]
    elif isinstance(fixed_anchor_value, (list, tuple)):
        fixed_anchor_labels = [str(item).strip() for item in fixed_anchor_value if str(item).strip()]
    else:
        return {
            "module": "environment_object_consistency",
            "error": "eoc_fixed_anchor_labels must be a list or pipe-delimited string",
        }
    depth_model_override = artifacts.get("eoc_depth_model")
    diameters_path = artifacts.get("eoc_unity_object_diameters_json")
    real_diameter_m: Dict[str, float] = {}
    diameter_artifact_error: Optional[str] = None
    if diameters_path:
        try:
            real_diameter_m = _load_object_diameters(str(diameters_path))
        except Exception as exc:
            diameter_artifact_error = str(exc)

    video_path = context.candidate_video
    if not video_path:
        return {"module": "environment_object_consistency", "error": "candidate_video is required"}

    assignment_gt = None
    assignment_poses = None
    assignment_intrinsics = None
    if fixed_anchor_labels:
        try:
            assignment_gt = _load_unity_object_transforms(
                str(artifacts.get("eoc_unity_object_transforms_json") or "")
            )
            assignment_poses = _load_camera_poses(
                str(artifacts.get("eoc_unity_poses_npy") or "")
            )
            assignment_intrinsics = _load_camera_intrinsics(
                str(artifacts.get("eoc_unity_intrinsics_npy") or "")
            )
        except Exception as exc:
            return {
                "module": "environment_object_consistency",
                "error": f"failed to load fixed-anchor assignment artifacts: {exc}",
            }

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
        fixed_anchor_labels=fixed_anchor_labels,
        unity_gt=assignment_gt,
        unity_poses=assignment_poses,
        unity_intrinsics=assignment_intrinsics,
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
        "fixed_anchor_labels": fixed_anchor_labels,
        "fixed_anchor_assignment": track.get("fixed_anchor_assignment"),
    }
    if diameter_artifact_error is not None:
        result["diameter_artifact_error"] = diameter_artifact_error

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

    if depth_model_override:
        depth_model_id = str(depth_model_override)
        result["depth_model_selection"] = "explicit_override"
    else:
        depth_model_id = _auto_select_depth_model(video_path, track, gt, poses, intrinsics, device)
        result["depth_model_selection"] = "auto"
    result["depth_model_used"] = depth_model_id

    try:
        cand_depth, cand_distance, cand_point3d = _estimate_candidate_depths_and_distances(
            video_path, track, depth_model_id, device, intrinsics
        )
    except Exception as exc:
        result["ground_truth_status"] = "error"
        result["ground_truth_note"] = f"candidate depth estimation failed: {exc}"
        result["depth_absrel_pct"] = None
        result["pairwise_distance_mae_pct"] = None
        return result

    if real_diameter_m:
        # Apparent-size depth (photogrammetric, no depth model) overrides
        # the monocular estimate for any label with a known real diameter --
        # see _estimate_depth_via_apparent_size's docstring for why (reflective
        # objects like MirrorRollingBall's chrome sphere confuse monocular
        # depth models structurally; apparent size does not depend on
        # surface appearance). Overrides depth AND the 3D points feeding
        # pairwise_distance_mae_pct -- using apparent-size depth for one and
        # leaving the other on the (worse, for these objects) monocular
        # estimate would silently desync the two reported metrics.
        size_depth_series, size_point3d_series = _estimate_depth_via_apparent_size(
            track, real_diameter_m, intrinsics, (track["height"], track["width"])
        )
        for label, vals in size_depth_series.items():
            if vals:
                cand_depth[label] = float(np.mean(vals))
                cand_point3d[label] = size_point3d_series[label]
        result["apparent_size_depth_labels"] = [
            label for label, vals in size_depth_series.items() if vals
        ]
        cand_distance = _pairwise_distances_from_point3d(
            cand_point3d, [o["label"] for o in track["objects"]]
        )

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
