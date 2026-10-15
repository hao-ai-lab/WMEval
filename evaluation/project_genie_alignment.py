"""Align Project Genie's downloaded video with browser action observations.

The hosted UI exposes a clean downloaded video, but its timeline is not tied to
the browser clock.  This module establishes that tie from multiple ordered SIFT
correspondences, validates the mapping conservatively, and trims the exact
native frames that correspond to the semantic browser trajectory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class NativeAlignmentError(RuntimeError):
    """The native video cannot be certified against the observed trajectory."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class DescribedFrame:
    frame_index: int
    descriptors: Any


def _crop_gray(image: Any, *, observed: bool) -> Any:
    height, width = image.shape[:2]
    if observed:
        # Remove the progress bar, controller overlays, watermark, and edge UI.
        return image[
            int(0.09 * height) : int(0.78 * height),
            int(0.05 * width) : int(0.95 * width),
        ]
    # The downloaded video has only a small lower watermark; retain most of its
    # wider field of view so local features can find their geometric matches.
    return image[
        int(0.02 * height) : int(0.92 * height),
        int(0.02 * width) : int(0.98 * width),
    ]


def _describe(sift: Any, image: Any, *, observed: bool) -> Any:
    crop = _crop_gray(image, observed=observed)
    _, descriptors = sift.detectAndCompute(crop, None)
    return descriptors


def _score(matcher: Any, query: Any, candidate: Any) -> float:
    if query is None or candidate is None or len(query) < 2 or len(candidate) < 2:
        return 0.0
    pairs = matcher.knnMatch(query, candidate, k=2)
    if not pairs:
        return 0.0
    good = sum(
        len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
        for pair in pairs
    )
    return float(good / len(query))


def load_semantic_endpoint(action_trace: Path) -> float:
    raw = json.loads(action_trace.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise NativeAlignmentError(f"{action_trace}: expected an action list")
    returns = [
        row
        for row in raw
        if isinstance(row, dict) and row.get("visual_return_status") == "validated"
    ]
    if len(returns) != 1:
        raise NativeAlignmentError(
            f"{action_trace}: expected exactly one validated visual return"
        )
    row = returns[0]
    explicit = row.get("visual_return_global_elapsed_sec")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        endpoint = float(explicit)
    else:
        accepted = row.get("accepted_at_sec")
        turn_elapsed = row.get("visual_return_elapsed_sec")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (accepted, turn_elapsed)
        ):
            raise NativeAlignmentError(
                f"{action_trace}: visual return lacks a global or derivable endpoint"
            )
        endpoint = float(accepted) + float(turn_elapsed)
    if endpoint <= 0:
        raise NativeAlignmentError(f"{action_trace}: semantic endpoint must be positive")
    return endpoint


def load_action_trace_endpoint(action_trace: Path) -> float:
    """Return the completed endpoint of a fixed-duration/calibrated replay."""
    raw = json.loads(action_trace.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise NativeAlignmentError(f"{action_trace}: expected a non-empty action list")
    last = raw[-1]
    endpoint = last.get("completed_at_sec") if isinstance(last, dict) else None
    if not isinstance(endpoint, (int, float)) or isinstance(endpoint, bool) or endpoint <= 0:
        raise NativeAlignmentError(
            f"{action_trace}: final action lacks a positive completed_at_sec"
        )
    return float(endpoint)


def select_observed_rows(
    manifest: Path, semantic_endpoint_sec: float, sample_count: int,
    *, eligible_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    if sample_count < 5:
        raise ValueError("sample_count must be at least five")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < sample_count:
        raise NativeAlignmentError(f"{manifest}: insufficient observed frames")
    usable: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "path" not in row or "elapsed_sec" not in row:
            raise NativeAlignmentError(f"{manifest}: malformed frame {index}")
        path = Path(str(row["path"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if eligible_indices is None or index in eligible_indices:
            usable.append({**row, "index": row.get("index", index),
                           "path": str(path), "elapsed_sec": float(row["elapsed_sec"])})
    if len(usable) < sample_count:
        raise NativeAlignmentError(f"{manifest}: insufficient eligible observed frames")
    elapsed = np.asarray([row["elapsed_sec"] for row in usable], dtype=np.float64)
    if np.any(np.diff(elapsed) <= 0):
        raise NativeAlignmentError(f"{manifest}: frame times are not strictly increasing")
    if elapsed[0] > 0.05 or elapsed[-1] + 0.05 < semantic_endpoint_sec:
        raise NativeAlignmentError(
            f"{manifest}: observed capture does not span semantic endpoint "
            f"{semantic_endpoint_sec:.6f}s"
        )
    targets = np.linspace(0.0, semantic_endpoint_sec, sample_count)
    # Select distinct ordered checkpoints even when the capture is sparse.
    # Independent nearest neighbours can select the same frame twice.
    costs = np.full((sample_count, len(usable)), np.inf)
    backs = np.full(costs.shape, -1, dtype=int)
    costs[0, elapsed <= .05] = np.abs(elapsed[elapsed <= .05])
    for sample in range(1, sample_count):
        for index in range(sample, len(usable)):
            if sample == sample_count - 1 and elapsed[index] + .05 < semantic_endpoint_sec:
                continue
            previous = int(np.argmin(costs[sample - 1, :index]))
            costs[sample, index] = costs[sample - 1, previous] + abs(elapsed[index] - targets[sample])
            backs[sample, index] = previous
    last = int(np.argmin(costs[-1]))
    if not np.isfinite(costs[-1, last]):
        raise NativeAlignmentError(f"{manifest}: no distinct checkpoints spanning the endpoint")
    indices = [last]
    for sample in range(sample_count - 1, 0, -1):
        indices.append(int(backs[sample, indices[-1]]))
    indices.reverse()
    return [usable[index] for index in indices]


def _video_info(capture: Any, cv2: Any) -> VideoInfo:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count < 2 or width <= 0 or height <= 0:
        raise NativeAlignmentError("native video metadata is invalid")
    return VideoInfo(fps=fps, frame_count=frame_count, width=width, height=height)


def _native_descriptors(
    native: Path, sift: Any, *, sample_fps: float
) -> tuple[VideoInfo, list[DescribedFrame], int]:
    import cv2

    capture = cv2.VideoCapture(str(native))
    if not capture.isOpened():
        raise NativeAlignmentError(f"cannot open native video: {native}")
    info = _video_info(capture, cv2)
    stride = max(1, int(round(info.fps / sample_fps)))
    described: list[DescribedFrame] = []
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            described.append(
                DescribedFrame(frame_index, _describe(sift, gray, observed=False))
            )
        frame_index += 1
    capture.release()
    if frame_index < info.frame_count - 1:
        raise NativeAlignmentError(
            f"native video decoded only {frame_index} of {info.frame_count} frames"
        )
    if len(described) < 10:
        raise NativeAlignmentError("native video produced too few alignment samples")
    return info, described, stride


def _refine_matches(
    native: Path,
    sift: Any,
    matcher: Any,
    queries: Sequence[Any],
    coarse_frames: Sequence[int],
    radius: int,
    observed_times: Sequence[float],
    fps: float,
) -> tuple[list[int], list[float]]:
    import cv2

    capture = cv2.VideoCapture(str(native))
    if not capture.isOpened():
        raise NativeAlignmentError(f"cannot reopen native video: {native}")
    candidate_rows: list[list[int]] = []
    score_rows: list[list[float]] = []
    try:
        for query, coarse in zip(queries, coarse_frames):
            start = max(0, coarse - radius)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            row_frames: list[int] = []
            row_scores: list[float] = []
            for frame_index in range(start, coarse + radius + 1):
                ok, image = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                row_frames.append(frame_index)
                row_scores.append(
                    _score(matcher, query, _describe(sift, gray, observed=False))
                )
            if not row_frames:
                raise NativeAlignmentError(
                    f"could not decode refinement window around frame {coarse}"
                )
            candidate_rows.append(row_frames)
            score_rows.append(row_scores)
    finally:
        capture.release()
    selected = ordered_sparse_path(
        score_rows,
        candidate_rows,
        observed_times,
        fps,
    )
    return (
        [candidate_rows[row][index] for row, index in enumerate(selected)],
        [score_rows[row][index] for row, index in enumerate(selected)],
    )


def ordered_sparse_path(
    score_rows: Sequence[Sequence[float]],
    candidate_frame_rows: Sequence[Sequence[int]],
    observed_times: Sequence[float],
    fps: float,
    *,
    minimum_speed_ratio: float = 0.15,
    maximum_speed_ratio: float = 2.0,
) -> list[int]:
    """Choose one frame per local window while preserving plausible chronology."""
    observed = np.asarray(observed_times, dtype=np.float64)
    if not (
        len(score_rows) == len(candidate_frame_rows) == len(observed)
        and len(observed) >= 2
    ):
        raise ValueError("refinement rows and observations must have equal length")
    if fps <= 0 or np.any(np.diff(observed) <= 0):
        raise ValueError("refinement needs positive fps and increasing observations")
    if not 0 < minimum_speed_ratio < maximum_speed_ratio:
        raise ValueError("invalid native/observed speed-ratio bounds")

    frames = [np.asarray(row, dtype=np.int64) for row in candidate_frame_rows]
    values = [np.asarray(row, dtype=np.float64) for row in score_rows]
    for row_frames, row_values in zip(frames, values):
        if (
            len(row_frames) == 0
            or len(row_frames) != len(row_values)
            or np.any(np.diff(row_frames) <= 0)
        ):
            raise ValueError("each refinement row needs matching, increasing candidates")

    totals = [np.full(len(row), -np.inf, dtype=np.float64) for row in values]
    backs = [np.full(len(row), -1, dtype=np.int32) for row in values]
    totals[0] = values[0].copy()
    for row_index in range(1, len(observed)):
        observed_delta = observed[row_index] - observed[row_index - 1]
        minimum_delta = max(
            1, int(np.floor(minimum_speed_ratio * observed_delta * fps))
        )
        maximum_delta = max(
            minimum_delta,
            int(np.ceil(maximum_speed_ratio * observed_delta * fps)),
        )
        for candidate_index, frame in enumerate(frames[row_index]):
            deltas = frame - frames[row_index - 1]
            valid = np.flatnonzero(
                (deltas >= minimum_delta) & (deltas <= maximum_delta)
            )
            if valid.size == 0:
                continue
            previous = totals[row_index - 1][valid]
            winner_offset = int(np.argmax(previous))
            winner = int(valid[winner_offset])
            if not np.isfinite(previous[winner_offset]):
                continue
            totals[row_index][candidate_index] = (
                previous[winner_offset] + values[row_index][candidate_index]
            )
            backs[row_index][candidate_index] = winner

    final = int(np.argmax(totals[-1]))
    if not np.isfinite(totals[-1][final]):
        raise NativeAlignmentError("no chronologically plausible refined alignment path")
    path = [final]
    for row_index in range(len(observed) - 1, 0, -1):
        previous = int(backs[row_index][path[-1]])
        if previous < 0:
            raise NativeAlignmentError("refined native alignment path is incomplete")
        path.append(previous)
    path.reverse()
    return path


def ordered_coarse_path(
    similarity: np.ndarray,
    candidate_frames: Sequence[int],
    observed_times: Sequence[float],
    fps: float,
    *,
    minimum_speed_ratio: float = 0.15,
    maximum_speed_ratio: float = 2.0,
) -> list[int]:
    """Select the strongest chronologically coherent native match sequence.

    A single view near the beginning/end of a circular trajectory is often
    ambiguous.  Selecting each nearest neighbour independently can therefore
    map an early observation to the later revisit.  Dynamic programming uses
    all checkpoints jointly and permits only forward progress whose local
    native/observed time ratio is physically plausible.
    """
    matrix = np.asarray(similarity, dtype=np.float64)
    frames = np.asarray(candidate_frames, dtype=np.int64)
    observed = np.asarray(observed_times, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(observed), len(frames)):
        raise ValueError("similarity dimensions do not match frames and observations")
    if len(observed) < 2 or len(frames) < 2 or fps <= 0:
        raise ValueError("ordered alignment needs multiple samples and positive fps")
    if np.any(np.diff(frames) <= 0) or np.any(np.diff(observed) <= 0):
        raise ValueError("candidate frames and observed times must be strictly increasing")
    if not 0 < minimum_speed_ratio < maximum_speed_ratio:
        raise ValueError("invalid native/observed speed-ratio bounds")

    scores = np.full(matrix.shape, -np.inf, dtype=np.float64)
    back = np.full(matrix.shape, -1, dtype=np.int32)
    scores[0] = matrix[0]
    for query_index in range(1, len(observed)):
        observed_delta = observed[query_index] - observed[query_index - 1]
        minimum_delta = max(1, int(np.floor(minimum_speed_ratio * observed_delta * fps)))
        maximum_delta = max(
            minimum_delta,
            int(np.ceil(maximum_speed_ratio * observed_delta * fps)),
        )
        for candidate_index, frame in enumerate(frames):
            deltas = frame - frames[:candidate_index]
            valid = np.flatnonzero(
                (deltas >= minimum_delta) & (deltas <= maximum_delta)
            )
            if valid.size == 0:
                continue
            previous_scores = scores[query_index - 1, valid]
            winner_offset = int(np.argmax(previous_scores))
            winner = int(valid[winner_offset])
            if not np.isfinite(previous_scores[winner_offset]):
                continue
            scores[query_index, candidate_index] = (
                previous_scores[winner_offset] + matrix[query_index, candidate_index]
            )
            back[query_index, candidate_index] = winner
    final = int(np.argmax(scores[-1]))
    if not np.isfinite(scores[-1, final]):
        raise NativeAlignmentError("no chronologically plausible native alignment path")
    path = [final]
    for query_index in range(len(observed) - 1, 0, -1):
        previous = int(back[query_index, path[-1]])
        if previous < 0:
            raise NativeAlignmentError("native alignment path is incomplete")
        path.append(previous)
    path.reverse()
    return path


def validate_alignment(
    observed_times: Sequence[float],
    native_times: Sequence[float],
    scores: Sequence[float],
    alternative_scores: Sequence[float],
    *,
    min_match: float = 0.10,
    min_margin: float = 0.02,
    minimum_local_speed_ratio: float = 0.10,
    maximum_local_speed_ratio: float = 2.0,
) -> tuple[str, list[str], dict[str, float | int | bool]]:
    observed = np.asarray(observed_times, dtype=np.float64)
    native = np.asarray(native_times, dtype=np.float64)
    match = np.asarray(scores, dtype=np.float64)
    alternatives = np.asarray(alternative_scores, dtype=np.float64)
    if not (
        len(observed) == len(native) == len(match) == len(alternatives) and len(observed) >= 5
    ):
        raise ValueError("alignment arrays must have the same length of at least five")
    slope, intercept = np.polyfit(observed, native, 1)
    fitted = slope * observed + intercept
    residual = float(np.sum((native - fitted) ** 2))
    total = float(np.sum((native - native.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    margins = match - alternatives
    increasing = bool(np.all(np.diff(native) > 0))
    matched_count = int(np.sum(match >= min_match))
    useful_margin_count = int(np.sum(margins >= min_margin))
    observed_span = float(observed[-1] - observed[0])
    native_span = float(native[-1] - native[0])
    span_ratio = native_span / observed_span if observed_span > 0 else 0.0
    local_speed_ratios = np.diff(native) / np.diff(observed)
    reasons: list[str] = []
    if not increasing:
        reasons.append("native matches are not strictly chronological")
    if matched_count != len(match):
        reasons.append(
            f"only {matched_count}/{len(match)} matches meet {min_match:.3f}"
        )
    if useful_margin_count < len(match) - 1:
        reasons.append(
            f"only {useful_margin_count}/{len(match)} matches have margin >= {min_margin:.3f}"
        )
    if np.any(local_speed_ratios < minimum_local_speed_ratio) or np.any(
        local_speed_ratios > maximum_local_speed_ratio
    ):
        reasons.append(
            "native/browser local speed ratio outside "
            f"[{minimum_local_speed_ratio:.3f}, {maximum_local_speed_ratio:.3f}]"
        )
    if not 0.35 <= span_ratio <= 2.0:
        reasons.append(f"native/observed duration ratio {span_ratio:.4f} is implausible")
    if match[0] < min_match or match[-1] < min_match:
        reasons.append("semantic start or endpoint lacks a strong native match")
    metrics: dict[str, float | int | bool] = {
        "sample_count": len(match),
        "strictly_increasing": increasing,
        "matched_sample_count": matched_count,
        "useful_margin_sample_count": useful_margin_count,
        "minimum_match": float(match.min()),
        "median_match": float(np.median(match)),
        "minimum_margin": float(margins.min()),
        "median_margin": float(np.median(margins)),
        "timeline_slope": float(slope),
        "timeline_intercept_sec": float(intercept),
        "timeline_r_squared": float(r_squared),
        "minimum_local_speed_ratio": float(local_speed_ratios.min()),
        "maximum_local_speed_ratio": float(local_speed_ratios.max()),
        "native_span_sec": native_span,
        "observed_span_sec": observed_span,
        "native_observed_span_ratio": float(span_ratio),
    }
    return ("validated" if not reasons else "insufficient_evidence"), reasons, metrics


def _trim_video(native: Path, output: Path, start_frame: int, end_frame: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to trim the native trajectory")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(native),
            "-vf",
            f"trim=start_frame={start_frame}:end_frame={end_frame + 1},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def align_native_video(
    native: Path,
    observed_manifest: Path,
    action_trace: Path,
    output_dir: Path,
    *,
    sample_count: int = 9,
    native_sample_fps: float = 4.0,
    endpoint_mode: str = "closed_loop_visual_return",
    allow_feature_sparse_capture: bool = False,
) -> dict[str, Any]:
    import cv2

    native = native.resolve()
    observed_manifest = observed_manifest.resolve()
    action_trace = action_trace.resolve()
    output_dir = output_dir.resolve()
    for path in (native, observed_manifest, action_trace):
        if not path.is_file():
            raise FileNotFoundError(path)
    if native_sample_fps <= 0:
        raise ValueError("native_sample_fps must be positive")
    if endpoint_mode == "closed_loop_visual_return":
        semantic_endpoint = load_semantic_endpoint(action_trace)
        claim_scope = "native_video_corresponds_to_closed_loop_browser_trajectory"
    elif endpoint_mode == "action_trace_end":
        semantic_endpoint = load_action_trace_endpoint(action_trace)
        claim_scope = "native_video_corresponds_to_fixed_or_calibrated_browser_trajectory"
    else:
        raise ValueError(f"unsupported endpoint_mode: {endpoint_mode}")
    sift = cv2.SIFT_create(nfeatures=1200)
    matcher = cv2.BFMatcher()
    eligibility = []
    eligible_indices = None
    if allow_feature_sparse_capture:
        # Eligibility depends only on the observed image's feature count,
        # never on how well it matches the generated video. Keep the original
        # manifest unchanged, report every exclusion, and retain all existing
        # correspondence/chronology/margin requirements and both endpoints.
        payload = json.loads(observed_manifest.read_text(encoding="utf-8"))
        eligible_indices = set()
        for index, row in enumerate(payload["frames"]):
            image = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise NativeAlignmentError(f"cannot read observed frame: {row['path']}")
            descriptors = _describe(sift, image, observed=True)
            count = 0 if descriptors is None else len(descriptors)
            eligibility.append({"index": index, "feature_count": count,
                                "eligible": count >= 100})
            if count >= 100:
                eligible_indices.add(index)
    observed_rows = select_observed_rows(
        observed_manifest, semantic_endpoint, sample_count,
        eligible_indices=eligible_indices,
    )
    queries: list[Any] = []
    for row in observed_rows:
        image = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise NativeAlignmentError(f"cannot read observed frame: {row['path']}")
        descriptors = _describe(sift, image, observed=True)
        if descriptors is None or len(descriptors) < 100:
            raise NativeAlignmentError(
                f"observed frame has too few features: {row['path']}"
            )
        queries.append(descriptors)
    info, candidates, stride = _native_descriptors(
        native, sift, sample_fps=native_sample_fps
    )
    similarity = np.asarray(
        [
            [_score(matcher, query, candidate.descriptors) for candidate in candidates]
            for query in queries
        ],
        dtype=np.float64,
    )
    observed_times = [float(row["elapsed_sec"]) for row in observed_rows]
    coarse_indices = ordered_coarse_path(
        similarity,
        [candidate.frame_index for candidate in candidates],
        observed_times,
        info.fps,
    )
    coarse_frames = [candidates[int(index)].frame_index for index in coarse_indices]
    refined_frames, refined_scores = _refine_matches(
        native,
        sift,
        matcher,
        queries,
        coarse_frames,
        radius=stride,
        observed_times=observed_times,
        fps=info.fps,
    )
    exclusion_frames = max(stride + 1, int(round(info.fps)))
    alternative_scores: list[float] = []
    for row_index, best_frame in enumerate(refined_frames):
        alternatives = [
            float(similarity[row_index, candidate_index])
            for candidate_index, candidate in enumerate(candidates)
            if abs(candidate.frame_index - best_frame) > exclusion_frames
        ]
        alternative_scores.append(max(alternatives, default=0.0))
    native_times = [frame / info.fps for frame in refined_frames]
    status, reasons, metrics = validate_alignment(
        observed_times, native_times, refined_scores, alternative_scores
    )
    matches = [
        {
            "sample_index": index,
            "observed_frame_index": int(row["index"]),
            "observed_elapsed_sec": float(row["elapsed_sec"]),
            "observed_sha256": str(row.get("sha256") or sha256(Path(row["path"]))),
            "native_frame_index": int(native_frame),
            "native_elapsed_sec": float(native_time),
            "sift_match": float(score),
            "alternative_match_outside_1s": float(alternative),
            "match_margin": float(score - alternative),
        }
        for index, (row, native_frame, native_time, score, alternative) in enumerate(
            zip(
                observed_rows,
                refined_frames,
                native_times,
                refined_scores,
                alternative_scores,
            )
        )
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "claim_scope": claim_scope,
        "endpoint_mode": endpoint_mode,
        "method": "ordered_multi_checkpoint_sift_v2",
        "validation_reasons": reasons,
        "native_video": str(native),
        "native_video_sha256": sha256(native),
        "native_video_info": {
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration_sec": info.duration_sec,
            "coarse_sample_stride_frames": stride,
        },
        "observed_manifest": str(observed_manifest),
        "observed_manifest_sha256": sha256(observed_manifest),
        "action_trace": str(action_trace),
        "action_trace_sha256": sha256(action_trace),
        "semantic_endpoint_sec": semantic_endpoint,
        "metrics": metrics,
        "matches": matches,
        "observed_feature_eligibility": eligibility,
    }
    report_path = output_dir / "native_alignment.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if status == "validated":
        start_frame = int(refined_frames[0])
        end_frame = int(refined_frames[-1])
        if end_frame <= start_frame:
            raise NativeAlignmentError("validated alignment has a non-positive frame span")
        output_video = output_dir / "trajectory_native.mp4"
        _trim_video(native, output_video, start_frame, end_frame)
        report["trajectory_native"] = str(output_video)
        report["trajectory_native_sha256"] = sha256(output_video)
        report["trajectory_native_start_frame"] = start_frame
        report["trajectory_native_end_frame_inclusive"] = end_frame
        report["trajectory_native_frame_count"] = end_frame - start_frame + 1
        report["trajectory_native_duration_sec"] = (end_frame - start_frame + 1) / info.fps
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "validated":
        raise NativeAlignmentError(
            "native timeline alignment did not validate: " + "; ".join(reasons)
        )
    report["report"] = str(report_path)
    report["report_sha256"] = sha256(report_path)
    return report


def alignment_receipt_fields(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "validated":
        raise NativeAlignmentError("refusing to use an unvalidated alignment report")
    return {
        "native_video_timeline_alignment": "validated_visual_correspondence",
        "native_alignment_report": report["report"],
        "native_alignment_report_sha256": report["report_sha256"],
        "evaluation_video": report["trajectory_native"],
        "evaluation_video_sha256": report["trajectory_native_sha256"],
        "evaluation_video_frame_count": report["trajectory_native_frame_count"],
        "evaluation_video_duration_sec": report["trajectory_native_duration_sec"],
        "evaluation_artifact_status": "native_action_aligned_ready",
    }


def update_receipt_with_alignment(receipt_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    fields = alignment_receipt_fields(report)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("native_video_sha256") not in {
        None,
        report.get("native_video_sha256"),
    }:
        raise NativeAlignmentError("receipt and alignment report name different native videos")
    if receipt.get("action_trace_sha256") not in {
        None,
        report.get("action_trace_sha256"),
    }:
        raise NativeAlignmentError("receipt and alignment report name different action traces")
    if receipt.get("observed_frames_sha256") not in {
        None,
        report.get("observed_manifest_sha256"),
    }:
        raise NativeAlignmentError("receipt and alignment report name different observed frames")
    receipt.update(
        {
            "native_video": report["native_video"],
            "native_video_sha256": report["native_video_sha256"],
            "action_trace": report["action_trace"],
            "action_trace_sha256": report["action_trace_sha256"],
            "observed_frames": report["observed_manifest"],
            "observed_frames_sha256": report["observed_manifest_sha256"],
        }
    )
    receipt.update(fields)
    receipt.update(
        {
            "status": "complete",
            "stage": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "native_video_status": "downloaded",
            "native_video_error": None,
        }
    )
    receipt.pop("error", None)
    receipt.pop("visible_controls", None)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
