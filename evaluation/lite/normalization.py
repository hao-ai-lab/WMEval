from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

# Single source of truth for the locked LiquidContainer normalization geometry.
# These mirror the v1 evaluator lock (`832 x 464 @ 16 fps`); the future
# execution layer should cross-check them against each case's protocol/lock
# rather than re-hardcoding the literals.
LIQUID_NORM_WIDTH = 832
LIQUID_NORM_HEIGHT = 464
LIQUID_NORM_FPS = 16

# ffmpeg can hang on malformed inputs; bound every normalization call.
DEFAULT_NORMALIZATION_TIMEOUT_SEC = 600.0
LIGHTING_NORM_FPS = 12
# Candidate params that only lighting-orbit normalization consumes (never runner kwargs).
LIGHTING_ZERO_FRAME_PARAM = "lighting_canonical_zero_frame"
LIGHTING_NORMALIZATION_PARAMS = frozenset({"lighting_time_mapping", LIGHTING_ZERO_FRAME_PARAM})


class NormalizationError(RuntimeError):
    """Candidate video normalization (ffmpeg) failed."""


def derive_lighting_time_mapping(
    source: str | Path, *, native_fps: float | None = None, canonical_zero_frame: int = 0,
    timeout_sec: float = DEFAULT_NORMALIZATION_TIMEOUT_SEC,
) -> dict:
    """Time mapping for a candidate without a generation receipt.

    ``native_fps`` defaults to the container's frame rate and
    ``canonical_zero_frame`` to the first frame; callers pass the model's
    declared values (``candidate_fps_override`` / ``lighting_canonical_zero_frame``)
    when they differ. normalize_lighting_window() still enforces a constant
    frame rate and ten full seconds.
    """
    fps = native_fps if native_fps is not None else _probe(source, timeout_sec=timeout_sec)["fps"]
    if abs(fps - round(fps)) < 1e-9:
        fps = int(round(fps))
    with Path(source).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"schema_version": 1, "native_fps": fps, "canonical_zero_frame": canonical_zero_frame,
            "video_sha256": digest}


def lighting_window_native_frames(native_fps: float) -> int:
    """Native frames whose timestamps n / native_fps fall inside [0, 10) s."""
    return math.ceil(10 * native_fps - 1e-6)


def normalize_lighting_window(
    source: str | Path, destination: str | Path, *, time_mapping: dict,
    timeout_sec: float = DEFAULT_NORMALIZATION_TIMEOUT_SEC,
) -> dict[str, float]:
    """Verified native timeline -> [0,10) at 12 fps, without playback stretch.

    The time mapping (generation receipt, or derived) supplies the native frame
    of the orbit's t=0 and the model's native fps. Native frame n is placed at
    (n - canonical_zero_frame) / native_fps; when the container is tagged with
    a different constant rate it is re-timed to native_fps, never stretched to
    fit. Reject short/VFR inputs before ffmpeg can conceal missing time by padding.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise NormalizationError("normalization must not overwrite the source video")
    if time_mapping.get("schema_version") != 1:
        raise NormalizationError("lighting_time_mapping schema_version must be 1")
    offset, fps = time_mapping.get("canonical_zero_frame"), time_mapping.get("native_fps")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise NormalizationError("canonical_zero_frame must be a nonnegative integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise NormalizationError("native_fps must be finite and positive")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if time_mapping.get("video_sha256") != digest:
        raise NormalizationError("lighting time mapping does not match candidate video hash")
    current = _probe(source, timeout_sec=timeout_sec)
    retime = abs(current["fps"] - fps) > 1e-6
    required = offset + lighting_window_native_frames(fps)
    if current["frames"] < required:
        raise NormalizationError(
            f"short lighting video: {current['frames']} native frames, need {required} "
            f"({offset} before the orbit + 10 s at {fps:g} fps)")
    if retime:
        # Frame index -> native time; the container's own rate is only checked for CFR.
        native_rate = Fraction(fps).limit_denominator(100000)
        timing = f"setpts=N/({native_rate.numerator}/{native_rate.denominator}*TB)"
    else:
        timing = "setpts=PTS-STARTPTS"
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                 "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)],
                                check=True, capture_output=True, text=True, timeout=timeout_sec)
        frames = json.loads(result.stdout)["frames"]
        pts = [float(f["best_effort_timestamp_time"]) for f in frames[:required]]
        step = 1 / current["fps"]
        if len(pts) != required or any(abs((p - pts[0]) - i * step) > 2e-5 for i, p in enumerate(pts)):
            raise NormalizationError("lighting receipt requires a constant-rate native timeline")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                        "-vf", f"trim=start_frame={offset}:end_frame={required},{timing},"
                        f"fps=fps={LIGHTING_NORM_FPS}:start_time=0:round=near",
                        "-frames:v", "120", "-an", "-c:v", "libx264", "-crf", "0",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
                       check=True, capture_output=True, text=True, timeout=timeout_sec)
        actual = _probe(destination, timeout_sec=timeout_sec)
    except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise NormalizationError(f"lighting window normalization failed: {exc}") from exc
    if actual != {**current, "fps": 12.0, "frames": 120}:
        raise NormalizationError(f"lighting window has unexpected geometry/timeline: {actual}")
    return {**actual, "source_fps": current["fps"], "retimed": retime}


def normalize_lighting_video(
    source: str | Path, destination: str | Path, *,
    timeout_sec: float = DEFAULT_NORMALIZATION_TIMEOUT_SEC,
) -> dict[str, float]:
    """Sample at 12 fps using timestamps, preserving size and playback speed.

    Apply BEFORE GroundingDINO/SAM tracking. A rounded frame stride does not
    normalize 16/17/20-fps inputs to 12 fps. Lossless encoding avoids adding
    another lossy compression pass to shadow-mask measurements.
    """
    destination = Path(destination)
    if Path(source).resolve() == destination.resolve():
        raise NormalizationError("normalization must not overwrite the source video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-vf", f"setpts=PTS-STARTPTS,fps=fps={LIGHTING_NORM_FPS}:start_time=0:round=near",
               "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_sec)
        actual = _probe(destination, timeout_sec=timeout_sec)
    except FileNotFoundError as exc:
        raise NormalizationError("ffmpeg/ffprobe executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise NormalizationError(f"ffmpeg/ffprobe timed out normalizing {source}") from exc
    except subprocess.CalledProcessError as exc:
        raise NormalizationError(f"lighting normalization failed: {(exc.stderr or '').strip()}") from exc
    if abs(actual["fps"] - LIGHTING_NORM_FPS) > 1e-6:
        raise NormalizationError(f"expected {LIGHTING_NORM_FPS} fps, got {actual['fps']}")
    return actual


def _scale_fps_filter() -> str:
    return (
        f"scale={LIQUID_NORM_WIDTH}:{LIQUID_NORM_HEIGHT}:flags=bilinear,"
        f"fps={LIQUID_NORM_FPS}"
    )


def normalize_liquid_video(
    source: str | Path,
    destination: str | Path,
    *,
    timeout_sec: float = DEFAULT_NORMALIZATION_TIMEOUT_SEC,
) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        _scale_fps_filter(),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise NormalizationError(
            "ffmpeg executable not found; install ffmpeg to normalize candidates"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise NormalizationError(
            f"ffmpeg timed out after {timeout_sec}s normalizing {source}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise NormalizationError(
            f"ffmpeg failed to normalize {source} (exit {exc.returncode}): {stderr}"
        ) from exc


# ── Reference-matched normalization (Table 4/7 cells) ──────────────────────
# `match_reference` resamples a candidate onto its case's reference video:
# spatially it is fit inside the reference frame and padded (no cropping, no
# aspect distortion); temporally its whole duration is stretched onto the
# reference duration and sampled at the reference fps for exactly the
# reference frame count. This is the same transform the Table 4/7 preparation
# applied to native model outputs, so frame k of the candidate is compared with
# frame k of the Unity reference and its ground truth.


def _probe(path: str | Path, *, timeout_sec: float) -> dict[str, float]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "default=noprint_wrappers=1", str(path),
    ]
    try:
        out = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout_sec
        ).stdout
    except FileNotFoundError as exc:
        raise NormalizationError("ffprobe executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise NormalizationError(f"ffprobe failed on {path}: {(exc.stderr or '').strip()}") from exc
    fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    numerator, _, denominator = fields["avg_frame_rate"].partition("/")
    fps = float(numerator) / float(denominator or 1)
    frames = int(fields["nb_read_frames"])
    if fps <= 0 or frames <= 0:
        raise NormalizationError(f"{path}: unreadable frame rate or frame count")
    return {"width": int(fields["width"]), "height": int(fields["height"]), "fps": fps, "frames": frames}


def normalize_to_reference(
    source: str | Path,
    destination: str | Path,
    *,
    reference: str | Path,
    timeout_sec: float = DEFAULT_NORMALIZATION_TIMEOUT_SEC,
) -> dict[str, float]:
    """Resample ``source`` to ``reference``'s resolution, fps and frame count."""
    target = _probe(reference, timeout_sec=timeout_sec)
    current = _probe(source, timeout_sec=timeout_sec)
    width, height = int(target["width"]), int(target["height"])
    destination_path = Path(destination)
    if (current["width"], current["height"], current["frames"]) == (width, height, int(target["frames"])) and abs(
        current["fps"] - target["fps"]
    ) <= 1e-3:
        # Already on the reference timeline: copy bytes instead of a lossy re-encode.
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
        return current
    stretch = (target["frames"] / target["fps"]) / (current["frames"] / current["fps"])
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setpts={stretch:.12f}*PTS,fps={target['fps']:.12g}"
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", video_filter, "-frames:v", str(int(target["frames"])), "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_sec)
    except FileNotFoundError as exc:
        raise NormalizationError("ffmpeg executable not found; install ffmpeg") from exc
    except subprocess.TimeoutExpired as exc:
        raise NormalizationError(f"ffmpeg timed out after {timeout_sec}s normalizing {source}") from exc
    except subprocess.CalledProcessError as exc:
        raise NormalizationError(
            f"ffmpeg failed to normalize {source} (exit {exc.returncode}): {(exc.stderr or '').strip()}"
        ) from exc
    actual = _probe(destination_path, timeout_sec=timeout_sec)
    if (actual["width"], actual["height"], actual["frames"]) != (width, height, int(target["frames"])) or abs(
        actual["fps"] - target["fps"]
    ) > 1e-3:
        raise NormalizationError(f"{destination_path}: normalized to {actual}, expected {target}")
    return actual
