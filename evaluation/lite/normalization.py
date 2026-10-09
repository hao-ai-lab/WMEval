from __future__ import annotations

import shutil
import subprocess
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


class NormalizationError(RuntimeError):
    """Candidate video normalization (ffmpeg) failed."""


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
