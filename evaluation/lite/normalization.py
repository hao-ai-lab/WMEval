from __future__ import annotations

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
