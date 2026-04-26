"""Evaluation modules for world-model video analysis."""

try:
    from .clip4clip import run_clip4clip
except ImportError:
    run_clip4clip = None  # type: ignore[assignment]

try:
    from .control_motion import analyze_control_motion_from_context
    from .lpips import analyze_lpips_from_context, compute_lpips_video_pair
except ImportError:
    analyze_control_motion_from_context = None  # type: ignore[assignment]
    analyze_lpips_from_context = None  # type: ignore[assignment]
    compute_lpips_video_pair = None  # type: ignore[assignment]

__all__ = [
    "analyze_control_motion_from_context",
    "analyze_lpips_from_context",
    "analyze_lighting_from_context",
    "analyze_motion_from_context",
    "analyze_shadow_from_context",
    "compute_lpips_video_pair",
    "run_clip4clip",
]
