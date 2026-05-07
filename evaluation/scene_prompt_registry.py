from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


SCENE_CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "sample_scene": {
        "scene_hints": (
            "Return the full collision timeline, not just the first event. "
            "Enumerate all distinct collision/contact episodes in chronological order. "
            "Do not omit later collisions just because an earlier one exists, and do not collapse distinct episodes into one event. "
            "Include depth-direction collisions as well as lateral left/right collisions when they are visually distinct episodes. "
            "When depth ambiguity exists, use side='front' for contact with an obstacle ahead in the forward camera-depth direction, and use side='back' for a later contact with an obstacle behind or after the moving object has already passed it. "
            "Only merge contacts that are clearly one continuous contact episode with no meaningful separation."
        ),
        "extraction_mode": "full_timeline",
        "relabel_strategy": "motion_first",
        "match_bias": "chronology",
        "pair_accept_cost": 0.42,
        "skip_cost": 0.26,
        "forbid_cross_class_matches": False,
    },
    "liquid_container": {
        "scene_hints": (
            "Sealed-bottle sloshing scene (the bottle has a cap; liquid stays inside). "
            "Evaluate (a) the fluid velocity field on the liquid mask, "
            "(b) volume conservation via per-frame mask area stability, "
            "(c) the liquid surface orientation relative to gravity, and "
            "(d) the dominant slosh oscillation frequency. "
            "Do NOT evaluate pour onset, fill-fraction depletion, or spill volume — "
            "no liquid leaves the container."
        ),
        "default_sam2_prompt": "the liquid inside the bottle",
        "default_camera_regime": "c2_static_side",
        "default_piv_window_size": 32,
        "default_piv_overlap": 16,
        "applicable_modules": [
            "liquid_velocity",
            "liquid_volume_conservation",
            "liquid_surface_orientation",
            "slosh_spectrum",
            "lpips",
            "control_motion",
            "clip4clip",
        ],
    },
}


def infer_scene_key(reference_video: str, candidate_video: str) -> Optional[str]:
    parts: list[str] = []
    for raw in (reference_video, candidate_video):
        p = Path(raw)
        parts.extend(segment.lower() for segment in p.parts if segment)
        parts.append(p.stem.lower())

    for key in SCENE_CONFIG_REGISTRY:
        if key.lower() in parts:
            return key
    return None


def resolve_scene_hints(
    *,
    scene_key: Optional[str],
    reference_video: str,
    candidate_video: str,
    explicit_hints: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    explicit = explicit_hints.strip() if isinstance(explicit_hints, str) else None
    resolved_key = scene_key.strip().lower() if isinstance(scene_key, str) and scene_key.strip() else None
    if resolved_key is None:
        resolved_key = infer_scene_key(reference_video, candidate_video)

    registry = SCENE_CONFIG_REGISTRY.get(resolved_key) if resolved_key else None
    registry_hints = (
        str(registry.get("scene_hints")).strip()
        if isinstance(registry, Mapping) and str(registry.get("scene_hints", "")).strip()
        else None
    )
    if explicit and registry_hints:
        return resolved_key, registry_hints + "\n" + explicit
    if explicit:
        return resolved_key, explicit
    if registry_hints:
        return resolved_key, registry_hints
    return resolved_key, None


def resolve_scene_config(
    *,
    scene_key: Optional[str],
    reference_video: str,
    candidate_video: str,
) -> tuple[Optional[str], dict[str, Any]]:
    resolved_key = scene_key.strip().lower() if isinstance(scene_key, str) and scene_key.strip() else None
    if resolved_key is None:
        resolved_key = infer_scene_key(reference_video, candidate_video)
    if resolved_key is None:
        return None, {}
    raw = SCENE_CONFIG_REGISTRY.get(resolved_key) or {}
    return resolved_key, dict(raw)
