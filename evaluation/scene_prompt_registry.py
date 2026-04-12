from __future__ import annotations

from pathlib import Path
from typing import Optional


SCENE_HINT_REGISTRY: dict[str, str] = {
    "sample_scene": (
        "Return the full collision timeline, not just the first event. "
        "Enumerate all distinct collision/contact episodes in chronological order. "
        "Do not omit later collisions just because an earlier one exists, and do not collapse distinct episodes into one event. "
        "Include depth-direction collisions as well as lateral left/right collisions when they are visually distinct episodes. "
        "When depth ambiguity exists, use side='front' for contact with an obstacle ahead in the forward camera-depth direction, and use side='back' for a later contact with an obstacle behind or after the moving object has already passed it. "
        "Only merge contacts that are clearly one continuous contact episode with no meaningful separation."
    ),
}


def infer_scene_key(reference_video: str, candidate_video: str) -> Optional[str]:
    parts: list[str] = []
    for raw in (reference_video, candidate_video):
        p = Path(raw)
        parts.extend(segment.lower() for segment in p.parts if segment)
        parts.append(p.stem.lower())

    for key in SCENE_HINT_REGISTRY:
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

    registry_hints = SCENE_HINT_REGISTRY.get(resolved_key) if resolved_key else None
    if explicit and registry_hints:
        return resolved_key, registry_hints + "\n" + explicit
    if explicit:
        return resolved_key, explicit
    if registry_hints:
        return resolved_key, registry_hints
    return resolved_key, None
