"""Resolve recorded artifact paths after a WMEval repository is relocated."""
from __future__ import annotations

from pathlib import Path


_REPO_MARKERS = ("runs", "experiments", "lite", "eval_sets")
# The Lite suite moved from lite/ to lite/; older receipts keep the old path.
_MOVED = {("eval_sets", "wmeval_lite_v1"): ("lite",)}


def resolve_recorded_path(
    value: str | Path,
    *,
    repo_root: Path,
    relative_to: Path | None = None,
) -> Path:
    """Resolve a receipt path locally without weakening its hash validation.

    Receipts retain the absolute generation path. A copied release bundle may
    provide that file at the same repository-relative suffix on another host.
    """
    recorded = Path(value).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    if not recorded.is_absolute() and relative_to is not None:
        candidate = (relative_to / recorded).resolve()
        if candidate.is_file():
            return candidate
    for marker in _REPO_MARKERS:
        if marker not in recorded.parts:
            continue
        suffix = recorded.parts[recorded.parts.index(marker):]
        for old, new in _MOVED.items():
            if suffix[:len(old)] == old:
                suffix = new + suffix[len(old):]
        candidate = (repo_root.resolve() / Path(*suffix)).resolve()
        if candidate.is_file():
            return candidate
    return recorded.resolve()
