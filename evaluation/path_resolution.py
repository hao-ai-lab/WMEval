from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_ptlflow_root(
    explicit_root: Optional[str] = None,
    *,
    repo_root: Optional[Path] = None,
) -> str:
    if explicit_root is not None and str(explicit_root).strip():
        return str(Path(str(explicit_root).strip()).expanduser().resolve())

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    resolved_root = Path(repo_root).resolve()
    candidate_bases = [resolved_root]
    if resolved_root.name == "WMEval":
        candidate_bases.append(resolved_root.parent)

    seen: set[Path] = set()
    for base in candidate_bases:
        if base in seen:
            continue
        seen.add(base)
        sibling = (base / "ptlflow").resolve()
        if (sibling / "synthetic_flow.py").is_file():
            return str(sibling)

    return "/workspace/ptlflow"
