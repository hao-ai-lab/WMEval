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
        repo_root = Path(__file__).resolve().parents[2]
    sibling = (repo_root / "ptlflow").resolve()
    if (sibling / "synthetic_flow.py").is_file():
        return str(sibling)

    return "/workspace/ptlflow"
