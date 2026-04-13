from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.path_resolution import resolve_ptlflow_root  # noqa: E402


def test_resolve_ptlflow_root_prefers_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "custom_ptlflow"
    explicit.mkdir()
    assert resolve_ptlflow_root(str(explicit), repo_root=tmp_path) == str(explicit.resolve())


def test_resolve_ptlflow_root_accepts_wmeval_root(tmp_path: Path) -> None:
    wmeval_root = tmp_path / "WMEval"
    ptlflow_root = tmp_path / "ptlflow"
    wmeval_root.mkdir()
    ptlflow_root.mkdir()
    (ptlflow_root / "synthetic_flow.py").write_text("# stub\n", encoding="utf-8")

    assert resolve_ptlflow_root(None, repo_root=wmeval_root) == str(ptlflow_root.resolve())


def test_resolve_ptlflow_root_accepts_parent_unity_root(tmp_path: Path) -> None:
    ptlflow_root = tmp_path / "ptlflow"
    ptlflow_root.mkdir()
    (ptlflow_root / "synthetic_flow.py").write_text("# stub\n", encoding="utf-8")

    assert resolve_ptlflow_root(None, repo_root=tmp_path) == str(ptlflow_root.resolve())


def test_resolve_ptlflow_root_falls_back_to_workspace() -> None:
    assert resolve_ptlflow_root(None, repo_root=Path("/tmp/definitely_missing_root")) == "/workspace/ptlflow"
