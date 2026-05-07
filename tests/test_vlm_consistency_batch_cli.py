"""CLI smoke tests for evaluation.modules.vlm_consistency.batch (no API calls).

The batch CLI is manifest-driven and exposes three subcommands:
``build_manifest``, ``generate_rubrics``, ``evaluate``. These tests only cover
argument parsing and the offline ``build_manifest`` flow against a fixture
view_dir; ``generate_rubrics`` and ``evaluate`` would call out to providers and
are exercised via the helper-level tests instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="OpenCV required (evaluation package import chain)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules.vlm_consistency import batch  # noqa: E402


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["batch", *args])


def test_main_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    _argv(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2


def test_build_manifest_requires_view_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    _argv(monkeypatch, "build_manifest")
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2


def test_generate_rubrics_requires_args(monkeypatch: pytest.MonkeyPatch) -> None:
    _argv(monkeypatch, "generate_rubrics", "--model", "dummy")
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2


def test_evaluate_requires_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _argv(
        monkeypatch,
        "evaluate",
        "--model", "dummy",
        "--output", str(tmp_path / "out.jsonl"),
    )
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2


def _build_fixture_view(tmp_path: Path) -> Path:
    view_dir = tmp_path / "view"
    (view_dir / "data" / "ChessBoard").mkdir(parents=True)
    (view_dir / "data" / "ChessBoard" / "Genie_ChessBoard_run0.mp4").write_bytes(b"")
    (view_dir / "scenes.csv").write_text(
        "scene_name,prompt\nChessBoard,A chess scene\n",
        encoding="utf-8",
    )
    return view_dir


def test_build_manifest_end_to_end_via_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    view_dir = _build_fixture_view(tmp_path)
    _argv(monkeypatch, "build_manifest", "--view_dir", str(view_dir))
    assert batch.main() == 0
    assert (view_dir / "manifest.jsonl").is_file()


def test_evaluate_missing_manifest_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "no_manifest.jsonl"
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()
    out = tmp_path / "out.jsonl"
    _argv(
        monkeypatch,
        "evaluate",
        "--manifest", str(missing),
        "--rubric_dir", str(rubric_dir),
        "--provider", "openai",
        "--model", "dummy",
        "--output", str(out),
    )
    with pytest.raises(FileNotFoundError):
        batch.main()
