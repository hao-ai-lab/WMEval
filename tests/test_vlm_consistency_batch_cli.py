"""CLI smoke tests for evaluation.modules.vlm_consistency.batch (no API calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="OpenCV required (evaluation package import chain)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules.vlm_consistency import batch  # noqa: E402


def test_main_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["batch"])
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2


def test_generate_rubrics_missing_csv_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "generate_rubrics", "--model", "dummy", "--csv", str(missing)],
    )
    assert batch.main() == 1


def test_evaluate_missing_csv_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "evaluate",
            "--model",
            "dummy",
            "--csv",
            str(missing),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert batch.main() == 1


def test_evaluate_no_model_dirs_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "p.csv"
    csv_path.write_text("Scene,Prompt\nBallRoller,x\n", encoding="utf-8")
    empty_data = tmp_path / "data"
    empty_data.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "evaluate",
            "--model",
            "dummy",
            "--csv",
            str(csv_path),
            "--data_dir",
            str(empty_data),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert batch.main() == 1
