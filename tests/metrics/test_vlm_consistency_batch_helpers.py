"""Offline tests for vlm_consistency manifest helpers and rubric cache lookup.

Tests pure helpers and the build_manifest end-to-end flow against a fixture
view_dir. No API calls and no video decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="OpenCV required (evaluation package import chain)")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules.vlm_consistency.batch import (  # noqa: E402
    _load_done_keys,
    _load_manifest,
    _read_csv,
    _resolve,
    _split_model_and_run,
    _write_manifest,
    cmd_build_manifest,
)
from evaluation.modules.vlm_consistency.rubric_cache import find_rubric  # noqa: E402


def test_split_model_and_run_typical_filename() -> None:
    assert _split_model_and_run(
        "Genie_ChessBoard_first_run0", "ChessBoard_first"
    ) == ("Genie", "run0")
    assert _split_model_and_run(
        "LingBot_RollingBall_run3", "RollingBall"
    ) == ("LingBot", "run3")


def test_split_model_and_run_scene_at_start_returns_empty_model() -> None:
    assert _split_model_and_run(
        "ChessBoard_first_run0", "ChessBoard_first"
    ) == ("", "")


def test_split_model_and_run_no_scene_match_returns_empty() -> None:
    assert _split_model_and_run("Random_Filename", "ChessBoard") == ("", "")


def test_resolve_relative_path(tmp_path: Path) -> None:
    base = tmp_path
    assert _resolve(base, "data/foo.mp4") == base / "data" / "foo.mp4"


def test_resolve_absolute_path(tmp_path: Path) -> None:
    abs_path = tmp_path / "abs.mp4"
    result = _resolve(tmp_path / "other_base", str(abs_path))
    assert result == abs_path


def test_resolve_none_returns_none(tmp_path: Path) -> None:
    assert _resolve(tmp_path, None) is None
    assert _resolve(tmp_path, "") is None


def test_read_csv_strips_whitespace(tmp_path: Path) -> None:
    csv_path = tmp_path / "p.csv"
    csv_path.write_text(
        "scene_name,prompt,ref_path\n"
        "  ChessBoard  ,  A black-and-white chess scene  ,refs/cb.png\n",
        encoding="utf-8",
    )
    rows = _read_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["scene_name"] == "ChessBoard"
    assert rows[0]["prompt"] == "A black-and-white chess scene"
    assert rows[0]["ref_path"] == "refs/cb.png"


def test_write_load_manifest_round_trip(tmp_path: Path) -> None:
    rows = [
        {"scene_name": "ChessBoard", "prompt": "p1", "video_path": "data/a.mp4",
         "ref_path": "refs/cb.png", "model_name": "Genie", "run_id": "run0"},
        {"scene_name": "ChessBoard", "prompt": "p1", "video_path": "data/b.mp4",
         "ref_path": "refs/cb.png", "model_name": "LingBot", "run_id": "run0"},
    ]
    out = tmp_path / "manifest.jsonl"
    _write_manifest(rows, out)
    base, loaded = _load_manifest(out)
    assert base == tmp_path.resolve()
    assert loaded == rows


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_manifest(tmp_path / "does_not_exist.jsonl")


def test_load_done_keys_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_done_keys(tmp_path / "no.jsonl") == set()


def test_load_done_keys_parses_existing_results(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    rows = [
        {"scene_name": "ChessBoard", "video_path": "data/a.mp4", "score": 0.9},
        {"scene_name": "RollingBall", "video_path": "data/b.mp4", "score": 0.8},
        "this is not json",
        {"scene_name": "", "video_path": "data/c.mp4"},
        {"scene_name": "OK", "video_path": ""},
    ]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")
    done = _load_done_keys(out)
    assert done == {
        ("ChessBoard", "data/a.mp4"),
        ("RollingBall", "data/b.mp4"),
    }


def _fake_view_dir(tmp_path: Path) -> Path:
    """Build a minimal view_dir matching the format build_manifest expects."""
    view_dir = tmp_path / "view"
    (view_dir / "data" / "ChessBoard").mkdir(parents=True)
    (view_dir / "data" / "RollingBall").mkdir(parents=True)
    (view_dir / "refs").mkdir()
    (view_dir / "data" / "ChessBoard" / "Genie_ChessBoard_run0.mp4").write_bytes(b"")
    (view_dir / "data" / "ChessBoard" / "LingBot_ChessBoard_run0.mp4").write_bytes(b"")
    (view_dir / "data" / "RollingBall" / "Genie_RollingBall_run0.mp4").write_bytes(b"")
    (view_dir / "refs" / "ChessBoard.png").write_bytes(b"")
    (view_dir / "scenes.csv").write_text(
        "scene_name,prompt,ref_path\n"
        "ChessBoard,A chess scene,refs/ChessBoard.png\n"
        "RollingBall,A ball scene,\n"
        "MissingScene,No data dir,\n",
        encoding="utf-8",
    )
    return view_dir


def test_cmd_build_manifest_writes_expected_rows(tmp_path: Path) -> None:
    view_dir = _fake_view_dir(tmp_path)
    args = argparse.Namespace(
        view_dir=str(view_dir), csv=None, data_dir=None, out=None,
    )
    rc = cmd_build_manifest(args)
    assert rc == 0

    manifest = view_dir / "manifest.jsonl"
    assert manifest.is_file()

    _, rows = _load_manifest(manifest)
    scene_to_rows = {}
    for r in rows:
        scene_to_rows.setdefault(r["scene_name"], []).append(r)

    assert set(scene_to_rows) == {"ChessBoard", "RollingBall"}
    assert len(scene_to_rows["ChessBoard"]) == 2
    assert len(scene_to_rows["RollingBall"]) == 1

    cb_genie = next(r for r in scene_to_rows["ChessBoard"]
                    if r["model_name"] == "Genie")
    assert cb_genie["run_id"] == "run0"
    assert cb_genie["prompt"] == "A chess scene"
    assert cb_genie["ref_path"] == "refs/ChessBoard.png"

    rb = scene_to_rows["RollingBall"][0]
    assert rb["ref_path"] in (None, "")


def test_cmd_build_manifest_returns_one_when_csv_missing(tmp_path: Path) -> None:
    view_dir = tmp_path / "view"
    (view_dir / "data").mkdir(parents=True)
    args = argparse.Namespace(
        view_dir=str(view_dir), csv=None, data_dir=None, out=None,
    )
    assert cmd_build_manifest(args) == 1


def test_cmd_build_manifest_returns_one_when_data_dir_missing(tmp_path: Path) -> None:
    view_dir = tmp_path / "view"
    view_dir.mkdir()
    (view_dir / "scenes.csv").write_text("scene_name,prompt\nA,p\n", encoding="utf-8")
    args = argparse.Namespace(
        view_dir=str(view_dir), csv=None, data_dir=None, out=None,
    )
    assert cmd_build_manifest(args) == 1


def test_find_rubric_matches_by_prompt_version(tmp_path: Path) -> None:
    rubric_dir = tmp_path / "rubrics"
    scene_dir = rubric_dir / "ChessBoard"
    scene_dir.mkdir(parents=True)
    (scene_dir / "openai_gpt-5-4_v1.json").write_text("{}", encoding="utf-8")
    (scene_dir / "openai_gpt-5-4_v0.json").write_text("{}", encoding="utf-8")

    found = find_rubric("ChessBoard", "v1", str(rubric_dir),
                        provider="openai", model="gpt-5.4")
    assert found is not None
    assert found.name == "openai_gpt-5-4_v1.json"


def test_find_rubric_returns_none_when_no_match(tmp_path: Path) -> None:
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()
    assert find_rubric("MissingScene", "v1", str(rubric_dir)) is None

    scene_dir = rubric_dir / "ChessBoard"
    scene_dir.mkdir()
    (scene_dir / "openai_gpt-5-4_v0.json").write_text("{}", encoding="utf-8")
    assert find_rubric("ChessBoard", "v999", str(rubric_dir)) is None
