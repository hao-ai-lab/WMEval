"""Offline tests for vlm_consistency batch helpers and rubric cache lookup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="OpenCV required (evaluation package import chain)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules.vlm_consistency.batch import (  # noqa: E402
    find_ref_image,
    find_videos_for_scene,
    load_prompts,
)
from evaluation.modules.vlm_consistency.rubric_cache import find_rubric  # noqa: E402


def test_load_prompts_reads_scene_and_prompt_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text(
        "Scene,Prompt\n"
        'BallRoller,"keep the ball rolling"\n'
        "RollingBall,simple scene\n",
        encoding="utf-8",
    )
    out = load_prompts(csv_path)
    assert out == {
        "BallRoller": "keep the ball rolling",
        "RollingBall": "simple scene",
    }


def test_find_ref_image_matches_scene_prefix(tmp_path: Path) -> None:
    ref = tmp_path / "BallRoller_image.png"
    ref.write_bytes(b"")
    assert find_ref_image("BallRoller", tmp_path) == ref
    assert find_ref_image("BallRoller extra", tmp_path) == ref


def test_find_videos_for_scene_strips_model_prefix_and_view_suffix(
    tmp_path: Path,
) -> None:
    unity = tmp_path / "Unity"
    unity.mkdir()
    v1 = unity / "Unity_BallRoller_thirdperson.mp4"
    v2 = unity / "Unity_RollingBall.mp4"
    v1.write_bytes(b"")
    v2.write_bytes(b"")

    ball = find_videos_for_scene("BallRoller", unity)
    assert ball == [v1]

    roll = find_videos_for_scene("RollingBall", unity)
    assert roll == [v2]


def test_find_rubric_resolves_expected_filename(tmp_path: Path) -> None:
    scene_dir = tmp_path / "BallRoller"
    scene_dir.mkdir(parents=True)
    path = scene_dir / "openai_gpt-5-4_v1.json"
    path.write_text("{}", encoding="utf-8")

    found = find_rubric(
        "BallRoller",
        "v1",
        str(tmp_path),
        provider="openai",
        model="gpt-5.4",
    )
    assert found == path
