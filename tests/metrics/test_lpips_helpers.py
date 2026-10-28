"""Unit tests for LPIPS helper functions (no API calls, no torch required for these)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cv2", reason="OpenCV required (evaluation.video_io imports cv2)")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.context import VideoMetadata  # noqa: E402
from evaluation.modules.lpips_causal_vlm import (  # noqa: E402
    build_judge_user_prompt,
    extract_json_object,
    filter_events_by_type,
    pair_collision_events_ref_first,
    paired_frames_for_reference_event_offset,
    paired_frames_per_video_contact,
    resolve_lpips_temporal_pairing,
)
from evaluation.video_io import (  # noqa: E402
    build_default_event_offsets,
    normalized_anchor_indices_from_fractions,
)


def test_normalized_anchor_indices_basic() -> None:
    assert normalized_anchor_indices_from_fractions(10, [0.0, 0.5, 1.0]) == [0, 4, 9]
    assert normalized_anchor_indices_from_fractions(1, [0.0, 1.0]) == [0, 0]
    assert normalized_anchor_indices_from_fractions(3, [0.5]) == [1]


def test_normalized_anchor_indices_empty() -> None:
    assert normalized_anchor_indices_from_fractions(100, []) == []
    assert normalized_anchor_indices_from_fractions(0, [0.0]) == []


def test_normalized_anchor_indices_fraction_out_of_range() -> None:
    with pytest.raises(ValueError, match="fraction"):
        normalized_anchor_indices_from_fractions(10, [1.1])


def test_build_default_event_offsets() -> None:
    assert build_default_event_offsets(2, 2) == [-2, -1, 0, 1, 2]
    assert build_default_event_offsets(0, 0) == [0]


def test_extract_json_object_plain() -> None:
    obj = extract_json_object('{"a": 1, "events": []}')
    assert obj == {"a": 1, "events": []}


def test_extract_json_object_fenced() -> None:
    text = '```json\n{"reasoning_steps": [], "events": [{"frame": 3}]}\n```'
    obj = extract_json_object(text)
    assert obj["events"][0]["frame"] == 3


def test_extract_json_object_embedded() -> None:
    text = 'Here you go: {"x": 2} thanks'
    assert extract_json_object(text) == {"x": 2}


def test_pair_collision_side_left() -> None:
    ref = [{"type": "collision", "side": "left", "frame": 10}]
    cand = [{"type": "collision", "side": "left", "frame": 12}]
    matched, ur, uc = pair_collision_events_ref_first(ref, cand)
    assert len(matched) == 1
    assert ur == [] and uc == []


def test_pair_collision_unknown_prefers_unknown() -> None:
    ref = [{"type": "collision", "side": "unknown", "frame": 1}]
    cand_left = [{"type": "collision", "side": "left", "frame": 2}]
    cand_unk = [{"type": "collision", "side": "unknown", "frame": 3}]
    matched, _, _ = pair_collision_events_ref_first(ref, cand_left + cand_unk)
    assert matched[0][1]["frame"] == 3


def test_pair_collision_unmatched() -> None:
    ref = [{"type": "collision", "side": "left", "frame": 1}]
    cand = [{"type": "collision", "side": "right", "frame": 2}]
    matched, ur, uc = pair_collision_events_ref_first(ref, cand)
    assert matched == []
    assert len(ur) == 1 and len(uc) == 1


def test_pair_collision_json_roundtrip() -> None:
    ref = json.loads('[{"type":"collision","side":"right","frame":5}]')
    cand = json.loads('[{"type":"collision","side":"right","frame":7}]')
    matched, _, _ = pair_collision_events_ref_first(ref, cand)
    assert len(matched) == 1


def test_build_judge_user_prompt_max_events_appended() -> None:
    base = build_judge_user_prompt([0, 10])
    one = build_judge_user_prompt([0, 10], max_events=1)
    two = build_judge_user_prompt([0, 10], max_events=2)
    assert "at most ONE" in one
    assert "earliest" in one.lower()
    assert one != base
    assert "at most 2" in two


def test_filter_events_by_type_accepts_bounce_and_substrings() -> None:
    ev = [
        {"type": "bounce", "frame": 1},
        {"type": "head_on_collision", "frame": 2},
        {"type": "other", "frame": 3},
    ]
    kept = filter_events_by_type(ev)
    assert [e["frame"] for e in kept] == [1, 2]


def test_resolve_lpips_temporal_pairing_auto() -> None:
    assert (
        resolve_lpips_temporal_pairing({}, event_source="both") == "per_video_contact"
    )
    assert resolve_lpips_temporal_pairing({}, event_source="reference") == "wall_clock"
    assert (
        resolve_lpips_temporal_pairing(
            {"lpips_causal_lpips_pairing": "wall_clock"}, event_source="both"
        )
        == "wall_clock"
    )


def test_paired_frames_per_video_contact_clips() -> None:
    ref = VideoMetadata("a", 30.0, 100, 1, 1, 100 / 30.0)
    cand = VideoMetadata("b", 20.0, 50, 1, 1, 50 / 20.0)
    r, c = paired_frames_per_video_contact(5, 10, -10, ref, cand)
    assert r == 0 and c == 0
    r2, c2 = paired_frames_per_video_contact(5, 10, 2, ref, cand)
    assert r2 == 7 and c2 == 12


def test_paired_frames_same_fps() -> None:
    ref = VideoMetadata("a", 30.0, 300, 1, 1, 10.0)
    cand = VideoMetadata("b", 30.0, 300, 1, 1, 10.0)
    r, c = paired_frames_for_reference_event_offset(100, 2, ref, cand)
    assert r == 102 and c == 102


def test_paired_frames_mismatched_fps() -> None:
    ref = VideoMetadata("a", 60.0, 121, 1, 1, 121 / 60.0)
    cand = VideoMetadata("b", 20.0, 41, 1, 1, 41 / 20.0)
    r, c = paired_frames_for_reference_event_offset(60, 0, ref, cand)
    assert r == 60
    assert c == 20


def test_paired_frames_offset_scales_by_ref_fps() -> None:
    ref = VideoMetadata("a", 30.0, 300, 1, 1, 10.0)
    cand = VideoMetadata("b", 60.0, 600, 1, 1, 10.0)
    r0, c0 = paired_frames_for_reference_event_offset(100, 0, ref, cand)
    r1, c1 = paired_frames_for_reference_event_offset(100, 1, ref, cand)
    assert r1 == r0 + 1
    assert c1 == c0 + 2


def test_causal_reference_mode_calls_vlm_once() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("lpips")

    from unittest.mock import patch

    from evaluation.modules.lpips_causal_vlm import analyze_causal_vlm_lpips
    from evaluation.video_io import build_evaluation_context

    ref_v = ROOT / "videos/unity/mirror_ball_unity_test.mp4"
    cand_v = ROOT / "videos/world_model/mirror_ball_genie_test.mp4"
    if not ref_v.is_file() or not cand_v.is_file():
        pytest.skip("Mirror ball fixture videos not present")

    calls: list[int] = []

    def fake_judge(
        video_path: str,
        max_grid_frames: int,
        model: str,
        **_: Any,
    ) -> tuple[dict, list[int], list]:
        calls.append(1)
        grid = [10, 50, 90]
        ev = [
            {
                "type": "collision",
                "side": "left",
                "frame": 50,
                "evidence_frame_indices": [10, 50],
            }
        ]
        return {"reasoning_steps": [], "events": ev}, grid, []

    ctx = build_evaluation_context(
        reference_video=str(ref_v),
        candidate_video=str(cand_v),
        eval_fps=4.0,
        max_pairs=8,
        text_prompt="",
    )
    ctx.artifacts["lpips_causal_event_source"] = "reference"

    import torch

    with patch(
        "evaluation.modules.lpips_causal_vlm.run_vlm_judge_on_video",
        side_effect=fake_judge,
    ):
        out = analyze_causal_vlm_lpips(ctx, torch.device("cpu"), ctx.artifacts)

    assert len(calls) == 1
    assert out.get("judge_candidate", {}).get("skipped") is True
    assert out["num_matched_events"] == 1
    assert "evaluation_interval" in out["lpips_per_matched_event"][0]
    row0 = out["lpips_per_matched_event"][0]
    assert row0["e_cand_contact_ref_frame"] == row0["e_ref"]
