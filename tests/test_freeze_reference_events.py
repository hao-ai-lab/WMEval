from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.modules.lpips_causal_vlm import load_frozen_reference_events
from tools.freeze_reference_events import select_consensus


def _pass(*events: tuple[int, str, float]) -> dict:
    return {
        "payload": {
            "events": [
                {"frame": frame, "side": side, "confidence": confidence, "type": "collision"}
                for frame, side, confidence in events
            ]
        }
    }


def test_consensus_keeps_only_events_a_majority_of_passes_agree_on() -> None:
    passes = [
        _pass((100, "front", 0.9), (300, "left", 0.5)),
        _pass((104, "front", 0.7)),
        _pass((98, "front", 0.8)),
        _pass(),
        _pass((302, "left", 0.6)),
    ]
    selected, supports = select_consensus(passes, min_support=3)
    assert [e["frame"] for e in selected] == [100]  # highest-confidence member of the front cluster
    by_frame = {s["frame"]: s for s in supports}
    assert by_frame[100]["support"] == 3 and by_frame[100]["kept"]
    assert by_frame[302]["support"] == 2 and not by_frame[302]["kept"]


def test_consensus_counts_each_pass_once_and_separates_sides() -> None:
    passes = [_pass((100, "front", 0.9), (103, "front", 0.8)), _pass((101, "back", 0.9))]
    selected, supports = select_consensus(passes, min_support=2)
    assert selected == []  # same-pass duplicates do not self-support; other side is a different contact
    assert sorted(s["support"] for s in supports) == [1, 1]


def test_consensus_of_nothing_is_empty() -> None:
    assert select_consensus([_pass(), _pass()], min_support=2) == ([], [])


def _write_frozen(directory: Path, reference: Path, **overrides) -> Path:
    document = {
        "schema_version": 1,
        "reference_video_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "selected": {
            "payload": {"events": [{"frame": 120, "side": "front"}], "reasoning_steps": ["r"]},
            "grid_frame_indices": [0, 60, 120],
        },
    }
    document.update(overrides)
    path = directory / "contact_events.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_frozen_events_round_trip_as_the_judge_would_return_them(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"unity reference")
    payload, indices, frames = load_frozen_reference_events(_write_frozen(tmp_path, reference), reference)
    assert payload["events"] == [{"frame": 120, "side": "front"}]
    assert payload["frozen_reference_events"] is True
    assert indices == [0, 60, 120] and frames == []


def test_frozen_event_survives_the_grid_check_when_relabeling_moved_its_evidence(tmp_path: Path) -> None:
    from evaluation.modules.lpips_causal_vlm import sanitize_events_to_grid

    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"unity reference")
    # Evidence from the side-relabel window (step 11) rather than the main grid.
    event = {"frame": 231, "side": "front", "evidence_frame_indices": [204, 215, 226, 237, 248, 259]}
    frozen = _write_frozen(
        tmp_path,
        reference,
        selected={"payload": {"events": [event]}, "grid_frame_indices": [216, 231, 247]},
    )
    payload, seen, _ = load_frozen_reference_events(frozen, reference)

    assert sanitize_events_to_grid(payload["events"], {216, 231, 247})[0] == []  # the bug
    kept, dropped = sanitize_events_to_grid(payload["events"], set(seen))
    assert [e["frame"] for e in kept] == [231] and dropped == []


def test_frozen_events_refuse_a_different_reference_video(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"unity reference")
    frozen = _write_frozen(tmp_path, reference)
    reference.write_bytes(b"a different video")
    with pytest.raises(ValueError, match="different reference video"):
        load_frozen_reference_events(frozen, reference)


def test_frozen_events_reject_an_unknown_schema(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported frozen-events schema"):
        load_frozen_reference_events(_write_frozen(tmp_path, reference, schema_version=99), reference)
