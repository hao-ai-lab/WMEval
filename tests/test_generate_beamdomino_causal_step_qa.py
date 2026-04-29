from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_beamdomino_causal_step_qa import (
    STAGE_TEMPLATES,
    _shuffled_choices,
    build_dataset,
)


_FOUR_EVENTS = [
    {"index": 1, "name": "rolling", "start_sec": 0.0, "end_sec": 2.4},
    {"index": 2, "name": "impact", "start_sec": 2.4, "end_sec": 2.8},
    {"index": 3, "name": "topple", "start_sec": 2.8, "end_sec": 3.6},
    {"index": 4, "name": "cascade", "start_sec": 3.6, "end_sec": 4.5},
]


def _qa_lookup(qa: dict) -> tuple[str, list[str]]:
    """Return the body text of the correct option, and the bodies of all
    options without their `A)`/`B)` prefix. Used to verify that shuffling
    didn't change which option is correct, only its position."""
    bodies = [c.split(") ", 1)[1] for c in qa["choices"]]
    correct_idx = ord(qa["correct_answer"]) - ord("A")
    return bodies[correct_idx], bodies


def test_shuffled_choices_preserves_correct_text():
    choices = ["A) alpha", "B) beta", "C) gamma", "D) delta"]
    new_choices, new_correct = _shuffled_choices(choices, "A", seed=42)
    bodies = [c.split(") ", 1)[1] for c in new_choices]
    assert sorted(bodies) == sorted(["alpha", "beta", "gamma", "delta"])
    correct_idx = ord(new_correct) - ord("A")
    assert bodies[correct_idx] == "alpha"


def test_shuffled_choices_is_deterministic():
    choices = ["A) one", "B) two", "C) three", "D) four"]
    a1 = _shuffled_choices(choices, "A", seed=12345)
    a2 = _shuffled_choices(choices, "A", seed=12345)
    assert a1 == a2


def test_shuffled_choices_letters_are_dense():
    """Output letters must be A, B, C, D — no gaps, regardless of permutation."""
    choices = ["A) one", "B) two", "C) three", "D) four"]
    new_choices, _ = _shuffled_choices(choices, "A", seed=7)
    letters = [c.split(")", 1)[0] for c in new_choices]
    assert letters == ["A", "B", "C", "D"]


def test_build_dataset_correct_answers_are_not_all_A():
    """The whole point of shuffling: across a full BeamDomino dataset, the
    correct letter must vary so the judge cannot exploit positional bias."""
    ds = build_dataset(video_id="movie_002", events=_FOUR_EVENTS)
    v = ds["videos"][0]
    letters = [qa["correct_answer"] for qa in v["descriptive_qas"]] + [
        qa["correct_answer"] for qa in v["causal_qas"]
    ]
    # 4 descriptive + 3 causal = 7 letters from {A,B,C,D}; if all match it's
    # almost certainly because the shuffle is a no-op (probability of all
    # seven landing on the same letter by chance is ~7/4**7 ≈ 4e-4).
    assert len(set(letters)) > 1, f"all correct answers were {letters[0]!r}"


def test_build_dataset_shuffle_preserves_template_correctness():
    """For every QA, the option pointed to by `correct_answer` must contain
    the same body text the template marked correct. Otherwise we've silently
    changed the right answer instead of just moving its position."""
    ds = build_dataset(video_id="movie_002", events=_FOUR_EVENTS)
    v = ds["videos"][0]

    def _expected_body(stage: str, qa_kind: str) -> str:
        tmpl = STAGE_TEMPLATES[stage]
        choices = tmpl[f"{qa_kind}_choices"]
        letter = tmpl[f"{qa_kind}_correct"]
        return choices[ord(letter) - ord("A")].split(") ", 1)[1]

    stage_by_index = {seg["index"]: seg["name"] for seg in _FOUR_EVENTS}
    for qa in v["descriptive_qas"]:
        correct_body, _ = _qa_lookup(qa)
        assert correct_body == _expected_body(
            stage_by_index[qa["segment_index"]], "descriptive"
        )
    for qa in v["causal_qas"]:
        correct_body, _ = _qa_lookup(qa)
        assert correct_body == _expected_body(
            stage_by_index[qa["segment_index"]], "causal"
        )


def test_build_dataset_is_deterministic_per_video_id():
    """Two builds of the same video_id+events must produce identical output —
    otherwise the OpenAI judge's prompt cache key (which hashes the prompt
    text including choice order) would invalidate on every run."""
    ds1 = build_dataset(video_id="trajectory_42", events=_FOUR_EVENTS)
    ds2 = build_dataset(video_id="trajectory_42", events=_FOUR_EVENTS)
    assert ds1 == ds2


def test_build_dataset_different_video_ids_get_different_shuffles():
    """The seed depends on video_id, so two different trajectories should
    not get identical choice ordering — that would defeat the bias spread."""
    ds_a = build_dataset(video_id="alpha", events=_FOUR_EVENTS)
    ds_b = build_dataset(video_id="bravo", events=_FOUR_EVENTS)
    a_letters = [qa["correct_answer"] for qa in ds_a["videos"][0]["causal_qas"]]
    b_letters = [qa["correct_answer"] for qa in ds_b["videos"][0]["causal_qas"]]
    # Vanishingly unlikely they coincidentally match across all causal QAs.
    assert a_letters != b_letters
