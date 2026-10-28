from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset._causal_qa_vlm import QADecoratorResult, decorate_qa
from tools.dataset.generate_beamdomino_causal_step_qa import (
    SHAPE_STAGE_CHAINS,
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


# --- shape-aware chain dispatch ----------------------------------------------


_THREE_EVENTS_DIRECT_STRIKE = [
    {"index": 1, "name": "rolling", "start_sec": 0.0, "end_sec": 0.6},
    {"index": 2, "name": "strike",  "start_sec": 0.6, "end_sec": 0.9},
    {"index": 3, "name": "cascade", "start_sec": 0.9, "end_sec": 1.5},
]

_FOUR_EVENTS_GLANCING = [
    {"index": 1, "name": "rolling", "start_sec": 0.0, "end_sec": 0.6},
    {"index": 2, "name": "glance",  "start_sec": 0.6, "end_sec": 1.2},
    {"index": 3, "name": "strike",  "start_sec": 1.2, "end_sec": 1.4},
    {"index": 4, "name": "cascade", "start_sec": 1.4, "end_sec": 2.0},
]


def test_build_dataset_direct_strike_shape_emits_three_segments():
    """direct_strike trajectories are 3 segments, not 4. The dataset must
    contain descriptive QAs for segs 1/2/3 and causal QAs for 2/3 only."""
    ds = build_dataset(
        video_id="far_001",
        events=_THREE_EVENTS_DIRECT_STRIKE,
        trajectory_shape="direct_strike",
    )
    v = ds["videos"][0]
    assert [s["index"] for s in v["segments"]] == [1, 2, 3]
    assert [q["segment_index"] for q in v["descriptive_qas"]] == [1, 2, 3]
    assert [q["segment_index"] for q in v["causal_qas"]] == [2, 3]


def test_build_dataset_glancing_shape_uses_glance_template_for_seg2():
    """Glancing trajectories use the `glance` template at seg 2, whose
    correct descriptive answer is "platform stays standing" — distinct from
    canonical's "ball strikes a leg" or direct_strike's "ball hits domino"."""
    ds = build_dataset(
        video_id="glance_001",
        events=_FOUR_EVENTS_GLANCING,
        trajectory_shape="glancing",
    )
    seg2_desc = next(
        q for q in ds["videos"][0]["descriptive_qas"] if q["segment_index"] == 2
    )
    correct_idx = ord(seg2_desc["correct_answer"]) - ord("A")
    assert "stays standing" in seg2_desc["choices"][correct_idx]


def test_build_dataset_shape_chain_length_mismatch_raises():
    """If the events list has the wrong number of segments for the declared
    shape, the generator must refuse rather than silently dropping or
    duplicating QAs."""
    with pytest.raises(ValueError, match="expects 4 segments"):
        build_dataset(
            video_id="x",
            events=_THREE_EVENTS_DIRECT_STRIKE,  # 3 events
            trajectory_shape="canonical",          # but canonical wants 4
        )


def test_shape_stage_chains_only_reference_known_stages():
    """Every stage name listed in SHAPE_STAGE_CHAINS must have a template,
    otherwise a build with that shape would crash on KeyError."""
    for shape, stages in SHAPE_STAGE_CHAINS.items():
        for s in stages:
            assert s in STAGE_TEMPLATES, f"shape {shape!r} references unknown stage {s!r}"


# --- VLM-mode (decorator) tests ----------------------------------------------


def _make_decorator(
    question_prefix: str,
    distractors: list[str],
    *,
    correct_override: str | None = None,
):
    """Returns a decorator callable that fakes VLM output: the question is
    `question_prefix + template_question` and the distractors are constant.

    By default it confirms the template's correct answer (returns
    `template_correct_body` verbatim). Pass `correct_override` to simulate
    a VLM that decides the frames contradict the template — `_build_qa`
    must accept the override and log it."""

    def _decorator(
        *,
        stage,
        qa_type,
        segment_index,
        start_sec,
        end_sec,
        template_question,
        template_correct_body,
    ):
        return QADecoratorResult(
            question=f"{question_prefix}{template_question}",
            correct=correct_override
            if correct_override is not None
            else template_correct_body,
            distractors=list(distractors),
        )

    return _decorator


def test_build_dataset_vlm_mode_uses_decorator_question_and_distractors():
    decorator = _make_decorator(
        "[VLM] ",
        [
            "fake distractor one",
            "fake distractor two",
            "fake distractor three",
        ],
    )
    ds = build_dataset(
        video_id="movie_002", events=_FOUR_EVENTS, decorator=decorator
    )
    v = ds["videos"][0]
    for qa in v["descriptive_qas"] + v["causal_qas"]:
        assert qa["question"].startswith("[VLM] ")
        bodies = [c.split(") ", 1)[1] for c in qa["choices"]]
        # 3 distractors from the decorator + the template's correct body
        assert sum(b in bodies for b in ["fake distractor one", "fake distractor two", "fake distractor three"]) == 3


def test_build_dataset_vlm_mode_uses_template_correct_when_decorator_confirms():
    """When the decorator returns `correct=template_correct_body` verbatim
    (the common case — VLM agrees with the template), the QA's
    `correct_answer` must point at the template's correct text."""
    decorator = _make_decorator("", ["d1", "d2", "d3"])
    ds = build_dataset(
        video_id="movie_002", events=_FOUR_EVENTS, decorator=decorator
    )
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


def test_build_dataset_vlm_mode_accepts_decorator_correction(capsys):
    """When the decorator (i.e. the VLM, in production) decides the
    template's correct answer is wrong for this trajectory, the override
    must flow through to `correct_answer` AND be logged to stderr so a
    human reviewer can spot-check it."""
    decorator = _make_decorator(
        "",
        ["d1", "d2", "d3"],
        correct_override="VLM-corrected ground truth",
    )
    ds = build_dataset(
        video_id="movie_002", events=_FOUR_EVENTS, decorator=decorator
    )
    v = ds["videos"][0]
    for qa in v["descriptive_qas"] + v["causal_qas"]:
        correct_body, _ = _qa_lookup(qa)
        assert correct_body == "VLM-corrected ground truth"

    err = capsys.readouterr().err
    # At least one override should have been logged. We don't assert on
    # exact count since segment 1 has no causal QA.
    assert "VLM overrode template correct answer" in err
    assert "movie_002" in err


# --- decorate_qa unit tests --------------------------------------------------


def _fake_frames(n: int = 4) -> np.ndarray:
    return (np.random.default_rng(0).random((n, 32, 32, 3)) * 255).astype("uint8")


def _make_openai_mock(
    question: str, correct: str, distractors: list[str]
) -> MagicMock:
    payload = json.dumps(
        {"question": question, "correct": correct, "distractors": distractors}
    )
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=payload))]
    )
    return client


def test_decorate_qa_caches_response(tmp_path: Path):
    """A second call with identical inputs must hit the cache, not the API.
    Otherwise running the generator twice on the same dataset would double
    the OpenAI bill for no reason."""
    client = _make_openai_mock(
        "rewritten Q",
        correct="rolling along the floor",
        distractors=["dist A", "dist B", "dist C"],
    )
    common = dict(
        client=client,
        model="gpt-5.4",
        frames=_fake_frames(4),
        stage="rolling",
        qa_type="descriptive",
        start_sec=0.0,
        end_sec=2.4,
        template_question="What is the floor ball doing?",
        template_correct_body="rolling along the floor",
        cache_dir=tmp_path / "cache",
        video_id="vid",
        segment_index=1,
    )
    out1 = decorate_qa(**common)
    out2 = decorate_qa(**common)
    assert out1.question == out2.question == "rewritten Q"
    assert out1.distractors == out2.distractors == ["dist A", "dist B", "dist C"]
    assert out1.correct == "rolling along the floor"
    assert client.chat.completions.create.call_count == 1


def test_decorate_qa_passes_through_template_correct_when_vlm_confirms(tmp_path: Path):
    """The common case: VLM returns `correct` matching the template body
    verbatim. `decorate_qa` must surface that text."""
    client = _make_openai_mock("Q", correct="THE TRUTH", distractors=["a", "b", "c"])
    out = decorate_qa(
        client=client,
        model="gpt-5.4",
        frames=_fake_frames(4),
        stage="impact",
        qa_type="causal",
        start_sec=2.4,
        end_sec=2.8,
        template_question="Why?",
        template_correct_body="THE TRUTH",
        cache_dir=tmp_path / "cache",
        video_id="vid",
        segment_index=2,
    )
    assert out.correct == "THE TRUTH"


def test_decorate_qa_emits_corrected_answer_when_vlm_overrides(tmp_path: Path, caplog):
    """When the VLM decides the template's correct answer is wrong for the
    actual frames, `decorate_qa` must surface the corrected text and log a
    warning so a human reviewer can spot-check it."""
    import logging as _logging

    client = _make_openai_mock(
        "Q",
        correct="actually it tipped left, not right",
        distractors=["a", "b", "c"],
    )
    with caplog.at_level(_logging.WARNING, logger="tools.dataset._causal_qa_vlm"):
        out = decorate_qa(
            client=client,
            model="gpt-5.4",
            frames=_fake_frames(4),
            stage="topple",
            qa_type="descriptive",
            start_sec=2.8,
            end_sec=3.6,
            template_question="What does the platform do?",
            template_correct_body="it tips to the right",
            cache_dir=tmp_path / "cache",
            video_id="vid",
            segment_index=3,
        )

    assert out.correct == "actually it tipped left, not right"
    assert any("VLM overrode template correct answer" in r.message for r in caplog.records)

    # Cache must persist the override so reruns honor the corrected answer.
    cache_blobs = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_blobs) == 1
    cached = json.loads(cache_blobs[0].read_text())
    assert cached["correct"] == "actually it tipped left, not right"
    assert cached["template_correct_body"] == "it tips to the right"


def test_decorate_qa_rejects_distractor_equal_to_emitted_correct(tmp_path: Path):
    """The model sometimes 'helpfully' includes the correct answer as a
    distractor. Treat that as a generation failure — caching it would
    corrupt every future rerun. Comparison is against the *emitted*
    correct, not the template, since the VLM may have overridden it."""
    client = _make_openai_mock(
        "Q",
        correct="rolling along the floor",
        distractors=["fake", "rolling along the floor", "other"],
    )
    with pytest.raises(ValueError, match="identical to the correct"):
        decorate_qa(
            client=client,
            model="gpt-5.4",
            frames=_fake_frames(4),
            stage="rolling",
            qa_type="descriptive",
            start_sec=0.0,
            end_sec=2.4,
            template_question="Q?",
            template_correct_body="rolling along the floor",
            cache_dir=tmp_path / "cache",
            video_id="vid",
            segment_index=1,
        )
    # And: the failure must NOT have written a cache entry; otherwise
    # subsequent runs would inherit the corrupt response without ever
    # retrying the model.
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_decorate_qa_rejects_wrong_distractor_count(tmp_path: Path):
    client = _make_openai_mock("Q", correct="correct", distractors=["only", "two"])
    with pytest.raises(ValueError, match="expected exactly 3"):
        decorate_qa(
            client=client,
            model="gpt-5.4",
            frames=_fake_frames(4),
            stage="rolling",
            qa_type="descriptive",
            start_sec=0.0,
            end_sec=2.4,
            template_question="Q?",
            template_correct_body="correct",
            cache_dir=tmp_path / "cache",
            video_id="vid",
            segment_index=1,
        )


def test_decorate_qa_cache_invalidates_on_num_frames_change(tmp_path: Path):
    """Bumping `num_frames` must miss the cache. Otherwise an old 4-frame
    answer would silently satisfy a request that should have used 8 frames."""
    client = _make_openai_mock("Q", correct="correct", distractors=["a", "b", "c"])
    base = dict(
        client=client,
        model="gpt-5.4",
        stage="rolling",
        qa_type="descriptive",
        start_sec=0.0,
        end_sec=2.4,
        template_question="Q?",
        template_correct_body="correct",
        cache_dir=tmp_path / "cache",
        video_id="vid",
        segment_index=1,
    )
    decorate_qa(frames=_fake_frames(4), **base)
    assert client.chat.completions.create.call_count == 1
    decorate_qa(frames=_fake_frames(8), **base)
    assert client.chat.completions.create.call_count == 2
