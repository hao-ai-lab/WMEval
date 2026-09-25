from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules._causal_step_judge import JudgeRequest
from evaluation.modules._causal_step_recorded_judge import (
    RecordedAnswersError,
    RecordedJudge,
    load_recorded_answers,
)


def _artifact(passes: list[dict[str, str]], video_id: str = "vid1") -> dict:
    """Build the on-disk artifact: one CausalStep answers payload per pass."""
    return {
        "video_id": video_id,
        "judge": {
            "model": "gpt-5.4",
            "num_frame_samples": 8,
            "prompt_version": "v1",
            "temperature": 0.0,
        },
        "passes": [
            {
                "video_id": video_id,
                "answers": [
                    {
                        "video_id": video_id,
                        "segment_index": int(key.split("|")[0]),
                        "qa_type": key.split("|")[1],
                        "answer": letter,
                    }
                    for key, letter in p.items()
                ],
            }
            for p in passes
        ],
    }


def _req(segment_index: int, qa_type: str, repeat_index: int = 0) -> JudgeRequest:
    return JudgeRequest(
        video_path="unused.mp4",
        segment_index=segment_index,
        qa_type=qa_type,
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        repeat_index=repeat_index,
    )


def test_recorded_judge_returns_the_letter_for_its_own_pass(tmp_path: Path):
    path = tmp_path / "judge_answers.json"
    path.write_text(
        json.dumps(_artifact([{"1|descriptive": "A"}, {"1|descriptive": "B"}]))
    )

    judge = RecordedJudge(load_recorded_answers(path))

    assert judge.answer(_req(1, "descriptive", repeat_index=0)) == "A"
    assert judge.answer(_req(1, "descriptive", repeat_index=1)) == "B"


def test_recorded_judge_rejects_a_repeat_it_has_no_pass_for(tmp_path: Path):
    """Running repeats=5 against 3 recorded passes must fail loudly.

    Silently wrapping around or reusing the last pass would fabricate
    agreement between passes and inflate CSR toward 0 or 100.
    """
    path = tmp_path / "judge_answers.json"
    path.write_text(json.dumps(_artifact([{"1|descriptive": "A"}])))

    judge = RecordedJudge(load_recorded_answers(path))

    with pytest.raises(RecordedAnswersError, match="repeat"):
        judge.answer(_req(1, "descriptive", repeat_index=1))


def test_recorded_judge_rejects_an_unrecorded_question(tmp_path: Path):
    path = tmp_path / "judge_answers.json"
    path.write_text(json.dumps(_artifact([{"1|descriptive": "A"}])))

    judge = RecordedJudge(load_recorded_answers(path))

    with pytest.raises(RecordedAnswersError, match="no recorded answer"):
        judge.answer(_req(2, "causal", repeat_index=0))


def test_load_recorded_answers_rejects_an_empty_pass_list(tmp_path: Path):
    path = tmp_path / "judge_answers.json"
    path.write_text(json.dumps(_artifact([])))

    with pytest.raises(RecordedAnswersError, match="at least one pass"):
        load_recorded_answers(path)


def test_recorded_answers_expose_pass_count_and_judge_settings(tmp_path: Path):
    """The settings travel with the answers so a run can report what produced
    them — they are not recoverable from the letters alone."""
    path = tmp_path / "judge_answers.json"
    path.write_text(
        json.dumps(_artifact([{"1|descriptive": "A"}, {"1|descriptive": "A"}]))
    )

    recorded = load_recorded_answers(path)

    assert recorded.num_passes == 2
    assert recorded.judge_settings["model"] == "gpt-5.4"
    assert recorded.judge_settings["num_frame_samples"] == 8


def test_write_then_load_round_trips(tmp_path: Path):
    """The writer emits exactly what the loader accepts — otherwise recording a
    fixture and consuming it are two formats that drift apart."""
    from evaluation.modules._causal_step_recorded_judge import write_recorded_answers

    path = tmp_path / "judge_answers.json"
    write_recorded_answers(
        path,
        passes=[
            {("vid1", 1, "descriptive"): "A", ("vid1", 2, "causal"): "C"},
            {("vid1", 1, "descriptive"): "B", ("vid1", 2, "causal"): "C"},
        ],
        video_id="vid1",
        judge_settings={"model": "gpt-5.4", "num_frame_samples": 8,
                        "prompt_version": "v1", "temperature": 0.0},
    )

    recorded = load_recorded_answers(path)
    assert recorded.num_passes == 2
    judge = RecordedJudge(recorded)
    assert judge.answer(_req(1, "descriptive", repeat_index=0)) == "A"
    assert judge.answer(_req(1, "descriptive", repeat_index=1)) == "B"
    assert judge.answer(_req(2, "causal", repeat_index=1)) == "C"


def test_write_rejects_passes_that_disagree_on_which_questions_exist(tmp_path: Path):
    """Every pass must answer the same question set.

    A pass missing a question would make that repeat fail at judge time with a
    stale-recording error, long after the recording was accepted as valid.
    """
    from evaluation.modules._causal_step_recorded_judge import write_recorded_answers

    with pytest.raises(RecordedAnswersError, match="same questions"):
        write_recorded_answers(
            tmp_path / "judge_answers.json",
            passes=[
                {("vid1", 1, "descriptive"): "A", ("vid1", 2, "causal"): "C"},
                {("vid1", 1, "descriptive"): "B"},
            ],
            video_id="vid1",
            judge_settings={"model": "gpt-5.4"},
        )


def test_collect_pass_answers_reads_every_pass_file_in_order(tmp_path: Path):
    """The runner writes pass 0 unsuffixed and pass N as `.rN` — the collector
    has to reassemble them in pass order, not filesystem order."""
    from evaluation.modules._causal_step_io import write_answers
    from evaluation.modules._causal_step_recorded_judge import collect_pass_answers

    for i, letter in enumerate(["A", "B", "C"]):
        suffix = "" if i == 0 else f".r{i}"
        write_answers(
            str(tmp_path / f"vid1_answers{suffix}.json"),
            {("vid1", 1, "descriptive"): letter},
            video_id="vid1",
        )

    passes = collect_pass_answers(tmp_path, video_id="vid1", repeats=3)

    assert [p[("vid1", 1, "descriptive")] for p in passes] == ["A", "B", "C"]


def test_collect_pass_answers_fails_on_a_missing_pass(tmp_path: Path):
    from evaluation.modules._causal_step_io import write_answers
    from evaluation.modules._causal_step_recorded_judge import collect_pass_answers

    write_answers(
        str(tmp_path / "vid1_answers.json"),
        {("vid1", 1, "descriptive"): "A"},
        video_id="vid1",
    )

    with pytest.raises(RecordedAnswersError, match="missing"):
        collect_pass_answers(tmp_path, video_id="vid1", repeats=2)
