"""Replay pre-recorded judge answers instead of calling a VLM.

The reference side of this benchmark is a constant, not a computation: the
reference video never changes, the QA never changes, and the judge model,
frame count and prompt version are pinned by the dimension. Recording the
answers once and shipping them inside the case bundle makes the reference a
verified fixture; the candidate side still runs the real pipeline, and the
protocol is untouched.

Judge answers are sampled rather than deterministic (the gpt-5 family rejects
a non-default `temperature`, so none is sent), which is why a recording holds
several independent passes: CSR over N passes is successes/N, a value a single
all-or-nothing pass cannot express.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from ._causal_step_judge import JudgeRequest


class RecordedAnswersError(ValueError):
    """Raised when a recording is malformed or does not cover a request."""


# Settings the dimension pins and that change what the judge answers. The
# temperature is left out: it is not configurable from the suite, and for the
# gpt-5 family none is sent at all.
PINNED_JUDGE_FIELDS: Tuple[str, ...] = ("model", "num_frame_samples", "prompt_version")


_AnswerKey = Tuple[int, str]


@dataclass(frozen=True)
class RecordedAnswers:
    """One video's recorded judge answers, one map per independent pass."""

    video_id: str
    passes: List[Mapping[_AnswerKey, str]]
    judge_settings: Mapping[str, Any]

    @property
    def num_passes(self) -> int:
        return len(self.passes)


def load_recorded_answers(path: str | Path) -> RecordedAnswers:
    """Read a `judge_answers.json` case artifact.

    Each entry of `passes` is a CausalStep answers payload — the same shape
    `_causal_step_io.write_answers` emits — so a recording can be assembled
    straight from a real run's output.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordedAnswersError(f"{path}: cannot read recording: {exc}") from exc

    if not isinstance(data, dict):
        raise RecordedAnswersError(f"{path}: expected a JSON object")

    video_id = data.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise RecordedAnswersError(f"{path}: missing 'video_id'")

    raw_passes = data.get("passes")
    if not isinstance(raw_passes, list) or not raw_passes:
        raise RecordedAnswersError(f"{path}: 'passes' must hold at least one pass")

    passes: List[Mapping[_AnswerKey, str]] = []
    for i, payload in enumerate(raw_passes):
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
            raise RecordedAnswersError(f"{path}: pass {i} has no 'answers' list")
        one: Dict[_AnswerKey, str] = {}
        for entry in payload["answers"]:
            try:
                key = (int(entry["segment_index"]), str(entry["qa_type"]))
                one[key] = str(entry["answer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordedAnswersError(
                    f"{path}: pass {i} has a malformed answer entry {entry!r}"
                ) from exc
        if not one:
            raise RecordedAnswersError(f"{path}: pass {i} is empty")
        passes.append(one)

    settings = data.get("judge")
    if not isinstance(settings, dict):
        raise RecordedAnswersError(
            f"{path}: missing 'judge' block — the settings that produced these "
            "answers are not recoverable from the letters alone"
        )

    return RecordedAnswers(video_id=video_id, passes=passes, judge_settings=settings)


def collect_pass_answers(
    answers_dir: str | Path,
    *,
    video_id: str,
    repeats: int,
) -> List[Mapping[Tuple[str, int, str], str]]:
    """Read back the per-pass answer files a run wrote, in pass order.

    `run_causal_step_for_video` writes pass 0 as ``<video_id>_answers.json``
    and pass N as ``<video_id>_answers.rN.json``; this is the inverse, so a
    recording is assembled from a real run rather than a re-implementation of
    the judging loop.
    """
    from ._causal_step_io import load_answers_as_map

    answers_dir = Path(answers_dir)
    out: List[Mapping[Tuple[str, int, str], str]] = []
    for i in range(repeats):
        suffix = "" if i == 0 else f".r{i}"
        path = answers_dir / f"{video_id}_answers{suffix}.json"
        if not path.is_file():
            raise RecordedAnswersError(f"missing pass {i}: expected {path}")
        out.append(load_answers_as_map(str(path)))
    return out


def write_recorded_answers(
    path: str | Path,
    *,
    passes: List[Mapping[Tuple[str, int, str], str]],
    video_id: str,
    judge_settings: Mapping[str, Any],
) -> None:
    """Write a `judge_answers.json` from N runner answer maps.

    `passes` entries are keyed the way `run_causal_step_for_video` returns
    them — ``(video_id, segment_index, qa_type)`` — so a recording can be
    assembled straight from real run output with no reshaping.
    """
    if not passes:
        raise RecordedAnswersError("need at least one pass to record")

    question_sets = [{(k[1], k[2]) for k in p} for p in passes]
    if any(qs != question_sets[0] for qs in question_sets[1:]):
        raise RecordedAnswersError(
            "every pass must answer the same questions; got "
            f"{[len(qs) for qs in question_sets]} questions across passes"
        )

    payload = {
        "video_id": video_id,
        "judge": dict(judge_settings),
        "passes": [
            {
                "video_id": video_id,
                "answers": [
                    {
                        "video_id": vid,
                        "segment_index": int(seg),
                        "qa_type": qa_type,
                        "answer": letter,
                    }
                    for (vid, seg, qa_type), letter in sorted(
                        p.items(), key=lambda kv: (kv[0][1], kv[0][2])
                    )
                ],
            }
            for p in passes
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def check_recording_matches_judge(
    recorded: RecordedAnswers, live_settings: Mapping[str, Any]
) -> None:
    """Refuse to pair a recording with a live judge configured differently.

    The reference side replays the recording while the candidate side runs the
    live judge, so any pinned setting that differs (model, frame count, prompt)
    makes the two sides answer under different protocols and the delta
    measures the config change instead of the world model.
    """
    mismatched = [
        f"{name}: recorded {recorded.judge_settings.get(name)!r}, "
        f"live {live_settings.get(name)!r}"
        for name in PINNED_JUDGE_FIELDS
        if recorded.judge_settings.get(name) != live_settings.get(name)
    ]
    if mismatched:
        raise RecordedAnswersError(
            f"the recording for {recorded.video_id!r} was made with a different "
            f"judge configuration ({'; '.join(mismatched)}). Re-record it with "
            "tools/dataset/record_causal_reference_answers.py, or restore the settings."
        )


class RecordedJudge:
    """A `Judge` that replays `RecordedAnswers` and never calls a model."""

    def __init__(self, recorded: RecordedAnswers) -> None:
        self._recorded = recorded

    @property
    def num_passes(self) -> int:
        return self._recorded.num_passes

    def answer(self, request: JudgeRequest) -> str:
        repeat = int(request.repeat_index)
        if not 0 <= repeat < self._recorded.num_passes:
            raise RecordedAnswersError(
                f"no recorded pass for repeat {repeat}: the recording for "
                f"{self._recorded.video_id!r} holds {self._recorded.num_passes} "
                "pass(es). Re-record with more passes, or lower the repeat count."
            )
        key = (int(request.segment_index), str(request.qa_type))
        try:
            return self._recorded.passes[repeat][key]
        except KeyError:
            raise RecordedAnswersError(
                f"no recorded answer for segment {key[0]} {key[1]!r} in pass "
                f"{repeat} of {self._recorded.video_id!r}. The recording is stale "
                "with respect to the QA dataset."
            ) from None
