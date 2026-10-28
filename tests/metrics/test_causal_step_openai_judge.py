from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT / "imports" / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

from evaluation.modules._causal_step_judge import JudgeRequest
from evaluation.modules._causal_step_openai_judge import (
    OpenAIJudge,
    build_prompt,
    sample_segment_frames,
)


def _fake_frames(n: int = 60) -> np.ndarray:
    return (np.random.default_rng(0).random((n, 64, 64, 3)) * 255).astype("uint8")


def test_sample_segment_frames_picks_four_evenly():
    frames = _fake_frames(60)
    picks = sample_segment_frames(frames, start_frame=10, end_frame=50, num_samples=4)
    assert picks.shape == (4, 64, 64, 3)


def test_sample_segment_frames_clamps_negative_start():
    """Defensive: a slightly-negative start (rounding drift) should clamp to 0."""
    frames = _fake_frames(20)
    picks = sample_segment_frames(frames, start_frame=-2, end_frame=10, num_samples=4)
    assert picks.shape == (4, 64, 64, 3)


def test_sample_segment_frames_clamps_end_past_video():
    """WM video may be shorter than Unity. end_frame past the end should clamp,
    and as long as start is still valid the sample should succeed."""
    frames = _fake_frames(30)
    picks = sample_segment_frames(frames, start_frame=20, end_frame=100, num_samples=4)
    assert picks.shape == (4, 64, 64, 3)


def test_sample_segment_frames_raises_when_segment_entirely_past_video():
    """If both start and end are past the last frame (segment not in the video at all)
    we must raise, not silently sample garbage frames."""
    frames = _fake_frames(20)
    with pytest.raises(ValueError, match="segment.*outside"):
        sample_segment_frames(frames, start_frame=50, end_frame=80, num_samples=4)


def test_build_prompt_contains_question_and_choices():
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="Why does the beam fall?",
        choices=["A) ball hits it", "B) wind", "C) magic"],
        start_sec=0.5,
        end_sec=1.2,
        segment_frames_start=15,
        segment_frames_end=36,
    )
    prompt = build_prompt(req)
    assert "Why does the beam fall?" in prompt
    assert "A) ball hits it" in prompt
    assert "B) wind" in prompt
    assert "Answer with exactly one letter" in prompt


def _windowed_frames_provider(num_video_frames: int = 60):
    """A test frames_provider matching the windowed signature used in production."""
    pool = _fake_frames(num_video_frames)

    def provider(path, *, start_frame, end_frame, num_samples):
        return sample_segment_frames(
            pool, start_frame=start_frame, end_frame=end_frame, num_samples=num_samples
        )

    return provider


def test_openai_judge_caches_and_parses(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The answer is B."))]
    )

    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=tmp_path / "cache",
        frames_provider=_windowed_frames_provider(60),
    )

    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="Why does the beam fall?",
        choices=["A) ball hits it", "B) wind", "C) magic"],
        start_sec=0.5,
        end_sec=1.2,
        segment_frames_start=15,
        segment_frames_end=36,
    )
    out = judge.answer(req)
    assert out == "B"
    assert fake_client.chat.completions.create.call_count == 1
    assert judge.parse_failures == 0

    # Second call should hit cache, not the API
    out2 = judge.answer(req)
    assert out2 == "B"
    assert fake_client.chat.completions.create.call_count == 1


def test_openai_judge_invalidates_cache_on_num_samples_change(tmp_path: Path):
    """Bumping num_frame_samples must change the cache key. Otherwise reruns
    silently keep the old answer that was based on fewer/different frames."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))]
    )
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        segment_frames_start=0,
        segment_frames_end=30,
    )
    cache_dir = tmp_path / "cache"

    j4 = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=cache_dir,
        frames_provider=_windowed_frames_provider(60),
        num_frame_samples=4,
    )
    j4.answer(req)
    assert fake_client.chat.completions.create.call_count == 1

    j8 = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=cache_dir,
        frames_provider=_windowed_frames_provider(60),
        num_frame_samples=8,
    )
    j8.answer(req)
    # Different num_frame_samples must miss the cache and call the API again.
    assert fake_client.chat.completions.create.call_count == 2


def test_openai_judge_uses_max_completion_tokens_for_gpt5(tmp_path: Path):
    """gpt-5 family rejects `max_tokens`/`temperature` in chat.completions.
    The judge must send `max_completion_tokens` and omit temperature, or the
    real API returns 400 BadRequest before we even get to parse the answer."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))]
    )

    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-5.4-mini",
        cache_dir=tmp_path / "cache",
        frames_provider=_windowed_frames_provider(60),
    )
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        segment_frames_start=0,
        segment_frames_end=30,
    )
    judge.answer(req)

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_openai_judge_uses_legacy_params_for_gpt4(tmp_path: Path):
    """gpt-4* still accepts `max_tokens`/`temperature` and is cheaper for
    short answers — don't change its request shape."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))]
    )

    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=tmp_path / "cache",
        frames_provider=_windowed_frames_provider(60),
    )
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        segment_frames_start=0,
        segment_frames_end=30,
    )
    judge.answer(req)

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "max_tokens" in kwargs
    assert "temperature" in kwargs
    assert "max_completion_tokens" not in kwargs


def test_openai_judge_does_not_cache_unparseable_response(tmp_path: Path):
    """An unparseable response must not poison the cache: a transient model
    glitch shouldn't lock the dataset to a permanently empty answer."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="I cannot tell."))]
    )
    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=tmp_path / "cache",
        frames_provider=_windowed_frames_provider(60),
    )
    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        segment_frames_start=0,
        segment_frames_end=30,
    )
    out = judge.answer(req)
    assert out == ""
    assert judge.parse_failures == 1

    # Second call should re-hit the API (no cache entry written).
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))]
    )
    out2 = judge.answer(req)
    assert out2 == "A"
    assert fake_client.chat.completions.create.call_count == 2


def test_cache_key_unchanged_for_default_repeat_index():
    """The existing on-disk answer cache holds real, paid-for judge responses.

    Adding repeat support must not invalidate it, so the key for a request at
    the default repeat index is pinned here. If this constant changes, every
    cached answer under runs/causal_step_cache silently becomes a miss.
    """
    from evaluation.modules._causal_step_openai_judge import _cache_key

    req = JudgeRequest(
        video_path="x.mp4",
        segment_index=1,
        qa_type="causal",
        question="q",
        choices=["A) yes", "B) no"],
        start_sec=0.0,
        end_sec=1.0,
        segment_frames_start=0,
        segment_frames_end=30,
    )
    key = _cache_key(
        req,
        model="gpt-4o",
        prompt=build_prompt(req),
        num_frame_samples=8,
        temperature=0.0,
        prompt_version="v1",
    )
    assert key == "caadffd5fdea8aa6152c7093b05a7cd8"


def test_openai_judge_repeat_index_misses_cache(tmp_path: Path):
    """Each repeat must be an independent judge draw.

    The paper's per-shape reference figures (e.g. Unity CSR 60 on glancing)
    are means over five passes of the same video. If repeats shared a cache
    key every pass would replay one answer and the mean would be degenerate.
    """
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))]
    )
    cache_dir = tmp_path / "cache"

    def _req(repeat_index: int) -> JudgeRequest:
        return JudgeRequest(
            video_path="x.mp4",
            segment_index=1,
            qa_type="causal",
            question="q",
            choices=["A) yes", "B) no"],
            start_sec=0.0,
            end_sec=1.0,
            segment_frames_start=0,
            segment_frames_end=30,
            repeat_index=repeat_index,
        )

    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-4o",
        cache_dir=cache_dir,
        frames_provider=_windowed_frames_provider(60),
        num_frame_samples=8,
    )
    judge.answer(_req(0))
    assert fake_client.chat.completions.create.call_count == 1
    judge.answer(_req(1))
    assert fake_client.chat.completions.create.call_count == 2
    # Re-asking the same repeat is still a cache hit.
    judge.answer(_req(1))
    assert fake_client.chat.completions.create.call_count == 2


def _judge(tmp_path, model):
    return OpenAIJudge(
        client=MagicMock(),
        model=model,
        cache_dir=tmp_path / "cache",
        frames_provider=lambda *a, **k: np.zeros((1, 4, 4, 3), dtype=np.uint8),
        num_frame_samples=8,
    )


def test_effective_temperature_is_none_when_the_request_omits_it(tmp_path):
    """gpt-5 / o-series reject a non-default `temperature`, so `answer` never
    sends one. Reporting the configured 0.0 would claim a setting that was not
    applied -- the calls actually run at the API default."""
    judge = _judge(tmp_path, "gpt-5.4")
    assert judge.effective_temperature is None


def test_effective_temperature_is_the_configured_value_when_it_is_sent(tmp_path):
    judge = _judge(tmp_path, "gpt-4o")
    assert judge.effective_temperature == pytest.approx(0.0)


def test_describe_settings_reports_what_was_actually_applied(tmp_path):
    judge = _judge(tmp_path, "gpt-5.4")
    settings = judge.describe_settings()
    assert settings["model"] == "gpt-5.4"
    assert settings["num_frame_samples"] == 8
    assert settings["prompt_version"] == "v1"
    assert settings["temperature"] is None
    # A bare null is ambiguous between "unset" and "default"; say which.
    assert "api default" in settings["temperature_note"].lower()


def test_describe_settings_keeps_the_temperature_for_older_models(tmp_path):
    settings = _judge(tmp_path, "gpt-4o").describe_settings()
    assert settings["temperature"] == pytest.approx(0.0)
    assert "temperature_note" not in settings


def test_openai_judge_logs_token_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    usage_log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("WMEVAL_VLM_USAGE_LOG", str(usage_log))
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="A"))],
        usage={"prompt_tokens": 9400, "completion_tokens": 120,
               "completion_tokens_details": {"reasoning_tokens": 100}},
    )
    judge = OpenAIJudge(
        client=fake_client,
        model="gpt-5.4",
        cache_dir=tmp_path / "cache",
        frames_provider=_windowed_frames_provider(60),
    )
    req = JudgeRequest(
        video_path="x.mp4", segment_index=1, qa_type="descriptive",
        question="What moves?", choices=["A) ball", "B) beam"],
        start_sec=0.5, end_sec=1.2, segment_frames_start=15, segment_frames_end=36,
    )
    assert judge.answer(req) == "A"
    judge.answer(req)  # served from the cache
    records = [json.loads(line) for line in usage_log.read_text().splitlines()]
    assert len(records) == 1
    assert (records[0]["model"], records[0]["input_tokens"], records[0]["reasoning_tokens"]) == ("gpt-5.4", 9400, 100)
