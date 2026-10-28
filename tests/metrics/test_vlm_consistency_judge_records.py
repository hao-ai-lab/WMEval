from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.modules.vlm_consistency import vlm_judge
from evaluation.modules.vlm_consistency.providers import openai as openai_provider
from evaluation.modules.vlm_consistency.schemas import Rubric


def _rubric() -> Rubric:
    return Rubric.model_validate(
        {
            "scene_name": "SceneB_first",
            "prompt_text": "A red car drives around a track.",
            "rubric_items": [
                {"rubric_id": "env_01", "agent": "environment", "rubric_text": "Is there a track?"},
                {"rubric_id": "obj_01", "agent": "object", "rubric_text": "Is the car red?"},
            ],
            "metadata": {
                "generator_provider": "openai",
                "generator_model": "gpt-5.4",
                "generated_at": "2026-01-01T00:00:00Z",
                "num_candidates_generated": 2,
            },
        }
    )


def _fake_provider() -> SimpleNamespace:
    def call_vlm(*, user_text: str, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        rubric_id = "env_01" if "### env_01" in user_text else "obj_01"
        record.update({"model_resolved": "gpt-5.4-2026-03-05", "response_id": f"resp_{rubric_id}"})
        return {
            "item_scores": [
                {
                    "rubric_id": rubric_id,
                    "reasoning": "visible",
                    "frame_evidence": ["frame_t=0.0s"],
                    "judged_score": 3,
                }
            ],
            "explanations": {"summary": "ok", "issues": []},
        }

    return SimpleNamespace(call_vlm=call_vlm, DEFAULT_MODEL="gpt-5.4")


def test_judge_scene_records_every_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_judge, "_get_provider_module", lambda provider: _fake_provider())
    frames = [
        ("ref_frame", base64.b64encode(b"ref image").decode("ascii")),
        ("frame_t=0.0s", base64.b64encode(b"frame zero").decode("ascii")),
    ]

    result = vlm_judge.judge_scene(
        _rubric(), frames, provider="openai", model="gpt-5.4", api_key="test", alpha=0.6
    )

    assert result.overall_score == 1.0
    calls = {call.agent: call for call in result.judge_calls}
    assert set(calls) == {"environment", "object"}
    for agent, call in calls.items():
        assert call.request["model"] == "gpt-5.4"
        assert call.request["system_prompt_sha256"] == vlm_judge.SYSTEM_PROMPT_SHA256[agent]
        assert "A red car drives around a track." in call.request["user_text"]
        # image payloads are kept as hashes of the decoded bytes, not inline
        assert call.request["images"] == [
            {"label": "ref_frame", "sha256": hashlib.sha256(b"ref image").hexdigest()},
            {"label": "frame_t=0.0s", "sha256": hashlib.sha256(b"frame zero").hexdigest()},
        ]
        assert call.response["model_resolved"] == "gpt-5.4-2026-03-05"
    assert calls["object"].response["response_id"] == "resp_obj_01"
    assert result.model_dump()["judge_calls"][0]["agent"] in {"environment", "object"}


def test_judge_scene_skips_the_call_for_an_agent_without_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_judge, "_get_provider_module", lambda provider: _fake_provider())
    rubric = _rubric()
    rubric.rubric_items = [item for item in rubric.rubric_items if item.agent == "environment"]

    result = vlm_judge.judge_scene(
        rubric, [("frame_t=0.0s", "Zg==")], provider="openai", api_key="test"
    )

    assert [call.agent for call in result.judge_calls] == ["environment"]


class _Response:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def json(self) -> dict[str, Any]:
        return {
            "id": "resp_123",
            "model": "gpt-5.4-2026-03-05",
            "created_at": 1760000000,
            "output_text": json.dumps({"answer": 1}),
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }


def test_openai_call_vlm_fills_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_provider.requests, "post", lambda *a, **k: _Response())
    monkeypatch.setattr(openai_provider, "_log_usage", lambda *a, **k: None)
    kwargs = dict(
        system_prompt="s", user_text="u", images_b64=[], schema={}, api_key="k", model="gpt-5.4"
    )

    record: dict[str, Any] = {}
    assert openai_provider.call_vlm(**kwargs, record=record) == {"answer": 1}
    assert record == {
        "provider": "openai",
        "model_requested": "gpt-5.4",
        "model_resolved": "gpt-5.4-2026-03-05",
        "response_id": "resp_123",
        "created_at": 1760000000,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "attempts": 1,
        "output_text": '{"answer": 1}',
    }
    # callers that do not ask for a record are unaffected
    assert openai_provider.call_vlm(**kwargs) == {"answer": 1}
