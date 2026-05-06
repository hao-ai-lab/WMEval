from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ._causal_step_judge import JudgeParseError, JudgeRequest, parse_letter

logger = logging.getLogger(__name__)


# Provider contract: hand back exactly `num_samples` evenly-spaced uint8 frames
# from `[start_frame, end_frame)`. Implementations are free to seek inside the
# video file rather than decoding the whole thing into RAM (see
# `evaluation.modules.causal_step._read_window_uint8`).
FramesProvider = Callable[..., np.ndarray]  # (path, *, start_frame, end_frame, num_samples)


def sample_segment_frames(
    frames: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    num_samples: int = 4,
) -> np.ndarray:
    """Pick `num_samples` evenly-spaced frames from ``[start_frame, end_frame)``.

    Tolerates a segment window that extends past the video (e.g. a world-model
    clip shorter than its Unity reference) by clamping both bounds into
    ``[0, len(frames))``. Raises ``ValueError`` only when the segment falls
    entirely outside the available frames — silent sampling there would hand
    the judge frames from the wrong time window.

    Kept as a public helper so test/in-memory FramesProviders can sample after
    decoding the entire array. Disk-backed providers should sample directly
    via seek (see `_read_window_uint8` in `causal_step.py`).
    """
    if end_frame <= start_frame:
        raise ValueError(f"bad segment: {start_frame}..{end_frame}")
    n = len(frames)
    if n == 0:
        raise ValueError("cannot sample from an empty frame array")
    if start_frame >= n:
        raise ValueError(
            f"segment [{start_frame}, {end_frame}) is entirely outside video of length {n}"
        )
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, n)
    idx = np.linspace(start_frame, end_frame - 1, num_samples).round().astype(int)
    return frames[idx]


def _encode_png_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _is_newer_openai_model(model: str) -> bool:
    """Return True for OpenAI models that reject `max_tokens`/`temperature`.

    gpt-5 family and the o-series reasoning models replaced `max_tokens` with
    `max_completion_tokens` and stopped accepting non-default `temperature`.
    Match by prefix so we cover gpt-5 / gpt-5.4 / gpt-5.4-mini / o1 / o3 etc.
    without needing to enumerate every variant.
    """
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def build_prompt(req: JudgeRequest) -> str:
    choices_block = "\n".join(req.choices)
    return (
        f"You are watching a short video segment (frames {req.segment_frames_start} "
        f"to {req.segment_frames_end}, roughly {req.start_sec:.2f}s to "
        f"{req.end_sec:.2f}s).\n\n"
        f"Question ({req.qa_type}):\n{req.question}\n\n"
        f"Choices:\n{choices_block}\n\n"
        f"Answer with exactly one letter (A, B, C, ...). No explanation."
    )


def _cache_key(
    req: JudgeRequest,
    *,
    model: str,
    prompt: str,
    num_frame_samples: int,
    temperature: float,
    prompt_version: str,
) -> str:
    """Hash everything that can change the model's answer.

    Including `num_frame_samples`, `temperature`, and `prompt_version` is the
    point: bumping any of these must invalidate cached entries, otherwise reruns
    silently keep returning answers from the old configuration.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(req.video_path.encode())
    h.update(f"|{req.segment_index}|{req.qa_type}".encode())
    h.update(b"\0")
    h.update(prompt.encode())
    h.update(b"\0")
    h.update(f"n={num_frame_samples}|t={temperature:.6f}|pv={prompt_version}".encode())
    return h.hexdigest()[:32]


@dataclass
class OpenAIJudge:
    client: Any  # openai.OpenAI
    model: str
    cache_dir: Path
    frames_provider: FramesProvider
    num_frame_samples: int = 4
    temperature: float = 0.0
    max_tokens: int = 8
    prompt_version: str = "v1"
    parse_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def answer(self, request: JudgeRequest) -> str:
        prompt = build_prompt(request)
        key = _cache_key(
            request,
            model=self.model,
            prompt=prompt,
            num_frame_samples=self.num_frame_samples,
            temperature=self.temperature,
            prompt_version=self.prompt_version,
        )
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["letter"]

        picks = self.frames_provider(
            request.video_path,
            start_frame=request.segment_frames_start,
            end_frame=request.segment_frames_end,
            num_samples=self.num_frame_samples,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in picks:
            b64 = _encode_png_b64(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        # gpt-5 family / o-series reject `max_tokens` and `temperature`; route
        # to `max_completion_tokens` and omit temperature on those models. For
        # gpt-5 we also bump the cap because reasoning/internal tokens count
        # toward `max_completion_tokens`, and `8` was already tight even for
        # gpt-4o on edge cases.
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if _is_newer_openai_model(self.model):
            request_kwargs["max_completion_tokens"] = max(self.max_tokens, 1024)
        else:
            request_kwargs["temperature"] = self.temperature
            request_kwargs["max_tokens"] = self.max_tokens
        resp = self.client.chat.completions.create(**request_kwargs)
        raw = resp.choices[0].message.content or ""
        try:
            letter = parse_letter(raw, choices=request.choices)
        except JudgeParseError as exc:
            self.parse_failures += 1
            logger.warning(
                "OpenAIJudge: failed to parse letter for video=%s segment=%s qa=%s: %s "
                "(raw=%r); not caching, future retries can succeed",
                request.video_path,
                request.segment_index,
                request.qa_type,
                exc,
                raw,
            )
            # Don't cache an unparseable response — a retransient model glitch
            # would otherwise lock the dataset to a permanently empty answer.
            return ""

        cache_file.write_text(
            json.dumps(
                {
                    "letter": letter,
                    "raw": raw,
                    "model": self.model,
                    "segment_index": request.segment_index,
                    "qa_type": request.qa_type,
                    "video_path": request.video_path,
                    "num_frame_samples": self.num_frame_samples,
                    "temperature": self.temperature,
                    "prompt_version": self.prompt_version,
                }
            )
        )
        return letter
