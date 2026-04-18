from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from ._causal_step_judge import Judge, JudgeParseError, JudgeRequest, parse_letter


FramesProvider = Callable[[str], np.ndarray]  # path -> (N,H,W,3) uint8


def sample_segment_frames(
    frames: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    num_samples: int = 4,
) -> np.ndarray:
    """Pick `num_samples` evenly-spaced frames from [start_frame, end_frame)."""
    if end_frame <= start_frame:
        raise ValueError(f"bad segment: {start_frame}..{end_frame}")
    end_frame = min(end_frame, len(frames))
    idx = np.linspace(start_frame, end_frame - 1, num_samples).round().astype(int)
    return frames[idx]


def _encode_png_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


def _cache_key(req: JudgeRequest, model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(req.video_path.encode())
    h.update(f"|{req.segment_index}|{req.qa_type}".encode())
    h.update(b"\0")
    h.update(prompt.encode())
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

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def answer(self, request: JudgeRequest) -> str:
        prompt = build_prompt(request)
        key = _cache_key(request, self.model, prompt)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["letter"]

        frames_all = self.frames_provider(request.video_path)
        picks = sample_segment_frames(
            frames_all,
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

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content or ""
        try:
            letter = parse_letter(raw, choices=request.choices)
        except JudgeParseError:
            letter = ""

        cache_file.write_text(
            json.dumps(
                {
                    "letter": letter,
                    "raw": raw,
                    "model": self.model,
                    "segment_index": request.segment_index,
                    "qa_type": request.qa_type,
                    "video_path": request.video_path,
                }
            )
        )
        return letter
