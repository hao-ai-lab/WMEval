"""VLM-based decoration of CausalStep QA templates (hybrid mode).

The static `STAGE_TEMPLATES` provide a strong prior — the question, an
initial guess at the correct answer, and a wording style. The VLM is shown
the segment frames and may:

- rewrite the question for visual specificity,
- *confirm* the template's correct answer (return it verbatim), OR
- *correct* the template's correct answer when the frames demonstrably
  contradict it (e.g., the platform tipped left when the template said
  right),
- generate 3 distractors grounded in what the frames actually show.

This trades the previous "VLM cannot corrupt GT" guarantee for the ability
to fix templates that are wrong for a particular trajectory's footage.
`decorate_qa` logs every override so a human reviewer can spot-check; it
still rejects malformed responses (wrong distractor count, distractor
identical to the emitted correct, duplicate distractors).

Why we send only `model`, `messages`, and `response_format`:
gpt-5 family deprecated `temperature`/`max_tokens` in favor of `verbosity` /
`reasoning_effort` / `max_completion_tokens`, but the names differ across
versions. Sending only the universally-accepted params keeps the call
compatible with both gpt-4* and gpt-5* model strings; users override the
default model via the CLI flag.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_DECORATE_PROMPT_TEMPLATE = """\
You are watching a short segment of a physics simulation video (the \
"BeamDomino" scene). The segment is labeled `{stage}` and runs roughly \
{start_sec:.2f}s to {end_sec:.2f}s within the trajectory.

We are asking the following multiple-choice question about this segment:

  Q: {question}

A static template suggests this correct answer:

  TEMPLATE_CORRECT: {correct_body}

Your tasks:
1. Optionally rewrite the question to be more visually specific to what the \
frames actually show (which side, which color, what direction, etc.). The \
rewritten question MUST still be answered correctly by whatever you decide \
the correct answer is in step 2.
2. Decide the correct answer based on what the frames actually show:
   - If the frames clearly support TEMPLATE_CORRECT, return it verbatim as \
`correct`.
   - If the frames clearly contradict TEMPLATE_CORRECT (e.g., the platform \
tipped left when the template said right, or a different object made \
contact than the template named), return a corrected `correct` that \
reflects what actually happens in the frames. Keep its grammatical style \
and length similar to TEMPLATE_CORRECT.
   - When in doubt, prefer TEMPLATE_CORRECT — only override when the frames \
make the template wording demonstrably wrong.
3. Generate exactly 3 distractor options. Each distractor must:
   - Describe a *different* outcome or cause than your `correct` answer
   - Be a plausible alternative that someone might pick if they had not \
watched the frames carefully
   - Be visually disprovable from the frames (a careful viewer can rule it \
out by what they see)
   - Be roughly the same length and grammatical style as your `correct` \
answer
   - Avoid trivia like "wind", "magic", "the camera moved", "scripted timer" \
— prefer near-misses about which object moved, in what direction, in what \
order, against what

Respond with a single JSON object of the form: \
{{"question": "...", "correct": "...", "distractors": ["...", "...", "..."]}}.
"""


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "correct": {"type": "string", "minLength": 1},
        "distractors": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["question", "correct", "distractors"],
    "additionalProperties": False,
}


@dataclass
class QADecoratorResult:
    """Output of one decorator call.

    `correct` is the answer the VLM endorses after viewing the frames. It
    will normally equal the template's correct-option body verbatim, but
    the VLM is allowed to override it when the frames demonstrably
    contradict the template wording. Override events are logged by
    `decorate_qa` so a human reviewer can spot-check.
    """

    question: str
    correct: str
    distractors: list[str]


def _encode_png_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cache_key(
    *,
    model: str,
    video_id: str,
    segment_index: int,
    qa_type: str,
    prompt: str,
    num_frames: int,
) -> str:
    """Hash everything that can change the VLM's output.

    Includes `num_frames` so bumping it forces a recompute; otherwise an old
    cache entry from a 4-frame run would silently satisfy an 8-frame request.
    """
    h = hashlib.sha256()
    h.update(f"{model}|{video_id}|{segment_index}|{qa_type}|n={num_frames}".encode())
    h.update(b"\0")
    h.update(prompt.encode())
    return h.hexdigest()[:32]


def build_prompt(
    *,
    stage: str,
    start_sec: float,
    end_sec: float,
    template_question: str,
    template_correct_body: str,
) -> str:
    return _DECORATE_PROMPT_TEMPLATE.format(
        stage=stage,
        start_sec=start_sec,
        end_sec=end_sec,
        question=template_question,
        correct_body=template_correct_body,
    )


def decorate_qa(
    *,
    client: Any,
    model: str,
    frames: np.ndarray,
    stage: str,
    qa_type: str,
    start_sec: float,
    end_sec: float,
    template_question: str,
    template_correct_body: str,
    cache_dir: Path,
    video_id: str,
    segment_index: int,
) -> QADecoratorResult:
    """Ask the VLM to rewrite a single QA. Cached on disk.

    The VLM may either confirm the template's correct answer (most common,
    return it verbatim) or override it when the frames clearly contradict
    the template wording. Overrides are logged at WARNING level so a human
    reviewer can spot-check.

    Raises if the VLM returns malformed JSON, the wrong number of
    distractors, a distractor that is verbatim-identical to the *emitted*
    correct answer, or duplicate distractors.
    """
    prompt = build_prompt(
        stage=stage,
        start_sec=start_sec,
        end_sec=end_sec,
        template_question=template_question,
        template_correct_body=template_correct_body,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        model=model,
        video_id=video_id,
        segment_index=segment_index,
        qa_type=qa_type,
        prompt=prompt,
        num_frames=len(frames),
    )
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        # Older cache entries (pre-correction-mode) lack `correct`; fall back
        # to the template body so they remain usable instead of forcing a
        # full recompute. New entries always include `correct`.
        cached_correct = data.get("correct", template_correct_body)
        return QADecoratorResult(
            question=data["question"],
            correct=cached_correct,
            distractors=list(data["distractors"]),
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        b64 = _encode_png_b64(frame)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "decorated_qa",
                "strict": True,
                "schema": _RESPONSE_SCHEMA,
            },
        },
    )
    raw = resp.choices[0].message.content or ""
    parsed = json.loads(raw)

    question = str(parsed["question"]).strip()
    correct = str(parsed["correct"]).strip()
    distractors = [str(d).strip() for d in parsed["distractors"]]
    if len(distractors) != 3:
        raise ValueError(
            f"VLM returned {len(distractors)} distractors, expected exactly 3"
        )
    correct_norm = correct.lower()
    for d in distractors:
        if d.lower() == correct_norm:
            raise ValueError(
                f"VLM emitted a distractor identical to the correct answer: {d!r}"
            )
    if len({d.lower() for d in distractors}) != 3:
        raise ValueError(f"VLM emitted duplicate distractors: {distractors!r}")

    if correct.strip().lower() != template_correct_body.strip().lower():
        logger.warning(
            "VLM overrode template correct answer for video=%s segment=%s "
            "qa_type=%s: template=%r -> vlm=%r",
            video_id,
            segment_index,
            qa_type,
            template_correct_body,
            correct,
        )

    cache_file.write_text(
        json.dumps(
            {
                "model": model,
                "video_id": video_id,
                "segment_index": segment_index,
                "qa_type": qa_type,
                "stage": stage,
                "question": question,
                "correct": correct,
                "distractors": distractors,
                "template_correct_body": template_correct_body,
                "num_frames": len(frames),
            },
            indent=2,
        )
    )
    return QADecoratorResult(
        question=question,
        correct=correct,
        distractors=distractors,
    )
