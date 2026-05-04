"""Generate a CausalStep QA dataset draft for a BeamDomino trajectory.

Reads a segments JSON produced by `detect_beamdomino_segments` (top-level
`events` list with `{index, name, start_sec, end_sec, ...}`) and emits a
CausalStep-loadable draft using per-stage templates. Output must be reviewed
by a human before evaluation.

Template wording is grounded in the canonical BeamDomino scene: 
one ball rolls along the floor, strikes a supporting leg 
of a small white platform that holds another ball on top,
the platform tips over, and a row of dominoes near it then falls in a
chain. Distractors are deliberately phrased as plausible near-misses
(mirror direction, wrong contact location, simultaneous-vs-sequential
fall) so the VLM judge cannot rule them out from text alone — it has to
look at the frames.

Usage:
    python tools/generate_beamdomino_causal_step_qa.py \\
        --segments path/to/segments.json \\
        --video-id trajectory_001 \\
        --out path/to/qa_draft.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# VLM mode lazy-imports `evaluation.modules.causal_step._read_window_uint8`,
# which lives at the repo root. Mirror the other tools' bootstrap so the
# import resolves regardless of cwd or whether the script is invoked as
# `python tools/...` vs `python -m tools.generate_beamdomino_causal_step_qa`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LETTER_PREFIX = re.compile(r"^[A-Z]\)\s*")


def _seed_for(video_id: str, segment_index: int, qa_type: str) -> int:
    """Deterministic per-QA seed.

    Same `(video_id, segment_index, qa_type)` triple always yields the same
    shuffle — important so the OpenAI judge's prompt (and therefore its cache
    key) doesn't change between reruns on the same dataset.
    """
    digest = hashlib.sha256(
        f"{video_id}|{segment_index}|{qa_type}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def _shuffled_choices(
    choices: list[str], correct_letter: str, *, seed: int
) -> tuple[list[str], str]:
    """Reorder multiple-choice options and recompute the correct letter.

    Templates author every option as `correct_answer="A"` for ergonomics,
    which would otherwise plant a positional bias the VLM judge can exploit
    (and which the README's review checklist used to require humans to fix
    by hand). We strip the existing `A)`/`B)`/... prefix, permute the bodies
    via a seeded RNG, re-letter from `A`, and report which letter the
    originally-correct body now sits at.
    """
    correct_idx = ord(correct_letter) - ord("A")
    if not 0 <= correct_idx < len(choices):
        raise ValueError(
            f"correct_letter {correct_letter!r} out of range for {len(choices)} choices"
        )
    bodies = [_LETTER_PREFIX.sub("", c, count=1) for c in choices]

    rng = random.Random(seed)
    perm = list(range(len(bodies)))
    rng.shuffle(perm)

    new_bodies = [bodies[perm[i]] for i in range(len(perm))]
    new_choices = [f"{chr(ord('A') + i)}) {b}" for i, b in enumerate(new_bodies)]
    new_correct = chr(ord("A") + perm.index(correct_idx))
    return new_choices, new_correct


STAGE_TEMPLATES: dict[str, dict[str, Any]] = {
    "rolling": {
        "descriptive": (
            "What is the ball on the floor doing during this segment, and how "
            "does it move relative to the rest of the scene?"
        ),
        "descriptive_choices": [
            "A) rolling steadily along the floor toward the central white "
            "platform that has another ball resting on top of it",
            "B) bouncing repeatedly in place on the floor with no horizontal "
            "movement",
            "C) rolling away from the platform, back out toward the foreground "
            "of the scene",
            "D) sitting motionless on the floor next to a row of dominoes",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why is the ball on the floor moving the way it is in this "
            "segment, when nothing else has visibly happened yet?"
        ),
        "causal_choices": [
            "A) it was given an initial horizontal velocity at the very start "
            "of the trajectory, before any other event in the scene",
            "B) a domino has already tipped over and pushed it forward",
            "C) the ball that was sitting on top of the platform jumped down "
            "and shoved it",
            "D) the floor itself is sloped, so the ball is sliding under "
            "gravity rather than rolling under its own momentum",
        ],
        "causal_correct": "A",
    },
    "impact": {
        "descriptive": (
            "What is happening between the rolling floor ball and the white "
            "platform during this segment?"
        ),
        "descriptive_choices": [
            "A) the floor ball collides with one of the platform's vertical "
            "supporting legs (not the top surface)",
            "B) the floor ball passes harmlessly underneath the platform "
            "without touching any part of it",
            "C) the floor ball stops a short distance away from the platform "
            "and never makes contact with it",
            "D) the floor ball bounces upward onto the top of the platform "
            "and comes to rest next to the ball already there",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why does the white platform (and the ball still resting on top "
            "of it) begin to move in this segment?"
        ),
        "causal_choices": [
            "A) the floor ball transferred momentum into one of the platform's "
            "supporting legs, removing support from that side and destabilizing "
            "the platform above it",
            "B) the ball resting on top of the platform shifted on its own "
            "and unbalanced the platform without any contact from below",
            "C) one of the dominoes far across the scene fell first and "
            "somehow disturbed the platform from a distance",
            "D) the platform is scripted to fall on a fixed timer regardless "
            "of whether anything actually hits it",
        ],
        "causal_correct": "A",
    },
    "topple": {
        "descriptive": (
            "What is the white platform — with the ball still on top of it — "
            "doing during this segment?"
        ),
        "descriptive_choices": [
            "A) tipping over toward one side as it falls, while the ball that "
            "was resting on top rolls off the tilting surface",
            "B) lifting straight upward off the floor without any rotation",
            "C) rotating in place around its own vertical axis while staying "
            "upright on its legs",
            "D) staying perfectly horizontal while only the ball on top hops "
            "up and down on its surface",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why is the platform tipping over toward this particular side, "
            "and not the opposite one?"
        ),
        "causal_choices": [
            "A) the supporting leg on this side was knocked out earlier, so "
            "the platform pivots over the legs that are still standing and "
            "falls toward the now-unsupported side under gravity",
            "B) the ball on top rolled across the surface first and weighted "
            "this side, causing the whole platform to tip that way",
            "C) a gust of wind from the opposite side pushed it over",
            "D) the dominoes on this side reached out and physically pulled "
            "the platform down toward themselves",
        ],
        "causal_correct": "A",
    },
    "cascade": {
        "descriptive": (
            "What are the dominoes near the fallen platform doing during "
            "this segment, and in what pattern?"
        ),
        "descriptive_choices": [
            "A) the closest domino is knocked over and falls into its "
            "neighbor, which then tips the next one, propagating a chain "
            "reaction along the row",
            "B) every domino in the line tips over at the same instant, in "
            "perfect unison rather than one after another",
            "C) the dominoes remain upright; only the balls and the platform "
            "are still moving at this point",
            "D) the dominoes lift off the floor and float upward instead of "
            "tipping over sideways",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why are the dominoes falling in this chain reaction during "
            "this segment?"
        ),
        "causal_choices": [
            "A) the falling platform — and/or the ball that rolled off the "
            "top of it — struck the first domino, whose fall then knocked "
            "over its neighbor, propagating the impulse along the row",
            "B) the floor ball rolled directly into the line of dominoes "
            "itself, with no involvement from the platform or the ball that "
            "had been on top",
            "C) every domino was animated to fall on a fixed timer, "
            "independent of any physical contact in the scene",
            "D) the floor tilted and the dominoes simply slid sideways "
            "together without ever tipping over",
        ],
        "causal_correct": "A",
    },
    # ── glancing-shape stages ────────────────────────────────────────────
    # `glance`: ball touches a leg/roof of the platform but the platform
    # remains standing — momentum transfer was insufficient to tip it. Used
    # as segment 2 in the glancing chain.
    "glance": {
        "descriptive": (
            "What happens between the rolling floor ball and the platform "
            "during this segment?"
        ),
        "descriptive_choices": [
            "A) the floor ball brushes against the platform but the platform "
            "stays standing on its legs without tipping over",
            "B) the floor ball cleanly knocks one of the legs out and the "
            "platform begins to fall in this segment",
            "C) the floor ball stops cleanly when it touches the platform "
            "and never leaves the platform's footprint",
            "D) the floor ball passes underneath the platform without "
            "touching any leg or surface",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why does the platform remain standing during this segment, "
            "even though the floor ball makes contact with it?"
        ),
        "causal_choices": [
            "A) the contact is a glancing one — the ball brushes the platform "
            "without delivering enough horizontal force to remove a leg's "
            "support, so the platform stays balanced",
            "B) one supporting leg has been removed but the ball on top "
            "shifts to the opposite side and re-balances the platform",
            "C) the platform is rigidly anchored to the floor and cannot tip "
            "regardless of what hits it",
            "D) the ball reverses direction on contact and pushes the "
            "platform back into place before it can tilt",
        ],
        "causal_correct": "A",
    },
    # `cascade_from_strike`: same observable chain reaction as `cascade`,
    # but the *cause* is the direct ball-on-domino hit, not a falling
    # platform. The descriptive QA matches `cascade` (chain reaction is the
    # same regardless of how it started), but the causal QA pins the cause
    # to the ball's earlier direct strike.
    "cascade_from_strike": {
        "descriptive": (
            "What are the dominoes near the platform doing during this "
            "segment, and in what pattern?"
        ),
        "descriptive_choices": [
            "A) the closest domino is knocked over and falls into its "
            "neighbor, which then tips the next one, propagating a chain "
            "reaction along the row",
            "B) every domino in the line tips over at the same instant, in "
            "perfect unison rather than one after another",
            "C) the dominoes remain upright; only the balls and the platform "
            "are still moving at this point",
            "D) the dominoes lift off the floor and float upward instead of "
            "tipping over sideways",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why are the dominoes falling in this chain reaction during "
            "this segment?"
        ),
        "causal_choices": [
            "A) the rolling floor ball had earlier struck the first domino "
            "directly, and that fall is now tipping each successive piece "
            "along the row",
            "B) the platform tipped over earlier and its falling surface "
            "knocked the first domino, propagating the impulse along the row",
            "C) every domino was animated to fall on a fixed timer, "
            "independent of any physical contact in the scene",
            "D) the floor tilted and the dominoes simply slid sideways "
            "together without ever tipping over",
        ],
        "causal_correct": "A",
    },
    # `strike`: ball hits a domino directly (no platform involvement in
    # this segment). Used as segment 2 in direct_strike and segment 3 in
    # glancing chains.
    "strike": {
        "descriptive": (
            "What contact occurs during this segment, and which object does "
            "the floor ball strike?"
        ),
        "descriptive_choices": [
            "A) the floor ball collides directly with a domino in the row, "
            "without first knocking the platform over into it",
            "B) the falling platform — having been tipped earlier in the "
            "trajectory — strikes the first domino in this segment",
            "C) the floor ball stops short of the dominoes and never makes "
            "contact with any of them in this segment",
            "D) the ball that had been on top of the platform drops onto a "
            "domino on its own, with no contact from the floor ball",
        ],
        "descriptive_correct": "A",
        "causal_question": (
            "Why does the first domino begin to fall during this segment?"
        ),
        "causal_choices": [
            "A) the rolling floor ball reached the row and struck a domino "
            "directly, transferring its momentum into that standing piece",
            "B) the platform tipped over earlier and its falling surface "
            "knocked into the first domino during this segment",
            "C) the ball that had been resting on top of the platform "
            "rolled off the platform and hit the dominoes itself",
            "D) the dominoes are scripted to begin falling on a fixed timer "
            "once the trajectory starts, without any direct contact",
        ],
        "causal_correct": "A",
    },
}


# trajectory_shape -> ordered list of stage names per segment index. The
# generator no longer trusts the per-segment `name` field on the segments
# JSON (that came from the detector); it instead picks the stage from this
# table based on `trajectory_shape`. That keeps the QA semantics tied to
# the *shape*, not to whatever stage label the detector happened to emit.
SHAPE_STAGE_CHAINS: dict[str, list[str]] = {
    "canonical":     ["rolling", "impact", "topple",  "cascade"],
    "glancing":      ["rolling", "glance", "strike",  "cascade_from_strike"],
    "direct_strike": ["rolling", "strike", "cascade_from_strike"],
}


# Decorator contract: called once per QA when in VLM mode. Receives the stage
# label, qa kind, segment metadata, and the original template's question +
# correct option body; returns the rewritten question, the correct body
# (which the decorator may override based on what the frames show), and 3
# distractors. Overrides are logged. See `tools/_causal_qa_vlm.py:decorate_qa`
# for the production implementation.
QADecorator = Callable[..., Any]


def _build_qa(
    *,
    template_question: str,
    template_choices: list[str],
    template_correct: str,
    video_id: str,
    segment_index: int,
    qa_kind: str,
    seg: dict[str, Any],
    stage: str,
    decorator: Optional[QADecorator],
) -> dict[str, Any]:
    """Assemble one QA dict, optionally routing through the VLM decorator.

    With `decorator=None` we use the template question, the template's
    correct answer, and the template distractors as-is. With a decorator,
    the question, the correct answer, and the 3 distractors all come from
    the VLM after it has viewed the segment frames. The decorator is
    expected to start from the template's correct body and only override
    it when the frames demonstrably contradict it; overrides are logged to
    stderr so human review remains tractable.
    """
    correct_idx = ord(template_correct) - ord("A")
    template_correct_body = template_choices[correct_idx].split(") ", 1)[1]

    if decorator is not None:
        result = decorator(
            stage=stage,
            qa_type=qa_kind,
            segment_index=segment_index,
            start_sec=float(seg["start_sec"]),
            end_sec=float(seg["end_sec"]),
            template_question=template_question,
            template_correct_body=template_correct_body,
        )
        question = result.question
        correct_body = result.correct
        distractor_bodies = list(result.distractors)
        if correct_body.strip().lower() != template_correct_body.strip().lower():
            print(
                f"[qa-gen] VLM overrode template correct answer "
                f"video_id={video_id} segment={segment_index} qa={qa_kind}: "
                f"template={template_correct_body!r} -> vlm={correct_body!r}",
                file=sys.stderr,
            )
    else:
        question = template_question
        correct_body = template_correct_body
        distractor_bodies = [
            ch.split(") ", 1)[1]
            for ch in template_choices
            if ch.split(") ", 1)[1] != correct_body
        ]

    pre_shuffle = [f"A) {correct_body}"] + [
        f"{chr(ord('A') + 1 + i)}) {body}" for i, body in enumerate(distractor_bodies)
    ]
    choices, correct_letter = _shuffled_choices(
        pre_shuffle, "A", seed=_seed_for(video_id, segment_index, qa_kind)
    )
    return {
        "segment_index": segment_index,
        "question": question,
        "choices": choices,
        "correct_answer": correct_letter,
    }


def build_dataset(
    *,
    video_id: str,
    events: list[dict[str, Any]],
    decorator: Optional[QADecorator] = None,
    trajectory_shape: Optional[str] = None,
) -> dict[str, Any]:
    """Build a CausalStep-schema dataset from a segments.json `events` list.

    CausalStep schema requires: 1-indexed contiguous segments, descriptive QA
    for every segment, causal QA for segments 2..n, and `correct_answer` keys.

    Choices are shuffled per QA via `_shuffled_choices` so the correct letter
    is no longer always `A`; the shuffle is seeded by `(video_id,
    segment_index, qa_type)` and is therefore stable across reruns.

    When `trajectory_shape` is provided, the per-segment stage is picked from
    `SHAPE_STAGE_CHAINS[shape]` rather than from each event's `name` field.
    This decouples the QA semantics from the detector's segment labels and
    is the supported way to wire shape-specific question chains. When
    `trajectory_shape` is None (legacy callers, including the existing test
    fixture), each event's `name` field selects the template directly.

    When `decorator` is provided (VLM mode), each question, its correct
    answer, and its 3 distractors are written by the model based on the
    segment's frames. The model is given the template's correct answer as
    a strong prior; if it overrides the prior (because the frames clearly
    contradict the template wording), `_build_qa` logs the override to
    stderr so a human reviewer can spot-check the change.
    """
    sorted_events = sorted(events, key=lambda e: e["index"])
    if trajectory_shape is not None:
        if trajectory_shape not in SHAPE_STAGE_CHAINS:
            raise ValueError(
                f"unknown trajectory_shape {trajectory_shape!r}; "
                f"expected one of {sorted(SHAPE_STAGE_CHAINS)}"
            )
        chain = SHAPE_STAGE_CHAINS[trajectory_shape]
        if len(chain) != len(sorted_events):
            raise ValueError(
                f"trajectory_shape {trajectory_shape!r} expects {len(chain)} "
                f"segments but got {len(sorted_events)}"
            )
    else:
        chain = None

    out_segments: list[dict[str, Any]] = []
    desc_qas: list[dict[str, Any]] = []
    causal_qas: list[dict[str, Any]] = []

    for seg in sorted_events:
        idx = int(seg["index"])
        stage = chain[idx - 1] if chain is not None else seg["name"]
        if stage not in STAGE_TEMPLATES:
            raise ValueError(f"unknown stage {stage!r} in segment {seg}")
        tmpl = STAGE_TEMPLATES[stage]

        out_segments.append(
            {
                "index": idx,
                "start_time": float(seg["start_sec"]),
                "end_time": float(seg["end_sec"]),
            }
        )
        desc_qas.append(
            _build_qa(
                template_question=tmpl["descriptive"],
                template_choices=tmpl["descriptive_choices"],
                template_correct=tmpl["descriptive_correct"],
                video_id=video_id,
                segment_index=idx,
                qa_kind="descriptive",
                seg=seg,
                stage=stage,
                decorator=decorator,
            )
        )
        if idx >= 2:
            causal_qas.append(
                _build_qa(
                    template_question=tmpl["causal_question"],
                    template_choices=tmpl["causal_choices"],
                    template_correct=tmpl["causal_correct"],
                    video_id=video_id,
                    segment_index=idx,
                    qa_kind="causal",
                    seg=seg,
                    stage=stage,
                    decorator=decorator,
                )
            )

    return {
        "videos": [
            {
                "video_id": video_id,
                "segments": out_segments,
                "descriptive_qas": desc_qas,
                "causal_qas": causal_qas,
            }
        ]
    }


def _probe_video_fps(video_path: Path) -> float:
    """Read FPS off the video. Falls back to 30 if the codec lies."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or fps <= 0:
        return 30.0
    return float(fps)


def _make_vlm_decorator(
    *,
    video_path: Path,
    video_id: str,
    model: str,
    cache_dir: Path,
    num_frames: int,
    fps: float,
) -> QADecorator:
    """Build a decorator closure that pulls frames from disk and calls OpenAI.

    Imports are lazy because the templates-mode CLI path must work without
    the openai SDK installed.
    """
    from openai import OpenAI

    from evaluation.modules.causal_step import _read_window_uint8
    from tools._causal_qa_vlm import decorate_qa

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for --mode vlm")
    client = OpenAI(api_key=api_key)

    def _decorator(
        *,
        stage: str,
        qa_type: str,
        segment_index: int,
        start_sec: float,
        end_sec: float,
        template_question: str,
        template_correct_body: str,
    ) -> Any:
        start_frame = int(round(start_sec * fps))
        end_frame = max(int(round(end_sec * fps)), start_frame + 1)
        frames = _read_window_uint8(
            str(video_path),
            start_frame=start_frame,
            end_frame=end_frame,
            num_samples=num_frames,
        )
        return decorate_qa(
            client=client,
            model=model,
            frames=frames,
            stage=stage,
            qa_type=qa_type,
            start_sec=start_sec,
            end_sec=end_sec,
            template_question=template_question,
            template_correct_body=template_correct_body,
            cache_dir=cache_dir,
            video_id=video_id,
            segment_index=segment_index,
        )

    return _decorator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["templates", "vlm"],
        default="templates",
        help="`templates` (default) emits the static templates; `vlm` asks a "
        "vision model to rewrite questions and generate distractors from the "
        "actual segment frames (correct answers stay template-pinned).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to the trajectory video. Required for --mode vlm.",
    )
    parser.add_argument(
        "--vlm-model",
        default="gpt-5.4",
        help="OpenAI model id for VLM mode (default: gpt-5.4).",
    )
    parser.add_argument(
        "--vlm-cache-dir",
        type=Path,
        default=None,
        help="Where decorated QAs are cached. Defaults to <out>.parent/qa_vlm_cache.",
    )
    parser.add_argument("--vlm-num-frames", type=int, default=4)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Override the video FPS (otherwise read from the file).",
    )
    args = parser.parse_args()

    segments_payload = json.loads(args.segments.read_text())
    events = segments_payload.get("events", segments_payload)
    if not isinstance(events, list):
        raise ValueError(
            f"expected segments.json to have top-level 'events' list, got {type(events).__name__}"
        )
    trajectory_shape = (
        segments_payload.get("trajectory_shape")
        if isinstance(segments_payload, dict)
        else None
    )

    decorator: Optional[QADecorator] = None
    if args.mode == "vlm":
        if args.video is None:
            parser.error("--video is required when --mode vlm")
        cache_dir = args.vlm_cache_dir or (args.out.parent / "qa_vlm_cache")
        fps = args.video_fps if args.video_fps else _probe_video_fps(args.video)
        decorator = _make_vlm_decorator(
            video_path=args.video,
            video_id=args.video_id,
            model=args.vlm_model,
            cache_dir=cache_dir,
            num_frames=args.vlm_num_frames,
            fps=fps,
        )

    dataset = build_dataset(
        video_id=args.video_id,
        events=events,
        decorator=decorator,
        trajectory_shape=trajectory_shape,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2))
    print(f"wrote draft QA dataset to {args.out}")


if __name__ == "__main__":
    main()
