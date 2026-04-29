"""Generate a CausalStep QA dataset draft for a BeamDomino trajectory.

Reads a segments JSON produced by `detect_beamdomino_segments` (top-level
`events` list with `{index, name, start_sec, end_sec, ...}`) and emits a
CausalStep-loadable draft using per-stage templates. Output must be reviewed
by a human (see docs/causal_step_qa_review_checklist.md) before evaluation.

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
import random
import re
from pathlib import Path
from typing import Any

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
}


def build_dataset(*, video_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a CausalStep-schema dataset from a segments.json `events` list.

    CausalStep schema requires: 1-indexed contiguous segments, descriptive QA
    for every segment, causal QA for segments 2..n, and `correct_answer` keys.

    Choices are shuffled per QA via `_shuffled_choices` so the correct letter
    is no longer always `A`; the shuffle is seeded by `(video_id,
    segment_index, qa_type)` and is therefore stable across reruns.
    """
    sorted_events = sorted(events, key=lambda e: e["index"])
    out_segments: list[dict[str, Any]] = []
    desc_qas: list[dict[str, Any]] = []
    causal_qas: list[dict[str, Any]] = []

    for seg in sorted_events:
        stage = seg["name"]
        if stage not in STAGE_TEMPLATES:
            raise ValueError(f"unknown stage {stage!r} in segment {seg}")
        tmpl = STAGE_TEMPLATES[stage]
        idx = int(seg["index"])

        out_segments.append(
            {
                "index": idx,
                "start_time": float(seg["start_sec"]),
                "end_time": float(seg["end_sec"]),
            }
        )
        desc_choices, desc_correct = _shuffled_choices(
            tmpl["descriptive_choices"],
            tmpl["descriptive_correct"],
            seed=_seed_for(video_id, idx, "descriptive"),
        )
        desc_qas.append(
            {
                "segment_index": idx,
                "question": tmpl["descriptive"],
                "choices": desc_choices,
                "correct_answer": desc_correct,
            }
        )
        if idx >= 2:
            causal_choices, causal_correct = _shuffled_choices(
                tmpl["causal_choices"],
                tmpl["causal_correct"],
                seed=_seed_for(video_id, idx, "causal"),
            )
            causal_qas.append(
                {
                    "segment_index": idx,
                    "question": tmpl["causal_question"],
                    "choices": causal_choices,
                    "correct_answer": causal_correct,
                }
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    segments_payload = json.loads(args.segments.read_text())
    events = segments_payload.get("events", segments_payload)
    if not isinstance(events, list):
        raise ValueError(
            f"expected segments.json to have top-level 'events' list, got {type(events).__name__}"
        )

    dataset = build_dataset(video_id=args.video_id, events=events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2))
    print(f"wrote draft QA dataset to {args.out}")


if __name__ == "__main__":
    main()
