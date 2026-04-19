"""Generate a CausalStep QA dataset draft for a BeamDomino trajectory.

Reads a segments JSON produced by `detect_beamdomino_segments` (top-level
`events` list with `{index, name, start_sec, end_sec, ...}`) and emits a
CausalStep-loadable draft using per-stage templates. Output must be reviewed
by a human (see docs/causal_step_qa_review_checklist.md) before evaluation.

Usage:
    python tools/generate_beamdomino_causal_step_qa.py \\
        --segments path/to/segments.json \\
        --video-id trajectory_001 \\
        --out path/to/qa_draft.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_TEMPLATES: dict[str, dict[str, Any]] = {
    "rolling": {
        "descriptive": "What is the ball doing in this segment?",
        "descriptive_choices": [
            "A) rolling toward the beam stack",
            "B) bouncing off the wall",
            "C) resting on the floor",
            "D) flying through the air",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why is the ball moving this way?",
        "causal_choices": [
            "A) it was launched with an initial velocity in the previous segment",
            "B) the beam pushed it",
            "C) a domino hit it",
            "D) gravity reversed",
        ],
        "causal_correct": "A",
    },
    "impact": {
        "descriptive": "What is happening between the ball and the beam stack?",
        "descriptive_choices": [
            "A) the ball collides with the beam stack",
            "B) the ball stops short of the stack",
            "C) the stack falls without being touched",
            "D) nothing is moving",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why did the beam stack start to move?",
        "causal_choices": [
            "A) the ball transferred momentum to the stack on impact",
            "B) the dominoes pushed the stack",
            "C) wind",
            "D) the camera moved",
        ],
        "causal_correct": "A",
    },
    "topple": {
        "descriptive": "What is the beam stack doing?",
        "descriptive_choices": [
            "A) toppling toward the dominoes",
            "B) standing upright",
            "C) disassembling into pieces",
            "D) vanishing",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why is the stack toppling in this direction?",
        "causal_choices": [
            "A) the ball's impact pushed it toward the dominoes",
            "B) the dominoes pulled it",
            "C) random initialization",
            "D) it was always falling",
        ],
        "causal_correct": "A",
    },
    "cascade": {
        "descriptive": "What are the dominoes doing?",
        "descriptive_choices": [
            "A) falling in a cascade",
            "B) standing still",
            "C) jumping upward",
            "D) sliding sideways",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why are the dominoes falling?",
        "causal_choices": [
            "A) the toppling beam stack struck the first domino and triggered a chain",
            "B) the ball rolled into them directly",
            "C) the floor tilted",
            "D) they were pre-animated",
        ],
        "causal_correct": "A",
    },
}


def build_dataset(*, video_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a CausalStep-schema dataset from a segments.json `events` list.

    CausalStep schema requires: 1-indexed contiguous segments, descriptive QA
    for every segment, causal QA for segments 2..n, and `correct_answer` keys.
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
        desc_qas.append(
            {
                "segment_index": idx,
                "question": tmpl["descriptive"],
                "choices": tmpl["descriptive_choices"],
                "correct_answer": tmpl["descriptive_correct"],
            }
        )
        if idx >= 2:
            causal_qas.append(
                {
                    "segment_index": idx,
                    "question": tmpl["causal_question"],
                    "choices": tmpl["causal_choices"],
                    "correct_answer": tmpl["causal_correct"],
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
