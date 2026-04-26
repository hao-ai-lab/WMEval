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
        "descriptive": "What is the floor ball doing in this segment?",
        "descriptive_choices": [
            "A) rolling along the floor toward the pillar-and-roof structure",
            "B) bouncing off a wall",
            "C) resting motionless on the floor",
            "D) flying through the air above the dominoes",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why is the floor ball moving this way?",
        "causal_choices": [
            "A) it was launched with an initial velocity at the start of the trajectory",
            "B) a falling domino kicked it forward",
            "C) the roof ball pushed it",
            "D) gravity reversed direction",
        ],
        "causal_correct": "A",
    },
    "impact": {
        "descriptive": "What is happening between the floor ball and the pillar structure?",
        "descriptive_choices": [
            "A) the floor ball collides with the pillars that support the roof",
            "B) the floor ball stops short of the pillars without touching them",
            "C) the roof falls on its own while the ball is still far away",
            "D) nothing in the scene is moving",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why did the roof and the ball on top of it start to move?",
        "causal_choices": [
            "A) the floor ball transferred momentum to the pillars, destabilizing the roof they were supporting",
            "B) the dominoes fell first and pushed the roof over",
            "C) wind blew the roof sideways",
            "D) the camera moved and only made it look like motion",
        ],
        "causal_correct": "A",
    },
    "topple": {
        "descriptive": "What is the roof (with the ball on top) doing?",
        "descriptive_choices": [
            "A) falling and toppling toward the line of dominoes",
            "B) staying balanced on the pillars",
            "C) lifting upward off the pillars",
            "D) disappearing from the scene",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why is the roof falling in this direction?",
        "causal_choices": [
            "A) the floor ball knocked the supporting pillars out, removing the roof's support on that side",
            "B) the dominoes reached up and pulled it down",
            "C) it was randomly initialized to fall",
            "D) it had been falling since the start of the video",
        ],
        "causal_correct": "A",
    },
    "cascade": {
        "descriptive": "What are the dominoes doing?",
        "descriptive_choices": [
            "A) falling one after another in a chain reaction",
            "B) standing upright and unmoved",
            "C) jumping straight upward off the floor",
            "D) sliding sideways without tipping over",
        ],
        "descriptive_correct": "A",
        "causal_question": "Why are the dominoes falling?",
        "causal_choices": [
            "A) the falling roof (and the ball on top of it) struck the first domino, which then knocked over the next, propagating the chain",
            "B) the floor ball rolled directly into the dominoes itself",
            "C) the floor tilted and they slid over",
            "D) they were pre-animated to fall regardless of any collision",
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
