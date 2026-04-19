from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT / "imports" / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

from evaluation.context import EvaluationContext, SampledVideoFrames, VideoMetadata
from evaluation.registry import run_modules


def _stub_frames() -> SampledVideoFrames:
    meta = VideoMetadata(
        path="stub.mp4",
        fps=30.0,
        frame_count=10,
        width=64,
        height=64,
        duration_sec=1.0,
    )
    arr = np.zeros((64, 64, 3), dtype="uint8")
    return SampledVideoFrames(
        metadata=meta,
        timestamps_sec=[0.0],
        frame_indices=[0],
        frames=[arr],
    )


def test_identity_delta_is_zero(tmp_path: Path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "vid1",
                        "segments": [
                            {"index": 1, "start_time": 0.0, "end_time": 0.33},
                            {"index": 2, "start_time": 0.33, "end_time": 0.66},
                            {"index": 3, "start_time": 0.66, "end_time": 1.0},
                        ],
                        "descriptive_qas": [
                            {
                                "segment_index": i,
                                "question": f"Q{i}?",
                                "choices": ["A) yes", "B) no"],
                                "correct_answer": "A",
                            }
                            for i in (1, 2, 3)
                        ],
                        "causal_qas": [
                            {
                                "segment_index": i,
                                "question": f"C{i}?",
                                "choices": ["A) cause", "B) other"],
                                "correct_answer": "A",
                            }
                            for i in (2, 3)
                        ],
                    }
                ]
            }
        )
    )

    fake_video = tmp_path / "unity.mp4"
    fake_video.write_bytes(b"")  # fake_always_a judge never reads the file

    ctx = EvaluationContext(
        reference_video=str(fake_video),
        candidate_video=str(fake_video),
        reference=_stub_frames(),
        candidate=_stub_frames(),
        shared_duration_sec=1.0,
        eval_fps=30.0,
        max_pairs=1,
        artifacts={
            "causal_step_qa_dataset": str(qa_path),
            "causal_step_video_id": "vid1",
            "causal_step_answers_cache_dir": str(tmp_path / "cache"),
            "causal_step_causal_fail_advance": True,
            "causal_step_judge": "fake_always_a",
        },
    )
    results = run_modules(context=ctx, modules=["causal_step"])
    out = results["causal_step"]
    assert "error" not in out, out
    for metric, value in out["delta"].items():
        assert abs(value) < 1e-9, f"{metric} delta not zero: {value}"
