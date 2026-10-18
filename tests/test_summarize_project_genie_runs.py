from __future__ import annotations

import json
from pathlib import Path

from evaluation.project_genie_alignment import sha256
from tools.summarize_project_genie_runs import build_summary, write_summary


def _write_run(root: Path, repeat_id: str, video_bytes: bytes) -> None:
    run = root / repeat_id
    run.mkdir(parents=True)
    video = run / "trajectory_native.mp4"
    native = run / "native.mp4"
    video.write_bytes(video_bytes)
    native.write_bytes(b"native-" + video_bytes)
    report = run / "native_alignment.json"
    report.write_text(
        json.dumps(
            {
                "status": "validated",
                "method": "ordered_multi_checkpoint_sift_v2",
                "metrics": {
                    "matched_sample_count": 9,
                    "strictly_increasing": True,
                    "minimum_match": 0.2,
                    "minimum_margin": 0.1,
                    "minimum_local_speed_ratio": 0.5,
                    "maximum_local_speed_ratio": 1.0,
                    "timeline_r_squared": 0.99,
                },
            }
        )
    )
    (run / "receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "repeat_id": repeat_id,
                "generation_mode": "new_world_generation",
                "table_repeat_eligible": True,
                "evaluation_artifact_status": "native_action_aligned_ready",
                "first_frame_sha256": "frame-hash",
                "actions_sha256": "action-hash",
                "observed_duration_sec": 8.0,
                "evaluation_video_frame_count": 120,
                "evaluation_video_duration_sec": 5.0,
                "native_video_sha256": sha256(native),
                "evaluation_video": str(video),
                "evaluation_video_sha256": sha256(video),
                "native_alignment_report": str(report),
                "native_alignment_report_sha256": sha256(report),
            }
        )
    )


def test_summary_validates_consistent_distinct_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(root, "run0", b"video-0")
    _write_run(root, "run1", b"video-1")
    summary = build_summary(root, ["run0", "run1"])
    assert summary["status"] == "validated"
    assert summary["generation_count"] == 2
    assert summary["aggregate"]["native_frame_count"]["mean"] == 120.0
    prefix = tmp_path / "out" / "validation"
    write_summary(summary, prefix)
    assert prefix.with_suffix(".json").is_file()
    assert prefix.with_suffix(".csv").is_file()
    assert "not a Table 7 metric result" in prefix.with_suffix(".md").read_text()


def test_summary_rejects_duplicate_native_outputs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(root, "run0", b"same")
    _write_run(root, "run1", b"same")
    summary = build_summary(root, ["run0", "run1"])
    assert summary["status"] == "invalid"
    assert "not unique" in " ".join(summary["errors"])


def test_summary_accepts_hash_validated_action_capture(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for index in range(2):
        repeat_id = f"run{index}"
        run = root / repeat_id
        run.mkdir(parents=True)
        video = run / "observed.mp4"
        trace = run / "action_trace.json"
        frames = run / "observed_frames.json"
        video.write_bytes(f"video-{index}".encode())
        trace.write_text("[]")
        frames.write_text('{"frames": []}')
        (run / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "repeat_id": repeat_id,
                    "generation_mode": "new_world_generation",
                    "table_repeat_eligible": True,
                    "evaluation_artifact_status": "action_aligned_capture_ready",
                    "trajectory_validation_status": "reconstructed_nominal_timing",
                    "first_frame_sha256": "frame-hash",
                    "actions_sha256": "action-hash",
                    "observed_duration_sec": 10.0,
                    "observed_frame_count": 75,
                    "observed_video": str(video),
                    "observed_video_sha256": sha256(video),
                    "action_trace": str(trace),
                    "action_trace_sha256": sha256(trace),
                    "observed_frames": str(frames),
                    "observed_frames_sha256": sha256(frames),
                }
            )
        )
    summary = build_summary(root, ["run0", "run1"])
    assert summary["status"] == "validated"
    assert summary["aggregate"]["evaluation_frame_count"]["mean"] == 75.0
