from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.project_genie_alignment import (
    NativeAlignmentError,
    load_action_trace_endpoint,
    load_semantic_endpoint,
    ordered_coarse_path,
    ordered_sparse_path,
    sha256,
    update_receipt_with_alignment,
    validate_alignment,
)
import numpy as np


def test_semantic_endpoint_prefers_explicit_global_time(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "visual_return_status": "validated",
                    "accepted_at_sec": 0.2,
                    "visual_return_elapsed_sec": 6.5,
                    "visual_return_global_elapsed_sec": 6.9,
                }
            ]
        )
    )
    assert load_semantic_endpoint(trace) == 6.9


def test_semantic_endpoint_supports_pre_global_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "visual_return_status": "validated",
                    "accepted_at_sec": 0.2,
                    "visual_return_elapsed_sec": 6.5,
                }
            ]
        )
    )
    assert load_semantic_endpoint(trace) == pytest.approx(6.7)


def test_semantic_endpoint_requires_one_validated_return(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text("[]")
    with pytest.raises(NativeAlignmentError, match="exactly one"):
        load_semantic_endpoint(trace)


def test_action_trace_endpoint_uses_last_completed_time(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {"op": "hold", "completed_at_sec": 1.5},
                {"op": "wait", "completed_at_sec": 4.25},
            ]
        )
    )
    assert load_action_trace_endpoint(trace) == 4.25


def test_alignment_validation_accepts_ordered_high_confidence_mapping() -> None:
    status, reasons, metrics = validate_alignment(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.5, 2.25, 3.0, 3.75, 4.5],
        [0.31, 0.28, 0.42, 0.36, 0.29],
        [0.08, 0.07, 0.05, 0.06, 0.08],
    )
    assert status == "validated"
    assert reasons == []
    assert metrics["strictly_increasing"] is True
    assert metrics["timeline_r_squared"] == pytest.approx(1.0)


def test_alignment_validation_rejects_nonchronological_mapping() -> None:
    status, reasons, _ = validate_alignment(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.5, 2.25, 2.0, 3.75, 4.5],
        [0.31, 0.28, 0.42, 0.36, 0.29],
        [0.08, 0.07, 0.05, 0.06, 0.08],
    )
    assert status == "insufficient_evidence"
    assert "not strictly chronological" in " ".join(reasons)


def test_alignment_validation_accepts_bounded_nonlinear_time_warp() -> None:
    status, reasons, metrics = validate_alignment(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 1.2, 1.7, 2.6, 3.8],
        [0.31, 0.28, 0.42, 0.36, 0.29],
        [0.08, 0.07, 0.05, 0.06, 0.08],
    )
    assert metrics["timeline_r_squared"] < 0.98
    assert status == "validated"
    assert reasons == []


def test_alignment_validation_rejects_implausible_local_time_warp() -> None:
    status, reasons, _ = validate_alignment(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 1.01, 1.7, 2.6, 3.8],
        [0.31, 0.28, 0.42, 0.36, 0.29],
        [0.08, 0.07, 0.05, 0.06, 0.08],
    )
    assert status == "insufficient_evidence"
    assert "local speed ratio" in " ".join(reasons)


def test_ordered_path_resolves_ambiguous_early_view_with_full_sequence() -> None:
    # Query 1 independently prefers the late revisit (column 8), but that
    # assignment cannot precede the distinctive middle and endpoint matches.
    similarity = np.full((5, 10), 0.01)
    similarity[0, 1] = 0.30
    similarity[1, 2] = 0.18
    similarity[1, 8] = 0.19
    similarity[2, 4] = 0.50
    similarity[3, 6] = 0.55
    similarity[4, 8] = 0.45
    path = ordered_coarse_path(
        similarity,
        list(range(10)),
        [0.0, 1.0, 2.0, 3.0, 4.0],
        fps=1.0,
        minimum_speed_ratio=0.1,
        maximum_speed_ratio=2.0,
    )
    assert path == [1, 2, 4, 6, 8]


def test_ordered_path_fails_without_forward_timeline() -> None:
    with pytest.raises(NativeAlignmentError, match="no chronologically plausible"):
        ordered_coarse_path(
            np.ones((3, 3)),
            [0, 1, 2],
            [0.0, 0.1, 0.2],
            fps=24.0,
            minimum_speed_ratio=1.0,
            maximum_speed_ratio=2.0,
        )


def test_ordered_sparse_path_prevents_independent_refinement_collapse() -> None:
    # Both middle observations prefer frame 10 independently. The joint path
    # must choose a slightly weaker later frame for the second observation.
    path = ordered_sparse_path(
        [
            [0.50, 0.20],
            [0.90, 0.80, 0.10],
            [0.95, 0.85, 0.20],
            [0.20, 0.60],
        ],
        [[0, 1], [9, 10, 11], [10, 11, 12], [20, 21]],
        [0.0, 1.0, 2.0, 3.0],
        fps=10.0,
        minimum_speed_ratio=0.1,
        maximum_speed_ratio=2.0,
    )
    selected = [
        [[0, 1], [9, 10, 11], [10, 11, 12], [20, 21]][row][index]
        for row, index in enumerate(path)
    ]
    assert selected == [0, 9, 10, 21]
    assert all(later > earlier for earlier, later in zip(selected, selected[1:]))


def test_alignment_finalizer_recovers_receipt_after_download_timeout(
    tmp_path: Path,
) -> None:
    native = tmp_path / "native.mp4"
    trace = tmp_path / "action_trace.json"
    observed = tmp_path / "observed_frames.json"
    evaluation = tmp_path / "trajectory_native.mp4"
    report_path = tmp_path / "native_alignment.json"
    for path, content in (
        (native, b"native"),
        (trace, b"trace"),
        (observed, b"observed"),
        (evaluation, b"evaluation"),
        (report_path, b"report"),
    ):
        path.write_bytes(content)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "error",
                "error": "TimeoutError: download",
                "native_video_sha256": sha256(native),
                "observed_frames_sha256": sha256(observed),
            }
        )
    )
    report = {
        "status": "validated",
        "native_video": str(native),
        "native_video_sha256": sha256(native),
        "action_trace": str(trace),
        "action_trace_sha256": sha256(trace),
        "observed_manifest": str(observed),
        "observed_manifest_sha256": sha256(observed),
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "trajectory_native": str(evaluation),
        "trajectory_native_sha256": sha256(evaluation),
        "trajectory_native_frame_count": 10,
        "trajectory_native_duration_sec": 1.0,
    }
    updated = update_receipt_with_alignment(receipt_path, report)
    assert updated["status"] == "complete"
    assert "error" not in updated
    assert updated["action_trace_sha256"] == sha256(trace)
    assert updated["evaluation_artifact_status"] == "native_action_aligned_ready"
