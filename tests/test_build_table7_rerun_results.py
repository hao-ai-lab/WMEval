from __future__ import annotations

import json
from pathlib import Path

from tools.build_table7_rerun_results import build_results
from evaluation.table_comparison import sha256


ROOT = Path(__file__).resolve().parents[1]


def test_build_results_maps_five_lite_rows_and_preserves_missing_distance(tmp_path: Path) -> None:
    runs = []
    rows = []
    for index in range(5):
        repeat_id = f"run{index}"
        video = tmp_path / f"{repeat_id}.mp4"
        video.write_bytes(repeat_id.encode())
        runs.append(
            {
                "repeat_id": repeat_id,
                "candidate_video": str(video),
                "candidate_video_sha256": sha256(video),
            }
        )
        rows.append(
            {
                "dimension": "environment_object_consistency",
                "case_id": "WanderingGarden/scene_first",
                "model": "project_genie",
                "seed": repeat_id,
                "status": "completed",
                "candidate_video": str(video),
                "evaluator_lock": {
                    "release_id": "wmeval-lite-v1",
                    "evaluator_fingerprint": "fixed",
                },
                "full_report": {
                    "device": "cpu",
                    "results": {
                        "environment_object_consistency": {
                            "depth_model_selection": "auto",
                            "depth_model_used": "depth-model",
                        }
                    },
                },
                "metrics": {
                    "depth_absrel_pct": 30 + index,
                    "pairwise_distance_mae_pct": None,
                    "object_permanence_pct": 80 + index,
                },
            }
        )
    receipt = tmp_path / "inputs.json"
    receipt.write_text(json.dumps({"status": "ready", "runs": runs}))
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_results(
        baseline_path=ROOT / "experiments/tables_4_7/paper_baseline.yaml",
        results_jsonl=jsonl,
        evaluation_inputs_path=receipt,
        source_case_id="WanderingGarden/scene_first",
        target_case_id="table7/WanderingGarden/360_in",
    )
    target = result["results"]["table7/WanderingGarden/360_in"]
    assert target["depth_absrel_pct"]["run4"] == 34.0
    assert target["object_permanence_pct"]["run0"] == 80.0
    assert all(value is None for value in target["dist_mae_pct"].values())
    assert len(result["provenance"]["incomplete_metrics"]) == 5
    assert result["provenance"]["evaluation_settings"]["device"] == "cpu"


def test_build_results_maps_mirror_character_consistency(tmp_path: Path) -> None:
    runs = []
    rows = []
    for index in range(5):
        repeat_id = f"run{index}"
        video = tmp_path / f"{repeat_id}.mp4"
        video.write_bytes(repeat_id.encode())
        runs.append(
            {
                "repeat_id": repeat_id,
                "candidate_video": str(video),
                "candidate_video_sha256": sha256(video),
            }
        )
        rows.append(
            {
                "dimension": "environment_object_consistency",
                "case_id": "MirrorRollingBall/prompt_third",
                "model": "project_genie",
                "seed": repeat_id,
                "status": "completed",
                "candidate_video": str(video),
                "evaluator_lock": {
                    "release_id": "wmeval-lite-v1",
                    "evaluator_fingerprint": "fixed",
                },
                "full_report": {
                    "device": "cpu",
                    "results": {"environment_object_consistency": {}},
                },
                "metrics": {
                    "pairwise_distance_mae_pct": None,
                    "character_consistency_pct": 50 + index,
                },
            }
        )
    receipt = tmp_path / "inputs.json"
    receipt.write_text(json.dumps({"status": "ready", "runs": runs}))
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = build_results(
        baseline_path=ROOT / "experiments/tables_4_7/paper_baseline.yaml",
        results_jsonl=jsonl,
        evaluation_inputs_path=receipt,
        source_case_id="MirrorRollingBall/prompt_third",
        target_case_id="table7/MirrorRollingBall/360_in",
    )
    target = result["results"]["table7/MirrorRollingBall/360_in"]
    assert target["character_consistency_pct"]["run4"] == 54.0
    assert all(value is None for value in target["dist_mae_pct"].values())


def test_build_results_rejects_mixed_evaluator_locks(tmp_path: Path) -> None:
    runs = []
    rows = []
    for index in range(5):
        repeat_id = f"run{index}"
        video = tmp_path / f"{repeat_id}.mp4"
        video.write_bytes(repeat_id.encode())
        runs.append(
            {
                "repeat_id": repeat_id,
                "candidate_video": str(video),
                "candidate_video_sha256": sha256(video),
            }
        )
        rows.append(
            {
                "dimension": "environment_object_consistency",
                "case_id": "WanderingGarden/scene_first",
                "model": "project_genie",
                "seed": repeat_id,
                "status": "completed",
                "candidate_video": str(video),
                "evaluator_lock": {
                    "release_id": "wmeval-lite-v1",
                    "evaluator_fingerprint": f"fingerprint-{index % 2}",
                },
                "full_report": {
                    "device": "cpu",
                    "results": {"environment_object_consistency": {}},
                },
                "metrics": {"depth_absrel_pct": 30 + index},
            }
        )
    receipt = tmp_path / "inputs.json"
    receipt.write_text(json.dumps({"status": "ready", "runs": runs}))
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))

    import pytest

    with pytest.raises(ValueError, match="one evaluator lock"):
        build_results(
            baseline_path=ROOT / "experiments/tables_4_7/paper_baseline.yaml",
            results_jsonl=jsonl,
            evaluation_inputs_path=receipt,
            source_case_id="WanderingGarden/scene_first",
            target_case_id="table7/WanderingGarden/360_in",
        )
