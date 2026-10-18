from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import action_backends
from evaluation.lite.suite_loader import load_suite
from evaluation.trajectories import load_trajectory, validate_trajectory
from tools import generate_synthetic_trajectories as gen
from tools.expected_jobs import main as expected_jobs
from tools.sync_lite_data import build_full_index, build_index, check_synthetic

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval_sets" / "wmeval_lite_v1"
MANIFEST = json.loads((SUITE / "synthetic" / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_is_consistent() -> None:
    rows = MANIFEST["trajectories"]
    assert MANIFEST["count"] == len(rows) >= 200
    assert sum(MANIFEST["by_family"].values()) == len(rows)
    ids = [r["synthetic_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert len({r["trajectory_id"] for r in rows}) == len(rows)
    dimensions = set(load_suite(SUITE).dimensions)
    core = load_suite(SUITE).all_case_ids()
    targets = {spec["target"] for spec in action_backends.registry()["action_targets"].values()}
    for r in rows:
        assert r["synthetic_id"] == r["path"] and r["path"].startswith(f"synthetic/{r['family']}/")
        assert r["conditioning_case"] in core
        assert r["action_target"] in targets
        assert r["dimensions"] and set(r["dimensions"]) <= dimensions
        assert {"trajectory.yaml", "action.npz", "actions.jsonl", "meta.json"} <= set(r["files"])
        assert r["reference_status"] == "pending_unity_render"


def test_full_index_extends_core() -> None:
    core, full = build_index(), build_full_index()
    assert full[: len(core)] == core
    assert len(full) == len(core) + MANIFEST["count"]
    assert {row["group"] for row in full[len(core):]} == {"synthetic"}


@pytest.mark.integration("case_data")
def test_generator_is_deterministic(tmp_path: Path) -> None:
    rows = gen.generate(tmp_path / "synthetic")
    assert gen.compare(rows, tmp_path) == []
    for r in rows[:: max(1, len(rows) // 25)]:
        traj = load_trajectory(tmp_path / r["path"] / "trajectory.yaml")
        validate_trajectory(traj, source=r["synthetic_id"])
        assert traj["paper_alignment"]["status"] == "synthetic"


@pytest.mark.integration("synthetic_data")
def test_downloaded_synthetic_files_match_manifest() -> None:
    n, missing, bad = check_synthetic()
    assert (n, missing, bad) == (MANIFEST["count"], [], [])


@pytest.mark.integration("case_data")
def test_expected_jobs_full_split_adds_gated_synthetic_jobs(tmp_path: Path) -> None:
    assert expected_jobs(["--model", "odyssey", "--split", "full", "--seeds", "42",
                          "--dimensions", "environment_object_consistency,kinematics_mechanics",
                          "--output", str(tmp_path), "--no-probe"]) == 0
    report = json.loads((tmp_path / "expected_jobs.json").read_text())
    assert report["split"] == "test_full"
    synthetic = [j for j in report["jobs"] if j["case_id"].startswith("synthetic/")]
    assert synthetic and {j["action_support"]["target"] for j in synthetic} == {"camera", "character"}
    assert all(set(j["dimensions"]) <= {"environment_object_consistency", "kinematics_mechanics"} for j in synthetic)
    job = synthetic[0]
    assert job["candidate_video"] == str(tmp_path / job["case_id"] / "42.mp4")
    assert "trajectory.yaml" in job["conditioning"]["action_files"]
    gated = {row["action_support"]["target"] for row in report["not_applicable"] if row["case_id"].startswith("synthetic/")}
    assert {"ego_vehicle", "cue_ball"} <= gated
