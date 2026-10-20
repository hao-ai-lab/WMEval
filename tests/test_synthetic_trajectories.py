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
    assert {tuple(row["suites"]) for row in full[len(core):]} == {("wmeval_v1",)}


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


def test_dataset_card_declares_every_index_column() -> None:
    # The Hub viewer infers column types per chunk; all-null stretches (synthetic rows have
    # no reference video) break that, so the card must declare every column explicitly.
    import yaml

    from tools.sync_lite_data import INDEX_COLUMNS

    header = (SUITE / "DATASET_CARD.md").read_text(encoding="utf-8").split("---")[1]
    (info,) = yaml.safe_load(header)["dataset_info"]
    assert [f["name"] for f in info["features"]] == list(INDEX_COLUMNS)
    assert all(f.get("dtype") == "string" or f.get("sequence") == "string" for f in info["features"])


def test_descriptions_match_actions_and_are_diverse() -> None:
    import numpy as np

    families = {fam["family"]: fam for fam in gen.families()}
    rows = MANIFEST["trajectories"]
    assert len({r["prompt"] for r in rows}) == len(rows)
    for r in rows:
        fam = families[r["family"]]
        # text first: the prompt ends with the action description, which parses back into the steps
        assert r["prompt"].endswith("\n\n" + r["action_description"])
        assert gen.parse_description(fam["grammar"], r["action_description"], fam["duration"]) == r["action_steps"]
        for old, _ in gen.SCENE_EDITS.get(r["conditioning_case"], []):
            assert old not in r["prompt"], f"{r['synthetic_id']} still describes the core case's action"
    for name, fam in families.items():
        plans = [r["action_steps"] for r in rows if r["family"] == name]
        signals = np.stack([gen.control_signal(fam, plan) for plan in plans]).astype(float)
        assert len({s.tobytes() for s in signals}) == len(plans), f"{name}: duplicate action sequences"
        dist = np.sqrt(((signals[:, None] - signals[None]) ** 2).mean(-1)) / np.sqrt((signals ** 2).mean())
        np.fill_diagonal(dist, np.inf)
        assert dist.min() > 0.25, f"{name}: two action sequences are near-duplicates ({dist.min():.2f})"


def test_parse_rejects_text_that_does_not_match_the_steps() -> None:
    fam = next(f for f in gen.families() if f["family"] == "camera_yaw")
    g = fam["grammar"]
    good = "From this starting view, the camera turns left at 30 degrees per second for 2 seconds (60 degrees in total)."
    assert gen.parse_description(g, good + "", 2.0) == [{"kind": "turn", "dir": "left", "speed": 30.0, "sec": 2.0}]
    with pytest.raises(ValueError):  # stated total contradicts speed x duration
        gen.parse_description(g, good.replace("60 degrees", "90 degrees"), 2.0)
    with pytest.raises(ValueError):  # the steps end early but the end-of-clip phrase is missing
        gen.parse_description(g, good, fam["duration"])
