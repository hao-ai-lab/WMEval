from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import action_backends
from evaluation.lite.suite_loader import load_suite
from evaluation.trajectories import load_trajectory, validate_trajectory
from tools.dataset import generate_synthetic_trajectories as gen
from tools.expected_jobs import main as expected_jobs
from tools.sync_lite_data import build_full_index, build_index, check_synthetic

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "lite"
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
        assert r["reference_status"] in {"pending_unity_render", "unity_rendered"}


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
    # Odyssey drives the car on RacingCar (action_backends.yaml), so ego_vehicle cases are jobs too
    assert synthetic and {j["action_support"]["target"] for j in synthetic} == {"camera", "character", "ego_vehicle"}
    assert all(set(j["dimensions"]) <= {"environment_object_consistency", "kinematics_mechanics"} for j in synthetic)
    job = synthetic[0]
    assert job["candidate_video"] == str(tmp_path / job["case_id"] / "42.mp4")
    assert "trajectory.yaml" in job["conditioning"]["action_files"]
    gated = {row["action_support"]["target"] for row in report["not_applicable"] if row["case_id"].startswith("synthetic/")}
    assert "cue_ball" in gated and "ego_vehicle" not in gated


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
        grammar = gen.grammar_for(fam, r["conditioning_case"])
        assert gen.parse_description(grammar, r["action_description"], fam["duration"]) == r["action_steps"]
        for old, _ in gen.SCENE_EDITS.get(r["conditioning_case"], []):
            assert old not in r["prompt"], f"{r['synthetic_id']} still describes the core case's action"
    for name, fam in families.items():
        plans = [(r["action_steps"], r["conditioning_case"]) for r in rows if r["family"] == name]
        signals = np.stack([gen.control_signal(fam, plan, case) for plan, case in plans]).astype(float)
        assert len({s.tobytes() for s in signals}) == len(plans), f"{name}: duplicate action sequences"
        if fam.get("signal_hold_sec"):
            # impulse families (one-step strikes): relative to the pair's own energy, since a
            # single strike's held signal is small against the family RMS
            energy = (signals ** 2).mean(-1)
            dist = np.sqrt(((signals[:, None] - signals[None]) ** 2).mean(-1) / ((energy[:, None] + energy[None]) / 2))
        else:
            dist = np.sqrt(((signals[:, None] - signals[None]) ** 2).mean(-1)) / np.sqrt((signals ** 2).mean())
        np.fill_diagonal(dist, np.inf)
        assert dist.min() > 0.25, f"{name}: two action sequences are near-duplicates ({dist.min():.2f})"


def test_ball_force_directions_are_named_from_the_camera() -> None:
    # Same controls, camera-relative words: BallRoller's camera looks along world +X, so a
    # world +Z push ("vertical" > 0) moves the ball screen-left.
    fam = next(f for f in gen.families() if f["family"] == "ball_force")
    world = [{"kind": "push", "dir": "forward", "strength": "full", "sec": 1.0}]
    shown = gen.shown_plan(fam, "BallRoller/ctrl", world)
    assert shown[0]["dir"] == "left"
    g = gen.grammar_for(fam, "BallRoller/ctrl")
    assert g.compile(shown) == fam["grammar"].compile(world)
    assert g.describe(shown, 6.0).startswith("As seen from the camera, the ball is pushed left")


def test_screening_swaps_only_rejected_plans() -> None:
    fam = next(f for f in gen.families() if f["family"] == "cue_ball")
    index = [f["family"] for f in gen.families()].index("cue_ball")
    base = gen.select_plans(fam, index)
    swapped = gen.select_plans(fam, index, [{"plan": base[3]}, {"plan": base[7]}])
    assert len(swapped) == len(base)
    assert [k for k, (a, b) in enumerate(zip(base, swapped)) if a != b] == [3, 7]
    assert base[3] not in swapped and base[7] not in swapped


def test_parse_rejects_text_that_does_not_match_the_steps() -> None:
    fam = next(f for f in gen.families() if f["family"] == "camera_yaw")
    g = fam["grammar"]
    good = "From this starting view, the camera turns left at 30 degrees per second for 2 seconds (60 degrees in total)."
    assert gen.parse_description(g, good + "", 2.0) == [{"kind": "turn", "dir": "left", "speed": 30.0, "sec": 2.0}]
    with pytest.raises(ValueError):  # stated total contradicts speed x duration
        gen.parse_description(g, good.replace("60 degrees", "90 degrees"), 2.0)
    with pytest.raises(ValueError):  # the steps end early but the end-of-clip phrase is missing
        gen.parse_description(g, good, fam["duration"])


def test_references_are_consistent_with_the_manifest() -> None:
    path = SUITE / "synthetic" / "references.json"
    if not path.is_file():
        pytest.skip("no Unity references published yet")
    doc = json.loads(path.read_text(encoding="utf-8"))
    refs, rows = doc["references"], {r["synthetic_id"]: r for r in MANIFEST["trajectories"]}
    assert doc["count"] == len(refs) and set(refs) <= set(rows)
    for sid, row in rows.items():
        ref = refs.get(sid)
        # meta.json (in the manifest) mirrors references.json, so --verify covers it
        assert row["reference_status"] == ("unity_rendered" if ref else "pending_unity_render")
        if ref is None:
            continue
        assert row["reference_video"] == f"{sid}/{ref['video']}"
        assert ref["video"] in ref["files"] and all(f.startswith("reference/") for f in ref["files"])
        weak = ref["first_frame_psnr_db"] is not None and ref["first_frame_psnr_db"] < doc["weak_first_frame_threshold_db"]
        assert ref["quality"] == ("weak_first_frame" if weak else "ok") == row["reference_quality"]
        assert "/Users/" not in json.dumps(ref)


def test_dataset_revision_is_pinned() -> None:
    from tools.sync_lite_data import pinned_revision

    rev = pinned_revision()
    assert rev is not None and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_index_metadata_when_run_as_a_script() -> None:
    # `python tools/sync_lite_data.py upload` has tools/ (not the repo root) on sys.path;
    # the index must still resolve evaluation.* metadata such as action_target.
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, sys.argv[1]); import sync_lite_data as s; "
            "print(sum(r['action_target'] is not None for r in s.build_index()))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "tools")],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert int(out) > 0
