from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation import action_backends
from evaluation.lite.candidate_discovery import load_candidate_manifest
from evaluation.trajectories import compile_trajectory, load_trajectory
from tools import prepare_prompt_case_inputs as prep


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_turn_is_registered_and_baked_into_every_prompt_case() -> None:
    canonical = prep.TABLE_ROOT / prep.TRAJECTORY
    trajectory = load_trajectory(canonical)
    compiled = compile_trajectory(canonical)
    assert float(compiled.values.sum()) / 20.0 == pytest.approx(-400.0)
    assert action_backends.action_target(trajectory) == "camera"
    for backend in ("lingbot_world_cam", "matrix_game_3"):
        assert action_backends.require_support(backend, trajectory) == "camera"

    cases = prep.prompt_cases()
    assert len(cases) == 6
    for case in cases:
        for source, name in prep.BAKED.items():
            assert _sha(case.artifacts_dir / name) == _sha(prep.TABLE_ROOT / source), (case.case_id, name)


def test_prepare_writes_runner_queues_and_a_lite_candidate_manifest(tmp_path: Path) -> None:
    cases = prep.prompt_cases()
    summary = prep.prepare(cases, tmp_path)

    for model in ("lingbot_world_cam", "matrix_game_3"):
        assert summary["models"][model]["generations"] == 30
        rows = [json.loads(line) for line in (tmp_path / "model_inputs" / model / "generation_queue.jsonl").read_text().splitlines()]
        assert {row["generation_repeat"] for row in rows} == {f"run{i}" for i in range(5)}
        assert all(row["conditioning_image"].endswith("reference/ref.png") for row in rows)

    for case in cases:
        # LingBot: one counter-clockwise turn in place (first person), two orbits
        # around the character while looking at it (third person)
        lingbot = tmp_path / "model_inputs/lingbot_world_cam" / case.case_id
        poses = np.load(lingbot / "poses.npy")
        intrinsics = np.load(lingbot / "intrinsics.npy")
        orbit = case.perspective == "third_person"
        assert poses.shape == (161, 4, 4) and intrinsics.shape == (161, 4)
        assert np.all(np.linalg.det(poses[:, :3, :3]) > 0)
        forward = poses[:, :3, 2].astype(np.float64)
        yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
        turns = 2 if orbit else 1
        assert yaw[-1] - yaw[0] == pytest.approx(-360.0 * turns * 160 / 161, abs=0.01)
        if orbit:
            offset = poses[:, :3, 3] - np.array([0.0, 0.12, 2.0])
            assert np.allclose(np.hypot(offset[:, 0], offset[:, 2]), 1.8, atol=1e-5)
            to_center = np.array([0.0, 0.0, 2.0]) - poses[:, :3, 3]
            to_center /= np.linalg.norm(to_center, axis=1, keepdims=True)
            assert np.allclose(np.einsum("ij,ij->i", forward, to_center), 1.0, atol=1e-5)
        else:
            assert not poses[:, :3, 3].any()
        expected = prep.LINGBOT_ORBIT_INTRINSICS if orbit else prep.LINGBOT_TURN_INTRINSICS
        assert np.array_equal(intrinsics, np.tile(expected, (161, 1)))

        # Matrix-Game 3.0: j (turn left) every frame, plus d (strafe) to orbit in third person
        matrix = tmp_path / "model_inputs/matrix_game_3" / case.case_id
        schedule = np.load(matrix / "schedule.npz")
        receipt = json.loads((matrix / "receipt.json").read_text())
        keyboard, mouse = schedule["keyboard"], schedule["mouse"]
        assert keyboard.shape == (177, 6) and mouse.shape == (177, 2)
        assert np.all(mouse[:, 1] == np.float32(-0.1)) and not mouse[:, 0].any()
        orbit = case.perspective == "third_person"
        assert np.all(keyboard[:, 3] == (1.0 if orbit else 0.0))
        assert not np.delete(keyboard, 3, axis=1).any()
        assert receipt["mapping"]["keys_per_frame"] == {"mouse": "j", "keyboard": "d" if orbit else "q"}
        assert receipt["environment_prompt"] == case.prompt

    for model in ("lingbot_world_cam", "matrix_game_3"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "model_inputs" / model / "generation_queue.jsonl").read_text().splitlines()
        ]
        assert {(row["generation_repeat"], row["seed"]) for row in rows} == {(f"run{i}", i) for i in range(5)}

    candidates = load_candidate_manifest(
        tmp_path / "candidates.jsonl", allowed_cases={case.case_id for case in cases}
    )
    assert len(candidates) == 60
    assert {row.seed for row in candidates} == {f"run{i}" for i in range(5)}


def test_runners_use_the_queue_seed_when_given() -> None:
    from tools import run_lingbot_table7, run_matrix_game3

    for runner, default in ((run_lingbot_table7, 42003), (run_matrix_game3, 45003)):
        assert runner.seed_for({"generation_repeat": "run3", "seed": 3}) == 3
        assert runner.seed_for({"generation_repeat": "run3"}) == default
