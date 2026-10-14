from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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
        lingbot = tmp_path / "model_inputs/lingbot_world_cam" / case.case_id
        poses = np.load(lingbot / "poses.npy")
        assert poses.shape == (161, 4, 4)
        assert np.all(np.linalg.det(poses[:, :3, :3]) > 0)
        forward = poses[:, :3, 2]
        yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
        assert yaw[-1] - yaw[0] == pytest.approx(-400.0, abs=10.0)
        assert np.load(lingbot / "intrinsics.npy").shape == (161, 4)

        matrix = tmp_path / "model_inputs/matrix_game_3" / case.case_id
        schedule = np.load(matrix / "schedule.npz")
        receipt = json.loads((matrix / "receipt.json").read_text())
        gain = receipt["mapping"]["visual_degrees_per_mouse_unit_per_frame"]
        assert schedule["keyboard"].shape == (177, 6) and not schedule["keyboard"].any()
        assert float(schedule["mouse"][:, 1].sum()) * gain == pytest.approx(-400.0, abs=1e-3)
        assert receipt["environment_prompt"] == case.prompt

    candidates = load_candidate_manifest(
        tmp_path / "candidates.jsonl", allowed_cases={case.case_id for case in cases}
    )
    assert len(candidates) == 60
    assert {row.seed for row in candidates} == {f"run{i}" for i in range(5)}


def test_unity_poses_must_perform_the_commanded_turn(tmp_path: Path) -> None:
    angles = np.deg2rad(-20.0 * np.arange(300) / 30.0)  # 20 deg/s, not the commanded 40
    poses = np.tile(np.eye(4, dtype=np.float32), (300, 1, 1))
    poses[:, 0, 0] = np.cos(angles)
    poses[:, 0, 2] = np.sin(angles)
    poses[:, 2, 0] = -np.sin(angles)
    poses[:, 2, 2] = np.cos(angles)
    np.save(tmp_path / "poses.npy", poses)
    np.save(tmp_path / "intrinsics.npy", np.eye(3, dtype=np.float32))
    case = SimpleNamespace(case_id="Fake/prompt_first", artifacts_dir=tmp_path)

    with pytest.raises(ValueError, match="turns at -20"):
        prep.unity_camera(case, load_trajectory(prep.TABLE_ROOT / prep.TRAJECTORY))
