from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_lingbot_table7_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_lingbot_table7_inputs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_yaw_offsets_include_exact_endpoint_and_perturbation_hold() -> None:
    full = MODULE.yaw_offsets("table7/Scene/360_in", -360.0)
    perturb = MODULE.yaw_offsets("table7/Scene/perturb_out", 45.0)
    assert full.shape == (161,)
    assert full[0] == 0.0
    assert full[-1] == -360.0
    assert perturb[0] == 0.0
    assert perturb[20] == 45.0  # 20 frames / 16 fps = 1.25 s
    assert np.all(perturb[20:] == 45.0)


def test_pose_yaw_selection_supports_reverse_direction_and_wrap() -> None:
    angles = np.deg2rad(np.arange(150) * 2.4)
    poses = np.stack([np.eye(4) for _ in angles])
    poses[:, 0, 2] = np.sin(angles)
    poses[:, 2, 2] = np.cos(angles)
    requested = np.array([0.0, -45.0, -180.0, -360.0])
    indices = MODULE.pose_yaw_source_indices(
        poses, start_frame=0, offsets_deg=requested
    )
    selected_yaw = np.rad2deg(
        np.arctan2(poses[indices, 0, 2], poses[indices, 2, 2])
    )
    circular_error = np.abs((selected_yaw - requested + 180.0) % 360.0 - 180.0)
    assert np.all(circular_error <= 1.3)


def test_unity_pose_conversion_preserves_relative_transform() -> None:
    angles = np.deg2rad(np.array([0.0, 45.0]))
    proper = np.stack([np.eye(4), np.eye(4)])
    proper[:, 0, 0] = np.cos(angles)
    proper[:, 0, 2] = np.sin(angles)
    proper[:, 2, 0] = -np.sin(angles)
    proper[:, 2, 2] = np.cos(angles)
    reflection = np.diag([1.0, 1.0, -1.0, 1.0])
    unity = reflection[None] @ proper
    converted, status = MODULE.proper_opencv_c2w(unity)
    assert status == "unity_world_z_reflection_to_right_handed"
    assert np.allclose(np.linalg.det(converted[:, :3, :3]), 1.0)
    original_relative = np.linalg.inv(unity[0]) @ unity[1]
    converted_relative = np.linalg.inv(converted[0]) @ converted[1]
    assert np.allclose(original_relative, converted_relative)


def test_plan_covers_all_twelve_table7_cells() -> None:
    plans = MODULE._plans()
    assert len(plans) == 12
    assert len({plan.case_id for plan in plans}) == 12
    assert {plan.case_id.split("/")[1] for plan in plans} == {
        "WanderingGarden",
        "BilliardGame",
        "MirrorRollingBall",
    }


def test_prompt_metadata_matches_table7_components() -> None:
    perturb = MODULE.prompt_metadata("table7/WanderingGarden/perturb_in")
    full_turn = MODULE.prompt_metadata("table7/MirrorRollingBall/360_out")
    assert "45-degree camera perturbation" in perturb["environment_prompt"]
    assert "inward-facing" in perturb["environment_prompt"]
    assert "full 360-degree camera rotation" in full_turn["environment_prompt"]
    assert "outward-facing" in full_turn["environment_prompt"]
    assert full_turn["character_prompt"] == "A reflective chrome sphere."
    assert perturb["prompt_provenance_status"] == "reconstructed"
