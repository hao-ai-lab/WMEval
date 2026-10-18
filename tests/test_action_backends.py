from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation import action_backends
from evaluation.trajectories import (
    integrate_rate_channel,
    integrated_total,
    load_trajectory,
    sample_held_values,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "experiments/tables_4_7"
TRAJECTORIES = sorted((TABLE_ROOT / "trajectories").glob("*.yaml"))


def _cases() -> list[dict]:
    return yaml.safe_load((TABLE_ROOT / "cases.yaml").read_text(encoding="utf-8"))["cases"]


RECONSTRUCTIONS = ROOT / "experiments/trajectory_reconstructions/trajectories"
PHYSICS_TRAJECTORIES = sorted(
    [*RECONSTRUCTIONS.glob("beam_domino_*.yaml"), *RECONSTRUCTIONS.glob("liquid_container_*.yaml")]
)


def test_every_canonical_trajectory_declares_exactly_one_action_target() -> None:
    declared = set(action_backends.registry()["action_targets"])
    lite_only = [RECONSTRUCTIONS / "chess_walk_third.yaml", RECONSTRUCTIONS / "wandering_garden_scene_first.yaml"]
    ids = {load_trajectory(path)["trajectory_id"] for path in [*TRAJECTORIES, *PHYSICS_TRAJECTORIES, *lite_only]}
    assert ids == declared


def test_physics_trajectories_are_unity_and_genie_only() -> None:
    assert len(PHYSICS_TRAJECTORIES) == 7
    for path in PHYSICS_TRAJECTORIES:
        trajectory = load_trajectory(path)
        target = action_backends.action_target(trajectory)
        assert target in {"rolling_ball", "liquid_container"}, path.name
        assert trajectory["adapters"]["project_genie"]["status"] != "unsupported", path.name
        assert action_backends.support_status("project_genie", target) is not None
        for name in ("matrix_game_2", "matrix_game_3", "lingbot_world_cam"):
            with pytest.raises(action_backends.UnsupportedAction):
                action_backends.require_support(name, trajectory)


def test_table4_actuates_scene_entities_and_table7_only_the_camera() -> None:
    for case in _cases():
        target = action_backends.action_target(load_trajectory(TABLE_ROOT / case["trajectory"]))
        if case["table"] == 4:
            assert target != "camera", case["case_id"]
        else:
            assert target == "camera", case["case_id"]


def test_every_requested_open_model_is_registered_with_its_native_timeline() -> None:
    models = yaml.safe_load((TABLE_ROOT / "remaining_models.yaml").read_text(encoding="utf-8"))["models"]
    for name in models:
        spec = action_backends.backend(name)
        assert int(spec["frames"]) > 0 and float(spec["fps"]) > 0
    for name in ("unity", "project_genie"):
        action_backends.backend(name)


def test_camera_only_backend_rejects_vehicle_control_with_recorded_reason() -> None:
    drive = load_trajectory(TABLE_ROOT / "trajectories/racing_car_drive.yaml")
    with pytest.raises(action_backends.UnsupportedAction, match="extrinsics"):
        action_backends.require_support("lingbot_world_cam", drive)
    assert action_backends.require_support("matrix_game_2", drive) == "ego_vehicle"
    assert action_backends.support_status("matrix_game_2", "ego_vehicle") == "target_entity_validation_required"


@pytest.mark.parametrize(
    "stem,trajectory_id",
    [
        ("ball_roller_collision_path", "controllability.ball_roller.collision_path_v1"),
        ("mirror_rolling_ball_axis_cycle", "controllability.mirror_rolling_ball.axis_cycle_v1"),
    ],
)
def test_ball_control_trajectories_actuate_the_ball(stem: str, trajectory_id: str) -> None:
    trajectory = load_trajectory(TABLE_ROOT / f"trajectories/{stem}.yaml")
    assert trajectory["trajectory_id"] == trajectory_id
    assert action_backends.action_target(trajectory) == "ball"


def test_object_free_backends_are_unsupported_for_ball_control_with_a_reason() -> None:
    trajectory = load_trajectory(TABLE_ROOT / "trajectories/ball_roller_collision_path.yaml")
    with pytest.raises(action_backends.UnsupportedAction, match="no object channel"):
        action_backends.require_support("waypoint15", trajectory)
    with pytest.raises(action_backends.UnsupportedAction, match="cannot move the ball"):
        action_backends.require_support("lingbot_world_cam", trajectory)
    for name in ("unity", "project_genie", "lingbot_world_act"):
        assert action_backends.require_support(name, trajectory) == "ball"


def test_project_genie_unsupported_targets_match_trajectory_adapter_status() -> None:
    for path in TRAJECTORIES:
        trajectory = load_trajectory(path)
        target = action_backends.action_target(trajectory)
        adapter_status = trajectory["adapters"]["project_genie"]["status"]
        registry_status = action_backends.support_status("project_genie", target)
        assert (adapter_status == "unsupported") == (registry_status is None), path.name


def test_camera_gain_selection_follows_backend_calibration_layout() -> None:
    assert action_backends.camera_gain("matrix_game_2", "table7/X/perturb_in", -45.0) == (10.1, "pooled_bidirectional")
    assert action_backends.camera_gain("matrix_game_3", "table7/X/perturb_in", -45.0) == (11.175, "perturb_negative")
    assert action_backends.camera_gain("matrix_game_3", "table7/X/360_out", 360.0) == (11.434, "full_turn_positive")


def test_registry_must_classify_every_target(tmp_path: Path) -> None:
    raw = yaml.safe_load(action_backends.REGISTRY_PATH.read_text(encoding="utf-8"))
    del raw["backends"]["lingbot_world_cam"]["unsupported"]["cue_ball"]
    broken = tmp_path / "action_backends.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must classify every target"):
        action_backends.registry(broken)


@pytest.mark.parametrize("fps,frames", [(12.0, 129), (16.0, 161), (17.0, 177)])
def test_rate_integration_preserves_commanded_yaw_on_any_native_timeline(fps: float, frames: int) -> None:
    for name in ("camera_perturb_in", "camera_360_out"):
        trajectory = load_trajectory(TABLE_ROOT / f"trajectories/{name}.yaml")
        increments = integrate_rate_channel(trajectory, "yaw_rate_deg_s", fps, frames)
        assert increments.shape == (frames,)
        assert np.isclose(increments.sum(), integrated_total(trajectory, "yaw_rate_deg_s"))


def test_held_sampling_zeroes_controls_after_the_trajectory_ends() -> None:
    values, names = sample_held_values(
        TABLE_ROOT / "compiled/racing_car_drive.action.npz", 12.0, 129, end_sec=10.0
    )
    assert names == ["steer", "throttle", "brake"]
    assert values.shape == (129, 3)
    assert np.all(values[np.arange(129) / 12.0 >= 10.0] == 0.0)
    assert values[24, 1] > 0  # 2.0 s: throttle segment


def test_complete_orbit_support_is_independent_of_yaw_target_support(tmp_path):
    orbit={'trajectory_id':'lighting.camera.look_at_orbit_v2','kind':'camera_look_at_orbit','target':'camera'}
    assert action_backends.require_support('lingbot_world_cam',orbit)=='camera'
    assert action_backends.support_status('matrix_game_2','camera') is not None
    with pytest.raises(action_backends.UnsupportedAction,match='metric'):
        action_backends.require_support('matrix_game_2',orbit)
    raw=yaml.safe_load(action_backends.REGISTRY_PATH.read_text())
    decision=raw['trajectory_support'][orbit['trajectory_id']]['backends']['matrix_game_2']
    decision.update(status='pending_calibration',reason='Test conversion is not audited yet.')
    pending=tmp_path/'pending.yaml';pending.write_text(yaml.safe_dump(raw))
    with pytest.raises(action_backends.IncompleteAction,match='pending_calibration'):
        action_backends.require_support('matrix_game_2',orbit,path=pending)
    decision.update(status='unsupported',reason='Test interface supplies yaw only.')
    path=tmp_path/'backends.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(action_backends.UnsupportedAction,match='yaw only'):
        action_backends.require_support('matrix_game_2',orbit,path=path)
    del raw['trajectory_support']
    raw['action_targets'][orbit['trajectory_id']]={'target':'camera'}
    path2=tmp_path/'target_only.yaml';path2.write_text(yaml.safe_dump(raw))
    with pytest.raises(action_backends.IncompleteAction,match='alone'):
        action_backends.require_support('matrix_game_2',orbit,path=path2)
