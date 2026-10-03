from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation.trajectories import (
    TrajectoryError,
    compile_trajectory,
    load_actions_jsonl,
    validate_trajectory,
    write_compiled_trajectory,
)


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "tables_4_7"


def test_manifest_has_every_table_cell() -> None:
    manifest = yaml.safe_load((ROOT / "cases.yaml").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert len(cases) == 16
    assert sum(case["table"] == 4 for case in cases) == 4
    assert sum(case["table"] == 7 for case in cases) == 12
    assert len({case["case_id"] for case in cases}) == 16
    assert manifest["prompt_provenance"]["status"] == "reconstructed"
    assert all((ROOT / case["prompt_source"]).resolve().is_file() for case in cases)
    table7_pairs = {(case["scene"], case["case_id"].rsplit("/", 1)[-1]) for case in cases if case["table"] == 7}
    assert table7_pairs == {
        (scene, variant)
        for scene in ("WanderingGarden", "BilliardGame", "MirrorRollingBall")
        for variant in ("perturb_in", "perturb_out", "360_in", "360_out")
    }
    billiard_table7 = [
        case
        for case in cases
        if case["table"] == 7 and case["scene"] == "BilliardGame"
    ]
    assert all("unity_scene_runs" in case["first_frame"] for case in billiard_table7)
    assert all(
        case["orientation_frame_status"] == "unity_reconstructed_validated"
        for case in billiard_table7
    )
    mirror_out = [
        case
        for case in cases
        if case["scene"] == "MirrorRollingBall"
        and case.get("paper_orientation") == "outward"
    ]
    assert all("prepared_assets" in case["first_frame"] for case in mirror_out)


def test_racing_trajectory_compiles_to_unity_array_and_genie_chords(tmp_path: Path) -> None:
    source = ROOT / "trajectories" / "racing_car_drive.yaml"
    compiled = compile_trajectory(source)
    assert compiled.channel_names == ("steer", "throttle", "brake")
    assert compiled.values.shape == (300, 3)
    np.testing.assert_array_equal(compiled.values[:30], np.zeros((30, 3), dtype=np.float32))
    assert compiled.project_genie_status == "nominal"
    assert {key for op in compiled.project_genie_ops for key in op.get("keys", [])} == {
        "KeyA", "KeyD", "KeyS", "KeyW"
    }

    manifest = write_compiled_trajectory(source, tmp_path)
    native = np.load(tmp_path / manifest["action_npy"], allow_pickle=False)
    np.testing.assert_array_equal(native, compiled.values)
    artifact = np.load(tmp_path / manifest["action_npz"], allow_pickle=False)
    assert artifact["values"].shape == (300, 3)
    metadata = json.loads(str(artifact["metadata_json"]))
    assert metadata["trajectory_id"] == "table4.racing_car.drive_v1"
    assert load_actions_jsonl(tmp_path / manifest["project_genie_actions"])
    summary = (tmp_path / manifest["actions_txt"]).read_text(encoding="utf-8")
    assert "steer throttle brake" in summary
    assert "Project Genie browser actions" in summary

    first_hash = manifest["action_npz_sha256"]
    second = write_compiled_trajectory(source, tmp_path)
    assert second["action_npz_sha256"] == first_hash
    assert second["action_npy_sha256"] == manifest["action_npy_sha256"]


@pytest.mark.parametrize(
    ("name", "expected_key", "active_ms"),
    [
        ("camera_perturb_in.yaml", "ArrowLeft", 1250),
        ("camera_perturb_out.yaml", "ArrowRight", 1250),
        ("camera_360_in.yaml", "ArrowLeft", 10000),
        ("camera_360_out.yaml", "ArrowRight", 10000),
    ],
)
def test_camera_trajectories_compile_to_expected_nominal_keys(
    name: str, expected_key: str, active_ms: int
) -> None:
    compiled = compile_trajectory(ROOT / "trajectories" / name)
    assert compiled.values.shape == (200, 1)
    assert compiled.project_genie_ops[0] == {
        "op": "hold",
        "keys": [expected_key],
        "ms": active_ms,
        "target_yaw_deg": -45.0 if expected_key == "ArrowLeft" and active_ms == 1250 else (
            -360.0 if expected_key == "ArrowLeft" else (45.0 if active_ms == 1250 else 360.0)
        ),
    }
    assert sum(op["ms"] for op in compiled.project_genie_ops) == 10000


def test_billiard_action_is_unity_ready_but_not_misrepresented_as_genie_control() -> None:
    compiled = compile_trajectory(ROOT / "trajectories" / "billiard_shot.yaml")
    assert compiled.values.shape == (150, 12)
    assert compiled.values[0, 10] == 1
    assert np.all(compiled.values[1:, 10] == 0)
    assert compiled.project_genie_status == "unsupported"
    assert compiled.project_genie_ops == ()


def test_first_person_billiard_ball_control_compiles_to_wasd() -> None:
    compiled = compile_trajectory(
        ROOT / "trajectories" / "billiard_first_person_ball_control.yaml"
    )
    assert compiled.values.shape == (300, 2)
    assert compiled.project_genie_status == "nominal"
    active = [op for op in compiled.project_genie_ops if op["op"] == "hold"]
    assert [op["keys"] for op in active] == [["KeyW"], ["KeyD"], ["KeyA"]]
    assert sum(op["ms"] for op in compiled.project_genie_ops) == 10000


def test_schema_rejects_gaps() -> None:
    raw = yaml.safe_load((ROOT / "trajectories" / "camera_360_in.yaml").read_text())
    raw["segments"][0]["start_sec"] = 0.1
    with pytest.raises(TrajectoryError, match="expected contiguous"):
        validate_trajectory(raw)
