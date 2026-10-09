from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/prepare_matrix_game_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_matrix_game_inputs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_camera_schedules_preserve_requested_integrated_yaw() -> None:
    manifest = MODULE.load_yaml(MODULE.TABLE_ROOT / "cases.yaml")
    cases = {row["case_id"]: row for row in manifest["cases"]}
    for model in MODULE.MODEL_SPECS:
        _, full_mouse, full_mapping = MODULE.schedule_for(
            model, cases["table7/WanderingGarden/360_out"]
        )
        _, perturb_mouse, perturb_mapping = MODULE.schedule_for(
            model, cases["table7/WanderingGarden/perturb_in"]
        )
        full_gain = full_mapping["visual_degrees_per_mouse_unit_per_frame"]
        perturb_gain = perturb_mapping["visual_degrees_per_mouse_unit_per_frame"]
        assert np.isclose((full_mouse[:, 1] * full_gain).sum(), 360.0)
        assert np.isclose((perturb_mouse[:, 1] * perturb_gain).sum(), -45.0)
        assert full_mapping["integrated_yaw_deg"] == 360.0
        assert perturb_mapping["integrated_yaw_deg"] == -45.0


def test_model_specific_schedule_shapes_and_table4_mapping() -> None:
    manifest = MODULE.load_yaml(MODULE.TABLE_ROOT / "cases.yaml")
    cases = {row["case_id"]: row for row in manifest["cases"]}
    mg2_keyboard, mg2_mouse, mapping = MODULE.schedule_for(
        "matrix_game_2", cases["table4/RacingCar/third"]
    )
    mg3_keyboard, mg3_mouse, _ = MODULE.schedule_for(
        "matrix_game_3", cases["table7/MirrorRollingBall/360_in"]
    )
    assert mg2_keyboard.shape == (129, 4)
    assert mg2_mouse.shape == (129, 2)
    assert mg3_keyboard.shape == (177, 6)
    assert mg3_mouse.shape == (177, 2)
    assert mapping["status"] == "target_entity_validation_required"
    assert set(np.unique(mg2_keyboard)).issubset({0.0, 1.0})
