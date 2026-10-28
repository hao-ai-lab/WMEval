from __future__ import annotations

import pytest

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "trajectories" / "audit_trajectory_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_trajectory_readiness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.integration("case_data")
def test_inventory_covers_every_lite_case_and_surfaces_known_gaps() -> None:
    result = MODULE.inventory(MODULE.DEFAULT_CASES)
    assert result["case_count"] == 33
    by_id = {row["case_id"]: row for row in result["cases"]}
    assert by_id["RacingCar/prompt_third"]["status"] == "ready"
    # Table 4/7 cells are first-class Lite cases with compiled trajectories
    # (numeric schedule + Project Genie browser program in actions.jsonl).
    for case_id in ("RacingCar/kinematics_third", "WanderingGarden/scene_360_in"):
        assert by_id[case_id]["effective_status"] == "native_machine_readable_ready"
        assert by_id[case_id]["backend_readiness"]["project_genie_browser"] == "native_executable_ready"
    assert "kinematics_mechanics" in by_id["RacingCar/kinematics_third"]["dimensions"]
    assert "kinematics_mechanics" not in by_id["RacingCar/prompt_third"]["dimensions"]
    # The two Lite-only object-consistency cases carry bundled trajectories
    # (exact reconstructions of their Unity captures).
    for case_id in ("WanderingGarden/scene_first", "ChessBoard/walk_third"):
        assert by_id[case_id]["status"] == "ready"
        assert by_id[case_id]["effective_status"] == "native_machine_readable_ready"
    assert by_id["BilliardGame/prompt_first"]["status"] == "underspecified"
    assert (
        by_id["BilliardGame/prompt_first"]["effective_status"]
        == "supplemental_reconstruction_ready"
    )
    # The one lighting case: the paper's native case, now with its Unity reference
    # orbit (trajectory.yaml + action.npz) next to the paper's hold-ArrowLeft script.
    lighting = by_id["SunRisesAndSets/lighting"]
    assert lighting["status"] == "ready"
    assert lighting["effective_status"] == "native_machine_readable_ready"
    assert lighting["dimensions"] == ["lighting_shading"]
    assert lighting["backend_readiness"]["project_genie_browser"] == "native_executable_ready"
    assert result["effective_status_counts"] == {
        "native_machine_readable_ready": 32,
        "supplemental_reconstruction_ready": 1,
    }
    assert result["backend_readiness_counts"] == {
        "unity_or_numeric_evaluator": {
            "native_numeric_ready": 32,
            "supplemental_numeric_ready": 1,
        },
        "project_genie_browser": {
            "native_executable_ready": 32,
            "supplemental_executable_ready": 1,
        },
    }
    assert by_id["BilliardGame/prompt_first"]["backend_readiness"] == {
        "unity_or_numeric_evaluator": "supplemental_numeric_ready",
        "project_genie_browser": "supplemental_executable_ready",
    }
    assert by_id["WanderingGarden/scene_first"]["backend_readiness"][
        "project_genie_browser"
    ] == "native_executable_ready"
    # BeamDomino's ball-force action is a compiled trajectory with a W-key
    # browser program (it was the known not_machine_executable gap).
    assert by_id["BeamDomino/causal"]["backend_readiness"][
        "project_genie_browser"
    ] == "native_executable_ready"
