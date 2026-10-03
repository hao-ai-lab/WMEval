from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_trajectory_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_trajectory_readiness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_inventory_covers_every_lite_case_and_surfaces_known_gaps() -> None:
    result = MODULE.inventory(MODULE.DEFAULT_CASES)
    assert result["case_count"] == 19
    by_id = {row["case_id"]: row for row in result["cases"]}
    assert by_id["RacingCar/prompt_third"]["status"] == "ready"
    assert "kinematics_mechanics" in by_id["RacingCar/prompt_third"]["dimensions"]
    assert by_id["WanderingGarden/scene_first"]["status"] == "underspecified"
    assert (
        by_id["WanderingGarden/scene_first"]["effective_status"]
        == "supplemental_reconstruction_ready"
    )
    assert by_id["BilliardGame/prompt_first"]["status"] == "underspecified"
    assert (
        by_id["BilliardGame/prompt_first"]["effective_status"]
        == "supplemental_reconstruction_ready"
    )
    assert by_id["ChessBoard/walk_third"]["status"] == "underspecified"
    assert (
        by_id["ChessBoard/walk_third"]["effective_status"]
        == "supplemental_reconstruction_ready"
    )
    assert by_id["SunRisesAndSets/lighting"]["status"] == "executable_script_only"
    assert (
        by_id["SunRisesAndSets/lighting"]["effective_status"]
        == "supplemental_reconstruction_ready"
    )
    assert result["effective_status_counts"] == {
        "native_machine_readable_ready": 15,
        "supplemental_reconstruction_ready": 4,
    }
    assert result["backend_readiness_counts"] == {
        "unity_or_numeric_evaluator": {
            "native_numeric_ready": 15,
            "supplemental_numeric_ready": 3,
            "supplemental_semantic_numeric_only": 1,
        },
        "project_genie_browser": {
            "native_executable_ready": 9,
            "not_machine_executable": 7,
            "supplemental_executable_ready": 3,
        },
    }
    assert by_id["BilliardGame/prompt_first"]["backend_readiness"] == {
        "unity_or_numeric_evaluator": "supplemental_numeric_ready",
        "project_genie_browser": "supplemental_executable_ready",
    }
    assert by_id["WanderingGarden/scene_first"]["backend_readiness"][
        "project_genie_browser"
    ] == "supplemental_executable_ready"
    assert by_id["BeamDomino/causal"]["backend_readiness"][
        "project_genie_browser"
    ] == "not_machine_executable"
