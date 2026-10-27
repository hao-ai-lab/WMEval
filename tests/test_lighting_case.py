"""SunRisesAndSets/lighting: the one lighting case (paper protocol + Unity reference orbit)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation.camera_orbit import orbit_poses, validate_orbit
from tools.build_lighting_case import CASE, CASE_ID, trajectory_spec
from tools.render_synthetic_references import LIGHTING_NATIVE_UNITY

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval_sets/wmeval_lite_v1"


def _unity_from_local(local: np.ndarray) -> np.ndarray:
    pivot = np.array([LIGHTING_NATIVE_UNITY["pivot_x"], 0.0, LIGHTING_NATIVE_UNITY["pivot_z"]])
    return pivot + np.diag([1.0, 1.0, -1.0]) @ local


def test_reference_orbit_starts_at_the_native_camera_rig_and_closes() -> None:
    spec = trajectory_spec()
    validate_orbit(spec)
    poses = orbit_poses(spec, np.array([0.0, 2.5, 10.0]))
    # ref.png is the scene's CameraOrbitAgent start view: lookAtTarget = TrainingArea,
    # cameraDistance 3, cameraYaw 45, cameraPitch 20 (Unity coordinates).
    target = np.array([0.3154, 1.28, 0.54422])
    yaw, pitch = math.radians(45.0), math.radians(20.0)
    agent = target + 3.0 * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    np.testing.assert_allclose(_unity_from_local(poses[0, :3, 3]), agent, atol=1e-12)
    forward = np.diag([1.0, 1.0, -1.0]) @ poses[0, :3, 2]
    np.testing.assert_allclose(forward, (target - agent) / np.linalg.norm(target - agent), atol=1e-12)
    np.testing.assert_allclose(poses[0], poses[-1], atol=1e-12)     # one full loop in 10 s
    assert spec["segments"][0]["values"]["orbit_phase_rate_deg_s"] == 36.0  # counter-clockwise
    assert spec["geometry"]["vertical_fov_deg"] == 60.0


@pytest.mark.integration("case_data")
def test_bundled_trajectory_matches_the_builder() -> None:
    bundled = yaml.safe_load((CASE / "reference/artifacts/trajectory.yaml").read_text())
    assert bundled == trajectory_spec()


def test_main_suite_scores_the_native_case_with_the_paper_protocol() -> None:
    suite = yaml.safe_load((SUITE / "suite.yaml").read_text())
    lighting = suite["dimensions"]["lighting_shading"]
    assert lighting["cases"] == [CASE_ID]
    assert lighting["modules"] == ["shadow_consistency"]
    assert lighting["requires_reference"] is False
    assert lighting["metric_config"] == {"eval_fps": 24.0, "max_pairs": 0}
    assert lighting["aggregate"]["headline"] == "shadow_direction_jitter_deg_per_frame"
    case = yaml.safe_load((CASE / "case.yaml").read_text())
    assert case["protocol"]["candidate_normalization"] == "none"
    assert case["module_params"] == {"shadow_consistency": {"shadow_text_prompt": "table. chair. lamp.",
                                                            "shadow_anchor_label": "table"}}
    # One lighting case, one suite: no companion suite scores it again.
    assert not [p for p in SUITE.glob("*/suite.yaml")]
    assert not (SUITE / "cases/SunRisesAndSets/lighting_orbit_v2").exists()
