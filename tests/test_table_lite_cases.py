"""Table 4/7 cells are first-class Lite cases evaluated by the single Lite path."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation.lite.normalization import normalize_to_reference
from evaluation.lite.suite import NORMALIZATION_TOKENS
from evaluation.trajectories import load_trajectory

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval_sets/wmeval_lite_v1"
TABLE_ROOT = ROOT / "experiments/tables_4_7"


def _lite_case_id(cell_id: str) -> str:
    table, scene, variant = cell_id.split("/")
    return f"{scene}/{'kinematics' if table == 'table4' else 'scene'}_{variant}"


def _active_cells() -> list[dict]:
    cases = yaml.safe_load((TABLE_ROOT / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if c.get("execution_scope", "active") == "active"]


def _suite_cases(dimension: str) -> list[str]:
    suite = yaml.safe_load((SUITE / "suite.yaml").read_text(encoding="utf-8"))
    return suite["dimensions"][dimension]["cases"]


def test_every_active_table_cell_is_wired_into_its_dimension() -> None:
    cells = _active_cells()
    assert len(cells) == 14
    for cell in cells:
        dimension = "kinematics_mechanics" if cell["table"] == 4 else "environment_object_consistency"
        assert _lite_case_id(cell["case_id"]) in _suite_cases(dimension), cell["case_id"]


def test_superseded_legacy_borrows_are_not_scored() -> None:
    assert "RacingCar/prompt_third" not in _suite_cases("kinematics_mechanics")
    assert "BilliardGame/prompt_first" not in _suite_cases("kinematics_mechanics")
    assert "BilliardGame/prompt_first" not in _suite_cases("environment_object_consistency")
    assert "MirrorRollingBall/prompt_third" not in _suite_cases("environment_object_consistency")


@pytest.mark.parametrize("cell", _active_cells(), ids=lambda c: c["case_id"])
def test_baked_bundle_carries_conditions_actions_and_ground_truth(cell: dict) -> None:
    case_dir = SUITE / "cases" / _lite_case_id(cell["case_id"])
    case = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    dimension = case["modules"][0]
    assert case["modules"] == [dimension]
    assert case["protocol"]["candidate_normalization"] == "match_reference"
    artifacts = case_dir / "reference" / "artifacts"
    for name in ("trajectory.yaml", "action.npz", "action.npy", "actions.txt", "actions.jsonl",
                 "poses.npy", "intrinsics.npy", "object_transforms.json", "table_cell.json"):
        assert (artifacts / name).is_file(), name
    assert (case_dir / "reference" / "ref.png").is_file()
    assert (case_dir / "reference" / "reference.mp4").is_file()
    provenance = json.loads((artifacts / "table_cell.json").read_text(encoding="utf-8"))
    assert provenance["table_cell"] == cell["case_id"]
    canonical = load_trajectory(TABLE_ROOT / cell["trajectory"])
    assert load_trajectory(artifacts / "trajectory.yaml") == canonical
    params = case["module_params"][dimension]
    prefix = "kin_" if dimension == "kinematics_mechanics" else "eoc_"
    for key in ("unity_poses_npy", "unity_intrinsics_npy", "unity_object_transforms_json"):
        assert (case_dir / params[prefix + key]).is_file()
    poses = np.load(artifacts / "poses.npy")
    intrinsics = np.load(artifacts / "intrinsics.npy")
    assert poses.shape[0] == intrinsics.shape[0]


def test_perturbation_reference_follows_the_commanded_timing() -> None:
    """Nominal protocol: ramp over the trajectory's yaw segment, then hold."""
    artifacts = SUITE / "cases/WanderingGarden/scene_perturb_in/reference/artifacts"
    mapping = json.loads((artifacts / "object_transforms.json").read_text(encoding="utf-8"))["trajectory_mapping"]
    assert mapping["mode"] == "perturbation"
    assert (mapping["action_start_sec"], mapping["action_end_sec"], mapping["timeline_end_sec"]) == (0.0, 1.25, 10.0)
    poses = np.load(artifacts / "poses.npy")
    # After the 1.25 s ramp the camera holds: every later pose is identical.
    hold_start = int(np.ceil(1.25 / 10.0 * poses.shape[0]))
    assert np.allclose(poses[hold_start:], poses[hold_start])


def test_match_reference_is_a_registered_normalization_token() -> None:
    assert NORMALIZATION_TOKENS["match_reference"] == "reference"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_match_reference_resamples_to_reference_geometry_and_timeline(tmp_path: Path) -> None:
    def make(path: Path, size: str, rate: int, seconds: float) -> None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size={size}:rate={rate}:duration={seconds}",
                        "-pix_fmt", "yuv420p", str(path)], check=True)

    reference, candidate = tmp_path / "ref.mp4", tmp_path / "cand.mp4"
    make(reference, "320x180", 15, 2.0)   # 30 frames
    make(candidate, "256x256", 24, 3.0)   # different size, rate and length
    result = normalize_to_reference(candidate, tmp_path / "out.mp4", reference=reference)
    assert (result["width"], result["height"], result["frames"]) == (320, 180, 30)
    assert abs(result["fps"] - 15.0) < 1e-3


@pytest.mark.integration("paper_checkout")
def test_lite_rows_export_into_the_paper_comparison_schema() -> None:
    from tools.export_lite_table_results import export

    rows = [
        {"case_id": "WanderingGarden/scene_360_in", "seed": f"run{i}", "status": "completed",
         "metrics": {"depth_absrel_pct": 100.0 + i, "pairwise_distance_mae_pct": 40.0, "object_permanence_pct": 10.0}}
        for i in range(5)
    ] + [
        {"case_id": "RacingCar/kinematics_third", "seed": "42", "status": "completed",
         "metrics": {"dtw_m_mean": 2.0, "fde_m_mean": 3.0, "error_rate_pct": 4.0}},
        {"case_id": "BallRoller/ctrl", "seed": "42", "status": "completed", "metrics": {"fde_m_mean": 9.9}},
    ]
    result = export(rows, model="New Model")
    cell = result["results"]["table7/WanderingGarden/360_in"]
    assert cell["depth_absrel_pct"] == {f"run{i}": 100.0 + i for i in range(5)}
    assert cell["dist_mae_pct"]["run4"] == 40.0
    racing = result["results"]["table4/RacingCar/third"]
    assert (racing["dtw"]["run0"], racing["fde"]["run0"], racing["error_rate_pct"]["run0"]) == (2.0, 3.0, 4.0)
    assert result["model"] == "New Model"
    assert result["provenance"]["cells"] == ["table4/RacingCar/third", "table7/WanderingGarden/360_in"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_match_reference_passes_through_an_already_matching_candidate(tmp_path: Path) -> None:
    video = tmp_path / "ref.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=15:duration=2", "-pix_fmt", "yuv420p", str(video)], check=True)
    out = tmp_path / "out.mp4"
    normalize_to_reference(video, out, reference=video)
    assert out.read_bytes() == video.read_bytes()  # no lossy re-encode
