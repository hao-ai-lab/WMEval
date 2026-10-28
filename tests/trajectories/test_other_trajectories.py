from __future__ import annotations

import pytest

from pathlib import Path

import numpy as np

from evaluation.trajectories import compile_trajectory, load_actions_jsonl
from tools.trajectories.prepare_other_trajectories import compile_manifest


ROOT = Path(__file__).resolve().parents[2] / "experiments" / "trajectory_reconstructions"


def test_chess_walk_compiles_exact_documented_capture_schedule(tmp_path: Path) -> None:
    compiled = compile_trajectory(ROOT / "trajectories" / "chess_walk_third.yaml")
    assert compiled.values.shape == (90, 4)
    np.testing.assert_array_equal(compiled.values[:25, 0], np.zeros(25))
    np.testing.assert_array_equal(compiled.values[:25, 1], np.ones(25))
    np.testing.assert_allclose(compiled.values[30], [0.8, 0.6, 1.0, 0.0])
    assert np.all(compiled.values[45:70, 3] == 1.0)
    assert np.all(compiled.values[70:, 0] == -0.5)

    copied = tmp_path / "reconstructions"
    copied.mkdir()
    (copied / "manifest.yaml").write_text((ROOT / "manifest.yaml").read_text())
    trajectories = copied / "trajectories"
    trajectories.mkdir()
    for source in (ROOT / "trajectories").glob("*.yaml"):
        (trajectories / source.name).write_text(source.read_text())
    report = compile_manifest(copied)
    assert report["status"] == "ready"
    assert report["case_count"] == 10
    assert load_actions_jsonl(
        copied / "compiled" / "chess_walk_third.actions.jsonl"
    )


def test_lighting_orbit_compiles_locked_ten_second_schedule() -> None:
    compiled = compile_trajectory(
        ROOT / "trajectories" / "sun_rises_and_sets_lighting.yaml"
    )
    assert compiled.values.shape == (160, 1)
    np.testing.assert_array_equal(compiled.values[:, 0], np.ones(160))
    assert compiled.timestamps_sec[-1] + 1.0 / 16.0 == 10.0


CASES = Path(__file__).resolve().parents[2] / "lite" / "cases"


@pytest.mark.integration("case_data")
def test_beam_domino_trajectories_reproduce_the_locked_action_npy() -> None:
    for case in ("causal", "causal_glancing", "causal_direct_strike"):
        compiled = compile_trajectory(ROOT / "trajectories" / f"beam_domino_{case}.yaml")
        legacy = np.load(CASES / "BeamDomino" / case / "reference" / "artifacts" / "action.npy")
        np.testing.assert_array_equal(compiled.values, legacy)


def test_liquid_trajectories_follow_the_nominal_12hz_profiles() -> None:
    def compiled(name: str):
        return compile_trajectory(ROOT / "trajectories" / f"liquid_container_fluid_{name}.yaml")

    assert not compiled("rest").values.any()
    impulse = compiled("impulse_decay").values
    assert impulse.shape == (240, 3)
    np.testing.assert_array_equal(np.flatnonzero(impulse[:, 2]), [24, 25])  # jump at 2.0 s
    np.testing.assert_array_equal(np.flatnonzero(impulse[:, 1]), [132, 133])  # strafe at 11.0 s
    step = compiled("step_hold").values
    assert np.all(step[24:120, 1] == 1.0) and not step[120:].any()
    chirp = compiled("chirp_strafe").values[:, 1]
    assert set(np.unique(chirp)) == {-1.0, 1.0}
    flips = np.flatnonzero(np.diff(chirp))
    assert np.diff(flips[:3]).min() > np.diff(flips[-3:]).max()  # frequency rises


@pytest.mark.integration("case_data")
def test_bundled_trajectory_files_match_the_compiled_sources() -> None:
    import hashlib
    import json

    manifest = json.loads((ROOT / "compiled" / "manifest.json").read_text())
    for row in manifest["cases"]:
        artifacts = CASES / row["case_id"] / "reference" / "artifacts"
        if not row["case_id"].startswith(("BeamDomino/", "LiquidContainer/")):
            continue
        compiled = row["compiled"]
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        assert digest(artifacts / "trajectory.yaml") == compiled["semantic_sha256"]
        assert digest(artifacts / "action.npz") == compiled["action_npz_sha256"]
        assert digest(artifacts / "actions.jsonl") == compiled["project_genie_actions_sha256"]
        cell = json.loads((artifacts / "table_cell.json").read_text())
        assert cell["trajectory_id"] == compiled["trajectory_id"]
        assert cell["backend_support"]["lingbot_world_cam"] is None
