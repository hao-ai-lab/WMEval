from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.trajectories import compile_trajectory, load_actions_jsonl
from tools.prepare_other_trajectories import compile_manifest


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "trajectory_reconstructions"


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
    assert report["case_count"] == 2
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
