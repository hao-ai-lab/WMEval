from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.trajectories import unity_trajectory_runner as runner
def test_export_evaluation_artifacts_matches_eoc_contract(
    tmp_path: Path, monkeypatch
) -> None:
    capture = tmp_path / "capture"
    (capture / "frames").mkdir(parents=True)
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    payload = {
        "num_frames": 2,
        "frames": [
            {"c2w": identity, "fx": 100, "fy": 100, "cx": 50, "cy": 51},
            {"c2w": identity, "fx": 100, "fy": 100, "cx": 50, "cy": 51},
        ],
        "anchors": {"Main tree": [1, 2, 3]},
    }
    (capture / "captured_data.json").write_text(json.dumps(payload))

    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        Path(command[-1]).write_bytes(b"video")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    artifacts = runner._export_evaluation_artifacts(capture, 15.0)

    artifact_dir = capture / "evaluation_artifacts"
    assert np.load(artifact_dir / "poses.npy").shape == (2, 4, 4)
    assert np.load(artifact_dir / "intrinsics.npy").tolist() == [
        [100.0, 100.0, 50.0, 51.0],
        [100.0, 100.0, 50.0, 51.0],
    ]
    transforms = json.loads((artifact_dir / "object_transforms.json").read_text())
    assert transforms["objects"]["tree"] == transforms["objects"]["Main tree"]
    assert set(artifacts) == {
        "reference_video",
        "object_transforms_json",
        "poses_npy",
        "intrinsics_npy",
    }
