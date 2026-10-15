from __future__ import annotations

import json
from pathlib import Path

from tools.expected_jobs import main

ROOT = Path(__file__).resolve().parents[1]


def test_expected_jobs_split_supported_and_not_applicable(tmp_path: Path) -> None:
    assert main(["--model", "odyssey", "--dimensions", "action_controllability,prompt_scene_consistency",
                 "--seeds", "42,1", "--output", str(tmp_path), "--no-probe"]) == 0
    report = json.loads((tmp_path / "expected_jobs.json").read_text())
    na = {row["case_id"] for row in report["not_applicable"]}
    assert {"BallRoller/ctrl", "BeamDomino/causal", "MirrorRollingBall/lpips_collision"} <= na
    job_cases = {job["case_id"] for job in report["jobs"]}
    assert "ChessBoard/prompt_first" in job_cases and not (job_cases & na)
    assert len(report["jobs"]) == 2 * len(job_cases)
    job = next(j for j in report["jobs"] if j["case_id"] == "ChessBoard/prompt_first")
    assert job["action_support"] == {"target": "camera", "status": "nominal_uncalibrated"}
    assert "trajectory.yaml" in job["conditioning"]["action_files"]
    manifest = (tmp_path / "candidates.template.jsonl").read_text().splitlines()
    assert len(manifest) == len(report["jobs"])
