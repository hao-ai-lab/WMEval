from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.plan_lingbot_adaptive_expansion import MOTION_PAIRS, SCENES, plan


def write_gate(root: Path, scene: str, motion: str, expand: bool) -> None:
    path = root / scene / motion / "five_run" / "variance_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repeat_count": 5,
                "expand_group_to_30": expand,
            }
        ),
        encoding="utf-8",
    )


def write_queue(path: Path) -> None:
    rows = []
    for scene in SCENES:
        for pair in MOTION_PAIRS:
            for motion in pair:
                for index in range(5):
                    rows.append(
                        {
                            "case_id": f"table7/{scene}/{motion}",
                            "generation_repeat": f"run{index}",
                            "candidate_video": f"outputs/{scene}/{motion}/run{index}/candidate.mp4",
                        }
                    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_plan_expands_both_members_when_one_cell_triggers(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    queue = tmp_path / "generation_queue.jsonl"
    output = tmp_path / "adaptive_expansion_plan.json"
    write_queue(queue)
    for scene in SCENES:
        for first, second in MOTION_PAIRS:
            write_gate(reports, scene, first, scene == "BilliardGame" and first == "perturb_in")
            write_gate(reports, scene, second, False)

    report = plan(reports_root=reports, queue=queue, output=output)

    assert report["selected_cases"] == [
        "table7/BilliardGame/perturb_in",
        "table7/BilliardGame/perturb_out",
    ]
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    counts = {
        case: sum(row["case_id"] == case for row in rows)
        for case in {row["case_id"] for row in rows}
    }
    assert counts["table7/BilliardGame/perturb_in"] == 30
    assert counts["table7/BilliardGame/perturb_out"] == 30
    assert counts["table7/BilliardGame/360_in"] == 5
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"


def test_plan_fails_closed_when_any_gate_is_missing(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    queue = tmp_path / "generation_queue.jsonl"
    write_queue(queue)
    for scene in SCENES:
        for first, second in MOTION_PAIRS:
            write_gate(reports, scene, first, False)
            if not (scene == "MirrorRollingBall" and second == "360_out"):
                write_gate(reports, scene, second, False)

    with pytest.raises(FileNotFoundError, match="missing five-run variance gate"):
        plan(
            reports_root=reports,
            queue=queue,
            output=tmp_path / "plan.json",
        )
