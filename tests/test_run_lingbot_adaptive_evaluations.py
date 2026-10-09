from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_lingbot_adaptive_evaluations as runner
from tools.run_lingbot_adaptive_evaluations import (
    cases_for_shard,
    load_selected_cases,
    total_generation_receipt_count,
)


def write_plan(path: Path, cases: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "model": "lingbot_world_cam",
                "selected_cases": cases,
                "selected_case_count": len(cases),
            }
        ),
        encoding="utf-8",
    )


def test_load_and_partition_selected_cases(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    cases = [
        "table7/WanderingGarden/perturb_out",
        "table7/WanderingGarden/perturb_in",
        "table7/MirrorRollingBall/360_in",
    ]
    write_plan(path, cases)

    selected = load_selected_cases(path)

    assert selected == sorted(cases)
    assert cases_for_shard(selected, 0, 2) == selected[::2]
    assert cases_for_shard(selected, 1, 2) == selected[1::2]


def test_load_selected_cases_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    write_plan(path, ["table4/RacingCar/third"])

    with pytest.raises(ValueError, match="unsupported adaptive case"):
        load_selected_cases(path)

    with pytest.raises(ValueError, match="invalid shard"):
        cases_for_shard([], 2, 2)


def test_total_generation_receipt_count_is_lingbot_table7_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUTS", tmp_path)
    for relative in (
        "table7/SceneA/perturb_in/run0/generation_receipt.json",
        "table7/SceneB/360_out/run29/generation_receipt.json",
        "table4/SceneA/third/run0/generation_receipt.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert total_generation_receipt_count() == 2
