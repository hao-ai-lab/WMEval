import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_open_model_table_evaluations import (
    action_trace_for,
    evaluation_environment,
    require_available_device,
    seed_extended_results_from_five_run,
)


def test_evaluation_environment_forces_local_cache_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    env = evaluation_environment()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_cpu_device_does_not_require_cuda() -> None:
    require_available_device("cpu")


@pytest.mark.parametrize(
    ("case_id", "target_entity"),
    [
        ("table4/RacingCar/third", "racing_car"),
        ("table4/BilliardGame/first", "white_cue_ball"),
    ],
)
def test_table4_trace_records_target_control(case_id: str, target_entity: str) -> None:
    receipt = Path("input_receipt.json")
    trace = action_trace_for(SimpleNamespace(case_id=case_id), receipt, "abc123")
    assert trace == [{
        "accepted_at_sec": 0.0,
        "completed_at_sec": 10.0,
        "action": "semantic_target_control",
        "target_entity": target_entity,
        "input_receipt": str(receipt),
        "input_receipt_sha256": "abc123",
    }]


def test_table7_perturbation_trace_keeps_yaw_then_hold() -> None:
    trace = action_trace_for(
        SimpleNamespace(case_id="table7/WanderingGarden/perturb_out"),
        Path("unused.json"),
        "unused",
    )
    assert trace == [
        {"accepted_at_sec": 0.0, "completed_at_sec": 1.25, "target_yaw_deg": 45.0},
        {"accepted_at_sec": 1.25, "completed_at_sec": 10.0, "action": "hold"},
    ]


def test_extended_cohort_reuses_exact_original_five(tmp_path: Path) -> None:
    source = tmp_path / "five_run" / "per_case_results.jsonl"
    target = tmp_path / "30_run" / "per_case_results.jsonl"
    source.parent.mkdir()
    rows = [
        {
            "dimension": "environment_object_consistency",
            "case_id": "MirrorRollingBall/prompt_third",
            "model": "matrix_game_3",
            "seed": f"run{index}",
            "status": "completed",
            "metrics": {"pairwise_distance_mae_pct": float(index)},
        }
        for index in range(5)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    plan = SimpleNamespace(
        lite_case_id="MirrorRollingBall/prompt_third",
        dimension="environment_object_consistency",
    )

    assert seed_extended_results_from_five_run(
        source, target, "matrix_game_3", plan
    ) == 5
    assert [json.loads(line) for line in target.read_text().splitlines()] == rows
    assert seed_extended_results_from_five_run(
        source, target, "matrix_game_3", plan
    ) == 0
    receipt = json.loads((target.parent / "five_run_seed_receipt.json").read_text())
    assert receipt["seeded_repeats"] == [f"run{index}" for index in range(5)]
