from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.assess_table7_paired_character_variance import load_values


def write_rows(path: Path, values: list[float]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "model": "model",
                    "case_id": "BilliardGame/prompt_first",
                    "dimension": "environment_object_consistency",
                    "seed": f"run{index}",
                    "status": "completed",
                    "metrics": {"character_consistency_pct": value},
                }
            )
            + "\n"
            for index, value in enumerate(values)
        ),
        encoding="utf-8",
    )


def test_load_values_requires_exact_complete_repeat_set(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_rows(path, [1, 2, 3, 4, 5])
    assert load_values(path, "model", "BilliardGame/prompt_first", 5) == {
        f"run{index}": float(index + 1) for index in range(5)
    }

    write_rows(path, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="expected paired-character repeats"):
        load_values(path, "model", "BilliardGame/prompt_first", 5)


def test_load_values_rejects_nonfinite_metric(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_rows(path, [1, 2, float("nan"), 4, 5])
    with pytest.raises(ValueError, match="invalid paired-character row"):
        load_values(path, "model", "BilliardGame/prompt_first", 5)
