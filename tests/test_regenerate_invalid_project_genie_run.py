from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/regenerate_invalid_project_genie_run.py"
SPEC = importlib.util.spec_from_file_location("regenerate_invalid_project_genie_run", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepared_command_preserves_case_specific_inputs_and_selects_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "runs/tables_4_7/prepared_assets/generation_queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps(
            {
                "case_id": "table7/WanderingGarden/perturb_out",
                "repeat_id": "run1",
                "status": "ready",
                "command": [
                    "python",
                    "run_project_genie_tables.py",
                    "run",
                    "--case",
                    "table7/WanderingGarden/perturb_out",
                    "--first-frame-override",
                    "table7/WanderingGarden/perturb_out=outward.png",
                    "--turn-calibration",
                    "table7/WanderingGarden/perturb_out=turn.json",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    command = MODULE._prepared_command(
        "table7/WanderingGarden/perturb_out", "run1"
    )

    assert "--first-frame-override" in command
    assert "--turn-calibration" in command
    assert command[-2:] == ["--repeat", "run1"]


def test_prepared_command_rejects_blocked_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "runs/tables_4_7/prepared_assets/generation_queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps(
            {
                "case_id": "table4/BilliardGame/first",
                "repeat_id": "run0",
                "status": "blocked",
                "command": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="not runnable"):
        MODULE._prepared_command("table4/BilliardGame/first", "run0")
