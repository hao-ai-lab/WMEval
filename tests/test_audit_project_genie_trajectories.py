from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_project_genie_trajectories.py"
SPEC = importlib.util.spec_from_file_location("audit_project_genie_trajectories", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_table7_policy_requires_calibrated_fixed_horizon() -> None:
    assert MODULE.expected_mode("table7/WanderingGarden/360_in") == (
        "calibrated_visual_periodicity"
    )
    assert MODULE.expected_mode("table4/RacingCar/first") == (
        "reconstructed_nominal_timing"
    )


def test_discovery_ignores_archives_and_calibration(tmp_path: Path) -> None:
    receipt = tmp_path / "table7" / "Scene" / "360_in" / "run0" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    archived = tmp_path / "stale_archive" / "table7" / "Scene" / "360_in" / "run0"
    archived.mkdir(parents=True)
    (archived / "receipt.json").write_text("{}")
    assert MODULE.discover_cells(tmp_path) == [
        ("table7/Scene/360_in", receipt.parents[1])
    ]
