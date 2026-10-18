#!/usr/bin/env python3
"""Plan and materialize LingBot's paired adaptive 30-run expansion.

Table 7 reports character consistency as an In/Out paired statistic.  Both
members of a perturbation/rotation pair therefore need the same repeat IDs.
If either five-run cell crosses its preregistered dispersion gate, extend both
members of that pair to 30 runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extend_open_model_generation_queue import extend


MODEL = "lingbot_world_cam"
SCENES = ("WanderingGarden", "BilliardGame", "MirrorRollingBall")
MOTION_PAIRS = (("perturb_in", "perturb_out"), ("360_in", "360_out"))


def gate_path(reports_root: Path, scene: str, motion: str) -> Path:
    return reports_root / scene / motion / "five_run" / "variance_gate.json"


def load_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing five-run variance gate: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("repeat_count") != 5:
        raise ValueError(f"invalid five-run variance gate: {path}")
    if not isinstance(value.get("expand_group_to_30"), bool):
        raise ValueError(f"missing boolean expand_group_to_30: {path}")
    return value


def plan(
    *, reports_root: Path, queue: Path, output: Path, dry_run: bool = False
) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    selected: set[str] = set()
    paired_decisions: list[dict[str, Any]] = []
    for scene in SCENES:
        for first, second in MOTION_PAIRS:
            first_id = f"table7/{scene}/{first}"
            second_id = f"table7/{scene}/{second}"
            first_gate = load_gate(gate_path(reports_root, scene, first))
            second_gate = load_gate(gate_path(reports_root, scene, second))
            gates[first_id] = first_gate
            gates[second_id] = second_gate
            triggered = bool(
                first_gate["expand_group_to_30"]
                or second_gate["expand_group_to_30"]
            )
            if triggered:
                selected.update((first_id, second_id))
            paired_decisions.append(
                {
                    "cases": [first_id, second_id],
                    "cell_triggers": {
                        first_id: first_gate["expand_group_to_30"],
                        second_id: second_gate["expand_group_to_30"],
                    },
                    "expand_pair_to_30": triggered,
                }
            )

    extension = None
    if not dry_run:
        extension = extend(queue, selected, 30)
    report = {
        "schema_version": 1,
        "model": MODEL,
        "policy": (
            "expand both In/Out cells to preserve identical repeat IDs when "
            "either member crosses its five-run variance gate"
        ),
        "all_gate_count": len(gates),
        "selected_cases": sorted(selected),
        "selected_case_count": len(selected),
        "paired_decisions": paired_decisions,
        "queue_extension": extension,
        "status": "dry_run" if dry_run else "complete",
    }
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model_inputs = ROOT / "runs/tables_4_7/model_inputs" / MODEL
    report = plan(
        reports_root=ROOT / "reports/table7" / MODEL,
        queue=model_inputs / "generation_queue.jsonl",
        output=model_inputs / "adaptive_expansion_plan.json",
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
