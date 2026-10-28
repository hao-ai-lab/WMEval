#!/usr/bin/env python3
"""Fail-closed completion audit for the requested open-model Table 4/7 scope."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_tables.export_open_model_tables_latex import (
    required_repeat_count,
    rows_for,
    table4_block,
    table7_block,
)
from tools.paper_tables.verify_open_model_generations import verify


TABLE7_CASES = tuple(
    f"table7/{scene}/{motion}"
    for scene in ("WanderingGarden", "BilliardGame", "MirrorRollingBall")
    for motion in ("perturb_in", "perturb_out", "360_in", "360_out")
)
ACTIVE_CASES = {
    "matrix_game_2": (
        "table4/RacingCar/third",
        "table4/BilliardGame/first",
        *TABLE7_CASES,
    ),
    "matrix_game_3": TABLE7_CASES,
    "lingbot_world_cam": TABLE7_CASES,
}


def audit_case(model: str, case_id: str) -> dict[str, Any]:
    repeat_count = required_repeat_count(model, case_id)
    generation = verify(model, repeat_count, {case_id})
    if generation["status"] != "complete":
        raise ValueError("; ".join(generation["errors"]))
    rows = rows_for(model, case_id)
    if len(rows) != repeat_count:
        raise ValueError(f"expected {repeat_count} completed evaluator rows, got {len(rows)}")
    return {
        "status": "complete",
        "model": model,
        "case_id": case_id,
        "repeat_count": repeat_count,
        "generation_count": generation["generation_count"],
        "evaluation_count": len(rows),
    }


def audit() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for model, case_ids in ACTIVE_CASES.items():
        for case_id in case_ids:
            try:
                cases.append(audit_case(model, case_id))
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                cases.append(
                    {
                        "status": "incomplete",
                        "model": model,
                        "case_id": case_id,
                        "error": str(exc),
                    }
                )
                errors.append(f"{model}/{case_id}: {exc}")

    blocks: dict[str, str] = {}
    try:
        blocks["table4_matrix_game_2"] = table4_block("matrix_game_2")[0]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Table 4 block: {exc}")
    for model in ("lingbot_world_cam", "matrix_game_3", "matrix_game_2"):
        try:
            blocks[f"table7_{model}"] = table7_block(model)[0]
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Table 7 {model} block: {exc}")

    return {
        "schema_version": 1,
        "status": "complete" if not errors else "incomplete",
        "scope": {
            "case_count": sum(len(value) for value in ACTIVE_CASES.values()),
            "lingbot_table4": "excluded_wrong_action_target",
            "models": list(ACTIVE_CASES),
        },
        "cases": cases,
        "paper_blocks_validated": sorted(blocks),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
