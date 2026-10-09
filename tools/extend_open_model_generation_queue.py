#!/usr/bin/env python3
"""Extend selected five-run model/cell queues to the adaptive 30-run cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lingbot_world_cam", "matrix_game_2", "matrix_game_3")


def extend(path: Path, selected_cases: set[str], repeat_count: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    case_order = list(dict.fromkeys(str(row["case_id"]) for row in rows))
    by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [row for row in rows if row["case_id"] == case_id] for case_id in case_order
    }
    missing = selected_cases - set(by_case)
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    output: list[dict[str, Any]] = []
    extended = 0
    for case_id in case_order:
        existing = {str(row["generation_repeat"]): row for row in by_case[case_id]}
        target = repeat_count if case_id in selected_cases else len(existing)
        template = existing.get("run0")
        if template is None:
            raise ValueError(f"{case_id}: missing run0 template")
        candidate = Path(str(template["candidate_video"]))
        if candidate.parent.name != "run0":
            raise ValueError(f"{case_id}: unexpected candidate path {candidate}")
        for index in range(target):
            repeat = f"run{index}"
            if repeat in existing:
                output.append(existing[repeat])
                continue
            row = dict(template)
            row["generation_repeat"] = repeat
            row["status"] = "waiting_for_adaptive_variance_expansion"
            row["candidate_video"] = str(candidate.parent.parent / repeat / candidate.name)
            output.append(row)
            extended += 1
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    return {
        "queue": str(path),
        "selected_cases": sorted(selected_cases),
        "repeat_count": repeat_count,
        "rows": len(output),
        "new_rows": extended,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--repeat-count", type=int, default=30)
    args = parser.parse_args()
    if args.repeat_count < 5:
        raise SystemExit("repeat count must be at least five")
    queue = ROOT / f"runs/tables_4_7/model_inputs/{args.model}/generation_queue.jsonl"
    report = extend(queue, set(args.case), args.repeat_count)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
