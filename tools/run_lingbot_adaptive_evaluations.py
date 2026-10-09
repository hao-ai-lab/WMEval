#!/usr/bin/env python3
"""Evaluate and statistically assess selected LingBot 30-run Table 7 cells.

The adaptive expansion planner preserves paired In/Out repeat IDs.  This
driver consumes that frozen plan, partitions selected cells across GPUs, waits
for receipt-complete generations, verifies every candidate, resumes the
30-run evaluator, and writes the preregistered first-five versus confirmation-
25 comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = "lingbot_world_cam"
PLAN = ROOT / "runs/tables_4_7/model_inputs" / MODEL / "adaptive_expansion_plan.json"
OUTPUTS = ROOT / "runs/tables_4_7/model_outputs" / MODEL
REPORTS = ROOT / "reports/table7" / MODEL

SCENE_CONFIG: dict[str, dict[str, Any]] = {
    "WanderingGarden": {
        "lite_case_id": "WanderingGarden/scene_first",
        "metrics": (
            "depth_absrel_pct",
            "pairwise_distance_mae_pct",
            "object_permanence_pct",
        ),
    },
    "BilliardGame": {
        "lite_case_id": "BilliardGame/prompt_first",
        "metrics": (
            "depth_absrel_pct",
            "pairwise_distance_mae_pct",
            "character_consistency_pct",
        ),
    },
    "MirrorRollingBall": {
        "lite_case_id": "MirrorRollingBall/prompt_third",
        "metrics": (
            "pairwise_distance_mae_pct",
            "character_consistency_pct",
        ),
    },
}


def load_selected_cases(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing adaptive expansion plan: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    selected = value.get("selected_cases")
    if (
        value.get("schema_version") != 1
        or value.get("model") != MODEL
        or value.get("status") != "complete"
        or not isinstance(selected, list)
        or value.get("selected_case_count") != len(selected)
    ):
        raise ValueError(f"invalid adaptive expansion plan: {path}")
    if len(selected) != len(set(selected)) or any(
        not isinstance(case, str) or case.count("/") != 2 for case in selected
    ):
        raise ValueError(f"invalid selected cases in adaptive expansion plan: {path}")
    for case in selected:
        table, scene, motion = case.split("/")
        if (
            table != "table7"
            or scene not in SCENE_CONFIG
            or motion not in {"perturb_in", "perturb_out", "360_in", "360_out"}
        ):
            raise ValueError(f"unsupported adaptive case: {case}")
    return sorted(selected)


def cases_for_shard(cases: list[str], shard_index: int, shard_count: int) -> list[str]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    return [case for index, case in enumerate(cases) if index % shard_count == shard_index]


def receipt_count(case_id: str) -> int:
    return sum(
        1
        for path in (OUTPUTS / case_id).glob("run*/generation_receipt.json")
        if path.is_file()
    )


def total_generation_receipt_count() -> int:
    return sum(
        1
        for path in OUTPUTS.glob("table7/*/*/run*/generation_receipt.json")
        if path.is_file()
    )


def run_checked(command: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def evaluate_case(case_id: str, poll_seconds: float, env: dict[str, str]) -> dict[str, Any]:
    _, scene, motion = case_id.split("/")
    while receipt_count(case_id) < 30:
        time.sleep(poll_seconds)

    run_checked(
        [
            sys.executable,
            str(ROOT / "tools/verify_open_model_generations.py"),
            "--model", MODEL,
            "--repeat-count", "30",
            "--case", case_id,
        ],
        env,
    )
    run_checked(
        [
            sys.executable,
            str(ROOT / "tools/run_open_model_table_evaluations.py"),
            "--model", MODEL,
            "--case", case_id,
            "--repeat-count", "30",
            "--device", "cuda",
        ],
        env,
    )
    config = SCENE_CONFIG[scene]
    cohort = REPORTS / scene / motion / "30_run"
    assess = [
        sys.executable,
        str(ROOT / "tools/assess_table_repeat_variance.py"),
        "--results", str(cohort / "per_case_results.jsonl"),
        "--model", MODEL,
        "--case-id", str(config["lite_case_id"]),
        "--dimension", "environment_object_consistency",
        "--output", str(cohort / "variance_comparison.json"),
    ]
    for metric in config["metrics"]:
        assess.extend(("--metric", metric))
    run_checked(assess, env)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "model": MODEL,
        "case_id": case_id,
        "repeat_count": 30,
        "variance_comparison": str(cohort / "variance_comparison.json"),
    }
    (cohort / "adaptive_evaluation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--wait-total-generation-receipts",
        type=int,
        default=0,
        help=(
            "defer all evaluation until this many LingBot Table 7 generation "
            "receipts exist; useful when the GPU remains available for tail helpers"
        ),
    )
    parser.add_argument(
        "--post-generation-settle-seconds",
        type=float,
        default=0.0,
        help=(
            "after the total-receipt gate opens, wait this long for the final "
            "generator to finish decode/offload cleanup before using the GPU"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.wait_total_generation_receipts < 0:
        raise SystemExit("--wait-total-generation-receipts must be nonnegative")
    if args.post_generation_settle_seconds < 0:
        raise SystemExit("--post-generation-settle-seconds must be nonnegative")
    try:
        cases = cases_for_shard(
            load_selected_cases(PLAN), args.shard_index, args.shard_count
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cases": cases,
                    "wait_total_generation_receipts": args.wait_total_generation_receipts,
                    "post_generation_settle_seconds": args.post_generation_settle_seconds,
                },
                indent=2,
            )
        )
        return 0

    while total_generation_receipt_count() < args.wait_total_generation_receipts:
        time.sleep(args.poll_seconds)
    if (
        args.wait_total_generation_receipts
        and args.post_generation_settle_seconds
    ):
        time.sleep(args.post_generation_settle_seconds)

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    completed = []
    for case in cases:
        completed.append(evaluate_case(case, args.poll_seconds, environment))
    print(
        json.dumps(
            {
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "cases": cases,
                "completed": completed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
