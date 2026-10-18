#!/usr/bin/env python3
"""Assess the paired In/Out character-consistency cells reported in Table 7."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assess_table_repeat_variance import assess
from tools.export_open_model_tables_latex import required_repeat_count


MODELS = ("lingbot_world_cam", "matrix_game_2", "matrix_game_3")
SCENE_CASES = {
    "BilliardGame": "BilliardGame/prompt_first",
    "MirrorRollingBall": "MirrorRollingBall/prompt_third",
}
REGIMES = ("perturb", "360")
METRIC = "character_consistency_pct"


def _results(model: str, scene: str, motion: str, repeat_count: int) -> Path:
    cohort = "30_run" if repeat_count == 30 else "five_run"
    return (
        ROOT
        / "reports/table7"
        / model
        / scene
        / motion
        / cohort
        / "per_case_results.jsonl"
    )


def _output(model: str, scene: str, regime: str, repeat_count: int) -> Path:
    cohort = "30_run" if repeat_count == 30 else "five_run"
    filename = "variance_comparison.json" if repeat_count == 30 else "variance_gate.json"
    return ROOT / "reports/table7" / model / scene / "paired_character" / regime / cohort / filename


def load_values(
    path: Path, model: str, lite_case_id: str, repeat_count: int
) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    selected: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("model") != model
            or row.get("case_id") != lite_case_id
            or row.get("dimension") != "environment_object_consistency"
        ):
            continue
        if row.get("status") != "completed":
            raise ValueError(f"non-completed paired-character row in {path}")
        seed = str(row.get("seed"))
        value = row.get("metrics", {}).get(METRIC)
        if (
            seed in selected
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"invalid paired-character row in {path}: {seed}")
        selected[seed] = float(value)
    expected = [f"run{index}" for index in range(repeat_count)]
    if sorted(selected, key=lambda seed: int(seed.removeprefix("run"))) != expected:
        raise ValueError(f"expected paired-character repeats {expected} in {path}")
    return selected


def required_pair_count(model: str, scene: str, regime: str) -> int:
    counts = {
        required_repeat_count(model, f"table7/{scene}/{regime}_in"),
        required_repeat_count(model, f"table7/{scene}/{regime}_out"),
    }
    if len(counts) != 1:
        raise ValueError(f"{model}/{scene}/{regime}: In/Out repeat counts are not paired")
    return counts.pop()


def assess_pair(model: str, scene: str, regime: str, repeat_count: int) -> dict[str, Any]:
    lite_case_id = SCENE_CASES[scene]
    first_case = f"table7/{scene}/{regime}_in"
    second_case = f"table7/{scene}/{regime}_out"
    first = load_values(_results(model, scene, f"{regime}_in", repeat_count), model, lite_case_id, repeat_count)
    second = load_values(_results(model, scene, f"{regime}_out", repeat_count), model, lite_case_id, repeat_count)
    if set(first) != set(second):
        raise ValueError(f"{model}/{scene}/{regime}: In/Out repeat IDs differ")
    ordered = [f"run{index}" for index in range(repeat_count)]
    paired = [(first[seed] + second[seed]) / 2.0 for seed in ordered]
    result = assess(paired, METRIC)
    if repeat_count == 30:
        for test in ("welch_t", "brown_forsythe"):
            result[test]["holm_pvalue"] = result[test]["pvalue"]
    report = {
        "schema_version": 1,
        "status": "complete",
        "model": model,
        "scene": scene,
        "regime": regime,
        "case_ids": [first_case, second_case],
        "repeat_count": repeat_count,
        "pairing": "per-repeat arithmetic mean of In and Out character consistency",
        "expand_group_to_30": result["expand_to_30"],
        "rule": {
            "percentage_metrics": "sample_std > 5 percentage points",
            "confirmation_comparison": "runs 0-4 versus independent runs 5-29",
            "tests": "two-sided Welch t and median-centered Levene/Brown-Forsythe; Holm within reported cell",
        },
        "metrics": {METRIC: result},
    }
    output = _output(model, scene, regime, repeat_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_one(model: str, scene: str, regime: str, wait: bool, poll_seconds: float) -> dict[str, Any]:
    while True:
        try:
            repeat_count = required_pair_count(model, scene, regime)
            return assess_pair(model, scene, regime, repeat_count)
        except (FileNotFoundError, ValueError):
            if not wait:
                raise
            time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--scene", choices=tuple(SCENE_CASES), action="append")
    parser.add_argument("--regime", choices=REGIMES, action="append")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    scenes = args.scene or list(SCENE_CASES)
    regimes = args.regime or list(REGIMES)
    reports = [
        run_one(args.model, scene, regime, args.wait, args.poll_seconds)
        for scene in scenes
        for regime in regimes
    ]
    print(json.dumps({"status": "complete", "reports": reports}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
