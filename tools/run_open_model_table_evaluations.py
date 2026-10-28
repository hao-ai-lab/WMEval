#!/usr/bin/env python3
"""Prepare and evaluate completed LingBot/Matrix-Game Table 4/7 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_table7_rerun_results import build_results as build_table7_results
from tools.prepare_project_genie_case_eval import prepare as prepare_case
from tools.run_prepared_table_evaluations import (
    PLANS,
    _billiard_kinematics_inputs,
    _billiard_unity_inputs,
    _racing_unity_params,
)


MODELS = ("lingbot_world_cam", "matrix_game_2", "matrix_game_3")
REPEATS = tuple(f"run{index}" for index in range(5))


def evaluation_environment() -> dict[str, str]:
    """Use the pinned local evaluator assets without remote metadata probes."""
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def require_available_device(device: str) -> None:
    """Fail before writing result rows when a requested CUDA device is absent."""
    if not device.lower().startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is unavailable in this process; run the "
            "evaluation outside the filesystem sandbox"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def queue_by_run(model: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = ROOT / f"runs/tables_4_7/model_inputs/{model}/generation_queue.jsonl"
    return {
        (row["case_id"], row["generation_repeat"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
        for row in [json.loads(line)]
    }


def action_trace_for(plan: Any, input_receipt: Path, actions_hash: str) -> list[dict[str, Any]]:
    """Build an auditable trace without treating target control as camera yaw."""
    if plan.case_id.startswith("table4/"):
        target_entity = {
            "RacingCar": "racing_car",
            "BilliardGame": "white_cue_ball",
        }.get(plan.case_id.split("/")[1])
        if target_entity is None:
            raise ValueError(f"unsupported Table 4 target trace: {plan.case_id}")
        return [{
            "accepted_at_sec": 0.0,
            "completed_at_sec": 10.0,
            "action": "semantic_target_control",
            "target_entity": target_entity,
            "input_receipt": str(input_receipt),
            "input_receipt_sha256": actions_hash,
        }]
    variant = plan.case_id.rsplit("/", 1)[-1]
    target_by_variant = {
        "perturb_in": -45.0,
        "perturb_out": 45.0,
        "360_in": -360.0,
        "360_out": 360.0,
    }
    if variant not in target_by_variant:
        raise ValueError(f"unsupported Table 7 camera trace: {plan.case_id}")
    action_end = 1.25 if variant.startswith("perturb_") else 10.0
    trace = [{
        "accepted_at_sec": 0.0,
        "completed_at_sec": action_end,
        "target_yaw_deg": target_by_variant[variant],
    }]
    if action_end < 10.0:
        trace.append({"accepted_at_sec": action_end, "completed_at_sec": 10.0, "action": "hold"})
    return trace


def write_compatibility_receipts(model: str, plan: Any, repeat_ids: tuple[str, ...] = REPEATS) -> Path:
    queue = queue_by_run(model)
    root = ROOT / f"runs/tables_4_7/evaluation_runs/{model}/{plan.case_id}"
    for repeat in repeat_ids:
        item = queue.get((plan.case_id, repeat))
        if item is None:
            raise ValueError(f"{model}/{plan.case_id}/{repeat}: missing queue item")
        candidate = ROOT / str(item["candidate_video"])
        generation_receipt = candidate.parent / "generation_receipt.json"
        generation = json.loads(generation_receipt.read_text(encoding="utf-8"))
        if generation.get("status") != "complete" or not candidate.is_file():
            raise ValueError(f"{model}/{plan.case_id}/{repeat}: generation is incomplete")
        digest = sha256(candidate)
        if generation.get("candidate_video_sha256") != digest:
            raise ValueError(f"{candidate}: hash differs from generation receipt")
        input_receipt = ROOT / str(item.get("receipt") or item["input_receipt"])
        actions_hash = sha256(input_receipt)
        first_hash = sha256(ROOT / str(item["conditioning_image"]))
        repeat_root = root / repeat
        repeat_root.mkdir(parents=True, exist_ok=True)
        trace_path = repeat_root / "action_trace.json"
        trace = action_trace_for(plan, input_receipt, actions_hash)
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "repeat_id": repeat,
            "generation_mode": "new_world_generation",
            "table_repeat_eligible": True,
            "evaluation_artifact_status": "native_action_aligned_ready",
            "evaluation_video": str(candidate),
            "evaluation_video_sha256": digest,
            "first_frame_sha256": first_hash,
            "actions_sha256": actions_hash,
            "action_trace": str(trace_path),
            "action_trace_sha256": sha256(trace_path),
            "character_prompt": item.get("character_prompt", ""),
            "source_generation_receipt": str(generation_receipt),
        }
        (repeat_root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return root


def prepare(model: str, plan: Any, repeat_ids: tuple[str, ...] = REPEATS) -> dict[str, Any]:
    if model == "lingbot_world_cam" and plan.case_id.startswith("table4/"):
        raise ValueError("LingBot Base Cam is excluded from Table 4")
    run_root = write_compatibility_receipts(model, plan, repeat_ids)
    output = ROOT / f"runs/tables_4_7/evaluation_prepared/{model}/{plan.case_id}"
    params = None
    reference = None
    if plan.preparation == "racing_unity_capture":
        params = _racing_unity_params(plan)
    elif plan.preparation == "billiard_kinematics_capture":
        params, reference = _billiard_kinematics_inputs(plan)
    elif plan.preparation == "billiard_unity_capture":
        params, reference = _billiard_unity_inputs(plan)
    return prepare_case(
        run_root,
        plan.lite_case_id,
        output,
        model=model,
        camera_trajectory_mode=plan.trajectory_mode,
        reference_start_frame=plan.reference_start_frame,
        reference_target_yaw_deg=plan.target_yaw_deg,
        candidate_params=params,
        reference_video_override=reference,
        repeat_ids=repeat_ids,
    )


def result_errors(path: Path, model: str, plan: Any, repeat_ids: tuple[str, ...] = REPEATS) -> list[str]:
    if not path.is_file():
        return [f"missing {path}"]
    selected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("model") == model and row.get("case_id") == plan.lite_case_id and row.get("dimension") == plan.dimension:
            selected[str(row.get("seed"))] = row
    errors = []
    if set(selected) != set(repeat_ids):
        errors.append(f"expected {list(repeat_ids)}, got {sorted(selected)}")
    errors.extend(f"{repeat}: {row.get('status')}: {row.get('error')}" for repeat, row in selected.items() if row.get("status") != "completed")
    return errors


def seed_extended_results_from_five_run(
    five_run_results: Path,
    extended_results: Path,
    model: str,
    plan: Any,
) -> int:
    """Seed a larger cohort with the already-frozen original five rows.

    The adaptive protocol defines runs 0--4 as the original cohort and runs
    5--29 as the independent confirmation cohort. Re-evaluating the first five
    would waste GPU time and could introduce evaluator nondeterminism, so copy
    their exact rows into the resumable extended report. Existing extended
    rows always win, making this safe to call again after interruption.
    """
    if not five_run_results.is_file():
        return 0
    source_rows = [
        json.loads(line)
        for line in five_run_results.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = {
        str(row.get("seed")): row
        for row in source_rows
        if row.get("model") == model
        and row.get("case_id") == plan.lite_case_id
        and row.get("dimension") == plan.dimension
        and row.get("status") == "completed"
        and str(row.get("seed")) in REPEATS
    }
    if set(selected) != set(REPEATS):
        return 0

    existing_rows = []
    if extended_results.is_file():
        existing_rows = [
            json.loads(line)
            for line in extended_results.read_text(encoding="utf-8").splitlines()
            if line
        ]
    existing_keys = {
        (
            row.get("dimension"),
            row.get("case_id"),
            row.get("model"),
            str(row.get("seed")),
        )
        for row in existing_rows
    }
    additions = []
    for repeat in REPEATS:
        row = selected[repeat]
        key = (plan.dimension, plan.lite_case_id, model, repeat)
        if key not in existing_keys:
            additions.append(row)
    if not additions:
        return 0
    extended_results.parent.mkdir(parents=True, exist_ok=True)
    rows = existing_rows + additions
    extended_results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "policy": "reuse_exact_original_five_for_adaptive_extension",
        "source": str(five_run_results),
        "source_sha256": sha256(five_run_results),
        "seeded_repeats": [str(row["seed"]) for row in additions],
    }
    (extended_results.parent / "five_run_seed_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(additions)


def write_table4_summary(results: Path, output: Path, model: str, case_id: str, repeat_ids: tuple[str, ...] = REPEATS) -> Path:
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("model") == model]
    metrics = {
        "dtw": "dtw_m_mean",
        "fde": "fde_m_mean",
        "exact_match_pct": "exact_match_pct",
        "error_rate_pct": "error_rate_pct",
    }
    summary: dict[str, Any] = {"schema_version": 1, "model": model, "case_id": case_id, "metrics": {}}
    for name, source in metrics.items():
        values = [row.get("metrics", {}).get(source) for row in rows]
        if len(values) != len(repeat_ids) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{case_id}: incomplete {source}")
        numeric = [float(value) for value in values]
        summary["metrics"][name] = {"values": dict(zip(repeat_ids, numeric)), "mean": statistics.fmean(numeric), "sample_std": statistics.stdev(numeric)}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rerun_results.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_one(model: str, plan: Any, device: str, prepare_only: bool, repeat_ids: tuple[str, ...] = REPEATS) -> dict[str, Any]:
    preparation = prepare(model, plan, repeat_ids)
    prepared = Path(str(preparation["candidate_manifest"])).parent
    cohort = "five_run" if len(repeat_ids) == 5 else f"{len(repeat_ids)}_run"
    report = ROOT / f"reports/{plan.case_id.split('/')[0]}/{model}/{plan.case_id.split('/')[1]}/{plan.case_id.split('/')[2]}/{cohort}"
    command = [
        sys.executable, str(ROOT / "tools/run_wmeval_lite.py"), "run",
        "--suite", str(ROOT / "lite"),
        "--candidate-manifest", str(preparation["candidate_manifest"]),
        "--dimensions", plan.dimension,
        "--device", device,
        "--resume", "--output-dir", str(report),
    ]
    row = {"model": model, "case_id": plan.case_id, "status": "prepared", "candidate_manifest": preparation["candidate_manifest"], "command": command}
    if prepare_only:
        return row
    require_available_device(device)
    if len(repeat_ids) > len(REPEATS):
        seed_extended_results_from_five_run(
            ROOT / f"reports/{plan.case_id.split('/')[0]}/{model}/{plan.case_id.split('/')[1]}/{plan.case_id.split('/')[2]}/five_run/per_case_results.jsonl",
            report / "per_case_results.jsonl",
            model,
            plan,
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        env=evaluation_environment(),
    )
    results = report / "per_case_results.jsonl"
    errors = result_errors(results, model, plan, repeat_ids)
    if completed.returncode or errors:
        row.update(status="evaluation_failed", returncode=completed.returncode, errors=errors)
        return row
    if plan.case_id.startswith("table7/") and len(repeat_ids) == 5:
        rerun = build_table7_results(
            baseline_path=ROOT / "experiments/protocol/paper_baseline.yaml",
            results_jsonl=results,
            evaluation_inputs_path=prepared / "evaluation_inputs.json",
            source_case_id=plan.lite_case_id,
            target_case_id=plan.case_id,
            model=model,
        )
        rerun_path = report / "rerun_results.json"
        rerun_path.write_text(json.dumps(rerun, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif plan.case_id.startswith("table4/"):
        rerun_path = write_table4_summary(results, report, model, plan.case_id, repeat_ids)
    else:
        rerun_path = results
    row.update(status="complete", rerun_results=str(rerun_path))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=5)
    args = parser.parse_args()
    selected = set(args.case)
    plans = [plan for plan in PLANS if not selected or plan.case_id in selected]
    if args.model == "lingbot_world_cam":
        plans = [plan for plan in plans if plan.case_id.startswith("table7/")]
    if args.repeat_count < 5:
        raise SystemExit("repeat count must be at least five")
    repeat_ids = tuple(f"run{index}" for index in range(args.repeat_count))
    rows = []
    exit_code = 0
    for plan in plans:
        try:
            row = run_one(args.model, plan, args.device, args.prepare_only, repeat_ids)
        except (FileNotFoundError, ValueError) as exc:
            row = {"model": args.model, "case_id": plan.case_id, "status": "waiting", "error": str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["status"] not in {"complete", "prepared"}:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
