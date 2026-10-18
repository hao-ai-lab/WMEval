#!/usr/bin/env python3
"""Prepare deterministic Matrix-Game 2/3 action schedules and five-run queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.trajectories import integrate_rate_channel, sample_held_values  # noqa: E402

TABLE_ROOT = ROOT / "experiments/tables_4_7"
DEFAULT_OUTPUT = ROOT / "runs/tables_4_7/model_inputs"
REPEATS = tuple(f"run{index}" for index in range(5))
# Native timelines, supported action targets and the frozen response-side
# camera gains (with their measurement provenance) live in
# experiments/tables_4_7/action_backends.yaml.
MODEL_SPECS = {name: action_backends.backend(name) for name in ("matrix_game_2", "matrix_game_3")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def prompt_for(case: dict[str, Any], manifest: dict[str, Any]) -> str:
    if case.get("environment_prompt"):
        return str(case["environment_prompt"])
    if int(case["table"]) == 4:
        case_doc = load_yaml((TABLE_ROOT / str(case["prompt_source"])).resolve())
        return str(case_doc["prompt"])
    components = load_yaml(TABLE_ROOT / str(manifest["prompt_components"]))
    variant = str(case["case_id"]).rsplit("/", 1)[-1]
    motion = "perturb" if variant.startswith("perturb") else "360"
    return " ".join(
        str(value).strip()
        for value in (
            components["scenes"][case["scene"]],
            components["orientations"][case["paper_orientation"]],
            components["motions"][motion],
        )
    )


def schedule_for(model: str, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = MODEL_SPECS[model]
    frames = int(spec["frames"])
    fps = float(spec["fps"])
    keyboard = np.zeros((frames, int(spec["keyboard_width"])), dtype=np.float32)
    mouse = np.zeros((frames, 2), dtype=np.float32)
    trajectory_path = TABLE_ROOT / str(case["trajectory"])
    trajectory = load_yaml(trajectory_path)
    target = action_backends.require_support(model, trajectory)
    mapping: dict[str, Any]

    if target == "camera":
        yaw_deg = integrate_rate_channel(trajectory, "yaw_rate_deg_s", fps, frames)
        integrated_yaw = float(yaw_deg.sum())
        visual_gain, calibration_key = action_backends.camera_gain(model, str(case["case_id"]), integrated_yaw)
        mouse[:, 1] = yaw_deg / visual_gain
        mapping = {
            "kind": "camera_yaw",
            "visual_degrees_per_mouse_unit_per_frame": visual_gain,
            "response_calibration_key": calibration_key,
            "upstream_geometric_degrees_per_mouse_unit_per_frame": spec.get("upstream_geometric_deg_per_mouse_unit"),
            "status": "response_calibrated_repeat_validated",
            "integrated_yaw_deg": integrated_yaw,
        }
        return keyboard, mouse, mapping

    values, names = sample_held_values(
        TABLE_ROOT / "compiled" / f"{trajectory_path.stem}.action.npz",
        fps,
        frames,
        end_sec=float(trajectory["duration_sec"]),
    )
    by_name = {name: values[:, index] for index, name in enumerate(names)}
    # Keyboard-only backends cannot express control magnitude: thresholded
    # WASD is a hypothesis about the controlled entity, not a validated mapping.
    if target == "ego_vehicle":
        steer = by_name["steer"]
        throttle = by_name["throttle"]
        brake = by_name["brake"]
        keyboard[:, 0] = throttle > 0.05
        keyboard[:, 1] = (throttle < -0.05) | (brake > 0.5)
        keyboard[:, 2] = steer < -0.05
        keyboard[:, 3] = steer > 0.05
        mapping = {"kind": "generic_wasd_to_car_hypothesis", "status": "target_entity_validation_required"}
    elif target == "cue_ball":
        forward = by_name["forward"]
        lateral = by_name["lateral"]
        keyboard[:, 0] = forward > 0.05
        keyboard[:, 1] = forward < -0.05
        keyboard[:, 2] = lateral < -0.05
        keyboard[:, 3] = lateral > 0.05
        mapping = {"kind": "generic_wasd_to_ball_hypothesis", "status": "target_entity_validation_required"}
    else:
        raise action_backends.UnsupportedAction(f"{model} has no keyboard mapping for {target} actions")
    return keyboard, mouse, mapping


def prepare_model(model: str, manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    model_root = output_root / model
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        if case.get("execution_scope") == "excluded_by_user":
            continue
        # Cells whose action target this backend cannot express are out of scope
        # (reported as unsupported), never compiled into a substitute action.
        trajectory_doc = load_yaml(TABLE_ROOT / str(case["trajectory"]))
        if action_backends.support_status(model, action_backends.action_target(trajectory_doc)) is None:
            continue
        keyboard, mouse, mapping = schedule_for(model, case)
        case_id = str(case["case_id"])
        case_root = model_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        schedule_path = case_root / "schedule.npz"
        np.savez(schedule_path, keyboard=keyboard, mouse=mouse)
        conditioning = (TABLE_ROOT / str(case["first_frame"])).resolve()
        if not conditioning.is_file():
            raise FileNotFoundError(conditioning)
        trajectory = (TABLE_ROOT / str(case["trajectory"])).resolve()
        receipt = {
            "schema_version": 1,
            "status": "prepared_calibration_gate_pending",
            "model": model,
            "case_id": case_id,
            "frames": MODEL_SPECS[model]["frames"],
            "fps": MODEL_SPECS[model]["fps"],
            "mapping": mapping,
            "schedule": str(schedule_path.relative_to(ROOT)),
            "schedule_sha256": sha256(schedule_path),
            "trajectory": str(trajectory.relative_to(ROOT)),
            "trajectory_sha256": sha256(trajectory),
            "conditioning_image": str(conditioning.relative_to(ROOT)),
            "conditioning_image_sha256": sha256(conditioning),
            "environment_prompt": prompt_for(case, manifest),
            "character_prompt": str(case.get("character_prompt", "")),
        }
        receipt_path = case_root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipt["receipt"] = str(receipt_path.relative_to(ROOT))
        rows.append(receipt)

    queue_path = model_root / "generation_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for repeat in REPEATS:
                item = {
                    **{key: row[key] for key in ("model", "case_id", "schedule", "conditioning_image", "environment_prompt", "character_prompt", "receipt")},
                    "generation_repeat": repeat,
                    "status": "waiting_for_calibration_gate",
                    "candidate_video": f"runs/tables_4_7/model_outputs/{model}/{row['case_id']}/{repeat}/candidate.mp4",
                }
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    return {
        "model": model,
        "case_count": len(rows),
        "generation_count": len(rows) * len(REPEATS),
        "queue": str(queue_path.relative_to(ROOT)),
        "queue_sha256": sha256(queue_path),
    }


def prepare_calibration(output_root: Path) -> dict[str, Any]:
    root = output_root / "matrix_game_2/calibration"
    root.mkdir(parents=True, exist_ok=True)
    frames = 21
    probes = {"no_action": 0.0}
    for amplitude in (0.05, 0.1, 0.2, 0.3):
        probes[f"yaw_left_{amplitude:g}"] = -amplitude
        probes[f"yaw_right_{amplitude:g}"] = amplitude
    rows = []
    for name, amplitude in probes.items():
        keyboard = np.zeros((frames, 4), dtype=np.float32)
        mouse = np.zeros((frames, 2), dtype=np.float32)
        mouse[:, 1] = amplitude
        path = root / f"{name}.npz"
        np.savez(path, keyboard=keyboard, mouse=mouse)
        rows.append({"probe": name, "amplitude": amplitude, "schedule": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    manifest = {"schema_version": 1, "status": "prepared", "frames": frames, "fps": float(MODEL_SPECS["matrix_game_2"]["fps"]), "latent_frames": 6, "repeats": list(REPEATS), "probes": rows}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = load_yaml(TABLE_ROOT / "cases.yaml")
    summaries = [prepare_model(model, manifest, args.output.resolve()) for model in MODEL_SPECS]
    calibration = prepare_calibration(args.output.resolve())
    print(json.dumps({"models": summaries, "matrix_game_2_calibration_probes": len(calibration["probes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
