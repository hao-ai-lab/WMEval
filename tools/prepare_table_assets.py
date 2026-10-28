#!/usr/bin/env python3
"""Materialize and audit generation assets for WMEval Tables 4 and 7.

The output is deliberately fail-closed: source-backed frames and actions are
copied into a self-contained bundle, while unresolved paper inputs become
machine-readable blockers. No missing view is synthesized or silently reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = REPO_ROOT / "experiments" / "tables_4_7"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "tables_4_7" / "prepared_assets"
PROJECT_GENIE_MIN_WIDTH = 1280
PROJECT_GENIE_MIN_HEIGHT = 704


def _media_tool(name: str) -> str:
    """Resolve a system binary or this repository's pinned local fallback."""
    system = shutil.which(name)
    if system:
        return system
    bundled = REPO_ROOT / ".local" / "bin" / name
    if bundled.is_file():
        return str(bundled)
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version 1 mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def _safe_id(case_id: str) -> str:
    return case_id.replace("/", "__")


def _png_dimensions(path: Path) -> tuple[int, int]:
    # PNG IHDR is fixed and avoids adding a new image-library dependency.
    raw = path.read_bytes()[:24]
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError(f"{path}: not a valid PNG")
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def _validate_orientation(root: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    check = spec.get("orientation_validation")
    if not check:
        return None
    poses_path = _resolve(root, str(check["poses"]))
    poses = np.load(poses_path, allow_pickle=False)
    base = int(check["base_frame"])
    derived = int(check["derived_frame"])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{poses_path}: expected Nx4x4 poses")
    if max(base, derived) >= len(poses):
        raise ValueError(f"{poses_path}: orientation frame is outside pose array")
    # In/out is a yaw relation. Ignore the shared pitch component of orbit
    # cameras; otherwise two horizontally opposing views with the same
    # downward pitch do not have a 3-D forward-vector dot product near -1.
    a = np.asarray(poses[base, (0, 2), 2], dtype=np.float64)
    b = np.asarray(poses[derived, (0, 2), 2], dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    threshold = float(check["maximum_forward_dot"])
    if dot > threshold:
        raise ValueError(
            f"orientation validation failed: forward dot {dot:.6f} > {threshold:.6f}"
        )
    return {
        "poses": str(poses_path),
        "poses_sha256": sha256(poses_path),
        "base_frame": base,
        "derived_frame": derived,
        "horizontal_forward_dot": dot,
        "maximum_forward_dot": threshold,
    }


def _materialize_frame(root: Path, spec: dict[str, Any], destination: Path) -> dict[str, Any]:
    kind = str(spec.get("kind"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    orientation = _validate_orientation(root, spec)
    if kind == "direct":
        source = _resolve(root, str(spec["path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        source_digest = sha256(source)
        expected_digest = spec.get("expected_sha256")
        if expected_digest is not None and source_digest != str(expected_digest):
            raise ValueError(
                f"{source}: conditioning frame SHA-256 {source_digest} does not "
                f"match pinned value {expected_digest}"
            )
        shutil.copy2(source, destination)
        source_record = {"source": str(source), "source_sha256": source_digest}
    elif kind == "video_frame":
        source = _resolve(root, str(spec["video"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        frame_index = int(spec["frame_index"])
        command = [
            _media_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(destination),
        ]
        subprocess.run(command, check=True)
        source_record = {
            "source": str(source),
            "source_sha256": sha256(source),
            "source_frame_index": frame_index,
        }
    else:
        raise ValueError(f"unsupported conditioning source kind: {kind}")
    width, height = _png_dimensions(destination)
    if width < PROJECT_GENIE_MIN_WIDTH or height < PROJECT_GENIE_MIN_HEIGHT:
        raise ValueError(
            f"{destination}: Project Genie conditioning images must be at least "
            f"{PROJECT_GENIE_MIN_WIDTH}x{PROJECT_GENIE_MIN_HEIGHT}, got "
            f"{width}x{height}"
        )
    return {
        **source_record,
        "path": str(destination.resolve()),
        "sha256": sha256(destination),
        "width": width,
        "height": height,
        "provenance": spec.get("provenance"),
        "source_status": spec.get("status"),
        "orientation_validation": orientation,
    }


def _prompt(root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> str:
    explicit = str(case.get("environment_prompt", "")).strip()
    if explicit:
        return explicit
    if int(case["table"]) == 4:
        source = _resolve(root, str(case["prompt_source"]))
        return str(yaml.safe_load(source.read_text(encoding="utf-8"))["prompt"])
    components = _load_yaml(_resolve(root, str(manifest["prompt_components"])))
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


def _copy_actions(root: Path, case: dict[str, Any], destination: Path) -> dict[str, Any]:
    stem = Path(str(case["trajectory"])).stem
    compiled = root / "compiled"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        f"{stem}.action.npy", f"{stem}.action.npz", f"{stem}.actions.jsonl",
        f"{stem}.actions.txt", f"{stem}.manifest.json",
    ]
    copied: dict[str, Any] = {}
    for name in names:
        source = compiled / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / name
        shutil.copy2(source, target)
        copied[name] = {"path": str(target.resolve()), "sha256": sha256(target)}
    action_manifest = json.loads((destination / f"{stem}.manifest.json").read_text(encoding="utf-8"))
    return {
        "trajectory_stem": stem,
        "trajectory_id": action_manifest["trajectory_id"],
        "alignment_status": action_manifest["alignment_status"],
        "project_genie_status": action_manifest["project_genie_status"],
        "browser_actions": str((destination / f"{stem}.actions.jsonl").resolve()),
        "files": copied,
    }


def _status(case: dict[str, Any], actions: dict[str, Any], frame: dict[str, Any] | None) -> tuple[str, list[str]]:
    if case.get("execution_scope", "active") != "active":
        return "excluded", [str(case["execution_scope"])]
    blockers: list[str] = []
    if frame is None:
        blockers.append("missing_orientation_specific_conditioning_frame")
    if actions["project_genie_status"] == "unsupported":
        blockers.append("project_genie_cannot_express_action_space")
    if blockers:
        return "blocked", blockers
    caveats = []
    if actions["alignment_status"] != "exact":
        caveats.append("trajectory_timeline_reconstructed")
    if frame and frame.get("source_status") != "repository_validated":
        caveats.append(str(frame.get("source_status")))
    if case.get("orientation_frame_status") == "unresolved":
        caveats.append("paper_orientation_anchor_unpublished")
    return ("prepared_with_caveats" if caveats else "prepared"), caveats


def _evaluation_contract(case: dict[str, Any]) -> dict[str, Any]:
    """Describe what the checked-in Lite evaluator can actually reproduce."""
    case_id = str(case["case_id"])
    scene = str(case["scene"])
    if int(case["table"]) == 4:
        lite = {
            ("RacingCar", "first_person"): "RacingCar/prompt_first",
            ("RacingCar", "third_person"): "RacingCar/prompt_third",
            ("BilliardGame", "first_person"): "BilliardGame/prompt_first",
            ("BilliardGame", "third_person"): "BilliardGame/prompt_first",
        }.get((scene, str(case["view"])))
        issues: list[str] = []
        if lite is None:
            issues.append("no_corresponding_lite_case")
        else:
            case_yaml = REPO_ROOT / "lite" / "cases" / lite / "case.yaml"
            case_doc = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
            modules = case_doc.get("modules", [])
            if "kinematics_mechanics" not in modules:
                issues.append("lite_case_does_not_enable_kinematics_mechanics")
            else:
                issues.append("metric_planar_dtw_fde_runtime_unity_calibrated")
                issues.append("mechanical_exact_match_error_rate_unity_mass_metric_trajectory")
        if scene == "RacingCar" and str(case["view"]) == "third_person":
            issues.append("matched_dynamic_private_unity_ground_truth_prepared")
        if scene == "RacingCar" and str(case["view"]) == "first_person":
            issues.append(
                "first_person_ego_car_is_camera_fixed_and_requires_video_ego_motion"
            )
            issues.append("paper_ego_camera_static_anchor_correction_unpublished")
        if scene == "BilliardGame" and str(case["view"]) == "third_person":
            issues.append("third_person_reuses_first_person_lite_case_with_runtime_overrides")
        return {
            "lite_case_id": lite,
            "dimension": "kinematics_mechanics",
            "status": (
                "blocked_paper_metric_parity"
                if lite is None or "lite_case_does_not_enable_kinematics_mechanics" in issues
                else "runnable_metric_reconstruction"
            ),
            "issues": issues,
        }

    lite = {
        "WanderingGarden": "WanderingGarden/scene_first",
        "MirrorRollingBall": "MirrorRollingBall/prompt_third",
        "BilliardGame": "BilliardGame/prompt_first",
    }[scene]
    issues = []
    status = "runnable_diagnostic_not_paper_comparable"
    if scene == "WanderingGarden":
        issues.append(
            "five_published_anchors_use_projection_assisted_canonical_assignment"
        )
        issues.append("published_detector_and_depth_backbone_settings_are_unavailable")
    elif scene == "MirrorRollingBall":
        issues.append("two_projection_assigned_foreground_sphere_anchors_added_from_unity_scene")
        issues.append("published_character_metric_uses_receipt_prompt_override")
    else:
        issues.append("paper_appendix_excludes_dynamic_scene_eoc_but_table_reports_distance")
        issues.append("third_person_camera_is_reconstructed_not_published")
        status = "blocked_paper_metric_parity"
    return {
        "lite_case_id": lite,
        "dimension": "environment_object_consistency",
        "status": status,
        "issues": issues,
        "source_case_id": case_id,
    }


def _run_command(
    case: dict[str, Any], frame: dict[str, Any], calibration: Path | None
) -> list[str]:
    command = [
        sys.executable,
        str((REPO_ROOT / "tools" / "run_project_genie_tables.py").resolve()),
        "run", "--resume", "--continue-on-error", "--allow-canvas-entry",
        "--auto-accept-upload-policy", "--allow-nominal-adapter",
        "--archive-stale-inputs", "--download-timeout-ms", "20000",
        "--case", str(case["case_id"]),
        "--first-frame-override", f"{case['case_id']}={frame['path']}",
    ]
    if int(case["table"]) == 7:
        variant = str(case["case_id"]).rsplit("/", 1)[-1]
        if calibration is not None:
            command.extend(["--turn-calibration", f"{case['case_id']}={calibration}"])
            if variant.endswith("_out"):
                command.append("--allow-opposite-turn-calibration")
            if str(case["view"]) == "third_person":
                command.append("--allow-cross-view-calibration")
            command.append("--allow-missing-native")
    else:
        command.append("--allow-missing-native")
    return command


def prepare(root: Path, output: Path) -> dict[str, Any]:
    manifest = _load_yaml(root / "cases.yaml")
    sources_doc = _load_yaml(root / "conditioning_sources.yaml")
    sources = sources_doc.get("sources", {})
    if set(sources) != {str(case["case_id"]) for case in manifest["cases"]}:
        raise ValueError("conditioning_sources.yaml must cover every case exactly")
    calibration_spec = sources_doc.get("backend_calibrations", {}).get("project_genie_yaw", {})
    calibration = _resolve(root, str(calibration_spec["path"])) if calibration_spec else None
    if calibration is not None and not calibration.is_file():
        calibration = None

    output.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        case_dir = output / "cases" / _safe_id(case_id)
        frame_spec = sources[case_id]
        frame: dict[str, Any] | None = None
        materialize_error: str | None = None
        if frame_spec.get("kind") != "missing":
            try:
                frame = _materialize_frame(root, frame_spec, case_dir / "conditioning.png")
            except Exception as exc:
                materialize_error = f"{type(exc).__name__}: {exc}"
        actions = _copy_actions(root, case, case_dir / "actions")
        status, issues = _status(case, actions, frame)
        if materialize_error:
            issues.append(f"conditioning_materialization_failed: {materialize_error}")
        if frame_spec.get("kind") == "missing":
            issues.append(str(frame_spec.get("reason", "source not configured")))
        prompt = _prompt(root, manifest, case)
        record = {
            "case_id": case_id,
            "table": int(case["table"]),
            "scene": str(case["scene"]),
            "view": str(case["view"]),
            "paper_orientation": case.get("paper_orientation"),
            "status": status,
            "issues": issues,
            "conditioning": frame,
            "actions": actions,
            "environment_prompt": prompt,
            "character_prompt": str(case.get("character_prompt", "")),
            "prompt_provenance_status": manifest.get("prompt_provenance", {}).get("status"),
            "evaluation": _evaluation_contract(case),
        }
        (case_dir / "case.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        case_rows.append(record)
        command = (
            _run_command(case, frame, calibration)
            if status not in {"blocked", "excluded"} and frame
            else None
        )
        if command:
            commands.append(command)
        for repeat_id in manifest["repeat_ids"]:
            queue.append(
                {
                    "case_id": case_id,
                    "repeat_id": str(repeat_id),
                    "status": "ready" if command else ("excluded" if status == "excluded" else "blocked"),
                    "issues": issues,
                    "command": command,
                }
            )

    status_counts: dict[str, int] = {}
    for row in case_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    evaluation_status_counts: dict[str, int] = {}
    for row in case_rows:
        evaluation = row["evaluation"]
        effective_status = (
            "blocked_generation_inputs"
            if row["status"] == "blocked"
            else ("excluded" if row["status"] == "excluded" else evaluation["status"])
        )
        evaluation_status_counts[effective_status] = (
            evaluation_status_counts.get(effective_status, 0)
            + len(manifest["repeat_ids"])
        )
    report = {
        "schema_version": 1,
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "case_count": len(case_rows),
        "repeat_count": len(queue),
        "status_counts": status_counts,
        "evaluation_status_counts": evaluation_status_counts,
        "backend_calibration": str(calibration) if calibration else None,
        "cases": case_rows,
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "generation_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "evaluation_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_rows:
            for repeat_id in manifest["repeat_ids"]:
                evaluation = row["evaluation"]
                effective_status = (
                    "blocked_generation_inputs"
                    if row["status"] == "blocked"
                    else ("excluded" if row["status"] == "excluded" else evaluation["status"])
                )
                handle.write(
                    json.dumps(
                        {
                            "case_id": row["case_id"],
                            "repeat_id": repeat_id,
                            "status": effective_status,
                            "generation_status": row["status"],
                            "evaluator_status": evaluation["status"],
                            "dimension": evaluation["dimension"],
                            "lite_case_id": evaluation["lite_case_id"],
                            "issues": list(
                                dict.fromkeys(row["issues"] + evaluation["issues"])
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    with (output / "asset_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "status", "issues", "conditioning_sha256"])
        writer.writeheader()
        for row in case_rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "status": row["status"],
                    "issues": "; ".join(row["issues"]),
                    "conditioning_sha256": (row["conditioning"] or {}).get("sha256", ""),
                }
            )
    agent = str((REPO_ROOT / "tools" / "project_genie_agent.py").resolve())
    script_lines = [
        "#!/bin/sh",
        "set -u",
        "failures=0",
        "",
        " ".join(shlex.quote(part) for part in [sys.executable, agent, "launch"])
        + " || exit $?",
        "",
    ]
    script_lines.extend(
        " ".join(shlex.quote(part) for part in command)
        + " || failures=$((failures + 1))"
        for command in commands
    )
    script_lines.extend(["", "exit \"$failures\""])
    runner = output / "run_ready_project_genie.sh"
    runner.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    runner.chmod(0o755)
    evaluation_driver = str(
        (REPO_ROOT / "tools" / "run_prepared_table_evaluations.py").resolve()
    )
    full_runner = output / "run_full_pipeline.sh"
    full_runner.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "set -u",
                "generation_code=0",
                f"{shlex.quote(str(runner.resolve()))} || generation_code=$?",
                "evaluation_code=0",
                " ".join(
                    shlex.quote(part)
                    for part in [
                        sys.executable,
                        evaluation_driver,
                        "--device",
                        "auto",
                        "--fail-incomplete",
                    ]
                )
                + " || evaluation_code=$?",
                "if [ \"$generation_code\" -ne 0 ]; then exit \"$generation_code\"; fi",
                "exit \"$evaluation_code\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    full_runner.chmod(0o755)
    md = [
        "# Table 4/7 prepared asset status", "",
        "Claim scope: paper-protocol reconstruction, not exact paper parity.", "",
        "Run `run_full_pipeline.sh` to resume browser generation and then prepare/evaluate every complete group.", "",
        (
            f"Generation queue: {sum(item['status'] == 'ready' for item in queue)} "
            f"runnable repeats and {sum(item['status'] == 'blocked' for item in queue)} "
            "explicit blockers. Evaluation queue: "
            f"{evaluation_status_counts.get('runnable_metric_reconstruction', 0)} "
            "runnable metric-reconstruction repeats, "
            f"{evaluation_status_counts.get('runnable_diagnostic_not_paper_comparable', 0)} "
            "runnable diagnostic repeats, "
            f"{evaluation_status_counts.get('blocked_paper_metric_parity', 0)} "
            "evaluator-parity blockers, and "
            f"{evaluation_status_counts.get('blocked_generation_inputs', 0)} "
            "generation-input blockers, and "
            f"{evaluation_status_counts.get('excluded', 0)} excluded repeats."
        ),
        "",
        (
            "Long evaluations checkpoint progress in "
            "`evaluation_automation_status.json`; rerunning the pipeline resumes "
            "completed rows by evaluator fingerprint."
        ),
        "",
        (
            "Run `uv run python tools/audit_table_reproduction.py` from the "
            "repository root to refresh the end-to-end reconstruction status."
        ),
        "",
        "| Case | Generation | Evaluation | Conditioning | Issues |",
        "|---|---|---|---|---|",
    ]
    for row in case_rows:
        digest = (row["conditioning"] or {}).get("sha256", "missing")
        evaluation = row["evaluation"]
        effective_evaluation = (
            "blocked_generation_inputs"
            if row["status"] == "blocked"
            else evaluation["status"]
        )
        combined_issues = list(
            dict.fromkeys(row["issues"] + evaluation["issues"])
        )
        md.append(
            f"| `{row['case_id']}` | {row['status']} | {effective_evaluation} | "
            f"`{digest[:12]}` | {'; '.join(combined_issues) or 'none'} |"
        )
    (output / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TABLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = prepare(args.root.resolve(), args.output_dir.resolve())
    print(json.dumps({key: report[key] for key in ("case_count", "repeat_count", "status_counts", "claim_scope")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
