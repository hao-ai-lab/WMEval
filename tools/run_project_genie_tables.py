#!/usr/bin/env python3
"""Plan or execute the resumable Project Genie Table 4/7 generation queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = REPO_ROOT / "experiments" / "protocol"
AGENT = REPO_ROOT / "tools" / "project_genie_agent.py"
ARCHIVABLE_STALE_STATUSES = {
    "input_mismatch_complete",
    "live_world_mismatch_complete",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_receipt_status(
    receipt_path: Path, expected_inputs: dict[str, Any] | None = None
) -> tuple[str, bool]:
    """Return a queue status only after validating the runnable artifact."""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_receipt", False
    status = str(receipt.get("status", "unknown"))
    if status != "complete":
        return status, False
    if receipt.get("reused_world_from") is not None or receipt.get(
        "generation_mode"
    ) == "revisit_existing_world" or receipt.get("table_repeat_eligible") is False:
        return "calibration_only_reused_world", False
    for key, expected in (expected_inputs or {}).items():
        if expected is not None and receipt.get(key) != expected:
            return "input_mismatch_complete", False
    page_url = receipt.get("page_url")
    if isinstance(page_url, str) and page_url:
        parsed = urlsplit(page_url)
        if parsed.path.endswith("/tools/projectgenie/w"):
            query = parse_qs(parsed.query)
            actual_prompt = query.get("w", [None])[0]
            actual_view = {
                "true": "third_person",
                "false": "first_person",
            }.get(query.get("use3p", [None])[0])
            expected_prompt = (expected_inputs or {}).get("environment_prompt")
            expected_view = (expected_inputs or {}).get("view")
            if (
                expected_prompt is not None
                and actual_prompt != expected_prompt
            ) or (expected_view is not None and actual_view != expected_view):
                return "live_world_mismatch_complete", False
    if receipt.get("trajectory_validation_status") not in {
        "calibrated_visual_periodicity",
        "calibrated_backend_adapter",
        "closed_loop_visual_phase",
        "reconstructed_nominal_timing",
    }:
        return "unvalidated_complete", False
    artifact_status = receipt.get("evaluation_artifact_status")
    if artifact_status == "native_action_aligned_ready":
        path_value = receipt.get("evaluation_video")
        expected_hash = receipt.get("evaluation_video_sha256")
        if receipt.get("native_video_timeline_alignment") != (
            "validated_visual_correspondence"
        ):
            return "invalid_evaluation_artifact", False
        alignment_path_value = receipt.get("native_alignment_report")
        alignment_hash = receipt.get("native_alignment_report_sha256")
        if not isinstance(alignment_path_value, str) or not isinstance(
            alignment_hash, str
        ):
            return "invalid_evaluation_artifact", False
        alignment_path = Path(alignment_path_value).expanduser()
        if not alignment_path.is_absolute():
            alignment_path = receipt_path.parent / alignment_path
        if not alignment_path.is_file() or sha256(alignment_path) != alignment_hash:
            return "invalid_evaluation_artifact", False
    elif artifact_status == "action_aligned_capture_ready":
        path_value = receipt.get("observed_video")
        expected_hash = receipt.get("observed_video_sha256")
    else:
        return "incomplete_evaluation_artifact", False
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        return "invalid_evaluation_artifact", False
    artifact = Path(path_value).expanduser()
    if not artifact.is_absolute():
        artifact = receipt_path.parent / artifact
    if not artifact.is_file() or sha256(artifact) != expected_hash:
        return "invalid_evaluation_artifact", False
    return "complete", True


def load_manifest(root: Path) -> dict[str, Any]:
    raw = yaml.safe_load((root / "cases.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"{root / 'cases.yaml'}: invalid table manifest")
    return raw


def table7_prompt(root: Path, manifest: dict[str, Any], case: dict[str, Any]) -> str:
    components_path = root / str(manifest.get("prompt_components", "prompt_components.yaml"))
    components = yaml.safe_load(components_path.read_text(encoding="utf-8"))
    scene = str(case["scene"])
    orientation = str(case["paper_orientation"])
    variant = str(case["case_id"]).rsplit("/", 1)[-1]
    motion = "perturb" if variant.startswith("perturb_") else "360"
    try:
        parts = [
            components["scenes"][scene],
            components["orientations"][orientation],
            components["motions"][motion],
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing Table 7 prompt component for {case['case_id']}: {exc}") from exc
    return " ".join(str(part).strip() for part in parts)


def task_plan(
    root: Path,
    output_root: Path,
    selected: set[str],
    first_frame_overrides: dict[str, Path] | None = None,
    selected_repeats: set[str] | None = None,
    turn_calibration_overrides: dict[str, Path] | None = None,
    closed_loop_yaw: bool = False,
    allow_nominal_adapter: bool = False,
) -> list[dict[str, Any]]:
    manifest = load_manifest(root)
    first_frame_overrides = first_frame_overrides or {}
    selected_repeats = selected_repeats or set()
    turn_calibration_overrides = turn_calibration_overrides or {}
    tasks: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        if selected and case_id not in selected:
            continue
        if case.get("execution_scope", "active") != "active":
            continue
        trajectory = (root / str(case["trajectory"])).resolve()
        compiled_actions = root / "compiled" / f"{trajectory.stem}.actions.jsonl"
        compiled_manifest = root / "compiled" / f"{trajectory.stem}.manifest.json"
        override = first_frame_overrides.get(case_id) or first_frame_overrides.get(str(case["scene"]))
        calibration = turn_calibration_overrides.get(case_id) or turn_calibration_overrides.get(
            str(case["scene"])
        )
        first_frame_raw = case.get("first_frame")
        prompt_source = (root / str(case["prompt_source"])).resolve()
        prompt = str(case.get("environment_prompt") or (
            table7_prompt(root, manifest, case)
            if case.get("table") == 7
            else yaml.safe_load(prompt_source.read_text(encoding="utf-8"))["prompt"]
        ))
        for repeat_id in manifest["repeat_ids"]:
            if selected_repeats and str(repeat_id) not in selected_repeats:
                continue
            out = output_root / case_id / str(repeat_id)
            receipt = out / "receipt.json"
            resolved_first_frame = override.resolve() if override else (
                (root / str(first_frame_raw)).resolve() if first_frame_raw else None
            )
            status = "pending"
            receipt_valid_complete = False
            has_semantic_yaw = False
            full_turn_only = False
            adapter_status: str | None = None
            compiled_actions_valid = False
            if compiled_actions.is_file() and compiled_actions.stat().st_size > 0:
                try:
                    action_rows = [
                        json.loads(line)
                        for line in compiled_actions.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
                    has_semantic_yaw = any(
                        "target_yaw_deg" in row for row in action_rows
                    )
                    full_turn_only = has_semantic_yaw and all(
                        abs(float(row["target_yaw_deg"])) >= 330.0
                        for row in action_rows
                        if "target_yaw_deg" in row
                    )
                    adapter_status = str(
                        json.loads(compiled_manifest.read_text(encoding="utf-8")).get(
                            "project_genie_status"
                        )
                    )
                    compiled_actions_valid = True
                except Exception:
                    compiled_actions_valid = False
            if receipt.is_file():
                expected_inputs: dict[str, Any] = {
                    "repeat_id": str(repeat_id),
                    "view": str(case["view"]),
                    "environment_prompt": str(prompt),
                    "character_prompt": str(case.get("character_prompt", "")),
                    "prompt_provenance_status": str(
                        manifest.get("prompt_provenance", {}).get("status", "missing")
                    ),
                    "actions_sha256": (
                        sha256(compiled_actions) if compiled_actions.is_file() else None
                    ),
                    "first_frame_sha256": (
                        sha256(resolved_first_frame)
                        if resolved_first_frame is not None
                        and resolved_first_frame.is_file()
                        else None
                    ),
                    "trajectory_validation_status": (
                        "calibrated_visual_periodicity"
                        if calibration is not None
                        else (
                            "closed_loop_visual_phase"
                            if closed_loop_yaw and full_turn_only
                            else (
                                "reconstructed_nominal_timing"
                                if allow_nominal_adapter
                                and adapter_status == "nominal"
                                else None
                            )
                        )
                    ),
                }
                status, receipt_valid_complete = completed_receipt_status(
                    receipt, expected_inputs
                )
            if receipt_valid_complete:
                pass
            elif resolved_first_frame is None:
                status = "blocked_missing_first_frame"
            elif not resolved_first_frame.is_file():
                status = "blocked_missing_first_frame"
            elif case.get("orientation_frame_status") == "unresolved" and override is None:
                status = "blocked_unresolved_orientation_frame"
            elif not compiled_actions.is_file() or compiled_actions.stat().st_size == 0:
                status = "blocked_unsupported_or_uncompiled_actions"
            elif not compiled_actions_valid:
                status = "blocked_invalid_compiled_actions"
            else:
                if closed_loop_yaw and has_semantic_yaw and not full_turn_only:
                    status = "blocked_closed_loop_only_supports_full_turn"
                elif (
                    adapter_status != "calibrated"
                    and calibration is None
                    and not (closed_loop_yaw and full_turn_only)
                    and not (allow_nominal_adapter and adapter_status == "nominal")
                ):
                    status = "blocked_uncalibrated_adapter"
                elif calibration is not None and not calibration.resolve().is_file():
                    status = "blocked_missing_turn_calibration"
                elif calibration is not None and not has_semantic_yaw:
                    status = "blocked_incompatible_turn_calibration"
            tasks.append(
                {
                    "case_id": case_id,
                    "repeat_id": str(repeat_id),
                    "status": status,
                    "first_frame": str(resolved_first_frame) if resolved_first_frame else None,
                    "first_frame_source": "override" if override else "manifest",
                    "actions": str(compiled_actions.resolve()),
                    "turn_calibration": str(calibration.resolve()) if calibration else None,
                    "closed_loop_yaw": bool(closed_loop_yaw and has_semantic_yaw),
                    "adapter_claim_scope": (
                        "reconstructed_nominal_timing"
                        if (
                            allow_nominal_adapter
                            and adapter_status == "nominal"
                            and calibration is None
                            and not (closed_loop_yaw and full_turn_only)
                        )
                        else "calibrated_or_closed_loop"
                    ),
                    "environment_prompt": str(prompt),
                    "character_prompt": str(case.get("character_prompt", "")),
                    "prompt_provenance_status": str(
                        manifest.get("prompt_provenance", {}).get("status", "missing")
                    ),
                    "view": str(case["view"]),
                    "output_dir": str(out.resolve()),
                }
            )
    unknown = selected - {str(case["case_id"]) for case in manifest["cases"]}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    known_keys = {str(case["scene"]) for case in manifest["cases"]} | {
        str(case["case_id"]) for case in manifest["cases"]
    }
    unknown_keys = set(first_frame_overrides) - known_keys
    if unknown_keys:
        raise ValueError(f"unknown first-frame override keys: {sorted(unknown_keys)}")
    unknown_calibration_keys = set(turn_calibration_overrides) - known_keys
    if unknown_calibration_keys:
        raise ValueError(
            f"unknown turn-calibration override keys: {sorted(unknown_calibration_keys)}"
        )
    unknown_repeats = selected_repeats - {str(item) for item in manifest["repeat_ids"]}
    if unknown_repeats:
        raise ValueError(f"unknown repeat ids: {sorted(unknown_repeats)}")
    return tasks


def parse_first_frame_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--first-frame-override must be SCENE=PATH")
        scene_or_case, raw_path = value.split("=", 1)
        if not scene_or_case or not raw_path:
            raise ValueError("--first-frame-override must be SCENE=PATH")
        overrides[scene_or_case] = Path(raw_path)
    return overrides


def archive_stale_task(
    task: dict[str, Any], output_root: Path, *, reason: str
) -> Path:
    """Recoverably rotate a completed slot whose declared inputs changed."""
    if reason not in ARCHIVABLE_STALE_STATUSES:
        raise ValueError(f"refusing to auto-archive non-input-stale status {reason!r}")
    root = output_root.resolve()
    source = Path(str(task["output_dir"])).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"run directory is outside output root: {source}") from exc
    if not source.is_dir():
        raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_parent = root / "stale_archive" / str(task["case_id"])
    archive_parent.mkdir(parents=True, exist_ok=True)
    base = f"{task['repeat_id']}_{stamp}"
    archive = archive_parent / base
    suffix = 1
    while archive.exists():
        archive = archive_parent / f"{base}_{suffix}"
        suffix += 1
    shutil.move(str(source), str(archive))
    record = {
        "schema_version": 1,
        "reason": reason,
        "source": str(source),
        "archive": str(archive),
        "case_id": task["case_id"],
        "repeat_id": task["repeat_id"],
        "replacement_inputs": {
            "first_frame": task.get("first_frame"),
            "actions": task.get("actions"),
            "environment_prompt": task.get("environment_prompt"),
            "character_prompt": task.get("character_prompt"),
            "view": task.get("view"),
        },
    }
    (archive / "stale_reason.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive


def run_task(
    task: dict[str, Any],
    *,
    port: int,
    resume: bool,
    allow_canvas_entry: bool,
    auto_accept_upload_policy: bool = False,
    reuse_current_world_from: Path | None = None,
    allow_missing_native: bool = False,
    download_timeout_ms: int = 90_000,
    capture_fps: float = 15.0,
    allow_cross_view_calibration: bool = False,
    allow_opposite_turn_calibration: bool = False,
) -> int:
    command = [
        sys.executable,
        str(AGENT),
        "--port", str(port),
        "run",
        "--first-frame", task["first_frame"],
        "--actions", task["actions"],
        "--environment-prompt", task["environment_prompt"],
        "--prompt-provenance-status", task["prompt_provenance_status"],
        "--view", task["view"],
        "--repeat-id", task["repeat_id"],
        "--output-dir", task["output_dir"],
        "--download-timeout-ms", str(download_timeout_ms),
        "--capture-fps", str(capture_fps),
        "--trajectory-claim-scope", task.get(
            "adapter_claim_scope", "calibrated_or_closed_loop"
        ),
    ]
    if task["character_prompt"]:
        command.extend(["--character-prompt", task["character_prompt"]])
    if task.get("turn_calibration"):
        command.extend(["--turn-calibration", task["turn_calibration"]])
    if allow_cross_view_calibration:
        command.append("--allow-cross-view-calibration")
    if allow_opposite_turn_calibration:
        command.append("--allow-opposite-turn-calibration")
    if task.get("closed_loop_yaw"):
        command.append("--closed-loop-yaw")
    if reuse_current_world_from is not None:
        command.extend(["--reuse-current-world-from", str(reuse_current_world_from.resolve())])
    if allow_missing_native:
        command.append("--allow-missing-native")
    if resume:
        command.append("--resume")
    if allow_canvas_entry:
        command.append("--allow-canvas-entry")
    if auto_accept_upload_policy:
        command.append("--auto-accept-upload-policy")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--root", type=Path, default=TABLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "project_genie")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repeat", action="append", default=[])
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--archive-stale-inputs",
        action="store_true",
        help=(
            "recoverably move completed slots with changed frame/prompt/view/action "
            "inputs under stale_archive before regenerating them"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-canvas-entry", action="store_true")
    parser.add_argument(
        "--auto-accept-upload-policy",
        action="store_true",
        help="explicitly authorize automatic acceptance of the upload policy dialog",
    )
    parser.add_argument("--reuse-current-world-from", type=Path)
    parser.add_argument("--allow-missing-native", action="store_true")
    parser.add_argument("--download-timeout-ms", type=int, default=90_000)
    parser.add_argument("--capture-fps", type=float, default=15.0)
    parser.add_argument("--closed-loop-yaw", action="store_true")
    parser.add_argument("--allow-cross-view-calibration", action="store_true")
    parser.add_argument("--allow-opposite-turn-calibration", action="store_true")
    parser.add_argument(
        "--allow-nominal-adapter",
        action="store_true",
        help=(
            "execute reconstructed fixed-duration browser controls and record "
            "their non-calibrated claim scope; unsupported actions remain blocked"
        ),
    )
    parser.add_argument(
        "--first-frame-override",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="use a freshly captured Unity frame for every case in SCENE",
    )
    parser.add_argument(
        "--turn-calibration",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="use a validated visual-yaw calibration report for a scene/case",
    )
    args = parser.parse_args()
    tasks = task_plan(
        args.root.resolve(),
        args.output_root.resolve(),
        set(args.case),
        parse_first_frame_overrides(args.first_frame_override),
        set(args.repeat),
        parse_first_frame_overrides(args.turn_calibration),
        args.closed_loop_yaw,
        args.allow_nominal_adapter,
    )
    if args.command == "plan":
        summary: dict[str, int] = {}
        for task in tasks:
            summary[task["status"]] = summary.get(task["status"], 0) + 1
        print(json.dumps({"task_count": len(tasks), "status_counts": summary, "tasks": tasks}, indent=2))
        return 0

    if args.reuse_current_world_from is not None and len(tasks) != 1:
        raise ValueError("--reuse-current-world-from requires exactly one selected task")

    failures = 0
    for task in tasks:
        if task["status"] == "complete":
            continue
        if task["status"].startswith("blocked_"):
            failures += 1
            if not args.continue_on_error:
                break
            continue
        if task["status"] in ARCHIVABLE_STALE_STATUSES and args.archive_stale_inputs:
            archive_stale_task(
                task, args.output_root, reason=str(task["status"])
            )
        code = run_task(
            task,
            port=args.port,
            resume=args.resume,
            allow_canvas_entry=args.allow_canvas_entry,
            auto_accept_upload_policy=args.auto_accept_upload_policy,
            reuse_current_world_from=args.reuse_current_world_from,
            allow_missing_native=args.allow_missing_native,
            download_timeout_ms=args.download_timeout_ms,
            capture_fps=args.capture_fps,
            allow_cross_view_calibration=args.allow_cross_view_calibration,
            allow_opposite_turn_calibration=args.allow_opposite_turn_calibration,
        )
        if code:
            failures += 1
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
