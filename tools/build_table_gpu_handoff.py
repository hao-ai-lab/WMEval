#!/usr/bin/env python3
"""Inventory, verify, and copy the minimal Table 4/7 GPU handoff assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.provenance_paths import resolve_recorded_path
from tools.run_prepared_table_evaluations import PLANS

DEFAULT_MANIFEST = REPO_ROOT / "experiments/protocol/gpu_handoff_manifest.json"
REPEATS = tuple(f"run{index}" for index in range(5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"handoff dependency is outside the repository: {path}") from exc


def _recorded(receipt: dict[str, Any], key: str, receipt_dir: Path) -> Path:
    value = receipt.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{receipt_dir / 'receipt.json'} lacks {key}")
    return resolve_recorded_path(
        value, repo_root=REPO_ROOT, relative_to=receipt_dir
    )


def _add(files: dict[str, set[str]], path: Path, role: str) -> None:
    relative = _repo_relative(path)
    files.setdefault(relative, set()).add(role)


def discover() -> tuple[dict[str, set[str]], list[str]]:
    files: dict[str, set[str]] = {}
    errors: list[str] = []
    for plan in PLANS:
        run_root = REPO_ROOT / "runs/project_genie" / plan.case_id
        receipts: list[tuple[Path, dict[str, Any]]] = []
        for repeat in REPEATS:
            receipt_path = run_root / repeat / "receipt.json"
            if not receipt_path.is_file():
                errors.append(f"missing {receipt_path.relative_to(REPO_ROOT)}")
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipts.append((receipt_path, receipt))
                _add(files, receipt_path, f"{plan.case_id}:{repeat}:receipt")
            except Exception as exc:
                errors.append(f"{receipt_path.relative_to(REPO_ROOT)}: {exc}")
        if len(receipts) != len(REPEATS):
            continue
        all_native = all(
            receipt.get("evaluation_artifact_status") == "native_action_aligned_ready"
            for _, receipt in receipts
        )
        for receipt_path, receipt in receipts:
            repeat = str(receipt.get("repeat_id"))
            status = receipt.get("evaluation_artifact_status")
            try:
                if status == "native_action_aligned_ready":
                    _add(files, _recorded(receipt, "evaluation_video", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:evaluation_video")
                    _add(files, _recorded(receipt, "native_alignment_report", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:alignment_report")
                elif status == "action_aligned_capture_ready":
                    _add(files, _recorded(receipt, "observed_video", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:observed_video")
                    _add(files, _recorded(receipt, "observed_frames", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:observed_frames")
                    _add(files, _recorded(receipt, "action_trace", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:action_trace")
                else:
                    raise ValueError(f"unsupported evaluation_artifact_status {status!r}")
                if not all_native:
                    _add(files, _recorded(receipt, "observed_video", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:common_browser_source")
                if plan.trajectory_mode == "perturbation":
                    _add(files, _recorded(receipt, "action_trace", receipt_path.parent),
                         f"{plan.case_id}:{repeat}:trajectory_timing")
            except Exception as exc:
                errors.append(f"{plan.case_id}:{repeat}: {exc}")

    capture_root = REPO_ROOT / "runs/tables_4_7/unity_scene_runs/captures"
    capture_files = {
        "table4__RacingCar__third": (
            "evaluation_artifacts/object_transforms.json",
            "evaluation_artifacts/poses.npy",
            "evaluation_artifacts/intrinsics.npy",
            "evaluation_artifacts/object_masses.json",
            "evaluation_artifacts/plane_calibration.json",
        ),
        "table4__BilliardGame__first": (
            "evaluation_artifacts/object_transforms.json",
            "evaluation_artifacts/poses.npy",
            "evaluation_artifacts/intrinsics.npy",
            "evaluation_artifacts/object_masses.json",
            "evaluation_artifacts/plane_calibration.json",
            "unity_groundtruth.mp4",
        ),
    }
    for plan in PLANS:
        if plan.preparation == "billiard_unity_capture":
            capture_files[plan.case_id.replace("/", "__")] = (
                "evaluation_artifacts/object_transforms.json",
                "evaluation_artifacts/poses.npy",
                "evaluation_artifacts/intrinsics.npy",
                "unity_groundtruth.mp4",
            )
    for capture, names in capture_files.items():
        for name in names:
            path = capture_root / capture / name
            if path.is_file():
                _add(files, path, f"unity_capture:{capture}")
            else:
                errors.append(f"missing {path.relative_to(REPO_ROOT)}")

    garden_receipt = REPO_ROOT / "runs/unity/garden_bridge_gt/capture_receipt.json"
    if garden_receipt.is_file():
        _add(files, garden_receipt, "unity_capture:garden:receipt")
        try:
            payload = json.loads(garden_receipt.read_text(encoding="utf-8"))
            for name, record in payload.get("evaluation_artifacts", {}).items():
                path = resolve_recorded_path(
                    record["path"], repo_root=REPO_ROOT,
                    relative_to=garden_receipt.parent,
                )
                _add(files, path, f"unity_capture:garden:{name}")
        except Exception as exc:
            errors.append(f"{garden_receipt.relative_to(REPO_ROOT)}: {exc}")
    else:
        errors.append(f"missing {garden_receipt.relative_to(REPO_ROOT)}")
    return files, errors


def build_manifest() -> dict[str, Any]:
    files, errors = discover()
    rows: list[dict[str, Any]] = []
    for relative, roles in sorted(files.items()):
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "roles": sorted(roles),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "status": "ready" if not errors else "incomplete",
        "branch": "lanxiang/release-cleanup",
        "source_commit_at_inventory": commit or None,
        "source_tree_dirty_at_inventory": dirty,
        "active_case_count": len(PLANS),
        "repeat_count_per_case": len(REPEATS),
        "active_generation_count": len(PLANS) * len(REPEATS),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "excluded_cells": ["table4/RacingCar/first", "table4/BilliardGame/third"],
        "errors": sorted(set(errors)),
        "files": rows,
    }


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in manifest.get("files", []):
        path = REPO_ROOT / str(row.get("path"))
        if not path.is_file():
            errors.append(f"missing {row.get('path')}")
            continue
        if path.stat().st_size != row.get("bytes"):
            errors.append(f"size mismatch {row.get('path')}")
        elif sha256(path) != row.get("sha256"):
            errors.append(f"SHA-256 mismatch {row.get('path')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--copy-to", type=Path,
        help="copy every manifest file under this destination repo root",
    )
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = verify_manifest(manifest)
    else:
        manifest = build_manifest()
        errors = list(manifest["errors"])
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.copy_to and not errors:
        destination = args.copy_to.expanduser().resolve()
        for row in manifest["files"]:
            source = REPO_ROOT / row["path"]
            target = destination / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        target_manifest = destination / args.manifest.relative_to(REPO_ROOT)
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, target_manifest)
    print(
        json.dumps(
            {
                "status": "ready" if not errors else "incomplete",
                "manifest": str(args.manifest.resolve()),
                "file_count": manifest.get("file_count"),
                "total_bytes": manifest.get("total_bytes"),
                "errors": errors,
                "copied_to": str(args.copy_to.resolve()) if args.copy_to else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
