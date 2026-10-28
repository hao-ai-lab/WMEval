#!/usr/bin/env python3
"""Relocate a Project Genie run without breaking path/hash provenance."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.project_genie_alignment import sha256


REPORT_HASH_FIELDS = {
    "native_video": "native_video_sha256",
    "observed_manifest": "observed_manifest_sha256",
    "action_trace": "action_trace_sha256",
    "trajectory_native": "trajectory_native_sha256",
}

RECEIPT_HASH_FIELDS = {
    "first_frame": "first_frame_sha256",
    "actions": "actions_sha256",
    "action_trace": "action_trace_sha256",
    "native_video": "native_video_sha256",
    "observed_frames": "observed_frames_sha256",
    "observed_video": "observed_video_sha256",
    "evaluation_video": "evaluation_video_sha256",
    "native_alignment_report": "native_alignment_report_sha256",
}


def _rewrite_prefixes(root: Path, old: str, new: str) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8")
        rewritten = content.replace(old, new)
        if rewritten != content:
            path.write_text(rewritten, encoding="utf-8")


def _refresh_hashes(payload: dict[str, Any], mapping: dict[str, str]) -> None:
    for path_key, hash_key in mapping.items():
        value = payload.get(path_key)
        if not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"recorded {path_key} does not exist: {path}")
        payload[hash_key] = sha256(path)


def relocate_run(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if source.parent != destination.parent:
        raise ValueError("archive destination must be a sibling of the run directory")
    if not (source / "receipt.json").is_file():
        raise FileNotFoundError(f"missing Project Genie receipt: {source / 'receipt.json'}")
    if destination.exists():
        raise FileExistsError(destination)

    old_prefix = str(source)
    source.rename(destination)
    _rewrite_prefixes(destination, old_prefix, str(destination))

    report_path = destination / "native_alignment.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _refresh_hashes(report, REPORT_HASH_FIELDS)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    receipt_path = destination / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _refresh_hashes(receipt, RECEIPT_HASH_FIELDS)
    screenshots = receipt.get("screenshots", {})
    if isinstance(screenshots, dict):
        for item in screenshots.values():
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            screenshot = Path(item["path"])
            if not screenshot.is_file():
                raise FileNotFoundError(f"recorded screenshot does not exist: {screenshot}")
            item["sha256"] = sha256(screenshot)
    receipt.update(
        {
            "archive_reason": "calibration_only_reused_world",
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "archived_from": old_prefix,
            "table_repeat_eligible": False,
        }
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stale = []
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".txt"}:
            if f"{old_prefix}/" in path.read_text(encoding="utf-8"):
                stale.append(str(path))
    if stale:
        raise RuntimeError(f"stale source paths remain after relocation: {stale}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = relocate_run(args.run_dir, args.archive_dir)
    print(
        json.dumps(
            {
                "archive_dir": str(args.archive_dir.expanduser().resolve()),
                "archive_reason": receipt["archive_reason"],
                "status": "relocated",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
