#!/usr/bin/env python3
"""Build an auditable validation table from Project Genie run receipts."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.project_genie_alignment import sha256
from evaluation.provenance_paths import resolve_recorded_path


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verified_file(
    receipt: dict[str, Any], path_key: str, hash_key: str, *, receipt_dir: Path
) -> Path:
    path_value = receipt.get(path_key)
    expected = receipt.get(hash_key)
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"receipt lacks {path_key}/{hash_key}")
    path = resolve_recorded_path(
        path_value, repo_root=REPO_ROOT, relative_to=receipt_dir
    )
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{path_key} is missing or hash-mismatched: {path}")
    return path


def build_summary(run_root: Path, repeat_ids: Sequence[str]) -> dict[str, Any]:
    run_root = run_root.resolve()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for repeat_id in repeat_ids:
        run_dir = run_root / repeat_id
        try:
            receipt = _read_object(run_dir / "receipt.json")
            artifact_status = receipt.get("evaluation_artifact_status")
            metrics: dict[str, Any] = {}
            if artifact_status == "native_action_aligned_ready":
                report_path = _verified_file(
                    receipt, "native_alignment_report", "native_alignment_report_sha256",
                    receipt_dir=run_dir,
                )
                video_path = _verified_file(
                    receipt, "evaluation_video", "evaluation_video_sha256",
                    receipt_dir=run_dir,
                )
                report = _read_object(report_path)
                metrics = report.get("metrics", {})
                artifact_checks = {
                    "native_artifact_ready": True,
                    "alignment_validated": report.get("status") == "validated",
                    "alignment_method_v2": report.get("method")
                    == "ordered_multi_checkpoint_sift_v2",
                    "nine_matches": metrics.get("matched_sample_count") == 9,
                    "strictly_increasing": metrics.get("strictly_increasing") is True,
                }
                evaluation_frame_count = receipt.get("evaluation_video_frame_count")
                evaluation_duration_sec = receipt.get("evaluation_video_duration_sec")
            elif artifact_status == "action_aligned_capture_ready":
                video_path = _verified_file(
                    receipt, "observed_video", "observed_video_sha256",
                    receipt_dir=run_dir,
                )
                _verified_file(
                    receipt, "action_trace", "action_trace_sha256", receipt_dir=run_dir
                )
                _verified_file(
                    receipt, "observed_frames", "observed_frames_sha256",
                    receipt_dir=run_dir,
                )
                artifact_checks = {
                    "action_capture_ready": True,
                    "action_timeline_declared": receipt.get(
                        "trajectory_validation_status"
                    )
                    in {
                        "reconstructed_nominal_timing",
                        "calibrated_visual_periodicity",
                        "calibrated_backend_adapter",
                        "closed_loop_visual_phase",
                    },
                }
                evaluation_frame_count = receipt.get("observed_frame_count")
                evaluation_duration_sec = receipt.get("observed_duration_sec")
            else:
                raise ValueError(f"unsupported evaluation artifact status {artifact_status!r}")
            checks = {
                "receipt_complete": receipt.get("status") == "complete",
                "repeat_id_matches": receipt.get("repeat_id") == repeat_id,
                "independent_generation": receipt.get("generation_mode")
                == "new_world_generation",
                "table_repeat_eligible": receipt.get("table_repeat_eligible") is True,
                **artifact_checks,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append(f"{repeat_id}: failed {', '.join(failed)}")
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "status": "validated" if not failed else "invalid",
                    "first_frame_sha256": receipt.get("first_frame_sha256"),
                    "actions_sha256": receipt.get("actions_sha256"),
                    "artifact_status": artifact_status,
                    "browser_duration_sec": receipt.get("observed_duration_sec"),
                    "native_frame_count": receipt.get("evaluation_video_frame_count"),
                    "native_duration_sec": receipt.get("evaluation_video_duration_sec"),
                    "native_video_sha256": receipt.get("native_video_sha256"),
                    "evaluation_frame_count": evaluation_frame_count,
                    "evaluation_duration_sec": evaluation_duration_sec,
                    "evaluation_video_sha256": sha256(video_path),
                    "evaluation_video": str(video_path),
                    "matched_checkpoints": metrics.get("matched_sample_count"),
                    "minimum_match": metrics.get("minimum_match"),
                    "minimum_margin": metrics.get("minimum_margin"),
                    "minimum_local_speed_ratio": metrics.get(
                        "minimum_local_speed_ratio"
                    ),
                    "maximum_local_speed_ratio": metrics.get(
                        "maximum_local_speed_ratio"
                    ),
                    "timeline_r_squared_diagnostic": metrics.get(
                        "timeline_r_squared"
                    ),
                    "upload_policy_handling_mode": receipt.get(
                        "upload_policy_handling_mode", "legacy_or_manual"
                    ),
                    "automated_policy_click_count": len(
                        receipt.get("upload_policy_acceptances", [])
                    ),
                }
            )
        except Exception as exc:
            errors.append(f"{repeat_id}: {exc}")

    if len(rows) == len(repeat_ids):
        first_hashes = {row["first_frame_sha256"] for row in rows}
        action_hashes = {row["actions_sha256"] for row in rows}
        evaluation_hashes = {row["evaluation_video_sha256"] for row in rows}
        if len(first_hashes) != 1:
            errors.append("repeat inputs do not share one first-frame hash")
        if len(action_hashes) != 1:
            errors.append("repeat inputs do not share one action hash")
        if len(evaluation_hashes) != len(rows):
            errors.append("evaluation outputs are not unique across repeats")

    def stats(key: str) -> dict[str, float] | None:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if len(values) != len(rows) or not values:
            return None
        return {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "schema_version": 1,
        "status": "validated" if not errors and len(rows) == len(repeat_ids) else "invalid",
        "run_root": str(run_root),
        "repeat_ids": list(repeat_ids),
        "generation_count": len(rows),
        "errors": errors,
        "aggregate": {
            "browser_duration_sec": stats("browser_duration_sec"),
            "native_duration_sec": stats("native_duration_sec"),
            "native_frame_count": stats("native_frame_count"),
            "evaluation_duration_sec": stats("evaluation_duration_sec"),
            "evaluation_frame_count": stats("evaluation_frame_count"),
        },
        "runs": rows,
    }


def write_summary(summary: dict[str, Any], output_prefix: Path) -> None:
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = summary["runs"]
    columns = list(rows[0]) if rows else ["repeat_id", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    visible = [
        "repeat_id",
        "status",
        "artifact_status",
        "browser_duration_sec",
        "evaluation_frame_count",
        "evaluation_duration_sec",
        "native_frame_count",
        "native_duration_sec",
        "matched_checkpoints",
        "minimum_match",
        "minimum_margin",
        "timeline_r_squared_diagnostic",
    ]
    lines = [
        "# Project Genie five-run validation",
        "",
        f"Status: **{summary['status']}**",
        "",
        "| " + " | ".join(visible) + " |",
        "| " + " | ".join("---" for _ in visible) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in visible) + " |")
    lines.extend(
        [
            "",
            "This table validates generation provenance and native/browser alignment. ",
            "It is not a Table 7 metric result and does not imply exact paper parity.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--repeat-id",
        action="append",
        dest="repeat_ids",
        default=[],
        help="repeat ID; defaults to run0 through run4",
    )
    args = parser.parse_args()
    repeat_ids = args.repeat_ids or [f"run{index}" for index in range(5)]
    summary = build_summary(args.run_root, repeat_ids)
    write_summary(summary, args.output_prefix)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
