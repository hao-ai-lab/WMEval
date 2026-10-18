#!/usr/bin/env python3
"""Audit Table 4 action-channel, timing, view, and repeat coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "experiments/tables_4_7"
RUN_ROOT = ROOT / "runs/project_genie/table4"


def _media_tool(name: str) -> str:
    """Resolve a system binary or this repository's pinned local fallback."""
    system = shutil.which(name)
    if system:
        return system
    bundled = ROOT / ".local" / "bin" / name
    if bundled.is_file():
        return str(bundled)
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_timeline(capture: Path) -> dict[str, Any]:
    """Read a full frame sequence or the compact package's ground-truth MP4."""
    frame_count = len(list((capture / "frames").glob("frame_*.png")))
    if frame_count == 300:
        return {
            "frame_count": frame_count,
            "duration_sec": frame_count / 30.0,
            "source": "png_sequence",
        }
    video = capture / "unity_groundtruth.mp4"
    if not video.is_file():
        transforms = capture / "evaluation_artifacts/object_transforms.json"
        if transforms.is_file():
            try:
                document = json.loads(transforms.read_text(encoding="utf-8"))
                frame_sets = [
                    tuple(int(index) for index in value["frames"])
                    for value in document["objects"].values()
                    if isinstance(value, dict) and value.get("frames")
                ]
                artifact_count = max(max(indices) for indices in frame_sets) + 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                artifact_count = 0
            if artifact_count:
                return {
                    "frame_count": artifact_count,
                    "duration_sec": artifact_count / 30.0,
                    "source": "evaluation_artifact_timeline",
                    "object_transforms": str(transforms),
                    "object_transforms_sha256": sha256(transforms),
                }
        return {
            "frame_count": frame_count,
            "duration_sec": frame_count / 30.0,
            "source": "incomplete_png_sequence",
        }
    command = [
        _media_tool("ffprobe"),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        stream = json.loads(completed.stdout)["streams"][0]
        count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
        video_count = int(count_value)
        duration = float(stream.get("duration", video_count / 30.0))
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, TypeError, ValueError):
        return {
            "frame_count": frame_count,
            "duration_sec": frame_count / 30.0,
            "source": "ffprobe_failed_png_fallback",
        }
    return {
        "frame_count": video_count,
        "duration_sec": duration,
        "source": "unity_groundtruth_mp4",
        "video": str(video),
        "video_sha256": sha256(video),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def state(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def trajectory_coverage(path: Path) -> dict[str, Any]:
    doc = load_yaml(path)
    channels = doc["channels"]
    segments = doc["segments"]
    coverage: dict[str, Any] = {}
    for name, spec in channels.items():
        durations: dict[str, float] = {}
        values: set[float] = set()
        for segment in segments:
            value = float(segment["values"][name])
            duration = float(segment["end_sec"]) - float(segment["start_sec"])
            values.add(value)
            key = state(value) if spec["kind"] == "continuous" else str(int(value))
            durations[key] = durations.get(key, 0.0) + duration
        expected = (
            {"negative", "zero", "positive"}
            if spec["kind"] == "continuous" and float(spec["min"]) < 0 < float(spec["max"])
            else ({"zero", "positive"} if spec["kind"] == "continuous" else {"0", "1"})
        )
        observed = set(durations)
        coverage[name] = {
            "kind": spec["kind"],
            "unit": spec["unit"],
            "values": sorted(values),
            "state_duration_sec": {key: round(value, 6) for key, value in sorted(durations.items())},
            "expected_states": sorted(expected),
            "observed_states": sorted(observed),
            "missing_states": sorted(expected - observed),
            "state_coverage_fraction": len(observed & expected) / len(expected),
        }
    compiled = TABLE_ROOT / "compiled" / f"{path.stem}.actions.jsonl"
    actions = [json.loads(line) for line in compiled.read_text().splitlines() if line.strip()]
    return {
        "trajectory_id": doc["trajectory_id"],
        "source": str(path),
        "duration_sec": float(doc["duration_sec"]),
        "control_hz": int(doc["control_hz"]),
        "segment_count": len(segments),
        "browser_action_count": len(actions),
        "browser_action_sha256": sha256(compiled),
        "channels": coverage,
        "project_genie_adapter_status": doc["adapters"]["project_genie"]["status"],
        "paper_alignment_status": doc["paper_alignment"]["status"],
        "paper_alignment_missing": doc["paper_alignment"]["missing"],
    }


def run_coverage(
    case_id: str, expected_action_hash: str, expected_first_frame_hash: str
) -> dict[str, Any]:
    _, scene, view = case_id.split("/")
    root = RUN_ROOT / scene / view
    rows: list[dict[str, Any]] = []
    for index in range(5):
        receipt_path = root / f"run{index}" / "receipt.json"
        if not receipt_path.is_file():
            rows.append({"repeat_id": f"run{index}", "status": "missing"})
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        video_value = receipt.get("evaluation_video") or receipt.get("observed_video")
        video = Path(video_value) if isinstance(video_value, str) else None
        trace_value = receipt.get("action_trace")
        trace = Path(trace_value) if isinstance(trace_value, str) else None
        rows.append(
            {
                "repeat_id": f"run{index}",
                "status": receipt.get("status"),
                "eligible": receipt.get("table_repeat_eligible") is not False,
                "view": receipt.get("view"),
                "action_hash_matches": receipt.get("actions_sha256") == expected_action_hash,
                "first_frame_hash_matches": receipt.get("first_frame_sha256")
                == expected_first_frame_hash,
                "action_trace_present": bool(trace and trace.is_file()),
                "evaluation_video_present": bool(video and video.is_file()),
                "duration_sec": receipt.get("observed_duration_sec"),
                "artifact_status": receipt.get("evaluation_artifact_status"),
            }
        )
    valid = [
        row
        for row in rows
        if row.get("status") == "complete"
        and row.get("eligible")
        and row.get("action_hash_matches")
        and row.get("first_frame_hash_matches")
        and row.get("action_trace_present")
        and row.get("evaluation_video_present")
    ]
    durations = [float(row["duration_sec"]) for row in valid if row.get("duration_sec") is not None]
    return {
        "required_repeats": 5,
        "valid_repeats": len(valid),
        "complete": len(valid) == 5,
        "duration_mean_sec": statistics.fmean(durations) if durations else None,
        "duration_sample_std_sec": statistics.stdev(durations) if len(durations) > 1 else None,
        "runs": rows,
    }


def case_gaps(case_id: str) -> tuple[str, list[str]]:
    if case_id == "table4/RacingCar/first":
        return (
            "generation_complete_metric_blocked",
            [
                "ego car is camera-fixed; object tracking alone cannot recover ego translation",
                "video-derived ego-motion estimator is required to avoid injecting Unity camera poses",
                "reverse throttle is not exercised",
            ],
        )
    if case_id == "table4/RacingCar/third":
        return (
            "metric_ready",
            ["reverse throttle is not exercised", "exact published numeric action timeline is unavailable"],
        )
    billiard_gaps = [
            "hosted Genie exposes navigation keys, not a guaranteed ball-force API",
            "visual review shows camera/table drift in some runs, so structural replay does not prove ball control",
            "backward control is not exercised",
            "cue angle, strength, hit point/spin, execute, and force-end are outside the standardized WASD protocol",
            "exact published numeric action timeline is unavailable",
    ]
    if case_id == "table4/BilliardGame/first":
        billiard_gaps.insert(
            2,
            "visual plane calibration survives only 1/300 frames while the Unity camera translates, so timestamp-matched Unity pose lifting can leak reference motion",
        )
        return "diagnostic_only_camera_pose_leakage_risk", billiard_gaps
    return "metric_ready_action_target_ambiguous", billiard_gaps


def build() -> dict[str, Any]:
    manifest = load_yaml(TABLE_ROOT / "cases.yaml")
    cases: dict[str, Any] = {}
    trajectory_cache: dict[Path, dict[str, Any]] = {}
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        if not case_id.startswith("table4/"):
            continue
        if case.get("execution_scope", "active") != "active":
            continue
        trajectory = (TABLE_ROOT / str(case["trajectory"])).resolve()
        trajectory_info = trajectory_cache.setdefault(trajectory, trajectory_coverage(trajectory))
        capture = ROOT / "runs/tables_4_7/unity_scene_runs/captures" / case_id.replace("/", "__")
        timeline = reference_timeline(capture)
        artifacts = capture / "evaluation_artifacts"
        first_frame = capture / "frames/frame_0000.png"
        required_artifacts = [
            "object_transforms.json",
            "poses.npy",
            "intrinsics.npy",
            "object_masses.json",
            "plane_calibration.json",
        ]
        evaluation_status, gaps = case_gaps(case_id)
        cases[case_id] = {
            "view": case["view"],
            "control_mode": case.get("control_mode"),
            "trajectory": trajectory_info,
            "unity_reference": {
                **timeline,
                "required_artifacts": {
                    name: (artifacts / name).is_file() for name in required_artifacts
                },
                "complete": timeline["frame_count"] == 300
                and all((artifacts / name).is_file() for name in required_artifacts),
            },
            "project_genie_runs": run_coverage(
                case_id,
                trajectory_info["browser_action_sha256"],
                sha256(first_frame) if first_frame.is_file() else "missing",
            ),
            "evaluation_status": evaluation_status,
            "gaps": gaps,
        }
    racing_hashes = {cases[key]["trajectory"]["browser_action_sha256"] for key in cases if "/RacingCar/" in key}
    return {
        "schema_version": 1,
        "status": "complete" if all(row["project_genie_runs"]["complete"] for row in cases.values()) else "incomplete",
        "claim_scope": "structural_action_coverage_and_declared_semantic_gaps",
        "table4_case_count": len(cases),
        "required_generation_count": 5 * len(cases),
        "valid_generation_count": sum(row["project_genie_runs"]["valid_repeats"] for row in cases.values()),
        "cross_view_action_identity": {
            "RacingCar": len(racing_hashes) == 1,
        },
        "cases": cases,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Table 4 action-trajectory coverage",
        "",
        f"Status: **{report['status']}**. Valid generations: "
        f"**{report['valid_generation_count']}/{report['required_generation_count']}**.",
        "",
        "Structural replay validity does not by itself prove that Project Genie applied a key to the prompted object. Semantic action-target limitations are listed separately.",
        "",
        "| Case | Unity | Genie repeats | Browser actions | Cross-channel coverage | Evaluation status |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case_id, case in report["cases"].items():
        channels = case["trajectory"]["channels"]
        channel_text = "; ".join(
            f"{name}: {','.join(info['observed_states'])}"
            for name, info in channels.items()
        )
        lines.append(
            f"| `{case_id}` | {'ready' if case['unity_reference']['complete'] else 'incomplete'} | "
            f"{case['project_genie_runs']['valid_repeats']}/5 | "
            f"{case['trajectory']['browser_action_count']} | {channel_text} | "
            f"`{case['evaluation_status']}` |"
        )
    lines.extend(["", "## Gaps", ""])
    for case_id, case in report["cases"].items():
        lines.append(f"### `{case_id}`")
        lines.append("")
        for gap in case["gaps"]:
            lines.append(f"- {gap}")
        lines.append("")
    lines.extend(
        [
            "## Cross-view standardization",
            "",
            f"- RacingCar first/third action files identical: **{str(report['cross_view_action_identity']['RacingCar']).lower()}**.",
            "- BilliardGame third-person is excluded from the active suite by user request.",
            "- Each active view has its own Unity conditioning frame and 300-frame ground-truth capture.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/tables_4_7/project_genie/table4_action_coverage",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build()
    json_path = output / "coverage.json"
    markdown_path = output / "coverage.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
