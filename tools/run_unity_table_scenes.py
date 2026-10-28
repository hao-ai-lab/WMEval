#!/usr/bin/env python3
"""Build and capture the eight reconstructed Table 4/7 Unity cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/protocol/unity_scene_runs.yaml"
DEFAULT_OUTPUT = ROOT / "runs/tables_4_7/unity_scene_runs"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("cases"), dict):
        raise ValueError(f"invalid Unity scene-run config: {path}")
    return config


def paths(config: dict[str, Any], output: Path, case: dict[str, Any]) -> tuple[Path, Path, Path]:
    unity_repo = (ROOT / str(config["unity_repo"])).resolve()
    project = unity_repo / str(case["project"])
    build_dir = output / "builds" / str(case["product"])
    app = build_dir / f"{case['product']}.app"
    binary = app / "Contents/MacOS" / str(case["product"])
    return project, app, binary


def build_projects(config: dict[str, Any], output: Path, selected: list[str]) -> None:
    editor = Path(str(config["unity_editor"]))
    if not editor.is_file():
        raise FileNotFoundError(f"Unity editor not found: {editor}")
    built: set[tuple[str, str]] = set()
    for case_id in selected:
        case = config["cases"][case_id]
        key = (str(case["project"]), str(case["product"]))
        if key in built:
            continue
        built.add(key)
        project, app, _binary = paths(config, output, case)
        build_dir = app.parent
        build_dir.mkdir(parents=True, exist_ok=True)
        log = output / "logs" / f"build_{case['product']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(editor), "-batchmode", "-quit", "-projectPath", str(project),
            "-executeMethod", "BuildScript.BuildMacOS", "-buildPath", str(build_dir),
            "-productName", str(case["product"]), "-scenes", str(case["scene"]),
            "-logFile", str(log),
        ]
        subprocess.run(command, check=True)
        if not app.is_dir():
            raise RuntimeError(f"Unity reported success but did not create {app}")


def encode_frames(capture_dir: Path, fps: float, count: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    output = capture_dir / "unity_groundtruth.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(capture_dir / "frames/frame_%04d.png"), "-frames:v", str(count),
         "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output)],
        check=True,
    )
    return output


def _project_world_point(
    point: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray | None:
    camera = np.linalg.inv(pose) @ np.append(point, 1.0)
    if camera[2] <= 1e-6:
        return None
    fx, fy, cx, cy = intrinsics
    return np.asarray(
        [fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy],
        dtype=np.float64,
    )


def _export_planar_calibration(
    artifact_dir: Path,
    first_frame: Path,
    rows: list[dict[str, Any]],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    kind: str,
) -> tuple[Path, Path]:
    """Export static visual correspondences from image pixels to Unity X/Z.

    Table-4 objects move on a horizontal plane.  Strong visual features in
    Unity frame zero are therefore useful metric anchors without adding
    artificial fiducials to the rendered scene.  The generated rollout starts
    from this conditioning frame; the evaluator tracks the same features and
    fits a robust per-frame image-to-plane homography.
    """
    image = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unreadable Unity first frame: {first_frame}")
    height, width = image.shape[:2]
    first = rows[0]
    if kind == "racing":
        moving_points = [np.asarray(first["car_pos"], dtype=np.float64)]
    else:
        moving_points = [
            np.asarray(point, dtype=np.float64)
            for point in first.get("balls", {}).values()
        ]
    plane_y = float(
        first.get("motion_plane_y", np.median([point[1] for point in moving_points]))
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_points = [
        np.asarray(point, dtype=np.float64)
        for point in first.get("metric_anchors", {}).values()
    ]
    mask = np.zeros(gray.shape, dtype=np.uint8) if marker_points else np.full(gray.shape, 255, dtype=np.uint8)
    marker_radius = max(18, min(width, height) // 30)
    for point in marker_points:
        uv = _project_world_point(point, poses[0], intrinsics[0])
        if uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
            cv2.circle(mask, tuple(np.rint(uv).astype(int)), marker_radius, 255, -1)
    margin = max(8, min(width, height) // 60)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    # A moving object must never become a calibration anchor.  Exclude a
    # generous disk around each frame-zero simulator projection.
    exclusion_radius = max(24, min(width, height) // 16)
    dynamic_exclusion_disks: list[dict[str, Any]] = []
    for point in moving_points:
        uv = _project_world_point(point, poses[0], intrinsics[0])
        if uv is not None:
            cv2.circle(mask, tuple(np.rint(uv).astype(int)), exclusion_radius, 0, -1)
            dynamic_exclusion_disks.append(
                {"center_uv": [round(float(uv[0]), 4), round(float(uv[1]), 4)], "radius_px": exclusion_radius}
            )

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=96,
        qualityLevel=0.01,
        minDistance=max(12, min(width, height) // 45),
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    anchors: list[dict[str, Any]] = []
    pose = poses[0]
    rotation = pose[:3, :3]
    origin = pose[:3, 3]
    max_horizontal_distance = 120.0 if kind == "racing" else 12.0
    if corners is not None:
        fx, fy, cx, cy = intrinsics[0]
        for uv_raw in corners.reshape(-1, 2):
            u, v = (float(uv_raw[0]), float(uv_raw[1]))
            ray_camera = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
            ray_world = rotation @ ray_camera
            if abs(ray_world[1]) <= 1e-8:
                continue
            scale = (plane_y - origin[1]) / ray_world[1]
            if scale <= 0:
                continue
            world = origin + scale * ray_world
            if float(np.linalg.norm(world[[0, 2]] - origin[[0, 2]])) > max_horizontal_distance:
                continue
            anchors.append(
                {
                    "id": f"anchor_{len(anchors):03d}",
                    "conditioning_uv": [round(u, 4), round(v, 4)],
                    "world_xz_m": [round(float(world[0]), 6), round(float(world[2]), 6)],
                }
            )

    calibration = {
        "schema_version": 1,
        "status": "ready" if len(anchors) >= 4 else "insufficient_visual_anchors",
        "method": "unity_conditioning_features_to_horizontal_plane",
        "unit": "meter",
        "plane": {"axis": "y", "value_m": plane_y, "output_axes": ["x", "z"]},
        "image_size": [width, height],
        "minimum_correspondences": 4,
        "anchor_region": "explicit_unity_metric_markers" if marker_points else "static_scene_features",
        "marker_world_xz_m": [
            [round(float(point[0]), 6), round(float(point[2]), 6)]
            for point in marker_points
        ],
        "marker_search_radius_px": marker_radius,
        "dynamic_exclusion_disks": dynamic_exclusion_disks,
        "anchors": anchors,
        "notes": [
            "Anchors are static Shi-Tomasi visual features from Unity frame zero.",
            "Moving simulator objects are excluded before feature selection.",
            "The evaluator tracks anchors in the generated video and estimates a per-frame RANSAC homography.",
        ],
    }
    calibration_path = artifact_dir / "plane_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")

    overlay = image.copy()
    for anchor in anchors:
        uv = tuple(np.rint(anchor["conditioning_uv"]).astype(int))
        cv2.circle(overlay, uv, 4, (0, 255, 255), -1, cv2.LINE_AA)
    overlay_path = artifact_dir / "plane_calibration_anchors.png"
    cv2.imwrite(str(overlay_path), overlay)
    return calibration_path, overlay_path


def export_artifacts(capture_dir: Path, kind: str, fps: float) -> dict[str, Any]:
    raw = json.loads((capture_dir / "gt.json").read_text(encoding="utf-8"))
    rows = raw["frames"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("capture ground truth has no usable frames")
    artifact_dir = capture_dir / "evaluation_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    poses = np.asarray([row["c2w"] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
    intrinsics = np.asarray([[row["fx"], row["fy"], row["cx"], row["cy"]] for row in rows])
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:, :2] <= 0):
        raise ValueError("capture contains invalid camera intrinsics")
    if not np.allclose(intrinsics[:, 0], intrinsics[:, 1], rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Unity RenderTexture capture requires square-pixel intrinsics (fx == fy)"
        )
    np.save(artifact_dir / "poses.npy", poses, allow_pickle=False)
    np.save(artifact_dir / "intrinsics.npy", intrinsics, allow_pickle=False)
    objects: dict[str, Any] = {}
    if kind == "racing":
        objects["car"] = {
            "frames": list(range(len(rows))),
            "positions_world": [row["car_pos"] for row in rows],
        }
    else:
        labels = {"Ball_0": "white ball", "Ball_1": "yellow ball", "Ball_2": "red ball"}
        for source, label in labels.items():
            objects[label] = {
                "frames": list(range(len(rows))),
                "positions_world": [row["balls"][source] for row in rows],
            }
    transforms = artifact_dir / "object_transforms.json"
    transforms.write_text(json.dumps({"schema_version": 1, "objects": objects}, indent=2) + "\n")
    masses = artifact_dir / "object_masses.json"
    if kind == "racing":
        mass = raw.get("car_mass_kg") if isinstance(raw, dict) else None
        if not isinstance(mass, (int, float)) or isinstance(mass, bool) or mass <= 0:
            raise ValueError("Racing capture ground truth lacks a positive car_mass_kg")
        mass_values = {"car": float(mass)}
    else:
        # The private CaromBilliard prefab and the locked Lite artifact both
        # declare the same real ball mass. Keep all three captured identities.
        mass_values = {label: 0.1701 for label in objects}
    masses.write_text(json.dumps(mass_values, indent=2, sort_keys=True) + "\n")
    video = encode_frames(capture_dir, fps, len(rows))
    first_frame = capture_dir / "frames/frame_0000.png"
    image = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if image is None or float(image.std()) < 2.0:
        raise RuntimeError(f"flat or unreadable Unity first frame: {first_frame}")
    calibration, calibration_overlay = _export_planar_calibration(
        artifact_dir, first_frame, rows, poses, intrinsics, kind
    )
    artifacts = [
        artifact_dir / "poses.npy",
        artifact_dir / "intrinsics.npy",
        transforms,
        masses,
        video,
        first_frame,
        calibration,
        calibration_overlay,
    ]
    return {path.name: {"path": str(path), "sha256": digest(path)} for path in artifacts}


def write_capture_status(
    output: Path, config: dict[str, Any], updated: list[dict[str, Any]]
) -> None:
    """Merge partial recaptures without dropping receipts for untouched cells."""
    receipt = output / "capture_status.json"
    by_case: dict[str, dict[str, Any]] = {}
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        for row in previous.get("cases", []):
            if isinstance(row, dict) and isinstance(row.get("case_id"), str):
                by_case[row["case_id"]] = row
    for row in updated:
        by_case[str(row["case_id"])] = row
    ordered = [by_case[case_id] for case_id in config["cases"] if case_id in by_case]
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": len(ordered),
                "complete_for_config": len(ordered) == len(config["cases"]),
                "cases": ordered,
            },
            indent=2,
        )
        + "\n"
    )


def capture_cases(config: dict[str, Any], output: Path, selected: list[str]) -> None:
    status: list[dict[str, Any]] = []
    for case_id in selected:
        case = config["cases"][case_id]
        _project, _app, binary = paths(config, output, case)
        if not binary.is_file():
            raise FileNotFoundError(f"build first: {binary}")
        capture_dir = output / "captures" / case_id.replace("/", "__")
        capture_dir.mkdir(parents=True, exist_ok=True)
        log = output / "logs" / f"capture_{case_id.replace('/', '__')}.log"
        # Keep the real graphics device active. Camera.Render into a
        # RenderTexture is window-independent, but -nographics disables Metal
        # itself and produces valid-sized, flat-gray PNGs.
        command = [str(binary), "-batchmode", "-outdir", str(capture_dir),
                   "-logFile", str(log), *[str(value) for value in case.get("runtime_args", [])]]
        subprocess.run(command, check=True)
        args = [str(value) for value in case.get("runtime_args", [])]
        fps = 30.0 if case["capture_kind"] == "racing" else 60.0
        if "-wmeval-fps" in args:
            fps = float(args[args.index("-wmeval-fps") + 1])
        artifacts = export_artifacts(capture_dir, str(case["capture_kind"]), fps)
        status.append({"case_id": case_id, "status": "ready", "command": command, "artifacts": artifacts})
    write_capture_status(output, config, status)


def export_existing(config: dict[str, Any], output: Path, selected: list[str]) -> None:
    status: list[dict[str, Any]] = []
    for case_id in selected:
        case = config["cases"][case_id]
        capture_dir = output / "captures" / case_id.replace("/", "__")
        args = [str(value) for value in case.get("runtime_args", [])]
        fps = 30.0 if case["capture_kind"] == "racing" else 60.0
        if "-wmeval-fps" in args:
            fps = float(args[args.index("-wmeval-fps") + 1])
        artifacts = export_artifacts(capture_dir, str(case["capture_kind"]), fps)
        status.append({"case_id": case_id, "status": "ready", "capture_fps": fps,
                       "artifacts": artifacts})
    write_capture_status(output, config, status)


def write_scripts(config: dict[str, Any], output: Path, selected: list[str]) -> None:
    scripts = output / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for case_id in selected:
        name = case_id.replace("/", "__")
        command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "capture",
                   "--case", case_id, "--output-dir", str(output)]
        path = scripts / f"run_{name}.sh"
        path.write_text("#!/bin/sh\nset -eu\n" + shlex.join(command) + "\n", encoding="utf-8")
        path.chmod(0o755)
    all_script = scripts / "run_all.sh"
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "all",
               "--output-dir", str(output)]
    all_script.write_text("#!/bin/sh\nset -eu\n" + shlex.join(command) + "\n", encoding="utf-8")
    all_script.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "capture", "export", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    selected = args.case or list(config["cases"])
    unknown = sorted(set(selected) - set(config["cases"]))
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    output = args.output_dir.resolve()
    write_scripts(config, output, selected)
    if args.command in {"build", "all"}:
        build_projects(config, output, selected)
    if args.command in {"capture", "all"}:
        capture_cases(config, output, selected)
    elif args.command == "export":
        export_existing(config, output, selected)
    print(json.dumps({"status": "ready", "command": args.command, "cases": selected,
                      "scripts": str(output / "scripts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
