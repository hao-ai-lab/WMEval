#!/usr/bin/env python3
"""Prepare an explicit look-at orbit for a lighting diagnostic (not a paper baseline).

World: right handed, +Y up, initial camera on +Z. Positive orbit angles
follow the right-hand rule around +Y (CCW viewed from above). Camera axes
are OpenCV: +X right, +Y down, +Z forward; poses are camera-to-world.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def orbit_poses(*, duration: float = 10.0, fps: float = 16.0,
                radius: float = 2.0, height: float = 1.6,
                target_height: float = 0.75, degrees: float = 360.0,
                vertical_fov: float = 60.0, width: int = 832,
                image_height: int = 480, seed_aspect: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [duration, fps, radius, height, target_height, degrees, vertical_fov, seed_aspect]
    if not np.isfinite(values).all() or min(duration, fps, radius) <= 0:
        raise ValueError("finite parameters and positive duration/fps/radius required")
    if not 0 < vertical_fov < 180 or min(width, image_height, seed_aspect) <= 0:
        raise ValueError("invalid camera intrinsics")
    count = round(duration * fps)
    if not np.isclose(count, duration * fps):
        raise ValueError("duration * fps must be integral")
    # Include both endpoints: 161 poses cover control times 0 through 10 s.
    times = np.arange(count + 1, dtype=np.float64) / fps
    angles = np.deg2rad(degrees) * times / duration
    positions = np.column_stack((radius * np.sin(angles),
                                 np.full(len(times), height), radius * np.cos(angles)))
    target = np.array([0.0, target_height, 0.0])
    forward = target - positions
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(forward, right)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, :3] = np.stack((right, down, forward), axis=-1)
    poses[:, :3, 3] = positions
    normalized_focal = 1 / (2 * np.tan(np.deg2rad(vertical_fov) / 2))
    # LingBot unconditionally treats K as 832x480, then scales x/y separately
    # to the seed aspect ratio. The bundled seed is SQUARE: compensate here
    # so the final square output has equal fx and fy (square pixels).
    intrinsics = np.tile([width * normalized_focal / seed_aspect,
                          image_height * normalized_focal, width / 2, image_height / 2], (len(times), 1))
    return times, poses, intrinsics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path, **kwargs) -> dict:
    kwargs = {**{name: p.default for name, p in inspect.signature(orbit_poses).parameters.items()}, **kwargs}
    times, poses, intrinsics = orbit_poses(**kwargs)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "poses.npy", poses.astype(np.float32), allow_pickle=False)
    np.save(output / "intrinsics.npy", intrinsics.astype(np.float32), allow_pickle=False)
    (output / "camera.jsonl").write_text("".join(
        json.dumps({"time_sec": float(t), "c2w": p.tolist()}) + "\n"
        for t, p in zip(times, poses)), encoding="utf-8")
    receipt = {
        "schema_version": 1, "status": "diagnostic_unvalidated",
        "target": "camera", "trajectory_id": "lighting.camera.look_at_orbit.diagnostic",
        "parameters": kwargs, "pose_convention": "right_handed_y_up_opencv_c2w",
        "intrinsics_convention": "fx_fy_cx_cy_in_832x480_pixels",
        "control_duration_sec": float(times[-1]), "poses": len(times),
        "sun": "fixed (scene/prompt constraint; not controlled by these camera arrays)",
        "assumptions": ["2 m scene scale and 60 degree vertical FOV are nominal, not calibrated from the seed image",
                        "161 frames at 16 fps encode 10.0625 s including the conditioning frame",
                        "This is a new diagnostic protocol, not a reconstruction of historical paper controls"],
        "files": {name: sha256(output / name) for name in ("poses.npy", "intrinsics.npy", "camera.jsonl")},
    }
    (output / "trajectory.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def preview(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    poses = np.load(output / "poses.npy")
    receipt = json.loads((output / "trajectory.json").read_text())
    params = receipt["parameters"]
    times = np.linspace(0, receipt["control_duration_sec"], len(poses))
    pos = poses[:, :3, 3]
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    ax = axes[0]
    ax.add_patch(Circle((0, 0), .65, color="#d5aa73"))
    ax.text(0, 0, "table", ha="center", va="center")
    ax.plot(pos[:, 0], -pos[:, 2], color="#17877d", lw=2)
    for j, i in enumerate(np.linspace(0, len(poses)-1, 5).astype(int)):
        x, z = pos[i, 0], -pos[i, 2]
        ax.scatter(x, z, color=colors[j], s=50, zorder=4)
        ax.annotate("", (x*.55, z*.55), (x,z), arrowprops=dict(arrowstyle="->", color=colors[j], lw=2))
        if j < 4:
            ax.text(x*1.2, z*1.2, f"{times[i]:g} s", ha="center", va="center")
    ax.set(xlim=(-2.8,2.8), ylim=(-2.8,2.8), xlabel="World X (m)", ylabel="World -Z (m)",
           title="Commanded camera path, viewed from above", aspect="equal")
    ax.grid(alpha=.15)
    ax = axes[1]
    radius = np.linalg.norm(pos[:, [0,2]], axis=1)
    to_target = np.array([0,params["target_height"],0]) - pos
    to_target /= np.linalg.norm(to_target,axis=1,keepdims=True)
    forward = poses[:,:3,2].astype(np.float64)
    forward /= np.linalg.norm(forward,axis=1,keepdims=True)
    error = np.rad2deg(np.arccos(np.clip(np.sum(to_target*forward,axis=1),-1,1)))
    ax.plot(times, radius, label="Distance to table axis (m)", lw=2)
    ax.plot(times, pos[:,1], label="Eye height (m)", lw=2)
    ax.plot(times, error, label="Look-at error (deg)", lw=2)
    ax.set(xlabel="Control time (s)", ylim=(-.15,2.5), title="Geometry checks from the actual pose arrays")
    ax.grid(alpha=.15); ax.legend(loc="upper center", fontsize=9)
    fig.suptitle(f"Lighting orbit diagnostic | {params['degrees']:g} degrees / {params['duration']:g} s / radius {params['radius']:g} m", fontsize=14)
    fig.savefig(output / "preview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/lighting_orbit_diagnostics/trajectory.yaml")
    parser.add_argument("--degrees", type=float)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    import yaml
    spec = yaml.safe_load(args.config.read_text())
    if spec.get("diagnostic_schema_version") != 1 or spec.get("kind") != "look_at_orbit" or spec.get("target") != "camera":
        raise ValueError("expected a camera look_at_orbit diagnostic")
    params = dict(spec["parameters"])
    if args.degrees is not None:
        params["degrees"] = args.degrees
    receipt = prepare(args.output, **params)
    receipt["semantic_source_sha256"] = sha256(args.config)
    receipt["compiler_sha256"] = sha256(Path(__file__))
    receipt["source_overrides"] = {} if args.degrees is None else {"degrees": args.degrees}
    (args.output / "trajectory.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if args.preview:
        preview(args.output)


if __name__ == "__main__":
    main()
