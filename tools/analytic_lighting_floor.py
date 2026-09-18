#!/usr/bin/env python3
"""Analytic geometry floor for the ``lighting_shading`` dimension.

Generates a synthetic "orbiting camera, fixed sun" scene analytically — no
Unity, no GPU, no renderer, numpy only — and scores it with the same
circular statistics ``shadow_consistency`` uses, to answer: what do MOSD and
SDJ read on a scene whose lighting is *physically perfect*?

The answer is not zero. Parallel world-space shadow rays converge under
perspective projection, so different objects' cast shadows do not share an
image-space azimuth even under a single exact fixed sun, and an orbiting
camera makes that disagreement move. The resulting MOSD and SDJ are an
irreducible geometric floor: the part of a world model's score that is
neither a lighting error nor a CV-pipeline error, but the metric's own
baseline. Read the ``SunRisesAndSets/lighting`` numbers against it.

The scene mirrors that case's seed frame at a coarse level (round table,
four stools, a few tabletop objects on a floor at y=0). Objects are points
at their anchor heights and shadows are ray-cast along the sun direction
onto the floor plane, so only MOSD/SDJ are defined here; SAS and SSD need
real shadow masks.

Usage::

    # the setting the SunRisesAndSets/lighting bundle documents
    # (elevation 25 deg, azimuth 120 deg) -> MOSD 11.15, SDJ 1.49
    python tools/analytic_lighting_floor.py

    # also write the per-frame log (this reproduces, byte for byte, the
    # analytic_orbit_log.json that used to ship inside the case bundle)
    python tools/analytic_lighting_floor.py --dump-log gt_orbit_log.json

    # how the floor moves with sun position
    python tools/analytic_lighting_floor.py \
        --sweep-elevations 15,25,35 --sweep-azimuths 0,60,120,180,240,300

The paper quotes the floor as a range (MOSD 8.6-11.2 deg, SDJ 1.4-2.0
deg/frame) measured over 21 sun azimuth/elevation settings; the exact grid
was not recorded in the generation harness, so pass your own with
``--sweep-*`` rather than assuming the flags below reproduce it.

Ported from the paper's generation harness on the ``WMEval-lighting-repro``
Modal volume (``harness/synthesize_orbit_gt.py`` +
``harness/step5_analytic_floor.py``). The geometry and the metric code are
carried over unchanged; only the matplotlib preview-video rendering was
dropped, which keeps this tool numpy-only and does not affect any number.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Scene: table at world origin (top at y=0.75), four stools around it, plus a
# few tabletop objects approximated as points at their bases.
SCENE_OBJECTS: dict[str, list[float]] = {
    "Table": [0.0, 0.75, 0.0],
    "Stool_N": [0.0, 0.45, +1.5],
    "Stool_S": [0.0, 0.45, -1.5],
    "Stool_E": [+1.5, 0.45, 0.0],
    "Stool_W": [-1.5, 0.45, 0.0],
    "Mug": [0.20, 0.78, 0.10],
    "Toaster": [-0.30, 0.80, 0.05],
    "Lamp": [0.0, 1.20, 0.0],
}

DEFAULT_SUN_AZIMUTH_DEG = 120.0
DEFAULT_SUN_ELEVATION_DEG = 25.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _project(
    p_world: np.ndarray,
    cam_pos: np.ndarray,
    R_cam: np.ndarray,
    focal_px: float,
    w: int,
    h: int,
) -> tuple[int, int] | None:
    """World point -> pixel coords (OpenCV convention, +Y down)."""
    p_cam = R_cam.T @ (p_world - cam_pos)
    x, y, z = p_cam
    if z <= 0.05:
        return None
    u = w / 2 + (x / z) * focal_px
    v = h / 2 - (y / z) * focal_px
    return int(round(u)), int(round(v))


def _shadow_anchor(
    p_obj_world: np.ndarray, sun_dir_world: np.ndarray, floor_y: float = 0.0
) -> np.ndarray | None:
    """Cast a ray from p_obj along sun_dir until it hits y=floor_y."""
    sy = sun_dir_world[1]
    if abs(sy) < 1e-6:
        return None  # sun on the horizon, infinite shadow
    t = (floor_y - p_obj_world[1]) / sy
    if t <= 0:
        return None  # sun below horizon for this geometry
    return p_obj_world + t * sun_dir_world


def build_orbit_log(
    *,
    num_frames: int = 120,
    width: int = 832,
    height: int = 480,
    fov_deg: float = 60.0,
    orbit_radius: float = 4.0,
    orbit_height: float = 2.5,
    orbit_degrees: float = 360.0,
    sun_azimuth_deg: float = DEFAULT_SUN_AZIMUTH_DEG,
    sun_elevation_deg: float = DEFAULT_SUN_ELEVATION_DEG,
) -> list[dict]:
    """Per-frame camera/light/object log for one fixed-sun camera orbit."""
    fov = np.radians(fov_deg)
    focal_px = (height / 2) / np.tan(fov / 2)

    # Sun: world-frame direction (toward sun, not direction of light flow).
    sa = np.radians(sun_azimuth_deg)
    se = np.radians(sun_elevation_deg)
    # +X = east, +Y = up, +Z = north. Light direction (flow) is opposite.
    toward_sun = np.array(
        [np.cos(se) * np.cos(sa), np.sin(se), np.cos(se) * np.sin(sa)]
    )
    light_dir = -toward_sun  # the direction light travels

    target = np.array([0.0, 0.6, 0.0])  # camera always looks at table centre

    log: list[dict] = []
    for t in range(num_frames):
        # Camera orbit: yaw swept linearly across orbit_degrees.
        yaw = np.radians(orbit_degrees * (t / max(1, num_frames - 1)))
        cam_pos = np.array(
            [
                orbit_radius * np.cos(yaw),
                orbit_height,
                orbit_radius * np.sin(yaw),
            ]
        )

        # Build a look-at rotation: camera local +Z toward target, +Y world up.
        forward = target - cam_pos
        forward /= np.linalg.norm(forward) + 1e-9
        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_world, forward)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(forward, right)
        # Columns of R_cam are camera basis in world coords.
        R_cam = np.stack([right, up, forward], axis=1)

        # Unity-style euler (Z-X-Y) for log compatibility.
        ry = np.degrees(np.arctan2(R_cam[0, 2], R_cam[2, 2]))
        rx = np.degrees(np.arcsin(-R_cam[1, 2]))
        rz = np.degrees(np.arctan2(R_cam[1, 0], R_cam[1, 1]))

        # Per-object 3D positions + their predicted shadow ground points.
        game_objects: list[dict] = []
        for name, position in SCENE_OBJECTS.items():
            p = np.array(position)
            shadow_p = _shadow_anchor(p, light_dir, floor_y=0.0)
            obj_uv = _project(p, cam_pos, R_cam, focal_px, width, height)
            shadow_uv = (
                None
                if shadow_p is None
                else _project(shadow_p, cam_pos, R_cam, focal_px, width, height)
            )
            game_objects.append(
                {
                    "name": name,
                    "position": {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                    },
                    "predicted_shadow_world": (
                        None
                        if shadow_p is None
                        else {
                            "x": float(shadow_p[0]),
                            "y": float(shadow_p[1]),
                            "z": float(shadow_p[2]),
                        }
                    ),
                    "object_uv": list(obj_uv) if obj_uv else None,
                    "shadow_uv": list(shadow_uv) if shadow_uv else None,
                }
            )

        game_objects.append(
            {
                "name": "Main Camera",
                "position": {
                    "x": float(cam_pos[0]),
                    "y": float(cam_pos[1]),
                    "z": float(cam_pos[2]),
                },
                "rotation": {"x": float(rx), "y": float(ry), "z": float(rz)},
            }
        )

        log.append(
            {
                "frame": t,
                "lightEulerAngles": {
                    "x": float(sun_elevation_deg),
                    "y": float(sun_azimuth_deg),
                    "z": 0.0,
                },
                "lightDirection": {
                    "x": float(light_dir[0]),
                    "y": float(light_dir[1]),
                    "z": float(light_dir[2]),
                },
                "intrinsics": {
                    "fx": float(focal_px),
                    "fy": float(focal_px),
                    "cx": float(width / 2),
                    "cy": float(height / 2),
                    "width": int(width),
                    "height": int(height),
                },
                "gameObjects": game_objects,
            }
        )
    return log


# ---------------------------------------------------------------------------
# Circular statistics — identical definitions to evaluation/modules/
# shadow_consistency.py, so the floor is directly comparable to the table.
# ---------------------------------------------------------------------------
def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    rads = np.radians(angles_deg)
    return float(np.degrees(math.atan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))))


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    rads = np.radians(angles_deg)
    R = math.sqrt(float(np.mean(np.sin(rads))) ** 2 + float(np.mean(np.cos(rads))) ** 2)
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(math.sqrt(-2.0 * math.log(R))))


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def analytic_floor(log: list[dict], anchor_name: str = "Table") -> dict:
    """MOSD/SDJ a physically perfect fixed sun produces on this orbit."""
    n_total = len(log)
    per_frame_az_mean: list[float] = []
    per_frame_spread: list[float] = []
    n_retained = 0
    n_with_shadow = 0

    for entry in log:
        gobjs = [g for g in entry["gameObjects"] if g["name"] != "Main Camera"]
        # scene retained iff the anchor object projects in front of the camera
        anchor = next((g for g in gobjs if g["name"] == anchor_name), None)
        if anchor and anchor.get("object_uv"):
            n_retained += 1
            retained = True
        else:
            retained = False

        az: list[float] = []
        for g in gobjs:
            ouv, suv = g.get("object_uv"), g.get("shadow_uv")
            if ouv is None or suv is None:
                continue
            dx = float(suv[0] - ouv[0])
            dy = float(suv[1] - ouv[1])
            if dx * dx + dy * dy < 4.0:
                continue
            az.append(math.degrees(math.atan2(dy, dx)))

        if retained and az:
            n_with_shadow += 1

        if not az:
            per_frame_az_mean.append(float("nan"))
            per_frame_spread.append(float("nan"))
        elif len(az) == 1:
            per_frame_az_mean.append(az[0])
            per_frame_spread.append(float("nan"))
        else:
            arr = np.asarray(az)
            per_frame_az_mean.append(_circular_mean_deg(arr))
            per_frame_spread.append(_circular_std_deg(arr))

    spread = np.asarray(per_frame_spread)
    valid = spread[~np.isnan(spread)]
    mosd = float(np.mean(valid)) if valid.size else None

    az_mean = np.asarray(per_frame_az_mean)
    steps = [
        abs(_angle_diff_deg(b, a))
        for a, b in zip(az_mean[:-1], az_mean[1:])
        if not (math.isnan(a) or math.isnan(b))
    ]
    sdj = float(np.median(steps)) if steps else None

    return {
        "module": "analytic_floor",
        "n_frames_total": n_total,
        "sun_elevation_deg": log[0]["lightEulerAngles"]["x"] if log else None,
        "sun_azimuth_deg": log[0]["lightEulerAngles"]["y"] if log else None,
        "metrics": {
            "scene_retention_rate_pct": round(
                100.0 * n_retained / max(1, n_total), 2
            ),
            "shadow_detection_rate_pct": (
                round(100.0 * n_with_shadow / max(1, n_retained), 2)
                if n_retained
                else None
            ),
            "multi_object_sun_disagreement_deg": (
                round(mosd, 2) if mosd is not None else None
            ),
            "shadow_direction_jitter_deg_per_frame": (
                round(sdj, 2) if sdj is not None else None
            ),
            "shadow_area_stability_pct": None,  # undefined without masks
            "shadow_shape_drift": None,  # undefined without masks
        },
        "note": (
            "SAS/SSD require shadow masks and are undefined on a point-based "
            "analytic log. MOSD/SDJ here are the pure-perspective geometric "
            "floor for a physically perfect single fixed sun."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _floats(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--orbit-radius", type=float, default=4.0)
    parser.add_argument("--orbit-height", type=float, default=2.5)
    parser.add_argument(
        "--orbit-degrees",
        type=float,
        default=360.0,
        help="Total yaw swept by the camera over the clip.",
    )
    parser.add_argument(
        "--sun-azimuth-deg",
        type=float,
        default=DEFAULT_SUN_AZIMUTH_DEG,
        help="World-frame sun azimuth (CCW from +X).",
    )
    parser.add_argument(
        "--sun-elevation-deg",
        type=float,
        default=DEFAULT_SUN_ELEVATION_DEG,
        help="World-frame sun elevation above horizon.",
    )
    parser.add_argument(
        "--anchor-name",
        default="Table",
        help="Object whose visibility defines scene retention.",
    )
    parser.add_argument(
        "--dump-log",
        default=None,
        help="Also write the per-frame log JSON to this path.",
    )
    parser.add_argument(
        "--out", default=None, help="Write the floor metrics JSON to this path."
    )
    parser.add_argument(
        "--sweep-azimuths",
        default=None,
        help="Comma-separated sun azimuths to sweep instead of a single run.",
    )
    parser.add_argument(
        "--sweep-elevations",
        default=None,
        help="Comma-separated sun elevations to sweep instead of a single run.",
    )
    args = parser.parse_args(argv)

    common = {
        "num_frames": args.num_frames,
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov_deg,
        "orbit_radius": args.orbit_radius,
        "orbit_height": args.orbit_height,
        "orbit_degrees": args.orbit_degrees,
    }

    azimuths = _floats(args.sweep_azimuths)
    elevations = _floats(args.sweep_elevations)
    if azimuths or elevations:
        azimuths = azimuths or [args.sun_azimuth_deg]
        elevations = elevations or [args.sun_elevation_deg]
        results = []
        print(f"{'elev':>6s} {'azim':>6s} {'MOSD':>8s} {'SDJ':>8s}")
        for elevation in elevations:
            for azimuth in azimuths:
                log = build_orbit_log(
                    sun_azimuth_deg=azimuth, sun_elevation_deg=elevation, **common
                )
                floor = analytic_floor(log, anchor_name=args.anchor_name)
                metrics = floor["metrics"]
                results.append(floor)
                print(
                    f"{elevation:6.1f} {azimuth:6.1f} "
                    f"{metrics['multi_object_sun_disagreement_deg']:8.2f} "
                    f"{metrics['shadow_direction_jitter_deg_per_frame']:8.2f}"
                )
        mosds = [
            r["metrics"]["multi_object_sun_disagreement_deg"]
            for r in results
            if r["metrics"]["multi_object_sun_disagreement_deg"] is not None
        ]
        sdjs = [
            r["metrics"]["shadow_direction_jitter_deg_per_frame"]
            for r in results
            if r["metrics"]["shadow_direction_jitter_deg_per_frame"] is not None
        ]
        print(
            f"\n{len(results)} setting(s): "
            f"MOSD {min(mosds):.2f}-{max(mosds):.2f} deg, "
            f"SDJ {min(sdjs):.2f}-{max(sdjs):.2f} deg/frame"
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

    log = build_orbit_log(
        sun_azimuth_deg=args.sun_azimuth_deg,
        sun_elevation_deg=args.sun_elevation_deg,
        **common,
    )
    if args.dump_log:
        dump_path = Path(args.dump_log)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
        print(f"wrote {dump_path}  ({len(log)} frames)")

    floor = analytic_floor(log, anchor_name=args.anchor_name)
    print(
        f"\n=== analytic geometric floor "
        f"(elevation {args.sun_elevation_deg} deg, "
        f"azimuth {args.sun_azimuth_deg} deg) ==="
    )
    for key, value in floor["metrics"].items():
        print(f"  {key:48s} {value}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(floor, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
