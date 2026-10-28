#!/usr/bin/env python3
"""Bake the active Table 4/7 cells into self-contained WMEval Lite cases.

Each active cell in ``experiments/protocol/cases.yaml`` becomes one Lite case
bundle, so Table 4 (kinematics & mechanics) and Table 7 (object & character
consistency) are evaluated by the same ``tools/run_wmeval_lite.py`` path as
every other dimension:

* ``reference/ref.png`` — the cell's exact conditioning frame;
* ``reference/artifacts/`` — the canonical semantic trajectory and its compiled
  forms (``trajectory.yaml``, ``action.npz``/``action.npy``, ``actions.txt``,
  ``actions.jsonl``), plus Unity poses, intrinsics, object transforms and, for
  Table 4, masses and plane calibration;
* ``reference/reference.mp4`` — the Unity reference on the evaluation timebase.

Camera cells whose Unity capture is a periodic orbit (WanderingGarden,
MirrorRollingBall) are re-indexed onto the *commanded* trajectory: a full turn
walks the orbit uniformly, a perturbation ramps over the trajectory's yaw
segment and then holds. A backend whose accepted action timing differs (e.g. a
browser model) may pass per-run ground-truth overrides in its candidate
manifest ``params``; the baked case is the nominal protocol.

Usage::

    python tools/dataset/build_table_lite_cases.py            # build all 14 cells
    python tools/dataset/build_table_lite_cases.py --cell table7/WanderingGarden/360_in
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.trajectories import integrated_total, load_trajectory  # noqa: E402
from tools.generation.prepare_matrix_game_inputs import prompt_for  # noqa: E402
from tools.paper_tables.prepare_project_genie_case_eval import (  # noqa: E402
    _load_camera_ground_truth,
    _write_mapped_ground_truth,
    pose_yaw_source_indices,
)
from tools.paper_tables.prepare_project_genie_table7_eval import (  # noqa: E402
    GARDEN_DEPTH_MODEL,
    PAPER_ANCHOR_DETECTION_PROMPT,
    PAPER_ANCHORS,
    _resample_reference,
    full_turn_source_indices,
    perturbation_source_indices,
)
from tools.paper_tables.run_prepared_table_evaluations import (  # noqa: E402
    PLANS,
    _billiard_kinematics_inputs,
    _billiard_unity_inputs,
    _racing_unity_params,
)

TABLE_ROOT = ROOT / "experiments/protocol"
CASES_ROOT = ROOT / "lite/cases"
GARDEN_RECEIPT = ROOT / "runs/unity/garden_bridge_gt/capture_receipt.json"
NO_CHARACTER = "no_character_configured_for_this_case_xyz."
PROTOCOL = {
    "candidate_normalization": "match_reference",
    "min_duration_sec": 1.0,
    "min_duration_ratio": 0.5,
    "duration_threshold_mode": "max",
}


def lite_case_id(cell_id: str) -> str:
    """``table4/RacingCar/third`` → ``RacingCar/kinematics_third``;
    ``table7/WanderingGarden/360_in`` → ``WanderingGarden/scene_360_in``."""
    table, scene, variant = cell_id.split("/")
    tag = "kinematics" if table == "table4" else "scene"
    return f"{scene}/{tag}_{variant}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination.exists() and destination.resolve() == source.resolve()):  # the bundle may be the source
        shutil.copyfile(source, destination)
    return _sha256(source)


def _frames_to_video(frames: list[Path], destination: Path, fps: float) -> None:
    """Encode an explicit frame sequence (repeats allowed) as the reference video."""
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "frames.txt"
        listing.write_text(
            "".join(f"file '{frame.resolve()}'\nduration {1.0 / fps:.9f}\n" for frame in frames)
            + f"file '{frames[-1].resolve()}'\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", str(listing), "-vf", f"fps={fps:.12g}", "-frames:v", str(len(frames)),
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(destination),
            ],
            check=True,
        )


def _video_frames(video: Path, workdir: Path) -> list[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-fps_mode", "passthrough", str(workdir / "f_%05d.png")],
        check=True,
    )
    return sorted(workdir.glob("f_*.png"))


def _probe(video: Path) -> dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "default=noprint_wrappers=1",
         str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    num, _, den = fields["avg_frame_rate"].partition("/")
    return {"width": int(fields["width"]), "height": int(fields["height"]),
            "fps": float(num) / float(den or 1), "frames": int(fields["nb_read_frames"])}


def _commanded_yaw_timing(trajectory: dict[str, Any]) -> tuple[float, float, float]:
    """(action start, action end, timeline end) of the trajectory's yaw motion."""
    moving = [s for s in trajectory["segments"] if abs(float(s["values"].get("yaw_rate_deg_s", 0.0))) > 0]
    return (
        min(float(s["start_sec"]) for s in moving),
        max(float(s["end_sec"]) for s in moving),
        float(trajectory["duration_sec"]),
    )


def _copy_actions(cell: dict[str, Any], artifacts: Path) -> dict[str, str]:
    trajectory_path = TABLE_ROOT / cell["trajectory"]
    stem = trajectory_path.stem
    compiled = TABLE_ROOT / "compiled"
    hashes = {"trajectory.yaml": _copy(trajectory_path, artifacts / "trajectory.yaml")}
    for suffix, name in ((".action.npz", "action.npz"), (".action.npy", "action.npy"),
                         (".actions.txt", "actions.txt"), (".actions.jsonl", "actions.jsonl")):
        hashes[name] = _copy(compiled / f"{stem}{suffix}", artifacts / name)
    return hashes


def _relative(params: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Point copied-artifact params at their bundle-relative paths."""
    return {key: mapping.get(key, value) for key, value in params.items()}


def _build_capture_cell(plan, cell, artifacts: Path, reference: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Billiard (Tables 4/7) and RacingCar (Table 4): Unity captured the exact motion."""
    if plan.preparation == "racing_unity_capture":
        params = _racing_unity_params(plan)
        video = (ROOT / "runs/tables_4_7/unity_scene_runs/captures/table4__RacingCar__third/unity_groundtruth.mp4")
    elif plan.preparation == "billiard_kinematics_capture":
        params, video = _billiard_kinematics_inputs(plan)
    else:
        template = yaml.safe_load((CASES_ROOT / plan.lite_case_id / "case.yaml").read_text(encoding="utf-8"))
        candidate, video = _billiard_unity_inputs(plan)
        params = {**template["module_params"]["environment_object_consistency"], **candidate}
    hashes = {"reference.mp4": _copy(Path(video), reference)}
    relative: dict[str, str] = {}
    for key, value in list(params.items()):
        if isinstance(value, str) and (value.endswith(".npy") or value.endswith(".json")) and Path(value).is_absolute():
            source = Path(value)
            name = source.name
            hashes[name] = _copy(source, artifacts / name)
            relative[key] = f"reference/artifacts/{name}"
    return _relative(params, relative), hashes


def _build_orbit_cell(plan, cell, artifacts: Path, reference: Path, workdir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """WanderingGarden / MirrorRollingBall: re-index a periodic Unity orbit onto the commanded yaw."""
    trajectory = load_trajectory(TABLE_ROOT / cell["trajectory"])
    target_yaw = integrated_total(trajectory, "yaw_rate_deg_s")
    if abs(target_yaw - plan.target_yaw_deg) > 1e-6:
        raise ValueError(f"{plan.case_id}: trajectory yaw {target_yaw} != plan {plan.target_yaw_deg}")
    start, end, timeline_end = _commanded_yaw_timing(trajectory)
    if plan.preparation == "garden":
        receipt = json.loads(GARDEN_RECEIPT.read_text(encoding="utf-8"))
        art = receipt["evaluation_artifacts"]

        def path(name: str) -> Path:
            return (GARDEN_RECEIPT.parent / art[name]["path"]).resolve()

        poses = np.load(path("poses_npy"), allow_pickle=False)
        intrinsics = np.load(path("intrinsics_npy"), allow_pickle=False)
        objects = json.loads(path("object_transforms_json").read_text(encoding="utf-8"))["objects"]
        width, height = (int(v) for v in receipt["resolution"])
        frame_count, fps = int(receipt["frame_count"]), float(receipt.get("capture_fps", 15.0))
        if plan.trajectory_mode == "full_turn":
            indices = full_turn_source_indices(
                source_count=poses.shape[0], output_count=frame_count,
                start_frame=plan.reference_start_frame, target_yaw_deg=target_yaw)
        else:
            indices, _ = perturbation_source_indices(
                source_count=poses.shape[0], output_count=frame_count,
                start_frame=plan.reference_start_frame, target_yaw_deg=target_yaw,
                action_trace=[{"target_yaw_deg": target_yaw, "accepted_at_sec": start, "completed_at_sec": end},
                              {"completed_at_sec": timeline_end}])
        sampled_poses, sampled_intrinsics, indices = _resample_reference(
            poses=poses, intrinsics=intrinsics, frame_count=frame_count,
            candidate_width=width, candidate_height=height,
            reference_width=width, reference_height=height, source_indices=indices)
        np.save(artifacts / "poses.npy", sampled_poses, allow_pickle=False)
        np.save(artifacts / "intrinsics.npy", sampled_intrinsics, allow_pickle=False)
        selected = {
            label: {"frames": list(range(frame_count)),
                    "positions_world": [objects[label]["positions_world"][min(i, len(objects[label]["positions_world"]) - 1)] for i in indices]}
            for label in PAPER_ANCHORS
        }
        (artifacts / "object_transforms.json").write_text(json.dumps(
            {"schema_version": 1, "source": "runs/unity/garden_bridge_gt (GardenBridge capture)",
             "timeline_mapping": "nearest_periodic_reference_pose_for_commanded_yaw",
             "trajectory_mapping": {"mode": plan.trajectory_mode, "start_frame": plan.reference_start_frame,
                                    "target_yaw_deg": target_yaw, "action_start_sec": start,
                                    "action_end_sec": end, "timeline_end_sec": timeline_end},
             "objects": selected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        frames = sorted((ROOT / "runs/unity/garden_bridge_gt/frames").glob("*.png"))
        params = {
            "eoc_text_prompt": PAPER_ANCHOR_DETECTION_PROMPT,
            "eoc_depth_model": GARDEN_DEPTH_MODEL,
            "eoc_fixed_anchor_labels": list(PAPER_ANCHORS),
            "eoc_character_prompt": NO_CHARACTER,
            "eoc_max_objects": 5,
        }
    else:  # MirrorRollingBall: the Lite prompt_third orbit is the Unity source
        template_root = CASES_ROOT / plan.lite_case_id
        camera_gt = _load_camera_ground_truth(template_root)
        template_video = next((template_root / "reference").glob("*.mp4"))
        info = _probe(template_video)
        frame_count, fps = int(info["frames"]), float(info["fps"])
        times = np.arange(frame_count, dtype=np.float64) / frame_count * timeline_end
        if plan.trajectory_mode == "full_turn":
            offsets = np.arange(frame_count, dtype=np.float64) / frame_count * target_yaw
        else:
            offsets = np.clip((times - start) / (end - start), 0.0, 1.0) * target_yaw
        indices = pose_yaw_source_indices(camera_gt["poses"], start_frame=plan.reference_start_frame,
                                          yaw_offsets_deg=offsets)
        _write_mapped_ground_truth(destination=artifacts, source=camera_gt, source_indices=indices,
                                   mapping={"mode": plan.trajectory_mode, "target_yaw_deg": target_yaw,
                                            "reference_start_frame": plan.reference_start_frame,
                                            "action_start_sec": start, "action_end_sec": end,
                                            "timeline_end_sec": timeline_end})
        frames = _video_frames(template_video, workdir / "mirror_frames")
        params = {k: v for k, v in camera_gt["params"].items() if not str(k).startswith("eoc_unity_")}
        if "diameters" in camera_gt["paths"]:
            _copy(camera_gt["paths"]["diameters"], artifacts / "object_diameters.json")
            params["eoc_unity_object_diameters_json"] = "reference/artifacts/object_diameters.json"
        character = str(cell.get("character_prompt", "")).strip()
        params["eoc_character_prompt"] = character or NO_CHARACTER
    _frames_to_video([frames[i] for i in indices], reference, fps)
    params.update({
        "eoc_unity_poses_npy": "reference/artifacts/poses.npy",
        "eoc_unity_intrinsics_npy": "reference/artifacts/intrinsics.npy",
        "eoc_unity_object_transforms_json": "reference/artifacts/object_transforms.json",
    })
    return params, {"reference_source_indices": json.dumps(list(map(int, indices)))}


def build_cell(cell: dict[str, Any], manifest: dict[str, Any], workdir: Path) -> str:
    plan = next(p for p in PLANS if p.case_id == cell["case_id"])
    case_id = lite_case_id(cell["case_id"])
    case_dir = CASES_ROOT / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    artifacts = case_dir / "reference" / "artifacts"
    artifacts.mkdir(parents=True)
    reference = case_dir / "reference" / "reference.mp4"
    provenance = {"ref.png": _copy((TABLE_ROOT / cell["first_frame"]).resolve(), case_dir / "reference" / "ref.png")}
    provenance.update(_copy_actions(cell, artifacts))
    if plan.preparation in {"garden", "lite"}:
        params, extra = _build_orbit_cell(plan, cell, artifacts, reference, workdir / case_id.replace("/", "__"))
    else:
        params, extra = _build_capture_cell(plan, cell, artifacts, reference)
    provenance.update(extra)
    dimension = plan.dimension
    trajectory = load_trajectory(TABLE_ROOT / cell["trajectory"])
    (artifacts / "table_cell.json").write_text(json.dumps({
        "schema_version": 1,
        "table_cell": cell["case_id"],
        "lite_case_id": case_id,
        "dimension": dimension,
        "trajectory_id": trajectory["trajectory_id"],
        "preparation": plan.preparation,
        "orientation_frame_status": cell.get("orientation_frame_status"),
        "paper_orientation": cell.get("paper_orientation"),
        "character_prompt": cell.get("character_prompt", ""),
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "sources": provenance,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    view = str(cell.get("view", ""))
    case_doc = {
        "schema_version": 2,
        "case_id": case_id,
        "scene": cell["scene"],
        "display_name": f"{cell['scene']} {cell['case_id'].split('/')[0]} {cell['case_id'].split('/')[-1]}",
        "prompt": prompt_for(cell, manifest),
        "perspective": view or None,
        "modules": [dimension],
        "reference": {
            "first_frame": "reference/ref.png",
            "video": "reference/reference.mp4",
            "checksums": "reference/checksums.sha256",
            "artifacts": {
                "actions_file": "reference/artifacts/actions.txt",
                "synthetic_action_dir": "reference/artifacts",
            },
        },
        "protocol": dict(PROTOCOL),
        "module_params": {dimension: params},
    }
    (case_dir / "case.yaml").write_text(yaml.safe_dump(case_doc, sort_keys=False, width=88), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "tools/dataset/build_lite_case.py"), str(case_dir)], check=True,
                   stdout=subprocess.DEVNULL)
    return case_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", default=[], help="Table cell id (default: all active)")
    args = parser.parse_args()
    manifest = yaml.safe_load((TABLE_ROOT / "cases.yaml").read_text(encoding="utf-8"))
    cells = [c for c in manifest["cases"] if c.get("execution_scope", "active") == "active"]
    if args.cell:
        cells = [c for c in cells if c["case_id"] in set(args.cell)]
    built = []
    with tempfile.TemporaryDirectory() as tmp:
        for cell in cells:
            built.append(build_cell(cell, manifest, Path(tmp)))
            print(f"built {built[-1]}  <- {cell['case_id']}")
    print(json.dumps({"built": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
