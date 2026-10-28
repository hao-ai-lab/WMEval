#!/usr/bin/env python3
"""Re-score stored kinematics rows with the extended (non-headline) metrics.

Reads ``per_case_results.jsonl`` rows of the ``kinematics_mechanics``
dimension, reuses their stored GroundingDINO/ByteTrack trajectories
(``full_report.results.kinematics_mechanics.trajectories``) -- no detector is
re-run -- and computes, against the *current* suite case artifacts:

* the legacy Unity-camera-lifted ``fde_m`` / ``dtw_m_mean`` (recomputed, to
  separate artifact changes from method changes);
* candidate-camera lifting ``fde_m_ego`` / ``dtw_m_ego`` and the robust
  mechanics (``robust_error_rate_pct`` ...), see
  :mod:`evaluation.modules.kinematics_robust`.

``--gt-case`` scores every row's candidate tracks against another case's
Unity ground truth (negative controls: a reference video scored against a
different trajectory of the same family).

Example::

    python tools/rescore_kinematics_extended.py \\
        --results reports/wmeval_lite/unity_selfmatch/per_case_results.jsonl \\
        --output /tmp/rescored.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from evaluation.modules import kinematics_mechanics as km  # noqa: E402

DIMENSION = "kinematics_mechanics"


def _case_kin_params(suite_root: str, case_id: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved ``kinematics_mechanics`` module params of a core or synthetic case."""
    if case_id in cache:
        return cache[case_id]
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = cache.setdefault("__suite__", load_suite(suite_root))
    if case_id.startswith("synthetic/"):
        from evaluation.lite.synthetic import build_case, load_trajectories

        trajectories = cache.setdefault("__trajectories__", {t.synthetic_id: t for t in load_trajectories(suite)})
        if case_id not in trajectories:
            raise KeyError(f"unknown synthetic case {case_id}")
        case = build_case(suite, trajectories[case_id])
    else:
        case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
    params = dict(case.module_params.get(DIMENSION) or {})
    params["__reference_video__"] = str(case.reference_video) if case.reference_video else None
    cache[case_id] = params
    return params


def _load_ground_truth(params: Dict[str, Any]) -> Dict[str, Any]:
    from evaluation.modules.environment_object_consistency import (
        _load_camera_intrinsics,
        _load_camera_poses,
    )

    gt = km._load_unity_object_transforms(str(params["kin_unity_object_transforms_json"]))
    poses = _load_camera_poses(str(params["kin_unity_poses_npy"]))
    intrinsics = _load_camera_intrinsics(str(params["kin_unity_intrinsics_npy"]))
    masses = (
        km._load_object_masses(str(params["kin_unity_object_masses_json"]))
        if params.get("kin_unity_object_masses_json")
        else {}
    )
    calibration = (
        km._load_plane_calibration(str(params["kin_plane_calibration_json"]))
        if params.get("kin_plane_calibration_json")
        else None
    )
    return {"gt": gt, "poses": poses, "intrinsics": intrinsics, "masses": masses, "calibration": calibration}


def _legacy_metric(trajectories, track_point, truth) -> Dict[str, Optional[float]]:
    """Legacy timestamp-matched Unity-camera lifting, recomputed."""
    if truth["calibration"] is None:
        return {"fde_m_mean": None, "dtw_m_mean": None}
    plane_y = float(truth["calibration"]["plane"]["value_m"])
    fde, dtw = [], []
    for label, points in trajectories.items():
        world = km._unity_trajectory(truth["gt"], label)
        if not world:
            continue
        pixels = {
            int(p["frame"]): np.asarray(
                p["bbox_bottom_center_uv"] if track_point == "bbox_bottom_center" else p["centroid_uv"],
                dtype=np.float64,
            )
            for p in points
        }
        lifted = km._map_pixels_to_unity_plane_by_pose(pixels, truth["poses"], truth["intrinsics"], plane_y)
        reference = {int(f): pos[[0, 2]] for f, pos in world}
        frames = sorted(set(lifted) & set(reference))
        if len(frames) < 2:
            continue
        hat = np.stack([lifted[f] for f in frames])
        ref = np.stack([reference[f] for f in frames])
        fde.append(km._fde(hat, ref))
        dtw.append(km._dtw_mean(hat, ref))
    return {
        "fde_m_mean": round(float(np.mean(fde)), 6) if fde else None,
        "dtw_m_mean": round(float(np.mean(dtw)), 6) if dtw else None,
    }


def rescore_row(
    row: Dict[str, Any],
    *,
    suite_root: str,
    cache: Dict[str, Any],
    gt_case: Optional[str] = None,
    case_map: Optional[Dict[str, str]] = None,
    ego_params: Optional[Dict[str, Any]] = None,
    robust_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = ((row.get("full_report") or {}).get("results") or {}).get(DIMENSION) or {}
    trajectories = result.get("trajectories") or {}
    case_id = (case_map or {}).get(row["case_id"], row["case_id"])
    truth_case = gt_case or case_id
    params = _case_kin_params(suite_root, truth_case, cache)
    candidate_params = _case_kin_params(suite_root, case_id, cache)
    truth = _load_ground_truth(params)
    track_point = str(result.get("track_point") or candidate_params.get("kin_track_point") or "centroid")
    video = result.get("video") or row.get("candidate_video")
    if video and not Path(video).exists():
        video = row.get("candidate_video")
    fps = float(result.get("fps") or 0.0)
    required = [s.strip() for s in str(candidate_params.get("kin_text_prompt", "")).split(".") if s.strip()]
    stored = row.get("metrics") or {}
    out: Dict[str, Any] = {
        "case_id": case_id,
        "stored_case_id": row["case_id"],
        "gt_case_id": truth_case,
        "model": row.get("model"),
        "seed": row.get("seed"),
        "candidate_video": video,
        "stored": {k: stored.get(k) for k in ("fde_m_mean", "dtw_m_mean", "error_rate_pct", "exact_match_pct")},
    }
    if not trajectories or fps <= 0:
        out["status"] = "no_stored_trajectories"
        return out
    out["legacy_recomputed"] = _legacy_metric(trajectories, track_point, truth)
    extended = km.extended_kinematics_from_tracks(
        trajectories,
        fps=fps,
        track_point=track_point,
        video_path=video,
        gt=truth["gt"],
        poses=truth["poses"],
        intrinsics=truth["intrinsics"],
        plane_calibration=truth["calibration"],
        masses=truth["masses"],
        required_labels=required,
        alignment_tolerance_frames=int(params.get("kin_event_alignment_tolerance_frames", 15)),
        ego_params=ego_params,
        robust_params=robust_params,
        reference_video=params.get("__reference_video__"),
    )
    out["status"] = extended.get("status")
    out["reference_camera_floor"] = extended.get("reference_camera_floor")
    out["extended_summary"] = extended.get("summary")
    out["camera_motion_estimate"] = extended.get("camera_motion_estimate")
    out["unity_camera"] = extended.get("unity_camera")
    out["labels"] = extended.get("labels")
    return out


def _rows(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("dimension") == DIMENSION:
                    yield row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="per_case_results.jsonl file(s)")
    parser.add_argument("--suite", default=str(REPO_ROOT / "lite"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", default=None, help="only rows of this case id (repeatable)")
    parser.add_argument("--gt-case", default=None, help="score against this case's Unity GT instead")
    parser.add_argument(
        "--case-map", action="append", default=[], metavar="OLD=NEW",
        help="treat stored case id OLD as current case NEW (e.g. RacingCar/prompt_third=RacingCar/kinematics_third)",
    )
    parser.add_argument("--ego-params", default="{}", help="JSON overrides of kinematics_robust.EGO_DEFAULTS")
    parser.add_argument("--robust-params", default="{}", help="JSON overrides of kinematics_robust.ROBUST_DEFAULTS")
    args = parser.parse_args(argv)
    case_map = dict(item.split("=", 1) for item in args.case_map)
    cache: Dict[str, Any] = {}
    count = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in _rows(args.results):
            case_id = case_map.get(row["case_id"], row["case_id"])
            if args.case and case_id not in args.case:
                continue
            try:
                out = rescore_row(
                    row, suite_root=args.suite, cache=cache, gt_case=args.gt_case, case_map=case_map,
                    ego_params=json.loads(args.ego_params), robust_params=json.loads(args.robust_params),
                )
            except Exception as exc:  # report and continue
                out = {"case_id": case_id, "model": row.get("model"), "status": "error", "error": repr(exc)}
            handle.write(json.dumps(out, default=float) + "\n")
            count += 1
            summary = out.get("extended_summary") or {}
            print(
                f"{out.get('case_id')} vs {out.get('gt_case_id')}: legacy fde "
                f"{(out.get('legacy_recomputed') or {}).get('fde_m_mean')} -> ego fde "
                f"{summary.get('fde_m_ego_mean')} robust_err {summary.get('robust_error_rate_pct')} "
                f"[{out.get('status')}]",
                flush=True,
            )
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
