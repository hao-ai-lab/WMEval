#!/usr/bin/env python3
"""Bake a canonical action trajectory into a Lite case bundle.

Copies ``trajectory.yaml`` (byte-identical to the hash-locked source in
``experiments/tables_4_7/trajectories``), its compiled ``action.npz`` and
``actions.jsonl``, and writes ``actions.json`` declaring the action target
(from ``experiments/tables_4_7/action_backends.yaml``) with source hashes. The
evaluation runner reads ``actions.json`` to report ``unsupported`` for backends
that cannot express the target, instead of scoring a substituted control.

Existing ``actions.txt`` / ``action.npy`` are left untouched. Re-lock the case
afterwards::

    python tools/bake_case_action.py BallRoller/ctrl ball_roller_collision_path
    python tools/build_lite_case.py lite/cases/BallRoller/ctrl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.trajectories import load_trajectory, write_compiled_trajectory  # noqa: E402

CASES_ROOT = ROOT / "lite/cases"
TRAJECTORIES = ROOT / "experiments/tables_4_7/trajectories"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bake(case_id: str, trajectory_stem: str) -> Path:
    source = TRAJECTORIES / f"{trajectory_stem}.yaml"
    trajectory = load_trajectory(source)
    registry = action_backends.registry()["action_targets"][trajectory["trajectory_id"]]
    artifacts = CASES_ROOT / case_id / "reference" / "artifacts"
    if not artifacts.is_dir():
        raise SystemExit(f"{artifacts} does not exist; author the case first")

    with tempfile.TemporaryDirectory() as tmp:
        write_compiled_trajectory(source, tmp)
        shutil.copyfile(source, artifacts / "trajectory.yaml")
        shutil.copyfile(Path(tmp) / f"{trajectory_stem}.action.npz", artifacts / "action.npz")
        shutil.copyfile(Path(tmp) / f"{trajectory_stem}.actions.jsonl", artifacts / "actions.jsonl")

    sources = {
        name: _sha256(artifacts / name)
        for name in ("trajectory.yaml", "action.npz", "actions.jsonl")
    }
    declaration = {
        "schema_version": 1,
        "lite_case_id": case_id,
        "trajectory_id": trajectory["trajectory_id"],
        "target": registry["target"],
        "frame": registry["frame"],
        "channels": list(trajectory["channels"]),
        "control_hz": trajectory["control_hz"],
        "duration_sec": trajectory["duration_sec"],
        "sources": sources,
    }
    output = artifacts / "actions.json"
    output.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_id", help="e.g. BallRoller/ctrl")
    parser.add_argument("trajectory", help="stem under experiments/tables_4_7/trajectories")
    args = parser.parse_args(argv)
    print(f"wrote {bake(args.case_id, args.trajectory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
