#!/usr/bin/env python3
"""Copy compiled reconstruction trajectories into their Lite case bundles.

For every case in ``experiments/trajectory_reconstructions/manifest.yaml`` whose
trajectory declares an action target in ``action_backends.yaml``, this writes the
same standardized action files the Table 4/7 bundles carry:

* ``reference/artifacts/trajectory.yaml`` — the authoritative semantic trajectory;
* ``reference/artifacts/action.npz`` — compiled timestamps, values, channel names;
* ``reference/artifacts/actions.jsonl`` — the Project Genie browser program;
* ``reference/artifacts/table_cell.json`` — provenance, the paper table cell, the
  action target and per-backend support (``null`` = unsupported, reported N/A).

The legacy ``action.npy`` / ``actions.txt`` stay untouched: evaluators read them
and their checksums are part of the locked contract. Run
``tools/prepare_other_trajectories.py`` first, then ``tools/build_lite_case.py``
on each touched case to re-lock it.

Usage::

    python tools/bundle_reconstructed_trajectories.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.trajectories import load_trajectory  # noqa: E402

RECON_ROOT = ROOT / "experiments/trajectory_reconstructions"
CASES_ROOT = ROOT / "eval_sets/wmeval_lite_v1/cases"

# Paper table each case reports into; trajectory shapes use the paper's names.
PAPER_CELLS = {
    "BeamDomino/causal": ("physical_causality", "tab:causality_per_traj/BeamDomino/canonical"),
    "BeamDomino/causal_glancing": ("physical_causality", "tab:causality_per_traj/BeamDomino/glancing"),
    "BeamDomino/causal_direct_strike": ("physical_causality", "tab:causality_per_traj/BeamDomino/direct_strike"),
    "LiquidContainer/fluid_rest": ("fluid_mechanics", "tab:fluid-headline/LiquidContainer/rest"),
    "LiquidContainer/fluid_impulse_decay": ("fluid_mechanics", "tab:fluid-headline/LiquidContainer/impulse_decay"),
    "LiquidContainer/fluid_chirp_strafe": ("fluid_mechanics", "tab:fluid-headline/LiquidContainer/chirp_strafe"),
    "LiquidContainer/fluid_step_hold": ("fluid_mechanics", "tab:fluid-headline/LiquidContainer/step_hold"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(case_id: str, trajectory_path: Path) -> Path:
    trajectory = load_trajectory(trajectory_path)
    target = action_backends.action_target(trajectory)
    dimension, cell = PAPER_CELLS[case_id]
    compiled = RECON_ROOT / "compiled"
    reference = CASES_ROOT / case_id / "reference"
    artifacts = reference / "artifacts"
    shutil.copyfile(trajectory_path, artifacts / "trajectory.yaml")
    shutil.copyfile(compiled / f"{trajectory_path.stem}.action.npz", artifacts / "action.npz")
    shutil.copyfile(compiled / f"{trajectory_path.stem}.actions.jsonl", artifacts / "actions.jsonl")
    sources = {
        path.name: _sha256(path)
        for path in sorted([*artifacts.iterdir(), *reference.glob("*.mp4"), *reference.glob("*.png")])
        if path.is_file() and path.name != "table_cell.json"
    }
    support = {
        name: action_backends.support_status(name, target)
        for name in action_backends.registry()["backends"]
    }
    (artifacts / "table_cell.json").write_text(json.dumps({
        "action_target": target,
        "backend_support": support,
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "dimension": dimension,
        "lite_case_id": case_id,
        "paper_alignment": trajectory["paper_alignment"]["status"],
        "schema_version": 1,
        "sources": sources,
        "table_cell": cell,
        "trajectory_id": trajectory["trajectory_id"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CASES_ROOT / case_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    manifest = yaml.safe_load((RECON_ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        if case_id not in PAPER_CELLS:
            continue
        print(bundle(case_id, RECON_ROOT / case["trajectory"]).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
