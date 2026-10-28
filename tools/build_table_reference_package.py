#!/usr/bin/env python3
"""Build a compact, verifiable Table 4/7 Unity reference-only package."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.provenance_paths import resolve_recorded_path

TABLE_ROOT = REPO_ROOT / "experiments/protocol"
HANDOFF_MANIFEST = TABLE_ROOT / "gpu_handoff_manifest.json"
AUDIT_MANIFEST = TABLE_ROOT / "reference_package_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "dist/reference-packages"
PACKAGE_ROOT = "wmeval-table4-table7-reference-v1"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)

CALIBRATION_RUNS = {
    "arrow_left": REPO_ROOT
    / "runs/project_genie/calibration/garden_bridge_arrow_left_attempt4",
    "arrow_right": REPO_ROOT
    / "runs/project_genie/calibration/garden_bridge_arrow_right",
}

REFERENCE_CASES = (
    "RacingCar/prompt_third",
    "BilliardGame/prompt_first",
    "WanderingGarden/scene_first",
    "MirrorRollingBall/prompt_third",
)

CONTRACT_PATHS = (
    "experiments/protocol/cases.yaml",
    "experiments/protocol/paper_baseline.yaml",
    "experiments/protocol/evaluator_protocol.yaml",
    "experiments/protocol/prompt_components.yaml",
    "experiments/protocol/conditioning_sources.yaml",
    "experiments/protocol/unity_scene_runs.yaml",
    "experiments/protocol/trajectories",
    "experiments/protocol/compiled",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _add_file(files: dict[str, dict[str, Any]], path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = _repo_relative(path)
    row = files.setdefault(relative, {"source": path, "roles": set()})
    row["roles"].add(role)


def discover_reference_files() -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    handoff = json.loads(HANDOFF_MANIFEST.read_text(encoding="utf-8"))
    for row in handoff["files"]:
        if any(str(role).startswith("unity_capture") for role in row["roles"]):
            path = REPO_ROOT / row["path"]
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"Unity handoff artifact changed: {row['path']}")
            _add_file(files, path, "unity_analytical_reference")

    cases = yaml.safe_load((TABLE_ROOT / "cases.yaml").read_text(encoding="utf-8"))
    for case in cases["cases"]:
        if case.get("execution_scope") == "excluded_by_user":
            continue
        frame = (TABLE_ROOT / str(case["first_frame"])).resolve()
        _add_file(files, frame, f"conditioning_frame:{case['case_id']}")

    case_root = REPO_ROOT / "lite/cases"
    for case_id in REFERENCE_CASES:
        for path in sorted((case_root / case_id).rglob("*")):
            if path.is_file():
                _add_file(files, path, f"lite_reference:{case_id}")

    for relative in CONTRACT_PATHS:
        path = REPO_ROOT / relative
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    _add_file(files, child, "evaluation_contract")
        else:
            _add_file(files, path, "evaluation_contract")
    return files


def calibration_jpegs(*, max_width: int, quality: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, source_dir in CALIBRATION_RUNS.items():
        samples_path = source_dir / "turn_samples.json"
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        for sample in samples:
            source = resolve_recorded_path(
                sample["path"], repo_root=REPO_ROOT, relative_to=source_dir
            )
            if sha256(source) != sample["sha256"]:
                raise ValueError(f"calibration source hash mismatch: {source}")
            with Image.open(source) as image:
                image = image.convert("RGB")
                if image.width > max_width:
                    height = round(image.height * max_width / image.width)
                    image = image.resize(
                        (max_width, height), resample=Image.Resampling.LANCZOS
                    )
                buffer = io.BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
            data = buffer.getvalue()
            rows.append(
                {
                    "archive_path": (
                        f"calibration_samples/{direction}/"
                        f"frame_{int(sample['index']):04d}.jpg"
                    ),
                    "data": data,
                    "direction": direction,
                    "sample_index": int(sample["index"]),
                    "elapsed_sec": float(sample["elapsed_sec"]),
                    "source_sha256": sample["sha256"],
                    "sha256": sha256_bytes(data),
                    "bytes": len(data),
                }
            )
    return rows


def _zip_write(archive: zipfile.ZipFile, relative: str, data: bytes) -> None:
    info = zipfile.ZipInfo(f"{PACKAGE_ROOT}/{relative}", date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def package_readme(manifest: dict[str, Any]) -> str:
    return f"""# WMEval Table 4/7 reference package v1

This package contains reference data only: no world-model generations and no
model checkpoints. It covers {manifest['active_case_count']} active cells.

Contents:

- exact generation conditioning images;
- Unity reference videos, camera poses/intrinsics, object trajectories,
  masses/diameters, and planar calibration anchors;
- semantic action trajectories and compiled browser controls;
- the paper baseline and evaluator protocol;
- {manifest['calibration_sample_count']} calibration JPEGs sampled at roughly
  one-second intervals from the successful Project Genie left/right yaw runs;
- original calibration JSON metadata.

The JPEG calibration frames are visual evidence only. Their source PNG hashes
and elapsed timestamps are in `calibration_samples/index.json`. Exact
conditioning images and numerical reference artifacts remain byte-identical.

Extract the archive and overlay its package root onto a checkout of
`lanxiang/release-cleanup`, then run:

```bash
unzip wmeval-table4-table7-reference-v1.zip -d /tmp/wmeval-reference
rsync -a /tmp/wmeval-reference/wmeval-table4-table7-reference-v1/ /path/to/WMEval/
cd /path/to/WMEval
.venv-gpu/bin/python tools/preflight_tables_4_7_gpu.py --reference-only
```

For a new model: calibrate its backend adapter, generate five independent
runs per active cell, write a candidate manifest, and evaluate using the fixed
Unity traces and `paper_baseline.yaml`. Reconstructed orientation/action
caveats in `cases.yaml` remain part of the claim scope.
"""


def build(output_dir: Path, *, max_width: int, quality: int) -> tuple[Path, dict[str, Any]]:
    source_files = discover_reference_files()
    samples = calibration_jpegs(max_width=max_width, quality=quality)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    source_rows = []
    for relative, item in sorted(source_files.items()):
        path = item["source"]
        source_rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "roles": sorted(item["roles"]),
            }
        )
    sample_index = [
        {key: row[key] for key in row if key != "data"} for row in samples
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package": PACKAGE_ROOT,
        "source_branch": "lanxiang/release-cleanup",
        "source_commit": commit or None,
        "active_case_count": 14,
        "excluded_cells": [
            "table4/RacingCar/first",
            "table4/BilliardGame/third",
        ],
        "source_file_count": len(source_rows),
        "calibration_sample_count": len(sample_index),
        "calibration_sampling": {
            "source_interval_sec": 1.0,
            "directions": sorted(CALIBRATION_RUNS),
            "jpeg_max_width": max_width,
            "jpeg_quality": quality,
            "purpose": "dense_visual_adapter_calibration_evidence",
        },
        "source_files": source_rows,
        "calibration_samples": sample_index,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{PACKAGE_ROOT}.zip"
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for row in source_rows:
            _zip_write(archive, row["path"], (REPO_ROOT / row["path"]).read_bytes())
        for metadata_name in ("receipt.json", "turn_samples.json", "turn_calibration.json", "unity_yaw_alignment.json"):
            for direction, source_dir in CALIBRATION_RUNS.items():
                path = source_dir / metadata_name
                if path.is_file():
                    _zip_write(
                        archive,
                        f"calibration_samples/{direction}/{metadata_name}",
                        path.read_bytes(),
                    )
        for row in samples:
            _zip_write(archive, row["archive_path"], row["data"])
        _zip_write(
            archive,
            "calibration_samples/index.json",
            (json.dumps(sample_index, indent=2, sort_keys=True) + "\n").encode(),
        )
        _zip_write(
            archive,
            "MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _zip_write(archive, "README.md", package_readme(manifest).encode())

    archive_hash = sha256(archive_path)
    sidecar = archive_path.with_suffix(".zip.sha256")
    sidecar.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
    manifest.update(
        {
            "archive": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_hash,
        }
    )
    AUDIT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive_path, manifest


def verify(archive_path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            errors.append(f"ZIP CRC failure: {bad}")
        manifest = json.loads(archive.read(f"{PACKAGE_ROOT}/MANIFEST.json"))
        for row in manifest["source_files"]:
            data = archive.read(f"{PACKAGE_ROOT}/{row['path']}")
            if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
                errors.append(f"source artifact mismatch: {row['path']}")
        for row in manifest["calibration_samples"]:
            data = archive.read(f"{PACKAGE_ROOT}/{row['archive_path']}")
            if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
                errors.append(f"calibration sample mismatch: {row['archive_path']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"), nargs="?", default="build")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--calibration-max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=84)
    args = parser.parse_args()
    if args.command == "build":
        archive, manifest = build(
            args.output_dir.resolve(),
            max_width=args.calibration_max_width,
            quality=args.jpeg_quality,
        )
        errors = verify(archive)
    else:
        archive = (args.archive or (args.output_dir / f"{PACKAGE_ROOT}.zip")).resolve()
        manifest = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))
        errors = verify(archive)
        if sha256(archive) != manifest.get("archive_sha256"):
            errors.append("archive SHA-256 differs from the checked-in audit manifest")
    print(
        json.dumps(
            {
                "status": "ready" if not errors else "invalid",
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256(archive),
                "source_file_count": manifest["source_file_count"],
                "calibration_sample_count": manifest["calibration_sample_count"],
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
