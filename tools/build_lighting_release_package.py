#!/usr/bin/env python3
"""Build and verify a standalone lighting dataset plus the public Lite scorer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "SunRisesAndSets/lighting_orbit_v2"
SEEDS = (42, 1, 2, 3, 4)
MODELS = {"LingBot": "lingbot_world_cam", "Matrix-Game3": "matrix_game_3",
          "HY-WorldPlay": "hy_worldplay"}
UNSUPPORTED = ("matrix_game_2", "waypoint", "project_genie")
SAFE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".cs"}
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"required regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def runtime_file(relative):
    """Python imports and an in-bundle results directory are not release inputs."""
    path = Path(relative)
    return '__pycache__' in path.parts or path.parts[0] in {'results', '.cache'}


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and not runtime_file(p.relative_to(root))}
    expected = set(manifest["files"]) | {"MANIFEST.json"}
    if actual != expected:
        raise ValueError(f"release inventory mismatch: {actual ^ expected}")
    for name, row in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {name}")
        path = root / relative
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"release checksum mismatch: {name}")
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    wanted = {(model, str(seed)) for model in MODELS for seed in SEEDS}
    identities = [(row["model"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)) or set(identities) != wanted:
        raise ValueError("release requires exactly five seeds for each supported backend")
    for row in rows:
        name = Path(row["candidate_video"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("candidate depends on an external path")
        if row["case_id"] != CASE_ID or sha(root / name) != row["params"]["lighting_time_mapping"]["video_sha256"]:
            raise ValueError("candidate mapping/hash mismatch")
    support = manifest["support"]
    if set(support) != set(MODELS.values()) | set(UNSUPPORTED):
        raise ValueError("all six backends must be classified")
    if any(v["status"] not in {"supported", "unsupported"} for v in support.values()):
        raise ValueError("unfinished capability decision in release")
    if any(support[key]['status'] != 'supported' for key in MODELS.values()) or any(
            support[key]['status'] != 'unsupported' for key in UNSUPPORTED):
        raise ValueError('release population differs from capability decisions')
    return {"status": "verified", "files": len(manifest["files"]),
            "formal_jobs": len(rows), "supported": 3, "unsupported": 3,
            "manifest_sha256": sha(root / "MANIFEST.json")}


def build(input_manifest, destination, repo=ROOT):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    input_manifest = Path(input_manifest).resolve()
    inputs = json.loads(input_manifest.read_text())
    def input_path(value):
        path = Path(value)
        return path if path.is_absolute() else input_manifest.parent / path
    rows = inputs["runs"]
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if len(rows) != 15 or {(r["model"], r["seed"]) for r in rows} != expected:
        raise ValueError("input inventory must contain the fifteen verified formal runs")
    support = yaml.safe_load((repo / "experiments/tables_4_7/action_backends.yaml").read_text())[
        "trajectory_support"]["lighting.camera.look_at_orbit_v2"]["backends"]
    support = {key: support[key] for key in (*MODELS.values(), *UNSUPPORTED)}
    if any(support[key]["status"] != "supported" for key in MODELS.values()) or any(
            support[key]["status"] != "unsupported" for key in UNSUPPORTED):
        raise ValueError("capability decisions do not match release population")
    destination.mkdir(parents=True)
    case_source = repo / "eval_sets/wmeval_lite_v1/cases" / CASE_ID
    case_dest = destination / "eval_sets/wmeval_lite_v1/cases" / CASE_ID
    for path in sorted(case_source.rglob("*")):
        if path.is_file():
            copy(path, case_dest / path.relative_to(case_source))
    # Ship public evaluation sources; runtime dependencies are pinned separately.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo,
        text=True).splitlines()
    for name in sorted(set(names)):
        p = Path(name)
        if p.suffix not in SAFE_SUFFIXES:
            continue
        if p.parts[0] not in {"evaluation", "harness", "tools", "configs"} and len(p.parts) != 1:
            continue
        copy(repo / p, destination / p)
    metadata = (
        "experiments/tables_4_7/action_backends.yaml",
        "experiments/lighting_orbit_v2/genie_controls_frozen.json",
        "experiments/lighting_orbit_v2/results/capability_audit.json",
        "experiments/lighting_orbit_v2/results/run_summary.json",
        "experiments/lighting_orbit_v2/results/genie_diagnostic_summary.json",
        "experiments/lighting_orbit_v2/results/genie_calibration.json",
        "experiments/lighting_orbit_v2/results/waypoint_round2_calibration.json",
        "experiments/lighting_orbit_v2/results/generation_case.yaml",
        "experiments/lighting_orbit_v2/hy_runtime_inventory.json",
        "experiments/lighting_orbit_v2/hy_glyph_inventory.json",
    )
    for name in metadata:
        copy(repo / name, destination / name)
    inventories = repo / 'experiments/lighting_orbit_v2/generation_model_inventory.json'
    if inventories.exists():
        copy(inventories, destination / inventories.relative_to(repo))
    dependencies = repo / 'experiments/lighting_orbit_v2/requirements-lighting-score.lock'
    if dependencies.exists():
        copy(dependencies, destination / dependencies.relative_to(repo))
    sources = repo / 'experiments/lighting_orbit_v2/source_provenance'
    for path in sorted(sources.glob('*.json')):
        copy(path, destination / path.relative_to(repo))
    evidence = repo / 'experiments/lighting_orbit_v2/results/release_verification'
    for path in sorted(evidence.glob('*.json*')):
        copy(path, destination / path.relative_to(repo))
    exact = repo / 'experiments/lighting_orbit_v2/results/same_gpu_verification'
    for path in sorted(exact.rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.log', '.yaml', '.py'}:
            copy(path, destination / path.relative_to(repo))
    cached = repo / 'eval_sets/wmeval_lite_v1/baselines/lighting_orbit_v2/fresh_results'
    for path in sorted(cached.glob('*.json*')):
        copy(path, destination / 'baseline_results' / path.name)
    suite = yaml.safe_load((repo / "eval_sets/wmeval_lite_v1/suite.yaml").read_text())
    dimension = suite["dimensions"]["lighting_shading"]
    dimension["cases"] = [CASE_ID]
    dimension["description"] = "Fixed-sun metric camera look-at orbit; native ten-second window sampled at 12fps."
    dimension["metric_config"]["eval_fps"] = 12.0
    suite = {"schema_version": 1, "suite_name": "WMEval lighting orbit release",
             "version": "lighting-orbit-v2", "dimensions": {"lighting_shading": dimension}}
    (destination / "eval_sets/wmeval_lite_v1/suite.yaml").write_text(
        yaml.safe_dump(suite, sort_keys=False))
    candidates, expected_scores, jobs = [], [], []
    for row in sorted(rows, key=lambda r: (r["model"], r["seed"])):
        backend = MODELS[row["model"]]
        target = destination / "candidates" / backend / f"seed_{row['seed']}"
        source_video = input_path(row["video"])
        receipt_path = input_path(row["receipt"])
        receipt = json.loads(receipt_path.read_text())
        video_sha = sha(source_video)
        if receipt["status"] != "completed" or receipt["video_sha256"] != video_sha:
            raise ValueError(f"unverified formal source: {row['model']}/{row['seed']}")
        copy(source_video, target / "video.mp4")
        mapping = receipt["lighting_time_mapping"]
        if mapping["video_sha256"] != video_sha:
            raise ValueError("receipt time mapping references a different video")
        # Exact source receipt is archived, not rewritten to claim a new generation.
        copy(receipt_path, target / "generation_receipt.json")
        candidates.append({"model": row["model"], "case_id": CASE_ID, "seed": str(row["seed"]),
            "candidate_video": (target / "video.mp4").relative_to(destination).as_posix(),
            "params": {"lighting_time_mapping": mapping, "shadow_hud_preset": "none"}})
        expected_scores.append({"model": row["model"], "case_id": CASE_ID,
                                "seed": str(row["seed"]), "metrics": row["metrics"]})
        jobs.append({"model": row["model"], "backend": backend, "case_id": CASE_ID,
                     "seed": row["seed"], "status": "completed"})
    (destination / "candidates.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in candidates))
    write_json(destination / "expected_scores.json", {"schema_version": 1,
        "comparison": "same pinned evaluator/environment; compare every per-seed metric",
        "rows": expected_scores})
    write_json(destination / "expected_jobs.json", {"schema_version": 1, "jobs": jobs,
        "not_applicable": [{"backend": key, "status": "unsupported",
                            "reason": support[key]["reason"]} for key in UNSUPPORTED]})
    # Diagnostic videos remain separate from formal candidate discovery.
    for entry in inputs.get("diagnostics", []):
        source = input_path(entry["directory"])
        target = destination / "diagnostics/genie" / source.name
        for name in ("video.mp4", "action_trace.json", "run.json", "download.json"):
            copy(source / name, target / name)
        write_json(target / "scope.json", {"status": "unsupported_for_main_protocol",
                   "role": "diagnostic_only", "video_sha256": sha(target / "video.mp4")})
    (destination / "README.md").write_text(
        "# WMEval lighting orbit v2\n\n"
        "Three supported backends, five fixed seeds each; three explicitly N/A backends. "
        "Genie diagnostic media never enter formal aggregation.\n\n"
        "Run from this extracted directory:\n\n"
        "    python tools/build_lighting_release_package.py verify --root .\n"
        "    python tools/run_wmeval_lite.py run --dimensions lighting_shading "
        "--candidate-manifest candidates.jsonl --output-dir results --device cuda\n"
        "    python tools/run_wmeval_lite.py run --dimensions lighting_shading "
        "--candidate-manifest candidates.jsonl --output-dir results --device cuda --resume\n\n"
        "If baseline_results/ is present, its frozen fresh scores can be resumed without GPU inference:\n\n"
        "    cp -R baseline_results results\n"
        "    python tools/run_wmeval_lite.py run --dimensions lighting_shading "
        "--candidate-manifest candidates.jsonl --output-dir results --device cpu --resume\n\n"
        "Cached replay requires PyTorch, NumPy, OpenCV and PyYAML installed, but loads no "
        "model weights and needs no GPU or GPT API key.\n\n"
        "Native videos are sampled at 12fps over exactly ten seconds. No padding or time stretching. "
        "Fresh evaluation needs the case-locked Torch 2.8/CUDA 12.8 environment, SAM2 source "
        "2b90b9f5ceec907a1c18123530e92e794ad901a4 and the verified GroundingDINO/SAM2 assets. "
        "Use tools/prepare_lighting_evaluator.py with a dedicated cache and set "
        "HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 after preparing it. No GPT API key is used. "
        "Generation receipts preserve original provenance; descriptive absolute paths inside receipts "
        "are archival commands, not dataset file dependencies. All evaluated files resolve within this bundle.\n")
    verification = exact / 'full_release/verification.json'
    if not verification.exists():
        verification = evidence / 'verification.json'
    if verification.exists():
        status = json.loads(verification.read_text())
        with (destination / 'README.md').open('a') as stream:
            stream.write('\nFresh-score acceptance status: ' + status['status'] + '. '
                'The archived verification report is authoritative. Dataset checksum verification '
                'does not imply exact GPU-score reproducibility. Do not replace the expected scores '
                'or relax their tolerance to hide a failed comparison.\n')
            if status['status'] == 'passed':
                stream.write('Strict fresh-score equality is verified on NVIDIA RTX PRO 6000 '
                    'Blackwell Server Edition, driver 595.91.07, with the case-locked dependencies. '
                    'The separate RTX PRO 4500 audit has small numerical differences and is retained. '
                    'All raw scored rows are preserved. baseline_results contains the corrected '
                    'derived case identity stamp, with source_evaluator_lock and an explicit stamp '
                    'correction audit; its scores and evaluator inputs are unchanged.\n')
    manifest = {"schema_version": 1, "release_id": "wmeval-lighting-orbit-v2",
                "case_id": CASE_ID, "support": support,
                "scoring": {"fps": 12, "seconds": 10, "frames": 120},
                "files": {}}
    for p in sorted(destination.rglob("*")):
        if p.is_file():
            manifest["files"][p.relative_to(destination).as_posix()] = {
                "bytes": p.stat().st_size, "sha256": sha(p)}
    write_json(destination / "MANIFEST.json", manifest)
    return verify(destination)


def archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    verify(root)
    if output.is_relative_to(root):
        raise ValueError("archive output must be outside its source bundle")
    with zipfile.ZipFile(output, "w") as zip:
        names = sorted(set(json.loads((root / 'MANIFEST.json').read_text())['files']) | {'MANIFEST.json'})
        for name in names:
            p = root / name
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.external_attr = 0o644 << 16
            zip.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_STORED)
    return {"archive": str(output), "bytes": output.stat().st_size, "sha256": sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build"); p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--repo", type=Path, default=ROOT)
    p = sub.add_parser("verify"); p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("archive"); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.inputs, args.output, args.repo) if args.command == "build" else (
        verify(args.root) if args.command == "verify" else archive(args.root, args.output))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
