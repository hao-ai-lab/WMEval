#!/usr/bin/env python3
"""Fetch (or publish) the WMEval Lite case data on the Hugging Face Hub.

The repository keeps each case's metadata -- ``case.yaml`` and the checksum
manifest ``reference/checksums.sha256`` -- while the case data itself (reference
videos, conditioning images, action trajectories and other artifacts under
``reference/``) lives in the dataset ``lmgame/WMEval-Lite`` at the same relative
paths. ``download`` writes the files back into ``eval_sets/wmeval_lite_v1`` and
verifies every case against its manifest.

    python tools/sync_lite_data.py download            # curated cases (test_core)
    python tools/sync_lite_data.py download --split full   # + 220 synthetic trajectories (test_full)
    python tools/sync_lite_data.py status              # what is present locally
    python tools/sync_lite_data.py upload --token-file ~/.credentials/hf_key   # maintainers

``HF_TOKEN`` (or ``--token-file``) is needed only for private datasets and uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "eval_sets/wmeval_lite_v1"
DEFAULT_REPO = "lmgame/WMEval-Lite"
CHECKSUMS = "checksums.sha256"
# Case data lives under each case's reference/ directory; every suite (a companion suite
# via `cases_root: ../cases`) uses the shared cases/ tree.
DATA_PATTERNS = ["cases/*/*/reference/**"]
# test_full adds the synthetic trajectory suite (tools/generate_synthetic_trajectories.py).
SYNTHETIC_PATTERNS = ["synthetic/*/*/*/**"]
SYNTHETIC_MANIFEST = SUITE_ROOT / "synthetic" / "manifest.json"


def _token(args: argparse.Namespace) -> str | None:
    if getattr(args, "token_file", None):
        return Path(args.token_file).expanduser().read_text().strip()
    return os.environ.get("HF_TOKEN") or None


def case_dirs(root: Path = SUITE_ROOT) -> list[Path]:
    return sorted(p.parent for p in root.glob("cases/*/*/case.yaml"))


def manifest(case_dir: Path) -> dict[str, str]:
    """``relative path -> sha256`` from the case's checksum manifest (paths relative to the case)."""
    path = case_dir / "reference" / CHECKSUMS
    if not path.is_file():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            entries[rel.strip()] = digest
    return entries


def check_case(case_dir: Path, *, verify_hashes: bool = True) -> tuple[int, list[str], list[str]]:
    """(files listed, missing files, hash mismatches) for one case."""
    entries = manifest(case_dir)
    missing, mismatched = [], []
    for rel, digest in entries.items():
        path = case_dir / rel
        if not path.is_file():
            missing.append(rel)
        elif verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(rel)
    return len(entries), missing, mismatched


def status(root: Path = SUITE_ROOT, *, verify_hashes: bool = False) -> dict[str, tuple[int, list[str], list[str]]]:
    return {str(d.relative_to(root)): check_case(d, verify_hashes=verify_hashes) for d in case_dirs(root)}


def cmd_status(args: argparse.Namespace) -> int:
    report = status(verify_hashes=args.verify)
    complete = sum(1 for n, miss, bad in report.values() if n and not miss and not bad)
    for case, (n, miss, bad) in report.items():
        if miss or bad:
            print(f"incomplete {case}: {len(miss)} missing, {len(bad)} mismatched of {n}")
    print(f"{complete}/{len(report)} case(s) have complete data")
    return 0 if complete == len(report) else 1


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    # only the trajectories this checkout's manifest lists (the Hub may hold other versions too)
    patterns = DATA_PATTERNS + ([f"{t}/**" for t in synthetic_files()] if args.split == "full" else [])
    revision = args.revision or pinned_revision()
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(SUITE_ROOT),
        allow_patterns=patterns,
        token=_token(args),
    )
    report = status(verify_hashes=True)
    bad = {c: r for c, r in report.items() if r[1] or r[2]}
    for case, (n, miss, mism) in bad.items():
        print(f"FAILED {case}: {len(miss)} missing, {len(mism)} mismatched of {n}")
    print(f"downloaded {args.repo}@{revision or 'main'}: {len(report) - len(bad)}/{len(report)} case(s) verified")
    failed = bool(bad)
    if args.split == "full":
        n, missing, mism = check_synthetic()
        broken = {"/".join(m.split("/")[:4]) for m in missing + mism}  # synthetic/<family>/<case>/<slug>
        print(f"synthetic trajectories: {n - len(broken)}/{n} verified (references included)")
        failed = failed or bool(missing or mism)
    return 1 if failed else 0


# The current curated suite is the core test split; a procedurally extended
# split can be added beside it later.
INDEX_NAME = "test_core.jsonl"
FULL_INDEX_NAME = "test_full.jsonl"
# The Hub dataset is shared by every branch; each checkout downloads the commit its metadata
# was published with, so a newer dataset version never breaks an older branch.
REVISION_FILE = "DATASET_REVISION"
SYNTHETIC_SUITE = "wmeval_v1"  # suite label of the synthetic (non-Lite) test_full rows
INDEX_COLUMNS = ("case_id", "scene", "variant", "group", "family", "action_target", "suites", "dimensions",
                 "perspective", "prompt", "conditioning_case", "reference_video", "first_frame", "trajectory",
                 "candidate_normalization", "reference_status")


def build_index(root: Path = SUITE_ROOT) -> list[dict]:
    """One row per suite case: the Hub viewer's ``test_core`` split (paths are dataset-relative)."""
    import yaml

    suites = {"wmeval_lite_v1": root}
    for path in sorted(root.glob("*/suite.yaml")):
        suites[f"wmeval_lite_v1/{path.parent.name}"] = path.parent
    rows: dict[str, dict] = {}
    for suite_name, suite_root in suites.items():
        doc = yaml.safe_load((suite_root / "suite.yaml").read_text(encoding="utf-8"))
        cases_root = (suite_root / str(doc.get("cases_root") or "cases")).resolve()
        for dim, spec in (doc.get("dimensions") or {}).items():
            for case_id in spec.get("cases") or []:
                case_dir = cases_root / case_id
                rel_dir = case_dir.relative_to(root.resolve()).as_posix()
                row = rows.get(rel_dir)
                if row is None:
                    case = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
                    ref = case.get("reference") or {}
                    art = case_dir / "reference" / "artifacts"

                    def rel(path):
                        return f"{rel_dir}/{path}" if path else None

                    row = rows[rel_dir] = {
                        "case_id": case_id,
                        "scene": case.get("scene") or case_id.split("/")[0],
                        "variant": case_id.split("/")[1],
                        "group": "core",
                        "family": "core",
                        "action_target": _declared_target(art),
                        "suites": [],
                        "dimensions": [],
                        "perspective": case.get("perspective"),
                        "prompt": case.get("prompt"),
                        "conditioning_case": case_id,
                        # the declared reference video, else the Unity video the bundle ships even when
                        # its metrics are reference-free (prompt-only cases), so every row links one
                        "reference_video": rel(ref.get("video")) or _bundled_video(case_dir, rel_dir),
                        "first_frame": rel(ref.get("first_frame") or ref.get("video")),
                        "trajectory": rel("reference/artifacts/trajectory.yaml") if (art / "trajectory.yaml").is_file() else None,
                        "candidate_normalization": (case.get("protocol") or {}).get("candidate_normalization"),
                        "reference_status": "unity_rendered",
                    }
                if suite_name not in row["suites"]:
                    row["suites"].append(suite_name)
                row["dimensions"].append(dim if suite_name == "wmeval_lite_v1" else f"{dim} ({suite_root.name})")
    return sorted(rows.values(), key=lambda r: r["case_id"] + ("~" if r["suites"] != ["wmeval_lite_v1"] else ""))


def _bundled_video(case_dir: Path, rel_dir: str) -> str | None:
    videos = sorted((case_dir / "reference").glob("*.mp4"))
    return f"{rel_dir}/reference/{videos[0].name}" if videos else None


def _declared_target(artifacts: Path) -> str | None:
    try:
        from evaluation.lite.action_support import declared_target

        return declared_target(artifacts)
    except Exception:  # noqa: BLE001 -- index metadata only
        return None


def synthetic_rows(root: Path = SUITE_ROOT) -> list[dict]:
    """test_full rows for the synthetic trajectories (from the tracked manifest)."""
    manifest = root / "synthetic" / "manifest.json"
    if not manifest.is_file():
        return []
    import json

    import yaml

    normalization: dict[str, str | None] = {}  # scored with the conditioning case's protocol
    rows = []
    for t in json.loads(manifest.read_text(encoding="utf-8"))["trajectories"]:
        cond = t["conditioning_case"]
        if cond not in normalization:
            case_yaml = root / "cases" / cond / "case.yaml"
            raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8")) if case_yaml.is_file() else None
            protocol = (raw or {}).get("protocol") or {}
            normalization[cond] = protocol.get("candidate_normalization")
        rows.append({
            "case_id": t["synthetic_id"], "scene": t["scene"], "variant": t["pattern"], "group": "synthetic",
            "family": t["family"], "action_target": t["action_target"], "suites": [SYNTHETIC_SUITE], "dimensions": t["dimensions"],
            "perspective": t["perspective"], "prompt": t["prompt"], "conditioning_case": t["conditioning_case"],
            "reference_video": t.get("reference_video"), "first_frame": t["first_frame"], "trajectory": f"{t['path']}/trajectory.yaml",
            "candidate_normalization": normalization[cond], "reference_status": t["reference_status"],
        })
    return rows


def build_full_index(root: Path = SUITE_ROOT) -> list[dict]:
    return build_index(root) + synthetic_rows(root)


def synthetic_files(root: Path = SUITE_ROOT) -> dict[str, dict[str, str]]:
    """trajectory path -> {file relative to it: sha256}: generated files plus its published Unity reference."""
    import json

    manifest = root / "synthetic" / "manifest.json"
    if not manifest.is_file():
        return {}
    out = {t["path"]: dict(t["files"]) for t in json.loads(manifest.read_text(encoding="utf-8"))["trajectories"]}
    references = root / "synthetic" / "references.json"
    if references.is_file():
        for sid, ref in json.loads(references.read_text(encoding="utf-8"))["references"].items():
            if sid in out:
                out[sid].update(ref["files"])
    return out


def check_synthetic(root: Path = SUITE_ROOT, *, verify_hashes: bool = True) -> tuple[int, list[str], list[str]]:
    """(trajectories listed, missing files, mismatched files) for the synthetic suite, references included."""
    files = synthetic_files(root)
    if not files:
        return 0, ["synthetic/manifest.json"], []
    missing, bad = [], []
    for traj, entries in files.items():
        for name, digest in entries.items():
            path = root / traj / name
            if not path.is_file():
                missing.append(f"{traj}/{name}")
            elif verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                bad.append(f"{traj}/{name}")
    return len(files), missing, bad


def pinned_revision(root: Path = SUITE_ROOT) -> str | None:
    """The Hub commit this checkout's metadata was published with (``DATASET_REVISION``)."""
    pin = root / REVISION_FILE
    return pin.read_text(encoding="utf-8").strip() or None if pin.is_file() else None


def cmd_upload(args: argparse.Namespace) -> int:
    import json
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi(token=_token(args))
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    card = SUITE_ROOT / "DATASET_CARD.md"
    if card.is_file():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset",
                        commit_message="Dataset card")
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in ((INDEX_NAME, build_index()), (FULL_INDEX_NAME, build_full_index())):
            index = Path(tmp) / name
            index.write_text("".join(json.dumps({k: r.get(k) for k in INDEX_COLUMNS}, ensure_ascii=False) + "\n"
                                     for r in rows), encoding="utf-8")
            api.upload_file(path_or_fileobj=str(index), path_in_repo=name, repo_id=args.repo, repo_type="dataset",
                            commit_message=f"Case index ({name[:-6]} split)")
    if args.card_only:
        print(f"updated the dataset card and {INDEX_NAME} on https://huggingface.co/datasets/{args.repo}")
        return 0
    report = status(verify_hashes=True)
    bad = {c: r for c, r in report.items() if r[1] or r[2]}
    if bad:
        print(f"refusing to upload: {len(bad)} case(s) incomplete locally, e.g. {next(iter(bad))}")
        return 1
    patterns = list(DATA_PATTERNS)
    if args.split == "full":
        n, missing, mism = check_synthetic()
        if missing or mism:
            print(f"refusing to upload: synthetic suite incomplete ({len(missing)} missing, {len(mism)} mismatched); "
                  "run tools/generate_synthetic_trajectories.py")
            return 1
        patterns += [f"{t}/**" for t in synthetic_files()]
    api.upload_folder(
        folder_path=str(SUITE_ROOT),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=patterns,
        # Other branches may still list older synthetic trajectories, so nothing is deleted
        # unless asked (once every branch has moved to the current manifest).
        delete_patterns=SYNTHETIC_PATTERNS if args.split == "full" and args.prune_synthetic else None,
        commit_message=args.message,
    )
    head = api.repo_info(args.repo, repo_type="dataset").sha
    (SUITE_ROOT / REVISION_FILE).write_text(head + "\n", encoding="utf-8")
    print(f"uploaded {len(report)} case(s) to https://huggingface.co/datasets/{args.repo}"
          f" ({'private' if args.private else 'public'}) at {head}; commit {REVISION_FILE} to pin it")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("download", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=DEFAULT_REPO)
        p.add_argument("--token-file")
    sub.choices["download"].add_argument("--revision", help=f"dataset revision (tag/commit); default: {REVISION_FILE}, else main")
    for name in ("download", "upload"):
        sub.choices[name].add_argument("--split", choices=("core", "full"), default="core",
                                       help="core = curated cases (test_core); full = core + synthetic trajectories (test_full)")
    sub.choices["upload"].add_argument("--private", action="store_true", help="create the dataset as private")
    sub.choices["upload"].add_argument("--message", default="Upload WMEval Lite case data")
    sub.choices["upload"].add_argument("--prune-synthetic", action="store_true",
                                       help="delete Hub synthetic trajectories that this manifest no longer lists")
    sub.choices["upload"].add_argument("--card-only", action="store_true",
                                       help="refresh only the dataset card and the case index")
    p_status = sub.add_parser("status")
    p_status.add_argument("--verify", action="store_true", help="also verify sha256")
    args = parser.parse_args(argv)
    return {"download": cmd_download, "upload": cmd_upload, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
