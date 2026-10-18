#!/usr/bin/env python3
"""Fetch (or publish) the WMEval Lite case data on the Hugging Face Hub.

The repository keeps each case's metadata -- ``case.yaml`` and the checksum
manifest ``reference/checksums.sha256`` -- while the case data itself (reference
videos, conditioning images, action trajectories and other artifacts under
``reference/``) lives in the dataset ``lmgame/WMEval-Lite`` at the same relative
paths. ``download`` writes the files back into ``eval_sets/wmeval_lite_v1`` and
verifies every case against its manifest.

    python tools/sync_lite_data.py download            # all suites (default)
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
# Case data lives under each case's reference/ directory; every suite (including the
# paper-native lighting suite, via `cases_root: ../cases`) uses the shared cases/ tree.
DATA_PATTERNS = ["cases/*/*/reference/**"]


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

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(SUITE_ROOT),
        allow_patterns=DATA_PATTERNS,
        token=_token(args),
    )
    report = status(verify_hashes=True)
    bad = {c: r for c, r in report.items() if r[1] or r[2]}
    for case, (n, miss, mism) in bad.items():
        print(f"FAILED {case}: {len(miss)} missing, {len(mism)} mismatched of {n}")
    print(f"downloaded {args.repo}@{args.revision or 'main'}: {len(report) - len(bad)}/{len(report)} case(s) verified")
    return 1 if bad else 0


# The current curated suite is the core test split; a procedurally extended
# split can be added beside it later.
INDEX_NAME = "test_core.jsonl"


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
                        "suites": [],
                        "dimensions": [],
                        "perspective": case.get("perspective"),
                        "prompt": case.get("prompt"),
                        "reference_video": rel(ref.get("video")),
                        "first_frame": rel(ref.get("first_frame")),
                        "trajectory": rel("reference/artifacts/trajectory.yaml") if (art / "trajectory.yaml").is_file() else None,
                        "candidate_normalization": (case.get("protocol") or {}).get("candidate_normalization"),
                    }
                if suite_name not in row["suites"]:
                    row["suites"].append(suite_name)
                row["dimensions"].append(dim if suite_name == "wmeval_lite_v1" else f"{dim} (paper-native)")
    return sorted(rows.values(), key=lambda r: r["case_id"] + ("~" if r["suites"] != ["wmeval_lite_v1"] else ""))


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
        index = Path(tmp) / INDEX_NAME
        index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build_index()), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(index), path_in_repo=INDEX_NAME, repo_id=args.repo, repo_type="dataset",
                        commit_message="Case index (test_core split)")
    if args.card_only:
        print(f"updated the dataset card and {INDEX_NAME} on https://huggingface.co/datasets/{args.repo}")
        return 0
    report = status(verify_hashes=True)
    bad = {c: r for c, r in report.items() if r[1] or r[2]}
    if bad:
        print(f"refusing to upload: {len(bad)} case(s) incomplete locally, e.g. {next(iter(bad))}")
        return 1
    api.upload_folder(
        folder_path=str(SUITE_ROOT),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=DATA_PATTERNS,
        commit_message=args.message,
    )
    print(f"uploaded {len(report)} case(s) to https://huggingface.co/datasets/{args.repo}"
          f" ({'private' if args.private else 'public'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("download", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=DEFAULT_REPO)
        p.add_argument("--token-file")
    sub.choices["download"].add_argument("--revision", help="dataset revision (tag/commit) to pin")
    sub.choices["upload"].add_argument("--private", action="store_true", help="create the dataset as private")
    sub.choices["upload"].add_argument("--message", default="Upload WMEval Lite case data")
    sub.choices["upload"].add_argument("--card-only", action="store_true",
                                       help="refresh only the dataset card and the case index")
    p_status = sub.add_parser("status")
    p_status.add_argument("--verify", action="store_true", help="also verify sha256")
    args = parser.parse_args(argv)
    return {"download": cmd_download, "upload": cmd_upload, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
