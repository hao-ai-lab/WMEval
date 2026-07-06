#!/usr/bin/env python3
"""Author or refresh a WMEval Lite case bundle.

Given a case directory whose ``reference/`` files are already in place and a
``case.yaml`` that may be missing its ``checksums.sha256`` / ``evaluator_lock``
values, this tool:

1. writes ``reference/checksums.sha256`` covering every file under
   ``reference/`` (excluding the checksum manifest itself), and
2. fills ``evaluator_lock.code_revision`` (current git HEAD) and
   ``evaluator_lock.evaluator_fingerprint`` (sha256 over the sorted checksum
   manifest + case protocol) unless already set and ``--relock`` is absent.

Usage::

    python tools/build_lite_case.py eval_sets/wmeval_lite_v1/cases/mirror_ball/third_person
    python tools/build_lite_case.py --relock eval_sets/wmeval_lite_v1/cases/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_checksums(case_dir: Path) -> Path:
    reference_root = case_dir / "reference"
    if not reference_root.is_dir():
        raise SystemExit(f"no reference/ directory under {case_dir}")
    checksums_path = reference_root / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(reference_root.rglob("*")):
        if not path.is_file() or path == checksums_path:
            continue
        relative = path.relative_to(case_dir).as_posix()
        lines.append(f"{_sha256_file(path)}  {relative}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums_path


def _fingerprint(case_dir: Path, raw: dict) -> str:
    checksums = (case_dir / "reference" / "checksums.sha256").read_text(
        encoding="utf-8"
    )
    payload = json.dumps(
        {"checksums": sorted(checksums.splitlines()), "protocol": raw.get("protocol")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def update_lock(case_dir: Path, *, relock: bool) -> None:
    case_yaml = case_dir / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{case_yaml} must contain a mapping")
    lock = raw.setdefault("evaluator_lock", {})
    if not isinstance(lock, dict):
        raise SystemExit(f"{case_yaml}: evaluator_lock must be a mapping")

    def fill(key: str, value: str) -> None:
        if relock or not str(lock.get(key) or "").strip():
            lock[key] = value

    fill("code_revision", _git_revision())
    fill("evaluator_fingerprint", _fingerprint(case_dir, raw))
    lock.setdefault("release_id", "wmeval-lite-v1")
    lock.setdefault("module_config", {})
    lock.setdefault("artifact_schema_versions", {})
    lock.setdefault("dependency_versions", {})

    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+", help="Case directories to (re)build")
    parser.add_argument(
        "--relock",
        action="store_true",
        help="Overwrite code_revision/evaluator_fingerprint even when set",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip load_case/validate_case_bundle after writing",
    )
    args = parser.parse_args(argv)

    for case_dir_arg in args.case_dirs:
        case_dir = Path(case_dir_arg).resolve()
        checksums_path = write_checksums(case_dir)
        update_lock(case_dir, relock=args.relock)
        print(f"wrote {checksums_path}")
        if not args.skip_validate:
            from evaluation.lite.case_loader import load_case
            from evaluation.lite.validation import validate_case_bundle

            case = load_case(case_dir, allowed_cases={_case_id(case_dir)})
            validate_case_bundle(case)
            print(f"validated {case.case_id}")
    return 0


def _case_id(case_dir: Path) -> str:
    return f"{case_dir.parent.name}/{case_dir.name}"


if __name__ == "__main__":
    sys.exit(main())
