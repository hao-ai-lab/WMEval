"""Case-bundle authoring: checksum manifest + evaluator lock refresh.

Shared by tools/build_lite_case.py and the generate-rubrics CLI path (which
writes rubric files into the checksummed reference/ area and must therefore
re-author the bundle afterwards).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
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
    """Rewrite reference/checksums.sha256 covering every file under reference/."""
    reference_root = case_dir / "reference"
    if not reference_root.is_dir():
        raise ValueError(f"no reference/ directory under {case_dir}")
    checksums_path = reference_root / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(reference_root.rglob("*")):
        if not path.is_file() or path == checksums_path:
            continue
        relative = path.relative_to(case_dir).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums_path


def _fingerprint(case_dir: Path, raw: dict) -> str:
    checksums = (case_dir / "reference" / "checksums.sha256").read_text(
        encoding="utf-8"
    )
    # The lock identifies the complete evaluator contract, not merely the
    # media checksum and normalization protocol.  In particular,
    # ``module_params`` carries prompts, thresholds, model IDs and analytical
    # ground-truth bindings; changing any of those must invalidate resumable
    # results.  Exclude only the self-referential lock block.
    case_config = {key: value for key, value in raw.items() if key != "evaluator_lock"}
    evaluator_sources = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted((REPO_ROOT / "evaluation").rglob("*.py"))
        if path.is_file()
    }
    payload = json.dumps(
        {
            "checksums": sorted(checksums.splitlines()),
            "case_config": case_config,
            "evaluator_sources": evaluator_sources,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def update_lock(case_dir: Path, *, relock: bool) -> None:
    case_yaml = case_dir / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{case_yaml} must contain a mapping")
    lock = raw.setdefault("evaluator_lock", {})
    if not isinstance(lock, dict):
        raise ValueError(f"{case_yaml}: evaluator_lock must be a mapping")

    def fill(key: str, value: str) -> None:
        if relock or not str(lock.get(key) or "").strip():
            lock[key] = value

    fill("code_revision", git_revision())
    # evaluator_fingerprint is DERIVED from checksums+protocol — recompute it
    # unconditionally so a bundle edit can never leave a silently stale stamp
    # (--relock only governs code_revision, which is a deliberate declaration
    # of when/at which revision the case was locked).
    lock["evaluator_fingerprint"] = _fingerprint(case_dir, raw)
    lock.setdefault("release_id", "wmeval-lite-v1")
    lock.setdefault("module_config", {})
    lock.setdefault("artifact_schema_versions", {})
    lock.setdefault("dependency_versions", {})

    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def author_case(case_dir: str | Path, *, relock: bool = False) -> Path:
    """Refresh checksums + evaluator lock after reference/ contents changed."""
    resolved = Path(case_dir).resolve()
    checksums_path = write_checksums(resolved)
    update_lock(resolved, relock=relock)
    return checksums_path
