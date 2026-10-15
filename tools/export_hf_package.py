#!/usr/bin/env python3
"""Export and verify a self-contained WMEval Lite dataset package.

The package is what goes to the Hugging Face dataset repo: everything a user
needs to *score* a model against the frozen Unity references, and nothing
else (no maintainer notes, no orphan case directories, no caches, nothing
git-ignored). The evaluator code itself is NOT in the package; MANIFEST.json
pins the evaluator code revision the package was exported from.

Usage::

    # Export the suite (plus companion suites / human_alignment / baselines)
    python tools/export_hf_package.py export \\
        --suite eval_sets/wmeval_lite_v1 --output /tmp/wmeval_lite_hf

    # Re-hash, scan for out-of-package references, run the suite validator
    python tools/export_hf_package.py verify /tmp/wmeval_lite_hf

What ``export`` copies (all discovered, nothing hard-coded per case):

* ``suite.yaml`` and every case bundle it lists under ``cases/``;
* every *companion suite*: a direct subdirectory of the suite root holding
  its own ``suite.yaml`` (e.g. ``lighting_paper_native/``), with the case
  bundles that suite lists;
* ``human_alignment/`` and ``baselines/`` (frozen scores and rollouts that
  ship with the suite; ``--no-baselines`` drops the latter);
* ``DATASET_CARD.md``, renamed to ``README.md`` (the Hub reads the card from
  ``README.md``).

Case directories present on disk but not listed by any suite, and other
suite-root files (``PENDING.md`` etc.), are reported and left out. Every path
a ``case.yaml`` declares (``reference.*``, ``reference.artifacts.*`` and the
known ``module_params`` path keys) must resolve inside its own case bundle;
export fails otherwise.

What ``verify`` checks:

1. integrity -- every MANIFEST.json entry exists with the recorded size and
   sha256; files on disk that the manifest does not list are reported;
2. self-containment -- every exported text file (yaml/yml/json/jsonl/md/txt)
   is scanned for references that would resolve outside the package
   (absolute paths, ``~``, ``../`` escapes, ``runs/``, ``modal://`` and Modal
   volume names, private repos). Each hit is classified as
   ``load_bearing`` (a path-valued field in a file the scorer reads --
   suite.yaml, case.yaml, anything in a case bundle: a real dependency),
   ``record_reference`` (the same in a record file such as a baseline score
   table or generation receipt: points at something not shipped, but
   nothing reads it when scoring), ``runtime_cache`` (a cwd-relative
   *output* location the evaluator writes to: not an input), or
   ``provenance`` (a YAML/txt comment, a description / note / command /
   source string, or Markdown prose). Relative path values that resolve to
   nothing inside the package are reported the same way;
3. the suite validator (``tools/run_wmeval_lite.py validate``) is run on the
   exported suite and on each companion suite, from inside the package
   directory, to prove the bundles load and checksum without the repo's
   ``eval_sets/``.

Exit status of ``verify`` is non-zero on any integrity error, load-bearing
hit, or validator failure (``--strict`` also fails on provenance hits).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_NAME = "tools/export_hf_package.py"
EXPORTER_VERSION = "1.0.0"
MANIFEST_NAME = "MANIFEST.json"
MANIFEST_FORMAT_VERSION = 1

# Suite-root directories that ship with the suite even though suite.yaml does
# not name them by path. ``baselines`` is optional (--no-baselines).
EXTRA_DIRS = ("human_alignment", "baselines")
DATASET_CARD = "DATASET_CARD.md"

# Never exported, never counted as "extra" by verify.
CACHE_DIR_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
     ".ipynb_checkpoints", ".cache", ".git"}
)
CACHE_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
CACHE_SUFFIXES = (".pyc", ".pyo", ".swp", ".tmp", "~")

TEXT_SUFFIXES = frozenset({".yaml", ".yml", ".json", ".jsonl", ".md", ".txt"})

# case.yaml module_params keys whose values are case-relative paths. Kept in
# sync with evaluation.lite.case_loader._MODULE_PARAM_PATH_KEYS (imported
# when available; this copy keeps the exporter usable without the package).
_FALLBACK_MODULE_PARAM_PATH_KEYS = frozenset(
    {
        "eoc_unity_object_transforms_json", "eoc_unity_poses_npy",
        "eoc_unity_intrinsics_npy", "eoc_unity_object_diameters_json",
        "kin_unity_object_transforms_json", "kin_unity_poses_npy",
        "kin_unity_intrinsics_npy", "kin_unity_object_masses_json",
        "lpips_reference_events_json", "kin_plane_calibration_json",
    }
)

_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class ExportError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cache_path(relative: PurePosixPath) -> bool:
    if any(part in CACHE_DIR_NAMES for part in relative.parts[:-1]):
        return True
    name = relative.name
    return name in CACHE_FILE_NAMES or name.endswith(CACHE_SUFFIXES)


def _module_param_path_keys() -> frozenset[str]:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from evaluation.lite.case_loader import _MODULE_PARAM_PATH_KEYS

        return frozenset(_MODULE_PARAM_PATH_KEYS)
    except Exception:  # pragma: no cover - exporter usable standalone
        return _FALLBACK_MODULE_PARAM_PATH_KEYS


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExportError(f"cannot parse {path}: {exc}") from exc


def suite_case_ids(suite_doc: Any) -> list[str]:
    """Case ids in first-appearance order across dimensions."""
    seen: dict[str, None] = {}
    dimensions = (suite_doc or {}).get("dimensions") or {}
    for spec in dimensions.values():
        for case_id in (spec or {}).get("cases") or []:
            seen.setdefault(str(case_id), None)
    return list(seen)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def git_head(repo: Path = REPO_ROOT) -> dict[str, Any]:
    head = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return {
        "revision": head.strip() if head else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def git_ignored(paths: list[Path], cwd: Path) -> set[Path]:
    """Subset of ``paths`` that git ignores (tracked files never count)."""
    if not paths or _git(["rev-parse", "--is-inside-work-tree"], cwd) is None:
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            cwd=cwd,
            input="\0".join(str(p) for p in paths) + "\0",
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    if result.returncode not in (0, 1):
        raise ExportError(f"git check-ignore failed: {result.stderr.strip()}")
    return {Path(item) for item in result.stdout.split("\0") if item}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


@dataclass
class SuitePlan:
    rel: str  # suite root relative to the package root ("." for the main one)
    root: Path  # source suite root
    doc: dict[str, Any]
    cases: list[str]
    unlisted_case_dirs: list[str] = field(default_factory=list)


def _filter_suite_doc(doc: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """suite.yaml restricted to ``keep``; empty dimensions are dropped."""
    out = dict(doc)
    dimensions = {}
    for name, spec in (doc.get("dimensions") or {}).items():
        cases = [c for c in (spec.get("cases") or []) if c in keep]
        if cases:
            dimensions[name] = {**spec, "cases": cases}
    out["dimensions"] = dimensions
    aliases = {}
    for name, mapping in (doc.get("legacy_case_aliases") or {}).items():
        if name in dimensions:
            kept = {k: v for k, v in (mapping or {}).items() if v in keep}
            if kept:
                aliases[name] = kept
    if "legacy_case_aliases" in doc:
        out["legacy_case_aliases"] = aliases
    return out


def _case_declared_paths(case_root: Path, doc: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, relative path) pairs a case.yaml declares."""
    out: list[tuple[str, str]] = []
    reference = doc.get("reference") or {}
    for key, value in reference.items():
        if key == "artifacts" and isinstance(value, dict):
            for name, rel in value.items():
                out.append((f"reference.artifacts.{name}", str(rel)))
        elif isinstance(value, str):
            out.append((f"reference.{key}", value))
    path_keys = _module_param_path_keys()
    for module, params in (doc.get("module_params") or {}).items():
        for key, value in (params or {}).items():
            if key in path_keys and isinstance(value, str):
                out.append((f"module_params.{module}.{key}", value))
    return out


def _check_case_bundle(case_root: Path, case_id: str) -> None:
    case_yaml = case_root / "case.yaml"
    if not case_yaml.is_file():
        raise ExportError(f"{case_id}: missing {case_yaml}")
    doc = _load_yaml(case_yaml) or {}
    problems = []
    resolved_root = case_root.resolve()
    for where, rel in _case_declared_paths(case_root, doc):
        if Path(rel).is_absolute():
            problems.append(f"{where}={rel!r} is absolute")
            continue
        target = (case_root / rel).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError:
            problems.append(f"{where}={rel!r} escapes the case bundle")
            continue
        if not target.exists():
            problems.append(f"{where}={rel!r} does not exist")
    if problems:
        raise ExportError(f"{case_id}: " + "; ".join(problems))


def plan_suites(suite_root: Path, cases: list[str] | None) -> list[SuitePlan]:
    main_yaml = suite_root / "suite.yaml"
    if not main_yaml.is_file():
        raise ExportError(f"no suite.yaml under {suite_root}")
    roots: list[tuple[str, Path]] = [(".", suite_root)]
    for child in sorted(suite_root.iterdir()):
        if child.is_dir() and child.name != "cases" and (child / "suite.yaml").is_file():
            roots.append((child.name, child))

    keep = set(cases) if cases else None
    if keep is not None:
        main_ids = set(suite_case_ids(_load_yaml(main_yaml)))
        unknown = sorted(keep - main_ids)
        if unknown:
            raise ExportError(f"--cases not listed by {main_yaml}: {unknown}")

    plans = []
    for rel, root in roots:
        doc = _load_yaml(root / "suite.yaml")
        if not isinstance(doc, dict):
            raise ExportError(f"{root / 'suite.yaml'} is not a mapping")
        ids = suite_case_ids(doc)
        if keep is not None:
            ids = [c for c in ids if c in keep]
            if not ids:
                continue  # companion suite has nothing in the subset
        on_disk = sorted(
            p.parent.relative_to(root / "cases").as_posix()
            for p in (root / "cases").glob("*/*/case.yaml")
        ) if (root / "cases").is_dir() else []
        listed_all = set(suite_case_ids(doc))
        plans.append(
            SuitePlan(
                rel=rel,
                root=root,
                doc=doc,
                cases=ids,
                unlisted_case_dirs=[c for c in on_disk if c not in listed_all],
            )
        )
    return plans


def _walk_files(directory: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in CACHE_DIR_NAMES)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def collect_export_files(
    suite_root: Path,
    plans: list[SuitePlan],
    *,
    include_baselines: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Map package-relative path -> source file, plus an exclusion report."""
    selected: dict[str, Path] = {}
    excluded: dict[str, Any] = {
        "unlisted_case_dirs": [],
        "suite_root_files": [],
        "cache_files": 0,
        "git_ignored": [],
    }

    def add_tree(source_dir: Path, package_prefix: PurePosixPath) -> None:
        for path in _walk_files(source_dir):
            rel = PurePosixPath(path.relative_to(source_dir).as_posix())
            if is_cache_path(rel):
                excluded["cache_files"] += 1
                continue
            selected[str(package_prefix / rel)] = path

    for plan in plans:
        prefix = PurePosixPath(plan.rel) if plan.rel != "." else PurePosixPath()
        if plan.rel == ".":
            excluded["unlisted_case_dirs"] += [f"cases/{c}" for c in plan.unlisted_case_dirs]
        else:
            excluded["unlisted_case_dirs"] += [
                f"{plan.rel}/cases/{c}" for c in plan.unlisted_case_dirs
            ]
        for case_id in plan.cases:
            case_root = plan.root / "cases" / case_id
            _check_case_bundle(case_root, case_id)
            add_tree(case_root, prefix / "cases" / case_id)
        # suite.yaml is written separately (it may be filtered for --cases);
        # companion suite roots carry nothing else we export.
        for other in sorted(plan.root.iterdir()):
            if plan.rel != "." and other.name not in ("suite.yaml", "cases"):
                excluded["suite_root_files"].append(f"{plan.rel}/{other.name}")

    companion_names = {p.rel for p in plans if p.rel != "."}
    for child in sorted(suite_root.iterdir()):
        name = child.name
        if name in ("suite.yaml", "cases") or name in companion_names:
            continue
        if (child / "suite.yaml").is_file():
            continue  # companion suite dropped by --cases
        if child.is_dir() and name in EXTRA_DIRS:
            if name == "baselines" and not include_baselines:
                excluded["suite_root_files"].append(name + "/ (--no-baselines)")
                continue
            add_tree(child, PurePosixPath(name))
        elif child.is_file() and name == DATASET_CARD:
            selected["README.md"] = child
        elif name in CACHE_FILE_NAMES or name in CACHE_DIR_NAMES:
            excluded["cache_files"] += 1
        else:
            excluded["suite_root_files"].append(name + ("/" if child.is_dir() else ""))

    for rel, source in selected.items():
        if source.is_symlink():
            raise ExportError(f"refusing to export symlink {source} (as {rel})")

    ignored = git_ignored(list(selected.values()), suite_root)
    if ignored:
        for rel in sorted(selected):
            if selected[rel] in ignored:
                excluded["git_ignored"].append(rel)
                del selected[rel]
    return selected, excluded


def _json_schema_fields(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(doc, dict):
        return {}
    return {
        key: doc[key]
        for key in ("schema_version", "schema", "format_version")
        if key in doc and isinstance(doc[key], (str, int, float))
    }


def collect_schema_versions(
    package_root: Path, plans: list[SuitePlan], files: Iterable[str]
) -> dict[str, Any]:
    suite_versions = {
        plan.rel: plan.doc.get("schema_version") for plan in plans
    }
    case_versions: Counter[str] = Counter()
    release_ids: set[str] = set()
    artifact_versions: dict[str, set[Any]] = defaultdict(set)
    for plan in plans:
        prefix = Path(plan.rel)
        for case_id in plan.cases:
            doc = _load_yaml(package_root / prefix / "cases" / case_id / "case.yaml") or {}
            case_versions[str(doc.get("schema_version"))] += 1
            lock = doc.get("evaluator_lock") or {}
            if lock.get("release_id"):
                release_ids.add(str(lock["release_id"]))
            for name, version in (lock.get("artifact_schema_versions") or {}).items():
                artifact_versions[str(name)].add(version)
    yaml_doc_versions: dict[str, set[Any]] = defaultdict(set)
    json_fields: dict[str, dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))
    for rel in files:
        path = package_root / rel
        name = PurePosixPath(rel).name
        if name in ("case.yaml", "suite.yaml"):
            continue
        if path.suffix == ".json":
            for key, value in _json_schema_fields(path).items():
                json_fields[name][key].add(value)
        elif path.suffix in (".yaml", ".yml"):
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, UnicodeDecodeError):
                continue
            if isinstance(doc, dict) and "schema_version" in doc:
                yaml_doc_versions[name].add(doc["schema_version"])

    def _sorted(values: set[Any]) -> list[Any]:
        return sorted(values, key=lambda v: (str(type(v)), str(v)))

    return {
        "suite_yaml_schema_version": suite_versions,
        "case_yaml_schema_version": dict(sorted(case_versions.items())),
        "evaluator_lock_release_ids": sorted(release_ids),
        "evaluator_lock_artifact_schema_versions": {
            k: _sorted(v) for k, v in sorted(artifact_versions.items())
        },
        "artifact_yaml_schema_version": {
            k: _sorted(v) for k, v in sorted(yaml_doc_versions.items())
        },
        "artifact_json_schema_fields": {
            name: {k: _sorted(v) for k, v in sorted(keys.items())}
            for name, keys in sorted(json_fields.items())
        },
    }


def _dataset_identity(main: SuitePlan, schema: dict[str, Any]) -> dict[str, Any]:
    doc = main.doc
    version = doc.get("release_version") or doc.get("version")
    for key in ("release_id", "dataset_id", "release"):
        if doc.get(key):
            return {"id": str(doc[key]), "id_source": f"suite.yaml:{key}",
                    "version": version, "suite_name": doc.get("suite_name")}
    release_ids = schema["evaluator_lock_release_ids"]
    if len(release_ids) == 1:
        return {"id": release_ids[0],
                "id_source": "case.yaml:evaluator_lock.release_id (uniform across cases)",
                "version": version, "suite_name": doc.get("suite_name")}
    slug = re.sub(r"[^a-z0-9]+", "-", str(doc.get("suite_name", "suite")).lower()).strip("-")
    return {"id": f"{slug}-{version}", "id_source": "suite.yaml:suite_name+version",
            "version": version, "suite_name": doc.get("suite_name")}


def export_package(
    suite_root: str | Path,
    output: str | Path,
    *,
    cases: list[str] | None = None,
    include_baselines: bool = True,
) -> dict[str, Any]:
    suite_root = Path(suite_root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ExportError(f"output directory {output} exists and is not empty")
    try:
        output.relative_to(suite_root)
    except ValueError:
        pass
    else:
        raise ExportError("output directory must not be inside the suite")

    plans = plan_suites(suite_root, cases)
    selected, excluded = collect_export_files(
        suite_root, plans, include_baselines=include_baselines
    )

    lfs = sorted(rel for rel, src in selected.items() if _is_lfs_pointer(src))
    if lfs:
        raise ExportError(f"git-lfs pointers, not content (run git lfs pull): {lfs}")

    output.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        target = output / plan.rel / "suite.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if cases:
            # Subset export: the comment banners cannot survive a YAML round
            # trip, which is acceptable for a preview/test package.
            filtered = _filter_suite_doc(plan.doc, set(plan.cases))
            target.write_text(
                "# Subset export of suite.yaml by tools/export_hf_package.py "
                f"(--cases {','.join(plan.cases)}); comments stripped.\n"
                + yaml.safe_dump(filtered, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(plan.root / "suite.yaml", target)
    for rel, source in selected.items():
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            rel = path.relative_to(output).as_posix()
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    schema = collect_schema_versions(output, plans, files)
    main = plans[0]
    manifest = {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "exporter": {"name": EXPORTER_NAME, "version": EXPORTER_VERSION},
        "dataset": _dataset_identity(main, schema),
        "evaluator_code": git_head(),
        "schema_versions": schema,
        "suites": [
            {
                "path": plan.rel,
                "suite_name": plan.doc.get("suite_name"),
                "version": plan.doc.get("version"),
                "cases": plan.cases,
                "dimensions": {
                    name: len([c for c in (spec.get("cases") or []) if c in plan.cases])
                    for name, spec in (plan.doc.get("dimensions") or {}).items()
                    if any(c in plan.cases for c in (spec.get("cases") or []))
                },
            }
            for plan in plans
        ],
        "subset_cases": sorted(cases) if cases else None,
        "excluded": excluded,
        "totals": {
            "files": len(files),
            "bytes": sum(row["bytes"] for row in files.values()),
        },
        "files": files,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as stream:
            return stream.read(len(_LFS_HEADER)) == _LFS_HEADER
    except OSError:
        return False


# --------------------------------------------------------------------------
# verify: self-containment scan
# --------------------------------------------------------------------------

# (kind, regex). Order matters only for reporting.
_ABS_PATH = re.compile(r"(?<![\w.:/~\\-])/(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]*")
_WIN_PATH = re.compile(r"(?<![\w])[A-Za-z]:\\[^\s\"']+")
SCAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("absolute_path", _ABS_PATH),
    ("absolute_path", _WIN_PATH),
    ("home_path", re.compile(r"(?<![\w/])~/[^\s\"'`]*")),
    ("parent_escape", re.compile(r"(?<![\w.])\.\./[^\s\"'`]*")),
    ("runs_dir", re.compile(r"(?<![\w./-])runs/[^\s\"'`]*")),
    ("modal", re.compile(r"modal://[^\s\"'`]*", re.IGNORECASE)),
    ("modal", re.compile(r"\bmodal volumes?\b[^\n]{0,40}", re.IGNORECASE)),
    ("modal_volume_name", re.compile(r"\bWMEval-[a-z0-9]+(?:-[a-z0-9]+)*\b")),
    ("private_repo", re.compile(r"\b[A-Za-z0-9_-]*Private\b(?:/[^\s\"'`]*)?")),
)

# A string under one of these keys is documentation, not a dependency.
PROVENANCE_KEYS = frozenset(
    {
        "description", "desc", "note", "notes", "comment", "comments",
        "provenance", "source", "sources", "source_note", "origin", "command",
        "cmd", "cli", "argv", "args", "generated_by", "generator", "history",
        "rationale", "reason", "remarks", "caveat", "caveats", "summary",
        "readme", "explanation", "documentation", "doc", "docs", "todo",
    }
)
_PROVENANCE_SUFFIXES = ("_note", "_notes", "_description", "_provenance",
                        "_source", "_comment", "_command", "_reason")

# Keys whose value is a location the evaluator *writes* to (relative to the
# caller's working directory), not an input the package must contain.
RUNTIME_CACHE_KEY = re.compile(r"(?:^|_)(cache_dir|cache_root|output_dir|out_dir|cache)$")

_PATHLIKE_SUFFIXES = frozenset(
    {".json", ".jsonl", ".yaml", ".yml", ".npy", ".npz", ".mp4", ".png",
     ".jpg", ".jpeg", ".txt", ".md", ".csv", ".sha256", ".pt", ".pth",
     ".safetensors", ".zip", ".py", ".mov", ".avi", ".webm", ".gif"}
)


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    kind: str
    match: str
    category: str  # load_bearing | record_reference | runtime_cache | provenance
    context: str  # key path, "comment" or "prose"


def _is_provenance_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in PROVENANCE_KEYS or lowered.endswith(_PROVENANCE_SUFFIXES)


def _walk_strings(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_strings(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_strings(value, path + (f"[{index}]",))
    elif isinstance(node, str):
        yield path, node


def _leaf_key(path: tuple[str, ...]) -> str:
    for part in reversed(path):
        if not part.startswith("["):
            return part
    return ""


def file_role(rel: str) -> str:
    """``evaluator_input`` for files the scorer reads (suite.yaml, case.yaml
    and everything inside a case bundle); ``record`` for everything else
    (frozen baseline scores, generation receipts, human-alignment data)."""
    parts = PurePosixPath(rel).parts
    if parts and parts[-1] in ("suite.yaml", "case.yaml"):
        return "evaluator_input"
    return "evaluator_input" if "cases" in parts[:-1] else "record"


def _classify_value(path: tuple[str, ...], value: str, role: str) -> tuple[str, str]:
    keypath = ".".join(path) or "<root>"
    if any(_is_provenance_key(part) for part in path if not part.startswith("[")):
        return "provenance", keypath
    if len(value.split()) > 1:
        return "provenance", keypath + " (prose)"
    if RUNTIME_CACHE_KEY.search(_leaf_key(path)):
        return "runtime_cache", keypath
    if role == "record":
        return "record_reference", keypath
    return "load_bearing", keypath


_CATEGORY_RANK = ("load_bearing", "record_reference", "runtime_cache", "provenance")


def _yaml_comment_start(line: str) -> int | None:
    """Index of a YAML comment '#' on this line (outside quotes), if any."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return index
    return None


def _parse_structured(path: Path, text: str) -> list[Any] | None:
    try:
        if path.suffix in (".yaml", ".yml"):
            return [d for d in yaml.safe_load_all(text)]
        if path.suffix == ".json":
            return [json.loads(text)]
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (yaml.YAMLError, ValueError):
        return None
    return None


def _on_line(value: str, line: str) -> bool:
    """Does (part of) this string value appear on this source line?"""
    if value in line or json.dumps(value)[1:-1] in line:
        return True
    # folded/plain multi-line YAML scalar: the line holds a fragment of it
    fragment = line.strip()
    if ":" in fragment and not fragment.startswith("-"):
        key, _, rest = fragment.partition(":")
        if key and " " not in key.strip():
            fragment = rest.strip()
    fragment = fragment.strip("'\"").lstrip("- ").strip()
    return bool(fragment) and " ".join(fragment.split()) in " ".join(value.split())


def scan_text_file(package_root: Path, rel: str) -> list[Hit]:
    path = package_root / rel
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    suffix = path.suffix
    role = file_role(rel)
    docs = _parse_structured(path, text) if suffix in (".yaml", ".yml", ".json", ".jsonl") else None
    strings: list[tuple[tuple[str, ...], str]] = []
    if docs is not None:
        for index, doc in enumerate(docs):
            prefix = (f"[{index}]",) if suffix == ".jsonl" or len(docs) > 1 else ()
            strings.extend(_walk_strings(doc, prefix))

    hits: list[Hit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        comment_at = None
        if suffix in (".yaml", ".yml"):
            comment_at = _yaml_comment_start(line)
        elif suffix == ".txt":
            stripped = line.lstrip()
            if stripped.startswith("#"):
                comment_at = len(line) - len(stripped)
        spans: list[tuple[int, int]] = []
        for kind, pattern in SCAN_PATTERNS:
            for match in pattern.finditer(line):
                token = match.group(0).rstrip(".,;:)]}>'\"`")
                start, end = match.start(), match.start() + len(token)
                if not token or any(s <= start and end <= e for s, e in spans):
                    continue  # same text already reported (e.g. inside a path)
                spans.append((start, end))
                if comment_at is not None and start >= comment_at:
                    category, context = "provenance", "comment"
                elif suffix == ".md":
                    category, context = "provenance", "prose"
                elif suffix == ".txt":
                    category = "load_bearing" if role == "evaluator_input" else "record_reference"
                    context = "text"
                elif docs is None:
                    category = "load_bearing" if role == "evaluator_input" else "record_reference"
                    context = "unparsed"
                else:
                    owners = [(p, v) for p, v in strings if token in v and _on_line(v, line)]
                    owners = owners or [(p, v) for p, v in strings if token in v]
                    if not owners:
                        # a match spanning a folded-scalar line break
                        category, context = "provenance", "unattributed"
                    else:
                        category, context = min(
                            (_classify_value(p, v, role) for p, v in owners),
                            key=lambda cv: _CATEGORY_RANK.index(cv[0]),
                        )
                hits.append(Hit(rel, lineno, kind, token, category, context))
    return hits


def _looks_like_relative_path(value: str) -> bool:
    if not value or any(ch.isspace() for ch in value) or "://" in value:
        return False
    if value.startswith(("/", "~")) or "/" not in value:
        return False
    return PurePosixPath(value).suffix.lower() in _PATHLIKE_SUFFIXES or value.endswith("/")


def unresolved_relative_paths(
    package_root: Path, rel: str, case_roots: list[Path]
) -> list[Hit]:
    """Path-valued fields whose relative path resolves to nothing in-package.

    A value may be relative to any directory between its file and the package
    root (case-relative paths inside reference/artifacts/*.json, manifest-
    relative paths in baselines/...); suite.yaml values may also be relative
    to each case root (e.g. ``rubric_dir``).
    """
    path = package_root / rel
    if path.suffix not in (".yaml", ".yml", ".json", ".jsonl"):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    docs = _parse_structured(path, text)
    if docs is None:
        return []
    root_resolved = package_root.resolve()
    bases = [path.parent, *[p for p in path.parent.parents if p == package_root or package_root in p.parents]]
    if path.name == "suite.yaml":
        bases += [r for r in case_roots if path.parent in r.parents]
    role = file_role(rel)
    hits = []
    lines = text.splitlines()
    reported: set[str] = set()
    for keypath, value in (item for doc in docs for item in _walk_strings(doc)):
        if value in reported or not _looks_like_relative_path(value):
            continue
        category, context = _classify_value(keypath, value, role)
        if category not in ("load_bearing", "record_reference"):
            continue
        resolved_inside = False
        for base in bases:
            candidate = (base / value).resolve()
            try:
                candidate.relative_to(root_resolved)
            except ValueError:
                continue
            if candidate.exists():
                resolved_inside = True
                break
        if not resolved_inside:
            reported.add(value)
            lineno = next((i for i, l in enumerate(lines, 1) if value in l), 0)
            hits.append(Hit(rel, lineno, "unresolved_relative_path", value, category, context))
    return hits


def scan_package(package_root: Path, files: Iterable[str]) -> list[Hit]:
    files = sorted(files)
    case_roots = sorted({
        (package_root / f).parent for f in files if PurePosixPath(f).name == "case.yaml"
    })
    hits: list[Hit] = []
    for rel in files:
        if PurePosixPath(rel).suffix.lower() not in TEXT_SUFFIXES:
            continue
        found = scan_text_file(package_root, rel)
        found += unresolved_relative_paths(package_root, rel, case_roots)
        hits.extend(sorted(set(found), key=lambda h: (h.line, h.kind, h.match)))
    return hits


# --------------------------------------------------------------------------
# verify: integrity + validator
# --------------------------------------------------------------------------


@dataclass
class VerifyReport:
    package: str
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    mismatched: list[dict[str, Any]] = field(default_factory=list)
    unsafe_entries: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    validator: list[dict[str, Any]] = field(default_factory=list)

    def by_category(self, category: str) -> list[Hit]:
        return [h for h in self.hits if h.category == category]

    @property
    def integrity_ok(self) -> bool:
        return not (self.missing or self.extra or self.mismatched or self.unsafe_entries)

    @property
    def validator_ok(self) -> bool:
        return all(row["returncode"] == 0 for row in self.validator)

    def ok(self, *, strict: bool = False) -> bool:
        if not (self.integrity_ok and self.validator_ok):
            return False
        if self.by_category("load_bearing"):
            return False
        return not (strict and self.by_category("provenance"))

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary"] = {
            "integrity_ok": self.integrity_ok,
            "validator_ok": self.validator_ok,
            "hits": dict(Counter(h.category for h in self.hits)),
        }
        return data


def run_validator(package_root: Path, suite_rel: str) -> dict[str, Any]:
    suite_dir = (package_root / suite_rel).resolve()
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_wmeval_lite.py"),
        "validate",
        "--suite",
        str(suite_dir),
    ]
    # cwd = the package, so nothing can silently resolve against the repo's
    # own eval_sets/ through a cwd-relative path.
    result = subprocess.run(
        command, cwd=package_root, capture_output=True, text=True, check=False
    )
    return {
        "suite": suite_rel,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-3:],
        "stderr_tail": result.stderr.strip().splitlines()[-20:],
    }


def verify_package(package: str | Path, *, run_validators: bool = True) -> VerifyReport:
    package_root = Path(package).resolve()
    manifest_path = package_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ExportError(f"no {MANIFEST_NAME} in {package_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = manifest.get("files") or {}
    report = VerifyReport(package=str(package_root))

    on_disk = set()
    for path in _walk_files(package_root):
        rel = PurePosixPath(path.relative_to(package_root).as_posix())
        if str(rel) == MANIFEST_NAME or is_cache_path(rel):
            continue
        on_disk.add(str(rel))

    for rel, row in sorted(entries.items()):
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            report.unsafe_entries.append(rel)
            continue
        path = package_root / rel
        if rel not in on_disk:
            report.missing.append(rel)
            continue
        if path.is_symlink():
            report.unsafe_entries.append(rel + " (symlink)")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != row.get("bytes") or digest != row.get("sha256"):
            report.mismatched.append(
                {"file": rel, "expected_sha256": row.get("sha256"),
                 "actual_sha256": digest, "expected_bytes": row.get("bytes"),
                 "actual_bytes": size}
            )
    report.extra = sorted(on_disk - set(entries))

    report.hits = scan_package(package_root, sorted(on_disk))

    if run_validators:
        for suite in manifest.get("suites") or [{"path": "."}]:
            report.validator.append(run_validator(package_root, suite["path"]))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{count} B"


def _print_hits(title: str, hits: list[Hit]) -> None:
    print(f"\n{title}: {len(hits)}")
    for hit in hits:
        print(f"  {hit.file}:{hit.line}: [{hit.kind}] {hit.match}  ({hit.context})")


def _cmd_export(args: argparse.Namespace) -> int:
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    try:
        manifest = export_package(
            args.suite, args.output, cases=cases,
            include_baselines=not args.no_baselines,
        )
    except ExportError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    totals = manifest["totals"]
    print(f"exported {totals['files']} files, {_human_bytes(totals['bytes'])} "
          f"to {Path(args.output).resolve()}")
    for suite in manifest["suites"]:
        print(f"  suite {suite['path']}: {len(suite['cases'])} case(s), "
              f"{len(suite['dimensions'])} dimension(s)")
    excluded = manifest["excluded"]
    if excluded["unlisted_case_dirs"]:
        print("  skipped case dirs not listed by any suite: "
              + ", ".join(excluded["unlisted_case_dirs"]))
    if excluded["suite_root_files"]:
        print("  skipped suite-root entries: " + ", ".join(excluded["suite_root_files"]))
    if excluded["git_ignored"]:
        print(f"  skipped {len(excluded['git_ignored'])} git-ignored file(s)")
    code = manifest["evaluator_code"]
    print(f"  evaluator code revision {code['revision']}"
          + (" (DIRTY working tree)" if code.get("dirty") else ""))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_package(args.package, run_validators=not args.no_validate)
    except ExportError as exc:
        print(f"verify failed: {exc}", file=sys.stderr)
        return 1
    print(f"package: {report.package}")
    print(f"integrity: missing={len(report.missing)} extra={len(report.extra)} "
          f"mismatched={len(report.mismatched)} unsafe={len(report.unsafe_entries)}")
    for label, rows in (("missing", report.missing), ("extra", report.extra),
                        ("unsafe", report.unsafe_entries)):
        for rel in rows:
            print(f"  {label}: {rel}")
    for row in report.mismatched:
        print(f"  mismatched: {row['file']}")

    sections = (
        ("load_bearing", "load-bearing out-of-package references (FAIL)"),
        ("record_reference", "record files pointing outside the package "
         "(baselines/receipts; not read when scoring)"),
        ("runtime_cache", "runtime cache/output locations (cwd-relative, written not read)"),
        ("provenance", "provenance-only mentions (comments/descriptions/prose)"),
    )
    for category, title in sections:
        hits = report.by_category(category)
        if args.brief and category in ("record_reference", "provenance"):
            print(f"\n{title}: {len(hits)}")
            for (file, kind), count in sorted(Counter((h.file, h.kind) for h in hits).items()):
                print(f"  {file}: {kind} x{count}")
        else:
            _print_hits(title, hits)

    for row in report.validator:
        status = "ok" if row["returncode"] == 0 else "FAIL"
        print(f"\nvalidator [{status}] suite {row['suite']}: "
              + " | ".join(row["stdout_tail"][-1:]))
        if row["returncode"] != 0:
            for line in row["stderr_tail"]:
                print(f"    {line}")
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_json(), indent=2) + "\n")
    ok = report.ok(strict=args.strict)
    print("\nverify: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_hf_package",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export a self-contained package")
    export_parser.add_argument("--suite", default="eval_sets/wmeval_lite_v1")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument(
        "--cases", default=None,
        help="Comma-separated case ids: export only these (suite.yaml is "
             "filtered and loses its comments)",
    )
    export_parser.add_argument("--no-baselines", action="store_true",
                               help="Leave the suite's baselines/ directory out")
    export_parser.set_defaults(func=_cmd_export)

    verify_parser = sub.add_parser("verify", help="Verify an exported package")
    verify_parser.add_argument("package")
    verify_parser.add_argument("--no-validate", action="store_true",
                               help="Skip running the suite validator")
    verify_parser.add_argument("--strict", action="store_true",
                               help="Also fail on provenance-only mentions")
    verify_parser.add_argument("--brief", action="store_true",
                               help="Collapse record/provenance hits to per-file counts")
    verify_parser.add_argument("--report", default=None,
                               help="Write the full report as JSON to this path")
    verify_parser.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
