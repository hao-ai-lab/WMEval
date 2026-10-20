#!/usr/bin/env python3
"""Preflight a WMEval Lite run without running any evaluator or hosted judge.

Checks, for the selected dimensions (default: all) of a suite:

* python      -- interpreter version (>= 3.11) vs. the case locks' ``python`` pins
* imports     -- runtime-only imports a dimension needs that no lock lists
                 (e.g. decord for SAM2's video predictor in fluid_mechanics)
* sam2 ext    -- SAM2's CUDA extension (``sam2._C``) when lighting_shading is selected
* deps        -- every ``evaluator_lock.dependency_versions`` entry vs. what is
                 installed (packages, plus cuda/gpu/nvidia_driver and
                 ``*_source_revision`` pins where they can be read locally)
* assets      -- evaluator weights of ``experiments/tables_4_7/
                 evaluator_assets.lock.json`` the selected cases use: present in
                 the local HF / torch caches with the locked sha256 (``--online``
                 additionally probes HF for missing ones, with a HEAD request)
* device      -- availability of the requested ``--device`` (cuda / mps / cpu)
* ffmpeg      -- ffmpeg/ffprobe on PATH (required when a case normalizes candidates)
* judge keys  -- presence of the hosted-judge API key a dimension needs
                 (environment or <repo>/.env; the key is never sent anywhere)
* suite       -- suite + case bundle validation (checksums unless --skip-checksums)

Prints a PASS/WARN/FAIL table and exits 1 when anything FAILs.  Version
mismatches against a lock are WARN (scores may drift) unless ``--strict``.

Usage::

    python tools/preflight_lite.py --suite eval_sets/wmeval_lite_v1 --device mps
    python tools/preflight_lite.py --dimensions lpips,fluid_mechanics --device cuda --strict
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import lock_case_dependencies as lcd  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MIN_PYTHON = (3, 11)
# Modules a dimension imports at run time that its pinned packages do not declare.
RUNTIME_IMPORTS = {
    "fluid_mechanics": ("decord",),  # sam2's video predictor on .mp4 input
}
JUDGE_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
HARDWARE_KEYS = ("cuda", "gpu", "nvidia_driver")


@dataclass(frozen=True)
class Row:
    status: str
    check: str
    detail: str


# ---------------------------------------------------------------------------
# Individual checks (each returns rows; kept free of argparse for testing)
# ---------------------------------------------------------------------------
def _cases_label(cases: Iterable[str], limit: int = 3) -> str:
    cases = sorted(set(cases))
    head = ", ".join(cases[:limit])
    return f"{head} (+{len(cases) - limit} more)" if len(cases) > limit else head


def check_python(locks: Mapping[str, Mapping[str, Any]], strict: bool = False) -> list[Row]:
    current = platform.python_version()
    rows: list[Row] = []
    if sys.version_info[:2] < MIN_PYTHON:
        rows.append(Row(FAIL, "python", f"{current} < supported {'.'.join(map(str, MIN_PYTHON))}"))
    pins: dict[str, list[str]] = {}
    for case_id, lock in locks.items():
        if lock.get("python"):
            pins.setdefault(str(lock["python"]), []).append(case_id)
    if not pins:
        rows.append(Row(PASS if not rows else rows[0].status, "python", f"{current} (no case pins python)"))
    for pinned, cases in sorted(pins.items()):
        width = len(pinned.split("."))
        if current.split(".")[:width] == pinned.split("."):
            status = PASS
        else:
            status = FAIL if strict else WARN
        rows.append(Row(status, "python", f"installed {current}, locked {pinned} [{_cases_label(cases)}]"))
    return rows


def _installed(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_revision(dist: str) -> str | None:
    """VCS commit a distribution was installed from (PEP 610 direct_url.json).

    ``dist`` may also be the import name: SAM2 from source installs as ``SAM-2``.
    """
    names = [dist, *importlib_metadata.packages_distributions().get(dist, [])]
    for name in names:
        try:
            text = importlib_metadata.distribution(name).read_text("direct_url.json")
            break
        except importlib_metadata.PackageNotFoundError:
            continue
    else:
        return None
    if not text:
        return None
    return (json.loads(text).get("vcs_info") or {}).get("commit_id")


def _hardware() -> dict[str, str | None]:
    info: dict[str, str | None] = {"cuda": None, "gpu": None, "nvidia_driver": None}
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip().splitlines()
            info["nvidia_driver"] = out[0].strip() if out else None
        except Exception:
            pass
    return info


def check_dependencies(
    locks: Mapping[str, Mapping[str, Any]],
    strict: bool = False,
    hardware: Callable[[], Mapping[str, str | None]] = _hardware,
) -> list[Row]:
    rows: list[Row] = []
    unlocked = sorted(case_id for case_id, lock in locks.items() if not lock)
    if unlocked:
        rows.append(Row(WARN, "deps", f"no dependency lock: {_cases_label(unlocked)}"))
    platforms: dict[str, list[str]] = {}
    expected: dict[tuple[str, str], list[str]] = {}
    for case_id, lock in locks.items():
        for key, value in lock.items():
            if key in ("python", "lock_source") or key.startswith(lcd.ASSET_PREFIX):
                continue
            if key == "platform":
                platforms.setdefault(str(value), []).append(case_id)
                continue
            expected.setdefault((key, str(value)), []).append(case_id)
    mismatch = FAIL if strict else WARN
    current_platform = f"{sys.platform}-{platform.machine()}"
    for pinned, cases in sorted(platforms.items()):
        status = PASS if pinned == current_platform else WARN
        rows.append(Row(status, "platform", f"running {current_platform}, locked {pinned} [{_cases_label(cases)}]"))
    hw: Mapping[str, str | None] | None = None
    for (key, want), cases in sorted(expected.items(), key=lambda item: (item[0][0] in HARDWARE_KEYS, item[0])):
        label = f"[{_cases_label(cases)}]"
        if key in HARDWARE_KEYS:
            hw = hw if hw is not None else hardware()
            have = hw.get(key)
            status = PASS if have == want else mismatch
            rows.append(Row(status, f"hw {key}", f"have {have or 'n/a'}, locked {want} {label}"))
            continue
        if key.endswith("_source_revision"):
            dist = key[: -len("_source_revision")]
            have = _source_revision(dist)
            if have is None:
                rows.append(Row(WARN, f"dep {key}", f"cannot read {dist}'s install revision; locked {want} {label}"))
            else:
                rows.append(Row(PASS if have == want else mismatch, f"dep {key}", f"have {have}, locked {want} {label}"))
            continue
        have = _installed(key)
        if have is None:
            rows.append(Row(FAIL, f"dep {key}", f"not installed; locked {want} {label}"))
        elif have == want:
            rows.append(Row(PASS, f"dep {key}", f"{have} {label}"))
        else:
            rows.append(Row(mismatch, f"dep {key}", f"installed {have}, locked {want} {label}"))
    return rows


def check_runtime_imports(
    dimension_ids: Iterable[str],
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
) -> list[Row]:
    needed: dict[str, list[str]] = {}
    for dim in dimension_ids:
        for module in RUNTIME_IMPORTS.get(dim, ()):
            needed.setdefault(module, []).append(dim)
    rows = []
    for module, dims in sorted(needed.items()):
        status = PASS if find_spec(module) is not None else FAIL
        detail = "importable" if status == PASS else "not installed (pip install it; see requirements-gpu.lock)"
        rows.append(Row(status, f"import {module}", f"{detail} [{', '.join(sorted(dims))}]"))
    return rows


LIGHTING_DOC = "docs/evaluation_dimensions.md#reproducing-the-published-lighting-scores"


def _import_sam2_extension() -> None:
    import torch  # noqa: F401  -- sam2._C links against torch's shared libraries
    importlib.import_module("sam2._C")


def check_lighting_sam2_extension(
    dimension_ids: Iterable[str],
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    import_extension: Callable[[], None] = _import_sam2_extension,
) -> list[Row]:
    """SAM2's CUDA extension: without it SAM2 skips mask hole filling and lighting scores drift.

    pip/uv report success when the extension fails to build, so import it directly.
    """
    if "lighting_shading" not in dimension_ids or find_spec("sam2") is None:
        return []
    try:
        import_extension()
    except Exception as exc:  # missing, or built against another torch
        return [Row(WARN, "sam2 CUDA extension",
                    f"sam2._C not importable ({type(exc).__name__}): SAM2 skips mask hole filling, so "
                    f"lighting scores differ from the published ones; rebuild SAM2 as in {LIGHTING_DOC} "
                    "[lighting_shading]")]
    return [Row(PASS, "sam2 CUDA extension", "sam2._C importable [lighting_shading]")]


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def _hf_hub() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _torch_checkpoints() -> Path:
    return Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"


def _hf_downloadable(model_id: str, revision: str, filename: str) -> bool:
    import urllib.request

    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "wmeval-preflight"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status < 400
    except Exception:
        return False


def _check_file(name: str, path: Path | None, sha256: str, hashes: bool, cache: dict[Path, str], missing: Row) -> Row:
    if path is None or not path.is_file():
        return missing
    if not hashes:
        return Row(PASS, f"asset {name}", f"present (hash not checked): {path}")
    got = _sha256(path, cache)
    if got == sha256:
        return Row(PASS, f"asset {name}", f"sha256 ok: {path}")
    return Row(FAIL, f"asset {name}", f"sha256 {got[:12]}.. != locked {sha256[:12]}..: {path}")


def check_asset(name: str, entry: Mapping[str, Any], *, hashes: bool = True, online: bool = False,
                cache: dict[Path, str] | None = None) -> list[Row]:
    cache = cache if cache is not None else {}
    if "files" in entry:  # vendored source tree
        root = REPO_ROOT / entry["vendored_path"]
        bad = [f for f, pin in entry["files"].items()
               if not (root / f).is_file() or _sha256(root / f, cache) != pin["sha256"]]
        if bad:
            return [Row(FAIL, f"asset {name}", f"vendored files differ from lock: {', '.join(bad)}")]
        return [Row(PASS, f"asset {name}", f"{len(entry['files'])} vendored files match @ {entry['source_revision'][:12]}")]
    if "package" in entry:  # pip package pin
        have = _installed(entry["package"])
        if have is None:
            return [Row(FAIL, f"asset {name}", f"{entry['package']} not installed; locked {entry['version']}")]
        return [Row(PASS if have == entry["version"] else WARN, f"asset {name}",
                    f"{entry['package']} {have}, locked {entry['version']}")]
    if "model_id" in entry:  # Hugging Face snapshot
        repo = _hf_hub() / ("models--" + entry["model_id"].replace("/", "--"))
        snapshot = repo / "snapshots" / entry["revision"] / entry["weight"]
        blob = repo / "blobs" / entry["sha256"]
        path = snapshot if snapshot.is_file() else (blob if blob.is_file() else None)
        if path is None and online:
            ok = _hf_downloadable(entry["model_id"], entry["revision"], entry["weight"])
            missing = Row(WARN if ok else FAIL, f"asset {name}",
                          f"not cached; {'downloadable' if ok else 'NOT reachable'} at "
                          f"{entry['model_id']}@{entry['revision'][:8]}")
        else:
            missing = Row(WARN, f"asset {name}", f"not cached ({entry['model_id']}@{entry['revision'][:8]}); "
                          "downloads on first use (rerun with --online to probe)")
        return [_check_file(name, path, entry["sha256"], hashes, cache, missing)]
    rows: list[Row] = []
    if "linear_weight" in entry:  # lpips: linear layers ship in the pip package + torchvision VGG16
        try:
            import lpips

            linear = Path(lpips.__file__).parent / "weights" / "v0.1" / Path(entry["linear_weight"]).name
        except Exception:
            linear = None
        rows.append(_check_file(f"{name}.linear", linear, entry["linear_sha256"], hashes, cache,
                                Row(FAIL, f"asset {name}.linear", "lpips package not importable")))
        rows.append(_check_file(f"{name}.backbone", _torch_checkpoints() / entry["backbone_weight"],
                                entry["backbone_sha256"], hashes, cache,
                                Row(WARN, f"asset {name}.backbone", f"{entry['backbone_weight']} not cached; torchvision downloads it on first use")))
        return rows
    # torch.hub / torchvision checkpoint
    rows.append(_check_file(name, _torch_checkpoints() / entry["weight"], entry["sha256"], hashes, cache,
                            Row(WARN, f"asset {name}", f"{entry['weight']} not cached; torch downloads it on first use")))
    if entry.get("source", "").count("/") == 1:  # torch.hub repo, e.g. facebookresearch/dinov2
        hub_dirs = list(_torch_checkpoints().parent.glob(entry["source"].replace("/", "_") + "_*"))
        if not hub_dirs:
            rows.append(Row(WARN, f"asset {name}.source", f"torch.hub repo {entry['source']} not cached; needs github.com on first use"))
    return rows


def check_assets(locks: Mapping[str, Mapping[str, Any]], derived: Mapping[str, set[str]],
                 assets_lock: Mapping[str, Any], *, hashes: bool = True, online: bool = False) -> list[Row]:
    """Asset pins named by case locks (or derived for unlocked cases) vs. the asset lock + local caches."""
    rows: list[Row] = []
    used: dict[str, list[str]] = {}
    for case_id, lock in locks.items():
        names = {k[len(lcd.ASSET_PREFIX):]: str(v) for k, v in lock.items() if k.startswith(lcd.ASSET_PREFIX)}
        for name in derived.get(case_id, set()) - set(names):
            names[name] = ""
        for name, pin in names.items():
            used.setdefault(name, []).append(case_id)
            entry = assets_lock.get(name)
            if entry is None:
                rows.append(Row(FAIL, f"asset {name}", f"{case_id} references an asset missing from the asset lock"))
            elif pin and pin != lcd.asset_pin(entry):
                rows.append(Row(FAIL, f"asset {name}", f"{case_id} locks {pin[:12]}.., asset lock says {lcd.asset_pin(entry)[:12]}.. (re-run lock_case_dependencies.py)"))
    cache: dict[Path, str] = {}
    for name in sorted(used):
        if name in assets_lock:
            rows.extend(check_asset(name, assets_lock[name], hashes=hashes, online=online, cache=cache))
    return rows


def check_device(requested: str | None) -> list[Row]:
    try:
        import torch
    except Exception as exc:
        return [Row(FAIL, "device", f"torch not importable: {exc}")]
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    available = ", ".join(n for n, ok in (("cuda", cuda), ("mps", mps), ("cpu", True)) if ok)
    want = (requested or "auto").lower()
    if want == "auto":
        if cuda:
            return [Row(PASS, "device", f"auto -> cuda ({torch.cuda.get_device_name(0)}); available: {available}")]
        hint = "; pass --device mps to use the Apple GPU" if mps else ""
        return [Row(WARN, "device", f"auto -> cpu (runner default picks cuda or cpu){hint}; available: {available}")]
    kind = want.split(":", 1)[0]
    ok = {"cuda": cuda, "mps": mps, "cpu": True}.get(kind)
    if ok is None:
        return [Row(FAIL, "device", f"unknown device {requested!r}; available: {available}")]
    detail = f"{requested} {'available' if ok else 'NOT available'}; available: {available}"
    if ok and kind == "cuda":
        detail += f" ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})"
    return [Row(PASS if ok else FAIL, "device", detail)]


def check_ffmpeg(needs_normalization: list[str], which: Callable[[str], str | None] = shutil.which) -> list[Row]:
    rows: list[Row] = []
    for tool in ("ffmpeg", "ffprobe"):
        path = which(tool)
        if path:
            try:
                first = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20).stdout.splitlines()
                version = first[0] if first else path
            except Exception:
                version = path
            rows.append(Row(PASS, tool, version))
        elif needs_normalization:
            rows.append(Row(FAIL, tool, f"not on PATH; needed to normalize {_cases_label(needs_normalization)}"))
        else:
            rows.append(Row(WARN, tool, "not on PATH (no selected case normalizes candidates)"))
    return rows


def judge_requirements(suite: Any, dimension_ids: Iterable[str], cases: Mapping[str, Any]) -> dict[str, list[str]]:
    """{env var: [dimension ids]} for hosted judges the selection would call."""
    needs: dict[str, list[str]] = {}
    for dimension_id in dimension_ids:
        dimension = suite.dimensions[dimension_id]
        if dimension.evaluator == "vlm_consistency":
            provider = str(dimension.vlm.get("provider", "openai"))
            needs.setdefault(JUDGE_KEYS.get(provider, f"<{provider} key>"), []).append(dimension_id)
            continue
        for case_id in dimension.cases:
            req = lcd.dimension_requirements(dimension, cases.get(case_id))
            if "openai" in req.imports:
                needs.setdefault(JUDGE_KEYS["openai"], []).append(dimension_id)
                break
    return needs


def check_judge_keys(needs: Mapping[str, list[str]], environ: Mapping[str, str] = os.environ) -> list[Row]:
    rows = []
    for var, dims in sorted(needs.items()):
        if environ.get(var, "").strip():
            rows.append(Row(PASS, f"judge {var}", f"set (presence only, not validated) for {', '.join(dims)}"))
        else:
            rows.append(Row(FAIL, f"judge {var}", f"missing; required by {', '.join(dims)}"))
    return rows


def check_suite(suite: Any, case_ids: Iterable[str], verify_checksums: bool) -> tuple[list[Row], dict[str, Any]]:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.validation import CaseValidationError, validate_case_bundle

    loaded: dict[str, Any] = {}
    failures: list[str] = []
    for case_id in sorted(case_ids):
        try:
            case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
            validate_case_bundle(case, verify_checksums=verify_checksums)
        except (ValueError, CaseValidationError) as exc:
            failures.append(f"{case_id}: {exc}")
            continue
        loaded[case_id] = case
    mode = "with checksums" if verify_checksums else "checksums skipped"
    rows = [Row(FAIL, "suite", failure) for failure in failures]
    if not failures:
        rows.append(Row(PASS, "suite", f"{len(loaded)} case bundle(s) valid ({mode})"))
    return rows, loaded


# ---------------------------------------------------------------------------
def render(rows: list[Row]) -> str:
    width = max([len("CHECK")] + [len(r.check) for r in rows])
    lines = [f"{'STATUS':<6}  {'CHECK':<{width}}  DETAIL", f"{'-' * 6}  {'-' * width}  {'-' * 40}"]
    lines += [f"{r.status:<6}  {r.check:<{width}}  {r.detail}" for r in rows]
    counts = {s: sum(r.status == s for r in rows) for s in (PASS, WARN, FAIL)}
    lines.append(f"\n{counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="eval_sets/wmeval_lite_v1", help="Suite root (default: %(default)s)")
    parser.add_argument("--dimensions", default=None, help="Comma-separated dimension ids (default: all)")
    parser.add_argument("--device", default="auto", help="cuda | cuda:N | mps | cpu | auto (default: %(default)s)")
    parser.add_argument("--strict", action="store_true", help="Treat locked-version mismatches as failures")
    parser.add_argument("--skip-checksums", action="store_true", help="Skip reference checksum verification")
    parser.add_argument("--skip-assets", action="store_true", help="Skip evaluator asset checks")
    parser.add_argument("--no-hash", action="store_true", help="Check asset presence only, not sha256")
    parser.add_argument("--online", action="store_true", help="HEAD-probe Hugging Face for assets missing from the cache")
    parser.add_argument("--json", default=None, help="Also write the rows as JSON to this path")
    args = parser.parse_args(argv)

    from evaluation.lite.suite_loader import load_suite
    from tools.run_wmeval_lite import _load_dotenv

    rows: list[Row] = []
    try:
        suite = load_suite(args.suite)
    except Exception as exc:
        rows.append(Row(FAIL, "suite", f"cannot load {args.suite}: {exc}"))
        print(render(rows))
        return 1
    dimension_ids = list(suite.dimensions)
    if args.dimensions:
        dimension_ids = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        unknown = [d for d in dimension_ids if d not in suite.dimensions]
        if unknown:
            parser.error(f"unknown dimension(s) {unknown}; available: {sorted(suite.dimensions)}")
    case_ids = sorted({c for d in dimension_ids for c in suite.dimensions[d].cases})

    suite_rows, cases = check_suite(suite, case_ids, verify_checksums=not args.skip_checksums)
    locks = {cid: dict(case.evaluator_lock.get("dependency_versions") or {}) for cid, case in cases.items()}

    rows += check_python(locks, strict=args.strict)
    rows += check_dependencies(locks, strict=args.strict)
    rows += check_runtime_imports(dimension_ids)
    rows += check_lighting_sam2_extension(dimension_ids)
    if not args.skip_assets:
        derived: dict[str, set[str]] = {}
        for d in dimension_ids:
            for cid in suite.dimensions[d].cases:
                if cid in cases and not locks[cid]:
                    derived.setdefault(cid, set()).update(
                        lcd.dimension_requirements(suite.dimensions[d], cases[cid]).assets)
        rows += check_assets(locks, derived, lcd.load_assets_lock(), hashes=not args.no_hash, online=args.online)
    rows += check_device(args.device)
    needs_norm = [cid for cid, case in cases.items()
                  if str(case.protocol.get("candidate_normalization", "none")) != "none"]
    rows += check_ffmpeg(needs_norm)
    _load_dotenv(REPO_ROOT)
    rows += check_judge_keys(judge_requirements(suite, dimension_ids, cases))
    rows += suite_rows

    print(f"WMEval Lite preflight: {args.suite} | dimensions: {', '.join(dimension_ids)} | "
          f"{len(case_ids)} case(s) | python {sys.executable}\n")
    print(render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ for r in rows], indent=2) + "\n", encoding="utf-8")
    return 1 if any(r.status == FAIL for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
