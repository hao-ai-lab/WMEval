#!/usr/bin/env python3
"""Fill ``evaluator_lock.dependency_versions`` for every live case of a Lite suite.

For each case listed by at least one dimension in ``suite.yaml`` the lock
records, from the interpreter running this tool (``importlib.metadata``):

* ``python`` -- the exact interpreter version, and ``platform``;
* one ``<distribution>: <version>`` entry per third-party package that the
  case's dimensions import on their scoring path (see ``module_requirements``);
* one ``asset:<name>: <pin>`` entry per evaluator asset of
  ``experiments/protocol/evaluator_assets.lock.json`` the dimensions load
  (the pin is the weight sha256, or the source revision for vendored code);
* ``lock_source`` -- marks the mapping as owned by this tool.

The package lists are derived by hand from the module code, following the
code path a dimension actually executes with the suite's metric_config and
the case's module_params, rather than every import the static closure can
reach.  Two consequences, both deliberate:

* ``import evaluation.modules.<x>`` runs ``evaluation/modules/__init__.py``,
  which imports torch/torchvision/lpips for every registry module.  Those
  incidental imports are only listed when the dimension computes with them.
* Config-gated branches are honoured: ``openpiv`` only when
  ``liquid_flow_estimator == 'openpiv'``; transformers/Pillow and the
  relative Depth Anything model for control_motion only when
  ``synthetic_use_depth``; the openai SDK for lpips only for the causal VLM
  sampling modes; depth models for kinematics only with ``kin_depth_lift``.

A non-empty lock that this tool did not write (no ``lock_source`` entry, e.g.
a hand-written hardware-pinned lock) is left untouched unless ``--force`` is
passed.  The tool is idempotent: a second run
in the same environment changes nothing.  Changed cases are re-authored with
``tools/build_lite_case.py`` (checksums + evaluator fingerprint) unless
``--no-rebuild`` is given.

Usage::

    python tools/lock_case_dependencies.py --suite lite
    python tools/lock_case_dependencies.py --check     # exit 1 if any lock is stale
    python tools/lock_case_dependencies.py --explain   # per-dimension requirements
"""
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCK_SOURCE = "tools/lock_case_dependencies.py"
ASSETS_LOCK_PATH = REPO_ROOT / "experiments" / "protocol" / "evaluator_assets.lock.json"
ASSET_PREFIX = "asset:"
META_KEYS = ("python", "platform", "lock_source")

CAUSAL_LPIPS_SAMPLING = frozenset(
    {"causal_vlm_collision", "causal", "vlm_collision", "causal_collision"}
)
DEPTH_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SAM2_SMALL = "facebook/sam2.1-hiera-small"


@dataclass
class Requirements:
    """Import names (top-level modules) and evaluator assets a code path uses."""

    imports: set[str] = field(default_factory=set)
    # Each tuple is "any one of these import names" (first installed wins).
    alternatives: set[tuple[str, ...]] = field(default_factory=set)
    assets: set[str] = field(default_factory=set)

    def update(self, other: "Requirements") -> "Requirements":
        self.imports |= other.imports
        self.alternatives |= other.alternatives
        self.assets |= other.assets
        return self


def _req(imports: Iterable[str] = (), assets: Iterable[str] = (), alternatives=()) -> Requirements:
    return Requirements(set(imports), set(alternatives), set(assets))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _depth_assets(override: Any) -> set[str]:
    """Depth checkpoints a run may load: the override, else both auto-select candidates."""
    if not override:
        return {"depth_anything_metric_outdoor", "depth_anything_metric_indoor"}
    return {
        DEPTH_OUTDOOR: {"depth_anything_metric_outdoor"},
        DEPTH_INDOOR: {"depth_anything_metric_indoor"},
    }.get(str(override), set())


# Packages the registry harness itself uses on every scored row: candidate and
# reference decoding in evaluation/video_io.py.
REGISTRY_HARNESS = _req({"numpy", "cv2"})
# vlm_consistency evaluator: frame_sampler (cv2/numpy), schemas (pydantic),
# provider clients (requests -- no vendor SDK).
VLM_CONSISTENCY = _req({"numpy", "cv2", "pydantic", "requests"})


def module_requirements(module: str, config: Mapping[str, Any]) -> Requirements:
    """Third-party imports + evaluator assets of one registry module's scoring path."""
    if module == "control_motion":
        # control_motion.py + flow_utils.py (torchvision RAFT large) + the
        # vendored PTLFlow synthetic-flow generator (numpy/torch).
        req = _req({"numpy", "cv2", "torch", "torchvision"}, {"raft_large", "ptlflow_synthetic"})
        if _truthy(config.get("synthetic_use_depth", False)):
            req.update(_req({"transformers", "PIL"}, {"depth_anything_relative_small"}))
        return req
    if module == "lpips":
        # lpips.py: lpips package (VGG via torchvision) on torch.
        req = _req({"numpy", "cv2", "torch", "torchvision", "lpips"}, {"lpips_vgg"})
        sampling = str(config.get("lpips_sampling", "temporal")).strip().lower()
        if sampling in CAUSAL_LPIPS_SAMPLING:
            # lpips_causal_vlm.py: RAFT flow proposals + the openai SDK judge.
            req.update(_req({"openai"}, {"raft_large"}))
        return req
    if module in {"environment_object_consistency"}:
        # GroundingDINO (transformers, PIL input), SAM2 tracking, DINOv2 via
        # torch.hub, transformers depth pipeline when Unity GT is bound.
        req = _req(
            {"numpy", "cv2", "torch", "transformers", "sam2", "PIL"},
            {"grounding_dino", "sam2", "dinov2"},
        )
        if config.get("eoc_unity_object_transforms_json") and config.get("eoc_unity_poses_npy"):
            req.assets |= _depth_assets(config.get("eoc_depth_model"))
        return req
    if module == "kinematics_mechanics":
        # GroundingDINO detection + ByteTrack (cjm-byte-track, else bytetracker).
        req = _req(
            {"numpy", "cv2", "torch", "transformers", "PIL"},
            {"grounding_dino"},
            alternatives={("cjm_byte_track", "bytetracker")},
        )
        if _truthy(config.get("kin_depth_lift", False)):
            req.assets |= _depth_assets(config.get("kin_depth_model"))
        return req
    if module in {
        "liquid_velocity",
        "liquid_volume_conservation",
        "liquid_surface_orientation",
        "slosh_spectrum",
    }:
        # _liquid_common.py: GroundingDINO box -> SAM2 mask propagation.
        req = _req({"numpy", "cv2", "torch", "transformers", "sam2", "PIL"}, {"grounding_dino"})
        if str(config.get("liquid_sam2_model", SAM2_SMALL)) == SAM2_SMALL:
            req.assets.add("sam2_small")
        if module == "liquid_velocity":
            estimator = str(config.get("liquid_flow_estimator", "raft")).strip().lower()
            if estimator == "openpiv":
                req.update(_req({"openpiv"}, {"openpiv"}))
            else:
                req.update(_req({"torchvision"}, {"raft_large"}))
        return req
    if module == "causal_step":
        # Frame extraction (cv2/numpy); vendored imports/CausalStep is stdlib
        # only; the OpenAI judge encodes frames with Pillow.
        req = _req({"numpy", "cv2"})
        if str(config.get("causal_step_judge", "openai")).strip().lower() == "openai":
            req.imports |= {"openai", "PIL"}
        return req
    if module == "shadow_consistency":
        return _req(
            {"numpy", "cv2", "torch", "transformers", "sam2", "PIL"}, {"grounding_dino", "sam2"}
        )
    raise KeyError(f"no dependency rule for registry module {module!r}; extend module_requirements()")


def dimension_config(dimension: Any, case: Any | None) -> dict[str, Any]:
    """dimension.metric_config < case.module_params[module] (no candidate params)."""
    merged: dict[str, Any] = dict(dimension.metric_config)
    if case is not None:
        for module in dimension.modules:
            params = case.module_params.get(module)
            if params:
                merged.update(dict(params))
    return merged


def dimension_requirements(dimension: Any, case: Any | None = None) -> Requirements:
    if dimension.evaluator == "vlm_consistency":
        return _req().update(VLM_CONSISTENCY)
    config = dimension_config(dimension, case)
    req = _req().update(REGISTRY_HARNESS)
    for module in dimension.modules:
        req.update(module_requirements(module, config))
    return req


def case_dimensions(suite: Any) -> dict[str, list[Any]]:
    by_case: dict[str, list[Any]] = {}
    for dimension in suite.dimensions.values():
        for case_id in dimension.cases:
            by_case.setdefault(case_id, []).append(dimension)
    return by_case


class MissingDependencyError(RuntimeError):
    pass


def _distributions_for(import_name: str, mapping: Mapping[str, list[str]]) -> list[str]:
    dists = sorted(set(mapping.get(import_name) or []), key=str.lower)
    return [d for d in dists if _version(d) is not None]


def _version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def load_assets_lock(path: Path = ASSETS_LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["assets"]


def asset_pin(entry: Mapping[str, Any]) -> str:
    """The identity recorded in a case lock for one evaluator asset."""
    for key in ("sha256", "linear_sha256", "source_revision", "revision", "version"):
        if entry.get(key):
            return str(entry[key])
    raise ValueError(f"asset entry has no pin: {entry}")


def resolve_lock(req: Requirements, assets: Mapping[str, Any]) -> dict[str, str]:
    mapping = importlib_metadata.packages_distributions()
    packages: dict[str, str] = {}
    missing: list[str] = []
    for import_name in sorted(req.imports):
        dists = _distributions_for(import_name, mapping)
        if not dists:
            missing.append(import_name)
        for dist in dists:
            packages[dist] = _version(dist)  # type: ignore[assignment]
    for options in sorted(req.alternatives):
        for import_name in options:
            dists = _distributions_for(import_name, mapping)
            if dists:
                for dist in dists:
                    packages[dist] = _version(dist)  # type: ignore[assignment]
                break
        else:
            missing.append(" | ".join(options))
    if missing:
        raise MissingDependencyError(f"not installed in {sys.executable}: {', '.join(missing)}")
    unknown = sorted(req.assets - set(assets))
    if unknown:
        raise KeyError(f"assets missing from {ASSETS_LOCK_PATH.name}: {unknown}")

    lock: dict[str, str] = {
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
    for dist in sorted(packages, key=str.lower):
        lock[dist] = packages[dist]
    for name in sorted(req.assets):
        lock[f"{ASSET_PREFIX}{name}"] = asset_pin(assets[name])
    lock["lock_source"] = LOCK_SOURCE
    return lock


def is_managed(existing: Mapping[str, Any]) -> bool:
    return not existing or existing.get("lock_source") == LOCK_SOURCE


def _explain(suite: Any) -> None:
    for dimension in suite.dimensions.values():
        req = dimension_requirements(dimension)
        alts = [" | ".join(a) for a in sorted(req.alternatives)]
        print(f"{dimension.dimension_id} ({dimension.evaluator}):")
        print(f"  imports: {', '.join(sorted(req.imports) + alts)}")
        print(f"  assets:  {', '.join(sorted(req.assets)) or '-'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="lite", help="Suite root (default: %(default)s)")
    parser.add_argument("--cases", default=None, help="Comma-separated case ids (default: every live case)")
    parser.add_argument("--check", action="store_true", help="Write nothing; exit 1 if any managed lock would change")
    parser.add_argument("--force", action="store_true", help="Overwrite non-empty locks this tool did not write")
    parser.add_argument("--no-rebuild", action="store_true", help="Do not re-run tools/build_lite_case.py on changed cases")
    parser.add_argument("--explain", action="store_true", help="Print per-dimension requirements and exit")
    args = parser.parse_args(argv)

    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(args.suite)
    if args.explain:
        _explain(suite)
        return 0
    assets = load_assets_lock()
    by_case = case_dimensions(suite)
    selected = sorted(by_case)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = sorted(wanted - set(by_case))
        if unknown:
            parser.error(f"not live cases of {args.suite}: {unknown}")
        selected = sorted(wanted)

    changed: list[Path] = []
    stale: list[str] = []
    for case_id in selected:
        case_dir = suite.cases_root / case_id
        case = load_case(case_dir, allowed_cases=suite.all_case_ids())
        req = _req()
        for dimension in by_case[case_id]:
            req.update(dimension_requirements(dimension, case))
        lock = resolve_lock(req, assets)

        case_yaml = case_dir / "case.yaml"
        raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
        existing = raw["evaluator_lock"].get("dependency_versions") or {}
        if not is_managed(existing) and not args.force:
            print(f"keep     {case_id}: existing lock not written by this tool ({len(existing)} keys)")
            continue
        if existing == lock:
            print(f"ok       {case_id}")
            continue
        stale.append(case_id)
        if args.check:
            print(f"stale    {case_id}")
            continue
        raw["evaluator_lock"]["dependency_versions"] = lock
        case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        changed.append(case_dir)
        dims = ",".join(d.dimension_id for d in by_case[case_id])
        print(f"locked   {case_id} [{dims}]: {len(lock) - len(META_KEYS)} entries")

    if args.check:
        print(f"\n{len(stale)} stale lock(s)")
        return 1 if stale else 0
    if changed and not args.no_rebuild:
        from tools.build_lite_case import main as build_main

        build_main([str(path) for path in changed])
    print(f"\n{len(changed)} case lock(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
