"""The synthetic trajectory set (``test_full``) as scoreable Lite cases.

Each synthetic trajectory (``synthetic/manifest.json``, written by
``tools/dataset/generate_synthetic_trajectories.py``) is replayed in the Unity scene of
its *conditioning* core case, so it is scored exactly like that case except
for what the trajectory itself changes:

* case id = ``synthetic_id`` (``synthetic/<family>/<Scene>__<variant>/<slug>``),
  root = the trajectory directory, reference video = its own
  ``reference/reference.mp4`` and Unity ground truth = its own
  ``reference/artifacts/`` (same file conventions as a core case);
* prompt, perspective, first frame and action target come from ``meta.json``;
* protocol (candidate normalization, duration floors) and evaluator lock come
  from the conditioning case's ``case.yaml``;
* module parameters are the conditioning case's, with artifact paths resolved
  against the trajectory directory, plus the per-conditioning-case overrides in
  suite.yaml ``synthetic.module_params``;
* the dimensions it is scored on are its manifest ``dimensions`` that the suite
  defines, minus any its reference declares not applicable in
  ``reference/artifacts/dimension_applicability.json`` (e.g. physical causality
  for a trajectory whose ball never starts a chain).

Reference files are checksummed in ``synthetic/references.json`` (written by
``tools/dataset/render_synthetic_references.py publish``) and the generated trajectory
files in the manifest; :func:`validate_synthetic_case` verifies both.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .case_loader import _MODULE_PARAM_PATH_KEYS, LiteCase, _freeze, load_case
from .suite_loader import Suite

SPLIT_CORE = "core"
SPLIT_SYNTHETIC = "synthetic"
SPLIT_FULL = "full"
SPLITS = (SPLIT_CORE, SPLIT_SYNTHETIC, SPLIT_FULL)

APPLICABILITY_FILENAME = "dimension_applicability.json"
QA_DATASET = "reference/artifacts/qa_dataset.json"
JUDGE_ANSWERS = "reference/artifacts/judge_answers.json"
REFERENCE_VIDEO = "reference/reference.mp4"
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# Dimension -> reference files it cannot run without (relative to the trajectory).
_REQUIRED_FILES = {
    "physical_causality": (QA_DATASET,),
    # SAM2/RAFT on the reference ship precomputed, like the core liquid cases
    "fluid_mechanics": tuple(
        f"reference/artifacts/{name}"
        for name in ("liquid_mask.npz", "bottle_mask.npz", "raft_pooled_flow.npz")
    ),
}


class SyntheticSuiteError(ValueError):
    pass


def includes_core(split: str) -> bool:
    return split in (SPLIT_CORE, SPLIT_FULL)


def includes_synthetic(split: str) -> bool:
    return split in (SPLIT_SYNTHETIC, SPLIT_FULL)


def check_split(split: str) -> str:
    if split not in SPLITS:
        raise SyntheticSuiteError(f"split must be one of {SPLITS}, got {split!r}")
    return split


@dataclass(frozen=True)
class SyntheticTrajectory:
    """One manifest row (plus its published reference record)."""

    synthetic_id: str
    path: str
    family: str
    scene: str
    pattern: str
    conditioning_case: str
    dimensions: tuple[str, ...]
    prompt: str
    perspective: str | None
    first_frame: str | None
    action_target: str | None
    reference_status: str
    reference_quality: str | None
    files: Mapping[str, str]
    reference_files: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class SyntheticCase:
    trajectory: SyntheticTrajectory
    case: LiteCase
    # Suite dimensions this trajectory is scored on.
    dimensions: tuple[str, ...]
    # Declared dimensions it is not scored on, with the reason.
    not_applicable: Mapping[str, str]
    # Registry modules of the scored dimensions (their inputs must exist).
    modules: tuple[str, ...] = ()
    # vlm_consistency dimension -> (scene, prompt_version, rubric_dir, provider, model)
    rubric_specs: Mapping[str, tuple[str, str, str, str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def row_fields(self) -> dict[str, Any]:
        """Extra per-case JSONL fields identifying a synthetic row."""
        t = self.trajectory
        return {
            "split": SPLIT_SYNTHETIC,
            "family": t.family,
            "conditioning_case": t.conditioning_case,
            "reference_quality": t.reference_quality,
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SyntheticSuiteError(f"cannot read {path}: {exc}") from exc


def _spec(suite: Suite):
    if suite.synthetic is None:
        raise SyntheticSuiteError(
            f"{suite.root / 'suite.yaml'} declares no `synthetic` section; "
            "this suite has no synthetic split"
        )
    return suite.synthetic


def load_trajectories(suite: Suite) -> list[SyntheticTrajectory]:
    spec = _spec(suite)
    manifest_path = suite.root / spec.manifest
    if not manifest_path.is_file():
        raise SyntheticSuiteError(
            f"{manifest_path} not found; run tools/dataset/generate_synthetic_trajectories.py"
        )
    references_path = suite.root / spec.references
    references = (
        _read_json(references_path).get("references", {})
        if references_path.is_file()
        else {}
    )
    out = []
    for row in _read_json(manifest_path).get("trajectories", []):
        ref = references.get(row["synthetic_id"]) or {}
        out.append(
            SyntheticTrajectory(
                synthetic_id=row["synthetic_id"],
                path=row["path"],
                family=row["family"],
                scene=row["scene"],
                pattern=row["pattern"],
                conditioning_case=row["conditioning_case"],
                dimensions=tuple(row.get("dimensions") or ()),
                prompt=row["prompt"],
                perspective=row.get("perspective"),
                first_frame=row.get("first_frame"),
                action_target=row.get("action_target"),
                reference_status=row.get("reference_status", "pending_unity_render"),
                reference_quality=row.get("reference_quality"),
                files=MappingProxyType(dict(row.get("files") or {})),
                reference_files=MappingProxyType(dict(ref.get("files") or {})),
            )
        )
    return out


def _resolve_module_params(
    conditioning: LiteCase,
    overrides: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for module, values in conditioning.module_params.items():
        resolved = dict(values)
        for key in _MODULE_PARAM_PATH_KEYS:
            value = resolved.get(key)
            if isinstance(value, str) and value.strip():
                # load_case resolved it against the core bundle; keep the same
                # bundle-relative path, now under the trajectory directory.
                try:
                    relative = Path(value).relative_to(conditioning.root)
                except ValueError:
                    continue
                resolved[key] = str((root / relative).resolve())
        params[module] = resolved
    for module, values in overrides.items():
        merged = params.setdefault(module, {})
        for key, value in values.items():
            if value is None:
                merged.pop(key, None)
            elif key in _MODULE_PARAM_PATH_KEYS and isinstance(value, str):
                merged[key] = str((root / value).resolve())
            else:
                merged[key] = value
    return params


def build_case(
    suite: Suite,
    trajectory: SyntheticTrajectory,
    *,
    conditioning: LiteCase | None = None,
) -> LiteCase:
    """The LiteCase a synthetic trajectory is scored as (see module docstring)."""
    spec = _spec(suite)
    if conditioning is None:
        conditioning = load_case(
            suite.cases_root / trajectory.conditioning_case,
            allowed_cases=suite.all_case_ids(),
        )
    root = (suite.root / trajectory.path).resolve()
    module_params = _resolve_module_params(
        conditioning,
        spec.module_params.get(trajectory.conditioning_case, {}),
        root,
    )
    artifacts: dict[str, Path] = {}
    qa = root / QA_DATASET
    if qa.is_file():
        artifacts["qa_dataset"] = qa
        videos = _read_json(qa).get("videos") or [{}]
        module_params.setdefault("causal_step", {})["causal_step_video_id"] = str(
            videos[0].get("video_id")
        )
        if (root / JUDGE_ANSWERS).is_file():
            artifacts["judge_answers"] = root / JUDGE_ANSWERS
    first_frame = None
    if trajectory.first_frame and trajectory.first_frame.lower().endswith(_IMAGE_SUFFIXES):
        first_frame = (suite.root / trajectory.first_frame).resolve()
    return replace(
        conditioning,
        root=root,
        case_id=trajectory.synthetic_id,
        scene=trajectory.scene,
        display_name=f"{trajectory.scene} {trajectory.family} {trajectory.pattern}",
        modules=tuple(module_params),
        reference_video=root / REFERENCE_VIDEO,
        baseline_path=None,
        checksums_path=(suite.root / spec.references).resolve(),
        prompt=trajectory.prompt,
        perspective=trajectory.perspective,
        first_frame=first_frame,
        module_params=_freeze(module_params, "module_params"),
        reference_artifacts=MappingProxyType(artifacts),
        action_target=trajectory.action_target,
        # Synthetic prompts are the conditioning scene description plus an
        # action sentence, so the scene's rubric applies; a trajectory whose
        # conditioning case has none ships its own under reference/artifacts.
        rubric_root=root if (root / "reference/artifacts/rubrics").is_dir() else conditioning.root,
    )


def _not_applicable(root: Path) -> dict[str, str]:
    path = root / "reference/artifacts" / APPLICABILITY_FILENAME
    if not path.is_file():
        return {}
    payload = _read_json(path)
    return {str(k): str(v) for k, v in (payload.get("not_applicable") or {}).items()}


def load_synthetic_cases(
    suite: Suite,
    *,
    dimensions: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
) -> dict[str, SyntheticCase]:
    """synthetic_id -> SyntheticCase for trajectories scored on ``dimensions``.

    Trajectories without a Unity reference yet are skipped (nothing to score).
    """
    wanted = set(dimensions) if dimensions is not None else set(suite.dimensions)
    wanted_families = set(families) if families is not None else None
    conditioning: dict[str, LiteCase] = {}
    out: dict[str, SyntheticCase] = {}
    for trajectory in load_trajectories(suite):
        if wanted_families is not None and trajectory.family not in wanted_families:
            continue
        if trajectory.reference_status != "unity_rendered":
            continue
        declared = [d for d in trajectory.dimensions if d in suite.dimensions and d in wanted]
        if not declared:
            continue
        cond_id = trajectory.conditioning_case
        if cond_id not in conditioning:
            conditioning[cond_id] = load_case(
                suite.cases_root / cond_id, allowed_cases=suite.all_case_ids()
            )
        case = build_case(suite, trajectory, conditioning=conditioning[cond_id])
        skipped = {d: r for d, r in _not_applicable(case.root).items() if d in declared}
        scored = tuple(d for d in declared if d not in skipped)
        rubric_specs = {
            d: (
                case.rubric_scene_name or case.scene,
                str(suite.dimensions[d].vlm["prompt_version"]),
                str((case.rubric_root or case.root) / str(suite.dimensions[d].vlm["rubric_dir"])),
                str(suite.dimensions[d].vlm["provider"]),
                str(suite.dimensions[d].vlm["model"]),
            )
            for d in scored
            if suite.dimensions[d].vlm
        }
        out[trajectory.synthetic_id] = SyntheticCase(
            trajectory=trajectory,
            case=case,
            dimensions=scored,
            not_applicable=MappingProxyType(skipped),
            modules=tuple(dict.fromkeys(m for d in scored for m in suite.dimensions[d].modules)),
            rubric_specs=MappingProxyType(rubric_specs),
        )
    return out


def missing_files(suite: Suite) -> list[str]:
    """Listed trajectory/reference files absent locally (data not downloaded)."""
    missing = []
    for t in load_trajectories(suite):
        for name in [*t.files, *t.reference_files]:
            if not (suite.root / t.path / name).is_file():
                missing.append(f"{t.path}/{name}")
    return missing


def validate_synthetic_case(entry: SyntheticCase, *, verify_checksums: bool = True) -> None:
    """Every listed file present (and matching its sha256), every artifact the
    scored dimensions need present, and the reference fully listed."""
    from .validation import CaseValidationError, _require_bundle_file, sha256_file

    t, case = entry.trajectory, entry.case
    if not t.reference_files:
        raise CaseValidationError("not listed in references.json")
    listed = {**t.files, **t.reference_files}
    for name, digest in sorted(listed.items()):
        path = case.root / name
        _require_bundle_file(path)
        if verify_checksums and sha256_file(path) != digest:
            raise CaseValidationError(f"checksum mismatch for {name}")
    on_disk = {
        p.relative_to(case.root).as_posix()
        for p in (case.root / "reference").rglob("*")
        if p.is_file()
    }
    unlisted = sorted(on_disk - set(t.reference_files))
    if unlisted:
        raise CaseValidationError(
            f"reference files not in references.json: {unlisted}"
        )
    needed: list[Path] = []
    for dimension in entry.dimensions:
        needed += [case.root / rel for rel in _REQUIRED_FILES.get(dimension, ())]
    for module in entry.modules:
        module_values = case.module_params.get(module) or {}
        needed += [
            Path(value)
            for key, value in module_values.items()
            if key in _MODULE_PARAM_PATH_KEYS and isinstance(value, str)
        ]
    for dimension in entry.dimensions:
        if dimension in entry.rubric_specs:
            from evaluation.modules.vlm_consistency.rubric_cache import find_rubric

            scene, version, rubric_dir, provider, model = entry.rubric_specs[dimension]
            if find_rubric(scene, version, rubric_dir, provider=provider, model=model) is None:
                raise CaseValidationError(
                    f"no {provider}/{model}/{version} rubric for "
                    f"{scene!r} under {rubric_dir}"
                )
    for path in needed:
        if not path.is_file():
            shown = path.relative_to(case.root) if path.is_relative_to(case.root) else path
            raise CaseValidationError(f"missing reference input {shown}")
