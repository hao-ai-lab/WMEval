"""Suite orchestration: the execution layer PR #19 deferred, absorbing the
run/aggregate skeleton of PR #16's ``run_lite_suite``.

Flow per work item (dimension, case, model, seed):
  suite.yaml → case.yaml (+bundle validation) → candidate discovery →
  candidate validation/normalization → adapter dispatch → JSONL row →
  aggregate summary.

Heavy imports (cv2-backed validation, the metric runner, ffmpeg
normalization) are deferred into the non-dry-run path so ``--dry-run`` only
needs PyYAML.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters.base import (
    STATUS_COMPLETED,
    STATUS_DRY_RUN,
    STATUS_ERROR,
    STATUS_INVALID_CANDIDATE,
    STATUS_MISSING_CANDIDATE,
    STATUS_UNSUPPORTED,
    CaseResult,
    DimensionEvaluator,
)
from .action_support import support_record, unsupported_reason
from evaluation import action_backends
from .aggregate import (
    build_aggregate,
    write_aggregate_json,
    write_summary_csv,
)
from .case_loader import LiteCase, load_case
from .candidate_discovery import (
    CandidateRow,
    discover_candidate_directory,
    load_candidate_manifest,
)
from .output import (
    AGGREGATE_CSV_FILENAME,
    AGGREGATE_JSON_FILENAME,
    PER_CASE_RESULTS_FILENAME,
    append_row,
    build_row,
    read_rows,
    row_key,
    write_rows,
)
from .suite_loader import (
    EVALUATOR_REGISTRY,
    EVALUATOR_VLM_CONSISTENCY,
    DimensionSpec,
    Suite,
    load_suite,
)

LIQUID_MODULES = frozenset(
    {
        "liquid_velocity",
        "liquid_volume_conservation",
        "liquid_surface_orientation",
        "slosh_spectrum",
    }
)

# candidate_normalization protocol tokens → normalizer. Tokens mapping to
# None are declared-but-identity (the candidate is used as-is); unknown
# tokens are an error so silent protocol drift is impossible.
NORMALIZATION_TOKENS: dict[str, str | None] = {
    "none": None,
    "beamdomino_native": None,
    "liquid_832x464_16fps": "liquid",
    # Resample to the case's reference video (fps, resolution, frame count);
    # used by the Table 4/7 cells so frame k aligns with Unity ground truth k.
    "match_reference": "reference",
    "lighting_fps12": "lighting",
    "lighting_orbit_10s_fps12": "lighting_orbit",
}


def _lighting_input_fingerprint(item: WorkItem) -> str | None:
    if item.case.protocol.get('candidate_normalization') != 'lighting_orbit_10s_fps12':
        return None
    video = item.candidate.candidate_video
    digest = None
    if video.is_file():
        with video.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    payload = {'candidate_sha256': digest, 'params': dict(item.candidate.params),
               'dimension_metric_config': dict(item.dimension.metric_config),
               'modules': item.dimension.modules}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

MODEL_PARAMS_FILENAME = "model.yaml"


class SuiteRunError(ValueError):
    pass


@dataclass(frozen=True)
class WorkItem:
    dimension: DimensionSpec
    case: LiteCase
    candidate: CandidateRow


def _load_model_params(model_root: Path) -> Mapping[str, Any]:
    """Optional per-model params (e.g. shadow_hud_preset) from model.yaml."""
    params_path = model_root / MODEL_PARAMS_FILENAME
    if not params_path.is_file():
        return {}
    import yaml

    raw = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise SuiteRunError(
            f"{params_path} must contain a mapping with string keys"
        )
    return raw


def _merge_candidate_params(
    candidate: CandidateRow, model_params: Mapping[str, Any]
) -> CandidateRow:
    if not model_params:
        return candidate
    merged = dict(model_params)
    merged.update(dict(candidate.params))
    return CandidateRow(
        model=candidate.model,
        case_id=candidate.case_id,
        seed=candidate.seed,
        candidate_video=candidate.candidate_video,
        params=merged,
    )


def _with_backend_defaults(candidate: CandidateRow) -> CandidateRow:
    """Fill params the backend registry knows, e.g. a model's real frame rate.

    Generators that write one frame per fixed action tick tag their mp4 with an
    unrelated fps; the registry records the real rate so users need not know to
    pass ``candidate_fps_override``. Explicit candidate/model params still win.
    """
    if "candidate_fps_override" in candidate.params:
        return candidate
    try:
        real_fps = action_backends.backend(candidate.model).get("real_fps")
    except KeyError:
        return candidate
    if real_fps is None:
        return candidate
    return CandidateRow(
        model=candidate.model,
        case_id=candidate.case_id,
        seed=candidate.seed,
        candidate_video=candidate.candidate_video,
        params={**candidate.params, "candidate_fps_override": float(real_fps)},
    )


def _load_cases(
    suite: Suite, case_ids: Sequence[str]
) -> dict[str, LiteCase]:
    cases: dict[str, LiteCase] = {}
    allowed = suite.all_case_ids()
    for case_id in case_ids:
        case = load_case(suite.cases_root / case_id, allowed_cases=allowed)
        if case.case_id != case_id:
            raise SuiteRunError(
                f"case directory {case_id} declares mismatched case_id "
                f"{case.case_id!r}"
            )
        cases[case_id] = case
    return cases


def _discover_candidates(
    *,
    model_root: str | Path | None,
    candidate_manifest: str | Path | None,
    allowed_cases: frozenset[str],
    suite_cases: frozenset[str] | None = None,
) -> tuple[list[CandidateRow], Mapping[str, Any]]:
    if (model_root is None) == (candidate_manifest is None):
        raise SuiteRunError(
            "exactly one of model_root or candidate_manifest is required"
        )
    if model_root is not None:
        root = Path(model_root)
        rows = discover_candidate_directory(root, allowed_cases=allowed_cases)
        return rows, _load_model_params(root)
    # Rows for suite cases outside the selected dimensions are skipped.
    rows = load_candidate_manifest(
        candidate_manifest, allowed_cases=(suite_cases or allowed_cases) | allowed_cases
    )
    return [row for row in rows if row.case_id in allowed_cases], {}


def _default_normalized_path(
    output_root: Path, candidate: CandidateRow
) -> Path:
    return (
        output_root
        / "normalized"
        / candidate.model
        / candidate.case_id
        / f"{candidate.seed}.mp4"
    )


def _normalize_candidate(
    *,
    case: LiteCase,
    candidate: CandidateRow,
    output_root: Path,
) -> Path:
    token = str(case.protocol["candidate_normalization"])
    if token not in NORMALIZATION_TOKENS:
        raise SuiteRunError(
            f"{case.case_id}: unknown candidate_normalization token {token!r}"
        )
    kind = NORMALIZATION_TOKENS[token]
    if kind is None:
        return candidate.candidate_video

    from .normalization import normalize_liquid_video, normalize_to_reference, normalize_lighting_video, normalize_lighting_window

    destination = _default_normalized_path(output_root, candidate)
    if kind == "lighting_orbit":
        mapping = candidate.params.get("lighting_time_mapping")
        if not isinstance(mapping, Mapping):
            raise SuiteRunError(f"{case.case_id}: candidate requires receipt-derived lighting_time_mapping")
        # This small artifact must be verified/rebuilt for changed receipt/input,
        # even when a previous candidate occupied the same output path.
        normalize_lighting_window(candidate.candidate_video, destination, time_mapping=dict(mapping))
        return destination
    if not destination.is_file():
        if kind == "reference":
            if case.reference_video is None:
                raise SuiteRunError(f"{case.case_id}: match_reference needs a reference video")
            normalize_to_reference(
                candidate.candidate_video, destination, reference=case.reference_video
            )
        elif kind == "lighting":
            normalize_lighting_video(candidate.candidate_video, destination)
        else:
            normalize_liquid_video(candidate.candidate_video, destination)
    return destination


def _install_liquid_artifacts_if_needed(
    *,
    dimension: DimensionSpec,
    case: LiteCase,
    installed: set[str],
) -> None:
    if not LIQUID_MODULES.intersection(dimension.modules):
        return
    if case.case_id in installed:
        return
    if not (case.artifacts_dir / "liquid_mask.npz").is_file():
        return

    from .reference_artifacts import install_liquid_reference_artifacts

    cache_root = str(dimension.metric_config.get("liquid_cache_root", "runs/liquid_cache"))
    install_liquid_reference_artifacts(case, Path(cache_root))
    installed.add(case.case_id)


def _resolve_adapter(
    dimension: DimensionSpec,
    adapters: Mapping[str, DimensionEvaluator],
) -> DimensionEvaluator:
    adapter = adapters.get(dimension.evaluator)
    if adapter is None:
        raise SuiteRunError(
            f"no adapter registered for evaluator {dimension.evaluator!r}"
        )
    return adapter


def _default_adapters(
    runner: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, DimensionEvaluator]:
    from .adapters.registry_adapter import RegistryAdapter
    from .adapters.vlm_consistency_adapter import VlmConsistencyAdapter

    return {
        EVALUATOR_REGISTRY: RegistryAdapter(runner=runner),
        EVALUATOR_VLM_CONSISTENCY: VlmConsistencyAdapter(),
    }


def run_suite(
    *,
    suite_root: str | Path,
    output_dir: str | Path,
    model_root: str | Path | None = None,
    candidate_manifest: str | Path | None = None,
    dimensions: Sequence[str] | None = None,
    device: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
    verify_checksums: bool = True,
    run_id: str = "run",
    runner: Callable[..., Mapping[str, Any]] | None = None,
    adapters: Mapping[str, DimensionEvaluator] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the requested dimensions over discovered candidates.

    Returns the aggregate summary (also written to ``output_dir`` together
    with the per-case JSONL, except in dry runs which write nothing).
    """
    emit = log or (lambda message: print(message, flush=True))
    suite = load_suite(suite_root)

    requested = tuple(dimensions) if dimensions else tuple(suite.dimensions)
    unknown = sorted(set(requested) - set(suite.dimensions))
    if unknown:
        raise SuiteRunError(f"unknown dimension(s): {unknown}")
    requested_dimensions = [suite.dimensions[dim_id] for dim_id in requested]

    case_ids = sorted(
        {case_id for dimension in requested_dimensions for case_id in dimension.cases}
    )
    cases = _load_cases(suite, case_ids)

    if not dry_run:
        from .validation import validate_case_bundle

        for case_id in case_ids:
            validate_case_bundle(cases[case_id], verify_checksums=verify_checksums)

    candidate_rows, model_params = _discover_candidates(
        model_root=model_root,
        candidate_manifest=candidate_manifest,
        allowed_cases=frozenset(case_ids),
        suite_cases=frozenset(
            case_id for dimension in suite.dimensions.values() for case_id in dimension.cases
        ),
    )
    candidate_rows = [
        _with_backend_defaults(_merge_candidate_params(row, model_params))
        for row in candidate_rows
    ]

    work_items = [
        WorkItem(dimension=dimension, case=cases[row.case_id], candidate=row)
        for dimension in requested_dimensions
        for row in candidate_rows
        if row.case_id in dimension.cases
    ]
    work_items.sort(
        key=lambda item: (
            item.dimension.dimension_id,
            item.candidate.case_id,
            item.candidate.model,
            item.candidate.seed,
        )
    )

    output_root = Path(output_dir).expanduser().resolve()
    results_path = output_root / PER_CASE_RESULTS_FILENAME

    input_fingerprints = {
        (item.dimension.dimension_id, item.candidate.case_id,
         item.candidate.model, item.candidate.seed): _lighting_input_fingerprint(item)
        for item in work_items
    } if not dry_run else {}

    done_keys = set()
    if resume and not dry_run:
        existing_rows = read_rows(results_path)
        current_rows: list[dict[str, Any]] = []
        stale_count = 0
        retryable_count = 0
        for row in existing_rows:
            case = cases.get(str(row.get("case_id") or ""))
            if case is None:
                current_rows.append(row)
                continue
            recorded_lock = row.get("evaluator_lock") or {}
            recorded_fingerprint = (
                recorded_lock.get("evaluator_fingerprint")
                if isinstance(recorded_lock, Mapping)
                else None
            )
            current_fingerprint = case.evaluator_lock.get("evaluator_fingerprint")
            expected_input = input_fingerprints.get(row_key(row))
            input_matches = expected_input is None or row.get('candidate_input_fingerprint') == expected_input
            if recorded_fingerprint == current_fingerprint and input_matches:
                # An interrupted or temporarily misconfigured run can leave an
                # error row behind.  Treating that row as done makes --resume
                # skip the item forever, so remove it before constructing the
                # completed-key set and let the item run again.
                if row.get("status") == STATUS_ERROR:
                    retryable_count += 1
                else:
                    current_rows.append(row)
            else:
                stale_count += 1
        if stale_count or retryable_count:
            write_rows(results_path, current_rows)
            if stale_count:
                emit(
                    f"[wmeval-lite] removed {stale_count} stale result row(s) "
                    "whose evaluator or lighting candidate input fingerprint changed"
                )
            if retryable_count:
                emit(
                    f"[wmeval-lite] removed {retryable_count} retryable error "
                    "row(s)"
                )
        done_keys = {row_key(row) for row in current_rows}

    resolved_adapters: Mapping[str, DimensionEvaluator] = (
        dict(adapters) if adapters is not None else _default_adapters(runner)
    )
    installed_liquid_cases: set[str] = set()
    dry_run_items: list[dict[str, Any]] = []

    emit(
        f"[wmeval-lite] {len(work_items)} work item(s) across "
        f"{len(requested)} dimension(s), {len(candidate_rows)} candidate row(s)"
    )

    for item in work_items:
        dimension = item.dimension
        case = item.case
        candidate = item.candidate
        key = (
            dimension.dimension_id,
            candidate.case_id,
            candidate.model,
            candidate.seed,
        )
        if key in done_keys:
            emit(f"[wmeval-lite] skip (resume): {'/'.join(key)}")
            continue

        reference_video = (
            str(case.reference_video)
            if dimension.requires_reference and case.reference_video is not None
            else None
        )
        row_common = {
            "suite_name": suite.suite_name,
            "suite_version": suite.version,
            "run_id": run_id,
            "dimension": dimension.dimension_id,
            "case_id": case.case_id,
            "scene": case.scene,
            "model": candidate.model,
            "seed": candidate.seed,
            "reference_video": reference_video,
            "evaluator_lock": case.evaluator_lock,
        }

        if dry_run:
            candidate_exists = candidate.candidate_video.is_file()
            dry_run_items.append(
                {
                    "dimension": dimension.dimension_id,
                    "case_id": case.case_id,
                    "model": candidate.model,
                    "seed": candidate.seed,
                    "status": STATUS_DRY_RUN,
                    "candidate_video": str(candidate.candidate_video),
                    "candidate_exists": candidate_exists,
                    "reference_video": reference_video,
                    "reference_exists": bool(
                        reference_video and Path(reference_video).is_file()
                    ),
                }
            )
            emit(
                f"[wmeval-lite] dry-run {dimension.dimension_id}/{case.case_id}: "
                f"candidate={candidate.candidate_video} exists={candidate_exists} "
                f"reference={reference_video}"
            )
            continue

        started = time.monotonic()
        try:
            result = _evaluate_item(
                suite=suite,
                item=item,
                output_root=output_root,
                device=device,
                adapters=resolved_adapters,
                installed_liquid_cases=installed_liquid_cases,
            )
        except Exception as exc:  # noqa: BLE001 — one bad item must not kill the run
            result = CaseResult(
                status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                full_report={"traceback": traceback.format_exc()},
            )
        timing = time.monotonic() - started

        row = build_row(
            **row_common,
            status=result.status,
            candidate_video=str(candidate.candidate_video),
            metrics=result.metrics,
            full_report=result.full_report,
            error=result.error,
            timing_sec=round(timing, 3),
        )
        if input_fingerprints.get(key) is not None:
            row['candidate_input_fingerprint'] = input_fingerprints[key]
            row['required_repeats'] = 5
        if result.input_contract is not None:
            row['input_contract'] = result.input_contract
        support = support_record(item.case.artifacts_dir, candidate.model)
        if support is not None:
            row['action_support'] = support
        row['metric_status'] = metric_status(result.status, result.metrics)
        none_metrics = [
            name for name, value in (result.metrics or {}).items()
            if value is None and result.status == STATUS_COMPLETED
        ]
        for name in none_metrics:
            row['metric_status'][name] = "unmeasurable"
        append_row(results_path, row)
        emit(
            f"[wmeval-lite] {dimension.dimension_id}/{case.case_id}/"
            f"{candidate.model}/{candidate.seed}: {result.status}"
            + (f" ({result.error})" if result.error else "")
        )

    if dry_run:
        return {
            "suite_name": suite.suite_name,
            "version": suite.version,
            "dry_run": True,
            "num_work_items": len(work_items),
            "work_items": dry_run_items,
        }

    all_rows = read_rows(results_path)
    aggregate = build_aggregate(suite, all_rows)
    write_aggregate_json(output_root / AGGREGATE_JSON_FILENAME, aggregate)
    write_summary_csv(output_root / AGGREGATE_CSV_FILENAME, suite, aggregate)
    emit(f"[wmeval-lite] wrote {results_path}")
    return aggregate


def _evaluate_item(
    *,
    suite: Suite,
    item: WorkItem,
    output_root: Path,
    device: str | None,
    adapters: Mapping[str, DimensionEvaluator],
    installed_liquid_cases: set[str],
) -> CaseResult:
    dimension = item.dimension
    case = item.case
    candidate = item.candidate

    reason = unsupported_reason(case.artifacts_dir, candidate.model)
    if reason is not None:
        return CaseResult(status=STATUS_UNSUPPORTED, error=reason)

    if not candidate.candidate_video.is_file():
        return CaseResult(
            status=STATUS_MISSING_CANDIDATE,
            error=f"candidate video not found: {candidate.candidate_video}",
        )

    from .validation import CandidateValidationError, validate_candidate_video

    protocol = case.protocol
    candidate_fps_override = candidate.params.get("candidate_fps_override")
    try:
        validated = validate_candidate_video(
            candidate.candidate_video,
            case.reference_video if dimension.requires_reference else None,
            min_duration_sec=float(protocol["min_duration_sec"]),
            min_duration_ratio=float(protocol["min_duration_ratio"]),
            candidate_fps_override=(
                float(candidate_fps_override) if candidate_fps_override is not None else None
            ),
        )
    except CandidateValidationError as exc:
        return CaseResult(status=STATUS_INVALID_CANDIDATE, error=str(exc))

    candidate_video = _normalize_candidate(
        case=case, candidate=candidate, output_root=output_root
    )

    _install_liquid_artifacts_if_needed(
        dimension=dimension, case=case, installed=installed_liquid_cases
    )

    adapter = _resolve_adapter(dimension, adapters)
    result = adapter.evaluate(
        suite=suite,
        dimension=dimension,
        case=case,
        candidate=candidate,
        candidate_video=candidate_video,
        device=device,
    )
    result.input_contract = input_contract(
        validated, normalization=str(protocol.get("candidate_normalization") or "none")
    )
    return result


# Tolerances for the input-contract report (scores are never altered by it).
_FPS_TOLERANCE = 0.5
_ASPECT_TOLERANCE = 0.01
_DURATION_TOLERANCE = 0.05


def input_contract(validated: Any, *, normalization: str) -> dict[str, Any]:
    """Candidate vs. reference video properties for one scored row.

    The case's reference video is the declared input spec. With
    ``candidate_normalization: none`` the evaluator consumes the candidate as
    supplied, so mismatches are flagged (``within_spec: false``) for the reader;
    normalized cases report the raw candidate before resampling.
    """
    candidate = validated.candidate
    reference = validated.reference

    def props(meta: Any) -> dict[str, Any]:
        return {
            "fps": round(float(meta.fps), 4),
            "width": int(meta.width),
            "height": int(meta.height),
            "frame_count": int(meta.frame_count),
            "duration_sec": round(float(meta.duration_sec), 4),
        }

    report: dict[str, Any] = {"normalization": normalization, "candidate": props(candidate)}
    if reference is None:
        report["within_spec"] = None
        return report
    report["reference"] = props(reference)
    flags = {
        "fps_mismatch": abs(candidate.fps - reference.fps) > _FPS_TOLERANCE,
        "resolution_mismatch": (candidate.width, candidate.height) != (reference.width, reference.height),
        "aspect_mismatch": abs(candidate.width / candidate.height - reference.width / reference.height)
        > _ASPECT_TOLERANCE * reference.width / reference.height,
        "duration_mismatch": abs(candidate.duration_sec - reference.duration_sec)
        > _DURATION_TOLERANCE * reference.duration_sec,
    }
    report["flags"] = flags
    report["duration_ratio"] = round(candidate.duration_sec / reference.duration_sec, 4)
    report["within_spec"] = (not any(flags.values())) if normalization == "none" else None
    return report


def metric_status(status: str, metrics: Mapping[str, Any] | None) -> dict[str, str]:
    """Per-metric status: ``ok`` / ``unmeasurable`` for completed rows, else the row status."""
    out: dict[str, str] = {}
    for name, value in (metrics or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if status != STATUS_COMPLETED:
            out[name] = status
        else:
            out[name] = "ok" if value == value and value not in (float("inf"), float("-inf")) else "unmeasurable"
    return out
