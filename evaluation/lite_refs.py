from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .runner import run_world_model_evaluation


DEFAULT_LITE_MANIFEST = "eval_sets/wmeval_lite_v1/manifest.json"


@dataclass(frozen=True)
class LiteDimension:
    dimension_id: str
    display_name: str
    description: str
    metric_config: Mapping[str, Any]
    test_cases: tuple["LiteTestCase", ...]


@dataclass(frozen=True)
class LiteTestCase:
    dimension_id: str
    test_case_id: str
    display_name: str
    legacy_ids: tuple[str, ...]
    prompt: str
    perspective: str
    reference_video: str
    reference_video_sha256: Optional[str]
    reference_artifacts: Mapping[str, Any]
    reference_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class LiteCandidate:
    dimension_id: str
    test_case_id: str
    world_model_name: str
    world_model_video: str
    run_id: str = ""


def repo_root_from_manifest(manifest_path: str | Path) -> Path:
    path = Path(manifest_path).expanduser().resolve()
    for parent in path.parents:
        if parent.name == "eval_sets":
            return parent.parent
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(
    raw_path: str | None,
    *,
    repo_root: Path,
    manifest_dir: Path,
) -> Optional[Path]:
    if raw_path is None or not str(raw_path).strip():
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    repo_relative = (repo_root / path).resolve()
    manifest_relative = (manifest_dir / path).resolve()
    if repo_relative.exists() or not manifest_relative.exists():
        return repo_relative
    return manifest_relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(path: Path, expected: Optional[str]) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def ensure_reference_video(
    test_case: LiteTestCase,
    *,
    repo_root: Path,
    manifest_dir: Path,
) -> Path:
    local = resolve_repo_path(
        test_case.reference_video,
        repo_root=repo_root,
        manifest_dir=manifest_dir,
    )
    if local is None or not local.exists():
        expected = str(local) if local is not None else test_case.reference_video
        raise FileNotFoundError(
            f"Reference video for {test_case.dimension_id}/{test_case.test_case_id} "
            f"not found: {expected}. Run tools/verify_wmeval_lite_refs.py."
        )
    _validate_sha256(local, test_case.reference_video_sha256)
    return local


def load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path).expanduser()
    if source.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {source}: {exc}") from exc
        return rows
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected JSONL or a JSON list: {source}")


def parse_lite_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Lite manifest must be a JSON object: {manifest_path}")

    dimensions_raw = data.get("dimensions")
    if not isinstance(dimensions_raw, Mapping) or not dimensions_raw:
        raise ValueError("Lite manifest requires a non-empty dimensions{} object")

    dimensions: List[LiteDimension] = []
    all_test_cases: List[LiteTestCase] = []
    for dimension_id, payload in dimensions_raw.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Dimension {dimension_id} must be an object")
        cases_raw = payload.get("test_cases")
        if not isinstance(cases_raw, list) or not cases_raw:
            raise ValueError(f"Dimension {dimension_id} requires non-empty test_cases[]")
        test_cases = tuple(
            _parse_test_case(
                dimension_id=str(dimension_id),
                row=row,
                index=index,
            )
            for index, row in enumerate(cases_raw, start=1)
        )
        case_ids = [case.test_case_id for case in test_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                f"Duplicate test_case_id values in dimension {dimension_id}: {case_ids}"
            )
        dimensions.append(
            LiteDimension(
                dimension_id=str(dimension_id),
                display_name=str(payload.get("display_name") or dimension_id),
                description=str(payload.get("description") or ""),
                metric_config=dict(payload.get("metric_config") or {}),
                test_cases=test_cases,
            )
        )
        all_test_cases.extend(test_cases)

    return {
        "suite_name": str(data.get("suite_name") or "WMEval Lite"),
        "version": str(data.get("version") or "v1"),
        "dimensions": tuple(dimensions),
        "test_cases": tuple(all_test_cases),
        "manifest_path": manifest_path,
    }


def _parse_test_case(*, dimension_id: str, row: Any, index: int) -> LiteTestCase:
    if not isinstance(row, Mapping):
        raise ValueError(f"{dimension_id} test case #{index} must be an object")
    test_case_id = str(row.get("test_case_id") or "").strip()
    if not test_case_id:
        raise ValueError(f"{dimension_id} test case #{index} missing test_case_id")
    legacy_raw = row.get("legacy_ids") or []
    if not isinstance(legacy_raw, list):
        raise ValueError(f"{dimension_id}/{test_case_id} legacy_ids must be a list")
    return LiteTestCase(
        dimension_id=dimension_id,
        test_case_id=test_case_id,
        display_name=str(row.get("display_name") or test_case_id).strip(),
        legacy_ids=tuple(str(item).strip() for item in legacy_raw if str(item).strip()),
        prompt=str(row.get("prompt") or "").strip(),
        perspective=str(row.get("perspective") or "").strip(),
        reference_video=str(row.get("reference_video") or "").strip(),
        reference_video_sha256=(
            str(row["reference_video_sha256"]).strip()
            if row.get("reference_video_sha256")
            else None
        ),
        reference_artifacts=dict(row.get("reference_artifacts") or {}),
        reference_metrics=dict(row.get("reference_metrics") or {}),
    )


def test_case_lookup(test_cases: Sequence[LiteTestCase]) -> Dict[Tuple[str, str], LiteTestCase]:
    lookup: Dict[Tuple[str, str], LiteTestCase] = {}
    for test_case in test_cases:
        lookup[(test_case.dimension_id, test_case.test_case_id)] = test_case
        for legacy_id in test_case.legacy_ids:
            lookup[(test_case.dimension_id, legacy_id)] = test_case
    return lookup


def resolve_candidate_target(row: Mapping[str, Any]) -> Tuple[str, str]:
    dimension_id = str(row.get("dimension") or row.get("dimension_id") or "").strip()
    test_case_id = str(
        row.get("test_case_id") or row.get("scene_id") or row.get("sample_id") or row.get("id") or ""
    ).strip()

    if not dimension_id:
        raise ValueError(f"Candidate row must include a dimension: {row}")
    if not test_case_id:
        raise ValueError(f"Candidate row must include a test_case_id: {row}")
    return dimension_id, test_case_id


def merge_candidate_rows(
    test_cases: Sequence[LiteTestCase],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> List[LiteCandidate]:
    by_key = test_case_lookup(test_cases)
    merged: List[LiteCandidate] = []
    for row in candidate_rows:
        dimension_id, test_case_id = resolve_candidate_target(row)
        if (dimension_id, test_case_id) not in by_key:
            raise ValueError(
                f"Candidate row references unknown dimension/test_case: "
                f"{dimension_id}/{test_case_id}"
            )
        world_model_video = str(
            row.get("world_model_video")
            or row.get("candidate_video")
            or row.get("genie3_video")
            or row.get("pred_video")
            or ""
        ).strip()
        if not world_model_video:
            raise ValueError(f"Candidate row missing world_model_video: {row}")
        merged.append(
            LiteCandidate(
                dimension_id=dimension_id,
                test_case_id=test_case_id,
                world_model_name=str(
                    row.get("world_model_name") or row.get("model_name") or ""
                ).strip(),
                world_model_video=world_model_video,
                run_id=str(row.get("run_id") or "").strip(),
            )
        )
    return merged


def _resolve_synthetic_action_file(
    test_case: LiteTestCase,
    *,
    repo_root: Path,
    manifest_dir: Path,
) -> Optional[Path]:
    artifacts = test_case.reference_artifacts
    action_dir = artifacts.get("synthetic_action_dir")
    if action_dir:
        dir_path = resolve_repo_path(str(action_dir), repo_root=repo_root, manifest_dir=manifest_dir)
        if dir_path is not None:
            action_npy = dir_path / "action.npy"
            if action_npy.exists():
                return action_npy
    action_file = artifacts.get("synthetic_action_file") or artifacts.get("actions_file")
    if action_file:
        return resolve_repo_path(str(action_file), repo_root=repo_root, manifest_dir=manifest_dir)
    return None


def _dimension_by_id(manifest: Mapping[str, Any]) -> Dict[str, LiteDimension]:
    return {dimension.dimension_id: dimension for dimension in manifest["dimensions"]}


def run_lite_suite(
    *,
    manifest_json: str | Path,
    output_dir: str | Path,
    candidate_jsonl: str | Path | None = None,
    dimensions: Sequence[str] | None = None,
    modules: Sequence[str] | None = None,
    eval_fps: float = 4.0,
    max_pairs: int = 32,
    device: Optional[str] = None,
    dry_run: bool = False,
    runner: Callable[..., Dict[str, Any]] = run_world_model_evaluation,
) -> Dict[str, Any]:
    manifest = parse_lite_manifest(manifest_json)
    manifest_path = Path(manifest["manifest_path"])
    repo_root = repo_root_from_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    dimension_map = _dimension_by_id(manifest)
    requested_dimensions = (
        tuple(dimensions)
        if dimensions
        else tuple(dimension.dimension_id for dimension in manifest["dimensions"])
    )
    unknown = sorted(set(requested_dimensions) - set(dimension_map))
    if unknown:
        raise ValueError(f"Unknown dimension(s): {unknown}")

    if candidate_jsonl is None:
        raise ValueError("candidate_jsonl is required for lite evaluation")
    candidates = merge_candidate_rows(manifest["test_cases"], load_json_or_jsonl(candidate_jsonl))
    candidates = [
        candidate
        for candidate in candidates
        if candidate.dimension_id in requested_dimensions
    ]
    by_key = test_case_lookup(manifest["test_cases"])

    out_root = Path(output_dir).expanduser().resolve()
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        test_case = by_key[(candidate.dimension_id, candidate.test_case_id)]
        dimension = dimension_map[candidate.dimension_id]
        metric_config = dict(dimension.metric_config)
        module_list = tuple(modules or metric_config.get("modules") or ("lpips",))
        eval_fps_value = float(metric_config.get("eval_fps", eval_fps))
        max_pairs_value = int(metric_config.get("max_pairs", max_pairs))

        if dry_run:
            reference_video = resolve_repo_path(
                test_case.reference_video,
                repo_root=repo_root,
                manifest_dir=manifest_dir,
            )
            candidate_video = resolve_repo_path(
                candidate.world_model_video,
                repo_root=repo_root,
                manifest_dir=manifest_dir,
            )
            rows.append(
                {
                    "dimension_id": candidate.dimension_id,
                    "dimension_name": dimension.display_name,
                    "test_case_id": test_case.test_case_id,
                    "display_name": test_case.display_name,
                    "world_model_name": candidate.world_model_name,
                    "run_id": candidate.run_id,
                    "reference_video": str(reference_video or test_case.reference_video),
                    "world_model_video": str(candidate_video or candidate.world_model_video),
                    "reference_exists": bool(reference_video and reference_video.exists()),
                    "candidate_exists": bool(candidate_video and candidate_video.exists()),
                    "status": "dry_run",
                }
            )
            continue

        reference_video = ensure_reference_video(
            test_case,
            repo_root=repo_root,
            manifest_dir=manifest_dir,
        )
        candidate_video = resolve_repo_path(
            candidate.world_model_video,
            repo_root=repo_root,
            manifest_dir=manifest_dir,
        )
        if candidate_video is None or not candidate_video.exists():
            rows.append(
                {
                    "dimension_id": candidate.dimension_id,
                    "dimension_name": dimension.display_name,
                    "test_case_id": test_case.test_case_id,
                    "display_name": test_case.display_name,
                    "world_model_name": candidate.world_model_name,
                    "run_id": candidate.run_id,
                    "status": "missing_candidate",
                    "error": f"world_model_video not found: {candidate.world_model_video}",
                }
            )
            continue

        synthetic_action_path = _resolve_synthetic_action_file(
            test_case,
            repo_root=repo_root,
            manifest_dir=manifest_dir,
        )
        runner_kwargs: Dict[str, Any] = {
            "video_path": str(candidate_video),
            "ref_video_path": str(reference_video),
            "text_prompt": test_case.prompt,
            "modules": module_list,
            "eval_fps": eval_fps_value,
            "max_pairs": max_pairs_value,
            "device": device,
            "synthetic_action_file": (
                str(synthetic_action_path) if synthetic_action_path is not None else None
            ),
            "synthetic_infer_actions": synthetic_action_path is None,
        }
        if "control_camera_compensation" in metric_config:
            runner_kwargs["control_camera_compensation"] = bool(
                metric_config["control_camera_compensation"]
            )
        if metric_config.get("control_global_motion_mode"):
            runner_kwargs["control_global_motion_mode"] = str(
                metric_config["control_global_motion_mode"]
            )
        if metric_config.get("control_bad_pixel_thresholds"):
            runner_kwargs["control_bad_pixel_thresholds"] = list(
                metric_config["control_bad_pixel_thresholds"]
            )

        report = runner(**runner_kwargs)
        lpips_result = dict(report.get("results", {}).get("lpips", {}))
        control_result = dict(report.get("results", {}).get("control_motion", {}))
        rows.append(
            {
                "dimension_id": candidate.dimension_id,
                "dimension_name": dimension.display_name,
                "test_case_id": test_case.test_case_id,
                "display_name": test_case.display_name,
                "world_model_name": candidate.world_model_name,
                "run_id": candidate.run_id,
                "prompt": test_case.prompt,
                "reference_video": str(reference_video),
                "world_model_video": str(candidate_video),
                "status": "completed",
                "reference_metrics": dict(test_case.reference_metrics),
                "lpips_mean": lpips_result.get("lpips_mean"),
                "lpips_median": lpips_result.get("lpips_median"),
                "control_motion": {
                    "num_frame_pairs": control_result.get("num_frame_pairs"),
                    "raw": control_result.get("raw"),
                    "camera_compensated": control_result.get("camera_compensated"),
                    "synthetic_vs_candidate": control_result.get("synthetic_vs_candidate"),
                },
                "full_report": report,
            }
        )

    aggregate = build_lite_aggregate(
        suite_name=manifest["suite_name"],
        version=manifest["version"],
        rows=rows,
    )
    if not dry_run:
        _write_jsonl(out_root / "per_case_results.jsonl", rows)
        (out_root / "aggregate_summary.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )
        _write_summary_csv(out_root / "aggregate_summary.csv", rows)
    return aggregate


def build_lite_aggregate(
    *,
    suite_name: str,
    version: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_dimension: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_dimension.setdefault(str(row.get("dimension_id") or ""), []).append(row)

    dimension_summaries: Dict[str, Any] = {}
    for dimension_id, dimension_rows in by_dimension.items():
        done = [row for row in dimension_rows if row.get("status") == "completed"]
        lpips_values = [
            float(row["lpips_mean"])
            for row in done
            if isinstance(row.get("lpips_mean"), (int, float))
        ]
        dimension_summaries[dimension_id] = {
            "dimension_name": dimension_rows[0].get("dimension_name") if dimension_rows else dimension_id,
            "num_rows": len(dimension_rows),
            "num_completed": len(done),
            "means": {
                "lpips_mean": (sum(lpips_values) / len(lpips_values)) if lpips_values else None,
            },
            "test_case_metrics": [
                {
                    "test_case_id": row.get("test_case_id"),
                    "display_name": row.get("display_name"),
                    "world_model_name": row.get("world_model_name"),
                    "run_id": row.get("run_id"),
                    "status": row.get("status"),
                    "lpips_mean": row.get("lpips_mean"),
                    "error": row.get("error"),
                }
                for row in dimension_rows
            ],
        }

    return {
        "suite_name": suite_name,
        "version": version,
        "label": f"{suite_name} w/ bundled reference",
        "num_rows": len(rows),
        "num_completed": len(completed),
        "dimensions": dimension_summaries,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "dimension_id",
        "dimension_name",
        "test_case_id",
        "display_name",
        "world_model_name",
        "run_id",
        "status",
        "lpips_mean",
        "lpips_median",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
