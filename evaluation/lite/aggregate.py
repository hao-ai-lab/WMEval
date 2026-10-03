"""Flat metric extraction and suite-level aggregation.

Two responsibilities:

1. ``extract_metrics``: flatten each metric module's raw report into a small
   flat mapping of headline scalars per dimension (the module reports
   themselves ride along untouched in ``full_report``).
2. ``build_aggregate``: two-level mean over completed rows — seeds average
   within a case first, then cases average within a dimension — labelled
   explicitly as ``mean_over_cases_of_mean_over_seeds`` in the summary.

No metric is renamed or recomputed here; extraction is read-only plucking of
already-reported values. Headline metric names come from suite.yaml
(``aggregate.headline``), so this module never invents composite scores.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from .suite_loader import DimensionSpec, Suite

AGGREGATION_LABEL = "mean_over_cases_of_mean_over_seeds"

# Delta metric families reported by causal_step ({unity, wm, delta} each).
_CAUSAL_STEP_METRICS = ("csr", "amcl", "mcl", "ws", "dua", "icra", "rf")

_SHADOW_METRIC_KEYS = (
    "scene_retention_rate_pct",
    "shadow_detection_rate_pct",
    "multi_object_sun_disagreement_deg",
    "shadow_direction_jitter_deg_per_frame",
    "shadow_area_stability_pct",
    "shadow_shape_drift",
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _get(mapping: Any, *path: str) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _extract_lpips(result: Mapping[str, Any]) -> dict[str, Any]:
    primary_source = result.get("lpips_primary_source")
    mean = _numeric(result.get("lpips_mean"))
    median = _numeric(result.get("lpips_median"))
    causal_context = any(
        key in result
        for key in (
            "num_reference_events",
            "num_candidate_events",
            "causal_lpips_skipped_reason",
        )
    )
    is_temporal_fallback = primary_source == "temporal_only" and causal_context
    return {
        # A causal row that fell back to temporal LPIPS is not averaged into
        # the causal headline. Its value remains available under an explicit
        # fallback metric and source field.
        "lpips_mean": None if is_temporal_fallback else mean,
        "lpips_median": None if is_temporal_fallback else median,
        "lpips_causal_mean": mean if primary_source == "causal" else None,
        "lpips_temporal_fallback_mean": mean if is_temporal_fallback else None,
        "lpips_primary_source": primary_source,
        "lpips_num_matched_events": _numeric(result.get("num_matched_events")),
        "lpips_num_reference_events": _numeric(result.get("num_reference_events")),
        "lpips_num_candidate_events": _numeric(result.get("num_candidate_events")),
        "lpips_matched_reference_event_fraction": _numeric(
            result.get("matched_reference_event_fraction")
        ),
        "lpips_causal_skipped_reason": result.get("causal_lpips_skipped_reason"),
    }


def _extract_control_motion(result: Mapping[str, Any]) -> dict[str, Any]:
    # Prefer the camera-compensated section when present; fall back to raw.
    section_name = "camera_compensated" if "camera_compensated" in result else "raw"
    section = result.get(section_name)
    metrics: dict[str, Any] = {
        "num_frame_pairs": _numeric(result.get("num_frame_pairs")),
        "epe_mean_raw": _numeric(_get(result, "raw", "epe_summary", "mean")),
        "epe_dynamic_mean_raw": _numeric(
            _get(result, "raw", "epe_dynamic_summary", "mean")
        ),
        "epe_mean_compensated": _numeric(
            _get(result, "camera_compensated", "epe_summary", "mean")
        ),
        "epe_dynamic_mean_compensated": _numeric(
            _get(result, "camera_compensated", "epe_dynamic_summary", "mean")
        ),
        # Paper-comparable RAFT decomposition (Table 3: "Raw Relative EPE
        # Dynamic Mean" / "Raw Cosine Similarity Dynamic Mean") -- computed by
        # control_motion.py all along but never previously extracted past
        # full_report, so it never reached metrics/aggregate/CSV output.
        "relative_epe_dynamic_mean_raw": _numeric(
            _get(result, "raw", "relative_epe_dynamic_summary", "mean")
        ),
        "cosine_similarity_dynamic_mean_raw": _numeric(
            _get(result, "raw", "cosine_similarity_dynamic_summary", "mean")
        ),
    }
    bad_pixel = _get(section, "bad_pixel_rates") if section else None
    if isinstance(bad_pixel, Mapping):
        for threshold_key, stats in bad_pixel.items():
            metrics[f"bad_pixel_rate_{threshold_key}_mean"] = _numeric(
                _get(stats, "summary", "mean")
            )
    synthetic = result.get("synthetic_vs_candidate")
    if isinstance(synthetic, Mapping):
        metrics["synthetic_epe_mean"] = _numeric(
            _get(synthetic, "epe_summary", "mean")
        )
    return metrics


def _extract_causal_step(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for family in ("unity", "wm", "delta"):
        payload = result.get(family)
        if not isinstance(payload, Mapping):
            continue
        for metric_name in _CAUSAL_STEP_METRICS:
            value = _numeric(payload.get(metric_name))
            if value is not None:
                metrics[f"{metric_name}_{family}"] = value
    return metrics


def _extract_liquid_velocity(result: Mapping[str, Any]) -> dict[str, Any]:
    # liquid_velocity's inner keys are generic (epe/cosine_similarity/...), so
    # they get a velocity_ prefix — the one deliberate renaming in extraction.
    return {
        "velocity_epe_mean": _numeric(_get(result, "epe", "mean")),
        "velocity_epe_median": _numeric(_get(result, "epe", "median")),
        "velocity_cosine_similarity": _numeric(result.get("cosine_similarity")),
        "velocity_divergence_residual": _numeric(result.get("divergence_residual")),
        "velocity_temporal_consistency": _numeric(result.get("temporal_consistency")),
    }


# Tables 5/11 print `dsigma_A` signed but `|dAdot|` and `|dtheta_rms|`
# absolute. The signed value is what the module computes and stays canonical;
# the absolute companion reproduces the paper's columns, which matters under
# pooling because a mean of signed deltas cancels across action profiles.
_ABS_DELTA_METRICS = ("mask_area_drift_per_sec", "surface_tilt_residual_deg")


def _extract_liquid_volume_conservation(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("mask_area_std_normalized", "mask_area_drift_per_sec"):
        for family in ("unity", "wm", "delta"):
            value = _numeric(_get(result, key, family))
            if value is not None:
                metrics[f"{key}_{family}"] = value
                if family == "delta" and key in _ABS_DELTA_METRICS:
                    metrics[f"{key}_abs_delta"] = abs(value)
    return metrics


def _extract_liquid_surface_orientation(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for family in ("unity", "wm", "delta"):
        value = _numeric(_get(result, "surface_tilt_residual_deg", family))
        if value is not None:
            metrics[f"surface_tilt_residual_deg_{family}"] = value
            if family == "delta":
                metrics["surface_tilt_residual_deg_abs_delta"] = abs(value)
    return metrics


def _extract_slosh_spectrum(result: Mapping[str, Any]) -> dict[str, Any]:
    # B.4 traces the meniscus topmost-pixel y-coordinate and Table 11 names L1
    # the "meniscus power-spectrum L1 distance", so the `meniscus` block is the
    # trace to report; `slosh_spectrum` keeps the less sensitive centroid
    # variant at the top level. Fall back to the top level for payloads that
    # predate the split, and keep the centroid figure under its own key.
    paper_trace = result.get("meniscus")
    source: Mapping[str, Any] = (
        paper_trace if isinstance(paper_trace, Mapping) else result
    )
    metrics = {
        "dominant_frequency_hz_unity": _numeric(
            _get(source, "dominant_frequency_hz", "unity")
        ),
        "dominant_frequency_hz_wm": _numeric(
            _get(source, "dominant_frequency_hz", "wm")
        ),
        "dominant_frequency_ratio": _numeric(
            _get(source, "dominant_frequency_hz", "ratio")
        ),
        "dominant_frequency_delta_hz": _numeric(
            _get(source, "dominant_frequency_hz", "delta_hz")
        ),
        "spectral_l1": _numeric(source.get("spectral_l1")),
    }
    centroid = result.get("centroid")
    if isinstance(centroid, Mapping):
        metrics["spectral_l1_centroid"] = _numeric(centroid.get("spectral_l1"))
    # Non-paper variant (B.4 mean-centers the trace and nothing else), kept as
    # a diagnostic.
    compensated = result.get("bulk_compensated")
    if isinstance(compensated, Mapping):
        metrics["spectral_l1_bulk_compensated"] = _numeric(
            compensated.get("spectral_l1")
        )
    return metrics


def _extract_shadow_consistency(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _numeric(result.get(key)) for key in _SHADOW_METRIC_KEYS}


def _extract_environment_object_consistency(result: Mapping[str, Any]) -> dict[str, Any]:
    # depth_absrel_pct / pairwise_distance_mae_pct are None while Unity
    # analytical ground truth (see PENDING.md) is unavailable; object
    # permanence and character consistency are reference-free and always
    # populated when detection succeeds.
    return {
        "depth_absrel_pct": _numeric(result.get("depth_absrel_pct")),
        "pairwise_distance_mae_pct": _numeric(result.get("pairwise_distance_mae_pct")),
        "object_permanence_pct": _numeric(result.get("object_permanence_pct")),
        "character_consistency_pct": _numeric(result.get("character_consistency_pct")),
    }


def _extract_kinematics_mechanics(result: Mapping[str, Any]) -> dict[str, Any]:
    # trajectory_fidelity / mechanical_quantities are per-label dicts (module
    # currently supports pending/proxy statuses rather than final paper
    # metrics -- see evaluation/modules/kinematics_mechanics.py docstring);
    # surface a single dimension-level summary: the best (lowest) FDE/DTW
    # proxy across tracked labels, when available.
    fidelity = result.get("trajectory_fidelity")
    fde_values: list[float] = []
    dtw_values: list[float] = []
    fde_m_values: list[float] = []
    dtw_m_values: list[float] = []
    if isinstance(fidelity, Mapping):
        for entry in fidelity.values():
            if isinstance(entry, Mapping):
                fde = _numeric(entry.get("fde_px"))
                dtw = _numeric(entry.get("dtw_px"))
                fde_m = _numeric(entry.get("fde_m"))
                dtw_m = _numeric(entry.get("dtw_m_mean"))
                if fde is not None:
                    fde_values.append(fde)
                if dtw is not None:
                    dtw_values.append(dtw)
                if fde_m is not None:
                    fde_m_values.append(fde_m)
                if dtw_m is not None:
                    dtw_m_values.append(dtw_m)
    coverage = result.get("trajectory_coverage")
    required_count = tracked_count = scored_count = missing_count = None
    fidelity_status = None
    if isinstance(coverage, Mapping):
        required_count = _numeric(coverage.get("required_label_count"))
        tracked_count = _numeric(coverage.get("tracked_required_label_count"))
        scored_count = _numeric(coverage.get("scored_required_label_count"))
        missing_count = _numeric(coverage.get("missing_required_label_count"))
        fidelity_status = coverage.get("status")
        # Never publish an average over a favorable subset of required labels.
        # Counts and status remain available so callers can distinguish this
        # from a detector/evaluator crash.
        if missing_count is not None and missing_count > 0:
            fde_values.clear()
            dtw_values.clear()
            fde_m_values.clear()
            dtw_m_values.clear()
    mechanical_summary = result.get("mechanical_summary")
    exact_match_pct = None
    error_rate_pct = None
    if isinstance(mechanical_summary, Mapping):
        exact_match_pct = _numeric(mechanical_summary.get("exact_match_pct"))
        error_rate_pct = _numeric(mechanical_summary.get("error_rate_pct"))
    return {
        "fde_px_mean": _mean(fde_values),
        "dtw_px_mean": _mean(dtw_values),
        "fde_m_mean": _mean(fde_m_values),
        "dtw_m_mean": _mean(dtw_m_values),
        "exact_match_pct": exact_match_pct,
        "error_rate_pct": error_rate_pct,
        "trajectory_fidelity_status": fidelity_status,
        "required_label_count": required_count,
        "tracked_required_label_count": tracked_count,
        "scored_required_label_count": scored_count,
        "missing_required_label_count": missing_count,
    }


MODULE_METRIC_EXTRACTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "lpips": _extract_lpips,
    "control_motion": _extract_control_motion,
    "causal_step": _extract_causal_step,
    "liquid_velocity": _extract_liquid_velocity,
    "liquid_volume_conservation": _extract_liquid_volume_conservation,
    "liquid_surface_orientation": _extract_liquid_surface_orientation,
    "slosh_spectrum": _extract_slosh_spectrum,
    "shadow_consistency": _extract_shadow_consistency,
    "environment_object_consistency": _extract_environment_object_consistency,
    "kinematics_mechanics": _extract_kinematics_mechanics,
}


def extract_metrics(
    dimension: DimensionSpec, results: Mapping[str, Any]
) -> dict[str, Any]:
    """Flatten the runner's per-module results for one dimension."""
    metrics: dict[str, Any] = {}
    for module_name in dimension.modules:
        module_result = results.get(module_name)
        if not isinstance(module_result, Mapping):
            continue
        extractor = MODULE_METRIC_EXTRACTORS.get(module_name)
        if extractor is None:
            continue
        metrics.update(extractor(module_result))
    return metrics


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_std(values: Sequence[float]) -> float | None:
    """Sample standard deviation (ddof=1); None below two observations."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _metric_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            names.update(
                key for key, value in metrics.items() if _numeric(value) is not None
            )
    return sorted(names)


def _two_level_stats(
    completed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Two-level statistics: seeds within a case, then cases within the dimension.

    Per metric: the headline mean (mean over case means) plus the dispersion a
    reviewer needs to judge it — spread across cases (sample std / min / max of
    case means) and per-case seed variability (sample std over seeds, None for
    single-seed cases). Deliberately no case-level bootstrap/CI: with a handful
    of cases per dimension a resampled interval would overstate rigor; the raw
    case means are reported instead.
    """
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in completed_rows:
        by_case.setdefault(str(row.get("case_id")), []).append(row)

    stats: dict[str, dict[str, Any]] = {}
    for metric_name in _metric_names(completed_rows):
        case_means: dict[str, float] = {}
        seed_std_by_case: dict[str, float | None] = {}
        for case_id in sorted(by_case):
            values = [
                value
                for row in by_case[case_id]
                if (value := _numeric(_get(row, "metrics", metric_name))) is not None
            ]
            if not values:
                continue
            case_means[case_id] = sum(values) / len(values)
            seed_std_by_case[case_id] = _sample_std(values)
        mean_values = list(case_means.values())
        stats[metric_name] = {
            "mean": _mean(mean_values),
            "std_across_cases": _sample_std(mean_values),
            "min_case_mean": min(mean_values) if mean_values else None,
            "max_case_mean": max(mean_values) if mean_values else None,
            "n_cases": len(mean_values),
            "case_means": case_means,
            "seed_std_by_case": seed_std_by_case,
        }
    return stats


def build_aggregate(suite: Suite, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]

    models: dict[str, Any] = {}
    model_names = sorted({str(row.get("model") or "") for row in rows})
    for model_name in model_names:
        model_rows = [row for row in rows if str(row.get("model") or "") == model_name]
        dimension_summaries: dict[str, Any] = {}
        for dimension_id, dimension in suite.dimensions.items():
            dimension_rows = [
                row for row in model_rows if row.get("dimension") == dimension_id
            ]
            if not dimension_rows:
                continue
            done = [row for row in dimension_rows if row.get("status") == "completed"]
            stats = _two_level_stats(done)
            means = {metric: payload["mean"] for metric, payload in stats.items()}
            headline: dict[str, Any] | None = None
            if dimension.aggregate is not None:
                headline_stats = stats.get(dimension.aggregate.headline, {})
                headline = {
                    "metric": dimension.aggregate.headline,
                    "value": headline_stats.get("mean"),
                    "std_across_cases": headline_stats.get("std_across_cases"),
                    "direction": dimension.aggregate.direction,
                }
            dimension_summaries[dimension_id] = {
                "display_name": dimension.display_name,
                "num_rows": len(dimension_rows),
                "num_cases": len({str(row.get("case_id")) for row in dimension_rows}),
                "num_completed": len(done),
                "headline": headline,
                "means": means,
                "stats": stats,
                "per_case": [
                    {
                        "case_id": row.get("case_id"),
                        "seed": row.get("seed"),
                        "status": row.get("status"),
                        "error": row.get("error"),
                        "metrics": row.get("metrics"),
                    }
                    for row in dimension_rows
                ],
            }
        models[model_name] = {"dimensions": dimension_summaries}

    return {
        "suite_name": suite.suite_name,
        "version": suite.version,
        "aggregation": AGGREGATION_LABEL,
        "num_rows": len(rows),
        "num_completed": len(completed),
        "models": models,
    }


def write_aggregate_json(path: str | Path, aggregate: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _extra_mean_names(dimension, headline_name: str) -> tuple[str, ...]:
    """`extra_means` minus the headline, which already has its own columns."""
    if dimension.aggregate is None:
        return ()
    return tuple(
        name for name in dimension.aggregate.extra_means if name != headline_name
    )


def write_summary_csv(
    path: str | Path, suite: Suite, aggregate: Mapping[str, Any]
) -> None:
    """One row per model; per dimension: headline mean + std across cases."""
    dimension_ids = list(suite.dimensions)
    fieldnames = ["model"]
    for dimension_id in dimension_ids:
        dimension = suite.dimensions[dimension_id]
        headline_name = (
            dimension.aggregate.headline if dimension.aggregate is not None else "n/a"
        )
        fieldnames.append(f"{dimension_id}:{headline_name}")
        fieldnames.append(f"{dimension_id}:{headline_name}_std")
        # `extra_means` names the metrics worth reading beside the headline.
        # Additive: the JSON `means` block still carries every metric the
        # modules emit, so listing one here surfaces it without hiding others.
        for extra in _extra_mean_names(dimension, headline_name):
            fieldnames.append(f"{dimension_id}:{extra}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for model_name, payload in sorted(aggregate.get("models", {}).items()):
            row: list[Any] = [model_name]
            dimensions = payload.get("dimensions", {})
            for dimension_id in dimension_ids:
                summary = dimensions.get(dimension_id) or {}
                headline = summary.get("headline") or {}
                row.append(headline.get("value"))
                row.append(headline.get("std_across_cases"))
                dimension = suite.dimensions[dimension_id]
                headline_name = (
                    dimension.aggregate.headline
                    if dimension.aggregate is not None
                    else "n/a"
                )
                means = summary.get("means") or {}
                for extra in _extra_mean_names(dimension, headline_name):
                    row.append(means.get(extra))
            writer.writerow(row)
