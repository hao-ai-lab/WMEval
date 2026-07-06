from __future__ import annotations

import csv
from pathlib import Path
from types import MappingProxyType

from evaluation.lite.aggregate import (
    AGGREGATION_LABEL,
    build_aggregate,
    extract_metrics,
    write_summary_csv,
)
from evaluation.lite.output import (
    ROW_SCHEMA_VERSION,
    append_row,
    build_row,
    load_done_keys,
    read_rows,
    row_key,
)
from evaluation.lite.suite_loader import AggregateSpec, DimensionSpec, Suite


def _suite_with(dimensions: dict[str, DimensionSpec]) -> Suite:
    return Suite(
        root=Path("/nonexistent"),
        suite_name="WMEval Lite Test",
        version="v-test",
        description="",
        dimensions=MappingProxyType(dimensions),
        legacy_case_aliases=MappingProxyType({}),
    )


def _dimension(
    dimension_id: str,
    *,
    modules: tuple[str, ...] = ("lpips",),
    headline: str = "lpips_mean",
    direction: str = "lower",
) -> DimensionSpec:
    return DimensionSpec(
        dimension_id=dimension_id,
        display_name=dimension_id,
        description="",
        evaluator="registry",
        requires_reference=True,
        cases=("scene_a/main",),
        modules=modules,
        aggregate=AggregateSpec(headline=headline, direction=direction),
    )


def _row(
    *,
    dimension: str = "toy_metric",
    case_id: str = "scene_a/main",
    model: str = "genie3",
    seed: str = "0001",
    status: str = "completed",
    metrics: dict | None = None,
) -> dict:
    return build_row(
        suite_name="WMEval Lite Test",
        suite_version="v-test",
        run_id="test",
        dimension=dimension,
        case_id=case_id,
        scene=case_id.split("/")[0],
        model=model,
        seed=seed,
        status=status,
        candidate_video=f"/videos/{model}/{case_id}/{seed}.mp4",
        reference_video="/refs/ref.mp4",
        evaluator_lock={
            "release_id": "wmeval-lite-test",
            "evaluator_fingerprint": "fp",
            "code_revision": "abc",
        },
        metrics=metrics if metrics is not None else {"lpips_mean": 0.5},
        full_report={"results": {}},
        timing_sec=1.0,
    )


def test_build_row_schema_fields() -> None:
    row = _row()
    assert set(row) == {
        "schema_version",
        "suite_name",
        "suite_version",
        "run_id",
        "dimension",
        "case_id",
        "scene",
        "model",
        "seed",
        "status",
        "error",
        "candidate_video",
        "reference_video",
        "evaluator_lock",
        "metrics",
        "full_report",
        "timing_sec",
    }
    assert row["schema_version"] == ROW_SCHEMA_VERSION
    # evaluator_lock summarized to the two reproducibility fields
    assert row["evaluator_lock"] == {
        "release_id": "wmeval-lite-test",
        "evaluator_fingerprint": "fp",
    }
    assert row_key(row) == ("toy_metric", "scene_a/main", "genie3", "0001")


def test_append_read_and_done_keys_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    first = _row(seed="0001")
    second = _row(seed="0002")
    append_row(path, first)
    append_row(path, second)

    rows = read_rows(path)
    assert [row["seed"] for row in rows] == ["0001", "0002"]
    assert load_done_keys(path) == {row_key(first), row_key(second)}
    assert load_done_keys(tmp_path / "missing.jsonl") == set()


def test_two_level_mean_over_seeds_then_cases() -> None:
    suite = _suite_with({"toy_metric": _dimension("toy_metric")})
    rows = [
        # case A: two seeds → case mean 0.3
        _row(case_id="scene_a/main", seed="0001", metrics={"lpips_mean": 0.2}),
        _row(case_id="scene_a/main", seed="0002", metrics={"lpips_mean": 0.4}),
        # case B: one seed → case mean 0.5
        _row(case_id="scene_b/main", seed="0001", metrics={"lpips_mean": 0.5}),
    ]

    aggregate = build_aggregate(suite, rows)

    summary = aggregate["models"]["genie3"]["dimensions"]["toy_metric"]
    # two-level: mean(0.3, 0.5) = 0.4, NOT flat mean(0.2, 0.4, 0.5) = 0.3667
    assert abs(summary["means"]["lpips_mean"] - 0.4) < 1e-9
    assert summary["headline"]["value"] == summary["means"]["lpips_mean"]
    assert summary["num_completed"] == 3
    assert aggregate["aggregation"] == AGGREGATION_LABEL


def test_aggregate_excludes_non_completed_rows_from_means() -> None:
    suite = _suite_with({"toy_metric": _dimension("toy_metric")})
    rows = [
        _row(seed="0001", metrics={"lpips_mean": 0.2}),
        _row(seed="0002", status="error", metrics={}),
    ]

    aggregate = build_aggregate(suite, rows)

    summary = aggregate["models"]["genie3"]["dimensions"]["toy_metric"]
    assert summary["num_rows"] == 2
    assert summary["num_completed"] == 1
    assert summary["means"]["lpips_mean"] == 0.2


def test_write_summary_csv_one_row_per_model(tmp_path: Path) -> None:
    suite = _suite_with({"toy_metric": _dimension("toy_metric")})
    rows = [
        _row(model="genie3", metrics={"lpips_mean": 0.2}),
        _row(model="other", metrics={"lpips_mean": 0.6}),
    ]
    aggregate = build_aggregate(suite, rows)

    csv_path = tmp_path / "aggregate_summary.csv"
    write_summary_csv(csv_path, suite, aggregate)

    with csv_path.open(newline="") as handle:
        parsed = list(csv.reader(handle))
    assert parsed[0] == ["model", "toy_metric:lpips_mean"]
    assert parsed[1] == ["genie3", "0.2"]
    assert parsed[2] == ["other", "0.6"]


def test_extract_metrics_control_motion() -> None:
    dimension = _dimension("controllability", modules=("control_motion",))
    results = {
        "control_motion": {
            "num_frame_pairs": 7,
            "raw": {
                "epe_summary": {"mean": 2.0},
                "epe_dynamic_summary": {"mean": 3.0},
                "bad_pixel_rates": {
                    "1px": {"summary": {"mean": 0.5}},
                    "3px": {"summary": {"mean": 0.2}},
                },
            },
            "camera_compensated": {
                "epe_summary": {"mean": 1.5},
                "epe_dynamic_summary": {"mean": 2.5},
                "bad_pixel_rates": {
                    "1px": {"summary": {"mean": 0.4}},
                    "3px": {"summary": {"mean": 0.1}},
                },
            },
            "synthetic_vs_candidate": {"epe_summary": {"mean": 4.0}},
        }
    }

    metrics = extract_metrics(dimension, results)

    assert metrics["num_frame_pairs"] == 7
    assert metrics["epe_mean_raw"] == 2.0
    assert metrics["epe_dynamic_mean_raw"] == 3.0
    assert metrics["epe_mean_compensated"] == 1.5
    assert metrics["epe_dynamic_mean_compensated"] == 2.5
    # bad-pixel rates come from the compensated section when present
    assert metrics["bad_pixel_rate_1px_mean"] == 0.4
    assert metrics["bad_pixel_rate_3px_mean"] == 0.1
    assert metrics["synthetic_epe_mean"] == 4.0


def test_extract_metrics_causal_step_families() -> None:
    dimension = _dimension("causal_step", modules=("causal_step",))
    results = {
        "causal_step": {
            "unity": {"csr": 0.9, "rf": 0.1},
            "wm": {"csr": 0.7, "rf": 0.3},
            "delta": {"csr": 0.2, "rf": 0.2},
        }
    }

    metrics = extract_metrics(dimension, results)

    assert metrics == {
        "csr_unity": 0.9,
        "rf_unity": 0.1,
        "csr_wm": 0.7,
        "rf_wm": 0.3,
        "csr_delta": 0.2,
        "rf_delta": 0.2,
    }


def test_extract_metrics_liquid_modules() -> None:
    dimension = _dimension(
        "liquid",
        modules=(
            "liquid_velocity",
            "liquid_volume_conservation",
            "liquid_surface_orientation",
            "slosh_spectrum",
        ),
    )
    results = {
        "liquid_velocity": {
            "epe": {"mean": 1.1, "median": 1.0},
            "cosine_similarity": 0.8,
            "divergence_residual": 0.05,
            "temporal_consistency": 0.9,
        },
        "liquid_volume_conservation": {
            "mask_area_std_normalized": {"unity": 0.01, "wm": 0.03, "delta": 0.02},
            "mask_area_drift_per_sec": {"unity": 0.001, "wm": 0.004, "delta": 0.003},
        },
        "liquid_surface_orientation": {
            "surface_tilt_residual_deg": {"unity": 1.0, "wm": 4.0, "delta": 3.0},
        },
        "slosh_spectrum": {
            "dominant_frequency_hz": {
                "unity": 1.2,
                "wm": 1.5,
                "ratio": 1.25,
                "delta_hz": 0.3,
            },
            "spectral_l1": 0.4,
        },
    }

    metrics = extract_metrics(dimension, results)

    assert metrics["velocity_epe_mean"] == 1.1
    assert metrics["velocity_cosine_similarity"] == 0.8
    assert metrics["mask_area_std_normalized_delta"] == 0.02
    assert metrics["mask_area_drift_per_sec_wm"] == 0.004
    assert metrics["surface_tilt_residual_deg_delta"] == 3.0
    assert metrics["dominant_frequency_delta_hz"] == 0.3
    assert metrics["spectral_l1"] == 0.4


def test_extract_metrics_ignores_unknown_modules_and_missing_results() -> None:
    dimension = _dimension("toy", modules=("lpips", "mystery_module"))
    metrics = extract_metrics(dimension, {"lpips": {"lpips_mean": 0.3}})
    assert metrics == {"lpips_mean": 0.3, "lpips_median": None}
