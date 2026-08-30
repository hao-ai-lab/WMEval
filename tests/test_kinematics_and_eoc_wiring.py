"""Cross-cutting wiring tests for the B.1/B.3 modules: aggregate extraction
and the registry_adapter metric-config whitelist (mirrors how
test_lite_schema_v2.py / test_lite_suite_runner.py exercise shadow_consistency's
wiring)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_extract_metrics_environment_object_consistency():
    from evaluation.lite.aggregate import extract_metrics
    from evaluation.lite.suite_loader import DimensionSpec

    dimension = DimensionSpec(
        dimension_id="scene_character_consistency",
        display_name="Scene & Character Consistency",
        description="test dimension",
        evaluator="registry",
        requires_reference=False,
        cases=(),
        modules=("environment_object_consistency",),
    )
    results = {
        "environment_object_consistency": {
            "module": "environment_object_consistency",
            "depth_absrel_pct": None,
            "pairwise_distance_mae_pct": None,
            "object_permanence_pct": 91.2,
            "character_consistency_pct": 87.5,
            "ground_truth_status": "pending",
        }
    }
    metrics = extract_metrics(dimension, results)
    assert metrics["object_permanence_pct"] == 91.2
    assert metrics["character_consistency_pct"] == 87.5
    assert metrics["depth_absrel_pct"] is None
    assert metrics["pairwise_distance_mae_pct"] is None


def test_extract_metrics_kinematics_mechanics():
    from evaluation.lite.aggregate import extract_metrics
    from evaluation.lite.suite_loader import DimensionSpec

    dimension = DimensionSpec(
        dimension_id="kinematics_mechanics",
        display_name="Kinematics & Mechanics",
        description="test dimension",
        evaluator="registry",
        requires_reference=False,
        cases=(),
        modules=("kinematics_mechanics",),
    )
    results = {
        "kinematics_mechanics": {
            "module": "kinematics_mechanics",
            "trajectory_fidelity": {
                "ball": {"status": "reprojection_proxy_2d", "fde_px": 4.0, "dtw_px": 12.0},
                "car": {"status": "reprojection_proxy_2d", "fde_px": 8.0, "dtw_px": 20.0},
            },
            "mechanical_quantities": None,
        }
    }
    metrics = extract_metrics(dimension, results)
    assert metrics["fde_px_mean"] == 6.0
    assert metrics["dtw_px_mean"] == 16.0


def test_metric_config_whitelist_accepts_new_module_keys():
    from evaluation.lite.adapters.registry_adapter import METRIC_CONFIG_KEYS

    for key in (
        "eoc_text_prompt",
        "eoc_character_prompt",
        "eoc_tau_det",
        "eoc_tau_mask",
        "eoc_tau_id",
        "eoc_max_objects",
        "eoc_depth_model",
        "eoc_gd_model",
        "eoc_unity_object_transforms_json",
        "eoc_unity_poses_npy",
        "eoc_unity_intrinsics_npy",
        "kin_text_prompt",
        "kin_static_anchor_labels",
        "kin_track_thresh",
        "kin_track_buffer",
        "kin_match_thresh",
        "kin_min_track_len",
        "kin_gd_threshold",
        "kin_gd_text_threshold",
        "kin_gd_model",
        "kin_unity_object_transforms_json",
        "kin_unity_poses_npy",
        "kin_unity_intrinsics_npy",
        "kin_unity_object_masses_json",
        # 2026-08-27: apparent-size depth (MirrorRollingBall's reflective
        # sphere) and the depth-lift/color-filter/multistage mechanics path
        # (BilliardGame) -- added to case.yaml but originally missed here,
        # so running either case through the real `run` CLI raised
        # MetricConfigError before any module code ever executed. Caught by
        # a post-fix sanity-check review, reproduced against the real CLI,
        # and fixed; kept in this whitelist test so it can't silently
        # regress the same way again.
        "eoc_unity_object_diameters_json",
        "kin_color_filter",
        "kin_color_hue_tolerance_deg",
        "kin_depth_lift",
        "kin_depth_lift_window",
        "kin_depth_model",
    ):
        assert key in METRIC_CONFIG_KEYS


def test_run_world_model_evaluation_accepts_new_module_kwargs():
    """`build_runner_kwargs`-adjacent smoke test: the runner's public
    signature must accept every whitelisted eoc_/kin_ key without raising a
    TypeError, mirroring how control_motion/shadow_consistency kwargs are
    already threaded through."""
    import inspect

    from evaluation.runner import run_world_model_evaluation

    params = inspect.signature(run_world_model_evaluation).parameters
    for key in (
        "eoc_text_prompt",
        "eoc_unity_object_transforms_json",
        "eoc_unity_object_diameters_json",
        "kin_text_prompt",
        "kin_unity_object_transforms_json",
        "kin_unity_object_masses_json",
        "kin_color_filter",
        "kin_color_hue_tolerance_deg",
        "kin_depth_lift",
        "kin_depth_lift_window",
        "kin_depth_model",
    ):
        assert key in params


def test_metric_config_keys_cover_every_artifacts_get_call():
    """Cross-check the whitelist against the modules' own source, rather
    than a second hand-maintained list: every `artifacts.get("kin_*"/
    "eoc_*", ...)` call in kinematics_mechanics.py/environment_object_
    consistency.py must name a key that's actually in METRIC_CONFIG_KEYS,
    or a case.yaml using it will pass code review and `validate` but fail
    the instant someone runs it through the real CLI -- exactly what
    happened with kin_depth_model/eoc_unity_object_diameters_json."""
    import re
    from pathlib import Path

    from evaluation.lite.adapters.registry_adapter import METRIC_CONFIG_KEYS

    root = Path(__file__).resolve().parents[1] / "evaluation" / "modules"
    pattern = re.compile(r'artifacts\.get\(\s*["\'](kin_[a-z_]+|eoc_[a-z_]+)["\']')
    missing: set[str] = set()
    for path in (root / "kinematics_mechanics.py", root / "environment_object_consistency.py"):
        text = path.read_text(encoding="utf-8")
        for key in pattern.findall(text):
            if key not in METRIC_CONFIG_KEYS:
                missing.add(key)
    assert not missing, (
        f"artifacts.get() reads these kin_*/eoc_* keys but METRIC_CONFIG_KEYS "
        f"doesn't whitelist them, so any case.yaml setting them will raise "
        f"MetricConfigError through the real runner: {sorted(missing)}"
    )
