from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evaluation.lite.suite_loader import SuiteValidationError, load_suite
from tests.lite_suite_test_utils import (
    DEFAULT_SUITE_YAML,
    build_default_suite_dir,
    write_reference_case,
    write_reference_free_case,
    write_suite,
)


def test_load_suite_parses_dimensions_and_aliases(tmp_path: Path) -> None:
    suite = load_suite(build_default_suite_dir(tmp_path / "suite"))

    assert suite.suite_name == "WMEval Lite Test"
    assert suite.version == "v-test"
    assert set(suite.dimensions) == {"toy_metric", "toy_promptcons"}

    metric = suite.dimensions["toy_metric"]
    assert metric.evaluator == "registry"
    assert metric.requires_reference is True
    assert metric.modules == ("lpips",)
    assert metric.metric_config["eval_fps"] == 4.0
    assert metric.aggregate is not None
    assert metric.aggregate.headline == "lpips_mean"
    assert metric.aggregate.direction == "lower"
    assert metric.aggregate.extra_means == ("lpips_median",)

    promptcons = suite.dimensions["toy_promptcons"]
    assert promptcons.evaluator == "vlm_consistency"
    assert promptcons.requires_reference is False
    # defaults merged in, model pinned by the manifest
    assert promptcons.vlm["provider"] == "openai"
    assert promptcons.vlm["model"] == "gpt-5.4"
    assert promptcons.vlm["prompt_version"] == "v1"
    assert promptcons.vlm["alpha"] == 0.6
    # case-relative rubric cache inside the checksummed bundle
    assert promptcons.vlm["rubric_dir"] == "reference/artifacts/rubrics"

    assert suite.all_case_ids() == {"scene_a/main", "scene_b/first"}
    assert suite.dimensions_for_case("scene_a/main") == ("toy_metric",)
    assert suite.legacy_case_aliases["toy_metric"]["old_scene_a"] == "scene_a/main"


def _mutated_suite(mutate) -> dict:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    mutate(payload)
    return payload


def _write_full(tmp_path: Path, payload: dict) -> Path:
    root = write_suite(tmp_path / "suite", payload)
    write_reference_case(root / "cases")
    write_reference_free_case(root / "cases")
    return root


def test_load_suite_rejects_unknown_evaluator(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"].update(evaluator="magic")
    )
    with pytest.raises(SuiteValidationError, match="evaluator must be one of"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_unwhitelisted_metric_config_key(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"]["metric_config"].update(
            lpips_smapling="typo"
        )
    )
    with pytest.raises(SuiteValidationError, match="sanctioned runner whitelist"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_vlm_dimension_without_model(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_promptcons"]["vlm"].pop("model")
    )
    with pytest.raises(SuiteValidationError, match="missing required key 'model'"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_registry_keys_on_vlm_dimension(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_promptcons"].update(modules=["lpips"])
    )
    with pytest.raises(SuiteValidationError, match="only valid for"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_vlm_key_on_registry_dimension(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"].update(vlm={"model": "x"})
    )
    with pytest.raises(SuiteValidationError, match="only valid for"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_bad_aggregate_direction(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"]["aggregate"].update(direction="up")
    )
    with pytest.raises(SuiteValidationError, match="direction must be one of"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_duplicate_cases(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"].update(
            cases=["scene_a/main", "scene_a/main"]
        )
    )
    with pytest.raises(SuiteValidationError, match="contains duplicates"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_malformed_case_id(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_metric"].update(cases=["scene_a"])
    )
    with pytest.raises(SuiteValidationError, match="'<scene>/<variant>'"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_rejects_alias_to_unknown_case(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["legacy_case_aliases"]["toy_metric"].update(bad="scene_x/none")
    )
    with pytest.raises(SuiteValidationError, match="not one of that dimension's cases"):
        load_suite(_write_full(tmp_path, payload))


def test_load_suite_requires_case_directories(tmp_path: Path) -> None:
    root = write_suite(tmp_path / "suite")
    write_reference_case(root / "cases")  # scene_b/first missing on purpose

    with pytest.raises(SuiteValidationError, match="scene_b/first"):
        load_suite(root)

    suite = load_suite(root, check_case_dirs=False)
    assert "scene_b/first" in suite.all_case_ids()


def test_load_suite_rejects_alpha_out_of_range(tmp_path: Path) -> None:
    payload = _mutated_suite(
        lambda p: p["dimensions"]["toy_promptcons"]["vlm"].update(alpha=1.5)
    )
    with pytest.raises(SuiteValidationError, match="alpha must be within"):
        load_suite(_write_full(tmp_path, payload))
