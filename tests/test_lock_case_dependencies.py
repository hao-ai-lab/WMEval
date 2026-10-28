from __future__ import annotations

import importlib.metadata as md
import platform
from pathlib import Path

import pytest
import yaml

from tests.lite_suite_test_utils import build_default_suite_dir
from tools import lock_case_dependencies as lcd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE = REPO_ROOT / "lite"


def _lock(case_dir: Path) -> dict:
    raw = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    return raw["evaluator_lock"]["dependency_versions"]


def test_locks_record_installed_versions_per_dimension_and_are_idempotent(tmp_path: Path) -> None:
    suite = build_default_suite_dir(tmp_path / "suite")
    assert lcd.main(["--suite", str(suite), "--no-rebuild"]) == 0

    registry = _lock(suite / "cases" / "scene_a" / "main")  # lpips, temporal sampling
    assert registry["python"] == platform.python_version()
    assert registry["lock_source"] == lcd.LOCK_SOURCE
    assert registry["torch"] == md.version("torch")
    assert registry["lpips"] == md.version("lpips")
    assert "openai" not in registry  # only the causal VLM sampling modes import it
    assert registry["asset:lpips_vgg"] == lcd.load_assets_lock()["lpips_vgg"]["linear_sha256"]

    vlm = _lock(suite / "cases" / "scene_b" / "first")
    assert {"pydantic", "requests", "numpy"} <= set(vlm)
    assert "torch" not in vlm and not any(key.startswith("asset:") for key in vlm)

    before = {p: p.read_bytes() for p in suite.rglob("case.yaml")}
    assert lcd.main(["--suite", str(suite), "--no-rebuild"]) == 0
    assert {p: p.read_bytes() for p in suite.rglob("case.yaml")} == before
    assert lcd.main(["--suite", str(suite), "--check"]) == 0


def test_foreign_locks_are_kept_unless_forced(tmp_path: Path) -> None:
    suite = build_default_suite_dir(tmp_path / "suite")
    case_yaml = suite / "cases" / "scene_a" / "main" / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    raw["evaluator_lock"]["dependency_versions"] = {"python": "3.12", "gpu": "pinned"}
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    lcd.main(["--suite", str(suite), "--no-rebuild"])
    assert _lock(case_yaml.parent) == {"python": "3.12", "gpu": "pinned"}
    lcd.main(["--suite", str(suite), "--no-rebuild", "--force"])
    assert _lock(case_yaml.parent)["lock_source"] == lcd.LOCK_SOURCE


def test_changed_cases_are_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.build_lite_case as build

    calls: list[list[str]] = []
    monkeypatch.setattr(build, "main", lambda argv: calls.append(argv) or 0)
    suite = build_default_suite_dir(tmp_path / "suite")
    lcd.main(["--suite", str(suite)])
    assert sorted(Path(p).name for p in calls[0]) == ["first", "main"]
    calls.clear()
    lcd.main(["--suite", str(suite)])
    assert calls == []


def test_config_gated_imports() -> None:
    assert "openpiv" not in lcd.module_requirements("liquid_velocity", {}).imports
    piv = lcd.module_requirements("liquid_velocity", {"liquid_flow_estimator": "openpiv"})
    assert "openpiv" in piv.imports and "raft_large" not in piv.assets
    causal = lcd.module_requirements("lpips", {"lpips_sampling": "causal_vlm_collision"})
    assert "openai" in causal.imports and "raft_large" in causal.assets
    depth = lcd.module_requirements("control_motion", {"synthetic_use_depth": True})
    assert "depth_anything_relative_small" in depth.assets
    assert "transformers" not in lcd.module_requirements("control_motion", {}).imports
    kin = lcd.module_requirements("kinematics_mechanics", {"kin_depth_lift": True, "kin_depth_model": lcd.DEPTH_INDOOR})
    assert kin.assets == {"grounding_dino", "depth_anything_metric_indoor"}
    with pytest.raises(KeyError):
        lcd.module_requirements("not_a_module", {})


def test_live_suite_locks_cover_each_cases_dimensions() -> None:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(SUITE)
    for case_id, dimensions in lcd.case_dimensions(suite).items():
        lock = dict(load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids()).evaluator_lock["dependency_versions"])
        assert lock, case_id
        assert lock.get("lock_source") == lcd.LOCK_SOURCE, case_id  # every live lock is tool-written
        case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
        req = lcd.Requirements()
        for dimension in dimensions:
            req.update(lcd.dimension_requirements(dimension, case))
        assert {k[len(lcd.ASSET_PREFIX):] for k in lock if k.startswith(lcd.ASSET_PREFIX)} == req.assets, case_id
        assert lock["python"] and lock["platform"]
        packages = {k for k in lock if k not in lcd.META_KEYS and not k.startswith(lcd.ASSET_PREFIX)}
        assert {"numpy"} <= packages, case_id
