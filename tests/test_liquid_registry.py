# WMEval/tests/test_liquid_registry.py
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_liquid_state_registered():
    from evaluation.registry import get_available_modules

    modules = get_available_modules()
    assert "liquid_state" in modules


def test_liquid_container_scene_in_registry():
    from evaluation.scene_prompt_registry import SCENE_CONFIG_REGISTRY

    assert "liquid_container" in SCENE_CONFIG_REGISTRY
    cfg = SCENE_CONFIG_REGISTRY["liquid_container"]
    assert cfg["default_pour_threshold_deg"] == 30.0
    assert cfg["default_k_pour"] == 0.05
    assert cfg["default_sam2_prompt"] == "the liquid inside the bottle"
    assert "liquid_state" in cfg["applicable_modules"]


def test_liquid_velocity_registered():
    from evaluation.registry import get_available_modules

    assert "liquid_velocity" in get_available_modules()
