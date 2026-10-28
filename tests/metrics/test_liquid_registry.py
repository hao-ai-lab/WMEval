# WMEval/tests/test_liquid_registry.py
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_liquid_velocity_registered():
    from evaluation.registry import get_available_modules

    assert "liquid_velocity" in get_available_modules()


def test_liquid_volume_conservation_registered():
    from evaluation.registry import get_available_modules

    assert "liquid_volume_conservation" in get_available_modules()


def test_liquid_surface_orientation_registered():
    from evaluation.registry import get_available_modules

    assert "liquid_surface_orientation" in get_available_modules()


def test_slosh_spectrum_registered():
    from evaluation.registry import get_available_modules

    assert "slosh_spectrum" in get_available_modules()


def test_liquid_container_scene_in_registry():
    from evaluation.scene_prompt_registry import SCENE_CONFIG_REGISTRY

    assert "liquid_container" in SCENE_CONFIG_REGISTRY
    cfg = SCENE_CONFIG_REGISTRY["liquid_container"]
    assert cfg["default_sam2_prompt"] == "the liquid inside the bottle"
    applicable = cfg["applicable_modules"]
    for name in (
        "liquid_velocity",
        "liquid_volume_conservation",
        "liquid_surface_orientation",
        "slosh_spectrum",
    ):
        assert name in applicable
    # No pour-related fields remain.
    for forbidden in (
        "default_pour_threshold_deg",
        "default_k_pour",
        "default_fill_calibration_method",
        "default_bottle_local_volume_ml",
    ):
        assert forbidden not in cfg
