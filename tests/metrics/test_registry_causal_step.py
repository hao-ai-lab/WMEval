from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.registry import MODULE_REGISTRY, get_available_modules


def test_causal_step_is_registered():
    assert "causal_step" in MODULE_REGISTRY
    assert "causal_step" in get_available_modules()
