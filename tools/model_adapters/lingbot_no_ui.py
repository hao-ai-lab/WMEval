#!/usr/bin/env python3
"""Run LingBot Base Cam without the diagnostic keyboard/camera UI overlay."""
from __future__ import annotations

import os
import sys
from pathlib import Path


SOURCE = Path(os.environ.get("LINGBOT_SOURCE", "/workspace/model_sources/lingbot-world"))
sys.path.insert(0, str(SOURCE))

import generate as upstream  # noqa: E402


_original_generate = upstream.wan.WanI2V.generate


def _generate_without_ui(self, *args, **kwargs):
    kwargs["vis_ui"] = False
    return _original_generate(self, *args, **kwargs)


upstream.wan.WanI2V.generate = _generate_without_ui


if __name__ == "__main__":
    upstream.generate(upstream._parse_args())
