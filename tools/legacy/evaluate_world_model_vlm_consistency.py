#!/usr/bin/env python3
"""VLM-judge consistency evaluation entry point.

Wraps :mod:`evaluation.modules.vlm_consistency.batch`, which exposes two
subcommands:

  generate_rubrics   Dual-agent rubric generation, cached per scene.
  evaluate           Score all videos under tests/vlm_judge/data/* against the
                     cached rubrics with a single VLM call per video.

Examples:
    # 1) Generate rubrics once per (provider, model)
    python tools/legacy/evaluate_world_model_vlm_consistency.py generate_rubrics \\
        --provider openai --model gpt-5.4

    # 2) Evaluate all generator-model directories
    python tools/legacy/evaluate_world_model_vlm_consistency.py evaluate \\
        --provider openai --model gpt-5.4 \\
        --output reports/world_model_vlm_consistency/results.json

Shell wrapper and pytest (repo root):
    bash scripts/evaluate_world_model_vlm_consistency.sh
    python -m pytest tests/test_vlm_consistency_batch_helpers.py \\
        tests/test_vlm_consistency_batch_cli.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.modules.vlm_consistency.batch import main


if __name__ == "__main__":
    sys.exit(main())
