"""Per-call VLM token accounting.

Every OpenAI call made by the evaluators (prompt_scene_consistency's judge,
lpips's causal-VLM judge, physical_causality's causal_step judge) reports its
``usage`` here. The record is printed to stderr and, when
``WMEVAL_VLM_USAGE_LOG`` names a file, appended to it as one JSON line --
``tools/run_wmeval_lite_modal.py`` sets that per row and sums the records so
spend can be checked before the next batch.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # noqa: BLE001 - best effort on SDK objects
            pass
    return dict(getattr(value, "__dict__", {}) or {})


def log_usage(provider: str, model: str, usage: Any, *, api: str = "") -> dict:
    """Normalise a Responses-API or Chat-Completions usage payload and log it."""
    raw = _as_dict(usage)
    details = _as_dict(raw.get("output_tokens_details") or raw.get("completion_tokens_details"))
    cached = _as_dict(raw.get("input_tokens_details") or raw.get("prompt_tokens_details"))
    record = {
        "ts": round(time.time(), 3),
        "provider": provider,
        "model": model,
        "api": api,
        "input_tokens": int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("output_tokens") or raw.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cached_input_tokens": int(cached.get("cached_tokens") or 0),
    }
    print(
        f"[vlm-usage] {provider}/{model} in={record['input_tokens']} "
        f"out={record['output_tokens']} reasoning={record['reasoning_tokens']}",
        file=sys.stderr,
        flush=True,
    )
    path = os.environ.get("WMEVAL_VLM_USAGE_LOG")
    if path:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass
    return record
