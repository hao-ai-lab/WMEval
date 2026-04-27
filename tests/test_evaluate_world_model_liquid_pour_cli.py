from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "evaluate_world_model_liquid_pour.py"


def test_cli_dry_run_with_missing_video_writes_error_row(tmp_path):
    """Manifest with one valid-shape entry whose video files don't exist:
    the CLI must still write a per_video row with `error`, exit 0."""
    if not CLI.exists():
        pytest.skip("CLI not yet implemented")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "video_id": "liquid_001",
        "unity_video": str(tmp_path / "missing_unity.mp4"),
        "wm_video": str(tmp_path / "missing_wm.mp4"),
        "unity_log": str(tmp_path / "missing_log.json"),
    }) + "\n")
    out = tmp_path / "out.json"
    cache = tmp_path / "cache"

    res = subprocess.run(
        [sys.executable, str(CLI),
         "--manifest", str(manifest),
         "--out", str(out),
         "--liquid-cache-root", str(cache),
         "--modules", "liquid_state"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    payload = json.loads(out.read_text())
    assert "per_video" in payload
    assert len(payload["per_video"]) == 1
    row = payload["per_video"][0]
    assert row.get("video_id") == "liquid_001"
    assert "error" in row
    assert payload["aggregate"]["n_errors"] == 1
