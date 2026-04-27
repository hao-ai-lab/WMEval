from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "evaluate_world_model_liquid_pour.py"


def test_cli_with_nonexistent_inputs_writes_error_row(tmp_path):
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
         "--liquid_cache_root", str(cache),
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
    assert row.get("manifest_line") == 1
    assert payload["aggregate"]["n_errors"] == 1


def test_cli_malformed_jsonl_records_line_number(tmp_path):
    """A malformed JSONL line must produce an error row carrying its 1-based line number."""
    if not CLI.exists():
        pytest.skip("CLI not yet implemented")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"video_id": "good", "unity_video": "x", "wm_video": "x", "unity_log": "x"}\n'
        "this is not json\n"
    )
    out = tmp_path / "out.json"
    cache = tmp_path / "cache"

    res = subprocess.run(
        [sys.executable, str(CLI),
         "--manifest", str(manifest),
         "--out", str(out),
         "--liquid_cache_root", str(cache),
         "--modules", "liquid_state"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    payload = json.loads(out.read_text())
    rows = payload["per_video"]
    assert len(rows) == 2
    bad = next(r for r in rows if r.get("manifest_line") == 2)
    assert "error" in bad and "bad JSON" in bad["error"]
    assert payload["aggregate"]["n_errors"] == 2


def test_cli_status_json_to_stdout(tmp_path):
    """Stdout carries a small status JSON, not the full report."""
    if not CLI.exists():
        pytest.skip("CLI not yet implemented")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")  # empty manifest: 0 trajectories
    out = tmp_path / "out.json"
    cache = tmp_path / "cache"

    res = subprocess.run(
        [sys.executable, str(CLI),
         "--manifest", str(manifest),
         "--out", str(out),
         "--liquid_cache_root", str(cache),
         "--modules", "liquid_state"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    status = json.loads(res.stdout)
    assert set(status.keys()) == {"out", "n_trajectories", "n_errors"}
    assert status["n_trajectories"] == 0
    assert status["n_errors"] == 0
