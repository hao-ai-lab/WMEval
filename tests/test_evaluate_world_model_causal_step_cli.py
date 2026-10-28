from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAUSALSTEP_ROOT = ROOT / "imports" / "CausalStep"
if str(CAUSALSTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAUSALSTEP_ROOT))

from tools.legacy.evaluate_world_model_causal_step import _run_manifest


def _tiny_qa_dataset(video_id: str = "vid1") -> dict:
    return {
        "videos": [
            {
                "video_id": video_id,
                "segments": [
                    {"index": 1, "start_time": 0.0, "end_time": 0.33},
                    {"index": 2, "start_time": 0.33, "end_time": 0.66},
                    {"index": 3, "start_time": 0.66, "end_time": 1.0},
                ],
                "descriptive_qas": [
                    {"segment_index": i, "question": f"Q{i}?", "choices": ["A) yes", "B) no"], "correct_answer": "A"}
                    for i in (1, 2, 3)
                ],
                "causal_qas": [
                    {"segment_index": i, "question": f"C{i}?", "choices": ["A) cause", "B) other"], "correct_answer": "A"}
                    for i in (2, 3)
                ],
            }
        ]
    }


def test_run_manifest_isolates_per_trajectory_errors(tmp_path: Path):
    """A bad trajectory (missing QA dataset) must not halt the batch. The failing
    entry is reported with {"error": ...} so the caller's "ok" filter works."""
    good_qa = tmp_path / "good_qa.json"
    good_qa.write_text(json.dumps(_tiny_qa_dataset("vid1")))

    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": "vid1",
                        "unity_video": str(fake_video),
                        "wm_video": str(fake_video),
                        "qa_dataset": str(good_qa),
                    }
                ),
                json.dumps(
                    {
                        "video_id": "vid_missing",
                        "unity_video": str(fake_video),
                        "wm_video": str(fake_video),
                        "qa_dataset": str(tmp_path / "does_not_exist.json"),
                    }
                ),
                "",  # blank line — must be skipped
                "{not json",  # malformed JSON — must be captured, not crash
            ]
        )
    )

    cache_root = tmp_path / "cache"
    per_video = _run_manifest(
        manifest_path=manifest,
        cache_root=cache_root,
        judge="fake_always_a",
        causalstep_root=None,
    )

    assert len(per_video) == 3, f"expected 3 entries (2 manifest + 1 malformed), got {len(per_video)}"

    good = [r for r in per_video if "error" not in r]
    bad = [r for r in per_video if "error" in r]
    assert len(good) == 1, f"expected 1 good trajectory, got {good}"
    assert good[0]["video_id"] == "vid1"
    assert "delta" in good[0]

    assert len(bad) == 2, f"expected 2 failures (missing QA + malformed JSON), got {bad}"
    # Failing trajectory with missing QA should preserve video_id for operator triage.
    missing_qa_failures = [r for r in bad if r.get("video_id") == "vid_missing"]
    assert missing_qa_failures, f"no entry carried video_id='vid_missing'; got {bad}"
