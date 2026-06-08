from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.lite_refs import (
    parse_lite_manifest,
    resolve_candidate_target,
    resolve_repo_path,
    repo_root_from_manifest,
    run_lite_suite,
)


def test_resolve_candidate_target_requires_dimension_and_test_case() -> None:
    assert resolve_candidate_target(
        {"dimension": "causal_lpips", "sample_id": "sample_scene"}
    ) == ("causal_lpips", "sample_scene")
    assert resolve_candidate_target(
        {"dimension_id": "controllability", "test_case_id": "ball_roller_third_person"}
    ) == ("controllability", "ball_roller_third_person")
    with pytest.raises(ValueError):
        resolve_candidate_target({"sample_id": "sample_scene"})
    with pytest.raises(ValueError):
        resolve_candidate_target({"dimension": "causal_lpips"})


def test_parse_lite_manifest_is_dimension_first(tmp_path: Path) -> None:
    bundle = tmp_path / "eval_sets" / "wmeval_lite_v1"
    bundle.mkdir(parents=True)
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "WMEval Lite Test",
                "version": "test",
                "dimensions": {
                    "causal_lpips": {
                        "display_name": "Causal LPIPS",
                        "metric_config": {"modules": ["lpips"]},
                        "test_cases": [
                            {
                                "test_case_id": "mirror_ball_third_person",
                                "display_name": "MirrorBall",
                                "legacy_ids": ["sample_scene"],
                                "prompt": "prompt",
                                "perspective": "third_person",
                                "reference_video": "eval_sets/wmeval_lite_v1/unity_references/mirror_ball_third_person_unity.mp4",
                                "reference_video_sha256": None,
                                "reference_artifacts": {},
                                "reference_metrics": {},
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_lite_manifest(manifest_path)

    assert parsed["dimensions"][0].dimension_id == "causal_lpips"
    case = parsed["test_cases"][0]
    assert case.test_case_id == "mirror_ball_third_person"
    assert case.perspective == "third_person"


def test_lite_suite_dry_run_resolves_bundled_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "eval_sets" / "wmeval_lite_v1" / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = parse_lite_manifest(manifest_path)
    manifest_dir = Path(manifest["manifest_path"]).parent
    root = repo_root_from_manifest(manifest_path)
    case = next(
        tc for tc in manifest["test_cases"] if tc.test_case_id == "mirror_ball_third_person"
    )
    resolved = resolve_repo_path(
        case.reference_video,
        repo_root=root,
        manifest_dir=manifest_dir,
    )
    assert resolved is not None
    assert resolved.exists()
    assert case.perspective == "third_person"


def test_lite_suite_dry_run_with_fake_candidate(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    ref_dir = repo_root / "eval_sets" / "wmeval_lite_v1" / "unity_references"
    ref_dir.mkdir(parents=True)
    ref = ref_dir / "mirror_ball_third_person_unity.mp4"
    ref.write_bytes(b"fake")

    manifest_path = repo_root / "eval_sets" / "wmeval_lite_v1" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "WMEval Lite Test",
                "version": "test",
                "dimensions": {
                    "causal_lpips": {
                        "display_name": "Causal LPIPS",
                        "metric_config": {"modules": ["lpips"]},
                        "test_cases": [
                            {
                                "test_case_id": "mirror_ball_third_person",
                                "display_name": "MirrorBall",
                                "legacy_ids": ["sample_scene"],
                                "prompt": "prompt",
                                "perspective": "third_person",
                                "reference_video": "eval_sets/wmeval_lite_v1/unity_references/mirror_ball_third_person_unity.mp4",
                                "reference_video_sha256": None,
                                "reference_artifacts": {},
                                "reference_metrics": {},
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cand = tmp_path / "candidate.mp4"
    cand.write_bytes(b"fake")
    candidate_jsonl = tmp_path / "candidates.jsonl"
    candidate_jsonl.write_text(
        json.dumps(
            {
                "dimension": "causal_lpips",
                "sample_id": "sample_scene",
                "world_model_name": "genie",
                "run_id": "run_01",
                "world_model_video": str(cand),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate = run_lite_suite(
        manifest_json=manifest_path,
        candidate_jsonl=candidate_jsonl,
        output_dir=tmp_path / "out",
        dry_run=True,
    )

    assert aggregate["label"] == "WMEval Lite Test w/ bundled reference"
    metrics = aggregate["dimensions"]["causal_lpips"]["test_case_metrics"][0]
    assert metrics["test_case_id"] == "mirror_ball_third_person"
    assert metrics["display_name"] == "MirrorBall"
