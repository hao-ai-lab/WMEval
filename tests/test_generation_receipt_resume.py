import json
from pathlib import Path

import pytest

from tools.generation.model_adapters.lingbot_batch import (
    acquire_generation_claim,
    generation_complete as lingbot_adapter_complete,
    release_generation_claim,
)
from tools.generation.run_lingbot_table7_batch import generation_complete as lingbot_complete
from tools.generation.run_matrix_game2 import generation_complete as matrix2_complete
from tools.generation.run_matrix_game3 import generation_complete as matrix3_complete


@pytest.mark.parametrize(
    "checker", [lingbot_adapter_complete, lingbot_complete, matrix2_complete, matrix3_complete]
)
def test_resume_requires_complete_matching_receipt(tmp_path: Path, checker) -> None:
    candidate = tmp_path / "run0" / "candidate.mp4"
    candidate.parent.mkdir()
    candidate.write_bytes(b"video")
    assert not checker(candidate)

    receipt = candidate.parent / "generation_receipt.json"
    receipt.write_text(
        json.dumps({"status": "complete", "candidate_video_sha256": "wrong"}),
        encoding="utf-8",
    )
    assert not checker(candidate)

    import hashlib

    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "candidate_video_sha256": hashlib.sha256(b"video").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert checker(candidate)


def test_lingbot_adapter_resume_checks_job_identity(tmp_path: Path) -> None:
    candidate = tmp_path / "run7" / "candidate.mp4"
    candidate.parent.mkdir()
    candidate.write_bytes(b"video")

    import hashlib

    receipt = {
        "status": "complete",
        "candidate_video_sha256": hashlib.sha256(b"video").hexdigest(),
        "model": "lingbot_world_cam",
        "case_id": "table7/MirrorRollingBall/perturb_in",
        "generation_repeat": "run7",
        "seed": 42007,
    }
    (candidate.parent / "generation_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    job = {
        "case_id": receipt["case_id"],
        "repeat": receipt["generation_repeat"],
        "receipt_static": {
            "model": receipt["model"],
            "case_id": receipt["case_id"],
            "generation_repeat": receipt["generation_repeat"],
            "seed": receipt["seed"],
        },
    }
    assert lingbot_adapter_complete(candidate, job)

    job["receipt_static"]["seed"] = 999
    assert not lingbot_adapter_complete(candidate, job)


def test_lingbot_generation_claim_is_exclusive_and_recoverable(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "run7" / "candidate.mp4"
    first = acquire_generation_claim(candidate)
    try:
        with pytest.raises(BlockingIOError):
            acquire_generation_claim(candidate, blocking=False)
    finally:
        release_generation_claim(first)

    second = acquire_generation_claim(candidate, blocking=False)
    release_generation_claim(second)
    assert (candidate.parent / "generation_claim.lock").is_file()
