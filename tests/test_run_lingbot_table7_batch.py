import json
import sys
from pathlib import Path

import pytest

from tools import run_lingbot_table7_batch as runner


def _queue_row(index: int) -> dict[str, object]:
    return {
        "case_id": f"table7/Scene/case_{index}",
        "generation_repeat": f"run{index}",
        "candidate_video": f"outputs/case_{index}/candidate.mp4",
        "conditioning_image": f"inputs/case_{index}/image.png",
        "poses": f"inputs/case_{index}/poses.npy",
        "intrinsics": f"inputs/case_{index}/intrinsics.npy",
        "environment_prompt": f"prompt {index}",
    }


def test_single_gpu_dry_run_uses_disjoint_shard(tmp_path: Path, monkeypatch, capsys) -> None:
    queue = tmp_path / "generation_queue.jsonl"
    queue.write_text(
        "".join(json.dumps(_queue_row(index)) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "QUEUE", queue)
    monkeypatch.setattr(runner, "BATCHES", tmp_path / "batches")
    monkeypatch.setattr(runner, "PYTHON", Path("/env/bin/python"))
    monkeypatch.setattr(runner, "SOURCE", Path("/source"))
    monkeypatch.setattr(runner, "CHECKPOINT", Path("/checkpoint"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lingbot_table7_batch.py",
            "--world-size",
            "1",
            "--gpu-index",
            "3",
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--dry-run",
        ],
    )

    assert runner.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generation_count"] == 2
    assert report["command"][0] == "/env/bin/python"
    assert "torch.distributed.run" not in report["command"]
    jobs_path = tmp_path / "batches/world1_shard1-of-2/jobs.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
    assert [job["case_id"] for job in jobs] == ["table7/Scene/case_1", "table7/Scene/case_3"]


def test_full_shard_excludes_cross_shard_early_reservations(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    queue = tmp_path / "generation_queue.jsonl"
    queue.write_text(
        "".join(json.dumps(_queue_row(index)) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    batches = tmp_path / "batches"
    reserved = batches / "world1_early_adaptive_other_gpu" / "jobs.jsonl"
    reserved.parent.mkdir(parents=True)
    reserved.write_text(
        json.dumps({"case_id": "table7/Scene/case_1", "repeat": "run1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "QUEUE", queue)
    monkeypatch.setattr(runner, "BATCHES", batches)
    monkeypatch.setattr(runner, "PYTHON", Path("/env/bin/python"))
    monkeypatch.setattr(runner, "SOURCE", Path("/source"))
    monkeypatch.setattr(runner, "CHECKPOINT", Path("/checkpoint"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lingbot_table7_batch.py",
            "--world-size",
            "1",
            "--gpu-index",
            "7",
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--dry-run",
        ],
    )

    assert runner.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generation_count"] == 1
    assert report["reserved_excluded"] == 1
    jobs_path = batches / "world1_shard1-of-2" / "jobs.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
    assert [(job["case_id"], job["repeat"]) for job in jobs] == [
        ("table7/Scene/case_3", "run3")
    ]


def test_eight_gpu_mode_must_start_at_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_lingbot_table7_batch.py", "--world-size", "8", "--gpu-index", "1"])
    with pytest.raises(SystemExit, match="must start"):
        runner.main()


def test_rejects_unsafe_batch_label(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lingbot_table7_batch.py",
            "--world-size",
            "1",
            "--gpu-index",
            "7",
            "--batch-label",
            "../overwrite",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit, match="--batch-label must contain"):
        runner.main()


def test_exact_job_keys_build_isolated_tail_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    queue = tmp_path / "generation_queue.jsonl"
    queue.write_text(
        "".join(json.dumps(_queue_row(index)) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "QUEUE", queue)
    monkeypatch.setattr(runner, "BATCHES", tmp_path / "batches")
    monkeypatch.setattr(runner, "PYTHON", Path("/env/bin/python"))
    monkeypatch.setattr(runner, "SOURCE", Path("/source"))
    monkeypatch.setattr(runner, "CHECKPOINT", Path("/checkpoint"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lingbot_table7_batch.py",
            "--world-size",
            "1",
            "--gpu-index",
            "7",
            "--batch-label",
            "tail_helper_gpu7",
            "--job-key",
            "table7/Scene/case_1::run1",
            "--job-key",
            "table7/Scene/case_3::run3",
            "--dry-run",
        ],
    )

    assert runner.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generation_count"] == 2
    jobs_path = tmp_path / "batches/world1_tail_helper_gpu7/jobs.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
    assert [(job["case_id"], job["repeat"]) for job in jobs] == [
        ("table7/Scene/case_1", "run1"),
        ("table7/Scene/case_3", "run3"),
    ]


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--job-key", "bad", "--batch-label", "tail"], "CASE_ID::RUN"),
        (
            ["--job-key", "table7/Scene/missing::run9", "--batch-label", "tail"],
            "unknown --job-key",
        ),
        (["--job-key", "table7/Scene/case_0::run0"], "requires an isolated"),
    ],
)
def test_exact_job_keys_fail_closed(
    tmp_path: Path, monkeypatch, arguments: list[str], message: str
) -> None:
    queue = tmp_path / "generation_queue.jsonl"
    queue.write_text(json.dumps(_queue_row(0)) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "QUEUE", queue)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lingbot_table7_batch.py",
            "--world-size",
            "1",
            "--gpu-index",
            "7",
            "--dry-run",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit, match=message):
        runner.main()
