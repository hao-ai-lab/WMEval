"""tools/fixture_backend.py drives the real runner without a model, GPU or API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.lite.candidate_discovery import load_candidate_manifest
from evaluation.lite.case_loader import load_case
from evaluation.lite.suite_loader import load_suite
from tools import run_wmeval_lite
from tools.fixture_backend import FixtureError, degrade_video, generate_fixtures, main


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "lite"
# Two of the smallest reference videos, both in the `lpips` dimension.
SMALL_CASES = ["BallRoller/ctrl", "LiquidContainer/fluid_rest"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(case_id: str) -> Path:
    suite = load_suite(SUITE)
    case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
    assert case.reference_video is not None
    return case.reference_video


def _runner_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    capsys.readouterr()
    assert run_wmeval_lite.main(argv) == 0
    out = capsys.readouterr().out
    return json.loads(out[out.index("{\n"):])


@pytest.mark.integration("case_data")
def test_identity_fixtures_drive_runner_validate_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = generate_fixtures(
        output_root=tmp_path / "fixtures",
        suite_root=SUITE,
        dimensions=["lpips"],
        cases=SMALL_CASES,
        log=lambda _msg: None,
    )
    assert result.skipped == []
    assert result.manifest == tmp_path / "fixtures" / "fixture_candidates.jsonl"
    assert result.model_root == tmp_path / "fixtures" / "fixture"

    raw_rows = [json.loads(line) for line in result.manifest.read_text().splitlines()]
    assert [list(row) for row in raw_rows] == [
        ["model", "case_id", "seed", "candidate_video"]
    ] * len(SMALL_CASES)
    rows = load_candidate_manifest(result.manifest, allowed_cases=SMALL_CASES)
    assert [(r.model, r.case_id, r.seed) for r in rows] == [
        ("fixture", case_id, "0") for case_id in SMALL_CASES
    ]
    for row in rows:
        expected = result.model_root / row.case_id / "0.mp4"
        assert row.candidate_video == expected
        assert _sha256(expected) == _sha256(_reference(row.case_id))

    assert run_wmeval_lite.main(
        ["validate", "--suite", str(SUITE), "--skip-checksums"]
    ) == 0

    plans = []
    for source in (
        ["--candidate-manifest", str(result.manifest)],
        ["--model-root", str(result.model_root)],
    ):
        plan = _runner_json(
            capsys,
            [
                "run",
                "--suite", str(SUITE),
                *source,
                "--dimensions", "lpips",
                "--output-dir", str(tmp_path / "report"),
                "--dry-run",
            ],
        )
        assert plan["dry_run"] is True
        assert plan["num_work_items"] == len(SMALL_CASES)
        items = plan["work_items"]
        assert {item["case_id"] for item in items} == set(SMALL_CASES)
        assert all(item["model"] == "fixture" for item in items)
        assert all(item["candidate_exists"] and item["reference_exists"] for item in items)
        plans.append(sorted((i["case_id"], i["candidate_video"]) for i in items))
    assert plans[0] == plans[1]
    # dry runs write nothing
    assert not (tmp_path / "report").exists()


def test_reference_free_cases_are_skipped_not_faked(tmp_path: Path) -> None:
    result = generate_fixtures(
        output_root=tmp_path,
        suite_root=SUITE,
        dimensions=["prompt_scene_consistency"],
        cases=["ChessBoard/prompt_first"],
        log=lambda _msg: None,
    )
    assert result.written == []
    assert result.skipped == [
        {
            "case_id": "ChessBoard/prompt_first",
            "reason": "case has no reference video (reference-free)",
        }
    ]
    assert result.manifest.read_text() == ""


def test_selection_errors(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="unknown dimension"):
        generate_fixtures(output_root=tmp_path, suite_root=SUITE, dimensions=["nope"])
    with pytest.raises(FixtureError, match="not in the selected dimension"):
        generate_fixtures(
            output_root=tmp_path,
            suite_root=SUITE,
            dimensions=["fluid_mechanics"],
            cases=["BallRoller/ctrl"],
        )
    assert main(["--output-root", str(tmp_path), "--model", "../x"]) == 2


def test_degraded_fixture_is_a_deterministic_perturbation(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    reference = _reference("BallRoller/ctrl")
    probe = tmp_path / "probe.mp4"
    reason = degrade_video(reference, probe)
    if reason is not None:
        pytest.skip(reason)

    runs = []
    for name in ("a", "b"):
        result = generate_fixtures(
            output_root=tmp_path / name,
            suite_root=SUITE,
            cases=["BallRoller/ctrl"],
            mode="degraded",
            seeds=["0", "1"],
            log=lambda _msg: None,
        )
        assert [row["seed"] for row in result.written] == ["0", "1"]
        runs.append(result.model_root / "BallRoller/ctrl/0.mp4")
    assert _sha256(runs[0]) == _sha256(runs[1])
    assert _sha256(runs[0]) != _sha256(reference)

    def frame_count_and_mean(path: Path) -> tuple[int, float]:
        capture = cv2.VideoCapture(str(path))
        count, total = 0, 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            total += float(frame.mean())
        capture.release()
        return count, total / max(count, 1)

    ref_frames, ref_mean = frame_count_and_mean(reference)
    out_frames, out_mean = frame_count_and_mean(runs[0])
    assert out_frames == ref_frames > 0
    assert out_mean > ref_mean  # brightness shift survived re-encoding
