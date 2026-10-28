from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.dataset import check_prompt_human_alignment as checker
def _fixture() -> dict[str, Any]:
    return json.loads(checker.DEFAULT_FIXTURE.read_text(encoding="utf-8"))


def test_fixture_reproduces_the_paper_alignment() -> None:
    fixture = _fixture()
    human = fixture["human_overall"]
    judge = fixture["paper_judge_overall"]
    generative = ["Genie", "LingBot", "MatrixGame3", "HY"]

    for name, models in (("all_models", ["Unity", *generative]), ("generative_models", generative)):
        result = checker.alignment(judge, human, models)
        expected = fixture["expected"][name]
        assert result["spearman_rho"] == expected["spearman_rho"]
        assert result["kendall_tau_b"] == expected["kendall_tau_b"]

    # pooled human scores as published in the paper appendix
    pooled = checker.pooled(human, ["Unity", *generative])
    assert {m: round(v, 3) for m, v in pooled.items()} == {
        "Unity": 0.904, "Genie": 0.780, "LingBot": 0.843, "MatrixGame3": 0.505, "HY": 0.522,
    }


LITE_NAMES = {"Genie": "genie", "LingBot": "lingbot", "MatrixGame3": "mg3", "HY": "hy"}


def _rows(judge: dict[str, dict[str, float]], *, with_unity: bool = True) -> list[dict[str, Any]]:
    rows = []
    for case, scores in judge.items():
        for study, lite in LITE_NAMES.items():
            for seed in ("42", "1", "2", "3", "4"):
                metrics = {"overall_score": scores[study]}
                if with_unity:
                    metrics["overall_score_unity"] = scores["Unity"]
                rows.append({"dimension": "prompt_scene_consistency", "status": "completed",
                             "case_id": case, "model": lite, "seed": seed, "metrics": metrics})
    return rows


def _run(tmp_path: Path, rows: list[dict[str, Any]]) -> int:
    results = tmp_path / "per_case_results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args = ["--results", str(results)]
    for study, lite in LITE_NAMES.items():
        args += ["--model", f"{study}={lite}"]
    return checker.main(args)


def test_checker_passes_on_the_paper_judge_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, _rows(_fixture()["paper_judge_overall"])) == 0
    out = capsys.readouterr().out
    assert "all_models: rho=0.9 tau_b=0.8" in out
    assert "generative_models: rho=0.8" in out


def test_checker_fails_when_the_judge_disagrees_with_humans(tmp_path: Path) -> None:
    judge = _fixture()["paper_judge_overall"]
    # swap the best- and worst-rated generative models in every case
    flipped = {
        case: {**scores, "LingBot": scores["MatrixGame3"], "MatrixGame3": scores["LingBot"]}
        for case, scores in judge.items()
    }
    assert _run(tmp_path, _rows(flipped)) == 1


def test_checker_without_frozen_unity_checks_generative_models_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, _rows(_fixture()["paper_judge_overall"], with_unity=False)) == 0
    out = capsys.readouterr().out
    assert "all_models" not in out
    assert "generative_models: rho=0.8" in out


def test_checker_refuses_incomplete_runs(tmp_path: Path) -> None:
    rows = [r for r in _rows(_fixture()["paper_judge_overall"])
            if not (r["model"] == "hy" and r["case_id"] == "RacingCar/prompt_third")]
    assert _run(tmp_path, rows) == 2
