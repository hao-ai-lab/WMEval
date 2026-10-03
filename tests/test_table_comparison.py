from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.table_comparison import ComparisonError, compare_results, load_baseline, result_template


BASELINE = Path(__file__).resolve().parents[1] / "experiments/tables_4_7/paper_baseline.yaml"


def test_paper_baseline_sources_and_table_shape_are_locked() -> None:
    baseline = load_baseline(BASELINE)
    assert set(baseline["tables"][4]["cases"]) == {
        "table4/RacingCar/first",
        "table4/RacingCar/third",
        "table4/BilliardGame/first",
        "table4/BilliardGame/third",
    }
    assert len(baseline["tables"][7]["cases"]) == 12
    assert len(result_template(baseline)["results"]) == 16


def test_comparison_uses_five_runs_and_sample_standard_deviation() -> None:
    baseline = load_baseline(BASELINE)
    rerun = result_template(baseline)
    for metrics in rerun["results"].values():
        for values in metrics.values():
            values.update({f"run{index}": float(index + 1) for index in range(5)})
    comparison = compare_results(baseline, rerun)
    assert comparison["complete"] is True
    complete = next(row for row in comparison["rows"] if row["status"] == "complete")
    assert complete["rerun_mean"] == 3.0
    assert complete["rerun_std"] == pytest.approx(1.5811388300841898)


def test_incomplete_results_fail_closed() -> None:
    baseline = load_baseline(BASELINE)
    rerun = result_template(baseline)
    with pytest.raises(ComparisonError, match="missing five numeric runs"):
        compare_results(baseline, rerun)
    partial = compare_results(baseline, rerun, allow_incomplete=True)
    assert partial["complete"] is False
    assert partial["pending_count"] == 48
