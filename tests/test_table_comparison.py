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
    # comparison semantics do not depend on the paper checkout being present
    baseline = load_baseline(BASELINE, missing_sources="warn", warn=lambda _m: None)
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
    baseline = load_baseline(BASELINE, missing_sources="warn", warn=lambda _m: None)
    rerun = result_template(baseline)
    with pytest.raises(ComparisonError, match="missing five numeric runs"):
        compare_results(baseline, rerun)
    partial = compare_results(baseline, rerun, allow_incomplete=True)
    assert partial["complete"] is False
    assert partial["pending_count"] == 48


def _baseline_with_sources(tmp_path: Path, sources: dict[str, dict[str, str]]) -> Path:
    import yaml

    data = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    data["sources"] = sources
    path = tmp_path / "paper_baseline.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_missing_paper_sources_warn_instead_of_failing_when_asked(tmp_path: Path) -> None:
    import hashlib

    present = tmp_path / "table.tex"
    present.write_text("\\begin{tabular}\n", encoding="utf-8")
    path = _baseline_with_sources(
        tmp_path,
        {
            "present": {"path": "table.tex",
                        "sha256": hashlib.sha256(present.read_bytes()).hexdigest()},
            "absent": {"path": "../no_paper_checkout/table.tex", "sha256": "0" * 64},
        },
    )
    with pytest.raises(ComparisonError, match="does not exist"):
        load_baseline(path)

    warnings: list[str] = []
    baseline = load_baseline(path, missing_sources="warn", warn=warnings.append)
    assert len(result_template(baseline)["results"]) == 16
    assert len(warnings) == 1 and "absent" in warnings[0]

    # a source that IS present must still match its pinned hash
    present.write_text("edited\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="hash changed"):
        load_baseline(path, missing_sources="warn", warn=warnings.append)
