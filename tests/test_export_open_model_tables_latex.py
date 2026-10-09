import json
from pathlib import Path

import pytest

from tools import export_open_model_tables_latex as exporter


def test_robustness_caption_replaces_nested_five_run_paper_caption() -> None:
    source = (
        "\\begin{table*}\n"
        "\\caption{Current paper. \\lxvalidation{Use runs 0--4.}}\n"
        "\\label{tab:worldmodel_consistency}\n"
        "\\begin{tabular}{lc}\nModel & 12.0 \\\\\n\\end{tabular}\n"
        "\\end{table*}\n"
    )
    cell = exporter.Cell(12.0, 1.0, 30, {f"run{i}": 12.0 for i in range(30)})
    updated = exporter.robustness_caption(source, 7, [cell, None])
    assert "Use runs 0--4" not in updated
    assert "marks 30 generations" in updated
    assert "unmarked estimates use five generations" in updated
    assert "N/D denotes" in updated
    assert "separate from the selected paper tables" in updated
    assert updated[updated.index("\\label"):] == source[source.index("\\label"):]


def test_robustness_caption_does_not_add_other_tables_cohort_markers() -> None:
    source = "\\caption{Old wording}\n\\label{tab:physical}\nBODY"
    cell = exporter.Cell(1.0, 0.1, 5, {f"run{i}": 1.0 for i in range(5)})
    updated = exporter.robustness_caption(source, 4, [cell])
    assert "Estimates use five generations" in updated
    assert "dagger" not in updated
    assert "N/D" not in updated
    with pytest.raises(ValueError, match="exactly one caption"):
        exporter.robustness_caption("\\label{tab:physical}", 4, [cell])


def test_optional_cell_fails_closed_when_any_repeat_is_undefined(
    tmp_path: Path, monkeypatch
) -> None:
    report = tmp_path / "per_case_results.jsonl"
    rows = []
    for index in range(5):
        rows.append(
            {
                "model": "model",
                "seed": f"run{index}",
                "status": "completed",
                "metrics": {"depth_absrel_pct": None if index == 3 else float(index)},
            }
        )
    report.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(exporter, "_report", lambda *_: report)
    monkeypatch.setattr(exporter, "required_repeat_count", lambda *_: 5)

    assert exporter.optional_cell("model", "table7/Scene/variant", "depth_absrel_pct") is None
    assert exporter._t7(None) == "N/D"
    with pytest.raises(ValueError, match="missing numeric"):
        exporter.cell("model", "table7/Scene/variant", "depth_absrel_pct")


def test_rows_for_does_not_fall_back_when_thirty_runs_are_required(
    tmp_path: Path, monkeypatch
) -> None:
    five = tmp_path / "five.jsonl"
    five.write_text(
        "".join(
            json.dumps({"model": "model", "seed": f"run{i}", "status": "completed"})
            + "\n"
            for i in range(5)
        ),
        encoding="utf-8",
    )
    missing_thirty = tmp_path / "missing-thirty.jsonl"
    monkeypatch.setattr(
        exporter,
        "_report",
        lambda _model, _case, cohort: missing_thirty if cohort == "30_run" else five,
    )
    monkeypatch.setattr(exporter, "required_repeat_count", lambda *_: 30)

    with pytest.raises(ValueError, match="required 30-run evaluation cohort is incomplete"):
        exporter.rows_for("model", "table7/Scene/variant")


def test_required_repeat_count_honors_gate_and_paired_queue(
    tmp_path: Path, monkeypatch
) -> None:
    gate = tmp_path / "variance_gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "model",
                "repeat_count": 5,
                "expand_group_to_30": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "_variance_report", lambda *_: gate)
    monkeypatch.setattr(exporter, "_queue_repeat_count", lambda *_: 30)
    assert exporter.required_repeat_count("model", "table7/Scene/variant") == 30

    gate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "model",
                "repeat_count": 5,
                "expand_group_to_30": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "_queue_repeat_count", lambda *_: 5)
    assert exporter.required_repeat_count("model", "table7/Scene/variant") == 30


def test_metric_values_rejects_nonfinite_numbers(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {
            "model": "model",
            "seed": f"run{i}",
            "status": "completed",
            "metrics": {"depth_absrel_pct": float("nan") if i == 2 else float(i)},
        }
        for i in range(5)
    ]
    monkeypatch.setattr(exporter, "rows_for", lambda *_: rows)

    assert exporter.metric_values(
        "model", "table7/Scene/variant", "depth_absrel_pct", allow_undefined=True
    ) is None
    with pytest.raises(ValueError, match="missing numeric"):
        exporter.metric_values("model", "table7/Scene/variant", "depth_absrel_pct")


def test_variance_comparison_requires_corrected_disjoint_cohort_tests(
    tmp_path: Path, monkeypatch
) -> None:
    comparison = tmp_path / "variance_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "model",
                "repeat_count": 30,
                "metrics": {
                    "depth_absrel_pct": {
                        "n": 30,
                        "mean_ci95": [1.0, 2.0],
                        "first5": {"mean": 1.0},
                        "confirmation25": {"mean": 1.5},
                        "welch_t": {"pvalue": 0.5, "holm_pvalue": 1.0},
                        "brown_forsythe": {"pvalue": 0.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "_variance_report", lambda *_: comparison)

    with pytest.raises(ValueError, match="incomplete 30-run statistics"):
        exporter._require_variance_comparison("model", "table7/Scene/variant")


def test_paired_cell_requires_reported_cell_statistics(monkeypatch) -> None:
    monkeypatch.setattr(
        exporter,
        "metric_values",
        lambda *_args, **_kwargs: {f"run{i}": float(i) for i in range(5)},
    )
    monkeypatch.setattr(
        exporter,
        "_require_paired_character_statistics",
        lambda model, scene, regime, repeat_count: (_ for _ in ()).throw(
            ValueError(f"{model}/{scene}/{regime}/{repeat_count}")
        ),
    )

    with pytest.raises(ValueError, match="model/BilliardGame/perturb/5"):
        exporter.paired_cell(
            "model",
            "table7/BilliardGame/perturb_in",
            "table7/BilliardGame/perturb_out",
            "character_consistency_pct",
        )
