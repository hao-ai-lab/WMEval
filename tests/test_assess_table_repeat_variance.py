from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/assess_table_repeat_variance.py"
SPEC = importlib.util.spec_from_file_location("assess_table_repeat_variance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_preregistered_five_run_thresholds() -> None:
    assert not MODULE.assess([50, 51, 49, 50, 50], "depth_absrel_pct")["expand_to_30"]
    assert MODULE.assess([40, 50, 60, 50, 50], "depth_absrel_pct")["expand_to_30"]
    assert not MODULE.assess([1.0, 1.1, 0.9, 1.0, 1.0], "dtw_m_mean")["expand_to_30"]
    assert MODULE.assess([0.5, 0.8, 1.2, 1.4, 1.1], "dtw_m_mean")["expand_to_30"]


def test_thirty_run_report_compares_disjoint_cohorts() -> None:
    result = MODULE.assess([1.0] * 5 + [1.1] * 25, "fde_m_mean")
    assert result["n"] == 30
    assert result["first5"]["mean"] == 1.0
    assert result["confirmation25"]["mean"] == 1.1
    assert "pvalue" in result["welch_t"]
    assert "pvalue" in result["brown_forsythe"]


def test_report_records_requested_metric_with_undefined_repeats(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "per_case_results.jsonl"
    output = tmp_path / "variance.json"
    rows = [
        {
            "model": "model",
            "case_id": "Scene/view",
            "dimension": "environment_object_consistency",
            "seed": f"run{index}",
            "status": "completed",
            "metrics": {
                "depth_absrel_pct": 10.0 + index,
                "pairwise_distance_mae_pct": None if index == 3 else 20.0 + index,
            },
        }
        for index in range(5)
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--results",
            str(results),
            "--model",
            "model",
            "--case-id",
            "Scene/view",
            "--dimension",
            "environment_object_consistency",
            "--metric",
            "depth_absrel_pct",
            "--metric",
            "pairwise_distance_mae_pct",
            "--output",
            str(output),
        ],
    )

    assert MODULE.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["unavailable_metrics"]["pairwise_distance_mae_pct"] == {
        "status": "undefined",
        "reason": "non_numeric_or_missing_values",
        "observed_numeric_count": 4,
        "required_repeat_count": 5,
        "missing_repeats": ["run3"],
    }
