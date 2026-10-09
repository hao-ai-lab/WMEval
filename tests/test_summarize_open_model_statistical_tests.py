from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_open_model_statistical_tests import flatten_report


def test_flatten_thirty_run_report_preserves_corrected_tests(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "model",
                "repeat_count": 30,
                "metrics": {
                    "metric": {
                        "n": 30,
                        "mean": 2.0,
                        "sample_std": 1.0,
                        "mean_ci95": [1.5, 2.5],
                        "first5": {"mean": 1.0},
                        "confirmation25": {"mean": 2.2},
                        "welch_t": {"pvalue": 0.01, "holm_pvalue": 0.03},
                        "brown_forsythe": {"pvalue": 0.2, "holm_pvalue": 0.4},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rows = flatten_report(
        model="model", group="group", kind="directional_cell", path=path, repeat_count=30
    )
    assert len(rows) == 1
    assert rows[0]["first5_mean"] == 1.0
    assert rows[0]["confirmation25_mean"] == 2.2
    assert rows[0]["holm_significant"] is True
