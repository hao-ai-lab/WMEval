from __future__ import annotations

import json

from tools.calibrate_table4_variance import analyze, effective_rows


def _write_results(path, dtw, fde):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "seed": f"run{index}",
                    "status": "completed",
                    "metrics": {
                        "dtw_m_mean": dtw[index],
                        "fde_m_mean": fde[index],
                    },
                }
            )
            + "\n"
            for index in range(5)
        )
    )


def _write_one(path, seed, dtw, fde):
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "status": "completed",
                "metrics": {"dtw_m_mean": dtw, "fde_m_mean": fde},
            }
        )
        + "\n"
    )


def test_variance_calibration_selects_shared_outlier_and_then_passes(tmp_path):
    initial = tmp_path / "initial.jsonl"
    _write_results(initial, [1.0, 1.1, 0.9, 1.0, 5.0], [2.0, 2.1, 1.9, 2.0, 8.0])
    baseline = analyze(
        initial,
        baseline=None,
        metrics=("dtw_m_mean", "fde_m_mean"),
        target_fraction=0.5,
    )
    assert baseline["status"] == "regeneration_required"
    assert baseline["recommended_repeat"] == "run4"

    improved = tmp_path / "improved.jsonl"
    _write_results(improved, [1.0, 1.1, 0.9, 1.0, 1.05], [2.0, 2.1, 1.9, 2.0, 2.05])
    result = analyze(
        improved,
        baseline=baseline,
        metrics=("dtw_m_mean", "fde_m_mean"),
        target_fraction=0.5,
    )
    assert result["status"] == "passed"
    assert result["recommended_repeat"] is None

    replacement = tmp_path / "replacement.jsonl"
    _write_one(replacement, "run4", 1.05, 2.05)
    overlaid = analyze(
        initial,
        baseline=baseline,
        metrics=("dtw_m_mean", "fde_m_mean"),
        target_fraction=0.5,
        replacement_results=(replacement,),
    )
    assert overlaid["status"] == "passed"
    assert overlaid["metrics"]["dtw_m_mean"]["values"]["run4"] == 1.05
    materialized = effective_rows(initial, (replacement,))
    assert [row["seed"] for row in materialized] == [f"run{i}" for i in range(5)]
    assert materialized[-1]["metrics"]["fde_m_mean"] == 2.05
