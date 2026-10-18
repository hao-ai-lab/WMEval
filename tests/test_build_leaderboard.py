from __future__ import annotations

from tools.build_leaderboard import build


def test_pairwise_win_rate_composite_ignores_missing_dimensions() -> None:
    paper = {
        "models": {"a": {"display": "A"}, "b": {"display": "B"}, "c": {"display": "C"}},
        "dimensions": {
            "lpips": {"metrics": {"x": "lower"}, "cells": {"k": {"x": {"a": 0.1, "b": 0.2, "c": 0.3}}}},
            "prompt_scene_consistency": {"metrics": {"o": "higher"}, "cells": {"k": {"o": {"a": 0.5, "b": 0.9}}}},
            "physical_causality": {"metrics": {"csr": "higher"}, "cells": {"k": {"csr": {"a": 3}}}},
        },
    }
    report = build(paper, {}, core_min_models=3)
    rows = {r["model"]: r for r in report["leaderboard"]}
    assert rows["a"]["dimension_scores"]["lpips"] == 100.0
    assert rows["c"]["dimension_scores"]["lpips"] == 0.0
    assert rows["a"]["dimension_scores"]["physical_causality"] is None  # single model: no comparison
    assert rows["a"]["composite"] == 50.0 and rows["a"]["coverage"] == 2
    assert rows["c"]["coverage"] == 1
    assert report["core_dimensions"] == ["lpips"]
