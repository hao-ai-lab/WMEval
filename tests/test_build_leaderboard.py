from __future__ import annotations

import pytest

from tools import build_leaderboard as bl
from tools.build_leaderboard import Entry, build, compare, lite_cells, rank_cell


def _paper(dimensions: dict, models=("a", "b", "c")) -> dict:
    return {"models": {m: {"display": m.upper()} for m in models}, "dimensions": dimensions}


def _rows(board: dict) -> dict:
    return {r["model"]: r for r in board["leaderboard"]}


def test_direction_is_respected() -> None:
    lower = rank_cell({"a": Entry(1.0, 0.0, 5), "b": Entry(2.0, 0.0, 5), "c": Entry(3.0, 0.0, 5)}, higher=False)[0]
    higher = rank_cell({"a": Entry(1.0, 0.0, 5), "b": Entry(2.0, 0.0, 5), "c": Entry(3.0, 0.0, 5)}, higher=True)[0]
    assert lower == {"a": 1, "b": 2, "c": 3}
    assert higher == {"a": 3, "b": 2, "c": 1}


def test_welch_tie_shares_the_averaged_rank() -> None:
    # a and b overlap (large std); c is far worse than both.
    ranks, ties = rank_cell({"a": Entry(10.0, 5.0, 5), "b": Entry(11.0, 5.0, 5), "c": Entry(50.0, 1.0, 5)},
                            higher=False)
    assert ranks == {"a": 1.5, "b": 1.5, "c": 3}
    assert ties == [["a", "b"]]
    # Same means with tiny spread separate.
    ranks, _ = rank_cell({"a": Entry(10.0, 0.1, 5), "b": Entry(11.0, 0.1, 5)}, higher=False)
    assert ranks == {"a": 1, "b": 2}


def test_bootstrap_over_seeds() -> None:
    noisy_a = Entry.from_samples([1.0, 5.0, 2.0, 6.0, 3.0])
    noisy_b = Entry.from_samples([2.0, 6.0, 3.0, 4.0, 5.0])
    assert compare(noisy_a, noisy_b, higher=True, key="k") == (0, "bootstrap")
    clear_a = Entry.from_samples([10.0, 10.5, 11.0, 10.2, 10.8])
    assert compare(clear_a, noisy_b, higher=True, key="k") == (1, "bootstrap")
    assert compare(clear_a, noisy_b, higher=False, key="k") == (-1, "bootstrap")


def test_single_sample_is_tested_against_the_other_spread() -> None:
    ref = Entry(10.0, 2.0, 5)
    assert compare(Entry.from_samples([11.0]), ref, higher=False)[0] == 0          # inside the spread
    assert compare(Entry.from_samples([30.0]), ref, higher=False) == (-1, "prediction")
    assert compare(Entry(1.0, None, 1), Entry(2.0, None, 1), higher=True) == (-1, "means")


def test_na_partial_coverage_single_model_dimension_and_like_for_like() -> None:
    paper = _paper({
        # b is N/A on cell k2; the cell with one model (k3) is not ranked at all.
        "lpips": {"metrics": {"x": "lower"}, "cells": {
            "k1": {"x": {"a": [0.1, 0.0], "b": [0.2, 0.0], "c": [0.3, 0.0]}},
            "k2": {"x": {"a": [0.5, 0.0], "c": [0.4, 0.0]}},
            "k3": {"x": {"b": [0.9, 0.0]}},
        }},
        "prompt_scene_consistency": {"metrics": {"o": "higher"}, "cells": {"k": {"o": {"a": [0.5, 0.0], "b": [0.9, 0.0]}}}},
        "physical_causality": {"metrics": {"csr": "higher"}, "cells": {"k": {"csr": {"a": [3, 1]}}}},
    })
    report = build(paper, {})
    lp = report["dimensions"]["lpips"]
    assert lp["mean_cell_rank"] == {"a": 1.5, "b": 2.0, "c": 2.0}
    # Pairwise: a beats b and b beats c on k1; a and c split k1/k2, so they tie.
    assert lp["rank"] == {"a": 1.5, "b": 2.0, "c": 2.5}
    assert lp["dimension_ties"] == [["a", "c"]]
    assert lp["percentile"] == {"a": 0.75, "b": 0.5, "c": 0.25}
    assert lp["cells"]["k3"]["x"]["values"]["b"]["rank"] is None

    pc = report["dimensions"]["physical_causality"]
    assert not pc["comparable"] and pc["percentile"] == {}

    rows = _rows(report)
    assert rows["a"]["coverage"] == 2                      # causality (single model) is not counted
    assert rows["a"]["dimension_ranks"]["physical_causality"] == {"rank": 1.0, "of": 1, "percentile": None}
    assert rows["a"]["overall_score"] == pytest.approx(0.375)  # mean(0.75 lpips, 0.0 prompt)
    assert rows["b"]["overall_score"] == pytest.approx(0.75)
    assert rows["c"]["overall_score"] == pytest.approx(0.25) and rows["c"]["coverage"] == 1
    assert [r["model"] for r in report["leaderboard"]] == ["b", "a", "c"]
    assert rows["b"]["overall_rank"] == 1

    # Like-for-like: only lpips ranks every listed model (c has no prompt score).
    assert report["like_for_like_dimensions"] == ["lpips"]
    assert rows["a"]["like_for_like_rank"] == 1 and rows["b"]["like_for_like_rank"] == 2


def test_overall_uses_ranks_not_raw_scores() -> None:
    # A huge raw margin in one dimension counts no more than a small one in another.
    paper = _paper({
        "lpips": {"metrics": {"x": "lower"}, "cells": {"k": {"x": {"a": [0.0, 0.0], "b": [1000.0, 0.0]}}}},
        "fluid_mechanics": {"metrics": {"y": "lower"}, "cells": {"k": {"y": {"a": [1.01, 0.0], "b": [1.0, 0.0]}}}},
    }, models=("a", "b"))
    rows = _rows(build(paper, {}))
    assert rows["a"]["overall_score"] == rows["b"]["overall_score"] == 0.5
    assert rows["a"]["overall_rank"] == rows["b"]["overall_rank"] == 1.5


def _row(dim, case, seed, metrics, status="completed"):
    return {"dimension": dim, "case_id": case, "seed": seed, "status": status, "metrics": metrics}


def test_lite_mapping_covers_all_dimensions() -> None:
    rows = [
        _row("environment_object_consistency", "BilliardGame/scene_360_in", "run0",
             {"depth_absrel_pct": 10.0, "pairwise_distance_mae_pct": 20.0, "character_consistency_pct": 80.0}),
        _row("environment_object_consistency", "BilliardGame/scene_360_out", "run0", {"character_consistency_pct": 60.0}),
        _row("environment_object_consistency", "MirrorRollingBall/scene_360_in", "run0",
             {"depth_absrel_pct": 5.0, "pairwise_distance_mae_pct": 7.0}),
        _row("kinematics_mechanics", "RacingCar/kinematics_third", "run0",
             {"dtw_m_mean": 0.5, "fde_m_mean": 0.9, "robust_error_rate_pct": 8.0}),
        _row("action_controllability", "BeamDomino/causal", "run0", {"relative_epe_dynamic_mean_raw": 0.3}),
        _row("lpips", "BallRoller/ctrl", "run0", {"lpips_causal_mean": None, "lpips_temporal_fallback_mean": 0.7}),
        _row("lpips", "MirrorRollingBall/lpips_collision", "run0", {"lpips_causal_mean": 0.6}),
        _row("lpips", "LiquidContainer/fluid_rest", "run0", {"lpips_mean": 0.4}),
        _row("fluid_mechanics", "LiquidContainer/fluid_rest", "run0",
             {"mask_area_std_normalized_delta": 0.1, "spectral_l1": 1.1}),
        _row("physical_causality", "BeamDomino/causal_glancing", "run0", {"csr_wm": 100.0, "amcl_delta": 0.5}),
        _row("lighting_shading", "SunRisesAndSets/lighting", "run0", {"shadow_direction_jitter_deg_per_frame": 2.0}),
        _row("lighting_shading", "SunRisesAndSets/lighting_orbit_v2", "run0", {"shadow_direction_jitter_deg_per_frame": 9.0}),
        _row("prompt_scene_consistency", "RacingCar/prompt_first", "run0", {"overall_score": 0.7, "env_score": 0.7}),
    ]
    out = lite_cells(rows)
    assert out["environment_object_consistency"]["billiard/360_in"] == {"depth": [10.0], "dist": [20.0]}
    assert out["environment_object_consistency"]["billiard/360"]["char"] == [80.0, 60.0]   # In/Out merged
    assert "depth" not in out["environment_object_consistency"]["mirror/360_in"]           # paper has no Mirror depth
    assert out["kinematics_mechanics"]["racing_car/third"]["robust_error"] == [8.0]
    assert out["action_controllability"]["beam_domino"]["rel_epe"] == [0.3]
    assert "ball_roller" not in out["lpips"]                                              # temporal fallback excluded
    assert out["lpips"]["mirror_ball"]["clpips"] == [0.6]
    assert out["fluid_mechanics"]["pooled"] == {"lpips": [0.4], "d_sigma_a": [0.1], "l1": [1.1]}
    assert out["physical_causality"]["beam_domino"] == {"csr": [100.0], "d_amcl": [0.5]}
    assert out["lighting_shading"]["sun_rises_and_sets"]["sdj"] == [2.0]                  # native case only
    assert out["prompt_scene_consistency"]["racing_car/first"] == {"env": [0.7], "overall": [0.7]}


def test_lite_later_runs_override_and_unsupported_drops_case() -> None:
    rows = [
        _row("action_controllability", "BallRoller/ctrl", "run0", {"relative_epe_dynamic_mean_raw": 0.1}),
        _row("prompt_scene_consistency", "ChessBoard/prompt_first", "run0", {"overall_score": 0.5}),
        # a later run: the model cannot express the case, and a re-judged seed replaces run0
        _row("action_controllability", "BallRoller/ctrl", "run0", {}, status="unsupported"),
        _row("prompt_scene_consistency", "ChessBoard/prompt_first", "run0", {"overall_score": 0.9}),
        _row("prompt_scene_consistency", "ChessBoard/prompt_first", "run1", {"overall_score": 0.8}),
    ]
    out = lite_cells(rows)
    assert "action_controllability" not in out
    assert out["prompt_scene_consistency"]["chessboard/first"]["overall"] == [0.9, 0.8]


def test_lite_model_is_ranked_on_paper_cells_only_and_extra_metrics_are_one_line(monkeypatch) -> None:
    paper = _paper({"kinematics_mechanics": {"metrics": {"fde": "lower"}, "cells": {
        "racing_car/third": {"fde": {"a": [3.5, 0.05], "b": [5.3, 0.15]}}}}}, models=("a", "b"))
    rows = [_row("kinematics_mechanics", "RacingCar/kinematics_third", s,
                 {"fde_m_mean": v, "fde_m_ego": v / 2}) for s, v in (("run0", 0.9), ("run1", 1.0))]
    report = build(paper, {"lite_model": rows})
    km = report["dimensions"]["kinematics_mechanics"]
    assert km["rank"] == {"lite_model": 1.0, "a": 2.0, "b": 3.0}
    assert set(km["cells"]["racing_car/third"]) == {"fde"}

    monkeypatch.setitem(bl.LITE_METRICS, "kinematics_mechanics",
                        {**bl.KINEMATICS_METRICS, "fde_ego": ("fde_m_ego", "lower")})
    report = build(paper, {"lite_model": rows, "other": [
        _row("kinematics_mechanics", "RacingCar/kinematics_third", "run0", {"fde_m_ego": 9.0})]})
    cell = report["dimensions"]["kinematics_mechanics"]["cells"]["racing_car/third"]
    assert cell["fde_ego"]["values"]["lite_model"]["rank"] == 1
    assert cell["fde_ego"]["values"]["other"]["rank"] == 2
    assert "a" not in cell["fde_ego"]["values"]                                            # paper rows N/A


def test_direction_conflict_between_paper_and_lite_is_an_error(monkeypatch) -> None:
    paper = _paper({"lpips": {"metrics": {"clpips": "higher"}, "cells": {"k": {"clpips": {"a": 1.0}}}}})
    with pytest.raises(ValueError):
        build(paper, {})


def test_markdown_renders_overall_like_for_like_and_dimension_tables() -> None:
    paper = _paper({
        "lpips": {"metrics": {"x": "lower"}, "cells": {"k": {"x": {"a": [0.1, 0.01], "b": [0.2, 0.01]}}}},
        "physical_causality": {"metrics": {"csr": "higher"}, "cells": {"k": {"csr": {"a": [3, 1]}}}},
    }, models=("a", "b"))
    text = bl.markdown(build(paper, {}))
    assert "## Overall" in text and "## Like-for-like" in text
    assert "| 1 | A | 1.000 | 1/8 |" in text
    assert "1/1*" in text
    assert "### cLPIPS — 1. A · 2. B" in text
    assert "| k · x ↓ | 0.10±0.01 ·1 | 0.20±0.01 ·2 |" in text


def test_lite_rows_for_cases_the_model_cannot_express_are_dropped() -> None:
    # A camera-only model's old kinematics rows predate the action-target gate; they
    # must not rank it on kinematics even though no newer run re-marked the case.
    from tools.build_leaderboard import lite_cells

    row = {"dimension": "kinematics_mechanics", "case_id": "RacingCar/kinematics_third", "seed": "run0",
           "status": "completed", "model": "odyssey", "metrics": {"fde_m_mean": 0.9, "dtw_m_mean": 0.5}}
    assert "kinematics_mechanics" not in lite_cells([row])
    assert "kinematics_mechanics" in lite_cells([{**row, "model": "project_genie"}])


def test_dimension_ties_when_cell_ranks_split() -> None:
    # a wins one cell and loses the other to b; c is last everywhere.
    paper = _paper({"lpips": {"metrics": {"x": "lower"}, "cells": {
        "k1": {"x": {"a": [0.1, 0.0], "b": [0.2, 0.0], "c": [0.9, 0.0]}},
        "k2": {"x": {"a": [0.3, 0.0], "b": [0.2, 0.0], "c": [0.9, 0.0]}},
        "k3": {"x": {"a": [0.1, 0.0], "b": [0.1, 0.0], "c": [0.9, 0.0]}},
    }}})
    lp = build(paper, {}, floors={})["dimensions"]["lpips"]
    assert lp["dimension_ties"] == [["a", "b"]]
    assert lp["rank"] == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_cells_at_or_below_the_self_match_floor_are_not_ranked() -> None:
    floors = {"environment_object_consistency": {
        "WanderingGarden/scene_360_in": {"depth": 93.0, "dist": 41.9, "dist_measurable": False},
        "BilliardGame/scene_360_in": {"depth": 22.0, "dist": 7.0, "dist_measurable": True},
    }}
    paper = _paper({"environment_object_consistency": {"metrics": {"depth": "lower", "dist": "lower"}, "cells": {
        "garden/360_in": {"depth": {"a": [80.0, 0.0], "b": [120.0, 0.0]}, "dist": {"a": [30.0, 0.0], "b": [60.0, 0.0]}},
        "billiard/360_in": {"depth": {"a": [50.0, 0.0], "b": [20.0, 0.0]},      # spread 30 > floor 22
                            "dist": {"a": [10.0, 0.0], "b": [14.0, 0.0]}},      # spread 4 < floor 7
    }}}, models=("a", "b"))
    eoc = build(paper, {}, floors=floors)["dimensions"]["environment_object_consistency"]
    assert set(eoc["floor_excluded"]) == {"garden/360_in · depth", "garden/360_in · dist", "billiard/360_in · dist"}
    assert "not measurable" in eoc["floor_excluded"]["garden/360_in · dist"]
    assert eoc["rank"] == {"a": 2.0, "b": 1.0}             # decided by Billiard depth alone
    assert eoc["cells"]["garden/360_in"]["depth"]["values"]["a"]["rank"] is None


def test_skipped_dimension_is_left_out_of_ranks_and_coverage() -> None:
    paper = _paper({
        "lpips": {"metrics": {"x": "lower"}, "cells": {"k": {"x": {"a": [0.1, 0.0], "b": [0.9, 0.0]}}}},
        "fluid_mechanics": {"metrics": {"y": "lower"}, "cells": {"k": {"y": {"a": [0.9, 0.0], "b": [0.1, 0.0]}}}},
    }, models=("a", "b"))
    report = build(paper, {}, floors={}, skip_dimensions=frozenset({"lpips"}))
    assert "lpips" not in report["dimensions"] and report["skipped_dimensions"] == ["lpips"]
    assert report["num_dimensions"] == 7
    assert [r["model"] for r in report["leaderboard"]] == ["b", "a"]


def test_readme_section_replaces_only_the_marked_block(tmp_path) -> None:
    paper = _paper({"fluid_mechanics": {"metrics": {"y": "lower"}, "cells": {"k": {"y": {"a": [0.9, 0.0], "b": [0.1, 0.0]}}}}},
                   models=("a", "b"))
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{bl.README_START}\nold table\n{bl.README_END}\nrest\n", encoding="utf-8")
    bl.write_readme(readme, bl.readme_section(build(paper, {}, floors={})))
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("\nrest\n") and "old table" not in text
    assert "| 1 | **B** |" in text and text.count(bl.README_START) == 1


def test_skipped_metric_is_left_out_of_the_dimension_rank() -> None:
    paper = _paper({"environment_object_consistency": {"metrics": {"depth": "lower", "char": "higher"}, "cells": {
        "billiard/360": {"char": {"a": [90.0, 0.0], "b": [10.0, 0.0]}},
        "billiard/360_in": {"depth": {"a": [90.0, 0.0], "b": [10.0, 0.0]}},
    }}}, models=("a", "b"))
    both = build(paper, {}, floors={})["dimensions"]["environment_object_consistency"]
    assert both["rank"] == {"a": 1.5, "b": 1.5}                      # one win each
    report = build(paper, {}, floors={}, skip_metrics=frozenset({("environment_object_consistency", "depth"),
                                                                 ("environment_object_consistency", "dist")}))
    eoc = report["dimensions"]["environment_object_consistency"]
    assert eoc["rank"] == {"a": 1.0, "b": 2.0} and "billiard/360_in" not in eoc["cells"]
    assert bl.character_only(report) and bl.label(report, "environment_object_consistency") == "Consistency"


def test_dropped_model_leaves_every_dimension() -> None:
    paper = _paper({"fluid_mechanics": {"metrics": {"y": "lower"}, "cells": {"k": {"y": {"a": [0.1, 0.0], "b": [0.9, 0.0]}}}},
                    "lighting_shading": {"metrics": {"z": "lower"}, "cells": {"k": {"z": {"a": [0.1, 0.0], "b": [0.5, 0.0], "c": [0.9, 0.0]}}}}})
    report = build(paper, {}, floors={}, drop_models=frozenset({"b"}))
    assert [r["model"] for r in report["leaderboard"]] == ["a", "c"] and report["dropped_models"] == ["b"]
    assert not report["dimensions"]["fluid_mechanics"]["comparable"]          # a alone: shown, not counted
    assert report["excluded"] == []
