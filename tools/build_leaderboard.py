#!/usr/bin/env python3
"""WMEval leaderboard: per-dimension rankings aggregated into an overall ranking.

Method (no cross-metric normalization is invented; only ranks are combined):

1. Cell ranks. Within a dimension, models are ranked separately on every
   (cell, metric) the paper reports for it (plus any Lite-only metric listed in
   LITE_METRICS), respecting the metric's direction. Two models that cannot be
   separated share the rank: with per-seed values on both sides (Lite runs, >= 2
   samples each) the seeds are bootstrapped and the pair is a tie unless one
   ordering holds in >= 95% of resamples; otherwise (paper mean +/- std over n
   generations) a two-sided Welch t-test at alpha = 0.05 decides, and a
   single-sample value is tested against the other model's spread (prediction
   interval). Rank = 1 + #models significantly better + 0.5 x #models tied, which
   is the averaged rank of a tie group. A model N/A on a (cell, metric) is not
   ranked there, and a (cell, metric) with one model has no comparison.
   A (cell, metric) below its measurement floor is shown but not ranked: the
   Unity reference scored against itself (``self_match_floor.json``, from
   SELF_MATCH_BOUNDS.md) has an error at least the spread of the model values,
   or pairwise distance is not measurable on the reference at all.
2. Dimension rank, with the same tie rule one level up. For each pair of models,
   the differences of their ranks on the (cell, metric)s both are ranked on are
   bootstrapped; the pair is ordered only if one mean rank is lower in >= 95% of
   resamples. Dimension rank = 1 + #models significantly better + 0.5 x #ties
   (the mean cell rank is reported alongside).
3. Overall. Each dimension rank becomes a percentile, (n - rank) / (n - 1):
   1.0 best ... 0.0 worst among the n models ranked there; a single-model
   dimension contributes no comparison. Overall score = mean percentile over the
   dimensions the model is ranked in; coverage = those dimensions / 8. The
   like-for-like score uses only the dimensions every listed model is ranked in.

Inputs: ``experiments/leaderboard/paper_scores.yaml`` (paper tables, mean +/- std)
plus Lite results for extra models (``--lite name=per_case_results.jsonl``,
repeatable; later files override earlier ones on the same dimension/case/seed,
and a case the latest run marks ``unsupported`` is N/A for that model). Lite
results are mapped onto the paper cells for all eight dimensions.

    python tools/build_leaderboard.py \\
        --lite odyssey=reports/wmeval_lite/odyssey/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_lpips/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_judge/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_v2/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_v2_prompt/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_v2_lighting_native/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_eoc_5seed_merged/per_case_results.jsonl \\
        --skip-dimension lpips \\
        --skip-metric environment_object_consistency:depth --skip-metric environment_object_consistency:dist \\
        --output reports/leaderboard --readme README.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run as a script: the action-target gate imports evaluation.*
    sys.path.insert(0, str(ROOT))
DEFAULT_PAPER = ROOT / "experiments/leaderboard/paper_scores.yaml"
DIMENSION_ORDER = (
    "environment_object_consistency", "action_controllability", "kinematics_mechanics",
    "fluid_mechanics", "physical_causality", "lighting_shading", "prompt_scene_consistency", "lpips",
)
SHORT = {
    "environment_object_consistency": "Consistency", "action_controllability": "Control",
    "kinematics_mechanics": "Kinematics", "fluid_mechanics": "Fluid", "physical_causality": "Causality",
    "lighting_shading": "Lighting", "prompt_scene_consistency": "Prompt", "lpips": "cLPIPS",
}
ALPHA = 0.05            # Welch / prediction-interval test level
BOOT_CONFIDENCE = 0.95  # bootstrap: an ordering must hold in this share of resamples
BOOT_RESAMPLES = 4000

# Kinematics metrics: paper metric (or a Lite-only metric) -> (Lite key, direction).
# A Lite-only metric added here is ranked among the models that report it (paper rows
# are N/A on it), e.g. add  "fde_ego": ("fde_m_ego", "lower"),  to include ego-frame FDE.
KINEMATICS_METRICS = {
    "dtw": ("dtw_m_mean", "lower"),
    "fde": ("fde_m_mean", "lower"),
    "robust_error": ("robust_error_rate_pct", "lower"),
}

# Lite per-case metric key for every leaderboard metric, by target dimension.
LITE_METRICS: dict[str, dict[str, tuple[str, str]]] = {
    "environment_object_consistency": {
        "depth": ("depth_absrel_pct", "lower"), "dist": ("pairwise_distance_mae_pct", "lower"),
        "char": ("character_consistency_pct", "higher"),
    },
    "kinematics_mechanics": KINEMATICS_METRICS,
    "action_controllability": {
        "rel_epe": ("relative_epe_dynamic_mean_raw", "lower"),
        "cos": ("cosine_similarity_dynamic_mean_raw", "higher"),
        "syn_cos": ("synthetic_mean_flow_cosine", "higher"),
        "syn_epe": ("synthetic_pixel_epe_mean", "lower"),
    },
    # Causal rows only: a generation with no matched event has lpips_causal_mean None (the
    # paper excludes such generations), never the temporal fallback.
    "lpips": {"clpips": ("lpips_causal_mean", "lower")},
    "physical_causality": {
        "csr": ("csr_wm", "higher"), "d_amcl": ("amcl_delta", "lower"), "d_mcl": ("mcl_delta", "lower"),
        "d_ws": ("ws_delta", "lower"), "d_dua": ("dua_delta", "lower"), "d_icra": ("icra_delta", "lower"),
    },
    "fluid_mechanics": {
        "d_sigma_a": ("mask_area_std_normalized_delta", "lower"),
        "d_a_dot": ("mask_area_drift_per_sec_abs_delta", "lower"),
        "d_theta": ("surface_tilt_residual_deg_abs_delta", "lower"),
        "l1": ("spectral_l1", "lower"),
        "lpips": ("lpips_mean", "lower"),  # whole-frame LPIPS from the lpips rows on LiquidContainer
    },
    "lighting_shading": {
        "scene_retention": ("scene_retention_rate_pct", "higher"),
        "shadow_detection": ("shadow_detection_rate_pct", "higher"),
        "mosd": ("multi_object_sun_disagreement_deg", "lower"),
        "sdj": ("shadow_direction_jitter_deg_per_frame", "lower"),
        "area_stability": ("shadow_area_stability_pct", "lower"),
        "shape_drift": ("shadow_shape_drift", "lower"),
    },
    "prompt_scene_consistency": {
        "env": ("env_score", "higher"), "obj": ("obj_score", "higher"), "overall": ("overall_score", "higher"),
    },
}

# Lite case -> paper cell.
_SCENES = {"WanderingGarden": "garden", "BilliardGame": "billiard", "MirrorRollingBall": "mirror"}
_PROMPT = {"ChessBoard": "chessboard", "MirrorRollingBall": "mirror", "RacingCar": "racing_car"}
_KINEMATICS = {"RacingCar/kinematics_third": "racing_car/third", "BilliardGame/kinematics_first": "billiard/first"}
_RAFT = {"BallRoller/ctrl": "ball_roller", "BeamDomino/causal": "beam_domino",
         "MirrorRollingBall/lpips_collision": "mirror_ball"}


def _routes(dimension: str, case_id: str) -> list[tuple[str, str, set[str] | None]]:
    """(target dimension, paper cell, metrics allowed or None = all) for one Lite row."""
    scene, _, variant = case_id.partition("/")
    if dimension == "environment_object_consistency" and scene in _SCENES and variant.startswith("scene_"):
        name = variant[len("scene_"):]                                   # perturb_in, 360_out, ...
        geometry = {"dist"} if scene == "MirrorRollingBall" else {"depth", "dist"}  # no Mirror depth in the paper
        out = [(dimension, f"{_SCENES[scene]}/{name}", geometry)]
        if scene != "WanderingGarden":                                   # In/Out merged, as in the paper
            out.append((dimension, f"{_SCENES[scene]}/{name.rsplit('_', 1)[0]}", {"char"}))
        return out
    if dimension == "prompt_scene_consistency" and scene in _PROMPT and variant.startswith("prompt_"):
        return [(dimension, f"{_PROMPT[scene]}/{variant[len('prompt_'):]}", None)]
    if dimension == "lighting_shading" and case_id == "SunRisesAndSets/lighting":   # the paper-native case only
        return [(dimension, "sun_rises_and_sets", None)]
    if dimension == "kinematics_mechanics" and case_id in _KINEMATICS:
        return [(dimension, _KINEMATICS[case_id], None)]
    if dimension == "action_controllability" and case_id in _RAFT:
        return [(dimension, _RAFT[case_id], None)]
    if dimension == "lpips" and case_id in _RAFT:
        return [(dimension, _RAFT[case_id], None)]
    if dimension == "lpips" and scene == "LiquidContainer":
        return [("fluid_mechanics", "pooled", {"lpips"})]
    if dimension == "fluid_mechanics" and scene == "LiquidContainer":
        return [(dimension, "pooled", {"d_sigma_a", "d_a_dot", "d_theta", "l1"})]
    if dimension == "physical_causality" and scene == "BeamDomino" and variant.startswith("causal"):
        return [(dimension, "beam_domino", None)]                        # pooled over the 3 shapes
    return []


def _gated_out(case_id: str, model: str) -> bool:
    """True when today's action-target gate marks ``model`` unsupported for ``case_id``.

    Older Lite rows can predate the gate (e.g. a camera-only model scored on a car or
    ball case); they must not count, even when no newer run re-marked the case."""
    from evaluation.lite.action_support import unsupported_reason

    artifacts = ROOT / "eval_sets/wmeval_lite_v1/cases" / case_id / "reference/artifacts"
    return bool(model) and artifacts.is_dir() and unsupported_reason(artifacts, model) is not None


def lite_cells(rows: list[dict]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """dimension -> cell -> metric -> samples (one per completed case x seed row).

    Later rows override earlier ones on the same (dimension, case, seed); a case whose
    latest rows include ``unsupported`` is dropped (the model cannot express it).
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        latest[(row["dimension"], row["case_id"], str(row.get("seed")))] = row
    unsupported = {(d, c) for (d, c, _), r in latest.items() if r.get("status") == "unsupported"}
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (dim, case, _), row in sorted(latest.items()):
        if row.get("status") != "completed" or (dim, case) in unsupported:
            continue
        if _gated_out(case, str(row.get("model") or "")):
            continue  # the model cannot express this case's action target under the current gate
        metrics = row.get("metrics") or {}
        for target, cell, allowed in _routes(dim, case):
            for metric, (key, _direction) in LITE_METRICS[target].items():
                value = metrics.get(key)
                if (allowed is None or metric in allowed) and isinstance(value, (int, float)) \
                        and not isinstance(value, bool) and math.isfinite(value):
                    out[target][cell][metric].append(float(value))
    return out


@dataclass(frozen=True)
class Entry:
    mean: float
    std: float | None   # sample std; None when unknown or n < 2
    n: int
    samples: tuple[float, ...] | None = None   # per-seed values (Lite)
    source: str = "paper"

    @classmethod
    def from_paper(cls, value, n: int) -> "Entry":
        if isinstance(value, (list, tuple)):         # [mean, std] or [mean, std, n] (n overrides the table's)
            n = int(value[2]) if len(value) > 2 else n
            return cls(float(value[0]), None if len(value) < 2 or value[1] is None else float(value[1]), n)
        return cls(float(value), None, n)

    @classmethod
    def from_samples(cls, values: list[float], source: str = "lite") -> "Entry":
        return cls(mean(values), stdev(values) if len(values) > 1 else None, len(values), tuple(values), source)


def _spread_known(e: Entry) -> bool:
    return e.std is not None and e.n >= 2


def summary_test_p(a: Entry, b: Entry) -> float:
    """Two-sided p-value that a and b differ, from mean/std/n only."""
    if a.mean == b.mean:
        return 1.0
    if _spread_known(a) and _spread_known(b):
        va, vb = a.std ** 2 / a.n, b.std ** 2 / b.n
        if va + vb == 0:
            return 0.0
        df = (va + vb) ** 2 / ((va ** 2 / (a.n - 1)) + (vb ** 2 / (b.n - 1)))
        return float(2 * stats.t.sf(abs(a.mean - b.mean) / math.sqrt(va + vb), df))
    if _spread_known(a) or _spread_known(b):
        # One single-sample value against the other model's spread (prediction interval).
        ref, one = (a, b) if _spread_known(a) else (b, a)
        if ref.std == 0:
            return 0.0
        t = abs(one.mean - ref.mean) / (ref.std * math.sqrt(1 + 1 / ref.n))
        return float(2 * stats.t.sf(t, ref.n - 1))
    return 0.0  # no spread on either side: the means decide


def compare(a: Entry, b: Entry, higher: bool, key: str = "") -> tuple[int, str]:
    """+1 if a is significantly better, -1 if b is, 0 if they tie; plus the test used."""
    sign = 1.0 if higher else -1.0
    if a.samples and b.samples and len(a.samples) > 1 and len(b.samples) > 1:
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        sa = rng.choice(a.samples, (BOOT_RESAMPLES, len(a.samples))).mean(axis=1)
        sb = rng.choice(b.samples, (BOOT_RESAMPLES, len(b.samples))).mean(axis=1)
        diff = sign * (sa - sb)
        share = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        if share >= BOOT_CONFIDENCE:
            return 1, "bootstrap"
        if share <= 1 - BOOT_CONFIDENCE:
            return -1, "bootstrap"
        return 0, "bootstrap"
    method = "welch" if _spread_known(a) and _spread_known(b) else (
        "prediction" if _spread_known(a) or _spread_known(b) else "means")
    if summary_test_p(a, b) >= ALPHA:
        return 0, method
    return (1 if sign * (a.mean - b.mean) > 0 else -1), method


def rank_cell(entries: dict[str, Entry], higher: bool, key: str = "") -> tuple[dict[str, float], list[list[str]]]:
    """Rank = 1 + #significantly better + 0.5 x #tied (the averaged rank of a tie group)."""
    models = sorted(entries)
    better = defaultdict(int)
    tied = defaultdict(int)
    ties = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            outcome, _ = compare(entries[a], entries[b], higher, f"{key}|{a}|{b}")
            if outcome > 0:
                better[b] += 1
            elif outcome < 0:
                better[a] += 1
            else:
                tied[a] += 1
                tied[b] += 1
                ties.append([a, b])
    return {m: 1 + better[m] + 0.5 * tied[m] for m in models}, ties


def averaged_ranks(scores: dict[str, float], lower_is_better: bool = True) -> dict[str, float]:
    """1..n ranks of the scores; exactly equal scores share the averaged rank."""
    order = sorted(scores, key=lambda m: (scores[m] if lower_is_better else -scores[m], m))
    out, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and math.isclose(scores[order[j + 1]], scores[order[i]], abs_tol=1e-9):
            j += 1
        for m in order[i:j + 1]:
            out[m] = (i + 1 + j + 1) / 2
        i = j + 1
    return out


FLOORS_PATH = ROOT / "eval_sets/wmeval_lite_v1/self_match_floor.json"
_CELL_SCENE = {"garden": "WanderingGarden", "billiard": "BilliardGame", "mirror": "MirrorRollingBall"}
DIM_TIE_RESAMPLES = 4000
DIM_TIE_CONFIDENCE = 0.95


def _floor_cases(cell: str) -> list[str]:
    scene, _, view = cell.partition("/")
    if scene not in _CELL_SCENE:
        return []
    views = [view] if view.endswith(("_in", "_out")) else [f"{view}_in", f"{view}_out"]   # char cells pool In/Out
    return [f"{_CELL_SCENE[scene]}/scene_{v}" for v in views]


def below_floor(dim: str, cell: str, metric: str, per: dict, floors: dict) -> str | None:
    """Why a (cell, metric) cannot separate models, or None.

    The Unity reference scored against itself gives the metric's own measurement error
    (SELF_MATCH_BOUNDS.md). When that error is at least the whole spread of model values,
    no two models differ by more than the metric's noise."""
    table = floors.get(dim) or {}
    cases = [table[c] for c in _floor_cases(cell) if c in table]
    if not cases or metric not in ("depth", "dist", "char") or len(per) < 2:
        return None
    if metric == "dist" and not all(c.get("dist_measurable") for c in cases):
        return "pairwise distance is not measurable on the Unity reference itself"
    vals = [c.get(metric) for c in cases]
    if any(v is None for v in vals):
        return None
    floor = sum(vals) / len(vals)
    noise = 100.0 - floor if metric == "char" else floor        # char is a consistency %, higher is better
    means = [e.mean for e in per.values()]
    spread = max(means) - min(means)
    if noise >= spread:
        return f"Unity self-match error ({noise:.1f}) is at least the spread of model values ({spread:.1f})"
    return None


def dimension_ranks(ranks_by_key: dict[str, dict[tuple, float]], seed: int = 0) -> tuple[dict[str, float], list[list[str]]]:
    """Dimension rank = 1 + #models significantly better + 0.5 x #ties, from paired cell ranks.

    For each pair, the (cell, metric) comparisons both are ranked on are bootstrapped; the
    pair is ordered only when one mean rank is lower in >= DIM_TIE_CONFIDENCE of resamples."""
    import numpy as np

    models = sorted(ranks_by_key)
    rng = np.random.default_rng(seed)
    better = {m: 0 for m in models}
    tie = {m: 0 for m in models}
    ties: list[list[str]] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            keys = sorted(set(ranks_by_key[a]) & set(ranks_by_key[b]))
            if not keys:
                continue
            d = np.array([ranks_by_key[a][k] - ranks_by_key[b][k] for k in keys])
            idx = rng.integers(0, len(d), size=(DIM_TIE_RESAMPLES, len(d)))
            means = d[idx].mean(axis=1)
            p_a = float((means < 0).mean())
            p_b = float((means > 0).mean())
            if p_a >= DIM_TIE_CONFIDENCE:
                better[b] += 1
            elif p_b >= DIM_TIE_CONFIDENCE:
                better[a] += 1
            else:
                tie[a] += 1
                tie[b] += 1
                ties.append([a, b])
    return {m: 1.0 + better[m] + 0.5 * tie[m] for m in models}, ties


def build(paper: dict, lite: dict[str, list[dict]], exclude: set[tuple[str, str]] = frozenset(),
          floors: dict | None = None, skip_dimensions: frozenset[str] = frozenset(),
          skip_metrics: frozenset[tuple[str, str]] = frozenset()) -> dict:
    order = [d for d in DIMENSION_ORDER if d not in skip_dimensions]
    if floors is None:
        floors = json.loads(FLOORS_PATH.read_text(encoding="utf-8")) if FLOORS_PATH.is_file() else {}
    dims = paper["dimensions"]
    models = {k: dict(v) for k, v in paper["models"].items()}
    directions: dict[str, dict[str, str]] = {}
    values: dict = {}
    for dim, spec in dims.items():
        directions[dim] = dict(spec["metrics"])
        n = int(spec.get("n", 5))
        values[dim] = {c: {m: {model: Entry.from_paper(v, n) for model, v in per.items() if (model, dim) not in exclude}
                           for m, per in cell.items()} for c, cell in spec["cells"].items()}
    for dim, metrics in LITE_METRICS.items():   # Lite-only metrics (e.g. new kinematics ones)
        if dim in directions:
            for metric, (_, direction) in metrics.items():
                if directions[dim].setdefault(metric, direction) != direction:
                    raise ValueError(f"{dim}.{metric}: paper direction {directions[dim][metric]} != Lite {direction}")
    for model, rows in lite.items():
        models.setdefault(model, {"display": model.replace("_", " ").title(), "note": "WMEval Lite runs"})
        for dim, cells in lite_cells(rows).items():
            if dim not in values or (model, dim) in exclude:
                continue
            for cell, metrics in cells.items():
                if cell not in values[dim]:      # only the cells the paper defines
                    continue
                for metric, vals in metrics.items():
                    values[dim][cell].setdefault(metric, {})[model] = Entry.from_samples(vals)

    for dim, metric in skip_metrics:                # left out of every cell before ranking
        for metrics in values.get(dim, {}).values():
            metrics.pop(metric, None)

    detail: dict = {}
    for dim in order:
        if dim not in values:
            continue
        cell_ranks: dict[str, list[float]] = defaultdict(list)
        ranks_by_key: dict[str, dict[tuple, float]] = defaultdict(dict)
        cells_out: dict = {}
        floor_excluded: dict = {}
        present = set()
        for cell, metrics in values[dim].items():
            for metric, per in metrics.items():
                if not per:
                    continue
                present.update(per)
                reason = below_floor(dim, cell, metric, per, floors)
                if reason:
                    floor_excluded[f"{cell} · {metric}"] = reason
                higher = directions[dim][metric] == "higher"
                ranks, ties = rank_cell(per, higher, f"{dim}|{cell}|{metric}") if len(per) > 1 and not reason else ({}, [])
                for m, r in ranks.items():
                    cell_ranks[m].append(r)
                    ranks_by_key[m][(cell, metric)] = r
                cells_out.setdefault(cell, {})[metric] = {
                    "direction": directions[dim][metric],
                    "values": {m: {"mean": round(e.mean, 6), "std": None if e.std is None else round(e.std, 6),
                                   "n": e.n, "source": e.source, "rank": ranks.get(m)} for m, e in sorted(per.items())},
                    "ties": ties,
                    **({"floor_excluded": reason} if reason else {}),
                }
        mean_rank = {m: mean(r) for m, r in cell_ranks.items()}
        dim_rank, dim_ties = dimension_ranks(ranks_by_key) if len(ranks_by_key) > 1 else ({}, [])
        if not dim_rank and len(present) == 1:   # single-model dimension: shown, never counted
            dim_rank = {next(iter(present)): 1.0}
        n_ranked = len(dim_rank)
        detail[dim] = {
            "table": dims[dim].get("table"),
            "models_present": sorted(present),
            "models_ranked": sorted(dim_rank, key=lambda m: (dim_rank[m], m)),
            "comparable": n_ranked >= 2,
            "mean_cell_rank": {m: round(v, 4) for m, v in mean_rank.items()},
            "rank_pairs": {m: len(r) for m, r in cell_ranks.items()},
            "dimension_ties": dim_ties,
            "floor_excluded": floor_excluded,
            "rank": dim_rank,
            "percentile": {m: round((n_ranked - r) / (n_ranked - 1), 4) for m, r in dim_rank.items()} if n_ranked >= 2 else {},
            "cells": cells_out,
        }

    ranked_dims = {m: [d for d in detail if m in detail[d]["percentile"]] for m in models}
    listed = [m for m in models if ranked_dims[m]]
    shared = [d for d in detail if detail[d]["comparable"] and listed and all(m in detail[d]["percentile"] for m in listed)]

    def score(m: str, dims_used: list[str]) -> float | None:
        vals = [detail[d]["percentile"][m] for d in dims_used if m in detail[d]["percentile"]]
        return round(mean(vals), 4) if vals else None

    overall = {m: score(m, ranked_dims[m]) for m in models}
    like = {m: score(m, shared) for m in listed} if shared else {}
    overall_rank = averaged_ranks({m: s for m, s in overall.items() if s is not None}, lower_is_better=False)
    like_rank = averaged_ranks(like, lower_is_better=False) if like else {}
    board = []
    for m in models:
        board.append({
            "model": m,
            "display": models[m]["display"],
            "overall_rank": overall_rank.get(m),
            "overall_score": overall[m],
            "coverage": len(ranked_dims[m]),
            "like_for_like_rank": like_rank.get(m),
            "like_for_like_score": like.get(m),
            "dimension_ranks": {d: {"rank": detail[d]["rank"][m], "of": len(detail[d]["rank"]),
                                    "percentile": detail[d]["percentile"].get(m)}
                                for d in detail if m in detail[d]["rank"]},
            "note": models[m].get("note"),
        })
    board.sort(key=lambda r: (r["overall_rank"] is None, r["overall_rank"] or 0, -r["coverage"], r["display"]))
    return {
        "schema_version": 2,
        "method": __doc__.split("\n\n")[1] + "\n\n" + "\n\n".join(__doc__.split("\n\n")[2:5]),
        "tests": {"alpha": ALPHA, "bootstrap_confidence": BOOT_CONFIDENCE, "bootstrap_resamples": BOOT_RESAMPLES},
        "num_dimensions": len(order),
        "like_for_like_dimensions": shared,
        "excluded": sorted(f"{m}:{d}" for m, d in exclude),
        "skipped_dimensions": [d for d in DIMENSION_ORDER if d in skip_dimensions],
        "skipped_metrics": sorted(f"{d}:{m}" for d, m in skip_metrics),
        "leaderboard": board,
        "dimensions": detail,
    }


def _decimals(v: float) -> int:
    text = repr(float(v))
    return 3 if "e" in text else min(3, len(text.partition(".")[2]))


def _row_places(values: dict) -> int | None:
    """Decimals of a (cell, metric) row: the finest precision transcribed from the paper."""
    paper = [x for v in values.values() if v["source"] == "paper" for x in (v["mean"], v["std"]) if x is not None]
    return max(map(_decimals, paper)) if paper else None


def _fmt_value(v: dict, places: int | None) -> str:
    """Mean ± std with shared decimals (row precision; by magnitude for Lite-only rows)."""
    if places is None:
        size = abs(v["mean"])
        places = 1 if size >= 10 else 2 if size >= 1 else 3
    text = f"{v['mean']:.{places}f}"
    return text if v["std"] is None else f"{text}±{v['std']:.{places}f}"


def label(report: dict, dim: str) -> str:
    skipped = set(report.get("skipped_metrics", []))
    if dim == "environment_object_consistency" and {f"{dim}:depth", f"{dim}:dist"} <= skipped:
        return "Character"
    return SHORT[dim]


def _rank(r: float) -> str:
    return f"{r:g}"


def markdown(report: dict, command: str | None = None) -> str:
    dims = [d for d in DIMENSION_ORDER if d in report["dimensions"]]
    n_dims = report["num_dimensions"]
    display = {r["model"]: r["display"] for r in report["leaderboard"]}
    lines = ["# WMEval leaderboard", "",
             "Per-dimension ranks from every (cell, metric) the paper reports, with statistical ties "
             "(seed bootstrap at 95% for per-seed Lite values, Welch t-test at α = 0.05 for paper mean ± std); "
             "cells below their Unity self-match floor are shown but not ranked; "
             "dimension rank = 1 + #better + 0.5 × #tied, where a pair is ordered only if its paired cell ranks "
             "favour one model in ≥ 95% of bootstrap resamples. Overall = mean of the dimension-rank percentiles "
             "(1 = best, 0 = worst among the models ranked in that dimension) over the dimensions a model is ranked in. "
             "Cells show rank/of; `—` not ranked (N/A or not run); `1/1*` single-model dimension (no comparison, not counted).",
             "", "## Overall", "",
             "| Rank | Model | Score | Coverage | " + " | ".join(label(report, d) for d in dims) + " |",
             "|" + "---|" * (4 + len(dims))]
    for row in report["leaderboard"]:
        cells = []
        for d in dims:
            dr = row["dimension_ranks"].get(d)
            cells.append("—" if dr is None else (f"{_rank(dr['rank'])}/{dr['of']}" + ("*" if dr["of"] < 2 else "")))
        score = "–" if row["overall_score"] is None else f"{row['overall_score']:.3f}"
        rank = "–" if row["overall_rank"] is None else _rank(row["overall_rank"])
        lines.append(f"| {rank} | {row['display']} | {score} | {row['coverage']}/{n_dims} | " + " | ".join(cells) + " |")

    shared = report["like_for_like_dimensions"]
    lines += ["", "## Like-for-like", ""]
    if not shared:
        lines.append("No dimension ranks every listed model.")
    else:
        lines += [f"Only the dimensions every listed model is ranked in: {', '.join(label(report, d) for d in shared)}.", "",
                  "| Rank | Model | Score | " + " | ".join(label(report, d) for d in shared) + " |",
                  "|" + "---|" * (3 + len(shared))]
        rows = sorted((r for r in report["leaderboard"] if r["like_for_like_rank"] is not None),
                      key=lambda r: (r["like_for_like_rank"], r["display"]))
        for row in rows:
            cells = [f"{_rank(row['dimension_ranks'][d]['rank'])}/{row['dimension_ranks'][d]['of']}" for d in shared]
            lines.append(f"| {_rank(row['like_for_like_rank'])} | {row['display']} | "
                         f"{row['like_for_like_score']:.3f} | " + " | ".join(cells) + " |")

    lines += ["", "## Per dimension", "",
              "Each cell: mean ± std (paper: over its n generations; Lite: over seeds, bare value = one sample) "
              "and the (cell, metric) rank after the tie test. ↑/↓ = higher/lower is better."]
    for d in dims:
        det = report["dimensions"][d]
        order = det["models_ranked"] + [m for m in det["models_present"] if m not in det["rank"]]
        summary = " · ".join(f"{_rank(det['rank'][m])}. {display[m]}" for m in det["models_ranked"])
        if not det["comparable"]:
            summary = f"single model ({', '.join(display[m] for m in det['models_present'])}) — no comparison"
        lines += ["", f"### {label(report, d)} — {summary}", "", f"Source: {str(det['table']).rstrip('.')}.", "",
                  "| Cell · metric | " + " | ".join(display[m] for m in order) + " |",
                  "|" + "---|" * (1 + len(order))]
        for cell, metrics in det["cells"].items():
            for metric, info in metrics.items():
                arrow = "↑" if info["direction"] == "higher" else "↓"
                places = _row_places(info["values"])
                cells = []
                for m in order:
                    v = info["values"].get(m)
                    if v is None:
                        cells.append("")
                        continue
                    text = _fmt_value(v, places)
                    cells.append(text + (f" ·{_rank(v['rank'])}" if v["rank"] is not None else ""))
                floor = " (below floor, not ranked)" if info.get("floor_excluded") else ""
                lines.append(f"| {cell} · {metric} {arrow}{floor} | " + " | ".join(cells) + " |")
        if det["comparable"]:
            lines.append("| **Dimension rank** (mean cell rank) | " + " | ".join(
                f"**{_rank(det['rank'][m])}** ({det['mean_cell_rank'][m]:.2f})" if m in det["rank"] else "—"
                for m in order) + " |")
            if det["dimension_ties"]:
                lines += ["", "Tied at the dimension level: " + "; ".join(
                    f"{display[a]} = {display[b]}" for a, b in det["dimension_ties"]) + "."]
        if det["floor_excluded"]:
            lines += ["", "Below measurement floor: " + "; ".join(
                f"{k} ({why})" for k, why in det["floor_excluded"].items()) + "."]
    if command:
        lines += ["", f"Generated with `{command}`."]
    return "\n".join(lines) + "\n"


README_START = "<!-- leaderboard:start (generated by tools/build_leaderboard.py --readme; do not edit by hand) -->"
README_END = "<!-- leaderboard:end -->"


def readme_section(report: dict) -> str:
    """Compact overall table for the repository README."""
    dims = [d for d in DIMENSION_ORDER if d in report["dimensions"]]
    n_dims = report["num_dimensions"]
    lines = [README_START, "",
             "| Rank | Model | Score ↑ | Coverage | " + " | ".join(label(report, d) for d in dims) + " |",
             "|" + "---|" * (4 + len(dims))]
    for row in report["leaderboard"]:
        cells = []
        for d in dims:
            dr = row["dimension_ranks"].get(d)
            cells.append("—" if dr is None else f"{_rank(dr['rank'])}/{dr['of']}" + ("*" if dr["of"] < 2 else ""))
        score = "–" if row["overall_score"] is None else f"{row['overall_score']:.3f}"
        rank = "–" if row["overall_rank"] is None else _rank(row["overall_rank"])
        lines.append(f"| {rank} | **{row['display']}** | {score} | {row['coverage']}/{n_dims} | " + " | ".join(cells) + " |")
    lines += ["",
              "Models are ranked separately in each dimension, on the metrics the paper reports for it, scene by scene. "
              "Models that cannot be told apart share a rank (bootstrap over seeds or Welch t-test per metric, then a paired "
              "bootstrap over scene ranks per dimension), and metrics whose Unity self-match error is as large as the spread "
              "between models are not ranked. Score = mean dimension percentile (1 = best, 0 = worst among the models ranked "
              "there) over the dimensions a model supports, so it combines ranks, never raw scores. Cells give rank/models "
              "ranked; `—` means the model cannot express that dimension's actions or was not run; `*` a single-model "
              "dimension, not counted."]
    skipped = [SHORT[d] for d in report.get("skipped_dimensions", [])]
    notes = [f"{', '.join(skipped)} is not included"] if skipped else []
    if label(report, "environment_object_consistency") == "Character":
        notes.append("object and character consistency is ranked on character identity only (depth and pairwise-distance "
                     "geometry are left out)")
    lines += ["", "Generated by `tools/build_leaderboard.py` from `experiments/leaderboard/paper_scores.yaml` and the "
              "Lite results of models scored after the paper" + ("; " + "; ".join(notes) if notes else "")
              + ". Its `--output` report lists every per-metric value and rank."]
    return "\n".join(lines + ["", README_END])


def write_readme(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = text.find(README_START.split(" (")[0]), text.find(README_END)
    if start < 0 or end < start:
        raise SystemExit(f"{path}: add the leaderboard markers first ({README_START.split(' (')[0]} ... {README_END})")
    path.write_text(text[:start] + section + text[end + len(README_END):], encoding="utf-8")


def load_lite(specs: list[str]) -> dict[str, list[dict]]:
    lite: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        name, path = spec.split("=", 1)
        lite[name].extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return dict(lite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--lite", action="append", default=[], help="model=per_case_results.jsonl (repeatable; later wins)")
    parser.add_argument("--exclude", action="append", default=[], help="model:dimension to leave out (repeatable)")
    parser.add_argument("--skip-dimension", action="append", default=[], choices=DIMENSION_ORDER,
                        help="leave a dimension out of the leaderboard entirely (repeatable)")
    parser.add_argument("--skip-metric", action="append", default=[], metavar="DIMENSION:METRIC",
                        help="leave one metric out of a dimension's ranking (repeatable)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--readme", type=Path, help="also rewrite the leaderboard section of this README")
    args = parser.parse_args(argv)
    paper = yaml.safe_load(args.paper.read_text(encoding="utf-8"))
    exclude = {tuple(spec.split(":", 1)) for spec in args.exclude}
    report = build(paper, load_lite(args.lite), exclude=exclude, skip_dimensions=frozenset(args.skip_dimension),
                   skip_metrics=frozenset(tuple(spec.split(":", 1)) for spec in args.skip_metric))
    command = "python tools/build_leaderboard.py " + " ".join(
        [f"--lite {s}" for s in args.lite] + [f"--exclude {s}" for s in args.exclude]
        + [f"--skip-dimension {d}" for d in args.skip_dimension]
        + [f"--skip-metric {m}" for m in args.skip_metric]
        + ([f"--output {args.output}"] if args.output else [])
        + ([f"--readme {args.readme}"] if args.readme else []))
    report["inputs"] = {"paper": str(args.paper.relative_to(ROOT) if args.paper.is_relative_to(ROOT) else args.paper),
                        "lite": args.lite}
    text = markdown(report, command)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output / "leaderboard.md").write_text(text, encoding="utf-8")
    if args.readme:
        write_readme(args.readme, readme_section(report))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
