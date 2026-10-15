#!/usr/bin/env python3
"""Composite WMEval leaderboard from per-cell headline values.

Scoring (no cross-metric normalization is invented):

* Within a dimension, every pair of models is compared on each (cell, metric)
  both report; the better value wins (ties count half). A model's dimension
  score is its pairwise win rate x 100 over those comparisons.
* The composite is the mean of the model's dimension scores over the dimensions
  where it has at least one comparison; ``coverage`` says how many. Dimensions
  where a model is N/A (unsupported, or not run) never count against it, and a
  dimension with a single scored model contributes no comparison.
* The core composite averages only the core dimensions (scored for at least
  ``--core-min-models`` models) and is reported only for models covering all of
  them, so it compares like with like.

Inputs: ``experiments/leaderboard/paper_scores.yaml`` (paper tables) plus
optional Lite results for extra models (``--lite name=per_case_results.jsonl``),
mapped onto the same cells (seed means of completed rows).

    python tools/build_leaderboard.py \\
        --lite odyssey=reports/wmeval_lite/odyssey/per_case_results.jsonl \\
        --lite odyssey=reports/wmeval_lite/odyssey_v2_prompt/per_case_results.jsonl \\
        --output reports/leaderboard
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]
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

# Lite case/metric -> paper cell/metric. Character consistency merges In/Out as the paper does.
_SCENES = {"WanderingGarden": "garden", "BilliardGame": "billiard", "MirrorRollingBall": "mirror"}
_PROMPT = {"ChessBoard": "chessboard", "MirrorRollingBall": "mirror", "RacingCar": "racing_car"}
LIGHTING_KEYS = {
    "scene_retention_rate_pct": "scene_retention", "shadow_detection_rate_pct": "shadow_detection",
    "multi_object_sun_disagreement_deg": "mosd", "shadow_direction_jitter_deg_per_frame": "sdj",
    "shadow_area_stability_pct": "area_stability", "shadow_shape_drift": "shape_drift",
}


def lite_cells(rows: list[dict]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """dimension -> cell -> metric -> values (one per completed seed row)."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("status") != "completed":
            continue
        dim, case, metrics = row["dimension"], row["case_id"], row.get("metrics") or {}
        scene, variant = case.split("/")
        def put(cell, metric, value):
            if isinstance(value, (int, float)) and value == value:
                out[dim][cell][metric].append(float(value))
        if dim == "environment_object_consistency" and scene in _SCENES and variant.startswith("scene_"):
            name = variant[len("scene_"):]                      # perturb_in, 360_out, ...
            cell = f"{_SCENES[scene]}/{name}"
            if scene != "MirrorRollingBall":                     # the paper reports no Mirror depth
                put(cell, "depth", metrics.get("depth_absrel_pct"))
            put(cell, "dist", metrics.get("pairwise_distance_mae_pct"))
            if scene != "WanderingGarden":
                put(f"{_SCENES[scene]}/{name.rsplit('_', 1)[0]}", "char", metrics.get("character_consistency_pct"))
        elif dim == "prompt_scene_consistency" and scene in _PROMPT:
            view = "first" if variant.endswith("first") else "third"
            put(f"{_PROMPT[scene]}/{view}", "overall", metrics.get("overall_score"))
        elif dim == "lighting_shading" and scene == "SunRisesAndSets":
            for key, metric in LIGHTING_KEYS.items():
                put("sun_rises_and_sets", metric, metrics.get(key))
    return out


def build(paper: dict, lite: dict[str, list[dict]], exclude: set[tuple[str, str]] = frozenset(),
          core_min_models: int = 5) -> dict:
    dims = paper["dimensions"]
    models = {k: dict(v) for k, v in paper["models"].items()}
    values: dict = {d: {c: {m: dict(per) for m, per in cell.items()} for c, cell in spec["cells"].items()}
                    for d, spec in dims.items()}
    for model, rows in lite.items():
        models.setdefault(model, {"display": model.replace("_", " ").title(), "note": "Lite runs"})
        for dim, cells in lite_cells(rows).items():
            if dim not in values:
                continue
            for cell, metrics in cells.items():
                for metric, vals in metrics.items():
                    if metric in dims[dim]["metrics"] and (model, dim) not in exclude:
                        values[dim].setdefault(cell, {}).setdefault(metric, {})[model] = mean(vals)

    scores: dict = {m: {} for m in models}
    detail: dict = {}
    for dim in DIMENSION_ORDER:
        if dim not in dims:
            continue
        directions = dims[dim]["metrics"]
        wins: dict = defaultdict(float)
        games: dict = defaultdict(int)
        for cell, metrics in values[dim].items():
            for metric, per in metrics.items():
                higher = directions[metric] == "higher"
                for a, b in combinations(sorted(per), 2):
                    va, vb = per[a], per[b]
                    games[a] += 1
                    games[b] += 1
                    if va == vb:
                        wins[a] += 0.5
                        wins[b] += 0.5
                    elif (va > vb) == higher:
                        wins[a] += 1
                    else:
                        wins[b] += 1
        scored = {m for cell in values[dim].values() for per in cell.values() for m in per}
        detail[dim] = {"models_scored": sorted(scored), "comparisons": {m: games[m] for m in scored}}
        for m in scored:
            scores[m][dim] = round(100.0 * wins[m] / games[m], 1) if games[m] else None

    core = [d for d in DIMENSION_ORDER if d in detail and len(detail[d]["models_scored"]) >= core_min_models]
    board = []
    for m, per in scores.items():
        covered = {d: s for d, s in per.items() if s is not None}
        core_scores = [per[d] for d in core if per.get(d) is not None]
        board.append({
            "model": m,
            "display": models[m]["display"],
            "composite": round(mean(covered.values()), 1) if covered else None,
            "core_composite": round(mean(core_scores), 1) if core and len(core_scores) == len(core) else None,
            "coverage": len(covered),
            "dimension_scores": per,
            "note": models[m].get("note"),
        })
    board.sort(key=lambda r: (r["composite"] is None, -(r["composite"] or 0), -r["coverage"]))
    return {"schema_version": 1, "method": __doc__.split("\n\n")[1], "core_dimensions": core,
            "excluded": sorted(f"{m}:{d}" for m, d in exclude), "leaderboard": board, "dimensions": detail}


def markdown(report: dict) -> str:
    dims = [d for d in DIMENSION_ORDER if d in report["dimensions"]]
    head = "| Rank | Model | Composite | Core | Coverage | " + " | ".join(SHORT[d] for d in dims) + " |"
    lines = [head, "|" + "---|" * (5 + len(dims))]
    rank = 0
    for row in report["leaderboard"]:
        rank += 1 if row["composite"] is not None else 0
        cells = []
        for d in dims:
            s = row["dimension_scores"].get(d, "absent")
            cells.append("—" if s == "absent" else ("single" if s is None else f"{s:.0f}"))
        lines.append(f"| {rank if row['composite'] is not None else '–'} | {row['display']} | "
                     f"{row['composite'] if row['composite'] is not None else '–'} | "
                     f"{row['core_composite'] if row['core_composite'] is not None else '–'} | {row['coverage']}/{len(dims)} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--lite", action="append", default=[], help="model=per_case_results.jsonl (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="model:dimension to leave out (repeatable)")
    parser.add_argument("--core-min-models", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paper = yaml.safe_load(args.paper.read_text(encoding="utf-8"))
    lite: dict[str, list[dict]] = defaultdict(list)
    for spec in args.lite:
        name, path = spec.split("=", 1)
        lite[name].extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    exclude = {tuple(spec.split(":", 1)) for spec in args.exclude}
    report = build(paper, dict(lite), exclude=exclude, core_min_models=args.core_min_models)
    text = markdown(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output / "leaderboard.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
