#!/usr/bin/env python3
"""Export adaptive robustness Tables 4/7 separately from the selected paper tables."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# The Git-backed Overleaf checkout is authoritative for paper integration.  The
# similarly named extracted ZIP directory is retained only as historical input.
PAPER = ROOT.parent / "-arxiv-WorldModel-Eval" / "table"
MODEL_NAMES = {
    "lingbot_world_cam": "LingBot-World (Cam)",
    "matrix_game_2": "Matrix-Game 2.0",
    "matrix_game_3": "Matrix-Game 3.0",
}
TABLE4_METRICS = {
    "dtw": "dtw_m_mean",
    "fde": "fde_m_mean",
    "exact_match_pct": "exact_match_pct",
    "error_rate_pct": "error_rate_pct",
}
TABLE7_METRICS = {
    "depth": "depth_absrel_pct",
    "distance": "pairwise_distance_mae_pct",
    "permanence": "object_permanence_pct",
    "character": "character_consistency_pct",
}
SCENES = ("WanderingGarden", "BilliardGame", "MirrorRollingBall")
VARIANTS = ("perturb_in", "perturb_out", "360_in", "360_out")


@dataclass(frozen=True)
class Cell:
    mean: float
    sample_std: float
    n: int
    values: dict[str, float]


def _report(model: str, case_id: str, cohort: str) -> Path:
    table, scene, variant = case_id.split("/")
    return ROOT / "reports" / table / model / scene / variant / cohort / "per_case_results.jsonl"


def _variance_report(model: str, case_id: str, cohort: str) -> Path:
    return _report(model, case_id, cohort).parent / (
        "variance_comparison.json" if cohort == "30_run" else "variance_gate.json"
    )


def _queue_repeat_count(model: str, case_id: str) -> int:
    queue = ROOT / "runs/tables_4_7/model_inputs" / model / "generation_queue.jsonl"
    if not queue.is_file():
        raise ValueError(f"{model}: missing frozen generation queue")
    repeats = [
        str(row["generation_repeat"])
        for line in queue.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("case_id") == case_id
    ]
    if repeats not in (
        [f"run{index}" for index in range(5)],
        [f"run{index}" for index in range(30)],
    ):
        raise ValueError(
            f"{model}/{case_id}: frozen queue must contain exactly runs 0--4 or 0--29"
        )
    return len(repeats)


def required_repeat_count(model: str, case_id: str) -> int:
    gate_path = _variance_report(model, case_id, "five_run")
    if not gate_path.is_file():
        raise ValueError(f"{model}/{case_id}: missing five-run variance gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("schema_version") != 1
        or gate.get("model") != model
        or gate.get("repeat_count") != 5
        or not isinstance(gate.get("expand_group_to_30"), bool)
    ):
        raise ValueError(f"{model}/{case_id}: invalid five-run variance gate")
    queued = _queue_repeat_count(model, case_id)
    # A paired counterpart can force this cell to 30 even if this cell's own
    # dispersion gate is false. Conversely, a true gate must never fall back
    # to five merely because queue materialization has not happened yet.
    return 30 if queued == 30 or gate["expand_group_to_30"] else 5


def _require_variance_comparison(model: str, case_id: str) -> None:
    path = _variance_report(model, case_id, "30_run")
    if not path.is_file():
        raise ValueError(f"{model}/{case_id}: missing 30-run variance comparison")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != 1
        or value.get("model") != model
        or value.get("repeat_count") != 30
        or not isinstance(value.get("metrics"), dict)
        or not value["metrics"]
    ):
        raise ValueError(f"{model}/{case_id}: invalid 30-run variance comparison")
    for metric, result in value["metrics"].items():
        tests = (result.get("welch_t"), result.get("brown_forsythe"))
        if (
            result.get("n") != 30
            or not isinstance(result.get("mean_ci95"), list)
            or len(result["mean_ci95"]) != 2
            or result.get("first5", {}).get("mean") is None
            or result.get("confirmation25", {}).get("mean") is None
            or any(
                not isinstance(test, dict)
                or "pvalue" not in test
                or "holm_pvalue" not in test
                for test in tests
            )
        ):
            raise ValueError(
                f"{model}/{case_id}: incomplete 30-run statistics for {metric}"
            )


def _require_paired_character_statistics(
    model: str, scene: str, regime: str, repeat_count: int
) -> None:
    cohort = "30_run" if repeat_count == 30 else "five_run"
    filename = "variance_comparison.json" if repeat_count == 30 else "variance_gate.json"
    path = ROOT / "reports/table7" / model / scene / "paired_character" / regime / cohort / filename
    if not path.is_file():
        raise ValueError(
            f"{model}/{scene}/{regime}: missing paired-character {repeat_count}-run statistics"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    metric = value.get("metrics", {}).get(TABLE7_METRICS["character"])
    if (
        value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("model") != model
        or value.get("scene") != scene
        or value.get("regime") != regime
        or value.get("repeat_count") != repeat_count
        or not isinstance(metric, dict)
        or metric.get("n") != repeat_count
    ):
        raise ValueError(f"{model}/{scene}/{regime}: invalid paired-character statistics")
    if repeat_count == 30:
        for test in ("welch_t", "brown_forsythe"):
            result = metric.get(test)
            if (
                not isinstance(metric.get("mean_ci95"), list)
                or len(metric["mean_ci95"]) != 2
                or metric.get("first5", {}).get("mean") is None
                or metric.get("confirmation25", {}).get("mean") is None
                or not isinstance(result, dict)
                or "pvalue" not in result
                or "holm_pvalue" not in result
            ):
                raise ValueError(
                    f"{model}/{scene}/{regime}: incomplete paired-character 30-run statistics"
                )


def _completed_rows(path: Path, model: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("model") == model and row.get("status") == "completed"]
    rows.sort(key=lambda row: int(str(row.get("seed", "run-1")).removeprefix("run")))
    return rows


def rows_for(model: str, case_id: str) -> list[dict[str, Any]]:
    expected = required_repeat_count(model, case_id)
    cohort = "30_run" if expected == 30 else "five_run"
    rows = _completed_rows(_report(model, case_id, cohort), model)
    seeds = [str(row.get("seed")) for row in rows]
    if len(rows) != expected or seeds != [f"run{index}" for index in range(expected)]:
        raise ValueError(
            f"{model}/{case_id}: required {expected}-run evaluation cohort is incomplete"
        )
    if expected == 30:
        _require_variance_comparison(model, case_id)
    return rows


def metric_values(
    model: str,
    case_id: str,
    metric: str,
    *,
    allow_undefined: bool = False,
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for row in rows_for(model, case_id):
        value = row.get("metrics", {}).get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            if allow_undefined:
                return None
            raise ValueError(f"{model}/{case_id}/{row.get('seed')}: missing numeric {metric}")
        values[str(row["seed"])] = float(value)
    return values


def summarize(values: dict[str, float]) -> Cell:
    numeric = list(values.values())
    if len(numeric) not in {5, 30}:
        raise ValueError(f"expected 5 or 30 values, got {len(numeric)}")
    return Cell(statistics.fmean(numeric), statistics.stdev(numeric), len(numeric), values)


def cell(model: str, case_id: str, metric: str) -> Cell:
    values = metric_values(model, case_id, metric)
    assert values is not None
    return summarize(values)


def optional_cell(model: str, case_id: str, metric: str) -> Cell | None:
    values = metric_values(model, case_id, metric, allow_undefined=True)
    return summarize(values) if values is not None else None


def paired_cell(model: str, first_case: str, second_case: str, metric: str) -> Cell | None:
    first = metric_values(model, first_case, metric, allow_undefined=True)
    second = metric_values(model, second_case, metric, allow_undefined=True)
    if first is None or second is None:
        return None
    if set(first) != set(second):
        raise ValueError(f"unpaired repeat IDs: {first_case} versus {second_case}")
    paired = {seed: (first[seed] + second[seed]) / 2.0 for seed in first}
    result = summarize(paired)
    _, first_scene, first_motion = first_case.split("/")
    _, second_scene, second_motion = second_case.split("/")
    first_regime = first_motion.removesuffix("_in")
    second_regime = second_motion.removesuffix("_out")
    if first_scene != second_scene or first_regime != second_regime:
        raise ValueError(f"invalid paired cases: {first_case} versus {second_case}")
    _require_paired_character_statistics(model, first_scene, first_regime, result.n)
    return result


def _mark(value: Cell) -> str:
    return r"\textsuperscript{\ensuremath{\dagger}}" if value.n == 30 else ""


def _t4(value: Cell, percent: bool = False) -> str:
    digits = 1 if percent else 2
    return f"${value.mean:.{digits}f} \\pm {value.sample_std:.{digits}f}${_mark(value)}"


def _t7(value: Cell | None) -> str:
    if value is None:
        return "N/D"
    return f"{value.mean:.1f}\\sdev{{{value.sample_std:.1f}}}{_mark(value)}"


def table4_block(model: str) -> tuple[str, dict[str, Any]]:
    race = "table4/RacingCar/third"
    billiard = "table4/BilliardGame/first"
    values = {
        name: {"race": cell(model, race, source), "billiard": cell(model, billiard, source)}
        for name, source in TABLE4_METRICS.items()
    }
    block = rf"""\multirow{{4}}{{*}}{{{MODEL_NAMES[model]}}}
& \multirow{{2}}{{*}}{{Kinematics}}
& DTW $\downarrow$
& N/A & {_t4(values['dtw']['race'])}
& {_t4(values['dtw']['billiard'])} & N/A \\
&
& FDE $\downarrow$
& N/A & {_t4(values['fde']['race'])}
& {_t4(values['fde']['billiard'])} & N/A \\
& \multirow{{2}}{{*}}{{Mechanical Laws}}
& Exact Match (\%) $\uparrow$
& N/A & {_t4(values['exact_match_pct']['race'], True)}
& {_t4(values['exact_match_pct']['billiard'], True)} & N/A \\
&
& Error Rate (\%) $\downarrow$
& N/A & {_t4(values['error_rate_pct']['race'], True)}
& {_t4(values['error_rate_pct']['billiard'], True)} & N/A \\
"""
    return block, values


def table7_block(model: str) -> tuple[str, dict[str, Any]]:
    case_ids = [f"table7/{scene}/{variant}" for scene in SCENES for variant in VARIANTS]
    depth = [optional_cell(model, case_id, TABLE7_METRICS["depth"]) for case_id in case_ids[:8]]
    distance = [optional_cell(model, case_id, TABLE7_METRICS["distance"]) for case_id in case_ids]
    permanence = [optional_cell(model, case_id, TABLE7_METRICS["permanence"]) for case_id in case_ids[:4]]
    char_cells = []
    for scene in SCENES[1:]:
        for regime in ("perturb", "360"):
            char_cells.append(paired_cell(
                model,
                f"table7/{scene}/{regime}_in",
                f"table7/{scene}/{regime}_out",
                TABLE7_METRICS["character"],
            ))
    d = [_t7(value) for value in depth]
    dist = [_t7(value) for value in distance]
    perm = [_t7(value) for value in permanence]
    char = [_t7(value) for value in char_cells]
    block = rf"""\multirow{{4}}{{*}}{{{MODEL_NAMES[model]}}}
& \multirow{{2}}{{*}}{{Env.}}
& Depth $\downarrow$
& {' & '.join(d[:4])}
& {' & '.join(d[4:])}
& N/A & N/A & N/A & N/A \\
&
& Dist. $\downarrow$
& {' & '.join(dist[:4])}
& {' & '.join(dist[4:8])}
& {' & '.join(dist[8:])} \\
&
\multirow{{2}}{{*}}{{Char.}}
& Obj. Perm. $\uparrow$
& {' & '.join(perm)}
& N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A \\
&
& Char. Cons. $\uparrow$
& \multicolumn{{2}}{{c}}{{N/A}} & \multicolumn{{2}}{{c}}{{N/A}}
& \multicolumn{{2}}{{c}}{{{char[0]}}} & \multicolumn{{2}}{{c}}{{{char[1]}}}
& \multicolumn{{2}}{{c}}{{{char[2]}}} & \multicolumn{{2}}{{c}}{{{char[3]}}} \\
"""
    return block, {"depth": depth, "distance": distance, "permanence": permanence, "character": char_cells}


def replace_block(source: str, display_name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"\\multirow\{{4\}}\{{\*\}}\{{{re.escape(display_name)}\}}.*?(?=\n\\midrule|\n\\bottomrule)",
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda _: replacement.rstrip(), source)
    if count != 1:
        raise ValueError(f"expected one source block for {display_name}, replaced {count}")
    return updated


def robustness_caption(source: str, table: int, cells: list[Cell | None]) -> str:
    """Set cohort-aware captions without relying on mutable paper wording."""
    title = {
        4: "Mechanical physical law evaluation.",
        7: "Environmental object and character consistency evaluation.",
    }[table]
    parts = [
        title,
        "Adaptive robustness analysis, separate from the selected paper tables.",
        r"Numeric cells report mean $\pm$ sample standard deviation.",
    ]
    if any(value is not None and value.n == 30 for value in cells):
        parts.append(
            r"$^{\dagger}$ marks 30 generations selected by the variance or paired-cell "
            "expansion rule; unmarked estimates use five generations."
        )
    else:
        parts.append("Estimates use five generations.")
    parts.append("Project Genie values are retained from the supplied paper.")
    parts.append("N/A denotes metrics outside the supported reporting protocol.")
    if any(value is None for value in cells):
        parts.append(
            "N/D denotes a metric without a finite measurement in every required "
            "repeat; no partial-cohort mean is reported."
        )
    if table == 7:
        parts.append(
            "In and Out denote inward- and outward-facing orientations; "
            "character scores average paired In/Out values within each repeat."
        )
    else:
        parts.append("LingBot-World (Cam) is excluded from target-object action evaluation.")
    caption = r"\caption{" + " ".join(parts) + "}\n"
    updated, count = re.subn(
        r"\\caption\{.*?\}\s*(?=\\label\{)",
        lambda _: caption,
        source,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"Table {table}: expected exactly one caption before its label")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables_4_7/final")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    table4 = (PAPER / "main-physical_laws.tex").read_text(encoding="utf-8")
    block4, evidence4 = table4_block("matrix_game_2")
    table4 = replace_block(table4, MODEL_NAMES["matrix_game_2"], block4)
    table4 = re.sub(
        r"\n\\midrule\n\n\\multirow\{4\}\{\*\}\{LingBot-World \(Cam\)\}.*?(?=\n\\midrule)",
        "\n\n% LingBot-World Base Cam is outside Table 4 target-action scope.\n",
        table4,
        flags=re.DOTALL,
    )

    table7 = (PAPER / "main-consistency.tex").read_text(encoding="utf-8")
    evidence7: dict[str, Any] = {}
    for model in ("lingbot_world_cam", "matrix_game_3", "matrix_game_2"):
        block, evidence = table7_block(model)
        table7 = replace_block(table7, MODEL_NAMES[model], block)
        evidence7[model] = evidence

    cells4: list[Cell | None] = [
        value for group in evidence4.values() for value in group.values()
    ]
    cells7: list[Cell | None] = []
    for model_evidence in evidence7.values():
        for values in model_evidence.values():
            cells7.extend(values)
    all_cells = cells4 + cells7
    numeric_cells = [value for value in all_cells if value is not None]
    table4 = robustness_caption(table4, 4, cells4)
    table7 = robustness_caption(table7, 7, cells7)

    (output / "table4.tex").write_text(table4, encoding="utf-8")
    (output / "table7.tex").write_text(table7, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "reporting_scope": "adaptive_robustness_not_current_paper_selection",
        "paper_selection": "Table 4 frozen by user; Table 7 uses original five-run cohorts",
        "table4": str(output / "table4.tex"),
        "table7": str(output / "table7.tex"),
        "thirty_run_cells_present": any(value.n == 30 for value in numeric_cells),
        "undefined_locked_evaluator_cells": sum(value is None for value in all_cells),
        "character_merge": "per-repeat arithmetic mean of paired In/Out results, then sample statistics over repeats",
        "table4_scope": "Matrix-Game 2 active RacingCar third-person and Billiard first-person; other columns N/A; LingBot Base Cam omitted",
    }
    (output / "export_receipt.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
