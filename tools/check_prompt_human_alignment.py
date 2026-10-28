#!/usr/bin/env python3
"""Check a prompt_scene_consistency run against the paper's human study.

The paper reports that the judge ranks models the way blinded human raters do
(Spearman rho = 0.90 over Unity + 4 generative models, 0.80 over the 4
generative models). ``lite/human_alignment/
prompt_scene_consistency.json`` freezes the human side of that comparison.
This tool recomputes the correlation for any Lite run over the same models'
videos -- e.g. after changing the judge model, prompts or rubrics -- and exits
non-zero when it falls below the paper's value.

Judge scores come from ``per_case_results.jsonl``: a model's case score is the
mean ``overall_score`` over its seeds, its pooled score the mean over the six
cases. The Unity reference comes from the frozen ``overall_score_unity``
carried by every row. Map Lite model names to the study's models with
``--model``::

    python tools/check_prompt_human_alignment.py \\
      --results reports/wmeval_lite/<run>/per_case_results.jsonl \\
      --model Genie=project_genie --model LingBot=lingbot_world_cam \\
      --model MatrixGame3=matrix_game_3 --model HY=hy_worldplay
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "lite/human_alignment/prompt_scene_consistency.json"
)
DIMENSION = "prompt_scene_consistency"


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; ties share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        k = i
        while k + 1 < len(order) and values[order[k + 1]] == values[order[i]]:
            k += 1
        for t in range(i, k + 1):
            result[order[t]] = (i + k) / 2 + 1
        i = k + 1
    return result


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return statistics.correlation(ranks(a), ranks(b))


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float:
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = len(a) * (len(a) - 1) / 2
    return (concordant - discordant) / (
        (pairs - ties_a) ** 0.5 * (pairs - ties_b) ** 0.5
    )


def pooled(per_case: Mapping[str, Mapping[str, float]], models: Sequence[str]) -> dict[str, float]:
    return {m: statistics.mean(scores[m] for scores in per_case.values()) for m in models}


def alignment(
    judge: Mapping[str, Mapping[str, float]],
    human: Mapping[str, Mapping[str, float]],
    models: Sequence[str],
) -> dict[str, Any]:
    """Model-level rank agreement between two {case: {model: score}} tables."""
    pj, ph = pooled(judge, models), pooled(human, models)
    a = [pj[m] for m in models]
    b = [ph[m] for m in models]
    return {
        "models": list(models),
        "judge": pj,
        "human": ph,
        "spearman_rho": round(spearman(a, b), 4),
        "kendall_tau_b": round(kendall_tau_b(a, b), 4),
    }


def judge_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[str],
    model_names: Mapping[str, str],
    reference_model: str,
) -> dict[str, dict[str, float]]:
    """{case: {study model: score}} from Lite rows; raises on missing cells."""
    seeds: dict[tuple[str, str], list[float]] = {}
    unity: dict[str, float] = {}
    by_lite_name = {lite: study for study, lite in model_names.items()}
    for row in rows:
        if row.get("dimension") != DIMENSION or row.get("status") != "completed":
            continue
        metrics = row.get("metrics") or {}
        case = row.get("case_id")
        if metrics.get("overall_score_unity") is not None:
            unity[case] = float(metrics["overall_score_unity"])
        study = by_lite_name.get(str(row.get("model")))
        if study is not None and metrics.get("overall_score") is not None:
            seeds.setdefault((case, study), []).append(float(metrics["overall_score"]))

    table: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for case in cases:
        table[case] = {}
        for study in model_names:
            values = seeds.get((case, study))
            if values:
                table[case][study] = statistics.mean(values)
            else:
                missing.append(f"{study} ({model_names[study]}) on {case}")
        if case in unity:
            table[case][reference_model] = unity[case]
    if missing:
        raise ValueError("no completed rows for: " + "; ".join(missing))
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", nargs="+", required=True, type=Path,
                        help="per_case_results.jsonl file(s)")
    parser.add_argument("--model", action="append", default=[], metavar="STUDY=LITE",
                        help="map a study model to its Lite model name (default: same name)")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    reference = fixture["reference_model"]
    generative = [m for m in fixture["models"] if m != reference]
    model_names = {m: m for m in generative}
    for item in args.model:
        study, _, lite = item.partition("=")
        if study not in model_names or not lite:
            parser.error(f"--model {item!r}: expected one of {generative} = <lite model>")
        model_names[study] = lite

    rows = [
        json.loads(line)
        for path in args.results
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    human = fixture["human_overall"]
    try:
        judge = judge_table(rows, cases=list(human), model_names=model_names,
                            reference_model=reference)
    except ValueError as exc:
        print(f"[alignment] {exc}", file=sys.stderr)
        return 2

    has_reference = all(reference in scores for scores in judge.values())
    checks = [("generative_models", generative)]
    if has_reference:
        checks.insert(0, ("all_models", [reference, *generative]))
    else:
        print(f"[alignment] no frozen {reference} scores in the rows; "
              "checking the generative models only")

    failed = False
    for name, models in checks:
        result = alignment(judge, human, models)
        expected = fixture["expected"][name]
        ok = result["spearman_rho"] >= expected["spearman_rho"]
        failed |= not ok
        print(f"[alignment] {name}: rho={result['spearman_rho']} "
              f"tau_b={result['kendall_tau_b']} (paper rho={expected['spearman_rho']}) "
              f"{'ok' if ok else 'BELOW PAPER'}")
        for m in models:
            print(f"    {fixture['models'][m]:22s} judge={result['judge'][m]:.4f} "
                  f"human={result['human'][m]:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
