#!/usr/bin/env python3
"""Judge noise floor for WMEval Lite's judge-scored dimensions.

Three judge-scored dimensions call a sampled hosted VLM (``gpt-5.4`` rejects a
non-default temperature, so every call is drawn at the API default):
``prompt_scene_consistency`` (rubric judge), ``physical_causality``
(CausalStep QA judge) and ``lpips`` (causal LPIPS: the judge finds contact
events on the candidate side; the reference side replays frozen events). This
tool measures how much the headline moves when *only the judge* is re-sampled:

``plan``
    Writes one strict candidate manifest per dimension where each video is
    listed K times (seed ``<seed>@r<k>``), so ``tools/run_wmeval_lite.py run``
    scores K independent judge repeats of the same file. ``--set core`` and
    ``--set synthetic`` list each case's Unity reference video as the candidate
    (the self-match floor); ``--set candidates --from <manifest>`` repeats a
    stored model's videos. ``physical_causality`` rows get their own
    ``causal_step_answers_cache_dir`` per repeat: the CausalStep judge caches
    answers by (video path, item, pass), so without it repeats 2..K would replay
    repeat 1.

``summarize``
    Reads the resulting ``per_case_results.jsonl`` files and writes per-case
    mean / std / min / max over the K repeats, the pooled within-video judge
    std for candidate repeats, the comparison with stored (single-pass) results
    of the same videos (``--stored``), and a minimum detectable difference
    (MDD): the smallest difference between two models' n-seed means that judge
    sampling alone does not produce at 95 % (two-sided),
    ``MDD_n = 1.96 * sqrt(2 / n) * sigma_judge``.

Run the judge through the NVIDIA gateway::

    python tools/measure_judge_floor.py plan --set core --repeats 5 \\
        --out reports/judge_noise_floor/core
    WMEVAL_VLM_USAGE_LOG=reports/judge_noise_floor/core/usage.jsonl \\
    python tools/judge_gateway.py --key-file ~/.credentials/nv_vlm_api_key -- run \\
        --candidate-manifest reports/judge_noise_floor/core/prompt_scene_consistency.jsonl \\
        --dimensions prompt_scene_consistency --output-dir reports/judge_noise_floor/core/prompt
    python tools/measure_judge_floor.py summarize \\
        --selfmatch reports/judge_noise_floor/core/*/per_case_results.jsonl \\
        --output lite/judge_noise_floor.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "lite"

JUDGE_DIMENSIONS = ("prompt_scene_consistency", "physical_causality", "lpips")
# Synthetic families whose trajectories are scored on a judge dimension.
JUDGE_FAMILIES = ("camera_yaw", "beam_domino", "ball_force")
REPEAT_SEP = "@r"
CAUSAL_CACHE_PARAM = "causal_step_answers_cache_dir"

# Metrics summarised per dimension; the first is the headline.
METRICS: dict[str, tuple[str, ...]] = {
    "prompt_scene_consistency": ("overall_score", "env_score", "obj_score"),
    "physical_causality": (
        "csr_delta", "csr_unity", "csr_wm", "amcl_delta", "ws_delta", "rf_delta",
        "dua_delta", "icra_delta", "amcl_wm", "ws_wm", "dua_wm", "icra_wm",
    ),
    "lpips": (
        "lpips_mean", "lpips_causal_mean", "lpips_temporal_fallback_mean",
        "lpips_matched_reference_event_fraction", "lpips_num_candidate_events",
        "lpips_num_matched_events",
    ),
}
Z95 = 1.959964


# --------------------------------------------------------------------------- plan
def repeat_seed(seed: str, k: int) -> str:
    return f"{seed}{REPEAT_SEP}{k}"


def base_seed(seed: str) -> str:
    """``run3@r2`` -> ``run3``; a seed without a repeat suffix is returned as is."""
    head, sep, tail = str(seed).rpartition(REPEAT_SEP)
    return head if sep and tail.isdigit() else str(seed)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def repeat_rows(
    rows: Iterable[Mapping[str, Any]],
    repeats: int,
    *,
    dimension: str,
    cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Each candidate row K times with seed ``<seed>@r<k>``.

    ``physical_causality`` rows get a per-(case, seed, repeat) answers cache so
    every repeat is a fresh judge draw.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    out: list[dict[str, Any]] = []
    for row in rows:
        for k in range(repeats):
            seed = repeat_seed(str(row["seed"]), k)
            new = {
                "model": row["model"],
                "case_id": row["case_id"],
                "seed": seed,
                "candidate_video": str(row["candidate_video"]),
            }
            params = dict(row.get("params") or {})
            if dimension == "physical_causality":
                if cache_root is None:
                    raise ValueError("physical_causality repeats need a cache_root")
                params[CAUSAL_CACHE_PARAM] = str(
                    Path(cache_root) / _slug(str(row["case_id"])) / _slug(f"{row['model']}_{seed}")
                )
            if params:
                new["params"] = params
            out.append(new)
    return out


def core_selfmatch_rows(suite_root: Path, dimension: str) -> list[dict[str, Any]]:
    """One ``unity`` row per core case of ``dimension``: its reference video as the candidate."""
    sys.path.insert(0, str(ROOT))
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(suite_root)
    rows = []
    for case_id in suite.dimensions[dimension].cases:
        case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
        video = case.reference_video
        if video is None:
            # Reference-free dimensions (prompt) do not declare a reference video;
            # the bundle still ships the Unity rollout its frozen scores were judged on.
            frozen = case.reference_artifacts.get("reference_scores")
            if frozen is not None:
                name = json.loads(Path(frozen).read_text(encoding="utf-8")).get("reference_video")
                video = case.root / str(name) if name else None
        if video is None or not Path(video).is_file():
            raise FileNotFoundError(f"{case_id}: no reference video to self-match")
        rows.append({"model": "unity", "case_id": case_id, "seed": "self", "candidate_video": str(video)})
    return rows


def select_synthetic(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    per_case: int,
    families: Sequence[str] = JUDGE_FAMILIES,
) -> list[Mapping[str, Any]]:
    """The first ``per_case`` rendered trajectories of every (family, conditioning case)
    that a judge dimension scores, in manifest order."""
    taken: dict[tuple[str, str], int] = defaultdict(int)
    picked = []
    for t in trajectories:
        if t.get("family") not in families or t.get("reference_status") != "unity_rendered":
            continue
        if not set(t.get("dimensions") or ()) & set(JUDGE_DIMENSIONS):
            continue
        key = (str(t["family"]), str(t["conditioning_case"]))
        if taken[key] >= per_case:
            continue
        taken[key] += 1
        picked.append(t)
    return picked


def synthetic_selfmatch_rows(suite_root: Path, dimension: str, per_case: int) -> list[dict[str, Any]]:
    manifest = json.loads((suite_root / "synthetic/manifest.json").read_text(encoding="utf-8"))
    rows = []
    for t in select_synthetic(manifest["trajectories"], per_case=per_case):
        if dimension not in (t.get("dimensions") or ()):
            continue
        video = suite_root / str(t["reference_video"])
        if not video.is_file():
            raise FileNotFoundError(f"{t['synthetic_id']}: missing {video}")
        rows.append({"model": "unity", "case_id": t["synthetic_id"], "seed": "self",
                     "candidate_video": str(video)})
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_plan(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dims = args.dimensions.split(",") if args.dimensions else list(JUDGE_DIMENSIONS)
    cache_root = (out / "causal_cache").resolve()
    for dim in dims:
        if args.set == "core":
            rows = core_selfmatch_rows(Path(args.suite), dim)
        elif args.set == "synthetic":
            rows = synthetic_selfmatch_rows(Path(args.suite), dim, args.per_case)
        else:
            if not args.from_manifest:
                raise SystemExit("--set candidates needs --from <manifest>")
            wanted = None
            if args.cases:
                wanted = set(args.cases.split(","))
            rows = [r for r in _read_jsonl(Path(args.from_manifest)) if wanted is None or r["case_id"] in wanted]
        if not rows:
            print(f"[plan] {dim}: no rows", file=sys.stderr)
            continue
        repeated = repeat_rows(rows, args.repeats, dimension=dim, cache_root=cache_root)
        path = out / f"{dim}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in repeated), encoding="utf-8")
        print(f"[plan] {dim}: {len(rows)} video(s) x {args.repeats} repeat(s) -> {path}")
    return 0


# --------------------------------------------------------------------------- summarize
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "std": round(stdev(values), 4) if len(values) > 1 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def mdd(sigma: float | None, n: int) -> float | None:
    """Smallest |mean_A - mean_B| over n seeds each that judge noise alone reaches
    with < 5 % probability (two-sided), when each seed is judged once."""
    if sigma is None or n < 1:
        return None
    return round(Z95 * math.sqrt(2.0 / n) * sigma, 4)


def group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], list[Mapping[str, Any]]]:
    """(dimension, case_id, model, base seed) -> the repeat rows (latest per seed wins)."""
    latest: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        latest[(row["dimension"], row["case_id"], str(row.get("model")), str(row.get("seed")))] = row
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (dim, case, model, seed), row in sorted(latest.items()):
        groups[(dim, case, model, base_seed(seed))].append(row)
    return groups


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status"))] += 1
    return dict(counts)


def summarize_video(dim: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-metric statistics over one video's judge repeats."""
    completed = [r for r in rows if r.get("status") == "completed"]
    out: dict[str, Any] = {"repeats": len(rows), "status": _status_counts(rows), "metrics": {}}
    for metric in METRICS.get(dim, ()):
        values = [float(r["metrics"][metric]) for r in completed
                  if _finite((r.get("metrics") or {}).get(metric))]
        if values:
            out["metrics"][metric] = describe(values)
    if dim == "lpips":
        sources: dict[str, int] = defaultdict(int)
        for r in completed:
            sources[str((r.get("metrics") or {}).get("lpips_primary_source"))] += 1
        out["lpips_primary_source"] = dict(sources)
        offsets = [o for o in (matched_event_offset_sec(r) for r in completed) if o is not None]
        if offsets:
            # |candidate contact time - frozen reference contact time| of the matched
            # events: chronological pairing always "matches" when both sides have an
            # event, so the matched fraction alone cannot show a wrong pairing.
            out["metrics"]["event_offset_sec"] = describe(offsets)
    return out


def matched_event_offset_sec(row: Mapping[str, Any]) -> float | None:
    """Mean |e_cand - e_ref| (seconds) over a causal-LPIPS row's matched events."""
    lp = (((row.get("full_report") or {}).get("results") or {}).get("lpips") or {})
    events = lp.get("lpips_per_matched_event") or []
    fps = float(((lp.get("reference_metadata") or {}).get("fps")) or 0.0)
    gaps = [abs(float(e["e_cand"]) - float(e["e_ref"])) for e in events
            if _finite(e.get("e_cand")) and _finite(e.get("e_ref"))]
    if not gaps or fps <= 0:
        return None
    return mean(gaps) / fps


def pooled_std(stds: Sequence[float | None], ns: Sequence[int]) -> float | None:
    """Pooled within-group std, sqrt(sum((n_i - 1) s_i^2) / sum(n_i - 1))."""
    num = den = 0.0
    for s, n in zip(stds, ns):
        if s is None or n < 2:
            continue
        num += (n - 1) * s * s
        den += n - 1
    return round(math.sqrt(num / den), 4) if den else None


def summarize_case(dim: str, videos: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Across the videos (seeds) of one case: pooled judge std per metric and the
    spread of the per-video means (generation + judge)."""
    out: dict[str, Any] = {}
    for metric in METRICS.get(dim, ()):
        stats = [v["metrics"][metric] for v in videos.values() if metric in v["metrics"]]
        if not stats:
            continue
        means = [s["mean"] for s in stats]
        out[metric] = {
            "videos": len(stats),
            "judge_std_pooled": pooled_std([s["std"] for s in stats], [s["n"] for s in stats]),
            "judge_range_max": round(max(s["max"] - s["min"] for s in stats), 4),
            "mean_of_video_means": round(mean(means), 4),
            "std_of_video_means": round(stdev(means), 4) if len(means) > 1 else None,
        }
    return out


def compare_stored(
    dim: str,
    videos: Mapping[str, dict[str, Any]],
    stored: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Re-scored repeat means vs a stored single pass of the same videos (headline metric)."""
    metric = METRICS[dim][0]
    pairs = []
    for seed, video in videos.items():
        row = stored.get(seed)
        value = ((row or {}).get("metrics") or {}).get(metric) if row else None
        if metric in video["metrics"] and _finite(value):
            pairs.append((seed, video["metrics"][metric], float(value)))
    if not pairs:
        return None
    diffs = [m["mean"] - v for _, m, v in pairs]
    inside = sum(1 for _, m, v in pairs if m["min"] <= v <= m["max"])
    z = []
    for _, m, v in pairs:
        if m["std"]:
            z.append((v - m["mean"]) / m["std"])
    return {
        "metric": metric,
        "videos": len(pairs),
        "stored_mean": round(mean(v for _, _, v in pairs), 4),
        "rescored_mean": round(mean(m["mean"] for _, m, _ in pairs), 4),
        "mean_diff_rescored_minus_stored": round(mean(diffs), 4),
        "mean_abs_diff": round(mean(abs(d) for d in diffs), 4),
        "stored_within_rescored_range": f"{inside}/{len(pairs)}",
        "stored_z_rms": round(math.sqrt(mean(x * x for x in z)), 3) if z else None,
        "per_seed": {s: {"rescored": m["mean"], "rescored_std": m["std"], "stored": v} for s, m, v in pairs},
    }


def summarize(
    selfmatch: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] = (),
    stored: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    seeds_for_mdd: int = 5,
) -> dict[str, Any]:
    """``stored`` maps a label (e.g. ``openai_2026-10-12``) to stored single-pass rows
    of the same candidate videos; each label gets its own ``vs_stored`` comparison."""
    report: dict[str, Any] = {"selfmatch": {}, "candidates": {}}
    for (dim, case, model, _seed), rows in group_rows(selfmatch).items():
        entry = summarize_video(dim, rows)
        headline = entry["metrics"].get(METRICS[dim][0], {})
        entry["mdd"] = {f"n{n}": mdd(headline.get("std"), n) for n in (1, seeds_for_mdd)}
        report["selfmatch"].setdefault(dim, {})[case] = entry

    stored_index: dict[str, dict[tuple[str, str, str], dict[str, Mapping[str, Any]]]] = {}
    for label, rows in (stored or {}).items():
        index = stored_index.setdefault(label, defaultdict(dict))
        for row in rows:
            if row.get("status") == "completed":
                index[(row["dimension"], row["case_id"], str(row.get("model")))][str(row.get("seed"))] = row

    per_case_videos: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (dim, case, model, seed), rows in group_rows(candidates).items():
        per_case_videos[(dim, case, model)][seed] = summarize_video(dim, rows)
    for (dim, case, model), videos in sorted(per_case_videos.items()):
        case_summary = summarize_case(dim, videos)
        headline = case_summary.get(METRICS[dim][0], {})
        entry = {
            "videos": videos,
            "summary": case_summary,
            "mdd": {f"n{n}": mdd(headline.get("judge_std_pooled"), n) for n in (1, seeds_for_mdd)},
        }
        for label, index in stored_index.items():
            cmp = compare_stored(dim, videos, index.get((dim, case, model), {}))
            if cmp is not None:
                entry.setdefault("vs_stored", {})[label] = cmp
        report["candidates"].setdefault(dim, {}).setdefault(model, {})[case] = entry
    report["floors"] = floors(report, seeds_for_mdd)
    return report


def synthetic_group(case_id: str) -> str | None:
    """``synthetic/<family>/<Scene>__<variant>/<slug>`` -> ``synthetic/<family>/<Scene>__<variant>``."""
    parts = str(case_id).split("/")
    return "/".join(parts[:3]) if parts[0] == "synthetic" and len(parts) >= 4 else None


def floors(report: Mapping[str, Any], seeds_for_mdd: int = 5) -> dict[str, Any]:
    """Headline floor per core case and per synthetic (family, conditioning case).

    ``sigma_judge`` is the larger of the reference self-match std and the pooled
    candidate-side judge std (a rubric judge is near-deterministic on a perfect
    reference, so the self-match std alone understates the noise on real models).
    """
    out: dict[str, Any] = {}
    for dim, cases in report.get("selfmatch", {}).items():
        headline = METRICS[dim][0]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case, entry in cases.items():
            stats = entry["metrics"].get(headline)
            group = synthetic_group(case)
            if group is not None:
                if stats:
                    groups[group].append(stats)
                continue
            cand_std = None
            for model_cases in report.get("candidates", {}).get(dim, {}).values():
                summary = (model_cases.get(case) or {}).get("summary", {}).get(headline) or {}
                if summary.get("judge_std_pooled") is not None:
                    cand_std = max(cand_std or 0.0, summary["judge_std_pooled"])
            self_std = (stats or {}).get("std")
            known = [v for v in (self_std, cand_std) if v is not None]
            sigma = max(known) if known else None
            out.setdefault(dim, {})[case] = {
                "metric": headline,
                "self_mean": (stats or {}).get("mean"),
                "self_std": self_std,
                "self_range": [(stats or {}).get("min"), (stats or {}).get("max")],
                "self_repeats": (stats or {}).get("n", 0),
                "candidate_judge_std": cand_std,
                "sigma_judge": sigma,
                "mdd_n1": mdd(sigma, 1),
                f"mdd_n{seeds_for_mdd}": mdd(sigma, seeds_for_mdd),
            }
        for group, stats in sorted(groups.items()):
            sigma = pooled_std([s["std"] for s in stats], [s["n"] for s in stats])
            out.setdefault(dim, {})[group] = {
                "metric": headline,
                "trajectories": len(stats),
                "self_mean": round(mean(s["mean"] for s in stats), 4),
                "self_mean_range": [min(s["mean"] for s in stats), max(s["mean"] for s in stats)],
                "self_std_pooled": sigma,
                "repeats_per_trajectory": max(s["n"] for s in stats),
                "mdd_n1": mdd(sigma, 1),
                f"mdd_n{seeds_for_mdd}": mdd(sigma, seeds_for_mdd),
            }
    return out


def _load_many(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths or ():
        rows.extend(_read_jsonl(Path(p)))
    return rows


def _parse_stored(specs: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """``label=path`` (repeatable; a bare path is labelled ``stored``)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs or ():
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = "stored", spec
        out[label].extend(_read_jsonl(Path(path)))
    return dict(out)


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """Drop per-video / per-seed detail, keeping per-case statistics."""
    out = json.loads(json.dumps(report))
    for models in out.get("candidates", {}).values():
        for cases in models.values():
            for entry in cases.values():
                entry.pop("videos", None)
                for cmp in (entry.get("vs_stored") or {}).values():
                    cmp.pop("per_seed", None)
    return out


def cmd_summarize(args: argparse.Namespace) -> int:
    report = summarize(
        _load_many(args.selfmatch),
        _load_many(args.candidates),
        _parse_stored(args.stored),
        seeds_for_mdd=args.seeds,
    )
    if args.compact:
        report = compact(report)
    report = {
        "schema_version": 1,
        "description": (
            "Judge-only variance of WMEval Lite's judge-scored dimensions: the same video "
            "re-judged K times (tools/measure_judge_floor.py). MDD_n = 1.96*sqrt(2/n)*sigma_judge."
        ),
        **({"note": args.note} if args.note else {}),
        **report,
    }
    text = json.dumps(report, indent=1, sort_keys=False) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[summarize] wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write K-repeat candidate manifests")
    p.add_argument("--set", choices=("core", "synthetic", "candidates"), required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suite", default=str(DEFAULT_SUITE))
    p.add_argument("--dimensions", help=f"comma-separated (default: {','.join(JUDGE_DIMENSIONS)})")
    p.add_argument("--per-case", type=int, default=2, help="synthetic trajectories per (family, case)")
    p.add_argument("--from", dest="from_manifest", help="candidate manifest to repeat (--set candidates)")
    p.add_argument("--cases", help="comma-separated case filter (--set candidates)")
    p.set_defaults(func=cmd_plan)
    s = sub.add_parser("summarize", help="per-case judge spread and MDD")
    s.add_argument("--selfmatch", nargs="*", default=[], help="per_case_results.jsonl of self-match repeats")
    s.add_argument("--candidates", nargs="*", default=[], help="per_case_results.jsonl of candidate repeats")
    s.add_argument("--stored", nargs="*", default=[],
                   help="label=per_case_results.jsonl: stored single-pass results of the same candidates")
    s.add_argument("--compact", action="store_true", help="omit per-video detail")
    s.add_argument("--note", help="free-text provenance recorded in the output (transport, date, spend)")
    s.add_argument("--seeds", type=int, default=5, help="seeds per model for MDD (default 5)")
    s.add_argument("--output")
    s.set_defaults(func=cmd_summarize)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
