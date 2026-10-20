#!/usr/bin/env python3
"""Author the reference-side inputs the synthetic trajectories' metrics need.

Unity renders (``tools/render_synthetic_references.py run``) give every
synthetic trajectory its video and ground truth. A few metrics also need inputs
that the core cases ship pre-computed in ``reference/artifacts/``; this tool
produces the same files for the synthetic references, with the same tools and
pinned settings, into the render source of truth
``runs/synthetic_unity/<family>/<Scene>__<variant>/<slug>/artifacts/``. Then
``render_synthetic_references.py publish`` copies them into the dataset tree
and ``synthetic/references.json``.

Subcommands::

    contact-events   lpips: judge the reference N times, keep the consensus
                     (tools/freeze_reference_events.py; ball_force, beam_domino)
    liquid-events    lpips: empty contact_events.json for liquid_container
                     (no judging; see the reason recorded in the file)
    causal-qa        physical_causality: segments from the Unity log
                     (tools/detect_beamdomino_segments.py), VLM-decorated QA
                     (tools/generate_beamdomino_causal_step_qa.py --mode vlm)
                     and recorded reference judge answers
                     (tools/record_causal_reference_answers.py); a trajectory
                     whose ball starts no chain is marked not applicable
    rubrics          prompt_scene_consistency: rubrics for conditioning cases
                     that have none (generated from the scene description)
    usage            calls, tokens and estimated cost from a usage log

fluid_mechanics masks/flow: ``tools/build_liquid_reference_artifacts.py --synthetic``.

Hosted-judge subcommands need OPENAI_API_KEY; set WMEVAL_VLM_USAGE_LOG to
record every call's tokens (``usage`` sums them).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SUITE_ROOT = ROOT / "eval_sets/wmeval_lite_v1"
RENDERS = ROOT / "runs/synthetic_unity"
EVENTS = "contact_events.json"
SEGMENTS = "causal_segments.json"
QA = "qa_dataset.json"
ANSWERS = "judge_answers.json"

# USD per 1M tokens (input, cached input, output incl. reasoning) used by `usage`
# to estimate spend. Assumed list prices; the token counts are exact.
PRICES = {
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}


def _suite():
    from evaluation.lite.suite_loader import load_suite

    return load_suite(SUITE_ROOT)


def _entries(suite, family: str, args: argparse.Namespace) -> list[Any]:
    """Synthetic cases of ``family`` (filters: --conditioning, --only, --limit)."""
    from evaluation.lite.synthetic import build_case, load_trajectories

    only = {s.strip() for s in (getattr(args, "only", None) or "").split(",") if s.strip()}
    out = []
    for t in load_trajectories(suite):
        if t.family != family:
            continue
        if getattr(args, "conditioning", None) and t.conditioning_case != args.conditioning:
            continue
        if only and t.pattern not in only:
            continue
        render = RENDERS / t.path.removeprefix("synthetic/")
        if not (render / "reference.mp4").is_file():
            print(f"skip {t.synthetic_id}: no render", flush=True)
            continue
        case = build_case(suite, t)
        # Judge the render itself (the published copy is identical after publish).
        case = dataclasses.replace(case, reference_video=render / "reference.mp4")
        out.append((t, case, render))
    limit = getattr(args, "limit", None)
    return out[:limit] if limit else out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- lpips
def cmd_contact_events(args: argparse.Namespace) -> int:
    from freeze_reference_events import freeze_case

    suite = _suite()
    entries = _entries(suite, args.family, args)
    print(f"[events] {len(entries)} {args.family} trajectory(ies), {args.passes} passes each", flush=True)
    for t, case, render in entries:
        out = render / "artifacts" / EVENTS
        if out.is_file() and not args.overwrite:
            doc = json.loads(out.read_text(encoding="utf-8"))
            if doc.get("reference_video_sha256") == _sha256(case.reference_video):
                print(f"[events] keep {t.synthetic_id}", flush=True)
                continue
        if args.dry_run:
            print(f"[events] would judge {t.synthetic_id}", flush=True)
            continue
        freeze_case(
            case, suite.dimensions["lpips"], passes=args.passes, min_support=None,
            allow_empty=True, workers=args.workers, output=out,
        )
    return 0


def cmd_liquid_events(args: argparse.Namespace) -> int:
    suite = _suite()
    lpips = suite.dimensions["lpips"].metric_config
    entries = _entries(suite, "liquid_container", args)
    for t, case, render in entries:
        doc = {
            "schema_version": 1,
            "case_id": t.synthetic_id,
            "reference_video_sha256": _sha256(case.reference_video),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge": {
                "provider": "openai",
                "model": lpips.get("lpips_vlm_model"),
                "coarse_model": lpips.get("lpips_vlm_coarse_model"),
                "prompt_version": None,
                "status": "not_judged",
            },
            "reason": (
                "not judged: the sealed bottle has no contact or collision episode "
                "(the four core LiquidContainer cases judged 5 passes each found 0 "
                "consensus events), and liquid LPIPS does not count toward the "
                "headline, which falls back to temporal LPIPS on these cases. "
                "Recorded empty, as the core liquid cases' consensus is."
            ),
            "consensus": {
                "rule": "events supported by at least min_support distinct passes "
                        "(frame-proximity clusters per side)",
                "passes": 0,
                "min_support": None,
                "cluster_gap_frames": None,
                "clusters": [],
            },
            "passes": [],
            "selected": {
                "payload": {"events": [], "reasoning_steps": []},
                "grid_frame_indices": [],
            },
        }
        out = render / "artifacts" / EVENTS
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[liquid-events] wrote {out.relative_to(ROOT)}", flush=True)
    return 0


# --------------------------------------------------------------------------- causal qa
def _video_id(synthetic_id: str) -> str:
    return synthetic_id.replace("/", "__")


def _segments(render: Path, synthetic_id: str) -> dict[str, Any]:
    from detect_beamdomino_segments import (
        SYNTHETIC_BALL_RADIUS_M,
        SYNTHETIC_CONTACT_MARGIN_M,
        SegmentDetectionError,
        detect_segments,
        log_from_synthetic_capture,
    )

    capture = json.loads((render / "artifacts/unity_gt.json").read_text(encoding="utf-8"))
    log = log_from_synthetic_capture(capture)
    record: dict[str, Any] = {
        "schema_version": 1,
        "synthetic_id": synthetic_id,
        "source": "artifacts/unity_gt.json via detect_beamdomino_segments.log_from_synthetic_capture "
                  "(geometric contacts) + detect_segments (default thresholds)",
        "contact_model": {"ball_radius_m": SYNTHETIC_BALL_RADIUS_M,
                          "contact_margin_m": SYNTHETIC_CONTACT_MARGIN_M},
    }
    try:
        segments = detect_segments(log)
    except SegmentDetectionError as exc:
        record.update({"trajectory_shape": None, "events": [], "reason": str(exc)})
        return record
    record.update(segments)
    record["source_trajectory_id"] = record.pop("trajectory_id", None)
    return record


def cmd_causal_qa(args: argparse.Namespace) -> int:
    suite = _suite()
    entries = _entries(suite, "beam_domino", args)
    stages = {"segments", "qa", "answers"} if args.stage == "all" else {args.stage}
    for t, case, render in entries:
        art = render / "artifacts"
        if "segments" in stages:
            record = _segments(render, t.synthetic_id)
            (art / SEGMENTS).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            applicability = art / "dimension_applicability.json"
            if record["trajectory_shape"] is None:
                applicability.write_text(json.dumps({
                    "schema_version": 1,
                    "not_applicable": {
                        "physical_causality": f"no causal chain in the Unity reference ({record['reason']})"
                    },
                }, indent=2) + "\n", encoding="utf-8")
                for stale in (QA, ANSWERS):
                    (art / stale).unlink(missing_ok=True)
            else:
                applicability.unlink(missing_ok=True)
            print(f"[causal-qa] {t.pattern}: {record['trajectory_shape']} "
                  f"{[(e['name'], round(e['start_sec'], 2), round(e['end_sec'], 2)) for e in record['events']]}"
                  + (f" -- {record['reason']}" if record.get("reason") else ""), flush=True)
        record = json.loads((art / SEGMENTS).read_text(encoding="utf-8"))
        if record["trajectory_shape"] is None:
            continue
        if "qa" in stages and (args.overwrite or not (art / QA).is_file()):
            if args.dry_run:
                print(f"[causal-qa] would generate QA for {t.pattern}")
            else:
                _generate_qa(record, case.reference_video, art / QA, _video_id(t.synthetic_id), args)
        if "answers" in stages and (args.overwrite or not (art / ANSWERS).is_file()):
            if args.dry_run:
                print(f"[causal-qa] would record {args.passes} reference passes for {t.pattern}")
            else:
                _record_answers(art / QA, case.reference_video, art / ANSWERS, args)
    return 0


def _generate_qa(record: dict, video: Path, out: Path, video_id: str, args: argparse.Namespace) -> None:
    from generate_beamdomino_causal_step_qa import _make_vlm_decorator, _probe_video_fps, build_dataset

    decorator = _make_vlm_decorator(
        video_path=video, video_id=video_id, model=args.qa_model,
        cache_dir=ROOT / "runs/synthetic_scoring/qa_vlm_cache",
        num_frames=args.qa_frames, fps=_probe_video_fps(video),
    )
    dataset = build_dataset(
        video_id=video_id, events=record["events"], decorator=decorator,
        trajectory_shape=record["trajectory_shape"],
    )
    out.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"[causal-qa] wrote {out.relative_to(ROOT)}", flush=True)


def _record_answers(qa_path: Path, video: Path, out: Path, args: argparse.Namespace) -> None:
    from record_causal_reference_answers import _build_judge, _video_fps

    from evaluation.modules._causal_step_recorded_judge import collect_pass_answers, write_recorded_answers
    from evaluation.modules._causal_step_runner import run_causal_step_for_video

    suite_cfg = _suite().dimensions["physical_causality"].metric_config
    judge = _build_judge(
        ROOT / "runs/causal_step_cache", model=str(suite_cfg["causal_step_vlm_model"]),
        num_frames=int(suite_cfg["causal_step_vlm_num_frames"]),
        prompt_version=str(suite_cfg.get("causal_step_vlm_prompt_version", "v1")),
    )
    video_id = json.loads(qa_path.read_text())["videos"][0]["video_id"]
    with tempfile.TemporaryDirectory() as tmp:
        run_causal_step_for_video(
            qa_dataset_path=qa_path, video_id=video_id, video_path=str(video), judge=judge,
            answers_out_dir=Path(tmp), causal_fail_advance=True, fps=_video_fps(video),
            repeats=args.passes,
        )
        passes = collect_pass_answers(Path(tmp), video_id=video_id, repeats=args.passes)
    write_recorded_answers(out, passes=passes, video_id=video_id, judge_settings=judge.describe_settings())
    print(f"[causal-qa] wrote {out.relative_to(ROOT)} ({args.passes} passes)", flush=True)


# --------------------------------------------------------------------------- rubrics
def cmd_rubrics(args: argparse.Namespace) -> int:
    from generate_synthetic_trajectories import scene_description

    from evaluation.lite.synthetic import load_synthetic_cases
    from evaluation.modules.vlm_consistency.rubric_cache import find_rubric, save_rubric
    from evaluation.modules.vlm_consistency.rubric_generator import generate_rubric
    from evaluation.modules.vlm_consistency.schemas import Rubric

    suite = _suite()
    dimension = suite.dimensions["prompt_scene_consistency"]
    cfg = dimension.vlm
    generated: dict[str, Any] = {}
    for case_id, entry in sorted(load_synthetic_cases(suite, dimensions=[dimension.dimension_id]).items()):
        cond = entry.trajectory.conditioning_case
        scene, version, rubric_dir, provider, model = entry.rubric_specs[dimension.dimension_id]
        cond_dir = suite.cases_root / cond / str(cfg["rubric_dir"])
        if find_rubric(scene, version, str(cond_dir), provider=provider, model=model):
            continue  # the conditioning case's rubric applies (same scene description)
        render_dir = RENDERS / entry.trajectory.path.removeprefix("synthetic/") / "artifacts/rubrics"
        if find_rubric(scene, version, str(render_dir), provider=provider, model=model) and not args.overwrite:
            print(f"[rubrics] keep {case_id}")
            continue
        if cond not in generated:
            if args.dry_run:
                print(f"[rubrics] would generate {scene} from {cond}'s scene description")
                generated[cond] = None
                continue
            generated[cond] = generate_rubric(
                prompt_text=scene_description(cond), scene_name=scene, provider=provider, model=model,
            )
            print(f"[rubrics] generated {scene} from {cond}", flush=True)
        if generated[cond] is not None:
            rubric = Rubric.model_validate_json(generated[cond].model_dump_json())
            path = save_rubric(rubric, str(render_dir), prompt_version=version)
            print(f"[rubrics] wrote {path.relative_to(ROOT)}", flush=True)
    return 0


# --------------------------------------------------------------------------- usage
def summarize_usage(path: Path) -> dict[str, Any]:
    per_model: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        m = per_model.setdefault(rec["model"], {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                                                "output_tokens": 0, "reasoning_tokens": 0})
        m["calls"] += 1
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
            m[key] += int(rec.get(key) or 0)
    total = 0.0
    for model, m in per_model.items():
        price_in, price_cached, price_out = PRICES.get(model.removeprefix("azure/openai/"), PRICES["gpt-5.4"])
        uncached = m["input_tokens"] - m["cached_input_tokens"]
        m["estimated_usd"] = round(
            (uncached * price_in + m["cached_input_tokens"] * price_cached + m["output_tokens"] * price_out) / 1e6, 4
        )
        total += m["estimated_usd"]
    return {"per_model": per_model, "calls": sum(m["calls"] for m in per_model.values()),
            "estimated_usd": round(total, 4), "prices_usd_per_1m_tokens": PRICES}


def cmd_usage(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_usage(Path(args.log)), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--only", help="comma-separated trajectory slugs (e.g. s00_roll_push)")
        p.add_argument("--conditioning", help="only trajectories of this conditioning case")
        p.add_argument("--limit", type=int)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("contact-events")
    common(p)
    p.add_argument("--family", required=True, choices=("ball_force", "beam_domino"))
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(func=cmd_contact_events)
    p = sub.add_parser("liquid-events")
    common(p)
    p.set_defaults(func=cmd_liquid_events)
    p = sub.add_parser("causal-qa")
    common(p)
    p.add_argument("--stage", choices=("segments", "qa", "answers", "all"), default="all")
    p.add_argument("--qa-model", default="gpt-5.4")
    p.add_argument("--qa-frames", type=int, default=4, help="frames per QA for the VLM decorator")
    p.add_argument("--passes", type=int, default=5, help="recorded reference judge passes")
    p.set_defaults(func=cmd_causal_qa)
    p = sub.add_parser("rubrics")
    common(p)
    p.set_defaults(func=cmd_rubrics)
    p = sub.add_parser("usage")
    p.add_argument("log")
    p.set_defaults(func=cmd_usage)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
