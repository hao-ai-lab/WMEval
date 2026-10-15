#!/usr/bin/env python3
"""Export the expected generation jobs for one model (case x seed), with N/A cases.

This is the model-neutral half of a new-model adapter: it reads the suite, asks
the action-backend registry whether the model can express each case's declared
action target, and writes one job per (case, seed) with everything the
model-specific generator needs -- first frame, prompt, the baked action files and
the reference video's timing. Cases the model cannot express are listed as
``not_applicable`` with the registry reason, so coverage can be checked later
(``repeat_policy.required_repeats`` in suite.yaml).

Output (``--output`` directory):

* ``expected_jobs.json`` -- ``{"jobs": [...], "not_applicable": [...], ...}``
  in the same spirit as ``baselines/lighting_orbit_v2/expected_jobs.json``;
* ``candidates.template.jsonl`` -- the candidate manifest the runner expects,
  with ``candidate_video`` set to ``<model-root>/<Scene>/<variant>/<seed>.mp4``.

Usage::

    python tools/expected_jobs.py --model my_model --output runs/my_model
    python tools/expected_jobs.py --model odyssey --dimensions prompt_scene_consistency \\
        --output runs/odyssey/prompt_jobs

Then generate each job's video at its ``candidate_video`` path and run
``tools/run_wmeval_lite.py run --candidate-manifest <output>/candidates.template.jsonl``.
See ``tools/model_adapters/TEMPLATE.md`` for the generator side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.lite.action_support import support_record, unsupported_reason  # noqa: E402
from evaluation.lite.case_loader import load_case  # noqa: E402
from evaluation.lite.suite_loader import load_suite  # noqa: E402

DEFAULT_SUITE = ROOT / "eval_sets/wmeval_lite_v1"
DEFAULT_SEEDS = ("42", "1", "2", "3", "4")
ACTION_FILES = ("trajectory.yaml", "action.npz", "actions.jsonl", "actions.json", "actions.txt", "action.npy")


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _reference_timing(video: Path) -> dict[str, Any] | None:
    try:
        from evaluation import video_io

        meta = video_io.get_video_metadata(str(video))
    except Exception:  # noqa: BLE001 -- timing is advisory; probing may be unavailable
        return None
    return {
        "fps": round(float(meta.fps), 4),
        "width": int(meta.width),
        "height": int(meta.height),
        "frame_count": int(meta.frame_count),
        "duration_sec": round(float(meta.duration_sec), 4) if meta.duration_sec else None,
    }


def build_jobs(
    *,
    suite_root: Path,
    model: str,
    dimensions: list[str] | None,
    seeds: list[str],
    model_root: Path,
    probe: bool = True,
) -> dict[str, Any]:
    suite = load_suite(suite_root)
    selected = dimensions or list(suite.dimensions)
    unknown = [d for d in selected if d not in suite.dimensions]
    if unknown:
        raise SystemExit(f"unknown dimension(s): {', '.join(unknown)}")
    case_dims: dict[str, list[str]] = {}
    for dimension_id in selected:
        for case_id in suite.dimensions[dimension_id].cases:
            case_dims.setdefault(case_id, []).append(dimension_id)

    jobs: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    for case_id in sorted(case_dims):
        case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
        artifacts = Path(case.artifacts_dir)
        support = support_record(artifacts, model)
        reason = unsupported_reason(artifacts, model)
        if reason is not None:
            not_applicable.append({"case_id": case_id, "dimensions": case_dims[case_id],
                                   "model": model, "action_support": support, "reason": reason})
            continue
        first_frame = getattr(case, "first_frame", None)
        conditioning = {
            "first_frame": _rel(Path(first_frame)) if first_frame else None,
            "first_frame_fallback": "frame 0 of the reference video" if not first_frame else None,
            "prompt": case.prompt,
            "perspective": getattr(case, "perspective", None),
            "action_files": {name: _rel(artifacts / name) for name in ACTION_FILES if (artifacts / name).is_file()},
            "reference_video": _rel(Path(case.reference_video)) if case.reference_video else None,
            "reference_timing": (_reference_timing(Path(case.reference_video))
                                 if probe and case.reference_video else None),
            "candidate_normalization": (case.protocol or {}).get("candidate_normalization", "none"),
        }
        for seed in seeds:
            jobs.append({
                "model": model,
                "case_id": case_id,
                "dimensions": case_dims[case_id],
                "seed": seed,
                "status": "expected",
                "action_support": support,
                "candidate_video": str(model_root / case_id / f"{seed}.mp4"),
                "conditioning": conditioning,
            })
    return {
        "schema_version": 1,
        "suite": suite.suite_name,
        "suite_version": suite.version,
        "model": model,
        "dimensions": selected,
        "seeds": seeds,
        "required_repeats": suite.required_repeats,
        "jobs": jobs,
        "not_applicable": not_applicable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", required=True, help="model / backend name (as in action_backends.yaml)")
    parser.add_argument("--dimensions", help="comma-separated (default: all)")
    parser.add_argument("--seeds", default=",".join(DEFAULT_SEEDS))
    parser.add_argument("--model-root", type=Path, help="where videos will be written (default: --output)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-probe", action="store_true", help="skip reading reference video timing")
    args = parser.parse_args(argv)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model_root = (args.model_root or output).resolve()
    report = build_jobs(
        suite_root=args.suite,
        model=args.model,
        dimensions=args.dimensions.split(",") if args.dimensions else None,
        seeds=[s.strip() for s in args.seeds.split(",") if s.strip()],
        model_root=model_root,
        probe=not args.no_probe,
    )
    (output / "expected_jobs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "candidates.template.jsonl").write_text(
        "".join(json.dumps({"model": j["model"], "case_id": j["case_id"], "seed": j["seed"],
                            "candidate_video": j["candidate_video"]}) + "\n" for j in report["jobs"]),
        encoding="utf-8",
    )
    print(json.dumps({"jobs": len(report["jobs"]), "not_applicable": len(report["not_applicable"]),
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
