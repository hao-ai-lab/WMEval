#!/usr/bin/env python3
"""WMEval Lite — unified benchmark suite CLI.

One command evaluates a world model's rollout videos across every suite
dimension (causal LPIPS, controllability, causal step, liquid physics,
prompt consistency, shadow consistency) and writes per-case JSONL plus an
aggregate summary.

Candidate layout (directory mode): ``<model_root>/<scene>/<variant>/<seed>.mp4``
where the model name is the directory name. Alternatively pass a strict JSONL
manifest (fields: model, case_id, seed, candidate_video[, params]).

Examples::

    # Validate the bundled suite data (checksums, bundle completeness)
    python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1

    # See every (dimension, case) the suite defines
    python tools/run_wmeval_lite.py list-cases --suite eval_sets/wmeval_lite_v1

    # Evaluate a model's rollouts across all dimensions
    python tools/run_wmeval_lite.py run \
        --suite eval_sets/wmeval_lite_v1 \
        --model-root runs/genie3 \
        --output-dir reports/wmeval_lite/genie3

    # Only the reference-free dimensions, resumable
    python tools/run_wmeval_lite.py run \
        --suite eval_sets/wmeval_lite_v1 \
        --model-root runs/genie3 \
        --dimensions prompt_consistency,shadow_consistency \
        --resume \
        --output-dir reports/wmeval_lite/genie3

    # Offline authoring: (re)generate prompt-consistency rubrics
    python tools/run_wmeval_lite.py generate-rubrics \
        --suite eval_sets/wmeval_lite_v1
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(repo_root: Path) -> None:
    """Load KEY=VALUE lines from <repo>/.env without overriding the
    environment (mirrors the eval_lite runner-script convention)."""
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _cmd_run(args: argparse.Namespace) -> int:
    from evaluation.lite.suite import run_suite

    run_id = args.run_id or _datetime.datetime.now(_datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )
    aggregate = run_suite(
        suite_root=args.suite,
        output_dir=args.output_dir,
        model_root=args.model_root,
        candidate_manifest=args.candidate_manifest,
        dimensions=_split_csv(args.dimensions),
        device=args.device,
        dry_run=args.dry_run,
        resume=args.resume,
        verify_checksums=not args.skip_checksums,
        run_id=run_id,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite
    from evaluation.lite.validation import CaseValidationError, validate_case_bundle

    suite = load_suite(args.suite)
    failures: list[str] = []
    for case_id in sorted(suite.all_case_ids()):
        try:
            case = load_case(
                suite.cases_root / case_id, allowed_cases=suite.all_case_ids()
            )
            validate_case_bundle(case, verify_checksums=not args.skip_checksums)
        except (ValueError, CaseValidationError) as exc:
            failures.append(f"{case_id}: {exc}")
            print(f"FAIL {case_id}: {exc}", file=sys.stderr)
        else:
            print(f"ok   {case_id}")
    if failures:
        print(f"\n{len(failures)} case(s) failed validation", file=sys.stderr)
        return 1
    print(f"\nAll {len(suite.all_case_ids())} case bundle(s) validated.")
    return 0


def _cmd_list_cases(args: argparse.Namespace) -> int:
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(args.suite)
    print(f"{suite.suite_name} {suite.version} — {len(suite.dimensions)} dimension(s)")
    for dimension_id, dimension in suite.dimensions.items():
        kind = (
            f"registry modules={list(dimension.modules)}"
            if dimension.evaluator == "registry"
            else f"vlm {dimension.vlm.get('provider')}/{dimension.vlm.get('model')}"
        )
        ref = "ref" if dimension.requires_reference else "reference-free"
        print(f"\n{dimension_id} ({dimension.display_name}) [{ref}; {kind}]")
        for case_id in dimension.cases:
            print(f"  - {case_id}")
    return 0


def _cmd_generate_rubrics(args: argparse.Namespace) -> int:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import EVALUATOR_VLM_CONSISTENCY, load_suite
    from evaluation.modules.vlm_consistency.rubric_cache import find_rubric, save_rubric
    from evaluation.modules.vlm_consistency.rubric_generator import generate_rubric

    suite = load_suite(args.suite)
    vlm_dimensions = [
        dimension
        for dimension in suite.dimensions.values()
        if dimension.evaluator == EVALUATOR_VLM_CONSISTENCY
    ]
    if not vlm_dimensions:
        print("suite has no vlm_consistency dimensions; nothing to do")
        return 0

    for dimension in vlm_dimensions:
        cfg = dimension.vlm
        provider = str(cfg["provider"])
        model = str(cfg["model"])
        prompt_version = str(cfg["prompt_version"])
        rubric_dir = suite.root / str(cfg["rubric_dir"])
        for case_id in dimension.cases:
            case = load_case(
                suite.cases_root / case_id, allowed_cases=suite.all_case_ids()
            )
            scene_name = case.rubric_scene_name or case.scene
            if case.prompt is None:
                print(
                    f"SKIP {dimension.dimension_id}/{case_id}: case.yaml has no prompt",
                    file=sys.stderr,
                )
                continue
            existing = find_rubric(
                scene_name, prompt_version, str(rubric_dir), provider=provider, model=model
            )
            if existing is not None and not args.overwrite:
                print(f"keep {scene_name}: {existing}")
                continue
            print(f"generate {scene_name} ({provider}/{model}/{prompt_version}) ...")
            rubric = generate_rubric(
                prompt_text=case.prompt,
                scene_name=scene_name,
                provider=provider,
                model=model,
            )
            path = save_rubric(rubric, str(rubric_dir), prompt_version=prompt_version)
            print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(REPO_ROOT)

    parser = argparse.ArgumentParser(
        prog="run_wmeval_lite",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_suite_arg(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--suite",
            default="eval_sets/wmeval_lite_v1",
            help="Suite root directory containing suite.yaml (default: %(default)s)",
        )

    run_parser = subparsers.add_parser(
        "run", help="Evaluate candidate rollouts across suite dimensions"
    )
    add_suite_arg(run_parser)
    run_parser.add_argument(
        "--model-root",
        default=None,
        help="Candidate directory: <model_root>/<scene>/<variant>/<seed>.mp4",
    )
    run_parser.add_argument(
        "--candidate-manifest",
        default=None,
        help="Strict candidate JSONL (model, case_id, seed, candidate_video[, params])",
    )
    run_parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated dimension ids (default: all)",
    )
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve suite/cases/candidates and report the work plan; no metrics",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip work items already present in the output JSONL",
    )
    run_parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA256 verification of reference bundles (faster; less strict)",
    )
    run_parser.set_defaults(func=_cmd_run)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate every case bundle referenced by the suite"
    )
    add_suite_arg(validate_parser)
    validate_parser.add_argument("--skip-checksums", action="store_true")
    validate_parser.set_defaults(func=_cmd_validate)

    list_parser = subparsers.add_parser(
        "list-cases", help="List dimensions and their cases"
    )
    add_suite_arg(list_parser)
    list_parser.set_defaults(func=_cmd_list_cases)

    rubric_parser = subparsers.add_parser(
        "generate-rubrics",
        help="Offline authoring: generate prompt-consistency rubrics into the suite",
    )
    add_suite_arg(rubric_parser)
    rubric_parser.add_argument("--overwrite", action="store_true")
    rubric_parser.set_defaults(func=_cmd_generate_rubrics)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
