#!/usr/bin/env python3
"""WMEval Lite — unified benchmark suite CLI.

One command evaluates a world model's rollout videos across every suite
dimension (causal LPIPS, controllability, causal step, liquid physics,
prompt consistency, shadow consistency) and writes per-case JSONL plus an
aggregate summary.

Candidate layout (directory mode): ``<model_root>/<scene>/<variant>/<seed>.mp4``
where the model name is the directory name; synthetic trajectories
(``--split synthetic|full``) live at
``<model_root>/synthetic/<family>/<Scene>__<variant>/<slug>/<seed>.mp4``.
Alternatively pass a strict JSONL manifest (fields: model, case_id, seed,
candidate_video[, params]); a synthetic case_id is its ``synthetic_id``.

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

    # Core cases plus the 220 synthetic trajectories (separate synthetic aggregate)
    python tools/run_wmeval_lite.py run \
        --suite eval_sets/wmeval_lite_v1 --split full \
        --model-root runs/genie3 \
        --output-dir reports/wmeval_lite/genie3

    # Only the reference-free dimensions, resumable
    python tools/run_wmeval_lite.py run \
        --suite eval_sets/wmeval_lite_v1 \
        --model-root runs/genie3 \
        --dimensions prompt_scene_consistency,lighting_shading \
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


def _missing_case_data(suite_root, case_ids=None) -> list[str]:
    """Cases whose manifest-listed reference files are absent (data not downloaded)."""
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(suite_root)
    missing = []
    for case_id in sorted(case_ids or suite.all_case_ids()):
        case_dir = suite.cases_root / case_id
        manifest = case_dir / "reference" / "checksums.sha256"
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip() and not (case_dir / line.split(None, 1)[1].strip()).is_file():
                missing.append(case_id)
                break
    return missing


def _data_hint(missing: list[str], split: str = "core") -> int:
    print(
        f"Case data not found for {len(missing)} case(s) (e.g. {missing[0]}). "
        "Download it from the Hugging Face Hub first:\n"
        "    python tools/sync_lite_data.py download"
        + (" --split full" if split != "core" else ""),
        file=sys.stderr,
    )
    return 2


def _missing_synthetic_data(suite_root, families=None) -> list[str]:
    """Synthetic trajectories (selected families) with listed files absent."""
    from evaluation.lite.suite_loader import load_suite
    from evaluation.lite.synthetic import load_trajectories

    suite = load_suite(suite_root)
    missing = []
    for t in load_trajectories(suite):
        if families and t.family not in families:
            continue
        if any(not (suite.root / t.path / name).is_file() for name in [*t.files, *t.reference_files]):
            missing.append(t.synthetic_id)
    return missing


def _cmd_run(args: argparse.Namespace) -> int:
    from evaluation.lite.suite import run_suite

    from evaluation.lite.synthetic import includes_core, includes_synthetic

    families = _split_csv(args.families)
    if not args.dry_run:
        from evaluation.lite.suite_loader import load_suite

        suite = load_suite(args.suite)
        dims = _split_csv(args.dimensions) or list(suite.dimensions)
        wanted = {c for d in dims if d in suite.dimensions for c in suite.dimensions[d].cases}
        missing = _missing_case_data(args.suite, wanted) if includes_core(args.split) else []
        if includes_synthetic(args.split):
            missing += _missing_synthetic_data(args.suite, families)
        if missing:
            return _data_hint(missing, args.split)

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
        split=args.split,
        families=families,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite
    from evaluation.lite.synthetic import includes_core, includes_synthetic
    from evaluation.lite.validation import CaseValidationError, validate_case_bundle

    suite = load_suite(args.suite)
    if not args.skip_checksums:
        missing = _missing_case_data(args.suite) if includes_core(args.split) else []
        if includes_synthetic(args.split):
            missing += _missing_synthetic_data(args.suite)
        if missing:
            return _data_hint(missing, args.split)
    failures: list[str] = []
    checked = 0
    if includes_synthetic(args.split):
        checked += _validate_synthetic(suite, args, failures)
    for case_id in (sorted(suite.all_case_ids()) if includes_core(args.split) else []):
        checked += 1
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
    print(f"\nAll {checked} case bundle(s) validated.")
    return 0


def _validate_synthetic(suite, args: argparse.Namespace, failures: list[str]) -> int:
    from evaluation.lite.synthetic import (
        SyntheticSuiteError,
        load_synthetic_cases,
        load_trajectories,
        validate_synthetic_case,
    )
    from evaluation.lite.validation import CaseValidationError

    trajectories = load_trajectories(suite)
    entries = load_synthetic_cases(suite)
    for t in trajectories:
        if t.synthetic_id not in entries:
            reason = (
                f"no Unity reference ({t.reference_status})"
                if t.reference_status != "unity_rendered"
                else "no dimension of this suite"
            )
            failures.append(f"{t.synthetic_id}: {reason}")
            print(f"FAIL {t.synthetic_id}: {reason}", file=sys.stderr)
    for case_id, entry in sorted(entries.items()):
        try:
            validate_synthetic_case(entry, verify_checksums=not args.skip_checksums)
        except (ValueError, CaseValidationError, SyntheticSuiteError) as exc:
            failures.append(f"{case_id}: {exc}")
            print(f"FAIL {case_id}: {exc}", file=sys.stderr)
        else:
            skipped = "".join(
                f" ({d} n/a: {reason})" for d, reason in sorted(entry.not_applicable.items())
            )
            print(f"ok   {case_id} [{', '.join(entry.dimensions)}]{skipped}")
    return len(trajectories)


def _cmd_list_cases(args: argparse.Namespace) -> int:
    from evaluation.lite.suite_loader import load_suite
    from evaluation.lite.synthetic import includes_core, includes_synthetic

    suite = load_suite(args.suite)
    if includes_synthetic(args.split):
        _list_synthetic(suite)
    if not includes_core(args.split):
        return 0
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

    # Case × dimension matrix: which dimensions evaluate each case.
    dimension_ids = list(suite.dimensions)
    all_cases = sorted(suite.all_case_ids())
    print("\nCase × dimension matrix:")
    for index, dimension_id in enumerate(dimension_ids, start=1):
        print(f"  d{index} = {dimension_id}")
    width = max(len(case_id) for case_id in all_cases) + 2
    header = " " * width + "  ".join(
        f"d{index}" for index in range(1, len(dimension_ids) + 1)
    )
    print(f"\n{header}")
    for case_id in all_cases:
        marks = "  ".join(
            "✓ " if case_id in suite.dimensions[dimension_id].cases else "· "
            for dimension_id in dimension_ids
        )
        print(f"{case_id:<{width}}{marks}")
    return 0


def _list_synthetic(suite) -> None:
    """Synthetic trajectories per family and dimension (scored / not applicable)."""
    from evaluation.lite.synthetic import load_synthetic_cases

    entries = load_synthetic_cases(suite)
    by_family: dict[str, list] = {}
    for entry in entries.values():
        by_family.setdefault(entry.trajectory.family, []).append(entry)
    print(f"synthetic split — {len(entries)} trajectory(ies) with a Unity reference")
    for family, members in sorted(by_family.items()):
        conditioning = sorted({e.trajectory.conditioning_case for e in members})
        weak = sum(1 for e in members if e.trajectory.reference_quality == "weak_first_frame")
        print(f"\n{family} ({len(members)}; conditioning {', '.join(conditioning)}"
              + (f"; {weak} weak_first_frame" if weak else "") + ")")
        for dimension_id in suite.dimensions:
            scored = sum(1 for e in members if dimension_id in e.dimensions)
            skipped = [e for e in members if dimension_id in e.not_applicable]
            if scored or skipped:
                print(f"  {dimension_id}: {scored} scored"
                      + (f", {len(skipped)} not applicable" if skipped else ""))
                for e in skipped:
                    print(f"    - {e.trajectory.synthetic_id}: {e.not_applicable[dimension_id]}")
    print()


def _cmd_generate_rubrics(args: argparse.Namespace) -> int:
    from evaluation.lite.authoring import author_case
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
        for case_id in dimension.cases:
            case = load_case(
                suite.cases_root / case_id, allowed_cases=suite.all_case_ids()
            )
            # case-relative: the rubric lives inside the checksummed bundle
            rubric_dir = case.root / str(cfg["rubric_dir"])
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
            # the bundle changed: refresh checksums + evaluator lock
            author_case(case.root, relock=True)
            print(f"wrote {path} (case re-authored)")
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

    def add_split_arg(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--split",
            choices=("core", "synthetic", "full"),
            default="core",
            help="core = curated cases (test_core); synthetic = the synthetic "
            "trajectories; full = both (test_full). Default: %(default)s",
        )

    run_parser = subparsers.add_parser(
        "run", help="Evaluate candidate rollouts across suite dimensions"
    )
    add_suite_arg(run_parser)
    run_parser.add_argument(
        "--model-root",
        default=None,
        help="Candidate directory: <model_root>/<scene>/<variant>/<seed>.mp4 "
        "(synthetic: <model_root>/synthetic/<family>/<Scene>__<variant>/<slug>/<seed>.mp4)",
    )
    add_split_arg(run_parser)
    run_parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated synthetic families to score (default: all)",
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
    add_split_arg(validate_parser)
    validate_parser.add_argument("--skip-checksums", action="store_true")
    validate_parser.set_defaults(func=_cmd_validate)

    list_parser = subparsers.add_parser(
        "list-cases", help="List dimensions and their cases"
    )
    add_suite_arg(list_parser)
    add_split_arg(list_parser)
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
