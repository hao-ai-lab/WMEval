#!/usr/bin/env python3
"""Judge each prompt case's Unity reference once and freeze its scores.

The prompt-consistency table reports the Unity render beside every model. The
hosted judge is non-deterministic, so judging the reference again on every run
would make it a moving target. This tool judges the bundled
``reference/unity_groundtruth.mp4`` N times with exactly the settings the
runtime uses (``VlmConsistencyAdapter.judge_video``), keeps every pass -- with
its judge requests and responses -- in ``reference/artifacts/reference_scores.json``
and reports the mean over passes. Runtime reads that file instead of
re-judging, and ignores it once the judge settings it records no longer match.

Requires OPENAI_API_KEY (explicit opt-in to hosted-judge calls)::

    python tools/dataset/freeze_prompt_reference_scores.py --passes 5 --register
    python tools/dataset/build_lite_case.py lite/cases/*/prompt_*
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.lite.adapters.vlm_consistency_adapter import (  # noqa: E402
    REFERENCE_SCORES_ARTIFACT,
    REFERENCE_SCORES_FILENAME,
    REFERENCE_SCORES_SCHEMA_VERSION,
    SCORE_KEYS,
    VlmConsistencyAdapter,
)

SUITE_ROOT = ROOT / "lite"
DIMENSION = "prompt_scene_consistency"
DEFAULT_REFERENCE_VIDEO = "reference/unity_groundtruth.mp4"


def reference_video(case: Any) -> Path:
    """The case's Unity render: ``reference.video`` when declared, else the
    bundled default (prompt cases are reference-free for candidates, so most
    do not declare it)."""
    video = case.reference_video or (case.root / DEFAULT_REFERENCE_VIDEO)
    if not Path(video).is_file():
        raise SystemExit(f"{case.case_id}: no Unity reference video at {video}")
    return Path(video)


def summarize(passes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean (the frozen score), sample std, min and max over passes, per score.
    A score a case has no rubric items for (obj_score on RacingCar_first) stays
    null."""
    summary: dict[str, dict[str, Any]] = {}
    for key in SCORE_KEYS:
        values = [float(p[key]) for p in passes if p.get(key) is not None]
        mean = sum(values) / len(values) if values else None
        std = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            if len(values) > 1
            else None
        )
        summary[key] = {
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(values),
        }
    return summary


def freeze(
    case_id: str,
    *,
    passes: int,
    workers: int = 5,
    suite_root: Path = SUITE_ROOT,
    dimension_id: str = DIMENSION,
    adapter: VlmConsistencyAdapter | None = None,
) -> Path:
    from evaluation.lite.authoring import sha256_file
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(suite_root)
    dimension = suite.dimensions[dimension_id]
    case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
    video = reference_video(case)
    judge = adapter or VlmConsistencyAdapter()
    print(
        f"[freeze] {case_id}: {passes} passes on {video.relative_to(case.root)}, "
        f"{dimension.vlm['provider']}/{dimension.vlm['model']} "
        f"prompt_version={dimension.vlm['prompt_version']}",
        flush=True,
    )

    def one_pass(number: int) -> dict[str, Any]:
        result = judge.judge_video(dimension=dimension, case=case, video=video)
        if result.status != "completed":
            raise SystemExit(f"{case_id} pass {number}: {result.status}: {result.error}")
        report = dict(result.full_report)
        identity = report.pop("judge")
        print(
            f"[freeze]   pass {number}: overall={result.metrics['overall_score']} "
            f"env={result.metrics['env_score']} obj={result.metrics['obj_score']}",
            flush=True,
        )
        return {
            "pass": number,
            **{key: result.metrics[key] for key in SCORE_KEYS},
            "identity": identity,
            "judge_result": report,
        }

    # Passes are independent judge calls; run them concurrently.
    with ThreadPoolExecutor(max_workers=max(1, min(workers, passes))) as pool:
        recorded = list(pool.map(one_pass, range(1, passes + 1)))

    identities = [p.pop("identity") for p in recorded]
    if any(identity != identities[0] for identity in identities):
        raise SystemExit(f"{case_id}: judge settings changed between passes")
    resolved = sorted(
        {
            str(call["response"].get("model_resolved"))
            for p in recorded
            for call in p["judge_result"].get("judge_calls") or []
            if call.get("response", {}).get("model_resolved")
        }
    )
    document = {
        "schema_version": REFERENCE_SCORES_SCHEMA_VERSION,
        "case_id": case_id,
        "dimension": dimension_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_video": video.relative_to(case.root).as_posix(),
        "reference_video_sha256": sha256_file(video),
        "judge": identities[0],
        "judge_model_resolved": resolved,
        "summary": summarize(recorded),
        "passes": recorded,
    }
    output = case.artifacts_dir / REFERENCE_SCORES_FILENAME
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    means = {key: document["summary"][key]["mean"] for key in SCORE_KEYS}
    print(f"[freeze] {case_id}: {means} -> {output}", flush=True)
    return output


def register_in_case(case_dir: Path) -> None:
    """Declare the frozen file under reference.artifacts (keeps other keys)."""
    case_yaml = case_dir / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    artifacts = raw["reference"].setdefault("artifacts", {})
    artifacts[REFERENCE_SCORES_ARTIFACT] = f"reference/artifacts/{REFERENCE_SCORES_FILENAME}"
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "case_ids", nargs="*", help=f"default: every {DIMENSION} case in suite.yaml"
    )
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5, help="concurrent judge passes")
    parser.add_argument(
        "--register", action="store_true", help="also declare the file in case.yaml"
    )
    args = parser.parse_args(argv)

    from evaluation.lite.suite_loader import load_suite
    from evaluation.modules.vlm_consistency.vlm_judge import _resolve_api_key
    from tools.run_wmeval_lite import _load_dotenv

    _load_dotenv(ROOT)
    suite = load_suite(SUITE_ROOT)
    dimension = suite.dimensions[DIMENSION]
    try:
        _resolve_api_key(str(dimension.vlm["provider"]))
    except RuntimeError as exc:
        raise SystemExit(f"{exc}: freezing runs the hosted judge") from exc

    for case_id in args.case_ids or list(dimension.cases):
        freeze(case_id, passes=args.passes, workers=args.workers)
        if args.register:
            register_in_case(suite.cases_root / case_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
