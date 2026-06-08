#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite_refs import DEFAULT_LITE_MANIFEST, run_lite_suite


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run WMEval Lite over bundled Unity references and candidate videos. "
            "Unity-free: reference videos ship with the dataset."
        )
    )
    ap.add_argument(
        "--manifest_json",
        default=str(REPO_ROOT / DEFAULT_LITE_MANIFEST),
        help="WMEval Lite manifest JSON",
    )
    ap.add_argument(
        "--candidate_jsonl",
        required=False,
        default=None,
        help="JSONL candidate rows with dimension/test_case_id and world_model_video",
    )
    ap.add_argument(
        "--dimensions",
        default="",
        help="Comma-separated dimension ids (causal_lpips,controllability). Default: all.",
    )
    ap.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "reports" / "wmeval_lite"),
        help="Directory for per-case JSONL and aggregate summaries",
    )
    ap.add_argument(
        "--modules",
        default="",
        help="Optional override: comma-separated modules (defaults come from each dimension)",
    )
    ap.add_argument("--eval_fps", type=float, default=4.0)
    ap.add_argument("--max_pairs", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve manifest/reference/candidate paths without running metrics",
    )

    args = ap.parse_args()
    dimensions = tuple(item.strip() for item in args.dimensions.split(",") if item.strip()) or None
    modules = tuple(item.strip() for item in args.modules.split(",") if item.strip()) or None
    aggregate = run_lite_suite(
        manifest_json=args.manifest_json,
        candidate_jsonl=args.candidate_jsonl,
        output_dir=args.output_dir,
        dimensions=dimensions,
        modules=modules,
        eval_fps=args.eval_fps,
        max_pairs=args.max_pairs,
        device=args.device,
        dry_run=args.dry_run,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
