#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite_refs import (
    DEFAULT_LITE_MANIFEST,
    ensure_reference_video,
    parse_lite_manifest,
    repo_root_from_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Verify bundled Unity reference videos for WMEval Lite. "
            "No Unity installation required."
        )
    )
    ap.add_argument(
        "--manifest_json",
        default=str(REPO_ROOT / DEFAULT_LITE_MANIFEST),
        help="WMEval Lite manifest JSON",
    )
    ap.add_argument(
        "--dimensions",
        default="",
        help="Comma-separated dimension ids (default: all in manifest)",
    )
    args = ap.parse_args()

    manifest = parse_lite_manifest(args.manifest_json)
    manifest_path = Path(manifest["manifest_path"])
    repo_root = repo_root_from_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    requested = {
        item.strip()
        for item in args.dimensions.split(",")
        if item.strip()
    }

    results = []
    missing = []
    for test_case in manifest["test_cases"]:
        if requested and test_case.dimension_id not in requested:
            continue
        try:
            path = ensure_reference_video(
                test_case,
                repo_root=repo_root,
                manifest_dir=manifest_dir,
            )
            results.append(
                {
                    "dimension_id": test_case.dimension_id,
                    "test_case_id": test_case.test_case_id,
                    "display_name": test_case.display_name,
                    "reference_video": str(path),
                    "status": "ok",
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{test_case.dimension_id}/{test_case.test_case_id}: {exc}")

    if missing:
        raise SystemExit(
            "Missing or invalid WMEval Lite reference assets:\n- " + "\n- ".join(missing)
        )

    print(json.dumps({"manifest": str(manifest_path), "assets": results}, indent=2))


if __name__ == "__main__":
    main()
