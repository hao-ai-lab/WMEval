#!/usr/bin/env python3
"""Author or refresh a WMEval Lite case bundle.

Given a case directory whose ``reference/`` files are already in place, this
tool rewrites ``reference/checksums.sha256`` covering every file under
``reference/``, recomputes ``evaluator_lock.evaluator_fingerprint``
(always — it is derived from the full case contract plus reference checksums
and must never go stale),
fills ``evaluator_lock.code_revision`` when empty (pass ``--relock`` to
refresh it to the current git HEAD), then validates the bundle.

Usage::

    python tools/build_lite_case.py lite/cases/BallRoller/ctrl
    python tools/build_lite_case.py --relock lite/cases/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _case_id(case_dir: Path) -> str:
    return f"{case_dir.parent.name}/{case_dir.name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+", help="Case directories to (re)build")
    parser.add_argument(
        "--relock",
        action="store_true",
        help="Refresh code_revision to the current git HEAD even when set "
        "(the fingerprint is always recomputed regardless)",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip load_case/validate_case_bundle after writing",
    )
    args = parser.parse_args(argv)

    from evaluation.lite.authoring import author_case

    for case_dir_arg in args.case_dirs:
        case_dir = Path(case_dir_arg).resolve()
        checksums_path = author_case(case_dir, relock=args.relock)
        print(f"wrote {checksums_path}")
        if not args.skip_validate:
            from evaluation.lite.case_loader import load_case
            from evaluation.lite.validation import validate_case_bundle

            case = load_case(case_dir, allowed_cases={_case_id(case_dir)})
            validate_case_bundle(case)
            print(f"validated {case.case_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
