#!/usr/bin/env python3
"""Create or render the five-run Table 4/7 paper comparison artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.table_comparison import compare_results, load_baseline, result_template, write_comparison


DEFAULT_BASELINE = REPO_ROOT / "experiments/tables_4_7/paper_baseline.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("template", "compare"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    baseline = load_baseline(args.baseline.resolve())
    if args.command == "template":
        output = (args.output or (REPO_ROOT / "experiments/tables_4_7/rerun_results.template.json")).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result_template(baseline), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        return 0
    if args.results is None:
        parser.error("compare requires --results")
    rerun = json.loads(args.results.read_text(encoding="utf-8"))
    comparison = compare_results(baseline, rerun, allow_incomplete=args.allow_incomplete)
    output_dir = (args.output or (REPO_ROOT / "runs/tables_4_7/comparison")).resolve()
    write_comparison(comparison, output_dir)
    print(json.dumps({"complete": comparison["complete"], "pending_count": comparison["pending_count"], "output": str(output_dir)}, indent=2))
    return 0 if comparison["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
