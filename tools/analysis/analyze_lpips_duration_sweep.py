#!/usr/bin/env python3
"""
Cumulative-window LPIPS analysis from existing WMEval LPIPS JSON reports.

For each report, uses the paired timestamps_sec and lpips_per_pair already
computed under fixed eval_fps / max_pairs. For k = 1..N, aggregates the
first k pairs — equivalent to asking: "if we had only evaluated through
timestamp k, what mean/median LPIPS would we report?"

This holds content and sampling policy fixed; only the evaluated time span
(growing from the start of the aligned clip) changes.

Note: Your report may not cover all of shared_duration_sec if max_pairs
caps the timeline (e.g. 64 frames at 4 Hz ≈ first 16s only). Re-run
LPIPS with a larger max_pairs or lower eval_fps to study longer windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_lpips_block(data: Dict[str, Any]) -> Dict[str, Any]:
    if "results" in data and isinstance(data["results"], dict):
        lp = data["results"].get("lpips")
        if isinstance(lp, dict):
            return lp
    if data.get("module") == "lpips" and "lpips_per_pair" in data:
        return data
    raise ValueError("JSON does not look like a WMEval LPIPS report (missing results.lpips).")


def cumulative_rows(
    report_path: Path,
    lp: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ts: Sequence[float] = lp.get("timestamps_sec") or []
    vals: Sequence[float] = lp.get("lpips_per_pair") or []
    if len(ts) != len(vals):
        raise ValueError(f"{report_path}: timestamps_sec and lpips_per_pair length mismatch")

    shared = lp.get("shared_duration_sec")
    eval_fps = lp.get("eval_fps")
    rows: List[Dict[str, Any]] = []
    for k in range(1, len(vals) + 1):
        prefix = list(vals[:k])
        rows.append(
            {
                "report": str(report_path),
                "scene_label": report_path.stem,
                "num_pairs": k,
                "window_end_sec": ts[k - 1],
                "shared_duration_sec": shared,
                "eval_fps": eval_fps,
                "lpips_mean": sum(prefix) / k,
                "lpips_median": float(statistics.median(prefix)),
                "lpips_std": float(statistics.pstdev(prefix)) if k > 1 else 0.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cumulative LPIPS vs. evaluated window from WMEval LPIPS JSON reports.",
    )
    ap.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="One or more *lpips*.json report paths",
    )
    ap.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Write combined CSV (default: print CSV to stdout)",
    )
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for path in args.reports:
        if not path.exists():
            print(f"Error: not found: {path}", file=sys.stderr)
            return 1
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lp = load_lpips_block(data)
        all_rows.extend(cumulative_rows(path, lp))

    fieldnames = [
        "report",
        "scene_label",
        "num_pairs",
        "window_end_sec",
        "shared_duration_sec",
        "eval_fps",
        "lpips_mean",
        "lpips_median",
        "lpips_std",
    ]
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {args.output_csv} ({len(all_rows)} rows)", file=sys.stderr)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
