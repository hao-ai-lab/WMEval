#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_manifest_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            yield row


def normalize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("id") or row.get("vid") or "").strip()
    reference_video = str(row.get("reference_video") or row.get("unity_video") or row.get("gt_video") or "").strip()
    world_model_video = str(
        row.get("world_model_video") or row.get("candidate_video") or row.get("genie3_video") or row.get("pred_video") or ""
    ).strip()
    world_model_name = str(row.get("world_model_name") or row.get("model_name") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    run_id = str(row.get("run_id") or "").strip()

    missing: List[str] = []
    if not sample_id:
        missing.append("sample_id")
    if not reference_video:
        missing.append("reference_video")
    if not world_model_video:
        missing.append("world_model_video")
    if missing:
        raise ValueError(f"Manifest row missing required fields {missing}: {row}")

    return {
        **row,
        "sample_id": sample_id,
        "reference_video": reference_video,
        "world_model_video": world_model_video,
        "world_model_name": world_model_name,
        "prompt": prompt,
        "run_id": run_id,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "world_model_name",
        "run_id",
        "reference_video",
        "world_model_video",
        "num_pairs",
        "shared_duration_sec",
        "lpips_mean",
        "lpips_median",
        "lpips_std",
        "lpips_min",
        "lpips_max",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute LPIPS (VGG backbone) between reference videos and world-model "
            "generated videos from a JSONL manifest."
        )
    )
    ap.add_argument("--manifest_jsonl", required=True, help="Input JSONL manifest")
    ap.add_argument("--output_jsonl", required=True, help="Per-pair LPIPS results")
    ap.add_argument("--summary_csv", default=None, help="Optional compact CSV summary")
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Timestamp sampling rate")
    ap.add_argument("--max_pairs", type=int, default=32, help="Maximum frame pairs per video pair")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    args = ap.parse_args()

    from metrics.lpips_vgg import compute_lpips_video_pair

    results: List[Dict[str, Any]] = []
    rows = [normalize_manifest_row(row) for row in load_manifest_rows(args.manifest_jsonl)]

    for idx, row in enumerate(rows, start=1):
        print(
            f"[{idx}/{len(rows)}] sample_id={row['sample_id']} "
            f"world_model={row['world_model_name'] or '-'} "
            f"run_id={row['run_id'] or '-'}",
            flush=True,
        )

        result = compute_lpips_video_pair(
            reference_video=row["reference_video"],
            candidate_video=row["world_model_video"],
            eval_fps=args.eval_fps,
            max_pairs=args.max_pairs,
            device=args.device,
        )
        results.append(
            {
                "sample_id": row["sample_id"],
                "world_model_name": row["world_model_name"],
                "run_id": row["run_id"],
                "prompt": row["prompt"],
                "reference_video": row["reference_video"],
                "world_model_video": row["world_model_video"],
                **result,
            }
        )

    output_jsonl = Path(args.output_jsonl)
    write_jsonl(output_jsonl, results)

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), results)

    print(
        json.dumps(
            {
                "manifest_jsonl": args.manifest_jsonl,
                "output_jsonl": str(output_jsonl),
                "summary_csv": args.summary_csv,
                "num_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
