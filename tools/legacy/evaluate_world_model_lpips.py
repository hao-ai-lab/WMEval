#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.runner import run_world_model_evaluation


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
        "lpips_sampling",
        "num_pairs",
        "shared_duration_sec",
        "lpips_mean",
        "lpips_median",
        "lpips_std",
        "lpips_min",
        "lpips_max",
        "matched_reference_event_fraction",
        "num_matched_events",
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
            "Compute LPIPS over a JSONL manifest using the shared world-model "
            "metrics orchestrator."
        )
    )
    ap.add_argument("--manifest_jsonl", required=True, help="Input JSONL manifest")
    ap.add_argument("--output_jsonl", required=True, help="Per-pair LPIPS results")
    ap.add_argument("--summary_csv", default=None, help="Optional compact CSV summary")
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Timestamp sampling rate")
    ap.add_argument("--max_pairs", type=int, default=32, help="Maximum frame pairs per video pair")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument(
        "--lpips_sampling",
        choices=["temporal", "dense", "causal_vlm_collision"],
        default="temporal",
        help="LPIPS anchor mode",
    )
    ap.add_argument(
        "--lpips_temporal_fractions",
        default="0,0.5,1",
        help='Comma-separated fractions in [0,1] for temporal mode',
    )
    ap.add_argument("--lpips_event_frames_before", type=int, default=2)
    ap.add_argument("--lpips_event_frames_after", type=int, default=2)
    ap.add_argument(
        "--lpips_event_offsets",
        default=None,
        help='Comma-separated offsets around contact (overrides before/after)',
    )
    ap.add_argument("--lpips_vlm_model", default="gpt-4o")
    ap.add_argument("--lpips_vlm_max_grid_frames", type=int, default=48)
    ap.add_argument("--lpips_vlm_prompt_version", default="causal_collision_v2")
    ap.add_argument(
        "--lpips_vlm_max_reference_events",
        type=int,
        default=None,
        help="Unity VLM: ask for at most N events (e.g. 1)",
    )
    ap.add_argument(
        "--lpips_vlm_reference_trim_sec",
        type=float,
        default=None,
        help="Reference VLM: sample keyframes only from first N seconds (e.g. 2)",
    )
    ap.add_argument(
        "--lpips_vlm_judge_hint",
        default="",
        help="Optional reference-judge hint (soft)",
    )
    ap.add_argument(
        "--lpips_causal_event_source",
        choices=["both", "reference", "unity"],
        default="both",
        help="causal_vlm_collision: unity/reference = judge reference only; both = two VLMs",
    )
    ap.add_argument(
        "--lpips_causal_lpips_pairing",
        choices=["auto", "wall_clock", "per_video_contact"],
        default="auto",
        help="causal: auto | wall_clock | per_video_contact (see evaluate_world_model_metrics)",
    )
    ap.add_argument(
        "--save_inspection_artifacts",
        action="store_true",
        help="Write QA frames under --inspection_root / run / sample",
    )
    ap.add_argument("--inspection_root", default="reports/world_model_lpips/qa")
    ap.add_argument(
        "--inspection_run_id",
        default="",
        help="If empty, each manifest row's run_id is used (fallback: batch)",
    )

    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    rows = [normalize_manifest_row(row) for row in load_manifest_rows(args.manifest_jsonl)]

    temporal_fractions = [
        float(x.strip())
        for x in args.lpips_temporal_fractions.split(",")
        if x.strip()
    ]
    event_offsets = None
    if args.lpips_event_offsets:
        event_offsets = [
            int(x.strip())
            for x in args.lpips_event_offsets.split(",")
            if x.strip()
        ]

    for idx, row in enumerate(rows, start=1):
        print(
            f"[{idx}/{len(rows)}] sample_id={row['sample_id']} "
            f"world_model={row['world_model_name'] or '-'} "
            f"run_id={row['run_id'] or '-'}",
            flush=True,
        )

        inspection_run_id = (
            args.inspection_run_id.strip()
            or (row["run_id"] or "").strip()
            or "batch"
        )
        report = run_world_model_evaluation(
            video_path=row["world_model_video"],
            ref_video_path=row["reference_video"],
            text_prompt=row["prompt"],
            modules=["lpips"],
            eval_fps=args.eval_fps,
            max_pairs=args.max_pairs,
            lpips_sampling=args.lpips_sampling,
            lpips_temporal_fractions=temporal_fractions,
            lpips_event_frames_before=args.lpips_event_frames_before,
            lpips_event_frames_after=args.lpips_event_frames_after,
            lpips_event_offsets=event_offsets,
            lpips_vlm_model=args.lpips_vlm_model,
            lpips_vlm_max_grid_frames=args.lpips_vlm_max_grid_frames,
            lpips_vlm_prompt_version=args.lpips_vlm_prompt_version,
            lpips_vlm_max_reference_events=args.lpips_vlm_max_reference_events,
            lpips_vlm_reference_trim_sec=args.lpips_vlm_reference_trim_sec,
            lpips_vlm_judge_hint=args.lpips_vlm_judge_hint,
            lpips_causal_event_source=args.lpips_causal_event_source,
            lpips_causal_lpips_pairing=(
                None
                if args.lpips_causal_lpips_pairing == "auto"
                else args.lpips_causal_lpips_pairing
            ),
            save_inspection_artifacts=args.save_inspection_artifacts,
            inspection_root=args.inspection_root,
            inspection_run_id=inspection_run_id,
            inspection_sample_id=row["sample_id"],
            device=args.device,
        )
        lpips_result = dict(report["results"].get("lpips", {}))
        results.append(
            {
                "sample_id": row["sample_id"],
                "world_model_name": row["world_model_name"],
                "run_id": row["run_id"],
                "prompt": row["prompt"],
                "reference_video": row["reference_video"],
                "world_model_video": row["world_model_video"],
                **lpips_result,
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
