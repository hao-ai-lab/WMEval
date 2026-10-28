#!/usr/bin/env python3
"""
CLIP4Clip-only evaluation: single video+text or JSONL manifest.
Sampling mode is controlled by --sample_mode (timestamp | frame_step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.modules.clip4clip import run_clip4clip
from evaluation.video_io import (
    SAMPLE_MODE_FRAME_STEP,
    SAMPLE_MODE_TIMESTAMP,
    build_evaluation_context,
)


def load_manifest_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON line {line_no} in {path}: {e}") from e


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    video = str(
        row.get("world_model_video")
        or row.get("candidate_video")
        or row.get("reference_video")
        or row.get("video")
        or ""
    ).strip()
    prompt = str(row.get("prompt") or row.get("text_prompt") or "").strip()
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    if not video:
        raise ValueError(f"Row missing video path: {row}")
    return {**row, "video": video, "prompt": prompt, "sample_id": sample_id or "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CLIP4Clip evaluation: single video+text or JSONL manifest. Use --sample_mode to choose sampling.",
    )
    ap.add_argument("--video", default=None, help="Single video path (mutually exclusive with --manifest_jsonl)")
    ap.add_argument("--text_prompt", default="", help="Text prompt for single-video mode")
    ap.add_argument("--manifest_jsonl", default=None, help="JSONL manifest (video/prompt per line)")
    ap.add_argument("--output_json", default=None, help="Output JSON for single-video mode")
    ap.add_argument("--output_jsonl", default=None, help="Output JSONL for manifest mode")
    ap.add_argument(
        "--sample_mode",
        choices=[SAMPLE_MODE_TIMESTAMP, SAMPLE_MODE_FRAME_STEP],
        default=SAMPLE_MODE_TIMESTAMP,
        help="timestamp: time-based sampling (same as LPIPS). frame_step: every N frames (cv_metrics style).",
    )
    ap.add_argument("--eval_fps", type=float, default=4.0, help="Sampling rate (timestamp: fps; frame_step: target_fps)")
    ap.add_argument("--max_pairs", type=int, default=32, help="Max frames per video")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")

    args = ap.parse_args()

    if args.video and args.manifest_jsonl:
        print("Error: use either --video or --manifest_jsonl, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.video and not args.manifest_jsonl:
        print("Error: provide --video or --manifest_jsonl.", file=sys.stderr)
        sys.exit(1)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: video not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        context = build_evaluation_context(
            reference_video=str(video_path),
            candidate_video=str(video_path),
            eval_fps=args.eval_fps,
            max_pairs=args.max_pairs,
            text_prompt=args.text_prompt or "A video",
            sample_mode=args.sample_mode,
        )
        result = run_clip4clip(context=context, device=args.device)
        out_path = args.output_json or "reports/clip4clip_report.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        if "error" in result:
            sys.exit(1)
        return

    # Manifest mode
    if not args.output_jsonl:
        print("Error: --manifest_jsonl requires --output_jsonl.", file=sys.stderr)
        sys.exit(1)
    rows = [normalize_row(r) for r in load_manifest_rows(args.manifest_jsonl)]
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row.get('sample_id', '')} {row['video']}", flush=True)
        context = build_evaluation_context(
            reference_video=row["video"],
            candidate_video=row["video"],
            eval_fps=args.eval_fps,
            max_pairs=args.max_pairs,
            text_prompt=row["prompt"] or "A video",
            sample_mode=args.sample_mode,
        )
        result = run_clip4clip(context=context, device=args.device)
        results.append({
            "sample_id": row.get("sample_id", ""),
            "video": row["video"],
            "prompt": row["prompt"],
            "sample_mode": args.sample_mode,
            **{k: v for k, v in result.items() if k not in ("reference_metadata", "candidate_metadata", "timestamps_sec", "candidate_frame_indices")},
        })
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output_jsonl} ({len(results)} rows)")
    print(json.dumps({"manifest_jsonl": args.manifest_jsonl, "output_jsonl": args.output_jsonl, "sample_mode": args.sample_mode, "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
