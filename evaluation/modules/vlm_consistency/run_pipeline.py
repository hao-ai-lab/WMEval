"""Pipeline orchestration with CLI subcommands.

Subcommands:
  generate_rubrics  -- dual-agent rubric generation with caching
  evaluate          -- unified VLM judging on all timestamped frames
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .frame_sampler import prepare_frames_for_vlm

from .rubric_cache import save_rubric
from .rubric_generator import generate_rubric
from .schemas import Rubric
from .vlm_judge import judge_scene


# ---------------------------------------------------------------------------
# generate_rubrics subcommand
# ---------------------------------------------------------------------------

def cmd_generate_rubrics(args: argparse.Namespace) -> int:
    """Generate, validate, and cache rubrics for scenes defined in a CSV."""
    csv_path = args.csv
    rubric_dir = args.rubric_dir
    provider = args.provider
    model = args.model
    gt_dir = args.gt_dir
    prompt_version = args.prompt_version

    if not Path(csv_path).is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    scenes = _load_scenes_from_csv(csv_path)
    if not scenes:
        print("Error: no scenes found in CSV", file=sys.stderr)
        return 1

    print(f"Generating rubrics for {len(scenes)} scene(s) "
          f"via {provider}/{model}...", file=sys.stderr)

    for scene_name, prompt_text in scenes:
        print(f"\n--- {scene_name} ---", file=sys.stderr)

        gt_frame_path = None
        if gt_dir:
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = Path(gt_dir) / f"{scene_name}_image{ext}"
                if candidate.is_file():
                    gt_frame_path = str(candidate)
                    break

        rubric = generate_rubric(
            prompt_text=prompt_text,
            scene_name=scene_name,
            provider=provider,
            model=model,
            gt_frame_path=gt_frame_path,
        )

        saved = save_rubric(rubric, rubric_dir, prompt_version)
        print(f"  Saved: {saved}", file=sys.stderr)

    print(f"\nDone. Rubrics cached in {rubric_dir}/", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# evaluate subcommand
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Unified evaluation: all timestamped frames, single VLM call."""
    rubric_path = args.rubric
    if not Path(rubric_path).is_file():
        print(f"Error: rubric not found: {rubric_path}", file=sys.stderr)
        return 1
    rubric = Rubric.model_validate_json(
        Path(rubric_path).read_text(encoding="utf-8")
    )

    video_path = args.video
    if not Path(video_path).is_file():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    provider = args.provider
    model = args.model
    alpha = args.alpha

    env_n = len(rubric.environment_items)
    obj_n = len(rubric.object_items)
    print(f"Rubric: {env_n} environment + {obj_n} object items", file=sys.stderr)

    # Sample frames uniformly across the full duration with timestamp labels
    sample_fps = args.num_frames / args.duration
    print(f"Sampling {args.num_frames} frames over {args.duration}s...", file=sys.stderr)
    all_frames, n_frames = prepare_frames_for_vlm(
        video_path=video_path,
        first_n_seconds=args.duration,
        sample_fps=sample_fps,
        max_frames=args.num_frames,
        start_seconds=0.0,
        gt_frame_path=args.gt_frame,
        label_prefix="frame",
        use_timestamp_labels=True,
    )
    print(f"Candidate frames: {n_frames}", file=sys.stderr)

    # Optional GT reference video frames
    gt_frames = None
    if args.gt_video and Path(args.gt_video).is_file():
        print(f"Sampling GT reference video frames from {args.gt_video}...", file=sys.stderr)
        gt_frames, gt_n = prepare_frames_for_vlm(
            video_path=args.gt_video,
            first_n_seconds=args.duration,
            sample_fps=sample_fps,
            max_frames=args.num_frames,
            start_seconds=0.0,
            label_prefix="gt_frame",
            use_timestamp_labels=True,
        )
        gt_frames = gt_frames if gt_frames else None
        print(f"GT video: {gt_n} frames", file=sys.stderr)

    print(f"Running unified VLM judge via {provider} (alpha={alpha})...",
          file=sys.stderr)
    result = judge_scene(
        rubric=rubric,
        all_frames=all_frames,
        provider=provider,
        model=model,
        alpha=alpha,
        gt_frames=gt_frames,
    )

    result_dict = result.model_dump()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {args.output}", file=sys.stderr)

    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scenes_from_csv(csv_path: str):
    """Load (scene_name, prompt) pairs from the test CSV."""
    scenes = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("scene_name", row.get("Scene", "")).strip()
            prompt = row.get("prompt", row.get("Prompt", "")).strip()
            if name and prompt:
                scenes.append((name, prompt))
    return scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vlm_consistency",
        description="Scene alignment: dual-agent rubric, unified judge",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # --- generate_rubrics ---
    gen_p = sub.add_parser(
        "generate_rubrics",
        help="Dual-agent rubric generation with caching",
    )
    gen_p.add_argument("--csv", required=True, help="CSV with scene_name and prompt columns")
    gen_p.add_argument("--rubric_dir", default="rubrics", help="Output rubric cache directory")
    gen_p.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai")
    gen_p.add_argument("--model", required=True, help="Model name")
    gen_p.add_argument("--gt_dir", default=None, help="Directory with GT initial frame images")
    gen_p.add_argument("--prompt_version", default="v1")

    # --- evaluate ---
    eval_p = sub.add_parser(
        "evaluate",
        help="Unified VLM evaluation on all timestamped frames",
    )
    eval_p.add_argument("--video", required=True, help="Path to generated video")
    eval_p.add_argument("--rubric", required=True, help="Path to cached rubric JSON")
    eval_p.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai")
    eval_p.add_argument("--model", default=None, help="Override default model")
    eval_p.add_argument("--gt_frame", default=None, help="GT initial frame image path")
    eval_p.add_argument("--gt_video", default=None, help="GT reference video path")
    eval_p.add_argument("--output", default=None, help="Output JSON path")
    eval_p.add_argument("--alpha", type=float, default=0.6,
                        help="Weight for environment component (default: 0.6)")
    eval_p.add_argument("--duration", type=float, default=10.0,
                        help="Video duration to sample in seconds (default: 10.0)")
    eval_p.add_argument("--num_frames", type=int, default=18,
                        help="Total frames to sample uniformly (default: 18)")

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate_rubrics":
        return cmd_generate_rubrics(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
