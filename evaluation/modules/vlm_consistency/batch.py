#!/usr/bin/env python3
"""Batch scene alignment evaluation using the dual-agent pipeline.

Directory layout expected:
  tests/vlm_judge/data/{ModelName}/*.mp4   — generated videos per model
  tests/vlm_judge/ref_img/*_image.png      — GT reference images per scene

Video-to-scene matching: a video file matches a scene if the scene name
(lowercased, stripped of spaces) starts with the file stem after removing
the model-name prefix and any trailing _firstperson / _thirdperson suffix.

Examples:
    # Generate rubrics (once per rubric model)
    python -m evaluation.modules.vlm_consistency.batch generate_rubrics \
        --provider openai --model gpt-5.4

    # Evaluate all models found under tests/vlm_judge/data/
    python -m evaluation.modules.vlm_consistency.batch evaluate \
        --provider openai --model gpt-5.4 \
        --output batch_result.json

    # Evaluate without reference image
    python -m evaluation.modules.vlm_consistency.batch evaluate \
        --provider openai --model gpt-5.4 --no_ref \
        --output batch_noref.json

    # Evaluate only specific models
    python -m evaluation.modules.vlm_consistency.batch evaluate \
        --provider openai --model gpt-5.4 \
        --model_names Genie Unity \
        --output batch_result.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .frame_sampler import prepare_frames_for_vlm
from . import Rubric, judge_scene, save_rubric
from .rubric_cache import find_rubric
from .rubric_generator import generate_rubric

TESTS_DIR = Path("tests/vlm_judge")
DEFAULT_CSV_PATH = TESTS_DIR / "Prompt - Basic_env.csv"
DEFAULT_DATA_DIR = TESTS_DIR / "data"
DEFAULT_REF_DIR = TESTS_DIR / "ref_img"
DEFAULT_RUBRIC_DIR = "rubrics"

SAMPLE_DURATION = 10.0
SAMPLE_FRAMES = 18


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def find_videos_for_scene(scene_name: str, model_dir: Path) -> List[Path]:
    """Find all .mp4 files in model_dir whose stem matches scene_name.

    Matching rule: after stripping the model-dir-name prefix and any
    _firstperson/_thirdperson style suffix, the remaining stem must be a
    prefix of the normalized scene name (or equal to it).
    """
    model_prefix = _normalize(model_dir.name)
    scene_norm = _normalize(scene_name)
    matches = []
    for mp4 in sorted(model_dir.glob("*.mp4")):
        stem = _normalize(mp4.stem)
        # strip model-name prefix (e.g. "unity", "matrix")
        if stem.startswith(model_prefix):
            stem = stem[len(model_prefix):]
        # strip known view/type suffixes
        for suffix in ("firstperson", "thirdperson", "actionsrun", "long"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        # match: scene name starts with the cleaned stem
        if stem and scene_norm.startswith(stem):
            matches.append(mp4)
    return matches


def find_ref_image(scene_name: str, ref_dir: Path) -> Optional[Path]:
    """Find *_image.{png,jpg} whose base name is a prefix of scene_name."""
    scene_norm = _normalize(scene_name)
    for img in sorted(ref_dir.glob("*_image.*")):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = _normalize(img.stem.replace("_image", "").replace("image", ""))
        if stem and scene_norm.startswith(stem):
            return img
    return None


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_prompts(csv_path: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_name", row.get("Scene", "")).strip()
            prompt = row.get("prompt", row.get("Prompt", "")).strip()
            if scene and prompt:
                prompts[scene] = prompt
    return prompts


# ---------------------------------------------------------------------------
# generate_rubrics
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    provider, model = args.provider, args.model
    ref_dir = Path(args.ref_dir)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    prompts = load_prompts(csv_path)

    print(f"Generating rubrics for {len(prompts)} scene(s) via "
          f"{provider}/{model}...", file=sys.stderr)

    for scene_name, prompt_text in prompts.items():
        print(f"\n--- {scene_name} ---", file=sys.stderr)
        gt_frame = None
        img = find_ref_image(scene_name, ref_dir)
        if img:
            gt_frame = str(img)
            print(f"  ref image: {img.name}", file=sys.stderr)

        rubric = generate_rubric(
            prompt_text=prompt_text,
            scene_name=scene_name,
            provider=provider,
            model=model,
            gt_frame_path=gt_frame,
        )
        saved = save_rubric(rubric, args.rubric_dir, args.prompt_version)
        print(f"  Saved: {saved}", file=sys.stderr)

    print(f"\nDone. Rubrics in {args.rubric_dir}/", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    provider, model = args.provider, args.model
    use_ref = not args.no_ref
    alpha = args.alpha
    rubric_provider = args.rubric_provider or provider
    rubric_model = args.rubric_model or model
    data_dir = Path(args.data_dir)
    ref_dir = Path(args.ref_dir)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # Discover model directories
    if args.model_names:
        model_dirs = [data_dir / n for n in args.model_names]
    else:
        model_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if not model_dirs:
        print(f"ERROR: no model directories found in {data_dir}", file=sys.stderr)
        return 1

    prompts = load_prompts(csv_path)
    ref_label = "with ref" if use_ref else "no ref"
    print(f"Models: {[d.name for d in model_dirs]}", file=sys.stderr)
    print(f"Scenes: {list(prompts.keys())}", file=sys.stderr)
    print(f"Judge: {provider}/{model} | rubrics: {rubric_provider}/{rubric_model} "
          f"| {ref_label} | alpha={alpha}", file=sys.stderr)

    all_results = []

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"WARNING: directory not found: {model_dir}", file=sys.stderr)
            continue

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Model: {model_dir.name}", file=sys.stderr)

        for scene_name in prompts:
            # Load rubric
            rubric_path = find_rubric(
                scene_name, args.prompt_version, args.rubric_dir,
                provider=rubric_provider, model=rubric_model,
            )
            if rubric_path is None:
                print(f"  [{scene_name}] WARNING: no rubric, skipping",
                      file=sys.stderr)
                all_results.append({
                    "model": model_dir.name,
                    "scene_name": scene_name,
                    "video_file": None,
                    "status": "missing_rubric",
                })
                continue

            rubric = Rubric.model_validate_json(
                rubric_path.read_text(encoding="utf-8")
            )

            # Find reference image
            gt_frame = None
            if use_ref:
                img = find_ref_image(scene_name, ref_dir)
                gt_frame = str(img) if img else None

            # Find matching videos
            videos = find_videos_for_scene(scene_name, model_dir)
            if not videos:
                print(f"  [{scene_name}] no video found — MISSING", file=sys.stderr)
                all_results.append({
                    "model": model_dir.name,
                    "scene_name": scene_name,
                    "video_file": None,
                    "status": "missing_video",
                })
                continue

            for vfile in videos:
                print(f"\n  [{scene_name}] {vfile.name}", file=sys.stderr)
                try:
                    all_frames, n_frames = prepare_frames_for_vlm(
                        video_path=str(vfile),
                        first_n_seconds=SAMPLE_DURATION,
                        sample_fps=SAMPLE_FRAMES / SAMPLE_DURATION,
                        max_frames=SAMPLE_FRAMES,
                        start_seconds=0.0,
                        gt_frame_path=gt_frame,
                        label_prefix="frame",
                        use_timestamp_labels=True,
                    )
                    print(f"    frames: {n_frames}", file=sys.stderr)

                    result = judge_scene(
                        rubric=rubric,
                        all_frames=all_frames,
                        provider=provider,
                        model=model,
                        alpha=alpha,
                    )

                    rd = result.model_dump()
                    rd["model"] = model_dir.name
                    rd["video_file"] = vfile.name
                    rd["status"] = "ok"
                    all_results.append(rd)

                    print(
                        f"    environment={result.environment_result.score:.4f}  "
                        f"object={result.object_result.score:.4f}  "
                        f"overall={result.overall_score:.4f}",
                        file=sys.stderr,
                    )

                except Exception as e:
                    print(f"    ERROR: {e}", file=sys.stderr)
                    all_results.append({
                        "model": model_dir.name,
                        "scene_name": scene_name,
                        "video_file": vfile.name,
                        "status": "error",
                        "error": str(e),
                    })

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} entries → {args.output}", file=sys.stderr)

    # Summary table
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"SUMMARY  judge={provider}/{model}  alpha={alpha}  {ref_label}",
          file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"{'Model':<12} {'Scene':<26} {'Video':<30} {'Env':>6} {'Obj':>6} "
          f"{'Overall':>8}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    for r in all_results:
        status = r.get("status", "")
        model_name = r.get("model", "")
        scene = r.get("scene_name", "")
        vfile = r.get("video_file") or ""
        if status == "ok":
            print(
                f"{model_name:<12} {scene:<26} {vfile:<30} "
                f"{r['environment_result']['score']:>6.3f} "
                f"{r['object_result']['score']:>6.3f} "
                f"{r['overall_score']:>8.4f}",
                file=sys.stderr,
            )
        else:
            print(f"{model_name:<12} {scene:<26} {vfile or '—':<30} "
                  f"{'':>6} {'':>6} {status:>8}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch scene alignment evaluation",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # generate_rubrics
    gen_p = sub.add_parser("generate_rubrics", help="Generate and cache rubrics")
    gen_p.add_argument("--provider", default="openai",
                       choices=["openai", "gemini", "anthropic"])
    gen_p.add_argument("--model", required=True)
    gen_p.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                       help=f"CSV with scene_name and prompt columns (default: {DEFAULT_CSV_PATH})")
    gen_p.add_argument("--rubric_dir", default=DEFAULT_RUBRIC_DIR)
    gen_p.add_argument("--ref_dir", default=str(DEFAULT_REF_DIR))
    gen_p.add_argument("--prompt_version", default="v1")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate all scenes across all models")
    eval_p.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "anthropic"])
    eval_p.add_argument("--model", required=True,
                        help="VLM judge model")
    eval_p.add_argument("--rubric_provider", default=None,
                        help="Provider used to generate rubrics (default: same as --provider)")
    eval_p.add_argument("--rubric_model", default=None,
                        help="Model used to generate rubrics (default: same as --model)")
    eval_p.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                        help=f"CSV with scene_name and prompt columns (default: {DEFAULT_CSV_PATH})")
    eval_p.add_argument("--rubric_dir", default=DEFAULT_RUBRIC_DIR)
    eval_p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Root directory containing model subdirs (default: {DEFAULT_DATA_DIR})")
    eval_p.add_argument("--ref_dir", default=str(DEFAULT_REF_DIR),
                        help=f"Directory with GT reference images (default: {DEFAULT_REF_DIR})")
    eval_p.add_argument("--model_names", nargs="+", default=None,
                        help="Limit to specific model subdirs, e.g. --model_names Genie Unity")
    eval_p.add_argument("--prompt_version", default="v1")
    eval_p.add_argument("--no_ref", action="store_true",
                        help="Skip GT reference image")
    eval_p.add_argument("--alpha", type=float, default=0.6,
                        help="Environment weight (default: 0.6)")
    eval_p.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.command == "generate_rubrics":
        return cmd_generate(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
