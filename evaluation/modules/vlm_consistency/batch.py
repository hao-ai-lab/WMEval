
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .frame_sampler import prepare_frames_for_vlm
from .rubric_cache import find_rubric, save_rubric
from .rubric_generator import generate_rubric
from .schemas import Rubric
from .vlm_judge import judge_scene

SAMPLE_DURATION = 10.0
SAMPLE_FRAMES = 18


def _read_csv(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return [
            {k.strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _split_model_and_run(stem: str, scene_name: str) -> Tuple[str, str]:
    idx = stem.find(scene_name)
    if idx <= 0:
        return "", ""
    model = stem[:idx].rstrip("_")
    rest = stem[idx + len(scene_name):].lstrip("_")
    return model, rest


def _write_manifest(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_manifest(manifest_path: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    base = manifest_path.parent.resolve()
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return base, rows


def _resolve(base: Path, rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (base / p)


def cmd_build_manifest(args: argparse.Namespace) -> int:
    view_dir = Path(args.view_dir).resolve()
    csv_path = Path(args.csv) if args.csv else view_dir / "scenes.csv"
    data_dir = Path(args.data_dir) if args.data_dir else view_dir / "data"
    out_path = Path(args.out) if args.out else view_dir / "manifest.jsonl"

    if not csv_path.is_file():
        print(f"ERROR: scenes.csv not found: {csv_path}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    base = out_path.parent.resolve()

    rows: List[Dict[str, Any]] = []
    skipped_scenes: List[str] = []
    for r in _read_csv(csv_path):
        scene = r.get("scene_name") or r.get("Scene") or ""
        prompt = r.get("prompt") or r.get("Prompt") or ""
        ref = r.get("ref_path") or ""
        if not scene or not prompt:
            continue

        scene_video_dir = data_dir / scene
        if not scene_video_dir.is_dir():
            skipped_scenes.append(scene)
            continue

        videos = sorted(scene_video_dir.glob("*.mp4"))
        if not videos:
            skipped_scenes.append(scene)
            continue

        ref_rel = ""
        if ref:
            ref_abs = (view_dir / ref).resolve() if not Path(ref).is_absolute() else Path(ref)
            if ref_abs.is_file():
                try:
                    ref_rel = str(ref_abs.relative_to(base))
                except ValueError:
                    ref_rel = str(ref_abs)

        for vid in videos:
            try:
                vid_rel = str(vid.resolve().relative_to(base))
            except ValueError:
                vid_rel = str(vid.resolve())
            model, run = _split_model_and_run(vid.stem, scene)
            rows.append({
                "scene_name": scene,
                "prompt": prompt,
                "ref_path": ref_rel or None,
                "video_path": vid_rel,
                "model_name": model,
                "run_id": run,
            })

    _write_manifest(rows, out_path)
    print(f"Wrote {len(rows)} entries -> {out_path}", file=sys.stderr)
    if skipped_scenes:
        print(f"Skipped scenes (no videos): {skipped_scenes}", file=sys.stderr)
    return 0


def cmd_generate_rubrics(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    base, rows = _load_manifest(manifest_path)

    rubric_dir = Path(args.rubric_dir).resolve()
    provider, model = args.provider, args.model
    prompt_version = args.prompt_version

    scenes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["scene_name"] not in scenes:
            scenes[r["scene_name"]] = r

    print(f"Manifest scenes: {len(scenes)} | rubric_dir: {rubric_dir} "
          f"| {provider}/{model}", file=sys.stderr)

    for scene, r in scenes.items():
        cached = find_rubric(scene, prompt_version, str(rubric_dir),
                             provider=provider, model=model)
        if cached and not args.overwrite:
            print(f"  [{scene}] cached -> {cached}", file=sys.stderr)
            continue

        print(f"\n=== {scene} ===", file=sys.stderr)

        rubric = generate_rubric(
            prompt_text=r["prompt"],
            scene_name=scene,
            provider=provider,
            model=model,
        )
        saved = save_rubric(rubric, str(rubric_dir), prompt_version)
        print(f"  saved -> {saved}", file=sys.stderr)

    return 0


def _load_done_keys(out_path: Path) -> set:
    done = set()
    if not out_path.is_file():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("scene_name", ""), row.get("video_path", ""))
            if key[0] and key[1]:
                done.add(key)
    return done


def cmd_evaluate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    base, rows = _load_manifest(manifest_path)

    rubric_dir = Path(args.rubric_dir).resolve()
    provider = args.provider
    judge_model = args.model
    rubric_provider = args.rubric_provider or provider
    rubric_model = args.rubric_model or judge_model
    alpha = args.alpha
    use_ref = not args.no_ref

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_path) if args.resume else set()
    if done:
        print(f"Resume: skipping {len(done)} already-evaluated videos", file=sys.stderr)

    rubric_cache: Dict[str, Optional[Path]] = {}

    print(f"Manifest rows: {len(rows)} | judge: {provider}/{judge_model} "
          f"| rubrics: {rubric_provider}/{rubric_model} | alpha={alpha}",
          file=sys.stderr)

    open_mode = "a" if args.resume and out_path.is_file() else "w"
    f_out = open(out_path, open_mode, encoding="utf-8")

    n_ok = n_skip = n_err = 0
    try:
        for r in rows:
            scene = r["scene_name"]
            vid_rel = r["video_path"]
            key = (scene, vid_rel)
            if key in done:
                n_skip += 1
                continue

            video_path = _resolve(base, vid_rel)
            if not video_path or not video_path.is_file():
                _emit(f_out, r, status="missing_video")
                n_err += 1
                continue

            if scene not in rubric_cache:
                rubric_cache[scene] = find_rubric(
                    scene, args.prompt_version, str(rubric_dir),
                    provider=rubric_provider, model=rubric_model,
                )
            rubric_path = rubric_cache[scene]
            if rubric_path is None:
                print(f"  [{scene}] no rubric cached, skip", file=sys.stderr)
                _emit(f_out, r, status="missing_rubric")
                n_err += 1
                continue

            rubric = Rubric.model_validate_json(
                rubric_path.read_text(encoding="utf-8")
            )

            gt = None
            if use_ref:
                ref = _resolve(base, r.get("ref_path"))
                gt = str(ref) if ref and ref.is_file() else None

            print(f"\n[{scene}] {video_path.name}", file=sys.stderr)
            try:
                all_frames, n_frames = prepare_frames_for_vlm(
                    video_path=str(video_path),
                    first_n_seconds=SAMPLE_DURATION,
                    sample_fps=SAMPLE_FRAMES / SAMPLE_DURATION,
                    max_frames=SAMPLE_FRAMES,
                    start_seconds=0.0,
                    gt_frame_path=gt,
                    label_prefix="frame",
                    use_timestamp_labels=True,
                )
                print(f"  frames: {n_frames}", file=sys.stderr)

                result = judge_scene(
                    rubric=rubric,
                    all_frames=all_frames,
                    provider=provider,
                    model=judge_model,
                    alpha=alpha,
                )
                rd = result.model_dump()
                rd["scene_name"] = scene
                rd["video_path"] = vid_rel
                rd["model_name"] = r.get("model_name", "")
                rd["run_id"] = r.get("run_id", "")
                rd["status"] = "ok"
                f_out.write(json.dumps(rd, ensure_ascii=False) + "\n")
                f_out.flush()
                n_ok += 1
                env_s = result.environment_result.score
                obj_s = result.object_result.score
                print(
                    f"  env={f'{env_s:.4f}' if env_s is not None else 'N/A'} "
                    f"obj={f'{obj_s:.4f}' if obj_s is not None else 'N/A'} "
                    f"overall={result.overall_score:.4f}",
                    file=sys.stderr,
                )
            except Exception as e:
                _emit(f_out, r, status="error", error=str(e))
                n_err += 1
                print(f"  ERROR: {e}", file=sys.stderr)
    finally:
        f_out.close()

    print(f"\nDone. ok={n_ok} skipped={n_skip} errors/missing={n_err} "
          f"-> {out_path}", file=sys.stderr)
    _print_summary(out_path)
    return 0


def _emit(f_out, manifest_row: Dict[str, Any], **extra: Any) -> None:
    rec = {
        "scene_name": manifest_row.get("scene_name", ""),
        "video_path": manifest_row.get("video_path", ""),
        "model_name": manifest_row.get("model_name", ""),
        "run_id": manifest_row.get("run_id", ""),
        **extra,
    }
    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f_out.flush()


def _print_summary(out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"\n{'='*92}", file=sys.stderr)
    print(f"{'Scene':<24} {'Model':<14} {'Run':<8} {'Env':>6} {'Obj':>6} "
          f"{'Overall':>8}  Status", file=sys.stderr)
    print("-" * 92, file=sys.stderr)
    for r in rows:
        if r.get("status") == "ok":
            env_s = r['environment_result']['score']
            obj_s = r['object_result']['score']
            print(
                f"{r['scene_name']:<24} {r.get('model_name',''):<14} "
                f"{r.get('run_id',''):<8} "
                f"{f'{env_s:.3f}' if env_s is not None else 'N/A':>6} "
                f"{f'{obj_s:.3f}' if obj_s is not None else 'N/A':>6} "
                f"{r['overall_score']:>8.4f}  ok",
                file=sys.stderr,
            )
        else:
            print(
                f"{r.get('scene_name',''):<24} {r.get('model_name',''):<14} "
                f"{r.get('run_id',''):<8} {'-':>6} {'-':>6} {'-':>8}  "
                f"{r.get('status','?')}",
                file=sys.stderr,
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manifest-driven batch scene alignment evaluation",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build_manifest", help="Scan a view dir -> manifest.jsonl")
    bp.add_argument("--view_dir", required=True,
                    help="Directory containing scenes.csv, refs/, data/")
    bp.add_argument("--csv", default=None, help="Override scenes.csv path")
    bp.add_argument("--data_dir", default=None, help="Override data/ root")
    bp.add_argument("--out", default=None,
                    help="Output manifest path (default: {view_dir}/manifest.jsonl)")

    gp = sub.add_parser("generate_rubrics",
                        help="Generate and cache rubrics for scenes in manifest")
    gp.add_argument("--manifest", required=True)
    gp.add_argument("--rubric_dir", required=True)
    gp.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini"])
    gp.add_argument("--model", required=True)
    gp.add_argument("--prompt_version", default="v1")
    gp.add_argument("--overwrite", action="store_true",
                    help="Re-generate even if a cached rubric exists")

    ep = sub.add_parser("evaluate", help="Run judge_scene on every manifest video")
    ep.add_argument("--manifest", required=True)
    ep.add_argument("--rubric_dir", required=True)
    ep.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini"])
    ep.add_argument("--model", required=True, help="VLM judge model")
    ep.add_argument("--rubric_provider", default=None,
                    help="Provider used to generate rubrics (default: --provider)")
    ep.add_argument("--rubric_model", default=None,
                    help="Model used to generate rubrics (default: --model)")
    ep.add_argument("--prompt_version", default="v1")
    ep.add_argument("--no_ref", action="store_true",
                    help="Do not pass GT reference image to the judge")
    ep.add_argument("--alpha", type=float, default=0.6,
                    help="Environment weight in overall score (default: 0.6)")
    ep.add_argument("--output", required=True,
                    help="Output JSONL path (one record per video)")
    ep.add_argument("--resume", action="store_true",
                    help="Append to --output and skip rows already present")

    args = ap.parse_args()
    if args.command == "build_manifest":
        return cmd_build_manifest(args)
    if args.command == "generate_rubrics":
        return cmd_generate_rubrics(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
