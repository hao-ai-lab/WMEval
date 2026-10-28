#!/usr/bin/env python3
"""Author the portable LiquidContainer reference artifacts for WM-Eval Lite.

Writes ``liquid_mask.npz`` / ``bottle_mask.npz`` / ``raft_pooled_flow.npz`` into
each ``cases/LiquidContainer/fluid_<profile>/reference/artifacts/`` directory via
``evaluation.lite.reference_artifacts.write_*`` -- the only writer path the Lite
trust layer accepts (``_read_arrays`` enforces an exact key set and dtypes, so a
hand-rolled ``np.savez`` is rejected at run time).

Two sources are supported:

``--from-cache`` (default)
    Transcode the production SAM2/RAFT pickles that already exist in a
    ``runs/liquid_cache`` directory. No SAM2, GroundingDINO or RAFT inference,
    therefore no GPU and no model downloads. Cache entries are located by
    recomputing their content+config hashes rather than by hardcoded filename,
    so a stale variant sitting next to the production one (e.g. a RAFT field
    computed at a looser ``min_mask_frac``) can never be picked up silently.

``--recompute``
    Run the live pipeline against the normalized reference video instead. Needed
    only if the cache is unavailable.

``--synthetic``
    Run the live pipeline (``--recompute``) for the synthetic ``liquid_container``
    trajectories, writing into their render source of truth
    ``runs/synthetic_unity/liquid_container/<case>/<slug>/artifacts/`` (``publish``
    with ``tools/render_synthetic_references.py`` copies them into the dataset).

The reference video must already be normalized to the locked 832x464 @ 16 fps
geometry (``evaluation.lite.normalization.normalize_liquid_video``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite.reference_artifacts import (  # noqa: E402
    write_liquid_mask,
    write_piv_field,
)
from evaluation.modules import _liquid_common as _lc  # noqa: E402

PROFILES = ("rest", "step_hold", "impulse_decay", "chirp_strafe")

# The locked v1 liquid evaluator config. These are the same values every
# fluid_* case.yaml carries under evaluator_lock.module_config.liquid; keeping
# one copy here means the tool and the lock cannot silently disagree.
SAM2_MODEL = "facebook/sam2.1-hiera-small"
LIQUID_PROMPT = "the liquid inside the bottle"
BOTTLE_PROMPT = "the glass bottle"
NORM_FPS = 16.0
LIQUID_MAX_PAIRS = None  # case.yaml spells this 0; _expected_liquid_lock maps 0 -> None
RAFT_WINDOW_SIZE = 32
RAFT_OVERLAP = 16
RAFT_MIN_MASK_FRAC = 0.5

# Cached reference runs were keyed off a video named `ref_norm.mp4`.
CACHE_STEM = "ref_norm"


class BuildError(RuntimeError):
    """A reference artifact could not be authored."""


def _mask_config_hash(prompt: str) -> str:
    return _lc._config_hash(
        {
            "sam2_model": SAM2_MODEL,
            "prompt": prompt,
            "fps": NORM_FPS,
            "max_pairs": LIQUID_MAX_PAIRS,
            "gdino_model": _lc._GDINO_MODEL_ID,
            "gdino_box_threshold": _lc._GDINO_BOX_THRESHOLD,
            "gdino_text_threshold": _lc._GDINO_TEXT_THRESHOLD,
        }
    )


def _raft_config_hash(mask: "_lc.LiquidMaskResult") -> str:
    mask_bytes_hash = hashlib.sha256(
        np.ascontiguousarray(mask.masks).tobytes()
    ).hexdigest()[:16]
    return _lc._config_hash(
        {
            "estimator": "raft_pooled_v1",
            "window_size": RAFT_WINDOW_SIZE,
            "overlap": RAFT_OVERLAP,
            "fps": NORM_FPS,
            "max_frames": LIQUID_MAX_PAIRS,
            "sam2_model": mask.sam2_model,
            "prompt": mask.prompt,
            "mask_hash": mask_bytes_hash,
            "min_mask_frac": float(RAFT_MIN_MASK_FRAC),
        }
    )


def _load_pickle(path: Path):
    if not path.is_file():
        raise BuildError(f"cache entry not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _content_hashes(frozen_metrics: Path) -> dict[str, str]:
    """profile -> normalized_content_hash, read from the frozen-reference table."""
    payload = json.loads(frozen_metrics.read_text())
    table = payload["frozen_reference_metrics"]
    return {
        profile: str(entry["normalized_content_hash"])
        for profile, entry in table.items()
        if profile in PROFILES
    }


def _video_frame_count(video: Path) -> int:
    import subprocess

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out)


def _expected_raft_shape(mask_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Mirror reference_artifacts._expected_raft_shape for a pre-flight check."""
    frames, height, width = mask_shape
    step = max(1, RAFT_WINDOW_SIZE - RAFT_OVERLAP)
    half = RAFT_WINDOW_SIZE // 2
    ys = np.arange(half, height - half + 1, step, dtype=int)
    xs = np.arange(half, width - half + 1, step, dtype=int)
    return (frames - 1, int(ys.size), int(xs.size))


def _load_from_cache(cache_dir: Path, content_hash: str):
    liquid_path = cache_dir / (
        f"{CACHE_STEM}.sam2_mask.{content_hash}.{_mask_config_hash(LIQUID_PROMPT)}.pkl"
    )
    bottle_path = cache_dir / (
        f"{CACHE_STEM}.sam2_mask.{content_hash}.{_mask_config_hash(BOTTLE_PROMPT)}.pkl"
    )
    liquid = _load_pickle(liquid_path)
    bottle = _load_pickle(bottle_path)
    raft_path = cache_dir / (
        f"{CACHE_STEM}.raft_pooled.{content_hash}.{_raft_config_hash(liquid)}.pkl"
    )
    field = _load_pickle(raft_path)
    return liquid, bottle, field, (liquid_path, bottle_path, raft_path)


def _recompute(video: Path, cache_root: Path):
    liquid = _lc.get_liquid_mask_video(
        video_path=str(video), sam2_model=SAM2_MODEL, prompt=LIQUID_PROMPT,
        fps=NORM_FPS, max_pairs=LIQUID_MAX_PAIRS, cache_dir=str(cache_root),
    )
    bottle = _lc.get_liquid_mask_video(
        video_path=str(video), sam2_model=SAM2_MODEL, prompt=BOTTLE_PROMPT,
        fps=NORM_FPS, max_pairs=LIQUID_MAX_PAIRS, cache_dir=str(cache_root),
    )
    field = _lc.get_raft_flow_field(
        video_path=str(video), mask=liquid, window_size=RAFT_WINDOW_SIZE,
        overlap=RAFT_OVERLAP, fps=NORM_FPS, max_frames=LIQUID_MAX_PAIRS,
        cache_dir=str(cache_root), min_mask_frac=RAFT_MIN_MASK_FRAC,
    )
    return liquid, bottle, field, (video, video, video)


def _verify(profile, liquid, bottle, field, video):
    if liquid.masks.shape != bottle.masks.shape:
        raise BuildError(
            f"{profile}: liquid/bottle mask shape mismatch "
            f"{liquid.masks.shape} vs {bottle.masks.shape}"
        )
    for label, mask, prompt in (
        ("liquid", liquid, LIQUID_PROMPT),
        ("bottle", bottle, BOTTLE_PROMPT),
    ):
        if mask.prompt != prompt:
            raise BuildError(f"{profile}: {label} mask prompt is {mask.prompt!r}")
        if mask.sam2_model != SAM2_MODEL:
            raise BuildError(f"{profile}: {label} mask model is {mask.sam2_model!r}")
        if float(mask.fps) != NORM_FPS:
            raise BuildError(f"{profile}: {label} mask fps is {mask.fps!r}")
    want = _expected_raft_shape(liquid.masks.shape)
    if field.u.shape != want:
        raise BuildError(
            f"{profile}: RAFT shape {field.u.shape}, expected {want} from mask "
            f"{liquid.masks.shape} at window={RAFT_WINDOW_SIZE}/overlap={RAFT_OVERLAP}"
        )
    n_video = _video_frame_count(video)
    if n_video != liquid.masks.shape[0]:
        raise BuildError(
            f"{profile}: reference video has {n_video} frames but the mask has "
            f"{liquid.masks.shape[0]} -- the video is not the one the mask was "
            f"computed from, or it is not normalized to {NORM_FPS} fps"
        )


def build_profile(profile, case_dir, cache_dir, content_hash, recompute, cache_root):
    video = case_dir / "reference" / "reference.mp4"
    if not video.is_file():
        raise BuildError(f"{profile}: missing normalized reference video {video}")

    if recompute:
        liquid, bottle, field, sources = _recompute(video, cache_root)
    else:
        liquid, bottle, field, sources = _load_from_cache(cache_dir, content_hash)

    _verify(profile, liquid, bottle, field, video)

    artifacts = case_dir / "reference" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_liquid_mask(artifacts / "liquid_mask.npz", liquid)
    write_liquid_mask(artifacts / "bottle_mask.npz", bottle)
    write_piv_field(artifacts / "raft_pooled_flow.npz", field)

    print(f"  {profile}:")
    print(f"    masks  {liquid.masks.shape} fps={liquid.fps} model={liquid.sam2_model}")
    print(f"    raft   {field.u.shape} valid_frac={field.valid.mean():.4f}")
    for path in sources:
        print(f"    src    {path.name}")
    for name in ("liquid_mask.npz", "bottle_mask.npz", "raft_pooled_flow.npz"):
        size = (artifacts / name).stat().st_size
        print(f"    wrote  {name}  ({size/1e6:.1f} MB)")


SYNTHETIC_RENDERS = REPO_ROOT / "runs/synthetic_unity/liquid_container"
ARTIFACT_NAMES = ("liquid_mask.npz", "bottle_mask.npz", "raft_pooled_flow.npz")


def build_synthetic(render_dir: Path, cache_root: Path, *, overwrite: bool = False) -> bool:
    """Compute the three reference artifacts for one synthetic render directory
    (``reference.mp4`` is already the normalized 832x464 @ 16 fps video)."""
    artifacts = render_dir / "artifacts"
    if not overwrite and all((artifacts / name).is_file() for name in ARTIFACT_NAMES):
        print(f"  {render_dir.name}: present, skipped")
        return False
    video = render_dir / "reference.mp4"
    if not video.is_file():
        raise BuildError(f"{render_dir}: missing reference.mp4")
    liquid, bottle, field, _ = _recompute(video, cache_root)
    _verify(render_dir.name, liquid, bottle, field, video)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_liquid_mask(artifacts / "liquid_mask.npz", liquid)
    write_liquid_mask(artifacts / "bottle_mask.npz", bottle)
    write_piv_field(artifacts / "raft_pooled_flow.npz", field)
    print(f"  {render_dir.parent.name}/{render_dir.name}: masks {liquid.masks.shape} "
          f"raft {field.u.shape} valid_frac={field.valid.mean():.4f}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-root", type=Path,
                        default=REPO_ROOT / "lite/cases/LiquidContainer",
                        help="directory holding the fluid_<profile> case bundles")
    parser.add_argument("--cache-dir", type=Path, required=False,
                        help="runs/liquid_cache directory holding the production pickles")
    parser.add_argument("--frozen-metrics", type=Path, required=False,
                        help="frozen_reference_metrics.json (source of the per-profile "
                             "normalized_content_hash used to locate cache entries)")
    parser.add_argument("--profiles", default=",".join(PROFILES),
                        help="comma-separated subset of " + ", ".join(PROFILES))
    parser.add_argument("--recompute", action="store_true",
                        help="run SAM2/RAFT live instead of reading the cache (needs GPU)")
    parser.add_argument("--recompute-cache-root", type=Path,
                        default=Path("runs/liquid_cache"),
                        help="where --recompute writes its own pickle cache")
    parser.add_argument("--synthetic", action="store_true",
                        help="recompute for the synthetic liquid_container renders under "
                             "runs/synthetic_unity (implies --recompute)")
    parser.add_argument("--only", default=None,
                        help="--synthetic: comma-separated trajectory slugs (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="--synthetic: recompute even when the artifacts exist")
    args = parser.parse_args()

    if args.synthetic:
        wanted = {s.strip() for s in (args.only or "").split(",") if s.strip()}
        dirs = sorted(p.parent for p in SYNTHETIC_RENDERS.glob("*/*/reference.mp4")
                      if not wanted or p.parent.name in wanted)
        print(f"building synthetic liquid reference artifacts for {len(dirs)} render(s)", flush=True)
        failures = []
        for render_dir in dirs:
            try:
                build_synthetic(render_dir, args.recompute_cache_root, overwrite=args.overwrite)
            except BuildError as exc:
                failures.append(str(exc))
                print(f"  {render_dir.name}: FAILED -- {exc}", flush=True)
        if failures:
            raise SystemExit(f"\n{len(failures)} render(s) failed")
        print("ok")
        return

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in PROFILES]
    if unknown:
        parser.error(f"unknown profiles: {unknown}; expected a subset of {list(PROFILES)}")

    content_hashes: dict[str, str] = {}
    if not args.recompute:
        if not args.cache_dir or not args.frozen_metrics:
            parser.error("--cache-dir and --frozen-metrics are required unless --recompute")
        content_hashes = _content_hashes(args.frozen_metrics)
        missing = [p for p in profiles if p not in content_hashes]
        if missing:
            parser.error(f"no normalized_content_hash for {missing} in {args.frozen_metrics}")

    print(f"building liquid reference artifacts ({'recompute' if args.recompute else 'from cache'})")
    failures = []
    for profile in profiles:
        case_dir = args.case_root / f"fluid_{profile}"
        try:
            build_profile(profile, case_dir, args.cache_dir,
                          content_hashes.get(profile), args.recompute,
                          args.recompute_cache_root)
        except BuildError as exc:
            failures.append(str(exc))
            print(f"  {profile}: FAILED -- {exc}")
    if failures:
        raise SystemExit(f"\n{len(failures)} profile(s) failed")
    print("ok")


if __name__ == "__main__":
    main()
