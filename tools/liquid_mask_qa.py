#!/usr/bin/env python3
"""SAM2 mask QA for the LiquidContainer scene.

Renders the liquid-mask and bottle-mask SAM2 segmentations as overlays at
first / middle / last frame for one or more videos. Run this BEFORE the
full eval to confirm the prompt actually segments what you think it does
on every video — a wrong mask poisons all four sloshing metrics silently.

Outputs a directory with overlay PNGs and a side-by-side strip per video,
plus a `manifest.json` summarising mask validity (how many frames had a
non-empty mask).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.modules import _liquid_common as _lc


def _overlay(frame_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    if mask.shape != frame_bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame_bgr.shape[1], frame_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = frame_bgr.copy()
    overlay = np.zeros_like(frame_bgr)
    overlay[mask] = color_bgr
    out = cv2.addWeighted(out, 1.0, overlay, 0.45, 0.0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, 1)
    return out


def _read_frame(video_path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} from {video_path}")
    return frame


def qa_one_video(
    *,
    video_path: str,
    out_dir: Path,
    sam2_model: str,
    liquid_prompt: str,
    bottle_prompt: str,
    cache_root: Path,
    max_pairs: int | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _lc._read_video_frames_uint8(video_path, max_frames=max_pairs)
    T = meta.shape[0]
    fps = float(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 0.0)

    liquid = _lc.get_liquid_mask_video(
        video_path=video_path, sam2_model=sam2_model, prompt=liquid_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )
    bottle = _lc.get_bottle_centroid_trace(
        video_path=video_path, sam2_model=sam2_model,
        bottle_prompt=bottle_prompt, fps=fps, max_pairs=max_pairs,
        cache_dir=cache_root,
    )
    bottle_masks_result = _lc.get_liquid_mask_video(
        video_path=video_path, sam2_model=sam2_model, prompt=bottle_prompt,
        fps=fps, max_pairs=max_pairs, cache_dir=cache_root,
    )

    indices = sorted({0, max(0, T // 2), max(0, T - 1)})
    strips: list[np.ndarray] = []
    for idx in indices:
        frame = _read_frame(video_path, idx)
        liq_mask = liquid.masks[idx] if idx < liquid.masks.shape[0] else np.zeros(frame.shape[:2], bool)
        bot_mask = bottle_masks_result.masks[idx] if idx < bottle_masks_result.masks.shape[0] else np.zeros(frame.shape[:2], bool)
        liq_overlay = _overlay(frame, liq_mask, (0, 220, 255))      # cyan-ish liquid
        bot_overlay = _overlay(frame, bot_mask, (255, 80, 80))      # blue bottle
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_liquid.png"), liq_overlay)
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_bottle.png"), bot_overlay)
        strip = np.hstack([frame, liq_overlay, bot_overlay])
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_strip.png"), strip)
        strips.append(strip)

    if strips:
        cv2.imwrite(str(out_dir / "summary_strip.png"), np.vstack(strips))

    liquid_validity = float(np.mean(liquid.masks.any(axis=(1, 2))))
    bottle_validity = float(np.mean(bottle.valid))

    return {
        "video": video_path,
        "frames": int(T),
        "fps": fps,
        "liquid_prompt": liquid_prompt,
        "bottle_prompt": bottle_prompt,
        "liquid_mask_validity": liquid_validity,
        "bottle_mask_validity": bottle_validity,
        "qa_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", help="One or more video paths to QA")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory for overlays")
    ap.add_argument("--liquid_prompt", default="the liquid inside the bottle")
    ap.add_argument("--bottle_prompt", default="the glass bottle")
    ap.add_argument("--sam2_model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--cache_root", type=Path, default=Path("runs/liquid_cache"))
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = use full video; otherwise cap decode at this many frames")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_pairs = int(args.max_frames) or None

    summaries: list[dict] = []
    for vpath in args.videos:
        sub = args.out_dir / Path(vpath).stem
        summary = qa_one_video(
            video_path=vpath,
            out_dir=sub,
            sam2_model=args.sam2_model,
            liquid_prompt=args.liquid_prompt,
            bottle_prompt=args.bottle_prompt,
            cache_root=args.cache_root,
            max_pairs=max_pairs,
        )
        summaries.append(summary)
        print(
            f"[QA] {vpath}: liquid_validity={summary['liquid_mask_validity']:.3f} "
            f"bottle_validity={summary['bottle_mask_validity']:.3f}  ->  {sub}"
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
