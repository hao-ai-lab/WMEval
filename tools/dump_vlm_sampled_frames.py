"""Dump the exact frames the VLM judge sees for a given video.

Reuses ``evaluation.modules.vlm_consistency.frame_sampler.sample_video_frames``
without modifying the judging pipeline. Frames are saved with filenames that
match the labels VLM evidence cites (``frame_t=X.Xs``), so ``frame_evidence``
strings can be looked up directly on disk.

Default sampling parameters mirror ``batch.py`` (SAMPLE_DURATION=10.0,
SAMPLE_FRAMES=18); override with CLI flags if a result was generated with
non-default settings.

Examples
--------
    # Default: 18 frames over the first 10 seconds, written to ./_frames/
    python tools/dump_vlm_sampled_frames.py path/to/video.mp4

    # Custom output dir + reference frame copy
    python tools/dump_vlm_sampled_frames.py video.mp4 \
        -o frames_out --ref-frame ref.png

    # Look up a single timestamp from VLM evidence
    python tools/dump_vlm_sampled_frames.py video.mp4 --only "frame_t=2.2s"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_sample_video_frames():
    """Load frame_sampler.sample_video_frames without triggering the heavy
    ``evaluation`` package __init__ (which imports torch, clip4clip, etc.).

    We register a stub ``evaluation`` package and pre-load the two submodules
    that ``frame_sampler`` actually needs (``context`` and ``video_io``), then
    load ``frame_sampler.py`` itself.
    """
    import importlib.util
    import types

    eval_dir = REPO_ROOT / "evaluation"

    # Stub the package so `from evaluation.X import Y` skips __init__.py
    pkg = types.ModuleType("evaluation")
    pkg.__path__ = [str(eval_dir)]  # marks it as a package
    sys.modules.setdefault("evaluation", pkg)

    def _load(modname: str, path: Path):
        if modname in sys.modules:
            return sys.modules[modname]
        spec = importlib.util.spec_from_file_location(modname, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("evaluation.context", eval_dir / "context.py")
    _load("evaluation.video_io", eval_dir / "video_io.py")

    sampler = _load(
        "evaluation._frame_sampler_standalone",
        eval_dir / "modules" / "vlm_consistency" / "frame_sampler.py",
    )
    return sampler.sample_video_frames


sample_video_frames = _load_sample_video_frames()


SAMPLE_DURATION = 10.0
SAMPLE_FRAMES = 18


def _label(ts: float) -> str:
    return f"frame_t={ts:.1f}s"


def dump_frames(
    video_path: Path,
    out_dir: Path,
    *,
    duration: float,
    num_frames: int,
    ref_frame: Optional[Path] = None,
    only_labels: Optional[List[str]] = None,
) -> List[Path]:
    import cv2

    sample_fps = num_frames / duration

    frames, timestamps, indices = sample_video_frames(
        str(video_path),
        first_n_seconds=duration,
        sample_fps=sample_fps,
        max_frames=num_frames,
    )

    if not frames:
        raise SystemExit(f"No frames sampled from {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    if ref_frame is not None:
        if not ref_frame.is_file():
            raise SystemExit(f"--ref-frame not found: {ref_frame}")
        dst = out_dir / f"ref_frame{ref_frame.suffix.lower() or '.png'}"
        if only_labels is None or "ref_frame" in only_labels:
            shutil.copyfile(ref_frame, dst)
            written.append(dst)

    for frame, ts, idx in zip(frames, timestamps, indices):
        label = _label(ts)
        if only_labels is not None and label not in only_labels:
            continue
        dst = out_dir / f"{label}__idx{idx}.png"
        ok = cv2.imwrite(str(dst), frame)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {dst}")
        written.append(dst)

    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("video", type=Path, help="Path to the generated video")
    p.add_argument(
        "-o", "--out-dir", type=Path, default=Path("_frames"),
        help="Directory to write frames into (default: ./_frames)",
    )
    p.add_argument(
        "--duration", type=float, default=SAMPLE_DURATION,
        help=f"first_n_seconds passed to sampler (default {SAMPLE_DURATION})",
    )
    p.add_argument(
        "--num-frames", type=int, default=SAMPLE_FRAMES,
        help=f"max_frames / target sample count (default {SAMPLE_FRAMES})",
    )
    p.add_argument(
        "--ref-frame", type=Path, default=None,
        help="Optional reference image to copy as ref_frame.* alongside dumps",
    )
    p.add_argument(
        "--only", action="append", default=None, metavar="LABEL",
        help="Restrict output to the given VLM label(s); repeatable. "
             "Example: --only 'frame_t=2.2s' --only ref_frame",
    )
    args = p.parse_args(argv)

    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")

    written = dump_frames(
        args.video,
        args.out_dir,
        duration=args.duration,
        num_frames=args.num_frames,
        ref_frame=args.ref_frame,
        only_labels=args.only,
    )

    print(f"Wrote {len(written)} file(s) to {args.out_dir}/", file=sys.stderr)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
