#!/usr/bin/env python3
"""No-GPU, no-API fixture "world model" for the WMEval Lite runner.

Writes one candidate video per (case, seed) for the requested suite dimensions
and a strict candidate manifest, so the full runner path (candidate discovery,
dry-run work plan, and — for the cheap dimensions — real metrics) can be
exercised without a model, a GPU or a hosted judge.

Modes
-----
``identity``  byte-copy of the case's reference video (read through the case
              loader).  Reference-matching dimensions should score ~perfectly.
``degraded``  the reference re-encoded with a deterministic perturbation
              (Gaussian blur sigma=2 + brightness +20/255) via ffmpeg, falling
              back to OpenCV.  Items are skipped (with a reason) when neither
              encoder is available.

Cases without a reference video (reference-free prompt cases) are skipped and
reported.

Layout (matches ``run_wmeval_lite.py run --model-root`` and the manifest)::

    <output-root>/<model>/<Scene>/<variant>/<seed>.mp4
    <output-root>/<model>_candidates.jsonl   # {"model","case_id","seed","candidate_video"}

Examples::

    python tools/fixture_backend.py --output-root /tmp/wm_fixtures \\
        --dimensions lpips --mode identity
    python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \\
        --candidate-manifest /tmp/wm_fixtures/fixture_candidates.jsonl \\
        --dimensions lpips --output-dir /tmp/wm_fixtures/report --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite.case_loader import load_case  # noqa: E402
from evaluation.lite.suite_loader import load_suite  # noqa: E402


MODES = ("identity", "degraded")
ENCODERS = ("auto", "ffmpeg", "opencv")
DEFAULT_MODEL = "fixture"
DEFAULT_SUITE = REPO_ROOT / "eval_sets" / "wmeval_lite_v1"

# Deterministic perturbation used by ``degraded`` (kept identical across encoders).
BLUR_SIGMA = 2.0
BRIGHTNESS_SHIFT = 20  # out of 255


class FixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class FixtureResult:
    written: list[dict[str, str]]
    skipped: list[dict[str, str]]
    manifest: Path
    model_root: Path


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _safe_component(value: str, what: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise FixtureError(f"{what} must be a single safe path component, got {value!r}")
    return value


def select_cases(
    suite_root: Path,
    dimensions: Sequence[str] | None = None,
    cases: Sequence[str] | None = None,
) -> list[str]:
    suite = load_suite(suite_root)
    requested = list(dimensions) if dimensions else list(suite.dimensions)
    unknown = sorted(set(requested) - set(suite.dimensions))
    if unknown:
        raise FixtureError(f"unknown dimension(s): {unknown}")
    selected = sorted(
        {case_id for dim in requested for case_id in suite.dimensions[dim].cases}
    )
    if cases:
        outside = sorted(set(cases) - set(selected))
        if outside:
            raise FixtureError(
                f"case(s) {outside} are not in the selected dimension(s) {requested}"
            )
        selected = sorted(set(cases))
    return selected


# --------------------------------------------------------------------------
# degraded encoders
# --------------------------------------------------------------------------


def _degrade_ffmpeg(source: Path, target: Path) -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg not found on PATH"
    command = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-vf", f"gblur=sigma={BLUR_SIGMA},eq=brightness={BRIGHTNESS_SHIFT / 255:.6f}",
        "-an", "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return f"ffmpeg failed: {completed.stderr.strip()[-300:]}"
    return None


def _degrade_opencv(source: Path, target: Path) -> str | None:
    try:
        import cv2
    except ImportError:
        return "OpenCV (cv2) is not installed"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return f"OpenCV cannot open {source}"
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        return "OpenCV VideoWriter (mp4v) unavailable"
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.GaussianBlur(frame, (0, 0), BLUR_SIGMA)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=BRIGHTNESS_SHIFT)
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        target.unlink(missing_ok=True)
        return f"OpenCV decoded no frames from {source}"
    return None


def degrade_video(source: Path, target: Path, encoder: str = "auto") -> str | None:
    """Write a perturbed copy of ``source``; return a skip reason or None."""
    order = {"auto": ("ffmpeg", "opencv"), "ffmpeg": ("ffmpeg",), "opencv": ("opencv",)}[
        encoder
    ]
    reasons = []
    for name in order:
        reason = (_degrade_ffmpeg if name == "ffmpeg" else _degrade_opencv)(source, target)
        if reason is None:
            return None
        reasons.append(reason)
    return "degraded mode unavailable: " + "; ".join(reasons)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def generate_fixtures(
    *,
    output_root: str | Path,
    suite_root: str | Path = DEFAULT_SUITE,
    dimensions: Sequence[str] | None = None,
    cases: Sequence[str] | None = None,
    mode: str = "identity",
    model: str = DEFAULT_MODEL,
    seeds: Sequence[str] = ("0",),
    encoder: str = "auto",
    manifest_path: str | Path | None = None,
    log=print,
) -> FixtureResult:
    if mode not in MODES:
        raise FixtureError(f"mode must be one of {MODES}, got {mode!r}")
    if encoder not in ENCODERS:
        raise FixtureError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    model = _safe_component(model, "model")
    seeds = [_safe_component(str(seed), "seed") for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise FixtureError(f"seeds must be nonempty and unique, got {seeds}")

    suite_root = Path(suite_root).resolve()
    output_root = Path(output_root).resolve()
    model_root = output_root / model
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output_root / f"{model}_candidates.jsonl"
    )

    suite = load_suite(suite_root)
    case_ids = select_cases(suite_root, dimensions, cases)
    allowed = suite.all_case_ids()

    written: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for case_id in case_ids:
        case = load_case(suite.cases_root / case_id, allowed_cases=allowed)
        reference = case.reference_video
        if reference is None or not reference.is_file():
            reason = (
                "case has no reference video (reference-free)"
                if reference is None
                else f"reference video missing: {reference}"
            )
            skipped.append({"case_id": case_id, "reason": reason})
            log(f"[fixture] skip {case_id}: {reason}")
            continue
        for seed in seeds:
            target = model_root / case_id / f"{seed}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "identity":
                shutil.copyfile(reference, target)
            else:
                partial = target.with_name(f".{target.stem}.partial.mp4")
                reason = degrade_video(reference, partial, encoder)
                if reason is not None:
                    skipped.append({"case_id": case_id, "seed": seed, "reason": reason})
                    log(f"[fixture] skip {case_id}/{seed}: {reason}")
                    continue
                os.replace(partial, target)
            try:
                video_ref = target.relative_to(manifest.parent).as_posix()
            except ValueError:
                video_ref = str(target)
            written.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "seed": seed,
                    "candidate_video": video_ref,
                }
            )
            log(f"[fixture] {mode} {case_id}/{seed} -> {target}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written),
        encoding="utf-8",
    )
    os.replace(tmp, manifest)
    return FixtureResult(
        written=written, skipped=skipped, manifest=manifest, model_root=model_root
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fixture_backend",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument(
        "--output-root",
        required=True,
        help="Fixtures go to <output-root>/<model>/<Scene>/<variant>/<seed>.mp4",
    )
    parser.add_argument(
        "--dimensions", default=None, help="Comma-separated dimension ids (default: all)"
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids to restrict to (must belong to the dimensions)",
    )
    parser.add_argument("--mode", choices=MODES, default="identity")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds (default: 0)")
    parser.add_argument("--encoder", choices=ENCODERS, default="auto")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path (default: <output-root>/<model>_candidates.jsonl)",
    )
    args = parser.parse_args(argv)

    try:
        result = generate_fixtures(
            output_root=args.output_root,
            suite_root=args.suite,
            dimensions=_split_csv(args.dimensions),
            cases=_split_csv(args.cases),
            mode=args.mode,
            model=args.model,
            seeds=_split_csv(args.seeds) or [],
            encoder=args.encoder,
            manifest_path=args.manifest,
        )
    except FixtureError as exc:
        print(f"fixture_backend: error: {exc}", file=sys.stderr)
        return 2

    print(
        f"\n[fixture] wrote {len(result.written)} candidate(s), skipped "
        f"{len(result.skipped)}; manifest: {result.manifest}"
    )
    print(f"[fixture] model root: {result.model_root}")
    if not result.written:
        print("[fixture] nothing generated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
