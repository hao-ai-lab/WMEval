#!/usr/bin/env python3
"""Record the reference-side judge answers for a physical_causality case.

The reference side of this dimension is a fixture, not a computation: the Unity
video, the QA dataset, and the judge settings are all pinned, so the answers are
a property of the benchmark rather than of a run. Recording them once and
shipping them in the case bundle is the same move already made for the liquid
dimension's SAM2/RAFT `.npz` artifacts.

Two things follow from recording several independent passes:

  * The reference numbers stop moving. Judge answers are sampled (the gpt-5
    family rejects a non-default `temperature`, so none is sent), so a single
    live pass makes `csr_unity` a coin flip.
  * CSR becomes expressible as a fraction. CSR is all-or-nothing per video, so
    over N passes it is successes/N rather than only 0 or 100.

Usage (needs OPENAI_API_KEY; ~19 QA x N passes of paid calls per case):

    python tools/dataset/record_causal_reference_answers.py \\
        lite/cases/BeamDomino/causal \\
        --passes 5

Then refresh the bundle checksums:

    python tools/dataset/build_lite_case.py lite/cases/BeamDomino/causal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modules._causal_step_recorded_judge import (  # noqa: E402
    collect_pass_answers,
    write_recorded_answers,
)
from evaluation.modules._causal_step_runner import run_causal_step_for_video  # noqa: E402

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_NUM_FRAMES = 8
DEFAULT_PROMPT_VERSION = "v1"
ARTIFACT_NAME = "judge_answers.json"


def _video_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0:
        raise SystemExit(f"could not read fps from {path}")
    return fps


def _build_judge(cache_dir: Path, *, model: str, num_frames: int, prompt_version: str):
    from openai import OpenAI

    from evaluation.modules._causal_step_openai_judge import OpenAIJudge
    from evaluation.modules.causal_step import _read_window_uint8

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    return OpenAIJudge(
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        model=model,
        cache_dir=cache_dir / "vlm_cache",
        frames_provider=_read_window_uint8,
        num_frame_samples=num_frames,
        prompt_version=prompt_version,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case_dir", type=Path, help="case bundle root")
    ap.add_argument("--passes", type=int, default=5,
                    help="independent judge passes to record (paper uses 5)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--cache-dir", type=Path, default=Path("runs/causal_step_cache"),
                    help="judge response cache; reused across passes and cases")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be recorded and exit")
    args = ap.parse_args(argv)

    case_dir: Path = args.case_dir
    qa_path = case_dir / "reference" / "artifacts" / "qa_dataset.json"
    video_path = case_dir / "reference" / "reference.mp4"
    out_path = case_dir / "reference" / "artifacts" / ARTIFACT_NAME
    for p in (qa_path, video_path):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    video_id = json.loads(qa_path.read_text())["videos"][0]["video_id"]
    n_qa = sum(
        len(v["descriptive_qas"]) + len(v["causal_qas"])
        for v in json.loads(qa_path.read_text())["videos"]
        if v["video_id"] == video_id
    )
    print(
        f"{case_dir.name}: video_id={video_id} qa_items={n_qa} passes={args.passes} "
        f"model={args.model} K={args.num_frames} -> up to "
        f"{n_qa * args.passes} judge calls (cached ones are free)"
    )
    if args.dry_run:
        return 0

    judge = _build_judge(
        args.cache_dir,
        model=args.model,
        num_frames=args.num_frames,
        prompt_version=args.prompt_version,
    )

    with tempfile.TemporaryDirectory() as tmp:
        answers_dir = Path(tmp)
        run_causal_step_for_video(
            qa_dataset_path=qa_path,
            video_id=video_id,
            video_path=str(video_path),
            judge=judge,
            answers_out_dir=answers_dir,
            causal_fail_advance=True,
            fps=_video_fps(video_path),
            repeats=args.passes,
        )
        passes = collect_pass_answers(
            answers_dir, video_id=video_id, repeats=args.passes
        )

    write_recorded_answers(
        out_path,
        passes=passes,
        video_id=video_id,
        # Ask the judge what it actually applied rather than restating the CLI
        # flags: `temperature` is omitted from the request for the gpt-5 family,
        # so hard-coding 0.0 here would stamp the recording with a setting that
        # was never sent.
        judge_settings=judge.describe_settings(),
    )
    agreement = len({tuple(sorted(p.items())) for p in passes})
    print(
        f"wrote {out_path} ({args.passes} passes, {agreement} distinct outcome(s))\n"
        f"now run: python tools/dataset/build_lite_case.py {case_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
