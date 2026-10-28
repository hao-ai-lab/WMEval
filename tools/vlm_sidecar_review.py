#!/usr/bin/env python3
"""Dev-time sidecar: ask a VLM judge whether B.1/B.3 module output is sensible.

This is NOT a production metric and is NOT wired into evaluation/registry.py
or the suite CLI. It exists only to help validate `environment_object_consistency`
and `kinematics_mechanics` while real Unity ground-truth artifacts are still
pending: run the module on a real
candidate video, sample a few frames, and ask a VLM judge whether the reported
numbers plausibly match what's visible in the video.

Uses NVIDIA's OpenAI-compatible inference gateway (switchyard) rather than
api.openai.com, so it works without an OPENAI_API_KEY during development.

Examples::

    python tools/vlm_sidecar_review.py \
        --module environment_object_consistency \
        --candidate-video lite/cases/BallRoller/ctrl/reference/reference.mp4 \
        --text-prompt "ball. wall."

    python tools/vlm_sidecar_review.py \
        --module kinematics_mechanics \
        --candidate-video path/to/rollout.mp4 \
        --text-prompt "ball. car."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_MODEL = "switchyard/openai/gpt-5.6-sol"
DEFAULT_KEY_PATH = Path.home() / ".credentials" / "openai_key"

METRIC_EXPLANATIONS = {
    "environment_object_consistency": (
        "Paper Appendix B.1 (Environment Object and Character Consistency). "
        "`object_permanence_pct`: % of frames each tracked anchor object stayed "
        "detectable (higher = object never vanished/got occluded unexpectedly). "
        "`character_consistency_pct`: % of frames the tracked character kept a "
        "DINOv2-embedding identity match to frame 0 (higher = same-looking "
        "character throughout, no identity drift/swap). `depth_absrel_pct` / "
        "`pairwise_distance_mae_pct`: relative error of estimated object depth / "
        "pairwise distances vs Unity ground truth (lower = more geometrically "
        "accurate; null if ground truth isn't wired for this case yet)."
    ),
    "kinematics_mechanics": (
        "Paper Appendix B.3 (Kinematics and Mechanics). `trajectory_fidelity`: "
        "FDE (final displacement error) and DTW (dynamic time warping distance) "
        "between the tracked object's reconstructed path and the Unity reference "
        "trajectory (lower = motion matches the reference more closely; null if "
        "ground truth isn't wired). `mechanical_quantities`: momentum/kinetic-"
        "energy ExactMatch % and relative ErrorRate vs Unity-derived ground truth "
        "(null if ground truth isn't wired). Note: this implementation currently "
        "does 2D pixel-space tracking, not full 3D metric-space reconstruction — "
        "the VLM judge should be asked whether the *qualitative* motion pattern "
        "(object moves smoothly, tracked consistently, no track ID swaps) looks "
        "sane, not whether absolute physical units are correct."
    ),
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_verdict": {
            "type": "string",
            "enum": ["plausible", "questionable", "implausible"],
        },
        "per_metric_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "reported_value": {"type": "string"},
                    "visual_agreement": {
                        "type": "string",
                        "enum": ["agrees", "unclear", "disagrees"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["metric", "reported_value", "visual_agreement", "reasoning"],
                "additionalProperties": False,
            },
        },
        "implementation_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Anything about the metric definitions/tracking that looks wrong or fragile from the frames alone, independent of the specific numbers.",
        },
        "summary": {"type": "string"},
    },
    "required": ["overall_verdict", "per_metric_findings", "implementation_concerns", "summary"],
    "additionalProperties": False,
}


def _load_api_key(key_path: Path) -> str:
    if not key_path.exists():
        raise FileNotFoundError(
            f"No API key at {key_path}. Pass --api-key-path or set it up first."
        )
    return key_path.read_text().strip()


def _sample_frames_b64(video_path: str, n: int = 6) -> List[str]:
    from evaluation.modules.vlm_consistency.frame_sampler import (
        encode_frame_b64,
        sample_video_frames,
    )

    frames, _ts, _idx = sample_video_frames(video_path, num_samples=n)
    return [encode_frame_b64(f) for f in frames]


def _run_module(module_name: str, candidate_video: str, text_prompt: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    # Build a minimal context; both modules only read context.candidate_video
    # and context.artifacts, so reference == candidate is fine for this sidecar.
    from evaluation.video_io import build_evaluation_context

    context = build_evaluation_context(
        reference_video=candidate_video,
        candidate_video=candidate_video,
        eval_fps=4.0,
        max_pairs=32,
        text_prompt=text_prompt,
    )
    context.artifacts.update(artifacts)

    if module_name == "environment_object_consistency":
        from evaluation.modules.environment_object_consistency import (
            analyze_environment_object_consistency_from_context,
        )

        context.artifacts.setdefault("eoc_text_prompt", text_prompt)
        return analyze_environment_object_consistency_from_context(context=context)

    if module_name == "kinematics_mechanics":
        from evaluation.modules.kinematics_mechanics import (
            analyze_kinematics_mechanics_from_context,
        )

        context.artifacts.setdefault("kin_text_prompt", text_prompt)
        return analyze_kinematics_mechanics_from_context(context=context)

    raise ValueError(f"Unknown module: {module_name!r}")


def _summarize_report_for_prompt(report: Dict[str, Any]) -> str:
    # Keep large nested detail (per-frame tracks, trajectories) out of the
    # prompt; the VLM judges plausibility, it doesn't need every coordinate.
    trimmed = dict(report)
    for verbose_key in ("trajectories", "character_consistency_detail", "objects"):
        if verbose_key in trimmed and trimmed[verbose_key] is not None:
            trimmed[verbose_key] = f"<omitted, {len(trimmed[verbose_key])} entries>"
    return json.dumps(trimmed, indent=2, default=str)


def call_vlm_judge(
    *,
    module_name: str,
    report: Dict[str, Any],
    frames_b64: List[str],
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    system_prompt = (
        "You are a careful reviewer sanity-checking a video-evaluation pipeline "
        "during development. You will see sampled frames from a video and a JSON "
        "report of metrics computed for that same video by an automated module. "
        "Judge whether the reported numbers are PLAUSIBLE given what you can see "
        "in the frames — you cannot verify exact values, but you can catch "
        "obviously wrong results (e.g. object_permanence_pct=100% when the object "
        "is clearly occluded/off-screen for several frames, or a trajectory metric "
        "reported as null when it should have data, or character_consistency "
        "claiming high identity match despite an obvious character swap). "
        f"Metric definitions for this module:\n{METRIC_EXPLANATIONS[module_name]}\n\n"
        "Respond ONLY by calling the structured output tool."
    )

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": f"Module report:\n```json\n{_summarize_report_for_prompt(report)}\n```"}
    ]
    for b64 in frames_b64:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "sidecar_review",
                "schema": REVIEW_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=2000,
    )
    return json.loads(response.output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", required=True, choices=["environment_object_consistency", "kinematics_mechanics"])
    parser.add_argument("--candidate-video", required=True)
    parser.add_argument("--text-prompt", default="")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--api-key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=None, help="Optional path to write the full JSON result.")
    args = parser.parse_args()

    api_key = _load_api_key(Path(args.api_key_path))

    print(f"[1/3] Running {args.module} on {args.candidate_video} ...", file=sys.stderr)
    report = _run_module(args.module, args.candidate_video, args.text_prompt, artifacts={})

    print("[2/3] Sampling frames for VLM review ...", file=sys.stderr)
    frames_b64 = _sample_frames_b64(args.candidate_video, n=args.num_frames)

    print(f"[3/3] Asking {args.model} to judge plausibility ...", file=sys.stderr)
    review = call_vlm_judge(
        module_name=args.module,
        report=report,
        frames_b64=frames_b64,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
    )

    result = {"module_report": report, "vlm_review": review}
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    print(f"\n=== VERDICT: {review['overall_verdict'].upper()} ===", file=sys.stderr)
    print(review["summary"], file=sys.stderr)


if __name__ == "__main__":
    main()
