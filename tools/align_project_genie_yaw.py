#!/usr/bin/env python3
"""Align Project Genie turn samples to a Unity reference-yaw sweep.

This is an evidence tool, not a permissive timing heuristic.  It uses a local
Depth Anything (DINOv2) backbone to compare scene layout across the rendered
Unity frame and the web-generated Project Genie view, then reports both raw
nearest neighbours and a direction-constrained phase path.  A report is only
labelled ``validated`` when the inferred path is monotonic, spans enough yaw,
and has a useful margin over its alternatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TimedImage:
    path: Path
    elapsed_sec: float


def load_samples(path: Path) -> list[TimedImage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"{path}: expected at least three samples")
    samples: list[TimedImage] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or "path" not in row or "elapsed_sec" not in row:
            raise ValueError(f"{path}: malformed sample {index}")
        image = Path(str(row["path"])).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        samples.append(TimedImage(image, float(row["elapsed_sec"])))
    if any(right.elapsed_sec <= left.elapsed_sec for left, right in zip(samples, samples[1:])):
        raise ValueError(f"{path}: elapsed_sec must be strictly increasing")
    return samples


def reference_frames(directory: Path, stride: int) -> list[Path]:
    if stride < 1:
        raise ValueError("reference stride must be positive")
    paths = sorted(directory.glob("frame_*.png"))[::stride]
    if len(paths) < 8:
        raise ValueError(f"{directory}: expected at least eight reference frames")
    return [path.resolve() for path in paths]


def _crop(image: Image.Image, kind: str) -> Image.Image:
    width, height = image.size
    if kind == "genie":
        # Remove Project Genie's progress bar, controls, watermark, and most of
        # the tutorial chrome while keeping the central generated viewport.
        box = (int(width * 0.10), int(height * 0.10), int(width * 0.90), int(height * 0.76))
    elif kind == "unity":
        # Match the vertical field used from the Genie viewport.  Unity frames
        # do not contain UI, so the horizontal extent remains intact.
        box = (0, int(height * 0.10), width, int(height * 0.76))
    else:
        raise ValueError(f"unknown image kind: {kind}")
    return image.crop(box).convert("RGB")


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode_images(
    paths: list[Path],
    *,
    kind: str,
    model_name: str,
    batch_size: int,
    processor_size: int,
) -> np.ndarray:
    """Return normalized global-plus-spatial DINOv2 descriptors."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, local_files_only=True)
    model.eval()
    descriptors: list[np.ndarray] = []
    with torch.inference_mode():
        for group in _chunks(paths, batch_size):
            images = [_crop(Image.open(path), kind) for path in group]
            inputs = processor(
                images=images,
                return_tensors="pt",
                size={"height": processor_size, "width": processor_size},
            )
            pixel_values = inputs["pixel_values"]
            output = model.backbone.forward_with_filtered_kwargs(
                pixel_values, output_hidden_states=True, output_attentions=False
            )
            tokens = output.hidden_states[-1]
            # DINOv2 uses a class token followed by patch tokens.
            patches = tokens[:, 1:, :]
            patch_height = pixel_values.shape[-2] // model.config.patch_size
            patch_width = pixel_values.shape[-1] // model.config.patch_size
            if patches.shape[1] != patch_height * patch_width:
                raise ValueError(
                    "unexpected backbone token grid: "
                    f"{patches.shape[1]} != {patch_height}*{patch_width}"
                )
            global_feature = functional.normalize(patches.mean(dim=1), dim=1)
            feature_map = patches.transpose(1, 2).reshape(
                patches.shape[0], patches.shape[2], patch_height, patch_width
            )
            spatial_feature = functional.adaptive_avg_pool2d(feature_map, (4, 8)).flatten(1)
            spatial_feature = functional.normalize(spatial_feature, dim=1)
            descriptor = functional.normalize(
                torch.cat((global_feature, spatial_feature), dim=1), dim=1
            )
            descriptors.append(descriptor.cpu().numpy().astype(np.float32))
    return np.concatenate(descriptors, axis=0)


def constrained_path(
    similarity: np.ndarray,
    *,
    direction: int,
    max_step: int,
    step_penalty: float,
) -> dict[str, Any]:
    """Find a monotonic, potentially multi-turn path through circular yaw bins.

    The path may advance from zero through two complete cycles.  A small linear
    transition penalty prevents weak semantic matches from inventing a large
    camera jump, while still allowing the data to select the speed.
    """
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    sample_count, yaw_bins = similarity.shape
    if max_step < 1 or max_step >= yaw_bins:
        raise ValueError("max_step must lie in [1, yaw_bins)")
    cycle_count = max(2, math.ceil(sample_count * max_step / yaw_bins) + 1)
    states = yaw_bins * cycle_count
    scores = np.full((sample_count, states), -np.inf, dtype=np.float64)
    back = np.full((sample_count, states), -1, dtype=np.int32)
    phase = np.arange(states) % yaw_bins
    if direction < 0:
        phase = (-phase) % yaw_bins
    scores[0] = similarity[0, phase]
    deltas = np.arange(max_step + 1, dtype=np.int32)
    penalties = step_penalty * deltas
    for sample_index in range(1, sample_count):
        for state in range(states):
            previous = state - deltas
            valid = previous >= 0
            previous = previous[valid]
            candidates = scores[sample_index - 1, previous] - penalties[valid]
            winner = int(np.argmax(candidates))
            scores[sample_index, state] = (
                candidates[winner] + similarity[sample_index, phase[state]]
            )
            back[sample_index, state] = int(previous[winner])
    final_state = int(np.argmax(scores[-1]))
    path = [final_state]
    for sample_index in range(sample_count - 1, 0, -1):
        path.append(int(back[sample_index, path[-1]]))
    path.reverse()
    phases = [int(phase[state]) for state in path]
    return {
        "direction": direction,
        "score": float(scores[-1, final_state]),
        "absolute_bins": path,
        "reference_bins": phases,
        "advanced_bins": int(path[-1] - path[0]),
        "step_bins": [int(right - left) for left, right in zip(path, path[1:])],
    }


def _top_matches(similarity: np.ndarray, count: int = 5) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for row in similarity:
        indices = np.argsort(row)[::-1][:count]
        result.append(
            [
                {"reference_bin": int(index), "similarity": float(row[index])}
                for index in indices
            ]
        )
    return result


def align(
    samples_path: Path,
    reference_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    reference_stride: int = 3,
    batch_size: int = 4,
    processor_size: int = 280,
    max_step: int | None = None,
    step_penalty: float = 0.002,
) -> dict[str, Any]:
    samples = load_samples(samples_path)
    references = reference_frames(reference_dir, reference_stride)
    query_features = encode_images(
        [sample.path for sample in samples],
        kind="genie",
        model_name=model_name,
        batch_size=batch_size,
        processor_size=processor_size,
    )
    reference_features = encode_images(
        references,
        kind="unity",
        model_name=model_name,
        batch_size=batch_size,
        processor_size=processor_size,
    )
    similarity = query_features @ reference_features.T
    yaw_bins = len(references)
    max_step = max_step or max(2, yaw_bins // 3)
    forward = constrained_path(
        similarity, direction=1, max_step=max_step, step_penalty=step_penalty
    )
    reverse = constrained_path(
        similarity, direction=-1, max_step=max_step, step_penalty=step_penalty
    )
    selected = forward if forward["score"] >= reverse["score"] else reverse
    alternative = reverse if selected is forward else forward
    degrees_per_bin = 360.0 / yaw_bins
    elapsed = samples[-1].elapsed_sec - samples[0].elapsed_sec
    span_deg = float(selected["advanced_bins"] * degrees_per_bin)
    score_margin_per_sample = float(
        (selected["score"] - alternative["score"]) / len(samples)
    )
    row_margins = np.sort(similarity, axis=1)[:, -1] - np.sort(similarity, axis=1)[:, -2]
    median_row_margin = float(np.median(row_margins))
    positive_steps = [step for step in selected["step_bins"] if step > 0]
    median_step = float(np.median(positive_steps)) if positive_steps else 0.0
    zero_step_fraction = float(np.mean(np.asarray(selected["step_bins"]) == 0))
    # These gates are deliberately conservative.  The cross-renderer feature
    # evidence must prefer a direction, make steady progress, and cover a full
    # turn before it can validate a 360-degree browser trajectory.
    validation_reasons: list[str] = []
    if span_deg < 330.0:
        validation_reasons.append(f"inferred span {span_deg:.1f} deg is below 330 deg")
    if score_margin_per_sample < 0.005:
        validation_reasons.append(
            f"direction margin {score_margin_per_sample:.4f} is below 0.005"
        )
    if median_row_margin < 0.002:
        validation_reasons.append(
            f"nearest-yaw median margin {median_row_margin:.4f} is below 0.002"
        )
    if zero_step_fraction > 0.6:
        validation_reasons.append(
            f"zero-step fraction {zero_step_fraction:.3f} exceeds 0.6"
        )
    estimated_rate = span_deg / elapsed if elapsed > 0 else 0.0
    return {
        "schema_version": 1,
        "status": "validated" if not validation_reasons else "insufficient_evidence",
        "validation_reasons": validation_reasons,
        "samples": str(samples_path.resolve()),
        "samples_sha256": sha256(samples_path),
        "reference_dir": str(reference_dir.resolve()),
        "reference_stride": reference_stride,
        "reference_frame_count": len(references),
        "reference_first_sha256": sha256(references[0]),
        "model": model_name,
        "processor_size": processor_size,
        "sample_count": len(samples),
        "elapsed_sec": elapsed,
        "degrees_per_reference_bin": degrees_per_bin,
        "inferred_span_deg": span_deg,
        "estimated_yaw_rate_deg_sec": estimated_rate,
        "estimated_full_turn_sec": 360.0 / estimated_rate if estimated_rate > 0 else None,
        "selected_direction": selected["direction"],
        "direction_score_margin_per_sample": score_margin_per_sample,
        "median_nearest_yaw_margin": median_row_margin,
        "median_positive_step_bins": median_step,
        "zero_step_fraction": zero_step_fraction,
        "selected_path": selected,
        "alternative_path": alternative,
        "top_matches": _top_matches(similarity),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference-stride", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--processor-size", type=int, default=280)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--step-penalty", type=float, default=0.002)
    args = parser.parse_args()
    report = align(
        args.samples.resolve(),
        args.reference_dir.resolve(),
        model_name=args.model,
        reference_stride=args.reference_stride,
        batch_size=args.batch_size,
        processor_size=args.processor_size,
        max_step=args.max_step,
        step_penalty=args.step_penalty,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
