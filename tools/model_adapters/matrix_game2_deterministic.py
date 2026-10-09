#!/usr/bin/env python3
"""Inject a deterministic per-frame action schedule into Matrix-Game 2."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    schedule = np.load(args.schedule, allow_pickle=False)
    keyboard = np.asarray(schedule["keyboard"], dtype=np.float32)
    mouse = np.asarray(schedule["mouse"], dtype=np.float32)
    expected_frames = (args.latent_frames - 1) * 4 + 1
    if keyboard.shape != (expected_frames, 4) or mouse.shape != (expected_frames, 2):
        raise ValueError(
            f"schedule must be keyboard=({expected_frames},4), mouse=({expected_frames},2); "
            f"got {keyboard.shape} and {mouse.shape}"
        )
    if args.latent_frames % 3:
        raise ValueError("Matrix-Game 2 latent frames must be divisible by its 3-frame block size")

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import inference as upstream  # noqa: PLC0415

    def deterministic_schedule(num_frames: int, num_samples_per_action: int = 4):
        del num_samples_per_action
        if num_frames != expected_frames:
            raise ValueError(f"upstream requested {num_frames} actions; expected {expected_frames}")
        return {
            "keyboard_condition": torch.from_numpy(keyboard.copy()),
            "mouse_condition": torch.from_numpy(mouse.copy()),
        }

    upstream.Bench_actions_universal = deterministic_schedule
    args.output.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(source / "inference.py"),
        "--config_path", str(args.config.resolve()),
        "--checkpoint_path", str(args.checkpoint.resolve()),
        "--img_path", str(args.image.resolve()),
        "--output_folder", str(args.output.resolve()),
        "--num_output_frames", str(args.latent_frames),
        "--seed", str(args.seed),
        "--pretrained_model_path", str(args.pretrained_model.resolve()),
    ]
    upstream.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
