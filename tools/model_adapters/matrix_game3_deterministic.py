#!/usr/bin/env python3
"""Inject a deterministic per-frame action schedule into Matrix-Game 3."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


class _ScheduleCursor:
    def __init__(self, keyboard: np.ndarray, mouse: np.ndarray):
        self.keyboard = keyboard
        self.mouse = mouse
        self.offset = 0

    def actions(self) -> dict[str, "_ScheduleSlice"]:
        return {
            "keyboard": _ScheduleSlice(self, "keyboard", advance=False),
            "mouse": _ScheduleSlice(self, "mouse", advance=True),
        }


class _ScheduleSlice:
    def __init__(self, cursor: _ScheduleCursor, name: str, advance: bool):
        self.cursor = cursor
        self.name = name
        self.advance = advance

    def repeat(self, frames: int, columns: int) -> torch.Tensor:
        if columns != 1:
            raise ValueError(f"unexpected repeat columns: {columns}")
        start = self.cursor.offset
        end = start + frames
        array = getattr(self.cursor, self.name)
        if end > len(array):
            raise ValueError(f"upstream requested actions through frame {end}; schedule has {len(array)}")
        result = torch.from_numpy(array[start:end].copy())
        if self.advance:
            self.cursor.offset = end
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    schedule = np.load(args.schedule, allow_pickle=False)
    keyboard = np.asarray(schedule["keyboard"], dtype=np.float32)
    mouse = np.asarray(schedule["mouse"], dtype=np.float32)
    if keyboard.shape != (177, 6) or mouse.shape != (177, 2):
        raise ValueError(f"Matrix-Game 3 schedule must be (177,6)/(177,2), got {keyboard.shape}/{mouse.shape}")
    cursor = _ScheduleCursor(keyboard, mouse)

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import pipeline.inference_interactive_pipeline as interactive  # noqa: PLC0415
    interactive.get_current_action = cursor.actions
    from diffusers.utils import export_to_video  # noqa: PLC0415

    def process_video_without_ui(input_video, output_video, config, mouse_icon_path, **kwargs):
        del config, mouse_icon_path, kwargs
        export_to_video([frame / 255.0 for frame in input_video], output_video, fps=17)

    interactive.process_video = process_video_without_ui
    # Upstream exposes --fa_version 0 but WanSelfAttention calls the raw
    # flash-only helper directly. Route that imported symbol through the
    # module's documented SDPA-aware dispatcher.
    import wan.modules.attention as attention_module  # noqa: PLC0415
    import wan.modules.model as model_module  # noqa: PLC0415
    model_module.flash_attention = attention_module.attention
    import generate as upstream  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    sys.argv = [
        str(source / "generate.py"),
        "--size", "704*1280",
        "--ckpt_dir", str(args.checkpoint.resolve()),
        "--output_dir", str(args.output.resolve()),
        "--fa_version", "0",
        "--use_int8",
        "--num_iterations", "4",
        "--num_inference_steps", "3",
        "--image", str(args.image.resolve()),
        "--prompt", args.prompt,
        "--save_name", "candidate",
        "--seed", str(args.seed),
        "--interactive",
        "--lightvae_pruning_rate", "0.5",
        "--vae_type", "mg_lightvae",
        "--ulysses_size", str(world_size),
    ]
    if world_size > 1:
        sys.argv.extend(["--dit_fsdp", "--t5_fsdp"])
    parsed = upstream._parse_args()
    upstream.generate(parsed)
    if cursor.offset != len(keyboard):
        raise ValueError(f"upstream consumed {cursor.offset} of {len(keyboard)} scheduled actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
