#!/usr/bin/env python3
"""Supply the canonical orbit to pinned MG3's native camera conditioning."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--conditioning', type=Path, required=True)
    args, native_args = parser.parse_known_args()
    data = np.load(args.conditioning, allow_pickle=False)
    poses, keyboard, mouse = (data[name] for name in ('poses', 'keyboard', 'mouse'))
    if poses.shape != (177, 4, 4) or keyboard.shape != (177, 6) or mouse.shape != (177, 2):
        raise ValueError('requires the complete 177-frame native orbit conditioning')
    seed_k = data['seed_intrinsics']
    sys.path.insert(0, str(args.source.resolve()))
    os.chdir(args.source)
    import pipeline.inference_pipeline as pipeline
    import generate as upstream

    def get_data(frames, height, width, image, device=None, dtype=None):
        if frames != 177 or image.size != (1024, 1024):
            raise ValueError('native clip count or square conditioning image changed')
        # The native latent canvas is rounded down to a multiple of 32.
        side = int(np.sqrt(height * width) // 32 * 32)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255
        pixels = functional.interpolate(pixels, size=(side, side), mode='bilinear',
                                        align_corners=False, antialias=True)
        pixels = (2 * pixels - 1).unsqueeze(2).to(device=device, dtype=dtype)
        return (pixels, torch.from_numpy(poses.copy()),
                torch.from_numpy(keyboard.copy())[None].to(device=device, dtype=dtype),
                torch.from_numpy(mouse.copy())[None].to(device=device, dtype=dtype))

    def get_intrinsics(height, width):
        if height != width:
            raise ValueError('square seed must keep its aspect ratio')
        return torch.tensor([seed_k[0, 0] * width / 1024, seed_k[1, 1] * height / 1024,
                             seed_k[0, 2] * width / 1024, seed_k[1, 2] * height / 1024])

    def process_video(video, output, controls, mouse_icon, **kwargs):
        from diffusers.utils import export_to_video
        export_to_video([frame.astype(np.float32) / 255 for frame in video], output, fps=17)

    pipeline.get_data = get_data
    pipeline.get_intrinsics = get_intrinsics
    pipeline.process_video = process_video
    sys.argv = [str(args.source / 'generate.py'), *native_args]
    parsed = upstream._parse_args()
    if parsed.interactive or parsed.num_iterations != 4:
        raise ValueError('requires the native pose pipeline with four iterations')
    if parsed.use_int8 or parsed.num_inference_steps != 3 or parsed.use_base_model:
        raise ValueError('requires the unquantized three-step distilled model')
    upstream.generate(parsed)


if __name__ == '__main__':
    main()
