#!/usr/bin/env python3
"""Make a timestamped contact sheet and metadata for a local diagnostic video."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def review(video: Path, output: Path, count: int = 8) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frames <= 0:
        raise ValueError(f"unreadable video: {video}")
    output.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, frames-1, count).round().astype(int)
    fig, axes = plt.subplots(2, (count+1)//2, figsize=(14, 7), squeeze=False,
                             layout="constrained")
    for ax in axes.flat: ax.axis('off')
    for ax, i in zip(axes.flat, indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: raise ValueError(f"failed to decode frame {i}")
        ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        ax.set_title(f"{i/fps:.2f} s | frame {i}")
    cap.release()
    fig.suptitle(video.stem + " | observed model output", fontsize=15)
    sheet = output / (video.stem + "_contact_sheet.png")
    fig.savefig(sheet,dpi=140); plt.close(fig)
    result = {"video":str(video.resolve()), "sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
              "fps":fps,"frames":frames,"duration_sec":frames/fps,"width":width,"height":height,
              "sampled_frame_indices":indices.tolist(),"contact_sheet":str(sheet.resolve())}
    (output/(video.stem+"_metadata.json")).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(review(args.video,args.output),indent=2))
