#!/usr/bin/env python3
"""Generate LingBot-World v2 (LingBot-World-Infinity) candidates on Modal.

LingBot-World v2 is image + prompt + camera-pose conditioned, like LingBot-World
Base Cam, so it reuses the Base Cam inputs from ``tools/generation/prepare_lingbot_table7_inputs.py``
(161-frame OpenCV camera-to-world poses and per-frame intrinsics). v2 reads its
intrinsics as given for a 480x832 image, so intrinsics written for the 1280x720
reference frame (cx ~ 640) are rescaled before upload; 480p ones are passed as is.

    modal profile activate hao-ai-lab
    python tools/generation/run_lingbot_v2_modal.py download              # weights -> Modal volume (once, ~86 GB)
    python tools/generation/run_lingbot_v2_modal.py run --queue runs/lingbot_world_v2/inputs/generation_queue.jsonl \\
        --out runs/lingbot_world_v2 --limit 1                   # feasibility: one video
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
APP = "wmeval-lingbot-v2"
REPO = "https://github.com/robbyant/lingbot-world-v2.git"
REPO_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
WEIGHTS_REPO = "robbyant/lingbot-world-v2-14b-causal-fast"
WEIGHTS_DIR = "/weights/lingbot-world-v2-14b-causal-fast"
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
              "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
FRAME_NUM = 161               # 10 s at the Base Cam 16 fps timeline (4n + 1)
SIZE = "480*832"
REF_W, REF_H, V2_W, V2_H = 1280, 720, 832, 480

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1")
    .pip_install(FLASH_ATTN)
    .pip_install("opencv-python>=4.9.0.80", "diffusers>=0.31.0", "transformers>=4.49.0,<=4.51.3", "tokenizers>=0.20.3",
                 "accelerate>=1.1.1", "tqdm", "imageio[ffmpeg]", "easydict", "ftfy", "imageio-ffmpeg",
                 "numpy>=1.23.5,<2", "scipy", "huggingface_hub", "hf_transfer")
    .run_commands(f"git clone {REPO} /lingbot && cd /lingbot && git checkout {REPO_COMMIT}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
weights = modal.Volume.from_name("wmeval-lingbot-v2-weights", create_if_missing=True)
app = modal.App(APP)


@app.function(image=image, volumes={"/weights": weights}, timeout=3 * 3600, cpu=8, memory=32768)
def download() -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(WEIGHTS_REPO, local_dir=WEIGHTS_DIR)
    weights.commit()
    total = sum(p.stat().st_size for p in Path(WEIGHTS_DIR).rglob("*") if p.is_file())
    return f"{WEIGHTS_REPO}: {total / 1e9:.1f} GB in {WEIGHTS_DIR}"


@app.function(image=image, volumes={"/weights": weights}, gpu="H100", timeout=3600, memory=131072, max_containers=8)
def generate(job: dict) -> dict:
    import io
    import subprocess
    import time
    import numpy as np

    work = Path("/tmp/job"); work.mkdir(parents=True, exist_ok=True)
    (work / "image.png").write_bytes(job["image"])
    np.save(work / "poses.npy", np.load(io.BytesIO(job["poses"])))
    np.save(work / "intrinsics.npy", np.load(io.BytesIO(job["intrinsics"])))
    out = work / "video.mp4"
    cmd = ["python", "generate.py", "--task", "i2v-A14B", "--size", SIZE, "--ckpt_dir", WEIGHTS_DIR,
           "--image", str(work / "image.png"), "--action_path", str(work), "--frame_num", str(FRAME_NUM),
           "--prompt", job["prompt"], "--base_seed", str(job["seed"]), "--local_attn_size", "18", "--sink_size", "6",
           "--convert_model_dtype", "--save_file", str(out)]
    if job.get("offload"):
        cmd += ["--offload_model", "True"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/lingbot", capture_output=True, text=True)
    log = (proc.stdout + proc.stderr)[-6000:]
    if proc.returncode != 0 or not out.is_file():
        return {"ok": False, "seconds": time.time() - t0, "log": log}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"ok": True, "seconds": time.time() - t0, "video": out.read_bytes(), "log": log[-2000:], "gpu_mem": smi.strip()}


def _job(row: dict, seed: int) -> dict:
    import io
    import numpy as np
    K = np.load(ROOT / row["intrinsics"]).astype(np.float32)
    if K[0, 2] > V2_W / 2 + 50:           # given for the 1280x720 reference frame (cx ~ 640): rescale to 832x480
        K[:, [0, 2]] *= V2_W / REF_W
        K[:, [1, 3]] *= V2_H / REF_H
    buf_k, buf_p = io.BytesIO(), io.BytesIO()
    np.save(buf_k, K); np.save(buf_p, np.load(ROOT / row["poses"]).astype(np.float32))
    prompt = " ".join(x for x in (row.get("environment_prompt", ""), row.get("character_prompt", "")) if x).strip()
    return {"image": (ROOT / row["conditioning_image"]).read_bytes(), "poses": buf_p.getvalue(),
            "intrinsics": buf_k.getvalue(), "prompt": prompt, "seed": seed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("--queue", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--case", action="append", default=[], help="only these case_ids (repeatable)")
    r.add_argument("--offload", action="store_true")
    args = ap.parse_args(argv)
    with modal.enable_output(), app.run():
        if args.cmd == "download":
            print(download.remote())
            return 0
        rows = [json.loads(l) for l in args.queue.read_text().splitlines() if l.strip()]
        rows = [x for x in rows if not args.case or x["case_id"] in args.case][: args.limit]
        jobs = []
        for row in rows:
            seed = int(row.get("base_seed", 42 + int(str(row["generation_repeat"]).removeprefix("run"))))
            jobs.append((row, {**_job(row, seed), "offload": args.offload}))
        for (row, _), res in zip(jobs, generate.map([j for _, j in jobs], return_exceptions=True)):
            dest = args.out / row["case_id"] / row["generation_repeat"]
            dest.mkdir(parents=True, exist_ok=True)
            if isinstance(res, Exception) or not res.get("ok"):
                (dest / "error.log").write_text(str(res) if isinstance(res, Exception) else res["log"])
                print(f"FAILED {row['case_id']} {row['generation_repeat']} -> {dest / 'error.log'}")
                continue
            (dest / "candidate.mp4").write_bytes(res["video"])
            (dest / "generation_receipt.json").write_text(json.dumps({
                "model": "lingbot_world_v2", "weights": WEIGHTS_REPO, "code": f"{REPO}@{REPO_COMMIT}",
                "size": SIZE, "frame_num": FRAME_NUM, "seed": int(row.get("base_seed", 42 + int(str(row['generation_repeat']).removeprefix('run')))),
                "case_id": row["case_id"], "seconds": round(res["seconds"], 1), "gpu_mem": res.get("gpu_mem")}, indent=1))
            print(f"ok {row['case_id']} {row['generation_repeat']}  {res['seconds']:.0f}s  gpu {res.get('gpu_mem')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
