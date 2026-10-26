#!/usr/bin/env python3
"""Generate LingBot-World Base (Act) controllability candidates on Modal.

The Act checkpoint (``robbyant/lingbot-world-base-act-preview``) takes image +
prompt + per-frame WASD (``action.npy``, columns w/a/s/d) plus camera poses whose
rotation sets the view. The three controllability cases drive the player ball with
a two-channel force; it is thresholded into WASD exactly as for Matrix-Game
(tools/run_matrix_game_ctrl_modal.py: forward/back -> W/S, lateral -> A/D at
|value| > 0.05), and the camera is held fixed (identity poses) because the Unity
reference camera is fixed. Conditioning frames are the same first frames the
Matrix-Game and Genie control runs use. Upstream draws a keyboard overlay onto the
video by default; tools/model_adapters/lingbot_no_ui.py disables it.

    modal profile activate hao-ai-lab
    python tools/run_lingbot_act_modal.py download          # weights -> Modal volume (once, ~75 GB)
    python tools/run_lingbot_act_modal.py run --limit 1     # feasibility: one video
    python tools/run_lingbot_act_modal.py run               # 3 cases x run0-4
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import modal
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP = "wmeval-lingbot-act"
REPO = "https://github.com/robbyant/lingbot-world.git"
REPO_COMMIT = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
WEIGHTS_REPO = "robbyant/lingbot-world-base-act-preview"
WEIGHTS_REV = "8eb7b95a69cd4c5b61ebb47840f697d2e153ceb7"
WEIGHTS_DIR = "/weights/lingbot-world-base-act-preview"   # upstream selects Act mode when 'act' is in the path
CASES = ("BallRoller/ctrl", "BeamDomino/causal", "MirrorRollingBall/lpips_collision")
SEEDS = {f"run{i}": 42 + i for i in range(5)}
FRAME_NUM = 161               # 10 s on the 16 fps container timeline (4n + 1)
FPS = 16.0
SIZE = "480*832"
FIRST_FRAMES = ROOT / "runs/matrix_game_ctrl/inputs/matrix_game_2"
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
              "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1")
    .pip_install(FLASH_ATTN)
    .pip_install("opencv-python>=4.9.0.80", "diffusers>=0.31.0", "transformers>=4.49.0,<=4.51.3", "tokenizers>=0.20.3",
                 "accelerate>=1.1.1", "tqdm", "imageio[ffmpeg]", "easydict", "ftfy", "imageio-ffmpeg",
                 "numpy>=1.23.5,<2", "scipy", "huggingface_hub", "hf_transfer")
    .run_commands(f"git clone {REPO} /lingbot && cd /lingbot && git checkout {REPO_COMMIT}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "LINGBOT_SOURCE": "/lingbot"})
    .add_local_file(str(ROOT / "tools/model_adapters/lingbot_no_ui.py"), "/adapters/lingbot_no_ui.py")
)
weights = modal.Volume.from_name("wmeval-lingbot-act-weights", create_if_missing=True)
app = modal.App(APP)


def schedule(case_id: str) -> tuple[np.ndarray, dict]:
    from evaluation.trajectories import sample_held_values
    npz = ROOT / "eval_sets/wmeval_lite_v1/cases" / case_id / "reference/artifacts/action.npz"
    end = float(np.load(npz)["timestamps_sec"][-1])
    values, names = sample_held_values(npz, FPS, FRAME_NUM, end_sec=end)
    ch = {str(n): values[:, i] for i, n in enumerate(names)}
    forward = ch.get("vertical", ch.get("z_force"))
    lateral = ch.get("horizontal", ch.get("x_force"))
    wasd = np.zeros((FRAME_NUM, 4), np.int32)
    wasd[:, 0] = forward > 0.05
    wasd[:, 1] = lateral < -0.05
    wasd[:, 2] = forward < -0.05
    wasd[:, 3] = lateral > 0.05
    return wasd, {"kind": "generic_wasd_to_ball_hypothesis", "threshold": 0.05, "columns": "wasd",
                  "channels": [str(n) for n in names], "action_end_sec": end,
                  "key_frames": {k: int(wasd[:, i].sum()) for i, k in enumerate("WASD")}}


def intrinsics() -> np.ndarray:
    # The upstream 480x832 examples' pinhole (fx, fy, cx, cy); only the fixed-camera rays depend on it.
    return np.tile(np.array([502.91, 503.11, 415.78, 239.78], np.float32), (FRAME_NUM, 1))


@app.function(image=image, volumes={"/weights": weights}, timeout=3 * 3600, cpu=8, memory=32768)
def download() -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REV, local_dir=WEIGHTS_DIR, ignore_patterns=["assets/*"])
    weights.commit()
    total = sum(p.stat().st_size for p in Path(WEIGHTS_DIR).rglob("*") if p.is_file())
    return f"{WEIGHTS_REPO}@{WEIGHTS_REV}: {total / 1e9:.1f} GB in {WEIGHTS_DIR}"


@app.function(image=image, volumes={"/weights": weights}, gpu="H100", timeout=3 * 3600, memory=196608, max_containers=15)
def generate(job: dict) -> dict:
    import subprocess
    import time

    work = Path("/tmp/job"); work.mkdir(parents=True, exist_ok=True)
    (work / "image.png").write_bytes(job["image"])
    np.save(work / "action.npy", np.load(io.BytesIO(job["action"])))
    np.save(work / "poses.npy", np.tile(np.eye(4, dtype=np.float32), (FRAME_NUM, 1, 1)))
    np.save(work / "intrinsics.npy", np.load(io.BytesIO(job["intrinsics"])))
    out = work / "video.mp4"
    cmd = ["python", "/adapters/lingbot_no_ui.py", "--task", "i2v-A14B", "--size", SIZE, "--ckpt_dir", WEIGHTS_DIR,
           "--image", str(work / "image.png"), "--action_path", str(work), "--frame_num", str(FRAME_NUM),
           "--prompt", job["prompt"], "--base_seed", str(job["seed"]), "--offload_model", "True",
           "--convert_model_dtype", "--save_file", str(out)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/lingbot", capture_output=True, text=True)
    log = (proc.stdout + proc.stderr)[-6000:]
    if proc.returncode != 0 or not out.is_file():
        return {"ok": False, "seconds": time.time() - t0, "log": log}
    return {"ok": True, "seconds": time.time() - t0, "video": out.read_bytes(), "log": log[-2000:]}


def main(argv: list[str] | None = None) -> int:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=ROOT / "runs/lingbot_world_act_ctrl")
    r.add_argument("--limit", type=int)
    r.add_argument("--case", action="append", default=[], help="only these case_ids (repeatable)")
    args = ap.parse_args(argv)
    with modal.enable_output(), app.run():
        if args.cmd == "download":
            print(download.remote())
            return 0
        jobs = []
        for case_id in CASES:
            if args.case and case_id not in args.case:
                continue
            wasd, mapping = schedule(case_id)
            buf_a, buf_k = io.BytesIO(), io.BytesIO()
            np.save(buf_a, wasd); np.save(buf_k, intrinsics())
            case = yaml.safe_load((ROOT / "eval_sets/wmeval_lite_v1/cases" / case_id / "case.yaml").read_text())
            frame = FIRST_FRAMES / case_id / "first_frame.png"
            for repeat, seed in SEEDS.items():
                if (args.out / case_id / repeat / "candidate.mp4").is_file():
                    continue
                jobs.append(((case_id, repeat, seed, mapping),
                             {"image": frame.read_bytes(), "action": buf_a.getvalue(), "intrinsics": buf_k.getvalue(),
                              "prompt": " ".join(str(case.get("prompt", "")).split()), "seed": seed}))
        jobs = jobs[: args.limit]
        print(f"{len(jobs)} jobs")
        for (case_id, repeat, seed, mapping), res in zip([m for m, _ in jobs],
                                                         generate.map([j for _, j in jobs], return_exceptions=True)):
            dest = args.out / case_id / repeat
            dest.mkdir(parents=True, exist_ok=True)
            if isinstance(res, Exception) or not res.get("ok"):
                (dest / "error.log").write_text(str(res) if isinstance(res, Exception) else res["log"])
                print(f"FAILED {case_id} {repeat} -> {dest / 'error.log'}", flush=True)
                continue
            (dest / "candidate.mp4").write_bytes(res["video"])
            (dest / "generation_receipt.json").write_text(json.dumps({
                "model": "lingbot_world_act", "weights": f"{WEIGHTS_REPO}@{WEIGHTS_REV}", "code": f"{REPO}@{REPO_COMMIT}",
                "size": SIZE, "frame_num": FRAME_NUM, "fps": FPS, "seed": seed, "case_id": case_id,
                "camera": "fixed (identity poses)", "mapping": mapping, "seconds": round(res["seconds"], 1)}, indent=1))
            print(f"ok {case_id} {repeat}  {res['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
