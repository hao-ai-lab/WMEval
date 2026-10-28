#!/usr/bin/env python3
"""Generate Matrix-Game 2.0 / 3.0 controllability candidates on Modal.

The three controllability cases (BallRoller/ctrl, BeamDomino/causal,
MirrorRollingBall/lpips_collision) drive the player ball with a two-channel force.
Both Matrix-Game models take WASD keyboard input, so the force channels are
thresholded into keys the same way tools/prepare_matrix_game_inputs.py maps the
BilliardGame cue ball (forward/back -> W/S, lateral -> A/D at |value| > 0.05; the
mouse is idle because the reference camera is fixed). Whether the keys move the
ball is what Synthetic-RAFT then measures.

    python tools/run_matrix_game_ctrl_modal.py prepare             # schedules + queues under runs/matrix_game_ctrl/inputs
    python tools/run_matrix_game_ctrl_modal.py download            # checkpoints -> Modal volume (once)
    python tools/run_matrix_game_ctrl_modal.py run --model matrix_game_2 [--limit 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = ("BallRoller/ctrl", "BeamDomino/causal", "MirrorRollingBall/lpips_collision")
SEEDS = {f"run{i}": 42 + i for i in range(5)}
SOURCE_REPO = "https://github.com/SkyworkAI/Matrix-Game.git"
SOURCE_REV = "756776d516233027631008761d4d40ddaea3e23a"
MODELS = {
    "matrix_game_2": {"repo": "Skywork/Matrix-Game-2.0", "rev": "08c42d1a99e57e79df60a964bb815618cdf460f8",
                      "frames": 129, "fps": 12.0, "keyboard": 4, "latent_frames": 33,
                      "skip": ["templerun_distilled_model/*"]},
    "matrix_game_3": {"repo": "Skywork/Matrix-Game-3.0", "rev": "382aa382a03cec057360755aa72c7735025770bb",
                      "frames": 177, "fps": 17.0, "keyboard": 6, "skip": ["base_model/*"]},
}
INPUTS = ROOT / "runs/matrix_game_ctrl/inputs"
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
              "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")


# ----------------------------------------------------------------------------- schedules
def schedule(model: str, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    from evaluation.trajectories import sample_held_values
    spec = MODELS[model]
    npz = ROOT / "lite/cases" / case_id / "reference/artifacts/action.npz"
    end = float(np.load(npz)["timestamps_sec"][-1])
    values, names = sample_held_values(npz, spec["fps"], spec["frames"], end_sec=end)
    ch = {str(n): values[:, i] for i, n in enumerate(names)}
    forward = ch.get("vertical", ch.get("z_force"))
    lateral = ch.get("horizontal", ch.get("x_force"))
    keyboard = np.zeros((spec["frames"], spec["keyboard"]), np.float32)
    keyboard[:, 0] = forward > 0.05
    keyboard[:, 1] = forward < -0.05
    keyboard[:, 2] = lateral < -0.05
    keyboard[:, 3] = lateral > 0.05
    mouse = np.zeros((spec["frames"], 2), np.float32)
    return keyboard, mouse, {"kind": "generic_wasd_to_ball_hypothesis", "threshold": 0.05,
                             "channels": [str(n) for n in names], "action_end_sec": end}


def prepare() -> None:
    import yaml
    for model in MODELS:
        rows = []
        for case_id in CASES:
            kb, mouse, mapping = schedule(model, case_id)
            d = INPUTS / model / case_id
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "schedule.npz", keyboard=kb, mouse=mouse)
            case_dir = ROOT / "lite/cases" / case_id
            case = yaml.safe_load((case_dir / "case.yaml").read_text())
            # Conditioning frame: the case's first_frame file, else frame 0 of the reference video
            # (the same rule tools/odyssey_agent.py applies).
            declared = (case.get("reference") or {}).get("first_frame")
            image = case_dir / declared if declared and (case_dir / declared).is_file() else d / "first_frame.png"
            if image == d / "first_frame.png" and not image.is_file():
                import subprocess
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(case_dir / "reference/reference.mp4"),
                                "-frames:v", "1", str(image)], check=True)
            (d / "receipt.json").write_text(json.dumps({"model": model, "case_id": case_id, "mapping": mapping,
                                                        "key_frames": {k: int(kb[:, i].sum()) for i, k in enumerate("WSAD")}}, indent=1))
            for repeat, seed in SEEDS.items():
                rows.append({"model": model, "case_id": case_id, "generation_repeat": repeat, "seed": seed,
                             "schedule": str((d / "schedule.npz").relative_to(ROOT)),
                             "conditioning_image": str(image.relative_to(ROOT)),
                             "prompt": str(case.get("prompt", "")).strip()})
        (INPUTS / model / "queue.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(model, len(rows), "jobs ->", INPUTS / model / "queue.jsonl")


# ----------------------------------------------------------------------------- Modal
weights = modal.Volume.from_name("wmeval-matrix-game-weights", create_if_missing=True)
base = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11").apt_install(
    "git", "ffmpeg", "libgl1", "libglib2.0-0")
mg2_image = (base
             .pip_install("torch==2.5.1", "torchvision==0.20.1")
             .pip_install(FLASH_ATTN)
             .pip_install("opencv-python>=4.9.0.80", "diffusers==0.31.0", "transformers==4.49.0", "tokenizers>=0.20.3",
                          "accelerate>=1.1.1", "tqdm", "imageio", "easydict", "ftfy", "imageio-ffmpeg", "numpy<2", "omegaconf",
                          "einops", "av", "safetensors", "open_clip_torch", "sentencepiece", "scikit-image",
                          "huggingface_hub", "hf_transfer", "git+https://github.com/openai/CLIP.git")
             .run_commands(f"git clone {SOURCE_REPO} /src && cd /src && git checkout {SOURCE_REV}")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
             .add_local_file(str(ROOT / "tools/model_adapters/matrix_game2_deterministic.py"), "/adapters/mg2.py"))
mg3_image = (base
             .pip_install("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0")
             .pip_install("opencv-python==4.10.0.84", "diffusers>=0.36.0", "transformers==4.57.3", "tokenizers>=0.20.3",
                          "accelerate>=1.1.1", "tqdm", "imageio[ffmpeg]", "easydict", "ftfy", "imageio-ffmpeg", "numpy==2.2.6",
                          "scipy", "einops", "pandas", "decord", "tensorboard", "trimesh", "safetensors==0.7.0", "huggingface_hub", "hf_transfer")
             .run_commands(f"git clone {SOURCE_REPO} /src && cd /src && git checkout {SOURCE_REV}")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
             .add_local_file(str(ROOT / "tools/model_adapters/matrix_game3_deterministic.py"), "/adapters/mg3.py"))
app = modal.App("wmeval-matrix-game-ctrl")


@app.function(image=mg2_image, volumes={"/weights": weights}, timeout=3 * 3600, cpu=8, memory=32768)
def download(model: str) -> str:
    from huggingface_hub import snapshot_download
    spec = MODELS[model]
    dest = f"/weights/{model}"
    snapshot_download(spec["repo"], revision=spec["rev"], local_dir=dest, ignore_patterns=spec["skip"])
    weights.commit()
    return f"{model}: {sum(p.stat().st_size for p in Path(dest).rglob('*') if p.is_file()) / 1e9:.1f} GB"


def _run(cmd: list[str], cwd: str, out: Path, video_name: str, t0: float) -> dict:
    import subprocess
    import time
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": cwd})
    log = (proc.stdout + proc.stderr)[-6000:]
    videos = sorted(out.rglob(video_name)) or sorted(p for p in out.rglob("*.mp4") if "icon" not in p.name)
    if proc.returncode != 0 or not videos:
        return {"ok": False, "seconds": time.time() - t0, "log": log}
    return {"ok": True, "seconds": time.time() - t0, "video": videos[0].read_bytes(), "log": log[-1500:]}


@app.function(image=mg2_image, volumes={"/weights": weights}, gpu="L40S", timeout=3600, memory=65536, max_containers=8)
def generate_mg2(job: dict) -> dict:
    import time
    work = Path("/tmp/job"); out = work / "out"; out.mkdir(parents=True, exist_ok=True)
    (work / "image.png").write_bytes(job["image"]); (work / "schedule.npz").write_bytes(job["schedule"])
    ck = "/weights/matrix_game_2"
    cmd = ["python", "/adapters/mg2.py", "--source", "/src/Matrix-Game-2", "--schedule", str(work / "schedule.npz"),
           "--config", "/src/Matrix-Game-2/configs/inference_yaml/inference_universal.yaml",
           "--checkpoint", f"{ck}/base_distilled_model/base_distill.safetensors", "--pretrained-model", ck,
           "--image", str(work / "image.png"), "--output", str(out), "--latent-frames", str(MODELS["matrix_game_2"]["latent_frames"]),
           "--seed", str(job["seed"])]
    return _run(cmd, "/src/Matrix-Game-2", out, "demo.mp4", time.time())


@app.function(image=mg3_image, volumes={"/weights": weights}, gpu="H100", timeout=3600, memory=131072, max_containers=8)
def generate_mg3(job: dict) -> dict:
    import time
    work = Path("/tmp/job"); out = work / "out"; out.mkdir(parents=True, exist_ok=True)
    (work / "image.png").write_bytes(job["image"]); (work / "schedule.npz").write_bytes(job["schedule"])
    cmd = ["python", "/adapters/mg3.py", "--source", "/src/Matrix-Game-3", "--schedule", str(work / "schedule.npz"),
           "--checkpoint", "/weights/matrix_game_3", "--image", str(work / "image.png"), "--prompt", job["prompt"] or " ",
           "--output", str(out), "--seed", str(job["seed"])]
    return _run(cmd, "/src/Matrix-Game-3", out, "candidate.mp4", time.time())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("--model", choices=sorted(MODELS), required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--case", action="append", default=[])
    r.add_argument("--out", type=Path, default=ROOT / "runs/matrix_game_ctrl")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare()
        return 0
    with modal.enable_output(), app.run():
        if args.cmd == "download":
            for line in download.map(list(MODELS)):
                print(line)
            return 0
        rows = [json.loads(l) for l in (INPUTS / args.model / "queue.jsonl").read_text().splitlines() if l.strip()]
        rows = [x for x in rows if not args.case or x["case_id"] in args.case][: args.limit]
        jobs = [{"image": (ROOT / x["conditioning_image"]).read_bytes(), "schedule": (ROOT / x["schedule"]).read_bytes(),
                 "prompt": x["prompt"], "seed": x["seed"]} for x in rows]
        fn = generate_mg2 if args.model == "matrix_game_2" else generate_mg3
        for row, res in zip(rows, fn.map(jobs, return_exceptions=True)):
            dest = args.out / args.model / row["case_id"] / row["generation_repeat"]
            dest.mkdir(parents=True, exist_ok=True)
            if isinstance(res, Exception) or not res.get("ok"):
                (dest / "error.log").write_text(str(res) if isinstance(res, Exception) else res["log"])
                print(f"FAILED {args.model} {row['case_id']} {row['generation_repeat']} -> {dest / 'error.log'}")
                continue
            (dest / "candidate.mp4").write_bytes(res["video"])
            (dest / "generation_receipt.json").write_text(json.dumps({
                "model": args.model, "weights": f"{MODELS[args.model]['repo']}@{MODELS[args.model]['rev']}",
                "source": f"{SOURCE_REPO}@{SOURCE_REV}", "case_id": row["case_id"], "seed": row["seed"],
                "schedule": row["schedule"], "seconds": round(res["seconds"], 1)}, indent=1))
            print(f"ok {args.model} {row['case_id']} {row['generation_repeat']} {res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
