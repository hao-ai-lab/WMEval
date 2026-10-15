#!/usr/bin/env python3
"""Run one bounded LingBot pose-orbit diagnostic using the existing Modal cache.

Prepare inputs with tools/prepare_lighting_orbit.py, then:
  WMEVAL_ORBIT_INPUT=/absolute/inputs modal run tools/run_lighting_orbit_modal.py
This never overwrites the paper's videos or evaluator case locks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get("WMEVAL_ORBIT_INPUT", ROOT / "runs/lighting_orbit/inputs"))
SOURCE_REVISION = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
app = modal.App("wmeval-lighting-orbit-diagnostic")
cache = modal.Volume.from_name("wm-hf-cache", create_if_missing=False)
outputs = modal.Volume.from_name("WMEval-lite-candidates", create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                 extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84", "scipy==1.15.3",
                 "diffusers==0.33.1", "transformers==4.51.3", "tokenizers==0.21.4",
                 "huggingface-hub==0.36.0", "accelerate==1.12.0", "imageio[ffmpeg]==2.37.0",
                 "easydict", "ftfy", "einops", "tqdm", "PyYAML")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
    .run_commands("git clone https://github.com/Robbyant/lingbot-world.git /workspace/lingbot-world",
                  f"cd /workspace/lingbot-world && git checkout {SOURCE_REVISION}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_CACHE": "/cache/hf", "PYTHONUNBUFFERED": "1",
          "LINGBOT_SOURCE": "/workspace/lingbot-world"})
    .add_local_dir(str(INPUT), "/workspace/orbit", copy=True)
    .add_local_file(str(ROOT / "tools/model_adapters/lingbot_no_ui.py"), "/workspace/lingbot_no_ui.py", copy=True)
    .add_local_file(str(ROOT / "eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting/reference/ref.png"), "/workspace/ref.png", copy=True)
    .add_local_file(str(ROOT / "eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting/case.yaml"), "/workspace/case.yaml", copy=True)
)


@app.function(image=image, gpu="H100:8", timeout=1800,
              volumes={"/cache/hf": cache, "/outputs": outputs})
def generate(seed: int, run_id: str) -> dict:
    import hashlib
    import subprocess
    import time
    import yaml
    import numpy as np

    root = Path("/outputs/lighting_orbit_diagnostics") / run_id
    if root.exists():
        raise FileExistsError(f"diagnostic already exists: {root}")
    ckpt = Path("/cache/hf/lingbot-world-base-cam")
    required = ("configuration.json", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                "low_noise_model/config.json", "high_noise_model/config.json")
    missing = [name for name in required if not (ckpt / name).exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint files missing at {ckpt}: {missing}")
    root.mkdir(parents=True)
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
    print(hardware, flush=True)
    prompt = yaml.safe_load(Path("/workspace/case.yaml").read_text())["prompt"]
    trajectory = json.loads(Path("/workspace/orbit/trajectory.json").read_text())
    degrees = trajectory["parameters"]["degrees"]
    if degrees != 360:
        prompt = prompt.replace("completing one full loop over the duration of the clip", f"covering an arc of {degrees:g} degrees over the duration of the clip")
    count = len(np.load("/workspace/orbit/poses.npy"))
    if count % 4 != 1:
        raise ValueError("LingBot requires 4n+1 frames")
    video = root / "video.mp4"
    command = ["torchrun", "--nproc_per_node=8", "/workspace/lingbot_no_ui.py",
               "--task", "i2v-A14B", "--size", "480*832", "--ckpt_dir", str(ckpt),
               "--image", "/workspace/ref.png", "--action_path", "/workspace/orbit",
               "--dit_fsdp", "--t5_fsdp", "--ulysses_size", "8", "--frame_num", str(count),
               "--sample_steps", "20", "--base_seed", str(seed), "--prompt", prompt,
               "--save_file", str(video)]
    receipt = {"status": "running", "seed": seed, "model": "robbyant/lingbot-world-base-cam",
               "source_revision": SOURCE_REVISION, "trajectory": trajectory,
               "hardware": hardware, "command": command,
               "checkpoint_provenance": "existing wm-hf-cache directory; HF revision unrecorded",
               "checkpoint_config_sha256": {name: hashlib.sha256((ckpt / name).read_bytes()).hexdigest()
                                             for name in required if name.endswith('.json')},
               "conditioning_sha256": hashlib.sha256(Path('/workspace/ref.png').read_bytes()).hexdigest()}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    outputs.commit()
    started = time.monotonic()
    with (root / "generation.log").open("w") as log:
        process = subprocess.Popen(command, cwd="/workspace/lingbot-world", stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        rc = process.wait()
    receipt.update(returncode=rc, elapsed_sec=time.monotonic()-started,
                   status="completed" if rc == 0 and video.exists() else "failed")
    if video.exists():
        receipt["video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    outputs.commit()
    if receipt["status"] != "completed":
        raise RuntimeError(f"generation failed; see {root}/generation.log")
    return receipt


@app.local_entrypoint()
def main(seed: int = 42, run_id: str = "lingbot_orbit360_20261010_seed42"):
    result = generate.remote(seed, run_id)
    print(json.dumps(result, indent=2))
