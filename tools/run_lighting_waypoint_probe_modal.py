#!/usr/bin/env python3
"""Small matched-seed Waypoint probes for lighting camera control.

These are diagnostic controls, not calibrated metric-space orbits. Keep
paper videos intact and inspect the paired translation/yaw response first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from tools.generate_waypoint15_candidates_modal import waypoint_image
else:
    # The runner module is imported again in the container. Its dependencies
    # are already installed; do not import the local-only image recipe there.
    waypoint_image = modal.Image.debian_slim()

app = modal.App("wmeval-lighting-waypoint-probes")
SOURCE_REVISION = "b3f1e725b222679a517632918cc78bba0c9fa433"
cache = modal.Volume.from_name("wm-hf-cache", create_if_missing=False)
outputs = modal.Volume.from_name("WMEval-lite-candidates", create_if_missing=False)
image = waypoint_image.pip_install(
    f"world_engine @ git+https://github.com/Overworldai/world_engine.git@{SOURCE_REVISION}",
    extra_options="--no-deps --force-reinstall",
)


@app.function(image=image, gpu="H100", timeout=1800,
              volumes={"/cache/hf": cache, "/outputs": outputs},
              secrets=[modal.Secret.from_name("huggingface")])
def generate(seed: int, run_id: str, steps: int, mode: str, mouse_x: float,
             strafe_duty: float) -> dict:
    import hashlib
    import importlib.metadata
    import subprocess
    import time
    import cv2
    import numpy as np
    import torch
    from world_engine import WorldEngine, CtrlInput

    source = importlib.metadata.distribution("world_engine").read_text("direct_url.json")
    if json.loads(source)["vcs_info"]["commit_id"] != SOURCE_REVISION:
        raise ValueError("world_engine source revision drifted")

    root = Path("/outputs/lighting_orbit_diagnostics") / run_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    if mode == "probes":
        programs = [("yaw_only", False, -.6), ("strafe_only", True, 0.), ("strafe_yaw", True, -.6)]
    elif mode == "combined":
        programs = [("combined", True, mouse_x)]
    elif mode == "yaw":
        programs = [("yaw", False, mouse_x)]
    elif mode == "yaw_sweep":
        programs = [("yaw_sweep", False, None)]
    elif mode == "compare_fallbacks":
        programs = [("combined_duty", True, mouse_x), ("yaw_sweep", False, None)]
    else:
        raise ValueError(mode)
    if not 0 <= strafe_duty <= 1 or not 1 <= steps <= 150:
        raise ValueError("invalid steps or duty")
    seed_path = Path("/workspace/cases/SunRisesAndSets/lighting/reference/ref.png")
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(seed_path)), (1280,720)), cv2.COLOR_BGR2RGB)
    seed_x4 = torch.from_numpy(np.repeat(rgb[None],4,axis=0))
    started = time.monotonic()
    engine = WorldEngine("Overworld/Waypoint-1.5-1B", quant=None, device="cuda")
    receipt = {"status":"running", "seed":seed, "model":"Overworld/Waypoint-1.5-1B",
               "world_engine_version":importlib.metadata.version("world_engine"),
               "source":source,
               "hardware":subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True),
               "conditioning_sha256":hashlib.sha256(seed_path.read_bytes()).hexdigest(),
               "seed_resize":"1280x1280 directly stretched to 1280x720, matching the paper Waypoint input",
               "native_fps":60,"frames_per_step":4,"conditioning_frames":4,
               "steps":steps,"mode":mode,"requested_mouse_x":mouse_x,
               "strafe_duty":strafe_duty,"programs":[],
               "checkpoint_provenance":"existing HF cache; revision not recorded in this diagnostic",
               "limitations":"Nominal keyboard/mouse probes; no measured metres, degrees or camera pose; no text conditioning"}
    print(receipt['hardware'],flush=True)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)); outputs.commit()
    for name, strafe, yaw in programs:
        path = root / f"{name}.mp4"
        torch.manual_seed(seed); np.random.seed(seed); engine.reset()
        writer = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                                   "-s","1280x720","-r","60","-i","-","-c:v","libx264","-crf","18",
                                   "-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE)
        engine.append_frame(seed_x4)
        writer.stdin.write(seed_x4.numpy().tobytes())
        controls=[]
        try:
            for i in range(steps):
                # Deterministic pulse-density D key: duty regulates nominal speed.
                active = strafe and int((i+1)*strafe_duty+1e-9)>int(i*strafe_duty+1e-9)
                actual_yaw = yaw if yaw is not None else -abs(mouse_x)*float(np.cos(2*np.pi*(i+.5)/steps))
                ctrl=CtrlInput(button={68} if active else set(),mouse=[actual_yaw,0.])
                frames=engine.gen_frame(ctrl=ctrl).cpu().numpy()
                writer.stdin.write(np.ascontiguousarray(frames,dtype=np.uint8).tobytes())
                controls.append({"step":i,"button":[68] if active else [],"mouse":[actual_yaw,0.]})
                if i % 15 == 0: print(f"{name}: {i}/{steps}; elapsed {time.monotonic()-started:.1f}s",flush=True)
        finally:
            writer.stdin.close(); rc=writer.wait()
        if rc != 0: raise RuntimeError("ffmpeg encoding failed")
        (root/f"{name}.controls.json").write_text(json.dumps(controls))
        receipt["programs"].append({"name":name,"strafe":strafe,"mouse_x":yaw,
                                   "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)); outputs.commit(); cache.commit()
    receipt.update(status="completed",elapsed_sec=time.monotonic()-started)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)); outputs.commit()
    return receipt


@app.local_entrypoint()
def main(seed: int=42, run_id: str="waypoint_probes_20261010_seed42", steps: int=45,
         mode: str="probes", mouse_x: float=-.6, strafe_duty: float=1.):
    print(json.dumps(generate.remote(seed,run_id,steps,mode,mouse_x,strafe_duty),indent=2))
