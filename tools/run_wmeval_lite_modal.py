#!/usr/bin/env python3
"""Run WMEval Lite dimensions on Modal GPUs, one container per candidate row.

The repo is baked into the image; candidate videos live on a Modal Volume.
Each (dimension-set, candidate row) becomes one ``run_row`` call that runs the
ordinary ``tools/run_wmeval_lite.py run`` CLI with a one-line manifest, writes
its ``per_case_results.jsonl`` under ``<volume>/<out_prefix>/<model>__<seed>/``,
and returns the rows. The local entrypoint fans out with ``.map`` and merges
everything into one local ``per_case_results.jsonl`` + aggregate summary
(the same two-level aggregation the CLI uses).

Default rows: the paper's lighting rollouts on the ``WMEval-lighting-repro``
volume (6 models x 5 seeds -> ``SunRisesAndSets/lighting``), with the
per-model ``shadow_hud_preset`` used for the paper numbers.

Examples::

    # smoke test: one small rollout
    modal run tools/run_wmeval_lite_modal.py --only lingbot_world --limit 1

    # whole lighting dimension (30 rows in parallel), merged output under
    # reports/wmeval_lite_modal/<run_id>/
    modal run tools/run_wmeval_lite_modal.py

    # any other manifest (container-side paths, e.g. /vol/...):
    modal run tools/run_wmeval_lite_modal.py --manifest my_rows.jsonl \
        --dimensions lighting_shading,prompt_scene_consistency
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/WMEval"
VOLUME_NAME = "WMEval-lighting-repro"
HF_CACHE_VOLUME = "wm-hf-cache"
CONTAINER_VOL = "/vol"
CONTAINER_HF_CACHE = "/cache/hf"
SUITE_REL = "eval_sets/wmeval_lite_v1"
LIGHTING_CASE_ID = "SunRisesAndSets/lighting"

# (manifest model name, volume file stem, seeds, hud preset). Presets match
# the paper's eval runs: only LingBot / Matrix-Game overlays are masked.
LIGHTING_FAMILIES: list[tuple[str, str, list[str], str]] = [
    ("project_genie", "genie_s{seed}", ["1", "2", "3", "4", "5"], "none"),
    ("lingbot_world", "lingbot_seed{seed}", ["42", "1", "2", "3", "4"], "lingbot"),
    ("matrix_game_2", "mg2_seed{seed}", ["42", "1", "2", "3", "4"], "mg2"),
    ("matrix_game_3", "mg3_seed{seed}", ["42", "1", "2", "3", "4"], "mg3"),
    ("hy_worldplay", "hy_seed{seed}", ["42", "1", "2", "3", "4"], "none"),
    ("waypoint15", "waypoint15_seed{seed}", ["42", "1", "2", "3", "4"], "none"),
]

TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

app = modal.App("wmeval-lite-runner")

candidates_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        extra_options=f"--index-url {TORCH_INDEX_URL}",
    )
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "scipy",
        "PyYAML>=6,<7",
        "transformers==4.51.3",
        "tokenizers>=0.20.3",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "lpips",
        "openai",
        "git+https://github.com/facebookresearch/sam2.git",
    )
    .env(
        {
            "HF_HUB_CACHE": CONTAINER_HF_CACHE,
            "HF_HOME": CONTAINER_HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        REMOTE_REPO,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".venv",
            ".venv/**",
            "**/__pycache__",
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache",
            ".pytest_cache/**",
            "**/.pytest_cache",
            "**/.pytest_cache/**",
            "**/.DS_Store",
            ".env",
            ".env.example",
            "WorldModelBenchScenePrivate",
            "WorldModelBenchScenePrivate/**",
            "reports",
            "reports/**",
            "runs",
            "runs/**",
            "tests/fixtures",
            "tests/fixtures/**",
            "**/*.zip",
            "**/*.docx",
        ],
    )
    .run_commands(
        f"cd {REMOTE_REPO} && python -c 'import evaluation.runner; import sam2; "
        f"import transformers; print(\"repo import ok\")'"
    )
)


def _load_dotenv_local(repo_root: Path) -> None:
    """Load KEY=VALUE lines from <repo>/.env into THIS (caller-side) process
    only -- never baked into the image (see the ``.env``/``.env.example``
    entries in the image's ``add_local_dir`` ignore list above). Mirrors
    ``tools/run_wmeval_lite.py``'s own ``_load_dotenv``."""
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def default_lighting_rows() -> list[dict]:
    rows: list[dict] = []
    for model, stem, seeds, hud in LIGHTING_FAMILIES:
        for seed in seeds:
            rows.append(
                {
                    "model": model,
                    "case_id": LIGHTING_CASE_ID,
                    "seed": seed,
                    "candidate_video": f"{CONTAINER_VOL}/rollouts/lighting/{stem.format(seed=seed)}.mp4",
                    "params": {"shadow_hud_preset": hud},
                }
            )
    return rows


@app.function(
    image=image,
    gpu="L40S",
    timeout=3 * 3600,
    volumes={CONTAINER_VOL: candidates_vol, CONTAINER_HF_CACHE: hf_cache_vol},
)
def run_row(row: dict, dimensions: str, out_prefix: str, run_id: str) -> dict:
    """Evaluate one candidate row through the ordinary Lite CLI."""
    import os
    import subprocess

    tag = f"{row['model']}__{row['seed']}"
    out_dir = f"{CONTAINER_VOL}/{out_prefix}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    manifest = f"/tmp/{tag}.jsonl"
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    cmd = [
        "python",
        "tools/run_wmeval_lite.py",
        "run",
        "--suite",
        SUITE_REL,
        "--candidate-manifest",
        manifest,
        "--dimensions",
        dimensions,
        "--output-dir",
        out_dir,
        "--device",
        "cuda",
        "--run-id",
        run_id,
    ]
    print(f"[{tag}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REMOTE_REPO, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
    print(f"[{tag}] exit={proc.returncode}\n{tail}", flush=True)

    rows: list[dict] = []
    results_path = Path(out_dir) / "per_case_results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    candidates_vol.commit()
    return {
        "tag": tag,
        "returncode": proc.returncode,
        "rows": rows,
        "log_tail": tail,
        "out_dir": out_dir,
    }


def _merge_locally(all_rows: list[dict], local_out: Path) -> None:
    """Write the merged JSONL + aggregate (same code path as the CLI)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluation.lite.aggregate import (
        build_aggregate,
        write_aggregate_json,
        write_summary_csv,
    )
    from evaluation.lite.output import (
        AGGREGATE_CSV_FILENAME,
        AGGREGATE_JSON_FILENAME,
        PER_CASE_RESULTS_FILENAME,
    )
    from evaluation.lite.suite_loader import load_suite

    local_out.mkdir(parents=True, exist_ok=True)
    results_path = local_out / PER_CASE_RESULTS_FILENAME
    with results_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    suite = load_suite(REPO_ROOT / SUITE_REL)
    aggregate = build_aggregate(suite, all_rows)
    write_aggregate_json(local_out / AGGREGATE_JSON_FILENAME, aggregate)
    write_summary_csv(local_out / AGGREGATE_CSV_FILENAME, suite, aggregate)
    print(f"merged {len(all_rows)} row(s) -> {results_path}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def main(
    manifest: str = "",
    dimensions: str = "lighting_shading",
    only: str = "",
    limit: int = 0,
    out_prefix: str = "",
    run_id: str = "",
    local_out: str = "",
):
    """Fan the rows out over Modal, then merge results locally.

    --manifest   local JSONL with container-side candidate paths (default: the
                 paper's lighting rollouts on the volume)
    --only       substring filter on "<model>__<seed>"
    --limit      keep only the first N rows (smoke tests)
    --out-prefix volume-relative output prefix (default eval/lite_runs/<run_id>)
    --local-out  local merge dir (default reports/wmeval_lite_modal/<run_id>)

    OPENAI_API_KEY (needed for lpips's causal_vlm_collision judge) is read
    from the environment or this repo's local .env and forwarded to Modal as
    an ephemeral Secret -- never baked into the image (see the
    ``.env``/``.env.example`` entries in the image's ``add_local_dir`` ignore
    list above).
    """
    run_id = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_prefix = out_prefix or f"eval/lite_runs/{run_id}"
    if manifest:
        rows = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = default_lighting_rows()
    if only:
        rows = [row for row in rows if only in f"{row['model']}__{row['seed']}"]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        print("no rows selected")
        return

    _load_dotenv_local(REPO_ROOT)
    target_fn = run_row
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        secret = modal.Secret.from_dict({"OPENAI_API_KEY": openai_key})
        target_fn = run_row.with_options(secrets=[secret])
    else:
        print(
            "warning: OPENAI_API_KEY not set (checked env + .env) -- "
            "lpips's causal-VLM-judge path will fail remotely for any row "
            "that needs it."
        )

    print(f"run_id={run_id} rows={len(rows)} dimensions={dimensions} out_prefix={out_prefix}")
    all_rows: list[dict] = []
    failures: list[str] = []
    for result in target_fn.map(
        rows, kwargs={"dimensions": dimensions, "out_prefix": out_prefix, "run_id": run_id}
    ):
        status = ", ".join(sorted({str(r.get("status")) for r in result["rows"]})) or "no rows"
        print(f"[{result['tag']}] exit={result['returncode']} status={status}")
        if result["returncode"] != 0 or not result["rows"]:
            failures.append(result["tag"])
            print(result["log_tail"])
        all_rows.extend(result["rows"])

    _merge_locally(all_rows, Path(local_out or REPO_ROOT / "reports" / "wmeval_lite_modal" / run_id))
    if failures:
        print(f"FAILED rows: {failures}")
    print(f"per-row outputs on volume {VOLUME_NAME}: {out_prefix}/<model>__<seed>/")
