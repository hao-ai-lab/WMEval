#!/usr/bin/env python3
"""Run WMEval Lite dimensions on Modal GPUs, one container per candidate row.

The repo is baked into the image; candidate videos live on Modal Volumes.
Each (dimension-set, candidate row) becomes one ``run_row`` call that runs the
ordinary ``tools/run_wmeval_lite.py run`` CLI with a one-line manifest, writes
its ``per_case_results.jsonl`` under
``WMEval-lite-candidates:<out_prefix>/<model>__<case_id>__<seed>/``, and
returns the rows. The local entrypoint fans out with ``.map`` and merges
everything into one local ``per_case_results.jsonl`` + aggregate summary (the
same two-level aggregation the CLI uses).

Candidate sources (pick one):

* ``--model-root /cand/<model>``  -- a model tree in the runner's own layout
  (``<model_root>/<scene>/<variant>/<seed>.mp4`` + optional ``model.yaml``) on
  the ``WMEval-lite-candidates`` volume, mounted at ``/cand``. Rows are
  discovered in a container with the suite's own discovery code.
* ``--manifest rows.jsonl``       -- an explicit JSONL manifest with
  container-side paths (``/cand/...`` or ``/vol/...``).
* default                          -- the paper's lighting rollouts on the
  ``WMEval-lighting-repro`` volume (mounted at ``/vol``), 6 models x 5 seeds,
  with the per-model ``shadow_hud_preset`` used for the paper numbers.

Examples::

    # the paper's lighting reproduction (30 rows)
    modal run tools/run_wmeval_lite_modal.py

    # every non-VLM dimension over a model tree
    modal run tools/run_wmeval_lite_modal.py --model-root /cand/waypoint15 \
        --dimensions action_controllability,environment_object_consistency,kinematics_mechanics,lighting_shading

    # smoke test: one row
    modal run tools/run_wmeval_lite_modal.py --model-root /cand/waypoint15 \
        --dimensions environment_object_consistency --only WanderingGarden --limit 1

OPENAI_API_KEY (prompt_scene_consistency's judge and lpips's causal-VLM
judge) is read from the environment or this repo's local ``.env`` and
forwarded to Modal as an ephemeral Secret -- never baked into the image. Every
VLM call's token usage is appended to ``<row output dir>/vlm_usage.jsonl``
(see ``evaluation/vlm_usage.py``) and summed into the run's ``vlm_usage``
totals so spend can be checked per row. ``--max-parallel N`` keeps only N
rows in flight (the judge's tokens-per-minute limit is per organisation) and
``--resume-from <per_case_results.jsonl>`` re-runs only the rows that did not
complete in an earlier run, carrying the completed ones into the new merge.
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
LIGHTING_VOLUME = "WMEval-lighting-repro"
CANDIDATES_VOLUME = "WMEval-lite-candidates"
HF_CACHE_VOLUME = "wm-hf-cache"
CONTAINER_VOL = "/vol"
CONTAINER_CAND = "/cand"
CONTAINER_HF_CACHE = "/cache/hf"
SUITE_REL = "lite"
LIGHTING_CASE_ID = "SunRisesAndSets/lighting"
USAGE_LOG_FILENAME = "vlm_usage.jsonl"

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

lighting_vol = modal.Volume.from_name(LIGHTING_VOLUME, create_if_missing=False)
candidates_vol = modal.Volume.from_name(CANDIDATES_VOLUME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
VOLUMES = {
    CONTAINER_VOL: lighting_vol,
    CONTAINER_CAND: candidates_vol,
    CONTAINER_HF_CACHE: hf_cache_vol,
}

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
        "cjm-byte-track",  # kinematics_mechanics tracker
        "pydantic",  # vlm_consistency schemas
        "requests",
        "git+https://github.com/facebookresearch/sam2.git",
        "decord",  # fluid_mechanics: SAM2 video predictor frame loading
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
            # local-only state: browser profiles with sign-in cookies, agent worktrees
            ".local",
            ".local/**",
            ".claude",
            ".claude/**",
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
        f"import transformers; import cjm_byte_track; print(\"repo import ok\")'"
    )
    # Pre-download LPIPS's VGG16 backbone weights into the image at build
    # time (~528MB), baked into this layer permanently. Without this,
    # evaluation/modules/lpips_causal_vlm.py's own
    # ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth existence check finds
    # nothing in a fresh container (this image has no torch-hub cache volume,
    # only the separate HF_HUB_CACHE one) and silently skips both causal and
    # temporal LPIPS scoring -- every row still reports status=completed with
    # every lpips_* metric null, no error, so the gap is easy to miss.
    # (TORCH_HOME is deliberately left at its default for the same reason:
    # that existence check is on the literal ~/.cache/torch path.)
    .run_commands(
        "python -c 'import lpips; lpips.LPIPS(net=\"vgg\"); "
        "print(\"vgg16 weights cached\")'"
    )
    # Same for environment_object_consistency's DINOv2 backbone (torch.hub,
    # ~90MB): bake it in instead of re-downloading it in every container.
    .run_commands(
        "python -c 'import torch; torch.hub.load(\"facebookresearch/dinov2\", "
        "\"dinov2_vits14\"); print(\"dinov2 cached\")'"
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


def _row_tag(row: dict) -> str:
    return f"{row['model']}__{row['case_id'].replace('/', '-')}__{row['seed']}"


def _sum_usage(records: list[dict]) -> dict:
    """Total prompt/completion/reasoning tokens per model over usage records."""
    totals: dict[str, dict[str, int]] = {}
    for rec in records:
        model = str(rec.get("model", "?"))
        bucket = totals.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        )
        bucket["calls"] += 1
        for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
            bucket[key] += int(rec.get(key) or 0)
    return totals


@app.function(image=image, timeout=600, volumes=VOLUMES)
def list_candidate_rows(model_root: str, dimensions: str) -> list[dict]:
    """Discover ``<model_root>/<scene>/<variant>/<seed>.mp4`` rows for the
    requested dimensions' cases, with model.yaml params merged in."""
    sys.path.insert(0, REMOTE_REPO)
    from evaluation.lite.candidate_discovery import discover_candidate_directory
    from evaluation.lite.suite import _load_model_params
    from evaluation.lite.suite_loader import load_suite

    suite = load_suite(Path(REMOTE_REPO) / SUITE_REL)
    wanted = [d.strip() for d in dimensions.split(",") if d.strip()]
    case_ids = {
        case_id
        for dim_id, dim in suite.dimensions.items()
        if not wanted or dim_id in wanted
        for case_id in dim.cases
    }
    root = Path(model_root)
    params = dict(_load_model_params(root))
    rows = []
    for row in discover_candidate_directory(root, allowed_cases=case_ids):
        rows.append(
            {
                "model": row.model,
                "case_id": row.case_id,
                "seed": row.seed,
                "candidate_video": str(row.candidate_video),
                "params": {**params, **dict(row.params)},
            }
        )
    return rows


@app.function(image=image, gpu="L40S", timeout=3 * 3600, volumes=VOLUMES)
def run_row(row: dict, dimensions: str, out_prefix: str, run_id: str) -> dict:
    """Evaluate one candidate row through the ordinary Lite CLI."""
    import subprocess

    tag = _row_tag(row)
    out_dir = f"{CONTAINER_CAND}/{out_prefix}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    manifest = f"/tmp/{tag}.jsonl"
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    usage_log = Path(out_dir) / USAGE_LOG_FILENAME

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
    env = {**os.environ, "WMEVAL_VLM_USAGE_LOG": str(usage_log)}
    proc = subprocess.run(cmd, cwd=REMOTE_REPO, capture_output=True, text=True, env=env)
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
    print(f"[{tag}] exit={proc.returncode}\n{tail}", flush=True)

    rows: list[dict] = []
    results_path = Path(out_dir) / "per_case_results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    usage: list[dict] = []
    if usage_log.is_file():
        for line in usage_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                usage.append(json.loads(line))
    candidates_vol.commit()
    hf_cache_vol.commit()
    return {
        "tag": tag,
        "returncode": proc.returncode,
        "rows": rows,
        "log_tail": tail,
        "out_dir": out_dir,
        "vlm_usage": usage,
    }


def _merge_locally(all_rows: list[dict], local_out: Path, usage: list[dict]) -> None:
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
    if usage:
        with (local_out / USAGE_LOG_FILENAME).open("w", encoding="utf-8") as handle:
            for rec in usage:
                handle.write(json.dumps(rec) + "\n")
        totals = _sum_usage(usage)
        (local_out / "vlm_usage_totals.json").write_text(
            json.dumps(totals, indent=2), encoding="utf-8"
        )
        print("VLM token usage per model:")
        for model, bucket in sorted(totals.items()):
            print(f"  {model}: {bucket}")


@app.local_entrypoint()
def main(
    manifest: str = "",
    model_root: str = "",
    dimensions: str = "lighting_shading",
    only: str = "",
    limit: int = 0,
    out_prefix: str = "",
    run_id: str = "",
    local_out: str = "",
    max_parallel: int = 0,
    resume_from: str = "",
):
    """Fan the rows out over Modal, then merge results locally.

    --model-root container path of a model tree on the candidates volume
    --manifest   local JSONL with container-side candidate paths
    --dimensions comma-separated dimension ids (default lighting_shading)
    --only       substring filter on "<model>__<case_id>__<seed>"
    --limit      keep only the first N rows (smoke tests)
    --out-prefix volume-relative output prefix (default eval/lite_runs/<run_id>)
    --local-out  local merge dir (default reports/wmeval_lite_modal/<run_id>)
    --max-parallel  run at most N rows at a time (VLM dimensions: keeps the
                 per-minute token rate under the OpenAI limit; 0 = all at once)
    --resume-from   a previous run's merged per_case_results.jsonl: rows that
                 completed there for every requested dimension are skipped and
                 carried into this run's merge, so only errored/missing rows
                 are re-evaluated
    """
    run_id = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_prefix = out_prefix or f"eval/lite_runs/{run_id}"
    if manifest and model_root:
        raise SystemExit("pass either --manifest or --model-root, not both")
    if model_root:
        rows = list_candidate_rows.remote(model_root, dimensions)
    elif manifest:
        rows = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = default_lighting_rows()
    if only:
        rows = [row for row in rows if only in _row_tag(row)]
    if limit > 0:
        rows = rows[:limit]
    carried: list[dict] = []
    if resume_from:
        previous = [
            json.loads(line)
            for line in Path(resume_from).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        requested = [d.strip() for d in dimensions.split(",") if d.strip()]
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from evaluation.lite.suite_loader import load_suite

        suite = load_suite(REPO_ROOT / SUITE_REL)
        completed = {
            (r["dimension"], r["case_id"], r["model"], r["seed"])
            for r in previous
            if r.get("status") == "completed"
        }
        remaining = []
        for row in rows:
            needed = [
                dim for dim in requested
                if row["case_id"] in suite.dimensions[dim].cases
            ]
            done = [dim for dim in needed if (dim, row["case_id"], row["model"], row["seed"]) in completed]
            if needed and len(done) == len(needed):
                carried.extend(
                    r for r in previous
                    if r.get("status") == "completed"
                    and r["dimension"] in needed
                    and (r["case_id"], r["model"], r["seed"]) == (row["case_id"], row["model"], row["seed"])
                )
            else:
                remaining.append(row)
        print(f"resume: {len(rows) - len(remaining)} row(s) already complete in {resume_from}, {len(remaining)} to run")
        rows = remaining
        if not rows:
            _merge_locally(carried, Path(local_out or REPO_ROOT / "reports" / "wmeval_lite_modal" / run_id), [])
            return
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
            "prompt_scene_consistency rows will be skipped and lpips's "
            "causal-VLM-judge path will fail remotely for any row that needs it."
        )

    print(f"run_id={run_id} rows={len(rows)} dimensions={dimensions} out_prefix={out_prefix}")
    for row in rows:
        print(f"  {_row_tag(row)}  {row['candidate_video']}  params={row.get('params', {})}")
    all_rows: list[dict] = list(carried)
    all_usage: list[dict] = []
    failures: list[str] = []
    chunk = max_parallel if max_parallel > 0 else len(rows)
    results = (
        result
        for start in range(0, len(rows), chunk)
        for result in target_fn.map(
            rows[start:start + chunk],
            kwargs={"dimensions": dimensions, "out_prefix": out_prefix, "run_id": run_id},
        )
    )
    for result in results:
        status = ", ".join(sorted({str(r.get("status")) for r in result["rows"]})) or "no rows"
        usage = result.get("vlm_usage") or []
        used = _sum_usage(usage)
        used_str = " ".join(
            f"{m}:{b['calls']}calls/{b['input_tokens']}in/{b['output_tokens']}out" for m, b in used.items()
        )
        print(f"[{result['tag']}] exit={result['returncode']} status={status} {used_str}")
        if result["returncode"] != 0 or not result["rows"]:
            failures.append(result["tag"])
            print(result["log_tail"])
        all_rows.extend(result["rows"])
        for rec in usage:
            all_usage.append({"tag": result["tag"], **rec})

    _merge_locally(
        all_rows,
        Path(local_out or REPO_ROOT / "reports" / "wmeval_lite_modal" / run_id),
        all_usage,
    )
    if failures:
        print(f"FAILED rows: {failures}")
    print(f"per-row outputs on volume {CANDIDATES_VOLUME}: {out_prefix}/<model>__<case_id>__<seed>/")
