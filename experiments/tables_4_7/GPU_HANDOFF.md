# Table 4/7 CUDA handoff

This branch can rerun all **14 active cells / 70 independent Project Genie
generations** on another Linux GPU machine. Table 4 RacingCar first-person and
BilliardGame third-person are intentionally excluded. The claim scope remains
`runnable_diagnostic_not_exact_paper_parity` because the paper did not publish
the original action timelines or spatial transform.

The required generated videos and Unity analytical captures live under the
gitignored `runs/` tree. A git checkout alone is therefore insufficient. The
checked-in `gpu_handoff_manifest.json` lists the minimal hashed evidence set:
319 files totaling about 585 MB at preparation time.

## 1. Verify or copy the evidence on the source machine

```bash
uv run python tools/build_table_gpu_handoff.py --verify

# Destination is the root of another checkout or a staging directory whose
# contents will be overlaid on the GPU checkout.
uv run python tools/build_table_gpu_handoff.py \
  --copy-to /path/to/gpu-checkout
```

The copy preserves repository-relative paths. Receipts retain their original
absolute paths for provenance, while readers safely relocate missing paths by
the `runs/`, `experiments/`, or `eval_sets/` suffix and still require the
recorded SHA-256.

## 2. Bootstrap the CUDA environment

Prerequisites: Linux x86_64, Python 3.10, `uv`, `git`, `ffmpeg`, an NVIDIA
driver compatible with CUDA 12.4, sufficient model-cache storage, and network
access for the first checkpoint download (or a pre-populated cache).

```bash
git switch lanxiang/release-cleanup
scripts/bootstrap_tables_4_7_gpu.sh
```

This creates `.venv-gpu`, installs Torch 2.6.0 / torchvision 0.21.0 from the
CUDA 12.4 wheel index, installs `requirements-gpu.lock`, validates every
handoff hash, checks evaluator imports, and fails if CUDA is unavailable.

Runtime model aliases required by the active cells are printed by preflight:

- `IDEA-Research/grounding-dino-tiny`
- `facebook/sam2.1-hiera-large`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- `facebookresearch/dinov2:dinov2_vits14`

Their immutable revision/hash lock remains a release P0 in `PENDING.md`; until
that is closed, an online or pre-populated cache is required and the model
cache must be retained with the results.

## 3. Run all active cells

```bash
WMEVAL_REPORT_SUFFIX=cuda_release scripts/run_tables_4_7_gpu.sh
```

The runner validates all five receipts per cell, normalizes candidate video to
the corresponding Unity timebase and image geometry, remaps analytical ground
truth, evaluates on CUDA, resumes safely, rejects missing required metrics, and
writes five-run comparisons under `reports/table4/` and `reports/table7/`.

For a preparation-only smoke test:

```bash
.venv-gpu/bin/python tools/preflight_tables_4_7_gpu.py
.venv-gpu/bin/python tools/run_prepared_table_evaluations.py \
  --device cuda --prepare-only --fail-incomplete
```

Do not regenerate Project Genie worlds on the GPU machine unless replacing an
invalid repeat. The handoff evaluates the already validated independent
generations. If a repeat is rejected, use the regeneration command recorded in
`evaluation_automation_status.json` on a browser-capable machine, rebuild the
handoff manifest, and transfer the replacement artifacts.
