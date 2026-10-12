# Lighting orbit v2 candidate dataset

This directory ships all fifteen first valid native rollouts: LingBot, Matrix-Game3 and HY-WorldPlay, seeds 42/1/2/3/4. candidates.jsonl uses paths relative to this directory. Generation receipts are exact source records; absolute command paths inside them are archival provenance, not input dependencies. expected_scores.json preserves the original RTX PRO 6000 scores. fresh_results contains a separate RTX PRO 4500 recomputation, not replacement reference values.

The full ten-second look-at protocol and three explicit N/A interface decisions are documented in experiments/lighting_orbit_v2/README.md. No Genie diagnostic or calibration probe enters the formal candidate manifest.

From the repository root, build the portable bundle without a private directory:

```sh
python tools/build_lighting_release_package.py build \
  --inputs eval_sets/wmeval_lite_v1/baselines/lighting_orbit_v2/release_sources.json \
  --output /tmp/wmeval-lighting-release
python tools/build_lighting_release_package.py verify --root /tmp/wmeval-lighting-release
python tools/build_lighting_release_package.py archive \
  --root /tmp/wmeval-lighting-release --output /tmp/wmeval-lighting-release.zip
cd /tmp/wmeval-lighting-release
cp -R baseline_results results
python tools/run_wmeval_lite.py run --dimensions lighting_shading \
  --candidate-manifest candidates.jsonl --output-dir results --device cpu --resume
```

The last command reuses the fifteen archived fresh rows and rebuilds their aggregates; it needs PyYAML but no GPU, model weights or GPT API key. This is cached-result replay, not a fresh inference test.

For fresh scoring, install Python 3.12, Torch 2.8.0/CUDA12.8, torchvision 0.23.0 and experiments/lighting_orbit_v2/requirements-lighting-score.lock. Install SAM2 from revision 2b90b9f5ceec907a1c18123530e92e794ad901a4. Populate a new dedicated HF cache using tools/prepare_lighting_evaluator.py --lock eval_sets/wmeval_lite_v1/cases/SunRisesAndSets/lighting_orbit_v2/reference/artifacts/evaluator_assets.lock.json --cache-dir /path/to/dedicated/hub, then export HF_HUB_CACHE to that directory and HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Run the same Lite command with --output-dir results --device cuda, without --resume.

Current acceptance: all fifteen videos recomputed; all normalized hashes unchanged; actual resume preserved the exact score-row hash. Strict1e-6 numerical comparison FAILED for 21/90 values on a different GPU. Maximum per-seed differences: MOSD 0.04 degrees, SDJ 0.17 degrees/frame, SAS 0.18 percentage points, SSD 0.0148; SRR/SDR unchanged. See experiments/lighting_orbit_v2/results/release_verification/. The new case remains outside the default suite until this numerical stability gap is addressed.
