# Compact Table 4/7 reference package

The reference-only package is intentionally separate from Project Genie
generations and model checkpoints. It contains everything fixed before a new
model is calibrated and generated:

- exact conditioning images for all 14 active cells;
- Unity reference videos and numerical poses, intrinsics, object transforms,
  masses/diameters, and planar anchors;
- semantic and compiled action trajectories;
- evaluator protocol and published comparison baseline;
- all 110 successful left/right yaw calibration samples at approximately
  one-second spacing, resized to 1280-wide JPEG for visual calibration;
- the original calibration JSON and source-PNG hashes.

Build and verify:

```bash
uv run python tools/paper_tables/build_table_reference_package.py build
uv run python tools/paper_tables/build_table_reference_package.py verify
```

Outputs are ignored by git:

```text
dist/reference-packages/wmeval-table4-table7-reference-v1.zip
dist/reference-packages/wmeval-table4-table7-reference-v1.zip.sha256
```

The checked-in `reference_package_manifest.json` records the archive checksum,
all exact source hashes, and all derived calibration-frame hashes.

## Complete two-machine workflow

The git branch contains code and manifests. The ZIP contains ignored/private
reference assets and must be transferred separately.

On the source machine:

```bash
git switch lanxiang/release-cleanup
git pull --ff-only origin lanxiang/release-cleanup

uv run python tools/paper_tables/build_table_reference_package.py build
uv run python tools/paper_tables/build_table_reference_package.py verify
```

On the GPU machine, obtain the branch before overlaying the reference package:

```bash
git clone https://github.com/hao-ai-lab/WMEval.git
cd WMEval
git fetch origin lanxiang/release-cleanup
git switch --track origin/lanxiang/release-cleanup
```

If the branch already exists locally:

```bash
git fetch origin
git switch lanxiang/release-cleanup
git pull --ff-only origin lanxiang/release-cleanup
```

## Distribution recommendation

For one known GPU machine, `rsync` over SSH is simplest and resumable:

```bash
rsync -ahP \
  dist/reference-packages/wmeval-table4-table7-reference-v1.zip* \
  USER@GPU_HOST:/path/to/downloads/
```

For repeated team downloads, upload the ZIP and checksum sidecar to private
object storage and share a time-limited signed URL. For example, with an
approved S3 bucket:

```bash
aws s3 cp dist/reference-packages/wmeval-table4-table7-reference-v1.zip \
  s3://BUCKET/wmeval/reference-v1/
aws s3 cp dist/reference-packages/wmeval-table4-table7-reference-v1.zip.sha256 \
  s3://BUCKET/wmeval/reference-v1/
aws s3 presign \
  s3://BUCKET/wmeval/reference-v1/wmeval-table4-table7-reference-v1.zip \
  --expires-in 604800
```

A GitHub Release asset is also technically suitable because the archive is
well below GitHub's per-asset limit, but these files contain renders derived
from `WorldModelBenchScenePrivate`. Do not upload them to a public release
until their distribution status is confirmed. A private object store or
direct `rsync` is the safer default.

## Download, verify, and install

```bash
curl -L -o wmeval-table4-table7-reference-v1.zip 'SIGNED_URL'
curl -L -o wmeval-table4-table7-reference-v1.zip.sha256 'SIGNED_CHECKSUM_URL'
sha256sum -c wmeval-table4-table7-reference-v1.zip.sha256

unzip wmeval-table4-table7-reference-v1.zip -d /tmp/wmeval-reference
rsync -a \
  /tmp/wmeval-reference/wmeval-table4-table7-reference-v1/ \
  /path/to/WMEval/

cd /path/to/WMEval
scripts/bootstrap_gpu.sh
.venv-gpu/bin/python tools/paper_tables/preflight_gpu.py --reference-only
```

On macOS, replace `sha256sum -c` with `shasum -a 256 -c`.

## New-model calibration and generation contract

After reference preflight succeeds:

1. Use the exact per-cell conditioning image declared in `cases.yaml`.
2. Calibrate the backend's action timing/gain against the Unity reference
   videos/traces and `calibration_samples/`; do not change the semantic action
   trajectory.
3. Record the backend mapping, timing, model/version, prompt hashes, input
   frame hash, and output-video hash.
4. Generate independent `run0` through `run4` for every selected active cell.
5. Normalize each output to the reference fps, resolution, duration, crop/pad,
   and timebase before evaluation. Preserve the original output separately.
6. Write candidate-manifest rows with `model`, Lite `case_id`, `seed`, and
   `candidate_video`; include case-specific analytical-reference overrides in
   `params` when required.
7. Run the relevant dimensions with `tools/run_wmeval_lite.py`, then compare
   the five-run mean and sample standard deviation (`ddof=1`) against
   `paper_baseline.yaml`.

The existing Project Genie orchestration demonstrates this contract in
`tools/paper_tables/run_prepared_table_evaluations.py`. A new backend should implement its
own generation adapter but reuse the fixed references and evaluator settings.
