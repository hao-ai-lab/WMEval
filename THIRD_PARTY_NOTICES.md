# Third-party notices

WMEval includes a small amount of vendored third-party code. It also
downloads pinned third-party model weights at evaluation time. This file lists
both, with their upstream sources.

Licenses are given only where a file in this repository states them. Every
other entry says **verify upstream**: check the license at the upstream source
before you redistribute the item or use it commercially. None of the model
weights below ship in this repository or in the WMEval Lite dataset. They are
downloaded from their upstream hosts on first use.

## Vendored code

| Path | What | Upstream | License |
| --- | --- | --- | --- |
| `evaluation/modules/vendor/ptlflow_synthetic/synthetic_flow.py`, `calibration.json` | Longuet-Higgins synthetic optical flow from game actions, used by the synthetic-flow controllability metrics (`control_motion`). Copied unmodified. | <https://github.com/shaoxiongduan/ptlflow>, a fork of <https://github.com/hmorimitsu/ptlflow> | Apache License 2.0, as stated in `evaluation/modules/vendor/ptlflow_synthetic/NOTICE.md`. The license text itself is not vendored next to the files. TODO(maintainers): add it. |

### Related in-repository code

| Path | What | Note |
| --- | --- | --- |
| `imports/CausalStep/` | Stepwise causal QA evaluator (schema, metrics, runner) used by `physical_causality`. It follows the CausalStep protocol (arXiv:2507.16878). | The repository history records this code as written here, and no upstream license or notice ships with it. If any code was copied from an upstream CausalStep release: verify upstream. |

## Pinned model weights

These are pinned in `experiments/protocol/evaluator_assets.lock.json`
(`schema_version: 1`, captured 2026-10-04). The retired lighting-orbit
protocol pinned its own subset in
`experiments/lighting_orbit/evaluator_assets.lock.json` (historical).

| Lock key | Upstream source | Pinned revision | Weight file | SHA-256 | License |
| --- | --- | --- | --- | --- | --- |
| `grounding_dino` | Hugging Face `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` | `model.safetensors` | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` | verify upstream |
| `depth_anything_metric_outdoor` | Hugging Face `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | `fd2c22027eaf20374204f14099b8341e1925ad39` | `model.safetensors` | `ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124` | verify upstream |
| `sam2` | Hugging Face `facebook/sam2.1-hiera-large` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` | `sam2.1_hiera_large.pt` | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | verify upstream |
| `dinov2` | `torch.hub` `facebookresearch/dinov2`, variant `dinov2_vits14` | source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | `dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | verify upstream |
| `raft_large` | torchvision `torchvision.models.optical_flow.raft_large`, weights `C_T_SKHT_V2` | torchvision release (see `requirements/gpu.lock`) | `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322` | verify upstream |
| `lpips_vgg` (backbone) | torchvision VGG16 backbone used by the `lpips` package (`net=vgg`) | torchvision release | `vgg16-397923af.pth` | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` | verify upstream |
| `lpips_vgg` (linear layers) | `lpips` pip package (`lpips==0.1.4` in `requirements/gpu.lock`), v0.1 linear layers | package release | `lpips/weights/v0.1/vgg.pth` | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` | verify upstream |

SAM2 *source code* is installed from <https://github.com/facebookresearch/sam2>
at revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`, as recorded in the
historical lighting-orbit lock (`sam2_source`). License: verify upstream.

### Models referenced in code or `suite.yaml` but not pinned in the lock

The evaluator can load these, but neither lock file records a revision or
SHA-256 for them.

| Model | Where it is used | License |
| --- | --- | --- |
| Hugging Face `facebook/sam2.1-hiera-small` | `fluid_mechanics` (`liquid_sam2_model` in `lite/suite.yaml`) and the liquid modules' default | verify upstream |
| Hugging Face `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | `evaluation/modules/environment_object_consistency.py` (indoor metric-depth default) | verify upstream |
| Hugging Face `Searchium-ai/clip4clip-webvid150k` | `evaluation/modules/clip4clip.py` (standalone CLIP4Clip evaluator; not a WMEval Lite dimension) | verify upstream |

## Hosted services

The VLM-judged dimensions call the OpenAI API: `prompt_scene_consistency`,
`physical_causality` and causal `lpips`. The judge models are pinned in
`lite/suite.yaml` (`gpt-5.4`, plus `gpt-5.4-mini` for the
LPIPS coarse pass). Using them is subject to the provider's terms. Nothing from
the provider is redistributed here.

## Python dependencies

Third-party Python packages are installed from the pinned lock files
(`requirements/gpu.lock`, `requirements/macos.lock`)
and are not redistributed. Examples are `torch`, `torchvision`,
`transformers`, `lpips`, `sam2`, `cjm-byte-track` and `opencv-python`. Each is
under its own license: verify upstream.
