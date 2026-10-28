---
pretty_name: WMEval Lite
license: other
license_name: unity-benchmark-scene-license-v1.0
license_link: https://huggingface.co/datasets/lmgame/WMEval-Lite/blob/main/LICENSE-UNITY.txt
language:
  - en
tags:
  - world-models
  - video-generation
  - interactive-video
  - evaluation
  - benchmark
  - unity
size_categories:
  - n<1K
dataset_info:
  - config_name: default
    features:
      - name: case_id
        dtype: string
      - name: scene
        dtype: string
      - name: variant
        dtype: string
      - name: group
        dtype: string
      - name: family
        dtype: string
      - name: action_target
        dtype: string
      - name: suites
        sequence: string
      - name: dimensions
        sequence: string
      - name: perspective
        dtype: string
      - name: prompt
        dtype: string
      - name: conditioning_case
        dtype: string
      - name: reference_video
        dtype: string
      - name: first_frame
        dtype: string
      - name: trajectory
        dtype: string
      - name: candidate_normalization
        dtype: string
      - name: reference_status
        dtype: string
configs:
  - config_name: default
    data_files:
      - split: test_full
        path: test_full.jsonl
      - split: test_core
        path: test_core.jsonl
---

# WMEval Lite (v1)


# WMEval Lite

<p align="center"> <a href="https://arxiv.org/pdf/2610.33894"><b>📜 Paper</b></a> | <a href="https://github.com/hao-ai-lab/WMEval"><b>💻 Code</b></a> | <a href="https://snyhlx.github.io/wmeval-blog/"><b>🌐 Project Page</b></a></p>

The Unity-free evaluation set of **WMEval**, a benchmark for interactive world models. Each case is a Unity scene rendered under a known action script. The dataset ships the **generation conditions** (prompt, conditioning frame, actions) and the **frozen Unity references** (videos, ground-truth geometry, frozen judge outputs): generate videos from the conditions with your model, then score them with the [evaluator](https://github.com/hao-ai-lab/WMEval). No Unity needed.

## Splits

| Split | Rows | Contents |
| --- | ---: | --- |
| `test_core` | 32 | Curated cases over 9 scenes and 8 dimensions (43 case–dimension pairs); the paper's numbers come from this split |
| `test_full` | 252 | `test_core` plus 220 synthetic trajectories (camera yaw and orbit, vehicle, cue ball, rolling ball, beam-domino ball, bottle, character), each with its own Unity reference; reported apart from the core headline |

Synthetic trajectories are built text first: an action description ("turns left at 30°/s for 2 s, then holds still") completes a core case's prompt and is compiled into its actions, starting from that case's conditioning frame.

## Layout

Files sit at the same paths as in the code repository, which keeps `suite.yaml`, each `case.yaml` and the checksum manifests; `python tools/sync_lite_data.py download` places them and verifies every checksum.

```
cases/<Scene>/<variant>/reference/
  reference.mp4                 # frozen Unity reference video
  ref.png                       # conditioning frame, where it differs from frame 0
  artifacts/                    # actions.txt, action.npy/npz, trajectory.yaml, actions.jsonl,
                                # Unity poses / intrinsics / object transforms / masses,
                                # rubrics, frozen judge outputs, liquid masks and flow
synthetic/<family>/<Scene>__<variant>/<pattern>/   # trajectory, compiled actions, meta.json, reference/
test_core.jsonl, test_full.jsonl                   # the split index shown in the viewer
audit/, protocol/, judge_noise_floor.json          # dataset audit, paper-rerun records, judge floor
```

## Dimensions

| Dimension | Cases | Reference video | Headline metric |
| --- | ---: | :---: | --- |
| Environment object & character consistency | 16 | yes | `object_permanence_pct` ↑ |
| Action controllability | 3 | yes | `relative_epe_dynamic_mean_raw` ↓ |
| Kinematics & mechanics | 3 | yes | `fde_m_mean` ↓ |
| Fluid mechanics | 4 | yes | `velocity_cosine_similarity` ↑ |
| Physical causality | 3 | yes | `csr_delta` ↓ |
| Lighting & shading | 1 | no | `shadow_direction_jitter_deg_per_frame` ↓ |
| Prompt-scene consistency | 6 | no | `overall_score` ↑ |
| Perceptual similarity (causal LPIPS) | 7 | yes | `lpips_mean` ↓ |

## Scoring a model

1. Generate one video per case and seed from `case.yaml` `prompt`, `reference/ref.png` (else frame 0 of the reference) and `reference/artifacts/` actions; match the reference's fps, resolution and duration, or follow the case's `protocol` block.
2. Place them at `<model_root>/<Scene>/<variant>/<seed>.mp4` (or list them in a JSONL `--candidate-manifest`).
3. Run from a checkout of the [evaluator](https://github.com/hao-ai-lab/WMEval):

```sh
python tools/run_wmeval_lite.py run --suite lite --model-root <model_root> --output-dir reports/<model> --resume
```

Reference-side judge outputs, Unity ground truth and fluid masks/flow are frozen in the bundles, so only the candidate side is computed. CV dimensions need a GPU and fetch pinned checkpoints on first use; `prompt_scene_consistency`, `physical_causality` and causal `lpips` call a hosted VLM judge (`OPENAI_API_KEY`, `gpt-5.4`). Deviations from the paper and known limitations: [`DIFFERENCES_FROM_PAPER.md`](DIFFERENCES_FROM_PAPER.md).

## License

The Unity benchmark scenes and their outputs, including the reference rollouts and Unity-logged quantities here, are licensed under the **Unity Benchmark Scene License v1.0** ([`LICENSE-UNITY.txt`](LICENSE-UNITY.txt)): they may be used only to produce or validate the benchmark's metrics, and not as training, fine-tuning or other input to any AI model, except as the reference against which a model's independently generated output is compared.

## Citation

```bibtex
@article{hu2026wmeval,
  title   = {WMEval: Evaluating Interactive World Models Grounded on Physical Simulations},
  author  = {Hu, Lanxiang and Huang, Yixin and Shankarampeta, Abhilash and Dai, Zilin and
             Wu, Yuyuan and Qian, Kun and Liu, Junze and Srinivasan, Arvind and Huo, Mingjia and
             Lin, Will and Zhong, Kai and Zhang, Weiqi and Wang, Tian and Cui, Qingjun and
             Rosing, Tajana and Zhang, Hao},
  journal = {arXiv preprint arXiv:2610.33894},
  year    = {2026}
}
```
