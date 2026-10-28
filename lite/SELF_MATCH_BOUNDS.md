# Unity self-match bounds (evaluator error floor)

Each case's Unity reference video was scored as if it were a candidate
(`model: unity`, one pass, 2026-10-15, `--device mps`). A perfect world model
cannot score better than this, so model differences smaller than these values
are within the evaluator's own error on that scene. The judge-scored dimensions
(prompt-scene consistency, physical causality, causal LPIPS) are covered by
repeated judge passes in [Judge-scored dimensions](#judge-scored-dimensions-2026-10-22)
at the end. Reproduce the first two sections:

```bash
python tools/run_wmeval_lite.py run --suite lite \
  --candidate-manifest runs/unity_selfmatch/candidates.jsonl \
  --dimensions environment_object_consistency,kinematics_mechanics \
  --output-dir reports/wmeval_lite/unity_selfmatch --device mps
```

(`runs/unity_selfmatch/candidates.jsonl` lists each case's `reference_video` as
the candidate.) "Measurable" is the anchor-validity diagnostic: at least two
anchors with 3D points and paired-frame coverage of at least 0.5.

## Environment object & character consistency

| Case | Depth AbsRel % | Pairwise dist. MAE % | Measurable (coverage) | Object permanence % | Character consistency % |
|---|---|---|---|---|---|
| BallRoller/ctrl | 61.8 | 60.4 | yes (1.0) | 100.0 | 100.0 |
| BilliardGame/scene_360_in | 22.9 | 9.1 | yes (0.7) | 100.0 | 77.3 |
| BilliardGame/scene_360_out | 22.2 | 38.9 | yes (0.8) | 100.0 | 85.3 |
| BilliardGame/scene_perturb_in | 22.6 | 36.0 | yes (1.0) | 100.0 | 100.0 |
| BilliardGame/scene_perturb_out | 29.3 | 7.4 | yes (1.0) | 100.0 | 100.0 |
| ChessBoard/walk_third | 42.1 | — | no (0.0) | 100.0 | 43.3 |
| MirrorRollingBall/scene_360_in | 44.5 | 37.3 | yes (0.5) | 56.2 | 56.7 |
| MirrorRollingBall/scene_360_out | 44.5 | 21.6 | yes (0.6) | 56.8 | 56.3 |
| MirrorRollingBall/scene_perturb_in | 42.1 | 33.2 | no (0.1) | 52.8 | 100.0 |
| MirrorRollingBall/scene_perturb_out | 34.8 | 31.7 | no (0.1) | 53.7 | 100.0 |
| RacingCar/prompt_third | 12.7 | — | no (0.0) | 100.0 | — |
| WanderingGarden/scene_360_in | 93.0 | 41.9 | no (0.1) | 20.5 | — |
| WanderingGarden/scene_360_out | 138.5 | 52.9 | no (0.1) | 20.0 | — |
| WanderingGarden/scene_first | 25.3 | 34.8 | no (0.0) | 4.5 | — |
| WanderingGarden/scene_perturb_in | 222.2 | 31.2 | no (0.1) | 22.5 | — |
| WanderingGarden/scene_perturb_out | 87.5 | 84.9 | no (0.0) | 37.6 | — |

Reading: WanderingGarden's depth and distance floors (87–222 % and 31–85 %) are
as large as the model values in Table 7, its anchors are detected in only
5–38 % of Unity frames, and pairwise distance is not measurable on any Garden
cell. Before comparing models on Garden depth / distance, check the Garden
bridge ground-truth alignment; until then treat those cells as not separable.
MirrorRollingBall perturbation cells are also not measurable for pairwise
distance (coverage 0.06–0.07).

## Kinematics & mechanics

| Case | FDE (m) | DTW (m) | Error rate % | Exact match % |
|---|---|---|---|---|
| BilliardGame/kinematics_first | 0.008 | 0.008 | 52.4 | 0.0 |
| RacingCar/kinematics_third | 0.733 | 0.603 | 29.2 | 0.0 |

Reading: the metric-space FDE / DTW floors are small relative to model errors
(Billiard 0.008 m; RacingCar 0.6–0.7 m vs. 2–5 m for models). The error-rate
floor (29–52 %) is far above the paper's model values (2–5 %), so the Lite
`error_rate_pct` does not yet match the paper's definition; reconcile before
claiming Table 4 parity on that column.

## Judge-scored dimensions (2026-10-22)

`prompt_scene_consistency`, `physical_causality` and `lpips` depend on a sampled
hosted judge (`gpt-5.4` rejects a non-default temperature, so every call is
drawn at the API default). Here only the judge is re-sampled: the same video is
scored K times, through the NVIDIA inference gateway (`tools/judge_gateway.py`,
model `azure/openai/gpt-5.4`, pinned prompts/rubrics/QA unchanged).

- **Reference self-score**: each case's Unity reference video scored as the
  candidate, K = 5 (core) or K = 3 (synthetic sample). For `physical_causality`
  one repeat is a full dimension run (`causal_step_judge_repeats: 5` passes,
  fresh answer cache per repeat), so its std is the std of the reported number.
- **Candidate judge spread**: Odyssey's 30 stored prompt videos (6 cases x 5
  seeds), K = 3 each; the pooled within-video std. Odyssey has no supported
  `lpips` / `physical_causality` rows (the action-target gate: no object
  channel), and no other model's BeamDomino / ball videos are stored locally, so
  the candidate side of those two dimensions is not measured.
- **sigma_judge** = max(reference self-score std, candidate judge std).
  **MDD_n** = 1.96 * sqrt(2/n) * sigma_judge: the smallest difference between two
  models' n-seed means (each seed judged once) that judge sampling alone produces
  less than 5 % of the time. MDD_1 compares single videos; MDD_5 is the paper's
  5-seed protocol. Judge noise only: generation (seed) variance comes on top.

Machine-readable: `judge_noise_floor.json` in the [Hub dataset](https://huggingface.co/datasets/lmgame/WMEval-Lite) (written here by `tools/dataset/measure_judge_floor.py`, not tracked in git). Raw rows, usage
logs and the full per-video summary: `reports/judge_noise_floor/` (not tracked).
Reproduce:

```bash
python tools/dataset/measure_judge_floor.py plan --set core --repeats 5 --out reports/judge_noise_floor/core
python tools/dataset/measure_judge_floor.py plan --set synthetic --repeats 3 --per-case 2 \
  --dimensions prompt_scene_consistency --out reports/judge_noise_floor/synthetic
python tools/dataset/measure_judge_floor.py plan --set synthetic --repeats 3 --per-case 3 \
  --dimensions lpips,physical_causality --out reports/judge_noise_floor/synthetic
python tools/dataset/measure_judge_floor.py plan --set candidates --repeats 3 \
  --from runs/odyssey/candidates_prompt_v2.jsonl --dimensions prompt_scene_consistency \
  --out reports/judge_noise_floor/odyssey
# each manifest, e.g.:
WMEVAL_VLM_USAGE_LOG=reports/judge_noise_floor/core/usage.jsonl \
python tools/judge_gateway.py --key-file ~/.credentials/nv_vlm_api_key -- run \
  --candidate-manifest reports/judge_noise_floor/core/lpips.jsonl --dimensions lpips \
  --output-dir reports/judge_noise_floor/runs/core_lpips --device mps   # (--split synthetic for synthetic)
python tools/dataset/measure_judge_floor.py summarize --compact \
  --selfmatch reports/judge_noise_floor/runs/{core,syn}_*/per_case_results.jsonl \
  --candidates reports/judge_noise_floor/runs/ody_*/per_case_results.jsonl \
  --stored nvidia_2026-10-15=reports/wmeval_lite/odyssey_v2_prompt/per_case_results.jsonl \
           openai_2026-10-12=reports/wmeval_lite/odyssey_judge/per_case_results.jsonl \
  --output lite/judge_noise_floor.json
```

Spend: 1,251 `gpt-5.4` calls, 18.2 M input tokens (8.4 M cached), 0.64 M
output tokens, about $36 at list prices (pilot $0.7, core $10.4, synthetic
$14.7, Odyssey re-score $10.4).

### Prompt-scene consistency (`overall_score`, higher is better)

| Case | Unity self-score, NVIDIA K=5: mean ± std [range] | Frozen OpenAI reference (5 passes, `reference_scores.json`) | Odyssey judge std (5 videos x K=3) | sigma_judge | MDD_1 | MDD_5 |
|---|---|---|---|---|---|---|
| ChessBoard/prompt_first | 0.984 ± 0.036 [0.92, 1.00] | 1.000 ± 0.000 | 0.025 | 0.036 | 0.099 | 0.044 |
| ChessBoard/prompt_third | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.018 | 0.018 | 0.050 | 0.022 |
| MirrorRollingBall/prompt_first | 0.953 ± 0.018 [0.93, 0.97] | 0.953 ± 0.018 | 0.022 | 0.022 | 0.062 | 0.028 |
| MirrorRollingBall/prompt_third | 0.995 ± 0.012 [0.97, 1.00] | 1.000 ± 0.000 | 0.032 | 0.032 | 0.088 | 0.039 |
| RacingCar/prompt_first | 0.827 ± 0.060 [0.80, 0.93] | 0.840 ± 0.112 | 0.038 | 0.060 | 0.165 | 0.074 |
| RacingCar/prompt_third | 0.892 ± 0.081 [0.80, 1.00] | 0.837 ± 0.053 | 0.023 | 0.081 | 0.224 | 0.100 |

Reading:
- One judge call moves `overall_score` by 0.02–0.04 (std) on ChessBoard and
  MirrorRollingBall and by 0.04–0.08 on RacingCar (the frozen OpenAI passes on
  RacingCar/prompt_first spread even more, 0.11, so its MDD_5 may be up to 0.14).
  A single video's score is good to about ±0.1.
- The Unity reference itself does not reach 1.0 on RacingCar (0.83–0.89) or on
  MirrorRollingBall/prompt_first (0.95, the object rubric): rubric strictness, a
  ceiling for every model, not a model error.
- Transport: NVIDIA and the frozen OpenAI passes agree on the reference within
  noise (per-case differences -0.016 … +0.055, average +0.003; no case is
  significant with Welch at n = 5 + 5). Odyssey re-score vs the stored results
  (same videos, same rubric and judge identity): vs the stored NVIDIA run of
  2026-10-15 (5 seeds) the per-case mean moved by -0.023 … +0.024 (average
  -0.005) and 23/30 stored single-pass scores fall inside the K=3 re-score range;
  vs the stored OpenAI run of 2026-10-12 (run0 only, 6 videos) per-video
  differences are -0.067 … +0.033 (average -0.007, mean |d| 0.033, i.e. one
  sigma). No transport bias is detectable; the transports are interchangeable
  at the resolution of the judge noise.

Paper models + Odyssey (6 models, 15 pairs per cell, 90 pairs): 32 pairwise
differences of 5-seed means are below MDD_5, i.e. inside judge noise alone:

| Cell | MDD_5 | Pairs within judge noise (difference of means) |
|---|---|---|
| ChessBoard/first | 0.044 | LingBot–Waypoint 0.003, Genie–LingBot 0.024, Genie–Waypoint 0.021, MG3–Odyssey 0.024 |
| ChessBoard/third | 0.022 | MG3–Waypoint 0.002, MG3–Odyssey 0.019, Odyssey–Waypoint 0.021 |
| MirrorRollingBall/first | 0.028 | HY–Waypoint 0.010, Genie–Odyssey 0.013 |
| MirrorRollingBall/third | 0.039 | HY–LingBot 0.005, HY–Genie 0.032, LingBot–Genie 0.037, MG3–Waypoint 0.011, Genie–Odyssey 0.016 |
| RacingCar/first | 0.074 | 8 of 15: every pair among Genie, LingBot, Waypoint, Odyssey except LingBot–Waypoint; MG3–HY, MG3–Waypoint, MG3–Odyssey |
| RacingCar/third | 0.100 | 10 of 15: every pair among Genie, LingBot, HY, Waypoint, Odyssey (only MG3 separates) |

All 32 are already ties under the leaderboard's Welch test (seed std 0.02–0.10
already contains one judge draw per seed), so no current rank rests on a
difference inside the judge floor. RacingCar is the cell where the prompt
dimension cannot order the top five models at all.

### Physical causality (`csr_delta`, lower is better; Unity side = recorded fixture)

| Case | csr_wm (Unity as candidate), K=5 runs x 5 passes | csr_unity (recorded) | csr_delta mean ± std [range] | Other deltas (mean ± std) | Items answered correctly (all 25 passes) | MDD_1 / MDD_5 (csr) |
|---|---|---|---|---|---|---|
| BeamDomino/causal | 100 ± 0 | 100 | 0 ± 0 | all 0 | 175/175 | 0 / 0 |
| BeamDomino/causal_direct_strike | 100 ± 0 | 100 | 0 ± 0 | dua 1.3 ± 3.0 | 124/125 (seg3 descriptive 24/25) | 0 / 0 |
| BeamDomino/causal_glancing | 84 ± 17 [60, 100] | 80 | -4 ± 17 [-20, +20] | amcl -0.04 ± 0.17, ws -0.16 ± 0.67, dua 0 ± 3.5 | 170/175 (seg1 descriptive 21/25, seg3 descriptive 24/25) | 46 / 21 |

Reading: canonical and direct-strike are judge-stable on the reference. On
glancing, one dimension run (5 passes) moves CSR by ±17 points because the
whole chain hinges on one descriptive item answered right 84 % of the time; a
csr_delta of 20 on glancing from one run is within judge noise, and 5-seed
means need about 21 points. NVIDIA (live) and the recorded OpenAI passes agree
(84 vs 80). The paper's only physical-causality row (Project Genie, CSR
20 ± 41, n = 15) has nothing to be compared with, so no current difference is
affected.

### LPIPS (causal; reference events frozen, candidate events judged live)

| Case | causal LPIPS self-score, K=5: mean ± std [range] | Primary source | Matched ref. fraction | Candidate events found | abs(t_cand - t_ref) of matched events (s) | MDD_1 / MDD_5 |
|---|---|---|---|---|---|---|
| BallRoller/ctrl | 0.0005 ± 0.0012 [0, 0.003] | causal 5/5 | 1.0 | 1.4 ± 0.5 | 0.03 ± 0.06 | 0.003 / 0.002 |
| BeamDomino/causal | **0.344 ± 0.011** [0.338, 0.363] | causal 5/5 | 1.0 | 1.8 ± 0.4 | **0.87 ± 0.14** | 0.030 / 0.013 |
| MirrorRollingBall/lpips_collision | 0.027 ± 0.006 [0.023, 0.038] | causal 5/5 | 1.0 | 4.0 ± 1.0 | **7.0 ± 0.7** | 0.018 / 0.008 |
| LiquidContainer/* (4, excluded from headline) | temporal fallback 0.010–0.017, std 0 | temporal_only 20/20 | 0 | 0–2 | — | — |

Reading: the judge-sampling spread is small, but two of the three headline
cases carry a **systematic** self-match error, which is the larger problem:
- BeamDomino/causal scores 0.34 against itself. Its frozen reference event set
  (`contact_events.json`, 5 passes, min support 3) kept only the later domino
  contact (frame 231); the pillar strike at frames 170–201 was split across
  side and frame clusters (support 1 + 2 + 1) and dropped. The live judge reports the
  pillar strike first, chronological pairing pairs it with the frozen domino
  event 0.8 s later, and LPIPS is taken between different moments. The paper's
  models score 0.60 / 0.61 here, so more than half of that number is this
  pairing artefact, and Project Genie vs LingBot (0.01 apart) is inside the
  judge noise (MDD_5 0.013).
- MirrorRollingBall/lpips_collision: the single frozen event is the final wall
  contact (about 8 s); the live judge finds 3–5 earlier contacts and the first
  one (0.5–2.3 s) is paired with it. LPIPS stays low only because the high,
  static camera makes distant frames look alike, so on this case the metric
  mostly measures appearance, not contact timing. "Matched fraction 1.0" hides
  both failures; `event_offset_sec` in `judge_noise_floor.json` exposes them.
- LiquidContainer: no frozen reference contacts, so the row is always the
  deterministic temporal fallback (the 0.01–0.017 floor is the re-encode of the
  832x464 / 16 fps normalised copy); the judge has no influence.
- The candidate-side judge is likely less stable than on the clean reference
  (the 2026-09-24 Waypoint run saw 0/1/3/0/0 contacts on one video), and that
  side was not measurable here. Treat these MDDs as lower bounds.

### Synthetic sample (K = 3)

| Family / conditioning case | Trajectories | Headline | Self-score mean [per-trajectory range] | Pooled judge std | MDD_1 / MDD_5 |
|---|---|---|---|---|---|
| camera_yaw / BilliardGame scene_360_in | 2 | overall | 1.000 [1.00, 1.00] | 0.000 | 0 / 0 |
| camera_yaw / ChessBoard prompt_first | 2 | overall | 0.993 [0.99, 1.00] | 0.016 | 0.045 / 0.020 |
| camera_yaw / ChessBoard prompt_third | 2 | overall | 0.989 [0.98, 1.00] | 0.027 | 0.075 / 0.034 |
| camera_yaw / MirrorRollingBall prompt_first | 2 | overall | 0.936 [0.92, 0.95] | 0.036 | 0.100 / 0.045 |
| camera_yaw / MirrorRollingBall prompt_third | 2 | overall | 0.993 [0.99, 1.00] | 0.016 | 0.045 / 0.020 |
| camera_yaw / RacingCar prompt_first | 2 | overall | 0.778 [0.73, 0.82] | 0.027 | 0.075 / 0.034 |
| camera_yaw / RacingCar prompt_third | 2 | overall | 0.817 [0.80, 0.84] | 0.030 | 0.083 / 0.037 |
| camera_yaw / WanderingGarden scene_360_in | 2 | overall | 0.848 [0.83, 0.86] | 0.058 | 0.160 / 0.071 |
| camera_yaw / WanderingGarden scene_360_out | 2 | overall | 0.760 [0.75, 0.77] | 0.023 | 0.064 / 0.029 |
| beam_domino / BeamDomino causal | 3 | csr_delta | 0 [0, 0] | 0 | 0 / 0 |
| beam_domino / BeamDomino causal | 3 | causal LPIPS | s00 0.161 (std 0); s01, s02 no frozen events (temporal only) | 0 | — |
| ball_force / BallRoller ctrl | 3 | causal LPIPS | 0.007 [0.006, 0.008] | 0.004 | 0.011 / 0.005 |
| ball_force / MirrorRollingBall lpips_collision | 3 | causal LPIPS | s03 0.004; s01, s05 no frozen events (temporal only) | 0.0002 | — |

Reading:
- The WanderingGarden 0.77 is reproducible (scene_360_out 0.75 / 0.77, env
  score 0.58–0.62 with the object score at 1.0): a rubric ceiling, not noise.
  WanderingGarden and RacingCar synthetic prompts cap a perfect model at
  0.75–0.86.
- beam_domino s00's LPIPS 0.161 against itself is deterministic across repeats
  (same pairing artefact as the core case: frozen event at frame 124, live
  first event at 109). 4 of the 9 sampled ball/beam trajectories have an empty
  frozen reference event set, so they can never produce causal LPIPS.
- beam_domino s01 / s02 CSR is uninformative: the segment-1 descriptive item is
  answered "D" in 15/15 passes on the Unity reference (keys C / B), so
  csr_unity = csr_wm = 0; the segment-3 causal item is a coin flip (7/15, 6/15),
  which moves icra_delta by -13 ± 6 and -20 ± 17 between the live judge and the
  recorded reference answers.

### What this means for comparisons

- Prompt-scene: treat two models' 5-seed means as indistinguishable when they
  differ by less than 0.02–0.04 (ChessBoard, MirrorRollingBall) or 0.07–0.10
  (RacingCar); single videos by less than 0.05–0.22.
- Physical causality: deltas on canonical / direct-strike are judge-stable on
  the reference; on glancing a 5-seed csr difference below about 21 points is
  judge noise.
- LPIPS: judge sampling is a small effect next to the systematic reference-
  event mismatch on BeamDomino/causal (floor 0.34) and the mis-pairing on
  MirrorRollingBall/lpips_collision. Re-freeze those reference events (side-
  agnostic clustering) or pair events by time rather than order before ranking
  models on these two cells.
