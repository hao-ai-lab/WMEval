# Unity self-match bounds (evaluator error floor)

Each case's Unity reference video was scored as if it were a candidate
(`model: unity`, one pass, 2026-10-15, `--device mps`). A perfect world model
cannot score better than this, so model differences smaller than these values
are within the evaluator's own error on that scene. Reproduce:

```bash
python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \
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
