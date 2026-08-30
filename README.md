# WMEval

Benchmark for evaluating interactive world models against Unity-based physical
simulations. Instead of judging generations on perceptual quality alone, WMEval
executes matched action trajectories in both Unity and the world model, then
compares them using simulator-derived ground truth — object states, camera
poses, masses, causal event traces — as the reference.

## Scenes

9 Unity scenes, each paired with deliberately designed action trajectories:

`BallRoller` · `BeamDomino` · `BilliardGame` · `ChessBoard` · `LiquidContainer` ·
`MirrorRollingBall` · `RacingCar` · `SunRisesAndSets` · `WanderingGarden`

## Dimensions

| Dimension | What it checks |
|---|---|
| Environment Object & Character Consistency | Object/character identity and spatial relations hold under camera motion (depth AbsRel, pairwise-distance MAE, object permanence, DINOv2 identity) |
| Action Controllability | Generated motion follows the given action signal (optical-flow RAFT decomposition) |
| Kinematics & Mechanics | Trajectories and mechanical quantities (momentum, energy) match the simulator |
| Fluid Mechanics | Liquid behavior obeys volume/momentum invariants |
| Physical Causality | Multi-stage causal chains resolve correctly |
| Lighting & Shading | Illumination and cast shadows stay physically consistent |
| Prompt-Scene Consistency | Generation matches the scene description |

## Repository layout

```
evaluation/modules/      metric implementations, one per dimension
evaluation/lite/         case-bundle loading, suite config, CLI adapters
eval_sets/wmeval_lite_v1/ suite.yaml (dimension → case wiring) + case bundles
tools/run_wmeval_lite.py  CLI: run, validate, list-cases, generate-rubrics
```

Each case bundle (`eval_sets/wmeval_lite_v1/cases/<Scene>/<variant>/`) pairs a
reference rollout with Unity-derived ground truth and is checksum-locked via
`tools/build_lite_case.py`.

## Usage

```bash
# Validate the bundled suite data
python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1

# Evaluate a model's rollouts across all wired dimensions
python tools/run_wmeval_lite.py run \
  --suite eval_sets/wmeval_lite_v1 \
  --model-root runs/<model> \
  --output-dir reports/wmeval_lite/<model>
```

Candidate videos are laid out as `<model_root>/<scene>/<variant>/<seed>.mp4`.

## Status

Not every dimension is live for every scene yet — see
`eval_sets/wmeval_lite_v1/suite.yaml` for current wiring and
`eval_sets/wmeval_lite_v1/PENDING.md` for what's still in progress.
