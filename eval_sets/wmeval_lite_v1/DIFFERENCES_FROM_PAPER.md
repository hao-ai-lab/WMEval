# WMEval Lite v1: differences from the paper

The Lite suite reproduces the paper's protocol without Unity. Where it differs,
the difference is listed here with its reason and its effect on numbers. Paper
tables are frozen; nothing below changes a published value unless it says so.

## Scope and roster

| Item | Paper | Lite v1 | Effect |
|---|---|---|---|
| Cases | per-table setups | 32 case bundles (14 Table 4/7 cells baked as `<Scene>/scene_*`, `<Scene>/kinematics_*`) | Same ground truth for the full-turn and captured Table 7 cells; Garden outward anchors are reconstructed |
| Lite-only cases | — | `ChessBoard/walk_third`, `WanderingGarden/scene_first` (object consistency); `BallRoller/ctrl` in LPIPS | Not in any paper table |
| Prompt-driven rows | causality: LingBot-World (Cam); fluid: LingBot-World, Matrix-Game 2.0 / 3.0 | Reported `unsupported` (no ball-force or bottle channel) | The paper revision drops these rows; HY-WorldPlay's fluid row is kept |
| Repeats | five generations | `repeat_policy.required_repeats: 5`; incomplete dimensions report `provisional_value` only | Partial runs never produce a headline |

## Actions

| Dimension | Lite trajectory | Difference from the paper's runs |
|---|---|---|
| Table 4/7 cells | canonical trajectories in `experiments/tables_4_7/trajectories/` | Commanded timing; browser models record realized timing separately |
| Prompt consistency | `prompt.camera.left_turn_v1` (10 s at 40 deg/s) | The paper's rows used per-model as-run camera inputs (`experiments/tables_4_7/prompt_as_run_inputs.yaml`); they are kept as-run. New models use the canonical turn |
| Causality | `ball_force_v1`, exact (bit-for-bit the capture's `action.npy`) | None |
| Fluid | `bottle_motion_v1`, nominal 12 Hz controller profile | The Unity capture stepped on wall-clock (~10 Hz); the legacy `action.npy` keeps that record |
| Controllability / LPIPS | `ball` trajectories (BallRoller, MirrorRollingBall); BeamDomino shares the causality trajectory | None for scoring; `candidate_normalization: none` kept for paper parity |
| Lighting | `lighting_orbit_v2` (fixed sun, 2 m look-at orbit) | The paper's native lighting protocol lives in `lighting_paper_native/` |

## Frozen judge outputs

| Output | Lite v1 | Paper | Effect |
|---|---|---|---|
| Causality reference answers | 5 judge passes on the Unity reference, after repairing four QA items (#34) | single protocol, original QA | Unity CSR 100 / 80 / 100 (Table 12: 100 / 60 / 100); Lite QA differs from the paper's |
| Causality estimator | 5 judge passes over one rollout per candidate | five rollouts | Makes fractional CSR expressible; does not reproduce Table 12 exactly |
| Prompt Unity scores | frozen, 5 passes on the shipped captures (mean 0.938) | earlier capture (0.917) | Unity row differs; model rows reproduce within mean abs. diff 0.023 |
| LPIPS contact events | frozen majority events (BallRoller, BeamDomino, MirrorRollingBall); LiquidContainer frozen with 0 events (5/5 passes) | re-judged per run | LingBot (Act) BeamDomino 0.547 offline vs. 0.61 (event-timing drift); LiquidContainer uses the temporal fallback |

## Evaluation behavior added after the paper

- Cases declare an action target; backends that cannot express it are
  `unsupported` (N/A), never scored with a substituted control.
- Every row records `input_contract` (candidate vs. reference fps, size,
  duration), `metric_status`, and `action_support` (e.g.
  `target_entity_validation_required` for Matrix-Game 2.0 object targets).
- `kin_plane_calibration_json` is resolved for the Table 4 cases, so kinematics
  reports metric units.

## Judge transport

The pinned judge is `gpt-5.4`. `tools/judge_gateway.py` can route the same
model through an OpenAI-compatible gateway; results produced that way record
`WMEVAL_JUDGE_TRANSPORT` where the tool supports it and should say so in their
notes.
