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
| Table 4/7 cells | canonical trajectories in `experiments/protocol/trajectories/` | Commanded timing; browser models record realized timing separately |
| Prompt consistency | `prompt.camera.left_turn_v1` (10 s at 40 deg/s) | The paper's rows used per-model as-run camera inputs (`experiments/protocol/prompt_as_run_inputs.yaml`); they are kept as-run. New models use the canonical turn |
| Causality | `ball_force_v1`, exact (bit-for-bit the capture's `action.npy`) | None |
| Fluid | `bottle_motion_v1`, nominal 12 Hz controller profile | The Unity capture stepped on wall-clock (~10 Hz); the legacy `action.npy` keeps that record |
| Controllability / LPIPS | `ball` trajectories (BallRoller, MirrorRollingBall); BeamDomino shares the causality trajectory | None for scoring; `candidate_normalization: none` kept for paper parity |
| Lighting | `SunRisesAndSets/lighting`: the paper's case, prompt, `ref.png` and per-model action records (`actions.txt`: hold ArrowLeft 10 s), scored with the paper's reference-free protocol | None for scoring (all 180 per-run values of the paper's lighting table reproduce exactly). Added: a Unity reference orbit (`reference/unity_orbit.mp4` + ground truth) rendered from the same first frame; the score does not read it. Its camera is the frame's own (radius 2.82 m, eye height 2.31 m about a tabletop pivot), not the prompt's approximate 2 m / 1.6 m |

## Frozen judge outputs

| Output | Lite v1 | Paper | Effect |
|---|---|---|---|
| Causality reference answers | 5 judge passes on the Unity reference, after repairing four QA items (#34) | single protocol, original QA | Unity CSR 100 / 80 / 100 (Table 12: 100 / 60 / 100); Lite QA differs from the paper's |
| Causality estimator | 5 judge passes over one rollout per candidate | five rollouts | Makes fractional CSR expressible; does not reproduce Table 12 exactly |
| Prompt Unity scores | frozen, 5 passes on the shipped captures (mean 0.938) | earlier capture (0.917) | Unity row differs; model rows reproduce within mean abs. diff 0.023 |
| LPIPS contact events | frozen majority events (BallRoller, BeamDomino, MirrorRollingBall); LiquidContainer frozen with 0 consensus events (no candidate contact reached 3/5 passes) | re-judged per run | LingBot (Act) BeamDomino 0.547 offline vs. 0.61 (event-timing drift); LiquidContainer uses the temporal fallback |

## Prompt corrections (2026-10-21 dataset audit)

The audit (`audit/README.md`, `tools/dataset/audit_dataset.py`) compared every core prompt with its
conditioning frame and Unity reference. Reference videos, first frames, actions and judge
outputs are unchanged; only these sentences of `case.yaml` `prompt` changed (evaluator
fingerprints re-authored with `tools/dataset/build_lite_case.py`). No prompt-consistency case
(`*/prompt_*`) changed, so rubrics and frozen prompt scores are unaffected; no other module
reads the prompt. The paper's runs used the old text (`experiments/protocol/prompt_components.yaml`
keeps the as-run Table 7 components).

| Case(s) | Old | New | Why |
|---|---|---|---|
| LiquidContainer/fluid_{rest,step_hold,impulse_decay,chirp_strafe} | "it rotates in response to control inputs but does not deform." / "casting a hard contact shadow on the floor directly under any object ..." | "it stays upright, never rotates and does not deform." / "casting a soft shadow on the floor to the left of any object ..." | No profile ever rotates the bottle (pitch/yaw/roll are forced to 0); the shadow is soft and falls to the left |
| SunRisesAndSets/lighting | "standing adult eye height, approximately 1.6 metres" / "a constant 2-metre radius ... keeping the table centred" / "lamp in the centre" / "ladder-back bar stools are arranged around the table" / "morning sun ... low above the eastern horizon, casting long ... shadows ... across the wooden floor" | "a raised eye height, approximately 2.3 metres ..., looking slightly down at the table" / "a constant radius of about 2.8 metres ... near the centre of the field of view" / "lamp near the edge of the tabletop" / "Two wooden ladder-back chairs stand at the table" / "sun ... high in the sky (about 68 degrees above the horizon), casting crisp ... shadows ... onto the tabletop and the wooden floor" | `ref.png` is the scene's CameraOrbitAgent view (radius 2.82 m, eye 2.31 m, pitch 20 deg, sun Euler x 68.2); two chairs; lamp at the tabletop edge |
| BilliardGame/scene_{360,perturb}_out | "outward-facing away from the scene's designated target objects and anchors." | "the paper's outward-condition start, seen from the far side of the room by the black-and-white wall painting; the table and all three balls are in view." | The outward conditioning frame shows the table and every ball |
| MirrorRollingBall/scene_{360,perturb}_out | same sentence | "the paper's outward-condition start; the chrome sphere is in the centre of the view, surrounded by yellow spheres." | The outward frame centres the target sphere |
| WanderingGarden/scene_{360,perturb}_out | "initially facing the main cherry tree." / "outward-facing away from the scene's designated target objects and anchors." | "initially facing away from the main cherry tree, toward the rockery waterfall and pond in front of a lit wooden building, with a stone lantern on the left." / "outward-facing, away from the main cherry tree." | The tree is not in the outward frame (it was described as faced) |
| MirrorRollingBall/lpips_collision | "low camera near a flat light gray floor. Center foreground: one ... chrome mirror ball ... stretching into the distance" / "soft shadows toward lower left" / "Dark horizon band and pale sky or wall above." | "a high camera looking diagonally down at a flat light gray floor enclosed by low dark walls. In the middle of the floor, a square grid of ... yellow spheres ...; one small ... chrome mirror ball (the player) sits at the centre of the grid." / "crisp shadows toward lower left" / "A pale blue-gray background above the walls." | Frame 0 is an elevated overview with a small ball in the grid centre |
| RacingCar/kinematics_third | "The road begins abruptly at a starting point, with no ground behind it" / "is parking exactly at the start of the road" / "... into the long empty highway." | "... at a starting point behind the camera, ..." / "is parked on the road a short distance ahead of its starting edge" / "...; a small red cube hovers above the road to the right of the car." | The road start is behind the camera (pavement continues behind the car into the foreground); the capture contains the scene's orbit-anchor cube |
| BallRoller/ctrl | "suitable for collecting pickups and weaving ..." | "suitable for weaving ..." | The scene has no pickups |
| BeamDomino/causal, causal_glancing, causal_direct_strike | "Keep the layout clean and readable so the full physics chain is visible in one shot." / "the camera follows ... throughout." | "Keep the layout clean and readable." / "... throughout, also after the ball has passed the structure and runs off the edge of the floating platform." | The follow camera leaves the structure and the ball falls off the platform in all three recordings |

Accepted as-is (documented in `audit/README.md`): RacingCar/kinematics_third's car ends against the
right barrier from 6.3 s (paper recording); ChessBoard/walk_third's follow camera clips through
foreground statues; the outward Garden frames contain no tree, so the tree anchor cannot be
detected at frame 0 on those Table 7 cells; BallRoller/ctrl's recording was driven more slowly
than its commanded 2.96 s key script (the ball is still accelerated until about 5.3 s), which is
left as is because the controllability actions are the paper-parity ones.

## BallRoller/ctrl kinematics (Lite-only cell, 2026-10-21)

Not in any paper table. Until now the cell tracked only the static dome and cube
(`kin_text_prompt: dome. cube.`), so it measured hallucinated obstacle motion, not kinematics.
It now tracks the controlled **white ball** (`kin_text_prompt: white ball.`, bottom-centre track
point, plane calibration for metric FDE/DTW, mass 1 kg) against new artifacts
`reference/artifacts/kin_{object_transforms.json,poses.npy,intrinsics.npy,object_masses.json,plane_calibration.json}`
built by `tools/dataset/build_ballroller_kinematics_gt.py`. The reference video is unchanged.

A Unity replay of the core key script cannot supply the ball's ground truth: in the committed
scene the rack differs from the recording (its untextured white balls and contact response), so
the replayed ball is within about 10 px of the recorded one for the first second (at 0.8x the
scene's force gain) but 100-400 px away after contact. The ball is therefore tracked in the core
reference video (colour blob, checked frame by frame) and lifted through the recovered Unity
camera (the render-and-compare fit that matches frame 0 at 25.3 dB) onto its centre height.
Self-match vs negative (the core reference vs a synthetic BallRoller reference as candidate):
FDE 0.24 vs 6.50 m, DTW 0.26 vs 2.70 m. The EOC and LPIPS artifacts of the case are unchanged.

## Evaluation behavior added after the paper

- Cases declare an action target; backends that cannot express it are
  `unsupported` (N/A), never scored with a substituted control.
- Every row records `input_contract` (candidate vs. reference fps, size,
  duration), `metric_status`, and `action_support` (e.g.
  `target_entity_validation_required` for Matrix-Game 2.0 object targets).
- `kin_plane_calibration_json` is resolved for the Table 4 cases, so kinematics
  reports metric units.
- The lighting case ships a Unity reference video and camera ground truth
  (`trajectory.yaml` look-at orbit, `action.npz`, per-frame poses, intrinsics,
  calibration, QA) that the paper never had; scoring is unchanged
  (reference-free). The paper's Matrix-Game 2.0 SRR prints 68.76 in its table
  while its per-seed data give 22.92 +/- 16.23 (a table typo; Lite reports the
  per-seed value).

## Judge transport

The pinned judge is `gpt-5.4`. `tools/judge_gateway.py` can route the same
model through an OpenAI-compatible gateway; results produced that way record
`WMEVAL_JUDGE_TRANSPORT` where the tool supports it and should say so in their
notes.

## Known limitations

- **Realized camera motion is not reported beside the scores.** Consistency cases command
  fixed yaw sweeps, but models realize different amounts of rotation (LingBot-World realized
  −17.7°/+41.4° for the ±45° perturbations and +54.9°/+277.9° for the full turns), so
  consistency scores are not normalized for how far the camera actually moved.
- **Target control is checked per backend, not per rollout.** `action_backends.yaml` refuses
  targets a backend cannot express, and RacingCar's road FDE is withheld when road alignment
  fails, but no per-run target-success rate is recorded.
- **Prompt cases.** The bundled trajectory is a 40 °/s left turn for 10 s; the paper models'
  inputs (`tools/generation/prepare_prompt_case_inputs.py`) reproduce their original rollouts instead.
  New models should compile the bundled turn.
- **Physical causality** grades one rollout over five judge passes, a different estimator
  from the paper's five rollouts: it makes fractional CSR expressible but does not reproduce
  Table 12 exactly.
- **Reproduction.** Re-scoring a paper model through the Lite cases and matching the
  published Table 4/7 means (`tools/paper_tables/export_lite_table_results.py` →
  `tools/paper_tables/compare_table_results.py compare`) has not been completed for every model.
