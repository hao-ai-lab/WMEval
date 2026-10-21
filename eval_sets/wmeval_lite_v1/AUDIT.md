# WMEval Lite v1 dataset audit (2026-10-21)

Every evaluation item -- the 32 core cases (`test_core`) and the 220 synthetic
trajectories (`test_full`) -- was checked against its Unity reference, fixed where it
was not sensible, and re-audited. Reproduce with `tools/audit_dataset.py`; results are in
`audit/` (`audit_summary.json`: counts and the initial findings; `audit_results.jsonl`:
the final per-item verdicts, GT flags and VLM checks).

## Method

Per item, two independent checks (`tools/audit_dataset.py run`):

* **Unity ground truth** (`gt`): object transforms projected through the recorded camera
  (does the tracked/controlled object move, and is it in view while moving), per-family
  physics checks (car stuck or against a barrier, lane changes and steering performed,
  ball leaving the BeamDomino platform while inputs are still commanded, causal segments
  from the Unity log, bottle tilt/hover, liquid masks), frozen-video runs while controls
  are active. Thresholds are constants at the top of the tool.
* **VLM judge** (`vlm`, NVIDIA endpoint `azure/openai/gpt-6-astra`, JSON mode): the
  full-resolution conditioning frame plus 28 frames spanning the reference (denser around
  strikes, jolts and jumps), the prompt, the commanded actions and what each scored
  dimension needs. It enumerates every prompt claim (supported / contradicted /
  never exhibited) and judges action match, prompt match, per-dimension meaningfulness and
  reference reliability (minor vs major).

An item is a *problem* if either source reports a major issue; every problem was then
checked by hand (contact sheets, GT traces) before fixing or accepting it.
`audit_dataset.py screen --family F` drives the synthetic fix loop: generate, render the
changed trajectories in Unity, publish, GT-check, and record failing plans in
`synthetic/screening.json`, which the generator swaps for the next-best plan of the same
stratum (other trajectories keep their plan and id).

## Initial findings (round 1)

Round 1 (`audit_summary.json` `round_1`): 167 of 252 items had at least one major flag,
47 minor only, 38 none. After manual review the confirmed problems were:

| Items | Category | Finding |
|---|---|---|
| core: LiquidContainer x4 | prompt | "rotates in response to control inputs" -- no profile rotates the bottle |
| core: SunRisesAndSets/lighting (+20 camera_orbit) | prompt | 2 m radius / 1.6 m eye / low morning sun / lamp in the centre / bar stools vs. the frame's 2.82 m / 2.31 m / 68 deg sun / lamp at the edge / two chairs |
| core: {Billiard,Mirror,Garden}/scene_{360,perturb}_out (+5 camera_yaw) | prompt | "outward-facing away from the targets" while the frame shows them; Garden "initially facing the cherry tree" while it is behind the camera |
| core: MirrorRollingBall/lpips_collision (+14 ball_force) | prompt | "low camera, chrome ball in the centre foreground" vs. an elevated overview with a small ball in the grid |
| core: RacingCar/kinematics_third (+28 ego_vehicle) | prompt | unmentioned red cube over the road |
| core: BeamDomino x3 (+25 beam_domino) | prompt | "full chain visible in one shot" -- the follow camera leaves the structure and the ball falls off the platform |
| core: BallRoller/ctrl | prompt / dimension | "collecting pickups" (none exist); kinematics tracks only the static dome and cube |
| cue_ball: 19/25 | dimension | tracked yellow ball never hit (15) or in view for < 0.5 s of its motion (4) |
| ego_vehicle: 19/28 | action / reliability | car steered into a roadside barrier and stuck (steer 0.3-1.0 turned it 45-160 deg; "lane changes" crossed the 24 m carriageway) |
| ball_force: 28/28 | action / prompt | directions named in world axes: BallRoller's camera looks along +X (a "forward" push moves screen-left), MirrorRollingBall's is rotated 45 deg |
| ball_force: 6/28 | dimension | controlled ball out of view for > 25% of the clip (one leaves the arena) |
| beam_domino: 17/25 | action / causality | ball rolls off the platform while pushes are still commanded for > 2 s (5 of them also start no causal chain; their causality opt-out was correct) |
| liquid_container: 25/25 | prompt / reliability | rotation claim; replay physics: forward/back motion tipped the bottle 46-58 deg and left it tipped (12), smaller permanent tilts (4), an upward jolt left it hovering (2) |
| camera_yaw: 19/45 | prompt | "turns ... to look directly behind" after an earlier turn does not end facing behind |
| camera_yaw: 5/45 (ChessBoard/prompt_first) | dimension | character consistency configured although the first-person camera has no visible character |
| character: 24/24 | reliability (accepted) | the follow camera clips through giant foreground statues, as in core ChessBoard/walk_third |

No near-duplicate plans were found (the generator's distinct-signal test holds).

## Fixes

**Core prompts** (17 cases): minimal sentence edits, listed old -> new in
`DIFFERENCES_FROM_PAPER.md` (Prompt corrections); evaluator fingerprints re-authored with
`tools/build_lite_case.py`. Reference videos, frames, actions and judge outputs untouched.

**BallRoller/ctrl kinematics** (Lite-only cell): now tracks the controlled white ball instead
of the static dome/cube, against `kin_*` artifacts from `tools/build_ballroller_kinematics_gt.py`
(see `DIFFERENCES_FROM_PAPER.md`). A Unity replay of the core key script could not provide the
ground truth -- the committed scene's rack differs from the recording, and after contact the
replayed ball is 100-400 px from the recorded one -- so the ball is tracked in the reference
video (checked frame by frame) and lifted through the recovered Unity camera, with a floor-plane
calibration for metric units. Self-match vs negative: FDE 0.24 vs 6.50 m, DTW 0.26 vs 2.70 m.

**Synthetic** (`tools/generate_synthetic_trajectories.py`, `synthetic/screening.json`):

* *Screening*: plans whose render fails a GT check are listed in `synthetic/screening.json`
  (210 render-verified rejections with reasons) and swapped, slot by slot, for the closest
  acceptable plan of the same stratum (other trajectories keep their plan and id).
  Replacements also follow a family rule where failures were systematic -- beam_domino: at
  most 5 s of pushing; ego_vehicle: at most one lane change or steer, not ending on a lane
  change. Replaced: beam_domino 17, ball_force 11 (the whole ball must stay in the fixed view:
  centre in view for >= 95% of the clip, never cut by the border in > 3% of frames),
  character 3 (a walk/sprint step blocked: < 40% of its nominal distance).
* *cue_ball grammar*: the yellow ball is the tracked object and the cue ball carries the
  camera (it is never in view, so tracking it instead is impossible). The first strike now
  goes straight at the yellow ball at light/medium strength, and a strike holds its input
  for one control step (v2 held it 0.5 s, so every following strike came 0.5 s later than
  the sentence implied). All 25 plans re-selected (held-impulse signal for the diversity
  selection), re-rendered and screened: yellow ball moves >= 0.15 m and is in view for
  >= 1 s of its motion, no scene reset (the game reset the table 0.1-0.3 s before the end in
  two renders), cue ball stays on the table, no two references with the same yellow-ball
  path or near-identical actions.
* *ego_vehicle grammar recalibrated* from the Unity GT (about 70 deg/s yaw per unit steer):
  steer 0.1/0.2/0.35 for 1-2 s, "changes lanes" = +/-(1.2/T^2) steering over 2-3 s. All 28
  plans re-selected and re-rendered, then screened: no barrier contact or stuck car, a lane
  change moves 1.5-9 m and ends within 8 deg of the road (no drift > 2 m while driving
  straight afterwards), a steer turns the car the commanded way.
* *liquid_container*: the replay now freezes the bottle's rotation and enables gravity
  (`freeze_rotation`/`use_gravity` schedule params, `WMEvalReplayCapture.cs` on the Unity
  branch; the committed scene's Rigidbody has neither, unlike the core recordings).
  Core-trajectory self-checks against the core videos: the jump in fluid_impulse_decay now
  falls back (PSNR returns to 40 dB after it) and fluid_step_hold returns to 40-43 dB after
  the strafe. All 25 re-rendered; SAM2/RAFT masks and empty contact events rebuilt.
* *Prompts*: ball_force directions named as seen from each case's camera ("As seen from
  the camera, the ball is pushed left ...", controls unchanged) and "receives no input"
  instead of "rolls without input" (the ball starts at rest); camera_yaw "turns left
  through a half turn"; scene descriptions follow the corrected core prompts (liquid adds
  that the camera travels with the bottle; beam_domino that the ball may roll off the
  platform).
* *Metric configuration*: ChessBoard/prompt_first (first person) no longer configures
  character consistency (`suite.yaml` `synthetic.module_params`).
* *Judge-authored inputs* re-run only for trajectories whose reference changed (NVIDIA
  gateway, pinned gpt-5.4): causal QA + 5 recorded passes and LPIPS contact events for 17
  beam_domino, contact events for 15 ball_force (6 + 9 in the follow-up round), the
  WanderingGarden outward rubric. Measured spend: $50.24 in 1116 gpt-5.4 calls ($36.23 in the
  first round, $14.01 in the follow-up; `build_synthetic_reference_inputs.py usage`).

## Accepted as-is

* Car-kinematics measurement design (candidate pixels lifted through the Unity camera, so
  a follow-camera trajectory cannot discriminate): out of scope, unchanged.
* BallRoller/ctrl action timing (core, found in the follow-up re-audit): the recording was
  driven more slowly than its commanded 2.96 s key script -- the ball is still accelerated
  (to ~4.4 m/s on the flat floor) until about 5.3 s, during the trajectory's zero-input tail.
  Fixing it means re-deriving the realized key timing, which would change the paper-parity
  controllability actions every model is generated from; left for a separate decision.
* RacingCar/kinematics_third (paper recording) ends against the right barrier from 6.3 s;
  physically consistent, so kept.
* ChessBoard/walk_third and the character family: the Cinemachine camera clips through
  foreground statues when it passes them, and the mannequin is small; the anchors (sphere,
  person) stay detectable, so the object-consistency score is still meaningful.
* WanderingGarden outward Table 7 cells (and their camera_yaw trajectories): the tree
  anchor is not in the first frame, so only lanterns/rocks are tracked from frame 0.
* BeamDomino (core and synthetic): the follow camera stays on the ball after it rolls off
  the floating platform; inputs may continue for up to 2 s after that, as in the core case.
* cue_ball: the camera rides on the cue ball with a fixed heading, so the yellow ball leaves
  the view after its first cushion; every reference keeps it in view for >= 1 s of its
  motion (GT), which the kinematics FDE/DTW is computed on.
* camera_orbit: lighting is reference-free; the Unity orbit (whose floor shadow is cropped
  at the bottom edge at times, as the pivot is 0.63 m off the table axis) is not scored.
* BeamDomino synthetic: the ball can be hidden behind the roof/platform for a moment, and
  the VLM judged direct-strike/glancing chains against the canonical one; the CausalStep QA
  is shape-specific (built from the Unity log), so these are not defects.
* LPIPS contact events with an empty consensus (judge passes disagree) on some
  beam_domino/ball_force references with visible contacts fall back to temporal LPIPS, as
  the core LiquidContainer cases do.
* Kinematics self-match vs negative on the final references (cue_ball s00 scored against
  itself vs against s12): FDE 0.009 vs 14.7 m, DTW 0.005 vs 6.4 m. The mechanics error rate
  does not separate them (52% vs 57%), a property of that metric, outside this audit.

## Re-audit

After the fixes every changed item was re-audited (rounds 2-8; the final verdict per item
is the latest round, all 252 records match the current videos and prompts) and the GT checks
were re-run on all 220 synthetic references: **0 major GT flags**. Final VLM verdicts:
112 ok, 108 minor-only (e.g. "the shadow is crisper than described", sub-second timing
uncertainty), **0 problems**, and 32 major VLM findings reviewed by hand and accepted for the
reasons in "Accepted as-is" (6 core, 6 beam_domino, 7 camera_orbit, 5 camera_yaw, 2
character, 6 cue_ball; listed with reasons in `audit_summary.json` `accepted`). VLM usage
over all rounds: 853 calls, 5.51 M input / 1.33 M output tokens (`azure/openai/gpt-6-astra`).
