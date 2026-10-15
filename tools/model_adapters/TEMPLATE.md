# Adding a new model to WMEval Lite

A new model needs only three things: a registry entry, a generator, and the Lite
runner. Nothing here changes a case's trajectory or reference.

1. **Register the backend** in `experiments/tables_4_7/action_backends.yaml`:
   interface, native frames / fps, and for every action target
   (`camera`, `ego_vehicle`, `cue_ball`, `ball`, `table_objects`, `rolling_ball`,
   `liquid_container`, `character`) either a `supports:` status or an
   `unsupported:` reason. Unsupported cases are reported N/A, never scored with a
   substituted control. Use `real_fps` when the container fps is not the
   generation rate.
2. **Export the jobs:**
   `python tools/expected_jobs.py --model <name> --output runs/<name>` writes
   `expected_jobs.json` (case x seed, with first frame, prompt, action files,
   reference timing) and `candidates.template.jsonl`.
3. **Generate** each job's video at its `candidate_video` path. Compile the
   case's `trajectory.yaml` / `action.npz` onto the model's native controls;
   never alter the trajectory. Record a receipt (inputs, seed, model revision)
   next to each video. Worked generators: `tools/prepare_matrix_game_inputs.py`,
   `tools/prepare_lingbot_table7_inputs.py`, `tools/generate_waypoint15_candidates_modal.py`,
   `tools/odyssey_agent.py` (browser).
4. **Score:** `python tools/preflight_lite.py` (if present), then
   `python tools/run_wmeval_lite.py run --candidate-manifest runs/<name>/candidates.template.jsonl --output-dir reports/wmeval_lite/<name> --resume`.
   A dimension headline is reported only when every applicable case has the
   suite's `repeat_policy.required_repeats` seeds; otherwise it is kept as
   `provisional_value`. Rows outside the reference's fps / size / duration are
   flagged in `input_contract` (`within_spec: false`).
