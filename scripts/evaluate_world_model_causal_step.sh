python tools/legacy/evaluate_world_model_causal_step.py \
  --manifest   runs/manifest.jsonl \
  --cache-root runs/cache \
  --judge      openai \
  --out        runs/causal_step_results.json
