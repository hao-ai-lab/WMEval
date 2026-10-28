python tools/legacy/evaluate_world_model_lpips.py \
  --manifest_jsonl /path/to/world_model_lpips_manifest.jsonl \
  --output_jsonl reports/world_model_lpips/results.jsonl \
  --summary_csv reports/world_model_lpips/summary.csv \
  --eval_fps 4 \
  --max_pairs 32
