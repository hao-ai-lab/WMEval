python tools/evaluate_world_model_metrics.py \
  --video /path/to/world_model_video.mp4 \
  --ref_video /path/to/reference_video.mp4 \
  --text_prompt "A ball rolls down a ramp" \
  --modules lpips \
  --output_json reports/world_model_metrics/report.json \
  --eval_fps 4 \
  --max_pairs 32
