python tools/legacy/visualize_world_model_diffusionlight.py \
  --video /workspace/WMEval/ground_truth_sun_rise.mp4 \
  --output_dir /workspace/WMEval/reports/diffusionlight_viz/ground_truth_sun_rise \
  --diffusionlight_hdr_dir /workspace/unity/output_sun_rise/hdr \
  --eval_fps 1 \
  --max_frames 10 \
  --device cuda

python tools/legacy/visualize_world_model_diffusionlight.py \
  --video /workspace/WMEval/mirror_ball_unity_test.mp4 \
  --output_dir /workspace/WMEval/reports/diffusionlight_viz/mirror_ball_unity_test \
  --eval_fps 2 \
  --max_frames 20 \
  --device cuda

python tools/legacy/visualize_world_model_diffusionlight.py \
  --video /workspace/WMEval/mirror_ball_genie_test.mp4 \
  --output_dir /workspace/WMEval/reports/diffusionlight_viz/mirror_ball_genie_test \
  --eval_fps 2 \
  --max_frames 20 \
  --device cuda
