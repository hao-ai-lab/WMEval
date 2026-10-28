python tools/legacy/groundingdino_bytetrack.py \
  --video /home/lah003/workspace/VideoScience/tmp/Genie2c72ec1ed75b42d5afccce4418d1ee02.mp4 \
  --text_prompt "ball" \
  --grounding_cfg /home/lah003/workspace/VideoScience/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /home/lah003/workspace/VideoScience/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --viz_fps 2 \
  --track_thresh 0.3 \
  --output_dir mechanics_tracking_frames/GenieRollingBall_0 \
  --output_json mechanics_tracking_frames/GenieRollingBall_0/reports/tracking_report.json

