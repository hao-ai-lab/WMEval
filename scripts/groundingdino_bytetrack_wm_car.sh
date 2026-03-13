python tools/groundingdino_bytetrack.py \
  --video /home/lah003/workspace/VideoScience/tmp/Geniee57f31076ebd4cbbbd7234559da0a799.mp4 \
  --text_prompt "car" \
  --grounding_cfg /home/lah003/workspace/VideoScience/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt /home/lah003/workspace/VideoScience/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --viz_fps 2 \
  --track_thresh 0.3 \
  --output_dir mechanics_tracking_frames/GenieRacingCar_0 \
  --output_json mechanics_tracking_frames/GenieRacingCar_0/reports/tracking_report.json

