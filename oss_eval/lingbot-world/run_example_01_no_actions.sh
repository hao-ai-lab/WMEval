export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 generate.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir /raid/lah003/models/lingbot-world-base-cam \
    --image examples/billiards_game_0/image.png \
    --dit_fsdp --t5_fsdp \
    --ulysses_size 4 \
    --frame_num 161 \
    --prompt "The environment is a billiard game. At the beginning, the white cue ball strikes the rest of the balls. All balls move according to physical laws, including momentum transfer and collisions. The tabletop has friction that gradually slows the balls down.
The provided image is used as the initial starting scene. All existing balls in the image are included in the simulation. No additional balls are added during the session.
The camera remains static for the entire video. The view always shows the full tabletop. No person or object blocks the view of the table."
