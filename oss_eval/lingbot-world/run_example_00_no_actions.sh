export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 generate.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir /raid/lah003/models/lingbot-world-base-cam \
    --image examples/00/image.jpg \
    --dit_fsdp --t5_fsdp \
    --ulysses_size 4 \
    --frame_num 161 \
    --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
