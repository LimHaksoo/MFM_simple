export CUDA_VISIBLE_DEVICES=4

python train.py \
    --single_root /data3/jisu/MFM/datasets/VisA \
    --single_name visa \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 8 \
    --num_vision_tokens 2 \
    --max_length 64 \
    --amp