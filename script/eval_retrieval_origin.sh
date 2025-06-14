#!/bin/bash

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1  --rdzv-backend=c10d --rdzv-endpoint=localhost:29501 ../Retrieval_origin.py \
    --config ../configs/Retrieval_flickr.yaml \
    --output_dir ../output/Retrieval_coco \
    --evaluate \
    --checkpoint ../output/origin-2491/checkpoint_best.pth
    # --checkpoint ../pretrain-weight/ALBEF_4M.pth
