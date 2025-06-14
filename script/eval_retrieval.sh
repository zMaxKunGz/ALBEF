#!/bin/bash

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=0

# Array of checkpoints
checkpoints=(
    "../output/Noqueue-2491/11-04-2025:13-52PUNCT/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:17-46all/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-07ADJ/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-08ADP/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-18ADV/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-23AUX/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-46DET/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-48NOUN/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:18-58PRON/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:19-01PROPN/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:19-07PUNCT/checkpoint_09.pth"
    # "../output/Noqueue-42/16-04-2025:19-07VERB/checkpoint_09.pth"
    # "../output/Masking100/03-05-2025:21-57ADJ/checkpoint_09.pth"
    # "../output/Masking100/03-05-2025:22-06ADV/checkpoint_09.pth"
    # "../output/Masking100/03-05-2025:22-06NOUN/checkpoint_09.pth"
    # "../output/Masking100/05-05-2025:07-40PROPN/checkpoint_09.pth"
    # "../output/Masking100/14-05-2025:21-15VERB/checkpoint_09.pth"
)

# Loop through each checkpoint
for checkpoint in "${checkpoints[@]}"
do
    torchrun --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29501 ../Retrieval.py \
        --config ../configs/Retrieval_flickr.yaml \
        --output_dir ../result/Retrieval_flickr \
        --evaluate \
        --checkpoint "$checkpoint"
done