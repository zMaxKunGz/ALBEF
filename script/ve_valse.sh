#!/bin/bash

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=1

# Array of checkpoints
checkpoints=(
    "../output/origin-2491/checkpoint_best.pth"
    "../pretrain-weight/ALBEF_4M.pth"
    # "../output/Noqueue-2491/10-04-2025:02-38AUX/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:13-52ADP/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:13-52DET/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:13-52PRON/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:13-52PUNCT/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:16-37ADJ/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:16-44ADV/checkpoint_09.pth"
    # "../output/Noqueue-2491/11-04-2025:16-44PROPN/checkpoint_09.pth"
    # "../output/Noqueue-2491/15-04-2025:02-17all/checkpoint_09.pth"
    # "../output/Noqueue-2491/16-03-2025:13-59NOUN/checkpoint_09.pth"
    # "../output/Noqueue-2491/26-03-2025:00-08VERB/checkpoint_09.pth"
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
    # "../output/Masking-100/03-05-2025:21-57ADJ/checkpoint_09.pth"
    # "../output/Masking-100/03-05-2025:22-06ADV/checkpoint_09.pth"
    # "../output/Masking-100/03-05-2025:22-06NOUN/checkpoint_09.pth"
    # "../output/Masking-100/05-05-2025:07-40PROPN/checkpoint_09.pth"
    # "../output/Masking-100/14-05-2025:21-15VERB/checkpoint_09.pth"
)

tasks=(
    "actant-swap"
    "actions-replacement"
    "coreference-hard"
    "coreference-standard"
    "counting-adversarial"
    "counting-hard"
    "counting-small-quant"
    "existence"
    "foil-it"
    "prurals"
    "relations"
)

for task in "${tasks[@]}"
do
    for checkpoint in "${checkpoints[@]}"
    do
        torchrun --nproc_per_node=1 ../VE.py \
            --config "../configs/VALSE_configs/VE-${task}.yaml" \
            --output_dir "../result/VE-${task}" \
            --evaluate \
            --type "${task}" \
            --checkpoint "$checkpoint"
    done
done
