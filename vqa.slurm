#!/bin/bash
#SBATCH -p gpu-limited 
#SBATCH -N 1 -c 16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gpus=4
#SBATCH -t 48:00:00 
#SBATCH -A lt200226
#SBATCH -J vqa-training
#SBATCH --array=0-4

module load Mamba/23.11.0-0
export WANDB_MODE="offline"
conda activate pytorch

# Define the list of part-of-speech tags.
checkpoints=(
#    "./output/Noqueue-2491/11-04-2025:13-52DET/checkpoint_09.pth"
#    "./output/Noqueue-2491/11-04-2025:16-44PROPN/checkpoint_09.pth"
#    "./output/Noqueue-2491/15-04-2025:02-17all/checkpoint_09.pth"
#    "./output/Noqueue-2491/16-03-2025:13-59NOUN/checkpoint_09.pth"
#    "./output/Noqueue-2491/26-03-2025:00-08VERB/checkpoint_09.pth"
    "./output/Noqueue-42/16-04-2025:18-07ADJ/checkpoint_09.pth"
    "./output/Noqueue-42/16-04-2025:18-08ADP/checkpoint_09.pth"
    "./output/Noqueue-42/16-04-2025:18-18ADV/checkpoint_09.pth"
    "./output/Noqueue-42/16-04-2025:18-23AUX/checkpoint_09.pth"
    "./output/Noqueue-42/16-04-2025:18-58PRON/checkpoint_09.pth"
)

checkpoint=${checkpoints[$SLURM_ARRAY_TASK_ID]}
echo "Running with checkpoint: ${checkpoint}"

torchrun --nnodes=1 --nproc_per_node=4 ./VQA.py \
    --config "./configs/VQA.yaml" \
    --output_dir "./output/VQA" \
    --checkpoint "$checkpoint"
conda deactivate
