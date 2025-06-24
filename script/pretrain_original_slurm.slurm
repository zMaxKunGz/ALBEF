#!/bin/bash
#SBATCH -p gpu 
#SBATCH -N 1 -c 16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gpus=4
#SBATCH -t 48:00:00 
#SBATCH -A lt200226
#SBATCH -J albef_mscoco

module load Mamba/23.11.0-0
export WANDB_MODE="offline"
conda activate pytorch

# Run the training script with the selected POS.
torchrun --nnodes=1 --nproc_per_node=4 ../Pretrain_ALBEF.py \
  --config ../configs/Pretrain.yaml \
  --output_dir ../output/origin \
  --pos all

conda deactivate
