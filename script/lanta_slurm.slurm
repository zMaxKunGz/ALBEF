#!/bin/bash
#SBATCH -p gpu-limited 
#SBATCH -N 1 -c 16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gpus=4
#SBATCH -t 48:00:00 
#SBATCH -A lt200226
#SBATCH -J albef_mscoco
#SBATCH --array=0

module load Mamba/23.11.0-0
export WANDB_MODE="offline"
conda activate pytorch

# Define the list of part-of-speech tags.
pos_array=('ADJ' 'ADP' 'ADV' 'AUX' 'DET' 'NOUN' 'PRON' 'PROPN' 'PUNCT' 'VERB')

# Select the POS based on the array index.
selected_pos=${pos_array[$SLURM_ARRAY_TASK_ID]}
echo "Running with pos: ${selected_pos}"

# Run the training script with the selected POS.
torchrun --nnodes=1 --nproc_per_node=4 ../Pretrain.py \
  --config ../configs/Pretrain.yaml \
  --output_dir ../output/NoMLM-42 \
  --pos all

conda deactivate
