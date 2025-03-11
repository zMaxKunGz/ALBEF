#!/bin/bash
#!/bin/bash
#SBATCH -p gpu 
#SBATCH -N 1 -c 16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gpus=4
#SBATCH -t 48:00:00 
#SBATCH -A pv812004
#SBATCH -J albef_mscoco

module load Mamba/23.11.0-0
export WANDB_MODE="offline"
conda activate pytorch
./pretrain_script.sh

conda deactivate
