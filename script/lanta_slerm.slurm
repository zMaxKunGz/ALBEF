#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 8                       # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1 	        # Specify number of tasks per node
#SBATCH --gpus=4		                # Specify total number of GPUs
#SBATCH -t 24:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A pv812004                     # Specify project name
#SBATCH -J albef_mscoco                 # Specify job name

module load Mamba/23.11.0-0
conda activate pytorch

./pretrain_script.sh

conda deactivate