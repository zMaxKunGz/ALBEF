#!/bin/bash
#SBATCH --job-name=albef_mscoco           # Job name
#SBATCH --output=%x_%j.out            # Standard output file
#SBATCH --error=%x_%j.err             # Standard error file
#SBATCH --partition=ASL-gpu           # Partition name for GPU jobs
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --gres=gpu:4                  # Number of GPUs needed
#SBATCH --time=24:00:00               # Maximum runtime (HH:MM:SS)

# Load any necessary modules
source ../pytorch/bin/activate

# Run your GPU job commands here
./script/pretrain_script.sh

deactivate