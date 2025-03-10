#!/bin/bash
#!/bin/bash
#SBATCH -p gpu 
#SBATCH -N 1 -c 16 
#SBATCH --ntasks-per-node=4 
#SBATCH --gpus=2 
#SBATCH -t 24:00:00 
#SBATCH -A pv812004
#SBATCH -J albef_mscoco

module load Mamba/23.11.0-0
conda activate pytorch
./pretrain_script.sh

conda deactivate
