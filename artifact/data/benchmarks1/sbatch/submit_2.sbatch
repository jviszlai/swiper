#!/bin/bash
#SBATCH --job-name=20250214_143749
#SBATCH --output=slurm/data/20250214_143749/logs/%a.out
#SBATCH --error=slurm/data/20250214_143749/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=1,7,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250214_143749/config.json" "slurm/data/20250214_143749/output" 604800