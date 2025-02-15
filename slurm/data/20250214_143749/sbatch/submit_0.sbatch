#!/bin/bash
#SBATCH --job-name=20250214_143749
#SBATCH --output=slurm/data/20250214_143749/logs/%a.out
#SBATCH --error=slurm/data/20250214_143749/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=4,10,16,22,28,34,40,46,52,58,64,70,76,82,88,94,100,106,112,118,124
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250214_143749/config.json" "slurm/data/20250214_143749/output" 604800