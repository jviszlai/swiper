#!/bin/bash
#SBATCH --job-name=20250212_152913
#SBATCH --output=slurm/data/20250212_152913/logs/%a.out
#SBATCH --error=slurm/data/20250212_152913/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=0,2,3,6,8,9,12,14,15,18,20,21,24,26,27,30,32,33,36,38,39,42,44,45,48,50,51,54,56,57,60,62,63,66,68,69,72,74,75,78,80,81,84,86,87,90,92,93,96,98,99,102,104,105,108,110,111,114,116,117,120,122,123
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250212_152913/config.json" "slurm/data/20250212_152913/output" 604800