#!/bin/bash
#SBATCH --job-name=20250212_153333
#SBATCH --output=slurm/data/20250212_153333/logs/%a.out
#SBATCH --error=slurm/data/20250212_153333/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21,24,25,26,27,30,31,32,33,36,37,38,39,42,43,44,45,48,49,50,51,54,55,56,57,60,61,62,63,66,67,68,69,72,73,74,75,78,79,80,81,84,85,86,87,90,91,92,93,96,97,98,99,102,103,104,105,108,109,110,111,114,115,116,117,120,121,122,123
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250212_153333/config.json" "slurm/data/20250212_153333/output" 604800