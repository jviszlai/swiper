#!/bin/bash
#SBATCH --job-name=20250214_143749
#SBATCH --output=slurm/data/20250214_143749/logs/%a.out
#SBATCH --error=slurm/data/20250214_143749/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65,66,68,69,71,72,74,75,77,78,80,81,83,84,86,87,89,90,92,93,95,96,98,99,101,102,104,105,107,108,110,111,113,114,116,117,119,120,122,123,125
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250214_143749/config.json" "slurm/data/20250214_143749/output" 604800